"""Chat Completions compat engine (openai SDK) — deepseek, moonshot, xai, openrouter.

One engine, four provider quirk-sets, dispatched flat on `row.provider`:

- deepseek: `max_tokens`; `reasoning_content` preserved in and replayed from
  the continuation artifact. Default-auto thinking-mode tool calls omit
  `tool_choice` and rely on DeepSeek's documented default; an explicit
  nondefault choice is rejected before dispatch.
- moonshot: `max_completion_tokens`; continuation = the COMPLETE native
  assistant message replayed verbatim, including `reasoning_content`
  (Preserved Thinking).
- xai: `max_completion_tokens`; native structured outputs (`response_format`
  json_schema) on `structured="native"` rows; `reasoning_content` continuity
  as deepseek (strip on resend).
- openrouter: `max_tokens`; the row's full routing pins as `provider` on EVERY
  call (no unpinned passthrough); ordered `reasoning_details` preserved
  verbatim into the artifact and replayed on assistant messages; upstream
  provider name from the response body; no `stream_options` (conflicts with
  require_parameters); in-band error objects on an HTTP-200 body.

Reasoning is NOT a quirk-set: the engine carries zero per-provider reasoning
shape knowledge. `row.reasoning[level]` is a self-describing request fragment
merged verbatim into the body; the shared `row_reasoning` owns which levels are
expressible, which keys join the `provider_options` collision set, and what
`CallMeta.native_reasoning` records.

The SDK owns the wire (serialization, transport, SSE, error envelopes); this
module owns classification against the shared taxonomy and the IR mapping.
ONE attempt per call: retryable trouble raises `TransientAttempt`; expected
non-retryable failures return outcome values with a fully populated `CallMeta`;
malformed envelopes raise `ProtocolDefect`; 401/403 raises `CredentialRejected`.
"""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, assert_never, cast

import httpx
import openai
from openai import omit
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall
from openai.types.completion_usage import CompletionUsage

from provider_runtime.engines import TransientAttempt
from provider_runtime.engines._common import (
    int_or_none,
    mapping_or_none,
    monotonic_ms,
    registry_invalid,
    response_content,
    retry_after_seconds,
    row_reasoning,
    str_or_none,
    validate_continuation,
)
from provider_runtime.engines._openai_common import (
    transient_connection,
    zero_env_client,
)
from provider_runtime.errors import (
    CredentialRejected,
    InvalidRequest,
    ProtocolDefect,
    RuntimeDefect,
    safe_provider_error_body_snippet,
)
from provider_runtime.registry import REGISTRY_REVISION, ModelRow
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    AttemptRecord,
    CallMeta,
    CallOutcome,
    CanonicalTool,
    CodecStreamEvent,
    ContinuationArtifact,
    ContinuationDelta,
    Failed,
    FinalAttempt,
    GenerateIntent,
    ImageBlock,
    Incomplete,
    InvalidStructuredOutput,
    InvalidToolArguments,
    PossiblyBillable,
    Presence,
    Present,
    PromptBlock,
    ProviderContextTooLarge,
    ProviderCredential,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTimeout,
    ResponsePayload,
    StreamStart,
    StrictJsonOutput,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextDelta,
    TextOutput,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallDone,
    ToolCallStart,
    ToolResultMessage,
    TransientCause,
    TransportUnavailable,
    UsageEvent,
    UserMessage,
    presence_of,
)

# The compat quirk-sets this engine serves.
type _Served = Literal["deepseek", "moonshot", "xai", "openrouter"]

# Every request key this engine writes itself: the body fields it maps from
# core intent fields (both output-cap spellings — the provider decides which),
# plus the kwargs it adds at the call sites. Two roles, one set. A
# provider_options key in it is an override, not an extension →
# InvalidRequest; the reasoning knob is row data, so every key IT can write
# joins that collision set per call (`_encode`). And because the body is built
# ON TOP of the fragment, a row whose fragment names one of these has its knob
# silently overwritten by the engine while CallMeta still reports the fragment
# as sent → registry_invalid.
_OWNED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "response_format",
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "provider",
    }
)


def _served_provider(row: ModelRow) -> _Served:
    match row.provider:
        case "deepseek" | "moonshot" | "xai" | "openrouter" as provider:
            return provider
        case other:
            raise registry_invalid(row, f"openai_chat engine does not serve provider {other!r}")


def _sequence_or_none(value: object) -> Sequence[object] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return None


# ---------------------------------------------------------------------------
# Encode — intent + row → messages array and extra-body params. The SDK
# serializes; unknown keys pass through untouched (verified against the
# installed SDK's transform).


@dataclass(frozen=True, slots=True)
class _Encoded:
    messages: list[dict[str, object]] = field(repr=False)  # may carry replayed payloads
    body: dict[str, object] = field(repr=False)
    native_reasoning: Presence[str]


def _encode(provider: _Served, row: ModelRow, intent: GenerateIntent) -> _Encoded:
    reasoning = row_reasoning(row, intent)
    fragment_collisions = sorted(_OWNED_KEYS & reasoning.owned_keys)
    if fragment_collisions:
        raise registry_invalid(
            row,
            f"reasoning fragment keys {fragment_collisions!r} would rewrite request fields "
            f"the engine sets itself",
        )
    body: dict[str, object] = dict(reasoning.fragment)
    match provider:
        case "moonshot" | "xai":
            body["max_completion_tokens"] = intent.max_output_tokens
        case "deepseek" | "openrouter":
            # deepseek documents max_tokens; openrouter routes max_tokens.
            body["max_tokens"] = intent.max_output_tokens
        case _:
            assert_never(provider)
    if intent.tools:
        body["tools"] = [_tool_definition(tool) for tool in intent.tools]
        # DeepSeek's thinking-mode tool guide requires the native assistant
        # reasoning to be replayed and demonstrates the provider-default tool
        # selection. Its thinking mode does not support tool_choice: omitting
        # the default `auto` is semantically exact, whereas omitting an
        # explicit nondefault choice would silently change caller intent.
        # Reject that unsupported combination before any network dispatch.
        if provider == "deepseek" and intent.reasoning != "none":
            if intent.tool_choice != "auto":
                raise InvalidRequest(
                    message=(
                        "DeepSeek thinking-mode tool calls do not support a nondefault "
                        f"tool_choice (got {intent.tool_choice!r})"
                    )
                )
        else:
            body["tool_choice"] = intent.tool_choice
    match intent.output:
        case TextOutput():
            pass
        case StrictJsonOutput() as output:
            match row.structured:
                case "native":
                    body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": output.name,
                            "schema": dict(output.schema),
                            "strict": True,
                        },
                    }
                case "json_mode":
                    # json_mode rows constrain to a JSON object; the caller's
                    # schema is enforced by validation (json_out), not the wire.
                    body["response_format"] = {"type": "json_object"}
                case _:
                    assert_never(row.structured)
        case _:
            assert_never(intent.output)
    if provider == "openrouter":
        body["provider"] = _routing_pins(row)
    owned = _OWNED_KEYS | reasoning.owned_keys
    for key in intent.provider_options:
        if key in owned:
            raise InvalidRequest(
                message=f"provider_options key {key!r} collides with a core field "
                f"the openai_chat engine maps itself"
            )
    body.update(intent.provider_options)
    return _Encoded(
        messages=_encode_messages(provider, row, intent),
        body=body,
        native_reasoning=reasoning.native_reasoning,
    )


def _routing_pins(row: ModelRow) -> dict[str, object]:
    match row.routing:
        case Present(value=routing):
            return {
                "only": list(routing.only),
                "order": list(routing.order),
                "allow_fallbacks": routing.allow_fallbacks,
                "require_parameters": routing.require_parameters,
                "data_collection": routing.data_collection,
                "zdr": routing.zdr,
                "quantizations": list(routing.quantizations),
            }
        case Absent():
            raise registry_invalid(row, "openrouter rows must pin routing")
        case _:
            assert_never(row.routing)


def _tool_definition(tool: CanonicalTool) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        },
    }


def _encode_messages(
    provider: _Served, row: ModelRow, intent: GenerateIntent
) -> list[dict[str, object]]:
    encoded: list[dict[str, object]] = []
    for message in intent.messages:
        match message:
            case SystemMessage(blocks=blocks):
                encoded.append({"role": "system", "content": "".join(b.text for b in blocks)})
            case UserMessage(blocks=blocks):
                encoded.append(_user_wire(blocks))
            case AssistantMessage() as assistant:
                encoded.append(_assistant_wire(provider, row, assistant, intent))
            case ToolResultMessage(call_id=call_id, output=output):
                # is_error has no Chat Completions representation.
                encoded.append({"role": "tool", "tool_call_id": call_id, "content": output})
            case _:
                assert_never(message)
    return encoded


def _user_wire(blocks: tuple[PromptBlock | ImageBlock, ...]) -> dict[str, object]:
    if all(isinstance(block, PromptBlock) for block in blocks):
        text = "".join(block.text for block in blocks if isinstance(block, PromptBlock))
        return {"role": "user", "content": text}
    parts: list[dict[str, object]] = []
    for block in blocks:
        match block:
            case PromptBlock(text=text):
                parts.append({"type": "text", "text": text})
            case ImageBlock(media_type=media_type, data=data):
                url = f"data:{media_type};base64,{b64encode(data).decode('ascii')}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            case _:
                assert_never(block)
    return {"role": "user", "content": parts}


def _assistant_wire(
    provider: _Served, row: ModelRow, message: AssistantMessage, intent: GenerateIntent
) -> dict[str, object]:
    match message.continuation:
        case Absent():
            return _assistant_from_fields(message)
        case Present(value=artifact):
            validate_continuation(artifact, row, intent)
            match provider:
                case "moonshot":
                    # Complete-native-message replay, verbatim — including
                    # reasoning_content (Preserved Thinking).
                    return dict(artifact.opaque_payload)
                case "deepseek":
                    # DeepSeek thinking-mode tool continuations require the
                    # complete native assistant message, including
                    # reasoning_content, on every subsequent request.
                    return dict(artifact.opaque_payload)
                case "xai":
                    # xAI accepts the same payload shape but reasoning_content
                    # is not replayable there.
                    payload = dict(artifact.opaque_payload)
                    payload.pop("reasoning_content", None)
                    return payload
                case "openrouter":
                    # Typed fields supply content/tool_calls; the artifact only
                    # carries the ordered reasoning_details, replayed verbatim.
                    encoded = _assistant_from_fields(message)
                    encoded["reasoning_details"] = _payload_reasoning_details(artifact)
                    return encoded
                case _:
                    assert_never(provider)
        case _:
            assert_never(message.continuation)


def _assistant_from_fields(message: AssistantMessage) -> dict[str, object]:
    # Empty text alongside tool calls is a null content on this wire.
    content: str | None = message.text if (message.text or not message.tool_calls) else None
    encoded: dict[str, object] = {"role": "assistant", "content": content}
    if message.tool_calls:
        encoded["tool_calls"] = [_tool_call_wire(call) for call in message.tool_calls]
    return encoded


def _tool_call_wire(call: ToolCall) -> dict[str, object]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                dict(call.arguments), separators=(",", ":"), ensure_ascii=False
            ),
        },
    }


def _payload_reasoning_details(artifact: ContinuationArtifact) -> list[dict[str, object]]:
    details = _sequence_or_none(artifact.opaque_payload.get("reasoning_details"))
    if details is None:
        raise InvalidRequest(
            message="openrouter continuation payload carries no reasoning_details array"
        )
    entries: list[dict[str, object]] = []
    for detail in details:
        entry = mapping_or_none(detail)
        if entry is None:
            raise InvalidRequest(
                message="openrouter continuation reasoning_details entry is not an object"
            )
        entries.append(dict(entry))
    return entries


# ---------------------------------------------------------------------------
# Usage — raw frames folded field-wise (later non-null wins, nothing erased),
# normalized ONCE into the cache-INCLUSIVE TokenUsage at the terminal.


def _fold_usage(
    base: Mapping[str, object] | None, new: Mapping[str, object] | None
) -> Mapping[str, object] | None:
    if new is None:
        return base
    if base is None:
        return dict(new)
    merged = dict(base)
    for key, value in new.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            inner = dict(existing)
            inner.update({k: v for k, v in value.items() if v is not None})
            merged[key] = inner
        elif value is not None:
            merged[key] = value
    return merged


def _usage_from_raw(provider: _Served, raw: Mapping[str, object]) -> TokenUsage:
    prompt_details = mapping_or_none(raw.get("prompt_tokens_details")) or {}
    completion_details = mapping_or_none(raw.get("completion_tokens_details")) or {}
    # cached_tokens nests under prompt_tokens_details or sits flat on the usage
    # object (Moonshot's form) — both surfaced; prompt_tokens is cache-inclusive
    # on this wire, so no ingress normalization is needed.
    cache_read = int_or_none(prompt_details.get("cached_tokens"))
    if cache_read is None:
        cache_read = int_or_none(raw.get("cached_tokens"))
    try:
        return TokenUsage.from_components(
            input_tokens=int_or_none(raw.get("prompt_tokens")) or 0,
            output_tokens=int_or_none(raw.get("completion_tokens")) or 0,
            total_tokens=presence_of(int_or_none(raw.get("total_tokens"))),
            reasoning_tokens=presence_of(int_or_none(completion_details.get("reasoning_tokens"))),
            cache_read_input_tokens=presence_of(cache_read),
            cache_write_input_tokens=presence_of(
                int_or_none(prompt_details.get("cache_write_tokens"))
            ),
        )
    except ValueError as error:
        raise ProtocolDefect(
            code="malformed_usage",
            message=f"{provider} usage is not valid token accounting: {error}",
        ) from error


# ---------------------------------------------------------------------------
# Tool-call decode — strict JSON-object parse, NO repair.


def _tool_arguments(
    raw: str, *, tool_name: str, call_id: str
) -> Mapping[str, object] | InvalidToolArguments:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return InvalidToolArguments(
            safe_detail=f"tool call {call_id} ({tool_name}): arguments are not valid JSON "
            f"({exc.msg} at char {exc.pos})"
        )
    if not isinstance(parsed, dict):
        return InvalidToolArguments(
            safe_detail=f"tool call {call_id} ({tool_name}): arguments are not a JSON object"
        )
    return parsed


def _decode_tool_calls(
    message: ChatCompletionMessage,
) -> tuple[ToolCall, ...] | InvalidToolArguments:
    calls: list[ToolCall] = []
    for entry in message.tool_calls or []:
        raw = entry.to_dict()
        call_id = str_or_none(raw.get("id")) or ""
        function = mapping_or_none(raw.get("function")) or {}
        name = str_or_none(function.get("name")) or ""
        if not call_id or not name:
            raise ProtocolDefect(
                code="malformed_tool_call", message="tool call is missing id or function.name"
            )
        arguments = _tool_arguments(
            str_or_none(function.get("arguments")) or "", tool_name=name, call_id=call_id
        )
        if isinstance(arguments, InvalidToolArguments):
            return arguments
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return tuple(calls)


@dataclass(slots=True)
class _ToolCallSlot:
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    started: bool = False


@dataclass(frozen=True, slots=True)
class _FinishedToolCall:
    tool_call: ToolCall
    # Exact wire shape rebuilt with the RAW accumulated argument string, for
    # the native-message continuation payload.
    native: Mapping[str, object]


class _ToolCallAccumulator:
    """Accumulates streamed tool-call deltas keyed by ``index``."""

    def __init__(self) -> None:
        self._slots: dict[int, _ToolCallSlot] = {}

    def apply(
        self, entries: Sequence[ChoiceDeltaToolCall]
    ) -> Iterator[ToolCallStart | ToolCallDelta]:
        for entry in entries:
            index = int_or_none(entry.index) or 0
            slot = self._slots.setdefault(index, _ToolCallSlot())
            if entry.id:
                slot.call_id = entry.id
            function = entry.function
            if function is not None and function.name:
                slot.name = function.name
            if slot.call_id and slot.name and not slot.started:
                slot.started = True
                yield ToolCallStart(call_id=slot.call_id, name=slot.name)
            fragment = function.arguments if function is not None else None
            if fragment:
                slot.arguments += fragment
                if slot.started:
                    yield ToolCallDelta(call_id=slot.call_id, arguments_delta=fragment)

    def finish(self) -> tuple[_FinishedToolCall, ...] | InvalidToolArguments:
        """Fold the accumulated slots. Pure — repeating it cannot erase calls."""
        finished: list[_FinishedToolCall] = []
        for index in sorted(self._slots):
            slot = self._slots[index]
            if not slot.call_id or not slot.name:
                raise ProtocolDefect(
                    code="malformed_tool_call",
                    message=f"streamed tool call at index {index} never carried id and function.name",
                )
            arguments = _tool_arguments(slot.arguments, tool_name=slot.name, call_id=slot.call_id)
            if isinstance(arguments, InvalidToolArguments):
                return arguments
            finished.append(
                _FinishedToolCall(
                    tool_call=ToolCall(id=slot.call_id, name=slot.name, arguments=arguments),
                    native={
                        "id": slot.call_id,
                        "type": "function",
                        "function": {"name": slot.name, "arguments": slot.arguments},
                    },
                )
            )
        return tuple(finished)


# ---------------------------------------------------------------------------
# Decode — terminal outcome construction. The output arm is determined by the
# intent's OutputSpec, never re-inferred from the wire.


def _terminal_outcome(
    *,
    finish_reason: str | None,
    meta: CallMeta,
    intent: GenerateIntent,
    text: str,
    tool_calls: tuple[ToolCall, ...],
    continuation: Presence[ContinuationArtifact],
) -> Succeeded | Incomplete | Failed:
    match finish_reason:
        case "stop" | "tool_calls":
            content = response_content(intent, text=text, tool_calls=tool_calls)
            if isinstance(content, InvalidStructuredOutput):
                return Failed(meta=meta, failure=content)
            return Succeeded(
                meta=meta, response=ResponsePayload(content=content, continuation=continuation)
            )
        case "length":
            return Incomplete(
                meta=meta,
                reason="max_output_tokens",
                status="provider_incomplete",
                safe_detail=Absent(),
            )
        case "content_filter":
            return Incomplete(
                meta=meta,
                reason="content_filter_partial",
                status="provider_incomplete",
                safe_detail=Absent(),
            )
        case _:
            raise ProtocolDefect(
                code="unknown_finish_reason",
                message=f"terminal finish_reason {finish_reason!r} is not recognized",
            )


def _continuation_from_message(
    provider: _Served, row: ModelRow, intent: GenerateIntent, message: ChatCompletionMessage
) -> Presence[ContinuationArtifact]:
    extra = message.model_extra or {}
    match provider:
        case "openrouter":
            details = _sequence_or_none(extra.get("reasoning_details"))
            if not details:
                return Absent()
            # Verbatim preservation: entries stored exactly as received, in order.
            payload: Mapping[str, object] = {"reasoning_details": list(details)}
        case "deepseek" | "moonshot" | "xai":
            if not str_or_none(extra.get("reasoning_content")) and not message.tool_calls:
                return Absent()
            # The payload is the complete native assistant message, verbatim.
            payload = message.to_dict()
        case _:
            assert_never(provider)
    return Present(
        ContinuationArtifact(
            target=intent.target, codec_id=row.continuation_codec, opaque_payload=payload
        )
    )


def _stream_continuation(
    provider: _Served,
    row: ModelRow,
    intent: GenerateIntent,
    *,
    text: str,
    reasoning: str,
    details: Sequence[object],
    finished: tuple[_FinishedToolCall, ...],
) -> Presence[ContinuationArtifact]:
    match provider:
        case "openrouter":
            if not details:
                return Absent()
            payload: dict[str, object] = {"reasoning_details": list(details)}
        case "deepseek" | "moonshot" | "xai":
            if not reasoning and not finished:
                return Absent()
            # Reconstruct the complete native assistant message for replay.
            payload = {"role": "assistant", "content": text if (text or not finished) else None}
            if reasoning:
                payload["reasoning_content"] = reasoning
            if finished:
                payload["tool_calls"] = [dict(call.native) for call in finished]
        case _:
            assert_never(provider)
    return Present(
        ContinuationArtifact(
            target=intent.target, codec_id=row.continuation_codec, opaque_payload=payload
        )
    )


# ---------------------------------------------------------------------------
# Error classification — SDK exception → TransientAttempt / defect / value.


def _classify_status(provider: _Served, exc: openai.APIStatusError) -> ProviderContextTooLarge:
    """Classify a non-2xx response: raises for everything except the one
    non-retryable expected failure (context overflow), which returns."""
    status = exc.status_code
    # The SDK unwraps body["error"] before attaching it (`body.get("error",
    # body)`), so exc.body is usually the inner error object already; tolerate
    # both shapes.
    body = mapping_or_none(exc.body)
    nested = mapping_or_none(body.get("error")) if body is not None else None
    error = nested if nested is not None else body
    snippet = safe_provider_error_body_snippet(dict(body) if body is not None else None)
    detail = f": {snippet}" if snippet else ""
    request_id = presence_of(exc.request_id)

    if status == 403 and provider == "openrouter" and _is_moderation_flagged(error):
        raise RuntimeDefect(
            origin="provider_response",
            code="input_moderation_flagged",
            message=f"openrouter flagged the input for moderation (HTTP {status}){detail}",
        )
    if status in (401, 403):
        raise CredentialRejected(
            message=f"{provider} rejected the platform credential (HTTP {status}){detail}"
        )
    if status == 402 or _mentions_quota(error):
        raise RuntimeDefect(
            origin="provider_http",
            code="quota_exhausted",
            message=f"{provider} reports exhausted quota/credits (HTTP {status}){detail}",
        )
    if status == 429:
        raise TransientAttempt(
            cause=ProviderRateLimit(retry_after=retry_after_seconds(exc.response.headers)),
            status_code=Present(status),
            provider_request_id=request_id,
            billability=PossiblyBillable(),
        )
    if status >= 500:
        raise TransientAttempt(
            cause=ProviderHttpUnavailable(),
            status_code=Present(status),
            provider_request_id=request_id,
            billability=PossiblyBillable(),
        )
    if _is_context_overflow(error):
        return ProviderContextTooLarge()
    if body is None:
        raise ProtocolDefect(
            code="unparseable_error_body",
            message=f"{provider} HTTP {status} error body is not a JSON object",
        )
    raise RuntimeDefect(
        origin="provider_http",
        code="unclassified_provider_error",
        message=f"{provider} returned unclassified HTTP {status}{detail}",
    )


def _is_moderation_flagged(error: Mapping[str, object] | None) -> bool:
    """A 403 body carrying moderation metadata is a per-request content failure,
    not a rejected platform credential."""
    if error is None:
        return False
    metadata = mapping_or_none(error.get("metadata"))
    if metadata is None:
        return False
    return _sequence_or_none(metadata.get("reasons")) is not None or "flagged_input" in metadata


def _mentions_quota(error: Mapping[str, object] | None) -> bool:
    if error is None:
        return False
    kind = (str_or_none(error.get("type")) or "") + (str_or_none(error.get("code")) or "")
    return "quota" in kind


def _is_context_overflow(error: Mapping[str, object] | None) -> bool:
    if error is None:
        return False
    if str_or_none(error.get("code")) == "context_length_exceeded":
        return True
    message = (str_or_none(error.get("message")) or "").lower()
    return "context length" in message or "token limit" in message or "maximum context" in message


def _classify_inband_error(provider: _Served, error: object) -> TransientCause:
    """Classify an in-band error object carried by an HTTP-200 body — the
    OpenRouter shape for an upstream that failed after the gateway accepted
    the request. 429-shaped → ProviderRateLimit; a DEFINITE 4xx code names a
    request the provider will refuse identically next time, so it raises.
    Everything else — including the upstream-failure envelopes that carry only
    a message and metadata — is an upstream that fell over: retryable."""
    parsed = mapping_or_none(error)
    raw_code = parsed.get("code") if parsed is not None else None
    code = int_or_none(raw_code)
    if code is None:
        digits = str_or_none(raw_code)
        code = int(digits) if digits is not None and digits.isdigit() else None
    if code == 429:
        return ProviderRateLimit(retry_after=Absent())
    if code is not None and 400 <= code < 500:
        snippet = safe_provider_error_body_snippet({"error": dict(parsed)} if parsed else None)
        raise ProtocolDefect(
            code="inband_provider_error",
            message=f"{provider} returned an unclassified in-band error on an HTTP 200 response"
            + (f": {snippet}" if snippet else ""),
        )
    return ProviderHttpUnavailable()


# ---------------------------------------------------------------------------
# Engine


class OpenAIChatEngine:
    """One attempt per call against a compat provider; the runtime owns retries."""

    def __init__(
        self, *, timeout_s: float = 600.0, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self._timeout_s = timeout_s
        self._http_client = http_client

    def _client(self, row: ModelRow, credential: ProviderCredential) -> openai.AsyncOpenAI:
        match row.base_url:
            case Present(value=base_url):
                pass
            case Absent():
                raise registry_invalid(row, "openai_chat rows must carry a base_url")
            case _:
                assert_never(row.base_url)
        return zero_env_client(
            api_key=credential.key,
            base_url=base_url,
            timeout_s=self._timeout_s,
            http_client=self._http_client,
        )

    async def generate(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> CallOutcome:
        provider = _served_provider(row)
        encoded = _encode(provider, row, intent)
        client = self._client(row, credential)
        started_ms = monotonic_ms()
        try:
            try:
                completion = await client.chat.completions.create(
                    model=row.model_id,
                    messages=cast("list[ChatCompletionMessageParam]", encoded.messages),
                    extra_body=encoded.body,
                )
            except openai.APIStatusError as exc:
                overflow = _classify_status(provider, exc)
                return Failed(
                    meta=self._error_meta(row, exc, encoded.native_reasoning, started_ms),
                    failure=overflow,
                )
            except openai.APIConnectionError as exc:
                raise transient_connection(exc) from exc
            except json.JSONDecodeError as exc:
                raise ProtocolDefect(
                    code="malformed_json",
                    message=f"{provider} 2xx response body is not valid JSON",
                ) from exc
            return self._decode_completion(
                provider, row, intent, completion, encoded.native_reasoning, started_ms
            )
        finally:
            if self._http_client is None:
                await client.close()

    async def stream(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> AsyncIterator[CodecStreamEvent]:
        provider = _served_provider(row)
        encoded = _encode(provider, row, intent)
        client = self._client(row, credential)
        started_ms = monotonic_ms()
        try:
            try:
                api_stream = await client.chat.completions.create(
                    model=row.model_id,
                    messages=cast("list[ChatCompletionMessageParam]", encoded.messages),
                    stream=True,
                    # openrouter: include_usage conflicts with require_parameters;
                    # the final chunk carries top-level usage anyway.
                    stream_options=(omit if provider == "openrouter" else {"include_usage": True}),
                    extra_body=encoded.body,
                )
            except openai.APIStatusError as exc:
                overflow = _classify_status(provider, exc)
                yield TerminalEvent(
                    outcome=Failed(
                        meta=self._error_meta(row, exc, encoded.native_reasoning, started_ms),
                        failure=overflow,
                    )
                )
                return
            except openai.APIConnectionError as exc:
                raise transient_connection(exc) from exc

            # Provider accepted the request (headers + 2xx) — the envelope opens.
            yield StreamStart()

            semantic = False
            request_id: str | None = None
            model: str | None = None
            upstream: str | None = None
            finish_reason: str | None = None
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            details: list[object] = []
            accumulator = _ToolCallAccumulator()
            finished: tuple[_FinishedToolCall, ...] = ()
            failure: InvalidToolArguments | None = None
            raw_usage: Mapping[str, object] | None = None

            def interrupted(cause: TransientCause) -> TransientAttempt:
                # What actually happened, on an open stream. Whether a
                # post-semantic transient is retryable — and the leaf that
                # then stands for it — is the runtime's call, not ours.
                return TransientAttempt(
                    cause=cause,
                    status_code=Present(200),
                    provider_request_id=presence_of(request_id),
                    billability=PossiblyBillable(),
                )

            try:
                async for chunk in api_stream:
                    if not isinstance(chunk, ChatCompletionChunk):
                        raise ProtocolDefect(
                            code="malformed_envelope",
                            message=f"{provider} stream chunk is not a chat.completion.chunk",
                        )
                    request_id = request_id or (str_or_none(chunk.id) or None)
                    model = model or (str_or_none(chunk.model) or None)
                    if provider == "openrouter":
                        upstream = (
                            str_or_none((chunk.model_extra or {}).get("provider")) or upstream
                        )

                    choice = chunk.choices[0] if chunk.choices else None
                    # Usage may ride top-level frames (all four) or choices[0]
                    # of the finish chunk (Moonshot) — fold whichever appear.
                    top_usage = (
                        chunk.usage.to_dict() if isinstance(chunk.usage, CompletionUsage) else None
                    )
                    choice_usage = (
                        mapping_or_none((choice.model_extra or {}).get("usage"))
                        if choice is not None
                        else None
                    )
                    folded = _fold_usage(_fold_usage(raw_usage, choice_usage), top_usage)
                    if folded is not None and folded is not raw_usage:
                        raw_usage = folded
                        yield UsageEvent(usage=_usage_from_raw(provider, folded))
                        semantic = True
                    if choice is None:
                        continue

                    delta = choice.delta if isinstance(choice.delta, ChoiceDelta) else None
                    if delta is not None:
                        for tool_event in accumulator.apply(delta.tool_calls or ()):
                            yield tool_event
                            semantic = True
                        text = str_or_none(delta.content) or ""
                        if text:
                            text_parts.append(text)
                            yield TextDelta(text=text)
                            semantic = True
                        delta_extra = delta.model_extra or {}
                        reasoning = str_or_none(delta_extra.get("reasoning_content")) or ""
                        if reasoning:
                            reasoning_parts.append(reasoning)
                        delta_details = _sequence_or_none(delta_extra.get("reasoning_details"))
                        if delta_details:
                            details.extend(delta_details)  # verbatim, in sequence

                    chunk_finish = str_or_none(choice.finish_reason)
                    # The FIRST terminal frame decides: a provider that repeats
                    # finish_reason on a trailing frame must not re-fold the
                    # accumulator or re-emit ToolCallDone.
                    if chunk_finish is not None and finish_reason is None:
                        if chunk_finish == "error":
                            # Off-spec openrouter shape: transient by contract
                            # even without a top-level error object.
                            raise interrupted(ProviderHttpUnavailable())
                        finish_reason = chunk_finish
                        finished_or_failure = accumulator.finish()
                        if isinstance(finished_or_failure, InvalidToolArguments):
                            failure = finished_or_failure
                            break
                        finished = finished_or_failure
                        for call in finished:
                            yield ToolCallDone(tool_call=call.tool_call)
                            semantic = True
            except TransientAttempt:
                raise
            except openai.APITimeoutError as exc:
                raise interrupted(ProviderTimeout()) from exc
            except openai.APIConnectionError as exc:
                raise interrupted(TransportUnavailable()) from exc
            except openai.APIError as exc:
                # In-band mid-stream error chunk (HTTP stays 200), surfaced by
                # the SDK as a bare APIError carrying the error object.
                raise interrupted(_classify_inband_error(provider, exc.body)) from exc
            except json.JSONDecodeError as exc:
                raise ProtocolDefect(
                    code="malformed_json",
                    message=f"{provider} stream chunk is not valid JSON",
                ) from exc
            except httpx.TimeoutException as exc:
                raise interrupted(ProviderTimeout()) from exc
            except httpx.TransportError as exc:
                raise interrupted(TransportUnavailable()) from exc

            def stream_meta(meta_model: str) -> CallMeta:
                return CallMeta(
                    provider=row.provider,
                    model=meta_model,
                    provider_request_id=presence_of(request_id),
                    upstream_provider=presence_of(upstream),
                    usage=Present(_usage_from_raw(provider, raw_usage))
                    if raw_usage is not None
                    else Absent(),
                    attempt_trace=(
                        AttemptRecord(
                            attempt=1,
                            signal=FinalAttempt(),
                            status_code=Present(200),
                            started_at_ms=started_ms,
                            ended_at_ms=monotonic_ms(),
                        ),
                    ),
                    billability=PossiblyBillable(),
                    native_reasoning=encoded.native_reasoning,
                    registry_revision=REGISTRY_REVISION,
                )

            if failure is not None:
                yield TerminalEvent(
                    outcome=Failed(meta=stream_meta(model or row.model_id), failure=failure)
                )
                return
            if finish_reason is None:
                # The stream ended without a terminal frame — a cut, not a
                # close. The SDK consumes `[DONE]` invisibly and ends iteration
                # the same way on byte exhaustion, so the old lane's `saw_done`
                # guarantee does not survive renting the wire: a truncation
                # after the finish frame but before a provider's trailing usage
                # frame decodes as a clean close with incomplete usage.
                raise interrupted(ProviderStreamInterrupted(partial_output=semantic))
            if model is None:
                raise ProtocolDefect(
                    code="missing_model", message=f"{provider} stream carried no model field"
                )

            text = "".join(text_parts)
            continuation = _stream_continuation(
                provider,
                row,
                intent,
                text=text,
                reasoning="".join(reasoning_parts),
                details=details,
                finished=finished,
            )
            if isinstance(continuation, Present):
                # AT MOST ONE, after all contributing items are final, before
                # the terminal.
                yield ContinuationDelta(artifact=continuation.value)
            yield TerminalEvent(
                outcome=_terminal_outcome(
                    finish_reason=finish_reason,
                    meta=stream_meta(model),
                    intent=intent,
                    text=text,
                    tool_calls=tuple(call.tool_call for call in finished),
                    continuation=continuation,
                )
            )
        finally:
            if self._http_client is None:
                await client.close()

    # -- meta construction ---------------------------------------------------

    def _error_meta(
        self,
        row: ModelRow,
        exc: openai.APIStatusError,
        native_reasoning: Presence[str],
        started_ms: int,
    ) -> CallMeta:
        # No envelope decoded: the row's model id stands in, usage is Absent.
        return CallMeta(
            provider=row.provider,
            model=row.model_id,
            provider_request_id=presence_of(exc.request_id),
            upstream_provider=Absent(),
            usage=Absent(),
            attempt_trace=(
                AttemptRecord(
                    attempt=1,
                    signal=FinalAttempt(),
                    status_code=Present(exc.status_code),
                    started_at_ms=started_ms,
                    ended_at_ms=monotonic_ms(),
                ),
            ),
            billability=PossiblyBillable(),
            native_reasoning=native_reasoning,
            registry_revision=REGISTRY_REVISION,
        )

    def _decode_completion(
        self,
        provider: _Served,
        row: ModelRow,
        intent: GenerateIntent,
        completion: object,
        native_reasoning: Presence[str],
        started_ms: int,
    ) -> CallOutcome:
        if not isinstance(completion, ChatCompletion):
            raise ProtocolDefect(
                code="malformed_envelope",
                message=f"{provider} 2xx response is not a chat.completion envelope",
            )
        inband = (completion.model_extra or {}).get("error")
        if inband is not None:
            # HTTP 200 carrying an error object: the upstream failed after the
            # gateway accepted the request. Same classifier as the stream arm.
            raise TransientAttempt(
                cause=_classify_inband_error(provider, inband),
                status_code=Present(200),
                provider_request_id=presence_of(str_or_none(completion.id) or None),
                billability=PossiblyBillable(),
            )
        model = str_or_none(completion.model)
        if not model:
            raise ProtocolDefect(
                code="missing_model", message=f"{provider} response carries no model field"
            )
        if not completion.choices:
            raise ProtocolDefect(
                code="missing_choices", message=f"{provider} response has no choices[0] object"
            )
        choice = completion.choices[0]
        message = choice.message
        if not isinstance(message, ChatCompletionMessage):
            raise ProtocolDefect(
                code="missing_message", message=f"{provider} choice has no message object"
            )
        upstream: Presence[str] = (
            presence_of(str_or_none((completion.model_extra or {}).get("provider")))
            if provider == "openrouter"
            else Absent()
        )
        meta = CallMeta(
            provider=row.provider,
            model=model,
            provider_request_id=presence_of(str_or_none(completion.id) or None),
            upstream_provider=upstream,
            usage=(
                Present(_usage_from_raw(provider, completion.usage.to_dict()))
                if isinstance(completion.usage, CompletionUsage)
                else Absent()
            ),
            attempt_trace=(
                AttemptRecord(
                    attempt=1,
                    signal=FinalAttempt(),
                    status_code=Present(200),
                    started_at_ms=started_ms,
                    ended_at_ms=monotonic_ms(),
                ),
            ),
            billability=PossiblyBillable(),
            native_reasoning=native_reasoning,
            registry_revision=REGISTRY_REVISION,
        )
        tool_calls = _decode_tool_calls(message)
        if isinstance(tool_calls, InvalidToolArguments):
            return Failed(meta=meta, failure=tool_calls)
        return _terminal_outcome(
            finish_reason=str_or_none(choice.finish_reason),
            meta=meta,
            intent=intent,
            text=str_or_none(message.content) or "",
            tool_calls=tool_calls,
            continuation=_continuation_from_message(provider, row, intent, message),
        )
