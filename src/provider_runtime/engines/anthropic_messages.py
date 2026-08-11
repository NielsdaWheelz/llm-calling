"""Anthropic Messages engine — anthropic proper on the anthropic SDK's native Messages API.

Wire obligations (spec §5/§6, anthropic_messages row):

- Caching: ONE inferred ``cache_control: {"type": "ephemeral"}`` breakpoint at
  the end of the conversation prefix — the final (new) turn excluded, falling
  back to the last system block on a first turn. Thinking blocks cannot carry
  the marker on this wire, so it lands on the last block that can. No caller
  annotation, no other markers.
- Reasoning: the row's reasoning map value is a self-describing wire fragment
  (anthropic rows: ``{"output_config": {"effort": "<level>"}}``) merged
  verbatim into the request — the engine carries no per-provider shape
  knowledge. ``native_reasoning`` records the fragment as compact sorted-keys
  JSON; a row without a reasoning knob (or an empty fragment) sends nothing
  and records ``Absent``.
- Continuations: the prior turn's thinking/redacted_thinking blocks (signatures
  intact) are captured verbatim into ``opaque_payload["blocks"]`` and replayed
  as the LEADING assistant content on the next turn — required for
  tool-use-with-thinking continuity. The payload is never parsed.
- Tools: ``input_schema`` closed with ``additionalProperties: false``;
  ``tool_choice`` only alongside tools.
- Output: SDK 0.121.0 wire fact — GA structured output is spelled
  ``output_config: {"format": {"type": "json_schema", "schema": ...}}``
  (``OutputConfigParam``), not ``output_format``; sent on ``structured="native"``
  rows only. ``json_mode`` rows have no Anthropic wire knob: nothing is sent
  and json_out validation owns the schema.
- Usage: Anthropic's wire ``input_tokens`` EXCLUDES cache components;
  ``TokenUsage.input_tokens`` is cache-INCLUSIVE, so ingress adds
  cache_read_input_tokens + cache_creation_input_tokens back in
  (cache_write = cache_creation_input_tokens).
- Refusals: pre-output refusal (reported usage with zero output tokens) is
  provider-confirmed unbilled → ConfirmedNonBillable; non-streamed refusal →
  Refused; streamed HTTP-200 stop_reason refusal → Incomplete(status="refused")
  with the partial output invalidated (no ContinuationDelta).
- One attempt: retryable trouble raises TransientAttempt (429 → rate limit with
  retry-after, 5xx/529/overloaded → unavailable, timeout/transport →
  timeout/unavailable, in-band overloaded/api_error stream events and mid-stream
  cuts → interrupted); expected non-retryable failures return outcome values
  with full CallMeta; malformed envelopes raise ProtocolDefect; 401/403 raises
  CredentialRejected.
"""

from __future__ import annotations

import json
import time
from base64 import b64encode
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NoReturn, assert_never

import anthropic
import httpx
from anthropic.types import (
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RedactedThinkingBlock,
    RefusalStopDetails,
    SignatureDelta,
    TextBlock,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types import (
    TextDelta as NativeTextDelta,
)
from anthropic.types.output_tokens_details import OutputTokensDetails

from provider_runtime.engines import TransientAttempt
from provider_runtime.errors import (
    CredentialRejected,
    InvalidRequest,
    ProtocolDefect,
    RuntimeDefect,
    safe_provider_error_body_snippet,
    sanitize_provider_text,
)
from provider_runtime.registry import REGISTRY_REVISION, ModelRow
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    AttemptRecord,
    Billability,
    CallMeta,
    CallOutcome,
    CodecStreamEvent,
    ConfirmedNonBillable,
    ContinuationArtifact,
    ContinuationDelta,
    Failed,
    FinalAttempt,
    GenerateIntent,
    ImageBlock,
    Incomplete,
    InvalidStructuredOutput,
    InvalidToolArguments,
    NotDispatched,
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
    Refused,
    ResponseContent,
    ResponsePayload,
    StreamStart,
    StrictJsonOutput,
    StructuredContent,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextContent,
    TextDelta,
    TextOutput,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallDone,
    ToolCallStart,
    ToolResultMessage,
    TransportUnavailable,
    UsageEvent,
    UserMessage,
    presence_of,
)

# Request-body keys this engine owns — mapped from core intent fields, or
# engine-inferred (cache_control, spec §5: no caller annotation). These plus
# the row's reasoning-fragment top-level keys form the provider_options
# collision set: naming one is an override, not an extension.
_OWNED_OPTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "max_tokens",
        "messages",
        "system",
        "cache_control",
        "tools",
        "tool_choice",
        "output_config",
        "stream",
    }
)

# Blocks that cannot carry cache_control on this wire.
_UNCACHEABLE_BLOCK_TYPES: Final[frozenset[str]] = frozenset({"thinking", "redacted_thinking"})

_SUCCESS_STOP_REASONS: Final[frozenset[str]] = frozenset({"end_turn", "tool_use", "stop_sequence"})
# 5xx-shaped in-band stream error types (overloaded_error ≙ 529); everything
# else in an error event is a ProtocolDefect.
_TRANSIENT_STREAM_ERROR_TYPES: Final[frozenset[str]] = frozenset({"overloaded_error", "api_error"})


def _monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def _int_or_none(value: object) -> int | None:
    # bool is an int subclass; token counts are never booleans.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Encode


@dataclass(frozen=True, slots=True)
class _EncodedRequest:
    # Carries prompt text and continuation payloads — never in repr.
    params: dict[str, Any] = field(repr=False)
    native_reasoning: Presence[str]


def _reasoning_fragment(
    row: ModelRow, intent: GenerateIntent
) -> tuple[dict[str, Any], Presence[str]]:
    """The row's self-describing reasoning wire fragment for the intent level,
    plus its ``native_reasoning`` stamp (compact sorted-keys JSON; Absent when
    nothing is sent)."""
    match row.reasoning:
        case Present(value=levels):
            if intent.reasoning not in levels:
                raise InvalidRequest(
                    message=f"reasoning level {intent.reasoning!r} is not declared for {row.ref!r}"
                )
            value = levels[intent.reasoning]
            if not isinstance(value, Mapping):
                raise RuntimeDefect(
                    origin="intent",
                    code="registry_invalid",
                    message=f"row {row.ref!r}: reasoning values must be request-fragment mappings",
                )
            fragment: dict[str, Any] = dict(value)
            if not fragment:
                return {}, Absent()
            return fragment, Present(json.dumps(fragment, sort_keys=True, separators=(",", ":")))
        case Absent():
            if intent.reasoning != "none":
                raise InvalidRequest(
                    message=f"row {row.ref!r} has no reasoning knob; "
                    f"level {intent.reasoning!r} is not expressible"
                )
            return {}, Absent()
        case _:
            assert_never(row.reasoning)


def _encode_request(row: ModelRow, intent: GenerateIntent) -> _EncodedRequest:
    fragment, native_reasoning = _reasoning_fragment(row, intent)
    collisions = sorted((_OWNED_OPTION_KEYS | fragment.keys()) & set(intent.provider_options))
    if collisions:
        raise InvalidRequest(
            message=f"provider_options keys {collisions!r} collide with engine-mapped request fields"
        )
    system_blocks, messages = _encode_messages(row, intent)
    params: dict[str, Any] = {
        "model": row.model_id,
        "max_tokens": intent.max_output_tokens,
        "messages": messages,
    }
    if system_blocks:
        params["system"] = system_blocks
    if intent.tools:
        params["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {**tool.parameters, "additionalProperties": False},
            }
            for tool in intent.tools
        ]
        params["tool_choice"] = {"type": intent.tool_choice}
    match intent.output:
        case StrictJsonOutput(schema=schema):
            match row.structured:
                case "native":
                    params["output_config"] = {
                        "format": {"type": "json_schema", "schema": dict(schema)}
                    }
                case "json_mode":
                    # No Anthropic wire knob; json_out validation owns the schema.
                    pass
                case _:
                    assert_never(row.structured)
        case TextOutput():
            pass
        case _:
            assert_never(intent.output)
    # The reasoning fragment merges verbatim. When the engine already maps the
    # same top-level key (output_config carries both the strict-output format
    # and the effort knob), the two mappings merge one level deep — still with
    # zero knowledge of the fragment's shape.
    for key, value in fragment.items():
        existing = params.get(key)
        params[key] = (
            {**existing, **value}
            if isinstance(existing, Mapping) and isinstance(value, Mapping)
            else value
        )
    if intent.provider_options:
        params["extra_body"] = dict(intent.provider_options)
    return _EncodedRequest(params=params, native_reasoning=native_reasoning)


def _encode_messages(
    row: ModelRow, intent: GenerateIntent
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    system_blocks: list[dict[str, object]] = []
    turns: list[tuple[str, list[dict[str, object]]]] = []
    pending_tool_results: list[dict[str, object]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            turns.append(("user", list(pending_tool_results)))
            pending_tool_results.clear()

    system_phase = True
    for message in intent.messages:
        match message:
            case SystemMessage(blocks=blocks):
                if not system_phase:
                    raise InvalidRequest(
                        message="anthropic system messages must precede all conversation turns "
                        "(the Messages API has one top-level system field)"
                    )
                # Anthropic rejects empty text blocks; drop them from the wire.
                system_blocks.extend(
                    {"type": "text", "text": block.text} for block in blocks if block.text
                )
            case UserMessage(blocks=blocks):
                system_phase = False
                flush_tool_results()
                content = [
                    _user_block_wire(block)
                    for block in blocks
                    if not (isinstance(block, PromptBlock) and not block.text)
                ]
                if not content:
                    raise InvalidRequest(
                        message="anthropic user turn has no non-empty content blocks to encode "
                        "(Anthropic rejects an empty content array)"
                    )
                turns.append(("user", content))
            case AssistantMessage():
                system_phase = False
                flush_tool_results()
                turns.append(("assistant", _assistant_content(message, row, intent)))
            case ToolResultMessage(call_id=call_id, output=output, is_error=is_error):
                system_phase = False
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": output,
                        "is_error": is_error,
                    }
                )
            case _:
                assert_never(message)
    flush_tool_results()
    # The one inferred cache breakpoint (spec §5): the end of the conversation
    # prefix, the final (new) turn excluded — falling back to the end of the
    # system prompt on a first turn. Thinking blocks cannot carry cache_control
    # on this wire, so the marker lands on the last block that can.
    prefix_blocks = [block for _, content in turns[:-1] for block in content]
    for block in reversed(prefix_blocks):
        if block.get("type") not in _UNCACHEABLE_BLOCK_TYPES:
            block["cache_control"] = {"type": "ephemeral"}
            break
    else:
        if system_blocks:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
    messages: list[dict[str, object]] = [
        {"role": role, "content": content} for role, content in turns
    ]
    return system_blocks, messages


def _user_block_wire(block: PromptBlock | ImageBlock) -> dict[str, object]:
    match block:
        case PromptBlock(text=text):
            return {"type": "text", "text": text}
        case ImageBlock(media_type=media_type, data=data):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64encode(data).decode("ascii"),
                },
            }
        case _:
            assert_never(block)


def _assistant_content(
    message: AssistantMessage, row: ModelRow, intent: GenerateIntent
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    match message.continuation:
        case Present(value=artifact):
            _validate_continuation(artifact, row, intent)
            # Thinking/redacted_thinking blocks lead the assistant turn VERBATIM.
            content.extend(_continuation_blocks(artifact))
        case Absent():
            pass
        case _:
            assert_never(message.continuation)
    if message.text:
        content.append({"type": "text", "text": message.text})
    for call in message.tool_calls:
        content.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": dict(call.arguments)}
        )
    if not content:
        # Anthropic rejects empty content arrays; silently dropping the turn
        # would misrepresent the conversation.
        raise InvalidRequest(
            message="anthropic assistant turn has no continuation blocks, text, "
            "or tool calls to encode"
        )
    return content


def _validate_continuation(
    artifact: ContinuationArtifact, row: ModelRow, intent: GenerateIntent
) -> None:
    if artifact.target != intent.target or artifact.codec_id != row.continuation_codec:
        raise InvalidRequest(
            message=(
                f"continuation artifact for {artifact.target.provider}/{artifact.target.model} "
                f"(codec {artifact.codec_id!r}) cannot replay to "
                f"{intent.target.provider}/{intent.target.model} "
                f"(codec {row.continuation_codec!r})"
            )
        )


def _continuation_blocks(artifact: ContinuationArtifact) -> list[dict[str, object]]:
    """Replay the payload's ordered thinking blocks verbatim — never parsed."""
    blocks = artifact.opaque_payload.get("blocks")
    if (
        not isinstance(blocks, Sequence)
        or isinstance(blocks, str | bytes)
        or not blocks
        or not all(isinstance(block, Mapping) for block in blocks)
    ):
        raise InvalidRequest(
            message="anthropic continuation artifact payload must carry the prior turn's "
            "ordered thinking blocks under 'blocks'"
        )
    return [dict(block) for block in blocks]


# ---------------------------------------------------------------------------
# Meta


def _meta(
    *,
    model: str,
    request_id: str | None,
    usage: Presence[TokenUsage],
    billability: Billability,
    native_reasoning: Presence[str],
    started_ms: int,
    status_code: Presence[int],
) -> CallMeta:
    return CallMeta(
        provider="anthropic",
        model=model,
        provider_request_id=presence_of(request_id),
        upstream_provider=Absent(),
        usage=usage,
        attempt_trace=(
            AttemptRecord(
                attempt=1,
                signal=FinalAttempt(),
                status_code=status_code,
                started_at_ms=started_ms,
                ended_at_ms=_monotonic_ms(),
            ),
        ),
        billability=billability,
        native_reasoning=native_reasoning,
        registry_revision=REGISTRY_REVISION,
    )


def _refusal_billability(usage: Presence[TokenUsage]) -> Billability:
    """Anthropic contract: a pre-output refusal is unbilled. The observable
    signal is reported usage with zero output tokens; anything else (billed
    partial output, or no reported usage) stays PossiblyBillable."""
    match usage:
        case Present(value=value) if value.output_tokens == 0:
            return ConfirmedNonBillable()
        case _:
            return PossiblyBillable()


def _refusal_detail(details: object) -> str:
    if isinstance(details, RefusalStopDetails):
        explanation = _str_or_none(details.explanation)
        if explanation:
            return sanitize_provider_text(explanation)
        category = _str_or_none(details.category)
        if category:
            return sanitize_provider_text(f"refusal category: {category}")
    return "provider refusal"


# ---------------------------------------------------------------------------
# SDK error classification (port of the old codec's classify_error)


def _terminal_http_failure(error: anthropic.APIStatusError) -> ProviderContextTooLarge:
    """Classify a non-2xx response.

    Returns the ONE expected terminal failure (context overflow); raises
    CredentialRejected / RuntimeDefect for terminal operator conditions and
    TransientAttempt for retryable ones.
    """
    status = error.status_code
    body = dict(error.body) if isinstance(error.body, Mapping) else None
    inner = body.get("error") if body else None
    message = (
        (_str_or_none(inner.get("message")) or "").lower() if isinstance(inner, Mapping) else ""
    )
    snippet = safe_provider_error_body_snippet(body) or f"HTTP {status}"

    if status == 429:
        raise TransientAttempt(
            cause=ProviderRateLimit(retry_after=_retry_after_seconds(error.response.headers)),
            status_code=Present(status),
            provider_request_id=presence_of(error.request_id),
            billability=PossiblyBillable(),
        )
    if status in (500, 502, 503, 504, 529) or error.type == "overloaded_error":
        raise TransientAttempt(
            cause=ProviderHttpUnavailable(),
            status_code=Present(status),
            provider_request_id=presence_of(error.request_id),
            billability=PossiblyBillable(),
        )
    if status == 413 or error.type == "request_too_large":
        return ProviderContextTooLarge()
    if error.type == "invalid_request_error" and "too long" in message:
        # e.g. "prompt is too long: N tokens > limit" — the documented
        # context-overflow shape on this provider (400, not 413).
        return ProviderContextTooLarge()
    if status in (401, 403):
        raise CredentialRejected(
            message=f"anthropic rejected the platform credential (HTTP {status}): {snippet}"
        )
    if status == 402 or error.type == "billing_error" or "credit balance is too low" in message:
        raise RuntimeDefect(
            origin="provider_http",
            code="quota_exhausted",
            message=f"anthropic reported exhausted quota/credit (HTTP {status}): {snippet}",
        )
    raise RuntimeDefect(
        origin="provider_http",
        code="unclassified_provider_error",
        message=f"anthropic returned an unclassified error (HTTP {status}): {snippet}",
    )


def _transient_connection(error: anthropic.APIConnectionError) -> TransientAttempt:
    if isinstance(error, anthropic.APITimeoutError):
        return TransientAttempt(
            cause=ProviderTimeout(),
            status_code=Absent(),
            provider_request_id=Absent(),
            billability=PossiblyBillable(),
        )
    # A pure pre-connect failure means no bytes reached the provider; every
    # other transport error implies the connection was at least opened.
    billability: Billability = (
        NotDispatched() if isinstance(error.__cause__, httpx.ConnectError) else PossiblyBillable()
    )
    return TransientAttempt(
        cause=TransportUnavailable(),
        status_code=Absent(),
        provider_request_id=Absent(),
        billability=billability,
    )


def _retry_after_seconds(headers: httpx.Headers) -> Presence[float]:
    raw = headers.get("retry-after")
    if raw is None:
        return Absent()
    try:
        seconds = float(raw)
    except ValueError:
        return Absent()
    return Present(seconds) if seconds >= 0 else Absent()


def _raise_inband_stream_error(
    error: anthropic.APIStatusError, *, status_code: int, request_id: str | None
) -> NoReturn:
    """An in-band error SSE frame on an HTTP-200 stream (the SDK surfaces it
    as APIStatusError mid-iteration)."""
    if error.type in _TRANSIENT_STREAM_ERROR_TYPES:
        raise TransientAttempt(
            cause=ProviderHttpUnavailable(),
            status_code=Present(status_code),
            provider_request_id=presence_of(request_id),
            billability=PossiblyBillable(),
        ) from error
    body = dict(error.body) if isinstance(error.body, Mapping) else None
    snippet = safe_provider_error_body_snippet(body)
    raise ProtocolDefect(
        code="stream_error_event",
        message=f"anthropic stream carried a non-transient error event: {snippet or error.type!r}",
    ) from error


# ---------------------------------------------------------------------------
# Usage — normalized ONCE at ingress; the fold spans message_start and
# message_delta frames on streams (one frame on generate).


@dataclass(slots=True)
class _UsageFold:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None
    thinking: int | None = None

    def absorb(self, frame: Usage | MessageDeltaUsage) -> None:
        self.input_tokens = _fold(self.input_tokens, frame.input_tokens)
        self.output_tokens = _fold(self.output_tokens, frame.output_tokens)
        self.cache_read = _fold(self.cache_read, frame.cache_read_input_tokens)
        self.cache_write = _fold(self.cache_write, frame.cache_creation_input_tokens)
        details = frame.output_tokens_details
        if isinstance(details, OutputTokensDetails):
            self.thinking = _fold(self.thinking, details.thinking_tokens)

    def presence(self) -> Presence[TokenUsage]:
        """Anthropic's wire input_tokens EXCLUDES cache components; normalize
        to the cache-INCLUSIVE TokenUsage.input_tokens invariant at ingress."""
        if self.input_tokens is None or self.output_tokens is None:
            return Absent()
        return Present(
            TokenUsage.from_components(
                input_tokens=self.input_tokens + (self.cache_read or 0) + (self.cache_write or 0),
                output_tokens=self.output_tokens,
                total_tokens=Absent(),
                reasoning_tokens=presence_of(self.thinking),
                cache_read_input_tokens=presence_of(self.cache_read),
                cache_write_input_tokens=presence_of(self.cache_write),
            )
        )


def _fold(current: int | None, incoming: object) -> int | None:
    value = _int_or_none(incoming)
    return value if value is not None else current


# ---------------------------------------------------------------------------
# Decode


def _continuation_of(
    row: ModelRow, intent: GenerateIntent, blocks: Sequence[Mapping[str, object]]
) -> Presence[ContinuationArtifact]:
    if not blocks:
        return Absent()
    return Present(
        ContinuationArtifact(
            target=intent.target,
            codec_id=row.continuation_codec,
            opaque_payload={"blocks": tuple(blocks)},
        )
    )


def _tool_call_of(block: ToolUseBlock) -> ToolCall | InvalidToolArguments:
    call_id = _str_or_none(block.id) or ""
    name = _str_or_none(block.name) or ""
    arguments = block.input
    if not isinstance(arguments, Mapping):
        return InvalidToolArguments(
            safe_detail=f"anthropic tool_use input for tool {name!r} (call {call_id}) "
            f"is not a JSON object"
        )
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _content_of(
    intent: GenerateIntent, *, text: str, tool_calls: tuple[ToolCall, ...]
) -> ResponseContent | InvalidStructuredOutput:
    """The output arm is the intent's OutputSpec, never re-inferred."""
    match intent.output:
        case TextOutput():
            return TextContent(text=text, tool_calls=tool_calls)
        case StrictJsonOutput():
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                return InvalidStructuredOutput(
                    safe_detail=f"structured output is not valid JSON "
                    f"({error.msg} at char {error.pos})"
                )
            if not isinstance(payload, dict):
                return InvalidStructuredOutput(
                    safe_detail=f"structured output parsed to {type(payload).__name__}, "
                    f"not a JSON object"
                )
            return StructuredContent(payload=payload, text=text)
        case _:
            assert_never(intent.output)


def _decode_response(
    row: ModelRow,
    intent: GenerateIntent,
    message: object,
    *,
    native_reasoning: Presence[str],
    started_ms: int,
) -> CallOutcome:
    # Typed `object` honestly: the SDK returns the raw text (a str) for 2xx
    # bodies served with a non-JSON content type in non-strict mode; that is a
    # malformed envelope at this boundary.
    if not isinstance(message, Message):
        raise ProtocolDefect(
            code="unparseable_response",
            message="anthropic response body is not a JSON message envelope",
        )
    model = _str_or_none(message.model)
    if model is None:
        raise ProtocolDefect(
            code="missing_model", message="anthropic response envelope is missing 'model'"
        )
    request_id = _str_or_none(message._request_id) or _str_or_none(message.id)
    fold = _UsageFold()
    if isinstance(message.usage, Usage):
        fold.absorb(message.usage)
    usage = fold.presence()
    stop_reason = message.stop_reason

    if stop_reason == "refusal":
        return Refused(
            meta=_meta(
                model=model,
                request_id=request_id,
                usage=usage,
                billability=_refusal_billability(usage),
                native_reasoning=native_reasoning,
                started_ms=started_ms,
                status_code=Present(200),
            ),
            safe_detail=_refusal_detail(message.stop_details),
        )

    meta = _meta(
        model=model,
        request_id=request_id,
        usage=usage,
        billability=PossiblyBillable(),
        native_reasoning=native_reasoning,
        started_ms=started_ms,
        status_code=Present(200),
    )

    if stop_reason == "max_tokens":
        return Incomplete(
            meta=meta,
            reason="max_output_tokens",
            status="provider_incomplete",
            safe_detail=Absent(),
        )
    if stop_reason == "model_context_window_exceeded":
        # Documented HTTP-200 terminal (SDK 0.121 StopReason): generation
        # crossed the model's context window — an expected failure value.
        return Failed(meta=meta, failure=ProviderContextTooLarge())
    if not isinstance(stop_reason, str) or stop_reason not in _SUCCESS_STOP_REASONS:
        # "pause_turn" lands here deliberately: it occurs only on server-tool
        # turns (web search etc.), and this wire never sends server tools —
        # CanonicalTool encodes client tools exclusively.
        raise ProtocolDefect(
            code="unknown_stop_reason",
            message=f"anthropic response carried unknown stop_reason {stop_reason!r}",
        )
    if not isinstance(message.content, list):
        raise ProtocolDefect(
            code="malformed_content",
            message="anthropic response 'content' is not a list of blocks",
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking_blocks: list[Mapping[str, object]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            text_parts.append(_str_or_none(block.text) or "")
        elif isinstance(block, ToolUseBlock):
            parsed = _tool_call_of(block)
            if isinstance(parsed, InvalidToolArguments):
                return Failed(meta=meta, failure=parsed)
            tool_calls.append(parsed)
        elif isinstance(block, ThinkingBlock | RedactedThinkingBlock):
            # Verbatim wire capture: construct-time field sets hold exactly the
            # wire keys, so exclude_unset dumps reproduce the block as it
            # arrived (signature intact).
            thinking_blocks.append(block.model_dump(mode="json", exclude_unset=True))
        else:
            raise ProtocolDefect(
                code="unknown_content_block",
                message="anthropic response carried unknown content block "
                f"{getattr(block, 'type', None)!r}",
            )

    content = _content_of(intent, text="".join(text_parts), tool_calls=tuple(tool_calls))
    if isinstance(content, InvalidStructuredOutput):
        return Failed(meta=meta, failure=content)
    continuation = _continuation_of(row, intent, thinking_blocks)
    return Succeeded(
        meta=meta, response=ResponsePayload(content=content, continuation=continuation)
    )


# ---------------------------------------------------------------------------
# Stream decode state


@dataclass(slots=True)
class _OpenToolCall:
    call_id: str
    name: str
    # Assembled model output — never in repr.
    arguments: str = field(default="", repr=False)


@dataclass(slots=True)
class _StreamState:
    request_id: str | None
    model: str | None = None
    in_band_id: str | None = None
    usage: _UsageFold = field(default_factory=_UsageFold)
    stop_reason: str | None = None
    stop_details: RefusalStopDetails | None = None
    # Assembled response text, tool arguments, and thinking blocks (the
    # continuation payload material) — never in repr.
    text_parts: list[str] = field(default_factory=list, repr=False)
    open_tools: dict[int, _OpenToolCall] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list, repr=False)
    thinking_by_index: dict[int, dict[str, object]] = field(default_factory=dict, repr=False)
    thinking_blocks: list[Mapping[str, object]] = field(default_factory=list, repr=False)
    semantic_emitted: bool = False


def _require_index(value: object) -> int:
    index = _int_or_none(value)
    if index is None:
        raise ProtocolDefect(
            code="malformed_stream_event",
            message="anthropic content_block event is missing the integer 'index'",
        )
    return index


def _start_content_block(
    state: _StreamState, frame: RawContentBlockStartEvent
) -> ToolCallStart | None:
    index = _require_index(frame.index)
    block = frame.content_block
    if isinstance(block, TextBlock):
        return None
    if isinstance(block, ToolUseBlock):
        call_id = _str_or_none(block.id) or ""
        name = _str_or_none(block.name) or ""
        state.open_tools[index] = _OpenToolCall(call_id=call_id, name=name)
        return ToolCallStart(call_id=call_id, name=name)
    if isinstance(block, ThinkingBlock | RedactedThinkingBlock):
        state.thinking_by_index[index] = block.model_dump(mode="json", exclude_unset=True)
        return None
    raise ProtocolDefect(
        code="unknown_content_block",
        message=f"anthropic stream carried unknown content block {getattr(block, 'type', None)!r}",
    )


def _absorb_delta(
    state: _StreamState, frame: RawContentBlockDeltaEvent
) -> TextDelta | ToolCallDelta | None:
    index = _require_index(frame.index)
    delta = frame.delta
    if isinstance(delta, NativeTextDelta):
        text = _str_or_none(delta.text) or ""
        state.text_parts.append(text)
        return TextDelta(text=text) if text else None
    if isinstance(delta, InputJSONDelta):
        open_tool = state.open_tools.get(index)
        if open_tool is None:
            raise ProtocolDefect(
                code="malformed_stream_event",
                message="anthropic input_json_delta arrived for an unknown tool block",
            )
        partial = _str_or_none(delta.partial_json) or ""
        open_tool.arguments += partial
        return (
            ToolCallDelta(call_id=open_tool.call_id, arguments_delta=partial) if partial else None
        )
    if isinstance(delta, ThinkingDelta | SignatureDelta):
        block = state.thinking_by_index.get(index)
        if block is None:
            raise ProtocolDefect(
                code="malformed_stream_event",
                message="anthropic thinking delta arrived for an unknown thinking block",
            )
        key, fragment = (
            ("thinking", delta.thinking)
            if isinstance(delta, ThinkingDelta)
            else ("signature", delta.signature)
        )
        existing = block.get(key)
        block[key] = (existing if isinstance(existing, str) else "") + (
            _str_or_none(fragment) or ""
        )
        return None
    raise ProtocolDefect(
        code="malformed_stream_event",
        message=f"anthropic stream carried unknown delta {getattr(delta, 'type', None)!r}",
    )


def _finish_content_block(
    state: _StreamState, index: int
) -> ToolCallDone | InvalidToolArguments | None:
    open_tool = state.open_tools.pop(index, None)
    if open_tool is not None:
        try:
            arguments = json.loads(open_tool.arguments or "{}")
        except json.JSONDecodeError:
            arguments = None
        if not isinstance(arguments, dict):
            # Strict JSON parse only; NO repair. An expected model failure the
            # caller folds into a Failed terminal.
            return InvalidToolArguments(
                safe_detail=f"anthropic tool_use arguments for tool {open_tool.name!r} "
                f"(call {open_tool.call_id}) did not parse as a JSON object"
            )
        tool_call = ToolCall(id=open_tool.call_id, name=open_tool.name, arguments=arguments)
        state.tool_calls.append(tool_call)
        return ToolCallDone(tool_call=tool_call)
    thinking = state.thinking_by_index.pop(index, None)
    if thinking is not None:
        state.thinking_blocks.append(thinking)
    return None


def _terminal_events(
    row: ModelRow,
    intent: GenerateIntent,
    state: _StreamState,
    *,
    native_reasoning: Presence[str],
    started_ms: int,
    status_code: int,
) -> list[CodecStreamEvent]:
    if state.model is None:
        raise ProtocolDefect(
            code="malformed_stream_event",
            message="anthropic stream reached message_stop without a message_start model",
        )
    usage = state.usage.presence()
    request_id = state.request_id or state.in_band_id

    if state.stop_reason == "refusal":
        # The four-kind stream terminal grammar has no Refused: an HTTP-200
        # stop_reason refusal terminates as incomplete+refused and the partial
        # output is invalidated (no ContinuationDelta).
        return [
            TerminalEvent(
                outcome=Incomplete(
                    meta=_meta(
                        model=state.model,
                        request_id=request_id,
                        usage=usage,
                        billability=_refusal_billability(usage),
                        native_reasoning=native_reasoning,
                        started_ms=started_ms,
                        status_code=Present(status_code),
                    ),
                    reason="content_filter_partial",
                    status="refused",
                    safe_detail=Present(_refusal_detail(state.stop_details)),
                )
            )
        ]

    meta = _meta(
        model=state.model,
        request_id=request_id,
        usage=usage,
        billability=PossiblyBillable(),
        native_reasoning=native_reasoning,
        started_ms=started_ms,
        status_code=Present(status_code),
    )

    if state.stop_reason == "max_tokens":
        return [
            TerminalEvent(
                outcome=Incomplete(
                    meta=meta,
                    reason="max_output_tokens",
                    status="provider_incomplete",
                    safe_detail=Absent(),
                )
            )
        ]
    if state.stop_reason == "model_context_window_exceeded":
        # Documented HTTP-200 terminal (SDK 0.121 StopReason): generation
        # crossed the model's context window — an expected failure value.
        return [TerminalEvent(outcome=Failed(meta=meta, failure=ProviderContextTooLarge()))]
    if state.stop_reason is None or state.stop_reason not in _SUCCESS_STOP_REASONS:
        # "pause_turn" lands here deliberately: it occurs only on server-tool
        # turns (web search etc.), and this wire never sends server tools —
        # CanonicalTool encodes client tools exclusively.
        raise ProtocolDefect(
            code="unknown_stop_reason",
            message=f"anthropic stream carried unknown stop_reason {state.stop_reason!r}",
        )

    content = _content_of(
        intent, text="".join(state.text_parts), tool_calls=tuple(state.tool_calls)
    )
    if isinstance(content, InvalidStructuredOutput):
        return [TerminalEvent(outcome=Failed(meta=meta, failure=content))]
    continuation = _continuation_of(row, intent, state.thinking_blocks)
    events: list[CodecStreamEvent] = []
    match continuation:
        case Present(value=artifact):
            # Exactly ONE ContinuationDelta, after all contributing blocks are
            # final and before the terminal.
            events.append(ContinuationDelta(artifact=artifact))
        case Absent():
            pass
        case _:
            assert_never(continuation)
    events.append(
        TerminalEvent(
            outcome=Succeeded(
                meta=meta, response=ResponsePayload(content=content, continuation=continuation)
            )
        )
    )
    return events


# ---------------------------------------------------------------------------
# Engine


class AnthropicMessagesEngine:
    """One attempt against Anthropic; the runtime owns retries and the envelope."""

    def __init__(
        self, *, timeout_s: float = 600.0, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self._timeout_s = timeout_s
        self._http_client = http_client

    def _client_for(
        self, row: ModelRow, credential: ProviderCredential
    ) -> anthropic.AsyncAnthropic:
        base_url = row.base_url.value if isinstance(row.base_url, Present) else None
        return anthropic.AsyncAnthropic(
            api_key=credential.key,
            base_url=base_url,
            timeout=self._timeout_s,
            max_retries=0,
            http_client=self._http_client,
        )

    async def generate(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> CallOutcome:
        encoded = _encode_request(row, intent)
        started_ms = _monotonic_ms()
        client = self._client_for(row, credential)
        try:
            try:
                message = await client.messages.create(**encoded.params)
            except anthropic.APIStatusError as error:
                overflow = _terminal_http_failure(error)
                return Failed(
                    meta=_meta(
                        model=row.model_id,
                        request_id=error.request_id,
                        usage=Absent(),
                        billability=PossiblyBillable(),
                        native_reasoning=encoded.native_reasoning,
                        started_ms=started_ms,
                        status_code=Present(error.status_code),
                    ),
                    failure=overflow,
                )
            except anthropic.APIConnectionError as error:
                raise _transient_connection(error) from error
            except json.JSONDecodeError as error:
                raise ProtocolDefect(
                    code="unparseable_response",
                    message=f"anthropic response body is not valid JSON: {error}",
                ) from error
        finally:
            if self._http_client is None:
                await client.close()
        return _decode_response(
            row, intent, message, native_reasoning=encoded.native_reasoning, started_ms=started_ms
        )

    async def stream(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> AsyncIterator[CodecStreamEvent]:
        encoded = _encode_request(row, intent)
        started_ms = _monotonic_ms()
        client = self._client_for(row, credential)
        events: anthropic.AsyncStream[Any] | None = None
        try:
            try:
                events = await client.messages.create(stream=True, **encoded.params)
            except anthropic.APIStatusError as error:
                overflow = _terminal_http_failure(error)
                # Terminal without StreamStart: the provider never accepted.
                yield TerminalEvent(
                    outcome=Failed(
                        meta=_meta(
                            model=row.model_id,
                            request_id=error.request_id,
                            usage=Absent(),
                            billability=PossiblyBillable(),
                            native_reasoning=encoded.native_reasoning,
                            started_ms=started_ms,
                            status_code=Present(error.status_code),
                        ),
                        failure=overflow,
                    )
                )
                return
            except anthropic.APIConnectionError as error:
                raise _transient_connection(error) from error

            status_code = events.response.status_code
            state = _StreamState(request_id=_str_or_none(events.response.headers.get("request-id")))
            yield StreamStart()
            try:
                async for raw in events:
                    if isinstance(raw, RawMessageStartEvent):
                        message = raw.message
                        if isinstance(message, Message):
                            state.model = _str_or_none(message.model) or state.model
                            state.in_band_id = _str_or_none(message.id)
                            if isinstance(message.usage, Usage):
                                state.usage.absorb(message.usage)
                    elif isinstance(raw, RawContentBlockStartEvent):
                        started = _start_content_block(state, raw)
                        if started is not None:
                            state.semantic_emitted = True
                            yield started
                    elif isinstance(raw, RawContentBlockDeltaEvent):
                        delta_event = _absorb_delta(state, raw)
                        if delta_event is not None:
                            state.semantic_emitted = True
                            yield delta_event
                    elif isinstance(raw, RawContentBlockStopEvent):
                        done = _finish_content_block(state, _require_index(raw.index))
                        if isinstance(done, InvalidToolArguments):
                            # Expected model failure after tool deltas: always
                            # terminal, never retried.
                            yield TerminalEvent(
                                outcome=Failed(
                                    meta=_meta(
                                        model=state.model or row.model_id,
                                        request_id=state.request_id or state.in_band_id,
                                        usage=state.usage.presence(),
                                        billability=PossiblyBillable(),
                                        native_reasoning=encoded.native_reasoning,
                                        started_ms=started_ms,
                                        status_code=Present(status_code),
                                    ),
                                    failure=done,
                                )
                            )
                            return
                        if done is not None:
                            state.semantic_emitted = True
                            yield done
                    elif isinstance(raw, RawMessageDeltaEvent):
                        delta = raw.delta
                        if delta is not None:
                            if isinstance(delta.stop_reason, str):
                                state.stop_reason = delta.stop_reason
                            if delta.stop_details is not None:
                                state.stop_details = delta.stop_details
                        if isinstance(raw.usage, MessageDeltaUsage):
                            state.usage.absorb(raw.usage)
                            folded = state.usage.presence()
                            if isinstance(folded, Present):
                                # Progressive telemetry, but still a
                                # retry-blocking semantic event.
                                state.semantic_emitted = True
                                yield UsageEvent(usage=folded.value)
                    elif isinstance(raw, RawMessageStopEvent):
                        for event in _terminal_events(
                            row,
                            intent,
                            state,
                            native_reasoning=encoded.native_reasoning,
                            started_ms=started_ms,
                            status_code=status_code,
                        ):
                            yield event
                        return
                    # Unknown event kinds are ignored (forward-compatible per
                    # Anthropic's streaming contract).
            except anthropic.APIStatusError as error:
                _raise_inband_stream_error(
                    error, status_code=status_code, request_id=state.request_id
                )
            except httpx.TimeoutException as error:
                raise TransientAttempt(
                    cause=ProviderTimeout(),
                    status_code=Present(status_code),
                    provider_request_id=presence_of(state.request_id),
                    billability=PossiblyBillable(),
                ) from error
            except httpx.TransportError as error:
                raise TransientAttempt(
                    cause=TransportUnavailable(),
                    status_code=Present(status_code),
                    provider_request_id=presence_of(state.request_id),
                    billability=PossiblyBillable(),
                ) from error
            except json.JSONDecodeError as error:
                raise ProtocolDefect(
                    code="malformed_stream_event",
                    message=f"anthropic stream frame is not valid JSON: {error}",
                ) from error
            # The SSE source ended without message_stop: a mid-stream cut.
            raise TransientAttempt(
                cause=ProviderStreamInterrupted(partial_output=state.semantic_emitted),
                status_code=Present(status_code),
                provider_request_id=presence_of(state.request_id),
                billability=PossiblyBillable(),
            )
        finally:
            if events is not None:
                await events.close()
            if self._http_client is None:
                await client.close()
