"""OpenAI Responses engine — OpenAI proper on the openai SDK's native Responses API.

Wire obligations (spec §6, openai_responses row):

- ``store: false`` on every call plus ``include: ["reasoning.encrypted_content"]``
  so multi-turn reasoning replays statelessly. The ordered ``response.output``
  items ARE the replay state: each is captured verbatim (wire keys only —
  construct-time field sets hold exactly the wire keys, so exclude_unset dumps
  reproduce them) into ``ContinuationArtifact.opaque_payload["output"]`` and
  spliced back as input items on the next turn. Typed text/tool_calls are never
  re-synthesized alongside a continuation.
- Reasoning: the row's reasoning map value is a self-describing wire fragment
  (openai rows: ``{"reasoning": {"effort": "<level>"}}``) merged verbatim into
  the request — the engine carries no per-provider shape knowledge. The shared
  ``row_reasoning`` owns which levels are expressible and what the row's knob
  may write; this engine only merges the fragment and stamps the result.
- Tools: function tools with ``strict: true`` and a closed top-level schema
  (``additionalProperties: false``); ``tool_choice`` only alongside tools.
- Output: ``text.format`` ``json_schema`` (strict) on ``structured="native"``
  rows, ``json_object`` on ``json_mode`` rows. When the intent asked for strict
  JSON the terminal text must strictly parse to an object (StructuredContent);
  anything else is Failed(InvalidStructuredOutput) — no repair.
- provider_options: forwarded verbatim via ``extra_body``; a key the engine
  itself maps from core intent fields raises InvalidRequest.
- One attempt: retryable trouble raises TransientAttempt (429 → rate limit with
  retry-after, 5xx → unavailable, timeout/transport → timeout/unavailable,
  in-band stream errors and mid-stream cuts → interrupted); expected
  non-retryable failures return outcome values with full CallMeta; malformed
  envelopes (including malformed usage) raise ProtocolDefect; 401/403 raises
  CredentialRejected.
- Base URL: resolved by the engine (row value, else the canonical OpenAI host).
  The lane reads zero env vars, so the SDK's OPENAI_BASE_URL default is never
  allowed to decide where a call goes.
"""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, NoReturn, assert_never

import httpx
import openai
import pydantic
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseError,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from provider_runtime.engines import TransientAttempt
from provider_runtime.engines._common import (
    int_or_none,
    monotonic_ms,
    registry_invalid,
    response_content,
    row_reasoning,
    str_or_none,
    validate_continuation,
)
from provider_runtime.engines._openai_common import (
    CANONICAL_BASE_URL,
    terminal_http_failure,
    transient_connection,
    zero_env_client,
)
from provider_runtime.errors import (
    InvalidRequest,
    ProtocolDefect,
    safe_provider_error_body_snippet,
    sanitize_provider_text,
)
from provider_runtime.registry import REGISTRY_REVISION
from provider_runtime.registry import _ModelRow as ModelRow
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    AttemptRecord,
    CallMeta,
    CallOutcome,
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
    ProviderCredential,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTimeout,
    Refused,
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
    TransportUnavailable,
    UserMessage,
    presence_of,
)

# Request-body keys this engine maps from core intent fields. These plus every
# top-level key the row's reasoning knob can write form the provider_options
# collision set: naming one is an override, not an extension.
_OWNED_OPTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "input",
        "max_output_tokens",
        "store",
        "include",
        "tools",
        "tool_choice",
        "text",
        "stream",
    }
)

# Every request key this engine writes itself — the params literal, the
# conditional tools/text branches, and the kwargs added at the call sites.
# The reasoning fragment is splatted into that literal, so a row naming one of
# these AT ANY LEVEL either overwrites the engine's value or is silently
# overwritten by it depending on ordering; both are a poisoned row, not a
# request.
_ENGINE_SET_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "input",
        "max_output_tokens",
        "store",
        "include",
        "tools",
        "tool_choice",
        "text",
        "extra_body",
        "stream",
    }
)


# ---------------------------------------------------------------------------
# Encode


@dataclass(frozen=True, slots=True)
class _EncodedRequest:
    # Carries prompt text and continuation payloads — never in repr.
    params: dict[str, Any] = field(repr=False)
    native_reasoning: Presence[str]


def _encode_request(row: ModelRow, intent: GenerateIntent) -> _EncodedRequest:
    reasoning = row_reasoning(row, intent)
    fragment_collisions = sorted(_ENGINE_SET_REQUEST_KEYS & reasoning.owned_keys)
    if fragment_collisions:
        raise registry_invalid(
            row,
            f"reasoning fragment keys {fragment_collisions!r} would rewrite request fields "
            f"the engine sets itself",
        )
    collisions = sorted((_OWNED_OPTION_KEYS | reasoning.owned_keys) & set(intent.provider_options))
    if collisions:
        raise InvalidRequest(
            message=f"provider_options keys {collisions!r} collide with engine-mapped request fields"
        )
    params: dict[str, Any] = {
        "model": row.model_id,
        "input": _encode_input(row, intent),
        "max_output_tokens": intent.max_output_tokens,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        # The reasoning fragment merges verbatim — the engine never inspects
        # its shape; its keys collide with nothing this engine maps.
        **reasoning.fragment,
    }
    if intent.tools:
        params["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": {**tool.parameters, "additionalProperties": False},
                "strict": True,
            }
            for tool in intent.tools
        ]
        params["tool_choice"] = intent.tool_choice
    match intent.output:
        case StrictJsonOutput(name=name, schema=schema):
            match row.structured:
                case "native":
                    params["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": name,
                            "schema": dict(schema),
                            "strict": True,
                        }
                    }
                case "json_mode":
                    params["text"] = {"format": {"type": "json_object"}}
                case _:
                    assert_never(row.structured)
        case TextOutput():
            pass
        case _:
            assert_never(intent.output)
    if intent.provider_options:
        params["extra_body"] = dict(intent.provider_options)
    return _EncodedRequest(params=params, native_reasoning=reasoning.native_reasoning)


def _encode_input(row: ModelRow, intent: GenerateIntent) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for message in intent.messages:
        match message:
            case SystemMessage(blocks=blocks):
                items.append(
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": block.text} for block in blocks],
                    }
                )
            case UserMessage(blocks=blocks):
                items.append(
                    {"role": "user", "content": [_encode_user_block(block) for block in blocks]}
                )
            case AssistantMessage(text=text, tool_calls=tool_calls, continuation=continuation):
                match continuation:
                    case Present(value=artifact):
                        validate_continuation(artifact, row, intent)
                        items.extend(_artifact_input_items(artifact))
                    case Absent():
                        if tool_calls:
                            raise InvalidRequest(
                                message=(
                                    "openai Responses cannot replay an assistant tool turn "
                                    "without its continuation artifact; typed tool_calls alone "
                                    "cannot reconstruct the required ordered "
                                    "function_call/reasoning items"
                                )
                            )
                        items.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text}],
                            }
                        )
                    case _:
                        assert_never(continuation)
            case ToolResultMessage(call_id=call_id, output=output):
                # The Responses wire has no error flag; is_error is not encodable.
                items.append({"type": "function_call_output", "call_id": call_id, "output": output})
            case _:
                assert_never(message)
    return items


def _encode_user_block(block: PromptBlock | ImageBlock) -> dict[str, object]:
    match block:
        case PromptBlock(text=text):
            return {"type": "input_text", "text": text}
        case ImageBlock(media_type=media_type, data=data):
            encoded = b64encode(data).decode("ascii")
            return {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{encoded}",
                "detail": "auto",
            }
        case _:
            assert_never(block)


def _artifact_input_items(artifact: ContinuationArtifact) -> list[dict[str, object]]:
    """Splice the payload's ordered output items back verbatim — never parsed."""
    items = artifact.opaque_payload.get("output")
    if (
        not isinstance(items, Sequence)
        or isinstance(items, str | bytes)
        or not items
        or not all(isinstance(item, Mapping) for item in items)
    ):
        raise InvalidRequest(
            message=(
                "openai continuation artifact payload must carry the prior turn's "
                "complete ordered response.output item list under 'output'"
            )
        )
    return [dict(item) for item in items]


# ---------------------------------------------------------------------------
# Meta


def _meta(
    *,
    model: str,
    request_id: str | None,
    usage: Presence[TokenUsage],
    native_reasoning: Presence[str],
    started_ms: int,
    status_code: Presence[int],
) -> CallMeta:
    # Every engine-constructed outcome follows a dispatched request; OpenAI
    # never confirms non-billing, so billability is always PossiblyBillable.
    return CallMeta(
        provider="openai",
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
                ended_at_ms=monotonic_ms(),
            ),
        ),
        billability=PossiblyBillable(),
        native_reasoning=native_reasoning,
        registry_revision=REGISTRY_REVISION,
    )


# ---------------------------------------------------------------------------
# Decode


def _decode_usage(raw: object) -> Presence[TokenUsage]:
    if not isinstance(raw, ResponseUsage):
        return Absent()
    input_details = raw.input_tokens_details
    output_details = raw.output_tokens_details
    cached = input_details.cached_tokens if isinstance(input_details, InputTokensDetails) else None
    cache_write = (
        input_details.cache_write_tokens if isinstance(input_details, InputTokensDetails) else None
    )
    reasoning = (
        output_details.reasoning_tokens if isinstance(output_details, OutputTokensDetails) else None
    )
    input_tokens = int_or_none(raw.input_tokens)
    output_tokens = int_or_none(raw.output_tokens)
    if input_tokens is None or output_tokens is None:
        # Usage the SDK surfaced without integer prompt/completion counts;
        # zeroing them would keep the cache components while dropping the
        # inclusive input they are a subset of.
        raise ProtocolDefect(
            code="malformed_usage",
            message="openai usage carries no integer input/output token counts",
        )
    # OpenAI's input_tokens is already cache-inclusive — no normalization needed.
    try:
        usage = TokenUsage.from_components(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=presence_of(int_or_none(raw.total_tokens)),
            reasoning_tokens=presence_of(int_or_none(reasoning)),
            cache_read_input_tokens=presence_of(int_or_none(cached)),
            cache_write_input_tokens=presence_of(int_or_none(cache_write)),
        )
    except ValueError as error:
        raise ProtocolDefect(
            code="malformed_usage", message=f"openai usage is not valid token accounting: {error}"
        ) from error
    return Present(usage)


def _dump_items(items: Sequence[object], *, code: str) -> list[Mapping[str, object]]:
    """Verbatim wire capture: construct-time field sets hold exactly the wire
    keys, so exclude_unset dumps reproduce each item as it arrived."""
    dumped: list[Mapping[str, object]] = []
    for item in items:
        if not isinstance(item, pydantic.BaseModel):
            raise ProtocolDefect(code=code, message="openai output item is not a JSON object")
        dumped.append(item.model_dump(mode="json", exclude_unset=True))
    return dumped


def _continuation_of(
    row: ModelRow, intent: GenerateIntent, items: Sequence[Mapping[str, object]]
) -> Presence[ContinuationArtifact]:
    if not items:
        return Absent()
    return Present(
        ContinuationArtifact(
            target=intent.target,
            codec_id=row.continuation_codec,
            opaque_payload={"output": tuple(items)},
        )
    )


def _parse_tool_call(
    item: ResponseFunctionToolCall, fallback_arguments: str = ""
) -> ToolCall | InvalidToolArguments:
    """Strict JSON parse only; NO repair. Parse failure is an expected model
    failure value the caller folds into Failed."""
    call_id = str_or_none(item.call_id) or str_or_none(item.id) or ""
    name = str_or_none(item.name) or ""
    arguments_raw = str_or_none(item.arguments)
    if arguments_raw is None:
        arguments_raw = fallback_arguments
    try:
        arguments = json.loads(arguments_raw)
    except json.JSONDecodeError as error:
        return InvalidToolArguments(
            safe_detail=(
                f"openai tool call {name!r} arguments are not valid JSON: "
                f"{sanitize_provider_text(str(error), limit=200)}"
            )
        )
    if not isinstance(arguments, Mapping):
        return InvalidToolArguments(
            safe_detail=(
                f"openai tool call {name!r} arguments are not a JSON object; "
                f"got {type(arguments).__name__}"
            )
        )
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _incomplete_reason(
    details: object,
) -> tuple[Literal["max_output_tokens", "content_filter_partial"], str]:
    """Map incomplete_details to (Incomplete.reason literal, native reason)."""
    reason = details.reason if isinstance(details, IncompleteDetails) else None
    native = str_or_none(reason) or ""
    if native == "max_output_tokens":
        return "max_output_tokens", native
    if native == "content_filter":
        return "content_filter_partial", native
    raise ProtocolDefect(
        code="unknown_incomplete_reason",
        message=f"openai incomplete response carries unknown reason {native!r}",
    )


def _decode_response(
    row: ModelRow,
    intent: GenerateIntent,
    response: Response,
    *,
    native_reasoning: Presence[str],
    started_ms: int,
) -> CallOutcome:
    if not isinstance(response, Response):
        # The SDK returns raw text for 2xx bodies served with a non-JSON
        # content type; that is a malformed envelope at this boundary.
        raise ProtocolDefect(
            code="unparseable_response",
            message="openai response body is not a JSON object envelope",
        )
    model = str_or_none(response.model)
    if model is None:
        raise ProtocolDefect(
            code="missing_model", message="openai response envelope is missing 'model'"
        )
    if not isinstance(response.output, list):
        raise ProtocolDefect(
            code="malformed_output", message="openai response 'output' is not a list of items"
        )
    dumped_items = _dump_items(response.output, code="malformed_output")
    request_id = str_or_none(response._request_id) or str_or_none(response.id)
    meta = _meta(
        model=model,
        request_id=request_id,
        usage=_decode_usage(response.usage),
        native_reasoning=native_reasoning,
        started_ms=started_ms,
        status_code=Present(200),
    )

    refusal = _collect_refusal(response.output)
    if refusal is not None:
        return Refused(meta=meta, safe_detail=sanitize_provider_text(refusal))

    status = response.status
    if status == "incomplete":
        reason, native = _incomplete_reason(response.incomplete_details)
        return Incomplete(
            meta=meta, reason=reason, status="provider_incomplete", safe_detail=Present(native)
        )
    if status != "completed":
        raise ProtocolDefect(
            code="unknown_response_status",
            message=f"openai response carries unknown terminal status {status!r}",
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for item in response.output:
        if isinstance(item, ResponseOutputMessage) and isinstance(item.content, list):
            for part in item.content:
                if isinstance(part, ResponseOutputText):
                    text_parts.append(str_or_none(part.text) or "")
        elif isinstance(item, ResponseFunctionToolCall):
            parsed = _parse_tool_call(item)
            if isinstance(parsed, InvalidToolArguments):
                return Failed(meta=meta, failure=parsed)
            tool_calls.append(parsed)
    content = response_content(intent, text="".join(text_parts), tool_calls=tuple(tool_calls))
    if isinstance(content, InvalidStructuredOutput):
        return Failed(meta=meta, failure=content)
    return Succeeded(
        meta=meta,
        response=ResponsePayload(
            content=content, continuation=_continuation_of(row, intent, dumped_items)
        ),
    )


def _collect_refusal(output_items: Sequence[object]) -> str | None:
    parts: list[str] = []
    for item in output_items:
        if not isinstance(item, ResponseOutputMessage) or not isinstance(item.content, list):
            continue
        for part in item.content:
            if isinstance(part, ResponseOutputRefusal):
                parts.append(str_or_none(part.refusal) or "")
    return "".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Stream decode state


@dataclass(slots=True)
class _OpenToolCall:
    call_id: str
    name: str
    arguments: str = ""


@dataclass(slots=True)
class _StreamState:
    request_id: str | None
    text_parts: list[str] = field(default_factory=list)
    refusal_parts: list[str] = field(default_factory=list)
    completed_items: list[Mapping[str, object]] = field(default_factory=list)
    open_calls: dict[str, _OpenToolCall] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    semantic_emitted: bool = False


def _register_tool_call(state: _StreamState, item: object) -> ToolCallStart | None:
    if not isinstance(item, ResponseFunctionToolCall):
        return None
    item_key = str_or_none(item.id) or ""
    call_id = str_or_none(item.call_id) or item_key
    name = str_or_none(item.name) or ""
    state.open_calls[item_key] = _OpenToolCall(call_id=call_id, name=name)
    return ToolCallStart(call_id=call_id, name=name)


def _accumulate_tool_arguments(
    state: _StreamState, frame: ResponseFunctionCallArgumentsDeltaEvent
) -> ToolCallDelta | None:
    item_key = str_or_none(frame.item_id) or ""
    open_call = state.open_calls.get(item_key)
    if open_call is None:
        return None
    delta = str_or_none(frame.delta) or ""
    if not delta:
        return None
    open_call.arguments += delta
    return ToolCallDelta(call_id=open_call.call_id, arguments_delta=delta)


def _finish_output_item(
    state: _StreamState, frame: ResponseOutputItemDoneEvent
) -> ToolCallDone | InvalidToolArguments | None:
    item = frame.item
    if not isinstance(item, pydantic.BaseModel):
        raise ProtocolDefect(
            code="malformed_stream_frame",
            message="openai response.output_item.done frame is missing 'item'",
        )
    state.completed_items.append(item.model_dump(mode="json", exclude_unset=True))
    if isinstance(item, ResponseFunctionToolCall):
        item_key = str_or_none(item.id) or ""
        open_call = state.open_calls.pop(item_key, None)
        parsed = _parse_tool_call(item, open_call.arguments if open_call else "")
        if isinstance(parsed, InvalidToolArguments):
            return parsed
        state.tool_calls.append(parsed)
        return ToolCallDone(tool_call=parsed)
    if isinstance(item, ResponseOutputMessage) and not state.refusal_parts:
        # Non-delta refusal path: a completed message item may carry the
        # refusal as a finished content part.
        for part in item.content or []:
            if isinstance(part, ResponseOutputRefusal):
                state.refusal_parts.append(str_or_none(part.refusal) or "")
    return None


def _terminal_events(
    row: ModelRow,
    intent: GenerateIntent,
    state: _StreamState,
    frame: ResponseCompletedEvent | ResponseIncompleteEvent,
    *,
    native_reasoning: Presence[str],
    started_ms: int,
    status_code: int,
) -> list[CodecStreamEvent]:
    envelope = frame.response
    if not isinstance(envelope, Response):
        raise ProtocolDefect(
            code="malformed_stream_frame",
            message=f"openai {frame.type} frame is missing 'response'",
        )
    model = str_or_none(envelope.model)
    if model is None:
        raise ProtocolDefect(
            code="missing_model", message=f"openai {frame.type} envelope is missing 'model'"
        )
    request_id = state.request_id or str_or_none(envelope.id)
    meta = _meta(
        model=model,
        request_id=request_id,
        usage=_decode_usage(envelope.usage),
        native_reasoning=native_reasoning,
        started_ms=started_ms,
        status_code=Present(status_code),
    )

    if isinstance(frame, ResponseIncompleteEvent):
        reason, native = _incomplete_reason(envelope.incomplete_details)
        return [
            TerminalEvent(
                outcome=Incomplete(
                    meta=meta,
                    reason=reason,
                    status="provider_incomplete",
                    safe_detail=Present(native),
                )
            )
        ]

    if state.refusal_parts:
        # A streamed refusal arrives as content parts on an HTTP-200 completed
        # stream; it folds into the four-kind stream terminal grammar as
        # Incomplete(status="refused") — partial output is discarded
        # downstream, so no ContinuationDelta is emitted.
        return [
            TerminalEvent(
                outcome=Incomplete(
                    meta=meta,
                    reason="content_filter_partial",
                    status="refused",
                    safe_detail=Present(sanitize_provider_text("".join(state.refusal_parts))),
                )
            )
        ]

    content = response_content(
        intent, text="".join(state.text_parts), tool_calls=tuple(state.tool_calls)
    )
    if isinstance(content, InvalidStructuredOutput):
        return [TerminalEvent(outcome=Failed(meta=meta, failure=content))]
    continuation = _continuation_of(row, intent, state.completed_items)
    events: list[CodecStreamEvent] = []
    match continuation:
        case Present(value=artifact):
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


def _raise_stream_error(
    *, code: str | None, message: str | None, state: _StreamState, status_code: int
) -> NoReturn:
    if code == "rate_limit_exceeded":
        raise TransientAttempt(
            cause=ProviderRateLimit(retry_after=Absent()),
            status_code=Present(status_code),
            provider_request_id=presence_of(state.request_id),
            billability=PossiblyBillable(),
        )
    if code == "server_error":
        raise TransientAttempt(
            cause=ProviderHttpUnavailable(),
            status_code=Present(status_code),
            provider_request_id=presence_of(state.request_id),
            billability=PossiblyBillable(),
        )
    snippet = safe_provider_error_body_snippet({"error": {"code": code, "message": message}})
    raise ProtocolDefect(
        code="provider_stream_failure",
        message=f"openai stream reported a terminal provider error: {snippet or code or '?'}",
    )


def _raise_sdk_stream_error(
    error: openai.APIError, *, state: _StreamState, status_code: int
) -> NoReturn:
    """The SDK decodes an in-band error itself for any frame carrying a
    top-level ``error`` key (gateway/proxy shape) and raises APIError before
    our own frame dispatch sees it; classify it exactly like our frames."""
    body = error.body if isinstance(error.body, Mapping) else {}
    _raise_stream_error(
        code=str_or_none(body.get("code")),
        message=str_or_none(body.get("message")) or sanitize_provider_text(error.message),
        state=state,
        status_code=status_code,
    )


# ---------------------------------------------------------------------------
# Engine


class OpenAIResponsesEngine:
    """One attempt against OpenAI proper; the runtime owns retries and the envelope."""

    def __init__(
        self, *, timeout_s: float = 600.0, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self._timeout_s = timeout_s
        self._http_client = http_client

    def _client_for(self, row: ModelRow, credential: ProviderCredential) -> openai.AsyncOpenAI:
        # An Absent row base_url resolves here, never in the SDK (which would
        # consult OPENAI_BASE_URL).
        base_url = row.base_url.value if isinstance(row.base_url, Present) else CANONICAL_BASE_URL
        return zero_env_client(
            api_key=credential.key,
            base_url=base_url,
            timeout_s=self._timeout_s,
            http_client=self._http_client,
        )

    async def generate(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> CallOutcome:
        encoded = _encode_request(row, intent)
        started_ms = monotonic_ms()
        client = self._client_for(row, credential)
        try:
            try:
                response = await client.responses.create(**encoded.params)
            except openai.APIStatusError as error:
                overflow = terminal_http_failure(error)
                return Failed(
                    meta=_meta(
                        model=row.model_id,
                        request_id=error.request_id,
                        usage=Absent(),
                        native_reasoning=encoded.native_reasoning,
                        started_ms=started_ms,
                        status_code=Present(error.status_code),
                    ),
                    failure=overflow,
                )
            except openai.APIConnectionError as error:
                raise transient_connection(error) from error
            except json.JSONDecodeError as error:
                raise ProtocolDefect(
                    code="unparseable_response",
                    message=f"openai response body is not valid JSON: {error}",
                ) from error
        finally:
            if self._http_client is None:
                await client.close()
        return _decode_response(
            row, intent, response, native_reasoning=encoded.native_reasoning, started_ms=started_ms
        )

    async def stream(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> AsyncIterator[CodecStreamEvent]:
        encoded = _encode_request(row, intent)
        started_ms = monotonic_ms()
        client = self._client_for(row, credential)
        events: openai.AsyncStream[Any] | None = None
        try:
            try:
                events = await client.responses.create(stream=True, **encoded.params)
            except openai.APIStatusError as error:
                overflow = terminal_http_failure(error)
                # Terminal without StreamStart: the provider never accepted.
                yield TerminalEvent(
                    outcome=Failed(
                        meta=_meta(
                            model=row.model_id,
                            request_id=error.request_id,
                            usage=Absent(),
                            native_reasoning=encoded.native_reasoning,
                            started_ms=started_ms,
                            status_code=Present(error.status_code),
                        ),
                        failure=overflow,
                    )
                )
                return
            except openai.APIConnectionError as error:
                raise transient_connection(error) from error

            status_code = events.response.status_code
            state = _StreamState(request_id=events.response.headers.get("x-request-id"))
            yield StreamStart()
            try:
                async for raw in events:
                    if isinstance(raw, ResponseCreatedEvent | ResponseInProgressEvent):
                        if state.request_id is None and isinstance(raw.response, Response):
                            state.request_id = str_or_none(raw.response.id)
                    elif isinstance(raw, ResponseTextDeltaEvent):
                        delta = str_or_none(raw.delta) or ""
                        if delta:
                            state.text_parts.append(delta)
                            state.semantic_emitted = True
                            yield TextDelta(text=delta)
                    elif isinstance(raw, ResponseRefusalDeltaEvent):
                        state.refusal_parts.append(str_or_none(raw.delta) or "")
                    elif isinstance(raw, ResponseOutputItemAddedEvent):
                        start = _register_tool_call(state, raw.item)
                        if start is not None:
                            state.semantic_emitted = True
                            yield start
                    elif isinstance(raw, ResponseFunctionCallArgumentsDeltaEvent):
                        arguments_delta = _accumulate_tool_arguments(state, raw)
                        if arguments_delta is not None:
                            state.semantic_emitted = True
                            yield arguments_delta
                    elif isinstance(raw, ResponseOutputItemDoneEvent):
                        done = _finish_output_item(state, raw)
                        if isinstance(done, InvalidToolArguments):
                            # Expected model failure after tool deltas: always
                            # terminal, never retried.
                            yield TerminalEvent(
                                outcome=Failed(
                                    meta=_meta(
                                        model=row.model_id,
                                        request_id=state.request_id,
                                        usage=Absent(),
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
                    elif isinstance(raw, ResponseCompletedEvent | ResponseIncompleteEvent):
                        for event in _terminal_events(
                            row,
                            intent,
                            state,
                            raw,
                            native_reasoning=encoded.native_reasoning,
                            started_ms=started_ms,
                            status_code=status_code,
                        ):
                            yield event
                        return
                    elif isinstance(raw, ResponseFailedEvent):
                        error = raw.response.error if isinstance(raw.response, Response) else None
                        _raise_stream_error(
                            code=error.code if isinstance(error, ResponseError) else None,
                            message=error.message if isinstance(error, ResponseError) else None,
                            state=state,
                            status_code=status_code,
                        )
                    elif isinstance(raw, ResponseErrorEvent):
                        _raise_stream_error(
                            code=str_or_none(raw.code),
                            message=str_or_none(raw.message),
                            state=state,
                            status_code=status_code,
                        )
                    # Other event kinds (content_part lifecycles, *.done text
                    # frames, reasoning summaries, ...) carry nothing we consume.
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
                    code="malformed_stream_frame",
                    message=f"openai stream frame is not valid JSON: {error}",
                ) from error
            except openai.APIError as error:
                _raise_sdk_stream_error(error, state=state, status_code=status_code)
            # The SSE source ended without a terminal frame: a mid-stream cut.
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
