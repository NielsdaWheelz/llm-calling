"""Moonshot Kimi direct codec (Chat Completions wire, `moonshot_chat`).

Wire facts (provider-facts.md): POST https://api.moonshot.ai/v1/chat/completions;
`max_completion_tokens` (NOT the deprecated `max_tokens`); top-level
`reasoning_effort` low|high|max; thinking + Preserved Thinking always on;
caching fully automatic (no wire fields — `finalize` is passthrough);
continuation = the COMPLETE prior assistant message replayed as-is, including
`reasoning_content` and `tool_calls`; NO sampling fields.

Non-stream usage is the top-level `usage` object (flat `cached_tokens` = cache
read). Streamed usage arrives on `choices[0].usage` of the finish_reason chunk;
`stream_options.include_usage` is sent belt-and-braces, so a trailing
top-level-usage chunk with empty choices is tolerated and folded too — the
terminal outcome folds whichever frames were present, preferring the most
complete field-wise picture.

Codecs always decode message content as ``TextContent`` verbatim: the response
output arm is determined by the PLAN (which knows whether strict output was
requested), never re-inferred from the wire, so structured-content promotion is
a runtime concern.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Final

from provider_runtime import _chat_completions_wire as wire
from provider_runtime._signals import ClassifiedError, TransientStreamError
from provider_runtime.catalog import ChatModelContract
from provider_runtime.errors import (
    CredentialRejected,
    PlanningDefect,
    ProtocolDefect,
    RuntimeDefect,
    safe_provider_error_body_snippet,
)
from provider_runtime.transport import SseEvent
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    CallMeta,
    CallOutcome,
    CodecStreamEvent,
    ContinuationArtifact,
    ContinuationDelta,
    DraftRequest,
    FinalizedProviderRequest,
    GenerateIntent,
    Incomplete,
    PossiblyBillable,
    Presence,
    Present,
    ProviderContextTooLarge,
    ProviderHttpUnavailable,
    ProviderName,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTarget,
    ResponsePayload,
    StreamStart,
    StrictJsonOutput,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextContent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDone,
    ToolResultMessage,
    UsageEvent,
    UserMessage,
    presence_of,
)

CODEC_ID: Final[str] = "moonshot_chat"
PROTOCOL: Final = "moonshot_chat"

_PROVIDER: Final[ProviderName] = "moonshot"
_URL: Final[str] = "https://api.moonshot.ai/v1/chat/completions"
_STRICT_SCHEMA_DIALECT: Final[str] = "chat_completions_response_format_json_schema"


# ---------------------------------------------------------------------------
# encode / finalize / stream_request


def encode(intent: GenerateIntent, contract: ChatModelContract) -> DraftRequest:
    native_reasoning = _native_reasoning(intent, contract)
    tool_definitions = tuple(wire.tool_definition(tool) for tool in intent.tools)
    output_format = _output_format(intent, contract)

    body: dict[str, object] = {
        "model": contract.target.model,
        "messages": _encode_messages(intent),
        "max_completion_tokens": intent.max_output_tokens,
        "reasoning_effort": native_reasoning,
    }
    if intent.tools:
        body["tools"] = list(tool_definitions)
        body["tool_choice"] = intent.tool_choice
    if output_format is not None:
        body["response_format"] = output_format

    return DraftRequest(
        target=intent.target,
        protocol=PROTOCOL,
        url=_URL,
        safe_headers={},
        native_reasoning=native_reasoning,
        provider_framing_overhead_tokens=contract.provider_framing_overhead_tokens,
        prefix_bytes=wire.cache_prefix_bytes(
            wire.stable_prefix_messages(intent.messages),
            tool_definitions,
            intent.tool_choice if intent.tools else None,
            output_format,
        ),
        body=wire.dump_body(body),
    )


def finalize(draft: DraftRequest, affinity: str) -> FinalizedProviderRequest:
    """Passthrough: Moonshot caching is automatic — no affinity wire field.

    The affinity value is retained plan-side for fingerprint/telemetry only."""
    return FinalizedProviderRequest(
        target=draft.target,
        protocol=draft.protocol,
        url=draft.url,
        method="POST",
        safe_headers=draft.safe_headers,
        body=draft.body,
    )


def stream_request(request: FinalizedProviderRequest) -> FinalizedProviderRequest:
    """NEW value for the streaming variant: adds `stream` plus belt-and-braces
    `stream_options.include_usage` (usage also arrives on choices[0] of the
    finish chunk); re-dumped with the identical serialization settings."""
    body = dict(wire.parse_json_object(request.body, what="moonshot finalized request body"))
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    return FinalizedProviderRequest(
        target=request.target,
        protocol=request.protocol,
        url=request.url,
        method=request.method,
        safe_headers=request.safe_headers,
        body=wire.dump_body(body),
    )


def _native_reasoning(intent: GenerateIntent, contract: ChatModelContract) -> str:
    native = contract.reasoning.native_mapping.get(intent.reasoning)
    if native is None:
        raise PlanningDefect(
            code="unsupported_reasoning_level",
            message=(
                f"reasoning level {intent.reasoning!r} has no native mapping for "
                f"{contract.target.provider}/{contract.target.model}"
            ),
        )
    return native


def _output_format(intent: GenerateIntent, contract: ChatModelContract) -> dict[str, object] | None:
    if not isinstance(intent.output, StrictJsonOutput):
        return None
    if contract.strict_schema_dialect != _STRICT_SCHEMA_DIALECT:
        raise PlanningDefect(
            code="unsupported_schema_dialect",
            message=(
                f"moonshot codec compiles {_STRICT_SCHEMA_DIALECT!r}, contract declares "
                f"{contract.strict_schema_dialect!r}"
            ),
        )
    return wire.response_format_json_schema(intent.output)


def _encode_messages(intent: GenerateIntent) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for message in intent.messages:
        match message:
            case SystemMessage(blocks=blocks):
                messages.append(wire.system_message("".join(block.text for block in blocks)))
            case UserMessage(blocks=blocks):
                messages.append(wire.user_message("".join(block.text for block in blocks)))
            case AssistantMessage() as assistant:
                messages.append(_assistant_wire(assistant, intent.target))
            case ToolResultMessage(call_id=call_id, output=output):
                messages.append(wire.tool_result_message(call_id, output))
    return messages


def _assistant_wire(message: AssistantMessage, target: ProviderTarget) -> dict[str, object]:
    match message.continuation:
        case Present(value=artifact):
            _validate_artifact(artifact, target)
            # Complete-native-message replay: the payload IS the prior assistant
            # message exactly as Moonshot returned it (incl. reasoning_content
            # and tool_calls); typed fields are NOT re-encoded.
            return dict(artifact.opaque_payload)
        case _:
            return wire.assistant_message(message.text, message.tool_calls)


def _validate_artifact(artifact: ContinuationArtifact, target: ProviderTarget) -> None:
    if artifact.target != target or artifact.codec_id != CODEC_ID:
        raise PlanningDefect(
            code="continuation_mismatch",
            message=(
                f"continuation artifact for {artifact.target.provider}/{artifact.target.model} "
                f"(codec {artifact.codec_id!r}) cannot replay to {target.provider}/{target.model} "
                f"(codec {CODEC_ID!r})"
            ),
        )


# ---------------------------------------------------------------------------
# decode (non-stream)


def decode_response(status: int, headers: Mapping[str, str], body: bytes) -> CallOutcome:
    del status, headers  # 2xx only; Moonshot has no confirmed request-id header
    data = wire.parse_json_object(body, what="moonshot response")
    model = _required_model(data, what="moonshot response")
    choice = wire.first_choice(data, what="moonshot response")
    message = wire.choice_message(choice, what="moonshot response")
    tool_calls = wire.message_tool_calls(message)
    raw_usage = wire.top_level_usage(data)  # non-stream usage location: top-level
    meta = _meta(
        model=model,
        request_id=wire.str_or_none(data.get("id")),
        usage=Present(wire.parse_usage(raw_usage)) if raw_usage is not None else Absent(),
    )
    return _terminal_outcome(
        finish_reason=wire.finish_reason_of(choice),
        meta=meta,
        text=wire.message_text(message),
        tool_calls=tool_calls,
        continuation=_continuation_from_message(message, model),
    )


def _continuation_from_message(
    message: Mapping[str, object], model: str
) -> Presence[ContinuationArtifact]:
    # Replay material = preserved thinking and/or tool calls; the payload is the
    # complete native assistant message dict, verbatim.
    if wire.str_or_none(message.get("reasoning_content")) or message.get("tool_calls"):
        return Present(_artifact(dict(message), model))
    return Absent()


def _artifact(payload: Mapping[str, object], model: str) -> ContinuationArtifact:
    return ContinuationArtifact(
        target=ProviderTarget(provider=_PROVIDER, model=model),
        codec_id=CODEC_ID,
        opaque_payload=payload,
    )


def _required_model(data: Mapping[str, object], *, what: str) -> str:
    model = wire.str_or_none(data.get("model"))
    if not model:
        raise ProtocolDefect(code="missing_model", message=f"{what} carries no model field")
    return model


def _meta(*, model: str, request_id: str | None, usage: Presence[TokenUsage]) -> CallMeta:
    return CallMeta(
        provider=_PROVIDER,
        model=model,
        provider_request_id=presence_of(request_id),
        upstream_provider=Absent(),
        usage=usage,
        attempt_trace=(),
        billability=PossiblyBillable(),
    )


def _terminal_outcome(
    *,
    finish_reason: str | None,
    meta: CallMeta,
    text: str,
    tool_calls: tuple[ToolCall, ...],
    continuation: Presence[ContinuationArtifact],
) -> Succeeded | Incomplete:
    match finish_reason:
        case "stop" | "tool_calls":
            return Succeeded(
                meta=meta,
                response=ResponsePayload(
                    content=TextContent(text=text, tool_calls=tool_calls),
                    continuation=continuation,
                ),
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
                message=f"moonshot terminal finish_reason {finish_reason!r} is not recognized",
            )


# ---------------------------------------------------------------------------
# decode (stream)


async def decode_stream(
    headers: Mapping[str, str], events: AsyncIterator[SseEvent]
) -> AsyncIterator[CodecStreamEvent]:
    del headers  # Moonshot has no confirmed request-id header; the chunk id is used
    yield StreamStart()

    request_id: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    saw_done = False
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    accumulator = wire.ToolCallAccumulator()
    finished_calls: tuple[wire.FinishedToolCall, ...] = ()
    raw_usage: Mapping[str, object] | None = None

    async for event in events:
        if wire.is_done(event):
            saw_done = True
            break
        chunk = wire.parse_chunk(event, what="moonshot stream chunk")
        request_id = request_id or wire.str_or_none(chunk.get("id"))
        model = model or wire.str_or_none(chunk.get("model"))

        top_usage = wire.top_level_usage(chunk)
        choice = wire.chunk_choice(chunk, what="moonshot stream chunk")
        choice_usage = wire.choice_usage(choice) if choice is not None else None
        # Usage on choices[0] of the finish chunk is the real Moonshot shape;
        # the include_usage trailing top-level chunk (empty choices) is
        # tolerated — fold whichever is present.
        folded = wire.fold_raw_usage(wire.fold_raw_usage(raw_usage, choice_usage), top_usage)
        if folded is not None and folded is not raw_usage:
            raw_usage = folded
            yield UsageEvent(usage=wire.parse_usage(raw_usage))
        if choice is None:
            continue

        delta = wire.chunk_delta(choice)
        for tool_event in accumulator.apply(wire.delta_tool_calls(delta)):
            yield tool_event
        text = wire.delta_content(delta)
        if text:
            text_parts.append(text)
            yield TextDelta(text=text)
        reasoning = wire.delta_reasoning_content(delta)
        if reasoning:
            reasoning_parts.append(reasoning)

        chunk_finish = wire.finish_reason_of(choice)
        if chunk_finish is not None:
            finish_reason = chunk_finish
            finished_calls = accumulator.finish()
            for finished in finished_calls:
                yield ToolCallDone(tool_call=finished.tool_call)

    if not saw_done or finish_reason is None:
        raise TransientStreamError(ProviderStreamInterrupted(partial_output=False))
    if model is None:
        raise ProtocolDefect(code="missing_model", message="moonshot stream carried no model field")

    text = "".join(text_parts)
    tool_calls = tuple(finished.tool_call for finished in finished_calls)
    continuation = _stream_continuation(
        model=model,
        text=text,
        reasoning="".join(reasoning_parts),
        finished_calls=finished_calls,
    )
    if isinstance(continuation, Present):
        yield ContinuationDelta(artifact=continuation.value)

    meta = _meta(
        model=model,
        request_id=request_id,
        usage=Present(wire.parse_usage(raw_usage)) if raw_usage is not None else Absent(),
    )
    yield TerminalEvent(
        outcome=_terminal_outcome(
            finish_reason=finish_reason,
            meta=meta,
            text=text,
            tool_calls=tool_calls,
            continuation=continuation,
        )
    )


def _stream_continuation(
    *,
    model: str,
    text: str,
    reasoning: str,
    finished_calls: tuple[wire.FinishedToolCall, ...],
) -> Presence[ContinuationArtifact]:
    if not reasoning and not finished_calls:
        return Absent()
    # Reconstruct the complete native assistant message for as-is replay.
    payload: dict[str, object] = {
        "role": "assistant",
        "content": text if (text or not finished_calls) else None,
    }
    if reasoning:
        payload["reasoning_content"] = reasoning
    if finished_calls:
        payload["tool_calls"] = [dict(finished.native) for finished in finished_calls]
    return Present(_artifact(payload, model))


# ---------------------------------------------------------------------------
# classify_error (non-2xx only)


def classify_error(status: int, headers: Mapping[str, str], body: bytes) -> ClassifiedError:
    parsed = _tolerant_json(body)
    error = wire.mapping_or_none(parsed.get("error")) if parsed is not None else None
    snippet = safe_provider_error_body_snippet(dict(parsed) if parsed is not None else None, None)
    detail = f": {snippet}" if snippet else ""

    if status in (401, 403):
        raise CredentialRejected(
            message=f"moonshot rejected the platform credential (HTTP {status}){detail}"
        )
    if status == 402 or _is_quota(error):
        raise RuntimeDefect(
            origin="provider_http",
            code="quota_exhausted",
            message=f"moonshot reports exhausted quota/billing (HTTP {status}){detail}",
        )
    if status == 429:
        return ProviderRateLimit(retry_after=wire.retry_after_seconds(headers))
    if status in (500, 502, 503, 504):
        return ProviderHttpUnavailable()
    if _is_context_overflow(error):
        return ProviderContextTooLarge()
    if parsed is None:
        raise ProtocolDefect(
            code="unparseable_error_body",
            message=f"moonshot HTTP {status} error body is not a JSON object",
        )
    raise RuntimeDefect(
        origin="provider_http",
        code="unclassified_provider_error",
        message=f"moonshot returned unclassified HTTP {status}{detail}",
    )


def _tolerant_json(body: bytes) -> Mapping[str, object] | None:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return wire.mapping_or_none(data)


def _is_quota(error: Mapping[str, object] | None) -> bool:
    if error is None:
        return False
    kind = (wire.str_or_none(error.get("type")) or "") + (wire.str_or_none(error.get("code")) or "")
    return "quota" in kind


def _is_context_overflow(error: Mapping[str, object] | None) -> bool:
    if error is None:
        return False
    if wire.str_or_none(error.get("code")) == "context_length_exceeded":
        return True
    message = (wire.str_or_none(error.get("message")) or "").lower()
    return "context length" in message or "token limit" in message
