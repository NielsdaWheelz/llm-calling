"""OpenRouter operator-route codec (Chat Completions wire, `openrouter_chat`).

One dedicated pinned-upstream contract — never a provider-branching generic
client. Wire facts (provider-facts.md / spec §7): POST
https://openrouter.ai/api/v1/chat/completions; routed `max_tokens` (NOT the
direct-only `max_completion_tokens`); `reasoning {"effort", "exclude": false}`;
a full routing block pinning the upstream endpoint with fallbacks off;
`session_id` injected at finalize (sticky upstream affinity); protocol headers
`X-OpenRouter-Cache: false` (no response cache) and
`X-OpenRouter-Metadata: enabled` (upstream attribution source).

Continuation: the response `reasoning_details` array is preserved VERBATIM in
the artifact payload and replayed unmodified in sequence on the assistant
message; typed fields supply content/tool_calls.

upstream_provider is the provider display identity: the last attempt's
provider, else the selected endpoint's provider, else the top-level `provider`
field; Absent when none is present. Endpoint routing slugs remain request-side
facts and are never conflated with this observed identity. The generation id
(in-band `id`) is the provider_request_id.

Mid-stream in-band error chunks ({"error": {code, message}}, HTTP stays 200)
raise TransientStreamError — ProviderRateLimit when 429-shaped, otherwise
ProviderHttpUnavailable. A finish chunk carrying finish_reason "error" is
transient by the same contract even when the top-level error object is
absent or malformed. The runtime retry policy for this codec is
OPENROUTER_SINGLE_ATTEMPT (max_attempts=1, a planner concern): classification
stays exact here and the single-attempt budget is enforced upstream.

Codecs always decode message content as ``TextContent`` verbatim: the response
output arm is determined by the PLAN, never re-inferred from the wire.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Final

from provider_runtime import _chat_completions_wire as wire
from provider_runtime._signals import ClassifiedError, TransientStreamError
from provider_runtime.catalog import ChatModelContract, OpenRouterPrefixContract
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
    TransientCause,
    UsageEvent,
    UserMessage,
    presence_of,
)

CODEC_ID: Final[str] = "openrouter_chat"
PROTOCOL: Final = "openrouter_chat"

_PROVIDER: Final[ProviderName] = "openrouter"
_URL: Final[str] = "https://openrouter.ai/api/v1/chat/completions"
_STRICT_SCHEMA_DIALECT: Final[str] = "chat_completions_response_format_json_schema"

_SAFE_HEADERS: Final[Mapping[str, str]] = {
    "X-OpenRouter-Cache": "false",
    "X-OpenRouter-Metadata": "enabled",
}


# ---------------------------------------------------------------------------
# encode / finalize / stream_request


def encode(intent: GenerateIntent, contract: ChatModelContract) -> DraftRequest:
    native_reasoning = _native_reasoning(intent, contract)
    tool_definitions = tuple(wire.tool_definition(tool) for tool in intent.tools)
    output_format = _output_format(intent, contract)

    body: dict[str, object] = {
        "model": contract.target.model,
        "messages": _encode_messages(intent),
        "max_tokens": intent.max_output_tokens,
        "reasoning": {"effort": native_reasoning, "exclude": False},
    }
    if intent.tools:
        body["tools"] = list(tool_definitions)
        body["tool_choice"] = intent.tool_choice
    if output_format is not None:
        body["response_format"] = output_format
    body["provider"] = _provider_routing(contract)

    return DraftRequest(
        target=intent.target,
        protocol=PROTOCOL,
        url=_URL,
        safe_headers=dict(_SAFE_HEADERS),
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
    """NEW value injecting the derived affinity as top-level `session_id`
    (sticky upstream cache affinity, §8); the draft is never mutated."""
    body = dict(wire.parse_json_object(draft.body, what="openrouter draft body"))
    body["session_id"] = affinity
    return FinalizedProviderRequest(
        target=draft.target,
        protocol=draft.protocol,
        url=draft.url,
        method="POST",
        safe_headers=draft.safe_headers,
        body=wire.dump_body(body),
    )


def stream_request(request: FinalizedProviderRequest) -> FinalizedProviderRequest:
    """NEW value for the streaming variant: adds ONLY `stream: true`.

    stream_options.include_usage is deliberately NOT sent (deprecated no-op on
    OpenRouter; conflicts with provider.require_parameters=true) — the final
    chunk carries top-level usage."""
    body = dict(wire.parse_json_object(request.body, what="openrouter finalized request body"))
    body["stream"] = True
    return FinalizedProviderRequest(
        target=request.target,
        protocol=request.protocol,
        url=request.url,
        method=request.method,
        safe_headers=request.safe_headers,
        body=wire.dump_body(body),
    )


def _provider_routing(contract: ChatModelContract) -> dict[str, object]:
    cache = contract.cache
    if not isinstance(cache, OpenRouterPrefixContract):
        raise PlanningDefect(
            code="cache_contract_mismatch",
            message=(
                f"openrouter codec requires OpenRouterPrefixContract, contract for "
                f"{contract.target.provider}/{contract.target.model} declares "
                f"{type(cache).__name__}"
            ),
        )
    pinned = cache.pinned_upstream
    # quantizations is belt-and-braces: the "<provider>/<variant>" endpoint-slug
    # form for provider.only/order is UNCONFIRMED upstream (provider-facts.md),
    # so the pinned variant is also sent as an explicit quantization filter.
    variant = pinned.rpartition("/")[2]
    return {
        "only": [pinned],
        "order": [pinned],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "quantizations": [variant],
    }


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
                f"openrouter codec compiles {_STRICT_SCHEMA_DIALECT!r}, contract declares "
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
    # Typed fields supply content/tool_calls on this wire; the artifact only
    # carries the reasoning_details array, replayed verbatim in sequence.
    encoded = wire.assistant_message(message.text, message.tool_calls)
    match message.continuation:
        case Present(value=artifact):
            _validate_artifact(artifact, target)
            encoded["reasoning_details"] = [dict(detail) for detail in _payload_details(artifact)]
        case _:
            pass
    return encoded


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


def _payload_details(artifact: ContinuationArtifact) -> tuple[Mapping[str, object], ...]:
    details = wire.sequence_or_none(artifact.opaque_payload.get("reasoning_details"))
    if details is None:
        raise PlanningDefect(
            code="continuation_mismatch",
            message="openrouter continuation payload carries no reasoning_details array",
        )
    entries: list[Mapping[str, object]] = []
    for detail in details:
        entry = wire.mapping_or_none(detail)
        if entry is None:
            raise PlanningDefect(
                code="continuation_mismatch",
                message="openrouter continuation reasoning_details entry is not an object",
            )
        entries.append(entry)
    return tuple(entries)


# ---------------------------------------------------------------------------
# decode (non-stream)


def decode_response(status: int, headers: Mapping[str, str], body: bytes) -> CallOutcome:
    del status, headers  # 2xx only; the in-band generation id is the request id
    data = wire.parse_json_object(body, what="openrouter response")
    model = _required_model(data, what="openrouter response")
    choice = wire.first_choice(data, what="openrouter response")
    message = wire.choice_message(choice, what="openrouter response")
    tool_calls = wire.message_tool_calls(message)
    raw_usage = wire.top_level_usage(data)  # usage location: top-level only
    meta = _meta(
        model=model,
        request_id=wire.str_or_none(data.get("id")),
        upstream_provider=_upstream_provider(
            wire.mapping_or_none(data.get("openrouter_metadata")),
            wire.str_or_none(data.get("provider")),
        ),
        usage=Present(wire.parse_usage(raw_usage)) if raw_usage is not None else Absent(),
    )
    details = _message_reasoning_details(message)
    return _terminal_outcome(
        finish_reason=wire.finish_reason_of(choice),
        meta=meta,
        text=wire.message_text(message),
        tool_calls=tool_calls,
        continuation=_continuation_from_details(details, model),
    )


def _message_reasoning_details(message: Mapping[str, object]) -> tuple[object, ...]:
    details = wire.sequence_or_none(message.get("reasoning_details"))
    return tuple(details) if details is not None else ()


def _continuation_from_details(
    details: tuple[object, ...], model: str
) -> Presence[ContinuationArtifact]:
    if not details:
        return Absent()
    # Verbatim preservation: entries are stored exactly as received, in sequence.
    return Present(
        ContinuationArtifact(
            target=ProviderTarget(provider=_PROVIDER, model=model),
            codec_id=CODEC_ID,
            opaque_payload={"reasoning_details": list(details)},
        )
    )


def _upstream_provider(
    metadata: Mapping[str, object] | None, provider_field: str | None
) -> Presence[str]:
    """Pinned extraction order: openrouter_metadata attempts[-1].endpoint (the
    endpoint that served the response), else attempts[-1].provider, else the
    first entry of openrouter_metadata.endpoints, else the top-level `provider`
    field, else Absent."""
    if metadata is not None:
        slug = _upstream_from_metadata(metadata)
        if slug is not None:
            return Present(slug)
    if provider_field:
        return Present(provider_field)
    return Absent()


def _upstream_from_metadata(metadata: Mapping[str, object]) -> str | None:
    attempts = wire.sequence_or_none(metadata.get("attempts"))
    if attempts:
        last = wire.mapping_or_none(attempts[-1])
        if last is not None:
            provider = wire.str_or_none(last.get("provider"))
            if provider:
                return provider
    endpoints = wire.mapping_or_none(metadata.get("endpoints"))
    available = wire.sequence_or_none(endpoints.get("available")) if endpoints else None
    if available:
        for raw_entry in available:
            entry = wire.mapping_or_none(raw_entry)
            if entry is not None and entry.get("selected") is True:
                provider = wire.str_or_none(entry.get("provider"))
                if provider:
                    return provider
    return None


def _required_model(data: Mapping[str, object], *, what: str) -> str:
    model = wire.str_or_none(data.get("model"))
    if not model:
        raise ProtocolDefect(code="missing_model", message=f"{what} carries no model field")
    return model


def _meta(
    *,
    model: str,
    request_id: str | None,
    upstream_provider: Presence[str],
    usage: Presence[TokenUsage],
) -> CallMeta:
    return CallMeta(
        provider=_PROVIDER,
        model=model,
        provider_request_id=presence_of(request_id),
        upstream_provider=upstream_provider,
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
                message=f"openrouter terminal finish_reason {finish_reason!r} is not recognized",
            )


# ---------------------------------------------------------------------------
# decode (stream)


async def decode_stream(
    headers: Mapping[str, str], events: AsyncIterator[SseEvent]
) -> AsyncIterator[CodecStreamEvent]:
    del headers  # the in-band generation id is the request id
    yield StreamStart()

    request_id: str | None = None
    model: str | None = None
    provider_field: str | None = None
    metadata: Mapping[str, object] | None = None
    finish_reason: str | None = None
    saw_done = False
    text_parts: list[str] = []
    reasoning_details: list[object] = []
    accumulator = wire.ToolCallAccumulator()
    finished_calls: tuple[wire.FinishedToolCall, ...] = ()
    raw_usage: Mapping[str, object] | None = None

    async for event in events:
        if wire.is_done(event):
            saw_done = True
            break
        chunk = wire.parse_chunk(event, what="openrouter stream chunk")
        error = wire.mapping_or_none(chunk.get("error"))
        if error is not None:
            # In-band mid-stream error (HTTP stays 200) — transient by contract.
            raise TransientStreamError(_inband_error_cause(error))
        request_id = request_id or wire.str_or_none(chunk.get("id"))
        model = model or wire.str_or_none(chunk.get("model"))
        provider_field = wire.str_or_none(chunk.get("provider")) or provider_field
        metadata = wire.mapping_or_none(chunk.get("openrouter_metadata")) or metadata

        top_usage = wire.top_level_usage(chunk)  # final chunk carries usage top-level
        folded = wire.fold_raw_usage(raw_usage, top_usage)
        if folded is not None and folded is not raw_usage:
            raw_usage = folded
            yield UsageEvent(usage=wire.parse_usage(raw_usage))

        choice = wire.chunk_choice(chunk, what="openrouter stream chunk")
        if choice is None:
            continue
        delta = wire.chunk_delta(choice)
        for tool_event in accumulator.apply(wire.delta_tool_calls(delta)):
            yield tool_event
        text = wire.delta_content(delta)
        if text:
            text_parts.append(text)
            yield TextDelta(text=text)
        delta_details = wire.sequence_or_none(delta.get("reasoning_details"))
        if delta_details:
            reasoning_details.extend(delta_details)  # verbatim, in sequence

        chunk_finish = wire.finish_reason_of(choice)
        if chunk_finish is not None:
            if chunk_finish == "error":
                # Off-spec case: finish_reason "error" without (or with a
                # malformed) top-level error object. Keys the transient on
                # finish_reason per codec-seam.md line 37, and avoids
                # strict-parsing possibly-truncated tool arguments below.
                raise TransientStreamError(
                    _inband_error_cause(error) if error is not None else ProviderHttpUnavailable()
                )
            finish_reason = chunk_finish
            finished_calls = accumulator.finish()
            for finished in finished_calls:
                yield ToolCallDone(tool_call=finished.tool_call)

    if not saw_done or finish_reason is None:
        raise TransientStreamError(ProviderStreamInterrupted(partial_output=False))
    if model is None:
        raise ProtocolDefect(
            code="missing_model", message="openrouter stream carried no model field"
        )

    continuation = _continuation_from_details(tuple(reasoning_details), model)
    if isinstance(continuation, Present):
        yield ContinuationDelta(artifact=continuation.value)

    meta = _meta(
        model=model,
        request_id=request_id,
        upstream_provider=_upstream_provider(metadata, provider_field),
        usage=Present(wire.parse_usage(raw_usage)) if raw_usage is not None else Absent(),
    )
    yield TerminalEvent(
        outcome=_terminal_outcome(
            finish_reason=finish_reason,
            meta=meta,
            text="".join(text_parts),
            tool_calls=tuple(finished.tool_call for finished in finished_calls),
            continuation=continuation,
        )
    )


def _inband_error_cause(error: Mapping[str, object]) -> TransientCause:
    """429-shaped in-band errors → ProviderRateLimit; 5xx-shaped and unknown →
    ProviderHttpUnavailable (error.code mirrors HTTP status)."""
    code = wire.int_or_none(error.get("code"))
    if code is None:
        raw = wire.str_or_none(error.get("code"))
        if raw is not None and raw.isdigit():
            code = int(raw)
    if code == 429:
        return ProviderRateLimit(retry_after=Absent())
    return ProviderHttpUnavailable()


# ---------------------------------------------------------------------------
# classify_error (non-2xx only)


def classify_error(status: int, headers: Mapping[str, str], body: bytes) -> ClassifiedError:
    parsed = _tolerant_json(body)
    error = wire.mapping_or_none(parsed.get("error")) if parsed is not None else None
    detail = _error_detail(parsed, error)

    if status == 401:
        raise CredentialRejected(
            message=f"openrouter rejected the platform credential (HTTP {status}){detail}"
        )
    if status == 403:
        if _is_moderation_flagged(error):
            raise RuntimeDefect(
                origin="provider_response",
                code="input_moderation_flagged",
                message=f"openrouter flagged the input for moderation (HTTP {status}){detail}",
            )
        # No moderation metadata: conservative fallback to the platform-
        # credential classification (preserves any undocumented auth-shaped 403).
        raise CredentialRejected(
            message=f"openrouter rejected the platform credential (HTTP {status}){detail}"
        )
    if status == 402:
        raise RuntimeDefect(
            origin="provider_http",
            code="quota_exhausted",
            message=f"openrouter reports insufficient credits (HTTP {status}){detail}",
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
            message=f"openrouter HTTP {status} error body is not a JSON object",
        )
    raise RuntimeDefect(
        origin="provider_http",
        code="unclassified_provider_error",
        message=f"openrouter returned unclassified HTTP {status}{detail}",
    )


def _is_moderation_flagged(error: Mapping[str, object] | None) -> bool:
    """A 403 body carries moderation metadata (`reasons` array and/or
    `flagged_input`) — a per-request content failure, not a rejected
    platform credential."""
    if error is None:
        return False
    metadata = wire.mapping_or_none(error.get("metadata"))
    if metadata is None:
        return False
    return wire.sequence_or_none(metadata.get("reasons")) is not None or "flagged_input" in metadata


def _error_detail(parsed: Mapping[str, object] | None, error: Mapping[str, object] | None) -> str:
    snippet = safe_provider_error_body_snippet(dict(parsed) if parsed is not None else None, None)
    parts = [snippet] if snippet else []
    if error is not None:
        # Upstream (non-500) errors carry error.metadata.provider_code.
        error_metadata = wire.mapping_or_none(error.get("metadata"))
        if error_metadata is not None:
            provider_code = error_metadata.get("provider_code")
            if isinstance(provider_code, str | int):
                parts.append(f"upstream provider_code={provider_code}")
    return f": {'; '.join(parts)}" if parts else ""


def _tolerant_json(body: bytes) -> Mapping[str, object] | None:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return wire.mapping_or_none(data)


def _is_context_overflow(error: Mapping[str, object] | None) -> bool:
    if error is None:
        return False
    message = (wire.str_or_none(error.get("message")) or "").lower()
    return "context length" in message or "token limit" in message or "maximum context" in message
