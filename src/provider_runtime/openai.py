"""OpenAI Responses codec (spec §7 OpenAI row; .dossiers/codec-seam.md contract).

Wire shape (POST https://api.openai.com/v1/responses):

- ``input`` items: system blocks keep role ``"system"`` (the old codec's exact
  role mapping — never ``"developer"``) with ``input_text`` content parts; user
  blocks are ``input_text`` parts; a typed assistant turn without continuation
  material is one ``output_text`` message item; tool results are
  ``function_call_output`` items (the Responses wire has no error flag, so
  ``ToolResultMessage.is_error`` is intentionally not encoded).
- Reasoning: ``reasoning: {"effort": <native>}`` is sent for EVERY declared
  level with its exact native value from the catalog mapping — identity strings
  incl. distinct ``"xhigh"``/``"max"``; ``"none"`` IS sent as effort ``"none"``,
  nothing is ever omitted. ``store: false`` plus
  ``include: ["reasoning.encrypted_content"]`` keep reasoning replayable
  statelessly; no other reasoning fields (mode/context/summary) are sent.
- Caching (explicit prefix): top-level ``prompt_cache_options
  {"mode": "explicit", "ttl": "30m"}`` and a ``prompt_cache_breakpoint
  {"mode": "explicit"}`` marker on the LAST content block of the leading stable
  prefix (per provider-facts: breakpoints live on content blocks). The derived
  affinity is injected at :func:`finalize` as top-level ``prompt_cache_key``.
- Tools: function tools with inlined, annotated JSON Schema and
  ``strict: true``; ``tool_choice`` is sent only alongside ``tools``.
- Strict output: ``text.format`` ``json_schema`` (name, schema, strict:true) —
  dialect ``"openai_text_format_json_schema"``. Decode returns the output TEXT
  only (``TextContent``): the output arm is determined by the PLAN, and decode
  signatures carry no plan, so ``StructuredContent`` construction (strict parse
  of the text) belongs to the plan-owning runtime — this codec never infers it.
- Continuation: an ``AssistantMessage.continuation`` artifact's
  ``opaque_payload["output"]`` carries the COMPLETE ordered ``response.output``
  item list from the prior turn and is replayed verbatim as input items — the
  SOLE wire source for that turn (typed text/tool_calls are never
  re-synthesized). Decode collects the complete ordered output list into ONE
  :class:`~provider_runtime.types.ContinuationArtifact`.
- Streaming: the finalized body carries no ``stream`` flag;
  :func:`stream_request` derives the streaming variant (``"stream": true``)
  as a NEW value with identical deterministic serialization.

Transcription (transport-special multipart, NOT FinalizedProviderRequest JSON
bytes): :func:`build_transcription_request` /
:func:`parse_transcription_response` preserve the old codec's httpx-facing
multipart contract for the runtime's transcribe port.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, assert_never

from provider_runtime._signals import (
    ClassifiedError,
    ExpectedFailureSignal,
    TransientStreamError,
)
from provider_runtime.catalog import ChatModelContract, OpenAIExplicitPrefixContract
from provider_runtime.errors import (
    CredentialRejected,
    PlanningDefect,
    ProtocolDefect,
    RuntimeDefect,
    safe_provider_error_body_snippet,
    sanitize_provider_text,
)
from provider_runtime.schema import to_json_schema
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
    Dynamic,
    FinalizedProviderRequest,
    GenerateIntent,
    Incomplete,
    InvalidToolArguments,
    PossiblyBillable,
    Presence,
    Present,
    ProviderContextTooLarge,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTarget,
    Refused,
    ResponsePayload,
    Stable,
    StreamStart,
    StrictJsonOutput,
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
    UserMessage,
    presence_of,
)

CODEC_ID: Final[str] = "openai_responses"

_RESPONSES_URL: Final[str] = "https://api.openai.com/v1/responses"
_TRANSCRIPTIONS_URL: Final[str] = "https://api.openai.com/v1/audio/transcriptions"

# In-band stream/terminal error codes that signal a retryable provider-side
# condition (everything else in a failed/error frame is a ProtocolDefect).
_TRANSIENT_STREAM_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {"server_error", "rate_limit_exceeded"}
)


# ---------------------------------------------------------------------------
# Deterministic serialization (seam: identical settings EVERYWHERE)


def _dumps(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _dump_bytes(value: object) -> bytes:
    return _dumps(value).encode("utf-8")


def _frame(component: bytes) -> bytes:
    return len(component).to_bytes(8, "big") + component


# ---------------------------------------------------------------------------
# JSON narrowing helpers (codec-private; raw nullable provider JSON stays here)


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: object) -> int | None:
    # bool is an int subclass; token counts are never booleans.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _item_list(value: object) -> list[Mapping[str, object]] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return None
    items: list[Mapping[str, object]] = []
    for entry in value:
        mapping = _as_mapping(entry)
        if mapping is None:
            return None
        items.append(mapping)
    return items


# ---------------------------------------------------------------------------
# encode / finalize / stream_request


def encode(intent: GenerateIntent, contract: ChatModelContract) -> DraftRequest:
    if contract.protocol != "openai_responses":
        raise PlanningDefect(
            code="protocol_mismatch",
            message=(
                f"openai codec received contract protocol {contract.protocol!r}; "
                f"expected 'openai_responses'"
            ),
        )
    if intent.target != contract.target:
        raise PlanningDefect(
            code="target_contract_mismatch",
            message=(
                f"intent target {intent.target.provider}/{intent.target.model} does not "
                f"match contract target {contract.target.provider}/{contract.target.model}"
            ),
        )
    cache = contract.cache
    if not isinstance(cache, OpenAIExplicitPrefixContract):
        raise PlanningDefect(
            code="unsupported_cache_contract",
            message=(
                f"openai codec requires OpenAIExplicitPrefixContract; got {type(cache).__name__}"
            ),
        )
    native_reasoning = contract.reasoning.native_mapping.get(intent.reasoning)
    if native_reasoning is None:
        raise PlanningDefect(
            code="unsupported_reasoning_level",
            message=(
                f"reasoning level {intent.reasoning!r} is not declared for "
                f"{contract.target.provider}/{contract.target.model}"
            ),
        )

    input_items, stable_prefix_items = _encode_input(intent)

    body: dict[str, object] = {
        "model": intent.target.model,
        "input": input_items,
        "max_output_tokens": intent.max_output_tokens,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": native_reasoning},
        "prompt_cache_options": {"mode": "explicit", "ttl": cache.ttl},
    }

    tool_entries: list[dict[str, object]] = [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": to_json_schema(
                tool.parameters, inline_defs=True, include_annotations=True
            ),
            "strict": True,
        }
        for tool in intent.tools
    ]
    if tool_entries:
        body["tools"] = tool_entries
        body["tool_choice"] = intent.tool_choice

    text_format: dict[str, object] | None = None
    match intent.output:
        case StrictJsonOutput(name=name, schema=schema):
            text_format = {
                "type": "json_schema",
                "name": name,
                "schema": to_json_schema(schema, inline_defs=True, include_annotations=True),
                "strict": True,
            }
            body["text"] = {"format": text_format}
        case TextOutput():
            pass
        case _:
            assert_never(intent.output)

    # Cache-affinity prefix bytes: length-framed exact native serializations in
    # the seam's fixed order — stable-prefix ITEMS (each the enclosing
    # {role, content: [stable parts]} input item, so role/message boundary
    # participate, not bare content parts), tool definitions, tool_choice,
    # output format (OpenAI's cache prefix includes tools and the text.format
    # schema). Fixed section tags separate the four component classes so a
    # stable text byte-equal to a tool-definition dump can never collide
    # across classes. The injected prompt_cache_key is excluded by
    # construction (it does not exist pre-finalize).
    components: list[bytes] = [b"stable"]
    components.extend(_dump_bytes(item) for item in stable_prefix_items)
    components.append(b"tools")
    components.extend(_dump_bytes(entry) for entry in tool_entries)
    components.append(b"tool_choice")
    if tool_entries:
        components.append(_dump_bytes(intent.tool_choice))
    components.append(b"format")
    if text_format is not None:
        components.append(_dump_bytes(text_format))
    prefix_bytes = b"".join(_frame(component) for component in components)

    return DraftRequest(
        target=intent.target,
        protocol="openai_responses",
        url=_RESPONSES_URL,
        safe_headers={},
        native_reasoning=native_reasoning,
        provider_framing_overhead_tokens=contract.provider_framing_overhead_tokens,
        prefix_bytes=prefix_bytes,
        body=_dump_bytes(body),
    )


def _encode_input(
    intent: GenerateIntent,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build the ordered ``input`` item list and collect the leading stable
    prefix projected at message granularity: each contributing item is
    ``{"role": role, "content": [stable parts only]}`` (its exact native
    placement, so role/message-boundary participate in the cache-affinity
    projection, not bare content parts) — the last stable part is stamped
    with the explicit breakpoint marker before serialization."""
    items: list[dict[str, object]] = []
    stable_prefix_items: list[dict[str, object]] = []
    prefix_open = True

    for message in intent.messages:
        match message:
            case SystemMessage() | UserMessage():
                role = "system" if isinstance(message, SystemMessage) else "user"
                parts: list[dict[str, object]] = []
                stable_parts: list[dict[str, object]] = []
                for block in message.blocks:
                    part: dict[str, object] = {"type": "input_text", "text": block.text}
                    parts.append(part)
                    match block.stability:
                        case Stable():
                            if prefix_open:
                                stable_parts.append(part)
                        case Dynamic():
                            prefix_open = False
                        case _:
                            assert_never(block.stability)
                items.append({"role": role, "content": parts})
                if stable_parts:
                    stable_prefix_items.append({"role": role, "content": stable_parts})
            case AssistantMessage(text=text, tool_calls=tool_calls, continuation=continuation):
                prefix_open = False
                match continuation:
                    case Present(value=artifact):
                        _validate_continuation(artifact, intent)
                        items.extend(_artifact_output_items(artifact))
                    case Absent():
                        if tool_calls:
                            raise PlanningDefect(
                                code="continuation_required_for_tool_replay",
                                message=(
                                    "openai Responses cannot replay an assistant tool turn "
                                    "without its continuation artifact; typed tool_calls "
                                    "alone cannot reconstruct the required ordered "
                                    "function_call/reasoning items"
                                ),
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
                prefix_open = False
                items.append({"type": "function_call_output", "call_id": call_id, "output": output})
            case _:
                assert_never(message)

    if stable_prefix_items:
        last_item_content = stable_prefix_items[-1]["content"]
        assert isinstance(last_item_content, list)
        last_item_content[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return items, stable_prefix_items


def _validate_continuation(artifact: ContinuationArtifact, intent: GenerateIntent) -> None:
    if artifact.target != intent.target or artifact.codec_id != CODEC_ID:
        raise PlanningDefect(
            code="continuation_mismatch",
            message=(
                f"continuation artifact for {artifact.target.provider}/"
                f"{artifact.target.model} (codec {artifact.codec_id!r}) cannot replay to "
                f"{intent.target.provider}/{intent.target.model} (codec {CODEC_ID!r})"
            ),
        )


def _artifact_output_items(artifact: ContinuationArtifact) -> list[dict[str, object]]:
    items = _item_list(artifact.opaque_payload.get("output"))
    if items is None or not items:
        raise PlanningDefect(
            code="invalid_continuation_payload",
            message=(
                "openai continuation artifact payload must carry the prior turn's "
                "complete ordered response.output item list under 'output'"
            ),
        )
    return [dict(item) for item in items]


def finalize(draft: DraftRequest, affinity: str) -> FinalizedProviderRequest:
    """Inject ``prompt_cache_key=affinity`` into a NEW finalized request.

    The draft is never mutated; the key is appended after all draft-time keys
    and the body is re-dumped with the identical deterministic settings."""
    parsed: dict[str, object] = json.loads(draft.body)
    parsed["prompt_cache_key"] = affinity
    return FinalizedProviderRequest(
        target=draft.target,
        protocol=draft.protocol,
        url=draft.url,
        method="POST",
        safe_headers=draft.safe_headers,
        body=_dump_bytes(parsed),
    )


def stream_request(request: FinalizedProviderRequest) -> FinalizedProviderRequest:
    """Derive the streaming variant of a finalized call as a NEW value.

    Adds ``"stream": true`` and re-dumps with the identical deterministic
    settings; url/headers/method are unchanged. ``generate()`` uses the
    finalized request as-is (no ``stream`` field)."""
    parsed: dict[str, object] = json.loads(request.body)
    parsed["stream"] = True
    return FinalizedProviderRequest(
        target=request.target,
        protocol=request.protocol,
        url=request.url,
        method=request.method,
        safe_headers=request.safe_headers,
        body=_dump_bytes(parsed),
    )


# ---------------------------------------------------------------------------
# Shared decode pieces


def _parse_json_object(body: bytes) -> Mapping[str, object]:
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProtocolDefect(
            code="unparseable_response",
            message=f"openai response body is not valid JSON: {error}",
        ) from error
    mapping = _as_mapping(parsed)
    if mapping is None:
        raise ProtocolDefect(
            code="unparseable_response",
            message=(f"openai response body is not a JSON object; got {type(parsed).__name__}"),
        )
    return mapping


def _decode_usage(raw: object) -> Presence[TokenUsage]:
    usage = _as_mapping(raw)
    if usage is None:
        return Absent()
    input_details = _as_mapping(usage.get("input_tokens_details")) or {}
    output_details = _as_mapping(usage.get("output_tokens_details")) or {}
    return Present(
        TokenUsage.from_components(
            input_tokens=_as_int(usage.get("input_tokens")) or 0,
            output_tokens=_as_int(usage.get("output_tokens")) or 0,
            total_tokens=presence_of(_as_int(usage.get("total_tokens"))),
            reasoning_tokens=presence_of(_as_int(output_details.get("reasoning_tokens"))),
            cache_read_input_tokens=presence_of(_as_int(input_details.get("cached_tokens"))),
            cache_write_input_tokens=presence_of(_as_int(input_details.get("cache_write_tokens"))),
        )
    )


def _parse_tool_call(item: Mapping[str, object], fallback_arguments: str = "") -> ToolCall:
    """Strict JSON parse only; NO repair. Parse failure is an expected model
    failure surfaced via ExpectedFailureSignal (the runtime folds it)."""
    call_id = _as_str(item.get("call_id")) or _as_str(item.get("id")) or ""
    name = _as_str(item.get("name")) or ""
    arguments_raw = _as_str(item.get("arguments"))
    if arguments_raw is None:
        arguments_raw = fallback_arguments
    try:
        arguments = json.loads(arguments_raw)
    except json.JSONDecodeError as error:
        raise ExpectedFailureSignal(
            InvalidToolArguments(
                safe_detail=(
                    f"openai tool call {name!r} arguments are not valid JSON: "
                    f"{sanitize_provider_text(str(error), limit=200)}"
                )
            )
        ) from error
    if not isinstance(arguments, Mapping):
        raise ExpectedFailureSignal(
            InvalidToolArguments(
                safe_detail=(
                    f"openai tool call {name!r} arguments are not a JSON object; "
                    f"got {type(arguments).__name__}"
                )
            )
        )
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _incomplete_reason(
    details: object,
) -> tuple[Literal["max_output_tokens", "content_filter_partial"], str]:
    """Map incomplete_details to (Incomplete.reason literal, native reason)."""
    native = _as_str((_as_mapping(details) or {}).get("reason")) or ""
    if native == "max_output_tokens":
        return "max_output_tokens", native
    if native == "content_filter":
        return "content_filter_partial", native
    raise ProtocolDefect(
        code="unknown_incomplete_reason",
        message=f"openai incomplete response carries unknown reason {native!r}",
    )


def _continuation_of(
    model: str, output_items: Sequence[Mapping[str, object]]
) -> Presence[ContinuationArtifact]:
    if not output_items:
        return Absent()
    return Present(
        ContinuationArtifact(
            target=ProviderTarget(provider="openai", model=model),
            codec_id=CODEC_ID,
            opaque_payload={"output": tuple(dict(item) for item in output_items)},
        )
    )


def _meta(
    *,
    model: str,
    request_id: str | None,
    usage: Presence[TokenUsage],
) -> CallMeta:
    return CallMeta(
        provider="openai",
        model=model,
        provider_request_id=presence_of(request_id),
        upstream_provider=Absent(),
        usage=usage,
        attempt_trace=(),
        billability=PossiblyBillable(),
    )


# ---------------------------------------------------------------------------
# decode_response (2xx only)


def decode_response(status: int, headers: Mapping[str, str], body: bytes) -> CallOutcome:
    envelope = _parse_json_object(body)
    model = _as_str(envelope.get("model"))
    if model is None:
        raise ProtocolDefect(
            code="missing_model",
            message="openai response envelope is missing 'model'",
        )
    request_id = headers.get("x-request-id") or _as_str(envelope.get("id"))
    meta = _meta(model=model, request_id=request_id, usage=_decode_usage(envelope.get("usage")))

    output_items = _item_list(envelope.get("output"))
    if output_items is None:
        raise ProtocolDefect(
            code="malformed_output",
            message="openai response 'output' is not a list of items",
        )

    refusal = _collect_refusal(output_items)
    if refusal is not None:
        return Refused(meta=meta, safe_detail=sanitize_provider_text(refusal))

    response_status = _as_str(envelope.get("status"))
    if response_status == "incomplete":
        reason, native = _incomplete_reason(envelope.get("incomplete_details"))
        return Incomplete(
            meta=meta,
            reason=reason,
            status="provider_incomplete",
            safe_detail=Present(native),
        )
    if response_status != "completed":
        raise ProtocolDefect(
            code="unknown_response_status",
            message=f"openai response carries unknown terminal status {response_status!r}",
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for item in output_items:
        item_type = item.get("type")
        if item_type == "message":
            for part in _item_list(item.get("content")) or []:
                if part.get("type") == "output_text":
                    text_parts.append(_as_str(part.get("text")) or "")
        elif item_type == "function_call":
            tool_calls.append(_parse_tool_call(item))
    text = "".join(text_parts)

    return Succeeded(
        meta=meta,
        response=ResponsePayload(
            content=TextContent(text=text, tool_calls=tuple(tool_calls)),
            continuation=_continuation_of(model, output_items),
        ),
    )


def _collect_refusal(output_items: Sequence[Mapping[str, object]]) -> str | None:
    parts: list[str] = []
    for item in output_items:
        if item.get("type") != "message":
            continue
        for part in _item_list(item.get("content")) or []:
            if part.get("type") == "refusal":
                parts.append(_as_str(part.get("refusal")) or "")
    return "".join(parts) if parts else None


# ---------------------------------------------------------------------------
# decode_stream


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


async def decode_stream(
    headers: Mapping[str, str], events: AsyncIterator[SseEvent]
) -> AsyncIterator[CodecStreamEvent]:
    state = _StreamState(request_id=headers.get("x-request-id"))
    yield StreamStart()

    async for sse in events:
        if sse.data == "[DONE]":
            continue
        frame = _parse_stream_frame(sse.data)
        frame_type = _as_str(frame.get("type"))

        if frame_type in ("response.created", "response.in_progress"):
            envelope = _as_mapping(frame.get("response")) or {}
            if state.request_id is None:
                state.request_id = _as_str(envelope.get("id"))
        elif frame_type == "response.output_text.delta":
            delta = _as_str(frame.get("delta")) or ""
            if delta:
                state.text_parts.append(delta)
                yield TextDelta(text=delta)
        elif frame_type == "response.refusal.delta":
            state.refusal_parts.append(_as_str(frame.get("delta")) or "")
        elif frame_type == "response.output_item.added":
            start = _register_tool_call(state, frame)
            if start is not None:
                yield start
        elif frame_type == "response.function_call_arguments.delta":
            delta_event = _accumulate_tool_arguments(state, frame)
            if delta_event is not None:
                yield delta_event
        elif frame_type == "response.output_item.done":
            done = _finish_output_item(state, frame)
            if done is not None:
                yield done
        elif frame_type in ("response.completed", "response.incomplete"):
            for event in _terminal_events(state, frame, frame_type):
                yield event
            return
        elif frame_type == "response.failed":
            envelope = _as_mapping(frame.get("response")) or {}
            _raise_stream_error(_as_mapping(envelope.get("error")) or {})
        elif frame_type == "error":
            _raise_stream_error(frame)
        # Other event types (content_part lifecycles, *.done text frames,
        # reasoning summaries, ...) carry no semantic payload we consume.

    raise TransientStreamError(ProviderStreamInterrupted(partial_output=False))


def _parse_stream_frame(data: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as error:
        raise ProtocolDefect(
            code="malformed_stream_frame",
            message=f"openai stream frame is not valid JSON: {error}",
        ) from error
    mapping = _as_mapping(parsed)
    if mapping is None:
        raise ProtocolDefect(
            code="malformed_stream_frame",
            message=f"openai stream frame is not a JSON object; got {type(parsed).__name__}",
        )
    return mapping


def _register_tool_call(state: _StreamState, frame: Mapping[str, object]) -> ToolCallStart | None:
    item = _as_mapping(frame.get("item")) or {}
    if item.get("type") != "function_call":
        return None
    item_key = _as_str(frame.get("item_id")) or _as_str(item.get("id")) or ""
    call_id = _as_str(item.get("call_id")) or item_key
    name = _as_str(item.get("name")) or ""
    state.open_calls[item_key] = _OpenToolCall(call_id=call_id, name=name)
    return ToolCallStart(call_id=call_id, name=name)


def _accumulate_tool_arguments(
    state: _StreamState, frame: Mapping[str, object]
) -> ToolCallDelta | None:
    item_key = _as_str(frame.get("item_id")) or ""
    open_call = state.open_calls.get(item_key)
    if open_call is None:
        return None
    delta = _as_str(frame.get("delta")) or ""
    if not delta:
        return None
    open_call.arguments += delta
    return ToolCallDelta(call_id=open_call.call_id, arguments_delta=delta)


def _finish_output_item(state: _StreamState, frame: Mapping[str, object]) -> ToolCallDone | None:
    item = _as_mapping(frame.get("item"))
    if item is None:
        raise ProtocolDefect(
            code="malformed_stream_frame",
            message="openai response.output_item.done frame is missing 'item'",
        )
    state.completed_items.append(item)
    item_type = item.get("type")
    if item_type == "function_call":
        item_key = _as_str(frame.get("item_id")) or _as_str(item.get("id")) or ""
        open_call = state.open_calls.pop(item_key, None)
        tool_call = _parse_tool_call(item, open_call.arguments if open_call else "")
        state.tool_calls.append(tool_call)
        return ToolCallDone(tool_call=tool_call)
    if item_type == "message" and not state.refusal_parts:
        # Non-delta refusal path: a completed message item may carry the
        # refusal as a finished content part.
        for part in _item_list(item.get("content")) or []:
            if part.get("type") == "refusal":
                state.refusal_parts.append(_as_str(part.get("refusal")) or "")
    return None


def _terminal_events(
    state: _StreamState, frame: Mapping[str, object], frame_type: str
) -> list[CodecStreamEvent]:
    envelope = _as_mapping(frame.get("response"))
    if envelope is None:
        raise ProtocolDefect(
            code="malformed_stream_frame",
            message=f"openai {frame_type} frame is missing 'response'",
        )
    model = _as_str(envelope.get("model"))
    if model is None:
        raise ProtocolDefect(
            code="missing_model",
            message=f"openai {frame_type} envelope is missing 'model'",
        )
    request_id = state.request_id or _as_str(envelope.get("id"))
    meta = _meta(model=model, request_id=request_id, usage=_decode_usage(envelope.get("usage")))

    if frame_type == "response.incomplete":
        reason, native = _incomplete_reason(envelope.get("incomplete_details"))
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
        # OpenAI's refusal is a content part on an HTTP-200 completed stream;
        # streamed refusal folds into the four-kind stream terminal grammar as
        # Incomplete(status="refused") — partial output is discarded downstream,
        # so no ContinuationDelta is emitted.
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

    events: list[CodecStreamEvent] = []
    continuation = _continuation_of(model, state.completed_items)
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
                meta=meta,
                response=ResponsePayload(
                    content=TextContent(
                        text="".join(state.text_parts),
                        tool_calls=tuple(state.tool_calls),
                    ),
                    continuation=continuation,
                ),
            )
        )
    )
    return events


def _raise_stream_error(error: Mapping[str, object]) -> None:
    code = _as_str(error.get("code")) or ""
    if code == "rate_limit_exceeded":
        raise TransientStreamError(ProviderRateLimit(retry_after=Absent()))
    if code in _TRANSIENT_STREAM_ERROR_CODES:
        raise TransientStreamError(ProviderHttpUnavailable())
    snippet = safe_provider_error_body_snippet({"error": dict(error)}, None)
    raise ProtocolDefect(
        code="provider_stream_failure",
        message=f"openai stream reported a terminal provider error: {snippet or code or '?'}",
    )


# ---------------------------------------------------------------------------
# classify_error (non-2xx only)


def classify_error(status: int, headers: Mapping[str, str], body: bytes) -> ClassifiedError:
    json_body: dict[str, object] | None
    try:
        parsed = json.loads(body)
        json_body = parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        json_body = None
    error = _as_mapping((json_body or {}).get("error")) or {}
    code = _as_str(error.get("code")) or ""
    error_type = _as_str(error.get("type")) or ""
    message = _as_str(error.get("message")) or ""
    snippet = safe_provider_error_body_snippet(json_body, None) or f"HTTP {status}"

    if status in (401, 403):
        raise CredentialRejected(
            message=f"openai rejected the platform credential (HTTP {status}): {snippet}"
        )
    if status == 402 or "insufficient_quota" in (code, error_type):
        raise RuntimeDefect(
            origin="provider_http",
            code="quota_exhausted",
            message=f"openai quota/billing exhausted (HTTP {status}): {snippet}",
        )
    if code == "context_length_exceeded" or "maximum context length" in message.lower():
        return ProviderContextTooLarge()
    if status == 429:
        return ProviderRateLimit(retry_after=_retry_after_seconds(headers))
    if status in (500, 502, 503, 504):
        return ProviderHttpUnavailable()
    raise RuntimeDefect(
        origin="provider_http",
        code="unclassified_provider_error",
        message=f"openai returned an unclassified error (HTTP {status}): {snippet}",
    )


def _retry_after_seconds(headers: Mapping[str, str]) -> Presence[float]:
    raw = headers.get("retry-after")
    if raw is None:
        return Absent()
    try:
        seconds = float(raw)
    except ValueError:
        return Absent()
    return Present(seconds) if seconds >= 0 else Absent()


# ---------------------------------------------------------------------------
# Transcription (transport-special multipart port; NOT a FinalizedProviderRequest)


@dataclass(frozen=True, slots=True)
class TranscriptionHttpRequest:
    """httpx-facing multipart shape for POST /v1/audio/transcriptions.

    The runtime's transcribe port dispatches it as
    ``client.post(url, data=dict(form_fields), files={"file": (filename,
    content, media_type)})`` with the transport-injected Authorization header —
    the old codec's exact multipart contract."""

    url: str
    form_fields: Mapping[str, str]
    filename: str
    content: bytes = field(repr=False)
    media_type: str = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    usage: Presence[TokenUsage]
    provider_request_id: Presence[str]


def build_transcription_request(
    *, model: str, filename: str, audio: bytes, media_type: str
) -> TranscriptionHttpRequest:
    return TranscriptionHttpRequest(
        url=_TRANSCRIPTIONS_URL,
        form_fields={"model": model, "response_format": "json"},
        filename=filename,
        content=audio,
        media_type=media_type,
    )


def parse_transcription_response(
    status: int, headers: Mapping[str, str], body: bytes
) -> TranscriptionResult:
    """Decode a 2xx transcription response; non-2xx goes through classify_error."""
    data = _parse_json_object(body)
    text = _as_str(data.get("text"))
    if text is None:
        raise ProtocolDefect(
            code="missing_transcription_text",
            message="openai transcription response did not include text",
        )
    usage_mapping = _as_mapping(data.get("usage"))
    usage: Presence[TokenUsage] = Absent()
    if usage_mapping is not None:
        usage = Present(
            TokenUsage.from_components(
                input_tokens=_as_int(usage_mapping.get("input_tokens")) or 0,
                output_tokens=_as_int(usage_mapping.get("output_tokens")) or 0,
                total_tokens=presence_of(_as_int(usage_mapping.get("total_tokens"))),
                reasoning_tokens=Absent(),
                cache_read_input_tokens=Absent(),
                cache_write_input_tokens=Absent(),
            )
        )
    request_id = headers.get("x-request-id") or _as_str(data.get("id"))
    return TranscriptionResult(
        text=text,
        usage=usage,
        provider_request_id=presence_of(request_id),
    )
