"""Anthropic Messages codec (protocol/codec id ``anthropic_messages``).

Full replacement of the pre-cutover ``AnthropicClient``. The old forced-tool
structured-output path and the thinking-budget machinery are DELETED: strict
JSON output rides the native, GA ``output_config.format`` (dialect
``anthropic_output_config_json_schema``, no beta header, thinking-compatible),
and no ``thinking`` field is ever sent — Fable's thinking is always on and
Sonnet 5 runs adaptive by default, so the codec emits only
``output_config.effort``.

Wire decisions
--------------
- Sampling: NEVER sent (Claude 400s on ``temperature``/``top_p``/``top_k``).
- Caching (spec §7 "explicit breakpoint ... optionally plus top-level
  automatic"): BOTH mechanisms are emitted on every request. The explicit
  breakpoint ``cache_control {"type": "ephemeral", "ttl": "5m"}`` is placed on
  the LAST block of the leading stable-prefix run — a system block, or, when
  stable content extends into leading user messages, the last stable content
  block. Additionally the constant top-level request field
  ``cache_control: {"type": "ephemeral"}`` opts into Anthropic's automatic
  mode (auto-breakpoint on the last cacheable block), which serves append-only
  chat tails beyond the stable prefix. This is the simplest spec-compliant
  form: the explicit breakpoint is deterministic (it feeds ``prefix_bytes``
  and the cache plan) and the automatic field is a byte-constant, so it never
  perturbs affinity.
- Reasoning: top-level ``output_config: {"effort": <native>}`` — identity
  mapping for low/medium/high/xhigh/max. No ``thinking`` field of any kind.
- Strict output: MERGED into the SAME ``output_config`` object:
  ``{"effort": X, "format": {"type": "json_schema", "schema": ...}}`` with the
  schema serialized via ``to_json_schema(inline_defs=True,
  include_annotations=True)``.
- Continuation: the artifact ``opaque_payload`` is ``{"blocks": [...]}`` — the
  prior turn's thinking/redacted_thinking content blocks verbatim. On encode
  they lead the assistant turn unchanged; typed fields supply the text and
  tool_use blocks (runtime-api assistant-turn rule).
- Streaming: ``stream_request`` derives the streaming variant of a finalized
  request by injecting ``"stream": true`` into the body (identical
  deterministic serialization); ``generate`` uses the finalized request as-is.
- Refusal billability: Anthropic confirms pre-output refusals are unbilled, so
  a refusal whose reported ``usage.output_tokens == 0`` is
  ``ConfirmedNonBillable``; any refusal with billed output tokens (mid-stream
  refusal) or without reported usage is ``PossiblyBillable``.
- Streamed refusal: the stream terminal grammar has no ``Refused`` — an
  HTTP-200 ``stop_reason: "refusal"`` on a stream terminates as
  ``Incomplete(reason="content_filter_partial", status="refused")`` with a
  Present safe detail, and the partial output (including any thinking blocks)
  is invalidated: no ``ContinuationDelta`` is emitted for a refused stream.
- Structured-output decode arm (cross-codec rule): decode returns
  ``TextContent`` ONLY. The seam's ``decode_response(status, headers, body)``
  and ``decode_stream(headers, events)`` signatures carry no plan context, and
  the output arm is determined by the PLAN, never re-inferred — so under
  ``StrictJsonOutput`` the schema-conformant JSON arrives as the
  ``TextContent`` text, and ``StructuredContent`` construction (strict parse,
  with Refused/Incomplete taking precedence) belongs to the plan-owning
  runtime.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Final

from provider_runtime._signals import (
    ClassifiedError,
    ExpectedFailureSignal,
    TransientStreamError,
)
from provider_runtime.catalog import ChatModelContract
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
    CanonicalTool,
    CodecStreamEvent,
    ConfirmedNonBillable,
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
    PromptBlock,
    ProviderContextTooLarge,
    ProviderHttpUnavailable,
    ProviderProtocol,
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
    UsageEvent,
    UserMessage,
)

CODEC_ID: Final[str] = "anthropic_messages"
PROTOCOL: Final[ProviderProtocol] = "anthropic_messages"

_URL: Final[str] = "https://api.anthropic.com/v1/messages"
_API_VERSION: Final[str] = "2023-06-01"

_EXPLICIT_BREAKPOINT: Final[Mapping[str, str]] = {"type": "ephemeral", "ttl": "5m"}
_AUTOMATIC_CACHE_CONTROL: Final[Mapping[str, str]] = {"type": "ephemeral"}

_SUCCESS_STOP_REASONS: Final[frozenset[str]] = frozenset({"end_turn", "tool_use", "stop_sequence"})
_THINKING_BLOCK_TYPES: Final[frozenset[str]] = frozenset({"thinking", "redacted_thinking"})
# 5xx-shaped in-band/streamed provider error types (overloaded_error ≙ 529).
_TRANSIENT_ERROR_TYPES: Final[frozenset[str]] = frozenset({"overloaded_error", "api_error"})


def _dumps(obj: object) -> str:
    # The one body serialization rule (codec seam): deterministic literal
    # construction order, compact separators, no ASCII escaping.
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _dump_bytes(obj: object) -> bytes:
    return _dumps(obj).encode("utf-8")


def _frame(component: bytes) -> bytes:
    return len(component).to_bytes(8, "big") + component


# ---------------------------------------------------------------------------
# encode / finalize / stream_request


def _tool_wire(tool: CanonicalTool) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": to_json_schema(tool.parameters, inline_defs=True, include_annotations=True),
    }


def _continuation_blocks(artifact: ContinuationArtifact) -> list[dict[str, object]]:
    raw = artifact.opaque_payload.get("blocks")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise PlanningDefect(
            code="invalid_continuation_payload",
            message=(
                "anthropic continuation artifact payload must carry a 'blocks' sequence of"
                " content blocks"
            ),
        )
    blocks: list[dict[str, object]] = []
    for index, block in enumerate(raw):
        if not isinstance(block, Mapping):
            raise PlanningDefect(
                code="invalid_continuation_payload",
                message=(f"anthropic continuation artifact payload block {index} is not a mapping"),
            )
        blocks.append({str(key): value for key, value in block.items()})
    return blocks


def _assistant_wire(message: AssistantMessage, intent: GenerateIntent) -> dict[str, object]:
    content: list[dict[str, object]] = []
    match message.continuation:
        case Present(value=artifact):
            if artifact.target != intent.target or artifact.codec_id != CODEC_ID:
                raise PlanningDefect(
                    code="continuation_mismatch",
                    message=(
                        "continuation artifact does not match the intent target/codec: "
                        f"artifact codec {artifact.codec_id!r} for "
                        f"{artifact.target.provider}/{artifact.target.model}, intent "
                        f"{CODEC_ID!r} for {intent.target.provider}/{intent.target.model}"
                    ),
                )
            # Thinking/redacted_thinking blocks lead the assistant turn VERBATIM.
            content.extend(_continuation_blocks(artifact))
        case Absent():
            pass
    if message.text:
        content.append({"type": "text", "text": message.text})
    for call in message.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": dict(call.arguments),
            }
        )
    if not content:
        # Anthropic rejects empty content arrays; there is no wire-legal way
        # to represent a turn with no continuation blocks, text, or tool
        # calls — omitting the message would silently misrepresent the
        # conversation, so this is a defect, not a silent drop.
        raise PlanningDefect(
            code="empty_assistant_turn",
            message=(
                "anthropic assistant turn has no continuation blocks, text, or tool calls to encode"
            ),
        )
    return {"role": "assistant", "content": content}


def _tool_result_wire(message: ToolResultMessage) -> dict[str, object]:
    return {
        "type": "tool_result",
        "tool_use_id": message.call_id,
        "content": message.output,
        "is_error": message.is_error,
    }


def _text_block_wire(block: PromptBlock) -> dict[str, object]:
    return {"type": "text", "text": block.text}


def encode(intent: GenerateIntent, contract: ChatModelContract) -> DraftRequest:
    if contract.protocol != PROTOCOL:
        raise PlanningDefect(
            code="protocol_mismatch",
            message=(
                f"anthropic codec received contract protocol {contract.protocol!r}; "
                f"expected {PROTOCOL!r}"
            ),
        )
    native_reasoning = contract.reasoning.native_mapping.get(intent.reasoning)
    if native_reasoning is None:
        raise PlanningDefect(
            code="unsupported_reasoning_level",
            message=(
                f"reasoning level {intent.reasoning!r} has no native mapping for "
                f"{intent.target.provider}/{intent.target.model}"
            ),
        )

    system_blocks: list[dict[str, object]] = []
    messages: list[dict[str, object]] = []
    # Wire-ordered (block dict, stability) pairs plus run-breakers (None) for
    # assistant/tool turns; the stable prefix is the leading Stable run.
    flattened: list[tuple[dict[str, object], Stable] | None] = []
    # Placement-scoped stable-prefix projection for cache-affinity framing:
    # the leading system-field blocks (if any) as ONE {"system": [...]} entry,
    # then each leading user message's stable content as its own
    # {"role": "user", "content": [...]} entry — so a role move (system vs.
    # leading user) or a message regrouping changes prefix_bytes. Shares the
    # same wire dict objects as `system_blocks`/`messages`, so the breakpoint
    # stamp below is visible here too.
    stable_prefix_projection: list[dict[str, object]] = []
    system_stable_blocks: list[dict[str, object]] = []
    prefix_open = True

    pending_tool_results: list[dict[str, object]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            messages.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    system_phase = True
    for message in intent.messages:
        match message:
            case SystemMessage(blocks=blocks):
                if not system_phase:
                    raise PlanningDefect(
                        code="misplaced_system_message",
                        message=(
                            "anthropic system messages must precede all conversation turns"
                            " (the Messages API has a single top-level system field)"
                        ),
                    )
                for block in blocks:
                    if not block.text:
                        # Anthropic rejects empty text blocks; drop it from
                        # the wire entirely rather than sending "". An empty
                        # dynamic block still ends the leading stable run —
                        # it carries no wire artifact, but its position in
                        # the message sequence still marks where the planner
                        # guaranteed stable prefix stops.
                        if isinstance(block.stability, Dynamic):
                            flattened.append(None)
                            prefix_open = False
                        continue
                    wire = _text_block_wire(block)
                    system_blocks.append(wire)
                    match block.stability:
                        case Stable() as stable:
                            flattened.append((wire, stable))
                            if prefix_open:
                                system_stable_blocks.append(wire)
                        case Dynamic():
                            flattened.append(None)
                            prefix_open = False
            case UserMessage(blocks=blocks):
                system_phase = False
                flush_tool_results()
                if system_stable_blocks:
                    stable_prefix_projection.append({"system": system_stable_blocks})
                    system_stable_blocks = []
                content: list[dict[str, object]] = []
                stable_content: list[dict[str, object]] = []
                for block in blocks:
                    if not block.text:
                        # Same empty-text drop as the system-phase loop above.
                        if isinstance(block.stability, Dynamic):
                            flattened.append(None)
                            prefix_open = False
                        continue
                    wire = _text_block_wire(block)
                    content.append(wire)
                    match block.stability:
                        case Stable() as stable:
                            flattened.append((wire, stable))
                            if prefix_open:
                                stable_content.append(wire)
                        case Dynamic():
                            flattened.append(None)
                            prefix_open = False
                if not content:
                    raise PlanningDefect(
                        code="empty_message_content",
                        message=(
                            "anthropic user turn has no non-empty content blocks to"
                            " encode (Anthropic rejects an empty content array)"
                        ),
                    )
                messages.append({"role": "user", "content": content})
                if stable_content:
                    stable_prefix_projection.append({"role": "user", "content": stable_content})
            case AssistantMessage():
                system_phase = False
                flush_tool_results()
                flattened.append(None)
                prefix_open = False
                messages.append(_assistant_wire(message, intent))
            case ToolResultMessage():
                system_phase = False
                flattened.append(None)
                prefix_open = False
                pending_tool_results.append(_tool_result_wire(message))
    flush_tool_results()
    if system_stable_blocks:
        stable_prefix_projection.append({"system": system_stable_blocks})

    # Explicit 5m breakpoint on the LAST block of the leading stable run. The
    # planner guarantees a non-empty contiguous stable prefix; the codec places
    # the breakpoint wherever that run ends (system block or leading stable
    # content block).
    stable_prefix: list[dict[str, object]] = []
    for entry in flattened:
        if entry is None:
            break
        stable_prefix.append(entry[0])
    if stable_prefix:
        stable_prefix[-1]["cache_control"] = dict(_EXPLICIT_BREAKPOINT)

    tools_wire = [_tool_wire(tool) for tool in intent.tools]

    output_config: dict[str, object] = {"effort": native_reasoning}
    format_wire: dict[str, object] | None = None
    match intent.output:
        case StrictJsonOutput(schema=schema):
            format_wire = {
                "type": "json_schema",
                "schema": to_json_schema(schema, inline_defs=True, include_annotations=True),
            }
            output_config["format"] = format_wire
        case TextOutput():
            pass

    body: dict[str, object] = {
        "model": intent.target.model,
        "max_tokens": intent.max_output_tokens,
    }
    if system_blocks:
        body["system"] = system_blocks
    body["messages"] = messages
    if tools_wire:
        body["tools"] = tools_wire
        body["tool_choice"] = {"type": intent.tool_choice}
    body["output_config"] = output_config
    # Top-level automatic caching (append-only chat) — a byte-constant field.
    body["cache_control"] = dict(_AUTOMATIC_CACHE_CONTROL)

    # Fixed section tags separate the four component classes (stable/tools/
    # tool_choice/format) so a stable text byte-equal to a tool-definition
    # dump can never collide across classes; the stable class is framed at
    # message/placement granularity (system field vs. messages[] role), not
    # bare block text, so a role move or message regrouping changes the bytes.
    prefix_components: list[bytes] = [_frame(b"stable")]
    prefix_components.extend(_frame(_dump_bytes(item)) for item in stable_prefix_projection)
    prefix_components.append(_frame(b"tools"))
    prefix_components.extend(_frame(_dump_bytes(tool)) for tool in tools_wire)
    prefix_components.append(_frame(b"tool_choice"))
    if tools_wire:
        prefix_components.append(_frame(_dump_bytes({"type": intent.tool_choice})))
    prefix_components.append(_frame(b"format"))
    if format_wire is not None:
        prefix_components.append(_frame(_dump_bytes(format_wire)))

    return DraftRequest(
        target=intent.target,
        protocol=PROTOCOL,
        url=_URL,
        safe_headers={"anthropic-version": _API_VERSION, "accept": "application/json"},
        native_reasoning=native_reasoning,
        provider_framing_overhead_tokens=contract.provider_framing_overhead_tokens,
        prefix_bytes=b"".join(prefix_components),
        body=_dump_bytes(body),
    )


def finalize(draft: DraftRequest, affinity: str) -> FinalizedProviderRequest:
    """Anthropic carries no injected affinity field: passthrough (body unchanged)."""
    del affinity
    return FinalizedProviderRequest(
        target=draft.target,
        protocol=draft.protocol,
        url=draft.url,
        method="POST",
        safe_headers=draft.safe_headers,
        body=draft.body,
    )


def stream_request(request: FinalizedProviderRequest) -> FinalizedProviderRequest:
    """The streaming variant of a finalized call: inject ``"stream": true``.

    Returns a NEW value with the identical deterministic serialization;
    ``generate`` uses the finalized request as-is.
    """
    body = _parse_json_object(request.body, context="finalized anthropic request body")
    streamed = dict(body)
    streamed["stream"] = True
    return FinalizedProviderRequest(
        target=request.target,
        protocol=request.protocol,
        url=request.url,
        method=request.method,
        safe_headers=request.safe_headers,
        body=_dump_bytes(streamed),
    )


# ---------------------------------------------------------------------------
# Shared decode helpers


def _parse_json_object(body: bytes, *, context: str) -> dict[str, object]:
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolDefect(
            code="malformed_provider_body",
            message=f"{context} is not valid JSON: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise ProtocolDefect(
            code="malformed_provider_body",
            message=f"{context} must be a JSON object; got {type(parsed).__name__}",
        )
    return parsed


def _int_field(raw: object) -> int | None:
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _usage_presence(raw: object) -> Presence[TokenUsage]:
    """Fold one Anthropic usage mapping into the normalized TokenUsage.

    Anthropic's wire ``input_tokens`` EXCLUDES cache read/write components,
    but ``TokenUsage.input_tokens`` is, by the codec-invariant convention
    documented on ``TokenUsage.from_components``, always the cache-INCLUSIVE
    total prompt token count. Normalize at ingress: the stored
    ``input_tokens`` is the raw wire value plus cache_read_input_tokens plus
    cache_creation_input_tokens, matching what OpenAI/Gemini/Moonshot/
    OpenRouter report natively. Anthropic also reports no total, so the
    derived-total branch of ``from_components`` applies: total = input
    (already inclusive) + output.
    """
    if not isinstance(raw, Mapping):
        return Absent()
    raw_input_tokens = _int_field(raw.get("input_tokens"))
    output_tokens = _int_field(raw.get("output_tokens"))
    if raw_input_tokens is None or output_tokens is None:
        return Absent()
    cache_read = _int_field(raw.get("cache_read_input_tokens"))
    cache_write = _int_field(raw.get("cache_creation_input_tokens"))
    input_tokens = raw_input_tokens + (cache_read or 0) + (cache_write or 0)
    return Present(
        TokenUsage.from_components(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=Absent(),
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Absent() if cache_read is None else Present(cache_read),
            cache_write_input_tokens=Absent() if cache_write is None else Present(cache_write),
        )
    )


def _request_id(headers: Mapping[str, str], in_band: object) -> Presence[str]:
    header = headers.get("request-id")
    if isinstance(header, str) and header:
        return Present(header)
    if isinstance(in_band, str) and in_band:
        return Present(in_band)
    return Absent()


def _meta(
    *,
    model: str,
    provider_request_id: Presence[str],
    usage: Presence[TokenUsage],
    billability: ConfirmedNonBillable | PossiblyBillable,
) -> CallMeta:
    return CallMeta(
        provider="anthropic",
        model=model,
        provider_request_id=provider_request_id,
        upstream_provider=Absent(),
        usage=usage,
        attempt_trace=(),
        billability=billability,
    )


def _refusal_billability(usage: Presence[TokenUsage]) -> ConfirmedNonBillable | PossiblyBillable:
    """Anthropic contract: a pre-output refusal is unbilled. The observable
    signal is reported usage with zero output tokens; anything else (billed
    partial output, or no reported usage) stays PossiblyBillable."""
    match usage:
        case Present(value=value) if value.output_tokens == 0:
            return ConfirmedNonBillable()
        case _:
            return PossiblyBillable()


def _refusal_detail(stop_details: object) -> str:
    if isinstance(stop_details, Mapping):
        explanation = stop_details.get("explanation")
        if isinstance(explanation, str) and explanation:
            return sanitize_provider_text(explanation)
        category = stop_details.get("category")
        if isinstance(category, str) and category:
            return sanitize_provider_text(f"refusal category: {category}")
    return "provider refusal"


def _tool_call_from_wire(block: Mapping[str, object]) -> ToolCall:
    call_id = block.get("id")
    name = block.get("name")
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise ProtocolDefect(
            code="malformed_content_block",
            message="anthropic tool_use block is missing string 'id'/'name' fields",
        )
    arguments = block.get("input")
    if not isinstance(arguments, Mapping):
        raise ExpectedFailureSignal(
            InvalidToolArguments(
                safe_detail=(
                    f"anthropic tool_use input for tool {name!r} (call {call_id}) is not a"
                    " JSON object"
                )
            )
        )
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _continuation_presence(
    thinking_blocks: Sequence[Mapping[str, object]], model: str
) -> Presence[ContinuationArtifact]:
    if not thinking_blocks:
        return Absent()
    return Present(
        ContinuationArtifact(
            target=ProviderTarget(provider="anthropic", model=model),
            codec_id=CODEC_ID,
            opaque_payload={"blocks": [dict(block) for block in thinking_blocks]},
        )
    )


# ---------------------------------------------------------------------------
# decode_response (non-stream, 2xx only)


def decode_response(status: int, headers: Mapping[str, str], body: bytes) -> CallOutcome:
    del status  # 2xx only by contract; non-2xx goes through classify_error.
    data = _parse_json_object(body, context="anthropic response body")
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise ProtocolDefect(
            code="malformed_provider_body",
            message="anthropic response body is missing the string 'model' field",
        )
    provider_request_id = _request_id(headers, data.get("id"))
    usage = _usage_presence(data.get("usage"))
    stop_reason = data.get("stop_reason")

    if stop_reason == "refusal":
        return Refused(
            meta=_meta(
                model=model,
                provider_request_id=provider_request_id,
                usage=usage,
                billability=_refusal_billability(usage),
            ),
            safe_detail=_refusal_detail(data.get("stop_details")),
        )

    if stop_reason == "max_tokens":
        return Incomplete(
            meta=_meta(
                model=model,
                provider_request_id=provider_request_id,
                usage=usage,
                billability=PossiblyBillable(),
            ),
            reason="max_output_tokens",
            status="provider_incomplete",
            safe_detail=Absent(),
        )

    if not isinstance(stop_reason, str) or stop_reason not in _SUCCESS_STOP_REASONS:
        raise ProtocolDefect(
            code="unknown_stop_reason",
            message=f"anthropic response carried unknown stop_reason {stop_reason!r}",
        )

    content = data.get("content")
    if not isinstance(content, list):
        raise ProtocolDefect(
            code="malformed_provider_body",
            message="anthropic response 'content' must be an array of blocks",
        )
    text_blocks: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking_blocks: list[Mapping[str, object]] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise ProtocolDefect(
                code="malformed_content_block",
                message="anthropic content block is not a JSON object",
            )
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ProtocolDefect(
                    code="malformed_content_block",
                    message="anthropic text block is missing the string 'text' field",
                )
            text_blocks.append(text)
        elif block_type == "tool_use":
            tool_calls.append(_tool_call_from_wire(block))
        elif isinstance(block_type, str) and block_type in _THINKING_BLOCK_TYPES:
            thinking_blocks.append(block)
        else:
            raise ProtocolDefect(
                code="unknown_content_block",
                message=f"anthropic response carried unknown content block type {block_type!r}",
            )

    return Succeeded(
        meta=_meta(
            model=model,
            provider_request_id=provider_request_id,
            usage=usage,
            billability=PossiblyBillable(),
        ),
        response=ResponsePayload(
            # TextContent only (cross-codec rule): under strict output the
            # schema-conformant JSON is this text; the runtime owns the parse.
            content=TextContent(text="".join(text_blocks), tool_calls=tuple(tool_calls)),
            continuation=_continuation_presence(thinking_blocks, model),
        ),
    )


# ---------------------------------------------------------------------------
# decode_stream


def _stream_event_payload(event: SseEvent) -> dict[str, object]:
    payload = _parse_json_object(event.data.encode("utf-8"), context="anthropic stream event data")
    return payload


def _raise_stream_error(payload: dict[str, object]) -> None:
    error = payload.get("error")
    error_type = error.get("type") if isinstance(error, Mapping) else None
    if isinstance(error_type, str) and error_type in _TRANSIENT_ERROR_TYPES:
        raise TransientStreamError(ProviderHttpUnavailable())
    snippet = safe_provider_error_body_snippet(payload, None)
    raise ProtocolDefect(
        code="stream_error_event",
        message=f"anthropic stream carried a non-transient error event: {snippet or error_type!r}",
    )


class _StreamState:
    """Mutable per-stream accumulation (codec-private; the yielded events and
    the terminal outcome are the only externally visible values)."""

    __slots__ = (
        "model",
        "in_band_id",
        "usage_data",
        "stop_reason",
        "stop_details",
        "texts_by_index",
        "text_order",
        "tool_by_index",
        "tool_calls",
        "thinking_by_index",
        "thinking_blocks",
        "saw_message_stop",
    )

    def __init__(self) -> None:
        self.model: str | None = None
        self.in_band_id: object = None
        self.usage_data: dict[str, object] = {}
        self.stop_reason: str | None = None
        self.stop_details: object = None
        self.texts_by_index: dict[int, list[str]] = {}
        self.text_order: list[int] = []
        self.tool_by_index: dict[int, dict[str, str]] = {}
        self.tool_calls: list[ToolCall] = []
        self.thinking_by_index: dict[int, dict[str, object]] = {}
        self.thinking_blocks: list[Mapping[str, object]] = []
        self.saw_message_stop = False


def _require_index(payload: Mapping[str, object]) -> int:
    index = payload.get("index")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ProtocolDefect(
            code="malformed_stream_event",
            message="anthropic content_block event is missing the integer 'index' field",
        )
    return index


async def decode_stream(
    headers: Mapping[str, str],
    events: AsyncIterator[SseEvent],
) -> AsyncIterator[CodecStreamEvent]:
    state = _StreamState()

    async for event in events:
        if event.event == "ping":
            continue
        payload = _stream_event_payload(event)
        raw_type = payload.get("type")
        event_type = raw_type if isinstance(raw_type, str) else event.event

        if event_type == "ping":
            continue
        if event_type == "error":
            _raise_stream_error(payload)
        elif event_type == "message_start":
            message = payload.get("message")
            if not isinstance(message, Mapping):
                raise ProtocolDefect(
                    code="malformed_stream_event",
                    message="anthropic message_start event is missing the 'message' object",
                )
            model = message.get("model")
            if isinstance(model, str) and model:
                state.model = model
            state.in_band_id = message.get("id")
            start_usage = message.get("usage")
            if isinstance(start_usage, Mapping):
                state.usage_data.update(start_usage)
            yield StreamStart()
        elif event_type == "content_block_start":
            index = _require_index(payload)
            block = payload.get("content_block")
            if not isinstance(block, Mapping):
                raise ProtocolDefect(
                    code="malformed_stream_event",
                    message="anthropic content_block_start is missing the 'content_block' object",
                )
            block_type = block.get("type")
            if block_type == "text":
                state.texts_by_index[index] = []
                state.text_order.append(index)
            elif block_type == "tool_use":
                call_id = block.get("id")
                name = block.get("name")
                if not isinstance(call_id, str) or not isinstance(name, str):
                    raise ProtocolDefect(
                        code="malformed_stream_event",
                        message=(
                            "anthropic tool_use content_block_start is missing string"
                            " 'id'/'name' fields"
                        ),
                    )
                state.tool_by_index[index] = {"id": call_id, "name": name, "json": ""}
                yield ToolCallStart(call_id=call_id, name=name)
            elif isinstance(block_type, str) and block_type in _THINKING_BLOCK_TYPES:
                state.thinking_by_index[index] = {str(key): value for key, value in block.items()}
            else:
                raise ProtocolDefect(
                    code="unknown_content_block",
                    message=(f"anthropic stream carried unknown content block type {block_type!r}"),
                )
        elif event_type == "content_block_delta":
            index = _require_index(payload)
            delta = payload.get("delta")
            if not isinstance(delta, Mapping):
                raise ProtocolDefect(
                    code="malformed_stream_event",
                    message="anthropic content_block_delta is missing the 'delta' object",
                )
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text")
                if not isinstance(text, str):
                    raise ProtocolDefect(
                        code="malformed_stream_event",
                        message="anthropic text_delta is missing the string 'text' field",
                    )
                state.texts_by_index.setdefault(index, [])
                if index not in state.text_order:
                    state.text_order.append(index)
                state.texts_by_index[index].append(text)
                if text:
                    yield TextDelta(text=text)
            elif delta_type == "input_json_delta":
                tool = state.tool_by_index.get(index)
                if tool is None:
                    raise ProtocolDefect(
                        code="malformed_stream_event",
                        message="anthropic input_json_delta arrived for an unknown tool block",
                    )
                partial = delta.get("partial_json")
                if not isinstance(partial, str):
                    raise ProtocolDefect(
                        code="malformed_stream_event",
                        message=(
                            "anthropic input_json_delta is missing the string 'partial_json' field"
                        ),
                    )
                tool["json"] += partial
                if partial:
                    yield ToolCallDelta(call_id=tool["id"], arguments_delta=partial)
            elif delta_type == "thinking_delta" or delta_type == "signature_delta":
                block_state = state.thinking_by_index.get(index)
                if block_state is None:
                    raise ProtocolDefect(
                        code="malformed_stream_event",
                        message=(f"anthropic {delta_type} arrived for an unknown thinking block"),
                    )
                key = "thinking" if delta_type == "thinking_delta" else "signature"
                fragment = delta.get(key)
                if not isinstance(fragment, str):
                    raise ProtocolDefect(
                        code="malformed_stream_event",
                        message=f"anthropic {delta_type} is missing the string {key!r} field",
                    )
                existing = block_state.get(key)
                block_state[key] = (existing if isinstance(existing, str) else "") + fragment
            else:
                raise ProtocolDefect(
                    code="malformed_stream_event",
                    message=f"anthropic stream carried unknown delta type {delta_type!r}",
                )
        elif event_type == "content_block_stop":
            index = _require_index(payload)
            tool = state.tool_by_index.pop(index, None)
            if tool is not None:
                raw_arguments = tool["json"] or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = None
                if not isinstance(arguments, dict):
                    raise ExpectedFailureSignal(
                        InvalidToolArguments(
                            safe_detail=(
                                f"anthropic tool_use arguments for tool {tool['name']!r} "
                                f"(call {tool['id']}) did not parse as a JSON object"
                            )
                        )
                    )
                tool_call = ToolCall(id=tool["id"], name=tool["name"], arguments=arguments)
                state.tool_calls.append(tool_call)
                yield ToolCallDone(tool_call=tool_call)
                continue
            thinking = state.thinking_by_index.pop(index, None)
            if thinking is not None:
                state.thinking_blocks.append(thinking)
        elif event_type == "message_delta":
            delta = payload.get("delta")
            if isinstance(delta, Mapping):
                stop_reason = delta.get("stop_reason")
                if isinstance(stop_reason, str):
                    state.stop_reason = stop_reason
                if "stop_details" in delta:
                    state.stop_details = delta.get("stop_details")
            delta_usage = payload.get("usage")
            if isinstance(delta_usage, Mapping):
                state.usage_data.update(delta_usage)
                folded = _usage_presence(state.usage_data)
                if isinstance(folded, Present):
                    yield UsageEvent(usage=folded.value)
        elif event_type == "message_stop":
            state.saw_message_stop = True
            break
        # Unknown event types are ignored (forward-compatible per Anthropic's
        # streaming contract; malformed payloads of KNOWN types defect above).

    if not state.saw_message_stop:
        # No terminal frame: transient interruption; the runtime rebuilds the
        # leaf with the true partial_output flag it tracks.
        raise TransientStreamError(ProviderStreamInterrupted(partial_output=False))

    if state.model is None:
        raise ProtocolDefect(
            code="malformed_stream_event",
            message="anthropic stream reached message_stop without a message_start model",
        )

    usage = _usage_presence(state.usage_data)
    provider_request_id = _request_id(headers, state.in_band_id)

    if state.stop_reason == "refusal":
        # Streamed refusal: the four-kind stream terminal grammar has no
        # Refused — terminate as incomplete+refused and invalidate the partial
        # output (no ContinuationDelta for a refused stream).
        yield TerminalEvent(
            outcome=Incomplete(
                meta=_meta(
                    model=state.model,
                    provider_request_id=provider_request_id,
                    usage=usage,
                    billability=_refusal_billability(usage),
                ),
                reason="content_filter_partial",
                status="refused",
                safe_detail=Present(_refusal_detail(state.stop_details)),
            )
        )
        return

    if state.stop_reason == "max_tokens":
        yield TerminalEvent(
            outcome=Incomplete(
                meta=_meta(
                    model=state.model,
                    provider_request_id=provider_request_id,
                    usage=usage,
                    billability=PossiblyBillable(),
                ),
                reason="max_output_tokens",
                status="provider_incomplete",
                safe_detail=Absent(),
            )
        )
        return

    if state.stop_reason is None or state.stop_reason not in _SUCCESS_STOP_REASONS:
        raise ProtocolDefect(
            code="unknown_stop_reason",
            message=f"anthropic stream carried unknown stop_reason {state.stop_reason!r}",
        )

    continuation = _continuation_presence(state.thinking_blocks, state.model)
    match continuation:
        case Present(value=artifact):
            # Exactly ONE ContinuationDelta, after all contributing native
            # items are final and before the terminal.
            yield ContinuationDelta(artifact=artifact)
        case Absent():
            pass

    text_blocks = ["".join(state.texts_by_index[index]) for index in state.text_order]
    yield TerminalEvent(
        outcome=Succeeded(
            meta=_meta(
                model=state.model,
                provider_request_id=provider_request_id,
                usage=usage,
                billability=PossiblyBillable(),
            ),
            response=ResponsePayload(
                # TextContent only (cross-codec rule); see decode_response.
                content=TextContent(text="".join(text_blocks), tool_calls=tuple(state.tool_calls)),
                continuation=continuation,
            ),
        )
    )


# ---------------------------------------------------------------------------
# classify_error (non-2xx only)


def _error_body(body: bytes) -> tuple[dict[str, object] | None, str | None, str]:
    """Best-effort parse: (json body, error.type, lower-cased error.message)."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None, ""
    if not isinstance(parsed, dict):
        return None, None, ""
    error = parsed.get("error")
    if not isinstance(error, Mapping):
        return parsed, None, ""
    error_type = error.get("type")
    message = error.get("message")
    return (
        parsed,
        error_type if isinstance(error_type, str) else None,
        message.lower() if isinstance(message, str) else "",
    )


def _retry_after_seconds(headers: Mapping[str, str]) -> Presence[float]:
    raw = headers.get("retry-after")
    if raw is None:
        return Absent()
    try:
        return Present(float(raw))
    except ValueError:
        return Absent()


def classify_error(status: int, headers: Mapping[str, str], body: bytes) -> ClassifiedError:
    json_body, error_type, error_message = _error_body(body)
    snippet = safe_provider_error_body_snippet(json_body, None) or sanitize_provider_text(
        body.decode("utf-8", errors="replace")
    )

    if status == 429:
        return ProviderRateLimit(retry_after=_retry_after_seconds(headers))
    if status in (500, 502, 503, 504, 529) or error_type == "overloaded_error":
        return ProviderHttpUnavailable()
    if status == 413 or error_type == "request_too_large":
        return ProviderContextTooLarge()
    if error_type == "invalid_request_error" and "too long" in error_message:
        # e.g. "prompt is too long: N tokens > limit" — the documented
        # context-overflow shape on this provider (400, not 413).
        return ProviderContextTooLarge()
    if status in (401, 403):
        raise CredentialRejected(
            message=f"anthropic rejected the platform credential (HTTP {status}): {snippet}"
        )
    if (
        status == 402
        or "credit balance is too low" in error_message
        or error_type == "billing_error"
    ):
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
