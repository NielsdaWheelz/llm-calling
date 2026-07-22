"""Gemini generateContent codec (``gemini_generate_content``).

Wire target: POST
``https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent``
(auth is transport-injected via ``x-goog-api-key``; this codec never sees the
credential).

Cross-codec streaming seam (binding, shared with every codec module):
``GenerateIntent`` carries NO streaming flag — the runtime's ``stream()`` vs
``generate()`` entry decides the mode. Each codec exposes
``stream_request(request) -> FinalizedProviderRequest`` returning a NEW frozen
value carrying the stream variant of the finalized plan. For Gemini that is a
pure URL rewrite to ``:streamGenerateContent?alt=sse`` (the body is identical
in both modes); the Chat-Completions/Responses codecs instead re-dump the body
with ``"stream": true`` injected. The runtime never rewrites plans itself.

Codec decisions (documented here because they are wire-observable):

- Reasoning: ``generationConfig.thinkingConfig.thinkingLevel`` with the
  catalog's native mapping (minimal/low/medium/high identity). The old
  per-model ``thinkingBudget`` tables are DELETED — gemini-3.5-flash speaks
  ``thinkingLevel`` natively.
- Tool schemas: ``functionDeclarations`` carry ``parametersJsonSchema`` — the
  JSON-Schema-native declaration field (parallel to
  ``generationConfig.responseJsonSchema``, dialect
  ``gemini_response_json_schema``). The canonical subset is natively
  expressible, so NOTHING is stripped: ``additionalProperties: false``,
  ``required``, ``anyOf`` nullable unions, and annotations all reach the wire
  verbatim. The old ``_strip_gemini_tool_schema_unsupported_fields`` behavior
  MUST NOT reappear; release certification validates provider acceptance.
- Caching: implicit only (``GeminiAutomaticPrefix``). NO cache field of any
  kind is sent; ``prefix_bytes`` is still computed per the codec seam because
  cache affinity is telemetry for Gemini.
- Continuation: a Succeeded model turn that contains functionCall parts or
  thoughtSignatures yields a ``ContinuationArtifact`` whose opaque payload is
  ``{"parts": <candidate content parts verbatim>}``. Replay sends the entire
  parts list back unchanged as one ``role: "model"`` turn (provider contract:
  "return the entire response with all parts back"; signature parts are never
  merged or re-built). Typed ``AssistantMessage.tool_calls`` must correspond
  1:1, in order, by name with the payload's functionCall parts — validated at
  encode; when the payload is present it is the SOLE wire source for the turn.
- Call ids: the Gemini wire has none, so decode synthesizes deterministic ids
  ``call_<index>`` in functionCall order, restarting at ``call_0`` on every
  response. Because those ids recur across turns, encode resolves
  ``ToolResultMessage.call_id`` against ONLY the most recent preceding
  assistant turn's typed tool calls (turn-scoped, not intent-wide), and
  consecutive tool results coalesce into ONE ``role: "user"`` turn.
- Blocked content: Gemini has no Anthropic-style refusal contract. A blocked
  finish reason (SAFETY / PROHIBITED_CONTENT / RECITATION / BLOCKLIST / SPII /
  IMAGE_SAFETY) and a blocked prompt (``promptFeedback.blockReason`` with no
  candidates) both map to ``Incomplete(reason="content_filter_partial",
  status="provider_incomplete")`` — in BOTH non-stream and stream decode.
  ``Refused`` is never constructed here.
- ``finishReason: "MALFORMED_FUNCTION_CALL"`` raises
  ``ExpectedFailureSignal(InvalidToolArguments)`` — the provider reporting an
  unusable tool call is the same expected failure as a strict argument-parse
  failure.
- provider_request_id: Absent() ALWAYS (no correlation id on this wire; the
  catalog row records ``provider_request_id_available=False``).
- Output arm: decode returns ``TextContent`` (for StrictJsonOutput plans the
  text IS the JSON document). The runtime owns the plan and therefore owns
  constructing ``StructuredContent`` — the codec seam signature carries no
  plan, and the output arm is never re-inferred from the response.
- ``CallMeta.model`` and the continuation artifact's target model are taken
  from the response's ``modelVersion`` echo. Live certification asserts the
  echo equals the catalog id (a divergent echo would fail continuation replay
  validation loudly at plan time, never silently).
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
    CodecStreamEvent,
    ContinuationArtifact,
    ContinuationDelta,
    DraftRequest,
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
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTarget,
    ResponsePayload,
    Stable,
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
    ToolCallStart,
    ToolResultMessage,
    UserMessage,
)

CODEC_ID: Final[str] = "gemini_generate_content"

_BASE_URL: Final = "https://generativelanguage.googleapis.com/v1beta/models"
_NONSTREAM_SUFFIX: Final = ":generateContent"
_STREAM_SUFFIX: Final = ":streamGenerateContent?alt=sse"

# Blocked-content finish reasons (see module docstring for the mapping rationale).
_CONTENT_FILTER_FINISH_REASONS: Final = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}
)


def _dumps(value: object) -> str:
    # The one body-serialization rule of the codec seam: key order is
    # construction order; never sorted.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _frame(raw: bytes) -> bytes:
    return len(raw).to_bytes(8, "big") + raw


def _framed(value: object) -> bytes:
    return _frame(_dumps(value).encode("utf-8"))


# ---------------------------------------------------------------------------
# encode / finalize / stream_request


def encode(intent: GenerateIntent, contract: ChatModelContract) -> DraftRequest:
    if contract.protocol != "gemini_generate_content":
        raise PlanningDefect(
            code="wrong_protocol",
            message=(
                f"gemini codec received a {contract.protocol!r} contract for target "
                f"{intent.target.provider}/{intent.target.model}"
            ),
        )
    native_level = contract.reasoning.native_mapping.get(intent.reasoning)
    if native_level is None:
        raise PlanningDefect(
            code="unsupported_reasoning_level",
            message=(
                f"reasoning level {intent.reasoning!r} is not supported by "
                f"{intent.target.provider}/{intent.target.model}; supported: "
                f"{sorted(contract.reasoning.native_mapping)}"
            ),
        )
    if intent.tools and isinstance(intent.output, StrictJsonOutput):
        raise PlanningDefect(
            code="tools_with_strict_output",
            message="tools combined with StrictJsonOutput must be rejected at plan time",
        )

    system_parts, contents = _encode_messages(intent)

    generation_config: dict[str, object] = {
        "maxOutputTokens": intent.max_output_tokens,
        "thinkingConfig": {"thinkingLevel": native_level},
    }
    output_schema_config: dict[str, object] | None = None
    if isinstance(intent.output, StrictJsonOutput):
        # StrictJsonOutput.name has no Gemini wire field; the schema itself is
        # the whole native encoding (dialect gemini_response_json_schema).
        schema_json = to_json_schema(
            intent.output.schema, inline_defs=True, include_annotations=True
        )
        output_schema_config = {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema_json,
        }
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseJsonSchema"] = schema_json

    declarations = [
        {
            "name": tool.name,
            "description": tool.description,
            # JSON-Schema-native, UNSTRIPPED (see module docstring).
            "parametersJsonSchema": to_json_schema(
                tool.parameters, inline_defs=True, include_annotations=True
            ),
        }
        for tool in intent.tools
    ]
    tool_config: dict[str, object] = {
        "functionCallingConfig": {"mode": "AUTO" if intent.tool_choice == "auto" else "NONE"}
    }

    body: dict[str, object] = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}
    body["generationConfig"] = generation_config
    if declarations:
        body["tools"] = [{"functionDeclarations": declarations}]
        body["toolConfig"] = tool_config

    # prefix_bytes: length-framed native serializations in the seam's fixed
    # order — stable-prefix projection (systemInstruction-vs-contents
    # placement plus per-content role, not bare parts), tool declarations,
    # tool_choice encoding, output-schema encoding. Fixed section tags
    # separate the four component classes so a stable text byte-equal to a
    # tool-definition dump can never collide across classes. Gemini's
    # implicit cache prefix includes tools, so they are affinity inputs; no
    # cache field ever reaches the wire.
    prefix = bytearray()
    prefix += _frame(b"stable")
    for item in _stable_prefix_projection(intent):
        prefix += _framed(item)
    prefix += _frame(b"tools")
    for declaration in declarations:
        prefix += _framed(declaration)
    prefix += _frame(b"tool_choice")
    if declarations:
        prefix += _framed(tool_config)
    prefix += _frame(b"format")
    if output_schema_config is not None:
        prefix += _framed(output_schema_config)

    return DraftRequest(
        target=intent.target,
        protocol="gemini_generate_content",
        url=f"{_BASE_URL}/{intent.target.model}{_NONSTREAM_SUFFIX}",
        safe_headers={},
        native_reasoning=native_level,
        provider_framing_overhead_tokens=contract.provider_framing_overhead_tokens,
        prefix_bytes=bytes(prefix),
        body=_dumps(body).encode("utf-8"),
    )


def finalize(draft: DraftRequest, affinity: str) -> FinalizedProviderRequest:
    """Passthrough: Gemini caching is implicit, so ``affinity`` is telemetry-only
    (recorded on the plan by the planner) and never injected into the body."""
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
    """Derive the streaming variant of a finalized request (cross-codec seam).

    Gemini selects streaming by URL — ``:streamGenerateContent?alt=sse`` — with
    a byte-identical body."""
    if not request.url.endswith(_NONSTREAM_SUFFIX):
        raise PlanningDefect(
            code="invalid_stream_derivation",
            message=f"gemini stream_request expects a {_NONSTREAM_SUFFIX} url",
        )
    return FinalizedProviderRequest(
        target=request.target,
        protocol=request.protocol,
        url=request.url.removesuffix(_NONSTREAM_SUFFIX) + _STREAM_SUFFIX,
        method=request.method,
        safe_headers=request.safe_headers,
        body=request.body,
    )


def _text_parts(blocks: tuple[PromptBlock, ...]) -> list[dict[str, object]]:
    return [{"text": block.text} for block in blocks] or [{"text": ""}]


def _encode_messages(
    intent: GenerateIntent,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    system_parts: list[dict[str, object]] = []
    contents: list[dict[str, object]] = []
    pending_results: list[dict[str, object]] = []
    # Turn-scoped: reset on every AssistantMessage. Gemini's wire has no call
    # ids, so decode synthesizes call_0, call_1, ... per response — the same
    # ids recur in every turn. Resolving against a single intent-wide map
    # would let a later turn's call_0 overwrite an earlier turn's, so
    # ToolResultMessage.call_id is resolved against ONLY the most recent
    # preceding assistant turn.
    turn_call_names: dict[str, str] = {}

    def flush_results() -> None:
        if pending_results:
            contents.append({"role": "user", "parts": list(pending_results)})
            pending_results.clear()

    for message in intent.messages:
        match message:
            case SystemMessage(blocks=blocks):
                flush_results()
                system_parts.extend({"text": block.text} for block in blocks)
            case UserMessage(blocks=blocks):
                flush_results()
                contents.append({"role": "user", "parts": _text_parts(blocks)})
            case AssistantMessage():
                flush_results()
                contents.append({"role": "model", "parts": _assistant_parts(message, intent)})
                turn_call_names = {}
                for call in message.tool_calls:
                    if call.id in turn_call_names:
                        raise PlanningDefect(
                            code="duplicate_tool_call_id",
                            message=(
                                f"assistant turn carries duplicate tool call id {call.id!r}; "
                                "tool call ids must be unique within a turn"
                            ),
                        )
                    turn_call_names[call.id] = call.name
            case ToolResultMessage(call_id=call_id, output=output, is_error=is_error):
                name = turn_call_names.get(call_id)
                if name is None:
                    raise PlanningDefect(
                        code="unknown_tool_call_id",
                        message=(
                            f"ToolResultMessage.call_id {call_id!r} does not match any "
                            "tool call in the preceding assistant turn"
                        ),
                    )
                pending_results.append(
                    {
                        "functionResponse": {
                            "name": name,
                            "response": {"error": output} if is_error else {"output": output},
                        }
                    }
                )
    flush_results()
    return system_parts, contents


def _assistant_parts(message: AssistantMessage, intent: GenerateIntent) -> list[object]:
    match message.continuation:
        case Present(value=artifact):
            return _replay_parts(artifact, message, intent)
        case _:
            parts: list[object] = []
            if message.text:
                parts.append({"text": message.text})
            parts.extend(
                {"functionCall": {"name": call.name, "args": dict(call.arguments)}}
                for call in message.tool_calls
            )
            return parts or [{"text": ""}]


def _replay_parts(
    artifact: ContinuationArtifact, message: AssistantMessage, intent: GenerateIntent
) -> list[object]:
    if artifact.target != intent.target or artifact.codec_id != CODEC_ID:
        raise PlanningDefect(
            code="continuation_mismatch",
            message=(
                "continuation artifact is bound to "
                f"{artifact.target.provider}/{artifact.target.model} via {artifact.codec_id!r}; "
                f"intent targets {intent.target.provider}/{intent.target.model} via {CODEC_ID!r}"
            ),
        )
    raw_parts = artifact.opaque_payload.get("parts")
    if not isinstance(raw_parts, Sequence) or isinstance(raw_parts, str | bytes):
        raise PlanningDefect(
            code="invalid_continuation_payload",
            message="gemini continuation payload must carry a 'parts' list of part objects",
        )
    parts: list[Mapping[str, object]] = []
    for part in raw_parts:
        if not isinstance(part, Mapping):
            raise PlanningDefect(
                code="invalid_continuation_payload",
                message="gemini continuation payload must carry a 'parts' list of part objects",
            )
        parts.append(part)
    payload_names = [
        function_call.get("name")
        for part in parts
        if isinstance(function_call := part.get("functionCall"), Mapping)
    ]
    typed_names = [call.name for call in message.tool_calls]
    if payload_names != typed_names:
        raise PlanningDefect(
            code="continuation_mismatch",
            message=(
                "typed tool calls must correspond 1:1 in order with the continuation "
                f"payload's functionCall parts (payload {payload_names}, typed {typed_names})"
            ),
        )
    # Verbatim replay: the payload is the sole wire source for this turn (the
    # provider requires the entire prior response's parts back, signatures and
    # all); the typed fields were validated against it above.
    return list(parts)


def _stable_prefix_projection(intent: GenerateIntent) -> list[dict[str, object]]:
    """Placement/role-scoped stable-prefix projection for cache-affinity
    framing: the leading systemInstruction text (if any) as ONE
    ``{"systemInstruction": [...]}`` entry, then each leading user content
    turn's stable parts as its own ``{"role": "user", "parts": [...]}`` entry
    — so systemInstruction-vs-contents placement and per-content role
    participate, and a role move or message regrouping changes the bytes.
    Stops at the first Dynamic block or the first non-system/user message
    (mirrors the wire-body walk; the planner owns contiguity/non-empty)."""
    projection: list[dict[str, object]] = []
    system_parts: list[dict[str, object]] = []
    for message in intent.messages:
        if isinstance(message, SystemMessage):
            for block in message.blocks:
                if not isinstance(block.stability, Stable):
                    if system_parts:
                        projection.append({"systemInstruction": system_parts})
                    return projection
                system_parts.append({"text": block.text})
            continue
        if system_parts:
            projection.append({"systemInstruction": system_parts})
            system_parts = []
        if not isinstance(message, UserMessage):
            return projection
        content_parts: list[dict[str, object]] = []
        for block in message.blocks:
            if not isinstance(block.stability, Stable):
                if content_parts:
                    projection.append({"role": "user", "parts": content_parts})
                return projection
            content_parts.append({"text": block.text})
        if content_parts:
            projection.append({"role": "user", "parts": content_parts})
    if system_parts:
        projection.append({"systemInstruction": system_parts})
    return projection


# ---------------------------------------------------------------------------
# decode (non-stream)


def decode_response(status: int, headers: Mapping[str, str], body: bytes) -> CallOutcome:
    del status, headers  # 2xx only; Gemini carries no header-borne request id.
    data = _parse_json_object(body, code="malformed_response_body")
    meta = _meta(model=_model_version(data), usage=_usage_from(data.get("usageMetadata")))

    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        block_reason = _prompt_block_reason(data)
        if block_reason is not None:
            return Incomplete(
                meta=meta,
                reason="content_filter_partial",
                status="provider_incomplete",
                safe_detail=Present(sanitize_provider_text(f"prompt blocked: {block_reason}")),
            )
        raise ProtocolDefect(
            code="missing_candidates",
            message="gemini response carried no candidates and no promptFeedback block reason",
        )
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise ProtocolDefect(code="malformed_candidate", message="candidates[0] is not an object")

    text, tool_calls, raw_parts = _decode_parts(candidate)
    finish_reason = candidate.get("finishReason")
    if not isinstance(finish_reason, str):
        raise ProtocolDefect(
            code="missing_finish_reason",
            message="gemini non-stream candidate carried no finishReason",
        )
    if finish_reason == "STOP":
        return Succeeded(
            meta=meta,
            response=ResponsePayload(
                content=TextContent(text=text, tool_calls=tool_calls),
                continuation=_continuation(raw_parts, tool_calls, meta.model),
            ),
        )
    return _non_stop_terminal(finish_reason, meta)


def _non_stop_terminal(finish_reason: str, meta: CallMeta) -> Incomplete:
    if finish_reason == "MAX_TOKENS":
        return Incomplete(
            meta=meta,
            reason="max_output_tokens",
            status="provider_incomplete",
            safe_detail=Absent(),
        )
    if finish_reason in _CONTENT_FILTER_FINISH_REASONS:
        return Incomplete(
            meta=meta,
            reason="content_filter_partial",
            status="provider_incomplete",
            safe_detail=Present(sanitize_provider_text(f"finishReason: {finish_reason}")),
        )
    if finish_reason == "MALFORMED_FUNCTION_CALL":
        raise ExpectedFailureSignal(
            InvalidToolArguments(safe_detail="gemini reported finishReason MALFORMED_FUNCTION_CALL")
        )
    raise ProtocolDefect(
        code="unknown_finish_reason",
        message=f"gemini finishReason {sanitize_provider_text(finish_reason)!r} is not modeled",
    )


def _decode_parts(
    candidate: Mapping[str, object],
) -> tuple[str, tuple[ToolCall, ...], list[Mapping[str, object]]]:
    content = candidate.get("content")
    raw_parts = content.get("parts", []) if isinstance(content, Mapping) else []
    if not isinstance(raw_parts, list):
        raise ProtocolDefect(
            code="malformed_candidate", message="candidate content.parts must be a list"
        )

    text_segments: list[str] = []
    tool_calls: list[ToolCall] = []
    parts: list[Mapping[str, object]] = []
    for part in raw_parts:
        if not isinstance(part, Mapping):
            raise ProtocolDefect(
                code="malformed_candidate", message="candidate part is not an object"
            )
        parts.append(part)
        if "functionCall" in part:
            tool_calls.append(_tool_call(part["functionCall"], index=len(tool_calls)))
        elif part.get("thought") is True:
            continue  # thought-summary parts are not visible output
        elif "text" in part:
            text_value = part["text"]
            if not isinstance(text_value, str):
                raise ProtocolDefect(
                    code="malformed_candidate", message="text part is not a string"
                )
            text_segments.append(text_value)
    return "".join(text_segments), tuple(tool_calls), parts


def _tool_call(function_call: object, *, index: int) -> ToolCall:
    if not isinstance(function_call, Mapping):
        raise ProtocolDefect(code="malformed_candidate", message="functionCall is not an object")
    name = function_call.get("name")
    if not isinstance(name, str) or not name:
        raise ProtocolDefect(code="malformed_candidate", message="functionCall.name is missing")
    arguments = function_call.get("args", {})
    if not isinstance(arguments, Mapping):
        # Strict parse only, NO repair: a non-object argument payload is the
        # expected invalid-tool-arguments failure.
        raise ExpectedFailureSignal(
            InvalidToolArguments(
                safe_detail=(
                    f"gemini functionCall {name!r} args is "
                    f"{type(arguments).__name__}, not an object"
                )
            )
        )
    # Wire ids do not exist on generateContent; ids are synthesized
    # deterministically in functionCall order.
    return ToolCall(id=f"call_{index}", name=name, arguments=dict(arguments))


def _continuation(
    parts: list[Mapping[str, object]], tool_calls: tuple[ToolCall, ...], model: str
) -> Presence[ContinuationArtifact]:
    if not tool_calls and not any("thoughtSignature" in part for part in parts):
        return Absent()
    return Present(
        ContinuationArtifact(
            target=ProviderTarget(provider="gemini", model=model),
            codec_id=CODEC_ID,
            opaque_payload={"parts": [dict(part) for part in parts]},
        )
    )


def _meta(*, model: str, usage: Presence[TokenUsage]) -> CallMeta:
    return CallMeta(
        provider="gemini",
        model=model,
        provider_request_id=Absent(),  # ALWAYS: no correlation id on this wire
        upstream_provider=Absent(),
        usage=usage,
        attempt_trace=(),
        billability=PossiblyBillable(),
    )


def _model_version(data: Mapping[str, object]) -> str:
    model_version = data.get("modelVersion")
    return model_version if isinstance(model_version, str) else ""


def _prompt_block_reason(data: Mapping[str, object]) -> str | None:
    prompt_feedback = data.get("promptFeedback")
    if isinstance(prompt_feedback, Mapping):
        block_reason = prompt_feedback.get("blockReason")
        if isinstance(block_reason, str) and block_reason:
            return block_reason
    return None


def _usage_from(metadata: object) -> Presence[TokenUsage]:
    if not isinstance(metadata, Mapping):
        return Absent()

    def count(key: str) -> int:
        value = metadata.get(key)
        return value if type(value) is int else 0

    def component(key: str) -> Presence[int]:
        value = metadata.get(key)
        return Present(value) if type(value) is int else Absent()

    return Present(
        TokenUsage.from_components(
            input_tokens=count("promptTokenCount"),
            output_tokens=count("candidatesTokenCount"),
            total_tokens=component("totalTokenCount"),
            reasoning_tokens=component("thoughtsTokenCount"),
            cache_read_input_tokens=component("cachedContentTokenCount"),
            cache_write_input_tokens=Absent(),  # implicit caching bills no writes
        )
    )


def _parse_json_object(raw: bytes | str, *, code: str) -> dict[str, object]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProtocolDefect(
            code=code, message=f"gemini payload is not valid JSON: {error}"
        ) from error
    if not isinstance(data, dict):
        raise ProtocolDefect(code=code, message="gemini payload is not a JSON object")
    return data


# ---------------------------------------------------------------------------
# decode (stream)


async def decode_stream(
    headers: Mapping[str, str], events: AsyncIterator[SseEvent]
) -> AsyncIterator[CodecStreamEvent]:
    """Decode data-only SSE ``GenerateContentResponse`` chunks.

    Text arrives as per-chunk deltas; functionCall parts arrive whole (so
    ToolCallStart and ToolCallDone are emitted together); the final chunk
    carries finishReason + the authoritative cumulative usageMetadata. Exactly
    one ContinuationDelta precedes a Succeeded terminal when the accumulated
    parts warrant replay. A stream that ends without a finishReason chunk
    raises TransientStreamError for the runtime retry boundary."""
    del headers  # no header-borne request id on this wire
    yield StreamStart()

    model = ""
    usage: Presence[TokenUsage] = Absent()
    text_segments: list[str] = []
    tool_calls: list[ToolCall] = []
    parts_accum: list[Mapping[str, object]] = []

    async for event in events:
        data = _parse_json_object(event.data, code="malformed_stream_frame")
        if chunk_model := _model_version(data):
            model = chunk_model
        chunk_usage = _usage_from(data.get("usageMetadata"))
        if isinstance(chunk_usage, Present):
            # Gemini reports cumulative counts; the latest frame folds all
            # previous ones and the terminal meta is built from it.
            usage = chunk_usage

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            block_reason = _prompt_block_reason(data)
            if block_reason is not None:
                yield TerminalEvent(
                    outcome=Incomplete(
                        meta=_meta(model=model, usage=usage),
                        reason="content_filter_partial",
                        status="provider_incomplete",
                        safe_detail=Present(
                            sanitize_provider_text(f"prompt blocked: {block_reason}")
                        ),
                    )
                )
                return
            continue  # usage-only / empty keep-alive frame
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            raise ProtocolDefect(
                code="malformed_stream_frame", message="candidates[0] is not an object"
            )

        chunk_text, chunk_calls, chunk_parts = _decode_parts(candidate)
        parts_accum.extend(chunk_parts)
        if chunk_text:
            text_segments.append(chunk_text)
            yield TextDelta(text=chunk_text)
        for call in chunk_calls:
            # Re-index across the whole stream (chunk-local indices restart).
            stream_call = ToolCall(
                id=f"call_{len(tool_calls)}", name=call.name, arguments=call.arguments
            )
            tool_calls.append(stream_call)
            yield ToolCallStart(call_id=stream_call.id, name=stream_call.name)
            yield ToolCallDone(tool_call=stream_call)

        finish_reason = candidate.get("finishReason")
        if finish_reason is None:
            continue
        if not isinstance(finish_reason, str):
            raise ProtocolDefect(
                code="malformed_stream_frame", message="finishReason is not a string"
            )
        meta = _meta(model=model, usage=usage)
        if finish_reason == "STOP":
            continuation = _continuation(parts_accum, tuple(tool_calls), model)
            if isinstance(continuation, Present):
                yield ContinuationDelta(artifact=continuation.value)
            yield TerminalEvent(
                outcome=Succeeded(
                    meta=meta,
                    response=ResponsePayload(
                        content=TextContent(
                            text="".join(text_segments), tool_calls=tuple(tool_calls)
                        ),
                        continuation=continuation,
                    ),
                )
            )
        else:
            yield TerminalEvent(outcome=_non_stop_terminal(finish_reason, meta))
        return

    raise TransientStreamError(ProviderStreamInterrupted(partial_output=False))


# ---------------------------------------------------------------------------
# classify_error (non-2xx only)


def classify_error(status: int, headers: Mapping[str, str], body: bytes) -> ClassifiedError:
    parsed = _parse_error_body(body)
    error = parsed.get("error") if parsed is not None else None
    error = error if isinstance(error, Mapping) else {}
    rpc_status = error.get("status")
    rpc_status = rpc_status if isinstance(rpc_status, str) else ""
    message = error.get("message")
    message = message if isinstance(message, str) else ""
    snippet = safe_provider_error_body_snippet(parsed, None) or sanitize_provider_text(
        body.decode("utf-8", errors="replace")
    )

    if status == 429 or rpc_status == "RESOURCE_EXHAUSTED":
        return ProviderRateLimit(retry_after=_retry_after(headers))
    if status in (500, 502, 503, 504) or rpc_status in (
        "UNAVAILABLE",
        "INTERNAL",
        "DEADLINE_EXCEEDED",
    ):
        return ProviderHttpUnavailable()
    if (
        status in (401, 403)
        or rpc_status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
        or "api key not valid" in message.lower()
    ):
        raise CredentialRejected(
            message=f"gemini rejected the platform credential (HTTP {status}): {snippet}"
        )
    if rpc_status == "INVALID_ARGUMENT" and "exceeds the maximum" in message.lower():
        # Provider-documented overflow shape (old-codec knowledge preserved):
        # both token-count and payload-size variants say "exceeds the maximum".
        return ProviderContextTooLarge()
    if parsed is None:
        raise ProtocolDefect(
            code="unparseable_error_body",
            message=f"gemini HTTP {status} error body could not be parsed: {snippet}",
        )
    raise RuntimeDefect(
        origin="provider_http",
        code="unclassified_provider_error",
        message=f"gemini HTTP {status} ({rpc_status or 'no rpc status'}): {snippet}",
    )


def _parse_error_body(body: bytes) -> dict[str, object] | None:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _retry_after(headers: Mapping[str, str]) -> Presence[float]:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return Present(float(value))
            except ValueError:
                return Absent()  # HTTP-date form: no seconds value to carry
    return Absent()
