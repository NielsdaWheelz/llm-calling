"""Gemini GenerateContent engine (google-genai SDK) — gemini.

The SDK owns the wire (serialization, transport, SSE, error envelopes); this
module owns classification against the shared taxonomy and the IR mapping.
ONE attempt per call: retryable trouble raises `TransientAttempt`; expected
non-retryable failures return outcome values with a fully populated `CallMeta`;
malformed envelopes raise `ProtocolDefect`; 401/403 raises `CredentialRejected`.

Engine decisions (wire-observable):

- Reasoning: `row.reasoning[level]` is a self-describing wire fragment — a
  mapping of GenerateContent config params merged verbatim into the request
  ({} = nothing sent). The ROW decides the shape (Gemini 3+ rows carry
  `thinking_config.thinking_level`, 2.5-era rows `thinking_budget`); the
  engine never hardcodes model generations. Fragment top-level keys join the
  provider_options collision set; `native_reasoning` records the fragment as
  compact sorted-keys JSON.
- Continuation: a Succeeded turn with functionCall parts or thoughtSignatures
  yields an artifact whose payload is `{"parts": <candidate parts verbatim>}`
  (json-mode dumps: camelCase keys, signatures as base64 — the wire shape).
  Replay sends the entire parts list back as one `role: "model"` turn; the
  payload is the SOLE wire source for that turn and is never inspected beyond
  the `parts` shape check (the SDK revalidates it; base64 signature strings
  round-trip byte-exact, verified against the installed SDK).
- Call ids: the wire has none; decode synthesizes deterministic `call_<index>`
  ids in functionCall order (restarting per response, re-indexed across a
  stream). Those ids recur across turns, so `ToolResultMessage.call_id`
  resolves against ONLY the most recent preceding assistant turn, and
  consecutive tool results coalesce into ONE `role: "user"` turn.
- Blocked content: Gemini has no refusal contract. Blocked finish reasons
  (SAFETY / PROHIBITED_CONTENT / RECITATION / BLOCKLIST / SPII / IMAGE_SAFETY)
  and blocked prompts (promptFeedback.blockReason, no candidates) both map to
  Incomplete(reason="content_filter_partial"); `Refused` is never constructed.
- `finishReason: MALFORMED_FUNCTION_CALL` / `UNEXPECTED_TOOL_CALL` →
  Failed(InvalidToolArguments) — the provider reporting an unusable tool call
  is the same expected failure as a strict argument-parse failure.
- provider_request_id: Absent() ALWAYS (registry correlation "none"; the
  body's responseId is not surfaced).
- provider_options are GenerateContentConfig fields (the SDK's config surface
  is the extension seam); keys the engine maps itself collide → InvalidRequest,
  and keys the SDK cannot express also raise InvalidRequest — this SDK
  validates its config closed, so there is no wire-blind passthrough.
- Usage: promptTokenCount is already cache-INCLUSIVE on this wire (per the
  SDK's own field contract), so no ingress normalization; thoughtsTokenCount
  → reasoning, cachedContentTokenCount → cache_read, no cache writes billed.
- AFC (the SDK's automatic-function-calling agent loop) is disabled on every
  call — never on the wire, it would break the one-attempt contract.
- The SDK's own retry machinery defaults to a single attempt (retry_options
  None → stop_after_attempt(1)); the engine leaves it unset — the runtime is
  the one retry owner.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, assert_never

import httpx
import pydantic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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
    ToolCallDone,
    ToolCallStart,
    ToolResultMessage,
    TransientCause,
    TransportUnavailable,
    UserMessage,
    presence_of,
)

# Config fields this engine maps from core intent fields, plus fields whose
# wire effect the decode contract forecloses (candidate_count: only
# candidates[0] is decoded, so a caller-requested N would be silently
# discarded). Both casings the SDK accepts; a provider_options key in this
# set — or among the row's reasoning-fragment keys, added per call — is an
# override, not an extension → InvalidRequest.
_OWNED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "system_instruction",
        "systemInstruction",
        "max_output_tokens",
        "maxOutputTokens",
        "tools",
        "tool_config",
        "toolConfig",
        "response_mime_type",
        "responseMimeType",
        "response_json_schema",
        "responseJsonSchema",
        "response_schema",
        "responseSchema",
        "candidate_count",
        "candidateCount",
        "http_options",
        "httpOptions",
        "automatic_function_calling",
        "automaticFunctionCalling",
    }
)

# Blocked-content finish reasons (see module docstring for the mapping).
_CONTENT_FILTER_FINISH_REASONS: Final = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}
)

# Pinned explicitly whenever row.base_url is Absent: genai.Client resolves an
# unset HttpOptions.base_url via GOOGLE_GEMINI_BASE_URL (see _base_url.get_base_url
# in the installed SDK) before ever reaching the mldev branch's own hardcoded
# default of the same host — zero-env is a behavioral guarantee, not just a
# source-text gate, so this engine never lets that fallback trigger.
_CANONICAL_BASE_URL: Final = "https://generativelanguage.googleapis.com/"


def _monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def _registry_invalid(row: ModelRow, detail: str) -> RuntimeDefect:
    return RuntimeDefect(
        origin="intent", code="registry_invalid", message=f"row {row.ref!r}: {detail}"
    )


# ---------------------------------------------------------------------------
# Encode — intent + row → typed SDK contents and config. The SDK serializes;
# every content is validated into a `Content` here so a validation failure
# inside the SDK call can only be response-side.


@dataclass(frozen=True, slots=True)
class _Encoded:
    contents: list[genai_types.Content] = field(repr=False)  # may carry replayed payloads
    config: genai_types.GenerateContentConfig = field(repr=False)
    native_reasoning: Presence[str]


def _encode(row: ModelRow, intent: GenerateIntent) -> _Encoded:
    fragment, native_reasoning = _reasoning_fragment(row, intent)
    system_parts, contents = _encode_contents(row, intent)
    config: dict[str, object] = {
        "max_output_tokens": intent.max_output_tokens,
        # SDK-side agent loop, never on the wire; disabled so the one-attempt
        # contract holds.
        "automatic_function_calling": {"disable": True},
    }
    config.update(fragment)
    if system_parts:
        config["system_instruction"] = {"parts": system_parts}
    if intent.tools:
        config["tools"] = [
            genai_types.Tool(
                function_declarations=[
                    genai_types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        # JSON-Schema-native declaration field: nothing stripped.
                        parameters_json_schema=dict(tool.parameters),
                    )
                    for tool in intent.tools
                ]
            )
        ]
        match intent.tool_choice:
            case "auto":
                mode = "AUTO"
            case "none":
                mode = "NONE"
            case _:
                assert_never(intent.tool_choice)
        config["tool_config"] = {"function_calling_config": {"mode": mode}}
    match intent.output:
        case TextOutput():
            pass
        case StrictJsonOutput() as output:
            config["response_mime_type"] = "application/json"
            match row.structured:
                case "native":
                    config["response_json_schema"] = dict(output.schema)
                case "json_mode":
                    # JSON-only constraint; the caller's schema is enforced by
                    # validation (json_out), not the wire.
                    pass
                case _:
                    assert_never(row.structured)
        case _:
            assert_never(intent.output)
    owned = _OWNED_KEYS | fragment.keys()
    for key in intent.provider_options:
        if key in owned:
            raise InvalidRequest(
                message=f"provider_options key {key!r} collides with a config field "
                f"the gemini_generate engine already sends"
            )
    config.update(intent.provider_options)
    try:
        validated = genai_types.GenerateContentConfig.model_validate(config)
    except pydantic.ValidationError:
        # Engine-built fields are typed by construction; only caller options
        # can fail here. Keys only — option values never enter the message.
        raise InvalidRequest(
            message=f"provider_options keys {sorted(intent.provider_options)} do not "
            f"validate as GenerateContent config fields"
        ) from None
    return _Encoded(contents=contents, config=validated, native_reasoning=native_reasoning)


def _reasoning_fragment(
    row: ModelRow, intent: GenerateIntent
) -> tuple[Mapping[str, object], Presence[str]]:
    match row.reasoning:
        case Absent():
            if intent.reasoning != "none":
                raise InvalidRequest(
                    message=f"row {row.ref!r} has no reasoning knob; "
                    f"level {intent.reasoning!r} is not expressible"
                )
            return {}, Absent()
        case Present(value=levels):
            if intent.reasoning not in levels:
                raise InvalidRequest(
                    message=f"reasoning level {intent.reasoning!r} is outside the levels "
                    f"row {row.ref!r} supports"
                )
            fragment = levels[intent.reasoning]
        case _:
            assert_never(row.reasoning)
    if not isinstance(fragment, Mapping):
        raise _registry_invalid(row, "gemini reasoning values must be config-fragment mappings")
    if not fragment:
        return {}, Absent()
    try:
        # Validated alone so a bad row defects as registry_invalid instead of
        # blaming the caller's provider_options at the merged validation.
        genai_types.GenerateContentConfig.model_validate(dict(fragment))
    except pydantic.ValidationError:
        raise _registry_invalid(
            row, "gemini reasoning fragment does not validate as GenerateContent config fields"
        ) from None
    return fragment, Present(json.dumps(dict(fragment), sort_keys=True, separators=(",", ":")))


def _encode_contents(
    row: ModelRow, intent: GenerateIntent
) -> tuple[list[dict[str, object]], list[genai_types.Content]]:
    system_parts: list[dict[str, object]] = []
    contents: list[genai_types.Content] = []
    pending_results: list[dict[str, object]] = []
    # Turn-scoped: reset on every AssistantMessage — synthesized call ids
    # recur across turns, so an intent-wide map would let a later turn's
    # call_0 shadow an earlier one.
    turn_call_names: dict[str, str] = {}

    def flush_results() -> None:
        if pending_results:
            contents.append(_content({"role": "user", "parts": list(pending_results)}))
            pending_results.clear()

    for message in intent.messages:
        match message:
            case SystemMessage(blocks=blocks):
                flush_results()
                system_parts.extend({"text": block.text} for block in blocks)
            case UserMessage(blocks=blocks):
                flush_results()
                contents.append(_content({"role": "user", "parts": _user_parts(blocks)}))
            case AssistantMessage() as assistant:
                flush_results()
                contents.append(_assistant_content(assistant, row, intent))
                turn_call_names = {}
                for call in assistant.tool_calls:
                    if call.id in turn_call_names:
                        raise InvalidRequest(
                            message=f"assistant turn carries duplicate tool call id {call.id!r}; "
                            "tool call ids must be unique within a turn"
                        )
                    turn_call_names[call.id] = call.name
            case ToolResultMessage(call_id=call_id, output=output, is_error=is_error):
                name = turn_call_names.get(call_id)
                if name is None:
                    raise InvalidRequest(
                        message=f"ToolResultMessage.call_id {call_id!r} does not match any "
                        "tool call in the preceding assistant turn"
                    )
                pending_results.append(
                    {
                        "function_response": {
                            "name": name,
                            "response": {"error": output} if is_error else {"output": output},
                        }
                    }
                )
            case _:
                assert_never(message)
    flush_results()
    return system_parts, contents


def _content(raw: Mapping[str, object]) -> genai_types.Content:
    # Engine-built dicts come from typed IR — a failure here is an engine bug
    # and raises raw (a defect, loudly).
    return genai_types.Content.model_validate(raw)


def _user_parts(blocks: tuple[PromptBlock | ImageBlock, ...]) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    for block in blocks:
        match block:
            case PromptBlock(text=text):
                parts.append({"text": text})
            case ImageBlock(media_type=media_type, data=data):
                parts.append({"inline_data": {"mime_type": media_type, "data": data}})
            case _:
                assert_never(block)
    return parts or [{"text": ""}]


def _assistant_content(
    message: AssistantMessage, row: ModelRow, intent: GenerateIntent
) -> genai_types.Content:
    match message.continuation:
        case Present(value=artifact):
            return _replay_content(artifact, row, intent)
        case Absent():
            parts: list[dict[str, object]] = []
            if message.text:
                parts.append({"text": message.text})
            parts.extend(
                {"function_call": {"name": call.name, "args": dict(call.arguments)}}
                for call in message.tool_calls
            )
            return _content({"role": "model", "parts": parts or [{"text": ""}]})
        case _:
            assert_never(message.continuation)


def _replay_content(
    artifact: ContinuationArtifact, row: ModelRow, intent: GenerateIntent
) -> genai_types.Content:
    if artifact.target != intent.target or artifact.codec_id != row.continuation_codec:
        raise InvalidRequest(
            message=f"continuation artifact for {artifact.target.provider}/"
            f"{artifact.target.model} (codec {artifact.codec_id!r}) cannot replay to "
            f"{intent.target.provider}/{intent.target.model} (codec {row.continuation_codec!r})"
        )
    parts = artifact.opaque_payload.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, str | bytes):
        raise InvalidRequest(
            message="gemini continuation payload must carry a 'parts' list of part objects"
        )
    try:
        # The payload is the sole wire source for this turn (the provider
        # requires the entire prior response's parts back, signatures and
        # all); the SDK revalidates it — never this engine.
        return genai_types.Content.model_validate({"role": "model", "parts": list(parts)})
    except pydantic.ValidationError:
        raise InvalidRequest(
            message="gemini continuation payload parts do not validate as native parts"
        ) from None


# ---------------------------------------------------------------------------
# Usage — cumulative frames folded field-wise: later non-zero wins (proto3
# JSON omits zero-valued fields, so 0 and null are indistinguishable on this
# wire), normalized ONCE into the cache-INCLUSIVE TokenUsage at the terminal.


@dataclass(slots=True)
class _UsageFold:
    prompt: int | None = None
    candidates: int | None = None
    total: int | None = None
    thoughts: int | None = None
    cached: int | None = None
    seen: bool = False

    def fold(self, metadata: genai_types.GenerateContentResponseUsageMetadata) -> None:
        self.seen = True
        self.prompt = metadata.prompt_token_count or self.prompt
        self.candidates = metadata.candidates_token_count or self.candidates
        self.total = metadata.total_token_count or self.total
        self.thoughts = metadata.thoughts_token_count or self.thoughts
        self.cached = metadata.cached_content_token_count or self.cached

    def usage(self) -> Presence[TokenUsage]:
        if not self.seen:
            return Absent()
        # promptTokenCount is already cache-inclusive on this wire; the
        # provider-reported total is authoritative.
        return Present(
            TokenUsage.from_components(
                input_tokens=self.prompt or 0,
                output_tokens=self.candidates or 0,
                total_tokens=presence_of(self.total),
                reasoning_tokens=presence_of(self.thoughts),
                cache_read_input_tokens=presence_of(self.cached),
                cache_write_input_tokens=Absent(),  # implicit caching bills no writes
            )
        )


def _usage_presence(
    metadata: genai_types.GenerateContentResponseUsageMetadata | None,
) -> Presence[TokenUsage]:
    if metadata is None:
        return Absent()
    fold = _UsageFold()
    fold.fold(metadata)
    return fold.usage()


# ---------------------------------------------------------------------------
# Decode — candidate parts → text/tool calls + verbatim part dicts for the
# continuation payload. The output arm follows the intent's OutputSpec.


def _decode_parts(
    candidate: genai_types.Candidate,
) -> tuple[str, tuple[ToolCall, ...], list[Mapping[str, object]]]:
    content = candidate.content
    raw_parts = content.parts if content is not None and content.parts is not None else []
    text_segments: list[str] = []
    calls: list[ToolCall] = []
    dumped: list[Mapping[str, object]] = []
    for part in raw_parts:
        # json-mode dump = the wire shape: camelCase keys, base64 signatures.
        dumped.append(part.model_dump(mode="json", by_alias=True, exclude_none=True))
        function_call = part.function_call
        if function_call is not None:
            if not function_call.name:
                raise ProtocolDefect(
                    code="malformed_candidate", message="functionCall.name is missing"
                )
            # No wire ids on generateContent: ids are synthesized
            # deterministically in functionCall order.
            calls.append(
                ToolCall(
                    id=f"call_{len(calls)}",
                    name=function_call.name,
                    arguments=dict(function_call.args or {}),
                )
            )
        elif part.thought:
            continue  # thought-summary parts are not visible output
        elif part.text is not None:
            text_segments.append(part.text)
    return "".join(text_segments), tuple(calls), dumped


def _response_content(
    intent: GenerateIntent, text: str, tool_calls: tuple[ToolCall, ...]
) -> ResponseContent | InvalidStructuredOutput:
    match intent.output:
        case TextOutput():
            return TextContent(text=text, tool_calls=tool_calls)
        case StrictJsonOutput():
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                return InvalidStructuredOutput(
                    safe_detail=f"structured output is not valid JSON ({exc.msg} at char {exc.pos})"
                )
            if not isinstance(payload, dict):
                return InvalidStructuredOutput(
                    safe_detail=f"structured output parsed to {type(payload).__name__}, "
                    f"not a JSON object"
                )
            return StructuredContent(payload=payload, text=text)
        case _:
            assert_never(intent.output)


def _continuation(
    parts: list[Mapping[str, object]],
    tool_calls: tuple[ToolCall, ...],
    row: ModelRow,
    intent: GenerateIntent,
) -> Presence[ContinuationArtifact]:
    if not tool_calls and not any("thoughtSignature" in part for part in parts):
        return Absent()
    return Present(
        ContinuationArtifact(
            target=intent.target,
            codec_id=row.continuation_codec,
            opaque_payload={"parts": list(parts)},
        )
    )


def _finish_value(candidate: genai_types.Candidate) -> str | None:
    finish = candidate.finish_reason
    return None if finish is None else str(finish.value)


def _block_reason(response: genai_types.GenerateContentResponse) -> str | None:
    feedback = response.prompt_feedback
    if feedback is None or feedback.block_reason is None:
        return None
    return str(feedback.block_reason.value)


def _blocked_incomplete(meta: CallMeta, detail: str) -> Incomplete:
    return Incomplete(
        meta=meta,
        reason="content_filter_partial",
        status="provider_incomplete",
        safe_detail=Present(sanitize_provider_text(detail)),
    )


def _non_stop_outcome(finish: str, meta: CallMeta) -> Incomplete | Failed:
    if finish == "MAX_TOKENS":
        return Incomplete(
            meta=meta,
            reason="max_output_tokens",
            status="provider_incomplete",
            safe_detail=Absent(),
        )
    if finish in _CONTENT_FILTER_FINISH_REASONS:
        return _blocked_incomplete(meta, f"finishReason: {finish}")
    if finish in ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"):
        # Both mean the model produced an unusable tool call — the same
        # expected failure as a strict argument-parse failure.
        return Failed(
            meta=meta,
            failure=InvalidToolArguments(safe_detail=f"gemini reported finishReason {finish}"),
        )
    raise ProtocolDefect(
        code="unknown_finish_reason",
        message=f"gemini finishReason {sanitize_provider_text(finish)!r} is not modeled",
    )


# ---------------------------------------------------------------------------
# Error classification — SDK exception → TransientAttempt / defect / value.


def _validation_summary(exc: pydantic.ValidationError) -> str:
    """Error locations and types only — response payload values (which the
    ValidationError's own message embeds) never enter a defect message."""
    return "; ".join(
        f"{'.'.join(str(item) for item in err['loc'])}: {err['type']}" for err in exc.errors()[:3]
    )


def _retry_after(exc: genai_errors.APIError) -> Presence[float]:
    """Numeric Retry-After seconds; the HTTP-date form has no consumer → Absent."""
    if not isinstance(exc.response, httpx.Response):
        return Absent()
    value = exc.response.headers.get("retry-after")
    if value is None:
        return Absent()
    try:
        return Present(float(value))
    except ValueError:
        return Absent()


def _classify_api_error(exc: genai_errors.APIError) -> ProviderContextTooLarge:
    """Classify a non-2xx response: raises for everything except the one
    non-retryable expected failure (context overflow), which returns."""
    code = exc.code if isinstance(exc.code, int) else None
    status = exc.status if isinstance(exc.status, str) else ""
    message = exc.message if isinstance(exc.message, str) else ""
    details = exc.details if isinstance(exc.details, dict) else None
    snippet = safe_provider_error_body_snippet(details)
    detail = f": {snippet}" if snippet else ""

    if code == 429 or status == "RESOURCE_EXHAUSTED":
        raise TransientAttempt(
            cause=ProviderRateLimit(retry_after=_retry_after(exc)),
            status_code=presence_of(code),
            provider_request_id=Absent(),
            billability=PossiblyBillable(),
        )
    if (code is not None and code >= 500) or status in (
        "UNAVAILABLE",
        "INTERNAL",
        "DEADLINE_EXCEEDED",
    ):
        raise TransientAttempt(
            cause=ProviderHttpUnavailable(),
            status_code=presence_of(code),
            provider_request_id=Absent(),
            billability=PossiblyBillable(),
        )
    if (
        code in (401, 403)
        or status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
        or "api key not valid" in message.lower()
    ):
        raise CredentialRejected(
            message=f"gemini rejected the platform credential (HTTP {code}){detail}"
        )
    if status == "INVALID_ARGUMENT" and "exceeds the maximum" in message.lower():
        # Provider-documented overflow shape: both the token-count and
        # payload-size variants say "exceeds the maximum".
        return ProviderContextTooLarge()
    raise RuntimeDefect(
        origin="provider_http",
        code="unclassified_provider_error",
        message=f"gemini HTTP {code} ({status or 'no rpc status'}){detail}",
    )


def _inband_cause(exc: genai_errors.APIError) -> TransientCause:
    """Mid-stream in-band error frames (HTTP stayed 200), code parity with
    `_classify_api_error`: 429 → ProviderRateLimit, 5xx →
    ProviderHttpUnavailable, anything else is a non-transient error inside a
    2xx stream → ProtocolDefect."""
    code = exc.code if isinstance(exc.code, int) else None
    if code == 429:
        return ProviderRateLimit(retry_after=Absent())
    if code is not None and code >= 500:
        return ProviderHttpUnavailable()
    snippet = safe_provider_error_body_snippet(
        exc.details if isinstance(exc.details, dict) else None
    )
    raise ProtocolDefect(
        code="inband_error_frame",
        message=f"gemini stream carried a non-transient in-band error frame "
        f"(code {code}){f': {snippet}' if snippet else ''}",
    ) from None


def _transport_attempt(exc: httpx.TransportError) -> TransientAttempt:
    # A pure pre-connect failure means no bytes reached the provider; every
    # other transport error implies the connection was at least opened.
    dispatched = not isinstance(exc, httpx.ConnectError)
    return TransientAttempt(
        cause=TransportUnavailable(),
        status_code=Absent(),
        provider_request_id=Absent(),
        billability=PossiblyBillable() if dispatched else NotDispatched(),
    )


def _timeout_attempt(exc: httpx.TimeoutException) -> TransientAttempt:
    # A connect timeout is a pure pre-connect failure — the handshake never
    # completed, so no request bytes reached the provider.
    dispatched = not isinstance(exc, httpx.ConnectTimeout)
    return TransientAttempt(
        cause=ProviderTimeout(),
        status_code=Absent(),
        provider_request_id=Absent(),
        billability=PossiblyBillable() if dispatched else NotDispatched(),
    )


# ---------------------------------------------------------------------------
# Engine


class GeminiGenerateEngine:
    """One attempt per call against Gemini; the runtime owns retries."""

    def __init__(
        self, *, timeout_s: float = 600.0, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self._timeout_s = timeout_s
        self._http_client = http_client

    def _client(self, row: ModelRow, credential: ProviderCredential) -> genai.Client:
        match row.base_url:
            case Present(value=base_url):
                pass
            case Absent():
                base_url = _CANONICAL_BASE_URL  # pinned, never SDK/env-resolved
            case _:
                assert_never(row.base_url)
        return genai.Client(
            vertexai=False,  # explicit: never let SDK env sniffing flip the backend
            api_key=credential.key,
            http_options=genai_types.HttpOptions(
                base_url=base_url,
                timeout=int(self._timeout_s * 1000),  # HttpOptions.timeout is milliseconds
                httpx_async_client=self._http_client,
            ),
        )

    async def generate(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> CallOutcome:
        encoded = _encode(row, intent)
        client = self._client(row, credential)
        started_ms = _monotonic_ms()
        try:
            try:
                response = await client.aio.models.generate_content(
                    model=row.model_id, contents=list(encoded.contents), config=encoded.config
                )
            except genai_errors.APIError as exc:
                overflow = _classify_api_error(exc)
                return Failed(
                    meta=_error_meta(row, exc, encoded.native_reasoning, started_ms),
                    failure=overflow,
                )
            except httpx.TimeoutException as exc:
                raise _timeout_attempt(exc) from exc
            except httpx.TransportError as exc:
                raise _transport_attempt(exc) from exc
            except json.JSONDecodeError as exc:
                raise ProtocolDefect(
                    code="malformed_json", message="gemini 2xx response body is not valid JSON"
                ) from exc
            except pydantic.ValidationError as exc:
                # Request-side validation cannot reach here: contents and
                # config were validated in _encode. The cause embeds response
                # payloads — never chained; locations + error types stand in.
                raise ProtocolDefect(
                    code="malformed_envelope",
                    message="gemini 2xx response does not validate as a "
                    f"GenerateContentResponse ({_validation_summary(exc)})",
                ) from None
            return _decode_response(row, intent, response, encoded.native_reasoning, started_ms)
        finally:
            # The SDK skips closing an injected httpx client.
            await client.aio.aclose()

    async def stream(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> AsyncIterator[CodecStreamEvent]:
        encoded = _encode(row, intent)
        client = self._client(row, credential)
        started_ms = _monotonic_ms()
        try:
            try:
                chunks = await client.aio.models.generate_content_stream(
                    model=row.model_id, contents=list(encoded.contents), config=encoded.config
                )
            except genai_errors.APIError as exc:
                overflow = _classify_api_error(exc)
                yield TerminalEvent(
                    outcome=Failed(
                        meta=_error_meta(row, exc, encoded.native_reasoning, started_ms),
                        failure=overflow,
                    )
                )
                return
            except httpx.TimeoutException as exc:
                raise _timeout_attempt(exc) from exc
            except httpx.TransportError as exc:
                raise _transport_attempt(exc) from exc

            # Provider accepted the request (headers + 2xx) — the envelope opens.
            yield StreamStart()

            semantic = False
            model = row.model_id
            fold = _UsageFold()
            text_segments: list[str] = []
            tool_calls: list[ToolCall] = []
            parts_accum: list[Mapping[str, object]] = []

            def interrupted(cause: TransientCause) -> TransientAttempt:
                # Post-semantic-output ANY transient is terminal for the
                # runtime; only the engine knows what it already yielded.
                final = ProviderStreamInterrupted(partial_output=True) if semantic else cause
                return TransientAttempt(
                    cause=final,
                    status_code=Present(200),
                    provider_request_id=Absent(),
                    billability=PossiblyBillable(),
                )

            def stream_meta() -> CallMeta:
                return _meta(
                    row,
                    model=model,
                    usage=fold.usage(),
                    native_reasoning=encoded.native_reasoning,
                    started_ms=started_ms,
                    status_code=Present(200),
                )

            try:
                async for chunk in chunks:
                    if chunk.model_version:
                        model = chunk.model_version
                    if chunk.usage_metadata is not None:
                        # Cumulative counts fold silently; UsageEvent is never
                        # emitted on this wire (usage rides most frames and
                        # would needlessly block pre-output stream retries).
                        fold.fold(chunk.usage_metadata)
                    candidates = chunk.candidates
                    if not candidates:
                        reason = _block_reason(chunk)
                        if reason is not None:
                            yield TerminalEvent(
                                outcome=_blocked_incomplete(
                                    stream_meta(), f"prompt blocked: {reason}"
                                )
                            )
                            return
                        continue  # usage-only / empty keep-alive frame
                    candidate = candidates[0]

                    chunk_text, chunk_calls, chunk_parts = _decode_parts(candidate)
                    parts_accum.extend(chunk_parts)
                    if chunk_text:
                        text_segments.append(chunk_text)
                        yield TextDelta(text=chunk_text)
                        semantic = True
                    for call in chunk_calls:
                        # functionCall parts arrive whole; re-index across the
                        # whole stream (chunk-local indices restart at 0).
                        stream_call = ToolCall(
                            id=f"call_{len(tool_calls)}", name=call.name, arguments=call.arguments
                        )
                        tool_calls.append(stream_call)
                        yield ToolCallStart(call_id=stream_call.id, name=stream_call.name)
                        yield ToolCallDone(tool_call=stream_call)
                        semantic = True

                    finish = _finish_value(candidate)
                    if finish is None:
                        continue
                    if finish == "STOP":
                        text = "".join(text_segments)
                        content = _response_content(intent, text, tuple(tool_calls))
                        if isinstance(content, InvalidStructuredOutput):
                            yield TerminalEvent(outcome=Failed(meta=stream_meta(), failure=content))
                            return
                        continuation = _continuation(parts_accum, tuple(tool_calls), row, intent)
                        if isinstance(continuation, Present):
                            # AT MOST ONE, after all contributing parts are
                            # final, before the terminal.
                            yield ContinuationDelta(artifact=continuation.value)
                        yield TerminalEvent(
                            outcome=Succeeded(
                                meta=stream_meta(),
                                response=ResponsePayload(
                                    content=content, continuation=continuation
                                ),
                            )
                        )
                    else:
                        yield TerminalEvent(outcome=_non_stop_outcome(finish, stream_meta()))
                    return
            except genai_errors.APIError as exc:
                # In-band error frame (HTTP stayed 200), surfaced by the SDK
                # mid-iteration.
                raise interrupted(_inband_cause(exc)) from exc
            except (genai_errors.UnknownApiResponseError, json.JSONDecodeError):
                # The SDK's frame-parse error embeds the raw frame verbatim —
                # never chained.
                raise ProtocolDefect(
                    code="malformed_stream_frame",
                    message="gemini stream frame is not valid JSON",
                ) from None
            except pydantic.ValidationError as exc:
                # The cause embeds frame payloads — never chained; locations +
                # error types stand in.
                raise ProtocolDefect(
                    code="malformed_stream_frame",
                    message="gemini stream frame does not validate as a "
                    f"GenerateContentResponse ({_validation_summary(exc)})",
                ) from None
            except httpx.TimeoutException as exc:
                raise interrupted(ProviderTimeout()) from exc
            except httpx.TransportError as exc:
                raise interrupted(TransportUnavailable()) from exc

            # The stream ended without a terminal frame — a cut, not a close.
            raise interrupted(ProviderStreamInterrupted(partial_output=False))
        finally:
            await client.aio.aclose()


# ---------------------------------------------------------------------------
# Meta + terminal construction.


def _meta(
    row: ModelRow,
    *,
    model: str,
    usage: Presence[TokenUsage],
    native_reasoning: Presence[str],
    started_ms: int,
    status_code: Presence[int],
) -> CallMeta:
    return CallMeta(
        provider=row.provider,
        model=model,
        provider_request_id=Absent(),  # ALWAYS: correlation "none" on this wire
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
        billability=PossiblyBillable(),
        native_reasoning=native_reasoning,
        registry_revision=REGISTRY_REVISION,
    )


def _error_meta(
    row: ModelRow,
    exc: genai_errors.APIError,
    native_reasoning: Presence[str],
    started_ms: int,
) -> CallMeta:
    # No envelope decoded: the row's model id stands in, usage is Absent.
    return _meta(
        row,
        model=row.model_id,
        usage=Absent(),
        native_reasoning=native_reasoning,
        started_ms=started_ms,
        # One normalization rule with _classify_api_error: presence_of for
        # non-int codes — never a fabricated status.
        status_code=presence_of(exc.code if isinstance(exc.code, int) else None),
    )


def _decode_response(
    row: ModelRow,
    intent: GenerateIntent,
    response: genai_types.GenerateContentResponse,
    native_reasoning: Presence[str],
    started_ms: int,
) -> CallOutcome:
    meta = _meta(
        row,
        # modelVersion echo when present; the row's wire id otherwise.
        model=response.model_version or row.model_id,
        usage=_usage_presence(response.usage_metadata),
        native_reasoning=native_reasoning,
        started_ms=started_ms,
        status_code=Present(200),
    )
    candidates = response.candidates
    if not candidates:
        reason = _block_reason(response)
        if reason is not None:
            return _blocked_incomplete(meta, f"prompt blocked: {reason}")
        raise ProtocolDefect(
            code="missing_candidates",
            message="gemini response carried no candidates and no promptFeedback block reason",
        )
    candidate = candidates[0]
    text, tool_calls, parts = _decode_parts(candidate)
    finish = _finish_value(candidate)
    if finish is None:
        raise ProtocolDefect(
            code="missing_finish_reason",
            message="gemini non-stream candidate carried no finishReason",
        )
    if finish != "STOP":
        return _non_stop_outcome(finish, meta)
    content = _response_content(intent, text, tool_calls)
    if isinstance(content, InvalidStructuredOutput):
        return Failed(meta=meta, failure=content)
    return Succeeded(
        meta=meta,
        response=ResponsePayload(
            content=content, continuation=_continuation(parts, tool_calls, row, intent)
        ),
    )
