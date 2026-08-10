"""Anthropic Messages engine conformance + fault injection (HTTP boundary via respx).

Request tests assert EXACT body dicts as the SDK puts them on the wire; decode
tests feed canned envelopes; stream tests feed raw SSE bytes. No internal
mocking anywhere — respx intercepts the SDK's own httpx transport.
"""

import json
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import replace

import httpx
import pytest
import respx

from provider_runtime.engines import TransientAttempt
from provider_runtime.engines.anthropic_messages import AnthropicMessagesEngine
from provider_runtime.errors import (
    CredentialRejected,
    InvalidRequest,
    ProtocolDefect,
    RuntimeDefect,
)
from provider_runtime.registry import REGISTRY_REVISION, ModelRow
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    Billability,
    CanonicalTool,
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
    OutputSpec,
    PossiblyBillable,
    Present,
    PromptBlock,
    PromptMessage,
    ProviderContextTooLarge,
    ProviderCredential,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTarget,
    ProviderTimeout,
    ReasoningLevel,
    Refused,
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
    ToolChoice,
    ToolResultMessage,
    TransportUnavailable,
    UsageEvent,
    UserMessage,
)

MESSAGES_URL = "https://api.anthropic.com/v1/messages"

REASONING_LEVELS: Mapping[ReasoningLevel, object] = {"low": 1024, "high": 8192}

ROW = ModelRow(
    ref="anthropic:claude-test",
    provider="anthropic",
    model_id="claude-test-1",
    engine="anthropic_messages",
    base_url=Present("https://api.anthropic.com"),
    context_window=1_000_000,
    max_output_tokens=64_000,
    modalities=frozenset({"text", "image"}),
    tools=True,
    streaming=True,
    structured="native",
    reasoning=Present(REASONING_LEVELS),
    continuation_codec="anthropic.v1",
    correlation="header",
    routing=Absent(),
)

TARGET = ProviderTarget(provider="anthropic", model="claude-test-1")
CREDENTIAL = ProviderCredential(provider="anthropic", key="sk-ant-test-not-a-real-key-123456")

SYSTEM = SystemMessage(blocks=(PromptBlock(text="You are terse."),))
USER = UserMessage(blocks=(PromptBlock(text="hi"),))

SEARCH_TOOL = CanonicalTool(
    name="search_library",
    description="Search the library",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

TEXT_OUTPUT = TextOutput()

VERDICT_OUTPUT = StrictJsonOutput(
    name="verdict",
    schema={
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
        "additionalProperties": False,
    },
)


def make_intent(
    *,
    messages: tuple[PromptMessage, ...] = (SYSTEM, USER),
    max_output_tokens: int = 128,
    reasoning: ReasoningLevel = "high",
    tools: tuple[CanonicalTool, ...] = (),
    tool_choice: ToolChoice = "auto",
    output: OutputSpec = TEXT_OUTPUT,
    provider_options: Mapping[str, object] | None = None,
    target: ProviderTarget = TARGET,
) -> GenerateIntent:
    return GenerateIntent(
        target=target,
        messages=messages,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        tools=tools,
        tool_choice=tool_choice,
        output=output,
        provider_options={} if provider_options is None else provider_options,
    )


# Anthropic's wire input_tokens EXCLUDES cache components: 20 uncached + 80
# cache-read + 8 cache-write must normalize to the inclusive 108.
def usage_body(*, output_tokens: int = 30) -> dict[str, object]:
    return {
        "input_tokens": 20,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": 80,
        "cache_creation_input_tokens": 8,
        "output_tokens_details": {"thinking_tokens": 12},
    }


EXPECTED_USAGE = TokenUsage(
    input_tokens=108,
    output_tokens=30,
    total_tokens=138,
    reasoning_tokens=Present(12),
    cache_read_input_tokens=Present(80),
    cache_write_input_tokens=Present(8),
)

THINKING_BLOCK: dict[str, object] = {
    "type": "thinking",
    "thinking": "pondering",
    "signature": "sig-1",
}
REDACTED_BLOCK: dict[str, object] = {"type": "redacted_thinking", "data": "opaque-redacted"}
TEXT_BLOCK: dict[str, object] = {"type": "text", "text": "hello"}
TOOL_USE_BLOCK: dict[str, object] = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "search_library",
    "input": {"query": "cats"},
}


def envelope(
    *,
    content: list[dict[str, object]],
    stop_reason: str | None = "end_turn",
    usage: dict[str, object] | None = None,
    stop_details: dict[str, object] | None = None,
    model: str | None = "claude-test-1",
    message_id: str = "msg_123",
) -> dict[str, object]:
    body: dict[str, object] = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
    }
    if model is not None:
        body["model"] = model
    if usage is not None:
        body["usage"] = usage
    if stop_details is not None:
        body["stop_details"] = stop_details
    return body


def mock_response(body: dict[str, object], *, request_id: str | None = "req_abc") -> httpx.Response:
    headers = {"content-type": "application/json"}
    if request_id is not None:
        headers["request-id"] = request_id
    return httpx.Response(200, headers=headers, json=body)


def request_body(route: respx.Route) -> dict[str, object]:
    request = route.calls.last.request
    parsed = json.loads(request.content)
    assert isinstance(parsed, dict), f"request body is not a JSON object: {request.content!r}"
    return parsed


def sse_bytes(frames: list[dict[str, object]]) -> bytes:
    chunks = [f"event: {frame['type']}\ndata: {json.dumps(frame)}\n\n".encode() for frame in frames]
    return b"".join(chunks)


def mock_stream(frames: list[dict[str, object]], *, request_id: str = "req_s1") -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream", "request-id": request_id},
        content=sse_bytes(frames),
    )


START_USAGE: dict[str, object] = {
    "input_tokens": 20,
    "output_tokens": 1,
    "cache_read_input_tokens": 80,
    "cache_creation_input_tokens": 8,
}
DELTA_USAGE: dict[str, object] = {
    "output_tokens": 30,
    "output_tokens_details": {"thinking_tokens": 12},
}


def message_start_frame(
    *,
    usage: dict[str, object] | None = None,
    model: str = "claude-test-1",
    message_id: str = "msg_s1",
) -> dict[str, object]:
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": dict(START_USAGE) if usage is None else usage,
        },
    }


def message_delta_frame(
    *,
    stop_reason: str | None,
    usage: dict[str, object] | None = None,
    stop_details: dict[str, object] | None = None,
) -> dict[str, object]:
    delta: dict[str, object] = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    return {
        "type": "message_delta",
        "delta": delta,
        "usage": dict(DELTA_USAGE) if usage is None else usage,
    }


def text_delta_frame(index: int, text: str) -> dict[str, object]:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }


async def collect_stream(
    engine: AnthropicMessagesEngine, intent: GenerateIntent, *, row: ModelRow = ROW
) -> list[CodecStreamEvent]:
    return [event async for event in engine.stream(row, intent, CREDENTIAL)]


def assert_meta(
    outcome: Succeeded | Refused | Incomplete | Failed,
    *,
    request_id: str = "req_abc",
    usage: TokenUsage | None = EXPECTED_USAGE,
    native_reasoning: str | None = "budget_tokens=8192",
    status_code: int = 200,
    model: str = "claude-test-1",
    billability: Billability | None = None,
) -> None:
    billability = PossiblyBillable() if billability is None else billability
    meta = outcome.meta
    assert meta.provider == "anthropic", f"meta.provider: {meta.provider!r}"
    assert meta.model == model, f"meta.model: {meta.model!r}"
    assert meta.provider_request_id == Present(request_id), (
        f"meta.provider_request_id: {meta.provider_request_id!r}"
    )
    assert meta.upstream_provider == Absent(), f"meta.upstream_provider: {meta.upstream_provider!r}"
    expected_usage = Absent() if usage is None else Present(usage)
    assert meta.usage == expected_usage, f"meta.usage: {meta.usage!r} != {expected_usage!r}"
    expected_native = Absent() if native_reasoning is None else Present(native_reasoning)
    assert meta.native_reasoning == expected_native, (
        f"meta.native_reasoning: {meta.native_reasoning!r} != {expected_native!r}"
    )
    assert meta.registry_revision == REGISTRY_REVISION, (
        f"meta.registry_revision: {meta.registry_revision!r}"
    )
    assert meta.billability == billability, (
        f"meta.billability: {meta.billability!r} != {billability!r}"
    )
    assert len(meta.attempt_trace) == 1, f"attempt_trace: {meta.attempt_trace!r}"
    record = meta.attempt_trace[0]
    assert record.attempt == 1, f"attempt: {record.attempt}"
    assert record.signal == FinalAttempt(), f"signal: {record.signal!r}"
    assert record.status_code == Present(status_code), f"status_code: {record.status_code!r}"
    assert record.ended_at_ms >= record.started_at_ms, f"attempt record times: {record!r}"


# ---------------------------------------------------------------------------
# Request conformance


@respx.mock
async def test_generate_sends_exact_request_body_headers_and_url() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert route.call_count == 1, f"expected exactly one dispatch; got {route.call_count}"
    request = route.calls.last.request
    assert request.headers["x-api-key"] == CREDENTIAL.key, (
        f"x-api-key header: {request.headers.get('x-api-key')!r}"
    )
    assert request.headers["anthropic-version"] == "2023-06-01", (
        f"anthropic-version header: {request.headers.get('anthropic-version')!r}"
    )
    body = request_body(route)
    assert body == {
        "model": "claude-test-1",
        "max_tokens": 128,
        "system": [
            {
                "type": "text",
                "text": "You are terse.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "thinking": {"type": "enabled", "budget_tokens": 8192},
    }, f"request body: {body!r}"


@respx.mock
async def test_generate_places_cache_breakpoint_on_last_system_block_only() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    system = SystemMessage(blocks=(PromptBlock(text="Rules."), PromptBlock(text="Corpus.")))
    outcome = await AnthropicMessagesEngine().generate(
        ROW, make_intent(messages=(system, USER)), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["system"] == [
        {"type": "text", "text": "Rules."},
        {"type": "text", "text": "Corpus.", "cache_control": {"type": "ephemeral"}},
    ], f"system: {body.get('system')!r}"


@respx.mock
async def test_generate_without_system_omits_system_and_breakpoint() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    outcome = await AnthropicMessagesEngine().generate(
        ROW, make_intent(messages=(USER,)), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert "system" not in body, f"system key must be omitted; body: {body!r}"


@respx.mock
async def test_generate_drops_empty_text_blocks_from_the_wire() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    system = SystemMessage(blocks=(PromptBlock(text="Rules."), PromptBlock(text="")))
    user = UserMessage(blocks=(PromptBlock(text=""), PromptBlock(text="hi")))
    outcome = await AnthropicMessagesEngine().generate(
        ROW, make_intent(messages=(system, user)), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["system"] == [
        {"type": "text", "text": "Rules.", "cache_control": {"type": "ephemeral"}}
    ], f"system: {body.get('system')!r}"
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], (
        f"messages: {body.get('messages')!r}"
    )


async def test_generate_rejects_user_turn_with_no_encodable_content() -> None:
    intent = make_intent(messages=(SYSTEM, UserMessage(blocks=(PromptBlock(text=""),))))
    with pytest.raises(InvalidRequest, match="user turn"):
        await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)


async def test_generate_rejects_system_message_after_conversation_turns() -> None:
    intent = make_intent(messages=(USER, SYSTEM))
    with pytest.raises(InvalidRequest, match="precede"):
        await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)


async def test_generate_rejects_empty_assistant_turn() -> None:
    intent = make_intent(
        messages=(SYSTEM, USER, AssistantMessage(text="", tool_calls=(), continuation=Absent()))
    )
    with pytest.raises(InvalidRequest, match="assistant turn"):
        await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)


@respx.mock
async def test_generate_maps_reasoning_level_to_row_thinking_budget() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    outcome = await AnthropicMessagesEngine().generate(
        ROW, make_intent(reasoning="low"), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}, (
        f"thinking: {body.get('thinking')!r}"
    )
    assert outcome.meta.native_reasoning == Present("budget_tokens=1024"), (
        f"native_reasoning: {outcome.meta.native_reasoning!r}"
    )


@respx.mock
async def test_generate_omits_thinking_when_row_has_no_reasoning_knob() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    row = replace(ROW, reasoning=Absent())
    outcome = await AnthropicMessagesEngine().generate(
        row, make_intent(reasoning="none"), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert "thinking" not in body, f"thinking key must be omitted; body: {body!r}"
    assert outcome.meta.native_reasoning == Absent(), (
        f"native_reasoning: {outcome.meta.native_reasoning!r}"
    )


async def test_generate_rejects_undeclared_reasoning_level() -> None:
    with pytest.raises(InvalidRequest, match="reasoning level 'max'"):
        await AnthropicMessagesEngine().generate(ROW, make_intent(reasoning="max"), CREDENTIAL)


async def test_generate_rejects_reasoning_level_on_knobless_row() -> None:
    row = replace(ROW, reasoning=Absent())
    with pytest.raises(InvalidRequest, match="no reasoning knob"):
        await AnthropicMessagesEngine().generate(row, make_intent(reasoning="high"), CREDENTIAL)


@respx.mock
async def test_generate_encodes_tools_with_closed_input_schema() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    intent = make_intent(tools=(SEARCH_TOOL,), tool_choice="none")
    outcome = await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["tools"] == [
        {
            "name": "search_library",
            "description": "Search the library",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        }
    ], f"tools: {body.get('tools')!r}"
    assert body["tool_choice"] == {"type": "none"}, f"tool_choice: {body.get('tool_choice')!r}"


@respx.mock
async def test_generate_without_tools_omits_tool_choice() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert "tools" not in body and "tool_choice" not in body, (
        f"tools/tool_choice must be omitted; body: {body!r}"
    )


@respx.mock
async def test_generate_encodes_strict_json_output_via_output_config_format() -> None:
    structured: dict[str, object] = {"type": "text", "text": '{"verdict": "yes"}'}
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[structured], usage=usage_body()))
    )
    outcome = await AnthropicMessagesEngine().generate(
        ROW, make_intent(output=VERDICT_OUTPUT), CREDENTIAL
    )
    body = request_body(route)
    # SDK 0.121.0 wire fact: GA structured output is output_config.format
    # (JSONOutputFormatParam), not a top-level output_format field.
    assert body["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"verdict": {"type": "string"}},
                "required": ["verdict"],
                "additionalProperties": False,
            },
        }
    }, f"output_config: {body.get('output_config')!r}"
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert outcome.response.content == StructuredContent(
        payload={"verdict": "yes"}, text='{"verdict": "yes"}'
    ), f"content: {outcome.response.content!r}"


@respx.mock
async def test_generate_json_mode_row_sends_no_output_knob() -> None:
    structured: dict[str, object] = {"type": "text", "text": '{"verdict": "no"}'}
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[structured], usage=usage_body()))
    )
    row = replace(ROW, structured="json_mode")
    outcome = await AnthropicMessagesEngine().generate(
        row, make_intent(output=VERDICT_OUTPUT), CREDENTIAL
    )
    body = request_body(route)
    assert "output_config" not in body, f"output_config must be omitted; body: {body!r}"
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert outcome.response.content == StructuredContent(
        payload={"verdict": "no"}, text='{"verdict": "no"}'
    ), f"content: {outcome.response.content!r}"


@respx.mock
async def test_generate_encodes_image_block_as_base64_source() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    image = ImageBlock(media_type="image/png", data=b"\x89PNG-fake")
    intent = make_intent(
        messages=(SYSTEM, UserMessage(blocks=(PromptBlock(text="what is this?"), image)))
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64encode(b"\x89PNG-fake").decode("ascii"),
                    },
                },
            ],
        }
    ], f"messages: {body.get('messages')!r}"


@respx.mock
async def test_generate_replays_continuation_blocks_verbatim_and_groups_tool_results() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    artifact = ContinuationArtifact(
        target=TARGET,
        codec_id="anthropic.v1",
        opaque_payload={"blocks": (THINKING_BLOCK, REDACTED_BLOCK)},
    )
    intent = make_intent(
        messages=(
            SYSTEM,
            USER,
            AssistantMessage(
                text="checking",
                tool_calls=(
                    ToolCall(id="toolu_1", name="search_library", arguments={"query": "cats"}),
                ),
                continuation=Present(artifact),
            ),
            ToolResultMessage(call_id="toolu_1", output="42", is_error=False),
            ToolResultMessage(call_id="toolu_2", output="boom", is_error=True),
        ),
        tools=(SEARCH_TOOL,),
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                THINKING_BLOCK,
                REDACTED_BLOCK,
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search_library",
                    "input": {"query": "cats"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "42",
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": "boom",
                    "is_error": True,
                },
            ],
        },
    ], f"messages must lead with the verbatim thinking blocks: {body.get('messages')!r}"


@pytest.mark.parametrize(
    "artifact",
    [
        ContinuationArtifact(
            target=ProviderTarget(provider="anthropic", model="claude-other"),
            codec_id="anthropic.v1",
            opaque_payload={"blocks": (THINKING_BLOCK,)},
        ),
        ContinuationArtifact(
            target=TARGET,
            codec_id="openai.v1",
            opaque_payload={"blocks": (THINKING_BLOCK,)},
        ),
    ],
    ids=["target-mismatch", "codec-mismatch"],
)
async def test_generate_rejects_continuation_bound_elsewhere(
    artifact: ContinuationArtifact,
) -> None:
    intent = make_intent(
        messages=(
            SYSTEM,
            USER,
            AssistantMessage(text="prior", tool_calls=(), continuation=Present(artifact)),
        )
    )
    with pytest.raises(InvalidRequest, match="cannot replay"):
        await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)


async def test_generate_rejects_continuation_without_blocks() -> None:
    artifact = ContinuationArtifact(target=TARGET, codec_id="anthropic.v1", opaque_payload={})
    intent = make_intent(
        messages=(
            SYSTEM,
            USER,
            AssistantMessage(text="prior", tool_calls=(), continuation=Present(artifact)),
        )
    )
    with pytest.raises(InvalidRequest, match="blocks"):
        await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)


@respx.mock
async def test_generate_forwards_provider_options_into_request_body() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    intent = make_intent(
        provider_options={"metadata": {"user_id": "u-1"}, "service_tier": "standard_only"}
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["metadata"] == {"user_id": "u-1"}, f"body: {body!r}"
    assert body["service_tier"] == "standard_only", f"body: {body!r}"


async def test_generate_rejects_provider_options_colliding_with_owned_keys() -> None:
    intent = make_intent(provider_options={"thinking": {"type": "disabled"}})
    with pytest.raises(InvalidRequest, match="thinking"):
        await AnthropicMessagesEngine().generate(ROW, intent, CREDENTIAL)


@respx.mock
async def test_generate_uses_sdk_default_base_url_when_row_base_url_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    row = replace(ROW, base_url=Absent())
    outcome = await AnthropicMessagesEngine().generate(row, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert route.call_count == 1, "request must hit the SDK default base URL"


# ---------------------------------------------------------------------------
# Response decode


@respx.mock
async def test_generate_decodes_success_usage_meta_and_continuation() -> None:
    content = [THINKING_BLOCK, REDACTED_BLOCK, TEXT_BLOCK]
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=content, usage=usage_body()))
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert_meta(outcome)
    assert outcome.response.content == TextContent(text="hello", tool_calls=()), (
        f"content: {outcome.response.content!r}"
    )
    continuation = outcome.response.continuation
    assert isinstance(continuation, Present), f"continuation: {continuation!r}"
    artifact = continuation.value
    assert artifact.target == TARGET, f"artifact.target: {artifact.target!r}"
    assert artifact.codec_id == "anthropic.v1", f"artifact.codec_id: {artifact.codec_id!r}"
    assert list(artifact.opaque_payload["blocks"]) == [THINKING_BLOCK, REDACTED_BLOCK], (  # type: ignore[arg-type]
        f"payload blocks must be the verbatim ordered wire blocks: {artifact.opaque_payload!r}"
    )


@respx.mock
async def test_generate_falls_back_to_envelope_id_when_header_missing() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(
            envelope(content=[TEXT_BLOCK], usage=usage_body()), request_id=None
        )
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert outcome.meta.provider_request_id == Present("msg_123"), (
        f"provider_request_id: {outcome.meta.provider_request_id!r}"
    )


@respx.mock
async def test_generate_parses_tool_calls() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(
            envelope(
                content=[TEXT_BLOCK, TOOL_USE_BLOCK], stop_reason="tool_use", usage=usage_body()
            )
        )
    )
    outcome = await AnthropicMessagesEngine().generate(
        ROW, make_intent(tools=(SEARCH_TOOL,)), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert outcome.response.content == TextContent(
        text="hello",
        tool_calls=(ToolCall(id="toolu_1", name="search_library", arguments={"query": "cats"}),),
    ), f"content: {outcome.response.content!r}"


@respx.mock
async def test_generate_non_object_tool_input_returns_failed_value() -> None:
    broken = dict(TOOL_USE_BLOCK, input="not-an-object")
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(
            envelope(content=[broken], stop_reason="tool_use", usage=usage_body())
        )
    )
    outcome = await AnthropicMessagesEngine().generate(
        ROW, make_intent(tools=(SEARCH_TOOL,)), CREDENTIAL
    )
    assert isinstance(outcome, Failed), f"outcome: {outcome!r}"
    assert isinstance(outcome.failure, InvalidToolArguments), f"failure: {outcome.failure!r}"
    assert "search_library" in outcome.failure.safe_detail, (
        f"safe_detail: {outcome.failure.safe_detail!r}"
    )
    assert_meta(outcome)


@respx.mock
async def test_generate_pre_output_refusal_is_confirmed_non_billable() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(
            envelope(
                content=[],
                stop_reason="refusal",
                usage=usage_body(output_tokens=0),
                stop_details={
                    "type": "refusal",
                    "category": "general_harms",
                    "explanation": "I can't help with that.",
                },
            )
        )
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Refused), f"outcome: {outcome!r}"
    assert outcome.safe_detail == "I can't help with that.", f"safe_detail: {outcome.safe_detail!r}"
    zero_output = replace(EXPECTED_USAGE, output_tokens=0, total_tokens=108)
    assert_meta(outcome, usage=zero_output, billability=ConfirmedNonBillable())


@respx.mock
async def test_generate_mid_output_refusal_stays_possibly_billable() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(
            envelope(
                content=[TEXT_BLOCK],
                stop_reason="refusal",
                usage=usage_body(),
                stop_details={"type": "refusal", "category": "cyber", "explanation": None},
            )
        )
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Refused), f"outcome: {outcome!r}"
    assert outcome.safe_detail == "refusal category: cyber", f"safe_detail: {outcome.safe_detail!r}"
    assert_meta(outcome, billability=PossiblyBillable())


@respx.mock
async def test_generate_max_tokens_maps_to_incomplete() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(
            envelope(content=[TEXT_BLOCK], stop_reason="max_tokens", usage=usage_body())
        )
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Incomplete), f"outcome: {outcome!r}"
    assert outcome.reason == "max_output_tokens", f"reason: {outcome.reason!r}"
    assert outcome.status == "provider_incomplete", f"status: {outcome.status!r}"
    assert outcome.safe_detail == Absent(), f"safe_detail: {outcome.safe_detail!r}"
    assert_meta(outcome)


@respx.mock
async def test_generate_unknown_stop_reason_is_protocol_defect() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(
            envelope(content=[TEXT_BLOCK], stop_reason="pause_turn", usage=usage_body())
        )
    )
    with pytest.raises(ProtocolDefect, match="pause_turn"):
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)


@respx.mock
async def test_generate_malformed_json_envelope_is_protocol_defect() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "application/json"}, content=b"{not json"
        )
    )
    with pytest.raises(ProtocolDefect, match="not valid JSON"):
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)


@respx.mock
async def test_generate_missing_model_is_protocol_defect() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], model=None, usage=usage_body()))
    )
    with pytest.raises(ProtocolDefect, match="model"):
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)


@respx.mock
async def test_generate_invalid_strict_json_text_returns_failed_value() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=mock_response(envelope(content=[TEXT_BLOCK], usage=usage_body()))
    )
    outcome = await AnthropicMessagesEngine().generate(
        ROW, make_intent(output=VERDICT_OUTPUT), CREDENTIAL
    )
    assert isinstance(outcome, Failed), f"outcome: {outcome!r}"
    assert isinstance(outcome.failure, InvalidStructuredOutput), f"failure: {outcome.failure!r}"
    assert "JSON" in outcome.failure.safe_detail, f"safe_detail: {outcome.failure.safe_detail!r}"
    assert_meta(outcome)


# ---------------------------------------------------------------------------
# Fault injection — generate


@respx.mock
async def test_generate_rate_limit_with_retry_after_raises_transient() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "7", "request-id": "req_429"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )
    )
    with pytest.raises(TransientAttempt) as exc_info:
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    attempt = exc_info.value
    assert attempt.cause == ProviderRateLimit(retry_after=Present(7.0)), f"cause: {attempt.cause!r}"
    assert attempt.status_code == Present(429), f"status_code: {attempt.status_code!r}"
    assert attempt.provider_request_id == Present("req_429"), (
        f"provider_request_id: {attempt.provider_request_id!r}"
    )
    assert attempt.billability == PossiblyBillable(), f"billability: {attempt.billability!r}"


@respx.mock
async def test_generate_rate_limit_without_retry_after_raises_transient() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
        )
    )
    with pytest.raises(TransientAttempt) as exc_info:
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.cause == ProviderRateLimit(retry_after=Absent()), (
        f"cause: {exc_info.value.cause!r}"
    )


@pytest.mark.parametrize("status", [503, 529])
@respx.mock
async def test_generate_5xx_raises_transient_unavailable(status: int) -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            status,
            json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
    )
    with pytest.raises(TransientAttempt) as exc_info:
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.cause == ProviderHttpUnavailable(), f"cause: {exc_info.value.cause!r}"
    assert exc_info.value.status_code == Present(status), (
        f"status_code: {exc_info.value.status_code!r}"
    )


@respx.mock
async def test_generate_timeout_raises_transient_timeout() -> None:
    respx.post(MESSAGES_URL).mock(side_effect=httpx.ReadTimeout("read timed out"))
    with pytest.raises(TransientAttempt) as exc_info:
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.cause == ProviderTimeout(), f"cause: {exc_info.value.cause!r}"
    assert exc_info.value.billability == PossiblyBillable(), (
        f"billability: {exc_info.value.billability!r}"
    )


@respx.mock
async def test_generate_connect_error_raises_transport_unavailable_not_dispatched() -> None:
    respx.post(MESSAGES_URL).mock(side_effect=httpx.ConnectError("no route to host"))
    with pytest.raises(TransientAttempt) as exc_info:
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.cause == TransportUnavailable(), f"cause: {exc_info.value.cause!r}"
    assert exc_info.value.billability == NotDispatched(), (
        f"billability: {exc_info.value.billability!r}"
    )


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (
            400,
            {
                "type": "invalid_request_error",
                "message": "prompt is too long: 200000 tokens > 100000 maximum",
            },
        ),
        (413, {"type": "request_too_large", "message": "Request body too large"}),
    ],
    ids=["400-too-long", "413-request-too-large"],
)
@respx.mock
async def test_generate_context_overflow_returns_failed_value(
    status: int, error: dict[str, object]
) -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            status,
            headers={"request-id": "req_ctx"},
            json={"type": "error", "error": error},
        )
    )
    outcome = await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Failed), f"outcome: {outcome!r}"
    assert outcome.failure == ProviderContextTooLarge(), f"failure: {outcome.failure!r}"
    assert_meta(outcome, request_id="req_ctx", usage=None, status_code=status)


@respx.mock
async def test_generate_credential_rejection_raises() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            401,
            json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}},
        )
    )
    with pytest.raises(CredentialRejected, match="401"):
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)


@respx.mock
async def test_generate_billing_error_is_quota_defect() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "billing_error", "message": "Your credit balance is too low"},
            },
        )
    )
    with pytest.raises(RuntimeDefect) as exc_info:
        await AnthropicMessagesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.code == "quota_exhausted", f"code: {exc_info.value.code!r}"


# ---------------------------------------------------------------------------
# Streaming


@respx.mock
async def test_stream_decodes_thinking_text_tool_usage_and_terminal() -> None:
    frames: list[dict[str, object]] = [
        message_start_frame(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "pondering"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig-1"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        text_delta_frame(1, "Hi"),
        text_delta_frame(1, " there"),
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "search_library",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": ' "cats"}'},
        },
        {"type": "content_block_stop", "index": 2},
        message_delta_frame(stop_reason="tool_use"),
        {"type": "message_stop"},
    ]
    route = respx.post(MESSAGES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(AnthropicMessagesEngine(), make_intent(tools=(SEARCH_TOOL,)))

    body = request_body(route)
    assert body["stream"] is True, f"stream flag missing from request body: {body!r}"

    expected_tool_call = ToolCall(id="toolu_1", name="search_library", arguments={"query": "cats"})
    assert events[0] == StreamStart(), f"first event: {events[0]!r}"
    assert events[1] == TextDelta(text="Hi"), f"{events[1]!r}"
    assert events[2] == TextDelta(text=" there"), f"{events[2]!r}"
    assert events[3] == ToolCallStart(call_id="toolu_1", name="search_library"), f"{events[3]!r}"
    assert events[4] == ToolCallDelta(call_id="toolu_1", arguments_delta='{"query":'), (
        f"{events[4]!r}"
    )
    assert events[5] == ToolCallDelta(call_id="toolu_1", arguments_delta=' "cats"}'), (
        f"{events[5]!r}"
    )
    assert events[6] == ToolCallDone(tool_call=expected_tool_call), f"{events[6]!r}"
    assert events[7] == UsageEvent(usage=EXPECTED_USAGE), (
        f"usage must fold message_start + message_delta frames: {events[7]!r}"
    )

    continuation_delta = events[8]
    assert isinstance(continuation_delta, ContinuationDelta), f"{continuation_delta!r}"
    artifact = continuation_delta.artifact
    assert artifact.target == TARGET and artifact.codec_id == "anthropic.v1", f"{artifact!r}"
    assert list(artifact.opaque_payload["blocks"]) == [THINKING_BLOCK], (  # type: ignore[arg-type]
        f"payload blocks must be the assembled thinking block: {artifact.opaque_payload!r}"
    )

    terminal = events[9]
    assert isinstance(terminal, TerminalEvent), f"{terminal!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded), f"terminal outcome: {outcome!r}"
    assert outcome.response.content == TextContent(
        text="Hi there", tool_calls=(expected_tool_call,)
    ), f"content: {outcome.response.content!r}"
    assert outcome.response.continuation == Present(artifact), (
        f"terminal continuation: {outcome.response.continuation!r}"
    )
    assert_meta(outcome, request_id="req_s1")
    assert len(events) == 10, f"unexpected extra events: {events!r}"


@respx.mock
async def test_stream_refusal_folds_into_incomplete_refused_terminal() -> None:
    frames: list[dict[str, object]] = [
        message_start_frame(usage={"input_tokens": 20, "output_tokens": 0}),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        },
        {"type": "content_block_stop", "index": 0},
        message_delta_frame(
            stop_reason="refusal",
            usage={"output_tokens": 0},
            stop_details={"type": "refusal", "category": None, "explanation": "Can't help."},
        ),
        {"type": "message_stop"},
    ]
    respx.post(MESSAGES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(AnthropicMessagesEngine(), make_intent())
    # UsageEvent is progressive telemetry; the refusal terminal must follow it
    # directly with NO ContinuationDelta (the partial output is invalidated).
    assert events[0] == StreamStart(), f"events: {events!r}"
    assert isinstance(events[1], UsageEvent), f"events: {events!r}"
    terminal = events[2]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Incomplete), f"terminal outcome: {outcome!r}"
    assert outcome.status == "refused", f"status: {outcome.status!r}"
    assert outcome.reason == "content_filter_partial", f"reason: {outcome.reason!r}"
    assert outcome.safe_detail == Present("Can't help."), f"safe_detail: {outcome.safe_detail!r}"
    refusal_usage = TokenUsage(
        input_tokens=20,
        output_tokens=0,
        total_tokens=20,
        reasoning_tokens=Absent(),
        cache_read_input_tokens=Absent(),
        cache_write_input_tokens=Absent(),
    )
    assert_meta(
        outcome,
        request_id="req_s1",
        usage=refusal_usage,
        billability=ConfirmedNonBillable(),
    )
    assert len(events) == 3, f"no ContinuationDelta on refusal: {events!r}"


@respx.mock
async def test_stream_max_tokens_incomplete_terminal() -> None:
    frames: list[dict[str, object]] = [
        message_start_frame(),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        text_delta_frame(0, "truncat"),
        {"type": "content_block_stop", "index": 0},
        message_delta_frame(stop_reason="max_tokens"),
        {"type": "message_stop"},
    ]
    respx.post(MESSAGES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(AnthropicMessagesEngine(), make_intent())
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Incomplete), f"terminal outcome: {outcome!r}"
    assert outcome.reason == "max_output_tokens", f"reason: {outcome.reason!r}"
    assert outcome.status == "provider_incomplete", f"status: {outcome.status!r}"
    assert_meta(outcome, request_id="req_s1")


@respx.mock
async def test_stream_midcut_after_semantic_output_raises_partial_interrupt() -> None:
    frames: list[dict[str, object]] = [
        message_start_frame(),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        text_delta_frame(0, "partial"),
    ]
    respx.post(MESSAGES_URL).mock(return_value=mock_stream(frames))
    events: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as exc_info:
        async for event in AnthropicMessagesEngine().stream(ROW, make_intent(), CREDENTIAL):
            events.append(event)
    assert events == [StreamStart(), TextDelta(text="partial")], f"events: {events!r}"
    assert exc_info.value.cause == ProviderStreamInterrupted(partial_output=True), (
        f"cause: {exc_info.value.cause!r}"
    )
    assert exc_info.value.billability == PossiblyBillable(), (
        f"billability: {exc_info.value.billability!r}"
    )


@respx.mock
async def test_stream_midcut_before_semantic_output_raises_clean_interrupt() -> None:
    respx.post(MESSAGES_URL).mock(return_value=mock_stream([message_start_frame()]))
    events: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as exc_info:
        async for event in AnthropicMessagesEngine().stream(ROW, make_intent(), CREDENTIAL):
            events.append(event)
    assert events == [StreamStart()], f"events: {events!r}"
    assert exc_info.value.cause == ProviderStreamInterrupted(partial_output=False), (
        f"cause: {exc_info.value.cause!r}"
    )


@respx.mock
async def test_stream_inband_overloaded_error_raises_transient() -> None:
    frames: list[dict[str, object]] = [
        message_start_frame(),
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]
    respx.post(MESSAGES_URL).mock(return_value=mock_stream(frames))
    events: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as exc_info:
        async for event in AnthropicMessagesEngine().stream(ROW, make_intent(), CREDENTIAL):
            events.append(event)
    assert events == [StreamStart()], f"events: {events!r}"
    assert exc_info.value.cause == ProviderHttpUnavailable(), f"cause: {exc_info.value.cause!r}"


@respx.mock
async def test_stream_unknown_inband_error_is_protocol_defect() -> None:
    frames: list[dict[str, object]] = [
        message_start_frame(),
        {"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}},
    ]
    respx.post(MESSAGES_URL).mock(return_value=mock_stream(frames))
    with pytest.raises(ProtocolDefect, match="error event"):
        async for _ in AnthropicMessagesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass


@respx.mock
async def test_stream_tool_argument_parse_failure_yields_failed_terminal() -> None:
    frames: list[dict[str, object]] = [
        message_start_frame(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "search_library",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query": '},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    respx.post(MESSAGES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(AnthropicMessagesEngine(), make_intent(tools=(SEARCH_TOOL,)))
    assert events[0] == StreamStart(), f"events: {events!r}"
    assert events[1] == ToolCallStart(call_id="toolu_1", name="search_library"), f"{events[1]!r}"
    assert events[2] == ToolCallDelta(call_id="toolu_1", arguments_delta='{"query": '), (
        f"{events[2]!r}"
    )
    terminal = events[3]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Failed), f"terminal outcome: {outcome!r}"
    assert isinstance(outcome.failure, InvalidToolArguments), f"failure: {outcome.failure!r}"
    assert len(events) == 4, f"stream must end at the failed terminal: {events!r}"


@respx.mock
async def test_stream_context_overflow_yields_failed_terminal_without_stream_start() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            400,
            headers={"request-id": "req_ctx"},
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "prompt is too long"},
            },
        )
    )
    events = await collect_stream(AnthropicMessagesEngine(), make_intent())
    assert len(events) == 1, f"expected a single terminal event: {events!r}"
    terminal = events[0]
    assert isinstance(terminal, TerminalEvent), f"{terminal!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Failed), f"terminal outcome: {outcome!r}"
    assert outcome.failure == ProviderContextTooLarge(), f"failure: {outcome.failure!r}"
    assert_meta(outcome, request_id="req_ctx", usage=None, status_code=400)


@respx.mock
async def test_stream_credential_rejection_raises_before_any_event() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            403,
            json={"type": "error", "error": {"type": "permission_error", "message": "forbidden"}},
        )
    )
    events: list[CodecStreamEvent] = []
    with pytest.raises(CredentialRejected, match="403"):
        async for event in AnthropicMessagesEngine().stream(ROW, make_intent(), CREDENTIAL):
            events.append(event)
    assert events == [], f"no events may precede credential rejection: {events!r}"


@respx.mock
async def test_stream_rate_limit_at_accept_raises_transient() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "3"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
        )
    )
    with pytest.raises(TransientAttempt) as exc_info:
        async for _ in AnthropicMessagesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass
    assert exc_info.value.cause == ProviderRateLimit(retry_after=Present(3.0)), (
        f"cause: {exc_info.value.cause!r}"
    )


@respx.mock
async def test_stream_malformed_frame_is_protocol_defect() -> None:
    content = sse_bytes([message_start_frame()]) + b"event: message_delta\ndata: {broken\n\n"
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=content
        )
    )
    with pytest.raises(ProtocolDefect, match="not valid JSON"):
        async for _ in AnthropicMessagesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass


@respx.mock
async def test_stream_strict_json_terminal_promotes_structured_content() -> None:
    frames: list[dict[str, object]] = [
        message_start_frame(),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        text_delta_frame(0, '{"verdict": "yes"}'),
        {"type": "content_block_stop", "index": 0},
        message_delta_frame(stop_reason="end_turn"),
        {"type": "message_stop"},
    ]
    respx.post(MESSAGES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(AnthropicMessagesEngine(), make_intent(output=VERDICT_OUTPUT))
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded), f"terminal outcome: {outcome!r}"
    assert outcome.response.content == StructuredContent(
        payload={"verdict": "yes"}, text='{"verdict": "yes"}'
    ), f"content: {outcome.response.content!r}"


# ---------------------------------------------------------------------------
# Continuation round-trip: decode → replay encodes verbatim


@respx.mock
async def test_continuation_round_trip_replays_decoded_blocks_verbatim() -> None:
    first_content = [THINKING_BLOCK, TOOL_USE_BLOCK]
    route = respx.post(MESSAGES_URL).mock(
        return_value=mock_response(
            envelope(content=first_content, stop_reason="tool_use", usage=usage_body())
        )
    )
    engine = AnthropicMessagesEngine()
    first = await engine.generate(ROW, make_intent(tools=(SEARCH_TOOL,)), CREDENTIAL)
    assert isinstance(first, Succeeded), f"first outcome: {first!r}"
    continuation = first.response.continuation
    assert isinstance(continuation, Present), f"continuation: {continuation!r}"
    content = first.response.content
    assert isinstance(content, TextContent), f"content: {content!r}"

    follow_up = make_intent(
        messages=(
            SYSTEM,
            USER,
            AssistantMessage(text="", tool_calls=content.tool_calls, continuation=continuation),
            ToolResultMessage(call_id="toolu_1", output="42", is_error=False),
        ),
        tools=(SEARCH_TOOL,),
    )
    second = await engine.generate(ROW, follow_up, CREDENTIAL)
    assert isinstance(second, Succeeded), f"second outcome: {second!r}"
    body = request_body(route)
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                THINKING_BLOCK,
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search_library",
                    "input": {"query": "cats"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "42",
                    "is_error": False,
                }
            ],
        },
    ], f"replayed messages must carry the decoded blocks verbatim: {body.get('messages')!r}"
