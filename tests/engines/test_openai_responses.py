"""OpenAI Responses engine conformance + fault injection (HTTP boundary via respx).

Request tests assert EXACT body dicts as the SDK puts them on the wire; decode
tests feed canned envelopes; stream tests feed raw SSE bytes. No internal
mocking anywhere — respx intercepts the SDK's own httpx transport.
"""

import json
from base64 import b64encode
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace

import httpx
import pytest
import respx

from provider_runtime.engines import TransientAttempt
from provider_runtime.engines.openai_responses import OpenAIResponsesEngine
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
    UserMessage,
)

RESPONSES_URL = "https://api.openai.com/v1/responses"

# Registry rows carry the real openai wire fragment per level — a self-describing
# request-parameter mapping the engine merges verbatim.
REASONING_LEVELS: Mapping[ReasoningLevel, object] = {
    "none": {"reasoning": {"effort": "none"}},
    "low": {"reasoning": {"effort": "low"}},
    "high": {"reasoning": {"effort": "high"}},
}

NATIVE_REASONING_HIGH = '{"reasoning":{"effort":"high"}}'

ROW = ModelRow(
    ref="openai:gpt-test",
    provider="openai",
    model_id="gpt-test-1",
    engine="openai_responses",
    base_url=Present("https://api.openai.com/v1"),
    context_window=400_000,
    max_output_tokens=64_000,
    modalities=frozenset({"text", "image"}),
    tools=True,
    streaming=True,
    structured="native",
    reasoning=Present(REASONING_LEVELS),
    continuation_codec="openai.v1",
    correlation="header",
    routing=Absent(),
)

# Every request-affecting environment variable the openai SDK reads on its own.
POISON_ENV = {
    "OPENAI_BASE_URL": "https://poisoned.invalid/v1",
    "OPENAI_ORG_ID": "org-poison",
    "OPENAI_PROJECT_ID": "proj-poison",
    "OPENAI_WEBHOOK_SECRET": "whsec-poison",
    "OPENAI_CUSTOM_HEADERS": "X-Poison: pwned",
}

TARGET = ProviderTarget(provider="openai", model="gpt-test-1")
CREDENTIAL = ProviderCredential(provider="openai", key="sk-test-not-a-real-key-1234567890")

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


def usage_body() -> dict[str, object]:
    return {
        "input_tokens": 120,
        "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 8},
        "output_tokens": 30,
        "output_tokens_details": {"reasoning_tokens": 12},
        "total_tokens": 150,
    }


EXPECTED_USAGE = TokenUsage(
    input_tokens=120,
    output_tokens=30,
    total_tokens=150,
    reasoning_tokens=Present(12),
    cache_read_input_tokens=Present(100),
    cache_write_input_tokens=Present(8),
)

TEXT_ITEM: dict[str, object] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
}

REASONING_ITEM: dict[str, object] = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [],
    "encrypted_content": "gAAAAB-enc-1",
}

FUNCTION_CALL_ITEM: dict[str, object] = {
    "id": "fc_1",
    "type": "function_call",
    "call_id": "call_1",
    "name": "search_library",
    "arguments": '{"query": "cats"}',
    "status": "completed",
}


def envelope(
    *,
    output: list[dict[str, object]],
    status: str = "completed",
    usage: dict[str, object] | None = None,
    incomplete_details: dict[str, object] | None = None,
    model: str | None = "gpt-test-1",
    response_id: str = "resp_123",
) -> dict[str, object]:
    body: dict[str, object] = {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": status,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if model is not None:
        body["model"] = model
    if usage is not None:
        body["usage"] = usage
    if incomplete_details is not None:
        body["incomplete_details"] = incomplete_details
    return body


def mock_response(body: dict[str, object], *, request_id: str | None = "req_abc") -> httpx.Response:
    headers = {"content-type": "application/json"}
    if request_id is not None:
        headers["x-request-id"] = request_id
    return httpx.Response(200, headers=headers, json=body)


def request_body(route: respx.Route) -> dict[str, object]:
    request = route.calls.last.request
    parsed = json.loads(request.content)
    assert isinstance(parsed, dict), f"request body is not a JSON object: {request.content!r}"
    return parsed


def sse_bytes(frames: list[dict[str, object]]) -> bytes:
    chunks = [f"event: {frame['type']}\ndata: {json.dumps(frame)}\n\n".encode() for frame in frames]
    return b"".join(chunks)


def stream_headers(request_id: str | None) -> dict[str, str]:
    headers = {"content-type": "text/event-stream"}
    if request_id is not None:
        headers["x-request-id"] = request_id
    return headers


def mock_stream(
    frames: list[dict[str, object]], *, request_id: str | None = "req_s1"
) -> httpx.Response:
    return httpx.Response(200, headers=stream_headers(request_id), content=sse_bytes(frames))


def mock_broken_stream(frames: list[dict[str, object]], error: Exception) -> httpx.Response:
    """A stream whose transport dies mid-body — the shape a real dropped
    connection takes, as opposed to a well-formed body that simply ends."""

    async def body() -> AsyncIterator[bytes]:
        yield sse_bytes(frames)
        raise error

    return httpx.Response(200, headers=stream_headers("req_s1"), content=body())


def created_frame(response_id: str = "resp_s1") -> dict[str, object]:
    return {
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": response_id, "status": "in_progress", "model": "gpt-test-1"},
    }


def completed_frame(
    *,
    output: list[dict[str, object]],
    usage: dict[str, object] | None = None,
    model: str = "gpt-test-1",
    response_id: str = "resp_s1",
) -> dict[str, object]:
    return {
        "type": "response.completed",
        "sequence_number": 99,
        "response": envelope(
            output=output, usage=usage, model=model, response_id=response_id, status="completed"
        ),
    }


async def collect_stream(
    engine: OpenAIResponsesEngine, intent: GenerateIntent, *, row: ModelRow = ROW
) -> list[CodecStreamEvent]:
    return [event async for event in engine.stream(row, intent, CREDENTIAL)]


def assert_meta(
    outcome: Succeeded | Refused | Incomplete | Failed,
    *,
    request_id: str = "req_abc",
    usage: TokenUsage | None = EXPECTED_USAGE,
    native_reasoning: str | None = NATIVE_REASONING_HIGH,
    status_code: int = 200,
    model: str = "gpt-test-1",
) -> None:
    meta = outcome.meta
    assert meta.provider == "openai", f"meta.provider: {meta.provider!r}"
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
    assert meta.billability == PossiblyBillable(), f"meta.billability: {meta.billability!r}"
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
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert route.call_count == 1, f"expected exactly one dispatch; got {route.call_count}"
    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {CREDENTIAL.key}", (
        f"authorization header: {request.headers.get('authorization')!r}"
    )
    body = request_body(route)
    assert body == {
        "model": "gpt-test-1",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": "You are terse."}]},
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ],
        "max_output_tokens": 128,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": "high"},
    }, f"request body: {body!r}"


@respx.mock
async def test_generate_omits_reasoning_when_row_has_no_reasoning_knob() -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    row = replace(ROW, reasoning=Absent())
    outcome = await OpenAIResponsesEngine().generate(row, make_intent(reasoning="none"), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert "reasoning" not in body, f"reasoning key must be omitted; body: {body!r}"
    assert outcome.meta.native_reasoning == Absent(), (
        f"native_reasoning: {outcome.meta.native_reasoning!r}"
    )


async def test_generate_rejects_undeclared_reasoning_level() -> None:
    with pytest.raises(InvalidRequest, match="reasoning level 'max'"):
        await OpenAIResponsesEngine().generate(ROW, make_intent(reasoning="max"), CREDENTIAL)


async def test_generate_rejects_a_reasoning_level_on_a_knobless_row() -> None:
    """A knobless row expresses only 'none'; an explicit level the row cannot
    put on the wire is refused, never silently dropped."""
    row = replace(ROW, reasoning=Absent())
    with pytest.raises(InvalidRequest, match="no reasoning knob"):
        await OpenAIResponsesEngine().generate(row, make_intent(reasoning="high"), CREDENTIAL)


@respx.mock
async def test_generate_merges_row_reasoning_fragment_verbatim() -> None:
    """The row owns the wire shape: whatever mapping it declares for the level
    is merged as-is and stamped as compact sorted-keys JSON."""
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    levels: Mapping[ReasoningLevel, object] = {
        "low": {"reasoning": {"effort": "low", "summary": "auto"}}
    }
    outcome = await OpenAIResponsesEngine().generate(
        replace(ROW, reasoning=Present(levels)), make_intent(reasoning="low"), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}, f"body: {body!r}"
    assert outcome.meta.native_reasoning == Present(
        '{"reasoning":{"effort":"low","summary":"auto"}}'
    ), f"native_reasoning: {outcome.meta.native_reasoning!r}"


@respx.mock
async def test_generate_reasoning_none_on_a_row_declaring_no_none_sends_nothing() -> None:
    """spec §14: "none" is the facade default, so it is callable on every row —
    a row that declares no "none" level sends no reasoning field and lets the
    provider's own default apply."""
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    levels: Mapping[ReasoningLevel, object] = {"low": {"reasoning": {"effort": "low"}}}
    outcome = await OpenAIResponsesEngine().generate(
        replace(ROW, reasoning=Present(levels)), make_intent(reasoning="none"), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert "reasoning" not in body, f"nothing may be sent; body: {body!r}"
    assert outcome.meta.native_reasoning == Absent(), (
        f"native_reasoning: {outcome.meta.native_reasoning!r}"
    )


@respx.mock
async def test_generate_declared_reasoning_none_still_sends_its_fragment() -> None:
    """A row that DOES declare "none" (openai spells it as an effort) sends it —
    the level is not special-cased away, only the undeclared case is."""
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(reasoning="none"), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["reasoning"] == {"effort": "none"}, f"body: {body!r}"
    assert outcome.meta.native_reasoning == Present('{"reasoning":{"effort":"none"}}'), (
        f"native_reasoning: {outcome.meta.native_reasoning!r}"
    )


async def test_generate_rejects_non_mapping_reasoning_value_as_registry_defect() -> None:
    levels: Mapping[ReasoningLevel, object] = {"high": "high"}
    with pytest.raises(RuntimeDefect, match="request-fragment mappings") as exc_info:
        await OpenAIResponsesEngine().generate(
            replace(ROW, reasoning=Present(levels)), make_intent(), CREDENTIAL
        )
    assert exc_info.value.code == "registry_invalid", f"code: {exc_info.value.code!r}"


async def test_generate_rejects_reasoning_fragment_colliding_with_an_engine_set_field() -> None:
    """The fragment is splatted into the params literal: a row naming a field
    the engine sets itself silently rewrites the call — here `store`, whose
    `false` is what makes reasoning replay stateless."""
    levels: Mapping[ReasoningLevel, object] = {
        "high": {"reasoning": {"effort": "high"}, "store": True}
    }
    with pytest.raises(RuntimeDefect, match="'store'") as exc_info:
        await OpenAIResponsesEngine().generate(
            replace(ROW, reasoning=Present(levels)), make_intent(), CREDENTIAL
        )
    assert exc_info.value.code == "registry_invalid", f"code: {exc_info.value.code!r}"


async def test_generate_rejects_provider_options_colliding_with_reasoning_fragment() -> None:
    intent = make_intent(provider_options={"reasoning": {"effort": "none"}})
    with pytest.raises(InvalidRequest, match="reasoning"):
        await OpenAIResponsesEngine().generate(ROW, intent, CREDENTIAL)


@respx.mock
async def test_generate_encodes_tools_strict_with_closed_schema() -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    intent = make_intent(tools=(SEARCH_TOOL,), tool_choice="none")
    outcome = await OpenAIResponsesEngine().generate(ROW, intent, CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["tools"] == [
        {
            "type": "function",
            "name": "search_library",
            "description": "Search the library",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ], f"tools: {body.get('tools')!r}"
    assert body["tool_choice"] == "none", f"tool_choice: {body.get('tool_choice')!r}"


@respx.mock
async def test_generate_without_tools_omits_tool_choice() -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert "tools" not in body and "tool_choice" not in body, (
        f"tools/tool_choice must be omitted; body: {body!r}"
    )


@respx.mock
async def test_generate_encodes_strict_json_output_as_native_text_format() -> None:
    structured_item: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": '{"verdict": "yes"}', "annotations": []}],
    }
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[structured_item], usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(
        ROW, make_intent(output=VERDICT_OUTPUT), CREDENTIAL
    )
    body = request_body(route)
    assert body["text"] == {
        "format": {
            "type": "json_schema",
            "name": "verdict",
            "schema": {
                "type": "object",
                "properties": {"verdict": {"type": "string"}},
                "required": ["verdict"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    }, f"text.format: {body.get('text')!r}"
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert outcome.response.content == StructuredContent(
        payload={"verdict": "yes"}, text='{"verdict": "yes"}'
    ), f"content: {outcome.response.content!r}"


@respx.mock
async def test_generate_json_mode_row_sends_json_object_format() -> None:
    structured_item: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": '{"verdict": "no"}', "annotations": []}],
    }
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[structured_item], usage=usage_body()))
    )
    row = replace(ROW, structured="json_mode")
    outcome = await OpenAIResponsesEngine().generate(
        row, make_intent(output=VERDICT_OUTPUT), CREDENTIAL
    )
    body = request_body(route)
    assert body["text"] == {"format": {"type": "json_object"}}, f"text: {body.get('text')!r}"
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert outcome.response.content == StructuredContent(
        payload={"verdict": "no"}, text='{"verdict": "no"}'
    ), f"content: {outcome.response.content!r}"


@respx.mock
async def test_generate_encodes_image_block_as_data_url() -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    image = ImageBlock(media_type="image/png", data=b"\x89PNG-fake")
    intent = make_intent(
        messages=(SYSTEM, UserMessage(blocks=(PromptBlock(text="what is this?"), image)))
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, intent, CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    expected_url = f"data:image/png;base64,{b64encode(b'\x89PNG-fake').decode('ascii')}"
    assert body["input"] == [
        {"role": "system", "content": [{"type": "input_text", "text": "You are terse."}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is this?"},
                {"type": "input_image", "image_url": expected_url, "detail": "auto"},
            ],
        },
    ], f"input: {body.get('input')!r}"


@respx.mock
async def test_generate_replays_continuation_payload_verbatim() -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    artifact = ContinuationArtifact(
        target=TARGET,
        codec_id="openai.v1",
        opaque_payload={"output": (REASONING_ITEM, FUNCTION_CALL_ITEM)},
    )
    intent = make_intent(
        messages=(
            SYSTEM,
            USER,
            AssistantMessage(
                text="prior",
                tool_calls=(
                    ToolCall(id="call_1", name="search_library", arguments={"query": "cats"}),
                ),
                continuation=Present(artifact),
            ),
            ToolResultMessage(call_id="call_1", output="42", is_error=False),
        )
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, intent, CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["input"] == [
        {"role": "system", "content": [{"type": "input_text", "text": "You are terse."}]},
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        REASONING_ITEM,
        FUNCTION_CALL_ITEM,
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
    ], f"input must splice payload items verbatim; got: {body.get('input')!r}"


@pytest.mark.parametrize(
    "artifact",
    [
        ContinuationArtifact(
            target=ProviderTarget(provider="openai", model="gpt-other"),
            codec_id="openai.v1",
            opaque_payload={"output": (REASONING_ITEM,)},
        ),
        ContinuationArtifact(
            target=TARGET,
            codec_id="anthropic.v1",
            opaque_payload={"output": (REASONING_ITEM,)},
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
        await OpenAIResponsesEngine().generate(ROW, intent, CREDENTIAL)


async def test_generate_rejects_continuation_without_output_items() -> None:
    artifact = ContinuationArtifact(target=TARGET, codec_id="openai.v1", opaque_payload={})
    intent = make_intent(
        messages=(
            SYSTEM,
            USER,
            AssistantMessage(text="prior", tool_calls=(), continuation=Present(artifact)),
        )
    )
    with pytest.raises(InvalidRequest, match="output"):
        await OpenAIResponsesEngine().generate(ROW, intent, CREDENTIAL)


async def test_generate_rejects_assistant_tool_calls_without_continuation() -> None:
    intent = make_intent(
        messages=(
            SYSTEM,
            USER,
            AssistantMessage(
                text="",
                tool_calls=(ToolCall(id="call_1", name="search_library", arguments={}),),
                continuation=Absent(),
            ),
            ToolResultMessage(call_id="call_1", output="42", is_error=False),
        )
    )
    with pytest.raises(InvalidRequest, match="continuation"):
        await OpenAIResponsesEngine().generate(ROW, intent, CREDENTIAL)


@respx.mock
async def test_generate_forwards_provider_options_into_request_body() -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    intent = make_intent(provider_options={"parallel_tool_calls": False, "prompt_cache_key": "k1"})
    outcome = await OpenAIResponsesEngine().generate(ROW, intent, CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    body = request_body(route)
    assert body["parallel_tool_calls"] is False, f"body: {body!r}"
    assert body["prompt_cache_key"] == "k1", f"body: {body!r}"


async def test_generate_rejects_provider_options_colliding_with_owned_keys() -> None:
    intent = make_intent(provider_options={"store": True})
    with pytest.raises(InvalidRequest, match="store"):
        await OpenAIResponsesEngine().generate(ROW, intent, CREDENTIAL)


@respx.mock
async def test_generate_ignores_openai_base_url_env_when_row_base_url_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-env dispatch: an Absent row base_url resolves to the canonical host
    in the engine, never to whatever OPENAI_BASE_URL says."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://poisoned.invalid/v1")
    poisoned = respx.post("https://poisoned.invalid/v1/responses").mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    row = replace(ROW, base_url=Absent())
    outcome = await OpenAIResponsesEngine().generate(row, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert route.call_count == 1, "request must hit the canonical OpenAI host"
    assert poisoned.call_count == 0, "OPENAI_BASE_URL must never reroute a call"


@respx.mock
async def test_generate_suppresses_every_ambient_sdk_env_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK reads these at client construction whenever the matching argument
    is omitted, and OPENAI_CUSTOM_HEADERS injects arbitrary headers into every
    request. None of them may touch the wire."""
    for name, value in POISON_ENV.items():
        monkeypatch.setenv(name, value)
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert route.call_count == 1, "request must hit the canonical OpenAI host"
    headers = route.calls.last.request.headers
    assert "x-poison" not in headers, f"OPENAI_CUSTOM_HEADERS reached the wire: {headers!r}"
    assert "openai-organization" not in headers, f"OPENAI_ORG_ID reached the wire: {headers!r}"
    assert "openai-project" not in headers, f"OPENAI_PROJECT_ID reached the wire: {headers!r}"


# ---------------------------------------------------------------------------
# Response decode


@respx.mock
async def test_generate_decodes_success_usage_meta_and_continuation() -> None:
    output = [REASONING_ITEM, TEXT_ITEM]
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=output, usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert_meta(outcome)
    assert outcome.response.content == TextContent(text="hello", tool_calls=()), (
        f"content: {outcome.response.content!r}"
    )
    continuation = outcome.response.continuation
    assert isinstance(continuation, Present), f"continuation: {continuation!r}"
    artifact = continuation.value
    assert artifact.target == TARGET, f"artifact.target: {artifact.target!r}"
    assert artifact.codec_id == "openai.v1", f"artifact.codec_id: {artifact.codec_id!r}"
    assert list(artifact.opaque_payload["output"]) == output, (  # type: ignore[arg-type]
        f"payload items must be the verbatim wire output: {artifact.opaque_payload!r}"
    )


@respx.mock
async def test_generate_falls_back_to_envelope_id_when_header_missing() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(
            envelope(output=[TEXT_ITEM], usage=usage_body()), request_id=None
        )
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert outcome.meta.provider_request_id == Present("resp_123"), (
        f"provider_request_id: {outcome.meta.provider_request_id!r}"
    )


@respx.mock
async def test_generate_parses_tool_calls_strictly() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(
            envelope(output=[FUNCTION_CALL_ITEM, TEXT_ITEM], usage=usage_body())
        )
    )
    outcome = await OpenAIResponsesEngine().generate(
        ROW, make_intent(tools=(SEARCH_TOOL,)), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"outcome: {outcome!r}"
    assert outcome.response.content == TextContent(
        text="hello",
        tool_calls=(ToolCall(id="call_1", name="search_library", arguments={"query": "cats"}),),
    ), f"content: {outcome.response.content!r}"


@respx.mock
async def test_generate_invalid_tool_arguments_returns_failed_value() -> None:
    broken = dict(FUNCTION_CALL_ITEM, arguments='{"query": ')
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[broken], usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(
        ROW, make_intent(tools=(SEARCH_TOOL,)), CREDENTIAL
    )
    assert isinstance(outcome, Failed), f"outcome: {outcome!r}"
    assert isinstance(outcome.failure, InvalidToolArguments), f"failure: {outcome.failure!r}"
    assert "search_library" in outcome.failure.safe_detail, (
        f"safe_detail: {outcome.failure.safe_detail!r}"
    )
    assert_meta(outcome)


@respx.mock
async def test_generate_maps_refusal_output_to_refused() -> None:
    refusal_item: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
    }
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[refusal_item], usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Refused), f"outcome: {outcome!r}"
    assert outcome.safe_detail == "I cannot help with that.", (
        f"safe_detail: {outcome.safe_detail!r}"
    )
    assert_meta(outcome)


@pytest.mark.parametrize(
    ("native_reason", "expected_reason"),
    [("max_output_tokens", "max_output_tokens"), ("content_filter", "content_filter_partial")],
)
@respx.mock
async def test_generate_maps_incomplete_statuses(native_reason: str, expected_reason: str) -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(
            envelope(
                output=[TEXT_ITEM],
                status="incomplete",
                usage=usage_body(),
                incomplete_details={"reason": native_reason},
            )
        )
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Incomplete), f"outcome: {outcome!r}"
    assert outcome.reason == expected_reason, f"reason: {outcome.reason!r}"
    assert outcome.status == "provider_incomplete", f"status: {outcome.status!r}"
    assert outcome.safe_detail == Present(native_reason), f"safe_detail: {outcome.safe_detail!r}"
    assert_meta(outcome)


@respx.mock
async def test_generate_unknown_incomplete_reason_is_protocol_defect() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(
            envelope(
                output=[TEXT_ITEM],
                status="incomplete",
                incomplete_details={"reason": "novel_reason"},
            )
        )
    )
    with pytest.raises(ProtocolDefect, match="novel_reason"):
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)


@respx.mock
async def test_generate_unknown_terminal_status_is_protocol_defect() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], status="queued"))
    )
    with pytest.raises(ProtocolDefect, match="queued"):
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)


@respx.mock
async def test_generate_strict_json_non_json_text_is_a_failed_value() -> None:
    """A provider answering a strict-JSON intent with unparseable output is an
    expected model failure — the leaf exists for exactly this — not a
    wire-protocol defect."""
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(
        ROW, make_intent(output=VERDICT_OUTPUT), CREDENTIAL
    )
    assert isinstance(outcome, Failed), f"outcome: {outcome!r}"
    assert isinstance(outcome.failure, InvalidStructuredOutput), f"failure: {outcome.failure!r}"
    assert "not valid JSON" in outcome.failure.safe_detail, (
        f"safe_detail: {outcome.failure.safe_detail!r}"
    )
    assert_meta(outcome)


@respx.mock
async def test_generate_strict_json_non_object_text_is_a_failed_value() -> None:
    scalar_item: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "[1, 2]", "annotations": []}],
    }
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[scalar_item], usage=usage_body()))
    )
    outcome = await OpenAIResponsesEngine().generate(
        ROW, make_intent(output=VERDICT_OUTPUT), CREDENTIAL
    )
    assert isinstance(outcome, Failed), f"outcome: {outcome!r}"
    assert isinstance(outcome.failure, InvalidStructuredOutput), f"failure: {outcome.failure!r}"
    assert "not a JSON object" in outcome.failure.safe_detail, (
        f"safe_detail: {outcome.failure.safe_detail!r}"
    )


@respx.mock
async def test_generate_malformed_json_envelope_is_protocol_defect() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "application/json"}, content=b"{not json"
        )
    )
    with pytest.raises(ProtocolDefect, match="not valid JSON"):
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)


@respx.mock
async def test_generate_missing_model_is_protocol_defect() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], model=None))
    )
    with pytest.raises(ProtocolDefect, match="model"):
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)


@respx.mock
async def test_generate_missing_output_is_protocol_defect() -> None:
    body = envelope(output=[])
    del body["output"]
    respx.post(RESPONSES_URL).mock(return_value=mock_response(body))
    with pytest.raises(ProtocolDefect, match="output"):
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)


@pytest.mark.parametrize(
    "usage",
    [
        {**usage_body(), "input_tokens": -5},
        {key: value for key, value in usage_body().items() if key != "input_tokens"},
        {**usage_body(), "output_tokens": "thirty"},
        {**usage_body(), "input_tokens_details": {"cached_tokens": -1}},
    ],
    ids=["negative-input", "missing-input", "non-int-output", "negative-cache-read"],
)
@respx.mock
async def test_generate_malformed_usage_is_protocol_defect(usage: dict[str, object]) -> None:
    """A 2xx envelope whose usage cannot be read as token accounting is a
    malformed envelope, not a silently zeroed count."""
    respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=[TEXT_ITEM], usage=usage))
    )
    with pytest.raises(ProtocolDefect, match="usage") as exc_info:
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.code == "malformed_usage", f"code: {exc_info.value.code!r}"
    assert exc_info.value.origin == "provider_response", f"origin: {exc_info.value.origin!r}"


# ---------------------------------------------------------------------------
# Fault injection — generate


@respx.mock
async def test_generate_rate_limit_with_retry_after_raises_transient() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "7", "x-request-id": "req_429"},
            json={"error": {"message": "slow down", "type": "rate_limit_error"}},
        )
    )
    with pytest.raises(TransientAttempt) as exc_info:
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    attempt = exc_info.value
    assert attempt.cause == ProviderRateLimit(retry_after=Present(7.0)), f"cause: {attempt.cause!r}"
    assert attempt.status_code == Present(429), f"status_code: {attempt.status_code!r}"
    assert attempt.provider_request_id == Present("req_429"), (
        f"provider_request_id: {attempt.provider_request_id!r}"
    )
    assert attempt.billability == PossiblyBillable(), f"billability: {attempt.billability!r}"


@respx.mock
async def test_generate_rate_limit_without_retry_after_raises_transient() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(429, json={"error": {"message": "slow down"}})
    )
    with pytest.raises(TransientAttempt) as exc_info:
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.cause == ProviderRateLimit(retry_after=Absent()), (
        f"cause: {exc_info.value.cause!r}"
    )


@respx.mock
async def test_generate_5xx_raises_transient_unavailable() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(503, json={"error": {"message": "overloaded"}})
    )
    with pytest.raises(TransientAttempt) as exc_info:
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.cause == ProviderHttpUnavailable(), f"cause: {exc_info.value.cause!r}"
    assert exc_info.value.status_code == Present(503), (
        f"status_code: {exc_info.value.status_code!r}"
    )


@respx.mock
async def test_generate_timeout_raises_transient_timeout() -> None:
    respx.post(RESPONSES_URL).mock(side_effect=httpx.ReadTimeout("read timed out"))
    with pytest.raises(TransientAttempt) as exc_info:
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.cause == ProviderTimeout(), f"cause: {exc_info.value.cause!r}"
    assert exc_info.value.billability == PossiblyBillable(), (
        f"a read timeout happened after the request was on the wire; "
        f"billability: {exc_info.value.billability!r}"
    )


@respx.mock
async def test_generate_connect_timeout_is_timeout_not_dispatched() -> None:
    """The SDK collapses every httpx timeout into APITimeoutError; the cause
    chain is the only place the pre-connect rule is still readable."""
    respx.post(RESPONSES_URL).mock(side_effect=httpx.ConnectTimeout("handshake timed out"))
    with pytest.raises(TransientAttempt) as exc_info:
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.cause == ProviderTimeout(), f"cause: {exc_info.value.cause!r}"
    assert exc_info.value.billability == NotDispatched(), (
        f"the handshake never completed, so no request bytes reached the provider; "
        f"billability: {exc_info.value.billability!r}"
    )


@respx.mock
async def test_generate_connect_error_raises_transport_unavailable_not_dispatched() -> None:
    respx.post(RESPONSES_URL).mock(side_effect=httpx.ConnectError("no route to host"))
    with pytest.raises(TransientAttempt) as exc_info:
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.cause == TransportUnavailable(), f"cause: {exc_info.value.cause!r}"
    assert exc_info.value.billability == NotDispatched(), (
        f"billability: {exc_info.value.billability!r}"
    )


@respx.mock
async def test_generate_context_overflow_returns_failed_value() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            400,
            headers={"x-request-id": "req_400"},
            json={
                "error": {
                    "message": "This model's maximum context length is 400000 tokens.",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                }
            },
        )
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Failed), f"outcome: {outcome!r}"
    assert outcome.failure == ProviderContextTooLarge(), f"failure: {outcome.failure!r}"
    assert_meta(outcome, request_id="req_400", usage=None, status_code=400)


@respx.mock
async def test_generate_context_overflow_detected_from_message_text() -> None:
    """No code on the body — the documented overflow sentence in the message is
    the only signal, and it must still resolve to the expected failure value."""
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            400,
            headers={"x-request-id": "req_400"},
            json={
                "error": {
                    "message": (
                        "This model's maximum context length is 400000 tokens, "
                        "however you requested 512000 tokens."
                    ),
                    "type": "invalid_request_error",
                }
            },
        )
    )
    outcome = await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert isinstance(outcome, Failed), f"outcome: {outcome!r}"
    assert outcome.failure == ProviderContextTooLarge(), f"failure: {outcome.failure!r}"
    assert_meta(outcome, request_id="req_400", usage=None, status_code=400)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (402, {"error": {"message": "payment required", "type": "billing_error"}}),
        (429, {"error": {"message": "You exceeded your quota", "code": "insufficient_quota"}}),
    ],
    ids=["402-payment-required", "429-insufficient-quota"],
)
@respx.mock
async def test_generate_quota_exhaustion_is_runtime_defect(
    status: int, body: dict[str, object]
) -> None:
    """Billing exhaustion is an operator fact, never a retryable rate limit —
    even when the provider signals it with HTTP 429."""
    respx.post(RESPONSES_URL).mock(return_value=httpx.Response(status, json=body))
    with pytest.raises(RuntimeDefect) as exc_info:
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.code == "quota_exhausted", f"code: {exc_info.value.code!r}"
    assert exc_info.value.origin == "provider_http", f"origin: {exc_info.value.origin!r}"


@respx.mock
async def test_generate_unclassified_4xx_is_runtime_defect() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            404,
            json={"error": {"message": "The model does not exist", "code": "model_not_found"}},
        )
    )
    with pytest.raises(RuntimeDefect) as exc_info:
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)
    assert exc_info.value.code == "unclassified_provider_error", f"code: {exc_info.value.code!r}"
    assert "model_not_found" in exc_info.value.message, f"message: {exc_info.value.message!r}"


@respx.mock
async def test_generate_credential_rejection_raises() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}},
        )
    )
    with pytest.raises(CredentialRejected, match="401"):
        await OpenAIResponsesEngine().generate(ROW, make_intent(), CREDENTIAL)


# ---------------------------------------------------------------------------
# Streaming


@respx.mock
async def test_stream_decodes_deltas_tool_calls_continuation_and_terminal() -> None:
    done_fc: dict[str, object] = {
        "id": "fc_item_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "search_library",
        "arguments": '{"query": "cats"}',
        "status": "completed",
    }
    done_reasoning = dict(REASONING_ITEM)
    done_message: dict[str, object] = {
        "id": "msg_s1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Hi there", "annotations": []}],
    }
    frames: list[dict[str, object]] = [
        created_frame(),
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "fc_item_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "search_library",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": "fc_item_1",
            "delta": '{"query":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 3,
            "output_index": 0,
            "item_id": "fc_item_1",
            "delta": ' "cats"}',
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "output_index": 0,
            "item": done_fc,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 5,
            "output_index": 1,
            "item": done_reasoning,
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 6,
            "output_index": 2,
            "item_id": "msg_s1",
            "content_index": 0,
            "logprobs": [],
            "delta": "Hi",
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 7,
            "output_index": 2,
            "item_id": "msg_s1",
            "content_index": 0,
            "logprobs": [],
            "delta": " there",
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 8,
            "output_index": 2,
            "item": done_message,
        },
        completed_frame(output=[done_fc, done_reasoning, done_message], usage=usage_body()),
    ]
    route = respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    intent = make_intent(tools=(SEARCH_TOOL,))
    events = await collect_stream(OpenAIResponsesEngine(), intent)

    body = request_body(route)
    assert body["stream"] is True, f"stream flag missing from request body: {body!r}"

    expected_tool_call = ToolCall(id="call_1", name="search_library", arguments={"query": "cats"})
    assert events[0] == StreamStart(), f"first event: {events[0]!r}"
    assert events[1] == ToolCallStart(call_id="call_1", name="search_library"), f"{events[1]!r}"
    assert events[2] == ToolCallDelta(call_id="call_1", arguments_delta='{"query":'), (
        f"{events[2]!r}"
    )
    assert events[3] == ToolCallDelta(call_id="call_1", arguments_delta=' "cats"}'), (
        f"{events[3]!r}"
    )
    assert events[4] == ToolCallDone(tool_call=expected_tool_call), f"{events[4]!r}"
    assert events[5] == TextDelta(text="Hi"), f"{events[5]!r}"
    assert events[6] == TextDelta(text=" there"), f"{events[6]!r}"

    continuation_delta = events[7]
    assert isinstance(continuation_delta, ContinuationDelta), f"{continuation_delta!r}"
    artifact = continuation_delta.artifact
    assert artifact.target == TARGET and artifact.codec_id == "openai.v1", f"{artifact!r}"
    assert list(artifact.opaque_payload["output"]) == [done_fc, done_reasoning, done_message], (  # type: ignore[arg-type]
        f"payload items must be the verbatim done items in order: {artifact.opaque_payload!r}"
    )

    terminal = events[8]
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
    assert len(events) == 9, f"unexpected extra events: {events!r}"


@respx.mock
async def test_stream_midcut_after_semantic_output_raises_partial_interrupt() -> None:
    frames = [
        created_frame(),
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "output_index": 0,
            "item_id": "msg_s1",
            "content_index": 0,
            "logprobs": [],
            "delta": "partial",
        },
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    events: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as exc_info:
        async for event in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
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
    respx.post(RESPONSES_URL).mock(return_value=mock_stream([created_frame()]))
    events: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as exc_info:
        async for event in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            events.append(event)
    assert events == [StreamStart()], f"events: {events!r}"
    assert exc_info.value.cause == ProviderStreamInterrupted(partial_output=False), (
        f"cause: {exc_info.value.cause!r}"
    )


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        (httpx.ReadTimeout("read timed out"), ProviderTimeout()),
        (httpx.RemoteProtocolError("peer closed connection mid-body"), TransportUnavailable()),
    ],
    ids=["read-timeout", "remote-protocol-error"],
)
@respx.mock
async def test_stream_transport_fault_mid_body_raises_transient(
    error: Exception, cause: ProviderTimeout | TransportUnavailable
) -> None:
    respx.post(RESPONSES_URL).mock(return_value=mock_broken_stream([created_frame()], error))
    events: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as exc_info:
        async for event in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            events.append(event)
    assert events == [StreamStart()], f"events: {events!r}"
    assert exc_info.value.cause == cause, f"cause: {exc_info.value.cause!r}"
    assert exc_info.value.status_code == Present(200), (
        f"status_code: {exc_info.value.status_code!r}"
    )
    assert exc_info.value.provider_request_id == Present("req_s1"), (
        f"provider_request_id: {exc_info.value.provider_request_id!r}"
    )
    assert exc_info.value.billability == PossiblyBillable(), (
        f"billability: {exc_info.value.billability!r}"
    )


def sdk_error_frame_bytes(error: dict[str, object]) -> bytes:
    """A frame carrying a top-level "error" key — the gateway/proxy shape the
    SDK decodes itself and raises as openai.APIError before our dispatch runs."""
    return b"event: error\ndata: " + json.dumps({"error": error}).encode() + b"\n\n"


@respx.mock
async def test_stream_sdk_decoded_error_frame_rate_limit_raises_transient() -> None:
    content = sse_bytes([created_frame()]) + sdk_error_frame_bytes(
        {"code": "rate_limit_exceeded", "message": "slow down"}
    )
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, headers=stream_headers("req_s1"), content=content)
    )
    with pytest.raises(TransientAttempt) as exc_info:
        async for _ in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass
    assert exc_info.value.cause == ProviderRateLimit(retry_after=Absent()), (
        f"cause: {exc_info.value.cause!r}"
    )
    assert exc_info.value.provider_request_id == Present("req_s1"), (
        f"provider_request_id: {exc_info.value.provider_request_id!r}"
    )


@respx.mock
async def test_stream_sdk_decoded_error_frame_defects_with_sanitized_text() -> None:
    content = sse_bytes([created_frame()]) + sdk_error_frame_bytes(
        {"message": "boom mid-stream from sk-not-a-real-key-1234567890"}
    )
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, headers=stream_headers("req_s1"), content=content)
    )
    with pytest.raises(ProtocolDefect) as exc_info:
        async for _ in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass
    message = exc_info.value.message
    assert exc_info.value.code == "provider_stream_failure", f"code: {exc_info.value.code!r}"
    assert "boom mid-stream" in message, f"message: {message!r}"
    assert "sk-not-a-real-key-1234567890" not in message, f"unredacted secret: {message!r}"


@respx.mock
async def test_stream_inband_error_event_rate_limit_raises_transient() -> None:
    frames: list[dict[str, object]] = [
        created_frame(),
        {
            "type": "error",
            "sequence_number": 1,
            "code": "rate_limit_exceeded",
            "message": "Too many tokens",
            "param": None,
        },
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    with pytest.raises(TransientAttempt) as exc_info:
        async for _ in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass
    assert exc_info.value.cause == ProviderRateLimit(retry_after=Absent()), (
        f"cause: {exc_info.value.cause!r}"
    )


@respx.mock
async def test_stream_failed_frame_server_error_raises_transient() -> None:
    frames: list[dict[str, object]] = [
        created_frame(),
        {
            "type": "response.failed",
            "sequence_number": 1,
            "response": {
                "id": "resp_s1",
                "status": "failed",
                "model": "gpt-test-1",
                "error": {"code": "server_error", "message": "The model had an error"},
            },
        },
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    with pytest.raises(TransientAttempt) as exc_info:
        async for _ in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass
    assert exc_info.value.cause == ProviderHttpUnavailable(), f"cause: {exc_info.value.cause!r}"


@respx.mock
async def test_stream_unknown_inband_error_is_protocol_defect() -> None:
    frames: list[dict[str, object]] = [
        created_frame(),
        {"type": "error", "sequence_number": 1, "code": "invalid_prompt", "message": "bad prompt"},
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    with pytest.raises(ProtocolDefect, match="invalid_prompt"):
        async for _ in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass


@respx.mock
async def test_stream_refusal_folds_into_incomplete_refused_terminal() -> None:
    frames: list[dict[str, object]] = [
        created_frame(),
        {
            "type": "response.refusal.delta",
            "sequence_number": 1,
            "output_index": 0,
            "item_id": "msg_s1",
            "content_index": 0,
            "delta": "I cannot ",
        },
        {
            "type": "response.refusal.delta",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": "msg_s1",
            "content_index": 0,
            "delta": "help with that.",
        },
        completed_frame(output=[], usage=usage_body()),
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(OpenAIResponsesEngine(), make_intent())
    assert events[0] == StreamStart(), f"events: {events!r}"
    terminal = events[1]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Incomplete), f"terminal outcome: {outcome!r}"
    assert outcome.status == "refused", f"status: {outcome.status!r}"
    assert outcome.reason == "content_filter_partial", f"reason: {outcome.reason!r}"
    assert outcome.safe_detail == Present("I cannot help with that."), (
        f"safe_detail: {outcome.safe_detail!r}"
    )
    assert len(events) == 2, f"no ContinuationDelta on refusal: {events!r}"


@respx.mock
async def test_stream_incomplete_terminal() -> None:
    frames: list[dict[str, object]] = [
        created_frame(),
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "output_index": 0,
            "item_id": "msg_s1",
            "content_index": 0,
            "logprobs": [],
            "delta": "truncat",
        },
        {
            "type": "response.incomplete",
            "sequence_number": 2,
            "response": envelope(
                output=[],
                status="incomplete",
                usage=usage_body(),
                incomplete_details={"reason": "max_output_tokens"},
                response_id="resp_s1",
            ),
        },
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(OpenAIResponsesEngine(), make_intent())
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Incomplete), f"terminal outcome: {outcome!r}"
    assert outcome.reason == "max_output_tokens", f"reason: {outcome.reason!r}"
    assert outcome.status == "provider_incomplete", f"status: {outcome.status!r}"
    assert_meta(outcome, request_id="req_s1")


@respx.mock
async def test_stream_falls_back_to_envelope_id_when_header_missing() -> None:
    frames = [created_frame(), completed_frame(output=[], usage=usage_body())]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames, request_id=None))
    events = await collect_stream(OpenAIResponsesEngine(), make_intent())
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded), f"terminal outcome: {outcome!r}"
    assert outcome.meta.provider_request_id == Present("resp_s1"), (
        f"provider_request_id: {outcome.meta.provider_request_id!r}"
    )


@respx.mock
async def test_stream_malformed_usage_is_protocol_defect() -> None:
    frames = [
        created_frame(),
        completed_frame(output=[], usage={**usage_body(), "output_tokens": -3}),
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    with pytest.raises(ProtocolDefect, match="usage") as exc_info:
        async for _ in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass
    assert exc_info.value.code == "malformed_usage", f"code: {exc_info.value.code!r}"


@respx.mock
async def test_stream_tool_argument_parse_failure_yields_failed_terminal() -> None:
    frames: list[dict[str, object]] = [
        created_frame(),
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "fc_item_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "search_library",
                "arguments": "",
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 2,
            "output_index": 0,
            "item": {
                "id": "fc_item_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "search_library",
                "arguments": '{"query": ',
                "status": "completed",
            },
        },
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(OpenAIResponsesEngine(), make_intent(tools=(SEARCH_TOOL,)))
    assert events[0] == StreamStart(), f"events: {events!r}"
    assert events[1] == ToolCallStart(call_id="call_1", name="search_library"), f"{events[1]!r}"
    terminal = events[2]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Failed), f"terminal outcome: {outcome!r}"
    assert isinstance(outcome.failure, InvalidToolArguments), f"failure: {outcome.failure!r}"
    assert len(events) == 3, f"stream must end at the failed terminal: {events!r}"


@respx.mock
async def test_stream_context_overflow_yields_failed_terminal_without_stream_start() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            400,
            headers={"x-request-id": "req_400"},
            json={"error": {"message": "too long", "code": "context_length_exceeded"}},
        )
    )
    events = await collect_stream(OpenAIResponsesEngine(), make_intent())
    assert len(events) == 1, f"expected a single terminal event: {events!r}"
    terminal = events[0]
    assert isinstance(terminal, TerminalEvent), f"{terminal!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Failed), f"terminal outcome: {outcome!r}"
    assert outcome.failure == ProviderContextTooLarge(), f"failure: {outcome.failure!r}"
    assert_meta(outcome, request_id="req_400", usage=None, status_code=400)


@respx.mock
async def test_stream_credential_rejection_raises_before_any_event() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(403, json={"error": {"message": "forbidden"}})
    )
    events: list[CodecStreamEvent] = []
    with pytest.raises(CredentialRejected, match="403"):
        async for event in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            events.append(event)
    assert events == [], f"no events may precede credential rejection: {events!r}"


@respx.mock
async def test_stream_rate_limit_at_accept_raises_transient() -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            429, headers={"retry-after": "3"}, json={"error": {"message": "slow down"}}
        )
    )
    with pytest.raises(TransientAttempt) as exc_info:
        async for _ in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass
    assert exc_info.value.cause == ProviderRateLimit(retry_after=Present(3.0)), (
        f"cause: {exc_info.value.cause!r}"
    )


@respx.mock
async def test_stream_malformed_frame_is_protocol_defect() -> None:
    content = sse_bytes([created_frame()]) + b"event: response.completed\ndata: {broken\n\n"
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=content
        )
    )
    with pytest.raises(ProtocolDefect, match="stream frame"):
        async for _ in OpenAIResponsesEngine().stream(ROW, make_intent(), CREDENTIAL):
            pass


@respx.mock
async def test_stream_strict_json_terminal_promotes_structured_content() -> None:
    done_message: dict[str, object] = {
        "id": "msg_s1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": '{"verdict": "yes"}', "annotations": []}],
    }
    frames: list[dict[str, object]] = [
        created_frame(),
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "output_index": 0,
            "item_id": "msg_s1",
            "content_index": 0,
            "logprobs": [],
            "delta": '{"verdict": "yes"}',
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 2,
            "output_index": 0,
            "item": done_message,
        },
        completed_frame(output=[done_message], usage=usage_body()),
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(OpenAIResponsesEngine(), make_intent(output=VERDICT_OUTPUT))
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded), f"terminal outcome: {outcome!r}"
    assert outcome.response.content == StructuredContent(
        payload={"verdict": "yes"}, text='{"verdict": "yes"}'
    ), f"content: {outcome.response.content!r}"


@respx.mock
async def test_stream_strict_json_non_json_terminal_is_a_failed_value() -> None:
    """Same conformance on the stream arm: unparseable strict-JSON output ends
    the envelope as Failed(InvalidStructuredOutput), never a defect — and no
    ContinuationDelta rides along with a failed terminal."""
    done_message: dict[str, object] = {
        "id": "msg_s1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "not json at all", "annotations": []}],
    }
    frames: list[dict[str, object]] = [
        created_frame(),
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "output_index": 0,
            "item_id": "msg_s1",
            "content_index": 0,
            "logprobs": [],
            "delta": "not json at all",
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 2,
            "output_index": 0,
            "item": done_message,
        },
        completed_frame(output=[done_message], usage=usage_body()),
    ]
    respx.post(RESPONSES_URL).mock(return_value=mock_stream(frames))
    events = await collect_stream(OpenAIResponsesEngine(), make_intent(output=VERDICT_OUTPUT))
    assert not any(isinstance(event, ContinuationDelta) for event in events), (
        f"a failed terminal carries no continuation; events: {events!r}"
    )
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent), f"events: {events!r}"
    outcome = terminal.outcome
    assert isinstance(outcome, Failed), f"terminal outcome: {outcome!r}"
    assert isinstance(outcome.failure, InvalidStructuredOutput), f"failure: {outcome.failure!r}"


# ---------------------------------------------------------------------------
# Continuation round-trip: decode → replay encodes verbatim


@respx.mock
async def test_continuation_round_trip_replays_decoded_items_verbatim() -> None:
    first_output = [REASONING_ITEM, FUNCTION_CALL_ITEM]
    route = respx.post(RESPONSES_URL).mock(
        return_value=mock_response(envelope(output=first_output, usage=usage_body()))
    )
    engine = OpenAIResponsesEngine()
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
            ToolResultMessage(call_id="call_1", output="42", is_error=False),
        ),
        tools=(SEARCH_TOOL,),
    )
    second = await engine.generate(ROW, follow_up, CREDENTIAL)
    assert isinstance(second, Succeeded), f"second outcome: {second!r}"
    body = request_body(route)
    assert body["input"] == [
        {"role": "system", "content": [{"type": "input_text", "text": "You are terse."}]},
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        REASONING_ITEM,
        FUNCTION_CALL_ITEM,
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
    ], f"replayed input must carry the decoded items verbatim: {body.get('input')!r}"
