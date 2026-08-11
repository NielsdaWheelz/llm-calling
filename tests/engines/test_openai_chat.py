"""openai_chat engine conformance + fault injection (respx at the HTTP boundary).

One engine, four provider quirk-sets (deepseek, moonshot, xai, openrouter) over
the openai SDK as a compat client. Fixture rows are constructed locally — tests
never depend on registry ROWS content. Covered per the freeze: exact request
body/header shapes, response decode, stream decode from raw SSE bytes,
continuation round-trips (verbatim replay / reasoning_content strip / verbatim
reasoning_details), the row's reasoning fragment merged verbatim, provider_options
passthrough vs collision, and the full fault-injection table.
"""

import json
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import replace

import httpx
import pytest
import respx

from provider_runtime.engines import TransientAttempt
from provider_runtime.engines.openai_chat import OpenAIChatEngine
from provider_runtime.errors import (
    CredentialRejected,
    InvalidRequest,
    ProtocolDefect,
    RuntimeDefect,
)
from provider_runtime.registry import REGISTRY_REVISION, ModelRow, OpenRouterRouting
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
    StreamStart,
    StrictJsonOutput,
    StructuredContent,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextContent,
    TextDelta,
    TextOutput,
    ToolCall,
    ToolCallDelta,
    ToolCallDone,
    ToolCallStart,
    ToolResultMessage,
    TransportUnavailable,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Fixture rows — local to this file by design (registry ROWS content is
# another agent's concern).

# A row's reasoning value is a self-describing wire fragment merged verbatim
# into the request; these are the real per-provider shapes, never synthetic
# stand-ins (an engine that cannot build a callable request must fail here).
DEEPSEEK_REASONING: Mapping[ReasoningLevel, object] = {
    "none": {"thinking": {"type": "disabled"}},
    "high": {"thinking": {"type": "enabled"}},
}
MOONSHOT_REASONING: Mapping[ReasoningLevel, object] = {
    # Omitting the knob is Moonshot's "off" — an empty fragment sends nothing.
    "none": {},
    "low": {"reasoning_effort": "low"},
    "high": {"reasoning_effort": "high"},
    "max": {"reasoning_effort": "max"},
}
XAI_REASONING: Mapping[ReasoningLevel, object] = {
    "low": {"reasoning_effort": "low"},
    "high": {"reasoning_effort": "high"},
}
OPENROUTER_REASONING: Mapping[ReasoningLevel, object] = {
    "low": {"reasoning": {"effort": "low"}},
    "high": {"reasoning": {"effort": "high"}},
}

DEEPSEEK_ROW = ModelRow(
    ref="deepseek:reasoner",
    provider="deepseek",
    model_id="deepseek-reasoner",
    engine="openai_chat",
    base_url=Present("https://api.deepseek.com/v1"),
    context_window=131_072,
    max_output_tokens=65_536,
    modalities=frozenset({"text"}),
    tools=True,
    streaming=True,
    structured="json_mode",
    reasoning=Present(DEEPSEEK_REASONING),
    continuation_codec="deepseek.v1",
    correlation="in_band",
    routing=Absent(),
)

MOONSHOT_ROW = ModelRow(
    ref="moonshot:kimi-k3",
    provider="moonshot",
    model_id="kimi-k3",
    engine="openai_chat",
    base_url=Present("https://api.moonshot.ai/v1"),
    context_window=1_048_576,
    max_output_tokens=131_072,
    modalities=frozenset({"text"}),
    tools=True,
    streaming=True,
    structured="json_mode",
    reasoning=Present(MOONSHOT_REASONING),
    continuation_codec="moonshot.v1",
    correlation="in_band",
    routing=Absent(),
)

XAI_ROW = ModelRow(
    ref="xai:grok-4",
    provider="xai",
    model_id="grok-4",
    engine="openai_chat",
    base_url=Present("https://api.x.ai/v1"),
    context_window=256_000,
    max_output_tokens=64_000,
    modalities=frozenset({"text", "image"}),
    tools=True,
    streaming=True,
    structured="native",
    reasoning=Present(XAI_REASONING),
    continuation_codec="xai.v1",
    correlation="in_band",
    routing=Absent(),
)

OPENROUTER_ROW = ModelRow(
    ref="openrouter:kimi-k3",
    provider="openrouter",
    model_id="moonshotai/kimi-k3",
    engine="openai_chat",
    base_url=Present("https://openrouter.ai/api/v1"),
    context_window=1_048_576,
    max_output_tokens=131_072,
    modalities=frozenset({"text"}),
    tools=True,
    streaming=True,
    structured="json_mode",
    reasoning=Present(OPENROUTER_REASONING),
    continuation_codec="openrouter.v1",
    correlation="in_band",
    routing=Present(
        OpenRouterRouting(only=("moonshotai",), order=("moonshotai",), quantizations=("int4",))
    ),
)

# A model with no reasoning knob at all (row.reasoning Absent).
KNOBLESS_ROW = ModelRow(
    ref="deepseek:chat",
    provider="deepseek",
    model_id="deepseek-chat",
    engine="openai_chat",
    base_url=Present("https://api.deepseek.com/v1"),
    context_window=131_072,
    max_output_tokens=8_192,
    modalities=frozenset({"text"}),
    tools=True,
    streaming=True,
    structured="json_mode",
    reasoning=Absent(),
    continuation_codec="deepseek.v1",
    correlation="in_band",
    routing=Absent(),
)

EXPECTED_PINS = {
    "only": ["moonshotai"],
    "order": ["moonshotai"],
    "allow_fallbacks": False,
    "require_parameters": True,
    "data_collection": "deny",
    "zdr": True,
    "quantizations": ["int4"],
}

SEARCH_TOOL = CanonicalTool(
    name="search",
    description="Search the corpus.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)

ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def credential_for(row: ModelRow) -> ProviderCredential:
    return ProviderCredential(provider=row.provider, key="test-key")


def intent_for(
    row: ModelRow,
    *,
    messages: tuple[PromptMessage, ...] | None = None,
    reasoning: ReasoningLevel = "high",
    tools: tuple[CanonicalTool, ...] = (),
    output: TextOutput | StrictJsonOutput | None = None,
    provider_options: dict[str, object] | None = None,
) -> GenerateIntent:
    return GenerateIntent(
        target=ProviderTarget(provider=row.provider, model=row.model_id),
        messages=messages
        or (SystemMessage((PromptBlock("be brief"),)), UserMessage((PromptBlock("hi"),))),
        max_output_tokens=512,
        reasoning=reasoning,
        tools=tools,
        tool_choice="auto",
        output=output or TextOutput(),
        provider_options=provider_options or {},
    )


def completion_body(
    *,
    model: str,
    message: dict[str, object] | None = None,
    finish_reason: str | None = "stop",
    usage: dict[str, object] | None = None,
    **top_level: object,
) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "resp-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message or {"role": "assistant", "content": "hello"},
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    body.update(top_level)
    return body


def chat_url(row: ModelRow) -> str:
    assert isinstance(row.base_url, Present), f"fixture row {row.ref} must carry a base_url"
    return row.base_url.value + "/chat/completions"


def mock_completion(row: ModelRow, body: dict[str, object], status: int = 200) -> respx.Route:
    return respx.post(chat_url(row)).mock(return_value=httpx.Response(status, json=body))


def last_request_json(route: respx.Route) -> dict[str, object]:
    assert route.called, "expected the engine to dispatch an HTTP request"
    parsed = json.loads(route.calls.last.request.content)
    assert isinstance(parsed, dict)
    return parsed


def sse_bytes(*events: dict[str, object] | str) -> bytes:
    frames = []
    for event in events:
        data = event if isinstance(event, str) else json.dumps(event)
        frames.append(f"data: {data}\n\n".encode())
    return b"".join(frames)


def mock_stream(row: ModelRow, payload: bytes) -> respx.Route:
    return respx.post(chat_url(row)).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=payload
        )
    )


async def collect(events: object) -> list[CodecStreamEvent]:
    collected: list[CodecStreamEvent] = []
    async for event in events:  # type: ignore[union-attr]
        collected.append(event)
    return collected


class CutByteStream(httpx.AsyncByteStream):
    """Yields the given chunks, then dies with a transport error (mid-stream cut)."""

    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        raise httpx.ReadError("connection cut mid-stream")


@pytest.fixture
def engine() -> OpenAIChatEngine:
    return OpenAIChatEngine()


# ---------------------------------------------------------------------------
# Request conformance — one test per provider quirk-set.


@respx.mock
async def test_moonshot_request_uses_max_completion_tokens_and_reasoning_effort(
    engine: OpenAIChatEngine,
) -> None:
    route = mock_completion(MOONSHOT_ROW, completion_body(model="kimi-k3"))
    await engine.generate(
        MOONSHOT_ROW, intent_for(MOONSHOT_ROW, reasoning="max"), credential_for(MOONSHOT_ROW)
    )
    body = last_request_json(route)
    assert body["model"] == "kimi-k3", f"body: {body}"
    assert body["max_completion_tokens"] == 512, (
        f"moonshot must use max_completion_tokens; body: {body}"
    )
    assert "max_tokens" not in body, f"moonshot must not send deprecated max_tokens; body: {body}"
    assert body["reasoning_effort"] == "max", (
        f"the row's max fragment merges verbatim; body: {body}"
    )
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ], f"body: {body}"
    auth = route.calls.last.request.headers["authorization"]
    assert auth == "Bearer test-key", f"credential must ride the Authorization header, got {auth!r}"


@respx.mock
async def test_deepseek_request_uses_max_tokens_and_row_reasoning_fragment(
    engine: OpenAIChatEngine,
) -> None:
    route = mock_completion(DEEPSEEK_ROW, completion_body(model="deepseek-reasoner"))
    outcome = await engine.generate(
        DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW, reasoning="high"), credential_for(DEEPSEEK_ROW)
    )
    body = last_request_json(route)
    assert body["max_tokens"] == 512, f"deepseek must use max_tokens; body: {body}"
    assert "reasoning_effort" not in body, f"the engine invents no knob shape; body: {body}"
    assert body["thinking"] == {"type": "enabled"}, (
        f"the row's fragment must merge verbatim into the body; body: {body}"
    )
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.native_reasoning == Present('{"thinking":{"type":"enabled"}}'), (
        f"native_reasoning is the fragment that went on the wire, as compact sorted-keys "
        f"JSON; got {outcome.meta.native_reasoning}"
    )


@respx.mock
async def test_deepseek_reasoning_none_sends_the_disabling_fragment(
    engine: OpenAIChatEngine,
) -> None:
    route = mock_completion(DEEPSEEK_ROW, completion_body(model="deepseek-reasoner"))
    outcome = await engine.generate(
        DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW, reasoning="none"), credential_for(DEEPSEEK_ROW)
    )
    body = last_request_json(route)
    assert body["thinking"] == {"type": "disabled"}, (
        f"'none' is whatever the row says it is — here, thinking off; body: {body}"
    )
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.native_reasoning == Present('{"thinking":{"type":"disabled"}}'), (
        f"got {outcome.meta.native_reasoning}"
    )


@respx.mock
async def test_empty_reasoning_fragment_sends_nothing_and_reports_absent(
    engine: OpenAIChatEngine,
) -> None:
    route = mock_completion(MOONSHOT_ROW, completion_body(model="kimi-k3"))
    outcome = await engine.generate(
        MOONSHOT_ROW, intent_for(MOONSHOT_ROW, reasoning="none"), credential_for(MOONSHOT_ROW)
    )
    body = last_request_json(route)
    assert "reasoning_effort" not in body, f"an empty fragment sends nothing; body: {body}"
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.native_reasoning == Absent(), (
        f"nothing was sent, so native_reasoning must be Absent, got {outcome.meta.native_reasoning}"
    )


async def test_non_mapping_reasoning_value_is_a_registry_defect(engine: OpenAIChatEngine) -> None:
    row = replace(MOONSHOT_ROW, reasoning=Present({"high": "high"}))
    with pytest.raises(RuntimeDefect) as excinfo:
        await engine.generate(row, intent_for(row), credential_for(row))
    assert excinfo.value.code == "registry_invalid", f"got {excinfo.value.code}"


@respx.mock
async def test_xai_request_native_structured_output_and_reasoning_effort(
    engine: OpenAIChatEngine,
) -> None:
    route = mock_completion(
        XAI_ROW,
        completion_body(
            model="grok-4",
            message={"role": "assistant", "content": '{"answer": "42"}'},
        ),
    )
    outcome = await engine.generate(
        XAI_ROW,
        intent_for(XAI_ROW, reasoning="low", output=StrictJsonOutput("answer", ANSWER_SCHEMA)),
        credential_for(XAI_ROW),
    )
    body = last_request_json(route)
    assert body["max_completion_tokens"] == 512, f"xai must use max_completion_tokens; body: {body}"
    assert body["reasoning_effort"] == "low", f"body: {body}"
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": ANSWER_SCHEMA, "strict": True},
    }, f"structured='native' rows must use json_schema; body: {body}"
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == StructuredContent(
        payload={"answer": "42"}, text='{"answer": "42"}'
    ), f"StrictJsonOutput must decode to StructuredContent, got {outcome.response.content}"


@respx.mock
async def test_json_mode_row_sends_json_object_response_format(engine: OpenAIChatEngine) -> None:
    route = mock_completion(
        MOONSHOT_ROW,
        completion_body(
            model="kimi-k3", message={"role": "assistant", "content": '{"answer": "x"}'}
        ),
    )
    await engine.generate(
        MOONSHOT_ROW,
        intent_for(MOONSHOT_ROW, output=StrictJsonOutput("answer", ANSWER_SCHEMA)),
        credential_for(MOONSHOT_ROW),
    )
    body = last_request_json(route)
    assert body["response_format"] == {"type": "json_object"}, (
        f"structured='json_mode' rows must use json_object, never json_schema; body: {body}"
    )


@respx.mock
async def test_openrouter_request_sends_pins_reasoning_and_max_tokens(
    engine: OpenAIChatEngine,
) -> None:
    route = mock_completion(OPENROUTER_ROW, completion_body(model="moonshotai/kimi-k3"))
    await engine.generate(
        OPENROUTER_ROW, intent_for(OPENROUTER_ROW, reasoning="high"), credential_for(OPENROUTER_ROW)
    )
    body = last_request_json(route)
    assert body["provider"] == EXPECTED_PINS, (
        f"the full routing pins object must ride EVERY openrouter call; body: {body}"
    )
    assert body["reasoning"] == {"effort": "high"}, (
        f"the row's unified-reasoning fragment merges verbatim; body: {body}"
    )
    assert "reasoning_effort" not in body, f"body: {body}"
    assert body["max_tokens"] == 512, f"openrouter must use routed max_tokens; body: {body}"


@respx.mock
async def test_openrouter_pins_ride_plain_text_calls_too(engine: OpenAIChatEngine) -> None:
    route = mock_completion(OPENROUTER_ROW, completion_body(model="moonshotai/kimi-k3"))
    await engine.generate(
        OPENROUTER_ROW,
        intent_for(OPENROUTER_ROW, reasoning="low", output=TextOutput()),
        credential_for(OPENROUTER_ROW),
    )
    body = last_request_json(route)
    assert body["provider"] == EXPECTED_PINS, (
        f"no unpinned passthrough — pins must be present even on minimal calls; body: {body}"
    )


@respx.mock
async def test_tools_and_tool_results_encode_to_chat_completions_wire(
    engine: OpenAIChatEngine,
) -> None:
    route = mock_completion(MOONSHOT_ROW, completion_body(model="kimi-k3"))
    messages: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("find it"),)),
        AssistantMessage(
            text="",
            tool_calls=(ToolCall(id="call-1", name="search", arguments={"query": "x"}),),
            continuation=Absent(),
        ),
        ToolResultMessage(call_id="call-1", output="found", is_error=False),
    )
    await engine.generate(
        MOONSHOT_ROW,
        intent_for(MOONSHOT_ROW, messages=messages, tools=(SEARCH_TOOL,)),
        credential_for(MOONSHOT_ROW),
    )
    body = last_request_json(route)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the corpus.",
                "parameters": SEARCH_TOOL.parameters,
            },
        }
    ], f"body: {body}"
    assert body["tool_choice"] == "auto", f"body: {body}"
    encoded = body["messages"]
    assert isinstance(encoded, list)
    assert encoded[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query":"x"}'},
            }
        ],
    }, f"empty text alongside tool calls must encode as null content; messages: {encoded}"
    assert encoded[2] == {"role": "tool", "tool_call_id": "call-1", "content": "found"}, (
        f"messages: {encoded}"
    )


@respx.mock
async def test_image_blocks_encode_as_data_url_content_parts(engine: OpenAIChatEngine) -> None:
    route = mock_completion(XAI_ROW, completion_body(model="grok-4"))
    png = b"\x89PNG"
    messages: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("look:"), ImageBlock(media_type="image/png", data=png))),
    )
    await engine.generate(
        XAI_ROW, intent_for(XAI_ROW, messages=messages, reasoning="low"), credential_for(XAI_ROW)
    )
    body = last_request_json(route)
    expected_url = "data:image/png;base64," + b64encode(png).decode("ascii")
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look:"},
                {"type": "image_url", "image_url": {"url": expected_url}},
            ],
        }
    ], f"body: {body}"


# ---------------------------------------------------------------------------
# provider_options — extension passthrough, never overrides.


@respx.mock
async def test_provider_options_unknown_keys_are_forwarded(engine: OpenAIChatEngine) -> None:
    route = mock_completion(DEEPSEEK_ROW, completion_body(model="deepseek-reasoner"))
    await engine.generate(
        DEEPSEEK_ROW,
        intent_for(DEEPSEEK_ROW, provider_options={"temperature": 0.2, "top_p": 0.9}),
        credential_for(DEEPSEEK_ROW),
    )
    body = last_request_json(route)
    assert body["temperature"] == 0.2, f"unknown provider_options must be forwarded; body: {body}"
    assert body["top_p"] == 0.9, f"body: {body}"


@pytest.mark.parametrize(
    ("row", "key"),
    [
        (MOONSHOT_ROW, "max_completion_tokens"),
        (OPENROUTER_ROW, "provider"),
        (OPENROUTER_ROW, "max_tokens"),
        (DEEPSEEK_ROW, "response_format"),
        (XAI_ROW, "messages"),
        # Row-mapped reasoning fragment keys are owned exactly like the
        # structural ones — whatever shape the row happens to use.
        (MOONSHOT_ROW, "reasoning_effort"),
        (XAI_ROW, "reasoning_effort"),
        (OPENROUTER_ROW, "reasoning"),
        (DEEPSEEK_ROW, "thinking"),
    ],
)
async def test_provider_options_owned_key_collision_raises_invalid_request(
    engine: OpenAIChatEngine, row: ModelRow, key: str
) -> None:
    with pytest.raises(InvalidRequest, match=key):
        await engine.generate(
            row, intent_for(row, provider_options={key: "boom"}), credential_for(row)
        )


@respx.mock
async def test_provider_options_cannot_silently_override_the_reasoning_fragment(
    engine: OpenAIChatEngine,
) -> None:
    # The regression this guards: a passthrough key that happens to be the
    # row's reasoning key would win the merge, leaving native_reasoning
    # describing something that never reached the wire.
    route = mock_completion(DEEPSEEK_ROW, completion_body(model="deepseek-reasoner"))
    with pytest.raises(InvalidRequest, match="thinking"):
        await engine.generate(
            DEEPSEEK_ROW,
            intent_for(
                DEEPSEEK_ROW,
                reasoning="high",
                provider_options={"thinking": {"type": "disabled"}},
            ),
            credential_for(DEEPSEEK_ROW),
        )
    assert not route.called, "the request must never be dispatched"


# ---------------------------------------------------------------------------
# Reasoning-level mapping.


async def test_reasoning_level_outside_row_mapping_raises_invalid_request(
    engine: OpenAIChatEngine,
) -> None:
    with pytest.raises(InvalidRequest, match="minimal"):
        await engine.generate(
            MOONSHOT_ROW,
            intent_for(MOONSHOT_ROW, reasoning="minimal"),
            credential_for(MOONSHOT_ROW),
        )


async def test_reasoning_on_knobless_row_raises_invalid_request(engine: OpenAIChatEngine) -> None:
    row = KNOBLESS_ROW
    with pytest.raises(InvalidRequest, match="reasoning"):
        await engine.generate(row, intent_for(row, reasoning="high"), credential_for(row))


@respx.mock
async def test_knobless_row_with_reasoning_none_sends_no_reasoning_field(
    engine: OpenAIChatEngine,
) -> None:
    row = KNOBLESS_ROW
    route = mock_completion(row, completion_body(model="deepseek-chat"))
    outcome = await engine.generate(row, intent_for(row, reasoning="none"), credential_for(row))
    body = last_request_json(route)
    assert "reasoning" not in body and "reasoning_effort" not in body, f"body: {body}"
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.native_reasoning == Absent()


# ---------------------------------------------------------------------------
# Response decode + CallMeta.


@respx.mock
async def test_success_decode_populates_meta_and_usage(engine: OpenAIChatEngine) -> None:
    mock_completion(
        MOONSHOT_ROW,
        completion_body(
            model="kimi-k3",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cached_tokens": 64,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        ),
    )
    outcome = await engine.generate(
        MOONSHOT_ROW, intent_for(MOONSHOT_ROW), credential_for(MOONSHOT_ROW)
    )
    assert isinstance(outcome, Succeeded), f"got {outcome}"
    assert outcome.response.content == TextContent(text="hello", tool_calls=())
    meta = outcome.meta
    assert meta.provider == "moonshot"
    assert meta.model == "kimi-k3"
    assert meta.provider_request_id == Present("resp-1"), (
        f"in-band id is the request id, got {meta.provider_request_id}"
    )
    assert meta.upstream_provider == Absent(), "only openrouter reports an upstream provider"
    assert meta.registry_revision == REGISTRY_REVISION
    assert meta.native_reasoning == Present('{"reasoning_effort":"high"}')
    assert meta.billability == PossiblyBillable()
    assert isinstance(meta.usage, Present)
    usage = meta.usage.value
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (100, 20, 120)
    assert usage.cache_read_input_tokens == Present(64), (
        f"moonshot flat cached_tokens must map to cache_read, got {usage.cache_read_input_tokens}"
    )
    assert usage.reasoning_tokens == Present(7)
    assert len(meta.attempt_trace) == 1, f"trace: {meta.attempt_trace}"
    record = meta.attempt_trace[0]
    assert record.attempt == 1
    assert isinstance(record.signal, FinalAttempt)
    assert record.status_code == Present(200)


@respx.mock
async def test_openrouter_upstream_provider_and_cache_details_decode(
    engine: OpenAIChatEngine,
) -> None:
    mock_completion(
        OPENROUTER_ROW,
        completion_body(
            model="moonshotai/kimi-k3",
            provider="Moonshot",
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 5,
                "total_tokens": 55,
                "cost": 0.0012,
                "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
            },
        ),
    )
    outcome = await engine.generate(
        OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
    )
    assert isinstance(outcome, Succeeded), f"got {outcome}"
    assert outcome.meta.upstream_provider == Present("Moonshot"), (
        f"upstream provider comes from the response body, got {outcome.meta.upstream_provider}"
    )
    assert isinstance(outcome.meta.usage, Present)
    usage = outcome.meta.usage.value
    assert usage.cache_read_input_tokens == Present(30)
    assert usage.cache_write_input_tokens == Present(10)


@respx.mock
async def test_tool_call_decode_strict_parses_arguments(engine: OpenAIChatEngine) -> None:
    mock_completion(
        XAI_ROW,
        completion_body(
            model="grok-4",
            finish_reason="tool_calls",
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-9",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query": "cats"}'},
                    }
                ],
            },
        ),
    )
    outcome = await engine.generate(
        XAI_ROW, intent_for(XAI_ROW, tools=(SEARCH_TOOL,)), credential_for(XAI_ROW)
    )
    assert isinstance(outcome, Succeeded), f"got {outcome}"
    assert outcome.response.content == TextContent(
        text="",
        tool_calls=(ToolCall(id="call-9", name="search", arguments={"query": "cats"}),),
    )


@respx.mock
async def test_invalid_tool_arguments_return_failed_value(engine: OpenAIChatEngine) -> None:
    mock_completion(
        XAI_ROW,
        completion_body(
            model="grok-4",
            finish_reason="tool_calls",
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-9",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query": '},
                    }
                ],
            },
        ),
    )
    outcome = await engine.generate(
        XAI_ROW, intent_for(XAI_ROW, tools=(SEARCH_TOOL,)), credential_for(XAI_ROW)
    )
    assert isinstance(outcome, Failed), f"strict parse, no repair — got {outcome}"
    assert isinstance(outcome.failure, InvalidToolArguments), f"got {outcome.failure}"
    assert outcome.meta.billability == PossiblyBillable()
    assert len(outcome.meta.attempt_trace) == 1


@respx.mock
async def test_finish_reason_length_and_content_filter_map_to_incomplete(
    engine: OpenAIChatEngine,
) -> None:
    mock_completion(
        DEEPSEEK_ROW, completion_body(model="deepseek-reasoner", finish_reason="length")
    )
    outcome = await engine.generate(
        DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW)
    )
    assert isinstance(outcome, Incomplete), f"got {outcome}"
    assert outcome.reason == "max_output_tokens"
    assert outcome.status == "provider_incomplete"

    respx.clear()
    mock_completion(
        DEEPSEEK_ROW, completion_body(model="deepseek-reasoner", finish_reason="content_filter")
    )
    outcome = await engine.generate(
        DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW)
    )
    assert isinstance(outcome, Incomplete), f"got {outcome}"
    assert outcome.reason == "content_filter_partial"


@respx.mock
async def test_unknown_finish_reason_raises_protocol_defect(engine: OpenAIChatEngine) -> None:
    mock_completion(DEEPSEEK_ROW, completion_body(model="deepseek-reasoner", finish_reason="weird"))
    with pytest.raises(ProtocolDefect, match="finish_reason"):
        await engine.generate(DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW))


@respx.mock
async def test_json_mode_text_that_is_not_json_fails_invalid_structured_output(
    engine: OpenAIChatEngine,
) -> None:
    mock_completion(
        MOONSHOT_ROW,
        completion_body(model="kimi-k3", message={"role": "assistant", "content": "not json"}),
    )
    outcome = await engine.generate(
        MOONSHOT_ROW,
        intent_for(MOONSHOT_ROW, output=StrictJsonOutput("answer", ANSWER_SCHEMA)),
        credential_for(MOONSHOT_ROW),
    )
    assert isinstance(outcome, Failed), f"got {outcome}"
    assert isinstance(outcome.failure, InvalidStructuredOutput), f"got {outcome.failure}"


@respx.mock
async def test_missing_choices_and_missing_model_raise_protocol_defect(
    engine: OpenAIChatEngine,
) -> None:
    respx.post(chat_url(DEEPSEEK_ROW)).mock(
        return_value=httpx.Response(
            200, json={"id": "x", "model": "deepseek-reasoner", "choices": []}
        )
    )
    with pytest.raises(ProtocolDefect):
        await engine.generate(DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW))

    respx.clear()
    body = completion_body(model="deepseek-reasoner")
    del body["model"]
    respx.post(chat_url(DEEPSEEK_ROW)).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(ProtocolDefect):
        await engine.generate(DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW))


@respx.mock
async def test_malformed_json_envelope_raises_protocol_defect(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(DEEPSEEK_ROW)).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "application/json"}, content=b"not json {"
        )
    )
    with pytest.raises(ProtocolDefect):
        await engine.generate(DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW))


# ---------------------------------------------------------------------------
# Continuations.


@respx.mock
async def test_deepseek_reasoning_content_preserved_and_stripped_on_resend(
    engine: OpenAIChatEngine,
) -> None:
    mock_completion(
        DEEPSEEK_ROW,
        completion_body(
            model="deepseek-reasoner",
            message={"role": "assistant", "content": "hello", "reasoning_content": "let me think"},
        ),
    )
    first = await engine.generate(
        DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW)
    )
    assert isinstance(first, Succeeded)
    continuation = first.response.continuation
    assert isinstance(continuation, Present), "reasoning_content must produce an artifact"
    artifact = continuation.value
    assert artifact.codec_id == DEEPSEEK_ROW.continuation_codec
    assert artifact.target == ProviderTarget(provider="deepseek", model="deepseek-reasoner")
    assert artifact.opaque_payload.get("reasoning_content") == "let me think", (
        "the artifact preserves reasoning_content"
    )

    respx.clear()
    route = mock_completion(DEEPSEEK_ROW, completion_body(model="deepseek-reasoner"))
    replay: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("hi"),)),
        AssistantMessage(text="hello", tool_calls=(), continuation=Present(artifact)),
        UserMessage((PromptBlock("and then?"),)),
    )
    await engine.generate(
        DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW, messages=replay), credential_for(DEEPSEEK_ROW)
    )
    body = last_request_json(route)
    messages = body["messages"]
    assert isinstance(messages, list)
    assert messages[1] == {"role": "assistant", "content": "hello"}, (
        f"reasoning_content must NEVER be resent to deepseek; messages: {messages}"
    )


@respx.mock
async def test_xai_reasoning_content_stripped_on_resend(engine: OpenAIChatEngine) -> None:
    mock_completion(
        XAI_ROW,
        completion_body(
            model="grok-4",
            message={"role": "assistant", "content": "hey", "reasoning_content": "hmm"},
        ),
    )
    first = await engine.generate(XAI_ROW, intent_for(XAI_ROW), credential_for(XAI_ROW))
    assert isinstance(first, Succeeded)
    continuation = first.response.continuation
    assert isinstance(continuation, Present)

    respx.clear()
    route = mock_completion(XAI_ROW, completion_body(model="grok-4"))
    replay: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("hi"),)),
        AssistantMessage(text="hey", tool_calls=(), continuation=continuation),
        UserMessage((PromptBlock("more"),)),
    )
    await engine.generate(XAI_ROW, intent_for(XAI_ROW, messages=replay), credential_for(XAI_ROW))
    messages = last_request_json(route)["messages"]
    assert isinstance(messages, list)
    assert messages[1] == {"role": "assistant", "content": "hey"}, (
        f"reasoning_content must not be resent to xai; messages: {messages}"
    )


@respx.mock
async def test_moonshot_continuation_replays_complete_native_message_verbatim(
    engine: OpenAIChatEngine,
) -> None:
    native_message = {
        "role": "assistant",
        "content": "done",
        "reasoning_content": "preserved thinking",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "x"}'},
            }
        ],
    }
    mock_completion(
        MOONSHOT_ROW,
        completion_body(model="kimi-k3", finish_reason="tool_calls", message=native_message),
    )
    first = await engine.generate(
        MOONSHOT_ROW, intent_for(MOONSHOT_ROW, tools=(SEARCH_TOOL,)), credential_for(MOONSHOT_ROW)
    )
    assert isinstance(first, Succeeded)
    continuation = first.response.continuation
    assert isinstance(continuation, Present), "reasoning + tool calls must produce an artifact"

    respx.clear()
    route = mock_completion(MOONSHOT_ROW, completion_body(model="kimi-k3"))
    replay: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("go"),)),
        AssistantMessage(text="done", tool_calls=(), continuation=continuation),
        ToolResultMessage(call_id="call-1", output="found", is_error=False),
    )
    await engine.generate(
        MOONSHOT_ROW,
        intent_for(MOONSHOT_ROW, messages=replay, tools=(SEARCH_TOOL,)),
        credential_for(MOONSHOT_ROW),
    )
    messages = last_request_json(route)["messages"]
    assert isinstance(messages, list)
    assert messages[1] == native_message, (
        "moonshot replays the COMPLETE native assistant message verbatim, including "
        f"reasoning_content (Preserved Thinking); got: {messages[1]}"
    )


@respx.mock
async def test_openrouter_reasoning_details_round_trip_verbatim(engine: OpenAIChatEngine) -> None:
    details = [
        {"type": "reasoning.encrypted", "data": "opaque-1", "index": 0},
        {"type": "reasoning.text", "text": "step two", "index": 1},
    ]
    mock_completion(
        OPENROUTER_ROW,
        completion_body(
            model="moonshotai/kimi-k3",
            message={"role": "assistant", "content": "ok", "reasoning_details": details},
        ),
    )
    first = await engine.generate(
        OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
    )
    assert isinstance(first, Succeeded)
    continuation = first.response.continuation
    assert isinstance(continuation, Present)
    assert continuation.value.opaque_payload == {"reasoning_details": details}, (
        f"ordered reasoning_details must be preserved verbatim; got {continuation.value.opaque_payload}"
    )

    respx.clear()
    route = mock_completion(OPENROUTER_ROW, completion_body(model="moonshotai/kimi-k3"))
    replay: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("hi"),)),
        AssistantMessage(text="ok", tool_calls=(), continuation=continuation),
        UserMessage((PromptBlock("next"),)),
    )
    await engine.generate(
        OPENROUTER_ROW, intent_for(OPENROUTER_ROW, messages=replay), credential_for(OPENROUTER_ROW)
    )
    messages = last_request_json(route)["messages"]
    assert isinstance(messages, list)
    assert messages[1] == {"role": "assistant", "content": "ok", "reasoning_details": details}, (
        f"reasoning_details replay verbatim on the assistant message; got {messages[1]}"
    )


async def test_continuation_bound_to_other_codec_or_target_raises_invalid_request(
    engine: OpenAIChatEngine,
) -> None:
    wrong_codec = ContinuationArtifact(
        target=ProviderTarget(provider="deepseek", model="deepseek-reasoner"),
        codec_id="moonshot.v1",
        opaque_payload={"role": "assistant", "content": "x"},
    )
    messages: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("hi"),)),
        AssistantMessage(text="x", tool_calls=(), continuation=Present(wrong_codec)),
    )
    with pytest.raises(InvalidRequest):
        await engine.generate(
            DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW, messages=messages), credential_for(DEEPSEEK_ROW)
        )

    wrong_target = ContinuationArtifact(
        target=ProviderTarget(provider="moonshot", model="kimi-k3"),
        codec_id="deepseek.v1",
        opaque_payload={"role": "assistant", "content": "x"},
    )
    messages = (
        UserMessage((PromptBlock("hi"),)),
        AssistantMessage(text="x", tool_calls=(), continuation=Present(wrong_target)),
    )
    with pytest.raises(InvalidRequest):
        await engine.generate(
            DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW, messages=messages), credential_for(DEEPSEEK_ROW)
        )


# ---------------------------------------------------------------------------
# Stream decode from raw SSE bytes.


@respx.mock
async def test_moonshot_stream_decodes_text_usage_and_continuation(
    engine: OpenAIChatEngine,
) -> None:
    route = mock_stream(
        MOONSHOT_ROW,
        sse_bytes(
            {
                "id": "s-1",
                "model": "kimi-k3",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": "think "}}
                ],
            },
            {"choices": [{"index": 0, "delta": {"reasoning_content": "hard"}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            {"choices": [{"index": 0, "delta": {"content": " world"}}]},
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 6,
                            "total_tokens": 17,
                            "cached_tokens": 4,
                        },
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 6,
                    "total_tokens": 17,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            },
            "[DONE]",
        ),
    )
    events = await collect(
        engine.stream(MOONSHOT_ROW, intent_for(MOONSHOT_ROW), credential_for(MOONSHOT_ROW))
    )
    body = last_request_json(route)
    assert body["stream"] is True, f"body: {body}"
    assert body["stream_options"] == {"include_usage": True}, (
        f"moonshot sends include_usage belt-and-braces; body: {body}"
    )

    kinds = [type(event).__name__ for event in events]
    assert kinds == [
        "StreamStart",
        "TextDelta",
        "TextDelta",
        "UsageEvent",
        "UsageEvent",
        "ContinuationDelta",
        "TerminalEvent",
    ], f"events: {kinds}"
    assert events[1] == TextDelta(text="Hello")
    assert events[2] == TextDelta(text=" world")

    continuation_event = events[-2]
    assert isinstance(continuation_event, ContinuationDelta)
    artifact = continuation_event.artifact
    assert artifact.codec_id == "moonshot.v1"
    assert artifact.opaque_payload == {
        "role": "assistant",
        "content": "Hello world",
        "reasoning_content": "think hard",
    }, f"reconstructed native message; got {artifact.opaque_payload}"

    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded), f"got {outcome}"
    assert outcome.response.content == TextContent(text="Hello world", tool_calls=())
    assert outcome.response.continuation == Present(artifact)
    assert outcome.meta.provider_request_id == Present("s-1")
    assert isinstance(outcome.meta.usage, Present), "terminal meta must fold all usage frames"
    usage = outcome.meta.usage.value
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (11, 6, 17)
    assert usage.cache_read_input_tokens == Present(4), f"usage: {usage}"
    assert usage.reasoning_tokens == Present(3), (
        f"the fold must merge the trailing usage frame's details; usage: {usage}"
    )


@respx.mock
async def test_stream_tool_calls_accumulate_by_index_and_strict_parse(
    engine: OpenAIChatEngine,
) -> None:
    mock_stream(
        XAI_ROW,
        sse_bytes(
            {
                "id": "s-2",
                "model": "grok-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-7",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": ""},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"query"'}}]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": ': "dogs"}'}}]
                        },
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            "[DONE]",
        ),
    )
    events = await collect(
        engine.stream(XAI_ROW, intent_for(XAI_ROW, tools=(SEARCH_TOOL,)), credential_for(XAI_ROW))
    )
    kinds = [type(event).__name__ for event in events]
    assert kinds == [
        "StreamStart",
        "ToolCallStart",
        "ToolCallDelta",
        "ToolCallDelta",
        "ToolCallDone",
        "ContinuationDelta",
        "TerminalEvent",
    ], f"events: {kinds}"
    assert events[1] == ToolCallStart(call_id="call-7", name="search")
    assert events[2] == ToolCallDelta(call_id="call-7", arguments_delta='{"query"')
    done = events[4]
    assert isinstance(done, ToolCallDone)
    assert done.tool_call == ToolCall(id="call-7", name="search", arguments={"query": "dogs"})
    continuation_event = events[5]
    assert isinstance(continuation_event, ContinuationDelta)
    native_calls = continuation_event.artifact.opaque_payload.get("tool_calls")
    assert native_calls == [
        {
            "id": "call-7",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "dogs"}'},
        }
    ], f"the artifact keeps the RAW accumulated argument string; got {native_calls}"


def tool_call_frame(index: int, **function: object) -> dict[str, object]:
    call: dict[str, object] = {"index": index, "function": function}
    return {"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}


@respx.mock
async def test_stream_interleaved_tool_calls_stay_separated_by_index(
    engine: OpenAIChatEngine,
) -> None:
    mock_stream(
        XAI_ROW,
        sse_bytes(
            {
                "id": "s-6",
                "model": "grok-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {"index": 0, "id": "call-a", "function": {"name": "search"}},
                                {"index": 1, "id": "call-b", "function": {"name": "search"}},
                            ],
                        },
                    }
                ],
            },
            tool_call_frame(0, arguments='{"query": "a'),
            tool_call_frame(1, arguments='{"query": "p'),
            tool_call_frame(0, arguments='pples"}'),
            tool_call_frame(1, arguments='ears"}'),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            "[DONE]",
        ),
    )
    events = await collect(
        engine.stream(XAI_ROW, intent_for(XAI_ROW, tools=(SEARCH_TOOL,)), credential_for(XAI_ROW))
    )
    assert [type(event).__name__ for event in events] == [
        "StreamStart",
        "ToolCallStart",
        "ToolCallStart",
        "ToolCallDelta",
        "ToolCallDelta",
        "ToolCallDelta",
        "ToolCallDelta",
        "ToolCallDone",
        "ToolCallDone",
        "ContinuationDelta",
        "TerminalEvent",
    ], f"events: {[type(e).__name__ for e in events]}"
    assert [event for event in events if isinstance(event, ToolCallDelta)] == [
        ToolCallDelta(call_id="call-a", arguments_delta='{"query": "a'),
        ToolCallDelta(call_id="call-b", arguments_delta='{"query": "p'),
        ToolCallDelta(call_id="call-a", arguments_delta='pples"}'),
        ToolCallDelta(call_id="call-b", arguments_delta='ears"}'),
    ], "fragments must land in the slot named by their index, never concatenated together"
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded), f"got {outcome}"
    assert outcome.response.content == TextContent(
        text="",
        tool_calls=(
            ToolCall(id="call-a", name="search", arguments={"query": "apples"}),
            ToolCall(id="call-b", name="search", arguments={"query": "pears"}),
        ),
    ), f"got {outcome.response.content}"


@respx.mock
async def test_repeated_finish_reason_frame_does_not_erase_accumulated_tool_calls(
    engine: OpenAIChatEngine,
) -> None:
    mock_stream(
        XAI_ROW,
        sse_bytes(
            {
                "id": "s-7",
                "model": "grok-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-c",
                                    "function": {"name": "search", "arguments": '{"query": "c"}'},
                                }
                            ]
                        },
                    }
                ],
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            },
            "[DONE]",
        ),
    )
    events = await collect(
        engine.stream(XAI_ROW, intent_for(XAI_ROW, tools=(SEARCH_TOOL,)), credential_for(XAI_ROW))
    )
    assert [type(event).__name__ for event in events] == [
        "StreamStart",
        "ToolCallStart",
        "ToolCallDelta",
        "ToolCallDone",
        "UsageEvent",
        "ContinuationDelta",
        "TerminalEvent",
    ], f"exactly one ToolCallDone per call; events: {[type(e).__name__ for e in events]}"
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded), f"got {outcome}"
    assert outcome.response.content == TextContent(
        text="", tool_calls=(ToolCall(id="call-c", name="search", arguments={"query": "c"}),)
    ), (
        "the terminal must agree with the ToolCallDone already delivered; got "
        f"{outcome.response.content}"
    )
    assert isinstance(outcome.response.continuation, Present)


@respx.mock
async def test_streamed_tool_arguments_that_never_parse_fail_the_terminal(
    engine: OpenAIChatEngine,
) -> None:
    mock_stream(
        XAI_ROW,
        sse_bytes(
            {
                "id": "s-8",
                "model": "grok-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-d",
                                    "function": {"name": "search", "arguments": '{"query": '},
                                }
                            ]
                        },
                    }
                ],
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            "[DONE]",
        ),
    )
    events = await collect(
        engine.stream(XAI_ROW, intent_for(XAI_ROW, tools=(SEARCH_TOOL,)), credential_for(XAI_ROW))
    )
    assert [type(event).__name__ for event in events] == [
        "StreamStart",
        "ToolCallStart",
        "ToolCallDelta",
        "TerminalEvent",
    ], f"no Done and no continuation for a call that never parsed; events: {events}"
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed), f"strict parse, no repair — got {outcome}"
    assert isinstance(outcome.failure, InvalidToolArguments), f"got {outcome.failure}"
    assert outcome.meta.model == "grok-4"
    assert outcome.meta.billability == PossiblyBillable()


@respx.mock
async def test_streamed_tool_call_without_id_or_name_is_a_protocol_defect(
    engine: OpenAIChatEngine,
) -> None:
    mock_stream(
        XAI_ROW,
        sse_bytes(
            {
                "id": "s-9",
                "model": "grok-4",
                "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            },
            tool_call_frame(0, arguments='{"query": "x"}'),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            "[DONE]",
        ),
    )
    with pytest.raises(ProtocolDefect) as excinfo:
        await collect(
            engine.stream(
                XAI_ROW, intent_for(XAI_ROW, tools=(SEARCH_TOOL,)), credential_for(XAI_ROW)
            )
        )
    assert excinfo.value.code == "malformed_tool_call", f"got {excinfo.value.code}"


@respx.mock
async def test_openrouter_finish_reason_error_is_transient(engine: OpenAIChatEngine) -> None:
    mock_stream(
        OPENROUTER_ROW,
        sse_bytes(
            {
                "id": "gen-3",
                "model": "moonshotai/kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            },
        ),
    )
    with pytest.raises(TransientAttempt) as excinfo:
        await collect(
            engine.stream(
                OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
            )
        )
    assert excinfo.value.cause == ProviderHttpUnavailable(), f"got {excinfo.value.cause}"
    assert excinfo.value.provider_request_id == Present("gen-3")


@respx.mock
async def test_openrouter_stream_collects_reasoning_details_and_upstream(
    engine: OpenAIChatEngine,
) -> None:
    route = mock_stream(
        OPENROUTER_ROW,
        sse_bytes(
            {
                "id": "gen-1",
                "model": "moonshotai/kimi-k3",
                "provider": "Moonshot",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_details": [{"type": "reasoning.encrypted", "data": "a"}]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_details": [{"type": "reasoning.text", "text": "b"}]},
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            },
            "[DONE]",
        ),
    )
    events = await collect(
        engine.stream(OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW))
    )
    body = last_request_json(route)
    assert body["provider"] == EXPECTED_PINS, f"pins must ride stream calls too; body: {body}"
    assert "stream_options" not in body, (
        f"openrouter must not send stream_options (conflicts with require_parameters); body: {body}"
    )

    continuation_events = [event for event in events if isinstance(event, ContinuationDelta)]
    assert len(continuation_events) == 1, f"events: {[type(e).__name__ for e in events]}"
    assert continuation_events[0].artifact.opaque_payload == {
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "a"},
            {"type": "reasoning.text", "text": "b"},
        ]
    }, "reasoning_details accumulate verbatim, in order"
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert isinstance(terminal.outcome, Succeeded)
    assert terminal.outcome.meta.upstream_provider == Present("Moonshot")
    assert terminal.outcome.meta.provider_request_id == Present("gen-1")


@respx.mock
async def test_stream_start_only_after_provider_acceptance(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(MOONSHOT_ROW)).mock(
        return_value=httpx.Response(429, json={"error": {"message": "slow down"}})
    )
    stream = engine.stream(MOONSHOT_ROW, intent_for(MOONSHOT_ROW), credential_for(MOONSHOT_ROW))
    with pytest.raises(TransientAttempt) as excinfo:
        await anext(stream)
    assert isinstance(excinfo.value.cause, ProviderRateLimit), f"got {excinfo.value.cause}"


@respx.mock
async def test_stream_cut_before_semantic_output_is_interrupted_not_partial(
    engine: OpenAIChatEngine,
) -> None:
    # The stream opens (role-only delta = not semantic) then ends with no
    # finish_reason and no [DONE].
    mock_stream(
        MOONSHOT_ROW,
        sse_bytes(
            {
                "id": "s-3",
                "model": "kimi-k3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            },
        ),
    )
    stream = engine.stream(MOONSHOT_ROW, intent_for(MOONSHOT_ROW), credential_for(MOONSHOT_ROW))
    events: list[CodecStreamEvent] = [await anext(stream)]
    assert isinstance(events[0], StreamStart)
    with pytest.raises(TransientAttempt) as excinfo:
        while True:
            events.append(await anext(stream))
    assert excinfo.value.cause == ProviderStreamInterrupted(partial_output=False), (
        f"no semantic output was yielded; got {excinfo.value.cause}; events: {events}"
    )


@respx.mock
async def test_stream_transport_cut_reports_the_transport_cause(
    engine: OpenAIChatEngine,
) -> None:
    first = sse_bytes(
        {"id": "s-4", "model": "kimi-k3", "choices": [{"index": 0, "delta": {"content": "Hel"}}]}
    )
    respx.post(chat_url(MOONSHOT_ROW)).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=CutByteStream((first,)),
        )
    )
    stream = engine.stream(MOONSHOT_ROW, intent_for(MOONSHOT_ROW), credential_for(MOONSHOT_ROW))
    seen: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as excinfo:
        async for event in stream:
            seen.append(event)
    assert any(isinstance(event, TextDelta) for event in seen), f"events: {seen}"
    assert excinfo.value.cause == TransportUnavailable(), (
        f"the engine reports what happened; whether a post-semantic transient is retryable "
        f"and which leaf stands for it is the runtime's call; got {excinfo.value.cause}"
    )
    assert excinfo.value.billability == PossiblyBillable()


@respx.mock
async def test_stream_cut_after_semantic_output_is_interrupted_with_partial_output(
    engine: OpenAIChatEngine,
) -> None:
    # Text was delivered, then the stream ended with no finish_reason: the
    # engine is the only party that can flag the partial output.
    mock_stream(
        MOONSHOT_ROW,
        sse_bytes(
            {
                "id": "s-5",
                "model": "kimi-k3",
                "choices": [{"index": 0, "delta": {"content": "Hel"}}],
            },
        ),
    )
    stream = engine.stream(MOONSHOT_ROW, intent_for(MOONSHOT_ROW), credential_for(MOONSHOT_ROW))
    seen: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as excinfo:
        async for event in stream:
            seen.append(event)
    assert any(isinstance(event, TextDelta) for event in seen), f"events: {seen}"
    assert excinfo.value.cause == ProviderStreamInterrupted(partial_output=True), (
        f"got {excinfo.value.cause}"
    )


@respx.mock
async def test_openrouter_inband_stream_error_pre_semantic_classifies_rate_limit(
    engine: OpenAIChatEngine,
) -> None:
    mock_stream(
        OPENROUTER_ROW,
        sse_bytes({"error": {"code": 429, "message": "rate limited"}}),
    )
    stream = engine.stream(
        OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
    )
    seen: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as excinfo:
        async for event in stream:
            seen.append(event)
    assert seen == [StreamStart()], f"only the envelope may precede the failure; got {seen}"
    assert excinfo.value.cause == ProviderRateLimit(retry_after=Absent()), (
        f"429-shaped in-band errors classify as rate limit; got {excinfo.value.cause}"
    )


@respx.mock
async def test_openrouter_inband_stream_error_post_semantic_reports_the_upstream_cause(
    engine: OpenAIChatEngine,
) -> None:
    mock_stream(
        OPENROUTER_ROW,
        sse_bytes(
            {
                "id": "gen-2",
                "model": "moonshotai/kimi-k3",
                "choices": [{"index": 0, "delta": {"content": "par"}}],
            },
            {"error": {"code": 502, "message": "upstream died"}},
        ),
    )
    stream = engine.stream(
        OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
    )
    seen: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as excinfo:
        async for event in stream:
            seen.append(event)
    assert any(isinstance(event, TextDelta) for event in seen), f"events: {seen}"
    assert excinfo.value.cause == ProviderHttpUnavailable(), f"got {excinfo.value.cause}"


@respx.mock
async def test_inband_stream_error_without_a_transient_code_is_a_protocol_defect(
    engine: OpenAIChatEngine,
) -> None:
    mock_stream(
        OPENROUTER_ROW,
        sse_bytes({"error": {"code": 400, "message": "no endpoints found sk-live-abcdefghij"}}),
    )
    stream = engine.stream(
        OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
    )
    with pytest.raises(ProtocolDefect) as excinfo:
        async for _ in stream:
            pass
    assert excinfo.value.code == "inband_provider_error", f"got {excinfo.value.code}"
    assert "sk-live-abcdefghij" not in excinfo.value.message, (
        f"the provider snippet must be sanitized; got {excinfo.value.message!r}"
    )


# ---------------------------------------------------------------------------
# Fault injection (generate).


@respx.mock
async def test_429_with_retry_after_raises_transient_rate_limit(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(MOONSHOT_ROW)).mock(
        return_value=httpx.Response(
            429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}
        )
    )
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(MOONSHOT_ROW, intent_for(MOONSHOT_ROW), credential_for(MOONSHOT_ROW))
    attempt = excinfo.value
    assert attempt.cause == ProviderRateLimit(retry_after=Present(7.0)), f"got {attempt.cause}"
    assert attempt.status_code == Present(429)
    assert attempt.billability == PossiblyBillable()


@respx.mock
async def test_429_without_retry_after_has_absent_delay(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(DEEPSEEK_ROW)).mock(
        return_value=httpx.Response(429, json={"error": {"message": "slow down"}})
    )
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW))
    assert excinfo.value.cause == ProviderRateLimit(retry_after=Absent()), (
        f"got {excinfo.value.cause}"
    )


@respx.mock
async def test_timeouts_raise_transient_provider_timeout(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(XAI_ROW)).mock(side_effect=httpx.ConnectTimeout("boom"))
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(XAI_ROW, intent_for(XAI_ROW), credential_for(XAI_ROW))
    assert excinfo.value.cause == ProviderTimeout(), f"got {excinfo.value.cause}"
    assert excinfo.value.status_code == Absent()

    respx.clear()
    respx.post(chat_url(XAI_ROW)).mock(side_effect=httpx.ReadTimeout("boom"))
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(XAI_ROW, intent_for(XAI_ROW), credential_for(XAI_ROW))
    assert excinfo.value.cause == ProviderTimeout(), f"got {excinfo.value.cause}"


@respx.mock
async def test_5xx_raises_transient_provider_unavailable(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(OPENROUTER_ROW)).mock(
        return_value=httpx.Response(503, json={"error": {"message": "down"}})
    )
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(
            OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
        )
    assert excinfo.value.cause == ProviderHttpUnavailable(), f"got {excinfo.value.cause}"
    assert excinfo.value.status_code == Present(503)


@respx.mock
async def test_connect_error_is_transport_unavailable_not_dispatched(
    engine: OpenAIChatEngine,
) -> None:
    respx.post(chat_url(DEEPSEEK_ROW)).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW))
    assert excinfo.value.cause == TransportUnavailable(), f"got {excinfo.value.cause}"
    assert excinfo.value.billability == NotDispatched(), (
        "a pure pre-connect failure means no bytes reached the provider"
    )


@respx.mock
async def test_mid_request_transport_error_is_possibly_billable(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(DEEPSEEK_ROW)).mock(side_effect=httpx.ReadError("broken pipe"))
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW))
    assert excinfo.value.cause == TransportUnavailable(), f"got {excinfo.value.cause}"
    assert excinfo.value.billability == PossiblyBillable(), (
        "the connection was opened, so the provider may have seen the request"
    )


@respx.mock
async def test_context_overflow_400_returns_failed_value_with_meta(
    engine: OpenAIChatEngine,
) -> None:
    respx.post(chat_url(MOONSHOT_ROW)).mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": "context_length_exceeded",
                    "message": "your prompt exceeds the context length",
                }
            },
        )
    )
    outcome = await engine.generate(
        MOONSHOT_ROW, intent_for(MOONSHOT_ROW), credential_for(MOONSHOT_ROW)
    )
    assert isinstance(outcome, Failed), f"got {outcome}"
    assert outcome.failure == ProviderContextTooLarge()
    meta = outcome.meta
    assert meta.provider == "moonshot"
    assert meta.model == "kimi-k3", "no envelope decoded — the row's model id stands in"
    assert meta.usage == Absent()
    assert meta.billability == PossiblyBillable()
    assert meta.registry_revision == REGISTRY_REVISION
    assert len(meta.attempt_trace) == 1
    assert meta.attempt_trace[0].status_code == Present(400)


@respx.mock
async def test_unclassified_400_raises_runtime_defect(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(MOONSHOT_ROW)).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad param"}})
    )
    with pytest.raises(RuntimeDefect) as excinfo:
        await engine.generate(MOONSHOT_ROW, intent_for(MOONSHOT_ROW), credential_for(MOONSHOT_ROW))
    assert excinfo.value.code == "unclassified_provider_error", f"got {excinfo.value.code}"


@respx.mock
async def test_401_and_403_raise_credential_rejected(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(DEEPSEEK_ROW)).mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    with pytest.raises(CredentialRejected):
        await engine.generate(DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW))

    respx.clear()
    respx.post(chat_url(XAI_ROW)).mock(
        return_value=httpx.Response(403, json={"error": {"message": "forbidden"}})
    )
    with pytest.raises(CredentialRejected):
        await engine.generate(XAI_ROW, intent_for(XAI_ROW), credential_for(XAI_ROW))


@respx.mock
async def test_openrouter_403_moderation_flag_is_not_credential_rejection(
    engine: OpenAIChatEngine,
) -> None:
    respx.post(chat_url(OPENROUTER_ROW)).mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "message": "flagged",
                    "metadata": {"reasons": ["violence"], "flagged_input": "…"},
                }
            },
        )
    )
    with pytest.raises(RuntimeDefect) as excinfo:
        await engine.generate(
            OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
        )
    assert excinfo.value.code == "input_moderation_flagged", f"got {excinfo.value.code}"
    assert not isinstance(excinfo.value, CredentialRejected)


@respx.mock
async def test_inband_error_on_a_200_body_is_transient_not_a_missing_model_defect(
    engine: OpenAIChatEngine,
) -> None:
    # OpenRouter answers 200 with an error object when the upstream fails
    # after acceptance — the same shape the stream arm already models.
    respx.post(chat_url(OPENROUTER_ROW)).mock(
        return_value=httpx.Response(
            200, json={"error": {"code": 502, "message": "upstream fell over"}}
        )
    )
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(
            OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
        )
    assert excinfo.value.cause == ProviderHttpUnavailable(), f"got {excinfo.value.cause}"
    assert excinfo.value.status_code == Present(200)
    assert excinfo.value.billability == PossiblyBillable()


@respx.mock
async def test_inband_rate_limit_on_a_200_body_classifies_as_rate_limit(
    engine: OpenAIChatEngine,
) -> None:
    respx.post(chat_url(OPENROUTER_ROW)).mock(
        return_value=httpx.Response(200, json={"error": {"code": "429", "message": "slow down"}})
    )
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(
            OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
        )
    assert excinfo.value.cause == ProviderRateLimit(retry_after=Absent()), (
        f"got {excinfo.value.cause}"
    )


@respx.mock
async def test_inband_error_without_a_transient_code_is_a_protocol_defect(
    engine: OpenAIChatEngine,
) -> None:
    respx.post(chat_url(OPENROUTER_ROW)).mock(
        return_value=httpx.Response(
            200, json={"error": {"code": 403, "message": "key sk-live-abcdefghij is not allowed"}}
        )
    )
    with pytest.raises(ProtocolDefect) as excinfo:
        await engine.generate(
            OPENROUTER_ROW, intent_for(OPENROUTER_ROW), credential_for(OPENROUTER_ROW)
        )
    assert excinfo.value.code == "inband_provider_error", f"got {excinfo.value.code}"
    assert "sk-live-abcdefghij" not in excinfo.value.message, (
        f"the provider snippet must be sanitized; got {excinfo.value.message!r}"
    )


@respx.mock
async def test_402_raises_quota_exhausted_defect(engine: OpenAIChatEngine) -> None:
    respx.post(chat_url(DEEPSEEK_ROW)).mock(
        return_value=httpx.Response(402, json={"error": {"message": "insufficient balance"}})
    )
    with pytest.raises(RuntimeDefect) as excinfo:
        await engine.generate(DEEPSEEK_ROW, intent_for(DEEPSEEK_ROW), credential_for(DEEPSEEK_ROW))
    assert excinfo.value.code == "quota_exhausted", f"got {excinfo.value.code}"
