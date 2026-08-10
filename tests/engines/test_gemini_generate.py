"""gemini_generate engine conformance + fault injection (respx at the HTTP boundary).

Native GenerateContent over the google-genai SDK. Fixture rows are constructed
locally — tests never depend on registry ROWS content. Covered per the freeze:
exact request body/header shapes, response decode, stream decode from raw SSE
bytes, continuation round-trip (ordered parts + thoughtSignatures replayed
verbatim), reasoning-level mapping from row.reasoning (thinkingLevel vs
thinkingBudget, chosen by the row), provider_options passthrough vs collision,
and the full fault-injection table.

Wire facts verified against the installed SDK (google-genai 1.75): the mldev
converter emits camelCase envelope keys but serializes ThinkingConfig and
FunctionDeclaration values with snake_case field names (proto3 JSON parsing
accepts both); bytes fields ride as base64 strings and round-trip through
part dict payloads unchanged.
"""

import json
from base64 import b64encode
from collections.abc import Mapping

import httpx
import pytest
import respx

from provider_runtime.engines import TransientAttempt
from provider_runtime.engines.gemini_generate import GeminiGenerateEngine
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
    ToolCallDone,
    ToolCallStart,
    ToolResultMessage,
    TransportUnavailable,
    UsageEvent,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Fixture rows — local to this file by design (registry ROWS content is
# another agent's concern).

# Gemini 3+ speaks thinkingLevel; the ROW carries the exact wire key/value.
LEVEL_REASONING: Mapping[ReasoningLevel, object] = {
    "low": {"thinkingLevel": "LOW"},
    "high": {"thinkingLevel": "HIGH"},
}
# 2.5-era speaks thinkingBudget; "none" maps to empty params (nothing sent).
BUDGET_REASONING: Mapping[ReasoningLevel, object] = {
    "none": {},
    "high": {"thinkingBudget": 24576},
}

LEVEL_ROW = ModelRow(
    ref="gemini:pro",
    provider="gemini",
    model_id="gemini-3-pro",
    engine="gemini_generate",
    base_url=Absent(),
    context_window=1_048_576,
    max_output_tokens=65_536,
    modalities=frozenset({"text", "image"}),
    tools=True,
    streaming=True,
    structured="native",
    reasoning=Present(LEVEL_REASONING),
    continuation_codec="gemini.v1",
    correlation="none",
    routing=Absent(),
)

BUDGET_ROW = ModelRow(
    ref="gemini:flash",
    provider="gemini",
    model_id="gemini-2.5-flash",
    engine="gemini_generate",
    base_url=Absent(),
    context_window=1_048_576,
    max_output_tokens=65_536,
    modalities=frozenset({"text", "image"}),
    tools=True,
    streaming=True,
    structured="native",
    reasoning=Present(BUDGET_REASONING),
    continuation_codec="gemini.v1",
    correlation="none",
    routing=Absent(),
)

# No reasoning knob at all; json_mode structured output.
KNOBLESS_ROW = ModelRow(
    ref="gemini:lite",
    provider="gemini",
    model_id="gemini-2.0-flash-lite",
    engine="gemini_generate",
    base_url=Absent(),
    context_window=1_048_576,
    max_output_tokens=8_192,
    modalities=frozenset({"text"}),
    tools=False,
    streaming=True,
    structured="json_mode",
    reasoning=Absent(),
    continuation_codec="gemini.v1",
    correlation="none",
    routing=Absent(),
)

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

CREDENTIAL = ProviderCredential(provider="gemini", key="test-key")


def intent_for(
    row: ModelRow,
    *,
    messages: tuple[PromptMessage, ...] | None = None,
    reasoning: ReasoningLevel = "high",
    tools: tuple[CanonicalTool, ...] = (),
    tool_choice: str = "auto",
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
        tool_choice="none" if tool_choice == "none" else "auto",
        output=output or TextOutput(),
        provider_options=provider_options or {},
    )


def generate_url(row: ModelRow) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{row.model_id}:generateContent"


def stream_url(row: ModelRow) -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{row.model_id}:streamGenerateContent"
    )


def response_body(
    *,
    parts: list[dict[str, object]] | None = None,
    finish_reason: str | None = "STOP",
    usage: dict[str, object] | None = None,
    model_version: str | None = "gemini-3-pro-echo",
    **top_level: object,
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "content": {"role": "model", "parts": parts or [{"text": "hello"}]}
    }
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    body: dict[str, object] = {"candidates": [candidate], "responseId": "resp-1"}
    if model_version is not None:
        body["modelVersion"] = model_version
    if usage is not None:
        body["usageMetadata"] = usage
    body.update(top_level)
    return body


def mock_generate(row: ModelRow, body: dict[str, object], status: int = 200) -> respx.Route:
    return respx.post(generate_url(row)).mock(return_value=httpx.Response(status, json=body))


def last_request_json(route: respx.Route) -> dict[str, object]:
    assert route.called, "expected the engine to dispatch an HTTP request"
    parsed = json.loads(route.calls.last.request.content)
    assert isinstance(parsed, dict)
    return parsed


def sse_bytes(*frames: dict[str, object]) -> bytes:
    return b"".join(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames)


def mock_stream(row: ModelRow, payload: bytes) -> respx.Route:
    return respx.post(stream_url(row)).mock(
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
def engine() -> GeminiGenerateEngine:
    return GeminiGenerateEngine()


# ---------------------------------------------------------------------------
# Request conformance.


@respx.mock
async def test_request_encodes_contents_config_and_credential_header(
    engine: GeminiGenerateEngine,
) -> None:
    route = mock_generate(LEVEL_ROW, response_body())
    outcome = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW, reasoning="high"), CREDENTIAL)
    body = last_request_json(route)
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}], f"body: {body}"
    assert body["systemInstruction"] == {"parts": [{"text": "be brief"}]}, (
        f"SystemMessage blocks must become systemInstruction parts; body: {body}"
    )
    config = body["generationConfig"]
    assert isinstance(config, dict)
    assert config["maxOutputTokens"] == 512, f"config: {config}"
    # The SDK serializes ThinkingConfig with snake_case field names (proto3
    # JSON accepts both casings); the row chose the thinkingLevel key.
    assert config["thinkingConfig"] == {"thinking_level": "HIGH"}, (
        f"row.reasoning['high'] must be forwarded into thinkingConfig; config: {config}"
    )
    assert "tools" not in body and "toolConfig" not in body, f"body: {body}"
    key = route.calls.last.request.headers["x-goog-api-key"]
    assert key == "test-key", f"credential must ride the x-goog-api-key header, got {key!r}"
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.native_reasoning == Present("thinkingLevel=HIGH"), (
        f"native_reasoning must record the exact row wire value, got {outcome.meta.native_reasoning}"
    )


@respx.mock
async def test_thinking_budget_row_sends_budget_and_reports_native_reasoning(
    engine: GeminiGenerateEngine,
) -> None:
    route = mock_generate(BUDGET_ROW, response_body(model_version="gemini-2.5-flash"))
    outcome = await engine.generate(
        BUDGET_ROW, intent_for(BUDGET_ROW, reasoning="high"), CREDENTIAL
    )
    config = last_request_json(route)["generationConfig"]
    assert isinstance(config, dict)
    assert config["thinkingConfig"] == {"thinking_budget": 24576}, (
        f"2.5-era rows speak thinkingBudget — the ROW decides the key; config: {config}"
    )
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.native_reasoning == Present("thinkingBudget=24576"), (
        f"got {outcome.meta.native_reasoning}"
    )


@respx.mock
async def test_reasoning_none_with_empty_params_sends_no_thinking_config(
    engine: GeminiGenerateEngine,
) -> None:
    route = mock_generate(BUDGET_ROW, response_body(model_version="gemini-2.5-flash"))
    outcome = await engine.generate(
        BUDGET_ROW, intent_for(BUDGET_ROW, reasoning="none"), CREDENTIAL
    )
    config = last_request_json(route)["generationConfig"]
    assert isinstance(config, dict)
    assert "thinkingConfig" not in config, f"'none' maps to empty params; config: {config}"
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.native_reasoning == Absent(), (
        f"nothing was sent, so native_reasoning must be Absent, got {outcome.meta.native_reasoning}"
    )


async def test_reasoning_level_outside_row_mapping_raises_invalid_request(
    engine: GeminiGenerateEngine,
) -> None:
    with pytest.raises(InvalidRequest, match="minimal"):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW, reasoning="minimal"), CREDENTIAL)


async def test_reasoning_on_knobless_row_raises_invalid_request(
    engine: GeminiGenerateEngine,
) -> None:
    with pytest.raises(InvalidRequest, match="reasoning"):
        await engine.generate(KNOBLESS_ROW, intent_for(KNOBLESS_ROW, reasoning="high"), CREDENTIAL)


@respx.mock
async def test_knobless_row_with_reasoning_none_sends_no_thinking_config(
    engine: GeminiGenerateEngine,
) -> None:
    route = mock_generate(KNOBLESS_ROW, response_body(model_version="gemini-2.0-flash-lite"))
    outcome = await engine.generate(
        KNOBLESS_ROW, intent_for(KNOBLESS_ROW, reasoning="none"), CREDENTIAL
    )
    config = last_request_json(route)["generationConfig"]
    assert isinstance(config, dict)
    assert "thinkingConfig" not in config, f"config: {config}"
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.native_reasoning == Absent()


@respx.mock
async def test_row_base_url_overrides_the_sdk_default(engine: GeminiGenerateEngine) -> None:
    row = ModelRow(
        ref="gemini:proxied",
        provider="gemini",
        model_id="gemini-3-pro",
        engine="gemini_generate",
        base_url=Present("https://gemini-proxy.example"),
        context_window=1_048_576,
        max_output_tokens=65_536,
        modalities=frozenset({"text"}),
        tools=True,
        streaming=True,
        structured="native",
        reasoning=Present(LEVEL_REASONING),
        continuation_codec="gemini.v1",
        correlation="none",
        routing=Absent(),
    )
    route = respx.post(
        "https://gemini-proxy.example/v1beta/models/gemini-3-pro:generateContent"
    ).mock(return_value=httpx.Response(200, json=response_body()))
    outcome = await engine.generate(row, intent_for(row), CREDENTIAL)
    assert route.called, "the row's base_url must be honored"
    assert isinstance(outcome, Succeeded)


@respx.mock
async def test_tools_and_tool_results_encode_to_generate_content_wire(
    engine: GeminiGenerateEngine,
) -> None:
    route = mock_generate(LEVEL_ROW, response_body())
    messages: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("find it"),)),
        AssistantMessage(
            text="",
            tool_calls=(
                ToolCall(id="call_0", name="search", arguments={"query": "cats"}),
                ToolCall(id="call_1", name="search", arguments={"query": "dogs"}),
            ),
            continuation=Absent(),
        ),
        ToolResultMessage(call_id="call_0", output="found cats", is_error=False),
        ToolResultMessage(call_id="call_1", output="kennel closed", is_error=True),
    )
    await engine.generate(
        LEVEL_ROW, intent_for(LEVEL_ROW, messages=messages, tools=(SEARCH_TOOL,)), CREDENTIAL
    )
    body = last_request_json(route)
    # The SDK serializes FunctionDeclaration with its snake_case json-schema
    # field; the schema itself reaches the wire unstripped.
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "search",
                    "description": "Search the corpus.",
                    "parameters_json_schema": SEARCH_TOOL.parameters,
                }
            ]
        }
    ], f"body: {body}"
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}, f"body: {body}"
    contents = body["contents"]
    assert isinstance(contents, list)
    assert contents[1] == {
        "role": "model",
        "parts": [
            {"functionCall": {"name": "search", "args": {"query": "cats"}}},
            {"functionCall": {"name": "search", "args": {"query": "dogs"}}},
        ],
    }, f"contents: {contents}"
    assert contents[2] == {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": "search", "response": {"output": "found cats"}}},
            {"functionResponse": {"name": "search", "response": {"error": "kennel closed"}}},
        ],
    }, f"consecutive tool results must coalesce into ONE user turn; contents: {contents}"


@respx.mock
async def test_tool_choice_none_sends_mode_none(engine: GeminiGenerateEngine) -> None:
    route = mock_generate(LEVEL_ROW, response_body())
    await engine.generate(
        LEVEL_ROW,
        intent_for(LEVEL_ROW, tools=(SEARCH_TOOL,), tool_choice="none"),
        CREDENTIAL,
    )
    body = last_request_json(route)
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}, f"body: {body}"


async def test_unknown_tool_result_call_id_raises_invalid_request(
    engine: GeminiGenerateEngine,
) -> None:
    messages: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("go"),)),
        AssistantMessage(
            text="",
            tool_calls=(ToolCall(id="call_0", name="search", arguments={}),),
            continuation=Absent(),
        ),
        ToolResultMessage(call_id="call_9", output="?", is_error=False),
    )
    with pytest.raises(InvalidRequest, match="call_9"):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW, messages=messages), CREDENTIAL)


async def test_duplicate_tool_call_ids_in_turn_raise_invalid_request(
    engine: GeminiGenerateEngine,
) -> None:
    messages: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("go"),)),
        AssistantMessage(
            text="",
            tool_calls=(
                ToolCall(id="call_0", name="search", arguments={}),
                ToolCall(id="call_0", name="search", arguments={}),
            ),
            continuation=Absent(),
        ),
    )
    with pytest.raises(InvalidRequest, match="call_0"):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW, messages=messages), CREDENTIAL)


@respx.mock
async def test_image_blocks_encode_as_inline_data_parts(engine: GeminiGenerateEngine) -> None:
    route = mock_generate(LEVEL_ROW, response_body())
    png = b"\x89PNG"
    messages: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("look:"), ImageBlock(media_type="image/png", data=png))),
    )
    await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW, messages=messages), CREDENTIAL)
    body = last_request_json(route)
    contents = body["contents"]
    assert isinstance(contents, list)
    parts = contents[0]["parts"]
    assert parts[0] == {"text": "look:"}, f"contents: {contents}"
    inline = parts[1]["inlineData"]
    assert inline["mimeType"] == "image/png", f"contents: {contents}"
    # The SDK base64-encodes bytes (urlsafe alphabet); these bytes decode back.
    assert inline["data"] in (
        b64encode(png).decode("ascii"),
        b64encode(png).decode("ascii").replace("+", "-").replace("/", "_"),
    ), f"contents: {contents}"


@respx.mock
async def test_structured_native_sends_mime_and_json_schema(engine: GeminiGenerateEngine) -> None:
    route = mock_generate(LEVEL_ROW, response_body(parts=[{"text": '{"answer": "42"}'}]))
    outcome = await engine.generate(
        LEVEL_ROW,
        intent_for(LEVEL_ROW, output=StrictJsonOutput("answer", ANSWER_SCHEMA)),
        CREDENTIAL,
    )
    config = last_request_json(route)["generationConfig"]
    assert isinstance(config, dict)
    assert config["responseMimeType"] == "application/json", f"config: {config}"
    assert config["responseJsonSchema"] == ANSWER_SCHEMA, (
        f"structured='native' rows send the caller schema verbatim; config: {config}"
    )
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == StructuredContent(
        payload={"answer": "42"}, text='{"answer": "42"}'
    ), f"StrictJsonOutput must decode to StructuredContent, got {outcome.response.content}"


@respx.mock
async def test_structured_json_mode_sends_mime_only(engine: GeminiGenerateEngine) -> None:
    route = mock_generate(
        KNOBLESS_ROW,
        response_body(parts=[{"text": '{"answer": "x"}'}], model_version="gemini-2.0-flash-lite"),
    )
    await engine.generate(
        KNOBLESS_ROW,
        intent_for(
            KNOBLESS_ROW, reasoning="none", output=StrictJsonOutput("answer", ANSWER_SCHEMA)
        ),
        CREDENTIAL,
    )
    config = last_request_json(route)["generationConfig"]
    assert isinstance(config, dict)
    assert config["responseMimeType"] == "application/json", f"config: {config}"
    assert "responseJsonSchema" not in config, (
        f"json_mode rows constrain to JSON only — the caller schema is enforced by "
        f"validation, not the wire; config: {config}"
    )


@respx.mock
async def test_structured_output_invalid_json_fails_invalid_structured_output(
    engine: GeminiGenerateEngine,
) -> None:
    mock_generate(LEVEL_ROW, response_body(parts=[{"text": "not json"}]))
    outcome = await engine.generate(
        LEVEL_ROW,
        intent_for(LEVEL_ROW, output=StrictJsonOutput("answer", ANSWER_SCHEMA)),
        CREDENTIAL,
    )
    assert isinstance(outcome, Failed), f"got {outcome}"
    assert isinstance(outcome.failure, InvalidStructuredOutput), f"got {outcome.failure}"


# ---------------------------------------------------------------------------
# provider_options — extension passthrough, never overrides.


@respx.mock
async def test_provider_options_config_fields_are_forwarded(engine: GeminiGenerateEngine) -> None:
    route = mock_generate(LEVEL_ROW, response_body())
    await engine.generate(
        LEVEL_ROW,
        intent_for(LEVEL_ROW, provider_options={"temperature": 0.25, "top_p": 0.9}),
        CREDENTIAL,
    )
    config = last_request_json(route)["generationConfig"]
    assert isinstance(config, dict)
    assert config["temperature"] == 0.25, f"provider_options must be forwarded; config: {config}"
    assert config["topP"] == 0.9, f"config: {config}"


@pytest.mark.parametrize(
    "key",
    [
        "max_output_tokens",
        "maxOutputTokens",
        "thinking_config",
        "thinkingConfig",
        "system_instruction",
        "tools",
        "tool_config",
        "response_mime_type",
        "response_json_schema",
        "response_schema",
        "http_options",
        "automatic_function_calling",
    ],
)
async def test_provider_options_owned_key_collision_raises_invalid_request(
    engine: GeminiGenerateEngine, key: str
) -> None:
    with pytest.raises(InvalidRequest, match=key):
        await engine.generate(
            LEVEL_ROW, intent_for(LEVEL_ROW, provider_options={key: "boom"}), CREDENTIAL
        )


async def test_provider_options_unknown_config_field_raises_invalid_request(
    engine: GeminiGenerateEngine,
) -> None:
    with pytest.raises(InvalidRequest, match="bogus_knob"):
        await engine.generate(
            LEVEL_ROW, intent_for(LEVEL_ROW, provider_options={"bogus_knob": 1}), CREDENTIAL
        )


# ---------------------------------------------------------------------------
# Response decode + CallMeta.


@respx.mock
async def test_success_decode_populates_meta_and_usage(engine: GeminiGenerateEngine) -> None:
    mock_generate(
        LEVEL_ROW,
        response_body(
            usage={
                "promptTokenCount": 100,  # cache-INCLUSIVE on this wire
                "candidatesTokenCount": 20,
                "totalTokenCount": 127,
                "thoughtsTokenCount": 7,
                "cachedContentTokenCount": 64,
            },
        ),
    )
    outcome = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert isinstance(outcome, Succeeded), f"got {outcome}"
    assert outcome.response.content == TextContent(text="hello", tool_calls=())
    meta = outcome.meta
    assert meta.provider == "gemini"
    assert meta.model == "gemini-3-pro-echo", "meta.model comes from the modelVersion echo"
    assert meta.provider_request_id == Absent(), (
        "correlation is 'none' on this wire — the body's responseId must NOT become "
        f"a request id; got {meta.provider_request_id}"
    )
    assert meta.upstream_provider == Absent()
    assert meta.registry_revision == REGISTRY_REVISION
    assert meta.native_reasoning == Present("thinkingLevel=HIGH")
    assert meta.billability == PossiblyBillable()
    assert isinstance(meta.usage, Present)
    usage = meta.usage.value
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (100, 20, 127), (
        f"promptTokenCount is already cache-inclusive and totalTokenCount is authoritative; "
        f"usage: {usage}"
    )
    assert usage.reasoning_tokens == Present(7), f"thoughtsTokenCount maps to reasoning: {usage}"
    assert usage.cache_read_input_tokens == Present(64), f"usage: {usage}"
    assert usage.cache_write_input_tokens == Absent(), (
        f"implicit caching bills no writes; usage: {usage}"
    )
    assert len(meta.attempt_trace) == 1, f"trace: {meta.attempt_trace}"
    record = meta.attempt_trace[0]
    assert record.attempt == 1
    assert isinstance(record.signal, FinalAttempt)
    assert record.status_code == Present(200)


@respx.mock
async def test_missing_model_version_falls_back_to_row_model_id(
    engine: GeminiGenerateEngine,
) -> None:
    mock_generate(LEVEL_ROW, response_body(model_version=None))
    outcome = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.model == "gemini-3-pro", f"got {outcome.meta.model}"


@respx.mock
async def test_tool_call_decode_synthesizes_deterministic_ids(
    engine: GeminiGenerateEngine,
) -> None:
    mock_generate(
        LEVEL_ROW,
        response_body(
            parts=[
                {"functionCall": {"name": "search", "args": {"query": "cats"}}},
                {"functionCall": {"name": "search", "args": {"query": "dogs"}}},
            ],
        ),
    )
    outcome = await engine.generate(
        LEVEL_ROW, intent_for(LEVEL_ROW, tools=(SEARCH_TOOL,)), CREDENTIAL
    )
    assert isinstance(outcome, Succeeded), f"got {outcome}"
    content = outcome.response.content
    assert isinstance(content, TextContent)
    assert content.tool_calls == (
        ToolCall(id="call_0", name="search", arguments={"query": "cats"}),
        ToolCall(id="call_1", name="search", arguments={"query": "dogs"}),
    ), f"the wire has no call ids — decode synthesizes call_<index>; got {content.tool_calls}"


@respx.mock
async def test_max_tokens_finish_maps_to_incomplete(engine: GeminiGenerateEngine) -> None:
    mock_generate(LEVEL_ROW, response_body(finish_reason="MAX_TOKENS"))
    outcome = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert isinstance(outcome, Incomplete), f"got {outcome}"
    assert outcome.reason == "max_output_tokens"
    assert outcome.status == "provider_incomplete"


@pytest.mark.parametrize("finish", ["SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST"])
@respx.mock
async def test_blocked_finish_reasons_map_to_content_filter_incomplete(
    engine: GeminiGenerateEngine, finish: str
) -> None:
    mock_generate(LEVEL_ROW, response_body(finish_reason=finish))
    outcome = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert isinstance(outcome, Incomplete), f"got {outcome}"
    assert outcome.reason == "content_filter_partial"
    assert outcome.status == "provider_incomplete"
    assert isinstance(outcome.safe_detail, Present)
    assert finish in outcome.safe_detail.value, f"got {outcome.safe_detail}"


@respx.mock
async def test_blocked_prompt_maps_to_content_filter_incomplete(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            200,
            json={
                "promptFeedback": {"blockReason": "SAFETY"},
                "usageMetadata": {"promptTokenCount": 4, "totalTokenCount": 4},
            },
        )
    )
    outcome = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert isinstance(outcome, Incomplete), f"a blocked prompt is Incomplete, got {outcome}"
    assert outcome.reason == "content_filter_partial"
    assert isinstance(outcome.safe_detail, Present)
    assert "SAFETY" in outcome.safe_detail.value, f"got {outcome.safe_detail}"
    assert isinstance(outcome.meta.usage, Present), "prompt-blocked usage still folds into meta"


@respx.mock
async def test_malformed_function_call_finish_returns_failed_invalid_tool_arguments(
    engine: GeminiGenerateEngine,
) -> None:
    mock_generate(LEVEL_ROW, response_body(finish_reason="MALFORMED_FUNCTION_CALL"))
    outcome = await engine.generate(
        LEVEL_ROW, intent_for(LEVEL_ROW, tools=(SEARCH_TOOL,)), CREDENTIAL
    )
    assert isinstance(outcome, Failed), f"got {outcome}"
    assert isinstance(outcome.failure, InvalidToolArguments), f"got {outcome.failure}"
    assert outcome.meta.billability == PossiblyBillable()


@respx.mock
async def test_unknown_finish_reason_raises_protocol_defect(engine: GeminiGenerateEngine) -> None:
    mock_generate(LEVEL_ROW, response_body(finish_reason="OTHER"))
    with pytest.raises(ProtocolDefect, match="OTHER"):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)


@respx.mock
async def test_missing_candidates_without_feedback_raises_protocol_defect(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(ProtocolDefect, match="candidates"):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)


@respx.mock
async def test_missing_finish_reason_raises_protocol_defect(engine: GeminiGenerateEngine) -> None:
    mock_generate(LEVEL_ROW, response_body(finish_reason=None))
    with pytest.raises(ProtocolDefect, match="finishReason"):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)


@respx.mock
async def test_malformed_json_envelope_raises_protocol_defect(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "application/json"}, content=b"not json {"
        )
    )
    with pytest.raises(ProtocolDefect):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)


@respx.mock
async def test_invalid_envelope_shape_raises_protocol_defect(engine: GeminiGenerateEngine) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(200, json={"candidates": [{"content": {"parts": "zzz"}}]})
    )
    with pytest.raises(ProtocolDefect):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)


# ---------------------------------------------------------------------------
# Continuations — ordered parts + thoughtSignatures, replayed verbatim.


@respx.mock
async def test_thought_signatures_round_trip_verbatim(engine: GeminiGenerateEngine) -> None:
    mock_generate(
        LEVEL_ROW,
        response_body(
            parts=[
                {"text": "calling now", "thoughtSignature": "c2ln"},
                {
                    "functionCall": {"name": "search", "args": {"query": "x"}},
                    "thoughtSignature": "c2lnMg==",
                },
            ],
        ),
    )
    first = await engine.generate(
        LEVEL_ROW, intent_for(LEVEL_ROW, tools=(SEARCH_TOOL,)), CREDENTIAL
    )
    assert isinstance(first, Succeeded)
    continuation = first.response.continuation
    assert isinstance(continuation, Present), "signature parts must produce an artifact"
    artifact = continuation.value
    assert artifact.codec_id == LEVEL_ROW.continuation_codec
    assert artifact.target == ProviderTarget(provider="gemini", model="gemini-3-pro")
    assert artifact.opaque_payload == {
        "parts": [
            {"text": "calling now", "thoughtSignature": "c2ln"},
            {
                "functionCall": {"name": "search", "args": {"query": "x"}},
                "thoughtSignature": "c2lnMg==",
            },
        ]
    }, f"ordered parts + signatures, verbatim; got {artifact.opaque_payload}"

    respx.clear()
    route = mock_generate(LEVEL_ROW, response_body())
    replay: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("go"),)),
        AssistantMessage(
            text="calling now",
            tool_calls=(ToolCall(id="call_0", name="search", arguments={"query": "x"}),),
            continuation=Present(artifact),
        ),
        ToolResultMessage(call_id="call_0", output="found", is_error=False),
    )
    await engine.generate(
        LEVEL_ROW, intent_for(LEVEL_ROW, messages=replay, tools=(SEARCH_TOOL,)), CREDENTIAL
    )
    contents = last_request_json(route)["contents"]
    assert isinstance(contents, list)
    assert contents[1] == {
        "role": "model",
        "parts": [
            {"text": "calling now", "thoughtSignature": "c2ln"},
            {
                "functionCall": {"name": "search", "args": {"query": "x"}},
                "thoughtSignature": "c2lnMg==",
            },
        ],
    }, f"the payload's parts are the SOLE wire source for the turn; contents: {contents}"
    assert contents[2] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "search", "response": {"output": "found"}}}],
    }, f"contents: {contents}"


@respx.mock
async def test_plain_text_response_has_no_continuation(engine: GeminiGenerateEngine) -> None:
    mock_generate(LEVEL_ROW, response_body(parts=[{"text": "just text"}]))
    outcome = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert isinstance(outcome, Succeeded)
    assert outcome.response.continuation == Absent(), (
        f"no signatures, no tool calls — nothing to replay; got {outcome.response.continuation}"
    )


async def test_continuation_bound_to_other_codec_or_target_raises_invalid_request(
    engine: GeminiGenerateEngine,
) -> None:
    wrong_codec = ContinuationArtifact(
        target=ProviderTarget(provider="gemini", model="gemini-3-pro"),
        codec_id="anthropic.v1",
        opaque_payload={"parts": [{"text": "x"}]},
    )
    messages: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("hi"),)),
        AssistantMessage(text="x", tool_calls=(), continuation=Present(wrong_codec)),
    )
    with pytest.raises(InvalidRequest):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW, messages=messages), CREDENTIAL)

    wrong_target = ContinuationArtifact(
        target=ProviderTarget(provider="gemini", model="gemini-2.5-flash"),
        codec_id="gemini.v1",
        opaque_payload={"parts": [{"text": "x"}]},
    )
    messages = (
        UserMessage((PromptBlock("hi"),)),
        AssistantMessage(text="x", tool_calls=(), continuation=Present(wrong_target)),
    )
    with pytest.raises(InvalidRequest):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW, messages=messages), CREDENTIAL)


async def test_continuation_payload_without_parts_raises_invalid_request(
    engine: GeminiGenerateEngine,
) -> None:
    broken = ContinuationArtifact(
        target=ProviderTarget(provider="gemini", model="gemini-3-pro"),
        codec_id="gemini.v1",
        opaque_payload={"reasoning": "not gemini shaped"},
    )
    messages: tuple[PromptMessage, ...] = (
        UserMessage((PromptBlock("hi"),)),
        AssistantMessage(text="x", tool_calls=(), continuation=Present(broken)),
    )
    with pytest.raises(InvalidRequest, match="parts"):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW, messages=messages), CREDENTIAL)


# ---------------------------------------------------------------------------
# Stream decode from raw SSE bytes.


@respx.mock
async def test_stream_decodes_text_tools_continuation_and_folds_usage(
    engine: GeminiGenerateEngine,
) -> None:
    route = mock_stream(
        LEVEL_ROW,
        sse_bytes(
            {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}],
                "modelVersion": "gemini-3-pro-echo",
                "usageMetadata": {"promptTokenCount": 11, "totalTokenCount": 11},
            },
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "internal plan", "thought": True},
                                {"text": "lo"},
                            ],
                        }
                    }
                ],
            },
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {"name": "search", "args": {"query": "x"}},
                                    "thoughtSignature": "c2ln",
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 6,
                    "totalTokenCount": 20,
                    "thoughtsTokenCount": 3,
                },
            },
        ),
    )
    events = await collect(
        engine.stream(LEVEL_ROW, intent_for(LEVEL_ROW, tools=(SEARCH_TOOL,)), CREDENTIAL)
    )
    assert route.called
    kinds = [type(event).__name__ for event in events]
    assert kinds == [
        "StreamStart",
        "TextDelta",
        "TextDelta",
        "ToolCallStart",
        "ToolCallDone",
        "ContinuationDelta",
        "TerminalEvent",
    ], f"events: {kinds}"
    assert not any(isinstance(event, UsageEvent) for event in events), (
        "gemini usage frames fold silently into the terminal meta (old-codec semantics); "
        f"events: {kinds}"
    )
    assert events[1] == TextDelta(text="Hel")
    assert events[2] == TextDelta(text="lo"), "thought-summary parts are not visible output"
    assert events[3] == ToolCallStart(call_id="call_0", name="search")
    done = events[4]
    assert isinstance(done, ToolCallDone)
    assert done.tool_call == ToolCall(id="call_0", name="search", arguments={"query": "x"})

    continuation_event = events[5]
    assert isinstance(continuation_event, ContinuationDelta)
    artifact = continuation_event.artifact
    assert artifact.codec_id == "gemini.v1"
    payload_parts = artifact.opaque_payload.get("parts")
    assert payload_parts == [
        {"text": "Hel"},
        {"text": "internal plan", "thought": True},
        {"text": "lo"},
        {"functionCall": {"name": "search", "args": {"query": "x"}}, "thoughtSignature": "c2ln"},
    ], f"ALL parts accumulate in order, thoughts and signatures included; got {payload_parts}"

    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded), f"got {outcome}"
    assert outcome.response.content == TextContent(
        text="Hello",
        tool_calls=(ToolCall(id="call_0", name="search", arguments={"query": "x"}),),
    )
    assert outcome.response.continuation == Present(artifact)
    meta = outcome.meta
    assert meta.model == "gemini-3-pro-echo"
    assert meta.provider_request_id == Absent()
    assert isinstance(meta.usage, Present), "terminal meta must fold all usage frames"
    usage = meta.usage.value
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (11, 6, 20), (
        f"cumulative frames fold field-wise, later values win; usage: {usage}"
    )
    assert usage.reasoning_tokens == Present(3), f"usage: {usage}"


@respx.mock
async def test_stream_blocked_prompt_yields_incomplete_terminal(
    engine: GeminiGenerateEngine,
) -> None:
    mock_stream(
        LEVEL_ROW,
        sse_bytes(
            {
                "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
                "usageMetadata": {"promptTokenCount": 4, "totalTokenCount": 4},
            },
        ),
    )
    events = await collect(engine.stream(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL))
    kinds = [type(event).__name__ for event in events]
    assert kinds == ["StreamStart", "TerminalEvent"], f"events: {kinds}"
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Incomplete), f"got {outcome}"
    assert outcome.reason == "content_filter_partial"
    assert isinstance(outcome.safe_detail, Present)
    assert "PROHIBITED_CONTENT" in outcome.safe_detail.value


@respx.mock
async def test_stream_429_at_acceptance_raises_before_stream_start(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(stream_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            429,
            json={"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED"}},
        )
    )
    stream = engine.stream(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    with pytest.raises(TransientAttempt) as excinfo:
        await anext(stream)
    assert isinstance(excinfo.value.cause, ProviderRateLimit), f"got {excinfo.value.cause}"


@respx.mock
async def test_stream_context_overflow_yields_failed_terminal(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(stream_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "The input token count exceeds the maximum number of tokens.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
    )
    events = await collect(engine.stream(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL))
    kinds = [type(event).__name__ for event in events]
    assert kinds == ["TerminalEvent"], f"a pre-acceptance failure has no envelope; events: {kinds}"
    terminal = events[0]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed), f"got {outcome}"
    assert outcome.failure == ProviderContextTooLarge()


@respx.mock
async def test_stream_ends_without_finish_reason_interrupted_not_partial(
    engine: GeminiGenerateEngine,
) -> None:
    # A usage-only frame is NOT semantic output; the stream then ends with no
    # finishReason frame — a cut, retryable.
    mock_stream(
        LEVEL_ROW,
        sse_bytes({"usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3}}),
    )
    stream = engine.stream(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    seen: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as excinfo:
        async for event in stream:
            seen.append(event)
    assert seen == [StreamStart()], f"only the envelope may precede the cut; got {seen}"
    assert excinfo.value.cause == ProviderStreamInterrupted(partial_output=False), (
        f"no semantic output was yielded; got {excinfo.value.cause}"
    )


@respx.mock
async def test_stream_transport_cut_after_semantic_output_is_partial(
    engine: GeminiGenerateEngine,
) -> None:
    first = sse_bytes({"candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}]})
    respx.post(stream_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=CutByteStream((first,)),
        )
    )
    stream = engine.stream(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    seen: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as excinfo:
        async for event in stream:
            seen.append(event)
    assert any(isinstance(event, TextDelta) for event in seen), f"events: {seen}"
    assert excinfo.value.cause == ProviderStreamInterrupted(partial_output=True), (
        f"semantic output was already yielded; got {excinfo.value.cause}"
    )
    assert excinfo.value.billability == PossiblyBillable()


@respx.mock
async def test_stream_inband_error_pre_semantic_classifies_rate_limit(
    engine: GeminiGenerateEngine,
) -> None:
    mock_stream(
        LEVEL_ROW,
        sse_bytes({"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
    )
    stream = engine.stream(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    seen: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as excinfo:
        async for event in stream:
            seen.append(event)
    assert seen == [StreamStart()], f"only the envelope may precede the failure; got {seen}"
    assert excinfo.value.cause == ProviderRateLimit(retry_after=Absent()), (
        f"429-shaped in-band errors classify as rate limit; got {excinfo.value.cause}"
    )


@respx.mock
async def test_stream_inband_error_post_semantic_is_partial(engine: GeminiGenerateEngine) -> None:
    mock_stream(
        LEVEL_ROW,
        sse_bytes(
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "par"}]}}]},
            {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}},
        ),
    )
    stream = engine.stream(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    seen: list[CodecStreamEvent] = []
    with pytest.raises(TransientAttempt) as excinfo:
        async for event in stream:
            seen.append(event)
    assert any(isinstance(event, TextDelta) for event in seen), f"events: {seen}"
    assert excinfo.value.cause == ProviderStreamInterrupted(partial_output=True)


@respx.mock
async def test_stream_malformed_frame_raises_protocol_defect(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(stream_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: {not json\n\n",
        )
    )
    stream = engine.stream(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    with pytest.raises(ProtocolDefect):
        async for _ in stream:
            pass


# ---------------------------------------------------------------------------
# Fault injection (generate).


@respx.mock
async def test_429_with_retry_after_raises_transient_rate_limit(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "7"},
            json={"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED"}},
        )
    )
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    attempt = excinfo.value
    assert attempt.cause == ProviderRateLimit(retry_after=Present(7.0)), f"got {attempt.cause}"
    assert attempt.status_code == Present(429)
    assert attempt.billability == PossiblyBillable()


@respx.mock
async def test_429_without_retry_after_has_absent_delay(engine: GeminiGenerateEngine) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            429,
            json={"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED"}},
        )
    )
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert excinfo.value.cause == ProviderRateLimit(retry_after=Absent()), (
        f"got {excinfo.value.cause}"
    )


@respx.mock
async def test_timeouts_raise_transient_provider_timeout(engine: GeminiGenerateEngine) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(side_effect=httpx.ConnectTimeout("boom"))
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert excinfo.value.cause == ProviderTimeout(), f"got {excinfo.value.cause}"
    assert excinfo.value.status_code == Absent()

    respx.clear()
    respx.post(generate_url(LEVEL_ROW)).mock(side_effect=httpx.ReadTimeout("boom"))
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert excinfo.value.cause == ProviderTimeout(), f"got {excinfo.value.cause}"


@respx.mock
async def test_5xx_raises_transient_provider_unavailable(engine: GeminiGenerateEngine) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            503, json={"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
        )
    )
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert excinfo.value.cause == ProviderHttpUnavailable(), f"got {excinfo.value.cause}"
    assert excinfo.value.status_code == Present(503)


@respx.mock
async def test_connect_error_is_transport_unavailable_not_dispatched(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert excinfo.value.cause == TransportUnavailable(), f"got {excinfo.value.cause}"
    assert excinfo.value.billability == NotDispatched(), (
        "a pure pre-connect failure means no bytes reached the provider"
    )


@respx.mock
async def test_mid_request_transport_error_is_possibly_billable(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(side_effect=httpx.ReadError("broken pipe"))
    with pytest.raises(TransientAttempt) as excinfo:
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert excinfo.value.cause == TransportUnavailable(), f"got {excinfo.value.cause}"
    assert excinfo.value.billability == PossiblyBillable(), (
        "the connection was opened, so the provider may have seen the request"
    )


@respx.mock
async def test_context_overflow_400_returns_failed_value_with_meta(
    engine: GeminiGenerateEngine,
) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "The input token count exceeds the maximum number of tokens.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
    )
    outcome = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert isinstance(outcome, Failed), f"got {outcome}"
    assert outcome.failure == ProviderContextTooLarge()
    meta = outcome.meta
    assert meta.provider == "gemini"
    assert meta.model == "gemini-3-pro", "no envelope decoded — the row's model id stands in"
    assert meta.usage == Absent()
    assert meta.billability == PossiblyBillable()
    assert meta.registry_revision == REGISTRY_REVISION
    assert len(meta.attempt_trace) == 1
    assert meta.attempt_trace[0].status_code == Present(400)


@respx.mock
async def test_unclassified_400_raises_runtime_defect(engine: GeminiGenerateEngine) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            400,
            json={"error": {"code": 400, "message": "bad field", "status": "INVALID_ARGUMENT"}},
        )
    )
    with pytest.raises(RuntimeDefect) as excinfo:
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert excinfo.value.code == "unclassified_provider_error", f"got {excinfo.value.code}"


@respx.mock
async def test_401_and_403_raise_credential_rejected(engine: GeminiGenerateEngine) -> None:
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            401,
            json={"error": {"code": 401, "message": "bad key", "status": "UNAUTHENTICATED"}},
        )
    )
    with pytest.raises(CredentialRejected):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)

    respx.clear()
    respx.post(generate_url(LEVEL_ROW)).mock(
        return_value=httpx.Response(
            403,
            json={"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}},
        )
    )
    with pytest.raises(CredentialRejected):
        await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)


# ---------------------------------------------------------------------------
# http_client injection seam.


async def test_injected_http_client_is_used_and_left_open() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=response_body())

    injected = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    engine = GeminiGenerateEngine(http_client=injected)
    first = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    second = await engine.generate(LEVEL_ROW, intent_for(LEVEL_ROW), CREDENTIAL)
    assert isinstance(first, Succeeded) and isinstance(second, Succeeded), f"got {first} / {second}"
    assert len(calls) == 2, f"both calls must ride the injected client; calls: {calls}"
    assert not injected.is_closed, "the engine must never close an injected http client"
    await injected.aclose()
