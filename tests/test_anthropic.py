import json
from pathlib import Path

import httpx
import pytest
import respx

from provider_runtime.anthropic import AnthropicClient
from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.types import (
    ModelCall,
    ModelMessage,
    ModelRef,
    ProviderArtifact,
    ReasoningConfig,
    StructuredOutputSpec,
    ToolCall,
    ToolResult,
    ToolSpec,
)

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures" / "anthropic"


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def request() -> ModelCall:
    return ModelCall(
        model=ModelRef(provider="anthropic", model="claude-3-opus-20240229"),
        messages=[
            ModelMessage(role="system", content="You are helpful."),
            ModelMessage(role="user", content="Hello!"),
        ],
        max_output_tokens=100,
        temperature=0.7,
    )


def metadata_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "language": {"type": "string"},
        },
        "required": ["title", "language"],
        "additionalProperties": False,
    }


@respx.mock
async def test_nonstream_success() -> None:
    respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        json=load_json("success_nonstream.json"),
    )

    async with httpx.AsyncClient() as http:
        response = await AnthropicClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert response.text == "Hello! How can I help you today?"
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 8
    assert response.usage.total_tokens == 18
    assert response.provider_request_id == "msg_test123"


@respx.mock
async def test_stream_success() -> None:
    respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        content=load_text("success_stream_chunks.txt"),
        headers={"content-type": "text/event-stream"},
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in AnthropicClient(http).generate_stream(
                request(), api_key="sk-test", timeout_s=30
            )
        ]

    assert chunks[-1].done is True
    assert chunks[-1].provider_request_id == "msg_test123"
    assert all(chunk.usage is None for chunk in chunks[:-1])
    assert "Hello" in "".join(chunk.delta_text for chunk in chunks)


@respx.mock
async def test_system_turn_uses_system_field() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        json=load_json("success_nonstream.json"),
    )

    async with httpx.AsyncClient() as http:
        await AnthropicClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["system"] == [{"type": "text", "text": "You are helpful."}]
    assert body["messages"] == [{"role": "user", "content": "Hello!"}]


@respx.mock
async def test_system_turn_can_mark_prompt_cache_breakpoint() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        json=load_json("success_nonstream.json"),
    )
    req = ModelCall(
        model=ModelRef(provider="anthropic", model="claude-3-opus-20240229"),
        messages=[
            ModelMessage(role="system", content="Stable.", cache_ttl="5m"),
            ModelMessage(role="system", content="Dynamic."),
            ModelMessage(role="user", content="Hello!"),
        ],
        max_output_tokens=100,
    )

    async with httpx.AsyncClient() as http:
        await AnthropicClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["system"] == [
        {
            "type": "text",
            "text": "Stable.",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        },
        {"type": "text", "text": "Dynamic."},
    ]


@respx.mock
async def test_structured_output_uses_forced_tool_and_parses_tool_input() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["content"] = [
        {
            "type": "tool_use",
            "id": "toolu_metadata",
            "name": "metadata_enrichment",
            "input": {"title": "The Book", "language": "en"},
        }
    ]
    response_json["stop_reason"] = "tool_use"
    route = respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        json=response_json,
    )
    req = ModelCall(
        model=ModelRef(provider="anthropic", model="claude-3-opus-20240229"),
        messages=[ModelMessage(role="user", content="Extract metadata.")],
        max_output_tokens=100,
        structured_output=StructuredOutputSpec(
            name="metadata_enrichment",
            schema=metadata_schema(),
        ),
    )

    async with httpx.AsyncClient() as http:
        response = await AnthropicClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [
        {
            "name": "metadata_enrichment",
            "description": "Return metadata_enrichment.",
            "input_schema": metadata_schema(),
        }
    ]
    assert body["tool_choice"] == {"type": "tool", "name": "metadata_enrichment"}
    assert response.text == ""
    assert response.structured_output == {"title": "The Book", "language": "en"}


async def test_structured_output_rejects_extended_thinking() -> None:
    req = ModelCall(
        model=ModelRef(provider="anthropic", model="claude-3-opus-20240229"),
        messages=[ModelMessage(role="user", content="Extract metadata.")],
        max_output_tokens=100,
        reasoning=ReasoningConfig(effort="high"),
        structured_output=StructuredOutputSpec(
            name="metadata_enrichment",
            schema=metadata_schema(),
        ),
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await AnthropicClient(http).generate(req, api_key="sk-test", timeout_s=30)

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST


@respx.mock
async def test_tool_use_in_nonstream_response_populates_tool_calls() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["content"] = [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "get_weather",
            "input": {"city": "SF"},
        },
    ]
    response_json["stop_reason"] = "tool_use"
    route = respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        json=response_json,
    )
    req = ModelCall(
        model=ModelRef(provider="anthropic", model="claude-3-opus-20240229"),
        messages=[ModelMessage(role="user", content="Weather?")],
        max_output_tokens=100,
        tools=(
            ToolSpec(
                name="get_weather",
                description="Look up weather.",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ),
        tool_choice="required",
    )

    async with httpx.AsyncClient() as http:
        response = await AnthropicClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "Look up weather.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    assert body["tool_choice"] == {"type": "any"}
    assert response.text == "Let me check."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "toolu_abc"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "SF"}


@respx.mock
async def test_tool_use_nonstream_non_object_arguments_raise_typed_error() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["content"] = [
        {
            "type": "tool_use",
            "id": "toolu_bad",
            "name": "get_weather",
            "input": ["not", "an", "object"],
        }
    ]
    response_json["stop_reason"] = "tool_use"
    respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        json=response_json,
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await AnthropicClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert exc_info.value.error_code == ModelCallErrorCode.TOOL_ARGUMENTS_INVALID
    assert exc_info.value.retryable is False


@respx.mock
async def test_tool_use_streaming_emits_tool_call_chunk() -> None:
    stream = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_tools","type":"message","role":"assistant","content":[],"model":"claude-3-opus-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_xyz","name":"get_weather","input":{}}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":"}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"SF\\"}"}}\n'
        "\n"
        "event: content_block_stop\n"
        'data: {"type":"content_block_stop","index":0}\n'
        "\n"
        "event: message_delta\n"
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":4}}\n'
        "\n"
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n'
        "\n"
    )
    respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        content=stream,
        headers={"content-type": "text/event-stream"},
    )
    req = ModelCall(
        model=ModelRef(provider="anthropic", model="claude-3-opus-20240229"),
        messages=[ModelMessage(role="user", content="Weather?")],
        max_output_tokens=100,
        tools=(
            ToolSpec(
                name="get_weather",
                description="Look up weather.",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ),
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in AnthropicClient(http).generate_stream(
                req, api_key="sk-test", timeout_s=30
            )
        ]

    tool_chunks = [c for c in chunks if c.tool_call is not None]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].tool_call == ToolCall(
        id="toolu_xyz", name="get_weather", arguments={"city": "SF"}
    )
    assert chunks[-1].done is True


@respx.mock
async def test_stream_thinking_blocks_yield_provider_artifact_chunks() -> None:
    stream = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_think","type":"message","role":"assistant","content":[],"model":"claude-opus-4-7","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me"}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" think."}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-abc"}}\n'
        "\n"
        "event: content_block_stop\n"
        'data: {"type":"content_block_stop","index":0}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque-bytes"}}\n'
        "\n"
        "event: content_block_stop\n"
        'data: {"type":"content_block_stop","index":1}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Answer."}}\n'
        "\n"
        "event: content_block_stop\n"
        'data: {"type":"content_block_stop","index":2}\n'
        "\n"
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n'
        "\n"
    )
    respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        content=stream,
        headers={"content-type": "text/event-stream"},
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in AnthropicClient(http).generate_stream(
                request(), api_key="sk-test", timeout_s=30
            )
        ]

    item_chunks = [
        chunk.provider_artifact for chunk in chunks if chunk.provider_artifact is not None
    ]
    assert [artifact.to_provider_payload() for artifact in item_chunks] == [
        {"type": "thinking", "thinking": "Let me think.", "signature": "sig-abc"},
        {"type": "redacted_thinking", "data": "opaque-bytes"},
    ]
    assert "".join(chunk.delta_text for chunk in chunks) == "Answer."
    assert chunks[-1].done is True


@respx.mock
async def test_assistant_provider_artifacts_lead_replayed_content() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").respond(
        200,
        json=load_json("success_nonstream.json"),
    )
    thinking = {"type": "thinking", "thinking": "Plan.", "signature": "sig-abc"}
    redacted = {"type": "redacted_thinking", "data": "opaque-bytes"}
    thinking_artifact = ProviderArtifact(
        provider="anthropic",
        model="claude-opus-4-7",
        purpose="thinking",
        payload=thinking,
    )
    redacted_artifact = ProviderArtifact(
        provider="anthropic",
        model="claude-opus-4-7",
        purpose="thinking",
        payload=redacted,
    )
    req = ModelCall(
        model=ModelRef(provider="anthropic", model="claude-opus-4-7"),
        messages=[
            ModelMessage(role="user", content="Weather?"),
            ModelMessage(
                role="assistant",
                content="Checking.",
                tool_calls=(ToolCall(id="toolu_1", name="get_weather", arguments={"city": "SF"}),),
                provider_artifacts=(thinking_artifact, redacted_artifact),
            ),
            ModelMessage(
                role="tool", tool_results=(ToolResult(call_id="toolu_1", output="sunny"),)
            ),
        ],
        max_output_tokens=2000,
        reasoning=ReasoningConfig(effort="high"),
    )

    async with httpx.AsyncClient() as http:
        await AnthropicClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["messages"][1] == {
        "role": "assistant",
        "content": [
            thinking,
            redacted,
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}},
        ],
    }


@respx.mock
async def test_nonstream_thinking_blocks_exposed_as_provider_artifacts() -> None:
    response_json = load_json("success_nonstream.json")
    thinking = {"type": "thinking", "thinking": "Plan.", "signature": "sig-abc"}
    response_json["content"] = [
        thinking,
        {"type": "text", "text": "Answer."},
    ]
    respx.post("https://api.anthropic.com/v1/messages").respond(200, json=response_json)

    async with httpx.AsyncClient() as http:
        response = await AnthropicClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert len(response.provider_artifacts) == 1
    artifact = response.provider_artifacts[0]
    assert artifact.provider == "anthropic"
    assert artifact.model == "claude-3-opus-20240229"
    assert artifact.purpose == "thinking"
    assert artifact.to_provider_payload() == thinking
    assert response.text == "Answer."


async def test_usage_parses_cache_tokens() -> None:
    async with httpx.AsyncClient() as http:
        usage = AnthropicClient(http)._parse_usage(
            {
                "input_tokens": 10,
                "output_tokens": 8,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 50,
            }
        )

    assert usage.input_tokens == 10
    assert usage.output_tokens == 8
    assert usage.cache_creation_input_tokens == 100
    assert usage.cache_read_input_tokens == 50
    assert usage.total_tokens == 168
