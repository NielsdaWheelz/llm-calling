import json
from pathlib import Path

import httpx
import pytest
import respx

from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.openai_compatible import OpenAICompatibleChatClient
from provider_runtime.types import (
    ModelCall,
    ModelMessage,
    ModelRef,
    ReasoningConfig,
    StructuredOutputSpec,
    ToolCall,
    ToolResult,
    ToolSpec,
)

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures" / "openai_compatible"


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def chat_client(http: httpx.AsyncClient) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        http, provider="openrouter", base_url="https://openrouter.test/v1"
    )


def request() -> ModelCall:
    return ModelCall(
        model=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.6"),
        messages=[ModelMessage(role="user", content="Hello!")],
        max_output_tokens=100,
        temperature=0.7,
    )


@respx.mock
async def test_nonstream_success() -> None:
    respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        json=load_json("success_nonstream.json"),
        headers={"x-request-id": "req-openrouter-123"},
    )

    async with httpx.AsyncClient() as http:
        response = await chat_client(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert response.text == "Hello from OpenAI-compatible."
    assert response.usage is not None
    assert response.usage.input_tokens == 9
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 14
    assert response.provider_request_id == "req-openrouter-123"


@respx.mock
async def test_stream_success() -> None:
    respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        content=load_text("success_stream_chunks.txt"),
        headers={"x-request-id": "req-openrouter-123", "content-type": "text/event-stream"},
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in chat_client(http).generate_stream(
                request(), api_key="sk-test", timeout_s=30
            )
        ]

    assert chunks[-1].done is True
    assert chunks[-1].provider_request_id == "req-openrouter-123"
    assert all(chunk.usage is None for chunk in chunks[:-1])
    assert "Hello from OpenAI-compatible." in "".join(chunk.delta_text for chunk in chunks)


@respx.mock
async def test_openrouter_reasoning_effort_is_forwarded() -> None:
    route = respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        json=load_json("success_nonstream.json"),
    )
    req = ModelCall(
        model=ModelRef(provider="openrouter", model="openai/gpt-oss-120b"),
        messages=[ModelMessage(role="user", content="Hello!")],
        max_output_tokens=100,
        temperature=0.7,
        reasoning=ReasoningConfig(effort="high"),
    )

    async with httpx.AsyncClient() as http:
        await chat_client(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["reasoning"] == {"effort": "high"}
    assert body["temperature"] == 0.7


@respx.mock
async def test_openrouter_cache_intent_is_stripped() -> None:
    route = respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        json=load_json("success_nonstream.json"),
    )
    req = ModelCall(
        model=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.6"),
        messages=[ModelMessage(role="system", content="Stable.", cache_ttl="5m")],
        max_output_tokens=100,
        prompt_cache_key="cache-key",
    )

    async with httpx.AsyncClient() as http:
        await chat_client(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert "cache_control" not in body


@respx.mock
async def test_nonstream_tool_calls() -> None:
    route = respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        json={
            "id": "chatcmpl-tool-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
    )
    req = ModelCall(
        model=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.6"),
        messages=[ModelMessage(role="user", content="Weather?")],
        max_output_tokens=100,
        tools=(
            ToolSpec(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ),
        tool_choice="auto",
    )

    async with httpx.AsyncClient() as http:
        response = await chat_client(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "strict": True,
            },
        }
    ]
    assert body["tool_choice"] == "auto"
    assert response.text == ""
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_abc"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}


@respx.mock
async def test_nonstream_malformed_tool_arguments_raise_typed_error() -> None:
    respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        json={
            "id": "chatcmpl-tool-bad",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": ',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )
    req = ModelCall(
        model=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.6"),
        messages=[ModelMessage(role="user", content="Weather?")],
        max_output_tokens=100,
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await chat_client(http).generate(req, api_key="sk-test", timeout_s=30)

    assert exc_info.value.error_code == ModelCallErrorCode.TOOL_ARGUMENTS_INVALID
    assert exc_info.value.retryable is False


@respx.mock
async def test_stream_tool_calls() -> None:
    stream_body = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_xyz",'
        '"function":{"name":"get_weather","arguments":"{\\"ci"}}]}}]}\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        '"function":{"arguments":"ty\\": \\"Paris\\"}"}}]},"finish_reason":"tool_calls"}],'
        '"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}\n'
        "data: [DONE]\n"
    )
    respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        content=stream_body,
        headers={"x-request-id": "req-tool-1", "content-type": "text/event-stream"},
    )
    req = ModelCall(
        model=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.6"),
        messages=[ModelMessage(role="user", content="Weather?")],
        max_output_tokens=100,
        tools=(
            ToolSpec(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in chat_client(http).generate_stream(
                req, api_key="sk-test", timeout_s=30
            )
        ]

    tool_chunks = [c for c in chunks if c.tool_call is not None]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].tool_call == ToolCall(
        id="call_xyz", name="get_weather", arguments={"city": "Paris"}
    )
    assert chunks[-1].done is True


@respx.mock
async def test_reasoning_content_not_read_and_not_replayed() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["choices"][0]["message"]["reasoning_content"] = "secret chain of thought"
    route = respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        json=response_json,
    )
    req = ModelCall(
        model=ModelRef(provider="openrouter", model="openai/gpt-oss-120b"),
        messages=[
            ModelMessage(role="user", content="Weather?"),
            ModelMessage(
                role="assistant",
                content="Checking.",
                tool_calls=(
                    ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"}),
                ),
            ),
            ModelMessage(role="tool", tool_results=(ToolResult(call_id="call_1", output="sunny"),)),
        ],
        max_output_tokens=100,
        reasoning=ReasoningConfig(effort="high"),
    )

    async with httpx.AsyncClient() as http:
        response = await chat_client(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["messages"][1] == {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }
    assert response.text == "Hello from OpenAI-compatible."


@respx.mock
async def test_stream_reasoning_content_not_in_delta_text() -> None:
    stream_body = (
        'data: {"choices":[{"delta":{"reasoning_content":"secret thought"}}]}\n'
        'data: {"choices":[{"delta":{"content":"Visible."}}]}\n'
        "data: [DONE]\n"
    )
    respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        content=stream_body,
        headers={"content-type": "text/event-stream"},
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in chat_client(http).generate_stream(
                request(), api_key="sk-test", timeout_s=30
            )
        ]

    assert "".join(chunk.delta_text for chunk in chunks) == "Visible."
    assert chunks[-1].done is True


@respx.mock
async def test_structured_output_uses_json_schema_response_format() -> None:
    route = respx.post("https://openrouter.test/v1/chat/completions").respond(
        200,
        json={
            "id": "chatcmpl-structured",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"title":"The Book"}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
    )
    req = ModelCall(
        model=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.6"),
        messages=[ModelMessage(role="user", content="Extract metadata.")],
        max_output_tokens=100,
        structured_output=StructuredOutputSpec(
            name="metadata_enrichment",
            schema={"type": "object", "properties": {}},
        ),
    )

    async with httpx.AsyncClient() as http:
        response = await chat_client(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "metadata_enrichment",
            "schema": {"type": "object", "properties": {}},
            "strict": True,
        },
    }
    assert response.structured_output == {"title": "The Book"}
