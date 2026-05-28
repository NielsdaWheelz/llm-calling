import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_calling.gemini import GeminiClient
from llm_calling.types import (
    LLMRequest,
    StructuredOutputSpec,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
)

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures" / "gemini"


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def request() -> LLMRequest:
    return LLMRequest(
        model_name="gemini-2.5-pro",
        messages=[
            Turn(role="system", content="You are helpful."),
            Turn(role="user", content="Hello!"),
            Turn(role="assistant", content="Hi."),
        ],
        max_tokens=100,
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
    }


@respx.mock
async def test_nonstream_success() -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    ).respond(200, json=load_json("success_nonstream.json"))

    async with httpx.AsyncClient() as http:
        response = await GeminiClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert response.text == "Hello! How can I help you today?"
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 8
    assert response.usage.total_tokens == 18
    assert response.provider_request_id is None


@respx.mock
async def test_stream_success() -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
    ).respond(200, content=load_text("success_stream_chunks.txt"))

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in GeminiClient(http).generate_stream(
                request(), api_key="sk-test", timeout_s=30
            )
        ]

    assert chunks[-1].done is True
    assert all(chunk.usage is None for chunk in chunks[:-1])
    assert "Hello" in "".join(chunk.delta_text for chunk in chunks)


@respx.mock
async def test_assistant_role_maps_to_model() -> None:
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    ).respond(200, json=load_json("success_nonstream.json"))

    async with httpx.AsyncClient() as http:
        await GeminiClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["systemInstruction"] == {"parts": [{"text": "You are helpful."}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Hello!"}]},
        {"role": "model", "parts": [{"text": "Hi."}]},
    ]


@respx.mock
async def test_structured_output_uses_response_json_schema_and_parses_response() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["candidates"][0]["content"]["parts"][0]["text"] = (
        '{"title":"The Book","language":"en"}'
    )
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    ).respond(200, json=response_json)
    req = LLMRequest(
        model_name="gemini-2.5-pro",
        messages=[Turn(role="user", content="Extract metadata.")],
        max_tokens=100,
        structured_output=StructuredOutputSpec(
            name="metadata_enrichment",
            schema=metadata_schema(),
        ),
    )

    async with httpx.AsyncClient() as http:
        response = await GeminiClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseJsonSchema"] == metadata_schema()
    assert response.structured_output == {"title": "The Book", "language": "en"}


@respx.mock
async def test_tool_call_nonstream_and_request_body() -> None:
    response_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8,
        },
    }
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    ).respond(200, json=response_json)
    req = LLMRequest(
        model_name="gemini-2.5-pro",
        messages=[
            Turn(role="user", content="What's the weather?"),
            Turn(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"}),),
            ),
            Turn(
                role="tool",
                tool_results=(ToolResult(call_id="call_1", output="sunny"),),
            ),
        ],
        max_tokens=100,
        tools=(
            ToolSpec(
                name="get_weather",
                description="Get the weather.",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ),
        tool_choice="required",
    )

    async with httpx.AsyncClient() as http:
        response = await GeminiClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ]
        }
    ]
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "What's the weather?"}]},
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}],
        },
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "get_weather", "response": {"output": "sunny"}}}
            ],
        },
    ]
    assert response.tool_calls == (
        ToolCall(id="get_weather", name="get_weather", arguments={"city": "Paris"}),
    )


@respx.mock
async def test_stream_yields_tool_call_chunk() -> None:
    stream = (
        'data: {"candidates":[{"content":{"parts":[{"functionCall":'
        '{"name":"get_weather","args":{"city":"Paris"}}}],"role":"model"},'
        '"index":0,"finishReason":"STOP"}],'
        '"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}\n\n'
    )
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
    ).respond(200, content=stream)

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in GeminiClient(http).generate_stream(
                request(), api_key="sk-test", timeout_s=30
            )
        ]

    tool_chunks = [c for c in chunks if c.tool_call is not None]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].tool_call == ToolCall(
        id="get_weather", name="get_weather", arguments={"city": "Paris"}
    )
    assert chunks[-1].done is True
