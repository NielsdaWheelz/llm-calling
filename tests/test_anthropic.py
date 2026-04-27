import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_calling.anthropic import AnthropicClient
from llm_calling.types import LLMRequest, Turn

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures" / "anthropic"


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def request() -> LLMRequest:
    return LLMRequest(
        model_name="claude-3-opus-20240229",
        messages=[
            Turn(role="system", content="You are helpful."),
            Turn(role="user", content="Hello!"),
        ],
        max_tokens=100,
        temperature=0.7,
    )


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
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 8
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
    assert body["system"] == "You are helpful."
    assert body["messages"] == [{"role": "user", "content": "Hello!"}]
