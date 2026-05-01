import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_calling.deepseek import DeepSeekClient
from llm_calling.types import LLMRequest, Turn

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures" / "deepseek"


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def request() -> LLMRequest:
    return LLMRequest(
        model_name="deepseek-chat",
        messages=[Turn(role="user", content="Hello!")],
        max_tokens=100,
        temperature=0.7,
    )


@respx.mock
async def test_nonstream_success() -> None:
    respx.post("https://api.deepseek.com/chat/completions").respond(
        200,
        json=load_json("success_nonstream.json"),
        headers={"x-request-id": "req-deepseek-123"},
    )

    async with httpx.AsyncClient() as http:
        response = await DeepSeekClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert response.text == "Hello from DeepSeek."
    assert response.usage is not None
    assert response.usage.input_tokens == 9
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 14
    assert response.provider_request_id == "req-deepseek-123"


@respx.mock
async def test_stream_success() -> None:
    respx.post("https://api.deepseek.com/chat/completions").respond(
        200,
        content=load_text("success_stream_chunks.txt"),
        headers={"x-request-id": "req-deepseek-123", "content-type": "text/event-stream"},
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in DeepSeekClient(http).generate_stream(
                request(), api_key="sk-test", timeout_s=30
            )
        ]

    assert chunks[-1].done is True
    assert chunks[-1].provider_request_id == "req-deepseek-123"
    assert all(chunk.usage is None for chunk in chunks[:-1])
    assert "Hello from DeepSeek." in "".join(chunk.delta_text for chunk in chunks)


@respx.mock
async def test_v4_reasoning_enables_thinking_and_omits_temperature() -> None:
    route = respx.post("https://api.deepseek.com/chat/completions").respond(
        200,
        json=load_json("success_nonstream.json"),
    )
    req = LLMRequest(
        model_name="deepseek-v4-pro",
        messages=[Turn(role="user", content="Hello!")],
        max_tokens=100,
        temperature=0.7,
        reasoning_effort="high",
    )

    async with httpx.AsyncClient() as http:
        await DeepSeekClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["thinking"] == {"type": "enabled"}
    assert "temperature" not in body
