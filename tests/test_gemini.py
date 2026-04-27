import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_calling.gemini import GeminiClient
from llm_calling.types import LLMRequest, Turn

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


@respx.mock
async def test_nonstream_success() -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    ).respond(200, json=load_json("success_nonstream.json"))

    async with httpx.AsyncClient() as http:
        response = await GeminiClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert response.text == "Hello! How can I help you today?"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 8
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
