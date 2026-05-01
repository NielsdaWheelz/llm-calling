import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_calling.openai import OpenAIClient
from llm_calling.types import LLMRequest, ReasoningEffort, Turn

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures" / "openai"


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def request(reasoning_effort: ReasoningEffort = "none") -> LLMRequest:
    return LLMRequest(
        model_name="gpt-5.4-mini",
        messages=[
            Turn(role="system", content="You are helpful."),
            Turn(role="user", content="Hello!"),
        ],
        max_tokens=100,
        temperature=0.7,
        reasoning_effort=reasoning_effort,
    )


@respx.mock
async def test_nonstream_success() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["incomplete_details"] = None
    response_json["usage"]["output_tokens_details"] = {"reasoning_tokens": 3}
    respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=response_json,
        headers={"x-request-id": "req-test-123"},
    )

    async with httpx.AsyncClient() as http:
        response = await OpenAIClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert response.text == "Hello! How can I help you today?"
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 8
    assert response.usage.total_tokens == 18
    assert response.usage.reasoning_tokens == 3
    assert response.provider_request_id == "req-test-123"
    assert response.status == "completed"
    assert response.incomplete_details is None


@respx.mock
async def test_stream_success() -> None:
    respx.post("https://api.openai.com/v1/responses").respond(
        200,
        content=load_text("success_stream_chunks.txt"),
        headers={"x-request-id": "req-test-123", "content-type": "text/event-stream"},
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in OpenAIClient(http).generate_stream(
                request(), api_key="sk-test", timeout_s=30
            )
        ]

    assert chunks[-1].done is True
    assert chunks[-1].provider_request_id == "req-test-123"
    assert chunks[-1].status == "completed"
    assert all(chunk.usage is None for chunk in chunks[:-1])
    assert "Hello" in "".join(chunk.delta_text for chunk in chunks)


@respx.mock
async def test_payload_omits_default_reasoning() -> None:
    route = respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=load_json("success_nonstream.json"),
    )

    async with httpx.AsyncClient() as http:
        response = await OpenAIClient(http).generate(
            request("default"), api_key="sk-test", timeout_s=30
        )

    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert "reasoning" not in body
    assert response.provider_request_id == "resp-test123"


@respx.mock
async def test_payload_includes_prompt_cache_key() -> None:
    route = respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=load_json("success_nonstream.json"),
    )
    req = LLMRequest(
        model_name="gpt-5.4-mini",
        messages=[
            Turn(role="system", content="You are helpful.", cache_ttl="5m"),
            Turn(role="user", content="Hello!"),
        ],
        max_tokens=100,
        prompt_cache_key="scope:abc123",
    )

    async with httpx.AsyncClient() as http:
        await OpenAIClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["prompt_cache_key"] == "scope:abc123"


@respx.mock
async def test_gpt5_payload_omits_temperature_and_maps_max_reasoning() -> None:
    route = respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=load_json("success_nonstream.json"),
    )

    async with httpx.AsyncClient() as http:
        await OpenAIClient(http).generate(request("max"), api_key="sk-test", timeout_s=30)

    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body["reasoning"] == {"effort": "xhigh"}
    assert "temperature" not in body


@respx.mock
async def test_nonstream_incomplete_preserves_status_and_usage() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["status"] = "incomplete"
    response_json["incomplete_details"] = {"reason": "max_output_tokens"}
    response_json["usage"]["output_tokens_details"] = {"reasoning_tokens": 11}
    respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=response_json,
        headers={"x-request-id": "req-incomplete-123"},
    )

    async with httpx.AsyncClient() as http:
        response = await OpenAIClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert response.status == "incomplete"
    assert response.incomplete_details == {"reason": "max_output_tokens"}
    assert response.provider_request_id == "req-incomplete-123"
    assert response.usage is not None
    assert response.usage.reasoning_tokens == 11


@respx.mock
async def test_stream_incomplete_yields_terminal_chunk() -> None:
    respx.post("https://api.openai.com/v1/responses").respond(
        200,
        content=(
            'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
            'data: {"type":"response.incomplete","response":{'
            '"id":"resp-incomplete",'
            '"status":"incomplete",'
            '"incomplete_details":{"reason":"max_output_tokens"},'
            '"usage":{"input_tokens":10,"output_tokens":8,'
            '"output_tokens_details":{"reasoning_tokens":4},"total_tokens":18}'
            "}}\n\n"
        ),
        headers={"content-type": "text/event-stream"},
    )

    chunks = []
    async with httpx.AsyncClient() as http:
        async for chunk in OpenAIClient(http).generate_stream(
            request(), api_key="sk-test", timeout_s=30
        ):
            chunks.append(chunk)

    assert [chunk.done for chunk in chunks] == [False, True]
    assert chunks[-1].status == "incomplete"
    assert chunks[-1].incomplete_details == {"reason": "max_output_tokens"}
    assert chunks[-1].provider_request_id == "resp-incomplete"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.reasoning_tokens == 4
