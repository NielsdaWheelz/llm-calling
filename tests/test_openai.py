import json
from pathlib import Path

import httpx
import pytest
import respx

from provider_runtime import DEFAULT_CATALOG, lower_generate_request
from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.openai import OpenAIClient
from provider_runtime.types import (
    ModelCall,
    ModelMessage,
    ModelRef,
    ProviderArtifact,
    ReasoningConfig,
    ReasoningEffort,
    StructuredOutputSpec,
    ToolCall,
    ToolResult,
    ToolSpec,
)

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures" / "openai"


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def request(reasoning_effort: ReasoningEffort = "none") -> ModelCall:
    return ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[
            ModelMessage(role="system", content="You are helpful."),
            ModelMessage(role="user", content="Hello!"),
        ],
        max_output_tokens=100,
        temperature=0.7,
        reasoning=ReasoningConfig(effort=reasoning_effort),
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
    req = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[
            ModelMessage(role="system", content="You are helpful.", cache_ttl="5m"),
            ModelMessage(role="user", content="Hello!"),
        ],
        max_output_tokens=100,
    )
    plan = lower_generate_request(
        req,
        DEFAULT_CATALOG.require_capabilities(req.model),
        streaming=False,
    )

    async with httpx.AsyncClient() as http:
        await OpenAIClient(http).generate(plan.call, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["prompt_cache_key"].startswith("pr-")


@respx.mock
async def test_structured_output_uses_json_schema_text_format_and_parses_response() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["output"][0]["content"][0]["text"] = '{"title":"The Book","language":"en"}'
    route = respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=response_json,
    )
    req = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[ModelMessage(role="user", content="Extract metadata.")],
        max_output_tokens=100,
        structured_output=StructuredOutputSpec(
            name="metadata_enrichment",
            schema=metadata_schema(),
        ),
    )

    async with httpx.AsyncClient() as http:
        response = await OpenAIClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "metadata_enrichment",
        "schema": metadata_schema(),
        "strict": True,
    }
    assert response.structured_output == {"title": "The Book", "language": "en"}


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
async def test_nonstream_tool_call_round_trip() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["output"] = [
        {
            "type": "function_call",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }
    ]
    route = respx.post("https://api.openai.com/v1/responses").respond(200, json=response_json)
    weather_tool = ToolSpec(
        name="get_weather",
        description="Get weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    req = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[
            ModelMessage(role="user", content="Weather in Paris?"),
            ModelMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(id="call_abc", name="get_weather", arguments={"city": "Paris"}),
                ),
            ),
            ModelMessage(
                role="tool", tool_results=(ToolResult(call_id="call_abc", output="sunny"),)
            ),
        ],
        max_output_tokens=100,
        tools=(weather_tool,),
        tool_choice="required",
    )

    async with httpx.AsyncClient() as http:
        response = await OpenAIClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": weather_tool.parameters,
            "strict": True,
        }
    ]
    assert body["tool_choice"] == "required"
    assert body["input"][1] == {
        "type": "function_call",
        "call_id": "call_abc",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }
    assert body["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": "sunny",
    }
    assert response.tool_calls == (
        ToolCall(id="call_abc", name="get_weather", arguments={"city": "Paris"}),
    )


@respx.mock
async def test_nonstream_malformed_tool_arguments_raise_typed_error() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["output"] = [
        {
            "type": "function_call",
            "call_id": "call_bad",
            "name": "get_weather",
            "arguments": "{not-json",
        }
    ]
    respx.post("https://api.openai.com/v1/responses").respond(200, json=response_json)

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await OpenAIClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert exc_info.value.error_code == ModelCallErrorCode.TOOL_ARGUMENTS_INVALID
    assert exc_info.value.retryable is False


@respx.mock
async def test_nonstream_repairable_tool_arguments_are_marked_repaired() -> None:
    response_json = load_json("success_nonstream.json")
    response_json["output"] = [
        {
            "type": "function_call",
            "call_id": "call_repaired",
            "name": "get_weather",
            "arguments": '{"city": "Paris",}',
        }
    ]
    respx.post("https://api.openai.com/v1/responses").respond(200, json=response_json)

    async with httpx.AsyncClient() as http:
        response = await OpenAIClient(http).generate(request(), api_key="sk-test", timeout_s=30)

    assert response.tool_calls == (
        ToolCall(
            id="call_repaired",
            name="get_weather",
            arguments={"city": "Paris"},
            argument_status="repaired",
        ),
    )


@respx.mock
async def test_stream_tool_call_yields_tool_call_chunk() -> None:
    respx.post("https://api.openai.com/v1/responses").respond(
        200,
        content=(
            'data: {"type":"response.output_item.added","item_id":"item_1",'
            '"item":{"type":"function_call","call_id":"call_xyz","name":"get_weather"}}\n\n'
            'data: {"type":"response.function_call_arguments.delta","item_id":"item_1",'
            '"delta":"{\\"city\\":"}\n\n'
            'data: {"type":"response.function_call_arguments.delta","item_id":"item_1",'
            '"delta":"\\"Berlin\\"}"}\n\n'
            'data: {"type":"response.output_item.done","item_id":"item_1",'
            '"item":{"type":"function_call","call_id":"call_xyz","name":"get_weather",'
            '"arguments":"{\\"city\\":\\"Berlin\\"}"}}\n\n'
            'data: {"type":"response.completed","response":{"id":"resp-tc",'
            '"status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        ),
        headers={"content-type": "text/event-stream"},
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in OpenAIClient(http).generate_stream(
                request(), api_key="sk-test", timeout_s=30
            )
        ]

    tool_chunks = [chunk for chunk in chunks if chunk.tool_call is not None]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].tool_call == ToolCall(
        id="call_xyz", name="get_weather", arguments={"city": "Berlin"}
    )
    assert chunks[-1].done is True
    assert chunks[-1].status == "completed"


@respx.mock
async def test_request_body_sets_store_false_and_includes_encrypted_reasoning() -> None:
    route = respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=load_json("success_nonstream.json"),
    )

    async with httpx.AsyncClient() as http:
        await OpenAIClient(http).generate(request("default"), api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]


@respx.mock
async def test_reasoning_item_replayed_before_function_call() -> None:
    route = respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=load_json("success_nonstream.json"),
    )
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "gAAAA-opaque",
    }
    artifact = ProviderArtifact(
        provider="openai",
        model="gpt-5.4-mini",
        purpose="reasoning",
        payload=reasoning_item,
    )
    req = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[
            ModelMessage(role="user", content="Weather in Paris?"),
            ModelMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="call_abc",
                        name="get_weather",
                        arguments={"city": "Paris"},
                        provider_metadata={"id": "fc_1"},
                    ),
                ),
                provider_artifacts=(artifact,),
            ),
            ModelMessage(
                role="tool", tool_results=(ToolResult(call_id="call_abc", output="sunny"),)
            ),
        ],
        max_output_tokens=100,
        reasoning=ReasoningConfig(effort="high"),
    )

    async with httpx.AsyncClient() as http:
        await OpenAIClient(http).generate(req, api_key="sk-test", timeout_s=30)

    body = json.loads(route.calls.last.request.content)
    assert body["input"][1] == reasoning_item
    assert body["input"][2] == {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_abc",
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }
    assert body["input"][3] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": "sunny",
    }


@respx.mock
async def test_rejects_mismatched_provider_artifact_before_request() -> None:
    route = respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=load_json("success_nonstream.json"),
    )
    req = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[
            ModelMessage(role="user", content="Weather?"),
            ModelMessage(
                role="assistant",
                provider_artifacts=(
                    ProviderArtifact(
                        provider="anthropic",
                        model="gpt-5.4-mini",
                        purpose="reasoning",
                        payload={"type": "reasoning", "encrypted_content": "opaque"},
                    ),
                ),
            ),
        ],
        max_output_tokens=100,
        reasoning=ReasoningConfig(effort="high"),
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as info:
            await OpenAIClient(http).generate(req, api_key="sk-test", timeout_s=30)

    assert info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert route.call_count == 0


@respx.mock
async def test_nonstream_reasoning_items_exposed_on_response() -> None:
    response_json = load_json("success_nonstream.json")
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "gAAAA-opaque",
    }
    response_json["output"] = [
        reasoning_item,
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        },
    ]
    respx.post("https://api.openai.com/v1/responses").respond(200, json=response_json)

    async with httpx.AsyncClient() as http:
        response = await OpenAIClient(http).generate(
            request("high"), api_key="sk-test", timeout_s=30
        )

    assert len(response.provider_artifacts) == 1
    artifact = response.provider_artifacts[0]
    assert artifact.provider == "openai"
    assert artifact.model == "gpt-5.4-mini"
    assert artifact.purpose == "reasoning"
    assert artifact.to_provider_payload() == reasoning_item
    assert response.tool_calls == (
        ToolCall(
            id="call_abc",
            name="get_weather",
            arguments={"city": "Paris"},
            provider_metadata={"id": "fc_1"},
        ),
    )


@respx.mock
async def test_stream_reasoning_item_yields_provider_artifact_chunk() -> None:
    respx.post("https://api.openai.com/v1/responses").respond(
        200,
        content=(
            'data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1",'
            '"summary":[],"encrypted_content":"gAAAA-opaque"}}\n\n'
            'data: {"type":"response.output_item.done","item":{"type":"function_call",'
            '"id":"fc_1","call_id":"call_xyz","name":"get_weather","arguments":"{}"}}\n\n'
            'data: {"type":"response.completed","response":{"id":"resp-rs",'
            '"status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        ),
        headers={"content-type": "text/event-stream"},
    )

    async with httpx.AsyncClient() as http:
        chunks = [
            chunk
            async for chunk in OpenAIClient(http).generate_stream(
                request("high"), api_key="sk-test", timeout_s=30
            )
        ]

    provider_artifacts = [
        chunk.provider_artifact for chunk in chunks if chunk.provider_artifact is not None
    ]
    assert [artifact.to_provider_payload() for artifact in provider_artifacts] == [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA-opaque"}
    ]
    tool_chunks = [chunk for chunk in chunks if chunk.tool_call is not None]
    assert tool_chunks[0].tool_call == ToolCall(
        id="call_xyz", name="get_weather", arguments={}, provider_metadata={"id": "fc_1"}
    )
    assert chunks[-1].done is True


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
