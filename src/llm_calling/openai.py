"""OpenAI Responses API client.

Endpoint:
- POST https://api.openai.com/v1/responses

Request body:
{
  "model": "<model_name>",
  "input": [
    {
      "role": "system" | "user" | "assistant",
      "content": [{"type": "input_text", "text": "..."}]
    }
  ],
  "max_output_tokens": 4096,
  "reasoning": {"effort": "none" | "minimal" | "low" | "medium" | "high" | "xhigh"},
  "stream": false
}

Response (non-stream):
- text: extracted from output[*].content[*].text where type=output_text
- usage: usage.input_tokens / usage.output_tokens / usage.total_tokens
- provider_request_id: x-request-id header or response id
"""

import json
from collections.abc import AsyncIterator

import httpx

from llm_calling.errors import LLMError, LLMErrorCode
from llm_calling.types import LLMChunk, LLMRequest, LLMResponse, LLMUsage

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def generate(
        self,
        req: LLMRequest,
        *,
        api_key: str,
        timeout_s: int,
    ) -> LLMResponse:
        headers = self._build_headers(api_key)
        body = self._build_request_body(req, stream=False)

        response = await self._client.post(
            OPENAI_RESPONSES_URL,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        response.raise_for_status()

        data = response.json()
        return self._parse_response(data, response.headers)

    async def generate_stream(
        self,
        req: LLMRequest,
        *,
        api_key: str,
        timeout_s: int,
    ) -> AsyncIterator[LLMChunk]:
        headers = self._build_headers(api_key)
        body = self._build_request_body(req, stream=True)

        async with self._client.stream(
            "POST",
            OPENAI_RESPONSES_URL,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        ) as response:
            response.raise_for_status()

            provider_request_id = response.headers.get("x-request-id")
            accumulated_usage: LLMUsage | None = None
            emitted_terminal = False

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    if not emitted_terminal:
                        yield LLMChunk(
                            delta_text="",
                            done=True,
                            usage=accumulated_usage,
                            provider_request_id=provider_request_id,
                        )
                    emitted_terminal = True
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type")

                if event_type == "response.output_text.delta":
                    delta_text = data.get("delta", "")
                    if delta_text:
                        yield LLMChunk(delta_text=delta_text, done=False)
                    continue

                if event_type == "response.created":
                    event_response = data.get("response") or {}
                    if provider_request_id is None:
                        provider_request_id = event_response.get("id")
                    continue

                if event_type in ("response.completed", "response.incomplete"):
                    event_response = data.get("response") or {}
                    if provider_request_id is None:
                        provider_request_id = event_response.get("id")

                    usage_data = event_response.get("usage") or data.get("usage")
                    if usage_data:
                        accumulated_usage = LLMUsage(
                            prompt_tokens=usage_data.get("input_tokens"),
                            completion_tokens=usage_data.get("output_tokens"),
                            total_tokens=usage_data.get("total_tokens"),
                        )

                    if not emitted_terminal:
                        emitted_terminal = True
                        yield LLMChunk(
                            delta_text="",
                            done=True,
                            usage=accumulated_usage,
                            provider_request_id=provider_request_id,
                        )
                    break

            if not emitted_terminal:
                raise LLMError(
                    LLMErrorCode.PROVIDER_DOWN,
                    "openai stream ended without terminal event",
                    provider="openai",
                )

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_body(self, req: LLMRequest, stream: bool) -> dict:
        body: dict = {
            "model": req.model_name,
            "input": [
                {
                    "role": turn.role,
                    "content": [{"type": "input_text", "text": turn.content}],
                }
                for turn in req.messages
            ],
            "max_output_tokens": req.max_tokens,
            "stream": stream,
        }

        if req.reasoning_effort == "none":
            body["reasoning"] = {"effort": "none"}
        elif req.reasoning_effort == "minimal":
            body["reasoning"] = {"effort": "minimal"}
        elif req.reasoning_effort == "low":
            body["reasoning"] = {"effort": "low"}
        elif req.reasoning_effort == "medium":
            body["reasoning"] = {"effort": "medium"}
        elif req.reasoning_effort == "high":
            body["reasoning"] = {"effort": "high"}
        elif req.reasoning_effort == "max":
            body["reasoning"] = {"effort": "xhigh"}
        else:
            raise ValueError(f"Unknown reasoning_effort: {req.reasoning_effort}")

        return body

    def _parse_response(self, data: dict, headers: httpx.Headers) -> LLMResponse:
        text_parts: list[str] = []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    text_parts.append(content_item.get("text", ""))

        usage = None
        usage_data = data.get("usage")
        if usage_data:
            usage = LLMUsage(
                prompt_tokens=usage_data.get("input_tokens"),
                completion_tokens=usage_data.get("output_tokens"),
                total_tokens=usage_data.get("total_tokens"),
            )

        provider_request_id = headers.get("x-request-id") or data.get("id")

        return LLMResponse(
            text="".join(text_parts),
            usage=usage,
            provider_request_id=provider_request_id,
        )
