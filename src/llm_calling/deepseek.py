"""DeepSeek chat completions client."""

import json
from collections.abc import AsyncIterator

import httpx

from llm_calling.errors import LLMError, LLMErrorCode
from llm_calling.types import LLMChunk, LLMRequest, LLMResponse, LLMUsage

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_V4_MODELS = {"deepseek-v4-pro", "deepseek-v4-flash"}


class DeepSeekClient:
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
            DEEPSEEK_CHAT_URL,
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
            DEEPSEEK_CHAT_URL,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        ) as response:
            response.raise_for_status()

            provider_request_id = response.headers.get("x-request-id")
            accumulated_usage: LLMUsage | None = None
            received_done = False

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    received_done = True
                    yield LLMChunk(
                        delta_text="",
                        done=True,
                        usage=accumulated_usage,
                        provider_request_id=provider_request_id,
                    )
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    if "usage" in data:
                        usage_data = data["usage"]
                        accumulated_usage = LLMUsage(
                            input_tokens=usage_data.get("prompt_tokens"),
                            output_tokens=usage_data.get("completion_tokens"),
                            total_tokens=usage_data.get("total_tokens"),
                            provider_usage=dict(usage_data),
                        )
                    continue

                delta = choices[0].get("delta", {})
                delta_text = delta.get("content", "")

                if "usage" in data:
                    usage_data = data["usage"]
                    accumulated_usage = LLMUsage(
                        input_tokens=usage_data.get("prompt_tokens"),
                        output_tokens=usage_data.get("completion_tokens"),
                        total_tokens=usage_data.get("total_tokens"),
                        provider_usage=dict(usage_data),
                    )

                if delta_text:
                    yield LLMChunk(delta_text=delta_text, done=False)

            if not received_done:
                raise LLMError(
                    LLMErrorCode.PROVIDER_DOWN,
                    "deepseek stream ended without [DONE] marker",
                    provider="deepseek",
                )

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_body(self, req: LLMRequest, stream: bool) -> dict:
        if req.structured_output is not None:
            raise LLMError(
                LLMErrorCode.BAD_REQUEST,
                "DeepSeek structured output is not implemented",
                provider="deepseek",
            )
        if req.prompt_cache_key is not None or any(
            turn.cache_ttl != "none" for turn in req.messages
        ):
            raise LLMError(
                LLMErrorCode.BAD_REQUEST,
                "DeepSeek does not support required prompt caching",
                provider="deepseek",
            )

        body: dict = {
            "model": req.model_name,
            "messages": [
                {
                    "role": turn.role,
                    "content": turn.content,
                }
                for turn in req.messages
            ],
            "max_tokens": req.max_tokens,
            "stream": stream,
        }

        uses_v4_thinking = req.model_name in DEEPSEEK_V4_MODELS and (
            req.reasoning_effort not in ("default", "none")
        )
        if req.temperature is not None and not uses_v4_thinking:
            body["temperature"] = req.temperature

        if req.model_name in DEEPSEEK_V4_MODELS and req.reasoning_effort != "default":
            body["thinking"] = {"type": "disabled" if req.reasoning_effort == "none" else "enabled"}

        return body

    def _parse_response(self, data: dict, headers: httpx.Headers) -> LLMResponse:
        choices = data.get("choices", [])
        if not choices:
            raise LLMError(
                LLMErrorCode.PROVIDER_DOWN,
                "deepseek response missing choices",
                provider="deepseek",
            )

        text = choices[0].get("message", {}).get("content", "")

        usage = None
        usage_data = data.get("usage")
        if usage_data:
            usage = LLMUsage(
                input_tokens=usage_data.get("prompt_tokens"),
                output_tokens=usage_data.get("completion_tokens"),
                total_tokens=usage_data.get("total_tokens"),
                provider_usage=dict(usage_data),
            )

        provider_request_id = headers.get("x-request-id") or data.get("id")

        return LLMResponse(
            text=text,
            usage=usage,
            provider_request_id=provider_request_id,
        )
