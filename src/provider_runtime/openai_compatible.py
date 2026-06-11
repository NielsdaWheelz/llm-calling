"""OpenAI-compatible chat completions client."""

import json
from collections.abc import AsyncIterator
from typing import Literal

import httpx

from provider_runtime.errors import ModelCallError, ModelCallErrorCode, raise_for_provider_error
from provider_runtime.tool_arguments import parse_tool_arguments
from provider_runtime.types import ModelCall, ModelChunk, ModelResponse, TokenUsage, ToolCall

OpenAICompatibleProvider = Literal["openrouter", "cloudflare"]


class OpenAICompatibleChatClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        provider: OpenAICompatibleProvider,
        base_url: str,
    ):
        self._client = client
        self._provider = provider
        self._url = f"{base_url.rstrip('/')}/chat/completions"

    async def generate(
        self,
        req: ModelCall,
        *,
        api_key: str,
        timeout_s: int,
    ) -> ModelResponse:
        headers = self._build_headers(api_key)
        body = self._build_request_body(req, stream=False)

        response = await self._client.post(
            self._url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        await raise_for_provider_error(response, self._provider)

        data = response.json()
        return self._parse_response(data, response.headers)

    async def generate_stream(
        self,
        req: ModelCall,
        *,
        api_key: str,
        timeout_s: int,
    ) -> AsyncIterator[ModelChunk]:
        headers = self._build_headers(api_key)
        body = self._build_request_body(req, stream=True)

        async with self._client.stream(
            "POST",
            self._url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        ) as response:
            await raise_for_provider_error(response, self._provider)

            provider_request_id = response.headers.get("x-request-id")
            accumulated_usage: TokenUsage | None = None
            received_done = False
            tool_call_acc: dict[int, dict] = {}

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    received_done = True
                    for idx in sorted(tool_call_acc):
                        acc = tool_call_acc[idx]
                        parsed_args = parse_tool_arguments(
                            acc["arguments"],
                            provider=self._provider,
                            tool_name=acc["name"],
                            call_id=acc["id"],
                        )
                        yield ModelChunk(
                            tool_call=ToolCall(
                                id=acc["id"], name=acc["name"], arguments=parsed_args
                            ),
                            done=False,
                        )
                    tool_call_acc.clear()
                    yield ModelChunk(
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

                if "usage" in data:
                    usage_data = data["usage"]
                    accumulated_usage = TokenUsage(
                        input_tokens=usage_data.get("prompt_tokens"),
                        output_tokens=usage_data.get("completion_tokens"),
                        total_tokens=usage_data.get("total_tokens"),
                        provider_usage=dict(usage_data),
                    )

                choices = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                delta_text = delta.get("content", "")

                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    acc = tool_call_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc_delta.get("id"):
                        acc["id"] = tc_delta["id"]
                    fn = tc_delta.get("function") or {}
                    if fn.get("name"):
                        acc["name"] = fn["name"]
                    if "arguments" in fn and fn["arguments"]:
                        acc["arguments"] += fn["arguments"]

                if choice.get("finish_reason") == "tool_calls" and tool_call_acc:
                    for idx in sorted(tool_call_acc):
                        acc = tool_call_acc[idx]
                        parsed_args = parse_tool_arguments(
                            acc["arguments"],
                            provider=self._provider,
                            tool_name=acc["name"],
                            call_id=acc["id"],
                        )
                        yield ModelChunk(
                            tool_call=ToolCall(
                                id=acc["id"], name=acc["name"], arguments=parsed_args
                            ),
                            done=False,
                        )
                    tool_call_acc.clear()

                if delta_text:
                    yield ModelChunk(delta_text=delta_text, done=False)

            if not received_done:
                raise ModelCallError(
                    ModelCallErrorCode.PROVIDER_DOWN,
                    f"{self._provider} stream ended without [DONE] marker",
                    provider=self._provider,
                    retryable=False,
                )

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_body(self, req: ModelCall, stream: bool) -> dict:
        messages: list[dict] = []
        for turn in req.messages:
            if turn.role == "tool":
                for tr in turn.tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.call_id,
                            "content": tr.output,
                        }
                    )
                continue
            if turn.role == "assistant" and turn.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": turn.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in turn.tool_calls
                        ],
                    }
                )
                continue
            messages.append({"role": turn.role, "content": turn.content})

        body: dict = {
            "model": req.model.model,
            "messages": messages,
            "max_tokens": req.max_output_tokens,
            "stream": stream,
        }

        if req.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "strict": t.strict,
                    },
                }
                for t in req.tools
            ]
            body["tool_choice"] = req.tool_choice

        if req.temperature is not None:
            body["temperature"] = req.temperature

        if req.structured_output is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": req.structured_output.name,
                    "schema": req.structured_output.schema,
                    "strict": req.structured_output.strict,
                },
            }

        if self._provider == "openrouter":
            if req.reasoning.effort not in ("default", "none"):
                effort = "high" if req.reasoning.effort == "max" else req.reasoning.effort
                body["reasoning"] = {"effort": effort}
        elif req.reasoning.effort not in ("default", "none"):
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                f"{self._provider} reasoning controls are not implemented",
                provider=self._provider,
            )

        return body

    def _parse_response(self, data: dict, headers: httpx.Headers) -> ModelResponse:
        choices = data.get("choices", [])
        if not choices:
            raise ModelCallError(
                ModelCallErrorCode.PROVIDER_DOWN,
                f"{self._provider} response missing choices",
                provider=self._provider,
                retryable=False,
            )

        message = choices[0].get("message", {}) or {}
        text = message.get("content") or ""
        structured_output = None
        if req_structured_text := text.strip():
            if req_structured_text.startswith("{"):
                try:
                    parsed = json.loads(req_structured_text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    structured_output = parsed

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_str = fn.get("arguments") or ""
            parsed_args = parse_tool_arguments(
                args_str,
                provider=self._provider,
                tool_name=fn.get("name") or "",
                call_id=tc.get("id") or "",
            )
            tool_calls.append(
                ToolCall(id=tc.get("id") or "", name=fn.get("name") or "", arguments=parsed_args)
            )

        usage = None
        usage_data = data.get("usage")
        if usage_data:
            usage = TokenUsage(
                input_tokens=usage_data.get("prompt_tokens"),
                output_tokens=usage_data.get("completion_tokens"),
                total_tokens=usage_data.get("total_tokens"),
                provider_usage=dict(usage_data),
            )

        provider_request_id = headers.get("x-request-id") or data.get("id")

        return ModelResponse(
            text=text,
            usage=usage,
            provider_request_id=provider_request_id,
            structured_output=structured_output,
            tool_calls=tuple(tool_calls),
        )
