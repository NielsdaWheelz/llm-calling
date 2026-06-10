"""DeepSeek chat completions client.

Reasoning invariant: DeepSeek reasoner `reasoning_content` must NOT be replayed on
continuation requests. This client never reads it (stream and non-stream parsing
consume only `content`/`tool_calls`), so assistant replays are stripped by
construction — keep it that way.
"""

import json
from collections.abc import AsyncIterator

import httpx

from llm_calling.errors import LLMError, LLMErrorCode, raise_for_provider_error
from llm_calling.types import LLMChunk, LLMRequest, LLMResponse, LLMUsage, ToolCall

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
        await raise_for_provider_error(response, "deepseek")

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
            await raise_for_provider_error(response, "deepseek")

            provider_request_id = response.headers.get("x-request-id")
            accumulated_usage: LLMUsage | None = None
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
                        try:
                            parsed_args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                        except json.JSONDecodeError:
                            parsed_args = {}
                        if not isinstance(parsed_args, dict):
                            parsed_args = {}
                        yield LLMChunk(
                            tool_call=ToolCall(
                                id=acc["id"], name=acc["name"], arguments=parsed_args
                            ),
                            done=False,
                        )
                    tool_call_acc.clear()
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

                if "usage" in data:
                    usage_data = data["usage"]
                    accumulated_usage = LLMUsage(
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
                        try:
                            parsed_args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                        except json.JSONDecodeError:
                            parsed_args = {}
                        if not isinstance(parsed_args, dict):
                            parsed_args = {}
                        yield LLMChunk(
                            tool_call=ToolCall(
                                id=acc["id"], name=acc["name"], arguments=parsed_args
                            ),
                            done=False,
                        )
                    tool_call_acc.clear()

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
                # Deliberately no reasoning_content here (see module docstring).
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
            "model": req.model_name,
            "messages": messages,
            "max_tokens": req.max_tokens,
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
                    },
                }
                for t in req.tools
            ]
            body["tool_choice"] = req.tool_choice

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

        message = choices[0].get("message", {}) or {}
        text = message.get("content") or ""

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_str = fn.get("arguments") or ""
            try:
                parsed_args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                parsed_args = {}
            if not isinstance(parsed_args, dict):
                parsed_args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id") or "", name=fn.get("name") or "", arguments=parsed_args)
            )

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
            tool_calls=tuple(tool_calls),
        )
