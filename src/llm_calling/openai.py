"""OpenAI Responses API client.

Endpoint:
- POST https://api.openai.com/v1/responses

Request body:
- reasoning is omitted when reasoning_effort="default"
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
- usage: usage.input_tokens / usage.output_tokens / usage.total_tokens /
  usage.output_tokens_details.reasoning_tokens
- status: status
- incomplete_details: incomplete_details
- provider_request_id: x-request-id header or response id
"""

import json
from collections.abc import AsyncIterator

import httpx

from llm_calling.errors import LLMError, LLMErrorCode
from llm_calling.types import LLMChunk, LLMRequest, LLMResponse, LLMUsage, ToolCall

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
        return self._parse_response(data, response.headers, structured=bool(req.structured_output))

    async def generate_stream(
        self,
        req: LLMRequest,
        *,
        api_key: str,
        timeout_s: int,
    ) -> AsyncIterator[LLMChunk]:
        if req.structured_output is not None:
            raise LLMError(
                LLMErrorCode.BAD_REQUEST,
                "OpenAI structured output streaming is not implemented",
                provider="openai",
            )
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
            tool_call_items: dict[str, dict] = {}

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
                            status="completed",
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

                if event_type == "response.output_item.added":
                    item = data.get("item") or {}
                    if item.get("type") == "function_call":
                        item_id = data.get("item_id") or item.get("id") or ""
                        tool_call_items[item_id] = {
                            "call_id": item.get("call_id") or "",
                            "name": item.get("name") or "",
                            "arguments": "",
                        }
                    continue

                if event_type == "response.function_call_arguments.delta":
                    item_id = data.get("item_id") or ""
                    if item_id in tool_call_items:
                        tool_call_items[item_id]["arguments"] += data.get("delta", "")
                    continue

                if event_type == "response.output_item.done":
                    item = data.get("item") or {}
                    if item.get("type") == "function_call":
                        item_id = data.get("item_id") or item.get("id") or ""
                        acc = tool_call_items.pop(item_id, None)
                        call_id = item.get("call_id") or (acc["call_id"] if acc else "")
                        name = item.get("name") or (acc["name"] if acc else "")
                        args_str = item.get("arguments")
                        if args_str is None:
                            args_str = acc["arguments"] if acc else ""
                        try:
                            parsed_args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            parsed_args = {}
                        if not isinstance(parsed_args, dict):
                            parsed_args = {}
                        yield LLMChunk(
                            tool_call=ToolCall(id=call_id, name=name, arguments=parsed_args),
                            done=False,
                        )
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

                    status = event_response.get("status") or (
                        "completed" if event_type == "response.completed" else "incomplete"
                    )
                    incomplete_details = event_response.get("incomplete_details")
                    usage_data = event_response.get("usage") or data.get("usage")
                    if usage_data:
                        accumulated_usage = self._parse_usage(usage_data)

                    if not emitted_terminal:
                        emitted_terminal = True
                        yield LLMChunk(
                            delta_text="",
                            done=True,
                            usage=accumulated_usage,
                            provider_request_id=provider_request_id,
                            status=status,
                            incomplete_details=incomplete_details,
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
        if req.prompt_cache_key is None and any(turn.cache_ttl != "none" for turn in req.messages):
            raise LLMError(
                LLMErrorCode.BAD_REQUEST,
                "OpenAI prompt cache turns require prompt_cache_key",
                provider="openai",
            )

        input_items: list[dict] = []
        for turn in req.messages:
            if turn.role == "tool":
                for tr in turn.tool_results:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tr.call_id,
                            "output": tr.output,
                        }
                    )
                continue
            if turn.content or not turn.tool_calls:
                input_items.append(
                    {
                        "role": turn.role,
                        "content": [{"type": "input_text", "text": turn.content}],
                    }
                )
            if turn.role == "assistant":
                for tc in turn.tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    )

        body: dict = {
            "model": req.model_name,
            "input": input_items,
            "max_output_tokens": req.max_tokens,
            "stream": stream,
        }

        if req.prompt_cache_key is not None:
            body["prompt_cache_key"] = req.prompt_cache_key
        if req.structured_output is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": req.structured_output.name,
                    "schema": req.structured_output.schema,
                    "strict": req.structured_output.strict,
                }
            }
        if req.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in req.tools
            ]
            body["tool_choice"] = req.tool_choice

        if req.reasoning_effort == "default":
            return body

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

    def _parse_response(
        self, data: dict, headers: httpx.Headers, *, structured: bool
    ) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in data.get("output", []):
            item_type = item.get("type")
            if item_type == "message":
                for content_item in item.get("content", []):
                    if content_item.get("type") == "output_text":
                        text_parts.append(content_item.get("text", ""))
            elif item_type == "function_call":
                args_str = item.get("arguments") or ""
                try:
                    parsed_args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    parsed_args = {}
                if not isinstance(parsed_args, dict):
                    parsed_args = {}
                tool_calls.append(
                    ToolCall(
                        id=item.get("call_id") or "",
                        name=item.get("name") or "",
                        arguments=parsed_args,
                    )
                )

        status = data.get("status")
        incomplete_details = data.get("incomplete_details")
        provider_request_id = headers.get("x-request-id") or data.get("id")
        text = "".join(text_parts)
        structured_output = None
        if structured and text.strip().startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                structured_output = parsed

        return LLMResponse(
            text=text,
            usage=self._parse_usage(data["usage"]) if data.get("usage") else None,
            provider_request_id=provider_request_id,
            status=status,
            incomplete_details=incomplete_details,
            structured_output=structured_output,
            tool_calls=tuple(tool_calls),
        )

    def _parse_usage(self, usage_data: dict) -> LLMUsage:
        output_tokens_details = usage_data.get("output_tokens_details") or {}
        input_tokens_details = usage_data.get("input_tokens_details") or {}
        cached_tokens = input_tokens_details.get("cached_tokens")
        return LLMUsage(
            input_tokens=usage_data.get("input_tokens"),
            output_tokens=usage_data.get("output_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            reasoning_tokens=output_tokens_details.get("reasoning_tokens"),
            cached_tokens=cached_tokens,
            cache_read_input_tokens=cached_tokens,
            provider_usage=dict(usage_data),
        )
