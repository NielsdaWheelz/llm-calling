"""Anthropic Messages API client.

Per PR-04 spec section 4.2:
- Endpoint: POST https://api.anthropic.com/v1/messages
- Headers: x-api-key: <key>, anthropic-version: 2023-06-01, Content-Type: application/json

ModelMessage conversion:
- System turn extracted to separate "system" field (Anthropic doesn't use system in messages array)
- Remaining turns mapped to messages with role preserved

Request body:
{
  "model": "<model_name>",
  "max_tokens": 1024,
  "temperature": 0.7,
  "system": "<system_prompt>",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

Response (non-stream):
{
  "id": "msg_...",
  "content": [{"type": "text", "text": "<output_text>"}],
  "usage": {
    "input_tokens": 100,
    "output_tokens": 50
  }
}

- text = concatenate all content[].text where type="text"
- usage.prompt_tokens = input_tokens
- usage.completion_tokens = output_tokens
- usage.total_tokens = sum
- provider_request_id = id

Streaming:
- Set "stream": true
- Events: event: content_block_delta with data: {"delta": {"text": "..."}}
- Terminal: event: message_stop
- Usage in event: message_delta at end
"""

import json
from collections.abc import AsyncIterator

import httpx

from provider_runtime.endpoints import ANTHROPIC_BASE_URL
from provider_runtime.errors import ModelCallError, ModelCallErrorCode, raise_for_provider_error
from provider_runtime.tool_arguments import parse_tool_arguments
from provider_runtime.types import (
    ModelCall,
    ModelChunk,
    ModelMessage,
    ModelResponse,
    TokenUsage,
    ToolCall,
)

ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_ADAPTIVE_THINKING_MODELS = {"claude-opus-4-7", "claude-sonnet-4-6"}


class AnthropicClient:
    def __init__(self, client: httpx.AsyncClient, *, base_url: str = ANTHROPIC_BASE_URL):
        self._client = client
        self._url = f"{base_url.rstrip('/')}/messages"

    async def generate(
        self,
        req: ModelCall,
        *,
        api_key: str,
        timeout_s: int,
    ) -> ModelResponse:
        """Non-streaming message generation."""
        headers = self._build_headers(api_key)
        body = self._build_request_body(req, stream=False)

        response = await self._client.post(
            self._url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        await raise_for_provider_error(response, "anthropic")

        data = response.json()
        return self._parse_response(data)

    async def generate_stream(
        self,
        req: ModelCall,
        *,
        api_key: str,
        timeout_s: int,
    ) -> AsyncIterator[ModelChunk]:
        """Streaming message generation using Server-Sent Events."""
        if req.structured_output is not None:
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                "Anthropic structured output streaming is not implemented",
                provider="anthropic",
            )
        headers = self._build_headers(api_key)
        body = self._build_request_body(req, stream=True)

        async with self._client.stream(
            "POST",
            self._url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        ) as response:
            await raise_for_provider_error(response, "anthropic")

            provider_request_id: str | None = None
            usage: TokenUsage | None = None
            received_stop = False
            usage_data: dict[str, object] = {}
            tool_blocks: dict[int, dict[str, object]] = {}
            thinking_blocks: dict[int, dict[str, object]] = {}

            async for line in response.aiter_lines():
                if not line:
                    continue

                # Anthropic SSE format: "event: <type>\ndata: {...}"
                if line.startswith("event: "):
                    event_type = line[7:]

                    if event_type == "message_stop":
                        received_stop = True
                        yield ModelChunk(
                            delta_text="",
                            done=True,
                            usage=usage,
                            provider_request_id=provider_request_id,
                        )
                        break
                    continue

                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type", "")

                # Handle message_start - extract request ID
                if event_type == "message_start":
                    message = data.get("message", {})
                    provider_request_id = message.get("id")
                    start_usage = message.get("usage")
                    if isinstance(start_usage, dict):
                        usage_data.update(start_usage)
                        usage = self._parse_usage(usage_data)
                    continue

                # Handle content_block_start - track tool_use and thinking blocks
                if event_type == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        index = data.get("index")
                        if isinstance(index, int):
                            tool_blocks[index] = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "json": "",
                            }
                    elif block.get("type") in ("thinking", "redacted_thinking"):
                        index = data.get("index")
                        if isinstance(index, int):
                            thinking_blocks[index] = dict(block)
                    continue

                # Handle content_block_delta - extract text, tool_use input json,
                # or thinking/signature deltas
                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        delta_text = delta.get("text", "")
                        if delta_text:
                            yield ModelChunk(delta_text=delta_text, done=False)
                    elif delta.get("type") == "input_json_delta":
                        index = data.get("index")
                        if isinstance(index, int) and index in tool_blocks:
                            tool_blocks[index]["json"] += delta.get("partial_json", "")
                    elif delta.get("type") in ("thinking_delta", "signature_delta"):
                        key = "thinking" if delta.get("type") == "thinking_delta" else "signature"
                        index = data.get("index")
                        if isinstance(index, int) and index in thinking_blocks:
                            block_state = thinking_blocks[index]
                            block_state[key] = str(block_state.get(key) or "") + str(
                                delta.get(key) or ""
                            )
                    continue

                # Handle content_block_stop - finalize tool_use and thinking blocks
                if event_type == "content_block_stop":
                    index = data.get("index")
                    if isinstance(index, int) and index in tool_blocks:
                        block_state = tool_blocks.pop(index)
                        raw_json = str(block_state["json"] or "{}")
                        arguments = parse_tool_arguments(
                            raw_json,
                            provider="anthropic",
                            tool_name=str(block_state["name"]),
                            call_id=str(block_state["id"]),
                        )
                        yield ModelChunk(
                            tool_call=ToolCall(
                                id=str(block_state["id"]),
                                name=str(block_state["name"]),
                                arguments=arguments,
                            ),
                            done=False,
                        )
                    elif isinstance(index, int) and index in thinking_blocks:
                        # One complete thinking/redacted_thinking block, verbatim.
                        yield ModelChunk(provider_artifact=thinking_blocks.pop(index), done=False)
                    continue

                # Handle message_delta - extract usage at end
                if event_type == "message_delta":
                    delta_usage = data.get("usage")
                    if isinstance(delta_usage, dict):
                        usage_data.update(delta_usage)
                        usage = self._parse_usage(usage_data)
                    continue

            if not received_stop:
                raise ModelCallError(
                    ModelCallErrorCode.PROVIDER_DOWN,
                    "Anthropic stream ended without message_stop event",
                    provider="anthropic",
                    retryable=False,
                )

    def _build_headers(self, api_key: str) -> dict[str, str]:
        """Build request headers."""
        return {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
        }

    def _build_request_body(self, req: ModelCall, stream: bool) -> dict:
        """Build request body from ModelCall.

        Extracts system turn to separate field.
        """
        if req.prompt_cache_key is not None:
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                "Anthropic does not support prompt_cache_key",
                provider="anthropic",
            )
        if req.structured_output is not None and req.reasoning.effort not in (
            "default",
            "none",
        ):
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                "Anthropic forced structured output is incompatible with extended thinking",
                provider="anthropic",
            )

        # Extract system prompt and non-system messages
        system_blocks = []
        messages = []

        for turn in req.messages:
            if turn.role == "system":
                # Anthropic uses a separate system field
                system_blocks.append(self._turn_to_text_block(turn))
            else:
                messages.append(self._turn_to_message(turn))

        body: dict = {
            "model": req.model.model,
            "max_tokens": req.max_output_tokens,
            "messages": messages,
            "stream": stream,
        }

        if system_blocks:
            body["system"] = system_blocks
        tools_payload: list[dict[str, object]] = []
        if req.structured_output is not None:
            tools_payload.append(
                {
                    "name": req.structured_output.name,
                    "description": f"Return {req.structured_output.name}.",
                    "input_schema": req.structured_output.schema,
                }
            )
            body["tool_choice"] = {"type": "tool", "name": req.structured_output.name}
        for tool in req.tools:
            tools_payload.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
            )
        if tools_payload:
            body["tools"] = tools_payload
        if req.tools and req.structured_output is None:
            if req.tool_choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif req.tool_choice == "none":
                body["tool_choice"] = {"type": "none"}
            elif req.tool_choice == "required":
                body["tool_choice"] = {"type": "any"}

        uses_adaptive_thinking = req.model.model in ANTHROPIC_ADAPTIVE_THINKING_MODELS and (
            req.reasoning.effort not in ("default", "none")
        )
        if req.temperature is not None and not uses_adaptive_thinking:
            body["temperature"] = req.temperature

        if req.reasoning.effort == "default":
            return body

        if req.reasoning.effort == "none":
            body["thinking"] = {"type": "disabled"}
            return body

        if req.model.model in ANTHROPIC_ADAPTIVE_THINKING_MODELS:
            if req.reasoning.effort in ("minimal", "low"):
                effort = "low"
            elif req.reasoning.effort == "medium":
                effort = "medium"
            elif req.reasoning.effort == "high":
                effort = "high"
            elif req.reasoning.effort == "max":
                effort = "xhigh"
            else:
                raise ValueError(f"Unknown reasoning_effort: {req.reasoning.effort}")

            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}
            return body

        if req.reasoning.effort == "minimal":
            budget_tokens = 1024
        elif req.reasoning.effort == "low":
            budget_tokens = 1536
        elif req.reasoning.effort == "medium":
            budget_tokens = 2048
        elif req.reasoning.effort == "high":
            budget_tokens = 3072
        elif req.reasoning.effort == "max":
            budget_tokens = 4000
        else:
            raise ValueError(f"Unknown reasoning_effort: {req.reasoning.effort}")

        max_allowed_budget = req.max_output_tokens - 1
        if max_allowed_budget < 1024:
            body["thinking"] = {"type": "disabled"}
        else:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(budget_tokens, max_allowed_budget),
            }

        return body

    def _turn_to_text_block(self, turn: ModelMessage) -> dict[str, object]:
        block: dict[str, object] = {"type": "text", "text": turn.content}
        if turn.cache_ttl == "none":
            return block
        if turn.cache_ttl not in ("5m", "1h"):
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                f"Unknown prompt cache ttl: {turn.cache_ttl}",
                provider="anthropic",
            )
        block["cache_control"] = {"type": "ephemeral", "ttl": turn.cache_ttl}
        return block

    def _turn_to_message(self, turn: ModelMessage) -> dict[str, object]:
        """Convert ModelMessage to Anthropic message format.

        Note: System turns are handled separately in _build_request_body.
        """
        if turn.cache_ttl != "none":
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                "Anthropic prompt cache is only supported on system turns",
                provider="anthropic",
            )
        if turn.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.output,
                        "is_error": result.is_error,
                    }
                    for result in turn.tool_results
                ],
            }
        if turn.role == "assistant" and (turn.tool_calls or turn.provider_artifacts):
            # Thinking blocks must lead the assistant turn, unmodified, before tool_use.
            content: list[dict[str, object]] = [dict(item) for item in turn.provider_artifacts]
            if turn.content:
                content.append({"type": "text", "text": turn.content})
            for call in turn.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            return {"role": "assistant", "content": content}
        return {
            "role": turn.role,
            "content": turn.content,
        }

    def _parse_response(self, data: dict) -> ModelResponse:
        """Parse non-streaming response."""
        # Extract text from content blocks
        content_blocks = data.get("content", [])
        text_parts = []
        structured_output = None
        tool_calls: list[ToolCall] = []
        provider_artifacts: list[dict[str, object]] = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") in ("thinking", "redacted_thinking"):
                provider_artifacts.append(block)
            elif block.get("type") == "tool_use":
                arguments = parse_tool_arguments(
                    block.get("input"),
                    provider="anthropic",
                    tool_name=str(block.get("name", "")),
                    call_id=str(block.get("id", "")),
                )
                if structured_output is None:
                    structured_output = arguments
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=arguments,
                    )
                )
        text = "".join(text_parts)

        # Extract usage - Anthropic uses input_tokens/output_tokens
        usage = None
        usage_data = data.get("usage")
        if usage_data:
            usage = self._parse_usage(usage_data)

        # Extract request ID from body
        provider_request_id = data.get("id")

        return ModelResponse(
            text=text,
            usage=usage,
            provider_request_id=provider_request_id,
            structured_output=structured_output,
            tool_calls=tuple(tool_calls),
            provider_artifacts=tuple(provider_artifacts),
        )

    def _parse_usage(self, usage_data: dict[str, object]) -> TokenUsage:
        input_tokens = _int_or_none(usage_data.get("input_tokens"))
        output_tokens = _int_or_none(usage_data.get("output_tokens"))
        cache_creation = _int_or_none(usage_data.get("cache_creation_input_tokens"))
        cache_read = _int_or_none(usage_data.get("cache_read_input_tokens"))
        total = sum(
            value
            for value in (input_tokens, output_tokens, cache_creation, cache_read)
            if value is not None
        )
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            provider_usage=dict(usage_data),
        )


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None
