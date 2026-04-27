"""Anthropic Messages API client.

Per PR-04 spec section 4.2:
- Endpoint: POST https://api.anthropic.com/v1/messages
- Headers: x-api-key: <key>, anthropic-version: 2023-06-01, Content-Type: application/json

Turn conversion:
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

from llm_calling.errors import LLMError, LLMErrorCode
from llm_calling.types import LLMChunk, LLMRequest, LLMResponse, LLMUsage, Turn

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_ADAPTIVE_THINKING_MODELS = {"claude-opus-4-7", "claude-sonnet-4-6"}


class AnthropicClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def generate(
        self,
        req: LLMRequest,
        *,
        api_key: str,
        timeout_s: int,
    ) -> LLMResponse:
        """Non-streaming message generation."""
        headers = self._build_headers(api_key)
        body = self._build_request_body(req, stream=False)

        response = await self._client.post(
            ANTHROPIC_MESSAGES_URL,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        response.raise_for_status()

        data = response.json()
        return self._parse_response(data)

    async def generate_stream(
        self,
        req: LLMRequest,
        *,
        api_key: str,
        timeout_s: int,
    ) -> AsyncIterator[LLMChunk]:
        """Streaming message generation using Server-Sent Events."""
        headers = self._build_headers(api_key)
        body = self._build_request_body(req, stream=True)

        async with self._client.stream(
            "POST",
            ANTHROPIC_MESSAGES_URL,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        ) as response:
            response.raise_for_status()

            provider_request_id: str | None = None
            usage: LLMUsage | None = None
            received_stop = False

            async for line in response.aiter_lines():
                if not line:
                    continue

                # Anthropic SSE format: "event: <type>\ndata: {...}"
                if line.startswith("event: "):
                    event_type = line[7:]

                    if event_type == "message_stop":
                        received_stop = True
                        yield LLMChunk(
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
                    continue

                # Handle content_block_delta - extract text
                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        delta_text = delta.get("text", "")
                        if delta_text:
                            yield LLMChunk(delta_text=delta_text, done=False)
                    continue

                # Handle message_delta - extract usage at end
                if event_type == "message_delta":
                    usage_data = data.get("usage", {})
                    if usage_data:
                        # Anthropic provides output_tokens in message_delta
                        usage = LLMUsage(
                            prompt_tokens=None,  # Not in delta
                            completion_tokens=usage_data.get("output_tokens"),
                            total_tokens=None,  # Compute later if needed
                        )
                    continue

            if not received_stop:
                raise LLMError(
                    LLMErrorCode.PROVIDER_DOWN,
                    "Anthropic stream ended without message_stop event",
                    provider="anthropic",
                )

    def _build_headers(self, api_key: str) -> dict[str, str]:
        """Build request headers."""
        return {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
        }

    def _build_request_body(self, req: LLMRequest, stream: bool) -> dict:
        """Build request body from LLMRequest.

        Extracts system turn to separate field.
        """
        # Extract system prompt and non-system messages
        system_prompt = None
        messages = []

        for turn in req.messages:
            if turn.role == "system":
                # Anthropic uses a separate system field
                system_prompt = turn.content
            else:
                messages.append(self._turn_to_message(turn))

        body: dict = {
            "model": req.model_name,
            "max_tokens": req.max_tokens,
            "messages": messages,
            "stream": stream,
        }

        if system_prompt:
            body["system"] = system_prompt

        uses_adaptive_thinking = (
            req.model_name in ANTHROPIC_ADAPTIVE_THINKING_MODELS and req.reasoning_effort != "none"
        )
        if req.temperature is not None and not uses_adaptive_thinking:
            body["temperature"] = req.temperature

        if req.reasoning_effort == "none":
            body["thinking"] = {"type": "disabled"}
            return body

        if req.model_name in ANTHROPIC_ADAPTIVE_THINKING_MODELS:
            if req.reasoning_effort in ("minimal", "low"):
                effort = "low"
            elif req.reasoning_effort == "medium":
                effort = "medium"
            elif req.reasoning_effort == "high":
                effort = "high"
            elif req.reasoning_effort == "max":
                effort = "xhigh"
            else:
                raise ValueError(f"Unknown reasoning_effort: {req.reasoning_effort}")

            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}
            return body

        if req.reasoning_effort == "minimal":
            budget_tokens = 1024
        elif req.reasoning_effort == "low":
            budget_tokens = 1536
        elif req.reasoning_effort == "medium":
            budget_tokens = 2048
        elif req.reasoning_effort == "high":
            budget_tokens = 3072
        elif req.reasoning_effort == "max":
            budget_tokens = 4000
        else:
            raise ValueError(f"Unknown reasoning_effort: {req.reasoning_effort}")

        max_allowed_budget = req.max_tokens - 1
        if max_allowed_budget < 1024:
            body["thinking"] = {"type": "disabled"}
        else:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(budget_tokens, max_allowed_budget),
            }

        return body

    def _turn_to_message(self, turn: Turn) -> dict[str, str]:
        """Convert Turn to Anthropic message format.

        Note: System turns are handled separately in _build_request_body.
        """
        return {
            "role": turn.role,
            "content": turn.content,
        }

    def _parse_response(self, data: dict) -> LLMResponse:
        """Parse non-streaming response."""
        # Extract text from content blocks
        content_blocks = data.get("content", [])
        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        text = "".join(text_parts)

        # Extract usage - Anthropic uses input_tokens/output_tokens
        usage = None
        usage_data = data.get("usage")
        if usage_data:
            input_tokens = usage_data.get("input_tokens")
            output_tokens = usage_data.get("output_tokens")
            total = None
            if input_tokens is not None and output_tokens is not None:
                total = input_tokens + output_tokens

            usage = LLMUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total,
            )

        # Extract request ID from body
        provider_request_id = data.get("id")

        return LLMResponse(
            text=text,
            usage=usage,
            provider_request_id=provider_request_id,
        )
