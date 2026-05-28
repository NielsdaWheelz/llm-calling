"""Gemini API client.

Per PR-04 spec section 4.3:
- Non-streaming: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
- Streaming: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse

Auth:
- Header: x-goog-api-key: <key>
- NEVER put key in query param
- NEVER log URL if key accidentally in query

Turn conversion:
- System turn → systemInstruction.parts[0].text
- "assistant" role → "model" role in Gemini
- Each turn's content → parts: [{"text": "..."}]

Request body:
{
  "contents": [
    {"role": "user", "parts": [{"text": "..."}]},
    {"role": "model", "parts": [{"text": "..."}]}
  ],
  "systemInstruction": {"parts": [{"text": "<system_prompt>"}]},
  "generationConfig": {
    "maxOutputTokens": 1024,
    "temperature": 0.7
  }
}

Response (non-stream):
{
  "candidates": [{
    "content": {"parts": [{"text": "<output_text>"}]}
  }],
  "usageMetadata": {
    "promptTokenCount": 100,
    "candidatesTokenCount": 50,
    "totalTokenCount": 150
  }
}

- text = concatenate candidates[0].content.parts[].text
- usage.prompt_tokens = promptTokenCount
- usage.completion_tokens = candidatesTokenCount
- usage.total_tokens = totalTokenCount
- provider_request_id = None (Gemini doesn't return one)

Streaming:
- Use :streamGenerateContent?alt=sse endpoint
- Each event: data: {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
- Terminal: last event has "finishReason": "STOP"
- Usage in final event's usageMetadata
"""

import json
from collections.abc import AsyncIterator

import httpx

from llm_calling.errors import LLMError, LLMErrorCode
from llm_calling.types import LLMChunk, LLMRequest, LLMResponse, LLMUsage, ToolCall, Turn

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_31_PRO_PREVIEW = "gemini-3.1-pro-preview"
GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"


class GeminiClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def generate(
        self,
        req: LLMRequest,
        *,
        api_key: str,
        timeout_s: int,
    ) -> LLMResponse:
        """Non-streaming content generation."""
        url = f"{GEMINI_BASE_URL}/{req.model_name}:generateContent"
        headers = self._build_headers(api_key)
        body = self._build_request_body(req)

        response = await self._client.post(
            url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        response.raise_for_status()

        data = response.json()
        return self._parse_response(data, structured=bool(req.structured_output))

    async def generate_stream(
        self,
        req: LLMRequest,
        *,
        api_key: str,
        timeout_s: int,
    ) -> AsyncIterator[LLMChunk]:
        """Streaming content generation using Server-Sent Events."""
        if req.structured_output is not None:
            raise LLMError(
                LLMErrorCode.BAD_REQUEST,
                "Gemini structured output streaming is not implemented",
                provider="gemini",
            )
        url = f"{GEMINI_BASE_URL}/{req.model_name}:streamGenerateContent?alt=sse"
        headers = self._build_headers(api_key)
        body = self._build_request_body(req)

        async with self._client.stream(
            "POST",
            url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        ) as response:
            response.raise_for_status()

            received_stop = False
            usage: LLMUsage | None = None

            async for line in response.aiter_lines():
                if not line:
                    continue

                # Gemini SSE format: "data: {...}"
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract text from candidates[0].content.parts[].text
                candidates = data.get("candidates", [])
                if candidates:
                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])

                    delta_text = ""
                    tool_calls: list[ToolCall] = []
                    for part in parts:
                        if "text" in part:
                            delta_text += part["text"]
                        elif "functionCall" in part:
                            fc = part["functionCall"]
                            name = fc.get("name", "")
                            args = fc.get("args") or {}
                            tool_calls.append(ToolCall(id=name, name=name, arguments=args))

                    # Check for finish reason
                    finish_reason = candidate.get("finishReason")

                    # Extract usage from final event
                    usage_metadata = data.get("usageMetadata")
                    if usage_metadata:
                        usage = LLMUsage(
                            input_tokens=usage_metadata.get("promptTokenCount"),
                            output_tokens=usage_metadata.get("candidatesTokenCount"),
                            total_tokens=usage_metadata.get("totalTokenCount"),
                            provider_usage=dict(usage_metadata),
                        )

                    if finish_reason == "STOP":
                        received_stop = True
                        # Yield any remaining text as non-terminal
                        if delta_text:
                            yield LLMChunk(delta_text=delta_text, done=False)
                        for tc in tool_calls:
                            yield LLMChunk(tool_call=tc, done=False)
                        # Then yield terminal chunk
                        yield LLMChunk(
                            delta_text="",
                            done=True,
                            usage=usage,
                            provider_request_id=None,  # Gemini doesn't provide request ID
                        )
                        break
                    else:
                        if delta_text:
                            yield LLMChunk(delta_text=delta_text, done=False)
                        for tc in tool_calls:
                            yield LLMChunk(tool_call=tc, done=False)

            if not received_stop:
                raise LLMError(
                    LLMErrorCode.PROVIDER_DOWN,
                    "Gemini stream ended without STOP finish reason",
                    provider="gemini",
                )

    def _build_headers(self, api_key: str) -> dict[str, str]:
        """Build request headers.

        Note: API key goes in header, NEVER in query param.
        """
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

    def _build_request_body(self, req: LLMRequest) -> dict:
        """Build request body from LLMRequest.

        Extracts system turn to systemInstruction and maps roles.
        """
        if req.prompt_cache_key is not None or any(
            turn.cache_ttl != "none" for turn in req.messages
        ):
            raise LLMError(
                LLMErrorCode.BAD_REQUEST,
                "Gemini cached content is not implemented for this request",
                provider="gemini",
            )

        # Build call_id → tool name lookup from assistant turns (Gemini matches
        # function responses by name, not call_id).
        call_id_to_name: dict[str, str] = {}
        for turn in req.messages:
            if turn.role == "assistant":
                for call in turn.tool_calls:
                    call_id_to_name[call.id] = call.name

        # Extract system prompt and non-system messages
        system_prompt = None
        contents = []

        for turn in req.messages:
            if turn.role == "system":
                system_prompt = turn.content
            else:
                contents.append(self._turn_to_content(turn, call_id_to_name))

        body: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": req.max_tokens,
            },
        }

        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        if req.temperature is not None:
            body["generationConfig"]["temperature"] = req.temperature
        if req.structured_output is not None:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseJsonSchema"] = req.structured_output.schema

        if req.tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in req.tools
                    ]
                }
            ]
            mode = {"auto": "AUTO", "none": "NONE", "required": "ANY"}[req.tool_choice]
            body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}

        if req.reasoning_effort == "default":
            return body

        if req.model_name == GEMINI_31_PRO_PREVIEW:
            if req.reasoning_effort in ("none", "minimal", "low"):
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "low"}
            elif req.reasoning_effort in ("medium", "high", "max"):
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "high"}
            else:
                raise ValueError(f"Unknown reasoning_effort: {req.reasoning_effort}")
            return body

        if req.model_name == GEMINI_3_FLASH_PREVIEW:
            if req.reasoning_effort in ("none", "minimal"):
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "minimal"}
            elif req.reasoning_effort == "low":
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "low"}
            elif req.reasoning_effort == "medium":
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "medium"}
            elif req.reasoning_effort in ("high", "max"):
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "high"}
            else:
                raise ValueError(f"Unknown reasoning_effort: {req.reasoning_effort}")
            return body

        if req.reasoning_effort == "none":
            return body
        if req.reasoning_effort == "minimal":
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "minimal"}
            return body
        if req.reasoning_effort == "low":
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "low"}
            return body
        if req.reasoning_effort == "medium":
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "medium"}
            return body
        if req.reasoning_effort == "high":
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "high"}
            return body
        if req.reasoning_effort == "max":
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "high"}
            return body
        raise ValueError(f"Unknown reasoning_effort: {req.reasoning_effort}")

    def _turn_to_content(self, turn: Turn, call_id_to_name: dict[str, str]) -> dict:
        """Convert Turn to Gemini content format.

        Note: Gemini uses "model" instead of "assistant" for the role, and
        identifies tool responses by function name (not call_id).
        """
        if turn.role == "tool":
            return {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": call_id_to_name.get(result.call_id, result.call_id),
                            "response": {"output": result.output},
                        }
                    }
                    for result in turn.tool_results
                ],
            }
        if turn.role == "assistant" and turn.tool_calls:
            parts: list[dict] = []
            if turn.content:
                parts.append({"text": turn.content})
            for call in turn.tool_calls:
                parts.append({"functionCall": {"name": call.name, "args": call.arguments}})
            return {"role": "model", "parts": parts}
        role = "model" if turn.role == "assistant" else turn.role
        return {
            "role": role,
            "parts": [{"text": turn.content}],
        }

    def _parse_response(self, data: dict, *, structured: bool) -> LLMResponse:
        """Parse non-streaming response."""
        # Extract text from candidates[0].content.parts[].text
        candidates = data.get("candidates", [])
        if not candidates:
            raise LLMError(
                LLMErrorCode.PROVIDER_DOWN,
                "Gemini response missing candidates",
                provider="gemini",
            )

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text_parts = [part.get("text", "") for part in parts if "text" in part]
        text = "".join(text_parts)
        tool_calls: list[ToolCall] = []
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                name = fc.get("name", "")
                args = fc.get("args") or {}
                tool_calls.append(ToolCall(id=name, name=name, arguments=args))
        structured_output = None
        if structured and text.strip().startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                structured_output = parsed

        # Extract usage
        usage = None
        usage_metadata = data.get("usageMetadata")
        if usage_metadata:
            usage = LLMUsage(
                input_tokens=usage_metadata.get("promptTokenCount"),
                output_tokens=usage_metadata.get("candidatesTokenCount"),
                total_tokens=usage_metadata.get("totalTokenCount"),
                provider_usage=dict(usage_metadata),
            )

        # Gemini doesn't return a request ID
        return LLMResponse(
            text=text,
            usage=usage,
            provider_request_id=None,
            structured_output=structured_output,
            tool_calls=tuple(tool_calls),
        )
