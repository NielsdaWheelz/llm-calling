"""Gemini API client.

Per PR-04 spec section 4.3:
- Non-streaming: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
- Streaming: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse

Auth:
- Header: x-goog-api-key: <key>
- NEVER put key in query param
- NEVER log URL if key accidentally in query

ModelMessage conversion:
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

import base64
import json
from collections.abc import AsyncIterator

import httpx

from provider_runtime._artifact_validation import validated_provider_artifacts
from provider_runtime.endpoints import GEMINI_BASE_URL
from provider_runtime.errors import ModelCallError, ModelCallErrorCode, raise_for_provider_error
from provider_runtime.structured_output import parse_required_structured_output
from provider_runtime.tool_arguments import parse_tool_arguments_with_status
from provider_runtime.types import (
    BinaryPart,
    ModelCall,
    ModelChunk,
    ModelMessage,
    ModelResponse,
    ProviderArtifact,
    TextPart,
    TokenUsage,
    ToolCall,
)

GEMINI_25_PRO = "gemini-2.5-pro"
GEMINI_25_FLASH = "gemini-2.5-flash"
GEMINI_31_PRO_PREVIEW = "gemini-3.1-pro-preview"
GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
_GEMINI_25_PRO_BUDGET_BY_EFFORT = {
    "minimal": 128,
    "low": 1024,
    "medium": 8192,
    "high": 16384,
    "max": 32768,
}
_GEMINI_25_FLASH_BUDGET_BY_EFFORT = {
    "none": 0,
    "minimal": 512,
    "low": 1024,
    "medium": 8192,
    "high": 16384,
    "max": 24576,
}


class GeminiClient:
    def __init__(self, client: httpx.AsyncClient, *, base_url: str = GEMINI_BASE_URL):
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def generate(
        self,
        req: ModelCall,
        *,
        api_key: str,
        timeout_s: float,
    ) -> ModelResponse:
        """Non-streaming content generation."""
        url = f"{self._base_url}/{req.model.model}:generateContent"
        headers = self._build_headers(api_key)
        body = self._build_request_body(req)

        response = await self._client.post(
            url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        await raise_for_provider_error(response, "gemini")

        data = response.json()
        return self._parse_response(
            data,
            structured=bool(req.structured_output),
            model=req.model.model,
        )

    async def generate_stream(
        self,
        req: ModelCall,
        *,
        api_key: str,
        timeout_s: float,
    ) -> AsyncIterator[ModelChunk]:
        """Streaming content generation using Server-Sent Events."""
        if req.structured_output is not None:
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                "Gemini structured output streaming is not implemented",
                provider="gemini",
            )
        url = f"{self._base_url}/{req.model.model}:streamGenerateContent?alt=sse"
        headers = self._build_headers(api_key)
        body = self._build_request_body(req)

        async with self._client.stream(
            "POST",
            url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        ) as response:
            await raise_for_provider_error(response, "gemini")

            received_terminal = False
            usage: TokenUsage | None = None

            async for line in response.aiter_lines():
                if not line:
                    continue

                # Gemini SSE format: "data: {...}"
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError as exc:
                    raise ModelCallError(
                        ModelCallErrorCode.PROVIDER_DOWN,
                        "Gemini stream event was not valid JSON",
                        provider="gemini",
                        retryable=False,
                    ) from exc

                # Extract text from candidates[0].content.parts[].text
                candidates = data.get("candidates", [])
                if candidates:
                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])

                    delta_text = ""
                    tool_calls: list[ToolCall] = []
                    provider_artifacts: list[ProviderArtifact] = []
                    for part in parts:
                        if part.get("thought"):
                            # Thought-summary parts are not visible output.
                            continue
                        if "text" in part:
                            delta_text += part["text"]
                        elif "functionCall" in part:
                            tool_call, artifact = _part_to_tool_call_and_artifact(
                                part, model=req.model.model
                            )
                            tool_calls.append(tool_call)
                            if artifact is not None:
                                provider_artifacts.append(artifact)

                    # Check for finish reason
                    finish_reason = candidate.get("finishReason")

                    # Extract usage from final event
                    usage_metadata = data.get("usageMetadata")
                    if usage_metadata:
                        usage = TokenUsage(
                            input_tokens=usage_metadata.get("promptTokenCount"),
                            output_tokens=usage_metadata.get("candidatesTokenCount"),
                            total_tokens=usage_metadata.get("totalTokenCount"),
                            provider_usage=dict(usage_metadata),
                        )

                    if finish_reason:
                        received_terminal = True
                        # Yield any remaining text as non-terminal
                        if delta_text:
                            yield ModelChunk(delta_text=delta_text, done=False)
                        for artifact in provider_artifacts:
                            yield ModelChunk(provider_artifact=artifact, done=False)
                        for tc in tool_calls:
                            yield ModelChunk(tool_call=tc, done=False)
                        # Then yield terminal chunk
                        yield ModelChunk(
                            delta_text="",
                            done=True,
                            usage=usage,
                            provider_request_id=None,  # Gemini doesn't provide request ID
                            status=("completed" if finish_reason == "STOP" else "incomplete"),
                            incomplete_details=(
                                None
                                if finish_reason == "STOP"
                                else {"finish_reason": finish_reason}
                            ),
                        )
                        break
                    else:
                        if delta_text:
                            yield ModelChunk(delta_text=delta_text, done=False)
                        for artifact in provider_artifacts:
                            yield ModelChunk(provider_artifact=artifact, done=False)
                        for tc in tool_calls:
                            yield ModelChunk(tool_call=tc, done=False)

            if not received_terminal:
                raise ModelCallError(
                    ModelCallErrorCode.PROVIDER_DOWN,
                    "Gemini stream ended without terminal finish reason",
                    provider="gemini",
                    retryable=False,
                )

    def _build_headers(self, api_key: str) -> dict[str, str]:
        """Build request headers.

        Note: API key goes in header, NEVER in query param.
        """
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

    def _build_request_body(self, req: ModelCall) -> dict:
        """Build request body from ModelCall.

        Extracts system turn to systemInstruction and maps roles.
        """
        if req.prompt_cache_key is not None or any(
            turn.cache_ttl != "none" for turn in req.messages
        ):
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
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
                contents.append(self._turn_to_content(turn, call_id_to_name, model=req.model.model))

        body: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": req.max_output_tokens,
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
                            "parameters": _gemini_tool_schema(tool.parameters),
                        }
                        for tool in req.tools
                    ]
                }
            ]
            mode = {"auto": "AUTO", "none": "NONE", "required": "ANY"}[req.tool_choice]
            body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}

        if req.model.model == GEMINI_25_PRO:
            budget = req.reasoning.budget_tokens
            if budget is None:
                effort = "minimal" if req.reasoning.effort == "default" else req.reasoning.effort
                budget = _GEMINI_25_PRO_BUDGET_BY_EFFORT.get(effort)
            elif req.reasoning.effort == "default":
                effort = "minimal"
            else:
                effort = req.reasoning.effort
            if budget is None:
                raise ValueError(f"Unknown reasoning_effort: {effort}")
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget}
            return body

        if req.model.model == GEMINI_25_FLASH:
            budget = req.reasoning.budget_tokens
            if budget is None:
                effort = "none" if req.reasoning.effort == "default" else req.reasoning.effort
                budget = _GEMINI_25_FLASH_BUDGET_BY_EFFORT.get(effort)
            elif req.reasoning.effort == "default":
                effort = "none"
            else:
                effort = req.reasoning.effort
            if budget is None:
                raise ValueError(f"Unknown reasoning_effort: {effort}")
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget}
            return body

        if req.reasoning.effort == "default":
            if req.model.model == GEMINI_31_PRO_PREVIEW:
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "low"}
                return body
            if req.model.model == GEMINI_3_FLASH_PREVIEW:
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "minimal"}
                return body

        if req.model.model == GEMINI_31_PRO_PREVIEW:
            if req.reasoning.effort in ("low", "medium"):
                level = req.reasoning.effort
            elif req.reasoning.effort in ("high", "max"):
                level = "high"
            else:
                raise ValueError(f"Unknown reasoning_effort: {req.reasoning.effort}")
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": level}
            return body

        if req.model.model == GEMINI_3_FLASH_PREVIEW:
            if req.reasoning.effort in ("minimal", "low", "medium", "high"):
                level = req.reasoning.effort
            elif req.reasoning.effort == "max":
                level = "high"
            else:
                raise ValueError(f"Unknown reasoning_effort: {req.reasoning.effort}")
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": level}
            return body

        raise ValueError(f"Unknown reasoning_effort: {req.reasoning.effort}")

    def _turn_to_content(
        self,
        turn: ModelMessage,
        call_id_to_name: dict[str, str],
        *,
        model: str,
    ) -> dict:
        """Convert ModelMessage to Gemini content format.

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
        if turn.role == "assistant" and (turn.tool_calls or turn.provider_artifacts):
            parts: list[dict] = []
            signature_by_call = _gemini_signature_by_call(turn, model=model)
            if turn.content:
                parts.append({"text": turn.content})
            for call in turn.tool_calls:
                part: dict[str, object] = {
                    "functionCall": {"name": call.name, "args": call.arguments}
                }
                signature = signature_by_call.get(call.id) or signature_by_call.get(call.name)
                if signature is not None:
                    part["thoughtSignature"] = signature
                parts.append(part)
            return {"role": "model", "parts": parts}
        role = "model" if turn.role == "assistant" else turn.role
        return {
            "role": role,
            "parts": _turn_parts(turn),
        }

    def _parse_response(self, data: dict, *, structured: bool, model: str) -> ModelResponse:
        """Parse non-streaming response."""
        # Extract text from candidates[0].content.parts[].text
        candidates = data.get("candidates", [])
        if not candidates:
            raise ModelCallError(
                ModelCallErrorCode.PROVIDER_DOWN,
                "Gemini response missing candidates",
                provider="gemini",
                retryable=False,
            )

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        text_parts = [
            part.get("text", "") for part in parts if "text" in part and not part.get("thought")
        ]
        text = "".join(text_parts)
        tool_calls: list[ToolCall] = []
        provider_artifacts: list[ProviderArtifact] = []
        for part in parts:
            if "functionCall" in part:
                tool_call, artifact = _part_to_tool_call_and_artifact(part, model=model)
                tool_calls.append(tool_call)
                if artifact is not None:
                    provider_artifacts.append(artifact)
        structured_output = None
        if structured:
            structured_output = parse_required_structured_output(text, provider="gemini")

        # Extract usage
        usage = None
        usage_metadata = data.get("usageMetadata")
        if usage_metadata:
            usage = TokenUsage(
                input_tokens=usage_metadata.get("promptTokenCount"),
                output_tokens=usage_metadata.get("candidatesTokenCount"),
                total_tokens=usage_metadata.get("totalTokenCount"),
                provider_usage=dict(usage_metadata),
            )

        finish_reason = candidate.get("finishReason")
        status = None
        incomplete_details = None
        if finish_reason:
            status = "completed" if finish_reason == "STOP" else "incomplete"
            if finish_reason != "STOP":
                incomplete_details = {"finish_reason": finish_reason}

        # Gemini doesn't return a request ID
        return ModelResponse(
            text=text,
            usage=usage,
            provider_request_id=None,
            status=status,
            incomplete_details=incomplete_details,
            structured_output=structured_output,
            tool_calls=tuple(tool_calls),
            provider_artifacts=tuple(provider_artifacts),
        )


def _gemini_tool_schema(schema: dict[str, object]) -> dict[str, object]:
    stripped = _strip_gemini_tool_schema_unsupported_fields(schema)
    return stripped if isinstance(stripped, dict) else {}


def _strip_gemini_tool_schema_unsupported_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_gemini_tool_schema_unsupported_fields(inner)
            for key, inner in value.items()
            if key != "additionalProperties"
        }
    if isinstance(value, list):
        return [_strip_gemini_tool_schema_unsupported_fields(inner) for inner in value]
    return value


def _part_to_tool_call_and_artifact(
    part: dict, *, model: str
) -> tuple[ToolCall, ProviderArtifact | None]:
    """Parse a functionCall part and capture Gemini thoughtSignature as an artifact."""
    fc = part["functionCall"]
    name = fc.get("name", "")
    call_id = fc.get("id") or name
    signature = part.get("thoughtSignature")
    parsed_args = parse_tool_arguments_with_status(
        fc.get("args"), provider="gemini", tool_name=name
    )
    tool_call = ToolCall(
        id=call_id,
        name=name,
        arguments=parsed_args.arguments,
        argument_status=parsed_args.status,
    )
    artifact = (
        ProviderArtifact(
            provider="gemini",
            model=model,
            purpose="signature",
            payload={
                "type": "gemini.thought_signature",
                "function_call_id": call_id,
                "function_name": name,
                "thoughtSignature": signature,
            },
        )
        if signature
        else None
    )
    return tool_call, artifact


def _turn_parts(turn: ModelMessage) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    if turn.content:
        parts.append({"text": turn.content})
    for part in turn.content_parts:
        if isinstance(part, TextPart):
            parts.append({"text": part.text})
        elif isinstance(part, BinaryPart):
            parts.append(
                {
                    "inlineData": {
                        "mimeType": part.media_type,
                        "data": base64.b64encode(part.data).decode("ascii"),
                    }
                }
            )
    return parts or [{"text": ""}]


def _gemini_signature_by_call(turn: ModelMessage, *, model: str) -> dict[str, str]:
    signatures: dict[str, str] = {}
    call_ids = {call.id for call in turn.tool_calls}
    call_names = {call.name for call in turn.tool_calls}
    for artifact in validated_provider_artifacts(
        turn.provider_artifacts,
        provider="gemini",
        model=model,
        purpose="signature",
    ):
        payload = artifact.to_provider_payload()
        signature = payload.get("thoughtSignature")
        if not isinstance(signature, str) or not signature:
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                "Gemini signature artifact replay requires thoughtSignature",
                provider="gemini",
                retryable=False,
            )
        matched = False
        for key, valid_keys in (
            (payload.get("function_call_id"), call_ids),
            (payload.get("function_name"), call_names),
        ):
            if isinstance(key, str) and key:
                if key in valid_keys:
                    signatures[key] = signature
                    matched = True
                continue
        if not matched:
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                "Gemini signature artifact replay requires a matching assistant tool call",
                provider="gemini",
                retryable=False,
            )
    return signatures
