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
  "reasoning": {"effort": "none" | "low" | "medium" | "high" | "xhigh"},
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

from provider_runtime._artifact_validation import validated_provider_artifacts
from provider_runtime.endpoints import OPENAI_BASE_URL
from provider_runtime.errors import ModelCallError, ModelCallErrorCode, raise_for_provider_error
from provider_runtime.structured_output import parse_required_structured_output
from provider_runtime.tool_arguments import parse_tool_arguments_with_status
from provider_runtime.types import (
    ModelCall,
    ModelChunk,
    ModelResponse,
    ProviderArtifact,
    TokenUsage,
    ToolCall,
    TranscriptionCall,
    TranscriptionResponse,
)

OPENAI_RESPONSES_URL = f"{OPENAI_BASE_URL}/responses"


class OpenAIClient:
    def __init__(self, client: httpx.AsyncClient, *, base_url: str = OPENAI_BASE_URL):
        self._client = client
        base = base_url.rstrip("/")
        self._url = f"{base}/responses"
        self._transcriptions_url = f"{base}/audio/transcriptions"

    async def generate(
        self,
        req: ModelCall,
        *,
        api_key: str,
        timeout_s: float,
    ) -> ModelResponse:
        headers = self._build_headers(api_key)
        body = self._build_request_body(req, stream=False)

        response = await self._client.post(
            self._url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        await raise_for_provider_error(response, "openai")

        data = response.json()
        return self._parse_response(
            data,
            response.headers,
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
        if req.structured_output is not None:
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                "OpenAI structured output streaming is not implemented",
                provider="openai",
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
            await raise_for_provider_error(response, "openai")

            provider_request_id = response.headers.get("x-request-id")
            accumulated_usage: TokenUsage | None = None
            emitted_terminal = False
            tool_call_items: dict[str, dict] = {}

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    if not emitted_terminal:
                        yield ModelChunk(
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
                except json.JSONDecodeError as exc:
                    raise ModelCallError(
                        ModelCallErrorCode.PROVIDER_DOWN,
                        "openai stream event was not valid JSON",
                        provider="openai",
                        retryable=False,
                    ) from exc

                event_type = data.get("type")

                if event_type == "response.output_text.delta":
                    delta_text = data.get("delta", "")
                    if delta_text:
                        yield ModelChunk(delta_text=delta_text, done=False)
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
                    if item.get("type") == "reasoning":
                        # Full reasoning item (incl. id and encrypted_content), verbatim.
                        yield ModelChunk(
                            provider_artifact=ProviderArtifact(
                                provider="openai",
                                model=req.model.model,
                                purpose="reasoning",
                                payload=dict(item),
                            ),
                            done=False,
                        )
                    elif item.get("type") == "function_call":
                        item_id = data.get("item_id") or item.get("id") or ""
                        acc = tool_call_items.pop(item_id, None)
                        call_id = item.get("call_id") or (acc["call_id"] if acc else "")
                        name = item.get("name") or (acc["name"] if acc else "")
                        args_str = item.get("arguments")
                        if args_str is None:
                            args_str = acc["arguments"] if acc else ""
                        parsed_args = parse_tool_arguments_with_status(
                            args_str,
                            provider="openai",
                            tool_name=name,
                            call_id=call_id,
                        )
                        yield ModelChunk(
                            tool_call=ToolCall(
                                id=call_id,
                                name=name,
                                arguments=parsed_args.arguments,
                                argument_status=parsed_args.status,
                                provider_metadata={"id": item["id"]} if item.get("id") else None,
                            ),
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
                        yield ModelChunk(
                            delta_text="",
                            done=True,
                            usage=accumulated_usage,
                            provider_request_id=provider_request_id,
                            status=status,
                            incomplete_details=incomplete_details,
                        )
                    break

            if not emitted_terminal:
                raise ModelCallError(
                    ModelCallErrorCode.PROVIDER_DOWN,
                    "openai stream ended without terminal event",
                    provider="openai",
                    retryable=False,
                )

    async def transcribe(
        self,
        req: TranscriptionCall,
        *,
        api_key: str,
        timeout_s: float,
    ) -> TranscriptionResponse:
        response = await self._client.post(
            self._transcriptions_url,
            headers=self._build_auth_headers(api_key),
            data={"model": req.model.model, "response_format": "json"},
            files={"file": (req.filename, req.audio, req.media_type)},
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        await raise_for_provider_error(response, "openai")
        data = response.json()
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str):
            raise ModelCallError(
                ModelCallErrorCode.PROVIDER_DOWN,
                "OpenAI transcription response did not include text",
                provider="openai",
                retryable=False,
            )
        return TranscriptionResponse(
            text=text,
            usage=_parse_transcription_usage(data),
            provider_request_id=response.headers.get("x-request-id")
            or (data.get("id") if isinstance(data, dict) else None),
        )

    def _build_auth_headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_body(self, req: ModelCall, stream: bool) -> dict:
        if req.prompt_cache_key is None and any(turn.cache_ttl != "none" for turn in req.messages):
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
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
            if turn.role == "assistant":
                # Replay captured reasoning items verbatim, in emission order: they must
                # precede the message/function_call items they originally preceded.
                for item in validated_provider_artifacts(
                    turn.provider_artifacts,
                    provider="openai",
                    model=req.model.model,
                    purpose="reasoning",
                ):
                    input_items.append(item.to_provider_payload())
            if turn.content or not turn.tool_calls:
                content_type = "output_text" if turn.role == "assistant" else "input_text"
                input_items.append(
                    {
                        "role": turn.role,
                        "content": [{"type": content_type, "text": turn.content}],
                    }
                )
            if turn.role == "assistant":
                for tc in turn.tool_calls:
                    fc_item: dict[str, object] = {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                    if tc.provider_metadata and "id" in tc.provider_metadata:
                        fc_item["id"] = tc.provider_metadata["id"]
                    input_items.append(fc_item)

        body: dict = {
            "model": req.model.model,
            "input": input_items,
            "max_output_tokens": req.max_output_tokens,
            "stream": stream,
            # Stateless reasoning continuity: never rely on server-stored state, and
            # request encrypted reasoning content so reasoning items can be replayed
            # verbatim across tool continuations. Sent unconditionally: reasoning-family
            # models emit reasoning items even when "reasoning" is omitted ("default"),
            # this client has no model-family registry, and both keys are no-ops for
            # non-reasoning models.
            "store": False,
            "include": ["reasoning.encrypted_content"],
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
                    "strict": t.strict,
                }
                for t in req.tools
            ]
            body["tool_choice"] = req.tool_choice

        if req.reasoning.effort == "default":
            return body

        if req.reasoning.effort == "none":
            body["reasoning"] = {"effort": "none"}
        elif req.reasoning.effort == "low":
            body["reasoning"] = {"effort": "low"}
        elif req.reasoning.effort == "medium":
            body["reasoning"] = {"effort": "medium"}
        elif req.reasoning.effort == "high":
            body["reasoning"] = {"effort": "high"}
        elif req.reasoning.effort == "max":
            body["reasoning"] = {"effort": "xhigh"}
        else:
            raise ValueError(f"Unknown reasoning_effort: {req.reasoning.effort}")

        return body

    def _parse_response(
        self,
        data: dict,
        headers: httpx.Headers,
        *,
        structured: bool,
        model: str,
    ) -> ModelResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        provider_artifacts: list[ProviderArtifact] = []
        for item in data.get("output", []):
            item_type = item.get("type")
            if item_type == "message":
                for content_item in item.get("content", []):
                    if content_item.get("type") == "output_text":
                        text_parts.append(content_item.get("text", ""))
            elif item_type == "reasoning":
                provider_artifacts.append(
                    ProviderArtifact(
                        provider="openai",
                        model=model,
                        purpose="reasoning",
                        payload=dict(item),
                    )
                )
            elif item_type == "function_call":
                args_str = item.get("arguments") or ""
                parsed_args = parse_tool_arguments_with_status(
                    args_str,
                    provider="openai",
                    tool_name=item.get("name") or "",
                    call_id=item.get("call_id") or "",
                )
                tool_calls.append(
                    ToolCall(
                        id=item.get("call_id") or "",
                        name=item.get("name") or "",
                        arguments=parsed_args.arguments,
                        argument_status=parsed_args.status,
                        provider_metadata={"id": item["id"]} if item.get("id") else None,
                    )
                )

        status = data.get("status")
        incomplete_details = data.get("incomplete_details")
        provider_request_id = headers.get("x-request-id") or data.get("id")
        text = "".join(text_parts)
        structured_output = None
        if structured:
            structured_output = parse_required_structured_output(text, provider="openai")

        return ModelResponse(
            text=text,
            usage=self._parse_usage(data["usage"]) if data.get("usage") else None,
            provider_request_id=provider_request_id,
            status=status,
            incomplete_details=incomplete_details,
            structured_output=structured_output,
            tool_calls=tuple(tool_calls),
            provider_artifacts=tuple(provider_artifacts),
        )

    def _parse_usage(self, usage_data: dict) -> TokenUsage:
        output_tokens_details = usage_data.get("output_tokens_details") or {}
        input_tokens_details = usage_data.get("input_tokens_details") or {}
        cached_tokens = input_tokens_details.get("cached_tokens")
        return TokenUsage(
            input_tokens=usage_data.get("input_tokens"),
            output_tokens=usage_data.get("output_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            reasoning_tokens=output_tokens_details.get("reasoning_tokens"),
            cached_tokens=cached_tokens,
            cache_read_input_tokens=cached_tokens,
            provider_usage=dict(usage_data),
        )


def _parse_transcription_usage(data: dict) -> TokenUsage | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = _int_or_none(usage.get("input_tokens"))
    output_tokens = _int_or_none(usage.get("output_tokens"))
    total_tokens = _int_or_none(usage.get("total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        provider_usage=dict(usage),
    )


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None
