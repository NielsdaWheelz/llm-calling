"""OpenAI-compatible chat completions client."""

import json
from collections.abc import AsyncIterator
from typing import Literal, cast

import httpx

from provider_runtime._artifact_validation import validated_provider_artifacts
from provider_runtime.errors import ModelCallError, ModelCallErrorCode, raise_for_provider_error
from provider_runtime.structured_output import parse_required_structured_output
from provider_runtime.tool_arguments import parse_tool_arguments_with_status
from provider_runtime.types import (
    ModelCall,
    ModelChunk,
    ModelResponse,
    ProviderArtifact,
    ProviderName,
    TokenUsage,
    ToolCall,
)

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
        self._provider: OpenAICompatibleProvider = provider
        self._url = f"{base_url.rstrip('/')}/chat/completions"

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
        await raise_for_provider_error(response, self._provider)

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
            finish_reason: str | None = None
            tool_call_acc: dict[int, dict] = {}

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    received_done = True
                    for idx in sorted(tool_call_acc):
                        acc = tool_call_acc[idx]
                        parsed_args = parse_tool_arguments_with_status(
                            acc["arguments"],
                            provider=self._provider,
                            tool_name=acc["name"],
                            call_id=acc["id"],
                        )
                        yield ModelChunk(
                            tool_call=ToolCall(
                                id=acc["id"],
                                name=acc["name"],
                                arguments=parsed_args.arguments,
                                argument_status=parsed_args.status,
                            ),
                            done=False,
                        )
                    tool_call_acc.clear()
                    yield ModelChunk(
                        delta_text="",
                        done=True,
                        usage=accumulated_usage,
                        provider_request_id=provider_request_id,
                        status=_status_from_finish_reason(finish_reason),
                        incomplete_details=_incomplete_details_from_finish_reason(finish_reason),
                    )
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError as exc:
                    raise ModelCallError(
                        ModelCallErrorCode.PROVIDER_DOWN,
                        f"{self._provider} stream event was not valid JSON",
                        provider=self._provider,
                        retryable=False,
                    ) from exc

                if "usage" in data:
                    accumulated_usage = self._parse_usage(data["usage"])

                choices = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta", {})
                delta_text = delta.get("content", "")

                for artifact in self._message_provider_artifacts(delta, model=req.model.model):
                    yield ModelChunk(provider_artifact=artifact, done=False)

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
                        parsed_args = parse_tool_arguments_with_status(
                            acc["arguments"],
                            provider=self._provider,
                            tool_name=acc["name"],
                            call_id=acc["id"],
                        )
                        yield ModelChunk(
                            tool_call=ToolCall(
                                id=acc["id"],
                                name=acc["name"],
                                arguments=parsed_args.arguments,
                                argument_status=parsed_args.status,
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
            if turn.role == "assistant" and (turn.tool_calls or turn.provider_artifacts):
                message: dict[str, object] = {
                    "role": "assistant",
                    "content": turn.content or None,
                }
                if turn.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in turn.tool_calls
                    ]
                message.update(
                    self._provider_artifact_message_fields(
                        turn.provider_artifacts,
                        model=req.model.model,
                    )
                )
                messages.append(message)
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
            effort = req.reasoning.effort
            if effort == "default":
                effort = "none"
            elif effort == "max":
                effort = "xhigh"
            body["reasoning"] = {"effort": effort, "exclude": True}
        elif req.reasoning.effort not in ("default", "none"):
            raise ModelCallError(
                ModelCallErrorCode.BAD_REQUEST,
                f"{self._provider} reasoning controls are not implemented",
                provider=self._provider,
            )

        return body

    def _parse_response(
        self,
        data: dict,
        headers: httpx.Headers,
        *,
        structured: bool,
        model: str,
    ) -> ModelResponse:
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
        provider_artifacts = self._message_provider_artifacts(message, model=model)
        structured_output = None
        if structured:
            structured_output = parse_required_structured_output(text, provider=self._provider)
        finish_reason = choices[0].get("finish_reason")
        finish_reason_str = str(finish_reason) if finish_reason else None

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_str = fn.get("arguments") or ""
            parsed_args = parse_tool_arguments_with_status(
                args_str,
                provider=self._provider,
                tool_name=fn.get("name") or "",
                call_id=tc.get("id") or "",
            )
            tool_calls.append(
                ToolCall(
                    id=tc.get("id") or "",
                    name=fn.get("name") or "",
                    arguments=parsed_args.arguments,
                    argument_status=parsed_args.status,
                )
            )

        usage = None
        usage_data = data.get("usage")
        if usage_data:
            usage = self._parse_usage(usage_data)

        provider_request_id = headers.get("x-request-id") or data.get("id")

        return ModelResponse(
            text=text,
            usage=usage,
            provider_request_id=provider_request_id,
            status=_status_from_finish_reason(finish_reason_str),
            incomplete_details=_incomplete_details_from_finish_reason(finish_reason_str),
            structured_output=structured_output,
            tool_calls=tuple(tool_calls),
            provider_artifacts=tuple(provider_artifacts),
        )

    def _parse_usage(self, usage_data: dict) -> TokenUsage:
        prompt_details = usage_data.get("prompt_tokens_details") or {}
        completion_details = usage_data.get("completion_tokens_details") or {}
        cached_tokens = prompt_details.get("cached_tokens")
        cache_write_tokens = prompt_details.get("cache_write_tokens")
        return TokenUsage(
            input_tokens=usage_data.get("prompt_tokens"),
            output_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            reasoning_tokens=completion_details.get("reasoning_tokens"),
            cache_creation_input_tokens=cache_write_tokens,
            cached_tokens=cached_tokens,
            cache_read_input_tokens=cached_tokens,
            provider_usage=dict(usage_data),
        )

    def _message_provider_artifacts(
        self,
        message: dict,
        *,
        model: str,
    ) -> list[ProviderArtifact]:
        artifacts: list[ProviderArtifact] = []
        reasoning_details = message.get("reasoning_details")
        if isinstance(reasoning_details, list):
            for detail in reasoning_details:
                if isinstance(detail, dict):
                    artifacts.append(
                        ProviderArtifact(
                            provider=cast(ProviderName, self._provider),
                            model=model,
                            purpose="reasoning",
                            payload=dict(detail),
                        )
                    )
        for field_name in ("reasoning", "reasoning_content"):
            value = message.get(field_name)
            if isinstance(value, str) and value:
                artifacts.append(
                    ProviderArtifact(
                        provider=cast(ProviderName, self._provider),
                        model=model,
                        purpose="reasoning",
                        payload={field_name: value},
                    )
                )
        return artifacts

    def _provider_artifact_message_fields(
        self,
        artifacts: tuple[ProviderArtifact, ...],
        *,
        model: str,
    ) -> dict[str, object]:
        reasoning_details: list[dict[str, object]] = []
        reasoning_parts: list[str] = []
        reasoning_content_parts: list[str] = []
        for artifact in validated_provider_artifacts(
            artifacts,
            provider=cast(ProviderName, self._provider),
            model=model,
            purpose="reasoning",
        ):
            payload = artifact.to_provider_payload()
            if _is_reasoning_detail(payload):
                reasoning_details.append(payload)
                continue
            details = payload.get("reasoning_details")
            if isinstance(details, list):
                reasoning_details.extend(detail for detail in details if isinstance(detail, dict))
                continue
            reasoning = payload.get("reasoning")
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
                continue
            reasoning_content = payload.get("reasoning_content")
            if isinstance(reasoning_content, str):
                reasoning_content_parts.append(reasoning_content)

        if reasoning_details:
            return {"reasoning_details": reasoning_details}
        if reasoning_parts:
            return {"reasoning": "".join(reasoning_parts)}
        if reasoning_content_parts:
            return {"reasoning_content": "".join(reasoning_content_parts)}
        return {}


def _is_reasoning_detail(payload: dict[str, object]) -> bool:
    return isinstance(payload.get("type"), str) and str(payload["type"]).startswith("reasoning.")


def _status_from_finish_reason(finish_reason: str | None) -> str | None:
    if finish_reason is None:
        return None
    if finish_reason in {"stop", "tool_calls"}:
        return "completed"
    return "incomplete"


def _incomplete_details_from_finish_reason(finish_reason: str | None) -> dict[str, object] | None:
    if finish_reason is None or _status_from_finish_reason(finish_reason) == "completed":
        return None
    return {"finish_reason": finish_reason}
