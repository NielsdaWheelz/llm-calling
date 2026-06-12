"""Provider-neutral request validation and high-level lowering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from provider_runtime.catalog import ModelCapability
from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.types import ModelCall, ModelMessage


@dataclass(frozen=True)
class GenerateRequestPlan:
    call: ModelCall
    stripped_cache: bool = False
    derived_prompt_cache_key: bool = False


def lower_generate_request(
    call: ModelCall,
    capabilities: ModelCapability,
    *,
    streaming: bool,
) -> GenerateRequestPlan:
    """Validate a call against model capabilities and lower optional intent."""
    _validate_generate_request(call, capabilities, streaming=streaming)
    lowered = _lower_prompt_cache(call, capabilities)
    return lowered


def _validate_generate_request(
    call: ModelCall,
    capabilities: ModelCapability,
    *,
    streaming: bool,
) -> None:
    if not capabilities.generation:
        raise _bad_request(capabilities, "generation is not supported")
    if streaming and not capabilities.streaming:
        raise _bad_request(capabilities, "streaming is not supported")
    if call.max_output_tokens < 1:
        raise _bad_request(capabilities, "max_output_tokens must be >= 1")
    if not call.messages:
        raise _bad_request(capabilities, "messages must be non-empty")
    if call.temperature is not None and not (0 <= call.temperature <= 2):
        raise _bad_request(capabilities, "temperature must be between 0 and 2")
    if call.reasoning.effort not in capabilities.reasoning_modes:
        raise _bad_request(
            capabilities,
            f"reasoning effort {call.reasoning.effort!r} is not supported",
        )
    _validate_reasoning_budget(call, capabilities)
    if capabilities.max_output_tokens is not None and (
        call.max_output_tokens > capabilities.max_output_tokens
    ):
        raise _bad_request(
            capabilities,
            f"max_output_tokens exceeds supported limit {capabilities.max_output_tokens}",
        )
    if call.structured_output is not None:
        supported = (
            capabilities.structured_output_streaming
            if streaming
            else capabilities.structured_output
        )
        if not supported:
            raise _bad_request(capabilities, "structured output is not supported")
        allowed_reasoning_modes = capabilities.structured_output_reasoning_modes
        if (
            allowed_reasoning_modes is not None
            and call.reasoning.effort not in allowed_reasoning_modes
        ):
            raise _bad_request(
                capabilities,
                "structured output is not supported with reasoning effort "
                f"{call.reasoning.effort!r}",
            )
    if call.tools and not capabilities.tool_calling:
        raise _bad_request(capabilities, "tool calling is not supported")
    if call.tool_choice == "required" and not capabilities.tool_choice_required:
        raise _bad_request(capabilities, "required tool choice is not supported")
    if (
        any(message.content_parts for message in call.messages)
        and not capabilities.multimodal_input
    ):
        raise _bad_request(capabilities, "content parts are not supported")
    if (
        any(message.provider_artifacts for message in call.messages)
        and not capabilities.raw_artifact_support
    ):
        raise _bad_request(capabilities, "provider artifact replay is not supported")


def _validate_reasoning_budget(call: ModelCall, capabilities: ModelCapability) -> None:
    budget_tokens = call.reasoning.budget_tokens
    if budget_tokens is None:
        return
    if budget_tokens == -1 and capabilities.reasoning_allows_dynamic_budget:
        return
    if budget_tokens in capabilities.reasoning_budget_tokens:
        return
    if capabilities.reasoning_budget_range is not None:
        min_budget, max_budget = capabilities.reasoning_budget_range
        if min_budget <= budget_tokens <= max_budget:
            return
    raise _bad_request(
        capabilities,
        f"reasoning budget {budget_tokens!r} is not supported",
    )


def _lower_prompt_cache(
    call: ModelCall,
    capabilities: ModelCapability,
) -> GenerateRequestPlan:
    cacheable_turns = [turn for turn in call.messages if turn.cache_ttl != "none"]
    if not cacheable_turns:
        return GenerateRequestPlan(_with_prompt_cache_key(call, None))

    if not capabilities.prompt_cache.supported:
        return GenerateRequestPlan(
            _with_prompt_cache_key(
                replace(
                    call,
                    messages=[_without_cache_ttl(turn) for turn in call.messages],
                ),
                None,
            ),
            stripped_cache=True,
        )

    allowed_ttls = set(capabilities.prompt_cache.ttl_options)
    lowered_messages = [
        turn
        if turn.cache_ttl == "none" or turn.cache_ttl in allowed_ttls
        else _without_cache_ttl(turn)
        for turn in call.messages
    ]
    if not any(turn.cache_ttl != "none" for turn in lowered_messages):
        return GenerateRequestPlan(
            _with_prompt_cache_key(
                replace(call, messages=lowered_messages),
                None,
            ),
            stripped_cache=True,
        )

    if capabilities.prompt_cache.requires_key:
        return GenerateRequestPlan(
            _with_prompt_cache_key(
                replace(call, messages=lowered_messages),
                _derive_prompt_cache_key(call, capabilities, lowered_messages),
            ),
            derived_prompt_cache_key=True,
        )

    return GenerateRequestPlan(
        _with_prompt_cache_key(
            replace(call, messages=lowered_messages),
            None,
        )
    )


def _with_prompt_cache_key(call: ModelCall, prompt_cache_key: str | None) -> ModelCall:
    lowered = replace(call)
    object.__setattr__(lowered, "prompt_cache_key", prompt_cache_key)
    return lowered


def _without_cache_ttl(turn: ModelMessage) -> ModelMessage:
    return turn if turn.cache_ttl == "none" else replace(turn, cache_ttl="none")


def _derive_prompt_cache_key(
    call: ModelCall,
    capabilities: ModelCapability,
    messages: list[ModelMessage],
) -> str:
    payload = {
        "provider": capabilities.provider,
        "model": capabilities.model,
        "messages": [
            {"role": message.role, "ttl": message.cache_ttl, "content": message.content}
            for message in messages
            if message.cache_ttl != "none"
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"pr-{hashlib.sha256(encoded).hexdigest()[:40]}"


def _bad_request(capabilities: ModelCapability, message: str) -> ModelCallError:
    return ModelCallError(
        ModelCallErrorCode.BAD_REQUEST,
        f"{capabilities.provider}/{capabilities.model}: {message}",
        provider=capabilities.provider,
        retryable=False,
    )
