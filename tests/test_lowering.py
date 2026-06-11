import pytest

from provider_runtime import (
    DEFAULT_CATALOG,
    ModelCall,
    ModelCallError,
    ModelCallErrorCode,
    ModelMessage,
    ModelRef,
    ReasoningConfig,
    StructuredOutputSpec,
    lower_generate_request,
)


def _cap(provider: str, model: str):
    return DEFAULT_CATALOG.require_capabilities(
        ModelRef(provider=provider, model=model)  # type: ignore[arg-type]
    )


def test_openai_cache_intent_derives_prompt_cache_key() -> None:
    call = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[ModelMessage(role="user", content="cache me", cache_ttl="5m")],
        max_output_tokens=100,
    )

    plan = lower_generate_request(call, _cap("openai", "gpt-5.4-mini"), streaming=False)

    assert plan.derived_prompt_cache_key is True
    assert plan.stripped_cache is False
    assert plan.call.prompt_cache_key is not None
    assert plan.call.prompt_cache_key.startswith("pr-")
    assert plan.call.messages[0].cache_ttl == "5m"


def test_unsupported_cache_intent_is_stripped_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.6"),
        messages=[ModelMessage(role="user", content="cache me", cache_ttl="1h")],
        max_output_tokens=100,
        prompt_cache_key="stale-key",
    )

    plan = lower_generate_request(
        call,
        _cap("openrouter", "moonshotai/kimi-k2.6"),
        streaming=False,
    )

    assert plan.stripped_cache is True
    assert plan.call.prompt_cache_key is None
    assert [message.cache_ttl for message in plan.call.messages] == ["none"]


def test_structured_output_unsupported_model_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="cloudflare", model="@cf/meta/llama-3.1-8b-instruct"),
        messages=[ModelMessage(role="user", content="json")],
        max_output_tokens=100,
        structured_output=StructuredOutputSpec(
            name="result",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        ),
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(
            call,
            _cap("cloudflare", "@cf/meta/llama-3.1-8b-instruct"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "structured output" in exc_info.value.message


def test_reasoning_unsupported_model_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="cloudflare", model="@cf/meta/llama-3.1-8b-instruct"),
        messages=[ModelMessage(role="user", content="think")],
        max_output_tokens=100,
        reasoning=ReasoningConfig(effort="high"),
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(
            call,
            _cap("cloudflare", "@cf/meta/llama-3.1-8b-instruct"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "reasoning effort" in exc_info.value.message
