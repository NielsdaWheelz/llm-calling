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
    TextPart,
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
        model=ModelRef(provider="cloudflare", model="@cf/openai/gpt-oss-20b"),
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
            _cap("cloudflare", "@cf/openai/gpt-oss-20b"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "structured output" in exc_info.value.message


def test_structured_output_reasoning_combination_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="anthropic", model="claude-sonnet-4-6"),
        messages=[ModelMessage(role="user", content="json")],
        max_output_tokens=100,
        reasoning=ReasoningConfig(effort="high"),
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
            _cap("anthropic", "claude-sonnet-4-6"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "structured output is not supported with reasoning effort" in exc_info.value.message


def test_reasoning_unsupported_model_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="cloudflare", model="@cf/openai/gpt-oss-20b"),
        messages=[ModelMessage(role="user", content="think")],
        max_output_tokens=100,
        reasoning=ReasoningConfig(effort="high"),
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(
            call,
            _cap("cloudflare", "@cf/openai/gpt-oss-20b"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "reasoning effort" in exc_info.value.message


def test_generation_unsupported_model_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="openai", model="text-embedding-3-small"),
        messages=[ModelMessage(role="user", content="not an embedding call")],
        max_output_tokens=1,
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(
            call,
            _cap("openai", "text-embedding-3-small"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "generation is not supported" in exc_info.value.message


def test_zero_generation_output_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[ModelMessage(role="user", content="hello")],
        max_output_tokens=0,
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(
            call,
            _cap("openai", "gpt-5.4-mini"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "max_output_tokens" in exc_info.value.message


def test_gemini_reasoning_budget_is_validated_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="gemini", model="gemini-2.5-pro"),
        messages=[ModelMessage(role="user", content="think")],
        max_output_tokens=100,
        reasoning=ReasoningConfig(effort="high", budget_tokens=0),
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(
            call,
            _cap("gemini", "gemini-2.5-pro"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "reasoning budget" in exc_info.value.message


def test_content_parts_unsupported_model_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[ModelMessage(role="user", content_parts=(TextPart(text="hello"),))],
        max_output_tokens=100,
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(
            call,
            _cap("openai", "gpt-5.4-mini"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "content parts" in exc_info.value.message
