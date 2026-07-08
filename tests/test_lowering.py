import pytest

from provider_runtime import (
    DEFAULT_CATALOG,
    ModelCall,
    ModelCallError,
    ModelCallErrorCode,
    ModelMessage,
    ModelRef,
    ProviderArtifact,
    ReasoningConfig,
    StructuredOutputSpec,
    TextPart,
    ToolSpec,
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


def test_openai_strict_tool_schema_is_normalized_before_provider_io() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "freshness_days": {"type": "integer", "nullable": True},
            "filters": {
                "type": "object",
                "properties": {
                    "kinds": {"type": "array", "items": {"type": "string"}},
                    "include_archived": {"type": "boolean"},
                },
                "required": ["kinds"],
            },
        },
        "required": ["query"],
    }
    call = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[ModelMessage(role="user", content="search")],
        max_output_tokens=100,
        tools=(ToolSpec(name="search", description="Search.", parameters=parameters),),
    )

    plan = lower_generate_request(call, _cap("openai", "gpt-5.4-mini"), streaming=False)

    assert "additionalProperties" not in parameters
    normalized = plan.call.tools[0].parameters
    assert normalized["additionalProperties"] is False
    assert normalized["required"] == ["query", "freshness_days", "filters"]
    assert normalized["properties"]["freshness_days"]["type"] == ["integer", "null"]
    assert "nullable" not in normalized["properties"]["freshness_days"]
    filters = normalized["properties"]["filters"]
    assert filters["additionalProperties"] is False
    assert filters["required"] == ["kinds", "include_archived"]
    assert filters["properties"]["include_archived"]["type"] == ["boolean", "null"]


def test_openai_structured_output_schema_is_normalized_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[ModelMessage(role="user", content="json")],
        max_output_tokens=100,
        structured_output=StructuredOutputSpec(
            name="result",
            schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["answer"],
            },
        ),
    )

    plan = lower_generate_request(call, _cap("openai", "gpt-5.4-mini"), streaming=False)

    assert plan.call.structured_output is not None
    schema = plan.call.structured_output.schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["answer", "confidence"]
    assert schema["properties"]["confidence"]["type"] == ["number", "null"]


def test_anthropic_tool_schema_is_not_openai_strictified() -> None:
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
    }
    call = ModelCall(
        model=ModelRef(provider="anthropic", model="claude-sonnet-4-6"),
        messages=[ModelMessage(role="user", content="search")],
        max_output_tokens=100,
        tools=(ToolSpec(name="search", description="Search.", parameters=parameters),),
    )

    plan = lower_generate_request(call, _cap("anthropic", "claude-sonnet-4-6"), streaming=False)

    assert plan.call.tools[0].parameters == parameters
    assert "additionalProperties" not in plan.call.tools[0].parameters


def test_openai_unstrictifiable_tool_schema_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[ModelMessage(role="user", content="map")],
        max_output_tokens=100,
        tools=(
            ToolSpec(
                name="map_tool",
                description="Dynamic map.",
                parameters={
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            ),
        ),
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(call, _cap("openai", "gpt-5.4-mini"), streaming=False)

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "map-like additionalProperties" in exc_info.value.message


def test_openai_additional_properties_true_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[ModelMessage(role="user", content="open object")],
        max_output_tokens=100,
        tools=(
            ToolSpec(
                name="open_object",
                description="Open object.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "additionalProperties": True,
                },
            ),
        ),
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(call, _cap("openai", "gpt-5.4-mini"), streaming=False)

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "additionalProperties=true" in exc_info.value.message


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


def test_kimi_structured_output_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="openrouter", model="moonshotai/kimi-k2.6"),
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
            _cap("openrouter", "moonshotai/kimi-k2.6"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "structured output" in exc_info.value.message


def test_gemini_preview_required_tool_choice_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="gemini", model="gemini-3.1-pro-preview"),
        messages=[ModelMessage(role="user", content="Use the tool.")],
        max_output_tokens=100,
        reasoning=ReasoningConfig(effort="low"),
        tools=(
            ToolSpec(
                name="lookup",
                description="Lookup a value.",
                parameters={"type": "object", "properties": {}},
            ),
        ),
        tool_choice="required",
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(
            call,
            _cap("gemini", "gemini-3.1-pro-preview"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "required tool choice" in exc_info.value.message


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


def test_provider_artifact_replay_unsupported_model_fails_before_provider_io() -> None:
    call = ModelCall(
        model=ModelRef(provider="cloudflare", model="@cf/openai/gpt-oss-20b"),
        messages=[
            ModelMessage(
                role="assistant",
                provider_artifacts=(
                    ProviderArtifact(
                        provider="cloudflare",
                        model="@cf/openai/gpt-oss-20b",
                        purpose="reasoning",
                        payload={"reasoning": "opaque"},
                    ),
                ),
            )
        ],
        max_output_tokens=100,
    )

    with pytest.raises(ModelCallError) as exc_info:
        lower_generate_request(
            call,
            _cap("cloudflare", "@cf/openai/gpt-oss-20b"),
            streaming=False,
        )

    assert exc_info.value.error_code == ModelCallErrorCode.BAD_REQUEST
    assert "provider artifact replay" in exc_info.value.message
