"""Live provider-runtime acceptance matrix.

These tests intentionally fail closed. They are excluded from the default test
suite and run only when selected with ``-m live_provider`` plus
``LLM_RUNTIME_LIVE=1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast

import httpx
import pytest

from provider_runtime import (
    ModelCall,
    ModelMessage,
    ModelRef,
    ModelRuntime,
    ProviderApiKey,
    ReasoningConfig,
    RetryPolicy,
    StructuredOutputSpec,
    ToolResult,
    ToolSpec,
)
from provider_runtime.catalog import DEFAULT_CATALOG, ModelCapability
from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.types import ProviderName, ReasoningEffort

pytestmark = [pytest.mark.asyncio, pytest.mark.live_provider]

_PROVIDER_ORDER: tuple[ProviderName, ...] = (
    "openai",
    "anthropic",
    "gemini",
    "openrouter",
    "cloudflare",
)
_PROVIDER_ENV: dict[ProviderName, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "cloudflare": ("CLOUDFLARE_AI_API_TOKEN", "CLOUDFLARE_AI_ACCOUNT_ID"),
}
_REASONING_ORDER: tuple[ReasoningEffort, ...] = (
    "max",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
    "default",
)
_ANTHROPIC_ADAPTIVE_THINKING_MODELS = frozenset(("claude-opus-4-7", "claude-sonnet-4-6"))
_ToolChoice = Literal["auto", "none", "required"]


@dataclass(frozen=True)
class LiveEnv:
    selected_providers: frozenset[ProviderName]

    def key_for(self, provider: ProviderName) -> ProviderApiKey:
        if provider not in self.selected_providers:
            pytest.skip(f"{provider} not selected by LLM_RUNTIME_LIVE_PROVIDERS")
        missing = [name for name in _PROVIDER_ENV[provider] if not os.environ.get(name)]
        if missing:
            pytest.fail(f"{provider} live-provider matrix requires env vars: {', '.join(missing)}")
        return ProviderApiKey(os.environ[_PROVIDER_ENV[provider][0]], source="platform")


@dataclass(frozen=True)
class ProviderCase:
    provider: ProviderName
    capability: ModelCapability

    @property
    def model(self) -> str:
        return self.capability.model


@pytest.fixture(scope="session")
def live_env() -> LiveEnv:
    if os.environ.get("LLM_RUNTIME_LIVE") != "1":
        pytest.fail("Set LLM_RUNTIME_LIVE=1 to run the live provider-runtime matrix")
    return LiveEnv(selected_providers=_selected_providers())


def _selected_providers() -> frozenset[ProviderName]:
    raw = os.environ.get("LLM_RUNTIME_LIVE_PROVIDERS")
    if not raw:
        return frozenset(_PROVIDER_ORDER)

    requested = {name.strip().lower() for name in raw.split(",") if name.strip()}
    known = set(_PROVIDER_ORDER)
    unknown = requested - known
    if unknown:
        pytest.fail(
            "Unknown LLM_RUNTIME_LIVE_PROVIDERS value(s): "
            f"{', '.join(sorted(unknown))}. Expected any of: {', '.join(_PROVIDER_ORDER)}"
        )
    return frozenset(cast(ProviderName, name) for name in requested)


def _provider_cases() -> tuple[ProviderCase, ...]:
    return tuple(
        ProviderCase(provider=provider, capability=_representative_capability(provider))
        for provider in _PROVIDER_ORDER
    )


def _representative_capability(provider: ProviderName) -> ModelCapability:
    key_probe_model = DEFAULT_CATALOG.key_probe_model(provider)
    entries = [
        entry
        for entry in DEFAULT_CATALOG.entries
        if entry.provider == provider and not entry.embeddings and entry.max_output_tokens != 0
    ]
    if key_probe_model:
        for entry in entries:
            if entry.model == key_probe_model:
                return entry
    if entries:
        return entries[0]
    raise AssertionError(f"No generation model configured for live provider {provider}")


def _reasoning_capability(provider: ProviderName) -> ModelCapability:
    entries = [
        entry
        for entry in DEFAULT_CATALOG.entries
        if entry.provider == provider and not entry.embeddings and entry.max_output_tokens != 0
    ]
    if provider == "anthropic":
        for entry in entries:
            if entry.model in _ANTHROPIC_ADAPTIVE_THINKING_MODELS:
                return entry
    return _representative_capability(provider)


def _runtime(http: httpx.AsyncClient) -> ModelRuntime:
    return ModelRuntime(
        http,
        cloudflare_account_id=os.environ.get("CLOUDFLARE_AI_ACCOUNT_ID"),
    )


def _call(
    case: ProviderCase,
    messages: list[ModelMessage],
    *,
    max_output_tokens: int = 96,
    reasoning: ReasoningEffort = "none",
    retry: RetryPolicy | None = None,
    structured_output: StructuredOutputSpec | None = None,
    tools: tuple[ToolSpec, ...] = (),
    tool_choice: _ToolChoice = "auto",
) -> ModelCall:
    return ModelCall(
        model=ModelRef(provider=case.provider, model=case.model),
        messages=messages,
        max_output_tokens=max_output_tokens,
        reasoning=ReasoningConfig(effort=reasoning),
        retry=retry or RetryPolicy(max_attempts=1, initial_delay_s=0),
        structured_output=structured_output,
        tools=tools,
        tool_choice=tool_choice,
    )


def _text_call(
    case: ProviderCase,
    prompt: str,
    *,
    max_output_tokens: int = 96,
    reasoning: ReasoningEffort = "none",
    retry: RetryPolicy | None = None,
) -> ModelCall:
    return _call(
        case,
        [ModelMessage(role="user", content=prompt)],
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        retry=retry,
    )


def _highest_reasoning_effort(capability: ModelCapability) -> ReasoningEffort:
    for effort in _REASONING_ORDER:
        if effort in capability.reasoning_modes:
            return effort
    raise AssertionError(
        f"No reasoning mode configured for {capability.provider}/{capability.model}"
    )


def _assert_text_response(case: ProviderCase, text: str) -> None:
    assert text.strip(), f"{case.provider}/{case.model} returned empty text"


def _assert_usage_if_claimed(case: ProviderCase, usage: object | None) -> None:
    if case.capability.usage_input_output_tokens:
        assert usage is not None, f"{case.provider}/{case.model} did not return usage"


@pytest.mark.parametrize("case", _provider_cases(), ids=lambda case: case.provider)
async def test_live_default_send(live_env: LiveEnv, case: ProviderCase) -> None:
    key = live_env.key_for(case.provider)
    async with httpx.AsyncClient() as http:
        response = await _runtime(http).generate(
            _text_call(
                case,
                "Reply with a short sentence containing the word nexus.",
                reasoning="default",
            ),
            key=key,
            timeout_s=60,
        )

    _assert_text_response(case, response.text)
    _assert_usage_if_claimed(case, response.usage)
    assert response.status not in {"failed", "incomplete", "error"}


@pytest.mark.parametrize("provider", _PROVIDER_ORDER)
async def test_live_highest_reasoning_send(live_env: LiveEnv, provider: ProviderName) -> None:
    capability = _reasoning_capability(provider)
    case = ProviderCase(provider=provider, capability=capability)
    effort = _highest_reasoning_effort(capability)
    if effort in ("default", "none"):
        pytest.skip(f"{provider}/{case.model} has no explicit reasoning mode above none")

    key = live_env.key_for(provider)
    async with httpx.AsyncClient() as http:
        response = await _runtime(http).generate(
            _text_call(
                case,
                "Answer in one sentence: what is two plus two?",
                max_output_tokens=160,
                reasoning=effort,
            ),
            key=key,
            timeout_s=90,
        )

    _assert_text_response(case, response.text)


@pytest.mark.parametrize("case", _provider_cases(), ids=lambda case: case.provider)
async def test_live_cacheable_prompt(live_env: LiveEnv, case: ProviderCase) -> None:
    if not case.capability.prompt_cache.supported:
        pytest.skip(f"{case.provider}/{case.model} does not support prompt caching")
    key = live_env.key_for(case.provider)

    messages = [
        ModelMessage(
            role="system",
            content="Stable live-provider cache prefix. Reply concisely.",
            cache_ttl=case.capability.prompt_cache.ttl_options[0],
        ),
        ModelMessage(role="user", content="Reply with the word cached once."),
    ]
    async with httpx.AsyncClient() as http:
        response = await _runtime(http).generate(
            _call(case, messages, max_output_tokens=48, reasoning="none"),
            key=key,
            timeout_s=60,
        )

    _assert_text_response(case, response.text)
    _assert_usage_if_claimed(case, response.usage)


@pytest.mark.parametrize("case", _provider_cases(), ids=lambda case: case.provider)
async def test_live_streaming_text(live_env: LiveEnv, case: ProviderCase) -> None:
    if not case.capability.streaming:
        pytest.skip(f"{case.provider}/{case.model} does not support streaming")
    key = live_env.key_for(case.provider)
    chunks = []
    terminal_seen = False

    async with httpx.AsyncClient() as http:
        async for chunk in _runtime(http).stream(
            _text_call(case, "Stream one short sentence.", max_output_tokens=64),
            key=key,
            timeout_s=60,
        ):
            if chunk.delta_text:
                chunks.append(chunk.delta_text)
            if chunk.done:
                terminal_seen = True
                _assert_usage_if_claimed(case, chunk.usage)

    assert terminal_seen, f"{case.provider}/{case.model} stream did not emit terminal chunk"
    _assert_text_response(case, "".join(chunks))


@pytest.mark.parametrize("case", _provider_cases(), ids=lambda case: case.provider)
async def test_live_forced_tool_call_and_continuation(
    live_env: LiveEnv,
    case: ProviderCase,
) -> None:
    if not case.capability.tool_calling or not case.capability.tool_choice_required:
        pytest.skip(f"{case.provider}/{case.model} does not support required tool calls")
    key = live_env.key_for(case.provider)
    tool = ToolSpec(
        name="lookup_weather",
        description="Look up a compact weather summary.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )
    user_turn = ModelMessage(role="user", content="Use the tool for weather in Paris.")

    async with httpx.AsyncClient() as http:
        runtime = _runtime(http)
        first = await runtime.generate(
            _call(
                case,
                [user_turn],
                max_output_tokens=128,
                reasoning="none",
                tools=(tool,),
                tool_choice="required",
            ),
            key=key,
            timeout_s=60,
        )
        assert first.tool_calls, f"{case.provider}/{case.model} did not return a tool call"
        tool_call = first.tool_calls[0]

        final = await runtime.generate(
            _call(
                case,
                [
                    user_turn,
                    ModelMessage(role="assistant", content=first.text, tool_calls=(tool_call,)),
                    ModelMessage(
                        role="tool",
                        tool_results=(
                            ToolResult(
                                call_id=tool_call.id, output="Paris weather: mild and clear."
                            ),
                        ),
                    ),
                ],
                max_output_tokens=96,
                reasoning="none",
            ),
            key=key,
            timeout_s=60,
        )

    _assert_text_response(case, final.text)


@pytest.mark.parametrize("case", _provider_cases(), ids=lambda case: case.provider)
async def test_live_structured_output_where_supported(
    live_env: LiveEnv,
    case: ProviderCase,
) -> None:
    if not case.capability.structured_output:
        pytest.skip(f"{case.provider}/{case.model} does not support structured output")
    key = live_env.key_for(case.provider)
    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "summary": {"type": "string"},
        },
        "required": ["ok", "summary"],
        "additionalProperties": False,
    }

    async with httpx.AsyncClient() as http:
        response = await _runtime(http).generate(
            _call(
                case,
                [ModelMessage(role="user", content="Return ok=true and a two-word summary.")],
                max_output_tokens=96,
                reasoning="none",
                structured_output=StructuredOutputSpec(
                    name="live_provider_result",
                    schema=schema,
                    strict=True,
                ),
            ),
            key=key,
            timeout_s=60,
        )

    assert isinstance(response.structured_output, dict), (
        f"{case.provider}/{case.model} did not return parsed structured output"
    )
    assert response.structured_output.get("ok") is True
    assert isinstance(response.structured_output.get("summary"), str)


@pytest.mark.parametrize("case", _provider_cases(), ids=lambda case: case.provider)
async def test_live_invalid_key_maps_to_invalid_key(live_env: LiveEnv, case: ProviderCase) -> None:
    live_env.key_for(case.provider)
    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await _runtime(http).generate(
                _text_call(case, "This call must fail before model output.", max_output_tokens=8),
                key=ProviderApiKey("invalid-live-provider-key", source="test"),
                timeout_s=30,
            )

    assert exc_info.value.error_code == ModelCallErrorCode.INVALID_KEY
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("case", _provider_cases(), ids=lambda case: case.provider)
async def test_live_timeout_maps_to_timeout(live_env: LiveEnv, case: ProviderCase) -> None:
    key = live_env.key_for(case.provider)
    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await _runtime(http).generate(
                _text_call(
                    case,
                    "This call intentionally uses an impossible timeout.",
                    max_output_tokens=8,
                    retry=RetryPolicy(max_attempts=2, initial_delay_s=0, max_delay_s=0),
                ),
                key=key,
                timeout_s=0,
            )

    assert exc_info.value.error_code == ModelCallErrorCode.TIMEOUT
    assert exc_info.value.retryable is True
