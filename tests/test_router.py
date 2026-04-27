import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_calling.errors import LLMError, LLMErrorCode
from llm_calling.router import LLMRouter
from llm_calling.types import LLMRequest, Turn

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures"


def request(provider: str) -> LLMRequest:
    model_name = {
        "openai": "gpt-5.4-mini",
        "anthropic": "claude-3-opus-20240229",
        "gemini": "gemini-2.5-pro",
        "deepseek": "deepseek-chat",
    }[provider]
    return LLMRequest(
        model_name=model_name,
        messages=[Turn(role="user", content="Hello!")],
        max_tokens=100,
    )


def endpoint(provider: str, model_name: str) -> str:
    if provider == "openai":
        return "https://api.openai.com/v1/responses"
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/messages"
    if provider == "gemini":
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        )
    if provider == "deepseek":
        return "https://api.deepseek.com/chat/completions"
    raise AssertionError(f"unknown provider in test: {provider}")


def fixture(provider: str, name: str) -> dict:
    return json.loads((FIXTURES / provider / name).read_text())


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "deepseek"])
@pytest.mark.parametrize(
    ("status_code", "fixture_name", "expected_code"),
    [
        (401, "error_401.json", LLMErrorCode.INVALID_KEY),
        (429, "error_429.json", LLMErrorCode.RATE_LIMIT),
        (400, "error_context_too_large.json", LLMErrorCode.CONTEXT_TOO_LARGE),
        (500, "error_500.json", LLMErrorCode.PROVIDER_DOWN),
        (404, "error_500.json", LLMErrorCode.MODEL_NOT_AVAILABLE),
    ],
)
@respx.mock
async def test_generate_maps_provider_errors(
    provider: str,
    status_code: int,
    fixture_name: str,
    expected_code: LLMErrorCode,
) -> None:
    req = request(provider)
    respx.post(endpoint(provider, req.model_name)).respond(
        status_code,
        json=fixture(provider, fixture_name),
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError) as exc_info:
            await LLMRouter(http).generate(provider, req, "sk-test")

    assert exc_info.value.error_code == expected_code
    assert exc_info.value.provider == provider


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "deepseek"])
@respx.mock
async def test_generate_maps_timeout(provider: str) -> None:
    req = request(provider)
    respx.post(endpoint(provider, req.model_name)).mock(side_effect=httpx.ReadTimeout("timeout"))

    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError) as exc_info:
            await LLMRouter(http).generate(provider, req, "sk-test")

    assert exc_info.value.error_code == LLMErrorCode.TIMEOUT


async def test_unknown_provider_is_model_not_available() -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError) as exc_info:
            await LLMRouter(http).generate("unknown", request("openai"), "sk-test")

    assert exc_info.value.error_code == LLMErrorCode.MODEL_NOT_AVAILABLE


async def test_disabled_provider_is_model_not_available() -> None:
    async with httpx.AsyncClient() as http:
        router = LLMRouter(http, enable_openai=False)
        with pytest.raises(LLMError) as exc_info:
            await router.generate("openai", request("openai"), "sk-test")

    assert exc_info.value.error_code == LLMErrorCode.MODEL_NOT_AVAILABLE
