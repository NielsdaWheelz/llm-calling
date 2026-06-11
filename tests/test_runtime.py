import json
from pathlib import Path

import httpx
import pytest
import respx

from provider_runtime import ModelRuntime, ProviderBaseUrls
from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.types import ModelCall, ModelMessage, ModelRef, RetryPolicy

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures"


def request(provider: str, *, retry: RetryPolicy | None = None) -> ModelCall:
    model_name = {
        "openai": "gpt-5.4-mini",
        "anthropic": "claude-3-opus-20240229",
        "gemini": "gemini-2.5-pro",
        "openrouter": "moonshotai/kimi-k2.6",
        "cloudflare": "@cf/meta/llama-3.1-8b-instruct",
    }[provider]
    return ModelCall(
        model=ModelRef(provider=provider, model=model_name),  # type: ignore[arg-type]
        messages=[ModelMessage(role="user", content="Hello!")],
        max_output_tokens=100,
        retry=retry or RetryPolicy(max_attempts=1),
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
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1/chat/completions"
    if provider == "cloudflare":
        return "https://cloudflare.test/v1/chat/completions"
    raise AssertionError(f"unknown provider in test: {provider}")


def fixture(provider: str, name: str) -> dict:
    folder = "openai_compatible" if provider in ("openrouter", "cloudflare") else provider
    return json.loads((FIXTURES / folder / name).read_text())


def runtime(http: httpx.AsyncClient) -> ModelRuntime:
    return ModelRuntime(http, cloudflare_base_url="https://cloudflare.test/v1")


def success_fixture(provider: str) -> dict:
    return fixture(provider, "success_nonstream.json")


async def test_capabilities_returns_catalog_entry() -> None:
    async with httpx.AsyncClient() as http:
        capabilities = runtime(http).capabilities(ModelRef(provider="openai", model="gpt-5.4-mini"))

    assert capabilities is not None
    assert capabilities.provider == "openai"
    assert capabilities.model == "gpt-5.4-mini"


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "openrouter", "cloudflare"])
@pytest.mark.parametrize(
    ("status_code", "fixture_name", "expected_code"),
    [
        (401, "error_401.json", ModelCallErrorCode.INVALID_KEY),
        (429, "error_429.json", ModelCallErrorCode.RATE_LIMIT),
        (400, "error_context_too_large.json", ModelCallErrorCode.CONTEXT_TOO_LARGE),
        (500, "error_500.json", ModelCallErrorCode.PROVIDER_DOWN),
        (404, "error_500.json", ModelCallErrorCode.MODEL_NOT_AVAILABLE),
    ],
)
@respx.mock
async def test_generate_maps_provider_errors(
    provider: str,
    status_code: int,
    fixture_name: str,
    expected_code: ModelCallErrorCode,
) -> None:
    req = request(provider)
    respx.post(endpoint(provider, req.model.model)).respond(
        status_code,
        json=fixture(provider, fixture_name),
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await runtime(http).generate(req, key="sk-test")

    assert exc_info.value.error_code == expected_code
    assert exc_info.value.provider == provider
    assert exc_info.value.status_code == status_code


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "openrouter", "cloudflare"])
@respx.mock
async def test_generate_maps_timeout(provider: str) -> None:
    req = request(provider)
    respx.post(endpoint(provider, req.model.model)).mock(side_effect=httpx.ReadTimeout("timeout"))

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await runtime(http).generate(req, key="sk-test")

    assert exc_info.value.error_code == ModelCallErrorCode.TIMEOUT


@pytest.mark.parametrize(
    "exc",
    [
        httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
        httpx.ResponseNotRead(),
        httpx.DecodingError("bad content-encoding"),
        TypeError("'NoneType' object is not subscriptable"),
        AttributeError("'NoneType' object has no attribute 'get'"),
    ],
)
@respx.mock
async def test_generate_wraps_transport_and_payload_exceptions(exc: Exception) -> None:
    req = request("openai")
    respx.post(endpoint("openai", req.model.model)).mock(side_effect=exc)

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await runtime(http).generate(req, key="sk-test")

    assert exc_info.value.error_code == ModelCallErrorCode.PROVIDER_DOWN
    assert type(exc).__name__ in exc_info.value.message
    assert str(exc) in exc_info.value.message


@respx.mock
async def test_stream_wraps_protocol_error() -> None:
    req = request("openai")
    respx.post(endpoint("openai", req.model.model)).mock(
        side_effect=httpx.RemoteProtocolError("peer closed connection")
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            async for _ in runtime(http).stream(req, key="sk-test"):
                pass

    assert exc_info.value.error_code == ModelCallErrorCode.PROVIDER_DOWN
    assert "peer closed connection" in exc_info.value.message


@respx.mock
async def test_generate_retries_retryable_errors_before_success() -> None:
    req = request("openai", retry=RetryPolicy(max_attempts=2, initial_delay_s=0))
    route = respx.post(endpoint("openai", req.model.model)).mock(
        side_effect=[
            httpx.Response(500, json=fixture("openai", "error_500.json")),
            httpx.Response(
                200,
                json=fixture("openai", "success_nonstream.json"),
                headers={"x-request-id": "req-after-retry"},
            ),
        ]
    )

    async with httpx.AsyncClient() as http:
        response = await runtime(http).generate(req, key="sk-test")

    assert route.call_count == 2
    assert response.provider_request_id == "req-after-retry"


@respx.mock
async def test_generate_does_not_retry_quota_exhaustion_429() -> None:
    req = request("openai", retry=RetryPolicy(max_attempts=3, initial_delay_s=0))
    route = respx.post(endpoint("openai", req.model.model)).respond(
        429,
        json={"error": {"code": "insufficient_quota", "type": "insufficient_quota"}},
        headers={"retry-after": "30", "x-request-id": "req-quota"},
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await runtime(http).generate(req, key="sk-test")

    assert route.call_count == 1
    assert exc_info.value.error_code == ModelCallErrorCode.QUOTA_EXCEEDED
    assert exc_info.value.retryable is False
    assert exc_info.value.retry_after_seconds == 30
    assert exc_info.value.provider_request_id == "req-quota"


@respx.mock
async def test_stream_retries_only_before_first_chunk() -> None:
    req = request("openai", retry=RetryPolicy(max_attempts=2, initial_delay_s=0))
    route = respx.post(endpoint("openai", req.model.model)).mock(
        side_effect=[
            httpx.Response(500, json=fixture("openai", "error_500.json")),
            httpx.Response(
                200,
                text=(FIXTURES / "openai" / "success_stream_chunks.txt").read_text(),
                headers={"content-type": "text/event-stream"},
            ),
        ]
    )

    async with httpx.AsyncClient() as http:
        chunks = [chunk async for chunk in runtime(http).stream(req, key="sk-test")]

    assert route.call_count == 2
    assert "".join(chunk.delta_text for chunk in chunks) == "Hello! How can I help?"
    assert chunks[-1].done is True


async def test_unknown_provider_is_model_not_available() -> None:
    req = ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini", route="unknown"),
        messages=[ModelMessage(role="user", content="Hello!")],
        max_output_tokens=100,
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await runtime(http).generate(req, key="sk-test")

    assert exc_info.value.error_code == ModelCallErrorCode.MODEL_NOT_AVAILABLE


async def test_disabled_provider_is_model_not_available() -> None:
    async with httpx.AsyncClient() as http:
        runtime = ModelRuntime(http, enable_openai=False)
        with pytest.raises(ModelCallError) as exc_info:
            await runtime.generate(request("openai"), key="sk-test")

    assert exc_info.value.error_code == ModelCallErrorCode.MODEL_NOT_AVAILABLE


@respx.mock
async def test_probe_key_uses_catalog_probe_model() -> None:
    route = respx.post("https://api.openai.com/v1/responses").respond(
        200,
        json=success_fixture("openai"),
        headers={"x-request-id": "req-probe"},
    )

    async with httpx.AsyncClient() as http:
        result = await runtime(http).probe_key(provider="openai", key="sk-test")

    assert result.ok is True
    assert result.model == "gpt-5.4-mini"
    assert result.provider_request_id == "req-probe"
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"
    assert json.loads(route.calls.last.request.content)["model"] == "gpt-5.4-mini"


@respx.mock
async def test_probe_key_returns_typed_invalid_key_result() -> None:
    respx.post("https://api.openai.com/v1/responses").respond(
        401,
        json=fixture("openai", "error_401.json"),
    )

    async with httpx.AsyncClient() as http:
        result = await runtime(http).probe_key(provider="openai", key="bad-key")

    assert result.ok is False
    assert result.error_code == ModelCallErrorCode.INVALID_KEY.value


@respx.mock
async def test_runtime_uses_configured_provider_base_urls() -> None:
    routes = {
        "openai": respx.post("https://openai-proxy.test/v1/responses").respond(
            200,
            json=success_fixture("openai"),
        ),
        "anthropic": respx.post("https://anthropic-proxy.test/v1/messages").respond(
            200,
            json=success_fixture("anthropic"),
        ),
        "gemini": respx.post(
            "https://gemini-proxy.test/v1beta/models/gemini-2.5-pro:generateContent"
        ).respond(200, json=success_fixture("gemini")),
        "openrouter": respx.post("https://openrouter-proxy.test/v1/chat/completions").respond(
            200,
            json=success_fixture("openrouter"),
        ),
        "cloudflare": respx.post("https://cloudflare-proxy.test/v1/chat/completions").respond(
            200,
            json=success_fixture("cloudflare"),
        ),
    }
    base_urls = ProviderBaseUrls(
        openai="https://openai-proxy.test/v1",
        anthropic="https://anthropic-proxy.test/v1",
        gemini="https://gemini-proxy.test/v1beta/models",
        openrouter="https://openrouter-proxy.test/v1",
        cloudflare="https://cloudflare-proxy.test/v1",
    )

    async with httpx.AsyncClient() as http:
        configured_runtime = ModelRuntime(http, base_urls=base_urls)
        for provider in routes:
            await configured_runtime.generate(request(provider), key="sk-test")

    assert {provider: route.called for provider, route in routes.items()} == {
        "openai": True,
        "anthropic": True,
        "gemini": True,
        "openrouter": True,
        "cloudflare": True,
    }
