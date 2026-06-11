import httpx
import pytest
import respx

from provider_runtime import ModelRuntime, ProviderApiKey, ProviderBaseUrls
from provider_runtime.embeddings import EmbeddingsClient
from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.types import EmbeddingCall, ModelRef, RetryPolicy

pytestmark = pytest.mark.asyncio

KEY = ProviderApiKey("sk-test", source="test")
CF_KEY = ProviderApiKey("cf-test", source="test")


def call(provider: str = "openai", *, retry: RetryPolicy | None = None) -> EmbeddingCall:
    return EmbeddingCall(
        model=ModelRef(provider=provider, model="text-embedding-3-small"),  # type: ignore[arg-type]
        inputs=["alpha", "beta"],
        dimensions=256,
        retry=retry or RetryPolicy(max_attempts=1),
    )


@respx.mock
async def test_openai_compatible_embeddings_success() -> None:
    respx.post("https://embeddings.test/v1/embeddings").respond(
        200,
        json={
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        },
        headers={"x-request-id": "req-emb-1"},
    )

    async with httpx.AsyncClient() as http:
        response = await EmbeddingsClient(
            http, provider="openai", base_url="https://embeddings.test/v1"
        ).embed(call(), api_key="sk-test", timeout_s=30)

    assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert respx.calls.last.request.content == (
        b'{"model":"text-embedding-3-small","input":["alpha","beta"],"dimensions":256}'
    )
    assert response.usage is not None
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens is None
    assert response.usage.total_tokens == 3
    assert response.provider_request_id == "req-emb-1"


@respx.mock
async def test_runtime_embeds_with_cloudflare_base_url() -> None:
    respx.post("https://cloudflare.test/v1/embeddings").respond(
        200,
        json={
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
    )

    async with httpx.AsyncClient() as http:
        response = await ModelRuntime(http, cloudflare_base_url="https://cloudflare.test/v1").embed(
            call("cloudflare"), key=CF_KEY
        )

    assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]


@respx.mock
async def test_runtime_embeds_with_openai_base_url() -> None:
    route = respx.post("https://openai-proxy.test/v1/embeddings").respond(
        200,
        json={
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
    )

    async with httpx.AsyncClient() as http:
        response = await ModelRuntime(
            http,
            base_urls=ProviderBaseUrls(openai="https://openai-proxy.test/v1"),
        ).embed(call(), key=KEY)

    assert route.called
    assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]


@respx.mock
async def test_terminal_parser_exceptions_do_not_retry_embedding_calls() -> None:
    route = respx.post("https://api.openai.com/v1/embeddings").mock(
        side_effect=TypeError("'NoneType' object is not subscriptable")
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await ModelRuntime(http).embed(
                call(retry=RetryPolicy(max_attempts=3, initial_delay_s=0)),
                key=KEY,
            )

    assert route.call_count == 1
    assert exc_info.value.retryable is False
    assert [attempt.status for attempt in exc_info.value.attempts] == ["terminal_error"]


@respx.mock
async def test_runtime_retries_retryable_embedding_errors_before_success() -> None:
    route = respx.post("https://api.openai.com/v1/embeddings").mock(
        side_effect=[
            httpx.Response(503, json={"error": {"message": "temporarily unavailable"}}),
            httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2]},
                        {"index": 1, "embedding": [0.3, 0.4]},
                    ],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            ),
        ]
    )

    async with httpx.AsyncClient() as http:
        response = await ModelRuntime(http).embed(
            call(retry=RetryPolicy(max_attempts=2, initial_delay_s=0)),
            key=KEY,
        )

    assert route.call_count == 2
    assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert [attempt.status for attempt in response.attempts] == ["retryable_error", "success"]
    assert response.attempts[0].error_code == ModelCallErrorCode.PROVIDER_DOWN.value
    assert response.retry_count == 1


async def test_runtime_rejects_unconfigured_embedding_provider() -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await ModelRuntime(http).embed(
                EmbeddingCall(
                    model=ModelRef(provider="anthropic", model="claude-3-opus-20240229"),
                    inputs=["x"],
                ),
                key=KEY,
            )

    assert exc_info.value.error_code == ModelCallErrorCode.MODEL_NOT_AVAILABLE
