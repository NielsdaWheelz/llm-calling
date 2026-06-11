"""Private concrete provider adapter runtime."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx

from provider_runtime.anthropic import AnthropicClient
from provider_runtime.cloudflare import CloudflareClient, cloudflare_ai_base_url
from provider_runtime.embeddings import EmbeddingsClient
from provider_runtime.endpoints import ANTHROPIC_BASE_URL, GEMINI_BASE_URL, OPENAI_BASE_URL
from provider_runtime.errors import ModelCallError, ModelCallErrorCode, classify_provider_error
from provider_runtime.gemini import GeminiClient
from provider_runtime.openai import OpenAIClient
from provider_runtime.openrouter import OPENROUTER_BASE_URL, OpenRouterClient
from provider_runtime.types import (
    EmbeddingCall,
    EmbeddingResponse,
    ModelCall,
    ModelChunk,
    ModelResponse,
    ProviderName,
)

DEFAULT_TIMEOUT_S = 45
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class _AdapterRuntime:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_gemini: bool = True,
        enable_openrouter: bool = True,
        enable_cloudflare: bool = True,
        openai_base_url: str = OPENAI_BASE_URL,
        anthropic_base_url: str = ANTHROPIC_BASE_URL,
        gemini_base_url: str = GEMINI_BASE_URL,
        openrouter_base_url: str = OPENROUTER_BASE_URL,
        cloudflare_base_url: str | None = None,
        cloudflare_account_id: str | None = None,
    ):
        if cloudflare_base_url is None and cloudflare_account_id:
            cloudflare_base_url = cloudflare_ai_base_url(cloudflare_account_id)
        self._openai = OpenAIClient(client, base_url=openai_base_url)
        self._anthropic = AnthropicClient(client, base_url=anthropic_base_url)
        self._gemini = GeminiClient(client, base_url=gemini_base_url)
        self._openrouter = OpenRouterClient(client, base_url=openrouter_base_url)
        self._cloudflare = (
            CloudflareClient(client, base_url=cloudflare_base_url)
            if cloudflare_base_url is not None
            else None
        )
        self._openai_embeddings = EmbeddingsClient(
            client, provider="openai", base_url=openai_base_url
        )
        self._cloudflare_embeddings = (
            EmbeddingsClient(client, provider="cloudflare", base_url=cloudflare_base_url)
            if cloudflare_base_url is not None
            else None
        )
        self._enable_openai = enable_openai
        self._enable_anthropic = enable_anthropic
        self._enable_gemini = enable_gemini
        self._enable_openrouter = enable_openrouter
        self._enable_cloudflare = enable_cloudflare

    def is_provider_available(self, provider: str) -> bool:
        if provider == "openai":
            return self._enable_openai
        if provider == "anthropic":
            return self._enable_anthropic
        if provider == "gemini":
            return self._enable_gemini
        if provider == "openrouter":
            return self._enable_openrouter
        if provider == "cloudflare":
            return self._enable_cloudflare and self._cloudflare is not None
        return False

    async def generate(
        self,
        call: ModelCall,
        *,
        key: str,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> ModelResponse:
        provider = self._resolve_provider(call.model.route or call.model.provider)
        client = self._resolve_client(provider)
        return await _retry_call(
            call.retry.max_attempts,
            call.retry.initial_delay_s,
            call.retry.max_delay_s,
            lambda: client.generate(call, api_key=key, timeout_s=timeout_s),
            wrap=lambda exc: _wrap_generate_error(provider, exc),
        )

    async def stream(
        self,
        call: ModelCall,
        *,
        key: str,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> AsyncIterator[ModelChunk]:
        provider = self._resolve_provider(call.model.route or call.model.provider)
        client = self._resolve_client(provider)
        attempts = max(1, call.retry.max_attempts)
        for attempt in range(1, attempts + 1):
            emitted_chunk = False
            try:
                async for chunk in client.generate_stream(call, api_key=key, timeout_s=timeout_s):
                    emitted_chunk = True
                    yield chunk
                return
            except Exception as raw_exc:
                exc = _wrap_stream_error(provider, raw_exc)
                if emitted_chunk or attempt >= attempts or not exc.retryable:
                    raise exc from raw_exc
                await _sleep_before_retry(
                    attempt=attempt,
                    error=exc,
                    initial_delay_s=call.retry.initial_delay_s,
                    max_delay_s=call.retry.max_delay_s,
                )

    async def embed(
        self,
        call: EmbeddingCall,
        *,
        key: str,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> EmbeddingResponse:
        provider = self._resolve_provider(call.model.route or call.model.provider)
        if provider == "openai":
            client = self._openai_embeddings
        elif provider == "cloudflare" and self._cloudflare_embeddings is not None:
            client = self._cloudflare_embeddings
        else:
            raise ModelCallError(
                ModelCallErrorCode.MODEL_NOT_AVAILABLE,
                f"Embeddings are not configured for provider {provider}",
                provider=provider,
                retryable=False,
            )

        return await _retry_call(
            call.retry.max_attempts,
            call.retry.initial_delay_s,
            call.retry.max_delay_s,
            lambda: client.embed(call, api_key=key, timeout_s=timeout_s),
            wrap=lambda exc: _wrap_embedding_error(provider, exc),
        )

    def _resolve_provider(self, provider: str) -> ProviderName:
        if provider == "openai":
            if self._enable_openai:
                return "openai"
            raise _provider_disabled(provider)
        if provider == "anthropic":
            if self._enable_anthropic:
                return "anthropic"
            raise _provider_disabled(provider)
        if provider == "gemini":
            if self._enable_gemini:
                return "gemini"
            raise _provider_disabled(provider)
        if provider == "openrouter":
            if self._enable_openrouter:
                return "openrouter"
            raise _provider_disabled(provider)
        if provider == "cloudflare":
            if self._enable_cloudflare and self._cloudflare is not None:
                return "cloudflare"
            raise _provider_disabled(provider)
        raise ModelCallError(
            ModelCallErrorCode.MODEL_NOT_AVAILABLE,
            f"Unknown provider: {provider}",
            provider=provider,
            retryable=False,
        )

    def _resolve_client(
        self, provider: ProviderName
    ) -> OpenAIClient | AnthropicClient | GeminiClient | OpenRouterClient | CloudflareClient:
        if provider == "openai":
            return self._openai
        if provider == "anthropic":
            return self._anthropic
        if provider == "gemini":
            return self._gemini
        if provider == "openrouter":
            return self._openrouter
        if provider == "cloudflare" and self._cloudflare is not None:
            return self._cloudflare
        raise _provider_disabled(provider)


async def _retry_call[T](
    max_attempts: int,
    initial_delay_s: float,
    max_delay_s: float,
    fn: Callable[[], Awaitable[T]],
    *,
    wrap: Callable[[Exception], ModelCallError],
) -> T:
    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as raw_exc:
            exc = wrap(raw_exc)
            if attempt >= attempts or not exc.retryable:
                raise exc from raw_exc
            await _sleep_before_retry(
                attempt=attempt,
                error=exc,
                initial_delay_s=initial_delay_s,
                max_delay_s=max_delay_s,
            )

    raise AssertionError("unreachable retry loop exit")


def _wrap_generate_error(provider: ProviderName, exc: Exception) -> ModelCallError:
    if isinstance(exc, httpx.TimeoutException):
        return ModelCallError(ModelCallErrorCode.TIMEOUT, "Request timed out", provider=provider)
    if isinstance(exc, httpx.HTTPStatusError):
        return _http_status_model_error(provider, exc)
    if isinstance(exc, httpx.NetworkError):
        return ModelCallError(ModelCallErrorCode.PROVIDER_DOWN, "Network error", provider=provider)
    if isinstance(exc, ModelCallError):
        return exc
    if isinstance(exc, (json.JSONDecodeError, KeyError, IndexError, ValueError)):
        return ModelCallError(
            ModelCallErrorCode.PROVIDER_DOWN,
            f"Response parsing error: {type(exc).__name__}",
            provider=provider,
            retryable=False,
        )
    if isinstance(exc, (httpx.HTTPError, httpx.StreamError, TypeError, AttributeError)):
        return ModelCallError(
            ModelCallErrorCode.PROVIDER_DOWN,
            f"Transport error: {type(exc).__name__}: {exc}",
            provider=provider,
        )
    return ModelCallError(
        ModelCallErrorCode.PROVIDER_DOWN,
        f"Unexpected runtime error: {type(exc).__name__}: {exc}",
        provider=provider,
    )


def _wrap_stream_error(provider: ProviderName, exc: Exception) -> ModelCallError:
    if isinstance(exc, httpx.TimeoutException):
        return ModelCallError(ModelCallErrorCode.TIMEOUT, "Stream timed out", provider=provider)
    if isinstance(exc, httpx.HTTPStatusError):
        return _http_status_model_error(provider, exc)
    if isinstance(exc, httpx.NetworkError):
        return ModelCallError(
            ModelCallErrorCode.PROVIDER_DOWN,
            "Network error during stream",
            provider=provider,
        )
    if isinstance(exc, ModelCallError):
        return exc
    if isinstance(exc, (json.JSONDecodeError, KeyError, IndexError, ValueError)):
        return ModelCallError(
            ModelCallErrorCode.PROVIDER_DOWN,
            f"Stream parsing error: {type(exc).__name__}",
            provider=provider,
            retryable=False,
        )
    if isinstance(exc, (httpx.HTTPError, httpx.StreamError, TypeError, AttributeError)):
        return ModelCallError(
            ModelCallErrorCode.PROVIDER_DOWN,
            f"Stream transport error: {type(exc).__name__}: {exc}",
            provider=provider,
        )
    return ModelCallError(
        ModelCallErrorCode.PROVIDER_DOWN,
        f"Unexpected stream error: {type(exc).__name__}: {exc}",
        provider=provider,
    )


def _wrap_embedding_error(provider: ProviderName, exc: Exception) -> ModelCallError:
    if isinstance(exc, httpx.TimeoutException):
        return ModelCallError(
            ModelCallErrorCode.TIMEOUT, "Embedding request timed out", provider=provider
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return _http_status_model_error(provider, exc)
    if isinstance(exc, ModelCallError):
        return exc
    if isinstance(exc, (json.JSONDecodeError, KeyError, IndexError, ValueError)):
        return ModelCallError(
            ModelCallErrorCode.PROVIDER_DOWN,
            f"Embedding response parsing error: {type(exc).__name__}",
            provider=provider,
            retryable=False,
        )
    if isinstance(exc, (httpx.HTTPError, httpx.StreamError, TypeError, AttributeError)):
        return ModelCallError(
            ModelCallErrorCode.PROVIDER_DOWN,
            f"Embedding transport error: {type(exc).__name__}: {exc}",
            provider=provider,
        )
    return ModelCallError(
        ModelCallErrorCode.PROVIDER_DOWN,
        f"Unexpected embedding error: {type(exc).__name__}: {exc}",
        provider=provider,
    )


def _http_status_model_error(provider: ProviderName, exc: httpx.HTTPStatusError) -> ModelCallError:
    response = exc.response
    code = classify_provider_error(
        provider,
        response.status_code,
        _parse_json_or_none(response),
        None,
    )
    return ModelCallError(
        code,
        f"Provider returned HTTP {response.status_code}",
        provider=provider,
        status_code=response.status_code,
        retry_after_seconds=_retry_after_seconds(response.headers.get("retry-after")),
        provider_request_id=response.headers.get("x-request-id")
        or response.headers.get("request-id"),
        retryable=response.status_code in _RETRYABLE_STATUS_CODES
        and code
        in {
            ModelCallErrorCode.RATE_LIMIT,
            ModelCallErrorCode.TIMEOUT,
            ModelCallErrorCode.PROVIDER_DOWN,
        },
    )


async def _sleep_before_retry(
    *,
    attempt: int,
    error: ModelCallError,
    initial_delay_s: float,
    max_delay_s: float,
) -> None:
    delay = error.retry_after_seconds
    if delay is None:
        delay = initial_delay_s * (2 ** max(0, attempt - 1))
    delay = min(delay, max_delay_s)
    if delay > 0:
        await asyncio.sleep(delay)


def _provider_disabled(provider: str) -> ModelCallError:
    return ModelCallError(
        ModelCallErrorCode.MODEL_NOT_AVAILABLE,
        f"Provider {provider} is disabled",
        provider=provider,
        retryable=False,
    )


def _parse_json_or_none(response: httpx.Response) -> dict | None:
    try:
        parsed = response.json()
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _retry_after_seconds(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None
