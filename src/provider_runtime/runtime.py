"""Public provider-runtime facade."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import cast

import httpx

from provider_runtime._adapter_runtime import DEFAULT_TIMEOUT_S, _AdapterRuntime
from provider_runtime.catalog import DEFAULT_CATALOG, ModelCapability, ModelCatalog
from provider_runtime.endpoints import ANTHROPIC_BASE_URL, GEMINI_BASE_URL, OPENAI_BASE_URL
from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.lowering import lower_generate_request
from provider_runtime.openrouter import OPENROUTER_BASE_URL
from provider_runtime.types import (
    EmbeddingCall,
    EmbeddingResponse,
    KeyProbeResult,
    ModelCall,
    ModelChunk,
    ModelMessage,
    ModelRef,
    ModelResponse,
    ProviderApiKey,
    ProviderName,
    ReasoningConfig,
    RetryPolicy,
)

_PROVIDERS: frozenset[str] = frozenset(
    ("openai", "anthropic", "gemini", "openrouter", "cloudflare")
)


@dataclass(frozen=True)
class ProviderBaseUrls:
    openai: str = OPENAI_BASE_URL
    anthropic: str = ANTHROPIC_BASE_URL
    gemini: str = GEMINI_BASE_URL
    openrouter: str = OPENROUTER_BASE_URL
    cloudflare: str | None = None


class ModelRuntime(_AdapterRuntime):
    """Validated public runtime facade over the current provider adapters."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        catalog: ModelCatalog = DEFAULT_CATALOG,
        base_urls: ProviderBaseUrls | None = None,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_gemini: bool = True,
        enable_openrouter: bool = True,
        enable_cloudflare: bool = True,
        openai_base_url: str | None = None,
        anthropic_base_url: str | None = None,
        gemini_base_url: str | None = None,
        openrouter_base_url: str | None = None,
        cloudflare_base_url: str | None = None,
        cloudflare_account_id: str | None = None,
    ):
        urls = base_urls or ProviderBaseUrls()
        self._catalog = catalog
        super().__init__(
            client,
            enable_openai=enable_openai,
            enable_anthropic=enable_anthropic,
            enable_gemini=enable_gemini,
            enable_openrouter=enable_openrouter,
            enable_cloudflare=enable_cloudflare,
            openai_base_url=openai_base_url or urls.openai,
            anthropic_base_url=anthropic_base_url or urls.anthropic,
            gemini_base_url=gemini_base_url or urls.gemini,
            openrouter_base_url=openrouter_base_url or urls.openrouter,
            cloudflare_base_url=cloudflare_base_url or urls.cloudflare,
            cloudflare_account_id=cloudflare_account_id,
        )

    def capabilities(self, model: ModelRef) -> ModelCapability | None:
        return self._catalog.capabilities(self._catalog_ref(model))

    async def probe_key(
        self,
        *,
        provider: ProviderName,
        key: ProviderApiKey,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> KeyProbeResult:
        model = self._catalog.key_probe_model(provider)
        if model is None:
            return KeyProbeResult(
                provider=provider,
                model="",
                ok=False,
                error_code=ModelCallErrorCode.MODEL_NOT_AVAILABLE.value,
            )
        call = ModelCall(
            model=ModelRef(provider=provider, model=model),
            messages=[ModelMessage(role="user", content="Reply with ok.")],
            max_output_tokens=8,
            reasoning=ReasoningConfig(effort="none"),
            retry=RetryPolicy(max_attempts=1),
        )
        try:
            response = await self.generate(call, key=key, timeout_s=timeout_s)
        except ModelCallError as exc:
            return KeyProbeResult(
                provider=provider,
                model=model,
                ok=False,
                error_code=exc.error_code.value,
                provider_request_id=exc.provider_request_id,
                attempts=exc.attempts,
            )
        return KeyProbeResult(
            provider=provider,
            model=model,
            ok=True,
            provider_request_id=response.provider_request_id,
            usage=response.usage,
            attempts=response.attempts,
        )

    async def generate(
        self,
        call: ModelCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> ModelResponse:
        capabilities = self._require_generate_capabilities(call, streaming=False)
        plan = lower_generate_request(call, capabilities, streaming=False)
        return await super().generate(plan.call, key=key, timeout_s=timeout_s)

    async def stream(
        self,
        call: ModelCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> AsyncIterator[ModelChunk]:
        capabilities = self._require_generate_capabilities(call, streaming=True)
        plan = lower_generate_request(call, capabilities, streaming=True)
        async for chunk in super().stream(plan.call, key=key, timeout_s=timeout_s):
            yield chunk

    async def embed(
        self,
        call: EmbeddingCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> EmbeddingResponse:
        capabilities = self._require_embedding_capabilities(call)
        if not capabilities.embeddings:
            raise ModelCallError(
                ModelCallErrorCode.MODEL_NOT_AVAILABLE,
                f"Embeddings are not configured for {capabilities.provider}/{capabilities.model}",
                provider=capabilities.provider,
                retryable=False,
            )
        return await super().embed(call, key=key, timeout_s=timeout_s)

    def _require_generate_capabilities(
        self,
        call: ModelCall,
        *,
        streaming: bool,
    ) -> ModelCapability:
        return self._require_capabilities(
            call.model, operation="stream" if streaming else "generate"
        )

    def _require_embedding_capabilities(self, call: EmbeddingCall) -> ModelCapability:
        return self._require_capabilities(call.model, operation="embed")

    def _require_capabilities(self, model: ModelRef, *, operation: str) -> ModelCapability:
        catalog_ref = self._catalog_ref(model)
        capabilities = self._catalog.capabilities(catalog_ref)
        if capabilities is None:
            raise ModelCallError(
                ModelCallErrorCode.MODEL_NOT_AVAILABLE,
                f"Unknown model for {operation}: {catalog_ref.provider}/{catalog_ref.model}",
                provider=catalog_ref.provider,
                retryable=False,
            )
        return capabilities

    def _catalog_ref(self, model: ModelRef) -> ModelRef:
        provider = model.route or model.provider
        if provider not in _PROVIDERS:
            raise ModelCallError(
                ModelCallErrorCode.MODEL_NOT_AVAILABLE,
                f"Unknown provider: {provider}",
                provider=provider,
                retryable=False,
            )
        return ModelRef(provider=cast(ProviderName, provider), model=model.model)
