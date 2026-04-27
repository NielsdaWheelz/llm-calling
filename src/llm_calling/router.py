"""Concrete provider router."""

import json
from collections.abc import AsyncIterator

import httpx

from llm_calling.anthropic import AnthropicClient
from llm_calling.deepseek import DeepSeekClient
from llm_calling.errors import LLMError, LLMErrorCode, classify_provider_error
from llm_calling.gemini import GeminiClient
from llm_calling.openai import OpenAIClient
from llm_calling.types import LLMChunk, LLMRequest, LLMResponse, ProviderName

DEFAULT_TIMEOUT_S = 45


class LLMRouter:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_gemini: bool = True,
        enable_deepseek: bool = True,
    ):
        self._openai = OpenAIClient(client)
        self._anthropic = AnthropicClient(client)
        self._gemini = GeminiClient(client)
        self._deepseek = DeepSeekClient(client)
        self._enable_openai = enable_openai
        self._enable_anthropic = enable_anthropic
        self._enable_gemini = enable_gemini
        self._enable_deepseek = enable_deepseek

    def is_provider_available(self, provider: str) -> bool:
        if provider == "openai":
            return self._enable_openai
        if provider == "anthropic":
            return self._enable_anthropic
        if provider == "gemini":
            return self._enable_gemini
        if provider == "deepseek":
            return self._enable_deepseek
        return False

    async def generate(
        self,
        provider: str,
        req: LLMRequest,
        api_key: str,
        *,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> LLMResponse:
        provider_name = self._resolve_provider(provider)
        client = self._resolve_client(provider_name)
        try:
            return await client.generate(req, api_key=api_key, timeout_s=timeout_s)
        except httpx.TimeoutException as exc:
            raise LLMError(LLMErrorCode.TIMEOUT, "Request timed out", provider=provider) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                classify_provider_error(
                    provider,
                    exc.response.status_code,
                    _parse_json_or_none(exc.response),
                    None,
                ),
                f"Provider returned HTTP {exc.response.status_code}",
                provider=provider,
            ) from exc
        except httpx.NetworkError as exc:
            raise LLMError(LLMErrorCode.PROVIDER_DOWN, "Network error", provider=provider) from exc
        except LLMError:
            raise
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(
                LLMErrorCode.PROVIDER_DOWN,
                f"Response parsing error: {type(exc).__name__}",
                provider=provider,
            ) from exc

    async def generate_stream(
        self,
        provider: str,
        req: LLMRequest,
        api_key: str,
        *,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> AsyncIterator[LLMChunk]:
        provider_name = self._resolve_provider(provider)
        client = self._resolve_client(provider_name)
        try:
            async for chunk in client.generate_stream(req, api_key=api_key, timeout_s=timeout_s):
                yield chunk
        except httpx.TimeoutException as exc:
            raise LLMError(LLMErrorCode.TIMEOUT, "Stream timed out", provider=provider) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                classify_provider_error(
                    provider,
                    exc.response.status_code,
                    _parse_json_or_none(exc.response),
                    None,
                ),
                f"Provider returned HTTP {exc.response.status_code}",
                provider=provider,
            ) from exc
        except httpx.NetworkError as exc:
            raise LLMError(
                LLMErrorCode.PROVIDER_DOWN,
                "Network error during stream",
                provider=provider,
            ) from exc
        except LLMError:
            raise
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(
                LLMErrorCode.PROVIDER_DOWN,
                f"Stream parsing error: {type(exc).__name__}",
                provider=provider,
            ) from exc

    def _resolve_provider(self, provider: str) -> ProviderName:
        if provider == "openai":
            if self._enable_openai:
                return "openai"
            raise LLMError(
                LLMErrorCode.MODEL_NOT_AVAILABLE,
                "Provider openai is disabled",
                provider=provider,
            )
        if provider == "anthropic":
            if self._enable_anthropic:
                return "anthropic"
            raise LLMError(
                LLMErrorCode.MODEL_NOT_AVAILABLE,
                "Provider anthropic is disabled",
                provider=provider,
            )
        if provider == "gemini":
            if self._enable_gemini:
                return "gemini"
            raise LLMError(
                LLMErrorCode.MODEL_NOT_AVAILABLE,
                "Provider gemini is disabled",
                provider=provider,
            )
        if provider == "deepseek":
            if self._enable_deepseek:
                return "deepseek"
            raise LLMError(
                LLMErrorCode.MODEL_NOT_AVAILABLE,
                "Provider deepseek is disabled",
                provider=provider,
            )
        raise LLMError(
            LLMErrorCode.MODEL_NOT_AVAILABLE,
            f"Unknown provider: {provider}",
            provider=provider,
        )

    def _resolve_client(
        self, provider: ProviderName
    ) -> OpenAIClient | AnthropicClient | GeminiClient | DeepSeekClient:
        if provider == "openai":
            return self._openai
        if provider == "anthropic":
            return self._anthropic
        if provider == "gemini":
            return self._gemini
        if provider == "deepseek":
            return self._deepseek


def _parse_json_or_none(response: httpx.Response) -> dict | None:
    try:
        parsed = response.json()
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
