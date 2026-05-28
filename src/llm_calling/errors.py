"""Provider error classification."""

from enum import StrEnum

import httpx


class LLMErrorCode(StrEnum):
    INVALID_KEY = "invalid_key"
    RATE_LIMIT = "rate_limit"
    CONTEXT_TOO_LARGE = "context_too_large"
    TIMEOUT = "timeout"
    PROVIDER_DOWN = "provider_down"
    BAD_REQUEST = "bad_request"
    MODEL_NOT_AVAILABLE = "model_not_available"


class LLMError(Exception):
    def __init__(
        self,
        error_code: LLMErrorCode,
        message: str,
        provider: str | None = None,
    ):
        self.error_code = error_code
        self.message = message
        self.provider = provider
        super().__init__(message)


async def raise_for_provider_error(response: httpx.Response, provider: str) -> None:
    """Raise LLMError with the provider's response body if status >= 400.

    Stream contexts close their body after raise_for_status(), so error
    detail is lost. Read explicitly here and surface it in the exception.
    """
    if response.status_code < 400:
        return
    try:
        await response.aread()
    except Exception:
        pass
    try:
        json_body = response.json()
    except Exception:
        json_body = None
    body_text = response.text if response.is_closed else ""
    snippet = (body_text or "").strip()[:500]
    code = classify_provider_error(
        provider,
        response.status_code,
        json_body if isinstance(json_body, dict) else None,
        None,
    )
    message = (
        f"{provider} HTTP {response.status_code}: {snippet}"
        if snippet
        else f"{provider} HTTP {response.status_code}"
    )
    raise LLMError(code, message, provider=provider)


def classify_provider_error(
    provider: str,
    status_code: int | None,
    json_body: dict | None,
    exception: Exception | None,
) -> LLMErrorCode:
    if exception is not None:
        exception_type = type(exception).__name__
        if "Timeout" in exception_type or "timeout" in str(exception).lower():
            return LLMErrorCode.TIMEOUT
        if "Network" in exception_type or "Connection" in exception_type:
            return LLMErrorCode.PROVIDER_DOWN

    if status_code is None:
        return LLMErrorCode.PROVIDER_DOWN

    if provider in ("openai", "deepseek"):
        return _classify_openai_error(status_code, json_body)
    if provider == "anthropic":
        return _classify_anthropic_error(status_code, json_body)
    if provider == "gemini":
        return _classify_gemini_error(status_code, json_body)
    return LLMErrorCode.PROVIDER_DOWN


def _classify_openai_error(status_code: int, json_body: dict | None) -> LLMErrorCode:
    if status_code in (401, 403):
        return LLMErrorCode.INVALID_KEY

    if status_code == 429:
        return LLMErrorCode.RATE_LIMIT

    if status_code == 404:
        return LLMErrorCode.MODEL_NOT_AVAILABLE

    if status_code >= 500:
        return LLMErrorCode.PROVIDER_DOWN

    if status_code == 400 and json_body:
        error = json_body.get("error", {})
        error_code = error.get("code", "")
        error_message = error.get("message", "").lower()

        if error_code == "context_length_exceeded":
            return LLMErrorCode.CONTEXT_TOO_LARGE
        if "maximum context length" in error_message:
            return LLMErrorCode.CONTEXT_TOO_LARGE
        if "model" in error_message and "not found" in error_message:
            return LLMErrorCode.MODEL_NOT_AVAILABLE

    if status_code is not None and status_code < 500:
        return LLMErrorCode.BAD_REQUEST

    return LLMErrorCode.PROVIDER_DOWN


def _classify_anthropic_error(status_code: int, json_body: dict | None) -> LLMErrorCode:
    if status_code in (401, 403):
        return LLMErrorCode.INVALID_KEY

    if status_code == 429:
        return LLMErrorCode.RATE_LIMIT

    if status_code == 404:
        return LLMErrorCode.MODEL_NOT_AVAILABLE

    if status_code >= 500:
        return LLMErrorCode.PROVIDER_DOWN

    if status_code == 400 and json_body:
        error = json_body.get("error", {})
        error_type = error.get("type", "")
        error_message = error.get("message", "").lower()

        if error_type == "invalid_request_error" and "too long" in error_message:
            return LLMErrorCode.CONTEXT_TOO_LARGE

    if status_code is not None and status_code < 500:
        return LLMErrorCode.BAD_REQUEST

    return LLMErrorCode.PROVIDER_DOWN


def _classify_gemini_error(status_code: int, json_body: dict | None) -> LLMErrorCode:
    body_str = str(json_body).lower() if json_body else ""

    if "api_key_invalid" in body_str:
        return LLMErrorCode.INVALID_KEY

    if status_code in (401, 403):
        return LLMErrorCode.INVALID_KEY

    if status_code == 429 or "resource_exhausted" in body_str:
        return LLMErrorCode.RATE_LIMIT

    if "exceeds the maximum" in body_str:
        return LLMErrorCode.CONTEXT_TOO_LARGE

    if status_code == 404 or "model not found" in body_str:
        return LLMErrorCode.MODEL_NOT_AVAILABLE

    if status_code >= 500:
        return LLMErrorCode.PROVIDER_DOWN

    if status_code is not None and status_code < 500:
        return LLMErrorCode.BAD_REQUEST

    return LLMErrorCode.PROVIDER_DOWN
