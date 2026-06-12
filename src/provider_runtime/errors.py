"""Provider error classification."""

from __future__ import annotations

import re
from enum import StrEnum

import httpx

from provider_runtime.types import RetryAttempt


class ModelCallErrorCode(StrEnum):
    INVALID_KEY = "invalid_key"
    RATE_LIMIT = "rate_limit"
    CONTEXT_TOO_LARGE = "context_too_large"
    TIMEOUT = "timeout"
    PROVIDER_DOWN = "provider_down"
    BAD_REQUEST = "bad_request"
    MODEL_NOT_AVAILABLE = "model_not_available"
    QUOTA_EXCEEDED = "quota_exceeded"
    TOOL_ARGUMENTS_INVALID = "tool_arguments_invalid"


class ModelCallError(Exception):
    def __init__(
        self,
        error_code: ModelCallErrorCode,
        message: str,
        provider: str | None = None,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        provider_request_id: str | None = None,
        retryable: bool | None = None,
        safe_body_snippet: str | None = None,
        attempts: tuple[RetryAttempt, ...] = (),
    ):
        self.error_code = error_code
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.provider_request_id = provider_request_id
        self.retryable = _is_retryable_error(error_code) if retryable is None else retryable
        self.safe_body_snippet = safe_body_snippet
        self.attempts = attempts
        super().__init__(message)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts) if self.attempts else 1

    @property
    def retry_count(self) -> int:
        return max(0, self.attempt_count - 1)

    def with_attempts(self, attempts: tuple[RetryAttempt, ...]) -> ModelCallError:
        self.attempts = attempts
        return self


async def raise_for_provider_error(response: httpx.Response, provider: str) -> None:
    """Raise ModelCallError with the provider's response body if status >= 400.

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
    snippet = _safe_body_snippet(body_text)
    code = classify_provider_error(
        provider,
        response.status_code,
        json_body if isinstance(json_body, dict) else None,
        None,
    )
    message = f"{provider} HTTP {response.status_code}"
    raise ModelCallError(
        code,
        message,
        provider=provider,
        status_code=response.status_code,
        retry_after_seconds=_retry_after_seconds(response.headers.get("retry-after")),
        provider_request_id=response.headers.get("x-request-id")
        or response.headers.get("request-id"),
        safe_body_snippet=snippet or None,
    )


def classify_provider_error(
    provider: str,
    status_code: int | None,
    json_body: dict | None,
    exception: Exception | None,
) -> ModelCallErrorCode:
    if exception is not None:
        exception_type = type(exception).__name__
        if "Timeout" in exception_type or "timeout" in str(exception).lower():
            return ModelCallErrorCode.TIMEOUT
        if "Network" in exception_type or "Connection" in exception_type:
            return ModelCallErrorCode.PROVIDER_DOWN

    if status_code is None:
        return ModelCallErrorCode.PROVIDER_DOWN

    if provider in ("openai", "openrouter", "cloudflare"):
        return _classify_openai_error(status_code, json_body)
    if provider == "anthropic":
        return _classify_anthropic_error(status_code, json_body)
    if provider == "gemini":
        return _classify_gemini_error(status_code, json_body)
    return ModelCallErrorCode.PROVIDER_DOWN


def _is_retryable_error(error_code: ModelCallErrorCode) -> bool:
    return error_code in {
        ModelCallErrorCode.RATE_LIMIT,
        ModelCallErrorCode.TIMEOUT,
        ModelCallErrorCode.PROVIDER_DOWN,
    }


def _retry_after_seconds(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "...redacted"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"), "...redacted"),
    (
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{10,}\b", re.IGNORECASE),
        "Bearer ...redacted",
    ),
    (
        re.compile(
            r"(?i)([?&](?:api[_-]?key|key|token|secret|access[_-]?token|"
            r"refresh[_-]?token|client[_-]?secret)=)[^&#\s]+"
        ),
        r"\1...redacted",
    ),
    (
        re.compile(
            r"(?i)(\"(?:api[_-]?key|x[_-]?api[_-]?key|key|token|secret|"
            r"authorization|access[_-]?token|refresh[_-]?token|client[_-]?secret)\""
            r"\s*:\s*\")[^\"]+(\")"
        ),
        r"\1...redacted\2",
    ),
    (
        re.compile(
            r"(?i)(\\\"(?:api[_-]?key|x[_-]?api[_-]?key|key|token|secret|"
            r"authorization|access[_-]?token|refresh[_-]?token|client[_-]?secret)\\\""
            r"\s*:\s*\\\")[^\\\"]+(\\\")"
        ),
        r"\1...redacted\2",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|x[_-]?api[_-]?key|key|token|secret|"
            r"authorization|access[_-]?token|refresh[_-]?token|client[_-]?secret)"
            r"\s*[:=]\s*[A-Za-z0-9._~+/=-]{8,}"
        ),
        r"\1=...redacted",
    ),
)


def sanitize_provider_text(text: str, *, limit: int = 500) -> str:
    snippet = (text or "").strip()[:limit]
    for pattern, replacement in _SECRET_PATTERNS:
        snippet = pattern.sub(replacement, snippet)
    return snippet


def _safe_body_snippet(body_text: str) -> str:
    return sanitize_provider_text(body_text, limit=500)


def _classify_openai_error(status_code: int, json_body: dict | None) -> ModelCallErrorCode:
    if status_code in (401, 403):
        return ModelCallErrorCode.INVALID_KEY

    if status_code in (408, 504):
        return ModelCallErrorCode.TIMEOUT

    if status_code == 429:
        error = (json_body or {}).get("error", {})
        # Billing exhaustion, not throughput: distinct because it is not retryable.
        if "insufficient_quota" in (error.get("code", ""), error.get("type", "")):
            return ModelCallErrorCode.QUOTA_EXCEEDED
        return ModelCallErrorCode.RATE_LIMIT

    if status_code == 404:
        return ModelCallErrorCode.MODEL_NOT_AVAILABLE

    if status_code >= 500:
        return ModelCallErrorCode.PROVIDER_DOWN

    if status_code == 400 and json_body:
        error = json_body.get("error", {})
        error_code = error.get("code", "")
        error_message = error.get("message", "").lower()

        if error_code == "context_length_exceeded":
            return ModelCallErrorCode.CONTEXT_TOO_LARGE
        if "maximum context length" in error_message:
            return ModelCallErrorCode.CONTEXT_TOO_LARGE
        if "model" in error_message and "not found" in error_message:
            return ModelCallErrorCode.MODEL_NOT_AVAILABLE

    if status_code is not None and status_code < 500:
        return ModelCallErrorCode.BAD_REQUEST

    return ModelCallErrorCode.PROVIDER_DOWN


def _classify_anthropic_error(status_code: int, json_body: dict | None) -> ModelCallErrorCode:
    if status_code in (401, 403):
        return ModelCallErrorCode.INVALID_KEY

    if status_code == 429:
        return ModelCallErrorCode.RATE_LIMIT

    if status_code == 404:
        return ModelCallErrorCode.MODEL_NOT_AVAILABLE

    if status_code >= 500:
        return ModelCallErrorCode.PROVIDER_DOWN

    if status_code == 400 and json_body:
        error = json_body.get("error", {})
        error_type = error.get("type", "")
        error_message = error.get("message", "").lower()

        if error_type == "invalid_request_error" and "too long" in error_message:
            return ModelCallErrorCode.CONTEXT_TOO_LARGE
        # Out-of-credit is a 400 invalid_request_error on this provider.
        if "credit balance is too low" in error_message:
            return ModelCallErrorCode.QUOTA_EXCEEDED

    if status_code is not None and status_code < 500:
        return ModelCallErrorCode.BAD_REQUEST

    return ModelCallErrorCode.PROVIDER_DOWN


def _classify_gemini_error(status_code: int, json_body: dict | None) -> ModelCallErrorCode:
    body_str = str(json_body).lower() if json_body else ""

    if "api_key_invalid" in body_str or ("api key" in body_str and "invalid" in body_str):
        return ModelCallErrorCode.INVALID_KEY

    if status_code in (401, 403):
        return ModelCallErrorCode.INVALID_KEY

    if status_code == 429 or "resource_exhausted" in body_str:
        return ModelCallErrorCode.RATE_LIMIT

    if "exceeds the maximum" in body_str:
        return ModelCallErrorCode.CONTEXT_TOO_LARGE

    if status_code == 404 or "model not found" in body_str:
        return ModelCallErrorCode.MODEL_NOT_AVAILABLE

    if status_code >= 500:
        return ModelCallErrorCode.PROVIDER_DOWN

    if status_code is not None and status_code < 500:
        return ModelCallErrorCode.BAD_REQUEST

    return ModelCallErrorCode.PROVIDER_DOWN
