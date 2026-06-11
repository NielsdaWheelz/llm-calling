import httpx
import pytest

from provider_runtime.errors import (
    ModelCallError,
    ModelCallErrorCode,
    classify_provider_error,
    raise_for_provider_error,
)


def test_unknown_provider_classifies_as_provider_down() -> None:
    assert classify_provider_error("unknown", 500, None, None) == ModelCallErrorCode.PROVIDER_DOWN


def test_network_exception_classifies_as_provider_down() -> None:
    assert (
        classify_provider_error("openai", None, None, ConnectionError("closed"))
        == ModelCallErrorCode.PROVIDER_DOWN
    )


def test_openai_429_insufficient_quota_classifies_as_quota_exceeded() -> None:
    body = {"error": {"code": "insufficient_quota", "type": "insufficient_quota"}}
    assert classify_provider_error("openai", 429, body, None) == ModelCallErrorCode.QUOTA_EXCEEDED
    assert (
        classify_provider_error("openrouter", 429, body, None) == ModelCallErrorCode.QUOTA_EXCEEDED
    )
    assert (
        classify_provider_error("cloudflare", 429, body, None) == ModelCallErrorCode.QUOTA_EXCEEDED
    )


def test_openai_429_without_quota_signal_classifies_as_rate_limit() -> None:
    assert classify_provider_error("openai", 429, None, None) == ModelCallErrorCode.RATE_LIMIT
    body = {"error": {"code": "rate_limit_exceeded", "type": "requests"}}
    assert classify_provider_error("openai", 429, body, None) == ModelCallErrorCode.RATE_LIMIT


def test_anthropic_credit_balance_400_classifies_as_quota_exceeded() -> None:
    body = {
        "error": {
            "type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API.",
        }
    }
    assert (
        classify_provider_error("anthropic", 400, body, None) == ModelCallErrorCode.QUOTA_EXCEEDED
    )


@pytest.mark.asyncio
async def test_provider_error_body_snippet_is_explicit_and_redacts_secret_tokens() -> None:
    response = httpx.Response(
        401,
        json={
            "error": {
                "message": (
                    "Bad keys sk-live-abcdefghijklmnopqrstuvwxyz1234567890, "
                    "AIzaSyabcdefghijklmnopqrstuvwxyz12345, and "
                    "Bearer very-secret-provider-token"
                )
            }
        },
        headers={"x-request-id": "req-secret"},
    )

    with pytest.raises(ModelCallError) as exc_info:
        await raise_for_provider_error(response, "openai")

    assert exc_info.value.provider_request_id == "req-secret"
    assert exc_info.value.message == "openai HTTP 401"
    assert str(exc_info.value) == "openai HTTP 401"
    assert exc_info.value.safe_body_snippet is not None
    assert "sk-live-abcdefghijklmnopqrstuvwxyz1234567890" not in exc_info.value.safe_body_snippet
    assert "AIzaSyabcdefghijklmnopqrstuvwxyz12345" not in exc_info.value.safe_body_snippet
    assert "Bearer very-secret-provider-token" not in exc_info.value.safe_body_snippet
    assert "...redacted" in exc_info.value.safe_body_snippet
