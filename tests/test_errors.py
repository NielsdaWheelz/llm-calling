from llm_calling.errors import LLMErrorCode, classify_provider_error


def test_unknown_provider_classifies_as_provider_down() -> None:
    assert classify_provider_error("unknown", 500, None, None) == LLMErrorCode.PROVIDER_DOWN


def test_network_exception_classifies_as_provider_down() -> None:
    assert (
        classify_provider_error("openai", None, None, ConnectionError("closed"))
        == LLMErrorCode.PROVIDER_DOWN
    )


def test_openai_429_insufficient_quota_classifies_as_quota_exceeded() -> None:
    body = {"error": {"code": "insufficient_quota", "type": "insufficient_quota"}}
    assert classify_provider_error("openai", 429, body, None) == LLMErrorCode.QUOTA_EXCEEDED
    assert classify_provider_error("deepseek", 429, body, None) == LLMErrorCode.QUOTA_EXCEEDED


def test_openai_429_without_quota_signal_classifies_as_rate_limit() -> None:
    assert classify_provider_error("openai", 429, None, None) == LLMErrorCode.RATE_LIMIT
    body = {"error": {"code": "rate_limit_exceeded", "type": "requests"}}
    assert classify_provider_error("openai", 429, body, None) == LLMErrorCode.RATE_LIMIT


def test_anthropic_credit_balance_400_classifies_as_quota_exceeded() -> None:
    body = {
        "error": {
            "type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API.",
        }
    }
    assert classify_provider_error("anthropic", 400, body, None) == LLMErrorCode.QUOTA_EXCEEDED
