from llm_calling.errors import LLMErrorCode, classify_provider_error


def test_unknown_provider_classifies_as_provider_down() -> None:
    assert classify_provider_error("unknown", 500, None, None) == LLMErrorCode.PROVIDER_DOWN


def test_network_exception_classifies_as_provider_down() -> None:
    assert (
        classify_provider_error("openai", None, None, ConnectionError("closed"))
        == LLMErrorCode.PROVIDER_DOWN
    )
