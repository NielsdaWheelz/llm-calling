import pytest

from provider_runtime import DEFAULT_CATALOG, ModelRef


def test_catalog_returns_per_model_capabilities() -> None:
    openai = DEFAULT_CATALOG.require_capabilities(ModelRef(provider="openai", model="gpt-5.5"))
    cloudflare = DEFAULT_CATALOG.require_capabilities(
        ModelRef(provider="cloudflare", model="@cf/meta/llama-3.1-8b-instruct")
    )

    assert openai.provider == "openai"
    assert openai.prompt_cache.mode == "keyed_ttl"
    assert openai.structured_output is True
    assert cloudflare.provider == "cloudflare"
    assert cloudflare.prompt_cache.supported is False
    assert cloudflare.structured_output is False


def test_catalog_uses_route_as_lookup_provider() -> None:
    routed = DEFAULT_CATALOG.require_capabilities(
        ModelRef(provider="openai", route="openrouter", model="openai/gpt-5.5")
    )

    assert routed.provider == "openrouter"
    assert routed.model == "openai/gpt-5.5"


def test_catalog_key_probe_models_are_provider_owned() -> None:
    assert DEFAULT_CATALOG.key_probe_model("openai") == "gpt-5.4-mini"
    assert DEFAULT_CATALOG.key_probe_model("anthropic") == "claude-haiku-4-5-20251001"
    assert DEFAULT_CATALOG.key_probe_model("gemini") == "gemini-3-flash-preview"
    assert DEFAULT_CATALOG.key_probe_model("openrouter") == "openai/gpt-5.4-mini"
    assert DEFAULT_CATALOG.key_probe_model("cloudflare") == "@cf/meta/llama-3.1-8b-instruct"


def test_catalog_rejects_unknown_models() -> None:
    with pytest.raises(KeyError):
        DEFAULT_CATALOG.require_capabilities(ModelRef(provider="openai", model="missing"))
