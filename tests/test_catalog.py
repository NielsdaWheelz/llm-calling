import pytest

from provider_runtime import DEFAULT_CATALOG, ModelRef


def test_catalog_returns_per_model_capabilities() -> None:
    openai = DEFAULT_CATALOG.require_capabilities(ModelRef(provider="openai", model="gpt-5.5"))
    cloudflare = DEFAULT_CATALOG.require_capabilities(
        ModelRef(provider="cloudflare", model="@cf/openai/gpt-oss-20b")
    )

    assert openai.provider == "openai"
    assert openai.generation is True
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
    assert DEFAULT_CATALOG.key_probe_model("cloudflare") == "@cf/openai/gpt-oss-20b"


def test_catalog_covers_ariel_research_and_vision_models() -> None:
    research = DEFAULT_CATALOG.require_capabilities(
        ModelRef(provider="openrouter", model="deepseek/deepseek-v3.2")
    )
    vision = DEFAULT_CATALOG.require_capabilities(
        ModelRef(provider="gemini", model="gemini-2.5-flash")
    )

    assert research.provider == "openrouter"
    assert vision.provider == "gemini"
    assert vision.multimodal_input is True


def test_catalog_keeps_generation_embeddings_and_transcription_disjoint() -> None:
    for entry in DEFAULT_CATALOG.entries:
        assert entry.generation != entry.embeddings or not entry.embeddings
        assert entry.generation != entry.transcription or not entry.transcription
        if entry.embeddings:
            assert entry.generation is False
            assert entry.max_output_tokens == 0
            assert entry.reasoning_modes == ("none",)
        if entry.transcription:
            assert entry.generation is False
            assert entry.max_output_tokens == 0
            assert entry.max_context_tokens == 0
            assert entry.reasoning_modes == ("none",)
        assert not (entry.embeddings and entry.transcription)


def test_catalog_omits_retired_and_foreign_provider_models() -> None:
    keys = {(entry.provider, entry.model) for entry in DEFAULT_CATALOG.entries}

    assert ("anthropic", "claude-3-opus-20240229") not in keys
    assert ("cloudflare", "text-embedding-3-small") not in keys
    assert ("cloudflare", "@cf/qwen/qwen3-embedding-0.6b") in keys


def test_gemini_reasoning_modes_do_not_overclaim_thinking_off() -> None:
    pro_25 = DEFAULT_CATALOG.require_capabilities(
        ModelRef(provider="gemini", model="gemini-2.5-pro")
    )
    pro_31 = DEFAULT_CATALOG.require_capabilities(
        ModelRef(provider="gemini", model="gemini-3.1-pro-preview")
    )
    flash_25 = DEFAULT_CATALOG.require_capabilities(
        ModelRef(provider="gemini", model="gemini-2.5-flash")
    )

    assert "none" not in pro_25.reasoning_modes
    assert "none" not in pro_31.reasoning_modes
    assert "none" in flash_25.reasoning_modes
    assert pro_25.reasoning_budget_range == (128, 32768)
    assert flash_25.reasoning_budget_range == (0, 24576)


def test_gemini_catalog_does_not_claim_provider_request_ids() -> None:
    gemini_entries = [entry for entry in DEFAULT_CATALOG.entries if entry.provider == "gemini"]

    assert gemini_entries
    assert all(entry.provider_request_id is False for entry in gemini_entries)


def test_catalog_rejects_unknown_models() -> None:
    with pytest.raises(KeyError):
        DEFAULT_CATALOG.require_capabilities(ModelRef(provider="openai", model="missing"))
