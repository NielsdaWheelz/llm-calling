"""Catalog row facts pinned exactly, plus construction-rejection behavior.

Every number here is transcribed from .dossiers/provider-facts.md. A failing
pin means either a provider fact changed (update the row
AND bump CATALOG_REVISION) or the catalog drifted from the researched facts.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import date

import pytest

from provider_runtime.catalog import (
    CATALOG,
    CATALOG_REVISION,
    AnthropicPrefixContract,
    Catalog,
    ChatModelContract,
    DirectCertification,
    EmbeddingContract,
    GeminiAutomaticPrefixContract,
    MoonshotKeyedPrefixContract,
    OpenAIExplicitPrefixContract,
    OpenRouterPrefixContract,
    OperatorCertified,
    OperatorUncertified,
    ReasoningContract,
    TranscriptionContract,
    check_catalog_freshness,
)
from provider_runtime.errors import RuntimeDefect
from provider_runtime.types import Absent, Present, ProviderName, ProviderTarget


def _chat(provider: ProviderName, model: str) -> ChatModelContract:
    return CATALOG.chat_contract(ProviderTarget(provider=provider, model=model))


def _with_chat(rows: tuple[ChatModelContract, ...]) -> Catalog:
    return Catalog(chat=rows, embeddings=CATALOG.embeddings, transcriptions=CATALOG.transcriptions)


# ---------------------------------------------------------------------------
# Portfolio shape


def test_catalog_revision_literal() -> None:
    assert CATALOG_REVISION == "cat-2026-07-28-r1"


def test_exactly_the_eight_chat_targets() -> None:
    targets = {(row.target.provider, row.target.model) for row in CATALOG.chat}
    assert targets == {
        ("openai", "gpt-5.6-sol"),
        ("openai", "gpt-5.6-terra"),
        ("openai", "gpt-5.6-luna"),
        ("anthropic", "claude-sonnet-5"),
        ("anthropic", "claude-fable-5"),
        ("gemini", "gemini-3.5-flash"),
        ("moonshot", "kimi-k3"),
        ("openrouter", "moonshotai/kimi-k3-20260715"),
    }, "chat portfolio must be exactly the 8 targets of the spec §4 portfolio"
    assert len(CATALOG.chat) == 8


def test_every_chat_row_has_current_verification_date_and_sources() -> None:
    for row in CATALOG.chat:
        expected = {
            "moonshot": date(2026, 7, 22),
            "openrouter": date(2026, 7, 28),
        }.get(row.target.provider, date(2026, 7, 20))
        assert row.verified_at == expected, row.target
        assert row.source_urls, row.target
        assert row.pricing.currency == "usd", row.target
        assert row.pricing.verified_at == expected, row.target


def test_native_mappings_are_identity_and_cover_levels_exactly() -> None:
    for row in CATALOG.chat:
        mapping = dict(row.reasoning.native_mapping)
        assert mapping == {level: level for level in row.reasoning.levels}, (
            f"{row.target}: native mapping must be identity over exactly the declared levels"
        )
        assert row.reasoning.provider_default in row.reasoning.levels, row.target


def test_reasoning_billed_inside_output_everywhere_with_zero_reserve() -> None:
    for row in CATALOG.chat:
        assert row.pricing.reasoning_billed_outside_output is False, row.target
        assert row.pricing.reasoning_reserve_tokens == 0, row.target


# ---------------------------------------------------------------------------
# OpenAI gpt-5.6 rows


@pytest.mark.parametrize(
    ("model", "input_rate", "cache_read", "cache_write", "output_rate"),
    [
        ("gpt-5.6-sol", 5_000_000, 500_000, 6_250_000, 30_000_000),
        ("gpt-5.6-terra", 2_500_000, 250_000, 3_125_000, 15_000_000),
        ("gpt-5.6-luna", 1_000_000, 100_000, 1_250_000, 6_000_000),
    ],
)
def test_gpt56_rows(
    model: str, input_rate: int, cache_read: int, cache_write: int, output_rate: int
) -> None:
    row = _chat("openai", model)
    assert row.protocol == "openai_responses"
    assert row.context_limit == 1_050_000
    assert row.output_limit == 128_000
    assert row.reasoning.levels == ("none", "low", "medium", "high", "xhigh", "max")
    assert row.reasoning.provider_default == "medium"
    assert row.cache == OpenAIExplicitPrefixContract(ttl="30m", minimum_prefix_tokens=1024)
    assert row.pricing.input_rate == input_rate
    assert row.pricing.cache_read_rate == cache_read
    assert row.pricing.cache_write_rate == cache_write
    assert row.pricing.output_rate == output_rate
    assert row.privacy.retention == "30d"
    assert row.privacy.zdr_eligible is True
    assert row.certification == DirectCertification()
    assert row.provider_request_id_available is True
    assert row.provider_framing_overhead_tokens == 64
    assert row.continuation_codec == "openai_responses"


def test_gpt56_xhigh_and_max_are_distinct_native_strings() -> None:
    mapping = _chat("openai", "gpt-5.6-sol").reasoning.native_mapping
    assert mapping["xhigh"] == "xhigh"
    assert mapping["max"] == "max"
    assert mapping["xhigh"] != mapping["max"], (
        "gpt-5.6 xhigh and max are DISTINCT native efforts; they must never collapse"
    )


# ---------------------------------------------------------------------------
# Anthropic rows


def test_claude_sonnet_5_row() -> None:
    row = _chat("anthropic", "claude-sonnet-5")
    assert row.protocol == "anthropic_messages"
    assert row.context_limit == 1_000_000
    assert row.output_limit == 128_000
    assert row.reasoning.levels == ("low", "medium", "high", "xhigh", "max")
    assert row.reasoning.provider_default == "high"
    assert row.cache == AnthropicPrefixContract(ttl="5m", minimum_prefix_tokens=1024)
    # INTRO rates through 2026-08-31: $2.00 / $0.20 / $2.50 (w5m) / $10.00 per 1M.
    assert row.pricing.input_rate == 2_000_000
    assert row.pricing.cache_read_rate == 200_000
    assert row.pricing.cache_write_rate == 2_500_000
    assert row.pricing.output_rate == 10_000_000
    assert row.privacy.retention == "standard"
    assert row.privacy.zdr_eligible is True
    assert row.certification == DirectCertification()
    assert row.provider_request_id_available is True
    assert row.strict_schema_dialect == "anthropic_output_config_json_schema"
    assert row.provider_framing_overhead_tokens == 32


def test_claude_fable_5_row() -> None:
    row = _chat("anthropic", "claude-fable-5")
    assert row.protocol == "anthropic_messages"
    assert row.context_limit == 1_000_000
    assert row.output_limit == 128_000
    assert row.reasoning.levels == ("low", "medium", "high", "xhigh", "max")
    assert row.reasoning.provider_default == "high"
    # Fable min cacheable prefix is 512 (sonnet is 1024).
    assert row.cache == AnthropicPrefixContract(ttl="5m", minimum_prefix_tokens=512)
    # $10.00 / $1.00 / $12.50 (w5m) / $50.00 per 1M.
    assert row.pricing.input_rate == 10_000_000
    assert row.pricing.cache_read_rate == 1_000_000
    assert row.pricing.cache_write_rate == 12_500_000
    assert row.pricing.output_rate == 50_000_000
    # Covered Model: 30-day retention required, ZDR unavailable.
    assert row.privacy.retention == "30d_required"
    assert row.privacy.zdr_eligible is False
    assert row.certification == DirectCertification()
    assert row.strict_schema_dialect == "anthropic_output_config_json_schema"


# ---------------------------------------------------------------------------
# Gemini row


def test_gemini_35_flash_row() -> None:
    row = _chat("gemini", "gemini-3.5-flash")
    assert row.protocol == "gemini_generate_content"
    assert row.context_limit == 1_048_576
    assert row.output_limit == 65_536
    assert row.reasoning.levels == ("minimal", "low", "medium", "high")
    assert row.reasoning.provider_default == "medium"
    assert row.cache == GeminiAutomaticPrefixContract(minimum_prefix_tokens=Present(4096))
    # $1.50 / $0.15 / implicit-cache no write billing / $9.00 per 1M.
    assert row.pricing.input_rate == 1_500_000
    assert row.pricing.cache_read_rate == 150_000
    assert row.pricing.cache_write_rate == 0
    assert row.pricing.output_rate == 9_000_000
    assert row.privacy.retention == "standard"
    assert row.privacy.zdr_eligible is True
    assert row.certification == DirectCertification()
    assert row.provider_request_id_available is False, (
        "gemini has no provider request id — the catalog must record that correlation fact"
    )


# ---------------------------------------------------------------------------
# Moonshot rows (direct + openrouter operator candidate)


def test_kimi_k3_direct_row() -> None:
    row = _chat("moonshot", "kimi-k3")
    assert row.protocol == "moonshot_chat"
    assert row.context_limit == 1_048_576
    assert row.output_limit == 131_072
    assert row.reasoning.levels == ("low", "high", "max")
    assert row.reasoning.provider_default == "max"
    assert row.cache == MoonshotKeyedPrefixContract(minimum_prefix_tokens=Absent())
    # $3.00 (cache-miss) / $0.30 (cache-hit) / automatic no write billing / $15.00 per 1M.
    assert row.pricing.input_rate == 3_000_000
    assert row.pricing.cache_read_rate == 300_000
    assert row.pricing.cache_write_rate == 0
    assert row.pricing.output_rate == 15_000_000
    assert row.privacy.retention == "standard"
    assert row.privacy.zdr_eligible is False
    assert row.certification == DirectCertification()
    assert row.provider_request_id_available is True


def test_openrouter_kimi_k3_operator_row() -> None:
    row = _chat("openrouter", "moonshotai/kimi-k3-20260715")
    assert row.protocol == "openrouter_chat"
    assert row.context_limit == 1_048_576
    assert row.output_limit == 131_072
    assert row.reasoning.levels == ("low", "high", "max")
    assert row.reasoning.provider_default == "max"
    assert row.cache == OpenRouterPrefixContract(
        pinned_upstream="moonshotai/mxfp4",
        canonical_revision="moonshotai/kimi-k3-20260715",
    )
    assert row.pricing.input_rate == 3_000_000
    assert row.pricing.cache_read_rate == 300_000
    assert row.pricing.cache_write_rate == 0
    assert row.pricing.output_rate == 15_000_000
    assert row.certification == OperatorUncertified(), (
        "the operator route stays OperatorUncertified until the paid certification "
        "produces an evidence artifact"
    )
    assert row.provider_request_id_available is True


# ---------------------------------------------------------------------------
# Embedding / transcription rows


def test_text_embedding_3_small_row() -> None:
    row = CATALOG.embedding_contract(
        ProviderTarget(provider="openai", model="text-embedding-3-small")
    )
    assert isinstance(row, EmbeddingContract)
    assert row.max_input_tokens == 8191
    assert row.input_rate == 20_000  # $0.02 per 1M
    assert row.source_urls
    assert len(CATALOG.embeddings) == 1


def test_gpt_4o_transcribe_row() -> None:
    row = CATALOG.transcription_contract(
        ProviderTarget(provider="openai", model="gpt-4o-transcribe")
    )
    assert isinstance(row, TranscriptionContract)
    assert row.input_rate == 2_500_000  # $2.50 per 1M audio-input tokens
    assert row.output_rate == 10_000_000  # $10.00 per 1M output tokens
    assert row.source_urls
    assert len(CATALOG.transcriptions) == 1


# ---------------------------------------------------------------------------
# Accessor defects


@pytest.mark.parametrize(
    "lookup",
    [
        CATALOG.chat_contract,
        CATALOG.embedding_contract,
        CATALOG.transcription_contract,
    ],
    ids=["chat", "embedding", "transcription"],
)
def test_unknown_target_raises_plan_defect(lookup: Callable[[ProviderTarget], object]) -> None:
    target = ProviderTarget(provider="openai", model="gpt-nonexistent")
    with pytest.raises(RuntimeDefect) as excinfo:
        lookup(target)
    assert excinfo.value.origin == "plan"
    assert excinfo.value.code == "unknown_target"
    assert "openai/gpt-nonexistent" in excinfo.value.message


def test_removed_legacy_targets_are_gone() -> None:
    for provider, model in (
        ("openai", "gpt-5.5"),
        ("anthropic", "claude-sonnet-4-6"),
        ("gemini", "gemini-2.5-flash"),
    ):
        with pytest.raises(RuntimeDefect):
            _chat(provider, model)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction rejection


def test_duplicate_target_rejected() -> None:
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((*CATALOG.chat, CATALOG.chat[0]))
    assert excinfo.value.code == "duplicate_target"
    assert "openai/gpt-5.6-sol" in excinfo.value.message


def test_native_mapping_missing_level_rejected() -> None:
    sol = _chat("openai", "gpt-5.6-sol")
    broken = replace(
        sol,
        reasoning=ReasoningContract(
            levels=sol.reasoning.levels,
            provider_default="medium",
            native_mapping={level: level for level in sol.reasoning.levels if level != "max"},
        ),
    )
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((broken,))
    assert excinfo.value.code == "invalid_reasoning_mapping"


def test_native_mapping_extra_key_rejected() -> None:
    gemini = _chat("gemini", "gemini-3.5-flash")
    broken = replace(
        gemini,
        reasoning=ReasoningContract(
            levels=gemini.reasoning.levels,
            provider_default="medium",
            native_mapping={**dict(gemini.reasoning.native_mapping), "max": "max"},
        ),
    )
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((broken,))
    assert excinfo.value.code == "invalid_reasoning_mapping"


def test_provider_default_outside_levels_rejected() -> None:
    kimi = _chat("moonshot", "kimi-k3")
    broken = replace(
        kimi,
        reasoning=ReasoningContract(
            levels=kimi.reasoning.levels,
            provider_default="medium",  # not in (low, high, max)
            native_mapping=kimi.reasoning.native_mapping,
        ),
    )
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((broken,))
    assert excinfo.value.code == "invalid_reasoning_mapping"


def test_direct_protocol_with_operator_certification_rejected() -> None:
    sol = _chat("openai", "gpt-5.6-sol")
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((replace(sol, certification=OperatorUncertified()),))
    assert excinfo.value.code == "certification_mismatch"


def test_openrouter_protocol_with_direct_certification_rejected() -> None:
    routed = _chat("openrouter", "moonshotai/kimi-k3-20260715")
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((replace(routed, certification=DirectCertification()),))
    assert excinfo.value.code == "certification_mismatch"


def test_operator_certified_row_matching_pin_facts_accepted() -> None:
    routed = _chat("openrouter", "moonshotai/kimi-k3-20260715")
    assert isinstance(routed.cache, OpenRouterPrefixContract)
    certified = replace(
        routed,
        certification=OperatorCertified(
            certified_pinned_upstream=routed.cache.pinned_upstream,
            certified_canonical_revision=routed.cache.canonical_revision,
            evidence_revision="ev-matching-1",
        ),
    )
    catalog = _with_chat((certified,))
    assert catalog.chat[0].certification == certified.certification


def test_operator_certified_row_re_pinned_to_new_revision_rejected() -> None:
    # A row re-pinned to a new canonical_revision while the OperatorCertified
    # evidence still carries the previous pin's facts must be rejected at
    # construction — planning must never pay traffic against a re-pinned
    # endpoint on stale certification evidence.
    routed = _chat("openrouter", "moonshotai/kimi-k3-20260715")
    assert isinstance(routed.cache, OpenRouterPrefixContract)
    re_pinned = replace(
        routed,
        cache=replace(routed.cache, canonical_revision="moonshotai/kimi-k3-20261001"),
        certification=OperatorCertified(
            certified_pinned_upstream=routed.cache.pinned_upstream,
            certified_canonical_revision=routed.cache.canonical_revision,
            evidence_revision="ev-stale-1",
        ),
    )
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((re_pinned,))
    assert excinfo.value.code == "certification_mismatch"


def test_operator_certified_row_with_non_openrouter_cache_rejected() -> None:
    routed = _chat("openrouter", "moonshotai/kimi-k3-20260715")
    broken = replace(
        routed,
        cache=MoonshotKeyedPrefixContract(minimum_prefix_tokens=Absent()),
        certification=OperatorCertified(
            certified_pinned_upstream="moonshotai/mxfp4",
            certified_canonical_revision="moonshotai/kimi-k3-20260715",
            evidence_revision="ev-wrong-cache-1",
        ),
    )
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((broken,))
    assert excinfo.value.code == "certification_mismatch"


def test_zero_input_rate_rejected() -> None:
    sol = _chat("openai", "gpt-5.6-sol")
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((replace(sol, pricing=replace(sol.pricing, input_rate=0)),))
    assert excinfo.value.code == "unpriced_selectable_route"


def test_zero_cache_write_rate_on_explicit_cache_row_rejected() -> None:
    sonnet = _chat("anthropic", "claude-sonnet-5")
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((replace(sonnet, pricing=replace(sonnet.pricing, cache_write_rate=0)),))
    assert excinfo.value.code == "unpriced_selectable_route"


def test_zero_cache_write_rate_on_automatic_cache_row_allowed() -> None:
    # Gemini/moonshot implicit caching legitimately bills no cache write.
    gemini = _chat("gemini", "gemini-3.5-flash")
    kimi = _chat("moonshot", "kimi-k3")
    catalog = _with_chat((gemini, kimi))
    assert catalog.chat_contract(gemini.target).pricing.cache_write_rate == 0


def test_empty_source_urls_rejected() -> None:
    sol = _chat("openai", "gpt-5.6-sol")
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((replace(sol, source_urls=()),))
    assert excinfo.value.code == "missing_source_urls"


def test_negative_framing_overhead_rejected() -> None:
    sol = _chat("openai", "gpt-5.6-sol")
    with pytest.raises(RuntimeDefect) as excinfo:
        _with_chat((replace(sol, provider_framing_overhead_tokens=-1),))
    assert excinfo.value.code == "invalid_framing_overhead"


# ---------------------------------------------------------------------------
# Freshness — data + explicit today, never import-time wall clock


def test_freshness_passes_on_verification_day() -> None:
    check_catalog_freshness(date(2026, 7, 20))


def test_freshness_passes_at_exactly_180_days_for_oldest_row() -> None:
    # Oldest rows (legacy embedding/transcription) verified 2026-06-11;
    # 2026-12-08 is exactly 180 days later — still fresh.
    check_catalog_freshness(date(2026, 12, 8))


def test_freshness_fails_just_past_180_days_of_oldest_row() -> None:
    with pytest.raises(RuntimeDefect) as excinfo:
        check_catalog_freshness(date(2026, 12, 9))
    assert excinfo.value.origin == "plan"
    assert excinfo.value.code == "stale_catalog_verification"
    assert "2026-06-11" in excinfo.value.message


def test_freshness_fails_loudly_a_year_out() -> None:
    with pytest.raises(RuntimeDefect) as excinfo:
        check_catalog_freshness(date(2027, 7, 20))
    assert excinfo.value.code == "stale_catalog_verification"


def test_freshness_accepts_explicit_catalog_argument() -> None:
    chat_only = Catalog(chat=CATALOG.chat, embeddings=(), transcriptions=())
    # All chat rows verified 2026-07-20 → fresh well past the legacy rows' horizon.
    check_catalog_freshness(date(2027, 1, 10), chat_only)
    with pytest.raises(RuntimeDefect):
        check_catalog_freshness(date(2027, 1, 20), chat_only)
