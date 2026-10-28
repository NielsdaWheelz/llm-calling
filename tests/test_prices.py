"""Indicative current-model prices and exact identity matching."""

from datetime import date

import pytest

from provider_runtime.prices import estimate_cost
from provider_runtime.registry import api_model_catalog
from provider_runtime.types import (
    Absent,
    AttemptRecord,
    CallMeta,
    FinalAttempt,
    PossiblyBillable,
    Present,
    ProviderName,
    TokenUsage,
)


def _meta(
    provider: ProviderName,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> CallMeta:
    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        reasoning_tokens=Absent(),
        cache_read_input_tokens=Present(cache_read),
        cache_write_input_tokens=Present(cache_write),
    )
    return CallMeta(
        provider=provider,
        model=model,
        provider_request_id=Absent(),
        usage=Present(usage),
        attempt_trace=(AttemptRecord(1, FinalAttempt(), Present(200), 0, 1),),
        billability=PossiblyBillable(),
        native_reasoning=Absent(),
        registry_revision="current",
    )


def test_cache_read_discount_and_write_surcharge() -> None:
    # Fable 5.1: 60k ordinary input + 40k cache read + 20k 5m write
    # surcharge + 10k output = 600k + 10k + 50k + 500k micros.
    estimate = estimate_cost(
        _meta(
            "anthropic", "claude-fable-5-1", 100_000, 10_000, cache_read=40_000, cache_write=20_000
        )
    )
    assert isinstance(estimate, Present)
    assert estimate.value.amount_usd_micros == 1_160_000
    assert estimate.value.as_of == date(2026, 10, 28)
    assert "official-provider-docs" in estimate.value.source


def test_only_current_exact_identity_has_an_estimate() -> None:
    assert isinstance(estimate_cost(_meta("openai", "gpt-6.1-sol", 1000, 100)), Present)
    assert isinstance(estimate_cost(_meta("openai", "gpt-5.6-sol", 1000, 100)), Absent)
    assert isinstance(estimate_cost(_meta("openai", "gpt-6.1-sol-old", 1000, 100)), Absent)


def test_unknown_high_context_tier_is_not_mispriced() -> None:
    assert isinstance(estimate_cost(_meta("openai", "gpt-6.1-sol", 272_001, 100)), Absent)
    assert isinstance(estimate_cost(_meta("xai", "grok-4.7", 200_001, 100)), Absent)


def test_deepseek_peak_rate_is_conservative() -> None:
    estimate = estimate_cost(_meta("deepseek", "deepseek-flash", 1_000_000, 1_000_000))
    assert isinstance(estimate, Present)
    assert estimate.value.amount_usd_micros == 1_500_000


def test_every_current_exact_model_has_a_short_context_estimate() -> None:
    for row in api_model_catalog().models:
        assert isinstance(
            estimate_cost(_meta(row.provider, row.dispatch.model_id, 1000, 100)), Present
        ), row.model_ref


@pytest.mark.parametrize(
    "provider,model,limit",
    [
        ("anthropic", "claude-haiku-5-5", 100_000),
        ("gemini", "gemini-3.1-pro-preview", 200_000),
    ],
)
def test_new_context_tiers_do_not_receive_short_context_prices(provider, model, limit) -> None:
    assert isinstance(estimate_cost(_meta(provider, model, limit, 100)), Present)
    assert estimate_cost(_meta(provider, model, limit + 1, 100)) == Absent()


def test_renamed_models_use_current_cache_prices() -> None:
    sol = estimate_cost(_meta("openai", "gpt-6.1-sol", 1000, 0, cache_read=1000))
    sonnet = estimate_cost(_meta("anthropic", "claude-sonnet-5-5", 1000, 0, cache_read=1000))
    assert isinstance(sol, Present) and sol.value.amount_usd_micros == 100
    assert isinstance(sonnet, Present) and sonnet.value.amount_usd_micros == 100


def test_xai_long_context_starts_at_the_published_boundary() -> None:
    assert isinstance(estimate_cost(_meta("xai", "grok-build-0.1", 199_999, 100)), Present)
    assert estimate_cost(_meta("xai", "grok-build-0.1", 200_000, 100)) == Absent()
