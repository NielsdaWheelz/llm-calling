"""Indicative current-model prices and exact identity matching."""

from datetime import date

from provider_runtime.prices import estimate_cost
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
    assert estimate.value.as_of == date(2026, 9, 25)
    assert "official-provider-docs" in estimate.value.source


def test_only_current_exact_identity_has_an_estimate() -> None:
    assert isinstance(estimate_cost(_meta("openai", "gpt-6-sol", 1000, 100)), Present)
    assert isinstance(estimate_cost(_meta("openai", "gpt-5.6-sol", 1000, 100)), Absent)
    assert isinstance(estimate_cost(_meta("openai", "gpt-6-sol-old", 1000, 100)), Absent)


def test_unknown_high_context_tier_is_not_mispriced() -> None:
    assert isinstance(estimate_cost(_meta("openai", "gpt-6-sol", 272_001, 100)), Absent)
    assert isinstance(estimate_cost(_meta("xai", "grok-4.7", 200_001, 100)), Absent)


def test_deepseek_peak_rate_is_conservative() -> None:
    estimate = estimate_cost(_meta("deepseek", "deepseek-flash", 1_000_000, 1_000_000))
    assert isinstance(estimate, Present)
    assert estimate.value.amount_usd_micros == 1_500_000
