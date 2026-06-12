from provider_runtime import DEFAULT_CATALOG, DEFAULT_PRICING_SOURCE, Pricing, TokenUsage
from provider_runtime.usage import estimate_catalog_cost, estimate_cost


def test_estimate_cost_uses_integer_micros_and_excludes_cache_from_full_input() -> None:
    cost = estimate_cost(
        TokenUsage(
            input_tokens=1000,
            output_tokens=2000,
            total_tokens=3000,
            reasoning_tokens=100,
            cache_creation_input_tokens=100,
            cached_tokens=400,
        ),
        Pricing(
            input_per_million=1.0,
            output_per_million=2.0,
            cached_input_per_million=0.25,
            cache_write_per_million_by_ttl={"5m": 1.25},
            reasoning_per_million=3.0,
            reasoning_billing_mode="separate",
        ),
        cache_write_ttl="5m",
    )

    assert cost.input_cost_usd_micros == 500
    assert cost.output_cost_usd_micros == 4000
    assert cost.cache_write_cost_usd_micros == 125
    assert cost.cache_read_cost_usd_micros == 100
    assert cost.reasoning_cost_usd_micros == 300
    assert cost.total_cost_usd_micros == 5025


def test_inclusive_reasoning_billing_does_not_double_count_reasoning_tokens() -> None:
    estimate = estimate_catalog_cost(
        TokenUsage(
            input_tokens=1000,
            output_tokens=2000,
            total_tokens=3000,
            reasoning_tokens=400,
        ),
        Pricing(
            input_per_million=1,
            output_per_million=2,
            reasoning_billing_mode="included_in_output",
            source_url="https://example.invalid/pricing",
            verified_at="2026-06-11",
        ),
    )

    assert estimate.status == "estimated"
    assert estimate.breakdown.reasoning_cost_usd_micros is None
    assert estimate.breakdown.total_cost_usd_micros == 5000


def test_estimate_cost_does_not_synthesize_total_for_unknown_prices() -> None:
    estimate = estimate_catalog_cost(
        TokenUsage(input_tokens=1000, output_tokens=1000, total_tokens=2000),
        Pricing(),
    )

    assert estimate.status == "missing_pricing"
    assert estimate.breakdown.input_cost_usd_micros is None
    assert estimate.breakdown.output_cost_usd_micros is None
    assert estimate.breakdown.total_cost_usd_micros is None


def test_estimate_catalog_cost_reports_missing_usage_before_pricing() -> None:
    estimate = estimate_catalog_cost(
        None,
        Pricing(input_per_million=1, output_per_million=2),
    )

    assert estimate.status == "missing_usage"
    assert estimate.breakdown.total_cost_usd_micros is None


def test_estimate_catalog_cost_reports_not_token_priced_for_provider_unit_pricing() -> None:
    estimate = estimate_catalog_cost(
        TokenUsage(input_tokens=1000, output_tokens=1000, total_tokens=2000),
        Pricing(unit="provider_units"),
    )

    assert estimate.status == "not_token_priced"
    assert estimate.breakdown.total_cost_usd_micros is None


def test_estimate_catalog_cost_reports_estimated_when_required_components_are_known() -> None:
    estimate = estimate_catalog_cost(
        TokenUsage(input_tokens=1000, output_tokens=1000, total_tokens=2000),
        Pricing(
            input_per_million=1,
            output_per_million=2,
            source_url="https://example.invalid/pricing",
            verified_at="2026-06-11",
        ),
        pricing_source="test-catalog",
    )

    assert estimate.policy == "catalog_pricing"
    assert estimate.status == "estimated"
    assert estimate.pricing_source == "test-catalog"
    assert estimate.breakdown.input_cost_usd_micros == 1000
    assert estimate.breakdown.output_cost_usd_micros == 2000
    assert estimate.breakdown.total_cost_usd_micros == 3000


def test_estimate_catalog_cost_requires_pricing_provenance() -> None:
    estimate = estimate_catalog_cost(
        TokenUsage(input_tokens=1000, output_tokens=1000, total_tokens=2000),
        Pricing(input_per_million=1, output_per_million=2),
        pricing_source="test-catalog",
    )

    assert estimate.status == "missing_pricing"
    assert estimate.pricing_source == "test-catalog"
    assert estimate.breakdown.input_cost_usd_micros is None
    assert estimate.breakdown.output_cost_usd_micros is None
    assert estimate.breakdown.total_cost_usd_micros is None


def test_pricing_snapshot_preserves_provenance_and_decimal_rates() -> None:
    pricing = Pricing(
        input_per_million="1.25",
        output_per_million="2.5",
        cached_input_per_million="0.125",
        cache_write_per_million_by_ttl={"5m": "1.5"},
        reasoning_billing_mode="included_in_output",
        source_url="https://example.invalid/pricing",
        verified_at="2026-06-11",
    )

    assert pricing.to_json() == {
        "input_per_million": "1.25",
        "output_per_million": "2.5",
        "cached_input_per_million": "0.125",
        "cache_write_per_million_by_ttl": {"5m": "1.5"},
        "reasoning_per_million": None,
        "reasoning_billing_mode": "included_in_output",
        "applies_up_to_input_tokens": None,
        "source_url": "https://example.invalid/pricing",
        "verified_at": "2026-06-11",
        "currency": "USD",
        "unit": "per_million_tokens",
    }


def test_threshold_pricing_fails_closed_above_supported_context() -> None:
    estimate = estimate_catalog_cost(
        TokenUsage(input_tokens=2001, output_tokens=1, total_tokens=2002),
        Pricing(
            input_per_million=1,
            output_per_million=2,
            applies_up_to_input_tokens=2000,
        ),
    )

    assert estimate.status == "missing_pricing"
    assert estimate.breakdown.total_cost_usd_micros is None


def test_default_catalog_prices_are_verified_or_missing() -> None:
    for capability in DEFAULT_CATALOG.entries:
        pricing_json = capability.pricing.to_json()
        has_prices = any(
            pricing_json[key] is not None
            for key in (
                "input_per_million",
                "output_per_million",
                "cached_input_per_million",
                "reasoning_per_million",
            )
        ) or bool(pricing_json["cache_write_per_million_by_ttl"])
        if has_prices:
            assert pricing_json["source_url"]
            assert pricing_json["verified_at"] == "2026-06-11"
        else:
            assert pricing_json["source_url"] is None


def test_default_catalog_pricing_source_name_is_stable() -> None:
    assert DEFAULT_PRICING_SOURCE == "provider_runtime.catalog.DEFAULT_CATALOG"
