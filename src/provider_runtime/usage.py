"""Usage normalization and deterministic catalog cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from provider_runtime.catalog import Pricing, ReasoningBillingMode
from provider_runtime.types import PromptCacheTTL, TokenUsage

CostPolicy = Literal["catalog_pricing"]
CostStatus = Literal["estimated", "missing_pricing", "missing_usage", "not_token_priced"]


@dataclass(frozen=True)
class CostBreakdown:
    input_cost_usd_micros: int | None
    output_cost_usd_micros: int | None
    cache_write_cost_usd_micros: int | None
    cache_read_cost_usd_micros: int | None
    reasoning_cost_usd_micros: int | None
    total_cost_usd_micros: int | None


@dataclass(frozen=True)
class CostEstimate:
    policy: CostPolicy
    status: CostStatus
    pricing_source: str
    pricing: Pricing
    breakdown: CostBreakdown


DEFAULT_PRICING_SOURCE = "provider_runtime.catalog.DEFAULT_CATALOG"


def estimate_catalog_cost(
    usage: TokenUsage | None,
    pricing: Pricing,
    *,
    cache_write_ttl: PromptCacheTTL | None = None,
    pricing_source: str = DEFAULT_PRICING_SOURCE,
) -> CostEstimate:
    """Return the shared advisory cost policy for a catalog-priced call."""
    if (
        usage is None
        or pricing.unit != "per_million_tokens"
        or not _pricing_has_provenance(pricing)
    ):
        breakdown = CostBreakdown(
            input_cost_usd_micros=None,
            output_cost_usd_micros=None,
            cache_write_cost_usd_micros=None,
            cache_read_cost_usd_micros=None,
            reasoning_cost_usd_micros=None,
            total_cost_usd_micros=None,
        )
        return CostEstimate(
            policy="catalog_pricing",
            status=_cost_status(usage, pricing, breakdown, cache_write_ttl=cache_write_ttl),
            pricing_source=pricing_source,
            pricing=pricing,
            breakdown=breakdown,
        )

    breakdown = estimate_cost(usage, pricing, cache_write_ttl=cache_write_ttl)
    status = _cost_status(usage, pricing, breakdown, cache_write_ttl=cache_write_ttl)
    return CostEstimate(
        policy="catalog_pricing",
        status=status,
        pricing_source=pricing_source,
        pricing=pricing,
        breakdown=breakdown,
    )


def estimate_cost(
    usage: TokenUsage | None,
    pricing: Pricing,
    *,
    cache_write_ttl: PromptCacheTTL | None = None,
) -> CostBreakdown:
    """Return advisory cost from normalized usage and catalog pricing.

    Missing token counts or missing prices yield ``None`` for that component
    and for the total. Returned values are integer USD micros.
    """
    if (
        usage is None
        or pricing.unit != "per_million_tokens"
        or not _pricing_applies_to_usage(usage, pricing)
    ):
        return CostBreakdown(
            input_cost_usd_micros=None,
            output_cost_usd_micros=None,
            cache_write_cost_usd_micros=None,
            cache_read_cost_usd_micros=None,
            reasoning_cost_usd_micros=None,
            total_cost_usd_micros=None,
        )

    cache_read_tokens = _first_present(usage.cached_tokens, usage.cache_read_input_tokens)
    cache_write_tokens = usage.cache_creation_input_tokens
    billable_input_tokens = usage.input_tokens
    if billable_input_tokens is not None:
        billable_input_tokens = max(
            0, billable_input_tokens - (cache_read_tokens or 0) - (cache_write_tokens or 0)
        )

    input_cost = _component_cost_usd_micros(billable_input_tokens, pricing.input_per_million)
    output_cost = _component_cost_usd_micros(usage.output_tokens, pricing.output_per_million)
    cache_write_cost = _component_cost_usd_micros(
        cache_write_tokens,
        _cache_write_price(pricing, cache_write_ttl),
    )
    cache_read_cost = _component_cost_usd_micros(
        cache_read_tokens, pricing.cached_input_per_million
    )
    reasoning_cost = _reasoning_cost_usd_micros(
        usage.reasoning_tokens,
        pricing.reasoning_per_million,
        pricing.reasoning_billing_mode,
    )

    components = [
        component
        for component in (
            input_cost,
            output_cost,
            cache_write_cost,
            cache_read_cost,
            reasoning_cost,
        )
        if component is not None
    ]
    total_cost = (
        sum(components)
        if components and not _has_missing_pricing(usage, pricing, cache_write_ttl=cache_write_ttl)
        else None
    )
    return CostBreakdown(
        input_cost_usd_micros=input_cost,
        output_cost_usd_micros=output_cost,
        cache_write_cost_usd_micros=cache_write_cost,
        cache_read_cost_usd_micros=cache_read_cost,
        reasoning_cost_usd_micros=reasoning_cost,
        total_cost_usd_micros=total_cost,
    )


def _cost_status(
    usage: TokenUsage | None,
    pricing: Pricing,
    breakdown: CostBreakdown,
    *,
    cache_write_ttl: PromptCacheTTL | None,
) -> CostStatus:
    if pricing.unit != "per_million_tokens":
        return "not_token_priced"
    if usage is None or not _has_any_usage_tokens(usage):
        return "missing_usage"
    if not _pricing_has_provenance(pricing):
        return "missing_pricing"
    if not _pricing_applies_to_usage(usage, pricing):
        return "missing_pricing"
    if _has_missing_pricing(usage, pricing, cache_write_ttl=cache_write_ttl):
        return "missing_pricing"
    return "estimated" if breakdown.total_cost_usd_micros is not None else "missing_pricing"


def _pricing_has_provenance(pricing: Pricing) -> bool:
    return bool(pricing.source_url and pricing.verified_at)


def _has_any_usage_tokens(usage: TokenUsage) -> bool:
    return any(
        value is not None
        for value in (
            usage.input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            usage.cache_creation_input_tokens,
            usage.cache_read_input_tokens,
            usage.cached_tokens,
        )
    )


def _pricing_applies_to_usage(usage: TokenUsage, pricing: Pricing) -> bool:
    if pricing.applies_up_to_input_tokens is None or usage.input_tokens is None:
        return True
    return usage.input_tokens <= pricing.applies_up_to_input_tokens


def _has_missing_pricing(
    usage: TokenUsage,
    pricing: Pricing,
    *,
    cache_write_ttl: PromptCacheTTL | None,
) -> bool:
    cache_read_tokens = _first_present(usage.cached_tokens, usage.cache_read_input_tokens)
    cache_write_tokens = usage.cache_creation_input_tokens
    billable_input_tokens = usage.input_tokens
    if billable_input_tokens is not None:
        billable_input_tokens = max(
            0, billable_input_tokens - (cache_read_tokens or 0) - (cache_write_tokens or 0)
        )
    return any(
        (
            _needs_price(billable_input_tokens, pricing.input_per_million),
            _needs_price(usage.output_tokens, pricing.output_per_million),
            _needs_price(cache_read_tokens, pricing.cached_input_per_million),
            _needs_price(cache_write_tokens, _cache_write_price(pricing, cache_write_ttl)),
            _reasoning_needs_price(
                usage.reasoning_tokens,
                pricing.reasoning_per_million,
                pricing.reasoning_billing_mode,
            ),
        )
    )


def _needs_price(tokens: int | None, price_per_million: object | None) -> bool:
    return tokens is not None and tokens > 0 and price_per_million is None


def _reasoning_needs_price(
    tokens: int | None,
    price_per_million: object | None,
    billing_mode: ReasoningBillingMode,
) -> bool:
    if tokens is None or tokens <= 0:
        return False
    if billing_mode == "separate":
        return price_per_million is None
    return billing_mode == "unknown"


def _component_cost_usd_micros(tokens: int | None, price_per_million: object | None) -> int | None:
    if tokens is None:
        return None
    if tokens == 0:
        return 0
    if price_per_million is None:
        return None
    return int(
        (Decimal(tokens) * Decimal(str(price_per_million))).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def _reasoning_cost_usd_micros(
    tokens: int | None,
    price_per_million: object | None,
    billing_mode: ReasoningBillingMode,
) -> int | None:
    if billing_mode != "separate":
        return None
    return _component_cost_usd_micros(tokens, price_per_million)


def _cache_write_price(
    pricing: Pricing,
    cache_write_ttl: PromptCacheTTL | None,
) -> object | None:
    if cache_write_ttl is None:
        return None
    return pricing.cache_write_per_million_by_ttl.get(cache_write_ttl)


def _first_present(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None
