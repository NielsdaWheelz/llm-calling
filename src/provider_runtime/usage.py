"""Usage normalization and deterministic catalog cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from provider_runtime.catalog import Pricing
from provider_runtime.types import TokenUsage

CostPolicy = Literal["catalog_pricing"]
CostStatus = Literal["estimated", "unknown"]


@dataclass(frozen=True)
class CostBreakdown:
    input_cost: float | None
    output_cost: float | None
    cached_input_cost: float | None
    reasoning_cost: float | None
    total_cost: float | None


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
    pricing_source: str = DEFAULT_PRICING_SOURCE,
) -> CostEstimate:
    """Return the shared advisory cost policy for a catalog-priced call."""
    breakdown = estimate_cost(usage, pricing)
    return CostEstimate(
        policy="catalog_pricing",
        status="estimated" if breakdown.total_cost is not None else "unknown",
        pricing_source=pricing_source,
        pricing=pricing,
        breakdown=breakdown,
    )


def estimate_cost(usage: TokenUsage | None, pricing: Pricing) -> CostBreakdown:
    """Return advisory cost from normalized usage and catalog pricing.

    Missing token counts or missing prices yield ``None`` for that component.
    The returned total is ``None`` if no component can be estimated.
    """
    if usage is None:
        return CostBreakdown(
            input_cost=None,
            output_cost=None,
            cached_input_cost=None,
            reasoning_cost=None,
            total_cost=None,
        )

    cached_input_tokens = usage.cached_tokens or usage.cache_read_input_tokens
    input_tokens = usage.input_tokens
    if input_tokens is not None and cached_input_tokens is not None:
        input_tokens = max(0, input_tokens - cached_input_tokens)

    input_cost = _component_cost(input_tokens, pricing.input_per_million)
    output_cost = _component_cost(usage.output_tokens, pricing.output_per_million)
    cached_input_cost = _component_cost(cached_input_tokens, pricing.cached_input_per_million)
    reasoning_cost = _component_cost(usage.reasoning_tokens, pricing.reasoning_per_million)

    components = [
        component
        for component in (input_cost, output_cost, cached_input_cost, reasoning_cost)
        if component is not None
    ]
    total_cost = sum(components) if components else None
    return CostBreakdown(
        input_cost=input_cost,
        output_cost=output_cost,
        cached_input_cost=cached_input_cost,
        reasoning_cost=reasoning_cost,
        total_cost=total_cost,
    )


def _component_cost(tokens: int | None, price_per_million: float | None) -> float | None:
    if tokens is None or price_per_million is None:
        return None
    return (tokens / 1_000_000) * price_per_million
