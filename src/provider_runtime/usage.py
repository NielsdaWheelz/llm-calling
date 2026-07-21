"""Terminal costing from frozen plan accounting.

`cost_from_accounting` computes the advisory cost breakdown for a terminal
call from the plan's frozen `Accounting` rates and the codec-normalized
`TokenUsage`. It deliberately imports nothing from the catalog (negative
gate): the planner froze the rates at plan time so terminal costing never
re-reads `PricingContract`.

Conventions:
- All amounts are integer USD micros; rates are USD micros per million tokens.
- A component the provider did not report is 0-cost — plain zeros, not owned
  absence: an unreported component is honestly "nothing billed on this line".
- Billable input subtracts cache read/write tokens (each only when Present)
  from `input_tokens`, floored at 0 — cache traffic is billed on its own lines
  at the cache rates.
- The reasoning line exists only when the plan froze
  `reasoning_billed_outside_output=True` and the provider reported reasoning
  tokens; it is billed at the OUTPUT rate (providers billing reasoning outside
  the output limit price it as output; no separate reasoning rate exists in
  `Accounting`). Providers billing reasoning inside output already include it
  in `output_tokens`, so the line stays 0 to avoid double counting.
- Each line is rounded to micros with `Decimal` `ROUND_HALF_UP`; the total is
  the sum of the rounded lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import assert_never

from provider_runtime.types import Absent, Accounting, Presence, Present, TokenUsage

_MILLION = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    input_cost_usd_micros: int
    output_cost_usd_micros: int
    cache_write_cost_usd_micros: int
    cache_read_cost_usd_micros: int
    reasoning_cost_usd_micros: int
    total_cost_usd_micros: int


def _tokens(maybe: Presence[int]) -> int:
    match maybe:
        case Present(value=value):
            return value
        case Absent():
            return 0
        case _:
            assert_never(maybe)


def _line_usd_micros(tokens: int, rate_usd_micros_per_million: int) -> int:
    if tokens == 0:
        return 0
    exact = Decimal(tokens) * Decimal(rate_usd_micros_per_million) / _MILLION
    return int(exact.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def cost_from_accounting(accounting: Accounting, usage: TokenUsage) -> CostBreakdown:
    cache_read_tokens = _tokens(usage.cache_read_input_tokens)
    cache_write_tokens = _tokens(usage.cache_write_input_tokens)
    billable_input_tokens = max(0, usage.input_tokens - cache_read_tokens - cache_write_tokens)

    input_cost = _line_usd_micros(billable_input_tokens, accounting.input_rate)
    output_cost = _line_usd_micros(usage.output_tokens, accounting.output_rate)
    cache_write_cost = _line_usd_micros(cache_write_tokens, accounting.cache_write_rate)
    cache_read_cost = _line_usd_micros(cache_read_tokens, accounting.cache_read_rate)
    reasoning_cost = (
        _line_usd_micros(_tokens(usage.reasoning_tokens), accounting.output_rate)
        if accounting.reasoning_billed_outside_output
        else 0
    )

    return CostBreakdown(
        input_cost_usd_micros=input_cost,
        output_cost_usd_micros=output_cost,
        cache_write_cost_usd_micros=cache_write_cost,
        cache_read_cost_usd_micros=cache_read_cost,
        reasoning_cost_usd_micros=reasoning_cost,
        total_cost_usd_micros=(
            input_cost + output_cost + cache_write_cost + cache_read_cost + reasoning_cost
        ),
    )


__all__ = ["CostBreakdown", "cost_from_accounting"]
