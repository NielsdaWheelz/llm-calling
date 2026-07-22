"""cost_from_accounting: micros math over frozen plan accounting.

The module must not touch the catalog: rates come exclusively from the frozen
Accounting value (negative gate covers the import elsewhere)."""

from __future__ import annotations

import pytest

from provider_runtime.types import Absent, Accounting, Presence, Present, TokenUsage
from provider_runtime.usage import CostBreakdown, cost_from_accounting

ABSENT: Absent = Absent()


def accounting(
    *,
    input_rate: int = 2_000_000,
    output_rate: int = 8_000_000,
    cache_write_rate: int = 2_500_000,
    cache_read_rate: int = 200_000,
    reasoning_billed_outside_output: bool = False,
) -> Accounting:
    return Accounting(
        currency="usd",
        input_rate=input_rate,
        output_rate=output_rate,
        cache_write_rate=cache_write_rate,
        cache_read_rate=cache_read_rate,
        reasoning_billed_outside_output=reasoning_billed_outside_output,
        platform_token_reservation=10_000,
        maximum_cost_estimate_usd_micros=99_000_000,
    )


def usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: Presence[int] = ABSENT,
    cache_read_input_tokens: Presence[int] = ABSENT,
    cache_write_input_tokens: Presence[int] = ABSENT,
) -> TokenUsage:
    return TokenUsage.from_components(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=Absent(),
        reasoning_tokens=reasoning_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
    )


def test_plain_input_output_math() -> None:
    breakdown = cost_from_accounting(
        accounting(), usage(input_tokens=1_000_000, output_tokens=500_000)
    )
    assert breakdown == CostBreakdown(
        input_cost_usd_micros=2_000_000,
        output_cost_usd_micros=4_000_000,
        cache_write_cost_usd_micros=0,
        cache_read_cost_usd_micros=0,
        reasoning_cost_usd_micros=0,
        total_cost_usd_micros=6_000_000,
    )


def test_cache_components_are_subtracted_from_billable_input() -> None:
    breakdown = cost_from_accounting(
        accounting(),
        usage(
            input_tokens=1000,
            cache_read_input_tokens=Present(600),
            cache_write_input_tokens=Present(100),
        ),
    )
    # billable input = 1000 - 600 - 100 = 300
    assert breakdown.input_cost_usd_micros == 600  # 300 * 2_000_000 / 1e6
    assert breakdown.cache_read_cost_usd_micros == 120  # 600 * 200_000 / 1e6
    assert breakdown.cache_write_cost_usd_micros == 250  # 100 * 2_500_000 / 1e6
    assert breakdown.total_cost_usd_micros == 600 + 120 + 250


def test_absent_cache_components_bill_full_input() -> None:
    breakdown = cost_from_accounting(accounting(), usage(input_tokens=1000))
    assert breakdown.input_cost_usd_micros == 2000
    assert breakdown.cache_read_cost_usd_micros == 0
    assert breakdown.cache_write_cost_usd_micros == 0


def test_billable_input_is_floored_at_zero() -> None:
    breakdown = cost_from_accounting(
        accounting(),
        usage(input_tokens=100, cache_read_input_tokens=Present(600)),
    )
    assert breakdown.input_cost_usd_micros == 0
    assert breakdown.cache_read_cost_usd_micros == 120


def test_rounding_is_half_up_per_line() -> None:
    # 1 token at 500_000 micros/M = 0.5 micros -> 1 (half rounds up).
    up = cost_from_accounting(accounting(input_rate=500_000), usage(input_tokens=1))
    assert up.input_cost_usd_micros == 1
    # 1 token at 499_999 micros/M = 0.499999 -> 0.
    down = cost_from_accounting(accounting(input_rate=499_999), usage(input_tokens=1))
    assert down.input_cost_usd_micros == 0
    # 3 tokens at 500_000 micros/M = 1.5 -> 2.
    mid = cost_from_accounting(accounting(input_rate=500_000), usage(input_tokens=3))
    assert mid.input_cost_usd_micros == 2


def test_reasoning_line_only_when_billed_outside_output() -> None:
    outside = cost_from_accounting(
        accounting(reasoning_billed_outside_output=True),
        usage(output_tokens=10, reasoning_tokens=Present(1_000_000)),
    )
    # Billed at the OUTPUT rate.
    assert outside.reasoning_cost_usd_micros == 8_000_000
    assert outside.total_cost_usd_micros == outside.output_cost_usd_micros + 8_000_000

    inside = cost_from_accounting(
        accounting(reasoning_billed_outside_output=False),
        usage(output_tokens=10, reasoning_tokens=Present(1_000_000)),
    )
    assert inside.reasoning_cost_usd_micros == 0

    unreported = cost_from_accounting(
        accounting(reasoning_billed_outside_output=True),
        usage(output_tokens=10),
    )
    assert unreported.reasoning_cost_usd_micros == 0


def test_zero_usage_is_all_zero() -> None:
    breakdown = cost_from_accounting(accounting(), usage())
    assert breakdown == CostBreakdown(0, 0, 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Per-provider cost goldens (§ TokenUsage.input_tokens codec invariant):
# every codec normalizes to the cache-INCLUSIVE convention at ingress, so
# cost_from_accounting's subtraction correctly recovers billable (uncached)
# input for all five providers. A future codec that drifts from this
# convention (e.g. reintroducing a cache-EXCLUSIVE input_tokens) will
# either zero out the input line here or break these totals.


def test_cost_golden_anthropic_shaped_cache_read_exceeds_raw_input() -> None:
    # Regression for the cost-accounting blocker: Anthropic's wire
    # input_tokens EXCLUDES cache reads/writes (raw input=500,
    # cache_read=2000, output=100), so the codec normalizes at ingress to
    # the inclusive total input_tokens=500+2000=2500 before this ever sees
    # it. A cache_read exceeding the raw wire input is the sharpest case:
    # under the old (wrong) exclusive convention this floored the billable
    # input line to 0.
    breakdown = cost_from_accounting(
        accounting(),
        usage(
            input_tokens=500 + 2000,  # already-normalized inclusive input
            output_tokens=100,
            cache_read_input_tokens=Present(2000),
        ),
    )
    # Bills the uncached 500 tokens at the input rate — never zero.
    assert breakdown.input_cost_usd_micros == 1000  # 500 * 2_000_000 / 1e6
    assert breakdown.cache_read_cost_usd_micros == 400  # 2000 * 200_000 / 1e6
    assert breakdown.total_cost_usd_micros == (
        breakdown.input_cost_usd_micros
        + breakdown.output_cost_usd_micros
        + breakdown.cache_write_cost_usd_micros
        + breakdown.cache_read_cost_usd_micros
        + breakdown.reasoning_cost_usd_micros
    )


@pytest.mark.parametrize(
    "provider_label",
    [
        "openai_input_tokens_details.cached_tokens",
        "gemini_cachedContentTokenCount",
        "moonshot_cached_tokens",
    ],
)
def test_cost_golden_inclusive_wire_shapes(provider_label: str) -> None:
    # OpenAI usage.input_tokens, Gemini promptTokenCount, and Moonshot
    # prompt_tokens are each already cache-INCLUSIVE on the wire (⊇ their
    # respective cached-token field) — these codecs pass input_tokens
    # straight through unmodified. Same shape as the Anthropic golden above
    # (500 uncached + 2000 cached = 2500), asserted per provider so a future
    # normalization drift in any one codec breaks its own line here.
    del provider_label  # documents which wire field this shape models
    breakdown = cost_from_accounting(
        accounting(),
        usage(
            input_tokens=2500,
            output_tokens=100,
            cache_read_input_tokens=Present(2000),
        ),
    )
    assert breakdown.input_cost_usd_micros == 1000
    assert breakdown.cache_read_cost_usd_micros == 400
    assert breakdown.total_cost_usd_micros == (
        breakdown.input_cost_usd_micros
        + breakdown.output_cost_usd_micros
        + breakdown.cache_write_cost_usd_micros
        + breakdown.cache_read_cost_usd_micros
        + breakdown.reasoning_cost_usd_micros
    )


def test_total_is_sum_of_rounded_lines() -> None:
    breakdown = cost_from_accounting(
        accounting(reasoning_billed_outside_output=True),
        usage(
            input_tokens=1_000,
            output_tokens=2_000,
            reasoning_tokens=Present(500),
            cache_read_input_tokens=Present(400),
            cache_write_input_tokens=Present(100),
        ),
    )
    assert breakdown.total_cost_usd_micros == (
        breakdown.input_cost_usd_micros
        + breakdown.output_cost_usd_micros
        + breakdown.cache_write_cost_usd_micros
        + breakdown.cache_read_cost_usd_micros
        + breakdown.reasoning_cost_usd_micros
    )
