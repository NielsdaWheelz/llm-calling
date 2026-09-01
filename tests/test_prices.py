"""Behavior tests for `provider_runtime.prices.estimate_cost`.

Expected micros are hand-computed from the vendored snapshot's rates — a price
refresh that changes a pinned rate is supposed to break these tests so the new
figures get reviewed with the diff.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from importlib import resources

from provider_runtime import registry
from provider_runtime.prices import estimate_cost
from provider_runtime.types import (
    Absent,
    AttemptRecord,
    CallMeta,
    CostEstimate,
    FinalAttempt,
    PossiblyBillable,
    Presence,
    Present,
    ProviderName,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Fixtures

_ABSENT: Absent = Absent()


def _usage(
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read: Presence[int] = _ABSENT,
    cache_write: Presence[int] = _ABSENT,
) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        reasoning_tokens=Absent(),
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )


def _meta(provider: ProviderName, model: str, usage: Presence[TokenUsage]) -> CallMeta:
    return CallMeta(
        provider=provider,
        model=model,
        provider_request_id=Absent(),
        upstream_provider=Absent(),
        usage=usage,
        attempt_trace=(
            AttemptRecord(
                attempt=1,
                signal=FinalAttempt(),
                status_code=Present(200),
                started_at_ms=0,
                ended_at_ms=1,
            ),
        ),
        billability=PossiblyBillable(),
        native_reasoning=Absent(),
        registry_revision="test-revision",
    )


def _micros(estimate: Presence[CostEstimate]) -> int:
    assert isinstance(estimate, Present), f"expected Present estimate, got {estimate!r}"
    return estimate.value.amount_usd_micros


def _snapshot() -> dict[str, object]:
    text = resources.files("provider_runtime").joinpath("prices_snapshot.json").read_text("utf-8")
    return json.loads(text)


# ---------------------------------------------------------------------------
# Estimation


def test_known_model_estimate_matches_hand_computed_micros() -> None:
    # claude-fable-5: input $10/MTok, cache read $1, cache-write surcharge
    # $2.5, output $50. Write tokens sit INSIDE input, pay the input rate
    # there, and the surcharge tops them up to Anthropic's $12.5 write price:
    #   (100_000 - 40_000) * 10  =   600_000 micros  (uncached + write input)
    #   40_000 * 1               =    40_000 micros  (cache reads)
    #   20_000 * 2.5             =    50_000 micros  (cache-write top-up)
    #   10_000 * 50              =   500_000 micros  (output)
    usage = _usage(100_000, 10_000, cache_read=Present(40_000), cache_write=Present(20_000))
    estimate = estimate_cost(_meta("anthropic", "claude-fable-5", Present(usage)))
    assert _micros(estimate) == 1_190_000, f"estimate was {estimate!r}"


def test_cache_read_tokens_are_a_discounted_subset_never_additive() -> None:
    # kimi-k3: input $3/MTok, cache read $0.3/MTok. Cached half is discounted,
    # not billed on top: 500 * 3 + 500 * 0.3 = 1_650 micros.
    cached = estimate_cost(
        _meta("moonshot", "kimi-k3", Present(_usage(1_000, 0, cache_read=Present(500))))
    )
    uncached = estimate_cost(_meta("moonshot", "kimi-k3", Present(_usage(1_000, 0))))
    assert _micros(cached) == 1_650, f"cached estimate was {cached!r}"
    assert _micros(uncached) == 3_000, f"uncached estimate was {uncached!r}"


def test_cache_read_exceeding_input_tokens_clamps_instead_of_crashing() -> None:
    # A provider misreport (cache_read > input_tokens) must yield a
    # well-formed estimate, not a crash: the uncached component clamps at 0
    # instead of going negative. kimi-k3: cache read $0.3/MTok →
    # 150 * 0.3 = 45 micros.
    usage = _usage(100, 0, cache_read=Present(150))
    estimate = estimate_cost(_meta("moonshot", "kimi-k3", Present(usage)))
    assert _micros(estimate) == 45, f"estimate was {estimate!r}"


def test_half_micros_round_up() -> None:
    # gemini-3.5-flash input $1.5/MTok: 3 tokens = 4.5 micros. Half up gives 5
    # (bankers' rounding would give 4).
    estimate = estimate_cost(_meta("gemini", "gemini-3.5-flash", Present(_usage(3, 0))))
    assert _micros(estimate) == 5, f"estimate was {estimate!r}"


def test_source_and_as_of_carry_the_snapshot_date() -> None:
    snapshot = _snapshot()
    snapshot_date = snapshot["snapshot_date"]
    assert isinstance(snapshot_date, str)
    estimate = estimate_cost(_meta("xai", "grok-4.5", Present(_usage(1_000, 1_000))))
    assert isinstance(estimate, Present), f"expected Present estimate, got {estimate!r}"
    assert estimate.value.source == f"genai-prices@{snapshot_date}", estimate.value
    assert estimate.value.as_of == date.fromisoformat(snapshot_date), estimate.value


# ---------------------------------------------------------------------------
# Matching — exact first, else the longest model key that prefixes meta.model.


def test_dated_model_id_prefix_matches_its_base_row() -> None:
    usage = Present(_usage(10_000, 1_000))
    dated = estimate_cost(_meta("anthropic", "claude-fable-5-20260301", usage))
    exact = estimate_cost(_meta("anthropic", "claude-fable-5", usage))
    assert _micros(dated) == _micros(exact), f"dated {dated!r} != exact {exact!r}"


def test_longest_prefix_row_wins() -> None:
    # gemini-3.5-flash-lite ($0.3/MTok input) is itself prefixed by
    # gemini-3.5-flash ($1.5/MTok); the longer, more specific key must win.
    estimate = estimate_cost(
        _meta("gemini", "gemini-3.5-flash-lite-preview-0801", Present(_usage(1_000, 0)))
    )
    assert _micros(estimate) == 300, f"estimate was {estimate!r}"


def test_unknown_model_returns_absent() -> None:
    estimate = estimate_cost(_meta("openai", "not-a-model", Present(_usage(1_000, 1_000))))
    assert estimate == Absent(), f"expected Absent, got {estimate!r}"


def test_openrouter_calls_estimate_as_absent() -> None:
    # Deliberate: upstream OpenRouter pricing depends on the routed endpoint,
    # so the snapshot vendors no openrouter rows.
    estimate = estimate_cost(
        _meta("openrouter", "moonshotai/kimi-k3-20260715", Present(_usage(1_000, 1_000)))
    )
    assert estimate == Absent(), f"expected Absent, got {estimate!r}"


def test_absent_usage_returns_absent_even_for_a_known_model() -> None:
    estimate = estimate_cost(_meta("anthropic", "claude-fable-5", Absent()))
    assert estimate == Absent(), f"expected Absent, got {estimate!r}"


# ---------------------------------------------------------------------------
# Snapshot invariants


def test_snapshot_rows_carry_every_consumed_field() -> None:
    snapshot = _snapshot()
    snapshot_date = snapshot["snapshot_date"]
    assert isinstance(snapshot_date, str)
    date.fromisoformat(snapshot_date)
    upstream = snapshot["upstream"]
    assert isinstance(upstream, str)
    assert upstream.startswith("pydantic/genai-prices@"), upstream
    rows = snapshot["rows"]
    assert isinstance(rows, list) and rows, "snapshot must carry rows"
    seen: set[tuple[str, str]] = set()
    for row in rows:
        assert isinstance(row, dict), row
        assert row["provider"] in {
            "openai",
            "anthropic",
            "gemini",
            "moonshot",
            "deepseek",
            "xai",
        }, f"unexpected provider in row {row!r}"
        assert isinstance(row["model"], str) and row["model"], row
        for rate_field in ("input_mtok_usd", "output_mtok_usd", "cache_read_mtok_usd"):
            assert Decimal(row[rate_field]) >= 0, f"negative {rate_field} in row {row!r}"
        Decimal(row["cache_write_surcharge_mtok_usd"])  # must parse as a Decimal
        key = (row["provider"], row["model"])
        assert key not in seen, f"duplicate snapshot row {key!r}"
        seen.add(key)


def test_snapshot_covers_the_registry_wire_model_ids() -> None:
    # Iterates the real registry rather than a hardcoded copy of its ids, so
    # this stays a live contract check across registry changes — openrouter is
    # deliberately excluded (its pricing depends on the routed endpoint).
    for row in registry._ROWS:
        if row.provider == "openrouter":
            continue
        estimate = estimate_cost(_meta(row.provider, row.model_id, Present(_usage(1_000, 1_000))))
        assert isinstance(estimate, Present), f"no snapshot price for {row.provider}:{row.model_id}"
