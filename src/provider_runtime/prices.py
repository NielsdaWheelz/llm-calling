"""Indicative cost estimation over the vendored genai-prices snapshot.

`estimate_cost` is pure: it reads `prices_snapshot.json` (vendored by
`tools/refresh_prices.py`; the library never fetches) and derives a
`CostEstimate` from a call's `CallMeta` on demand — indicative, never
authoritative, never stored on `CallMeta`.

Matching rule (genai-prices' match clauses, simplified): rows are keyed by
(provider, model); an exact model match wins, else the row whose model key is
the LONGEST prefix of `meta.model` (so a dated wire id like
"claude-fable-5-20260301" matches its base row, while
"gemini-3.5-flash-lite-…" picks flash-lite over flash). No row → `Absent()`.

Money: snapshot rates are USD per million tokens — numerically identical to
micros per token — carried as `Decimal` end to end; the total rounds half up
to whole micros. `TokenUsage.input_tokens` is cache-INCLUSIVE, so cache reads
are subtracted before the input rate applies (a discounted subset, never
additive) and cache writes — which already pay the input rate inside
`input_tokens` — add only the vendored write SURCHARGE. Absent cache fields
count as zero tokens.

Layering: imports from `types` only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from importlib import resources
from typing import assert_never

from provider_runtime.types import (
    Absent,
    CallMeta,
    CostEstimate,
    Presence,
    Present,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Snapshot — loaded once at import; a malformed snapshot fails loudly here.


@dataclass(frozen=True, slots=True)
class _Rates:
    input_mtok: Decimal
    output_mtok: Decimal
    cache_read_mtok: Decimal
    cache_write_surcharge_mtok: Decimal


def _load_snapshot() -> tuple[date, dict[tuple[str, str], _Rates]]:
    text = resources.files("provider_runtime").joinpath("prices_snapshot.json").read_text("utf-8")
    snapshot = json.loads(text)
    rates = {
        (row["provider"], row["model"]): _Rates(
            input_mtok=Decimal(row["input_mtok_usd"]),
            output_mtok=Decimal(row["output_mtok_usd"]),
            cache_read_mtok=Decimal(row["cache_read_mtok_usd"]),
            cache_write_surcharge_mtok=Decimal(row["cache_write_surcharge_mtok_usd"]),
        )
        for row in snapshot["rows"]
    }
    return date.fromisoformat(snapshot["snapshot_date"]), rates


_SNAPSHOT_DATE, _RATES = _load_snapshot()
_SOURCE = f"genai-prices@{_SNAPSHOT_DATE.isoformat()}"


def _match_rates(provider: str, model: str) -> Presence[_Rates]:
    exact = _RATES.get((provider, model))
    if exact is not None:
        return Present(exact)
    best: tuple[int, _Rates] | None = None
    for (row_provider, row_model), rates in _RATES.items():
        if row_provider != provider or not model.startswith(row_model):
            continue
        if best is None or len(row_model) > best[0]:
            best = (len(row_model), rates)
    return Absent() if best is None else Present(best[1])


# ---------------------------------------------------------------------------
# Estimation


def _tokens(count: Presence[int]) -> int:
    match count:
        case Present(value=value):
            return value
        case Absent():
            return 0
        case _:
            assert_never(count)


def _amount_usd_micros(usage: TokenUsage, rates: _Rates) -> int:
    cache_read = _tokens(usage.cache_read_input_tokens)
    cache_write = _tokens(usage.cache_write_input_tokens)
    amount = (
        (usage.input_tokens - cache_read) * rates.input_mtok
        + cache_read * rates.cache_read_mtok
        + cache_write * rates.cache_write_surcharge_mtok
        + usage.output_tokens * rates.output_mtok
    )
    return int(amount.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def estimate_cost(meta: CallMeta) -> Presence[CostEstimate]:
    """Estimate the USD cost of a terminal call; Absent when unpriceable.

    Absent when the snapshot has no row for (meta.provider, meta.model) — all
    openrouter calls, deliberately: upstream pricing depends on the routed
    endpoint — or when `meta.usage` is Absent.
    """
    match meta.usage:
        case Absent():
            return Absent()
        case Present(value=usage):
            pass
        case _:
            assert_never(meta.usage)
    matched = _match_rates(meta.provider, meta.model)
    match matched:
        case Absent():
            return Absent()
        case Present(value=rates):
            return Present(
                CostEstimate(
                    amount_usd_micros=_amount_usd_micros(usage, rates),
                    source=_SOURCE,
                    as_of=_SNAPSHOT_DATE,
                )
            )
        case _:
            assert_never(matched)
