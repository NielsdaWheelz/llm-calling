"""Refresh the vendored price snapshot from pydantic/genai-prices.

Usage: ``uv run tools/refresh_prices.py``. Network is allowed HERE only — the
library (`provider_runtime.prices`) reads the vendored snapshot and never
fetches. Refreshes are manual; review the diff before committing and bump
nothing else — `estimate_cost` picks up the new `snapshot_date` automatically.

Snapshot shape (minimal — only what `prices.py` consumes, not an upstream
mirror)::

    {
      "snapshot_date": "<ISO date the refresh ran>",
      "upstream": "pydantic/genai-prices@<commit sha> prices/data.json",
      "rows": [
        {
          "provider": "<ProviderName>",
          "model": "<upstream model id — exact-then-longest-prefix match key>",
          "input_mtok_usd": "<Decimal string>",
          "output_mtok_usd": "<Decimal string>",
          "cache_read_mtok_usd": "<Decimal string>",
          "cache_write_surcharge_mtok_usd": "<Decimal string>"
        },
        ...
      ]
    }

Normalization applied (upstream schema → this snapshot):

- Providers: only the six that map onto our ``ProviderName`` are kept
  (openai, anthropic, google→gemini, moonshotai→moonshot, deepseek,
  x-ai→xai). No openrouter rows — upstream OpenRouter pricing depends on the
  routed endpoint, so openrouter calls estimate as Absent by design.
- Models without both ``input_mtok`` and ``output_mtok`` (embeddings, image,
  moderation) are dropped — `estimate_cost` only prices generate calls.
- Tiered prices are flattened to the base rate (long-context tier surcharges
  ignored; the estimate is indicative, never authoritative).
- Conditional price lists are resolved as of the snapshot date: the last
  ``start_date``-constrained entry with ``start_date`` <= today wins
  (upstream's "last active" rule). A time-of-day constraint (a daily UTC
  window, e.g. DeepSeek's off-peak discount hours) has no request timestamp
  to resolve against, so it is picked as the flat rate only when its window
  covers MORE than half the day — that is the standard/peak rate, never the
  minority off-peak discount, which never overrides the otherwise-chosen
  entry.
- ``cache_read_mtok_usd`` defaults to the input rate when upstream prices no
  cache read (no discount) — exactly upstream's semantics for unpriced units.
- ``cache_write_surcharge_mtok_usd`` is upstream ``cache_write_mtok`` MINUS
  the input rate (0 when unpriced). Our ``TokenUsage.input_tokens`` is
  cache-INCLUSIVE (read and write), and `estimate_cost` only subtracts
  cache_read before applying the input rate — so write tokens already pay the
  input rate and this surcharge tops them up to upstream's full write price,
  reproducing genai-prices' own
  ``(input - cache_read - cache_write)*input + cache_write*cache_write_mtok``
  to the micro.

Rates are USD per million tokens — numerically identical to micros per token —
serialized as Decimal strings so the library's micros arithmetic stays exact.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date
from datetime import time as _time
from decimal import Decimal
from pathlib import Path

_REPO = "pydantic/genai-prices"
_DATA_PATH = "prices/data.json"
_COMMIT_URL = f"https://api.github.com/repos/{_REPO}/commits/main"
_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent / "src/provider_runtime/prices_snapshot.json"
)

# Upstream provider id → our ProviderName. Everything else is dropped.
_PROVIDER_NAMES: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "moonshotai": "moonshot",
    "deepseek": "deepseek",
    "x-ai": "xai",
}


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "provider-runtime-refresh-prices"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def _base_rate(value: object) -> Decimal:
    """Flatten a rate to Decimal: plain number, or tiered {base, tiers} → base."""
    if isinstance(value, dict):
        value = value["base"]
    if isinstance(value, Decimal | int):
        return Decimal(value)
    raise ValueError(f"unexpected rate shape: {value!r}")


def _window_hours(constraint: dict[str, object]) -> float:
    """Hours a daily UTC ``start_time``..``end_time`` window covers, wrapping midnight."""
    start = _time.fromisoformat(str(constraint["start_time"]).removesuffix("Z"))
    end = _time.fromisoformat(str(constraint["end_time"]).removesuffix("Z"))
    start_h = start.hour + start.minute / 60 + start.second / 3600
    end_h = end.hour + end.minute / 60 + end.second / 3600
    span = end_h - start_h
    return span if span > 0 else span + 24.0


def _resolve_prices(prices: object, today: date) -> dict[str, object] | None:
    """Pick the price set in force today; None when no entry applies.

    Upstream rule: the LAST ``start_date``-constrained entry with
    ``start_date`` <= today wins. A time-of-day constraint (a daily UTC
    window) has no request timestamp to resolve against, so it is picked as
    the flat rate only when its window covers MORE than half the day — that
    is the standard/peak rate, and the unconstrained fallback is the minority
    off-peak discount. A window covering half the day or less is a genuine
    off-peak discount and never overrides the otherwise-chosen entry.
    """
    if isinstance(prices, dict):
        return prices
    assert isinstance(prices, list)
    chosen: dict[str, object] | None = None
    for entry in prices:
        constraint = entry.get("constraint")
        if constraint is None:
            chosen = entry["prices"]
        elif "start_date" in constraint:
            if date.fromisoformat(constraint["start_date"]) <= today:
                chosen = entry["prices"]
        elif "start_time" in constraint and "end_time" in constraint:
            if _window_hours(constraint) > 12.0:
                chosen = entry["prices"]
            # else: a minority off-peak window — never overrides.
        else:
            raise ValueError(f"unrecognized price constraint shape: {constraint!r}")
    return chosen


def _rows(upstream: list[dict[str, object]], today: date) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for provider in upstream:
        provider_name = _PROVIDER_NAMES.get(str(provider["id"]))
        if provider_name is None:
            continue
        models = provider["models"]
        assert isinstance(models, list)
        for model in models:
            prices = _resolve_prices(model["prices"], today)
            if prices is None:
                continue
            if "input_mtok" not in prices or "output_mtok" not in prices:
                continue  # embeddings/image/moderation — not generate-shaped
            key = (provider_name, str(model["id"]))
            if key in seen:
                raise ValueError(f"duplicate upstream model {key!r}")
            seen.add(key)
            input_rate = _base_rate(prices["input_mtok"])
            cache_read = (
                _base_rate(prices["cache_read_mtok"]) if "cache_read_mtok" in prices else input_rate
            )
            cache_write_surcharge = (
                _base_rate(prices["cache_write_mtok"]) - input_rate
                if "cache_write_mtok" in prices
                else Decimal(0)
            )
            rows.append(
                {
                    "provider": provider_name,
                    "model": str(model["id"]),
                    "input_mtok_usd": str(input_rate),
                    "output_mtok_usd": str(_base_rate(prices["output_mtok"])),
                    "cache_read_mtok_usd": str(cache_read),
                    "cache_write_surcharge_mtok_usd": str(cache_write_surcharge),
                }
            )
    rows.sort(key=lambda row: (row["provider"], row["model"]))
    return rows


def main() -> None:
    commit = json.loads(_fetch(_COMMIT_URL))
    sha = str(commit["sha"])
    data_url = f"https://raw.githubusercontent.com/{_REPO}/{sha}/{_DATA_PATH}"
    upstream = json.loads(_fetch(data_url), parse_float=Decimal)
    today = date.today()
    snapshot = {
        "snapshot_date": today.isoformat(),
        "upstream": f"{_REPO}@{sha} {_DATA_PATH}",
        "rows": _rows(upstream, today),
    }
    _SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(snapshot['rows'])} rows to {_SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
