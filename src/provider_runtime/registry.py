"""Model registry — hand-curated capability rows and their resolution.

Rows are contract facts (context windows, output caps, native reasoning wire
values, correlation, OpenRouter routing pins); the runtime resolves every call
through them and engines never hardcode model behavior. `REGISTRY_REVISION`
is stamped into every `CallMeta` and bumped on any row change.

Layering: imports from `types` and `errors` only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from provider_runtime.errors import InvalidRequest, RuntimeDefect
from provider_runtime.types import (
    Absent,
    Presence,
    Present,
    ProviderName,
    ProviderTarget,
    ReasoningLevel,
)

# ---------------------------------------------------------------------------
# Rows

type EngineId = Literal["openai_responses", "openai_chat", "anthropic_messages", "gemini_generate"]

# The provider→engine dialect table (spec §6): OpenAI proper on the native
# Responses API; deepseek/moonshot/xai/openrouter on the openai SDK as a
# compatibility client; anthropic and gemini on their native SDKs.
_ENGINE_FOR_PROVIDER: Mapping[ProviderName, EngineId] = {
    "openai": "openai_responses",
    "anthropic": "anthropic_messages",
    "gemini": "gemini_generate",
    "moonshot": "openai_chat",
    "openrouter": "openai_chat",
    "deepseek": "openai_chat",
    "xai": "openai_chat",
}

_REASONING_LEVELS: frozenset[str] = frozenset(get_args(ReasoningLevel.__value__))


@dataclass(frozen=True, slots=True)
class OpenRouterRouting:
    """Routing/privacy pins sent on every OpenRouter call — no unpinned passthrough."""

    only: tuple[str, ...]
    order: tuple[str, ...]
    quantizations: tuple[str, ...]
    allow_fallbacks: Literal[False] = False
    require_parameters: Literal[True] = True
    data_collection: Literal["deny"] = "deny"
    zdr: Literal[True] = True

    def __post_init__(self) -> None:
        # Zero pins would be unpinned passthrough with fallbacks merely
        # disabled — the state this type exists to make unrepresentable.
        for label, pins in (
            ("only", self.only),
            ("order", self.order),
            ("quantizations", self.quantizations),
        ):
            if not pins:
                raise ValueError(f"OpenRouterRouting.{label} must name at least one pin")


@dataclass(frozen=True, slots=True)
class ModelRow:
    ref: str  # "provider:model-nickname" — e.g. "anthropic:claude-opus-5"
    provider: ProviderName
    model_id: str  # wire model id
    engine: EngineId
    base_url: Presence[str]  # Absent → SDK default
    context_window: int
    max_output_tokens: int
    modalities: frozenset[Literal["text", "image"]]
    tools: bool
    streaming: bool
    structured: Literal["native", "json_mode"]
    # Exact native reasoning wire value per supported level; Absent when the
    # model has no reasoning knob.
    reasoning: Presence[Mapping[ReasoningLevel, object]]
    continuation_codec: str  # codec_id continuations bind to
    correlation: Literal["header", "in_band", "none"]
    routing: Presence[OpenRouterRouting]  # Present iff provider == "openrouter"


# Bumped on any row change; stamped into every CallMeta (ledger-consumed).
REGISTRY_REVISION: str = "2026-08-09.1"

ROWS: tuple[ModelRow, ...] = ()


# ---------------------------------------------------------------------------
# Resolution


def resolve(ref: str) -> ModelRow:
    """Look up a row by its "provider:nickname" ref."""
    for row in ROWS:
        if row.ref == ref:
            return row
    raise InvalidRequest(message=f"unknown model ref {ref!r}")


def resolve_target(target: ProviderTarget) -> ModelRow:
    """Look up a row by exact (provider, wire model id) match."""
    for row in ROWS:
        if row.provider == target.provider and row.model_id == target.model:
            return row
    raise InvalidRequest(
        message=f"no registry row for provider {target.provider!r} model {target.model!r}"
    )


# ---------------------------------------------------------------------------
# Row invariants — enforced once at module import; a violated row is a defect.


def _invalid(detail: str) -> RuntimeDefect:
    return RuntimeDefect(origin="intent", code="registry_invalid", message=detail)


def _validate_rows(rows: tuple[ModelRow, ...]) -> None:
    seen_refs: set[str] = set()
    seen_targets: set[tuple[ProviderName, str]] = set()
    for row in rows:
        if row.ref in seen_refs:
            raise _invalid(f"duplicate ref {row.ref!r}")
        seen_refs.add(row.ref)
        # (provider, model_id) is the resolve_target key — two rows sharing it
        # would make resolution row-order-dependent.
        target_key = (row.provider, row.model_id)
        if target_key in seen_targets:
            raise _invalid(f"duplicate (provider, model_id) {target_key!r}")
        seen_targets.add(target_key)
        if row.context_window <= 0 or row.max_output_tokens <= 0:
            raise _invalid(f"row {row.ref!r} token limits must be positive")
        prefix, sep, nickname = row.ref.partition(":")
        if sep != ":" or prefix != row.provider or not nickname:
            raise _invalid(f"ref {row.ref!r} must be '{row.provider}:<model-nickname>'")
        if row.provider == "openrouter":
            match row.routing:
                case Present(value=routing):
                    # Belt over the Literal types: the negative gate "every
                    # openrouter row pins allow_fallbacks=False" holds at
                    # import, not just under pyright.
                    if routing.allow_fallbacks or not routing.require_parameters:
                        raise _invalid(f"openrouter row {row.ref!r} must pin routing")
                case Absent():
                    raise _invalid(f"openrouter row {row.ref!r} must pin routing")
        if row.provider != "openrouter" and not isinstance(row.routing, Absent):
            raise _invalid(f"non-openrouter row {row.ref!r} must not carry routing")
        if isinstance(row.reasoning, Present):
            unknown_levels = sorted(set(row.reasoning.value) - _REASONING_LEVELS)
            if unknown_levels:
                raise _invalid(
                    f"row {row.ref!r} reasoning keys {unknown_levels!r} are outside ReasoningLevel"
                )
        expected_engine = _ENGINE_FOR_PROVIDER[row.provider]
        if row.engine != expected_engine:
            raise _invalid(
                f"row {row.ref!r} engine {row.engine!r} must be {expected_engine!r} "
                f"for provider {row.provider!r}"
            )


_validate_rows(ROWS)
