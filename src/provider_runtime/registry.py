"""Model registry — hand-curated capability rows and their resolution.

Rows are contract facts (context windows, output caps, native reasoning wire
values, correlation, OpenRouter routing pins); the runtime resolves every call
through them and engines never hardcode model behavior. `REGISTRY_REVISION`
is stamped into every `CallMeta` and bumped on any row change.

Curation policy (what earns and shapes a row):
- One screen per provider. A model gets a row only with a real owner: current
  Nexus consumption or the spec §12 seven-provider acceptance matrix.
- `reasoning` maps ONLY provider-documented levels to exact native wire
  values — a str is the effort/level knob string sent verbatim; a mapping is
  the exact request fragment for config-object switches (deepseek "none" →
  {"thinking": {"type": "disabled"}}). No invented levels: a model without
  "xhigh" simply omits the key; a model that can't disable reasoning omits
  "none".
- `context_window`/`max_output_tokens` are honest provider-doc figures; where
  the provider publishes no separate output cap the shared context window is
  the recorded bound.
- `continuation_codec` is the freeze table's "<provider>.v1"; the suffix bumps
  only when a provider's opaque payload shape changes incompatibly.
- openrouter rows pin their upstream completely (no unpinned passthrough);
  every other row carries no routing.

Layering: imports from `types` and `errors` only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, get_args

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
REGISTRY_REVISION: str = "2026-08-10.1"

_TEXT_ONLY: Final[frozenset[Literal["text", "image"]]] = frozenset({"text"})
_TEXT_AND_IMAGE: Final[frozenset[Literal["text", "image"]]] = frozenset({"text", "image"})

# ---------------------------------------------------------------------------
# openai — ported from catalog.py (docs verified 2026-07-20). Responses
# `reasoning.effort` strings verbatim; xhigh and max are DISTINCT efforts.
# Image input: developers.openai.com/api/docs/guides/images-vision (all
# gpt-5.6 tiers, re-checked 2026-08-10).

_GPT56_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def _gpt56_row(model_id: str) -> ModelRow:
    return ModelRow(
        ref=f"openai:{model_id}",
        provider="openai",
        model_id=model_id,
        engine="openai_responses",
        base_url=Absent(),
        context_window=1_050_000,
        max_output_tokens=128_000,
        modalities=_TEXT_AND_IMAGE,
        tools=True,
        streaming=True,
        structured="native",  # Responses text.format json_schema
        reasoning=Present(_GPT56_REASONING),
        continuation_codec="openai.v1",
        correlation="header",  # x-request-id
        routing=Absent(),
    )


# ---------------------------------------------------------------------------
# anthropic — ported from catalog.py (docs verified 2026-07-20). Top-level
# output_config.effort strings; no off switch (fable: adaptive thinking is
# always on — an engine fact, not a row fact). Vision on both rows:
# platform.claude.com models overview (re-checked 2026-08-10).

_CLAUDE_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def _claude_row(model_id: str) -> ModelRow:
    return ModelRow(
        ref=f"anthropic:{model_id}",
        provider="anthropic",
        model_id=model_id,
        engine="anthropic_messages",
        base_url=Absent(),
        context_window=1_000_000,
        max_output_tokens=128_000,
        modalities=_TEXT_AND_IMAGE,
        tools=True,
        streaming=True,
        structured="native",  # output_format json_schema (GA)
        reasoning=Present(_CLAUDE_REASONING),
        continuation_codec="anthropic.v1",
        correlation="header",  # request-id
        routing=Absent(),
    )


# ---------------------------------------------------------------------------
# gemini — ported from catalog.py (docs verified 2026-07-20). 3.5 takes
# thinkingConfig.thinkingLevel strings (a 2.5 row would carry thinkingBudget
# ints instead — value shape selects the knob, spec §6). No correlation id on
# this wire.

_GEMINI_35_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}

_GEMINI_35_FLASH: Final = ModelRow(
    ref="gemini:gemini-3.5-flash",
    provider="gemini",
    model_id="gemini-3.5-flash",
    engine="gemini_generate",
    base_url=Absent(),
    context_window=1_048_576,
    max_output_tokens=65_536,
    modalities=_TEXT_AND_IMAGE,
    tools=True,
    streaming=True,
    structured="native",  # responseJsonSchema
    reasoning=Present(_GEMINI_35_REASONING),
    continuation_codec="gemini.v1",
    correlation="none",
    routing=Absent(),
)

# ---------------------------------------------------------------------------
# moonshot — ported from catalog.py (docs verified 2026-07-22). Direct wire
# `reasoning_effort`; output cap is the provider's documented default
# max_completion_tokens (its hard max equals the context size). K3 is natively
# multimodal (platform.kimi.ai vision guide, re-checked 2026-08-10): base64
# image parts on the same model id.

_KIMI_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "low": "low",
    "high": "high",
    "max": "max",
}

_KIMI_K3: Final = ModelRow(
    ref="moonshot:kimi-k3",
    provider="moonshot",
    model_id="kimi-k3",
    engine="openai_chat",
    base_url=Present("https://api.moonshot.ai/v1"),
    context_window=1_048_576,
    max_output_tokens=131_072,
    modalities=_TEXT_AND_IMAGE,
    tools=True,
    streaming=True,
    structured="json_mode",  # spec §5: json_out = JSON-mode + pydantic on moonshot
    reasoning=Present(_KIMI_REASONING),
    continuation_codec="moonshot.v1",
    correlation="in_band",  # response/chunk id; no confirmed header
    routing=Absent(),
)

# ---------------------------------------------------------------------------
# openrouter — ported from catalog.py (endpoint metadata verified 2026-07-20):
# wire id is the pinned canonical revision; the upstream endpoint slug is sent
# as only+order and its variant re-sent as an explicit quantization filter
# (belt over the UNCONFIRMED slug form). Unified `reasoning.effort` strings.
# Text-only until the pinned upstream's image acceptance is live-verified.

_OPENROUTER_KIMI_PIN: Final = "moonshotai/int4"

_OPENROUTER_KIMI_K3: Final = ModelRow(
    ref="openrouter:kimi-k3",
    provider="openrouter",
    model_id="moonshotai/kimi-k3-20260715",
    engine="openai_chat",
    base_url=Present("https://openrouter.ai/api/v1"),
    context_window=1_048_576,
    max_output_tokens=131_072,
    modalities=_TEXT_ONLY,
    tools=True,
    streaming=True,
    structured="json_mode",  # kimi upstream; require_parameters pins honesty
    reasoning=Present(_KIMI_REASONING),
    continuation_codec="openrouter.v1",
    correlation="in_band",  # generation id
    routing=Present(
        OpenRouterRouting(
            only=(_OPENROUTER_KIMI_PIN,),
            order=(_OPENROUTER_KIMI_PIN,),
            quantizations=("int4",),
        )
    ),
)

# ---------------------------------------------------------------------------
# deepseek — NEW (first-time coverage), verified 2026-08-10 against
# api-docs.deepseek.com (models & pricing, thinking-mode guide, chat-completion
# reference): model ids deepseek-v4-pro / deepseek-v4-flash (deepseek-chat /
# deepseek-reasoner retired 2026-07-24); 1,000,000-token context and a
# 384,000-token output cap for both, thinking and non-thinking alike;
# response_format supports json_object only (no json_schema) → json_mode; the
# hosted API is text-only; `reasoning_effort` accepts low|high|max on flash
# but only high|max on pro (low silently upgrades — omitted, no invented
# levels); thinking is on by default and disabled ONLY via the request
# fragment {"thinking": {"type": "disabled"}}.

_DEEPSEEK_THINKING_OFF: Final[Mapping[str, object]] = {"thinking": {"type": "disabled"}}

_DEEPSEEK_V4_FLASH_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "none": _DEEPSEEK_THINKING_OFF,
    "low": "low",
    "high": "high",
    "max": "max",
}

_DEEPSEEK_V4_PRO_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "none": _DEEPSEEK_THINKING_OFF,
    "high": "high",
    "max": "max",
}


def _deepseek_row(model_id: str, reasoning: Mapping[ReasoningLevel, object]) -> ModelRow:
    return ModelRow(
        ref=f"deepseek:{model_id}",
        provider="deepseek",
        model_id=model_id,
        engine="openai_chat",
        base_url=Present("https://api.deepseek.com"),
        context_window=1_000_000,
        max_output_tokens=384_000,
        modalities=_TEXT_ONLY,
        tools=True,
        streaming=True,
        structured="json_mode",
        reasoning=Present(reasoning),
        continuation_codec="deepseek.v1",
        correlation="in_band",  # chat-completions body id
        routing=Absent(),
    )


# ---------------------------------------------------------------------------
# xai — NEW (first-time coverage), verified 2026-08-10 against docs.x.ai
# (models/grok-4.5, guides/reasoning): flagship grok-4.5 (no current mini);
# 500,000-token context; xAI publishes no separate output cap — the shared
# window is the only documented bound; text+image input; function calling and
# structured outputs; `reasoning_effort` low|medium|high (default high),
# reasoning cannot be disabled.

_GROK_45_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}

_GROK_45: Final = ModelRow(
    ref="xai:grok-4.5",
    provider="xai",
    model_id="grok-4.5",
    engine="openai_chat",
    base_url=Present("https://api.x.ai/v1"),
    context_window=500_000,
    max_output_tokens=500_000,
    modalities=_TEXT_AND_IMAGE,
    tools=True,
    streaming=True,
    structured="native",  # structured outputs (spec §5 lists xai native)
    reasoning=Present(_GROK_45_REASONING),
    continuation_codec="xai.v1",
    correlation="in_band",  # chat-completions body id
    routing=Absent(),
)

ROWS: tuple[ModelRow, ...] = (
    _gpt56_row("gpt-5.6-sol"),
    _gpt56_row("gpt-5.6-terra"),
    _gpt56_row("gpt-5.6-luna"),
    _claude_row("claude-sonnet-5"),
    _claude_row("claude-fable-5"),
    _GEMINI_35_FLASH,
    _KIMI_K3,
    _OPENROUTER_KIMI_K3,
    _deepseek_row("deepseek-v4-pro", _DEEPSEEK_V4_PRO_REASONING),
    _deepseek_row("deepseek-v4-flash", _DEEPSEEK_V4_FLASH_REASONING),
    _GROK_45,
)


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
        if row.max_output_tokens > row.context_window:
            raise _invalid(
                f"row {row.ref!r} max_output_tokens {row.max_output_tokens} exceeds "
                f"context_window {row.context_window}"
            )
        if "text" not in row.modalities:
            raise _invalid(f"row {row.ref!r} modalities must include 'text'")
        # Continuations bind to the codec id; a codec owned by another provider
        # would replay opaque payloads across wire dialects.
        if not row.continuation_codec.startswith(f"{row.provider}."):
            raise _invalid(
                f"row {row.ref!r} continuation_codec {row.continuation_codec!r} must start "
                f"with '{row.provider}.'"
            )
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
