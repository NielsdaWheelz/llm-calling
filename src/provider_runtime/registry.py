"""Model registry — hand-curated capability rows and their resolution.

Rows are contract facts (context windows, output caps, native reasoning wire
values, correlation, OpenRouter routing pins); the runtime resolves every call
through them and engines never hardcode model behavior. `REGISTRY_REVISION`
is stamped into every `CallMeta` and bumped on any row change.

Curation policy (what earns and shapes a row):
- One screen per provider. A model gets a row only with a real owner: current
  Nexus consumption or the spec §12 seven-provider acceptance matrix.
- `reasoning` maps ONLY provider-documented levels to self-describing wire
  fragments: a `Mapping[str, JSON]` of request parameters the engine merges
  verbatim into its request (SDK-known keys as kwargs, the rest through the
  SDK's extra_body/config escape hatch). Engines hold zero per-provider
  reasoning shape knowledge, so the whole knob — key path and value — lives
  here. No invented levels: a model without "xhigh" simply omits the key; a
  model that can't disable reasoning omits "none".
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


def _invalid(detail: str) -> RuntimeDefect:
    return RuntimeDefect(origin="intent", code="registry_invalid", message=detail)


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
                raise _invalid(f"OpenRouterRouting.{label} must name at least one pin")


@dataclass(frozen=True, slots=True)
class ModelRow:
    ref: str  # "provider:model-nickname" — e.g. "anthropic:claude-opus-5"
    provider: ProviderName
    model_id: str  # wire model id
    engine: EngineId
    base_url: Presence[str]  # Absent → engine pins the canonical host, never env-resolved
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
REGISTRY_REVISION: str = "2026-08-10.2"

_TEXT_ONLY: Final[frozenset[Literal["text", "image"]]] = frozenset({"text"})
_TEXT_AND_IMAGE: Final[frozenset[Literal["text", "image"]]] = frozenset({"text", "image"})

# ---------------------------------------------------------------------------
# openai — developers.openai.com/api/docs/guides/reasoning +
# /api/docs/models/gpt-5.6-{sol,terra,luna} (verified 2026-08-10): the Responses
# knob is the top-level `reasoning` object; all three tiers accept
# none|low|medium|high|xhigh|max (no "minimal"), xhigh and max are DISTINCT
# efforts, default high. Image input: /api/docs/guides/images-vision.

_GPT56_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "none": {"reasoning": {"effort": "none"}},
    "low": {"reasoning": {"effort": "low"}},
    "medium": {"reasoning": {"effort": "medium"}},
    "high": {"reasoning": {"effort": "high"}},
    "xhigh": {"reasoning": {"effort": "xhigh"}},
    "max": {"reasoning": {"effort": "max"}},
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
# anthropic — platform.claude.com/docs/en/build-with-claude/effort (verified
# 2026-08-10): the knob is the request-level `output_config.effort`, levels
# low|medium|high|xhigh|max on both Sonnet 5 and Fable 5 (default high). Effort
# is not a thinking switch and has no "off" value, so no "none" key. Vision on
# both rows: platform.claude.com models overview (re-checked 2026-08-10).

_CLAUDE_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "low": {"output_config": {"effort": "low"}},
    "medium": {"output_config": {"effort": "medium"}},
    "high": {"output_config": {"effort": "high"}},
    "xhigh": {"output_config": {"effort": "xhigh"}},
    "max": {"output_config": {"effort": "max"}},
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
        structured="native",  # output_config.format json_schema (GA)
        reasoning=Present(_CLAUDE_REASONING),
        continuation_codec="anthropic.v1",
        correlation="header",  # request-id
        routing=Absent(),
    )


# ---------------------------------------------------------------------------
# gemini — ai.google.dev/gemini-api/docs/thinking + /docs/whats-new-gemini-3.5
# (verified 2026-08-10): 3.5 Flash takes thinking_config.thinking_level
# minimal|low|medium|high (default medium); 1,048,576 in / 65,536 out. A 2.5
# row would carry {"thinking_config": {"thinking_budget": N}} instead — the two
# cannot be sent together (400) and the fragment picks exactly one, so the
# engine needs no generation knowledge (spec §6). No correlation id on this wire.

_GEMINI_35_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "minimal": {"thinking_config": {"thinking_level": "minimal"}},
    "low": {"thinking_config": {"thinking_level": "low"}},
    "medium": {"thinking_config": {"thinking_level": "medium"}},
    "high": {"thinking_config": {"thinking_level": "high"}},
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
# moonshot — platform.kimi.ai/docs/api/chat (verified 2026-08-10): K3 always
# reasons and takes the top-level `reasoning_effort` low|high|max (default max),
# so no "none" key. Output cap is the provider's documented default
# max_completion_tokens (its hard max equals the context size). K3 is natively
# multimodal (platform.kimi.ai vision guide, re-checked 2026-08-10): base64
# image parts on the same model id.

_KIMI_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "low": {"reasoning_effort": "low"},
    "high": {"reasoning_effort": "high"},
    "max": {"reasoning_effort": "max"},
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
# openrouter — endpoint metadata verified 2026-08-10 against
# https://openrouter.ai/api/v1/models/moonshotai/kimi-k3/endpoints: the first-party
# endpoint is "Moonshot AI | moonshotai/kimi-k3-20260715", tag "moonshotai/mxfp4",
# quantization "mxfp4", context 1,048,576 (K3 ships natively in MXFP4; the int4
# variant was K2-era and no K3 endpoint serves it). The wire id is that pinned
# dated revision; the endpoint tag is sent as only+order and its quantization
# re-sent as an explicit filter. The unified `reasoning.effort` object is the
# gateway knob (endpoint supported_parameters lists "reasoning"). Text-only
# until the pinned upstream's image acceptance is live-verified.

_OPENROUTER_KIMI_PIN: Final = "moonshotai/mxfp4"

_OPENROUTER_KIMI_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "low": {"reasoning": {"effort": "low"}},
    "high": {"reasoning": {"effort": "high"}},
    "max": {"reasoning": {"effort": "max"}},
}

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
    reasoning=Present(_OPENROUTER_KIMI_REASONING),
    continuation_codec="openrouter.v1",
    correlation="in_band",  # generation id
    routing=Present(
        OpenRouterRouting(
            only=(_OPENROUTER_KIMI_PIN,),
            order=(_OPENROUTER_KIMI_PIN,),
            quantizations=("mxfp4",),
        )
    ),
)

# ---------------------------------------------------------------------------
# deepseek — NEW (first-time coverage), verified 2026-08-10 against
# api-docs.deepseek.com (models & pricing, guides/thinking_mode, chat-completion
# reference): model ids deepseek-v4-pro / deepseek-v4-flash (deepseek-chat /
# deepseek-reasoner retired 2026-07-24); 1,000,000-token context and a
# 384,000-token output cap for both, thinking and non-thinking alike;
# response_format supports json_object only (no json_schema) → json_mode; the
# hosted API is text-only. The knob is two parameters, both named here: the
# top-level `reasoning_effort` low|high|max and the `thinking` switch
# ({"type": "enabled"|"disabled"}, enabled by default, effort default high) —
# the docs' own example sends them together. Flash honors all three efforts;
# pro maps low→high, so "low" is omitted (no invented levels).

_DEEPSEEK_V4_FLASH_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "none": {"thinking": {"type": "disabled"}},
    "low": {"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
    "high": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    "max": {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
}

_DEEPSEEK_V4_PRO_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "none": {"thinking": {"type": "disabled"}},
    "high": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    "max": {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
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
# (developers/grok-4-5, developers/model-capabilities/text/reasoning): flagship
# grok-4.5 (no current mini); 500,000-token context; xAI publishes no separate
# output cap — the shared window is the only documented bound; text+image input;
# function calling and structured outputs; the chat-completions knob is the
# top-level `reasoning_effort` low|medium|high (default high) and reasoning
# cannot be disabled, so no "none" key.

_GROK_45_REASONING: Final[Mapping[ReasoningLevel, object]] = {
    "low": {"reasoning_effort": "low"},
    "medium": {"reasoning_effort": "medium"},
    "high": {"reasoning_effort": "high"},
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


def _validate_reasoning(row: ModelRow) -> None:
    """Declared levels only, each mapping to a fragment an engine can merge.

    Engines merge `row.reasoning[level]` verbatim into their request and stamp
    it as JSON; a non-mapping, non-string-keyed, or empty value is a curation
    defect that would otherwise surface as a call-time defect on first dispatch.
    """
    if not isinstance(row.reasoning, Present):
        return
    levels = row.reasoning.value
    if not levels:
        raise _invalid(f"row {row.ref!r} declares a reasoning knob with no levels")
    unknown_levels = sorted(set(levels) - _REASONING_LEVELS)
    if unknown_levels:
        raise _invalid(
            f"row {row.ref!r} reasoning keys {unknown_levels!r} are outside ReasoningLevel"
        )
    for level, fragment in levels.items():
        if (
            not isinstance(fragment, Mapping)
            or not fragment
            or any(not isinstance(key, str) for key in fragment)
        ):
            raise _invalid(
                f"row {row.ref!r} level {level!r} must map to a non-empty string-keyed "
                f"request fragment, got {fragment!r}"
            )


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
            if not isinstance(row.routing, Present):
                raise _invalid(f"openrouter row {row.ref!r} must pin routing")
            # The two pins that make "no implicit rerouting" true on the wire:
            # fixed in the type, re-checked here because a row is data and this
            # is where curation defects are caught.
            routing = row.routing.value
            if routing.allow_fallbacks or not routing.require_parameters:
                raise _invalid(
                    f"openrouter row {row.ref!r} must pin allow_fallbacks=False and "
                    f"require_parameters=True; got allow_fallbacks="
                    f"{routing.allow_fallbacks!r}, require_parameters="
                    f"{routing.require_parameters!r}"
                )
        elif isinstance(row.routing, Present):
            raise _invalid(f"non-openrouter row {row.ref!r} must not carry routing")
        _validate_reasoning(row)
        expected_engine = _ENGINE_FOR_PROVIDER[row.provider]
        if row.engine != expected_engine:
            raise _invalid(
                f"row {row.ref!r} engine {row.engine!r} must be {expected_engine!r} "
                f"for provider {row.provider!r}"
            )
        # openai_chat serves compatibility hosts only: the SDK would otherwise
        # aim at api.openai.com, so an Absent base_url defects at client
        # construction on the first call.
        if row.engine == "openai_chat" and not isinstance(row.base_url, Present):
            raise _invalid(f"openai_chat row {row.ref!r} must carry a base_url")


_validate_rows(ROWS)
