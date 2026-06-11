"""Shared model capability catalog for provider-runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from provider_runtime.types import ModelRef, PromptCacheTTL, ProviderName, ReasoningEffort

PromptCacheMode = Literal["none", "turn_ttl", "keyed_ttl"]
type PriceValue = Decimal | int | float | str
PricingUnit = Literal["per_million_tokens", "provider_units"]
ReasoningBillingMode = Literal["included_in_output", "separate", "not_billed", "unknown"]


@dataclass(frozen=True)
class PromptCacheCapability:
    mode: PromptCacheMode
    ttl_options: tuple[PromptCacheTTL, ...] = ()
    requires_key: bool = False
    affinity_hints: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return self.mode != "none"


@dataclass(frozen=True)
class Pricing:
    input_per_million: PriceValue | None = None
    output_per_million: PriceValue | None = None
    cached_input_per_million: PriceValue | None = None
    cache_write_per_million_by_ttl: dict[PromptCacheTTL, PriceValue] = field(default_factory=dict)
    reasoning_per_million: PriceValue | None = None
    reasoning_billing_mode: ReasoningBillingMode = "unknown"
    source_url: str | None = None
    verified_at: str | None = None
    currency: str = "USD"
    unit: PricingUnit = "per_million_tokens"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_per_million", _decimal_or_none(self.input_per_million))
        object.__setattr__(self, "output_per_million", _decimal_or_none(self.output_per_million))
        object.__setattr__(
            self,
            "cached_input_per_million",
            _decimal_or_none(self.cached_input_per_million),
        )
        object.__setattr__(
            self, "reasoning_per_million", _decimal_or_none(self.reasoning_per_million)
        )
        object.__setattr__(
            self,
            "cache_write_per_million_by_ttl",
            {
                ttl: price
                for ttl, raw_price in self.cache_write_per_million_by_ttl.items()
                if (price := _decimal_or_none(raw_price)) is not None
            },
        )

    def to_json(self) -> dict[str, object]:
        return {
            "input_per_million": _decimal_string(self.input_per_million),
            "output_per_million": _decimal_string(self.output_per_million),
            "cached_input_per_million": _decimal_string(self.cached_input_per_million),
            "cache_write_per_million_by_ttl": {
                ttl: _decimal_string(price)
                for ttl, price in self.cache_write_per_million_by_ttl.items()
            },
            "reasoning_per_million": _decimal_string(self.reasoning_per_million),
            "reasoning_billing_mode": self.reasoning_billing_mode,
            "source_url": self.source_url,
            "verified_at": self.verified_at,
            "currency": self.currency,
            "unit": self.unit,
        }


def _decimal_or_none(value: PriceValue | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _decimal_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return format(value, "f")
    return format(Decimal(str(value)), "f")


@dataclass(frozen=True)
class RouteCapability:
    route: ProviderName
    provider: ProviderName


@dataclass(frozen=True)
class ModelCapability:
    provider: ProviderName
    model: str
    routes: tuple[RouteCapability, ...]
    default_route: ProviderName
    key_probe_model: str
    reasoning_modes: tuple[ReasoningEffort, ...]
    reasoning_budget_tokens: tuple[int, ...] = ()
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    prompt_cache: PromptCacheCapability = field(
        default_factory=lambda: PromptCacheCapability("none")
    )
    streaming: bool = True
    tool_calling: bool = True
    tool_choice_required: bool = True
    structured_output: bool = True
    structured_output_streaming: bool = False
    reasoning_continuation: bool = False
    multimodal_input: bool = False
    embeddings: bool = False
    transcription: bool = False
    provider_request_id: bool = True
    usage_input_output_tokens: bool = True
    usage_reasoning_tokens: bool = False
    usage_cache_read_write_tokens: bool = False
    raw_artifact_support: bool = False
    retryable_errors: tuple[str, ...] = ("rate_limit", "timeout", "provider_down")
    default_timeout_s: int = 45
    max_timeout_s: int = 180
    pricing: Pricing = field(default_factory=Pricing)


class ModelCatalog:
    """Immutable lookup wrapper for per-model capabilities."""

    def __init__(self, entries: tuple[ModelCapability, ...]):
        self._entries = entries
        self._by_key = {(entry.provider, entry.model): entry for entry in entries}
        self._probe_models = {
            entry.provider: entry.key_probe_model
            for entry in entries
            if entry.model == entry.key_probe_model
        }
        for entry in entries:
            self._probe_models.setdefault(entry.provider, entry.key_probe_model)

    @property
    def entries(self) -> tuple[ModelCapability, ...]:
        return self._entries

    def capabilities(self, model: ModelRef) -> ModelCapability | None:
        provider = model.route or model.provider
        return self._by_key.get((provider, model.model))

    def require_capabilities(self, model: ModelRef) -> ModelCapability:
        capability = self.capabilities(model)
        if capability is None:
            raise KeyError(
                f"Unknown model capability: {model.route or model.provider}/{model.model}"
            )
        return capability

    def key_probe_model(self, provider: ProviderName) -> str | None:
        return self._probe_models.get(provider)


def _cap(
    provider: ProviderName,
    model: str,
    *,
    reasoning_modes: tuple[ReasoningEffort, ...],
    key_probe_model: str,
    max_context_tokens: int,
    max_output_tokens: int,
    prompt_cache: PromptCacheCapability | None = None,
    structured_output: bool = True,
    embeddings: bool = False,
    transcription: bool = False,
    multimodal_input: bool = False,
    raw_artifact_support: bool = False,
    usage_reasoning_tokens: bool = False,
    usage_cache_read_write_tokens: bool = False,
    reasoning_continuation: bool = False,
) -> ModelCapability:
    return ModelCapability(
        provider=provider,
        model=model,
        routes=(RouteCapability(route=provider, provider=provider),),
        default_route=provider,
        key_probe_model=key_probe_model,
        reasoning_modes=reasoning_modes,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        prompt_cache=prompt_cache or PromptCacheCapability("none"),
        structured_output=structured_output,
        embeddings=embeddings,
        transcription=transcription,
        multimodal_input=multimodal_input,
        raw_artifact_support=raw_artifact_support,
        usage_reasoning_tokens=usage_reasoning_tokens,
        usage_cache_read_write_tokens=usage_cache_read_write_tokens,
        reasoning_continuation=reasoning_continuation,
    )


_OPENAI_REASONING: tuple[ReasoningEffort, ...] = (
    "default",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "max",
)
_ANTHROPIC_REASONING: tuple[ReasoningEffort, ...] = (
    "default",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "max",
)


DEFAULT_CATALOG = ModelCatalog(
    (
        _cap(
            "openai",
            "gpt-5.5",
            reasoning_modes=_OPENAI_REASONING,
            key_probe_model="gpt-5.4-mini",
            max_context_tokens=400000,
            max_output_tokens=128000,
            prompt_cache=PromptCacheCapability("keyed_ttl", ("5m", "1h"), requires_key=True),
            raw_artifact_support=True,
            usage_reasoning_tokens=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
        ),
        _cap(
            "openai",
            "gpt-5.4-mini",
            reasoning_modes=_OPENAI_REASONING,
            key_probe_model="gpt-5.4-mini",
            max_context_tokens=400000,
            max_output_tokens=128000,
            prompt_cache=PromptCacheCapability("keyed_ttl", ("5m", "1h"), requires_key=True),
            raw_artifact_support=True,
            usage_reasoning_tokens=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
        ),
        _cap(
            "openai",
            "text-embedding-3-small",
            reasoning_modes=("none",),
            key_probe_model="gpt-5.4-mini",
            max_context_tokens=8191,
            max_output_tokens=0,
            structured_output=False,
            embeddings=True,
        ),
        _cap(
            "openai",
            "gpt-4o-transcribe",
            reasoning_modes=("none",),
            key_probe_model="gpt-5.4-mini",
            max_context_tokens=0,
            max_output_tokens=0,
            structured_output=False,
            transcription=True,
        ),
        _cap(
            "anthropic",
            "claude-3-opus-20240229",
            reasoning_modes=_ANTHROPIC_REASONING,
            key_probe_model="claude-haiku-4-5-20251001",
            max_context_tokens=200000,
            max_output_tokens=8192,
            prompt_cache=PromptCacheCapability("turn_ttl", ("5m", "1h")),
            raw_artifact_support=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
        ),
        _cap(
            "anthropic",
            "claude-opus-4-7",
            reasoning_modes=_ANTHROPIC_REASONING,
            key_probe_model="claude-haiku-4-5-20251001",
            max_context_tokens=1000000,
            max_output_tokens=32000,
            prompt_cache=PromptCacheCapability("turn_ttl", ("5m", "1h")),
            raw_artifact_support=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
        ),
        _cap(
            "anthropic",
            "claude-sonnet-4-6",
            reasoning_modes=_ANTHROPIC_REASONING,
            key_probe_model="claude-haiku-4-5-20251001",
            max_context_tokens=1000000,
            max_output_tokens=32000,
            prompt_cache=PromptCacheCapability("turn_ttl", ("5m", "1h")),
            raw_artifact_support=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
        ),
        _cap(
            "anthropic",
            "claude-haiku-4-5-20251001",
            reasoning_modes=_ANTHROPIC_REASONING,
            key_probe_model="claude-haiku-4-5-20251001",
            max_context_tokens=200000,
            max_output_tokens=8192,
            prompt_cache=PromptCacheCapability("turn_ttl", ("5m", "1h")),
            raw_artifact_support=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
        ),
        _cap(
            "gemini",
            "gemini-2.5-pro",
            reasoning_modes=("default", "none", "minimal", "low", "medium", "high", "max"),
            key_probe_model="gemini-3-flash-preview",
            max_context_tokens=1048576,
            max_output_tokens=65536,
            raw_artifact_support=True,
            reasoning_continuation=True,
        ),
        _cap(
            "gemini",
            "gemini-2.5-flash",
            reasoning_modes=("default", "none", "minimal", "low", "medium", "high", "max"),
            key_probe_model="gemini-3-flash-preview",
            max_context_tokens=1048576,
            max_output_tokens=65536,
            multimodal_input=True,
            raw_artifact_support=True,
            reasoning_continuation=True,
        ),
        _cap(
            "gemini",
            "gemini-3.1-pro-preview",
            reasoning_modes=("default", "none", "minimal", "low", "medium", "high", "max"),
            key_probe_model="gemini-3-flash-preview",
            max_context_tokens=1048576,
            max_output_tokens=65536,
            raw_artifact_support=True,
            reasoning_continuation=True,
        ),
        _cap(
            "gemini",
            "gemini-3-flash-preview",
            reasoning_modes=("default", "none", "minimal", "low", "medium", "high", "max"),
            key_probe_model="gemini-3-flash-preview",
            max_context_tokens=1048576,
            max_output_tokens=65536,
            raw_artifact_support=True,
            reasoning_continuation=True,
        ),
        _cap(
            "openrouter",
            "moonshotai/kimi-k2.6",
            reasoning_modes=("default", "none", "minimal", "low", "medium", "high", "max"),
            key_probe_model="openai/gpt-5.4-mini",
            max_context_tokens=128000,
            max_output_tokens=32000,
        ),
        _cap(
            "openrouter",
            "deepseek/deepseek-v3.2",
            reasoning_modes=("default", "none", "minimal", "low", "medium", "high", "max"),
            key_probe_model="openai/gpt-5.4-mini",
            max_context_tokens=128000,
            max_output_tokens=32000,
        ),
        _cap(
            "openrouter",
            "openai/gpt-5.5",
            reasoning_modes=_OPENAI_REASONING,
            key_probe_model="openai/gpt-5.4-mini",
            max_context_tokens=128000,
            max_output_tokens=32000,
        ),
        _cap(
            "openrouter",
            "openai/gpt-5.4-mini",
            reasoning_modes=_OPENAI_REASONING,
            key_probe_model="openai/gpt-5.4-mini",
            max_context_tokens=128000,
            max_output_tokens=32000,
        ),
        _cap(
            "cloudflare",
            "@cf/meta/llama-3.1-8b-instruct",
            reasoning_modes=("default", "none"),
            key_probe_model="@cf/meta/llama-3.1-8b-instruct",
            max_context_tokens=8192,
            max_output_tokens=4096,
            structured_output=False,
        ),
        _cap(
            "cloudflare",
            "text-embedding-3-small",
            reasoning_modes=("none",),
            key_probe_model="@cf/meta/llama-3.1-8b-instruct",
            max_context_tokens=8192,
            max_output_tokens=0,
            structured_output=False,
            embeddings=True,
        ),
    )
)
