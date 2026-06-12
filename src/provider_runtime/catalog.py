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
    applies_up_to_input_tokens: int | None = None
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
        if self.applies_up_to_input_tokens is not None and self.applies_up_to_input_tokens <= 0:
            raise ValueError("Pricing.applies_up_to_input_tokens must be > 0")

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
            "applies_up_to_input_tokens": self.applies_up_to_input_tokens,
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
    generation: bool = True
    reasoning_budget_tokens: tuple[int, ...] = ()
    reasoning_budget_range: tuple[int, int] | None = None
    reasoning_allows_dynamic_budget: bool = False
    reasoning_reserve_tokens: dict[ReasoningEffort, int] = field(default_factory=dict)
    structured_output_reasoning_modes: tuple[ReasoningEffort, ...] | None = None
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
    generation: bool = True,
    reasoning_budget_tokens: tuple[int, ...] = (),
    reasoning_budget_range: tuple[int, int] | None = None,
    reasoning_allows_dynamic_budget: bool = False,
    reasoning_reserve_tokens: dict[ReasoningEffort, int] | None = None,
    structured_output_reasoning_modes: tuple[ReasoningEffort, ...] | None = None,
    prompt_cache: PromptCacheCapability | None = None,
    streaming: bool = True,
    tool_calling: bool = True,
    tool_choice_required: bool = True,
    structured_output: bool = True,
    structured_output_streaming: bool = False,
    embeddings: bool = False,
    transcription: bool = False,
    multimodal_input: bool = False,
    raw_artifact_support: bool = False,
    provider_request_id: bool = True,
    usage_input_output_tokens: bool = True,
    usage_reasoning_tokens: bool = False,
    usage_cache_read_write_tokens: bool = False,
    reasoning_continuation: bool = False,
    pricing: Pricing | None = None,
) -> ModelCapability:
    if not generation:
        streaming = False
        tool_calling = False
        tool_choice_required = False
    return ModelCapability(
        provider=provider,
        model=model,
        routes=(RouteCapability(route=provider, provider=provider),),
        default_route=provider,
        key_probe_model=key_probe_model,
        generation=generation,
        reasoning_modes=reasoning_modes,
        reasoning_budget_tokens=reasoning_budget_tokens,
        reasoning_budget_range=reasoning_budget_range,
        reasoning_allows_dynamic_budget=reasoning_allows_dynamic_budget,
        reasoning_reserve_tokens=reasoning_reserve_tokens or {},
        structured_output_reasoning_modes=structured_output_reasoning_modes,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        prompt_cache=prompt_cache or PromptCacheCapability("none"),
        streaming=streaming,
        tool_calling=tool_calling,
        tool_choice_required=tool_choice_required,
        structured_output=structured_output,
        structured_output_streaming=structured_output_streaming,
        embeddings=embeddings,
        transcription=transcription,
        multimodal_input=multimodal_input,
        raw_artifact_support=raw_artifact_support,
        provider_request_id=provider_request_id,
        usage_input_output_tokens=usage_input_output_tokens,
        usage_reasoning_tokens=usage_reasoning_tokens,
        usage_cache_read_write_tokens=usage_cache_read_write_tokens,
        reasoning_continuation=reasoning_continuation,
        pricing=pricing or Pricing(),
    )


_OPENAI_REASONING: tuple[ReasoningEffort, ...] = (
    "default",
    "none",
    "low",
    "medium",
    "high",
    "max",
)
_OPENAI_REASONING_RESERVES: dict[ReasoningEffort, int] = {
    "default": 2048,
    "none": 0,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "max": 16384,
}
_OPENROUTER_OPENAI_REASONING: tuple[ReasoningEffort, ...] = (
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
_GEMINI_25_PRO_REASONING: tuple[ReasoningEffort, ...] = (
    "default",
    "minimal",
    "low",
    "medium",
    "high",
    "max",
)
_GEMINI_25_FLASH_REASONING: tuple[ReasoningEffort, ...] = (
    "default",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "max",
)
_GEMINI_31_PRO_REASONING: tuple[ReasoningEffort, ...] = (
    "default",
    "low",
    "medium",
    "high",
    "max",
)
_GEMINI_3_FLASH_REASONING: tuple[ReasoningEffort, ...] = (
    "default",
    "minimal",
    "low",
    "medium",
    "high",
    "max",
)

_VERIFIED_AT = "2026-06-11"
_OPENAI_PRICING_URL = "https://openai.com/api/pricing/"
_OPENAI_DEVELOPER_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
_ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
_ANTHROPIC_MODELS_URL = "https://platform.claude.com/docs/en/about-claude/models/overview"
_GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_CLOUDFLARE_PRICING_URL = "https://developers.cloudflare.com/workers-ai/platform/pricing/"


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
            reasoning_reserve_tokens=_OPENAI_REASONING_RESERVES,
            raw_artifact_support=True,
            usage_reasoning_tokens=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
            pricing=Pricing(
                input_per_million="5",
                output_per_million="30",
                cached_input_per_million="0.5",
                reasoning_billing_mode="included_in_output",
                applies_up_to_input_tokens=270000,
                source_url=_OPENAI_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "openai",
            "gpt-5.4-mini",
            reasoning_modes=_OPENAI_REASONING,
            key_probe_model="gpt-5.4-mini",
            max_context_tokens=400000,
            max_output_tokens=128000,
            prompt_cache=PromptCacheCapability("keyed_ttl", ("5m", "1h"), requires_key=True),
            reasoning_reserve_tokens=_OPENAI_REASONING_RESERVES,
            raw_artifact_support=True,
            usage_reasoning_tokens=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
            pricing=Pricing(
                input_per_million="0.75",
                output_per_million="4.5",
                cached_input_per_million="0.075",
                reasoning_billing_mode="included_in_output",
                applies_up_to_input_tokens=270000,
                source_url=_OPENAI_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "openai",
            "text-embedding-3-small",
            reasoning_modes=("none",),
            key_probe_model="gpt-5.4-mini",
            max_context_tokens=8191,
            max_output_tokens=0,
            generation=False,
            structured_output=False,
            embeddings=True,
            pricing=Pricing(
                input_per_million="0.02",
                output_per_million="0",
                reasoning_billing_mode="not_billed",
                source_url="https://developers.openai.com/api/docs/models/text-embedding-3-small",
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "openai",
            "gpt-4o-transcribe",
            reasoning_modes=("none",),
            key_probe_model="gpt-5.4-mini",
            max_context_tokens=0,
            max_output_tokens=0,
            generation=False,
            structured_output=False,
            transcription=True,
            pricing=Pricing(
                input_per_million="2.5",
                output_per_million="10",
                reasoning_billing_mode="not_billed",
                source_url=_OPENAI_DEVELOPER_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "anthropic",
            "claude-opus-4-8",
            reasoning_modes=_ANTHROPIC_REASONING,
            key_probe_model="claude-haiku-4-5-20251001",
            max_context_tokens=1000000,
            max_output_tokens=128000,
            prompt_cache=PromptCacheCapability("turn_ttl", ("5m", "1h")),
            structured_output_reasoning_modes=("default", "none"),
            raw_artifact_support=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
            pricing=Pricing(
                input_per_million="5",
                output_per_million="25",
                cached_input_per_million="0.5",
                cache_write_per_million_by_ttl={"5m": "6.25", "1h": "10"},
                reasoning_billing_mode="included_in_output",
                source_url=_ANTHROPIC_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "anthropic",
            "claude-sonnet-4-6",
            reasoning_modes=_ANTHROPIC_REASONING,
            key_probe_model="claude-haiku-4-5-20251001",
            max_context_tokens=1000000,
            max_output_tokens=64000,
            prompt_cache=PromptCacheCapability("turn_ttl", ("5m", "1h")),
            structured_output_reasoning_modes=("default", "none"),
            raw_artifact_support=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
            pricing=Pricing(
                input_per_million="3",
                output_per_million="15",
                cached_input_per_million="0.3",
                cache_write_per_million_by_ttl={"5m": "3.75", "1h": "6"},
                reasoning_billing_mode="included_in_output",
                source_url=_ANTHROPIC_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "anthropic",
            "claude-haiku-4-5-20251001",
            reasoning_modes=_ANTHROPIC_REASONING,
            key_probe_model="claude-haiku-4-5-20251001",
            max_context_tokens=200000,
            max_output_tokens=64000,
            prompt_cache=PromptCacheCapability("turn_ttl", ("5m", "1h")),
            structured_output_reasoning_modes=("default", "none"),
            raw_artifact_support=True,
            usage_cache_read_write_tokens=True,
            reasoning_continuation=True,
            pricing=Pricing(
                input_per_million="1",
                output_per_million="5",
                cached_input_per_million="0.1",
                cache_write_per_million_by_ttl={"5m": "1.25", "1h": "2"},
                reasoning_billing_mode="included_in_output",
                source_url=_ANTHROPIC_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "gemini",
            "gemini-2.5-pro",
            reasoning_modes=_GEMINI_25_PRO_REASONING,
            key_probe_model="gemini-3-flash-preview",
            max_context_tokens=1048576,
            max_output_tokens=65536,
            reasoning_budget_tokens=(128, 1024, 8192, 16384, 32768),
            reasoning_budget_range=(128, 32768),
            reasoning_allows_dynamic_budget=True,
            raw_artifact_support=True,
            provider_request_id=False,
            reasoning_continuation=True,
            pricing=Pricing(
                input_per_million="1.25",
                output_per_million="10",
                cached_input_per_million="0.125",
                reasoning_billing_mode="included_in_output",
                applies_up_to_input_tokens=200000,
                source_url=_GEMINI_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "gemini",
            "gemini-2.5-flash",
            reasoning_modes=_GEMINI_25_FLASH_REASONING,
            key_probe_model="gemini-3-flash-preview",
            max_context_tokens=1048576,
            max_output_tokens=65536,
            reasoning_budget_tokens=(0, 512, 1024, 8192, 16384, 24576),
            reasoning_budget_range=(0, 24576),
            reasoning_allows_dynamic_budget=True,
            multimodal_input=True,
            raw_artifact_support=True,
            provider_request_id=False,
            reasoning_continuation=True,
            pricing=Pricing(
                input_per_million="0.30",
                output_per_million="2.50",
                cached_input_per_million="0.03",
                reasoning_billing_mode="included_in_output",
                source_url=_GEMINI_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "gemini",
            "gemini-3.1-pro-preview",
            reasoning_modes=_GEMINI_31_PRO_REASONING,
            key_probe_model="gemini-3-flash-preview",
            max_context_tokens=1048576,
            max_output_tokens=65536,
            tool_choice_required=False,
            raw_artifact_support=True,
            provider_request_id=False,
            reasoning_continuation=True,
            pricing=Pricing(
                input_per_million="2",
                output_per_million="12",
                cached_input_per_million="0.2",
                reasoning_billing_mode="included_in_output",
                applies_up_to_input_tokens=200000,
                source_url=_GEMINI_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "gemini",
            "gemini-3-flash-preview",
            reasoning_modes=_GEMINI_3_FLASH_REASONING,
            key_probe_model="gemini-3-flash-preview",
            max_context_tokens=1048576,
            max_output_tokens=65536,
            raw_artifact_support=True,
            provider_request_id=False,
            reasoning_continuation=True,
            pricing=Pricing(
                input_per_million="0.50",
                output_per_million="3",
                cached_input_per_million="0.05",
                reasoning_billing_mode="included_in_output",
                source_url=_GEMINI_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "openrouter",
            "moonshotai/kimi-k2.6",
            reasoning_modes=("default", "none", "minimal", "low", "medium", "high", "max"),
            key_probe_model="openai/gpt-5.4-mini",
            max_context_tokens=262144,
            max_output_tokens=32000,
            structured_output=False,
            pricing=Pricing(
                input_per_million="0.67",
                output_per_million="3.39",
                cached_input_per_million="0.14",
                reasoning_billing_mode="included_in_output",
                source_url=_OPENROUTER_MODELS_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "openrouter",
            "deepseek/deepseek-v3.2",
            reasoning_modes=("default", "none", "minimal", "low", "medium", "high", "max"),
            key_probe_model="openai/gpt-5.4-mini",
            max_context_tokens=131072,
            max_output_tokens=32000,
            pricing=Pricing(
                input_per_million="0.2288",
                output_per_million="0.3432",
                reasoning_billing_mode="included_in_output",
                source_url=_OPENROUTER_MODELS_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "openrouter",
            "openai/gpt-5.5",
            reasoning_modes=_OPENROUTER_OPENAI_REASONING,
            key_probe_model="openai/gpt-5.4-mini",
            max_context_tokens=1050000,
            max_output_tokens=32000,
            pricing=Pricing(
                input_per_million="5",
                output_per_million="30",
                cached_input_per_million="0.5",
                reasoning_billing_mode="included_in_output",
                source_url=_OPENROUTER_MODELS_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "openrouter",
            "openai/gpt-5.4-mini",
            reasoning_modes=_OPENROUTER_OPENAI_REASONING,
            key_probe_model="openai/gpt-5.4-mini",
            max_context_tokens=400000,
            max_output_tokens=32000,
            pricing=Pricing(
                input_per_million="0.75",
                output_per_million="4.5",
                cached_input_per_million="0.075",
                reasoning_billing_mode="included_in_output",
                source_url=_OPENROUTER_MODELS_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "cloudflare",
            "@cf/openai/gpt-oss-20b",
            reasoning_modes=("default", "none"),
            key_probe_model="@cf/openai/gpt-oss-20b",
            max_context_tokens=128000,
            max_output_tokens=4096,
            streaming=False,
            tool_calling=False,
            tool_choice_required=False,
            structured_output=False,
            pricing=Pricing(
                input_per_million="0.20",
                output_per_million="0.30",
                reasoning_billing_mode="included_in_output",
                source_url=_CLOUDFLARE_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
        _cap(
            "cloudflare",
            "@cf/qwen/qwen3-embedding-0.6b",
            reasoning_modes=("none",),
            key_probe_model="@cf/openai/gpt-oss-20b",
            max_context_tokens=8192,
            max_output_tokens=0,
            generation=False,
            structured_output=False,
            embeddings=True,
            usage_input_output_tokens=False,
            pricing=Pricing(
                input_per_million="0.012",
                output_per_million="0",
                reasoning_billing_mode="not_billed",
                source_url=_CLOUDFLARE_PRICING_URL,
                verified_at=_VERIFIED_AT,
            ),
        ),
    )
)
