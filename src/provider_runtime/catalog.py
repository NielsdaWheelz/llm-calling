"""Model catalog — exact provider contracts (spec §4 portfolio, §5 contract, §8 cache).

Every rate, limit, enum, native mapping, and minimum-prefix figure below is
transcribed from ``.dossiers/provider-facts.md`` (verified 2026-07-20 against
official provider docs). The catalog is a transcription surface, never a place
to remember provider facts; drift is caught by release certification, which
re-verifies the sources.

Pricing is encoded as integer **usd micros per million tokens** ($/1M × 1e6
exactly; $5.00/M ⇒ 5_000_000). Rates stay integral for every row so no float
ever enters accounting.

Staleness: ``verified_at`` is encoded as data. A wall-clock check at
construction would make the package start failing spontaneously in a running
deployment, so freshness enforcement belongs to the CERTIFICATION command,
which calls :func:`check_catalog_freshness` with an explicit ``today``.
[Deviation from the spec §5 "stale source verification rejected at
construction" phrasing — adjudicated: construction stays deterministic and
wall-clock-free; the certification command owns time.]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal, assert_never

from provider_runtime.errors import RuntimeDefect
from provider_runtime.types import (
    Absent,
    Presence,
    Present,
    ProviderProtocol,
    ProviderTarget,
    ReasoningLevel,
)

# ---------------------------------------------------------------------------
# Contract value types


@dataclass(frozen=True, slots=True)
class ReasoningContract:
    # native_mapping keys == levels exactly (catalog-construction-validated);
    # xhigh and max are DISTINCT native strings for gpt-5.6.
    levels: tuple[ReasoningLevel, ...]
    provider_default: ReasoningLevel
    native_mapping: Mapping[ReasoningLevel, str]


# CacheContract — a tagged union declaring mechanism + parameters +
# minimum-prefix + billing facts (§8). "Enabled" means the correct provider
# mechanism is present; it does not promise a hit below the provider's minimum
# prefix length.


@dataclass(frozen=True, slots=True)
class OpenAIExplicitPrefixContract:
    # prompt_cache_options {mode: explicit, ttl: "30m"} — 30m is the only TTL
    # and a MINIMUM lifetime, not exact or maximum retention.
    ttl: Literal["30m"]
    minimum_prefix_tokens: int


@dataclass(frozen=True, slots=True)
class AnthropicPrefixContract:
    # cache_control {"type": "ephemeral", "ttl": "5m"}; a 1h tier exists but the
    # runtime uses only 5m.
    ttl: Literal["5m"]
    minimum_prefix_tokens: int


@dataclass(frozen=True, slots=True)
class GeminiAutomaticPrefixContract:
    # Implicit provider-side caching: no wire control is invented; live
    # certification proves reported cache reads (§8).
    minimum_prefix_tokens: Presence[int]


@dataclass(frozen=True, slots=True)
class MoonshotKeyedPrefixContract:
    # Moonshot automatically manages cache contents, while prompt_cache_key
    # provides the stable request affinity needed to improve hit rates.
    minimum_prefix_tokens: Presence[int]


@dataclass(frozen=True, slots=True)
class OpenRouterPrefixContract:
    # pinned_upstream: the provider.only endpoint slug; canonical_revision: the
    # catalog's pinned model revision. The planner's certification gate matches
    # a paid-certification artifact against both.
    pinned_upstream: str
    canonical_revision: str


type CacheContract = (
    OpenAIExplicitPrefixContract
    | AnthropicPrefixContract
    | GeminiAutomaticPrefixContract
    | MoonshotKeyedPrefixContract
    | OpenRouterPrefixContract
)


@dataclass(frozen=True, slots=True)
class PricingContract:
    # Integer usd micros per million tokens ($/1M × 1e6 exactly).
    currency: Literal["usd"]
    input_rate: int
    output_rate: int
    cache_read_rate: int
    # 0 is legal only for implicit-cache mechanisms (no write billing).
    cache_write_rate: int
    # False everywhere today: every current provider bills reasoning/thinking
    # tokens inside output. reasoning_reserve_tokens is the extra reservation
    # the planner adds only when reasoning IS billed outside output.
    reasoning_billed_outside_output: bool
    reasoning_reserve_tokens: int
    source_url: str
    verified_at: date


@dataclass(frozen=True, slots=True)
class PrivacyContract:
    retention: str
    zdr_eligible: bool


@dataclass(frozen=True, slots=True)
class DirectCertification:
    """A first-party provider route certified by direct release certification."""


@dataclass(frozen=True, slots=True)
class OperatorCertified:
    # The pin facts the paid certification actually exercised, transcribed
    # from the evidence artifact. Catalog construction requires these to
    # match the row's OpenRouterPrefixContract exactly (§8), so a row
    # re-pinned to a new upstream or revision can't keep planning traffic
    # against stale evidence.
    certified_pinned_upstream: str
    certified_canonical_revision: str
    # Immutable id of the paid certification artifact (endpoint-metadata
    # snapshot, probe generation ids, observed billed cache usage).
    evidence_revision: str


@dataclass(frozen=True, slots=True)
class OperatorUncertified:
    """Hidden operator row: representable (§4) but unusable until certified."""


type Certification = DirectCertification | OperatorCertified | OperatorUncertified


@dataclass(frozen=True, slots=True)
class ChatModelContract:
    target: ProviderTarget
    protocol: ProviderProtocol
    context_limit: int
    output_limit: int
    reasoning: ReasoningContract
    cache: CacheContract
    continuation_codec: str
    strict_schema_dialect: str
    # Gemini: False — §5 request correlation is guaranteed only "where the
    # provider supports it".
    provider_request_id_available: bool
    privacy: PrivacyContract
    pricing: PricingContract
    source_urls: tuple[str, ...]
    verified_at: date
    certification: Certification
    # Conservative certification-validated allowance added to the bytes-as-
    # tokens input bound for provider message framing (§6).
    provider_framing_overhead_tokens: int


@dataclass(frozen=True, slots=True)
class EmbeddingContract:
    target: ProviderTarget
    max_input_tokens: int
    # usd micros per million input tokens.
    input_rate: int
    source_urls: tuple[str, ...]
    verified_at: date


@dataclass(frozen=True, slots=True)
class TranscriptionContract:
    target: ProviderTarget
    # usd micros per million tokens (audio-token input / text-token output).
    input_rate: int
    output_rate: int
    source_urls: tuple[str, ...]
    verified_at: date


# ---------------------------------------------------------------------------
# Catalog


@dataclass(frozen=True, slots=True)
class Catalog:
    chat: tuple[ChatModelContract, ...]
    embeddings: tuple[EmbeddingContract, ...]
    transcriptions: tuple[TranscriptionContract, ...]

    def __post_init__(self) -> None:
        _validate_catalog(self)

    def chat_contract(self, target: ProviderTarget) -> ChatModelContract:
        for row in self.chat:
            if row.target == target:
                return row
        raise RuntimeDefect(
            origin="plan",
            code="unknown_target",
            message=(
                f"No chat model contract for target {_label(target)}; "
                f"known chat targets: {_known(self.chat)}"
            ),
        )

    def embedding_contract(self, target: ProviderTarget) -> EmbeddingContract:
        for row in self.embeddings:
            if row.target == target:
                return row
        raise RuntimeDefect(
            origin="plan",
            code="unknown_target",
            message=(
                f"No embedding contract for target {_label(target)}; "
                f"known embedding targets: {_known(self.embeddings)}"
            ),
        )

    def transcription_contract(self, target: ProviderTarget) -> TranscriptionContract:
        for row in self.transcriptions:
            if row.target == target:
                return row
        raise RuntimeDefect(
            origin="plan",
            code="unknown_target",
            message=(
                f"No transcription contract for target {_label(target)}; "
                f"known transcription targets: {_known(self.transcriptions)}"
            ),
        )


def _label(target: ProviderTarget) -> str:
    return f"{target.provider}/{target.model}"


def _known(
    rows: tuple[ChatModelContract, ...]
    | tuple[EmbeddingContract, ...]
    | tuple[TranscriptionContract, ...],
) -> str:
    return ", ".join(_label(row.target) for row in rows) or "<none>"


# ---------------------------------------------------------------------------
# Construction validation (spec §5: the catalog rejects malformed rows at
# construction; staleness alone is deferred to check_catalog_freshness).


def _validate_catalog(catalog: Catalog) -> None:
    seen: set[tuple[str, str]] = set()
    for target in (
        *(row.target for row in catalog.chat),
        *(row.target for row in catalog.embeddings),
        *(row.target for row in catalog.transcriptions),
    ):
        key = (target.provider, target.model)
        if key in seen:
            raise RuntimeDefect(
                origin="plan",
                code="duplicate_target",
                message=f"Catalog contains duplicate target {_label(target)}",
            )
        seen.add(key)
    for chat_row in catalog.chat:
        _validate_chat_row(chat_row)
    for embedding_row in catalog.embeddings:
        _require_source_urls(embedding_row.source_urls, _label(embedding_row.target))
        _require_positive_rate(embedding_row.input_rate, "input_rate", _label(embedding_row.target))
    for transcription_row in catalog.transcriptions:
        _require_source_urls(transcription_row.source_urls, _label(transcription_row.target))
        _require_positive_rate(
            transcription_row.input_rate, "input_rate", _label(transcription_row.target)
        )
        _require_positive_rate(
            transcription_row.output_rate, "output_rate", _label(transcription_row.target)
        )


def _validate_chat_row(row: ChatModelContract) -> None:
    label = _label(row.target)
    reasoning = row.reasoning
    if len(set(reasoning.levels)) != len(reasoning.levels):
        raise RuntimeDefect(
            origin="plan",
            code="invalid_reasoning_mapping",
            message=f"{label}: reasoning levels contain duplicates: {reasoning.levels}",
        )
    if reasoning.provider_default not in reasoning.levels:
        raise RuntimeDefect(
            origin="plan",
            code="invalid_reasoning_mapping",
            message=(
                f"{label}: provider_default {reasoning.provider_default!r} is not one of "
                f"the declared levels {reasoning.levels}"
            ),
        )
    if set(reasoning.native_mapping) != set(reasoning.levels) or len(
        reasoning.native_mapping
    ) != len(reasoning.levels):
        raise RuntimeDefect(
            origin="plan",
            code="invalid_reasoning_mapping",
            message=(
                f"{label}: native_mapping keys {sorted(reasoning.native_mapping)} must equal "
                f"the declared levels {reasoning.levels} exactly"
            ),
        )
    for rate_name, rate in (
        ("input_rate", row.pricing.input_rate),
        ("output_rate", row.pricing.output_rate),
        ("cache_read_rate", row.pricing.cache_read_rate),
    ):
        _require_positive_rate(rate, rate_name, label)
    if row.pricing.cache_write_rate < 0:
        raise RuntimeDefect(
            origin="plan",
            code="unpriced_selectable_route",
            message=(f"{label}: cache_write_rate must be >= 0; got {row.pricing.cache_write_rate}"),
        )
    if row.pricing.cache_write_rate == 0 and not _implicit_cache_billing(row.cache):
        raise RuntimeDefect(
            origin="plan",
            code="unpriced_selectable_route",
            message=(
                f"{label}: cache_write_rate may be 0 only for implicit/automatic cache "
                f"mechanisms; cache contract is {type(row.cache).__name__}"
            ),
        )
    _require_source_urls(row.source_urls, label)
    if row.provider_framing_overhead_tokens < 0:
        raise RuntimeDefect(
            origin="plan",
            code="invalid_framing_overhead",
            message=(
                f"{label}: provider_framing_overhead_tokens must be >= 0; "
                f"got {row.provider_framing_overhead_tokens}"
            ),
        )
    _validate_certification_pairing(row, label)


def _implicit_cache_billing(cache: CacheContract) -> bool:
    match cache:
        case (
            GeminiAutomaticPrefixContract()
            | MoonshotKeyedPrefixContract()
            | OpenRouterPrefixContract()
        ):
            return True
        case OpenAIExplicitPrefixContract() | AnthropicPrefixContract():
            return False
        case _:
            assert_never(cache)


def _validate_certification_pairing(row: ChatModelContract, label: str) -> None:
    match row.protocol:
        case "openrouter_chat":
            match row.certification:
                case DirectCertification():
                    raise RuntimeDefect(
                        origin="plan",
                        code="certification_mismatch",
                        message=(
                            f"{label}: openrouter_chat routes require OperatorCertified or "
                            f"OperatorUncertified, not DirectCertification"
                        ),
                    )
                case OperatorCertified(
                    certified_pinned_upstream=certified_pinned_upstream,
                    certified_canonical_revision=certified_canonical_revision,
                ):
                    if not isinstance(row.cache, OpenRouterPrefixContract):
                        raise RuntimeDefect(
                            origin="plan",
                            code="certification_mismatch",
                            message=(
                                f"{label}: OperatorCertified requires an "
                                f"OpenRouterPrefixContract cache contract; got "
                                f"{type(row.cache).__name__}"
                            ),
                        )
                    if (
                        certified_pinned_upstream != row.cache.pinned_upstream
                        or certified_canonical_revision != row.cache.canonical_revision
                    ):
                        raise RuntimeDefect(
                            origin="plan",
                            code="certification_mismatch",
                            message=(
                                f"{label}: OperatorCertified evidence pins "
                                f"pinned_upstream={certified_pinned_upstream!r} "
                                f"canonical_revision={certified_canonical_revision!r}, which "
                                f"does not match the row's cache contract "
                                f"pinned_upstream={row.cache.pinned_upstream!r} "
                                f"canonical_revision={row.cache.canonical_revision!r}"
                            ),
                        )
                case OperatorUncertified():
                    pass
                case _:
                    assert_never(row.certification)
        case (
            "openai_responses" | "anthropic_messages" | "gemini_generate_content" | "moonshot_chat"
        ):
            if not isinstance(row.certification, DirectCertification):
                raise RuntimeDefect(
                    origin="plan",
                    code="certification_mismatch",
                    message=(
                        f"{label}: direct protocol {row.protocol!r} requires "
                        f"DirectCertification; got {type(row.certification).__name__}"
                    ),
                )
        case _:
            assert_never(row.protocol)


def _require_positive_rate(rate: int, rate_name: str, label: str) -> None:
    if rate <= 0:
        raise RuntimeDefect(
            origin="plan",
            code="unpriced_selectable_route",
            message=f"{label}: {rate_name} must be > 0 usd micros/M; got {rate}",
        )


def _require_source_urls(source_urls: tuple[str, ...], label: str) -> None:
    if not source_urls:
        raise RuntimeDefect(
            origin="plan",
            code="missing_source_urls",
            message=f"{label}: source_urls must be non-empty",
        )


# ---------------------------------------------------------------------------
# Row data — transcribed from .dossiers/provider-facts.md (verified 2026-07-20).

# Bump on ANY row change (rate, limit, level, mapping, URL, certification, …);
# flows into FinalizedProviderCall.catalog_revision and the nexus ledger.
CATALOG_REVISION: Final[str] = "cat-2026-07-28-r1"

_CHAT_VERIFIED_AT: Final[date] = date(2026, 7, 20)
_MOONSHOT_VERIFIED_AT: Final[date] = date(2026, 7, 22)
_OPENROUTER_VERIFIED_AT: Final[date] = date(2026, 7, 28)
# Embedding/transcription rows carry the old catalog's facts and verification
# date forward unchanged; next certification re-verifies them.
_LEGACY_VERIFIED_AT: Final[date] = date(2026, 6, 11)

_OPENAI_PRICING_URL: Final = "https://openai.com/api/pricing/"
_OPENAI_DEVELOPER_PRICING_URL: Final = "https://developers.openai.com/api/docs/pricing"
_OPENAI_EMBEDDING_MODEL_URL: Final = (
    "https://developers.openai.com/api/docs/models/text-embedding-3-small"
)
_ANTHROPIC_PRICING_URL: Final = "https://platform.claude.com/docs/en/about-claude/pricing"
_ANTHROPIC_MODELS_URL: Final = "https://platform.claude.com/docs/en/about-claude/models/overview"
_GEMINI_PRICING_URL: Final = "https://ai.google.dev/gemini-api/docs/pricing"
_MOONSHOT_API_URL: Final = "https://platform.kimi.ai/docs/api/chat"
_MOONSHOT_PRICING_URL: Final = "https://platform.kimi.ai/docs/pricing/chat-k3"
_OPENROUTER_MODELS_URL: Final = "https://openrouter.ai/api/v1/models"

# Conservative certification-validated framing-overhead allowances (§6).
_OPENAI_RESPONSES_FRAMING_OVERHEAD: Final = 64
_DEFAULT_FRAMING_OVERHEAD: Final = 32

_GPT56_LEVELS: Final[tuple[ReasoningLevel, ...]] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
# Wire = reasoning effort strings verbatim; xhigh and max are DISTINCT levels.
_GPT56_NATIVE_MAPPING: Final[Mapping[ReasoningLevel, str]] = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

_ANTHROPIC_LEVELS: Final[tuple[ReasoningLevel, ...]] = ("low", "medium", "high", "xhigh", "max")
# Wire = top-level output_config.effort; complete set, no "none".
_ANTHROPIC_NATIVE_MAPPING: Final[Mapping[ReasoningLevel, str]] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

_GEMINI_LEVELS: Final[tuple[ReasoningLevel, ...]] = ("minimal", "low", "medium", "high")
# Wire = thinkingConfig.thinkingLevel.
_GEMINI_NATIVE_MAPPING: Final[Mapping[ReasoningLevel, str]] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}

_KIMI_LEVELS: Final[tuple[ReasoningLevel, ...]] = ("low", "high", "max")
# Wire = reasoning_effort (direct) / reasoning.effort (openrouter). low/high
# shipped ~2026-07-18 — every release certification re-probes acceptance.
_KIMI_NATIVE_MAPPING: Final[Mapping[ReasoningLevel, str]] = {
    "low": "low",
    "high": "high",
    "max": "max",
}


def _gpt56_contract(
    model: str,
    *,
    input_rate: int,
    cache_read_rate: int,
    cache_write_rate: int,
    output_rate: int,
) -> ChatModelContract:
    return ChatModelContract(
        target=ProviderTarget(provider="openai", model=model),
        protocol="openai_responses",
        context_limit=1_050_000,
        output_limit=128_000,
        reasoning=ReasoningContract(
            levels=_GPT56_LEVELS,
            # The provider's OWN default effort per OpenAI docs. Nexus profile
            # defaults (§4: fast=low, balanced=medium, deep=high) are an
            # llm_profiles concern, never a catalog fact.
            provider_default="medium",
            native_mapping=_GPT56_NATIVE_MAPPING,
        ),
        cache=OpenAIExplicitPrefixContract(ttl="30m", minimum_prefix_tokens=1024),
        continuation_codec="openai_responses",
        strict_schema_dialect="openai_text_format_json_schema",
        provider_request_id_available=True,
        # 30-day default abuse retention; ZDR-eligible.
        privacy=PrivacyContract(retention="30d", zdr_eligible=True),
        pricing=PricingContract(
            currency="usd",
            input_rate=input_rate,
            output_rate=output_rate,
            cache_read_rate=cache_read_rate,
            cache_write_rate=cache_write_rate,
            # OpenAI reasoning tokens are billed as output tokens.
            reasoning_billed_outside_output=False,
            reasoning_reserve_tokens=0,
            source_url=_OPENAI_PRICING_URL,
            verified_at=_CHAT_VERIFIED_AT,
        ),
        source_urls=(_OPENAI_PRICING_URL, _OPENAI_DEVELOPER_PRICING_URL),
        verified_at=_CHAT_VERIFIED_AT,
        certification=DirectCertification(),
        provider_framing_overhead_tokens=_OPENAI_RESPONSES_FRAMING_OVERHEAD,
    )


_GPT56_SOL: Final = _gpt56_contract(
    "gpt-5.6-sol",
    # $5.00 / $0.50 / $6.25 / $30.00 per 1M.
    input_rate=5_000_000,
    cache_read_rate=500_000,
    cache_write_rate=6_250_000,
    output_rate=30_000_000,
)

# Terra pricing page: >272K input tokens → 2× input / 1.5× output surcharge
# (sol/luna applicability UNCONFIRMED). Flat-encoded here per the provider-facts
# note; certification probes the surcharge scope.
_GPT56_TERRA: Final = _gpt56_contract(
    "gpt-5.6-terra",
    # $2.50 / $0.25 / $3.125 / $15.00 per 1M.
    input_rate=2_500_000,
    cache_read_rate=250_000,
    cache_write_rate=3_125_000,
    output_rate=15_000_000,
)

_GPT56_LUNA: Final = _gpt56_contract(
    "gpt-5.6-luna",
    # $1.00 / $0.10 / $1.25 / $6.00 per 1M.
    input_rate=1_000_000,
    cache_read_rate=100_000,
    cache_write_rate=1_250_000,
    output_rate=6_000_000,
)

_CLAUDE_SONNET_5: Final = ChatModelContract(
    target=ProviderTarget(provider="anthropic", model="claude-sonnet-5"),
    protocol="anthropic_messages",
    context_limit=1_000_000,
    output_limit=128_000,
    reasoning=ReasoningContract(
        levels=_ANTHROPIC_LEVELS,
        provider_default="high",
        native_mapping=_ANTHROPIC_NATIVE_MAPPING,
    ),
    cache=AnthropicPrefixContract(ttl="5m", minimum_prefix_tokens=1024),
    continuation_codec="anthropic_messages",
    strict_schema_dialect="anthropic_output_config_json_schema",
    provider_request_id_available=True,
    privacy=PrivacyContract(retention="standard", zdr_eligible=True),
    # INTRO rates through 2026-08-31: $2.00 / $0.20 / $2.50 (cache-write-5m) /
    # $10.00 per 1M. A 1h cache-write tier ($4.00/M intro) exists but the
    # runtime uses only 5m. From 2026-09-01 the rates switch to
    # $3.00 / $0.30 / $3.75 / $15.00 — recertification updates this row (and
    # bumps CATALOG_REVISION).
    pricing=PricingContract(
        currency="usd",
        input_rate=2_000_000,
        output_rate=10_000_000,
        cache_read_rate=200_000,
        cache_write_rate=2_500_000,
        # output_config.effort governs all output tokens (thinking included).
        reasoning_billed_outside_output=False,
        reasoning_reserve_tokens=0,
        source_url=_ANTHROPIC_PRICING_URL,
        verified_at=_CHAT_VERIFIED_AT,
    ),
    source_urls=(_ANTHROPIC_PRICING_URL, _ANTHROPIC_MODELS_URL),
    verified_at=_CHAT_VERIFIED_AT,
    certification=DirectCertification(),
    provider_framing_overhead_tokens=_DEFAULT_FRAMING_OVERHEAD,
)

# Fable: adaptive thinking is ALWAYS ON (thinking:{type:"disabled"} is
# rejected); the behavioral handling lives in the anthropic codec, not here.
_CLAUDE_FABLE_5: Final = ChatModelContract(
    target=ProviderTarget(provider="anthropic", model="claude-fable-5"),
    protocol="anthropic_messages",
    context_limit=1_000_000,
    output_limit=128_000,
    reasoning=ReasoningContract(
        levels=_ANTHROPIC_LEVELS,
        provider_default="high",
        native_mapping=_ANTHROPIC_NATIVE_MAPPING,
    ),
    cache=AnthropicPrefixContract(ttl="5m", minimum_prefix_tokens=512),
    continuation_codec="anthropic_messages",
    strict_schema_dialect="anthropic_output_config_json_schema",
    provider_request_id_available=True,
    # Covered Model: 30-day retention REQUIRED, ZDR unavailable. Deployment
    # records informed acceptance via NEXUS_FABLE_RETENTION_ACCEPTED_AT (§4).
    privacy=PrivacyContract(retention="30d_required", zdr_eligible=False),
    # $10.00 / $1.00 / $12.50 (cache-write-5m; 1h tier $20.00/M unused) /
    # $50.00 per 1M.
    pricing=PricingContract(
        currency="usd",
        input_rate=10_000_000,
        output_rate=50_000_000,
        cache_read_rate=1_000_000,
        cache_write_rate=12_500_000,
        reasoning_billed_outside_output=False,
        reasoning_reserve_tokens=0,
        source_url=_ANTHROPIC_PRICING_URL,
        verified_at=_CHAT_VERIFIED_AT,
    ),
    source_urls=(_ANTHROPIC_PRICING_URL, _ANTHROPIC_MODELS_URL),
    verified_at=_CHAT_VERIFIED_AT,
    certification=DirectCertification(),
    provider_framing_overhead_tokens=_DEFAULT_FRAMING_OVERHEAD,
)

_GEMINI_35_FLASH: Final = ChatModelContract(
    target=ProviderTarget(provider="gemini", model="gemini-3.5-flash"),
    protocol="gemini_generate_content",
    context_limit=1_048_576,
    output_limit=65_536,
    reasoning=ReasoningContract(
        levels=_GEMINI_LEVELS,
        provider_default="medium",
        native_mapping=_GEMINI_NATIVE_MAPPING,
    ),
    cache=GeminiAutomaticPrefixContract(minimum_prefix_tokens=Present(4096)),
    continuation_codec="gemini_generate_content",
    strict_schema_dialect="gemini_response_json_schema",
    # No provider request id — §5 correlation is guaranteed only where the
    # provider supports it.
    provider_request_id_available=False,
    privacy=PrivacyContract(retention="standard", zdr_eligible=True),
    # $1.50 / $0.15 (implicit/context cache read) / — / $9.00 per 1M.
    # cache_write_rate=0: implicit caching has no write billing (explicit-cache
    # storage pricing exists but the runtime never creates explicit caches).
    pricing=PricingContract(
        currency="usd",
        input_rate=1_500_000,
        output_rate=9_000_000,
        cache_read_rate=150_000,
        cache_write_rate=0,
        # Output price includes thinking tokens.
        reasoning_billed_outside_output=False,
        reasoning_reserve_tokens=0,
        source_url=_GEMINI_PRICING_URL,
        verified_at=_CHAT_VERIFIED_AT,
    ),
    source_urls=(_GEMINI_PRICING_URL,),
    verified_at=_CHAT_VERIFIED_AT,
    certification=DirectCertification(),
    provider_framing_overhead_tokens=_DEFAULT_FRAMING_OVERHEAD,
)

_KIMI_K3: Final = ChatModelContract(
    target=ProviderTarget(provider="moonshot", model="kimi-k3"),
    protocol="moonshot_chat",
    context_limit=1_048_576,
    # Default max_completion_tokens; the provider's hard max equals the context
    # size (1_048_576) — the default is encoded as the output limit.
    output_limit=131_072,
    reasoning=ReasoningContract(
        levels=_KIMI_LEVELS,
        provider_default="max",
        native_mapping=_KIMI_NATIVE_MAPPING,
    ),
    # Automatic cache management with stable request affinity supplied through
    # prompt_cache_key; minimum prefix and TTL are UNCONFIRMED by provider docs.
    cache=MoonshotKeyedPrefixContract(minimum_prefix_tokens=Absent()),
    continuation_codec="moonshot_chat",
    strict_schema_dialect="chat_completions_response_format_json_schema",
    provider_request_id_available=True,
    # Retention/ZDR posture UNCONFIRMED in provider docs → zdr_eligible False
    # until certification establishes otherwise.
    privacy=PrivacyContract(retention="standard", zdr_eligible=False),
    # $3.00 (cache-miss input) / $0.30 (cache-hit read) / — / $15.00 per 1M.
    # cache_write_rate=0: automatic caching has no write billing.
    pricing=PricingContract(
        currency="usd",
        input_rate=3_000_000,
        output_rate=15_000_000,
        cache_read_rate=300_000,
        cache_write_rate=0,
        reasoning_billed_outside_output=False,
        reasoning_reserve_tokens=0,
        source_url=_MOONSHOT_PRICING_URL,
        verified_at=_MOONSHOT_VERIFIED_AT,
    ),
    source_urls=(_MOONSHOT_API_URL, _MOONSHOT_PRICING_URL),
    verified_at=_MOONSHOT_VERIFIED_AT,
    certification=DirectCertification(),
    provider_framing_overhead_tokens=_DEFAULT_FRAMING_OVERHEAD,
)

# Hidden operator candidate (§4): absent from /llm-profiles, unselectable by
# chat. OperatorUncertified until the paid release certification proves
# low|high|max acceptance, the pinned upstream, and a NON-ZERO BILLED cache
# read (the endpoint metadata claims supports_implicit_caching=false, which
# contradicts Moonshot's automatic caching — only the paid probe settles it);
# certification then produces the OperatorCertified evidence artifact.
_OPENROUTER_KIMI_K3: Final = ChatModelContract(
    target=ProviderTarget(provider="openrouter", model="moonshotai/kimi-k3-20260715"),
    protocol="openrouter_chat",
    context_limit=1_048_576,
    output_limit=131_072,
    reasoning=ReasoningContract(
        levels=_KIMI_LEVELS,
        provider_default="max",
        native_mapping=_KIMI_NATIVE_MAPPING,
    ),
    cache=OpenRouterPrefixContract(
        pinned_upstream="moonshotai/mxfp4",
        canonical_revision="moonshotai/kimi-k3-20260715",
    ),
    continuation_codec="openrouter_chat",
    strict_schema_dialect="chat_completions_response_format_json_schema",
    provider_request_id_available=True,
    # Upstream is Moonshot; same UNCONFIRMED retention posture as the direct row.
    privacy=PrivacyContract(retention="standard", zdr_eligible=False),
    # Pinned endpoint metadata: prompt $3.00 / cache read $0.30 / cache write
    # not listed (0) / completion $15.00 per 1M.
    pricing=PricingContract(
        currency="usd",
        input_rate=3_000_000,
        output_rate=15_000_000,
        cache_read_rate=300_000,
        cache_write_rate=0,
        reasoning_billed_outside_output=False,
        reasoning_reserve_tokens=0,
        source_url=_OPENROUTER_MODELS_URL,
        verified_at=_OPENROUTER_VERIFIED_AT,
    ),
    source_urls=(_OPENROUTER_MODELS_URL,),
    verified_at=_OPENROUTER_VERIFIED_AT,
    certification=OperatorUncertified(),
    provider_framing_overhead_tokens=_DEFAULT_FRAMING_OVERHEAD,
)

_TEXT_EMBEDDING_3_SMALL: Final = EmbeddingContract(
    target=ProviderTarget(provider="openai", model="text-embedding-3-small"),
    max_input_tokens=8191,
    # $0.02 per 1M input tokens.
    input_rate=20_000,
    source_urls=(_OPENAI_EMBEDDING_MODEL_URL,),
    verified_at=_LEGACY_VERIFIED_AT,
)

_GPT_4O_TRANSCRIBE: Final = TranscriptionContract(
    target=ProviderTarget(provider="openai", model="gpt-4o-transcribe"),
    # $2.50 per 1M audio-input tokens / $10.00 per 1M output tokens.
    input_rate=2_500_000,
    output_rate=10_000_000,
    source_urls=(_OPENAI_DEVELOPER_PRICING_URL,),
    verified_at=_LEGACY_VERIFIED_AT,
)

CATALOG: Final[Catalog] = Catalog(
    chat=(
        _GPT56_SOL,
        _GPT56_TERRA,
        _GPT56_LUNA,
        _CLAUDE_SONNET_5,
        _CLAUDE_FABLE_5,
        _GEMINI_35_FLASH,
        _KIMI_K3,
        _OPENROUTER_KIMI_K3,
    ),
    embeddings=(_TEXT_EMBEDDING_3_SMALL,),
    transcriptions=(_GPT_4O_TRANSCRIBE,),
)


# ---------------------------------------------------------------------------
# Freshness — certification-command concern, never import-time wall-clock.

_MAX_VERIFICATION_AGE_DAYS: Final = 180


def check_catalog_freshness(today: date, catalog: Catalog = CATALOG) -> None:
    """Raise if any row's verification is more than 180 days older than ``today``.

    Called by the certification command with an explicit date; the package
    itself never reads the wall clock, so an aging deployment keeps planning
    while certification (which runs on every release) is what fails loudly.
    """
    for label, verified_at in (
        *((_label(row.target), row.verified_at) for row in catalog.chat),
        *((_label(row.target), row.verified_at) for row in catalog.embeddings),
        *((_label(row.target), row.verified_at) for row in catalog.transcriptions),
    ):
        age_days = (today - verified_at).days
        if age_days > _MAX_VERIFICATION_AGE_DAYS:
            raise RuntimeDefect(
                origin="plan",
                code="stale_catalog_verification",
                message=(
                    f"{label}: catalog facts verified {verified_at.isoformat()} are "
                    f"{age_days} days old as of {today.isoformat()} "
                    f"(limit {_MAX_VERIFICATION_AGE_DAYS} days); re-verify provider "
                    f"sources and bump CATALOG_REVISION"
                ),
            )
