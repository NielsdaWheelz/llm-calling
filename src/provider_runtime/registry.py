"""The current executable API model catalog and private native configurations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

from provider_runtime.errors import InvalidRequest, RuntimeDefect
from provider_runtime.types import (
    Absent,
    ApiDispatchFacts,
    ApiModelCatalog,
    ApiModelFacts,
    ApiReasoningFacts,
    EngineId,
    JsonModeStructuredOutput,
    NativeStructuredOutput,
    Presence,
    Present,
    ProviderName,
    ProviderTarget,
    ReasoningKey,
    RetirementFacts,
    SourceCitation,
    UpgradeFacts,
    canonical_json_bytes,
    freeze_json_object,
)

REGISTRY_REVISION: Final = "2026-10-28.1"
_BACKEND_CONTRACT_REVISION: Final = "provider-runtime.api-model-catalog.v5"
_KEY = re.compile(r"[a-z0-9]+(?:[_/][a-z0-9]+)*\Z")
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
GPT6_MODEL_IDS: Final = ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna")


@dataclass(frozen=True, slots=True)
class _SourcedReasoningDefault:
    value: ReasoningKey
    source: SourceCitation


@dataclass(frozen=True, slots=True)
class _ReasoningRow:
    facts: ApiReasoningFacts
    fragment: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ModelRow:
    ref: str
    provider: ProviderName
    model_id: str
    engine: EngineId
    base_url: Presence[str]
    context_window: int
    max_output_tokens: int  # chat() request budget; not a source capacity fact
    modalities: frozenset[Literal["text", "image"]]
    tools: bool
    streaming: bool
    structured: Literal["native", "json_mode"]
    reasoning: Presence[Mapping[ReasoningKey, _ReasoningRow]]
    source_default_reasoning: Presence[_SourcedReasoningDefault]
    upgrade: Presence[UpgradeFacts]
    retirement: Presence[RetirementFacts]
    continuation_codec: str
    correlation: Literal["header", "in_band", "none"]
    source_max_output_tokens: Presence[int] = Absent()


def _source(url: str) -> SourceCitation:
    return SourceCitation(url=url, verified_on=date(2026, 10, 28))


_OPENAI_SOURCE = _source("https://developers.openai.com/api/docs/guides/reasoning")
_CLAUDE_SOURCE = _source("https://platform.claude.com/docs/en/build-with-claude/effort")
_GEMINI_SOURCE = _source("https://ai.google.dev/gemini-api/docs/thinking")
_DEEPSEEK_SOURCE = _source("https://api-docs.deepseek.com/guides/thinking_mode/")
_XAI_SOURCE = _source("https://docs.x.ai/developers/models/grok-4.7")
_VISION: Final[frozenset[Literal["text", "image"]]] = frozenset({"text", "image"})
_MODEL_LABELS: Final[Mapping[str, str]] = {
    "gpt-6-astra": "GPT-6 Astra",
    "gpt-6.1-sol": "GPT-6.1 Sol",
    "gpt-6.1-terra": "GPT-6.1 Terra",
    "gpt-6-luna": "GPT-6 Luna",
    "claude-fable-5-1": "Claude Fable 5.1",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "gemini-4-argon": "Gemini 4 Argon",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro · preview",
    "deepseek-flash": "DeepSeek V4.1 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "grok-4.7": "Grok 4.7",
    "grok-build-0.1": "Grok Build 0.1 · beta",
}


def _default(key: str, source: SourceCitation) -> Presence[_SourcedReasoningDefault]:
    return Present(_SourcedReasoningDefault(key, source))


def _reasoning(
    mode: str | None, effort: str, order: int, fragment: Mapping[str, object]
) -> tuple[str, _ReasoningRow]:
    key = f"{mode}/{effort}" if mode is not None else effort
    if mode == "disabled":
        label = f"thinking off · {effort} effort"
    elif mode is not None:
        label = f"{mode} · {effort}"
    elif effort == "default":
        label = "provider default"
    else:
        label = "off" if effort == "none" else effort
    return key, _ReasoningRow(ApiReasoningFacts(key, label, mode, effort, order), fragment)


def _openai_row(model: str) -> _ModelRow:
    efforts = _EFFORTS if model in ("gpt-6-astra", "gpt-6.1-sol") else ("none", *_EFFORTS)
    reasoning = dict(
        _reasoning(mode, effort, order, {"reasoning": {"mode": mode, "effort": effort}})
        for mode in ("standard", "pro")
        for order, effort in enumerate(efforts)
    )
    return _ModelRow(
        ref=f"openai:{model}",
        provider="openai",
        model_id=model,
        engine="openai_responses",
        base_url=Absent(),
        context_window=1_050_000,
        max_output_tokens=128_000,
        source_max_output_tokens=Present(128_000),
        modalities=_VISION,
        tools=True,
        streaming=True,
        structured="native",
        reasoning=Present(reasoning),
        source_default_reasoning=(
            Absent() if model == "gpt-6-astra" else _default("standard/medium", _OPENAI_SOURCE)
        ),
        upgrade=Absent(),
        retirement=Absent(),
        continuation_codec="openai.v2",
        correlation="header",
    )


def _claude_row(model: str) -> _ModelRow:
    modes: list[tuple[str, tuple[str, ...]]] = [("adaptive", _EFFORTS)]
    if model == "claude-sonnet-5-5":
        modes.append(("between_tools", _EFFORTS[:3]))
    elif model == "claude-haiku-5-5":
        modes.append(("disabled", _EFFORTS[:3]))
    reasoning = dict(
        _reasoning(
            mode, effort, order, {"thinking": {"type": mode}, "output_config": {"effort": effort}}
        )
        for mode, efforts in modes
        for order, effort in enumerate(efforts)
    )
    default = (
        "adaptive/medium" if model in ("claude-opus-5-5", "claude-haiku-5-5") else "adaptive/high"
    )
    return _ModelRow(
        ref=f"anthropic:{model}",
        provider="anthropic",
        model_id=model,
        engine="anthropic_messages",
        base_url=Absent(),
        context_window=1_000_000,
        max_output_tokens=128_000,
        source_max_output_tokens=Present(128_000),
        modalities=_VISION,
        tools=True,
        streaming=True,
        structured="native",
        reasoning=Present(reasoning),
        source_default_reasoning=_default(default, _CLAUDE_SOURCE),
        upgrade=Absent(),
        retirement=Absent(),
        continuation_codec="anthropic.v2",
        correlation="header",
    )


def _gemini_row(model: str) -> _ModelRow:
    efforts = (
        ("minimal", "low", "medium", "high")
        if model == "gemini-3.5-flash-lite"
        else ("low", "medium", "high")
    )
    default = {
        "gemini-4-argon": "medium",
        "gemini-3.8-flash": "medium",
        "gemini-3.5-flash-lite": "minimal",
        "gemini-3.1-pro-preview": "high",
    }[model]
    return _ModelRow(
        ref=f"gemini:{model}",
        provider="gemini",
        model_id=model,
        engine="gemini_generate",
        base_url=Absent(),
        context_window=1_048_576,
        # Argon shares its window across input, thinking and output. Keep one
        # bounded request below both its streaming and non-streaming hop caps.
        max_output_tokens=32_768 if model == "gemini-4-argon" else 65_536,
        source_max_output_tokens=Present(1_048_576 if model == "gemini-4-argon" else 65_536),
        modalities=_VISION,
        tools=True,
        streaming=True,
        structured="native",
        reasoning=Present(
            dict(
                _reasoning(None, effort, order, {"thinking_config": {"thinking_level": effort}})
                for order, effort in enumerate(efforts)
            )
        ),
        source_default_reasoning=_default(default, _GEMINI_SOURCE),
        upgrade=Absent(),
        retirement=Absent(),
        continuation_codec="gemini.v2",
        correlation="none",
    )


def _deepseek_row(model: str) -> _ModelRow:
    return _ModelRow(
        ref=f"deepseek:{model}",
        provider="deepseek",
        model_id=model,
        engine="openai_chat",
        base_url=Present("https://api.deepseek.com"),
        # The /models metadata gives exact binary limits; the request budget stays conservative.
        context_window=1_048_576,
        max_output_tokens=384_000,
        source_max_output_tokens=Present(393_216),
        modalities=_VISION if model == "deepseek-flash" else frozenset({"text"}),
        tools=True,
        streaming=True,
        structured="json_mode",
        reasoning=Present(
            dict(
                _reasoning(
                    None,
                    effort,
                    order,
                    {"thinking": {"type": "disabled"}}
                    if effort == "none"
                    else {"thinking": {"type": "enabled"}, "reasoning_effort": effort},
                )
                for order, effort in enumerate(("none", "low", "high", "max"))
            )
        ),
        source_default_reasoning=_default("high", _DEEPSEEK_SOURCE),
        upgrade=Absent(),
        retirement=Absent(),
        continuation_codec="deepseek.v2",
        correlation="in_band",
    )


def _xai_row(model: str) -> _ModelRow:
    build = model == "grok-build-0.1"
    return _ModelRow(
        ref=f"xai:{model}",
        provider="xai",
        model_id=model,
        engine="openai_chat",
        base_url=Present("https://api.x.ai/v1"),
        context_window=256_000 if build else 500_000,
        max_output_tokens=32_000,
        source_max_output_tokens=Absent(),
        modalities=_VISION,
        tools=True,
        streaming=True,
        structured="native",
        reasoning=Present(
            dict([_reasoning(None, "default", 0, {})])
            if build
            else dict(
                _reasoning(None, effort, order, {"reasoning_effort": effort})
                for order, effort in enumerate(("low", "medium", "high", "xhigh"))
            )
        ),
        source_default_reasoning=_default(
            "default" if build else "high",
            _source("https://docs.x.ai/developers/models/grok-build-0.1") if build else _XAI_SOURCE,
        ),
        upgrade=Absent(),
        retirement=Absent(),
        continuation_codec="xai.v2",
        correlation="in_band",
    )


_ROWS: Final[tuple[_ModelRow, ...]] = (
    *(_openai_row(model) for model in GPT6_MODEL_IDS),
    *(
        _claude_row(model)
        for model in (
            "claude-fable-5-1",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
        )
    ),
    *(
        _gemini_row(model)
        for model in (
            "gemini-4-argon",
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
        )
    ),
    *(_deepseek_row(model) for model in ("deepseek-flash", "deepseek-v4-pro")),
    *(_xai_row(model) for model in ("grok-4.7", "grok-build-0.1")),
)


def _resolve(ref: str) -> _ModelRow:
    for row in _ROWS:
        if row.ref == ref:
            return row
    raise InvalidRequest(message=f"unknown model ref {ref!r}")


def _resolve_target(target: ProviderTarget) -> _ModelRow:
    for row in _ROWS:
        if row.provider == target.provider and row.model_id == target.model:
            return row
    raise InvalidRequest(
        message=f"no registry row for provider {target.provider!r} model {target.model!r}"
    )


def _fingerprint(domain: bytes, value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        domain
        + b"\x00"
        + canonical_json_bytes(freeze_json_object(value, context="catalog fingerprint"))
    ).hexdigest()


def _row_fingerprint(row: _ModelRow) -> str:
    reasoning = row.reasoning.value if isinstance(row.reasoning, Present) else {}
    default = (
        row.source_default_reasoning.value.value
        if isinstance(row.source_default_reasoning, Present)
        else None
    )
    return _fingerprint(
        b"provider-runtime.row.v5",
        {
            "ref": row.ref,
            "provider": row.provider,
            "offered": True,
            "model_id": row.model_id,
            "label": _MODEL_LABELS[row.model_id],
            "engine": row.engine,
            "base_url": row.base_url.value if isinstance(row.base_url, Present) else None,
            "context_window": row.context_window,
            "max_output_tokens": row.max_output_tokens,
            "source_max_output_tokens": (
                row.source_max_output_tokens.value
                if isinstance(row.source_max_output_tokens, Present)
                else None
            ),
            "modalities": tuple(sorted(row.modalities)),
            "tools": row.tools,
            "streaming": row.streaming,
            "structured": row.structured,
            "reasoning": {key: choice.fragment for key, choice in reasoning.items()},
            "presentation": tuple(
                (
                    key,
                    choice.facts.label,
                    choice.facts.mode,
                    choice.facts.effort,
                    choice.facts.effort_order,
                )
                for key, choice in reasoning.items()
            ),
            "default": default,
            "continuation_codec": row.continuation_codec,
            "correlation": row.correlation,
        },
    )


def _api_model_facts(row: _ModelRow) -> ApiModelFacts:
    reasoning = row.reasoning.value if isinstance(row.reasoning, Present) else {}
    return ApiModelFacts(
        model_ref=row.ref,
        label=_MODEL_LABELS[row.model_id],
        provider=row.provider,
        offered=True,
        dispatch=ApiDispatchFacts(
            model_id=row.model_id,
            engine=row.engine,
            base_url=row.base_url,
            correlation=row.correlation,
        ),
        upgrade=row.upgrade,
        retirement=row.retirement,
        context_window=row.context_window,
        max_output_tokens=row.source_max_output_tokens,
        input_modalities=tuple(m for m in ("text", "image") if m in row.modalities),
        tools=row.tools,
        streaming=row.streaming,
        structured=NativeStructuredOutput()
        if row.structured == "native"
        else JsonModeStructuredOutput(),
        reasoning=tuple(choice.facts for choice in reasoning.values()),
        source_default_reasoning=(
            Present(row.source_default_reasoning.value.value)
            if isinstance(row.source_default_reasoning, Present)
            else Absent()
        ),
        continuation_codec=row.continuation_codec,
        row_fingerprint=_row_fingerprint(row),
    )


def api_model_catalog() -> ApiModelCatalog:
    models = tuple(_api_model_facts(row) for row in _ROWS)
    return ApiModelCatalog(
        backend_contract_revision=_BACKEND_CONTRACT_REVISION,
        registry_revision=REGISTRY_REVISION,
        definition_revision=_fingerprint(
            b"provider-runtime.catalog.v5",
            {"rows": tuple(model.row_fingerprint for model in models)},
        ),
        models=models,
    )


def _validate_rows() -> None:
    seen: set[tuple[ProviderName, str]] = set()
    engines: Mapping[ProviderName, EngineId] = {
        "openai": "openai_responses",
        "anthropic": "anthropic_messages",
        "gemini": "gemini_generate",
        "deepseek": "openai_chat",
        "xai": "openai_chat",
    }
    for row in _ROWS:
        identity = (row.provider, row.model_id)
        if identity in seen or row.ref != f"{row.provider}:{row.model_id}":
            raise RuntimeDefect(
                origin="intent",
                code="registry_invalid",
                message="duplicate or malformed model identity",
            )
        seen.add(identity)
        if row.engine != engines[row.provider] or not row.continuation_codec.startswith(
            f"{row.provider}."
        ):
            raise RuntimeDefect(
                origin="intent", code="registry_invalid", message=f"invalid dispatch for {row.ref}"
            )
        if (
            row.context_window < row.max_output_tokens
            or row.max_output_tokens <= 0
            or (
                isinstance(row.source_max_output_tokens, Present)
                and (
                    row.source_max_output_tokens.value < row.max_output_tokens
                    or row.source_max_output_tokens.value > row.context_window
                )
            )
            or "text" not in row.modalities
        ):
            raise RuntimeDefect(
                origin="intent", code="registry_invalid", message=f"invalid limits for {row.ref}"
            )
        if not isinstance(row.reasoning, Present) or not row.reasoning.value:
            raise RuntimeDefect(
                origin="intent",
                code="registry_invalid",
                message=f"missing configurations for {row.ref}",
            )
        for key, choice in row.reasoning.value.items():
            if (
                not 1 <= len(key) <= 64
                or _KEY.fullmatch(key) is None
                or choice.facts.key != key
                or (
                    not choice.fragment
                    and not (
                        row.provider == "xai"
                        and row.model_id == "grok-build-0.1"
                        and tuple(row.reasoning.value) == ("default",)
                        and choice.facts
                        == ApiReasoningFacts("default", "provider default", None, "default", 0)
                        and isinstance(row.source_default_reasoning, Present)
                        and row.source_default_reasoning.value.value == "default"
                    )
                )
            ):
                raise RuntimeDefect(
                    origin="intent",
                    code="registry_invalid",
                    message=f"invalid configuration for {row.ref}",
                )
            freeze_json_object(choice.fragment, context=f"{row.ref}/{key}")
        if (
            isinstance(row.source_default_reasoning, Present)
            and row.source_default_reasoning.value.value not in row.reasoning.value
        ):
            raise RuntimeDefect(
                origin="intent", code="registry_invalid", message=f"invalid default for {row.ref}"
            )


_validate_rows()

__all__ = ["GPT6_MODEL_IDS", "api_model_catalog"]
