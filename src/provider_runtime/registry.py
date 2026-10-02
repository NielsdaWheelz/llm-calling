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

REGISTRY_REVISION: Final = "2026-09-25.2"
_BACKEND_CONTRACT_REVISION: Final = "provider-runtime.api-model-catalog.v3"
_KEY = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)*\Z")
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
GPT6_MODEL_IDS: Final = ("gpt-6-astra", "gpt-6-sol", "gpt-6-luna")


@dataclass(frozen=True, slots=True)
class _SourcedReasoningDefault:
    value: ReasoningKey
    source: SourceCitation


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
    reasoning: Presence[Mapping[ReasoningKey, object]]
    source_default_reasoning: Presence[_SourcedReasoningDefault]
    upgrade: Presence[UpgradeFacts]
    retirement: Presence[RetirementFacts]
    continuation_codec: str
    correlation: Literal["header", "in_band", "none"]
    source_max_output_tokens: Presence[int] = Absent()


def _source(url: str) -> SourceCitation:
    return SourceCitation(url=url, verified_on=date(2026, 9, 25))


_OPENAI_SOURCE = _source("https://developers.openai.com/api/docs/guides/reasoning")
_CLAUDE_SOURCE = _source("https://platform.claude.com/docs/en/build-with-claude/effort")
_GEMINI_SOURCE = _source("https://ai.google.dev/gemini-api/docs/latest-model")
_DEEPSEEK_SOURCE = _source("https://api-docs.deepseek.com/guides/thinking_mode/")
_XAI_SOURCE = _source("https://docs.x.ai/developers/models/grok-4.7")
_VISION: Final[frozenset[Literal["text", "image"]]] = frozenset({"text", "image"})
_MODEL_LABELS: Final[Mapping[str, str]] = {
    "gpt-6-astra": "GPT-6 Astra",
    "gpt-6-sol": "GPT-6 Sol",
    "gpt-6-luna": "GPT-6 Luna",
    "claude-fable-5-1": "Claude Fable 5.1",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "deepseek-flash": "DeepSeek Flash v4.1",
    "grok-4.7": "Grok 4.7",
}


def _default(key: str, source: SourceCitation) -> Presence[_SourcedReasoningDefault]:
    return Present(_SourcedReasoningDefault(key, source))


def _openai_row(model: str) -> _ModelRow:
    efforts = _EFFORTS if model == "gpt-6-astra" else ("none", *_EFFORTS)
    reasoning = {
        f"{mode}/{effort}": {"reasoning": {"mode": mode, "effort": effort}}
        for mode in ("standard", "pro")
        for effort in efforts
    }
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
    modes = ("adaptive", "disabled") if model == "claude-sonnet-5" else ("adaptive",)
    reasoning = {
        f"{mode}/{effort}": {"thinking": {"type": mode}, "output_config": {"effort": effort}}
        for mode in modes
        for effort in _EFFORTS
    }
    default = "adaptive/medium" if model == "claude-opus-5-5" else "adaptive/high"
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


_ROWS: Final[tuple[_ModelRow, ...]] = (
    *(_openai_row(model) for model in GPT6_MODEL_IDS),
    *(_claude_row(model) for model in ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5")),
    _ModelRow(
        ref="gemini:gemini-3.8-flash",
        provider="gemini",
        model_id="gemini-3.8-flash",
        engine="gemini_generate",
        base_url=Absent(),
        context_window=1_048_576,
        max_output_tokens=65_536,
        source_max_output_tokens=Present(65_536),
        modalities=_VISION,
        tools=True,
        streaming=True,
        structured="native",
        reasoning=Present(
            {
                effort: {"thinking_config": {"thinking_level": effort}}
                for effort in ("low", "medium", "high")
            }
        ),
        source_default_reasoning=_default("medium", _GEMINI_SOURCE),
        upgrade=Absent(),
        retirement=Absent(),
        continuation_codec="gemini.v2",
        correlation="none",
    ),
    _ModelRow(
        ref="deepseek:deepseek-flash",
        provider="deepseek",
        model_id="deepseek-flash",
        engine="openai_chat",
        base_url=Present("https://api.deepseek.com"),
        # The /models metadata gives exact binary limits; the request budget stays conservative.
        context_window=1_048_576,
        max_output_tokens=384_000,
        source_max_output_tokens=Present(393_216),
        modalities=_VISION,
        tools=True,
        streaming=True,
        structured="json_mode",
        reasoning=Present(
            {
                "none": {"thinking": {"type": "disabled"}},
                **{
                    effort: {"thinking": {"type": "enabled"}, "reasoning_effort": effort}
                    for effort in ("low", "high", "max")
                },
            }
        ),
        source_default_reasoning=_default("high", _DEEPSEEK_SOURCE),
        upgrade=Absent(),
        retirement=Absent(),
        continuation_codec="deepseek.v2",
        correlation="in_band",
    ),
    _ModelRow(
        ref="xai:grok-4.7",
        provider="xai",
        model_id="grok-4.7",
        engine="openai_chat",
        base_url=Present("https://api.x.ai/v1"),
        context_window=500_000,
        max_output_tokens=4096,
        source_max_output_tokens=Absent(),
        modalities=_VISION,
        tools=True,
        streaming=True,
        structured="native",
        reasoning=Present(
            {effort: {"reasoning_effort": effort} for effort in ("low", "medium", "high", "xhigh")}
        ),
        source_default_reasoning=_default("high", _XAI_SOURCE),
        upgrade=Absent(),
        retirement=Absent(),
        continuation_codec="xai.v2",
        correlation="in_band",
    ),
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


def _option_label(key: str) -> str:
    mode, sep, effort = key.partition("/")
    if sep:
        if mode == "disabled":
            return f"thinking off · {effort} effort"
        return f"{mode} · {effort}"
    return "off" if key == "none" else key


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
        b"provider-runtime.row.v3",
        {
            "ref": row.ref,
            "provider": row.provider,
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
            "reasoning": reasoning,
            "labels": tuple((key, _option_label(key)) for key in reasoning),
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
        reasoning=tuple(ApiReasoningFacts(key=key, label=_option_label(key)) for key in reasoning),
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
            b"provider-runtime.catalog.v3",
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
        for key, fragment in row.reasoning.value.items():
            if (
                not 1 <= len(key) <= 64
                or _KEY.fullmatch(key) is None
                or not isinstance(fragment, Mapping)
                or not fragment
            ):
                raise RuntimeDefect(
                    origin="intent",
                    code="registry_invalid",
                    message=f"invalid configuration for {row.ref}",
                )
            freeze_json_object(fragment, context=f"{row.ref}/{key}")
        if (
            isinstance(row.source_default_reasoning, Present)
            and row.source_default_reasoning.value.value not in row.reasoning.value
        ):
            raise RuntimeDefect(
                origin="intent", code="registry_invalid", message=f"invalid default for {row.ref}"
            )


_validate_rows()

__all__ = ["GPT6_MODEL_IDS", "api_model_catalog"]
