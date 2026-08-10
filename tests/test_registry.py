"""Tests for the model registry mechanism: resolution and row invariants."""

import re
from collections.abc import Mapping

import pytest

from provider_runtime.errors import InvalidRequest, RuntimeDefect
from provider_runtime.registry import (
    REGISTRY_REVISION,
    EngineId,
    ModelRow,
    OpenRouterRouting,
    _validate_rows,
    resolve,
    resolve_target,
)
from provider_runtime.types import (
    Absent,
    Presence,
    Present,
    ProviderName,
    ProviderTarget,
    ReasoningLevel,
)

_ABSENT = Absent()


def _row(
    *,
    ref: str = "anthropic:claude-opus-5",
    provider: ProviderName = "anthropic",
    model_id: str = "claude-opus-5-20260101",
    engine: EngineId = "anthropic_messages",
    reasoning: Presence[Mapping[ReasoningLevel, object]] = _ABSENT,
    routing: Presence[OpenRouterRouting] = _ABSENT,
    context_window: int = 200_000,
) -> ModelRow:
    return ModelRow(
        ref=ref,
        provider=provider,
        model_id=model_id,
        engine=engine,
        base_url=Absent(),
        context_window=context_window,
        max_output_tokens=64_000,
        modalities=frozenset({"text"}),
        tools=True,
        streaming=True,
        structured="native",
        reasoning=reasoning,
        continuation_codec="anthropic_messages",
        correlation="header",
        routing=routing,
    )


def _pinned_routing() -> OpenRouterRouting:
    return OpenRouterRouting(
        only=("moonshotai",),
        order=("moonshotai",),
        quantizations=("fp8",),
    )


# ---------------------------------------------------------------------------
# Resolution


def test_resolve_unknown_ref_raises_invalid_request() -> None:
    with pytest.raises(InvalidRequest) as exc_info:
        resolve("nonexistent:model")
    assert exc_info.value.origin == "intent"
    assert exc_info.value.code == "invalid_request"
    assert "nonexistent:model" in exc_info.value.message, (
        "the defect message must name the unknown ref for operator diagnosis"
    )


def test_resolve_target_unknown_target_raises_invalid_request() -> None:
    with pytest.raises(InvalidRequest) as exc_info:
        resolve_target(ProviderTarget(provider="openai", model="gpt-nonexistent"))
    assert exc_info.value.origin == "intent"
    assert exc_info.value.code == "invalid_request"
    assert "gpt-nonexistent" in exc_info.value.message, (
        "the defect message must name the unknown target for operator diagnosis"
    )


def test_registry_revision_is_a_dated_revision() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}\.\d+", REGISTRY_REVISION), (
        f"REGISTRY_REVISION must be a dated revision like '2026-08-09.1', got {REGISTRY_REVISION!r}"
    )


# ---------------------------------------------------------------------------
# Row invariants — enforced at module import via _validate_rows


def test_a_well_formed_row_set_validates() -> None:
    reasoning_map: Mapping[ReasoningLevel, object] = {"low": "low", "high": "high"}
    rows = (
        _row(),
        _row(
            ref="openai:gpt-5.6",
            provider="openai",
            model_id="gpt-5.6-terra",
            engine="openai_responses",
            reasoning=Present(reasoning_map),
        ),
        _row(
            ref="openrouter:kimi-k3",
            provider="openrouter",
            model_id="moonshotai/kimi-k3",
            engine="openai_chat",
            routing=Present(_pinned_routing()),
        ),
    )
    assert _validate_rows(rows) is None, "a well-formed row set must validate without raising"


def test_duplicate_refs_are_rejected_at_validation() -> None:
    rows = (_row(), _row(model_id="claude-opus-5-20260201"))
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows(rows)
    assert exc_info.value.origin == "intent"
    assert exc_info.value.code == "registry_invalid"
    assert "anthropic:claude-opus-5" in exc_info.value.message, (
        "the defect message must name the duplicated ref"
    )


def test_ref_must_be_provider_prefixed_nickname() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(ref="claude-opus-5"),))
    assert exc_info.value.code == "registry_invalid"

    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(ref="openai:claude-opus-5"),))
    assert exc_info.value.code == "registry_invalid", (
        "a ref whose prefix names a different provider than the row must be rejected"
    )


def test_openrouter_row_without_routing_is_rejected() -> None:
    row = _row(
        ref="openrouter:kimi-k3",
        provider="openrouter",
        model_id="moonshotai/kimi-k3",
        engine="openai_chat",
    )
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((row,))
    assert exc_info.value.code == "registry_invalid"
    assert "routing" in exc_info.value.message, (
        "openrouter rows without pinned routing must be named as the violation"
    )


def test_non_openrouter_row_with_routing_is_rejected() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(routing=Present(_pinned_routing())),))
    assert exc_info.value.code == "registry_invalid"
    assert "routing" in exc_info.value.message


def test_reasoning_keys_outside_reasoning_level_are_rejected() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(reasoning=Present({"turbo": "think-harder"})),))  # type: ignore
    assert exc_info.value.code == "registry_invalid"
    assert "turbo" in exc_info.value.message, (
        "the defect message must name the out-of-union reasoning key"
    )


def test_engine_must_match_the_providers_dialect() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(engine="openai_chat"),))
    assert exc_info.value.code == "registry_invalid"
    assert "anthropic_messages" in exc_info.value.message, (
        "the defect message must name the expected engine for the row's provider"
    )


def test_openrouter_routing_rejects_zero_pins() -> None:
    with pytest.raises(ValueError, match="must name at least one pin"):
        OpenRouterRouting(only=(), order=("moonshotai",), quantizations=("fp8",))
    with pytest.raises(ValueError, match="must name at least one pin"):
        OpenRouterRouting(only=("moonshotai",), order=("moonshotai",), quantizations=())


def test_duplicate_provider_model_id_identity_is_rejected() -> None:
    rows = (
        _row(),
        _row(ref="anthropic:opus-nickname-two"),
    )
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows(rows)
    assert exc_info.value.code == "registry_invalid"
    assert "model_id" in exc_info.value.message, (
        "(provider, model_id) is the resolve_target key and must be unique"
    )


def test_nonpositive_token_limits_are_rejected() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(context_window=0),))
    assert exc_info.value.code == "registry_invalid"
    assert "positive" in exc_info.value.message
