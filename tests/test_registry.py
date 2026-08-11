"""Tests for the model registry: resolution, row invariants, and shipped content."""

import json
import re
from collections.abc import Mapping
from typing import Literal, get_args

import pytest

from provider_runtime.errors import InvalidRequest, RuntimeDefect
from provider_runtime.registry import (
    REGISTRY_REVISION,
    ROWS,
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
_TEXT: frozenset[Literal["text", "image"]] = frozenset({"text"})
_LEVELS: frozenset[str] = frozenset(get_args(ReasoningLevel.__value__))


def _row(
    *,
    ref: str = "anthropic:claude-opus-5",
    provider: ProviderName = "anthropic",
    model_id: str = "claude-opus-5-20260101",
    engine: EngineId = "anthropic_messages",
    base_url: Presence[str] = _ABSENT,
    reasoning: Presence[Mapping[ReasoningLevel, object]] = _ABSENT,
    routing: Presence[OpenRouterRouting] = _ABSENT,
    context_window: int = 200_000,
    modalities: frozenset[Literal["text", "image"]] = _TEXT,
    continuation_codec: str | None = None,
) -> ModelRow:
    return ModelRow(
        ref=ref,
        provider=provider,
        model_id=model_id,
        engine=engine,
        base_url=base_url,
        context_window=context_window,
        max_output_tokens=64_000,
        modalities=modalities,
        tools=True,
        streaming=True,
        structured="native",
        reasoning=reasoning,
        continuation_codec=continuation_codec or f"{provider}.v1",
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
    reasoning_map: Mapping[ReasoningLevel, object] = {
        "low": {"reasoning": {"effort": "low"}},
        "high": {"reasoning": {"effort": "high"}},
    }
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
            base_url=Present("https://openrouter.ai/api/v1"),
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
        base_url=Present("https://openrouter.ai/api/v1"),
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
        _validate_rows((_row(reasoning=Present({"turbo": {"effort": "turbo"}})),))  # type: ignore
    assert exc_info.value.code == "registry_invalid"
    assert "turbo" in exc_info.value.message, (
        "the defect message must name the out-of-union reasoning key"
    )


def test_reasoning_values_that_engines_cannot_merge_are_rejected() -> None:
    # Engines merge the value verbatim as request parameters: a bare level
    # string or an empty fragment is a call-time defect, caught at import here.
    bare_string: Mapping[ReasoningLevel, object] = {"high": "high"}
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(reasoning=Present(bare_string)),))
    assert exc_info.value.code == "registry_invalid"
    assert "'high'" in exc_info.value.message, (
        f"the defect message must name the offending level: {exc_info.value.message!r}"
    )

    empty_fragment: Mapping[ReasoningLevel, object] = {"high": {}}
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(reasoning=Present(empty_fragment)),))
    assert exc_info.value.code == "registry_invalid", (
        "a level mapping to an empty fragment declares a knob that sends nothing"
    )


def test_reasoning_knob_without_levels_is_rejected() -> None:
    # Present({}) is strictly worse than Absent: every level, including "none",
    # becomes inexpressible while the row still claims a reasoning knob.
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(reasoning=Present({})),))
    assert exc_info.value.code == "registry_invalid"
    assert "no levels" in exc_info.value.message, exc_info.value.message


def test_openai_chat_row_without_base_url_is_rejected() -> None:
    row = _row(
        ref="moonshot:kimi-k3", provider="moonshot", model_id="kimi-k3", engine="openai_chat"
    )
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((row,))
    assert exc_info.value.code == "registry_invalid"
    assert "base_url" in exc_info.value.message, (
        "openai_chat serves compatibility hosts; without a base_url the SDK would call OpenAI"
    )


def test_engine_must_match_the_providers_dialect() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(engine="openai_chat"),))
    assert exc_info.value.code == "registry_invalid"
    assert "anthropic_messages" in exc_info.value.message, (
        "the defect message must name the expected engine for the row's provider"
    )


def test_openrouter_routing_rejects_zero_pins() -> None:
    # Zero pins is unpinned passthrough with fallbacks merely disabled — the
    # same curation defect taxonomy as every other row invariant.
    with pytest.raises(RuntimeDefect, match="must name at least one pin") as exc_info:
        OpenRouterRouting(only=(), order=("moonshotai",), quantizations=("fp8",))
    assert (exc_info.value.origin, exc_info.value.code) == ("intent", "registry_invalid")
    with pytest.raises(RuntimeDefect, match="must name at least one pin"):
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


def test_output_cap_exceeding_context_window_is_rejected() -> None:
    # Helper rows carry max_output_tokens=64_000; a 1_000-token window inverts the pair.
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(context_window=1_000),))
    assert exc_info.value.code == "registry_invalid"
    assert "context_window" in exc_info.value.message, (
        "an output cap above the context window is a curation typo and must be named"
    )


def test_codec_bound_to_a_different_provider_is_rejected() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(continuation_codec="openai.v1"),))
    assert exc_info.value.code == "registry_invalid"
    assert "continuation_codec" in exc_info.value.message, (
        "a codec id not owned by the row's provider would bind continuations across providers"
    )


def test_row_without_text_modality_is_rejected() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        _validate_rows((_row(modalities=frozenset({"image"})),))
    assert exc_info.value.code == "registry_invalid"
    assert "text" in exc_info.value.message


# ---------------------------------------------------------------------------
# Content — the shipped ROWS. Every row's provider facts were re-verified
# against the vendor docs on 2026-08-10 (source URLs live in the registry's
# per-provider comments); what the reasoning knob puts on the wire is a
# provider-doc fact plus the live matrix, not a unit-test constant, so the
# tests below check the properties engines depend on rather than restating
# fragments.

# Every target Nexus constructs today (nexus-web llm_profiles.py) must resolve.
_NEXUS_TARGETS: tuple[tuple[ProviderName, str], ...] = (
    ("openai", "gpt-5.6-luna"),
    ("openai", "gpt-5.6-terra"),
    ("openai", "gpt-5.6-sol"),
    ("anthropic", "claude-sonnet-5"),
    ("anthropic", "claude-fable-5"),
    ("gemini", "gemini-3.5-flash"),
    ("moonshot", "kimi-k3"),
)


def test_every_row_resolves_by_ref_and_by_wire_identity() -> None:
    # Callers reach a row either way: by nickname ref (facade) or by the
    # (provider, wire model id) a continuation replays to.
    for row in ROWS:
        assert resolve(row.ref) is row, f"ref {row.ref!r} did not resolve to its own row"
        target = ProviderTarget(provider=row.provider, model=row.model_id)
        assert resolve_target(target) is row, (
            f"({row.provider}, {row.model_id}) resolved to a different row than {row.ref!r}"
        )


def test_every_nexus_consumed_target_has_a_row() -> None:
    for provider, model in _NEXUS_TARGETS:
        row = resolve_target(ProviderTarget(provider=provider, model=model))
        assert (row.provider, row.model_id) == (provider, model), (
            f"Nexus target ({provider}, {model}) resolved to the wrong row {row.ref!r}"
        )


def test_all_seven_providers_are_covered() -> None:
    covered = {row.provider for row in ROWS}
    expected = set(get_args(ProviderName.__value__))
    assert covered == expected, (
        f"the spec's acceptance matrix needs every provider callable; "
        f"missing: {sorted(expected - covered)}, unexpected: {sorted(covered - expected)}"
    )


def test_continuation_codecs_follow_the_provider_codec_table() -> None:
    for row in ROWS:
        assert row.continuation_codec == f"{row.provider}.v1", (
            f"row {row.ref!r} codec {row.continuation_codec!r} must be the freeze table's "
            f"'{row.provider}.v1' (version bumps only on incompatible payload changes)"
        )


def test_base_urls_are_absent_for_native_sdks_and_pinned_for_compat() -> None:
    expected: dict[ProviderName, str | None] = {
        "openai": None,
        "anthropic": None,
        "gemini": None,
        "deepseek": "https://api.deepseek.com",
        "moonshot": "https://api.moonshot.ai/v1",
        "xai": "https://api.x.ai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    for row in ROWS:
        want = expected[row.provider]
        got = None if isinstance(row.base_url, Absent) else row.base_url.value
        assert got == want, f"row {row.ref!r} base_url: expected {want!r}, got {got!r}"


def test_structured_output_dialect_per_provider() -> None:
    # spec §5: native strict schema on openai/anthropic/gemini/xai; JSON-mode +
    # pydantic validation on deepseek/moonshot (and the kimi-upstream openrouter row).
    expected: dict[ProviderName, str] = {
        "openai": "native",
        "anthropic": "native",
        "gemini": "native",
        "xai": "native",
        "deepseek": "json_mode",
        "moonshot": "json_mode",
        "openrouter": "json_mode",
    }
    for row in ROWS:
        assert row.structured == expected[row.provider], (
            f"row {row.ref!r} structured: expected {expected[row.provider]!r}, "
            f"got {row.structured!r}"
        )


def test_correlation_facts_per_provider() -> None:
    expected: dict[ProviderName, str] = {
        "openai": "header",  # x-request-id
        "anthropic": "header",  # request-id
        "gemini": "none",  # no correlation id on this wire
        "moonshot": "in_band",  # response/chunk id; no confirmed header
        "openrouter": "in_band",  # generation id
        "deepseek": "in_band",  # chat-completions body id
        "xai": "in_band",  # chat-completions body id
    }
    for row in ROWS:
        assert row.correlation == expected[row.provider], (
            f"row {row.ref!r} correlation: expected {expected[row.provider]!r}, "
            f"got {row.correlation!r}"
        )


def test_every_row_supports_tools_and_streaming() -> None:
    for row in ROWS:
        assert row.tools and row.streaming, (
            f"row {row.ref!r}: every curated model documents tool calling and streaming; "
            f"got tools={row.tools}, streaming={row.streaming}"
        )


def test_modalities_are_verified_facts() -> None:
    # deepseek: hosted API is text-only (docs verified 2026-08-10); the
    # openrouter row stays text-only until the pinned upstream's image
    # acceptance is live-verified. Everything else documents image input.
    for row in ROWS:
        expect_image = row.provider not in ("deepseek", "openrouter")
        assert ("image" in row.modalities) == expect_image, (
            f"row {row.ref!r} modalities {sorted(row.modalities)} contradict provider docs"
        )


def test_context_and_output_caps_match_provider_docs() -> None:
    # xai publishes no separate output cap for grok-4.5 — the shared window is
    # the only documented bound. deepseek V4 documents 384k output on both tiers.
    expected = {
        "openai:gpt-5.6-sol": (1_050_000, 128_000),
        "openai:gpt-5.6-terra": (1_050_000, 128_000),
        "openai:gpt-5.6-luna": (1_050_000, 128_000),
        "anthropic:claude-sonnet-5": (1_000_000, 128_000),
        "anthropic:claude-fable-5": (1_000_000, 128_000),
        "gemini:gemini-3.5-flash": (1_048_576, 65_536),
        "moonshot:kimi-k3": (1_048_576, 131_072),
        "openrouter:kimi-k3": (1_048_576, 131_072),
        "deepseek:deepseek-v4-pro": (1_000_000, 384_000),
        "deepseek:deepseek-v4-flash": (1_000_000, 384_000),
        "xai:grok-4.5": (500_000, 500_000),
    }
    assert set(expected) == {row.ref for row in ROWS}, (
        "every shipped row's token limits are a provider-doc fact and must be listed here; "
        f"unlisted: {sorted({row.ref for row in ROWS} - set(expected))}, "
        f"stale: {sorted(set(expected) - {row.ref for row in ROWS})}"
    )
    for ref, (context_window, max_output_tokens) in expected.items():
        row = resolve(ref)
        assert (row.context_window, row.max_output_tokens) == (
            context_window,
            max_output_tokens,
        ), (
            f"row {ref!r}: expected {(context_window, max_output_tokens)}, "
            f"got {(row.context_window, row.max_output_tokens)}"
        )


def test_every_declared_level_maps_to_a_distinct_wire_fragment() -> None:
    # Engines hold zero reasoning shape knowledge: they merge row.reasoning[level]
    # verbatim into the request and stamp it into CallMeta.native_reasoning as
    # compact sorted-keys JSON. A value that is not a non-empty string-keyed
    # JSON object cannot be merged, and two levels sharing a fragment means one
    # of them silently sends the other's request.
    for row in ROWS:
        if isinstance(row.reasoning, Absent):
            continue
        levels = row.reasoning.value
        assert levels, f"row {row.ref!r} claims a reasoning knob but declares no levels"
        assert set(levels) <= _LEVELS, (
            f"row {row.ref!r} declares levels outside ReasoningLevel: {sorted(set(levels) - _LEVELS)}"
        )
        seen: dict[str, ReasoningLevel] = {}
        for level, fragment in levels.items():
            assert isinstance(fragment, Mapping) and fragment, (
                f"row {row.ref!r} level {level!r} must map to a non-empty request fragment, "
                f"got {fragment!r}"
            )
            stamp = json.dumps(fragment, sort_keys=True, separators=(",", ":"))
            assert stamp not in seen, (
                f"row {row.ref!r} levels {seen[stamp]!r} and {level!r} put the identical "
                f"fragment {stamp} on the wire"
            )
            seen[stamp] = level


def test_openrouter_row_pins_a_live_endpoint() -> None:
    # openrouter.ai/api/v1/models/moonshotai/kimi-k3/endpoints (fetched
    # 2026-08-10): the first-party endpoint is tag "moonshotai/mxfp4",
    # quantization "mxfp4", serving "moonshotai/kimi-k3-20260715". With
    # only+order+quantizations+require_parameters all pinned, a tag or
    # quantization that no endpoint serves makes every call unroutable.
    row = resolve("openrouter:kimi-k3")
    assert row.model_id == "moonshotai/kimi-k3-20260715", (
        f"the wire id must stay the pinned dated revision, got {row.model_id!r}"
    )
    match row.routing:
        case Present(value=routing):
            assert routing.only == routing.order == ("moonshotai/mxfp4",), (
                f"only and order must both name the certified endpoint tag: {routing!r}"
            )
            assert routing.quantizations == ("mxfp4",), (
                "the endpoint's quantization is re-sent as an explicit filter; K3 ships "
                f"natively in MXFP4 (int4 was K2-era): {routing.quantizations!r}"
            )
        case Absent():
            pytest.fail("openrouter:kimi-k3 must carry routing pins")
