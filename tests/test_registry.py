"""Tests for the model registry: resolution, row invariants, and shipped content."""

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


def _row(
    *,
    ref: str = "anthropic:claude-opus-5",
    provider: ProviderName = "anthropic",
    model_id: str = "claude-opus-5-20260101",
    engine: EngineId = "anthropic_messages",
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
        base_url=Absent(),
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
# Content — the shipped ROWS. Facts: old catalog.py port (provider docs
# verified 2026-07-20) for openai/anthropic/gemini/moonshot/openrouter;
# docs.deepseek.com + docs.x.ai (verified 2026-08-10) for deepseek/xai.

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


def test_resolve_returns_the_shipped_row_for_a_ref() -> None:
    row = resolve("moonshot:kimi-k3")
    assert row.provider == "moonshot", f"expected the moonshot row, got {row!r}"
    assert row.model_id == "kimi-k3"
    assert row.engine == "openai_chat"
    assert row.base_url == Present("https://api.moonshot.ai/v1"), (
        "moonshot rides the openai SDK as a compat client and must carry its own base_url"
    )


def test_resolve_target_returns_the_row_matching_wire_identity() -> None:
    terra = resolve_target(ProviderTarget(provider="openai", model="gpt-5.6-terra"))
    assert terra.ref == "openai:gpt-5.6-terra", (
        f"(openai, gpt-5.6-terra) must resolve to its own row, got {terra.ref!r}"
    )
    # The openrouter wire id is the pinned dated slug, not the nickname.
    pinned = resolve_target(
        ProviderTarget(provider="openrouter", model="moonshotai/kimi-k3-20260715")
    )
    assert pinned.ref == "openrouter:kimi-k3", (
        f"the openrouter row must be keyed by its pinned wire model id, got {pinned.ref!r}"
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


def test_ported_context_and_output_caps_match_the_catalog() -> None:
    expected = {
        "openai:gpt-5.6-sol": (1_050_000, 128_000),
        "openai:gpt-5.6-terra": (1_050_000, 128_000),
        "openai:gpt-5.6-luna": (1_050_000, 128_000),
        "anthropic:claude-sonnet-5": (1_000_000, 128_000),
        "anthropic:claude-fable-5": (1_000_000, 128_000),
        "gemini:gemini-3.5-flash": (1_048_576, 65_536),
        "moonshot:kimi-k3": (1_048_576, 131_072),
        "openrouter:kimi-k3": (1_048_576, 131_072),
    }
    for ref, (context_window, max_output_tokens) in expected.items():
        row = resolve(ref)
        assert (row.context_window, row.max_output_tokens) == (
            context_window,
            max_output_tokens,
        ), (
            f"row {ref!r}: expected {(context_window, max_output_tokens)}, "
            f"got {(row.context_window, row.max_output_tokens)}"
        )


def test_reasoning_maps_carry_exact_native_wire_values() -> None:
    # Native values are the provider's own effort/level strings, verbatim; a
    # model without a level simply omits the key (no invented levels).
    expected: dict[str, Present[Mapping[ReasoningLevel, object]]] = {
        # openai: Responses reasoning.effort; xhigh and max are DISTINCT efforts.
        "openai:gpt-5.6-sol": Present(
            {
                "none": "none",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "xhigh",
                "max": "max",
            }
        ),
        # anthropic: output_config.effort; no off switch.
        "anthropic:claude-fable-5": Present(
            {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}
        ),
        # gemini 3.5: thinkingConfig.thinkingLevel strings (not 2.5 thinkingBudget ints).
        "gemini:gemini-3.5-flash": Present(
            {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high"}
        ),
        # kimi k3: reasoning_effort direct / reasoning.effort via openrouter.
        "moonshot:kimi-k3": Present({"low": "low", "high": "high", "max": "max"}),
        "openrouter:kimi-k3": Present({"low": "low", "high": "high", "max": "max"}),
    }
    for ref, want in expected.items():
        row = resolve(ref)
        assert row.reasoning == want, (
            f"row {ref!r} reasoning map drifted from the catalog's verified native values: "
            f"expected {want!r}, got {row.reasoning!r}"
        )
    assert resolve("openai:gpt-5.6-terra").reasoning == expected["openai:gpt-5.6-sol"]
    assert resolve("openai:gpt-5.6-luna").reasoning == expected["openai:gpt-5.6-sol"]
    assert resolve("anthropic:claude-sonnet-5").reasoning == expected["anthropic:claude-fable-5"]


def test_deepseek_rows_carry_verified_facts() -> None:
    # docs verified 2026-08-10: 1M context / 384K output for both hosted V4
    # models; thinking enabled by default; reasoning_effort low|high|max on
    # flash, high|max only on pro; {"thinking": {"type": "disabled"}} is the
    # exact off-switch fragment on the openai-format wire.
    thinking_off = {"thinking": {"type": "disabled"}}
    flash = resolve("deepseek:deepseek-v4-flash")
    pro = resolve("deepseek:deepseek-v4-pro")
    for row in (flash, pro):
        assert (row.context_window, row.max_output_tokens) == (1_000_000, 384_000), (
            f"row {row.ref!r}: expected (1_000_000, 384_000), "
            f"got {(row.context_window, row.max_output_tokens)}"
        )
        assert row.engine == "openai_chat"
    assert flash.reasoning == Present(
        {"none": thinking_off, "low": "low", "high": "high", "max": "max"}
    ), f"flash reasoning drifted: {flash.reasoning!r}"
    assert pro.reasoning == Present({"none": thinking_off, "high": "high", "max": "max"}), (
        f"pro accepts only high|max effort (docs 2026-08-10); a 'low' key would silently "
        f"upgrade the caller's request: {pro.reasoning!r}"
    )


def test_xai_row_carries_verified_facts() -> None:
    row = resolve("xai:grok-4.5")
    assert row.model_id == "grok-4.5"
    assert row.engine == "openai_chat"
    assert (row.context_window, row.max_output_tokens) == (500_000, 500_000), (
        "xAI publishes no separate output cap for grok-4.5; the shared 500k window is "
        f"the only documented bound — got {(row.context_window, row.max_output_tokens)}"
    )
    assert row.reasoning == Present({"low": "low", "medium": "medium", "high": "high"}), (
        f"grok-4.5 reasoning_effort is low|medium|high and cannot be disabled (no 'none' "
        f"key): {row.reasoning!r}"
    )


def test_openrouter_row_pins_the_certified_upstream() -> None:
    row = resolve("openrouter:kimi-k3")
    assert row.model_id == "moonshotai/kimi-k3-20260715", (
        "the openrouter wire id must stay the catalog's pinned canonical revision"
    )
    match row.routing:
        case Present(value=routing):
            assert routing.only == ("moonshotai/int4",), routing
            assert routing.order == ("moonshotai/int4",), routing
            assert routing.quantizations == ("int4",), (
                "the pinned variant is re-sent as an explicit quantization filter "
                f"(belt over the endpoint-slug pin): {routing.quantizations!r}"
            )
        case Absent():
            pytest.fail("openrouter:kimi-k3 must carry routing pins")
