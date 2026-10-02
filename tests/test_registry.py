"""Current executable catalog contract, including complete private lowering."""

from dataclasses import replace

import pytest

from provider_runtime.engines._common import row_reasoning
from provider_runtime.errors import InvalidRequest
from provider_runtime.registry import (
    GPT6_MODEL_IDS,
    _resolve,
    _resolve_target,
    api_model_catalog,
)
from provider_runtime.types import (
    Absent,
    ApiReasoningFacts,
    GenerateIntent,
    Present,
    PromptBlock,
    ProviderTarget,
    TextOutput,
    UserMessage,
)

EXPECTED = {
    "openai:gpt-6-astra": 10,
    "openai:gpt-6-sol": 12,
    "openai:gpt-6-luna": 12,
    "anthropic:claude-fable-5-1": 5,
    "anthropic:claude-opus-5-5": 5,
    "anthropic:claude-sonnet-5": 10,
    "gemini:gemini-3.8-flash": 3,
    "deepseek:deepseek-flash": 4,
    "xai:grok-4.7": 4,
}


def _intent(ref: str, key: str) -> GenerateIntent:
    row = _resolve(ref)
    return GenerateIntent(
        target=ProviderTarget(row.provider, row.model_id),
        messages=(UserMessage((PromptBlock("hello"),)),),
        max_output_tokens=4096,
        reasoning=key,
        tools=(),
        tool_choice="auto",
        output=TextOutput(),
    )


def test_exact_current_catalog_and_shared_gpt6_identity() -> None:
    catalog = api_model_catalog()
    rows = {row.model_ref: row for row in catalog.models}
    assert {row.ref for row in map(_resolve, EXPECTED)} == set(EXPECTED)
    assert set(rows) == set(EXPECTED)
    assert GPT6_MODEL_IDS == ("gpt-6-astra", "gpt-6-sol", "gpt-6-luna")
    assert {row.dispatch.model_id for row in catalog.models if row.provider == "openai"} == set(
        GPT6_MODEL_IDS
    )
    assert sum(len(row.reasoning) for row in catalog.models) == 65
    assert all(len(rows[ref].reasoning) == count for ref, count in EXPECTED.items())
    assert all(row.label and row.row_fingerprint for row in catalog.models)
    assert catalog.definition_revision == api_model_catalog().definition_revision


def test_source_capacity_is_not_inferred_from_a_request_budget() -> None:
    expected = {
        **{f"openai:{model}": (1_050_000, Present(128_000)) for model in GPT6_MODEL_IDS},
        **{
            f"anthropic:{model}": (1_000_000, Present(128_000))
            for model in ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5")
        },
        "gemini:gemini-3.8-flash": (1_048_576, Present(65_536)),
        "deepseek:deepseek-flash": (1_048_576, Present(393_216)),
        "xai:grok-4.7": (500_000, Absent()),
    }
    assert {
        row.model_ref: (row.context_window, row.max_output_tokens)
        for row in api_model_catalog().models
    } == expected
    assert _resolve("deepseek:deepseek-flash").max_output_tokens == 384_000
    assert _resolve("xai:grok-4.7").max_output_tokens == 4096


def test_options_are_complete_content_and_defaults_are_sourced() -> None:
    for row in api_model_catalog().models:
        assert len({option.key for option in row.reasoning}) == len(row.reasoning)
        assert all(option.label and 1 <= len(option.key) <= 64 for option in row.reasoning)
        assert all(not hasattr(option, "native_wire_fragment") for option in row.reasoning)
        match row.source_default_reasoning:
            case Present(value=key):
                assert key in {option.key for option in row.reasoning}
            case Absent():
                assert row.model_ref == "openai:gpt-6-astra"
    assert [o.key for o in _resolve_facts("openai:gpt-6-astra").reasoning] == [
        f"{mode}/{effort}"
        for mode in ("standard", "pro")
        for effort in ("low", "medium", "high", "xhigh", "max")
    ]
    assert _resolve_facts("anthropic:claude-sonnet-5").reasoning[-1] == ApiReasoningFacts(
        key="disabled/max", label="thinking off · max effort"
    )


def _resolve_facts(ref: str):
    return next(row for row in api_model_catalog().models if row.model_ref == ref)


@pytest.mark.parametrize("ref", EXPECTED)
def test_every_declared_key_lowers_to_one_nonempty_native_configuration(ref: str) -> None:
    row = _resolve(ref)
    facts = _resolve_facts(ref)
    for option in facts.reasoning:
        lowered = row_reasoning(row, _intent(ref, option.key))
        assert lowered.fragment
        assert isinstance(lowered.native_reasoning, Present)


def test_composite_and_single_axis_lowering() -> None:
    assert row_reasoning(
        _resolve("openai:gpt-6-sol"), _intent("openai:gpt-6-sol", "pro/none")
    ).fragment == {"reasoning": {"mode": "pro", "effort": "none"}}
    assert row_reasoning(
        _resolve("anthropic:claude-sonnet-5"), _intent("anthropic:claude-sonnet-5", "disabled/max")
    ).fragment == {"thinking": {"type": "disabled"}, "output_config": {"effort": "max"}}
    assert row_reasoning(
        _resolve("deepseek:deepseek-flash"), _intent("deepseek:deepseek-flash", "none")
    ).fragment == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize(
    "ref,key",
    [
        ("openai:gpt-6-astra", "standard/none"),
        ("openai:gpt-6-sol", "high"),
        ("anthropic:claude-opus-5-5", "disabled/high"),
        ("gemini:gemini-3.8-flash", "minimal"),
        ("deepseek:deepseek-flash", "medium"),
        ("xai:grok-4.7", "max"),
    ],
)
def test_undeclared_configuration_rejected(ref: str, key: str) -> None:
    with pytest.raises(InvalidRequest, match="reasoning key"):
        row_reasoning(_resolve(ref), _intent(ref, key))


@pytest.mark.parametrize(
    "target",
    [
        ProviderTarget("openai", "gpt-5.6-sol"),
        ProviderTarget("gemini", "gemini-3.5-flash"),
        ProviderTarget("deepseek", "deepseek-v4-pro"),
        ProviderTarget("xai", "grok-4.5"),
    ],
)
def test_retired_exact_target_rejected(target: ProviderTarget) -> None:
    with pytest.raises(InvalidRequest, match="no registry row"):
        _resolve_target(target)


def test_catalog_fingerprint_rotates_on_private_native_mapping_change() -> None:
    from provider_runtime.registry import _row_fingerprint

    row = _resolve("openai:gpt-6-sol")
    assert isinstance(row.reasoning, Present)
    changed = replace(
        row,
        reasoning=Present(
            {**row.reasoning.value, "standard/low": {"reasoning": {"mode": "pro", "effort": "low"}}}
        ),
    )
    assert _row_fingerprint(changed) != _row_fingerprint(row)
