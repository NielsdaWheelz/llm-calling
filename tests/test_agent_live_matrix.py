from __future__ import annotations

import pytest

from tests.live.agent_matrix import (
    MatrixSelectionError,
    ModelReasoningCase,
    claude_model_reasoning_cases,
    codex_model_reasoning_cases,
    parse_claude_models,
)


def test_claude_model_list_is_exact_ordered_and_unique() -> None:
    assert parse_claude_models("claude-opus,claude-sonnet") == (
        "claude-opus",
        "claude-sonnet",
    )
    for malformed in (None, "", "claude-opus,", "claude-opus, claude-sonnet"):
        with pytest.raises(MatrixSelectionError):
            parse_claude_models(malformed)
    with pytest.raises(MatrixSelectionError, match="duplicates"):
        parse_claude_models("claude-opus,claude-opus")


def test_codex_matrix_uses_each_models_own_efforts() -> None:
    assert codex_model_reasoning_cases(
        ("narrow", "wide", "plain"),
        (("narrow", ("low",)), ("wide", ("low", "high")), ("plain", ())),
    ) == (
        ModelReasoningCase("narrow", "low"),
        ModelReasoningCase("wide", "low"),
        ModelReasoningCase("wide", "high"),
        ModelReasoningCase("plain", None),
    )


def test_codex_matrix_refuses_incomplete_model_mapping() -> None:
    with pytest.raises(MatrixSelectionError, match="one reasoning-effort mapping per model"):
        codex_model_reasoning_cases(("one", "two"), (("one", ("low",)),))


def test_claude_matrix_is_complete_cross_product() -> None:
    assert claude_model_reasoning_cases(("opus", "sonnet"), ("low", "high")) == (
        ModelReasoningCase("opus", "low"),
        ModelReasoningCase("opus", "high"),
        ModelReasoningCase("sonnet", "low"),
        ModelReasoningCase("sonnet", "high"),
    )
