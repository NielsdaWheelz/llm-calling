"""Pure selection rules for the paid local-agent certification matrix."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MODEL_NAME = re.compile(r"[^\x00-\x20\x7f,]{1,256}\Z")


class MatrixSelectionError(ValueError):
    """The operator input or discovered matrix cannot prove complete coverage."""


@dataclass(frozen=True, slots=True)
class ModelReasoningCase:
    model: str
    reasoning_effort: str | None


def parse_claude_models(raw: str | None) -> tuple[str, ...]:
    """Decode the required exact, ordered Claude model list."""
    if raw is None or not raw:
        raise MatrixSelectionError("Claude live certification requires an explicit model list")
    models = tuple(raw.split(","))
    if any(_MODEL_NAME.fullmatch(model) is None for model in models):
        raise MatrixSelectionError(
            "Claude live model names must be comma-free, whitespace-free native identifiers"
        )
    if len(models) != len(set(models)):
        raise MatrixSelectionError("Claude live model list contains duplicates")
    return models


def codex_model_reasoning_cases(
    models: tuple[str, ...] | None,
    model_reasoning_efforts: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[ModelReasoningCase, ...]:
    """Build every discovered Codex model with only the efforts that model supports."""
    if models is None or not models:
        raise MatrixSelectionError("Codex live certification requires enumerated models")
    effort_map = dict(model_reasoning_efforts)
    if len(effort_map) != len(model_reasoning_efforts) or set(effort_map) != set(models):
        raise MatrixSelectionError(
            "Codex live certification requires one reasoning-effort mapping per model"
        )
    cases: list[ModelReasoningCase] = []
    for model in models:
        efforts = effort_map[model]
        if not efforts:
            cases.append(ModelReasoningCase(model, None))
            continue
        cases.extend(ModelReasoningCase(model, effort) for effort in efforts)
    return tuple(cases)


def claude_model_reasoning_cases(
    models: tuple[str, ...], reasoning_efforts: tuple[str, ...] | None
) -> tuple[ModelReasoningCase, ...]:
    """Build the explicit Claude model cross-product with every discovered effort."""
    if not models:
        raise MatrixSelectionError("Claude live certification requires at least one model")
    if reasoning_efforts is None:
        raise MatrixSelectionError("Claude live certification requires enumerated efforts")
    if not reasoning_efforts:
        return tuple(ModelReasoningCase(model, None) for model in models)
    return tuple(
        ModelReasoningCase(model, effort) for model in models for effort in reasoning_efforts
    )
