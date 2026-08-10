"""Pure selection rules for the paid local-agent certification matrix."""

from __future__ import annotations

import re

_MODEL_NAME = re.compile(r"[^\x00-\x20\x7f,]{1,256}\Z")


class MatrixSelectionError(ValueError):
    """The operator input cannot prove the coverage it claims."""


def parse_model_list(raw: str | None) -> tuple[str, ...]:
    """Decode an optional exact, ordered model list from operator input.

    An absent or empty value selects the backend's default model — with the
    capability matrix deleted there is no discovery surface to enumerate models
    from, so any wider coverage must be named explicitly by the operator.
    """
    if raw is None or not raw:
        return ()
    models = tuple(raw.split(","))
    if any(_MODEL_NAME.fullmatch(model) is None for model in models):
        raise MatrixSelectionError(
            "live model names must be comma-free, whitespace-free native identifiers"
        )
    if len(models) != len(set(models)):
        raise MatrixSelectionError("live model list contains duplicates")
    return models
