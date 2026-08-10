from __future__ import annotations

import pytest

from tests.live.agent_matrix import MatrixSelectionError, parse_model_list


def test_an_absent_model_list_selects_the_backend_default() -> None:
    assert parse_model_list(None) == ()
    assert parse_model_list("") == ()


def test_model_lists_are_exact_ordered_and_unique() -> None:
    assert parse_model_list("model-a,model-b") == ("model-a", "model-b")
    for malformed in ("model-a,", "model-a, model-b"):
        with pytest.raises(MatrixSelectionError):
            parse_model_list(malformed)
    with pytest.raises(MatrixSelectionError, match="duplicates"):
        parse_model_list("model-a,model-a")
