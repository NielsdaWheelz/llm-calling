from __future__ import annotations

import pytest

from provider_runtime.agent_runtime._structured_output import (
    OutputSchemaMismatch,
    freeze_structured_output,
    parse_structured_output,
)
from provider_runtime.agent_runtime.types import FrozenJsonDict


def test_text_output_is_strictly_parsed_and_frozen() -> None:
    value = parse_structured_output('{"answer":"yes","score":1.5,"flags":[true,false]}')

    assert isinstance(value, FrozenJsonDict), (
        f"parsed structured output should be FrozenJsonDict, got {type(value).__name__}"
    )
    assert value == {"answer": "yes", "score": 1.5, "flags": (True, False)}
    with pytest.raises(TypeError):
        value["answer"] = "no"  # type: ignore[index]


def test_sdk_value_is_copied_before_freezing() -> None:
    source = {"answer": "no", "score": 2}

    value = freeze_structured_output(source)
    source["answer"] = "yes"

    assert value == {"answer": "no", "score": 2}, (
        "a later mutation of the SDK value must not reach the frozen copy"
    )


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"answer":"yes","answer":"no"}',
        '{"score":NaN}',
        '[{"answer":"yes"}]',
        '"just a string"',
    ],
)
def test_json_syntax_duplicates_constants_and_nonobject_roots_are_rejected(text: str) -> None:
    with pytest.raises(OutputSchemaMismatch) as exc_info:
        parse_structured_output(text)

    assert text not in str(exc_info.value), (
        "structured-output failures must not echo model text into the error"
    )


def test_nonjson_native_values_are_one_safe_named_signal() -> None:
    with pytest.raises(OutputSchemaMismatch):
        freeze_structured_output(["not", "an", "object"])
    with pytest.raises(OutputSchemaMismatch):
        freeze_structured_output({"score": float("inf")})
    with pytest.raises(OutputSchemaMismatch):
        freeze_structured_output(object())
