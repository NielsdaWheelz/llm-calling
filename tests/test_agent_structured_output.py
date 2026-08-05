from __future__ import annotations

import pytest

from provider_runtime.agent_runtime._structured_output import (
    OutputSchemaMismatch,
    parse_structured_output,
    validate_structured_output,
)
from provider_runtime.agent_runtime.types import FrozenJsonDict
from provider_runtime.schema import parse_canonical_schema

SCHEMA = parse_canonical_schema(
    {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no"]},
            "score": {"type": "number"},
            "details": {
                "anyOf": [
                    {"$ref": "#/$defs/details"},
                    {"type": "null"},
                ]
            },
        },
        "required": ["answer", "score", "details"],
        "additionalProperties": False,
        "$defs": {
            "details": {
                "type": "object",
                "properties": {"flags": {"type": "array", "items": {"type": "boolean"}}},
                "required": ["flags"],
                "additionalProperties": False,
            }
        },
    }
)


def test_text_output_is_strictly_parsed_validated_and_frozen() -> None:
    value = parse_structured_output(
        '{"answer":"yes","score":1.5,"details":{"flags":[true,false]}}',
        SCHEMA,
    )

    assert isinstance(value, FrozenJsonDict)
    assert value == {
        "answer": "yes",
        "score": 1.5,
        "details": {"flags": (True, False)},
    }
    with pytest.raises(TypeError):
        value["answer"] = "no"  # type: ignore[index]


def test_sdk_value_is_copied_before_validation() -> None:
    source = {"answer": "no", "score": 2, "details": None}

    value = validate_structured_output(source, SCHEMA)
    source["answer"] = "yes"

    assert value == {"answer": "no", "score": 2, "details": None}


@pytest.mark.parametrize(
    "value",
    [
        {"answer": "maybe", "score": 1, "details": None},
        {"answer": "yes", "score": True, "details": None},
        {"answer": "yes", "score": 1},
        {"answer": "yes", "score": 1, "details": None, "extra": "x"},
        {"answer": "yes", "score": 1, "details": {"flags": [1]}},
        ["not", "an", "object"],
    ],
)
def test_schema_mismatches_are_one_safe_named_signal(value: object) -> None:
    with pytest.raises(OutputSchemaMismatch) as exc_info:
        validate_structured_output(value, SCHEMA)

    assert repr(value) not in str(exc_info.value)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"answer":"yes","answer":"no","score":1,"details":null}',
        '{"answer":"yes","score":NaN,"details":null}',
        '[{"answer":"yes","score":1,"details":null}]',
        '{"answer":"yes","score":' + ("9" * 5_000) + ',"details":null}',
    ],
)
def test_json_syntax_duplicates_constants_and_nonobject_roots_are_rejected(text: str) -> None:
    with pytest.raises(OutputSchemaMismatch):
        parse_structured_output(text, SCHEMA)
