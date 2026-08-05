"""Strict structured-output decoding over the canonical schema subset."""

from __future__ import annotations

import json
from typing import assert_never, cast

from provider_runtime.agent_runtime.errors import InvalidAgentRequest
from provider_runtime.agent_runtime.types import (
    FrozenJsonDict,
    JsonValue,
    freeze_json_value,
)
from provider_runtime.schema import (
    ArrayNode,
    CanonicalJsonSchema,
    Node,
    NullableUnion,
    NullNode,
    ObjectNode,
    Ref,
    ScalarNode,
)
from provider_runtime.types import Present


class OutputSchemaMismatch(Exception):
    """Expected model-output failure; adapters convert it to a terminal value."""


class _InvalidJson(Exception):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise _InvalidJson
        value[key] = child
    return value


def _reject_constant(_value: str) -> object:
    raise _InvalidJson


def parse_structured_output(text: str, schema: CanonicalJsonSchema) -> FrozenJsonDict:
    """Parse one strict JSON object and validate it without repair or coercion."""
    if type(text) is not str:
        raise OutputSchemaMismatch("structured output was not text JSON")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (_InvalidJson, ValueError, RecursionError):
        raise OutputSchemaMismatch("structured output was not strict JSON") from None
    return validate_structured_output(value, schema)


def validate_structured_output(value: object, schema: CanonicalJsonSchema) -> FrozenJsonDict:
    """Freeze and validate one SDK/native structured value against ``schema``."""
    if not isinstance(schema, CanonicalJsonSchema):
        raise TypeError("schema must be CanonicalJsonSchema")
    try:
        frozen = freeze_json_value(value, context="structured_output")
    except InvalidAgentRequest:
        raise OutputSchemaMismatch("structured output was not finite JSON data") from None
    _validate_node(frozen, schema.root, schema, path="$", resolving=())
    if not isinstance(frozen, FrozenJsonDict):
        raise OutputSchemaMismatch("structured output root was not an object")
    return frozen


def _validate_node(
    value: JsonValue,
    node: Node,
    schema: CanonicalJsonSchema,
    *,
    path: str,
    resolving: tuple[str, ...],
) -> None:
    match node:
        case ObjectNode(properties=properties):
            if not isinstance(value, FrozenJsonDict):
                raise OutputSchemaMismatch(f"structured output type mismatch at {path}")
            expected = set(properties)
            if set(value) != expected:
                raise OutputSchemaMismatch(f"structured output object shape mismatch at {path}")
            for name, child_schema in properties.items():
                _validate_node(
                    cast(JsonValue, value[name]),
                    child_schema,
                    schema,
                    path=f"{path}.{name}",
                    resolving=resolving,
                )
        case ArrayNode(items=items):
            if not isinstance(value, tuple):
                raise OutputSchemaMismatch(f"structured output type mismatch at {path}")
            for index, child in enumerate(value):
                _validate_node(
                    child,
                    items,
                    schema,
                    path=f"{path}[{index}]",
                    resolving=resolving,
                )
        case ScalarNode(type=scalar_type, enum=enum):
            compatible = {
                "string": type(value) is str,
                "integer": type(value) is int,
                "number": type(value) in (int, float),
                "boolean": type(value) is bool,
            }[scalar_type]
            if not compatible:
                raise OutputSchemaMismatch(f"structured output type mismatch at {path}")
            if isinstance(enum, Present) and value not in enum.value:
                raise OutputSchemaMismatch(f"structured output enum mismatch at {path}")
        case NullableUnion(non_null=non_null):
            if value is not None:
                _validate_node(
                    value,
                    non_null,
                    schema,
                    path=path,
                    resolving=resolving,
                )
        case Ref(name=name):
            if name in resolving:
                raise TypeError("canonical schema contains a recursive ref")
            _validate_node(
                value,
                schema.defs[name],
                schema,
                path=path,
                resolving=(*resolving, name),
            )
        case NullNode():
            if value is not None:
                raise OutputSchemaMismatch(f"structured output type mismatch at {path}")
        case _:
            assert_never(node)


__all__ = [
    "OutputSchemaMismatch",
    "parse_structured_output",
    "validate_structured_output",
]
