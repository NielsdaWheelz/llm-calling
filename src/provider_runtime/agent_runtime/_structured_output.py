"""Strict structured-output decoding.

Schema enforcement is native: both SDKs receive the caller's plain JSON Schema
through their public output-schema option and their backend enforces it. This
module owns the residual boundary work — parsing the final answer as *strict*
JSON (no repair, no coercion, no duplicate keys) and freezing native structured
values — and reports any miss as an expected model-output failure.
"""

from __future__ import annotations

import json

from .errors import InvalidAgentRequest
from .types import FrozenJsonDict, freeze_json_value


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


def parse_structured_output(text: str) -> FrozenJsonDict:
    """Parse one strict JSON object from the model's final text."""
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
    return freeze_structured_output(value)


def freeze_structured_output(value: object) -> FrozenJsonDict:
    """Freeze one SDK/native structured value; the root must be a JSON object."""
    try:
        frozen = freeze_json_value(value, context="structured_output")
    except InvalidAgentRequest:
        raise OutputSchemaMismatch("structured output was not finite JSON data") from None
    if not isinstance(frozen, FrozenJsonDict):
        raise OutputSchemaMismatch("structured output root was not an object")
    return frozen


__all__ = [
    "OutputSchemaMismatch",
    "freeze_structured_output",
    "parse_structured_output",
]
