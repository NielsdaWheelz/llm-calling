"""Tool-call argument parsing shared by provider adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import json_repair

from provider_runtime.errors import ModelCallError, ModelCallErrorCode

ToolArgumentStatus = Literal["valid", "repaired"]


@dataclass(frozen=True)
class ParsedToolArguments:
    arguments: dict[str, object]
    status: ToolArgumentStatus


def parse_tool_arguments(
    raw: object,
    *,
    provider: str,
    tool_name: str = "",
    call_id: str = "",
) -> dict[str, object]:
    """Parse provider tool arguments and fail closed on malformed payloads."""
    return parse_tool_arguments_with_status(
        raw,
        provider=provider,
        tool_name=tool_name,
        call_id=call_id,
    ).arguments


def parse_tool_arguments_with_status(
    raw: object,
    *,
    provider: str,
    tool_name: str = "",
    call_id: str = "",
) -> ParsedToolArguments:
    """Parse provider tool arguments, repairing once when JSON text is malformed."""
    if raw is None or raw == "":
        return ParsedToolArguments({}, "valid")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            status: ToolArgumentStatus = "valid"
        except json.JSONDecodeError:
            try:
                parsed = json_repair.loads(raw)
            except Exception as repair_exc:
                raise _invalid(provider, tool_name=tool_name, call_id=call_id) from repair_exc
            status = "repaired"
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
        status = "valid"
    else:
        raise _invalid(provider, tool_name=tool_name, call_id=call_id)

    if not isinstance(parsed, dict):
        raise _invalid(provider, tool_name=tool_name, call_id=call_id)
    return ParsedToolArguments(parsed, status)


def _invalid(provider: str, *, tool_name: str, call_id: str) -> ModelCallError:
    label = tool_name or call_id or "tool call"
    return ModelCallError(
        ModelCallErrorCode.TOOL_ARGUMENTS_INVALID,
        f"{provider} returned malformed arguments for {label}",
        provider=provider,
        retryable=False,
    )
