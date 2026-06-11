"""Tool-call argument parsing shared by provider adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping

from provider_runtime.errors import ModelCallError, ModelCallErrorCode


def parse_tool_arguments(
    raw: object,
    *,
    provider: str,
    tool_name: str = "",
    call_id: str = "",
) -> dict[str, object]:
    """Parse provider tool arguments and fail closed on malformed payloads."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _invalid(provider, tool_name=tool_name, call_id=call_id) from exc
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        raise _invalid(provider, tool_name=tool_name, call_id=call_id)

    if not isinstance(parsed, dict):
        raise _invalid(provider, tool_name=tool_name, call_id=call_id)
    return parsed


def _invalid(provider: str, *, tool_name: str, call_id: str) -> ModelCallError:
    label = tool_name or call_id or "tool call"
    return ModelCallError(
        ModelCallErrorCode.TOOL_ARGUMENTS_INVALID,
        f"{provider} returned malformed arguments for {label}",
        provider=provider,
        retryable=False,
    )
