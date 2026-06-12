"""Structured-output parsing invariants shared by provider adapters."""

from __future__ import annotations

import json

from provider_runtime.errors import ModelCallError, ModelCallErrorCode


def parse_required_structured_output(
    text: str,
    *,
    provider: str,
) -> dict[str, object]:
    """Parse the provider text as the required structured-output object."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelCallError(
            ModelCallErrorCode.BAD_REQUEST,
            f"{provider} structured output was not valid JSON",
            provider=provider,
            retryable=False,
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelCallError(
            ModelCallErrorCode.BAD_REQUEST,
            f"{provider} structured output was not a JSON object",
            provider=provider,
            retryable=False,
        )
    return parsed
