"""Value rules shared by the two modules that validate request inputs.

`types.py` imports `policy.py`, so a rule both need can be owned by neither.
"""

from __future__ import annotations

import re
from typing import cast

from .errors import InvalidAgentRequest

ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def require_tuple(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise InvalidAgentRequest(f"{field_name} must be a tuple")
    return value


def require_unique_strings(value: object, field_name: str) -> tuple[str, ...]:
    items = require_tuple(value, field_name)
    if any(type(item) is not str or not item for item in items):
        raise InvalidAgentRequest(f"{field_name} entries must be non-empty strings")
    if len(items) != len(set(items)):
        raise InvalidAgentRequest(f"{field_name} must not contain duplicate entries")
    return cast(tuple[str, ...], items)


__all__ = ["ENVIRONMENT_NAME", "require_tuple", "require_unique_strings"]
