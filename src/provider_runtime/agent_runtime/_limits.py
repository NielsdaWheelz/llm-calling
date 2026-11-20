"""Shared bounded-ingress primitives for native agent output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum

# Both adapters read the same untrusted native stream, so they answer to one set of bounds:
# a limit that differed between the two lanes would be an accident rather than a policy.
_OPERATION_TIMEOUT_SECONDS = 30.0
_MAX_EVENT_COUNT = 100_000
_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_MAX_MESSAGE_ITEMS = 100_000
_MAX_TURN_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_EVENT_TEXT_BYTES = 4 * 1024 * 1024
_MAX_FINAL_TEXT_BYTES = 16 * 1024 * 1024
_MAX_DIAGNOSTICS = 256


class OutputLimitExceeded(Exception):
    """A bounded agent stream exceeded its configured limit."""

    def __init__(self, limit: int, *, unit: str = "units", incoming: int | None = None) -> None:
        super().__init__(f"agent output exceeded its {limit}-unit limit")
        self.limit = limit
        self.unit = unit
        self.incoming = incoming


def bounded_payload_size(
    value: object,
    limit: int,
    *,
    max_items: int,
    max_depth: int = 32,
) -> int:
    """Measure an SDK value before any adapter materializes an owned JSON copy.

    Provider SDK objects are untrusted ingress.  Walking their already-materialized object
    graph lets adapters reject a pathological message before ``asdict``/``model_dump`` or an
    owned freezer duplicates it.  The item and depth bounds also cover structures whose byte
    representation is deceptively small but whose Python object graph is expensive to traverse.
    """

    total = 0
    items = 0
    active: set[int] = set()

    def visit(child: object, depth: int) -> None:
        nonlocal items, total
        if depth > max_depth:
            raise OutputLimitExceeded(max_depth, unit="depth", incoming=depth)
        items += 1
        if items > max_items:
            raise OutputLimitExceeded(max_items, unit="items", incoming=items)
        if child is None or type(child) in (bool, float):
            total += 8
        elif type(child) is int:
            if not -(2**63) <= child <= 2**63 - 1:
                raise OutputLimitExceeded(
                    64,
                    unit="signed_bits",
                    incoming=(child if child >= 0 else ~child).bit_length() + 1,
                )
            total += len(str(child))
        elif isinstance(child, Enum):
            visit(child.value, depth + 1)
        elif isinstance(child, str):
            if len(child) > limit:
                raise OutputLimitExceeded(limit, unit="bytes")
            total += len(child.encode("utf-8"))
        elif isinstance(child, Mapping):
            identity = id(child)
            if identity in active:
                raise OutputLimitExceeded(0, unit="cycles", incoming=1)
            active.add(identity)
            try:
                for key, item in child.items():
                    if not isinstance(key, str):
                        raise OutputLimitExceeded(limit)
                    visit(key, depth + 1)
                    visit(item, depth + 1)
            finally:
                active.remove(identity)
        elif isinstance(child, (tuple, list)):
            identity = id(child)
            if identity in active:
                raise OutputLimitExceeded(0, unit="cycles", incoming=1)
            active.add(identity)
            try:
                for item in child:
                    visit(item, depth + 1)
            finally:
                active.remove(identity)
        elif is_dataclass(child) and not isinstance(child, type):
            identity = id(child)
            if identity in active:
                raise OutputLimitExceeded(0, unit="cycles", incoming=1)
            active.add(identity)
            try:
                for item in fields(child):
                    visit(item.name, depth + 1)
                    visit(getattr(child, item.name), depth + 1)
            finally:
                active.remove(identity)
        elif hasattr(child, "__dict__"):
            visit(vars(child), depth + 1)
        else:
            total += len(type(child).__name__)
        if total > limit:
            raise OutputLimitExceeded(limit, unit="bytes")

    visit(value, 0)
    return total


__all__ = ["OutputLimitExceeded", "bounded_payload_size"]
