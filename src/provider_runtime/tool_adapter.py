"""Lower frozen portable tool plans to provider-runtime's native tool IR."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, NoReturn, assert_never

from llm_tools import (
    Discoverable,
    FrozenToolPlan,
    HostTable,
    Native,
    ToolId,
    canonical_json_bytes,
    published_tool_ids,
)

from provider_runtime.types import CanonicalTool, ToolCall

_WIRE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_MAX_REJECTED_NAME_BYTES = 64
_MAX_JSON_DEPTH = 64

__all__ = [
    "CanonicalToolCall",
    "InvalidArgumentsReason",
    "PublishedTools",
    "RejectedToolArguments",
    "RejectedToolCall",
    "ToolCallResolution",
    "ToolPublication",
    "lower_tools",
]


class _FrozenJsonObject(dict[str, object]):
    """A provider-serializable dict with no mutation surface."""

    def __new__(
        cls,
        value: Mapping[str, object],
    ) -> _FrozenJsonObject:
        instance = dict.__new__(cls)
        dict.update(instance, value)
        return instance

    def __init__(self, value: Mapping[str, object]) -> None:
        pass

    def __copy__(self) -> _FrozenJsonObject:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenJsonObject:
        memo[id(self)] = self
        return self

    def __setitem__(self, key: str, value: object) -> NoReturn:
        raise TypeError("frozen JSON objects cannot be mutated")

    def __delitem__(self, key: str) -> NoReturn:
        raise TypeError("frozen JSON objects cannot be mutated")

    def __ior__(self, value: object) -> NoReturn:
        raise TypeError("frozen JSON objects cannot be mutated")

    def clear(self) -> NoReturn:
        raise TypeError("frozen JSON objects cannot be mutated")

    def pop(self, key: str, default: object = None) -> NoReturn:
        raise TypeError("frozen JSON objects cannot be mutated")

    def popitem(self) -> NoReturn:
        raise TypeError("frozen JSON objects cannot be mutated")

    def setdefault(self, key: str, default: object = None) -> NoReturn:
        raise TypeError("frozen JSON objects cannot be mutated")

    def update(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("frozen JSON objects cannot be mutated")


@dataclass(frozen=True, slots=True)
class ToolPublication:
    """The complete request-scoped publication input."""

    plan: FrozenToolPlan
    revealed_targets: tuple[ToolId, ...]

    def __post_init__(self) -> None:
        targets = tuple(self.revealed_targets)
        if len(set(targets)) != len(targets):
            raise ValueError("revealed targets must be unique")
        object.__setattr__(self, "revealed_targets", targets)


@dataclass(frozen=True, slots=True)
class CanonicalToolCall:
    """A provider call narrowed back to executable canonical identity."""

    provider_call_id: str
    tool_id: ToolId
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_arguments(self.arguments)
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class RejectedToolCall:
    """A non-executable provider name retained only for bounded audit."""

    provider_call_id: str
    raw_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_name", _bounded_utf8(self.raw_name))


type InvalidArgumentsReason = Literal["InvalidJson", "InputTooLarge"]


@dataclass(frozen=True, slots=True)
class RejectedToolArguments:
    """Known canonical tool with arguments rejected before host invocation."""

    provider_call_id: str
    tool_id: ToolId
    reason: InvalidArgumentsReason


type ToolCallResolution = CanonicalToolCall | RejectedToolCall | RejectedToolArguments


@dataclass(frozen=True, slots=True)
class PublishedTools:
    """One immutable native publication and its request-local reverse index."""

    tools: tuple[CanonicalTool, ...]
    _canonical_by_wire_name: Mapping[str, ToolId] = field(repr=False)
    _max_input_bytes_by_tool_id: Mapping[ToolId, int] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(
            self,
            "_canonical_by_wire_name",
            MappingProxyType(dict(self._canonical_by_wire_name)),
        )
        object.__setattr__(
            self,
            "_max_input_bytes_by_tool_id",
            MappingProxyType(dict(self._max_input_bytes_by_tool_id)),
        )

    def decode_tool_call(self, call: ToolCall) -> ToolCallResolution:
        tool_id = self._canonical_by_wire_name.get(call.name)
        if tool_id is None:
            return RejectedToolCall(
                provider_call_id=call.id,
                raw_name=call.name,
            )
        try:
            _validate_arguments(call.arguments)
            input_bytes = len(canonical_json_bytes({"type": "ParsedJson", "value": call.arguments}))
        except (TypeError, ValueError):
            return RejectedToolArguments(
                provider_call_id=call.id,
                tool_id=tool_id,
                reason="InvalidJson",
            )
        if input_bytes > self._max_input_bytes_by_tool_id[tool_id]:
            return RejectedToolArguments(
                provider_call_id=call.id,
                tool_id=tool_id,
                reason="InputTooLarge",
            )
        arguments = _freeze_mapping(call.arguments)
        return CanonicalToolCall(
            provider_call_id=call.id,
            tool_id=tool_id,
            arguments=arguments,
        )


def lower_tools(publication: ToolPublication) -> PublishedTools:
    """Publish exactly the tools selected by one frozen plan."""
    plan = publication.plan
    tool_ids = _publication_tool_ids(publication)

    tools: list[CanonicalTool] = []
    reverse: dict[str, ToolId] = {}
    max_input_bytes: dict[ToolId, int] = {}
    for tool_id in tool_ids:
        spec = plan.catalog_view.spec(tool_id)
        binding = plan.catalog_view.binding(tool_id)
        grant = plan.grant(tool_id)
        if spec.tool_contract_revision != grant.tool_contract_revision:
            raise ValueError(f"stale frozen tool contract for {tool_id!s}")
        if binding.policy_revision != grant.policy_revision:
            raise ValueError(f"stale frozen binding policy for {tool_id!s}")

        wire_name = _wire_name(tool_id)
        if wire_name in reverse:
            raise ValueError(f"provider tool-name collision: {wire_name!r}")
        reverse[wire_name] = tool_id
        max_input_bytes[tool_id] = grant.limits.max_input_bytes
        tools.append(
            CanonicalTool(
                name=wire_name,
                description=spec.documentation.text,
                parameters=_freeze_mapping(spec.input_schema.presentation),
            )
        )

    return PublishedTools(
        tools=tuple(tools),
        _canonical_by_wire_name=MappingProxyType(reverse),
        _max_input_bytes_by_tool_id=MappingProxyType(max_input_bytes),
    )


def _publication_tool_ids(publication: ToolPublication) -> tuple[ToolId, ...]:
    plan = publication.plan
    match plan.exposure:
        case Native():
            if publication.revealed_targets:
                raise ValueError("Native publication cannot carry revealed targets")
            return tuple(grant.id for grant in plan.profile.ordered_grants)
        case Discoverable():
            return published_tool_ids(plan, publication.revealed_targets)
        case HostTable():
            raise ValueError("HostTable plans are never provider-published")
        case other:
            assert_never(other)


def _wire_name(tool_id: ToolId) -> str:
    canonical = str(tool_id)
    try:
        ToolId(canonical)
    except ValueError as exc:
        raise ValueError(f"invalid canonical tool id: {canonical!r}") from exc
    alias = canonical.replace(".", "__")
    if len(alias.encode("ascii")) > 64:
        raise ValueError(f"provider tool name exceeds 64 bytes: {canonical!r}")
    if not _WIRE_NAME.fullmatch(alias):
        raise ValueError(f"provider tool name violates common grammar: {alias!r}")
    return alias


def _bounded_utf8(value: str) -> str:
    encoded = value.encode("utf-8", errors="replace")[:_MAX_REJECTED_NAME_BYTES]
    return encoded.decode("utf-8", errors="ignore")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return _FrozenJsonObject({key: _freeze_json(child) for key, child in value.items()})


def _validate_arguments(value: Mapping[str, object]) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_JSON_DEPTH:
            raise ValueError("JSON exceeds the maximum nesting depth")
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                raise TypeError("JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list | tuple):
            stack.extend((child, depth + 1) for child in current)
        elif isinstance(current, float) and not math.isfinite(current):
            raise TypeError("JSON numbers must be finite")
        elif current is not None and not isinstance(current, bool | int | float | str):
            raise TypeError(f"value is not JSON: {type(current).__name__}")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(child) for child in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("JSON numbers must be finite")
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"value is not JSON: {type(value).__name__}")
