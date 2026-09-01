"""Lower a frozen canonical tool plan to Codex MCP publication and observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from llm_tools import FrozenToolPlan, ToolId

from provider_runtime.tool_adapter import (
    ToolPublication,
    _publication_tool_ids,
    _wire_name,
)
from provider_runtime.types import JsonValue

from .events import AgentToolUse
from .types import CredentialRef, HeaderReference, McpServerSpec


@dataclass(frozen=True, slots=True)
class McpToolPublication:
    plan: FrozenToolPlan
    server_name: str
    url: str
    bearer: CredentialRef
    revealed_targets: tuple[ToolId, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.plan, FrozenToolPlan):
            raise ValueError("McpToolPublication.plan must be FrozenToolPlan")
        if not isinstance(self.bearer, CredentialRef) or self.bearer.kind == "local_account":
            raise ValueError("McpToolPublication.bearer must be a named credential reference")
        targets = tuple(self.revealed_targets)
        if len(set(targets)) != len(targets):
            raise ValueError("McpToolPublication.revealed_targets must be unique")
        object.__setattr__(self, "revealed_targets", targets)


@dataclass(frozen=True, slots=True)
class CanonicalMcpToolObservation:
    tool_call_id: str
    tool_id: ToolId
    phase: Literal["started", "updated", "completed"]
    payload: JsonValue
    succeeded: bool | None


@dataclass(frozen=True, slots=True)
class RejectedMcpToolObservation:
    tool_call_id: str
    raw_name: str


type McpToolObservation = CanonicalMcpToolObservation | RejectedMcpToolObservation


@dataclass(frozen=True, slots=True)
class PublishedMcpTools:
    server: McpServerSpec
    _canonical_by_observed_name: Mapping[str, ToolId] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_canonical_by_observed_name",
            MappingProxyType(dict(self._canonical_by_observed_name)),
        )

    def observe(self, event: AgentToolUse) -> McpToolObservation:
        tool_id = self._canonical_by_observed_name.get(event.name)
        if tool_id is None:
            return RejectedMcpToolObservation(
                tool_call_id=event.tool_call_id,
                raw_name=_bounded_utf8(event.name),
            )
        return CanonicalMcpToolObservation(
            tool_call_id=event.tool_call_id,
            tool_id=tool_id,
            phase=event.phase,
            payload=event.payload,
            succeeded=event.succeeded,
        )


def lower_mcp_tools(publication: McpToolPublication) -> PublishedMcpTools:
    """Publish exactly one plan's tools through one authenticated HTTPS MCP server."""
    common = ToolPublication(
        plan=publication.plan,
        revealed_targets=publication.revealed_targets,
    )
    tool_ids = _publication_tool_ids(common)
    aliases = tuple(_wire_name(tool_id) for tool_id in tool_ids)
    if len(set(aliases)) != len(aliases):
        raise ValueError("MCP tool-name collision")
    server = McpServerSpec(
        name=publication.server_name,
        transport="streamable_http",
        url=publication.url,
        header_refs=(HeaderReference(name="Authorization", source=publication.bearer),),
        required=True,
        allowed_tools=aliases,
    )
    return PublishedMcpTools(
        server=server,
        _canonical_by_observed_name=MappingProxyType(
            {
                f"{publication.server_name}/{alias}": tool_id
                for alias, tool_id in zip(aliases, tool_ids, strict=True)
            }
        ),
    )


def _bounded_utf8(value: str) -> str:
    encoded = value.encode("utf-8", errors="replace")[:64]
    return encoded.decode("utf-8", errors="ignore")


__all__ = [
    "CanonicalMcpToolObservation",
    "McpToolObservation",
    "McpToolPublication",
    "PublishedMcpTools",
    "RejectedMcpToolObservation",
    "lower_mcp_tools",
]
