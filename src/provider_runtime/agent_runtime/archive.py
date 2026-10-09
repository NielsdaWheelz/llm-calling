"""Read-only native conversation evidence. This boundary grants no execution."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from .errors import AgentRuntimeError, InvalidAgentRequest

ARCHIVE_CONTRACT_REVISION = "native-archive.v1"
INTERNAL_THREAD_SOURCE = "jarvis.cognition"
MEMORY_TOOL_NAMES = frozenset(
    (
        "memory_view",
        "memory_zoom",
        "memory_date",
        "memory_search",
        "memory_open",
        "memory_save_note",
    )
)

type ArchiveProvider = Literal["codex", "claude"]
type ArchiveRole = Literal["owner", "assistant", "peer", "tool", "host", "unknown"]
type ArchiveKind = Literal[
    "text",
    "tool_call",
    "tool_result",
    "attachment",
    "context",
    "memory_reference",
    "gap",
    "revision",
]
type ArchiveErrorCode = Literal[
    "unavailable",
    "unsupported",
    "event_too_large",
    "history_changed",
    "activation_boundary_lost",
    "source_conflict",
]


class ArchiveError(AgentRuntimeError):
    """Content-free capture failure; absence never masquerades as empty evidence."""

    def __init__(self, code: ArchiveErrorCode) -> None:
        super().__init__(f"native archive {code}", code=code)


class ArchiveMissing(ArchiveError):
    """A direct native observation proved absence, rather than a transport failure."""

    def __init__(self) -> None:
        super().__init__("unavailable")


@dataclass(frozen=True, slots=True)
class ArchiveHome:
    provider: ArchiveProvider
    state_root: Path
    codex_endpoint: Path | None = None
    memory_server: str = "jarvis-memory"

    def __post_init__(self) -> None:
        if (
            self.provider not in ("codex", "claude")
            or not isinstance(self.state_root, Path)
            or not self.state_root.is_absolute()
        ):
            raise InvalidAgentRequest("archive home requires a provider and absolute state root")
        if self.provider == "codex" and (
            self.codex_endpoint is None or not self.codex_endpoint.is_absolute()
        ):
            raise InvalidAgentRequest("Codex archive requires an absolute app-server endpoint")
        if self.provider == "claude" and self.codex_endpoint is not None:
            raise InvalidAgentRequest("Claude archive reads transcripts, not a Codex endpoint")
        if not self.memory_server or len(self.memory_server.encode("utf-8")) > 256:
            raise InvalidAgentRequest("archive memory server binding must be bounded and nonempty")


@dataclass(frozen=True, slots=True)
class ArchiveCapabilities:
    provider_version: str
    enumerate: bool
    read: bool
    internal_marking: bool
    memory_tool_recognition: bool
    child_result_recognition: bool


@dataclass(frozen=True, slots=True)
class ArchiveConversation:
    native_id: str
    created_at: datetime
    updated_at: datetime
    archived: bool
    internal: bool
    relation: Literal["child", "fork"] | None = None
    parent_native_id: str | None = None
    inherited_through: str | None = None
    working_directory: str | None = None
    head_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveInventory:
    conversations: tuple[ArchiveConversation, ...]
    next_page: str | None
    complete: bool
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class ArchiveEvent:
    native_event_id: str
    native_digest: str
    turn_id: str | None
    native_parent_id: str | None
    role: ArchiveRole
    kind: ArchiveKind
    text: str
    attributes: dict[str, object]
    occurred_at: datetime | None


@dataclass(frozen=True, slots=True)
class ArchiveRead:
    events: tuple[ArchiveEvent, ...]
    next: str | None
    caught_up: bool
    observed_at: datetime


def mark_internal_session(state_root: Path, native_id: str) -> None:
    """Record a launcher-chosen Claude identity before the process can persist it."""
    from uuid import UUID

    try:
        valid_id = type(native_id) is str and str(UUID(native_id)) == native_id
    except ValueError:
        valid_id = False
    if not isinstance(state_root, Path) or not state_root.is_absolute() or not valid_id:
        raise InvalidAgentRequest("internal archive marking requires an absolute root and UUID")
    directory = state_root / ".provider-runtime" / "archive-internal"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    (directory / native_id).touch(mode=0o600, exist_ok=True)


async def archive_capabilities(home: ArchiveHome) -> ArchiveCapabilities:
    if home.provider == "codex":
        from ._codex_archive import capabilities

        return await capabilities(home)
    from ._process import ProcessLimits, capture_process_output
    from .errors import ExecutableUnavailable, ProtocolDefect

    executable = shutil.which("claude")
    if executable is None:
        raise ArchiveError("unavailable")
    try:
        version = await capture_process_output(
            (executable, "--version"),
            cwd=home.state_root,
            environment={
                "PATH": os.environ.get("PATH", ""),
                "CLAUDE_CONFIG_DIR": str(home.state_root),
            },
            limits=ProcessLimits(max_stderr_bytes=1024, termination_grace_seconds=0.25),
            startup_timeout_seconds=10,
            max_stdout_bytes=1024,
            executable_label="Claude CLI",
            purpose="archive provider version",
        )
    except (ExecutableUnavailable, ProtocolDefect, OSError):
        raise ArchiveError("unavailable") from None
    return ArchiveCapabilities(version, True, True, True, True, True)


async def archive_list(
    home: ArchiveHome, page: str | None = None, *, with_heads: bool = False
) -> ArchiveInventory:
    """Assemble one bounded inventory; provider pagination stays inside the codec."""
    if page is not None or type(with_heads) is not bool:
        raise InvalidAgentRequest("archive inventories use one complete request")
    if home.provider == "codex":
        from ._codex_archive import inventory

        return await inventory(home, with_heads=with_heads)
    from ._claude_archive import inventory

    return await asyncio.to_thread(inventory, home, with_heads=with_heads)


async def archive_read(
    home: ArchiveHome, native_id: str, after: str | None = None, from_event_id: str | None = None
) -> ArchiveRead:
    if not native_id or len(native_id.encode("utf-8")) > 256:
        raise InvalidAgentRequest("native conversation identity must be bounded and nonempty")
    if any(
        value is not None and (not value or len(value.encode("utf-8")) > 256)
        for value in (after, from_event_id)
    ):
        raise InvalidAgentRequest("archive boundary identities must be bounded and nonempty")
    if home.provider == "codex":
        from ._codex_archive import read

        return await read(home, native_id, after=after, checkpoint=from_event_id)
    from ._claude_archive import read

    return await asyncio.to_thread(read, home, native_id, after=after, checkpoint=from_event_id)
