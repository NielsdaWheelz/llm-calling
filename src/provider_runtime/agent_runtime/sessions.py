"""Session identity, discovery values, and one-active-turn state."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .capabilities import AgentCapabilityScope
from .errors import (
    ConcurrentTurn,
    InvalidAgentRequest,
    ProtocolDefect,
    SessionMismatch,
    SessionUnavailable,
)
from .types import AgentSessionRef, CredentialRef

if TYPE_CHECKING:
    from .events import AgentEvent


def fingerprint_path(path: str | Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        raise InvalidAgentRequest("fingerprinted paths must be absolute")
    return hashlib.sha256(str(value.resolve()).encode("utf-8")).hexdigest()


def validate_session_ref(
    ref: AgentSessionRef,
    scope: AgentCapabilityScope,
    *,
    state_root_fingerprint: str,
    cwd: str | Path,
    cwd_scopes_sessions: bool,
) -> None:
    """Reject caller-scope mismatches; native absence is adapter-owned."""
    if ref.backend != scope.backend:
        raise SessionMismatch("AgentSessionRef.backend does not match the requesting scope")
    if ref.transport != scope.transport:
        raise SessionMismatch("AgentSessionRef.transport does not match the requesting scope")
    if ref.profile_key != scope.auth.profile_key:
        raise SessionMismatch("AgentSessionRef.profile_key does not match the requesting scope")
    if ref.state_root_fingerprint != state_root_fingerprint:
        raise SessionMismatch(
            "AgentSessionRef.state_root_fingerprint does not match the requesting scope"
        )
    if cwd_scopes_sessions and ref.cwd_fingerprint != fingerprint_path(cwd):
        raise SessionMismatch("AgentSessionRef.cwd_fingerprint does not match the requesting scope")


@dataclass(frozen=True, slots=True)
class CodexSessionFilters:
    archived: bool | None = None

    def __post_init__(self) -> None:
        if self.archived is not None and type(self.archived) is not bool:
            raise InvalidAgentRequest("CodexSessionFilters.archived must be bool when present")


@dataclass(frozen=True, slots=True)
class ClaudeSessionFilters:
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_string_tuple(self.tags, "ClaudeSessionFilters.tags")


type SessionFilters = CodexSessionFilters | ClaudeSessionFilters


@dataclass(frozen=True, slots=True)
class SessionQuery:
    scope: AgentCapabilityScope
    cursor: str | None = None
    limit: int = 50
    native: SessionFilters | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, AgentCapabilityScope):
            raise InvalidAgentRequest("SessionQuery.scope must be AgentCapabilityScope")
        _validate_page(self.cursor, self.limit, "SessionQuery")
        if self.native is not None:
            if self.scope.backend == "codex" and not isinstance(self.native, CodexSessionFilters):
                raise InvalidAgentRequest("codex session queries require CodexSessionFilters")
            if self.scope.backend == "claude" and not isinstance(self.native, ClaudeSessionFilters):
                raise InvalidAgentRequest("claude session queries require ClaudeSessionFilters")


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    name: str | None = None
    archived: bool | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name is not None and (type(self.name) is not str or not self.name):
            raise InvalidAgentRequest("SessionMetadata.name must be non-empty when present")
        if self.archived is not None and type(self.archived) is not bool:
            raise InvalidAgentRequest("SessionMetadata.archived must be bool when present")
        _validate_string_tuple(self.tags, "SessionMetadata.tags")


@dataclass(frozen=True, slots=True)
class SessionSummary:
    ref: AgentSessionRef
    metadata: SessionMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.ref, AgentSessionRef):
            raise InvalidAgentRequest("SessionSummary.ref must be AgentSessionRef")
        if not isinstance(self.metadata, SessionMetadata):
            raise InvalidAgentRequest("SessionSummary.metadata must be SessionMetadata")


@dataclass(frozen=True, slots=True)
class SessionPage:
    sessions: tuple[SessionSummary, ...]
    continuation_cursor: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sessions, tuple) or any(
            not isinstance(item, SessionSummary) for item in self.sessions
        ):
            raise InvalidAgentRequest("SessionPage.sessions must be a tuple of SessionSummary")
        _validate_cursor(self.continuation_cursor, "SessionPage.continuation_cursor")


@dataclass(frozen=True, slots=True)
class SessionReadOptions:
    auth: CredentialRef
    cursor: str | None = None
    limit: int = 50
    include_turns: bool = True
    include_items: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.auth, CredentialRef):
            raise InvalidAgentRequest("SessionReadOptions.auth must be CredentialRef")
        _validate_page(self.cursor, self.limit, "SessionReadOptions")
        if type(self.include_turns) is not bool or type(self.include_items) is not bool:
            raise InvalidAgentRequest("SessionReadOptions include flags must be bool")
        if self.include_items and not self.include_turns:
            raise InvalidAgentRequest("SessionReadOptions.include_items requires include_turns")


def validate_read_session_auth(ref: AgentSessionRef, options: SessionReadOptions) -> None:
    if ref.profile_key != options.auth.profile_key:
        raise SessionMismatch(
            "read-session auth profile does not match AgentSessionRef.profile_key"
        )


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    ref: AgentSessionRef
    metadata: SessionMetadata
    items: tuple[AgentEvent, ...]
    continuation_cursor: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, AgentSessionRef):
            raise InvalidAgentRequest("SessionSnapshot.ref must be AgentSessionRef")
        if not isinstance(self.metadata, SessionMetadata):
            raise InvalidAgentRequest("SessionSnapshot.metadata must be SessionMetadata")
        if not isinstance(self.items, tuple):
            raise InvalidAgentRequest("SessionSnapshot.items must be a tuple of AgentEvent")
        if self.items:
            from .events import AgentEvent

            if any(not isinstance(item, AgentEvent) for item in self.items):
                raise InvalidAgentRequest("SessionSnapshot.items must contain AgentEvent values")
        _validate_cursor(self.continuation_cursor, "SessionSnapshot.continuation_cursor")


def _validate_string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise InvalidAgentRequest(f"{field_name} must be a tuple")
    if any(type(item) is not str or not item for item in value):
        raise InvalidAgentRequest(f"{field_name} entries must be non-empty strings")
    if len(value) != len(set(value)):
        raise InvalidAgentRequest(f"{field_name} must not contain duplicates")


def _validate_cursor(cursor: object, field_name: str) -> None:
    if cursor is not None and (type(cursor) is not str or not cursor):
        raise InvalidAgentRequest(f"{field_name} must be non-empty when present")


def _validate_page(cursor: object, limit: object, owner: str) -> None:
    _validate_cursor(cursor, f"{owner}.cursor")
    if type(limit) is not int or limit <= 0:
        raise InvalidAgentRequest(f"{owner}.limit must be a positive integer")


class AgentSession:
    """A native session handle with lazy ref completion and turn exclusion."""

    __slots__ = ("_ref", "_state_lock", "_turn_active", "_closed", "__weakref__")

    def __init__(self, ref: AgentSessionRef | None = None) -> None:
        if ref is not None and not isinstance(ref, AgentSessionRef):
            raise InvalidAgentRequest("AgentSession ref must be AgentSessionRef when present")
        self._ref = ref
        self._state_lock = threading.Lock()
        self._turn_active = False
        self._closed = False

    @property
    def ref_is_complete(self) -> bool:
        with self._state_lock:
            return self._ref is not None

    @property
    def ref(self) -> AgentSessionRef:
        with self._state_lock:
            ref = self._ref
        if ref is None:
            raise ProtocolDefect(
                "AgentSession ref is not complete before session_started",
                code="incomplete_session_ref",
            )
        return ref

    def complete_ref(self, ref: AgentSessionRef) -> None:
        if not isinstance(ref, AgentSessionRef):
            raise ProtocolDefect("session_started carried an invalid AgentSessionRef")
        with self._state_lock:
            if self._ref is not None:
                raise ProtocolDefect(
                    "AgentSession ref was completed more than once",
                    code="duplicate_session_ref",
                )
            self._ref = ref

    def begin_turn(self) -> None:
        with self._state_lock:
            if self._closed:
                raise SessionUnavailable("AgentSession has been closed")
            if self._turn_active:
                raise ConcurrentTurn()
            self._turn_active = True

    def begin_close(self) -> bool:
        """Atomically close the handle and report whether a turn was active."""
        with self._state_lock:
            if self._closed:
                return self._turn_active
            self._closed = True
            return self._turn_active

    def end_turn(self) -> None:
        with self._state_lock:
            if not self._turn_active:
                raise ProtocolDefect(
                    "AgentSession.end_turn called without an active turn",
                    code="inactive_turn_end",
                )
            self._turn_active = False
