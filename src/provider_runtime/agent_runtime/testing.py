"""No-network agent-runtime doubles for application tests.

``NoNetworkAgentRuntime`` fails loudly on every operation that could reach an
agent backend. ``ScriptedAgentRuntime`` replays queued public values and
records reference-safe calls without retaining approval handlers, cancellation
signals, or resolved credential material.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from provider_runtime.types import CancelSignal

from .events import AgentEvent, AgentTerminal
from .sessions import (
    AgentSession,
    SessionPage,
    SessionQuery,
    SessionReadOptions,
    SessionSnapshot,
)
from .types import (
    AgentSessionRef,
    AgentSessionRequest,
    ApprovalHandler,
    TurnRequest,
)

type AgentRuntimeOperation = Literal[
    "list_sessions",
    "read_session",
    "open_session",
    "run_turn",
    "stream_turn",
]
type CapturedAgentSubject = (
    SessionQuery
    | tuple[AgentSessionRef, SessionReadOptions]
    | AgentSessionRequest
    | tuple[AgentSession, TurnRequest]
)


@dataclass(frozen=True, slots=True)
class CapturedAgentCall:
    operation: AgentRuntimeOperation
    subject: CapturedAgentSubject
    approvals_supplied: bool = False
    cancel_supplied: bool = False


def _unexpected(operation: AgentRuntimeOperation, subject: object) -> str:
    route = "unknown"
    if isinstance(subject, SessionQuery):
        route = f"{subject.backend}/{subject.transport}"
    elif isinstance(subject, AgentSessionRequest):
        route = f"{subject.backend}/{subject.transport}"
    elif isinstance(subject, tuple) and subject:
        first = subject[0]
        if isinstance(first, AgentSessionRef):
            route = f"{first.backend}/{first.transport}"
        elif isinstance(first, AgentSession) and first.ref_is_complete:
            route = f"{first.ref.backend}/{first.ref.transport}"
    return f"Unexpected agent-runtime {operation} in test: {route}"


class NoNetworkAgentRuntime:
    """Runtime double that fails on any attempted agent operation."""

    async def __aenter__(self) -> NoNetworkAgentRuntime:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def list_sessions(self, query: SessionQuery) -> SessionPage:
        raise AssertionError(_unexpected("list_sessions", query))

    async def read_session(
        self, ref: AgentSessionRef, options: SessionReadOptions
    ) -> SessionSnapshot:
        raise AssertionError(_unexpected("read_session", (ref, options)))

    async def open_session(self, request: AgentSessionRequest) -> AgentSession:
        raise AssertionError(_unexpected("open_session", request))

    async def run_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None = None,
        cancel: CancelSignal | None = None,
    ) -> AgentTerminal:
        del approvals, cancel
        raise AssertionError(_unexpected("run_turn", (session, request)))

    async def stream_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None = None,
        cancel: CancelSignal | None = None,
    ) -> AsyncIterator[AgentEvent]:
        del approvals, cancel
        raise AssertionError(_unexpected("stream_turn", (session, request)))
        yield  # pragma: no cover - preserves the async-iterator port

    async def close(self) -> None:
        """Close the structural runtime double; no external resource exists."""


@dataclass(slots=True)
class _Scripts:
    pages: deque[SessionPage]
    snapshots: deque[SessionSnapshot]
    sessions: deque[AgentSession]
    streams: deque[tuple[AgentEvent, ...]]


def _validated_script(script: Sequence[AgentEvent]) -> tuple[AgentEvent, ...]:
    events = tuple(script)
    if not events:
        raise AssertionError("Scripted agent-runtime stream must not be empty")
    for index, event in enumerate(events):
        if isinstance(event, AgentTerminal) and index != len(events) - 1:
            raise AssertionError("Scripted agent-runtime stream has events after terminal")
    if not isinstance(events[-1], AgentTerminal):
        raise AssertionError("Scripted agent-runtime stream must end with AgentTerminal")
    return events


class ScriptedAgentRuntime(NoNetworkAgentRuntime):
    """Queued, deterministic implementation of the public agent-runtime ports."""

    def __init__(
        self,
        *,
        session_pages: Iterable[SessionPage] = (),
        session_snapshots: Iterable[SessionSnapshot] = (),
        sessions: Iterable[AgentSession] = (),
        stream_scripts: Iterable[Sequence[AgentEvent]] = (),
    ) -> None:
        self.calls: list[CapturedAgentCall] = []
        self._scripts = _Scripts(
            pages=deque(session_pages),
            snapshots=deque(session_snapshots),
            sessions=deque(sessions),
            streams=deque(_validated_script(script) for script in stream_scripts),
        )

    async def list_sessions(self, query: SessionQuery) -> SessionPage:
        self.calls.append(CapturedAgentCall("list_sessions", query))
        return _pop(self._scripts.pages, "list_sessions")

    async def read_session(
        self, ref: AgentSessionRef, options: SessionReadOptions
    ) -> SessionSnapshot:
        self.calls.append(CapturedAgentCall("read_session", (ref, options)))
        return _pop(self._scripts.snapshots, "read_session")

    async def open_session(self, request: AgentSessionRequest) -> AgentSession:
        self.calls.append(CapturedAgentCall("open_session", request))
        return _pop(self._scripts.sessions, "open_session")

    async def run_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None = None,
        cancel: CancelSignal | None = None,
    ) -> AgentTerminal:
        subject = (session, request)
        self.calls.append(
            CapturedAgentCall(
                "run_turn",
                subject,
                approvals_supplied=approvals is not None,
                cancel_supplied=cancel is not None,
            )
        )
        terminal: AgentEvent | None = None
        async for event in self._replay_stream(session):
            terminal = event
        if not isinstance(terminal, AgentTerminal):  # guarded by script validation
            raise AssertionError("Scripted agent-runtime stream had no terminal event")
        return terminal

    async def stream_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None = None,
        cancel: CancelSignal | None = None,
    ) -> AsyncIterator[AgentEvent]:
        subject = (session, request)
        self.calls.append(
            CapturedAgentCall(
                "stream_turn",
                subject,
                approvals_supplied=approvals is not None,
                cancel_supplied=cancel is not None,
            )
        )
        async for event in self._replay_stream(session):
            yield event

    async def _replay_stream(self, session: AgentSession) -> AsyncIterator[AgentEvent]:
        script = _pop(self._scripts.streams, "stream_turn")
        terminal = script[-1]
        assert isinstance(terminal, AgentTerminal)
        if not session.ref_is_complete:
            session.complete_ref(terminal.session_ref)
        elif session.ref != terminal.session_ref:
            raise AssertionError("Scripted agent-runtime terminal ref differs from session.ref")
        session.begin_turn()
        try:
            for event in script:
                yield event
        finally:
            session.end_turn()


def _pop[T](queue: deque[T], operation: AgentRuntimeOperation) -> T:
    try:
        return queue.popleft()
    except IndexError:
        raise AssertionError(f"No scripted agent-runtime {operation} value queued") from None


__all__ = [
    "AgentRuntimeOperation",
    "CapturedAgentCall",
    "NoNetworkAgentRuntime",
    "ScriptedAgentRuntime",
]
