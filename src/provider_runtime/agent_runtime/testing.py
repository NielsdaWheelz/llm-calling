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

from provider_runtime.agent_runtime.capabilities import (
    AgentCapabilities,
    AgentCapabilityScope,
)
from provider_runtime.agent_runtime.events import (
    AgentEvent,
    terminal_event_to_result,
)
from provider_runtime.agent_runtime.sessions import (
    AgentSession,
    SessionPage,
    SessionQuery,
    SessionReadOptions,
    SessionSnapshot,
)
from provider_runtime.agent_runtime.types import (
    AgentResult,
    AgentSessionRef,
    AgentSessionRequest,
    ApprovalHandler,
    TurnRequest,
)
from provider_runtime.types import CancelSignal

type AgentRuntimeOperation = Literal[
    "capabilities",
    "list_sessions",
    "read_session",
    "open_session",
    "run_turn",
    "stream_turn",
]
type CapturedAgentSubject = (
    AgentCapabilityScope
    | SessionQuery
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
    if isinstance(subject, AgentCapabilityScope):
        route = f"{subject.backend}/{subject.transport}"
    elif isinstance(subject, SessionQuery):
        route = f"{subject.scope.backend}/{subject.scope.transport}"
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

    async def capabilities(self, scope: AgentCapabilityScope) -> AgentCapabilities:
        raise AssertionError(_unexpected("capabilities", scope))

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
    ) -> AgentResult:
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
    capabilities: deque[AgentCapabilities]
    pages: deque[SessionPage]
    snapshots: deque[SessionSnapshot]
    sessions: deque[AgentSession]
    streams: deque[tuple[AgentEvent, ...]]


def _validated_script(script: Sequence[AgentEvent]) -> tuple[AgentEvent, ...]:
    events = tuple(script)
    if not events:
        raise AssertionError("Scripted agent-runtime stream must not be empty")
    session_ref = events[0].session_ref
    turn_id = events[0].turn_id
    turn_started = False
    for expected_seq, event in enumerate(events, start=1):
        if event.seq != expected_seq:
            raise AssertionError("Scripted agent-runtime stream sequence must be gap-free")
        if event.session_ref != session_ref or event.turn_id != turn_id:
            raise AssertionError("Scripted agent-runtime stream identity must not change")
        if event.kind == "session_started":
            if expected_seq != 1:
                raise AssertionError("Scripted session_started must be the first event")
        elif event.kind == "turn_started":
            if turn_started:
                raise AssertionError("Scripted agent-runtime stream has two turn_started events")
            turn_started = True
        elif not turn_started:
            raise AssertionError("Scripted agent-runtime content requires turn_started first")
        if event.kind in ("turn_completed", "turn_failed", "turn_cancelled"):
            if expected_seq != len(events):
                raise AssertionError("Scripted agent-runtime stream has events after terminal")
    if not turn_started:
        raise AssertionError("Scripted agent-runtime stream requires turn_started")
    if events[-1].kind not in ("turn_completed", "turn_failed", "turn_cancelled"):
        raise AssertionError("Scripted agent-runtime stream must end with a terminal event")
    terminal_event_to_result(events[-1])
    return events


class ScriptedAgentRuntime(NoNetworkAgentRuntime):
    """Queued, deterministic implementation of the public agent-runtime ports."""

    def __init__(
        self,
        *,
        capabilities: Iterable[AgentCapabilities] = (),
        session_pages: Iterable[SessionPage] = (),
        session_snapshots: Iterable[SessionSnapshot] = (),
        sessions: Iterable[AgentSession] = (),
        stream_scripts: Iterable[Sequence[AgentEvent]] = (),
    ) -> None:
        self.calls: list[CapturedAgentCall] = []
        self._scripts = _Scripts(
            capabilities=deque(capabilities),
            pages=deque(session_pages),
            snapshots=deque(session_snapshots),
            sessions=deque(sessions),
            streams=deque(_validated_script(script) for script in stream_scripts),
        )
        self._started_sessions: set[AgentSession] = set()

    async def capabilities(self, scope: AgentCapabilityScope) -> AgentCapabilities:
        self.calls.append(CapturedAgentCall("capabilities", scope))
        return _pop(self._scripts.capabilities, "capabilities")

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
    ) -> AgentResult:
        subject = (session, request)
        self.calls.append(
            CapturedAgentCall(
                "run_turn",
                subject,
                approvals_supplied=approvals is not None,
                cancel_supplied=cancel is not None,
            )
        )
        terminal = None
        async for event in self._replay_stream(session):
            terminal = event
        if terminal is None:  # guarded by script validation
            raise AssertionError("Scripted agent-runtime stream had no terminal event")
        return terminal_event_to_result(terminal)

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
        first_stream = session not in self._started_sessions
        has_session_started = script[0].kind == "session_started"
        if first_stream != has_session_started:
            expected = "requires" if first_stream else "must not repeat"
            raise AssertionError(f"Scripted agent-runtime session {expected} session_started")
        first_ref = script[0].session_ref
        if session.ref_is_complete:
            if session.ref != first_ref:
                raise AssertionError("Scripted agent-runtime event ref differs from session.ref")
        elif has_session_started:
            session.complete_ref(first_ref)
        else:  # guarded by first-stream grammar, kept explicit for future changes
            raise AssertionError("Scripted agent-runtime session ref is incomplete")

        session.begin_turn()
        try:
            if first_stream:
                self._started_sessions.add(session)
            for event in script:
                if event.session_ref != session.ref:
                    raise AssertionError(
                        "Scripted agent-runtime event ref differs from session.ref"
                    )
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
