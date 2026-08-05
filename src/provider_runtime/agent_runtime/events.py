"""Strict normalized agent events and terminal stream invariants."""

from __future__ import annotations

import math
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Literal, assert_never

from provider_runtime.agent_runtime.errors import (
    InvalidAgentRequest,
    MissingTerminalEvent,
    ProtocolDefect,
)
from provider_runtime.agent_runtime.types import (
    AGENT_FAILURE_CAUSES,
    AgentFailureCause,
    AgentResult,
    AgentSessionRef,
    AgentTransport,
    ApprovalDecision,
    ApprovalRequest,
    Backend,
    FrozenJsonDict,
    JsonObject,
    JsonValue,
    require_frozen_json,
)

type AgentEventKind = Literal[
    "session_started",
    "turn_started",
    "text_delta",
    "reasoning",
    "tool_started",
    "tool_updated",
    "tool_completed",
    "approval_requested",
    "approval_answered",
    "file_change",
    "usage",
    "diagnostic",
    "unknown",
    "native_retry_observed",
    "turn_completed",
    "turn_failed",
    "turn_cancelled",
]
type TerminalEventKind = Literal["turn_completed", "turn_failed", "turn_cancelled"]
type SessionScopeEventKind = Literal["diagnostic", "unknown"]

TERMINAL_EVENT_KINDS: tuple[TerminalEventKind, ...] = (
    "turn_completed",
    "turn_failed",
    "turn_cancelled",
)
SESSION_SCOPE_EVENT_KINDS: tuple[SessionScopeEventKind, ...] = ("diagnostic", "unknown")
"""The non-terminal kinds whose subject is the session rather than one turn.

`turn_started` orders *turn* content, so the two kinds that name no turn-scoped subject are
the two that may precede it. `DiagnosticData` carries a code, a message and detail about the
environment, the account or the session — SDKs may report configuration, remote-control,
MCP startup, or account-limit notifications in exactly this window with no turn identity.
`UnknownData` is empty by construction: an unrecognized notification
has no declared scope, and refusing it here would turn any additive backend protocol change
into a hard failure of a lane rather than the passthrough the event contract promises.

Every other non-terminal kind names something that only exists inside a turn, and each one
stays gated: `text_delta`/`reasoning` are the turn's output, `tool_started`/`tool_updated`/
`tool_completed`/`file_change`/`approval_requested`/`approval_answered` are its work, `usage`
is a per-turn backend report, and `native_retry_observed` counts retries of a turn. A backend
that reported any of those before opening a turn is misframing
its own protocol, which is a defect and must keep raising.
"""


def _non_empty(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ProtocolDefect(f"{field} must be a non-empty string")


def _owned_json(value: object, field: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ProtocolDefect(f"{field} numbers must be finite")
    if value is None or type(value) in (bool, int, float, str):
        return
    if isinstance(value, tuple):
        for item in value:
            _owned_json(item, field)
        return
    if isinstance(value, FrozenJsonDict):
        _validate_frozen_json(value, field)
        return
    raise ProtocolDefect(f"{field} must be immutable JSON data")


def _diagnostics(value: object, field: str) -> None:
    if not isinstance(value, tuple) or any(type(item) is not str or not item for item in value):
        raise ProtocolDefect(f"{field} must be a tuple of non-empty strings")
    if len(value) != len(set(value)):
        raise ProtocolDefect(f"{field} must not contain duplicates")


def _validate_frozen_json(value: object, field: str) -> None:
    try:
        require_frozen_json(value, field)
    except InvalidAgentRequest:
        raise ProtocolDefect(f"{field} must be recursively frozen JSON") from None


@dataclass(frozen=True, slots=True)
class SessionStartedData:
    pass


@dataclass(frozen=True, slots=True)
class TurnStartedData:
    pass


@dataclass(frozen=True, slots=True)
class TextDeltaData:
    text: str

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise ProtocolDefect("TextDeltaData.text must be a string")


@dataclass(frozen=True, slots=True)
class ReasoningData:
    text: str
    visibility: Literal["summary", "full"]

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise ProtocolDefect("ReasoningData.text must be a string")
        if self.visibility not in ("summary", "full"):
            raise ProtocolDefect("ReasoningData.visibility must be summary or full")


@dataclass(frozen=True, slots=True)
class ToolStartedData:
    tool_call_id: str
    name: str
    arguments: JsonObject | None = None

    def __post_init__(self) -> None:
        _non_empty(self.tool_call_id, "ToolStartedData.tool_call_id")
        _non_empty(self.name, "ToolStartedData.name")
        if self.arguments is not None and not isinstance(self.arguments, FrozenJsonDict):
            raise ProtocolDefect("ToolStartedData.arguments must be frozen JSON when present")
        if self.arguments is not None:
            _validate_frozen_json(self.arguments, "ToolStartedData.arguments")


@dataclass(frozen=True, slots=True)
class ToolUpdatedData:
    tool_call_id: str
    update: JsonObject

    def __post_init__(self) -> None:
        _non_empty(self.tool_call_id, "ToolUpdatedData.tool_call_id")
        if not isinstance(self.update, FrozenJsonDict):
            raise ProtocolDefect("ToolUpdatedData.update must be frozen JSON")
        _validate_frozen_json(self.update, "ToolUpdatedData.update")


@dataclass(frozen=True, slots=True)
class ToolCompletedData:
    tool_call_id: str
    output: JsonValue | None
    succeeded: bool

    def __post_init__(self) -> None:
        _non_empty(self.tool_call_id, "ToolCompletedData.tool_call_id")
        _owned_json(self.output, "ToolCompletedData.output")
        if type(self.succeeded) is not bool:
            raise ProtocolDefect("ToolCompletedData.succeeded must be bool")


@dataclass(frozen=True, slots=True)
class ApprovalRequestedData:
    request: ApprovalRequest

    def __post_init__(self) -> None:
        if not isinstance(self.request, ApprovalRequest):
            raise ProtocolDefect("ApprovalRequestedData.request must be ApprovalRequest")


@dataclass(frozen=True, slots=True)
class ApprovalAnsweredData:
    decision: ApprovalDecision

    def __post_init__(self) -> None:
        if self.decision not in ("allow", "deny", "abort"):
            raise ProtocolDefect("ApprovalAnsweredData.decision is invalid")


@dataclass(frozen=True, slots=True)
class FileChangeData:
    """One file change a backend proposed, with the outcome it reported for it.

    `status` is required and has no default on purpose. Codex's `PatchApplyStatus` is
    `inProgress | completed | failed | declined`, so a change event that omitted its outcome
    would make a declined patch indistinguishable from an applied one — exactly the ambiguity
    that made the Codex adapter suppress both. A default would let a future call site
    reintroduce that ambiguity by forgetting one keyword; a required discriminant cannot be
    forgotten. Backends that only report changes after the fact pass `"applied"` explicitly.
    """

    path: str
    change: Literal["created", "modified", "deleted"]
    status: Literal["in_progress", "applied", "failed", "declined"]
    diff: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.path, "FileChangeData.path")
        if self.change not in ("created", "modified", "deleted"):
            raise ProtocolDefect("FileChangeData.change is invalid")
        if self.status not in ("in_progress", "applied", "failed", "declined"):
            raise ProtocolDefect("FileChangeData.status is invalid")
        if self.diff is not None and type(self.diff) is not str:
            raise ProtocolDefect("FileChangeData.diff must be a string when present")


@dataclass(frozen=True, slots=True)
class UsageData:
    usage: JsonObject

    def __post_init__(self) -> None:
        if not isinstance(self.usage, FrozenJsonDict):
            raise ProtocolDefect("UsageData.usage must be frozen JSON")
        _validate_frozen_json(self.usage, "UsageData.usage")


@dataclass(frozen=True, slots=True)
class DiagnosticData:
    code: str
    message: str
    detail: JsonObject | None = None

    def __post_init__(self) -> None:
        _non_empty(self.code, "DiagnosticData.code")
        _non_empty(self.message, "DiagnosticData.message")
        if self.detail is not None and not isinstance(self.detail, FrozenJsonDict):
            raise ProtocolDefect("DiagnosticData.detail must be frozen JSON when present")
        if self.detail is not None:
            _validate_frozen_json(self.detail, "DiagnosticData.detail")


@dataclass(frozen=True, slots=True)
class UnknownData:
    pass


@dataclass(frozen=True, slots=True)
class NativeRetryObservedData:
    attempt: int

    def __post_init__(self) -> None:
        if type(self.attempt) is not int or self.attempt <= 0:
            raise ProtocolDefect("NativeRetryObservedData.attempt must be positive")


@dataclass(frozen=True, slots=True)
class TurnCompletedData:
    final_text: str
    structured_output: JsonValue | None = None
    usage: JsonObject | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.final_text) is not str:
            raise ProtocolDefect("TurnCompletedData.final_text must be a string")
        _owned_json(self.structured_output, "TurnCompletedData.structured_output")
        if self.usage is not None and not isinstance(self.usage, FrozenJsonDict):
            raise ProtocolDefect("TurnCompletedData.usage must be frozen JSON when present")
        if self.usage is not None:
            _validate_frozen_json(self.usage, "TurnCompletedData.usage")
        _diagnostics(self.diagnostics, "TurnCompletedData.diagnostics")


@dataclass(frozen=True, slots=True)
class TurnFailedData:
    failure: AgentFailureCause
    final_text: str = ""
    structured_output: JsonValue | None = None
    usage: JsonObject | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.failure not in AGENT_FAILURE_CAUSES:
            raise ProtocolDefect("TurnFailedData.failure is invalid")
        if type(self.final_text) is not str:
            raise ProtocolDefect("TurnFailedData.final_text must be a string")
        _owned_json(self.structured_output, "TurnFailedData.structured_output")
        if self.usage is not None and not isinstance(self.usage, FrozenJsonDict):
            raise ProtocolDefect("TurnFailedData.usage must be frozen JSON when present")
        if self.usage is not None:
            _validate_frozen_json(self.usage, "TurnFailedData.usage")
        _diagnostics(self.diagnostics, "TurnFailedData.diagnostics")


@dataclass(frozen=True, slots=True)
class TurnCancelledData:
    final_text: str = ""
    structured_output: JsonValue | None = None
    usage: JsonObject | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.final_text) is not str:
            raise ProtocolDefect("TurnCancelledData.final_text must be a string")
        _owned_json(self.structured_output, "TurnCancelledData.structured_output")
        if self.usage is not None and not isinstance(self.usage, FrozenJsonDict):
            raise ProtocolDefect("TurnCancelledData.usage must be frozen JSON when present")
        if self.usage is not None:
            _validate_frozen_json(self.usage, "TurnCancelledData.usage")
        _diagnostics(self.diagnostics, "TurnCancelledData.diagnostics")


type AgentEventData = (
    SessionStartedData
    | TurnStartedData
    | TextDeltaData
    | ReasoningData
    | ToolStartedData
    | ToolUpdatedData
    | ToolCompletedData
    | ApprovalRequestedData
    | ApprovalAnsweredData
    | FileChangeData
    | UsageData
    | DiagnosticData
    | UnknownData
    | NativeRetryObservedData
    | TurnCompletedData
    | TurnFailedData
    | TurnCancelledData
)

_DATA_BY_KIND: dict[AgentEventKind, type[AgentEventData]] = {
    "session_started": SessionStartedData,
    "turn_started": TurnStartedData,
    "text_delta": TextDeltaData,
    "reasoning": ReasoningData,
    "tool_started": ToolStartedData,
    "tool_updated": ToolUpdatedData,
    "tool_completed": ToolCompletedData,
    "approval_requested": ApprovalRequestedData,
    "approval_answered": ApprovalAnsweredData,
    "file_change": FileChangeData,
    "usage": UsageData,
    "diagnostic": DiagnosticData,
    "unknown": UnknownData,
    "native_retry_observed": NativeRetryObservedData,
    "turn_completed": TurnCompletedData,
    "turn_failed": TurnFailedData,
    "turn_cancelled": TurnCancelledData,
}


@dataclass(frozen=True, slots=True)
class AgentEvent:
    schema_version: Literal["agent-event.v1"]
    seq: int
    backend: Backend
    transport: AgentTransport
    session_ref: AgentSessionRef
    turn_id: str
    kind: AgentEventKind
    data: AgentEventData
    native_type: str | None = None
    native_payload: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "agent-event.v1":
            raise ProtocolDefect("AgentEvent.schema_version must be agent-event.v1")
        if type(self.seq) is not int or self.seq <= 0:
            raise ProtocolDefect("AgentEvent.seq must be a positive integer")
        if not isinstance(self.session_ref, AgentSessionRef):
            raise ProtocolDefect("AgentEvent.session_ref must be AgentSessionRef")
        if self.backend != self.session_ref.backend or self.transport != self.session_ref.transport:
            raise ProtocolDefect("AgentEvent route does not match its session ref")
        _non_empty(self.turn_id, "AgentEvent.turn_id")
        expected_data = _DATA_BY_KIND.get(self.kind)
        if expected_data is None or not isinstance(self.data, expected_data):
            raise ProtocolDefect("AgentEvent kind and data variant do not match")
        if self.native_type is not None:
            _non_empty(self.native_type, "AgentEvent.native_type")
        if self.native_payload is not None and not isinstance(self.native_payload, FrozenJsonDict):
            raise ProtocolDefect("AgentEvent.native_payload must be frozen JSON")
        if self.native_payload is not None:
            _validate_frozen_json(self.native_payload, "AgentEvent.native_payload")


async def validate_event_stream(
    source: AsyncIterator[AgentEvent],
) -> AsyncGenerator[AgentEvent, None]:
    """Validate one stream before releasing its terminal event."""
    expected_seq = 1
    session_ref: AgentSessionRef | None = None
    turn_id: str | None = None
    turn_started = False
    terminal: AgentEvent | None = None
    async for event in source:
        if terminal is not None:
            raise ProtocolDefect(
                "agent stream emitted an event after its terminal",
                code="event_after_terminal",
            )
        if event.seq != expected_seq:
            raise ProtocolDefect(
                "agent stream sequence was not gap-free",
                code="invalid_event_sequence",
            )
        expected_seq += 1
        if session_ref is None:
            session_ref = event.session_ref
            turn_id = event.turn_id
        elif event.session_ref != session_ref or event.turn_id != turn_id:
            raise ProtocolDefect(
                "agent stream changed session or turn identity",
                code="event_identity_changed",
            )
        if event.kind == "session_started":
            if event.seq != 1:
                raise ProtocolDefect(
                    "session_started must be the first event",
                    code="misordered_session_started",
                )
        elif event.kind == "turn_started":
            if turn_started:
                raise ProtocolDefect(
                    "turn_started was emitted twice", code="duplicate_turn_started"
                )
            turn_started = True
        elif event.kind in TERMINAL_EVENT_KINDS:
            # A terminal is legal at any position once the stream has opened: cancellation and
            # the runtime-enforced timeout can land between session_started and turn_started,
            # and the spec requires that window to end in a terminal value, not a defect.
            pass
        elif event.kind in SESSION_SCOPE_EVENT_KINDS:
            # Session-scope events are legal wherever the backend reports them, including
            # before the turn opens; see SESSION_SCOPE_EVENT_KINDS for why these two and no
            # others. Reordering them into the turn would hide the ordering the operator needs.
            pass
        elif not turn_started:
            raise ProtocolDefect(
                "turn content preceded turn_started",
                code="missing_turn_started",
            )
        if event.kind in TERMINAL_EVENT_KINDS:
            terminal = event
        else:
            yield event
    if terminal is None:
        raise MissingTerminalEvent()
    yield terminal


def terminal_event_to_result(event: AgentEvent) -> AgentResult:
    match event.data:
        case TurnCompletedData(
            final_text=final_text,
            structured_output=structured_output,
            usage=usage,
            diagnostics=diagnostics,
        ):
            return AgentResult(
                status="succeeded",
                failure=None,
                final_text=final_text,
                structured_output=structured_output,
                session_ref=event.session_ref,
                turn_id=event.turn_id,
                usage=usage,
                diagnostics=diagnostics,
                terminal_native_type=event.native_type,
                terminal_native_payload=event.native_payload,
            )
        case TurnFailedData(
            failure=failure,
            final_text=final_text,
            structured_output=structured_output,
            usage=usage,
            diagnostics=diagnostics,
        ):
            return AgentResult(
                status="failed",
                failure=failure,
                final_text=final_text,
                structured_output=structured_output,
                session_ref=event.session_ref,
                turn_id=event.turn_id,
                usage=usage,
                diagnostics=diagnostics,
                terminal_native_type=event.native_type,
                terminal_native_payload=event.native_payload,
            )
        case TurnCancelledData(
            final_text=final_text,
            structured_output=structured_output,
            usage=usage,
            diagnostics=diagnostics,
        ):
            return AgentResult(
                status="cancelled",
                failure=None,
                final_text=final_text,
                structured_output=structured_output,
                session_ref=event.session_ref,
                turn_id=event.turn_id,
                usage=usage,
                diagnostics=diagnostics,
                terminal_native_type=event.native_type,
                terminal_native_payload=event.native_payload,
            )
        case (
            SessionStartedData()
            | TurnStartedData()
            | TextDeltaData()
            | ReasoningData()
            | ToolStartedData()
            | ToolUpdatedData()
            | ToolCompletedData()
            | ApprovalRequestedData()
            | ApprovalAnsweredData()
            | FileChangeData()
            | UsageData()
            | DiagnosticData()
            | UnknownData()
            | NativeRetryObservedData()
        ):
            raise ProtocolDefect(
                "terminal projection requires a terminal event",
                code="nonterminal_projection",
            )
        case _:
            assert_never(event.data)


__all__ = [
    "SESSION_SCOPE_EVENT_KINDS",
    "TERMINAL_EVENT_KINDS",
    "AgentEvent",
    "AgentEventData",
    "AgentEventKind",
    "ApprovalAnsweredData",
    "ApprovalRequestedData",
    "DiagnosticData",
    "FileChangeData",
    "NativeRetryObservedData",
    "ReasoningData",
    "SessionScopeEventKind",
    "SessionStartedData",
    "TerminalEventKind",
    "TextDeltaData",
    "ToolCompletedData",
    "ToolStartedData",
    "ToolUpdatedData",
    "TurnCancelledData",
    "TurnCompletedData",
    "TurnFailedData",
    "TurnStartedData",
    "UnknownData",
    "UsageData",
    "terminal_event_to_result",
    "validate_event_stream",
]
