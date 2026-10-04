"""Prepared native turns: immutable submission, terminal, and control evidence."""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from .errors import InvalidAgentRequest, ProtocolDefect
from .types import AgentSessionRef, ContentPart, JsonObject, JsonValue, require_frozen_json

if TYPE_CHECKING:
    from .events import AgentEvent, AgentTerminal


def _non_empty(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise InvalidAgentRequest(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class AgentAttempt:
    attempt_id: str
    request_digest: str

    def __post_init__(self) -> None:
        _non_empty(self.attempt_id, "AgentAttempt.attempt_id")
        if type(self.request_digest) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", self.request_digest
        ):
            raise InvalidAgentRequest("AgentAttempt.request_digest must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class AgentTurnRef:
    session_ref: AgentSessionRef
    native_turn_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.session_ref, AgentSessionRef) or self.session_ref.backend != "codex":
            raise InvalidAgentRequest("AgentTurnRef requires a Codex session ref")
        _non_empty(self.native_turn_id, "AgentTurnRef.native_turn_id")


@dataclass(frozen=True, slots=True)
class AgentResultRef:
    session_ref: AgentSessionRef
    native_result_id: str | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.session_ref, AgentSessionRef)
            or self.session_ref.backend != "claude"
        ):
            raise InvalidAgentRequest("AgentResultRef requires a Claude session ref")
        if self.native_result_id is not None:
            _non_empty(self.native_result_id, "AgentResultRef.native_result_id")


@dataclass(frozen=True, slots=True)
class AgentNotSubmitted:
    attempt: AgentAttempt
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AgentAttempt):
            raise InvalidAgentRequest("AgentNotSubmitted.attempt must be AgentAttempt")
        _non_empty(self.reason, "AgentNotSubmitted.reason")


@dataclass(frozen=True, slots=True)
class AgentAccepted:
    attempt: AgentAttempt
    turn: AgentTurnRef

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AgentAttempt) or not isinstance(self.turn, AgentTurnRef):
            raise InvalidAgentRequest("AgentAccepted requires attempt and native turn identity")


@dataclass(frozen=True, slots=True)
class AgentUncertain:
    attempt: AgentAttempt
    turn: AgentTurnRef | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AgentAttempt):
            raise InvalidAgentRequest("AgentUncertain.attempt must be AgentAttempt")
        if self.turn is not None and not isinstance(self.turn, AgentTurnRef):
            raise InvalidAgentRequest("AgentUncertain.turn must be AgentTurnRef when supplied")
        _non_empty(self.reason, "AgentUncertain.reason")


type AgentSubmission = AgentNotSubmitted | AgentAccepted | AgentUncertain


@dataclass(frozen=True, slots=True)
class NativeTerminalEvidence:
    attempt: AgentAttempt
    native_ref: AgentTurnRef | AgentResultRef
    seal_revision: Literal["codex-turn-completed.v1", "claude-result.v1"]
    origin: Literal["native"] = field(default="native", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AgentAttempt):
            raise ProtocolDefect("native terminal evidence omitted its submitted attempt")
        if isinstance(self.native_ref, AgentTurnRef):
            expected = "codex-turn-completed.v1"
        elif isinstance(self.native_ref, AgentResultRef):
            expected = "claude-result.v1"
        else:
            raise ProtocolDefect("native terminal evidence has invalid native identity")
        if self.seal_revision != expected:
            raise ProtocolDefect("native terminal evidence seal differs from its backend")


@dataclass(frozen=True, slots=True)
class LocalStopEvidence:
    submission: AgentAccepted | AgentUncertain
    reason: str
    origin: Literal["local_stop"] = field(default="local_stop", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.submission, AgentAccepted | AgentUncertain):
            raise ProtocolDefect("local stop requires accepted or unresolved submission evidence")
        _non_empty(self.reason, "LocalStopEvidence.reason")


@dataclass(frozen=True, slots=True)
class RawAgentOutput:
    value: JsonValue

    def __post_init__(self) -> None:
        require_frozen_json(self.value, "RawAgentOutput.value")


@dataclass(frozen=True, slots=True)
class AgentTurnControls:
    rpc_seconds: float
    pending_calls: int
    pending_call_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.rpc_seconds) not in (int, float)
            or not math.isfinite(self.rpc_seconds)
            or self.rpc_seconds <= 0
        ):
            raise InvalidAgentRequest("AgentTurnControls.rpc_seconds must be positive and finite")
        for name in ("pending_calls", "pending_call_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise InvalidAgentRequest(f"AgentTurnControls.{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class AgentControlReceipt:
    operation: Literal["steer", "interrupt"]
    request_id: str
    turn: AgentTurnRef | None
    disposition: Literal["not_sent", "rejected", "accepted", "unknown"]
    input_id: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.operation not in ("steer", "interrupt"):
            raise ProtocolDefect("unknown native control operation")
        if self.disposition not in ("not_sent", "rejected", "accepted", "unknown"):
            raise ProtocolDefect("unknown native control disposition")
        _non_empty(self.request_id, "AgentControlReceipt.request_id")
        _non_empty(self.reason, "AgentControlReceipt.reason")
        if self.turn is not None and not isinstance(self.turn, AgentTurnRef):
            raise ProtocolDefect("AgentControlReceipt.turn must be AgentTurnRef when supplied")
        if self.input_id is not None:
            _non_empty(self.input_id, "AgentControlReceipt.input_id")
        if self.operation == "interrupt" and self.input_id is not None:
            raise ProtocolDefect("interrupt control cannot identify a steered input")


@dataclass(frozen=True, slots=True)
class AgentCloseResult:
    local_closed: bool
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.local_closed) is not bool:
            raise ProtocolDefect("AgentCloseResult.local_closed must be bool")
        if type(self.diagnostics) is not tuple or any(
            type(item) is not str or not item for item in self.diagnostics
        ):
            raise ProtocolDefect("AgentCloseResult.diagnostics must be non-empty strings")


class AgentTurn(Protocol):
    @property
    def attempt(self) -> AgentAttempt: ...

    @property
    def submission(self) -> AgentSubmission | None: ...

    @property
    def terminal(self) -> AgentTerminal | None: ...

    @property
    def submitted_request(self) -> JsonObject: ...

    def events(self) -> AsyncIterator[AgentEvent]: ...

    async def submit(self) -> AgentSubmission: ...

    async def reply(self, call: AgentToolCall, result: AgentToolReply) -> None: ...

    async def steer(
        self, *, input_id: str, input: tuple[ContentPart, ...]
    ) -> AgentControlReceipt: ...

    async def interrupt(self) -> AgentControlReceipt: ...

    def revoke(self) -> None: ...

    async def close(self) -> AgentCloseResult: ...


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    turn: AgentTurnRef
    call_id: str
    reply_token: str
    name: str
    arguments: JsonValue

    def __post_init__(self) -> None:
        if not isinstance(self.turn, AgentTurnRef):
            raise ProtocolDefect("AgentToolCall.turn must be AgentTurnRef")
        _non_empty(self.call_id, "AgentToolCall.call_id")
        _non_empty(self.reply_token, "AgentToolCall.reply_token")
        _non_empty(self.name, "AgentToolCall.name")
        require_frozen_json(self.arguments, "AgentToolCall.arguments")


@dataclass(frozen=True, slots=True)
class AgentToolReply:
    text: str
    success: bool

    def __post_init__(self) -> None:
        if type(self.text) is not str or type(self.success) is not bool:
            raise InvalidAgentRequest("AgentToolReply requires text and a bool tool outcome")


@dataclass(frozen=True, slots=True)
class AgentMessage:
    turn: AgentTurnRef
    message_id: str
    phase: Literal["commentary", "final_answer", "unknown"]
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.turn, AgentTurnRef):
            raise ProtocolDefect("AgentMessage.turn must be AgentTurnRef")
        _non_empty(self.message_id, "AgentMessage.message_id")
        if self.phase not in ("commentary", "final_answer", "unknown"):
            raise ProtocolDefect("unknown native message phase")
        if type(self.text) is not str:
            raise ProtocolDefect("AgentMessage.text must be a string")


@dataclass(frozen=True, slots=True)
class AgentInputRecorded:
    turn: AgentTurnRef
    input_id: str
    item_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.turn, AgentTurnRef):
            raise ProtocolDefect("AgentInputRecorded.turn must be AgentTurnRef")
        _non_empty(self.input_id, "AgentInputRecorded.input_id")
        _non_empty(self.item_id, "AgentInputRecorded.item_id")
