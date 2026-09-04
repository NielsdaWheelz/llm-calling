"""The six normalized agent stream events and the terminal grammar.

The event vocabulary is deliberately closed at six kinds. First-class kinds carry
owned, validated data; every native frame that has no first-class kind travels as
``AgentNative`` with a bounded, recursively redacted payload. Exactly one
``AgentTerminal`` ends every started turn.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Literal

from provider_runtime.types import Absent, Presence, Present, TokenUsage

from .errors import (
    InvalidAgentRequest,
    MissingTerminalEvent,
    ProtocolDefect,
)
from .types import (
    AgentSessionRef,
    ApprovalDecision,
    ApprovalRequest,
    FrozenJsonDict,
    JsonObject,
    JsonValue,
    require_frozen_json,
)

type AgentFailureCause = Literal[
    "backend_failed",
    "turn_timeout",
    "output_limit_exceeded",
    "approval_unanswered",
    "output_schema_violation",
]
AGENT_FAILURE_CAUSES: tuple[AgentFailureCause, ...] = (
    "backend_failed",
    "turn_timeout",
    "output_limit_exceeded",
    "approval_unanswered",
    "output_schema_violation",
)
type ToolUsePhase = Literal["started", "updated", "completed"]
type AgentTerminalStatus = Literal["succeeded", "failed", "cancelled"]


def _non_empty(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ProtocolDefect(f"{field} must be a non-empty string")


def _validate_frozen_json(value: object, field: str) -> None:
    try:
        require_frozen_json(value, field)
    except InvalidAgentRequest:
        raise ProtocolDefect(f"{field} must be recursively frozen JSON") from None


# ---------------------------------------------------------------------------
# Terminal failure values — expected failures as values, never raises.


@dataclass(frozen=True, slots=True)
class AgentQuotaExhausted:
    """The subscription pool refused the turn.

    Block-and-stop is the whole contract: this lane never enables API-rate
    overflow and never forwards API-key credentials, so there is nothing to
    fall back to and nothing to retry.
    """


@dataclass(frozen=True, slots=True)
class AgentFailure:
    cause: AgentFailureCause

    def __post_init__(self) -> None:
        if self.cause not in AGENT_FAILURE_CAUSES:
            raise ProtocolDefect(f"AgentFailure.cause {self.cause!r} is unknown")


type AgentTerminalFailure = AgentQuotaExhausted | AgentFailure


# ---------------------------------------------------------------------------
# The six event kinds.


@dataclass(frozen=True, slots=True)
class AgentText:
    """One chunk of the assistant's visible output text."""

    text: str

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise ProtocolDefect("AgentText.text must be a string")


@dataclass(frozen=True, slots=True)
class AgentToolUse:
    """One observation of a model-initiated tool action.

    ``payload`` is the phase's own data — arguments on ``started``, progress on
    ``updated``, output on ``completed`` — owned frozen JSON, not a redacted
    diagnostic view. ``succeeded`` exists exactly on ``completed``; a completion
    without an outcome would make a failed action indistinguishable from an
    applied one.
    """

    tool_call_id: str
    name: str
    phase: ToolUsePhase
    payload: JsonValue = None
    succeeded: bool | None = None

    def __post_init__(self) -> None:
        _non_empty(self.tool_call_id, "AgentToolUse.tool_call_id")
        _non_empty(self.name, "AgentToolUse.name")
        if self.phase not in ("started", "updated", "completed"):
            raise ProtocolDefect(f"AgentToolUse.phase {self.phase!r} is unknown")
        _validate_frozen_json(self.payload, "AgentToolUse.payload")
        if self.phase == "completed":
            if type(self.succeeded) is not bool:
                raise ProtocolDefect("completed AgentToolUse requires a bool succeeded")
        elif self.succeeded is not None:
            raise ProtocolDefect("succeeded is only valid on a completed AgentToolUse")


@dataclass(frozen=True, slots=True)
class AgentUsage:
    """One invocation-to-date usage snapshot, normalized to the provider noun.

    Snapshots within a turn are not additive. Even when a backend reports
    thread- or session-cumulative state, adapters project only usage attributable
    to this ``AgentRuntime`` invocation.
    """

    usage: TokenUsage

    def __post_init__(self) -> None:
        if not isinstance(self.usage, TokenUsage):
            raise ProtocolDefect("AgentUsage.usage must be TokenUsage")


@dataclass(frozen=True, slots=True)
class AgentPermissionRequest:
    """One answered unsafe-action confirmation.

    The interaction happens through the caller's ``ApprovalHandler`` (or the
    policy's own deny/allow); the stream event is the auditable record and
    therefore carries the decision that was made.
    """

    request: ApprovalRequest
    decision: ApprovalDecision

    def __post_init__(self) -> None:
        if not isinstance(self.request, ApprovalRequest):
            raise ProtocolDefect("AgentPermissionRequest.request must be ApprovalRequest")
        if self.decision not in ("allow", "deny", "abort"):
            raise ProtocolDefect("AgentPermissionRequest.decision is invalid")


@dataclass(frozen=True, slots=True)
class AgentNative:
    """One native frame with no first-class kind, bounded and recursively redacted.

    ``payload`` must come from :func:`provider_runtime.agent_runtime.auth.
    redact_native_payload`: string values are sanitized, credential-shaped keys
    are dropped, and depth/item/byte bounds are enforced before anything is
    retained.
    """

    native_type: str
    payload: JsonObject

    def __post_init__(self) -> None:
        _non_empty(self.native_type, "AgentNative.native_type")
        if not isinstance(self.payload, FrozenJsonDict):
            raise ProtocolDefect("AgentNative.payload must be frozen JSON")
        _validate_frozen_json(self.payload, "AgentNative.payload")


@dataclass(frozen=True, slots=True)
class AgentTerminal:
    """The exactly-once terminal value of a started turn.

    ``usage`` is invocation-local on every status. It never includes historical
    thread/session usage, including when the upstream protocol is cumulative.
    ``Absent`` means the provider supplied no safely attributable usage.
    """

    status: AgentTerminalStatus
    failure: AgentTerminalFailure | None
    final_text: str
    session_ref: AgentSessionRef
    structured_output: JsonValue | None = None
    usage: Presence[TokenUsage] = Absent()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("succeeded", "failed", "cancelled"):
            raise ProtocolDefect(f"AgentTerminal.status {self.status!r} is unknown")
        if self.status == "failed":
            if not isinstance(self.failure, AgentQuotaExhausted | AgentFailure):
                raise ProtocolDefect("failed AgentTerminal requires a typed failure value")
        elif self.failure is not None:
            raise ProtocolDefect("only a failed AgentTerminal may carry a failure")
        if type(self.final_text) is not str:
            raise ProtocolDefect("AgentTerminal.final_text must be a string")
        if not isinstance(self.session_ref, AgentSessionRef):
            raise ProtocolDefect("AgentTerminal.session_ref must be AgentSessionRef")
        if self.structured_output is not None:
            _validate_frozen_json(self.structured_output, "AgentTerminal.structured_output")
        match self.usage:
            case Present(value=usage) if not isinstance(usage, TokenUsage):
                raise ProtocolDefect("AgentTerminal.usage must be Presence[TokenUsage]")
            case Present() | Absent():
                pass
            case _:
                raise ProtocolDefect("AgentTerminal.usage must be Presence[TokenUsage]")
        if not isinstance(self.diagnostics, tuple) or any(
            type(item) is not str or not item for item in self.diagnostics
        ):
            raise ProtocolDefect("AgentTerminal.diagnostics must be non-empty strings")
        if len(self.diagnostics) != len(set(self.diagnostics)):
            raise ProtocolDefect("AgentTerminal.diagnostics must not contain duplicates")


type AgentEvent = (
    AgentText | AgentToolUse | AgentUsage | AgentPermissionRequest | AgentNative | AgentTerminal
)

# The closed kind set, spelled once for isinstance checks and gate tests.
AGENT_EVENT_KINDS: tuple[type, ...] = (
    AgentText,
    AgentToolUse,
    AgentUsage,
    AgentPermissionRequest,
    AgentNative,
    AgentTerminal,
)


async def validate_event_stream(
    source: AsyncIterator[AgentEvent],
) -> AsyncGenerator[AgentEvent, None]:
    """Enforce the terminal grammar: exactly one ``AgentTerminal``, last.

    The terminal is held back until the source ends so a post-terminal frame is
    a defect the consumer sees *instead of* a terminal, never after one.
    """
    terminal: AgentTerminal | None = None
    async for event in source:
        if terminal is not None:
            raise ProtocolDefect(
                "agent stream emitted an event after its terminal",
                code="event_after_terminal",
            )
        if not isinstance(event, AGENT_EVENT_KINDS):
            raise ProtocolDefect(
                "agent stream emitted a value outside the closed event union",
                code="unknown_event_kind",
            )
        if isinstance(event, AgentTerminal):
            terminal = event
        else:
            yield event
    if terminal is None:
        raise MissingTerminalEvent()
    yield terminal


__all__ = [
    "AGENT_EVENT_KINDS",
    "AGENT_FAILURE_CAUSES",
    "AgentEvent",
    "AgentFailure",
    "AgentFailureCause",
    "AgentNative",
    "AgentPermissionRequest",
    "AgentQuotaExhausted",
    "AgentTerminal",
    "AgentTerminalFailure",
    "AgentTerminalStatus",
    "AgentText",
    "AgentToolUse",
    "AgentUsage",
    "ToolUsePhase",
    "validate_event_stream",
]
