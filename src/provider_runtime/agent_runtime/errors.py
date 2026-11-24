"""Typed errors and defects for the agent-runtime lane.

This hierarchy is intentionally separate from the provider HTTP runtime. Agent
configuration and availability failures are expected errors; broken native
protocol invariants are defects.
"""

from __future__ import annotations

from typing import Literal


class AgentRuntimeError(Exception):
    """Expected, modelable agent-runtime failure."""

    code: str
    message: str

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class InvalidAgentRequest(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_agent_request")


class UnsupportedCapability(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsupported_capability")


class CredentialUnavailable(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="credential_unavailable")


class CredentialRejected(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="credential_rejected")


class ExecutableUnavailable(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="executable_unavailable")


class SdkUnavailable(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="sdk_unavailable")


class McpConfigurationError(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="mcp_configuration_error")


class McpUnavailable(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="mcp_unavailable")


class SessionMismatch(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="session_mismatch")


class SessionUnavailable(AgentRuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="session_unavailable")


class ConcurrentTurn(AgentRuntimeError):
    def __init__(self, message: str = "the agent session already has an active turn") -> None:
        super().__init__(message, code="concurrent_turn")


class TurnNotStarted(AgentRuntimeError):
    """An expected stop before a backend supplied persistable turn identity."""

    reason: Literal["cancelled", "turn_timeout"]

    def __init__(self, reason: Literal["cancelled", "turn_timeout"]) -> None:
        messages = {
            "cancelled": "agent turn was cancelled before native identity was established",
            "turn_timeout": "agent turn timed out before native identity was established",
        }
        self.reason = reason
        super().__init__(messages[reason], code="turn_not_started")


class AgentRuntimeDefect(Exception):
    """Broken agent-runtime invariant; never a product-facing result."""

    code: str
    message: str

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ProtocolDefect(AgentRuntimeDefect):
    def __init__(self, message: str, *, code: str = "protocol_defect") -> None:
        super().__init__(message, code=code)


type CodexUnfinishedItemReason = Literal[
    "open_authority", "pending_input", "unproved_order", "summary_mismatch", "incomplete_identity"
]
type CodexSummaryMatch = Literal["match", "mismatch", "absent", "unchecked"]


class CodexUnfinishedItems(ProtocolDefect):
    """Content-free evidence for rejection of a successful but unbalanced terminal."""

    def __init__(
        self,
        *,
        reason: CodexUnfinishedItemReason,
        open_display_items: int,
        open_other_items: int,
        pending_calls: int,
        pending_inputs: int,
        pending_server_requests: int,
        summary: CodexSummaryMatch,
    ) -> None:
        super().__init__("turn completed with unfinished Codex item lifecycles")
        self.reason = reason
        self.open_display_items = open_display_items
        self.open_other_items = open_other_items
        self.pending_calls = pending_calls
        self.pending_inputs = pending_inputs
        self.pending_server_requests = pending_server_requests
        self.summary = summary


class ConflictingSubmissionEvidence(ProtocolDefect):
    """The same submission carries both native acceptance and definitive refusal."""

    def __init__(self) -> None:
        super().__init__("stock both accepted and refused the same turn/start")


class MissingTerminalEvent(ProtocolDefect):
    def __init__(self, message: str = "agent stream ended without a terminal event") -> None:
        super().__init__(message, code="missing_terminal_event")
