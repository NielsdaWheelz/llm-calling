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


class MissingTerminalEvent(ProtocolDefect):
    def __init__(self, message: str = "agent stream ended without a terminal event") -> None:
        super().__init__(message, code="missing_terminal_event")
