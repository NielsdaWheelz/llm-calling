"""The agent runtime is a second public surface, never a root compatibility facade."""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import ModuleType
from typing import get_args

import provider_runtime
import provider_runtime.agent_runtime as agent_runtime
from provider_runtime.agent_runtime import events


def test_every_agent_all_name_is_importable_sorted_and_unique() -> None:
    missing = [name for name in agent_runtime.__all__ if not hasattr(agent_runtime, name)]

    assert not missing, f"agent_runtime.__all__ names missing from the package: {missing}"
    assert agent_runtime.__all__ == sorted(agent_runtime.__all__)
    assert len(agent_runtime.__all__) == len(set(agent_runtime.__all__))


def test_agent_public_surface_is_exact() -> None:
    assert set(agent_runtime.__all__) == {
        "AgentCapabilities",
        "AgentCapabilityScope",
        "AgentEvent",
        "AgentEventData",
        "AgentEventKind",
        "AgentFailureCause",
        "AgentOutputSpec",
        "AgentResult",
        "AgentRuntime",
        "AgentRuntimeConfig",
        "AgentRuntimeDefect",
        "AgentRuntimeError",
        "AgentRuntimeOperation",
        "AgentSession",
        "AgentSessionRef",
        "AgentSessionRequest",
        "AgentTarget",
        "AgentTransport",
        "ApiTarget",
        "ApprovalAnsweredData",
        "ApprovalDecision",
        "ApprovalHandler",
        "ApprovalMode",
        "ApprovalRequest",
        "ApprovalRequestedData",
        "Backend",
        "BuiltinToolFamily",
        "CallTarget",
        "CapturedAgentCall",
        "ClaudeNativeOptions",
        "ClaudeSessionFilters",
        "CodexNativeOptions",
        "CodexSessionFilters",
        "ConcurrentTurn",
        "ContentPart",
        "CredentialKind",
        "CredentialRef",
        "CredentialRejected",
        "CredentialUnavailable",
        "DiagnosticData",
        "DiscoveryOperation",
        "EnvironmentReference",
        "ExecutableUnavailable",
        "FileChangeData",
        "FileContent",
        "FilesystemMode",
        "ForkSession",
        "FrozenJsonDict",
        "HeaderReference",
        "ImageContent",
        "InstructionRole",
        "InvalidAgentRequest",
        "JsonObject",
        "JsonScalar",
        "JsonSchemaAgentOutput",
        "JsonValue",
        "McpConfigurationError",
        "McpServerSpec",
        "McpTransport",
        "McpUnavailable",
        "MissingTerminalEvent",
        "NativeOptions",
        "NativeRetryObservedData",
        "NetworkMode",
        "NewSession",
        "NoNetworkAgentRuntime",
        "PermissionPolicy",
        "PermissionPolicyPatch",
        "ProtocolDefect",
        "ReasoningData",
        "ReasoningSpec",
        "ReasoningSummary",
        "ResumeSession",
        "SESSION_SCOPE_EVENT_KINDS",
        "ScriptedAgentRuntime",
        "SdkUnavailable",
        "SecretResolver",
        "SessionFilters",
        "SessionMetadata",
        "SessionMismatch",
        "SessionOpen",
        "SessionOperation",
        "SessionPage",
        "SessionQuery",
        "SessionReadOptions",
        "SessionScopeEventKind",
        "SessionSnapshot",
        "SessionStartedData",
        "SessionSummary",
        "SessionUnavailable",
        "TERMINAL_EVENT_KINDS",
        "TerminalEventKind",
        "TextAgentOutput",
        "TextContent",
        "TextDeltaData",
        "ToolCompletedData",
        "ToolStartedData",
        "ToolUpdatedData",
        "TurnCancelledData",
        "TurnCompletedData",
        "TurnFailedData",
        "TurnNotStarted",
        "TurnOverride",
        "TurnRequest",
        "TurnStartedData",
        "UnknownData",
        "UnsafeConfirmation",
        "UnsafeDimension",
        "UnsupportedCapability",
        "UsageData",
        "agent_target_to_session_request",
        "api_target_to_provider_target",
        "freeze_json_object",
        "freeze_json_value",
        "ref_from_json",
        "ref_to_json",
        "thaw_json_value",
        "tool_is_allowed",
    }


def test_both_stream_grammar_constants_are_exported_and_stay_the_validator_s_own() -> None:
    """The exported grammar constants are the rule `validate_event_stream` enforces.

    `AgentEventKind` and `TerminalEventKind` were exported from the start, so a consumer could
    *name* a kind in an annotation but had to hard-code `("turn_completed", "turn_failed",
    "turn_cancelled")` to branch on one at runtime — and a restated closed set is a copy that
    drifts. Both closed subsets of the stream contract are therefore exported together:
    `TERMINAL_EVENT_KINDS`, which ends a stream and is in bijection with `AgentResult.status`,
    and `SESSION_SCOPE_EVENT_KINDS`, the only non-terminal kinds legal before `turn_started`.
    Exporting one without the other would publish half a grammar to a consumer — including a
    caller-supplied `adapters=` implementation, which has to satisfy the whole of it.

    Re-export is identity, never a copy, or the sanctioned surface could disagree with the
    module that raises `ProtocolDefect` on it.
    """
    assert agent_runtime.TERMINAL_EVENT_KINDS is events.TERMINAL_EVENT_KINDS
    assert agent_runtime.SESSION_SCOPE_EVENT_KINDS is events.SESSION_SCOPE_EVENT_KINDS

    every_kind = set(get_args(agent_runtime.AgentEventKind.__value__))
    terminal = set(agent_runtime.TERMINAL_EVENT_KINDS)
    session_scope = set(agent_runtime.SESSION_SCOPE_EVENT_KINDS)

    assert terminal, "the terminal grammar must not be empty"
    assert session_scope, "the session-scope grammar must not be empty"
    assert terminal <= every_kind
    assert session_scope <= every_kind
    assert not terminal & session_scope, "a kind cannot be both terminal and session-scoped"
    assert terminal == set(get_args(agent_runtime.TerminalEventKind.__value__))
    assert session_scope == set(get_args(agent_runtime.SessionScopeEventKind.__value__))


def test_agent_runtime_is_not_a_root_compatibility_surface() -> None:
    assert "AgentRuntime" not in provider_runtime.__all__
    assert "AgentSessionRequest" not in provider_runtime.__all__
    assert "CallTarget" not in provider_runtime.__all__


def test_importing_agent_surface_does_not_import_optional_sdk() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import provider_runtime.agent_runtime; "
                "assert 'openai_codex' not in sys.modules; "
                "assert 'claude_agent_sdk' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_adapter_and_private_modules_are_not_agent_exports() -> None:
    for name in (
        "codex_sdk",
        "claude_sdk",
        "_codex_launcher",
        "_claude_launcher",
        "_limits",
        "_process",
        "_structured_output",
    ):
        assert name not in agent_runtime.__all__, f"{name} must stay unexported"


def test_deleted_legacy_agent_facades_stay_absent() -> None:
    for module in ("agent", "agents", "agent_client", "agent_facade"):
        try:
            importlib.import_module(f"provider_runtime.{module}")
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"legacy agent facade provider_runtime.{module} is importable")


def test_no_name_becomes_public_on_the_agent_package_without_being_declared() -> None:
    """`__all__` is the surface, not a summary of it.

    The exact-surface set above is hand-written, so it only catches a name someone removed or
    renamed. This catches the other direction — the one the last two phases actually produced,
    where a private helper is promoted and re-exported without being declared — because it
    needs no maintenance to stay true. Submodules bound by the package's own imports are not
    exports; `test_adapter_and_private_modules_are_not_agent_exports` owns those.
    """
    undeclared = sorted(
        name
        for name, value in vars(agent_runtime).items()
        if not name.startswith("_")
        and name not in set(agent_runtime.__all__)
        and not isinstance(value, ModuleType)
    )

    assert undeclared == [], f"public agent names missing from __all__: {undeclared}"
