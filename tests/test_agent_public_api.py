"""The agent runtime is a second public surface, never a root compatibility facade."""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import ModuleType

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
        "AgentEvent",
        "AgentFailure",
        "AgentFailureCause",
        "AgentNative",
        "AgentOutputSpec",
        "AgentPermissionRequest",
        "AgentQuotaExhausted",
        "AgentRuntime",
        "AgentRuntimeConfig",
        "AgentRuntimeDefect",
        "AgentRuntimeError",
        "AgentRuntimeOperation",
        "AgentSession",
        "AgentSessionRef",
        "AgentSessionRequest",
        "AgentTerminal",
        "AgentTerminalFailure",
        "AgentTerminalStatus",
        "AgentText",
        "AgentToolUse",
        "AgentTransport",
        "AgentUsage",
        "ApprovalDecision",
        "ApprovalHandler",
        "ApprovalMode",
        "ApprovalRequest",
        "Backend",
        "CapturedAgentCall",
        "ClaudeNativeOptions",
        "CodexNativeOptions",
        "ConcurrentTurn",
        "ContentPart",
        "CredentialKind",
        "CredentialRef",
        "CredentialRejected",
        "CredentialUnavailable",
        "EnvironmentReference",
        "ExecutableUnavailable",
        "FileContent",
        "FilesystemMode",
        "ForkSession",
        "FrozenJsonDict",
        "HeaderReference",
        "ImageContent",
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
        "NetworkMode",
        "NewSession",
        "NoNetworkAgentRuntime",
        "PermissionPolicy",
        "PermissionPolicyPatch",
        "ProtocolDefect",
        "ReasoningSpec",
        "ReasoningSummary",
        "ResumeSession",
        "ScriptedAgentRuntime",
        "SdkUnavailable",
        "SecretResolver",
        "SessionMetadata",
        "SessionMismatch",
        "SessionOpen",
        "SessionPage",
        "SessionQuery",
        "SessionReadOptions",
        "SessionSnapshot",
        "SessionSummary",
        "SessionUnavailable",
        "TextAgentOutput",
        "TextContent",
        "TurnNotStarted",
        "TurnRequest",
        "UnsafeConfirmation",
        "UnsafeDimension",
        "UnsupportedCapability",
        "freeze_json_object",
        "freeze_json_value",
        "ref_from_json",
        "ref_to_json",
        "thaw_json_value",
        "tool_is_allowed",
    }


def test_the_event_union_is_exported_whole() -> None:
    """The six kinds are one closed vocabulary; exporting a subset would publish half a grammar."""
    assert agent_runtime.AgentText is events.AgentText
    assert agent_runtime.AgentTerminal is events.AgentTerminal
    assert events.AGENT_EVENT_KINDS == (
        events.AgentText,
        events.AgentToolUse,
        events.AgentUsage,
        events.AgentPermissionRequest,
        events.AgentNative,
        events.AgentTerminal,
    )
    for kind in events.AGENT_EVENT_KINDS:
        assert kind.__name__ in agent_runtime.__all__, f"{kind.__name__} must be exported"


def test_agent_runtime_is_not_a_root_compatibility_surface() -> None:
    root_all = getattr(provider_runtime, "__all__", ())
    assert "AgentRuntime" not in root_all
    assert "AgentSessionRequest" not in root_all
    assert "CallTarget" not in root_all


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
        "_private_files",
        "_process",
        "_structured_output",
    ):
        assert name not in agent_runtime.__all__, f"{name} must stay unexported"


def test_deleted_modules_stay_absent() -> None:
    for module in ("agent", "agents", "agent_client", "agent_facade", "agent_runtime.capabilities"):
        try:
            importlib.import_module(f"provider_runtime.{module}")
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"deleted module provider_runtime.{module} is importable")


def test_no_name_becomes_public_on_the_agent_package_without_being_declared() -> None:
    """`__all__` is the surface, not a summary of it.

    The exact-surface set above is hand-written, so it only catches a name someone removed or
    renamed. This catches the other direction, where a private helper is promoted and
    re-exported without being declared, because it needs no maintenance to stay true.
    Submodules bound by the package's own imports are not exports;
    `test_adapter_and_private_modules_are_not_agent_exports` owns those.
    """
    undeclared = sorted(
        name
        for name, value in vars(agent_runtime).items()
        if not name.startswith("_")
        and name not in set(agent_runtime.__all__)
        and not isinstance(value, ModuleType)
    )

    assert undeclared == [], f"public agent names missing from __all__: {undeclared}"
