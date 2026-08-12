from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from provider_runtime.agent_runtime import (
    AgentFailure,
    AgentNative,
    AgentQuotaExhausted,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentSessionRequest,
    AgentTerminal,
    AgentText,
    AgentToolUse,
    AgentUsage,
    CodexNativeOptions,
    CredentialRef,
    ForkSession,
    JsonSchemaAgentOutput,
    McpServerSpec,
    NewSession,
    PermissionPolicy,
    PermissionPolicyPatch,
    ReasoningSpec,
    ResumeSession,
    SessionMetadata,
    SessionQuery,
    SessionReadOptions,
    SessionSnapshot,
    TextContent,
    TurnRequest,
    UnsafeConfirmation,
)
from provider_runtime.agent_runtime import codex_sdk as codex_sdk_module
from provider_runtime.agent_runtime._codex_launcher import ensure_codex_launcher
from provider_runtime.agent_runtime.codex_sdk import CodexSdkAdapter
from provider_runtime.agent_runtime.errors import (
    CredentialRejected,
    CredentialUnavailable,
    ProtocolDefect,
    SdkUnavailable,
    SessionUnavailable,
    UnsupportedCapability,
)
from provider_runtime.types import Absent, Present, TokenUsage

ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class FakeTextInput:
    text: str


@dataclass(slots=True)
class FakeLocalImageInput:
    path: str


@dataclass(slots=True)
class FakeConfig:
    codex_bin: str | None = None
    config_overrides: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] | None = None
    client_name: str = ""
    client_title: str = ""
    client_version: str = ""


def sdk_state(module: ModuleType) -> dict[str, Any]:
    return cast(dict[str, Any], module.state)  # type: ignore[attr-defined]


def notification(method: str, payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(method=method, payload=payload)


class FakeTurn:
    def __init__(self, module: ModuleType, thread_id: str, prompt: str, kwargs: dict[str, object]):
        state = sdk_state(module)
        state["turn_counter"] += 1
        self.id = f"turn-{state['turn_counter']}"
        self._module = module
        self._thread_id = thread_id
        self._prompt = prompt
        self._kwargs = kwargs
        self._interrupted = asyncio.Event()

    async def interrupt(self) -> dict[str, object]:
        self._interrupted.set()
        sdk_state(self._module)["interrupts"].append(self.id)
        return {}

    def _scoped(self, extra: dict[str, object]) -> dict[str, object]:
        return {"threadId": self._thread_id, "turnId": self.id, **extra}

    def _turn_completed(self, status: str, **extra: object) -> SimpleNamespace:
        return notification(
            "turn/completed",
            {
                "threadId": self._thread_id,
                "turn": {"id": self.id, "items": [], "status": status, **extra},
            },
        )

    async def stream(self):
        if self._prompt == "backend failure":
            raise RuntimeError("backend unavailable")
        if self._prompt == "quota text failure":
            raise RuntimeError("Monthly usage limit reached for this account")

        yield notification(
            "turn/started",
            {
                "threadId": self._thread_id,
                "turn": {"id": self.id, "items": [], "status": "inProgress"},
            },
        )
        if self._prompt == "hang":
            await self._interrupted.wait()
            yield self._turn_completed("interrupted")
            return
        if self._prompt == "interrupted":
            yield self._turn_completed("interrupted")
            return
        if self._prompt == "identity drift":
            yield notification(
                "item/agentMessage/delta",
                {
                    "threadId": "thread-intruder",
                    "turnId": self.id,
                    "itemId": "message-1",
                    "delta": "hello",
                },
            )
            return
        if self._prompt == "oversized tool output":
            yield notification(
                "item/completed",
                self._scoped(
                    {
                        "item": {
                            "id": "command-oversized",
                            "type": "commandExecution",
                            "status": "completed",
                            "aggregatedOutput": "x" * 2_048,
                        }
                    }
                ),
            )
            return
        if self._prompt == "quota error latch":
            yield notification(
                "error",
                self._scoped(
                    {
                        "willRetry": True,
                        "error": {
                            "message": "usage limit reached",
                            "codexErrorInfo": "usageLimitExceeded",
                        },
                    }
                ),
            )
            yield self._turn_completed("failed")
            return
        if self._prompt == "quota turn error":
            yield self._turn_completed(
                "failed",
                error={"message": "budget spent", "codexErrorInfo": "sessionBudgetExceeded"},
            )
            return
        if self._prompt == "failed turn":
            yield self._turn_completed("failed", error={"message": "backend fell over"})
            return
        if self._prompt.startswith("mcp"):
            item: dict[str, object] = {
                "id": "mcp-1",
                "type": "mcpToolCall",
                "server": "rogue" if self._prompt == "mcp unconfigured" else "docs",
                "tool": "delete" if self._prompt == "mcp denied tool" else "search",
                "status": "inProgress",
                "arguments": {"query": "usage"},
            }
            yield notification("item/started", self._scoped({"item": item}))
            if self._prompt == "mcp unfinished":
                yield self._turn_completed("completed")
                return
            yield notification(
                "item/mcpToolCall/progress",
                self._scoped({"itemId": "mcp-1", "message": "searching"}),
            )
            yield notification(
                "item/completed",
                self._scoped({"item": {**item, "status": "completed", "result": {"hits": 2}}}),
            )
            yield self._turn_completed("completed")
            return
        if self._prompt == "declined patch":
            change = {"path": "README.md", "kind": {"type": "update"}, "diff": "@@ declined"}
            patch = {
                "id": "patch-1",
                "type": "fileChange",
                "changes": [change],
                "status": "inProgress",
            }
            yield notification("item/started", self._scoped({"item": patch}))
            yield notification(
                "item/completed",
                self._scoped({"item": {**patch, "status": "declined"}}),
            )
            yield self._turn_completed("completed")
            return
        if self._prompt == "rich":
            yield notification(
                "item/reasoning/summaryTextDelta",
                self._scoped({"itemId": "reason-1", "summaryIndex": 0, "delta": "Inspecting"}),
            )
            command = {
                "id": "command-1",
                "type": "commandExecution",
                "command": "git status --short",
                "cwd": "/repo",
                "status": "inProgress",
            }
            yield notification("item/started", self._scoped({"item": command}))
            yield notification(
                "item/commandExecution/outputDelta",
                self._scoped({"itemId": "command-1", "delta": "clean\n"}),
            )
            yield notification(
                "item/completed",
                self._scoped(
                    {"item": {**command, "status": "completed", "aggregatedOutput": "clean\n"}}
                ),
            )
            change = {"path": "README.md", "kind": {"type": "update"}, "diff": "@@ changed"}
            patch = {
                "id": "patch-1",
                "type": "fileChange",
                "changes": [change],
                "status": "inProgress",
            }
            yield notification("item/started", self._scoped({"item": patch}))
            yield notification(
                "item/fileChange/patchUpdated",
                self._scoped({"itemId": "patch-1", "changes": [change]}),
            )
            yield notification(
                "item/completed",
                self._scoped({"item": {**patch, "status": "completed"}}),
            )

        text = (
            "not strict json {"
            if self._prompt == "schema violation"
            else '{"answer":"ok"}'
            if self._kwargs.get("output_schema") is not None
            else "Second turn."
            if self._prompt == "second"
            else "Inspection complete."
        )
        yield notification(
            "item/agentMessage/delta",
            self._scoped({"itemId": "message-1", "delta": text}),
        )
        usage_total: dict[str, object] = (
            {
                "inputTokens": 100,
                "cachedInputTokens": 40,
                "cacheWriteInputTokens": 10,
                "outputTokens": 20,
                "reasoningOutputTokens": 5,
                "totalTokens": 120,
            }
            if self._prompt == "rich"
            else {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}
        )
        yield notification(
            "thread/tokenUsage/updated",
            self._scoped({"tokenUsage": {"last": dict(usage_total), "total": usage_total}}),
        )
        if self._prompt == "rich":
            yield notification("account/rateLimits/updated", {"rateLimits": {"primary": 10}})
            yield notification(
                "thread/status/changed", {"threadId": self._thread_id, "status": "idle"}
            )
        yield self._turn_completed("completed")


class FakeThread:
    def __init__(self, module: ModuleType, thread_id: str):
        self._module = module
        self.id = thread_id

    async def turn(self, inputs: list[object], **kwargs: object) -> FakeTurn:
        prompt = "\n".join(item.text for item in inputs if isinstance(item, FakeTextInput))
        sdk_state(self._module)["turn_calls"].append(
            {"thread_id": self.id, "inputs": inputs, "kwargs": kwargs}
        )
        return FakeTurn(self._module, self.id, prompt, dict(kwargs))

    async def read(self, *, include_turns: bool = False) -> dict[str, object]:
        threads = cast(dict[str, dict[str, object]], sdk_state(self._module)["threads"])
        thread = dict(threads[self.id])
        thread["turns"] = [] if not include_turns else thread.get("turns", [])
        return {"thread": thread}


def fake_sdk(
    *,
    account_type: str | None = "chatgpt",
    version: str = "0.144.4",
    server_version: str = "0.144.4 (Ubuntu 24.4.0; x86_64) unknown",
) -> ModuleType:
    module = ModuleType("openai_codex")
    module.__dict__.update(
        {
            "__version__": version,
            "TextInput": FakeTextInput,
            "LocalImageInput": FakeLocalImageInput,
            "CodexConfig": FakeConfig,
            "ApprovalMode": SimpleNamespace(deny_all="deny_all", auto_review="auto_review"),
            "Sandbox": SimpleNamespace(
                read_only="read-only",
                workspace_write="workspace-write",
                full_access="full-access",
            ),
        }
    )
    module.state = {  # type: ignore[attr-defined]
        "account_type": account_type,
        "server_version": server_version,
        "clients": [],
        "calls": [],
        "threads": {},
        "thread_counter": 0,
        "turn_counter": 0,
        "turn_calls": [],
        "interrupts": [],
    }

    class FakeAsyncCodex:
        def __init__(self, config: FakeConfig):
            self.config = config
            self.metadata = {
                "serverInfo": {"name": "codex", "version": sdk_state(module)["server_version"]},
                "userAgent": "codex/0.144.4",
                "platformFamily": "unix",
                "platformOs": "linux",
            }
            self.closed = False
            sdk_state(module)["clients"].append(self)

        async def __aenter__(self):
            return self

        async def close(self) -> None:
            self.closed = True

        async def account(self) -> dict[str, object]:
            kind = sdk_state(module)["account_type"]
            if kind is None:
                return {"account": None, "requiresOpenaiAuth": True}
            return {
                "account": {"type": kind, "email": "person@example.com"},
                "requiresOpenaiAuth": True,
            }

        async def thread_list(self, **kwargs: object) -> dict[str, object]:
            state = sdk_state(module)
            state["calls"].append(("list", kwargs))
            threads = cast(dict[str, dict[str, object]], state["threads"])
            return {"data": list(threads.values()), "nextCursor": None}

        async def thread_start(self, **kwargs: object) -> FakeThread:
            state = sdk_state(module)
            state["thread_counter"] += 1
            thread_id = f"thread-{state['thread_counter']}"
            state["calls"].append(("start", kwargs))
            state["threads"][thread_id] = {
                "id": thread_id,
                "cwd": kwargs["cwd"],
                "name": "Fixture thread",
                "cliVersion": "0.144.4",
                "turns": [],
            }
            return FakeThread(module, thread_id)

        async def thread_resume(self, thread_id: str, **kwargs: object) -> FakeThread:
            state = sdk_state(module)
            state["calls"].append(("resume", {"thread_id": thread_id, **kwargs}))
            if thread_id not in state["threads"]:
                state["threads"][thread_id] = {
                    "id": thread_id,
                    "cwd": self.config.cwd or "/repo",
                    "name": "Fixture thread",
                    "cliVersion": "0.144.4",
                    "turns": [],
                }
            return FakeThread(module, thread_id)

        async def thread_fork(self, thread_id: str, **kwargs: object) -> FakeThread:
            state = sdk_state(module)
            state["thread_counter"] += 1
            fork_id = f"thread-{state['thread_counter']}"
            state["calls"].append(("fork", {"thread_id": thread_id, **kwargs}))
            state["threads"][fork_id] = {
                "id": fork_id,
                "cwd": kwargs["cwd"],
                "name": "Fixture fork",
                "cliVersion": "0.144.4",
                "turns": [],
            }
            return FakeThread(module, fork_id)

    module.__dict__["AsyncCodex"] = FakeAsyncCodex
    return module


def fake_runtime_package(*, version: str = "0.144.4") -> ModuleType:
    module = ModuleType("codex_cli_bin")
    module.__dict__.update(
        {
            "__version__": version,
            "PACKAGE_NAME": "openai-codex-cli-bin",
            "bundled_codex_path": lambda: Path(sys.executable),
            "bundled_path_dir": lambda: None,
        }
    )
    return module


def install_codex_modules(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    runtime_module: ModuleType,
) -> None:
    original = importlib.import_module

    def load(name: str, package: str | None = None) -> ModuleType:
        if name == "openai_codex":
            return module
        if name == "codex_cli_bin":
            return runtime_module
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", load)


@pytest.fixture
def installed_codex_sdk(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = fake_sdk()
    install_codex_modules(monkeypatch, module, fake_runtime_package())
    return module


def auth() -> CredentialRef:
    return CredentialRef(kind="local_account", profile_key="personal")


def request(tmp_path: Path, **changes: object) -> AgentSessionRequest:
    value = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=auth(),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=PermissionPolicy(allowed_tools=("*",)),
        model="native-model",
    )
    return replace(value, **changes)


def runtime(tmp_path: Path) -> AgentRuntime:
    state = tmp_path / "state"
    state.mkdir(mode=0o700, exist_ok=True)
    # The Claude adapter is a separate lane; the Codex tests register only their own route.
    return AgentRuntime(
        AgentRuntimeConfig(state_root_base=state),
        adapters=(CodexSdkAdapter(),),
    )


def turn(prompt: str, **changes: object) -> TurnRequest:
    return replace(TurnRequest(input=(TextContent(prompt),)), **changes)


def mcp_policy() -> PermissionPolicy:
    return PermissionPolicy(
        filesystem="full_access",
        network="unrestricted",
        allowed_tools=("*",),
        unsafe_confirmation=UnsafeConfirmation(("filesystem_full_access", "network_unrestricted")),
    )


def docs_server() -> McpServerSpec:
    return McpServerSpec(
        name="docs",
        transport="streamable_http",
        url="https://mcp.example.com/api",
        allowed_tools=("search",),
        denied_tools=("delete",),
    )


def start_call(module: ModuleType) -> dict[str, Any]:
    return next(payload for name, payload in sdk_state(module)["calls"] if name == "start")


def payload_dict(event: AgentToolUse) -> dict[str, object]:
    payload = event.payload
    assert isinstance(payload, Mapping), f"expected an object payload, got {payload!r}"
    return dict(payload)


def test_codex_launcher_replaces_the_sdk_merged_environment(tmp_path: Path) -> None:
    backend_root = tmp_path / "codex"
    backend_root.mkdir(mode=0o700)
    state_root = backend_root / "personal"
    state_root.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.write_text(
        f"#!{sys.executable} -I\n"
        "import json, os, pathlib, sys\n"
        "parent_env = pathlib.Path(f'/proc/{os.getppid()}/environ').read_bytes().decode(errors='replace')\n"
        "print(json.dumps({'env': dict(os.environ), 'parent_env': parent_env, "
        "'args': sys.argv[1:]}), flush=True)\n",
        encoding="utf-8",
    )
    target.chmod(0o700)
    launcher = ensure_codex_launcher(
        state_root,
        target,
        ("KEEP",),
        interpreter=sys.executable,
    )
    environment = dict(os.environ)
    environment.update({"KEEP": "selected", "MUST_NOT_LEAK": "ambient-secret"})

    completed = subprocess.run(
        (str(launcher), "owned-argument"),
        env=environment,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert payload["env"]["KEEP"] == "selected"
    assert "MUST_NOT_LEAK" not in payload["env"]
    assert "ambient-secret" not in payload["parent_env"]
    assert payload["args"] == ["owned-argument"]
    assert completed.returncode == -signal.SIGKILL


@pytest.mark.skipif(
    importlib.util.find_spec("openai_codex") is not None,
    reason="the codex-sdk extra is installed; the no-extras CI job runs this by node id",
)
async def test_absent_sdk_extra_is_sdk_unavailable(tmp_path: Path) -> None:
    with pytest.raises(SdkUnavailable, match="install the 'codex-sdk' extra"):
        async with runtime(tmp_path) as selected:
            await selected.open_session(request(tmp_path))


async def test_open_session_points_the_sdk_at_the_owned_launcher(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        assert session.ref.native_session_id == "thread-1"

    config = cast(FakeConfig, sdk_state(installed_codex_sdk)["clients"][0].config)
    assert config.cwd == str(tmp_path.resolve())
    assert config.codex_bin is not None, "the SDK must be pointed at the owned launcher"
    launcher = Path(config.codex_bin)
    assert launcher.name.startswith("codex-launcher-"), f"unexpected codex_bin {launcher}"
    source = launcher.read_text(encoding="utf-8")
    assert config.env is not None
    for name in config.env:
        assert f"'{name}'" in source, f"launcher must embed the child environment name {name}"
    assert config.env["CODEX_HOME"].endswith("codex/personal")


async def test_disabled_builtin_tools_emit_the_complete_certified_codex_policy(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        await selected.open_session(
            request(
                tmp_path,
                native=CodexNativeOptions(web_search=False, builtin_tools="disabled"),
            )
        )

    config = cast(FakeConfig, sdk_state(installed_codex_sdk)["clients"][0].config)
    assert config.config_overrides == ('forced_login_method="chatgpt"',)
    assert start_call(installed_codex_sdk)["config"] == {
        "apps": {"_default": {"enabled": False}},
        "features": {
            "apply_patch_streaming_events": False,
            "apps": False,
            "artifact": False,
            "auth_elicitation": False,
            "browser_use": False,
            "browser_use_external": False,
            "browser_use_full_cdp_access": False,
            "code_mode": False,
            "code_mode_host": False,
            "code_mode_only": False,
            "computer_use": False,
            "chronicle": False,
            "current_time_reminder": False,
            "default_mode_request_user_input": False,
            "deferred_executor": False,
            "enable_fanout": False,
            "enable_mcp_apps": False,
            "exec_permission_approvals": False,
            "goals": False,
            "guardian_approval": False,
            "hooks": False,
            "image_generation": False,
            "in_app_browser": False,
            "memories": False,
            "mentions_v2": False,
            "multi_agent": False,
            "multi_agent_v2": False,
            "non_prefixed_mcp_tool_names": False,
            "plugins": False,
            "plugin_sharing": False,
            "remote_plugin": False,
            "request_permissions_tool": False,
            "rollout_budget": False,
            "shell_snapshot": False,
            "shell_tool": False,
            "shell_zsh_fork": False,
            "skill_mcp_dependency_install": False,
            "standalone_web_search": False,
            "terminal_visualization_instructions": False,
            "token_budget": False,
            "tool_call_mcp_elicitation": False,
            "tool_suggest": False,
            "unified_exec": False,
            "unified_exec_zsh_fork": False,
            "web_search_cached": False,
            "web_search_request": False,
            "workspace_dependencies": False,
        },
        "include_apps_instructions": False,
        "include_collaboration_mode_instructions": False,
        "include_environment_context": False,
        "include_permissions_instructions": False,
        "mcp_servers": {},
        "shell_environment_policy": {"exclude": [], "inherit": "core"},
        "skills": {"bundled": {"enabled": False}, "include_instructions": False},
        "tools": {"experimental_request_user_input": {"enabled": False}},
        "web_search": "disabled",
    }


@pytest.mark.parametrize(
    ("sdk_version", "runtime_version", "server_version"),
    (
        ("0.145.0", "0.145.0", "0.145.0 (linux; x86_64)"),
        ("0.144.4", "0.145.0", "0.145.0 (linux; x86_64)"),
        ("0.144.4", "0.144.4", "0.145.0 (linux; x86_64)"),
    ),
)
async def test_disabled_builtin_tools_reject_uncertified_runtime_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sdk_version: str,
    runtime_version: str,
    server_version: str,
) -> None:
    module = fake_sdk(version=sdk_version, server_version=server_version)
    install_codex_modules(monkeypatch, module, fake_runtime_package(version=runtime_version))

    with pytest.raises(UnsupportedCapability, match="builtin tool policy"):
        async with runtime(tmp_path) as selected:
            await selected.open_session(
                request(tmp_path, native=CodexNativeOptions(builtin_tools="disabled"))
            )

    assert all(client.closed for client in sdk_state(module)["clients"]), (
        "a client opened before server-version rejection must still be closed"
    )


async def test_api_key_session_auth_is_rejected_before_any_client_starts(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    api_key_auth = CredentialRef(
        kind="api_key_environment", profile_key="personal", name="OPENAI_API_KEY"
    )
    with pytest.raises(UnsupportedCapability, match="local ChatGPT auth"):
        async with runtime(tmp_path) as selected:
            await selected.open_session(request(tmp_path, auth=api_key_auth))

    assert sdk_state(installed_codex_sdk)["clients"] == [], (
        "an API-key credential must be refused before the SDK client starts"
    )


async def test_non_chatgpt_account_is_credential_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = fake_sdk(account_type="apiKey")
    install_codex_modules(monkeypatch, module, fake_runtime_package())
    with pytest.raises(CredentialRejected, match="ChatGPT"):
        async with runtime(tmp_path) as selected:
            await selected.open_session(request(tmp_path))

    clients = sdk_state(module)["clients"]
    assert clients and all(client.closed for client in clients), (
        "the probing client must be closed after the rejection"
    )


async def test_missing_local_account_is_credential_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_codex_modules(monkeypatch, fake_sdk(account_type=None), fake_runtime_package())
    with pytest.raises(CredentialUnavailable, match="no authenticated local account"):
        async with runtime(tmp_path) as selected:
            await selected.open_session(request(tmp_path))


async def test_sdk_error_translation_drops_the_untrusted_exception_chain(
    tmp_path: Path,
    installed_codex_sdk: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "Bearer fixture-secret-value-that-must-not-leak"

    async def rejected(_self: object) -> dict[str, object]:
        raise RuntimeError(f"authorization={secret}")

    monkeypatch.setattr(installed_codex_sdk.AsyncCodex, "account", rejected)

    with pytest.raises(CredentialUnavailable) as captured:
        async with runtime(tmp_path) as selected:
            await selected.open_session(request(tmp_path))

    rendered = "".join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert secret not in rendered


async def test_policy_mappings_outside_the_codex_sandbox_presets_are_rejected(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        with pytest.raises(UnsupportedCapability, match="approval"):
            await selected.open_session(
                request(tmp_path, policy=PermissionPolicy(approval="ask", allowed_tools=("*",)))
            )
        with pytest.raises(UnsupportedCapability, match="read_only sandbox has no network"):
            await selected.open_session(
                request(
                    tmp_path,
                    policy=PermissionPolicy(
                        filesystem="read_only",
                        network="unrestricted",
                        allowed_tools=("*",),
                        unsafe_confirmation=UnsafeConfirmation(("network_unrestricted",)),
                    ),
                )
            )
        with pytest.raises(UnsupportedCapability, match="full_access"):
            await selected.open_session(
                request(
                    tmp_path,
                    policy=PermissionPolicy(
                        filesystem="full_access",
                        allowed_tools=("*",),
                        unsafe_confirmation=UnsafeConfirmation(("filesystem_full_access",)),
                    ),
                )
            )
        with pytest.raises(UnsupportedCapability, match="allowlist"):
            await selected.open_session(
                request(
                    tmp_path,
                    policy=PermissionPolicy(
                        network="allowlist",
                        network_allowlist=("api.example.com",),
                        allowed_tools=("*",),
                    ),
                )
            )
        with pytest.raises(UnsupportedCapability, match="tool filters"):
            await selected.open_session(request(tmp_path, policy=PermissionPolicy()))

    assert sdk_state(installed_codex_sdk)["clients"] == [], (
        "unmappable policies must be refused before the SDK client starts"
    )


@pytest.mark.parametrize(
    ("policy", "network_access"),
    [
        (PermissionPolicy(filesystem="workspace_write", allowed_tools=("*",)), False),
        (
            PermissionPolicy(
                filesystem="workspace_write",
                network="unrestricted",
                allowed_tools=("*",),
                unsafe_confirmation=UnsafeConfirmation(("network_unrestricted",)),
            ),
            True,
        ),
    ],
    ids=("network_disabled", "network_unrestricted"),
)
async def test_workspace_write_probes_bubblewrap_and_maps_the_sandbox_network_toggle(
    tmp_path: Path,
    installed_codex_sdk: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    policy: PermissionPolicy,
    network_access: bool,
) -> None:
    async def unavailable(*, cwd: Path, environment: Mapping[str, str]) -> bool:
        return False

    monkeypatch.setattr(codex_sdk_module, "bubblewrap_network_namespace_available", unavailable)
    async with runtime(tmp_path) as selected:
        with pytest.raises(UnsupportedCapability, match="bubblewrap"):
            await selected.open_session(request(tmp_path, policy=policy))
    assert sdk_state(installed_codex_sdk)["clients"] == [], (
        "the probe must fail closed before any client or thread is started"
    )

    async def available(*, cwd: Path, environment: Mapping[str, str]) -> bool:
        return True

    monkeypatch.setattr(codex_sdk_module, "bubblewrap_network_namespace_available", available)
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path, policy=policy))
        result = await selected.run_turn(session, turn("plain"))

    assert result.status == "succeeded"
    start = start_call(installed_codex_sdk)
    assert start["sandbox"] == "workspace-write"
    assert start["config"]["sandbox_workspace_write"] == {
        "writable_roots": [str(tmp_path.resolve())],
        "network_access": network_access,
    }


async def test_sdk_version_drift_warns_and_the_client_still_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = fake_sdk(version="0.145.0", server_version="0.145.0 (linux; x86_64)")
    install_codex_modules(monkeypatch, module, fake_runtime_package(version="0.145.0"))

    with pytest.warns(RuntimeWarning, match="certified"):
        async with runtime(tmp_path) as selected:
            session = await selected.open_session(request(tmp_path))
            result = await selected.run_turn(session, turn("plain"))

    assert result.status == "succeeded", "version drift must not block the behavioral probe"


async def test_runtime_package_version_drift_warns_and_the_client_still_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = fake_sdk(server_version="0.145.0 (linux; x86_64)")
    install_codex_modules(monkeypatch, module, fake_runtime_package(version="0.145.0"))

    with pytest.warns(RuntimeWarning, match="bundled runtime"):
        async with runtime(tmp_path) as selected:
            session = await selected.open_session(request(tmp_path))
            result = await selected.run_turn(session, turn("plain"))

    assert result.status == "succeeded"


async def test_server_version_drift_warns_and_the_client_still_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = fake_sdk(server_version="0.146.0 (Ubuntu 24.4.0; x86_64) unknown")
    install_codex_modules(monkeypatch, module, fake_runtime_package())

    with pytest.warns(RuntimeWarning, match="server reported"):
        async with runtime(tmp_path) as selected:
            session = await selected.open_session(request(tmp_path))
            result = await selected.run_turn(session, turn("plain"))

    assert result.status == "succeeded"


async def test_a_full_turn_streams_the_closed_event_grammar(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        events = [event async for event in selected.stream_turn(session, turn("rich"))]
        second = await selected.run_turn(session, turn("second"))

    kinds = [type(event).__name__ for event in events]
    assert kinds == [
        "AgentNative",  # turn/started
        "AgentNative",  # reasoning summary delta
        "AgentToolUse",  # commandExecution started
        "AgentToolUse",  # commandExecution updated
        "AgentToolUse",  # commandExecution completed
        "AgentToolUse",  # fileChange started
        "AgentToolUse",  # fileChange patchUpdated
        "AgentToolUse",  # fileChange completed
        "AgentText",
        "AgentUsage",
        "AgentNative",  # unknown method passthrough
        "AgentNative",  # thread/status/changed
        "AgentNative",  # turn/completed
        "AgentTerminal",
    ], f"unexpected event sequence: {kinds}"

    # Every native frame without a first-class kind travels, with no per-method noise filter:
    # a method this adapter has no opinion about is exactly what AgentNative is for.
    natives = [event for event in events if isinstance(event, AgentNative)]
    assert [native.native_type for native in natives] == [
        "turn/started",
        "item/reasoning/summaryTextDelta",
        "account/rateLimits/updated",
        "thread/status/changed",
        "turn/completed",
    ]

    tool_events = [event for event in events if isinstance(event, AgentToolUse)]
    assert [(event.name, event.phase) for event in tool_events] == [
        ("commandExecution", "started"),
        ("commandExecution", "updated"),
        ("commandExecution", "completed"),
        ("fileChange", "started"),
        ("fileChange", "updated"),
        ("fileChange", "completed"),
    ]
    command_started, command_updated, command_completed = tool_events[:3]
    assert command_started.tool_call_id == "command-1"
    assert payload_dict(command_started) == {"command": "git status --short", "cwd": "/repo"}
    assert command_started.succeeded is None, "succeeded exists only on completed tool use"
    assert payload_dict(command_updated) == {"output_delta": "clean\n"}
    assert command_completed.payload == "clean\n"
    assert command_completed.succeeded is True

    patch_started, patch_updated, patch_completed = tool_events[3:]
    started_payload = payload_dict(patch_started)
    assert started_payload["status"] == "in_progress"
    changes = started_payload["changes"]
    assert isinstance(changes, tuple) and len(changes) == 1
    change = cast(Mapping[str, object], changes[0])
    assert change["path"] == "README.md" and change["diff"] == "@@ changed"
    assert "changes" in payload_dict(patch_updated)
    assert patch_completed.succeeded is True

    text_events = [event for event in events if isinstance(event, AgentText)]
    assert [event.text for event in text_events] == ["Inspection complete."]

    expected_usage = TokenUsage(
        input_tokens=100,  # cache-inclusive on the OpenAI wire
        output_tokens=20,
        total_tokens=120,
        reasoning_tokens=Present(5),
        cache_read_input_tokens=Present(40),
        cache_write_input_tokens=Present(10),
    )
    usage_events = [event for event in events if isinstance(event, AgentUsage)]
    assert [event.usage for event in usage_events] == [expected_usage]

    terminal = events[-1]
    assert isinstance(terminal, AgentTerminal)
    assert terminal.status == "succeeded"
    assert terminal.failure is None
    assert terminal.final_text == "Inspection complete."
    assert terminal.session_ref == session.ref
    assert terminal.usage == Present(expected_usage)
    assert terminal.structured_output is None

    assert second.final_text == "Second turn.", "per-turn accumulation state must reset"
    assert second.usage == Present(
        TokenUsage(
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Absent(),
            cache_write_input_tokens=Absent(),
        )
    ), "counts the wire omits must be absent, not zero"


async def test_structured_output_passes_the_plain_schema_through_natively(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    output = JsonSchemaAgentOutput(name="answer", schema=ANSWER_SCHEMA)
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(
            request(tmp_path, output=output, reasoning=ReasoningSpec(effort="low"))
        )
        result = await selected.run_turn(session, turn("plain"))

    assert result.status == "succeeded"
    assert result.final_text == '{"answer":"ok"}'
    structured = cast(Mapping[str, object], result.structured_output)
    assert dict(structured) == {"answer": "ok"}
    turn_kwargs = sdk_state(installed_codex_sdk)["turn_calls"][0]["kwargs"]
    assert turn_kwargs["output_schema"] == ANSWER_SCHEMA, (
        "the plain JSON Schema must reach the SDK unchanged"
    )
    assert turn_kwargs["effort"] == "low"
    start = start_call(installed_codex_sdk)
    assert start["approval_mode"] == "deny_all"
    assert start["sandbox"] == "read-only"


async def test_structured_output_violation_is_a_failed_terminal_value(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    output = JsonSchemaAgentOutput(name="answer", schema=ANSWER_SCHEMA)
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path, output=output))
        result = await selected.run_turn(session, turn("schema violation"))

    assert result.status == "failed"
    assert result.failure == AgentFailure("output_schema_violation")
    assert result.final_text == "not strict json {"
    assert result.structured_output is None


async def test_provider_review_maps_to_the_public_auto_reviewer(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    policy = PermissionPolicy(approval="provider_review", allowed_tools=("*",))
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path, policy=policy))
        result = await selected.run_turn(session, turn("plain"))

    assert result.status == "succeeded"
    assert start_call(installed_codex_sdk)["approval_mode"] == "auto_review"


async def test_quota_exhaustion_latches_from_error_notifications(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        events = [event async for event in selected.stream_turn(session, turn("quota error latch"))]

    error_natives = [
        event for event in events if isinstance(event, AgentNative) and event.native_type == "error"
    ]
    assert len(error_natives) == 1, f"the error frame must surface as AgentNative: {events}"
    terminal = events[-1]
    assert isinstance(terminal, AgentTerminal)
    assert terminal.status == "failed"
    assert terminal.failure == AgentQuotaExhausted()
    assert "usage limit reached" in terminal.diagnostics


async def test_quota_exhaustion_reads_the_failed_turn_error(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        result = await selected.run_turn(session, turn("quota turn error"))

    assert result.status == "failed"
    assert result.failure == AgentQuotaExhausted()


async def test_non_quota_turn_failures_are_backend_failed(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        result = await selected.run_turn(session, turn("failed turn"))

    assert result.status == "failed"
    assert result.failure == AgentFailure("backend_failed")


async def test_backend_exceptions_become_failed_terminal_values(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        result = await selected.run_turn(session, turn("backend failure"))
    assert result.status == "failed"
    assert result.failure == AgentFailure("backend_failed")
    assert "backend unavailable" in result.diagnostics

    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        result = await selected.run_turn(session, turn("quota text failure"))
    assert result.status == "failed"
    assert result.failure == AgentQuotaExhausted(), (
        "quota-shaped backend exceptions must map to the quota terminal value"
    )


async def test_native_interruption_maps_to_a_cancelled_terminal(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        result = await selected.run_turn(session, turn("interrupted"))

    assert result.status == "cancelled"
    assert result.failure is None


async def test_runtime_timeouts_interrupt_the_native_turn(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        result = await selected.run_turn(session, turn("hang", timeout_seconds=0.05))

    assert result.status == "failed"
    assert result.failure == AgentFailure("turn_timeout")
    assert sdk_state(installed_codex_sdk)["interrupts"], (
        "the timeout must reach the SDK as a native turn interrupt"
    )


async def test_oversized_native_output_fails_closed_and_releases_the_session(
    tmp_path: Path,
    installed_codex_sdk: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex_sdk_module, "_MAX_MESSAGE_BYTES", 1_024)

    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        result = await selected.run_turn(session, turn("oversized tool output"))
        with pytest.raises(SessionUnavailable):
            await selected.run_turn(session, turn("plain"))

    assert result.status == "failed"
    assert result.failure == AgentFailure("output_limit_exceeded")
    assert sdk_state(installed_codex_sdk)["interrupts"], (
        "the oversized turn must be interrupted natively before teardown"
    )


async def test_thread_identity_drift_is_a_protocol_defect(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        with pytest.raises(ProtocolDefect, match="thread identity"):
            await selected.run_turn(session, turn("identity drift"))


async def test_mcp_tool_calls_carry_their_registered_identity(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(
            request(tmp_path, policy=mcp_policy(), mcp_servers=(docs_server(),))
        )
        events = [event async for event in selected.stream_turn(session, turn("mcp"))]

    tool_events = [event for event in events if isinstance(event, AgentToolUse)]
    assert [(event.phase, event.name) for event in tool_events] == [
        ("started", "docs/search"),
        ("updated", "docs/search"),
        ("completed", "docs/search"),
    ], f"MCP tool events must carry the registered server/tool identity: {tool_events}"
    started, progress, completed = tool_events
    assert started.tool_call_id == "mcp-1"
    assert payload_dict(started) == {"query": "usage"}
    assert payload_dict(progress) == {"message": "searching"}
    assert completed.succeeded is True
    assert payload_dict(completed) == {"hits": 2}

    start = start_call(installed_codex_sdk)
    assert start["config"]["mcp_servers"] == {
        "docs": {
            "url": "https://mcp.example.com/api",
            "required": True,
            "enabled_tools": ["search"],
            "disabled_tools": ["delete"],
        }
    }


async def test_mcp_tool_calls_outside_the_configured_policy_are_defects(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    for prompt, match in (
        ("mcp unconfigured", "unconfigured server"),
        ("mcp denied tool", "exact tool policy"),
        ("mcp unfinished", "active MCP tool calls"),
    ):
        async with runtime(tmp_path) as selected:
            session = await selected.open_session(
                request(tmp_path, policy=mcp_policy(), mcp_servers=(docs_server(),))
            )
            with pytest.raises(ProtocolDefect, match=match):
                await selected.run_turn(session, turn(prompt))


async def test_declined_file_changes_complete_unsuccessfully(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        events = [event async for event in selected.stream_turn(session, turn("declined patch"))]

    tool_events = [event for event in events if isinstance(event, AgentToolUse)]
    assert [(event.name, event.phase) for event in tool_events] == [
        ("fileChange", "started"),
        ("fileChange", "completed"),
    ]
    started, completed = tool_events
    assert payload_dict(started)["status"] == "in_progress"
    assert completed.succeeded is False, "a declined patch is a completed action that did not apply"
    terminal = events[-1]
    assert isinstance(terminal, AgentTerminal) and terminal.status == "succeeded"


async def test_session_instructions_reach_both_native_channels(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    """Neither instruction may be dropped: the SDK has a channel for each.

    `base_instructions` replaces Codex's built-in base prompt and `developer_instructions`
    carries the developer-role text, so a session that names either runs with it applied.
    """
    async with runtime(tmp_path) as selected:
        await selected.open_session(
            request(
                tmp_path,
                system=(TextContent("never delete files"),),
                developer=(TextContent("prefer small diffs"),),
            )
        )

    start = start_call(installed_codex_sdk)
    assert start["base_instructions"] == "never delete files"
    assert start["developer_instructions"] == "prefer small diffs"


async def test_turn_overrides_and_caller_approvals_are_refused(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async def approve(_request: object) -> str:
        return "allow"

    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        with pytest.raises(UnsupportedCapability, match="approval callbacks"):
            await selected.run_turn(session, turn("plain"), approvals=cast(Any, approve))
        # The refused turn never became identifiable, so block-and-stop released the session.
        with pytest.raises(SessionUnavailable):
            await selected.run_turn(session, turn("plain"))

    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        # The narrowing algebra accepts the patch; the route still cannot apply it to a
        # thread that is already started, so the turn is refused before any billable work.
        with pytest.raises(UnsupportedCapability, match="reconfigure policy"):
            await selected.run_turn(
                session, turn("plain", policy=PermissionPolicyPatch(approval="deny"))
            )


async def test_resume_and_fork_enforce_native_thread_identity(
    tmp_path: Path,
    installed_codex_sdk: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        ref = session.ref

    async def resume_other(_self: object, thread_id: str, **_kwargs: object) -> FakeThread:
        return FakeThread(installed_codex_sdk, "thread-imposter")

    monkeypatch.setattr(installed_codex_sdk.AsyncCodex, "thread_resume", resume_other)
    with pytest.raises(SessionUnavailable, match="different native thread"):
        async with runtime(tmp_path) as selected:
            await selected.open_session(request(tmp_path, open=ResumeSession(ref)))

    async def fork_same(_self: object, thread_id: str, **_kwargs: object) -> FakeThread:
        return FakeThread(installed_codex_sdk, thread_id)

    monkeypatch.setattr(installed_codex_sdk.AsyncCodex, "thread_fork", fork_same)
    with pytest.raises(ProtocolDefect, match="fork did not mint"):
        async with runtime(tmp_path) as selected:
            await selected.open_session(request(tmp_path, open=ForkSession(ref)))


async def test_resume_fork_and_metadata_only_discovery(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        first = await selected.run_turn(session, turn("plain"))
        page = await selected.list_sessions(
            SessionQuery(backend="codex", transport="sdk", auth=auth())
        )
        snapshot = await selected.read_session(session.ref, SessionReadOptions(auth=auth()))
        resumed = await selected.open_session(request(tmp_path, open=ResumeSession(session.ref)))
        second = await selected.run_turn(resumed, turn("second"))
        forked = await selected.open_session(request(tmp_path, open=ForkSession(session.ref)))

    assert first.status == "succeeded" and first.final_text == "Inspection complete."
    assert second.final_text == "Second turn."
    assert [summary.ref.native_session_id for summary in page.sessions] == [
        session.ref.native_session_id
    ]
    assert page.sessions[0].metadata == SessionMetadata(name="Fixture thread")
    list_kwargs = next(
        kwargs for name, kwargs in sdk_state(installed_codex_sdk)["calls"] if name == "list"
    )
    assert list_kwargs == {"archived": None, "cursor": None, "limit": 50}, (
        "discovery must not forward native filters"
    )
    assert snapshot == SessionSnapshot(
        ref=session.ref, metadata=SessionMetadata(name="Fixture thread")
    ), "read_session is metadata-only"
    assert forked.ref.native_session_id != session.ref.native_session_id, (
        "a fork must mint a new native thread identity"
    )
