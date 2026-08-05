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
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from provider_runtime.agent_runtime import (
    AgentCapabilityScope,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentSessionRequest,
    CodexNativeOptions,
    CredentialRef,
    JsonSchemaAgentOutput,
    NewSession,
    PermissionPolicy,
    ReasoningSpec,
    ResumeSession,
    SessionQuery,
    SessionReadOptions,
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
    SdkUnavailable,
    UnsupportedCapability,
)
from provider_runtime.schema import parse_canonical_schema


@dataclass(slots=True)
class FakeTextInput:
    text: str


@dataclass(slots=True)
class FakeLocalImageInput:
    path: str


@dataclass(slots=True)
class FakeConfig:
    codex_bin: str | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None
    client_name: str = ""
    client_title: str = ""
    client_version: str = ""


class FakeTurn:
    def __init__(self, module: ModuleType, thread_id: str, prompt: str, kwargs: dict[str, object]):
        state = cast(dict[str, Any], module.state)  # type: ignore[attr-defined]
        state["turn_counter"] += 1
        self.id = f"turn-{state['turn_counter']}"
        self._module = module
        self._thread_id = thread_id
        self._prompt = prompt
        self._kwargs = kwargs
        self._interrupted = asyncio.Event()

    async def interrupt(self) -> dict[str, object]:
        self._interrupted.set()
        cast(dict[str, Any], self._module.state)["interrupts"].append(self.id)  # type: ignore[attr-defined]
        return {}

    async def stream(self):
        if self._prompt == "hang":
            await self._interrupted.wait()
            yield notification(
                "turn/completed",
                {
                    "threadId": self._thread_id,
                    "turn": {"id": self.id, "items": [], "status": "interrupted"},
                },
            )
            return
        if self._prompt == "backend failure":
            raise RuntimeError("backend unavailable")

        yield notification(
            "turn/started",
            {
                "threadId": self._thread_id,
                "turn": {"id": self.id, "items": [], "status": "inProgress"},
            },
        )
        if self._prompt == "oversized tool output":
            yield notification(
                "item/completed",
                {
                    "threadId": self._thread_id,
                    "turnId": self.id,
                    "item": {
                        "id": "command-oversized",
                        "type": "commandExecution",
                        "status": "completed",
                        "aggregatedOutput": "x" * 2_048,
                    },
                },
            )
            return
        if self._prompt == "rich":
            yield notification(
                "item/reasoning/summaryTextDelta",
                {
                    "threadId": self._thread_id,
                    "turnId": self.id,
                    "itemId": "reason-1",
                    "summaryIndex": 0,
                    "delta": "Inspecting",
                },
            )
            command = {
                "id": "command-1",
                "type": "commandExecution",
                "command": "git status --short",
                "cwd": "/repo",
                "status": "inProgress",
            }
            yield notification(
                "item/started",
                {"threadId": self._thread_id, "turnId": self.id, "item": command},
            )
            yield notification(
                "item/commandExecution/outputDelta",
                {
                    "threadId": self._thread_id,
                    "turnId": self.id,
                    "itemId": "command-1",
                    "delta": "clean\n",
                },
            )
            yield notification(
                "item/completed",
                {
                    "threadId": self._thread_id,
                    "turnId": self.id,
                    "item": {**command, "status": "completed", "aggregatedOutput": "clean\n"},
                },
            )
            change = {
                "path": "README.md",
                "kind": {"type": "update"},
                "diff": "@@ changed",
            }
            yield notification(
                "item/completed",
                {
                    "threadId": self._thread_id,
                    "turnId": self.id,
                    "item": {
                        "id": "patch-1",
                        "type": "fileChange",
                        "changes": [change],
                        "status": "completed",
                    },
                },
            )

        text = (
            '{"answer":"ok"}'
            if self._kwargs.get("output_schema") is not None
            else "Second turn."
            if self._prompt == "second"
            else "Inspection complete."
        )
        yield notification(
            "item/agentMessage/delta",
            {
                "threadId": self._thread_id,
                "turnId": self.id,
                "itemId": "message-1",
                "delta": text,
            },
        )
        yield notification(
            "thread/tokenUsage/updated",
            {
                "threadId": self._thread_id,
                "turnId": self.id,
                "tokenUsage": {
                    "last": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                    "total": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                },
            },
        )
        yield notification(
            "turn/completed",
            {
                "threadId": self._thread_id,
                "turn": {"id": self.id, "items": [], "status": "completed"},
            },
        )


class FakeThread:
    def __init__(self, module: ModuleType, thread_id: str):
        self._module = module
        self.id = thread_id

    async def turn(self, inputs: list[object], **kwargs: object) -> FakeTurn:
        prompt = "\n".join(item.text for item in inputs if isinstance(item, FakeTextInput))
        cast(dict[str, Any], self._module.state)["turn_calls"].append(  # type: ignore[attr-defined]
            {"thread_id": self.id, "inputs": inputs, "kwargs": kwargs}
        )
        return FakeTurn(self._module, self.id, prompt, dict(kwargs))

    async def read(self, *, include_turns: bool = False) -> dict[str, object]:
        threads = cast(dict[str, dict[str, object]], self._module.state["threads"])  # type: ignore[attr-defined]
        thread = dict(threads[self.id])
        thread["turns"] = [] if not include_turns else thread.get("turns", [])
        return {"thread": thread}


def notification(method: str, payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(method=method, payload=payload)


def fake_sdk(*, account_type: str = "chatgpt", version: str = "0.144.4") -> ModuleType:
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
                "serverInfo": {
                    "name": "codex",
                    "version": "0.144.4 (Ubuntu 24.4.0; x86_64) unknown",
                },
                "userAgent": "codex/0.144.4",
                "platformFamily": "unix",
                "platformOs": "linux",
            }
            self.closed = False
            cast(dict[str, Any], module.state)["clients"].append(self)  # type: ignore[attr-defined]

        async def __aenter__(self):
            return self

        async def close(self) -> None:
            self.closed = True

        async def account(self) -> dict[str, object]:
            kind = cast(dict[str, Any], module.state)["account_type"]  # type: ignore[attr-defined]
            return {
                "account": {"type": kind, "email": "person@example.com"},
                "requiresOpenaiAuth": True,
            }

        async def models(self) -> dict[str, object]:
            return {
                "data": [
                    {
                        "model": "native-model",
                        "supportedReasoningEfforts": [
                            {"reasoningEffort": "low"},
                            {"reasoningEffort": "high"},
                        ],
                    }
                ],
                "nextCursor": None,
            }

        async def thread_list(self, **kwargs: object) -> dict[str, object]:
            cast(dict[str, Any], module.state)["calls"].append(("list", kwargs))  # type: ignore[attr-defined]
            threads = cast(dict[str, dict[str, object]], module.state["threads"])  # type: ignore[attr-defined]
            return {"data": list(threads.values()), "nextCursor": None}

        async def thread_start(self, **kwargs: object) -> FakeThread:
            state = cast(dict[str, Any], module.state)  # type: ignore[attr-defined]
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
            state = cast(dict[str, Any], module.state)  # type: ignore[attr-defined]
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
            state = cast(dict[str, Any], module.state)  # type: ignore[attr-defined]
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


@pytest.fixture
def installed_codex_sdk(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = fake_sdk()
    runtime_module = fake_runtime_package()
    original = importlib.import_module

    def load(name: str, package: str | None = None) -> ModuleType:
        if name == "openai_codex":
            return module
        if name == "codex_cli_bin":
            return runtime_module
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", load)
    return module


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
    from dataclasses import replace

    return replace(value, **changes)


def runtime(tmp_path: Path) -> AgentRuntime:
    state = tmp_path / "state"
    state.mkdir(mode=0o700, exist_ok=True)
    return AgentRuntime(AgentRuntimeConfig(state_root_base=state))


async def test_capabilities_report_the_sdk_route_and_matched_versions(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        capabilities = await selected.capabilities(
            AgentCapabilityScope(backend="codex", transport="sdk", auth=auth())
        )

    assert capabilities.sdk_version == "0.144.4"
    assert capabilities.executable_version == "0.144.4"
    assert capabilities.session_operations == ("new", "resume", "fork")
    assert capabilities.discovery_operations == ("list", "read")
    assert capabilities.models == ("native-model",)
    assert capabilities.approval_modes == ("deny", "provider_review")
    assert capabilities.reports_effective_effort is False


async def test_sdk_session_streams_structured_output_and_reuses_subscription_auth(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    output = JsonSchemaAgentOutput(
        name="answer",
        schema=parse_canonical_schema(
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        ),
    )
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(
            request(tmp_path, output=output, reasoning=ReasoningSpec(effort="low"))
        )
        result = await selected.run_turn(session, TurnRequest(input=(TextContent("answer"),)))

    assert result.status == "succeeded"
    assert result.final_text == '{"answer":"ok"}'
    structured = cast(Mapping[str, object], result.structured_output)
    assert dict(structured) == {"answer": "ok"}
    calls = cast(dict[str, Any], installed_codex_sdk.state)["calls"]  # type: ignore[attr-defined]
    start = next(payload for name, payload in calls if name == "start")
    assert start["approval_mode"] == "deny_all"
    assert start["sandbox"] == "read-only"


async def test_provider_review_maps_to_the_public_auto_reviewer(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    policy = PermissionPolicy(approval="provider_review", allowed_tools=("*",))
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path, policy=policy))
        result = await selected.run_turn(session, TurnRequest(input=(TextContent("rich"),)))

    assert result.status == "succeeded"
    calls = cast(dict[str, Any], installed_codex_sdk.state)["calls"]  # type: ignore[attr-defined]
    start = next(payload for name, payload in calls if name == "start")
    assert start["approval_mode"] == "auto_review"


async def test_sdk_resume_list_read_and_interrupt(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        first = await selected.run_turn(session, TurnRequest(input=(TextContent("first"),)))
        page = await selected.list_sessions(
            SessionQuery(scope=AgentCapabilityScope(backend="codex", transport="sdk", auth=auth()))
        )
        snapshot = await selected.read_session(
            session.ref,
            SessionReadOptions(auth=auth(), include_turns=False, include_items=False),
        )
        resumed = await selected.open_session(request(tmp_path, open=ResumeSession(session.ref)))
        cancelled = await selected.run_turn(
            resumed,
            TurnRequest(input=(TextContent("hang"),), timeout_seconds=0.01),
        )

    assert first.status == "succeeded"
    assert page.sessions[0].ref.native_session_id == session.ref.native_session_id
    assert snapshot.metadata.name == "Fixture thread"
    assert cancelled.status == "failed"
    assert cancelled.failure == "turn_timeout"
    assert cast(dict[str, Any], installed_codex_sdk.state)["interrupts"]  # type: ignore[attr-defined]


async def test_caller_approval_and_untyped_network_combinations_are_rejected(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    with pytest.raises(UnsupportedCapability, match="approval"):
        async with runtime(tmp_path) as selected:
            await selected.open_session(
                request(tmp_path, policy=PermissionPolicy(approval="ask", allowed_tools=("*",)))
            )

    confirmation = UnsafeConfirmation(("network_unrestricted",))
    with pytest.raises(UnsupportedCapability, match="network"):
        async with runtime(tmp_path) as selected:
            await selected.open_session(
                request(
                    tmp_path,
                    policy=PermissionPolicy(
                        network="unrestricted",
                        allowed_tools=("*",),
                        unsafe_confirmation=confirmation,
                    ),
                )
            )


async def test_non_chatgpt_auth_and_wrong_sdk_version_fail_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        CodexSdkAdapter,
        "_load_runtime_package",
        staticmethod(fake_runtime_package),
    )
    monkeypatch.setattr(
        CodexSdkAdapter, "_load_sdk", staticmethod(lambda: fake_sdk(account_type="apiKey"))
    )
    with pytest.raises(CredentialRejected, match="ChatGPT"):
        async with runtime(tmp_path) as selected:
            await selected.capabilities(
                AgentCapabilityScope(backend="codex", transport="sdk", auth=auth())
            )

    monkeypatch.setattr(
        CodexSdkAdapter, "_load_sdk", staticmethod(lambda: fake_sdk(version="9.9.9"))
    )
    with pytest.raises(SdkUnavailable, match="0.144.4"):
        async with runtime(tmp_path) as selected:
            await selected.capabilities(
                AgentCapabilityScope(backend="codex", transport="sdk", auth=auth())
            )


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
            await selected.capabilities(
                AgentCapabilityScope(backend="codex", transport="sdk", auth=auth())
            )

    rendered = "".join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert secret not in rendered


async def test_native_notification_is_bounded_before_tool_output_is_materialized(
    tmp_path: Path,
    installed_codex_sdk: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex_sdk_module, "_MAX_MESSAGE_BYTES", 1_024)

    async with runtime(tmp_path) as selected:
        session = await selected.open_session(request(tmp_path))
        result = await selected.run_turn(
            session,
            TurnRequest(input=(TextContent("oversized tool output"),)),
        )

    assert result.status == "failed"
    assert result.failure == "output_limit_exceeded"


@pytest.mark.skipif(
    importlib.util.find_spec("openai_codex") is not None,
    reason="the codex-sdk extra is installed; the no-extras CI job runs this by node id",
)
async def test_absent_sdk_extra_is_sdk_unavailable(tmp_path: Path) -> None:
    with pytest.raises(SdkUnavailable, match="install the 'codex-sdk' extra"):
        async with runtime(tmp_path) as selected:
            await selected.capabilities(
                AgentCapabilityScope(backend="codex", transport="sdk", auth=auth())
            )


async def test_sdk_config_carries_session_scoped_mcp_and_native_options(
    tmp_path: Path, installed_codex_sdk: ModuleType
) -> None:
    policy = PermissionPolicy(
        filesystem="full_access",
        network="unrestricted",
        allowed_tools=("*",),
        unsafe_confirmation=UnsafeConfirmation(("filesystem_full_access", "network_unrestricted")),
    )
    async with runtime(tmp_path) as selected:
        session = await selected.open_session(
            request(
                tmp_path,
                policy=policy,
                native=CodexNativeOptions(web_search=True),
            )
        )
        assert (
            await selected.run_turn(session, TurnRequest(input=(TextContent("ok"),)))
        ).status == "succeeded"

    calls = cast(dict[str, Any], installed_codex_sdk.state)["calls"]  # type: ignore[attr-defined]
    config = next(payload for name, payload in calls if name == "start")["config"]
    assert config["web_search"] == "live"
