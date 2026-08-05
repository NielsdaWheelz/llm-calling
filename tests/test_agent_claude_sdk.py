from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import signal
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

import provider_runtime.agent_runtime.claude_sdk as claude_module
from provider_runtime.agent_runtime._claude_launcher import (
    OwnedProcessGroup,
    ensure_claude_launcher,
    launcher_directory,
)
from provider_runtime.agent_runtime.auth import (
    AuthEnvironmentRequest,
    build_child_environment,
    child_home_directory,
)
from provider_runtime.agent_runtime.capabilities import (
    AgentCapabilityScope,
    validate_session_capabilities,
)
from provider_runtime.agent_runtime.claude_sdk import ClaudeSdkAdapter
from provider_runtime.agent_runtime.errors import (
    AgentRuntimeDefect,
    InvalidAgentRequest,
    McpUnavailable,
    MissingTerminalEvent,
    ProtocolDefect,
    SdkUnavailable,
    SessionUnavailable,
    UnsupportedCapability,
)
from provider_runtime.agent_runtime.events import (
    AgentEvent,
    DiagnosticData,
    terminal_event_to_result,
    validate_event_stream,
)
from provider_runtime.agent_runtime.policy import (
    PermissionPolicy,
    PermissionPolicyPatch,
    UnsafeConfirmation,
)
from provider_runtime.agent_runtime.sessions import (
    AgentSession,
    SessionQuery,
    SessionReadOptions,
    fingerprint_path,
)
from provider_runtime.agent_runtime.types import (
    AgentSessionRef,
    AgentSessionRequest,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    ClaudeNativeOptions,
    CredentialRef,
    ForkSession,
    JsonSchemaAgentOutput,
    McpServerSpec,
    NewSession,
    ResumeSession,
    TextContent,
    TurnRequest,
    thaw_json_value,
)
from provider_runtime.schema import parse_canonical_schema

FIXTURES = Path(__file__).parent / "fixtures" / "agent_runtime" / "claude"
CLAUDE_EXECUTABLE = (
    Path(__file__).parent / "fixtures" / "agent_runtime" / "claude" / "fake_claude_code.py"
)
AUTH = CredentialRef(kind="local_account", profile_key="fixture")
APPROVAL_CASES: list[dict[str, Any]] = json.loads((FIXTURES / "approval_cases.json").read_text())
FIXTURE_TOOLS = ("Read", "Write")


class ClaudeSDKError(Exception):
    """Stands in for the base class of the pinned SDK's own error hierarchy."""


class CLIConnectionError(ClaudeSDKError):
    pass


class ProcessError(ClaudeSDKError):
    pass


@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, object]


@dataclass
class TaskNotificationMessage(SystemMessage):
    """One of the six typed `SystemMessage` subclasses shipped by claude-agent-sdk 0.2.130."""

    task_id: str = ""
    status: str = ""
    output_file: str = ""
    summary: str = ""
    uuid: str | None = None
    session_id: str | None = None


@dataclass
class StreamEvent:
    uuid: str
    session_id: str
    event: dict[str, object]
    parent_tool_use_id: str | None = None


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, object]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: object = None
    is_error: bool | None = None


@dataclass
class AssistantMessage:
    content: list[object]
    model: str
    parent_tool_use_id: str | None = None
    error: str | None = None
    usage: dict[str, object] | None = None
    message_id: str | None = None
    stop_reason: str | None = None
    session_id: str | None = None
    uuid: str | None = None


@dataclass
class UserMessage:
    content: str | list[object]
    uuid: str | None = None
    parent_tool_use_id: str | None = None
    tool_use_result: dict[str, object] | None = None


@dataclass
class RateLimitInfo:
    status: str
    resets_at: int | None = None
    rate_limit_type: str | None = None
    utilization: float | None = None


@dataclass
class RateLimitEvent:
    rate_limit_info: RateLimitInfo
    uuid: str
    session_id: str


@dataclass
class ModelUsage:
    """`claude_agent_sdk.types.ModelUsage`, which `ResultMessage.model_usage` maps names to."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class DeferredToolUse:
    id: str
    name: str
    input: dict[str, object]


@dataclass
class ResultMessage:
    """Every field `parse_message` fills on a 0.2.130 `ResultMessage`, in its own order."""

    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    stop_reason: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, object] | None = None
    result: str | None = None
    structured_output: object = None
    model_usage: dict[str, ModelUsage] | None = None
    permission_denials: list[object] | None = None
    deferred_tool_use: DeferredToolUse | None = None
    errors: list[str] | None = None
    api_error_status: int | None = None
    uuid: str | None = None
    terminal_reason: str | None = None


@dataclass
class PermissionResultAllow:
    behavior: str = "allow"
    updated_input: dict[str, object] | None = None


@dataclass
class PermissionResultDeny:
    behavior: str = "deny"
    message: str = ""
    interrupt: bool = False


class ClaudeAgentOptions:
    add_dirs: list[str]
    allowed_tools: list[str]
    can_use_tool: Callable[[str, dict[str, Any], Any], Awaitable[object]]
    cli_path: str
    cwd: str
    disallowed_tools: list[str]
    env: dict[str, str]
    fork_session: bool
    include_partial_messages: bool
    mcp_servers: dict[str, object]
    model: str | None
    output_format: dict[str, object] | None
    permission_mode: str
    resume: str | None
    sandbox: dict[str, object]
    tools: list[str]

    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)


def load_message(value: dict[str, object]) -> object:
    """Mirror `claude_agent_sdk._internal.message_parser.parse_message`.

    In particular the pinned SDK returns `None` for a wire `type` it does not recognize and
    `ClaudeSDKClient` drops it, so the double must never hand the adapter a message the real
    dependency could not deliver.
    """
    kind = value["message_type"]
    fields: dict[str, Any] = {key: child for key, child in value.items() if key != "message_type"}
    if kind == "SystemMessage":
        return SystemMessage(**fields)
    if kind == "TaskNotificationMessage":
        return TaskNotificationMessage(**fields)
    if kind == "StreamEvent":
        return StreamEvent(**fields)
    if kind == "RateLimitEvent":
        fields["rate_limit_info"] = RateLimitInfo(**fields["rate_limit_info"])
        return RateLimitEvent(**fields)
    if kind in ("AssistantMessage", "UserMessage"):
        blocks = []
        content = fields.get("content")
        if isinstance(content, list):
            constructors = {
                "TextBlock": TextBlock,
                "ThinkingBlock": ThinkingBlock,
                "ToolUseBlock": ToolUseBlock,
                "ToolResultBlock": ToolResultBlock,
            }
            for block in content:
                block_fields = dict(block)
                constructor = constructors[block_fields.pop("block_type")]
                blocks.append(constructor(**block_fields))
            fields["content"] = blocks
        return (AssistantMessage if kind == "AssistantMessage" else UserMessage)(**fields)
    if kind == "ResultMessage":
        # `parse_message` types these two members rather than passing the wire dicts through.
        usage = fields.get("model_usage")
        if isinstance(usage, dict):
            fields["model_usage"] = {name: ModelUsage(**values) for name, values in usage.items()}
        deferred = fields.get("deferred_tool_use")
        if isinstance(deferred, dict):
            fields["deferred_tool_use"] = DeferredToolUse(**deferred)
        return ResultMessage(**fields)
    return None


def approval_case(name: str) -> dict[str, Any]:
    return next(case for case in APPROVAL_CASES if case["name"] == name)


class ClaudeSDKClient:
    """The SDK double, including the one thing only a real process can stand in for.

    `connect()` spawns `options.cli_path` exactly the way claude-agent-sdk 0.2.130 does —
    `anyio.open_process(cmd, ...)` with no `start_new_session` — because the adapter's
    process-group ownership is a property of the resulting process tree. A double that only
    exposed a `pid` attribute would let the lane's whole cleanup obligation pass untested,
    which is how it stayed broken.
    """

    instances: list[ClaudeSDKClient] = []
    mcp_status: dict[str, object] = {"mcpServers": []}
    init_overrides: dict[str, object] = {}
    bypass_launcher: bool = False
    # The name of an approval case to dispatch on the control channel just before a replayed
    # corpus reaches its result. The SDK's control-request handler runs on its own task and
    # does not consult the message stream, so a request Claude Code wrote before an interrupt
    # landed is delivered exactly there — on the tail nobody is reading.
    tail_permission: str | None = None

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self.options = options
        self.fixture = "success"
        self.query_calls: list[str] = []
        self.interrupt_calls = 0
        self.disconnected = False
        self.permission_results: list[object] = []
        self.stream_failure: BaseException | None = None
        self.process: asyncio.subprocess.Process | None = None
        self._transport: SimpleNamespace | None = None
        self._active: AsyncIterator[object] | None = None
        self.instances.append(self)

    async def connect(self, prompt=None) -> None:
        assert prompt is None
        executable = CLAUDE_EXECUTABLE if self.bypass_launcher else Path(self.options.cli_path)
        self.process = await asyncio.create_subprocess_exec(
            str(executable),
            "--output-format",
            "stream-json",
            "--verbose",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self.options.cwd,
            env={**os.environ, **self.options.env},
        )
        self._transport = SimpleNamespace(_process=self.process)

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self.query_calls.append(prompt)
        if prompt.startswith("fixture:"):
            self.fixture = prompt.removeprefix("fixture:")

    def _init_message(self, session_id: str) -> SystemMessage:
        """Report the effective configuration the way Claude Code 2.1.220 `tAr()` does."""
        return SystemMessage(
            subtype="init",
            data={**self._init_data(session_id), **self.init_overrides},
        )

    def _init_data(self, session_id: str) -> dict[str, object]:
        return cast(
            dict[str, object],
            {
                "cwd": str(self.options.cwd),
                "session_id": session_id,
                "tools": list(self.options.tools),
                "mcp_servers": [],
                "model": self.options.model,
                "permissionMode": self.options.permission_mode,
                "slash_commands": [],
                "apiKeySource": "none",
                "claude_code_version": "2.1.220",
                "output_style": "default",
                "agents": [],
                "skills": [],
                "plugins": [],
                "capabilities": ["interrupt_receipt_v1", "msg_lifecycle_v1"],
                "analytics_disabled": False,
                "product_feedback_disabled": False,
                "uuid": "11111111-1111-4111-8111-111111111141",
                "fast_mode_state": "off",
                "fast_mode_disabled_reason": "sdk_opt_in_required",
            },
        )

    async def _ask_permission(self, case: dict[str, Any]) -> object:
        result = await self.options.can_use_tool(
            case["tool_name"],
            dict(case["input"]),
            SimpleNamespace(**case["context"]),
        )
        self.permission_results.append(result)
        if getattr(result, "interrupt", False):
            await self.interrupt()
        return result

    async def receive_response(self):
        """Filter the client's one shared message stream, exactly as 0.2.130 does.

        `ClaudeSDKClient.receive_response` is a thin wrapper over `receive_messages()` that
        stops after a `ResultMessage` (client.py:571); the stream itself belongs to the
        client, not to the call. So a caller that abandons this iterator mid-turn leaves the
        rest of that turn queued, and the next call resumes it — which is precisely how an
        interrupted turn's tail becomes the next turn's stream. A double that started a
        fresh, independent stream per call could not reproduce that at all.
        """
        if self._active is None:
            self._active = self._messages()
        async for message in self._active:
            if isinstance(message, ResultMessage):
                # The turn's stream is finished the moment its result is *produced*: the real
                # client's queue has no memory of who consumed it, so a caller that abandons
                # this iterator after the result leaves nothing behind for the next one.
                self._active = None
                yield message
                return
            yield message

    async def _messages(self):
        if self.fixture.startswith("approval_"):
            case = approval_case(self.fixture.removeprefix("approval_"))
            session_id = "0198a200-0000-7000-8000-000000000041"
            yield self._init_message(session_id)
            # `Query._spawn_control_request_handler` dispatches `can_use_tool` on a task the
            # SDK spawns itself, concurrently with this message stream. Awaiting it inline
            # here would be the one ordering the real SDK never produces.
            asking = asyncio.create_task(self._ask_permission(case))
            yield AssistantMessage(
                content=[
                    TextBlock(text="Requesting permission."),
                    ToolUseBlock(
                        id=case["context"]["tool_use_id"],
                        name=case["tool_name"],
                        input=dict(case["input"]),
                    ),
                ],
                model="native-model",
                session_id=session_id,
                message_id="msg-approval-1",
                uuid="assistant-approval-1",
                usage={"input_tokens": 10, "output_tokens": 2},
            )
            result = await asking
            interrupted = getattr(result, "interrupt", False)
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id=session_id,
                result="Permission handled.",
                terminal_reason="aborted_tools" if interrupted else "completed",
                uuid="result-approval-1",
            )
            return
        if self.fixture in ("structured_failure", "structured_success"):
            session_id = "0198a200-0000-7000-8000-000000000051"
            valid = self.fixture == "structured_success"
            yield self._init_message(session_id)
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id=session_id,
                result='{"answer": "ok"}' if valid else '{"answer": 3}',
                structured_output={"answer": "ok"} if valid else {"answer": 3},
                terminal_reason="completed",
                uuid="result-structured-1",
            )
            return
        values = json.loads((FIXTURES / f"{self.fixture}.json").read_text())
        for index, value in enumerate(values):
            if self.stream_failure is not None and index == 2:
                raise self.stream_failure
            message = load_message(value)
            if message is None:
                continue
            if isinstance(message, SystemMessage) and message.subtype == "init":
                message.data["cwd"] = str(self.options.cwd)
            if self.tail_permission is not None and isinstance(message, ResultMessage):
                case, self.tail_permission = approval_case(self.tail_permission), None
                await self._ask_permission(case)
            yield message

    async def interrupt(self) -> None:
        self.interrupt_calls += 1

    async def disconnect(self) -> None:
        """Release the child the way the SDK's own transport close does: terminate, then wait."""
        self.disconnected = True
        process, self.process, self._transport = self.process, None, None
        if process is None:
            return
        with suppress(ProcessLookupError):
            process.terminate()
        await process.wait()

    async def get_mcp_status(self) -> dict[str, object]:
        return self.mcp_status

    async def set_model(self, model: str) -> None:
        self.options.model = model

    async def set_permission_mode(self, mode: str) -> None:
        self.options.permission_mode = mode


def fake_sdk() -> SimpleNamespace:
    ClaudeSDKClient.instances.clear()
    ClaudeSDKClient.mcp_status = {"mcpServers": []}
    ClaudeSDKClient.init_overrides = {}
    return SimpleNamespace(
        __version__="0.2.130",
        ClaudeAgentOptions=ClaudeAgentOptions,
        ClaudeSDKClient=ClaudeSDKClient,
        ClaudeSDKError=ClaudeSDKError,
        CLIConnectionError=CLIConnectionError,
        ProcessError=ProcessError,
        SystemMessage=SystemMessage,
        StreamEvent=StreamEvent,
        TextBlock=TextBlock,
        ThinkingBlock=ThinkingBlock,
        ToolUseBlock=ToolUseBlock,
        ToolResultBlock=ToolResultBlock,
        AssistantMessage=AssistantMessage,
        UserMessage=UserMessage,
        RateLimitEvent=RateLimitEvent,
        ResultMessage=ResultMessage,
        PermissionResultAllow=PermissionResultAllow,
        PermissionResultDeny=PermissionResultDeny,
        list_sessions=lambda **_kwargs: [],
        get_session_messages=lambda *_args, **_kwargs: [],
        get_session_info=lambda *_args, **_kwargs: None,
    )


@pytest.fixture
def installed_sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    module = fake_sdk()
    original = importlib.import_module

    def load(name: str, package: str | None = None):
        if name == "claude_agent_sdk":
            return module
        return original(name, package)

    async def network_sandbox_available(**_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(importlib, "import_module", load)
    monkeypatch.setattr(claude_module.shutil, "which", lambda _name: str(CLAUDE_EXECUTABLE))
    monkeypatch.setattr(claude_module, "_network_sandbox_available", network_sandbox_available)
    return module


def environment(tmp_path: Path) -> dict[str, str]:
    """The child environment the runtime hands the adapter, built by its real owner.

    Hand-authoring this map is what let the adapter grow its own PATH/HOME workaround: a
    fixture that omitted what `auth.build_child_environment` actually supplies made the
    omission look like the adapter's problem to solve. The directory tree is created the way
    `AgentRuntime._secure_state_root` creates it — every component the runtime owns at 0700,
    including the one *above* the state root — because the launcher this lane writes depends
    on that guarantee and a laxer fixture would hide a real deployment failure.
    """
    state_root = tmp_path / "state" / "claude" / "fixture"
    for component in (state_root.parent, state_root, child_home_directory(state_root)):
        component.mkdir(mode=0o700, parents=True, exist_ok=True)
        component.chmod(0o700)
    return build_child_environment(
        AuthEnvironmentRequest(
            backend="claude",
            credential=AUTH,
            inherited_environment={"TERM": "dumb"},
            allowed_environment=("TERM",),
            state_root=state_root,
        )
    )


def session_request(
    tmp_path: Path, *, auth: CredentialRef = AUTH, **changes: object
) -> AgentSessionRequest:
    cwd = tmp_path / "repo"
    cwd.mkdir(exist_ok=True)
    request = AgentSessionRequest(
        backend="claude",
        transport="sdk",
        auth=auth,
        open=NewSession(),
        cwd=str(cwd.resolve()),
        policy=PermissionPolicy(),
        model="native-model",
    )
    return replace(request, **changes)


async def drain(stream: AsyncIterator[AgentEvent], *, scheduled: bool = False) -> list[AgentEvent]:
    """Collect one validated stream, optionally giving concurrent SDK tasks room to run.

    `AgentRuntime._interruptible_stream` does a full `asyncio.wait` round trip between every
    adapter event, so a task the SDK spawned can always run between two of them. `scheduled`
    reproduces that here; without it a unit test only ever sees the single-task ordering.
    """
    events: list[AgentEvent] = []
    iterator = validate_event_stream(stream).__aiter__()
    while True:
        if scheduled:
            for _ in range(8):
                await asyncio.sleep(0)
        try:
            events.append(await anext(iterator))
        except StopAsyncIteration:
            return events


def fixture_policy(**changes: object) -> PermissionPolicy:
    return replace(PermissionPolicy(allowed_tools=FIXTURE_TOOLS), **changes)


async def stream_fixture(
    tmp_path: Path, name: str, installed_sdk: SimpleNamespace
) -> list[AgentEvent]:
    adapter = ClaudeSdkAdapter()
    try:
        session = await open_session(
            adapter,
            session_request(tmp_path, policy=fixture_policy()),
            environment=environment(tmp_path),
        )
        return await drain(
            adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent(f"fixture:{name}"),)),
                approvals=None,
            )
        )
    finally:
        await adapter.close()


async def until(condition: Callable[[], bool], *, timeout_seconds: float = 5.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.01)
    return condition()


def stopped(pid: int) -> bool:
    """Whether one pid is gone or is a zombie nobody here can reap.

    A descendant of the launched child is nobody's child once its own parent dies, so it is
    reaped by init and not by this process. `kill(pid, 0)` succeeds against a zombie, so the
    only honest liveness answer on Linux comes from the process state in `/proc`.
    """
    try:
        line = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return line.rsplit(") ", 1)[1].split(" ", 1)[0] == "Z"


def test_importing_adapter_does_not_import_optional_sdk() -> None:
    assert "claude_agent_sdk" not in sys.modules


async def test_the_launcher_turns_the_sdk_child_into_a_group_this_runtime_can_reap(
    tmp_path: Path,
) -> None:
    """The whole point of the launcher, proved against real processes.

    claude-agent-sdk 0.2.130 spawns `cli_path` with `anyio.open_process` and no
    `start_new_session`, so without the launcher the child joins *this* process's group:
    there is nothing safe to signal, and Claude Code's own children — a Bash-tool command, a
    stdio MCP server — survive every teardown. The descendant here ignores SIGTERM and only
    reports readiness once its handler is installed, so a passing run cannot be one where the
    first polite signal happened to win the race.
    """
    values = environment(tmp_path)
    state_root = Path(values["CLAUDE_CONFIG_DIR"])
    launcher = ensure_claude_launcher(
        state_root, str(CLAUDE_EXECUTABLE), interpreter=sys.executable
    )

    unlaunched = await asyncio.create_subprocess_exec(
        str(CLAUDE_EXECUTABLE), stdout=asyncio.subprocess.DEVNULL
    )
    try:
        assert os.getpgid(unlaunched.pid) == os.getpgid(0)
        with pytest.raises(ProtocolDefect, match="own process group"):
            await OwnedProcessGroup.adopt(unlaunched.pid, timeout_seconds=0.05)
    finally:
        unlaunched.kill()
        await unlaunched.wait()

    ready = tmp_path / "ready"
    child = await asyncio.create_subprocess_exec(
        str(launcher),
        "--spawn-descendant",
        "--ready-file",
        str(ready),
        stdout=asyncio.subprocess.DEVNULL,
    )
    assert await until(ready.exists)
    leader, group_id, descendant = (int(value) for value in ready.read_text().split())

    assert leader == child.pid == group_id != os.getpgid(0)
    assert os.getpgid(descendant) == child.pid
    owned = await OwnedProcessGroup.adopt(child.pid, timeout_seconds=2.0)

    # Establish that the descendant really is the hard case before crediting the escalation
    # with ending it. Without this, a run in which the readiness protocol had regressed and
    # the descendant died to the polite signal would look exactly like a run in which the
    # SIGKILL escalation worked, and the escalation would be untested.
    os.killpg(child.pid, signal.SIGTERM)
    assert await child.wait() == -signal.SIGTERM
    await asyncio.sleep(0.2)
    assert not stopped(descendant)

    await owned.terminate(grace_seconds=0.2)
    assert await until(lambda: stopped(descendant))


async def test_the_launcher_is_private_and_outside_everything_the_child_can_write(
    tmp_path: Path,
) -> None:
    """A launcher a session could rewrite would run the next launch outside the sandbox.

    `CLAUDE_CONFIG_DIR` and the child `HOME` under it are directories Claude Code writes to
    by design, so the launcher lives in the runtime-owned directory above them, which no
    child environment variable names and no sandbox root covers.
    """
    values = environment(tmp_path)
    state_root = Path(values["CLAUDE_CONFIG_DIR"])
    launcher = ensure_claude_launcher(
        state_root, str(CLAUDE_EXECUTABLE), interpreter=sys.executable
    )

    assert launcher.parent == launcher_directory(state_root) == state_root.parent
    assert not launcher.is_relative_to(state_root)
    assert launcher.stat().st_mode & 0o777 == 0o700
    assert str(CLAUDE_EXECUTABLE) in launcher.read_text()
    # Content-addressed, so a second runtime writing the same launcher is not a race.
    assert (
        ensure_claude_launcher(state_root, str(CLAUDE_EXECUTABLE), interpreter=sys.executable)
        == launcher
    )

    launcher.write_text("#!/bin/sh\nexec /bin/true\n")
    rewritten = ensure_claude_launcher(
        state_root, str(CLAUDE_EXECUTABLE), interpreter=sys.executable
    )
    assert "os.setsid()" in rewritten.read_text()


async def test_a_state_root_the_runtime_did_not_secure_refuses_to_carry_a_launcher(
    tmp_path: Path,
) -> None:
    values = environment(tmp_path)
    state_root = Path(values["CLAUDE_CONFIG_DIR"])
    state_root.parent.chmod(0o755)

    with pytest.raises(AgentRuntimeDefect, match="privately permissioned"):
        ensure_claude_launcher(state_root, str(CLAUDE_EXECUTABLE), interpreter=sys.executable)


async def test_a_child_that_is_not_its_own_group_leader_fails_the_open_closed(
    installed_sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No session may run while its descendants would be unreachable at teardown."""
    monkeypatch.setattr(claude_module, "_GROUP_ADOPTION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(ClaudeSDKClient, "bypass_launcher", True)
    adapter = ClaudeSdkAdapter()
    try:
        with pytest.raises(ProtocolDefect) as captured:
            await open_session(
                adapter, session_request(tmp_path), environment=environment(tmp_path)
            )
    finally:
        await adapter.close()

    assert captured.value.code == "process_group_missing"
    client = ClaudeSDKClient.instances[-1]
    assert client.disconnected is True
    assert client.process is None


async def test_missing_optional_dependency_is_a_typed_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = importlib.import_module

    def missing(name: str, package: str | None = None):
        if name == "claude_agent_sdk":
            raise ModuleNotFoundError(name)
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(SdkUnavailable, match="claude-agent-sdk"):
        await ClaudeSdkAdapter().capabilities(
            AgentCapabilityScope(backend="claude", transport="sdk", auth=AUTH),
            environment=environment(tmp_path),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("claude_agent_sdk") is not None,
    reason="the claude-sdk extra is installed; the no-extras CI job runs this by node id",
)
async def test_absent_sdk_extra_is_sdk_unavailable(tmp_path: Path) -> None:
    """Prove the real absent-module path, with no import machinery patched at all.

    Every other test in this file substitutes `importlib.import_module`, so it behaves the
    same whether or not the extra is installed. Only this one can fail when `_load_sdk` stops
    raising the precise typed error the spec's no-extras job asserts on.
    """
    adapter = ClaudeSdkAdapter()
    scope = AgentCapabilityScope(backend="claude", transport="sdk", auth=AUTH)
    with pytest.raises(SdkUnavailable, match="install the 'claude-sdk' extra"):
        await adapter.capabilities(scope, environment=environment(tmp_path))
    with pytest.raises(SdkUnavailable, match="install the 'claude-sdk' extra"):
        await open_session(adapter, session_request(tmp_path), environment=environment(tmp_path))


async def test_capabilities_are_sdk_native_and_auth_identity_is_honest(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    capabilities = await ClaudeSdkAdapter().capabilities(
        AgentCapabilityScope(backend="claude", transport="sdk", auth=AUTH),
        environment=environment(tmp_path),
    )

    assert capabilities.sdk_version == "0.2.130"
    assert capabilities.session_operations == ("new", "resume", "fork")
    assert capabilities.discovery_operations == ()
    assert capabilities.additional_dirs is True
    assert capabilities.network_allowlist is True
    assert capabilities.max_turns is False
    assert capabilities.turn_overrides == ()
    assert capabilities.persistent_turn_overrides == ()
    assert capabilities.reports_auth_identity is False
    assert capabilities.reports_effective_effort is False
    assert capabilities.builtin_tool_families == ("file_read", "file_write", "command")
    assert capabilities.builtin_tool_names == (
        ("file_read", ("Read", "Glob", "Grep")),
        ("file_write", ("Write", "Edit", "NotebookEdit")),
        ("command", ("Bash",)),
    )


@pytest.mark.parametrize(
    ("family", "name"),
    [(family, name) for family, names in claude_module._BUILTIN_TOOL_NAMES for name in names],
)
async def test_every_published_native_tool_name_is_one_a_policy_may_actually_name(
    installed_sdk: SimpleNamespace, tmp_path: Path, family: str, name: str
) -> None:
    """`builtin_tool_names` carries accepted names, so every one of them must be accepted.

    This is the disagreement class the audit was about: a capability report that advertises
    what the adapter then refuses. A caller has no other source for the exact spellings —
    `builtin_tool_families` is a normalized cross-backend vocabulary and `system/init` only
    echoes the tools the session already asked for — so an advertised name the policy
    validator rejects would strand it with no working value at all.
    """
    adapter = ClaudeSdkAdapter()
    try:
        await open_session(
            adapter,
            session_request(tmp_path, policy=PermissionPolicy(allowed_tools=(name,))),
            environment=environment(tmp_path),
        )
        options = ClaudeSDKClient.instances[-1].options
    finally:
        await adapter.close()

    assert options.tools == [name]
    # The approval path must classify the name the same way the table files it, because
    # `filesystem: read_only` refuses `command`/`file_change` outright: a published file
    # write that the approval path called an ordinary tool use would run under read-only.
    operation = ClaudeSdkAdapter._operation(name)
    assert (
        operation
        == {
            "file_read": "tool_use",
            "file_write": "file_change",
            "command": "command",
        }[family]
    )


async def test_refused_tool_names_are_never_advertised_as_accepted(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    """The other half of the same invariant: what the adapter refuses stays out of the table.

    `WebFetch`/`WebSearch` reach the network through Claude Code's own client rather than the
    sandbox this lane always engages, so `_validate_policy_mapping` refuses them. Advertising
    either — as a name, or as a `web_fetch`/`web_search` family a caller would then look for
    a name in — would be the capability report promising a route no request can take.
    """
    capabilities = await ClaudeSdkAdapter().capabilities(
        AgentCapabilityScope(backend="claude", transport="sdk", auth=AUTH),
        environment=environment(tmp_path),
    )
    published = {name for _family, names in capabilities.builtin_tool_names for name in names}

    assert published.isdisjoint(claude_module._REFUSED_TOOL_NAMES)
    assert set(capabilities.builtin_tool_families) == {
        family for family, _names in capabilities.builtin_tool_names
    }
    for refused in claude_module._REFUSED_TOOL_NAMES:
        adapter = ClaudeSdkAdapter()
        with pytest.raises(UnsupportedCapability):
            await open_session(
                adapter,
                session_request(tmp_path, policy=PermissionPolicy(allowed_tools=(refused,))),
                environment=environment(tmp_path),
            )
        await adapter.close()


async def test_api_key_auth_is_rejected_before_sdk_client_side_effects(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    auth = CredentialRef(
        kind="api_key_environment", profile_key="fixture", name="ANTHROPIC_API_KEY"
    )
    adapter = ClaudeSdkAdapter()
    with pytest.raises(UnsupportedCapability, match="authentication identity"):
        await open_session(
            adapter,
            session_request(tmp_path, auth=auth),
            environment={**environment(tmp_path), "ANTHROPIC_API_KEY": "fixture-secret"},
        )
    assert ClaudeSDKClient.instances == []


async def test_sdk_options_scrub_ambient_credentials_and_fail_closed_sandbox(
    installed_sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "ambient-oauth")
    additional = (tmp_path / "shared").resolve()
    additional.mkdir()
    adapter = ClaudeSdkAdapter()
    try:
        await open_session(
            adapter,
            session_request(tmp_path, additional_dirs=(str(additional),)),
            environment=environment(tmp_path),
        )
        options = ClaudeSDKClient.instances[-1].options
        options_any: Any = options
    finally:
        await adapter.close()

    assert options.env["ANTHROPIC_API_KEY"] == ""
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert options.env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert options.env["CLAUDE_CONFIG_DIR"].endswith("/claude/fixture")
    assert options.sandbox["failIfUnavailable"] is True
    assert options.add_dirs == [str(additional)]
    assert (
        Path(options_any.cli_path)
        .read_text()
        .endswith("os.execv(_EXECUTABLE, [_EXECUTABLE, *sys.argv[1:]])\n\n\nmain()\n")
    )


async def test_child_process_controls_are_usable_rather_than_blanked(
    installed_sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SDK merges its env over `os.environ`, so a blanked `PATH` wins and resolves nothing."""
    monkeypatch.setenv("PATH", "/ambient/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "ambient-home"))
    monkeypatch.setenv("EDITOR", "ambient-editor")
    values = environment(tmp_path)
    adapter = ClaudeSdkAdapter()
    try:
        await open_session(adapter, session_request(tmp_path), environment=values)
        options = ClaudeSDKClient.instances[-1].options
    finally:
        await adapter.close()

    state_root = Path(values["CLAUDE_CONFIG_DIR"])
    # The runtime-owned values reach the child unchanged; the ambient ones never do.
    assert options.env["PATH"] == values["PATH"] != "/ambient/bin"
    assert options.env["HOME"] == str(child_home_directory(state_root))
    assert child_home_directory(state_root).is_dir()
    assert options.env["LC_ALL"] == "C.UTF-8"
    # Everything the child is not authorized to see is still blanked, not inherited.
    assert options.env["EDITOR"] == ""


@pytest.mark.parametrize("filesystem", ["read_only", "workspace_write"])
@pytest.mark.parametrize("approval", ["deny", "ask", "allow"])
async def test_sdk_never_uses_behavior_changing_permission_modes(
    installed_sdk: SimpleNamespace,
    tmp_path: Path,
    filesystem: Literal["read_only", "workspace_write"],
    approval: Literal["deny", "ask", "allow"],
) -> None:
    confirmation = UnsafeConfirmation(("approval_allow",)) if approval == "allow" else None
    policy = PermissionPolicy(
        filesystem=filesystem,
        approval=approval,
        allowed_tools=("Read",),
        unsafe_confirmation=confirmation,
    )
    adapter = ClaudeSdkAdapter()
    try:
        await open_session(
            adapter, session_request(tmp_path, policy=policy), environment=environment(tmp_path)
        )
        options = ClaudeSDKClient.instances[-1].options
    finally:
        await adapter.close()

    assert options.permission_mode == "default"
    assert options.permission_mode not in ("plan", "bypassPermissions", "dontAsk")
    assert options.allowed_tools == []
    assert options.tools == ["Read"]


@pytest.mark.parametrize("tool", ["WebFetch", "WebSearch", "Read*"])
async def test_sdk_rejects_unenforceable_tool_availability_before_client_start(
    installed_sdk: SimpleNamespace, tmp_path: Path, tool: str
) -> None:
    adapter = ClaudeSdkAdapter()
    with pytest.raises(UnsupportedCapability):
        await open_session(
            adapter,
            session_request(tmp_path, policy=PermissionPolicy(allowed_tools=(tool,))),
            environment=environment(tmp_path),
        )
    assert ClaudeSDKClient.instances == []


async def test_sdk_rejects_per_turn_policy_it_cannot_reconfigure(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(
            tmp_path,
            policy=PermissionPolicy(filesystem="workspace_write", allowed_tools=("Read", "Edit")),
        ),
        environment=environment(tmp_path),
    )
    try:
        with pytest.raises(UnsupportedCapability, match="policy"):
            _ = [
                event
                async for event in adapter.stream_turn(
                    session,
                    TurnRequest(
                        input=(TextContent("fixture:success"),),
                        policy=PermissionPolicyPatch(
                            filesystem="read_only", allowed_tools=("Read",)
                        ),
                    ),
                    approvals=None,
                )
            ]
    finally:
        await adapter.close()


async def test_success_fixture_normalizes_visible_thinking_tools_files_and_usage(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    events = await stream_fixture(tmp_path, "success", installed_sdk)

    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    assert events[0].kind == "session_started"
    assert events[-1].kind == "turn_completed"
    assert terminal_event_to_result(events[-1]).final_text == "Inspection complete."
    assert {event.kind for event in events} >= {
        "turn_started",
        "text_delta",
        "reasoning",
        "tool_started",
        "tool_completed",
        "file_change",
        "usage",
        "unknown",
        "diagnostic",
    }
    # An unmodelled `system` subtype is the forward-compatibility surface the pinned SDK can
    # actually deliver, and its payload must survive redaction with no credential in it.
    unknown = next(event for event in events if event.kind == "unknown")
    assert unknown.native_type == "SystemMessage:compact_boundary"
    assert unknown.native_payload is not None
    wire = json.dumps(thaw_json_value(unknown.native_payload))
    assert "secret-value-123456" not in wire
    assert "future-secret-token-value" not in wire
    # Failed background work must not vanish from the stream.
    task_diagnostic = next(
        event
        for event in events
        if isinstance(event.data, DiagnosticData) and event.data.code == "claude_task_notification"
    )
    assert task_diagnostic.native_type == "SystemMessage:task_notification"


async def test_terminal_native_payload_keeps_every_field_the_sdk_reports(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    """`redact_native_payload` treats the field list as an allowlist, so an omission is a loss.

    `total_cost_usd`, `model_usage`, `permission_denials`, and `deferred_tool_use` all exist
    on `claude_agent_sdk.types.ResultMessage` and are all filled by `parse_message`; leaving
    them out silently dropped them from the one event the spec says retains native data.
    """
    events = await stream_fixture(tmp_path, "success", installed_sdk)

    terminal = events[-1]
    assert terminal.native_payload is not None
    payload = thaw_json_value(terminal.native_payload)
    assert isinstance(payload, dict)
    assert payload["total_cost_usd"] == 0.0731
    # The per-model counts survive `redact_native_payload`'s sensitive-key rule: they are
    # backend-reported usage, not auth material, and the rule exempts a `token` word that
    # sits beside a counting word (`input`/`output`). See tests/test_agent_auth.py for both
    # directions of that predicate.
    assert payload["model_usage"] == {
        "native-model": {
            "input_tokens": 140,
            "output_tokens": 32,
            "cost_usd": 0.0731,
        }
    }
    assert payload["permission_denials"] == [
        {"tool_name": "Bash", "tool_use_id": "tool-denied-1", "tool_input": {"command": "ls"}}
    ]
    assert payload["deferred_tool_use"] == {
        "id": "tool-deferred-1",
        "name": "Read",
        "input": {"file_path": "/workspace/repo/README.md"},
    }


async def test_an_interrupted_turn_does_not_hand_its_tail_to_the_next_turn(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    """The session survives a soft interrupt, so its abandoned messages must not.

    `receive_response()` filters the client's one shared stream, so the interrupted turn's
    remaining messages — its own `ResultMessage` included — are what the next turn reads.
    Without the drain the second turn here terminates on the *first* turn's result.
    """
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, policy=fixture_policy()),
        environment=environment(tmp_path),
    )
    try:
        stream = adapter.stream_turn(
            session,
            TurnRequest(input=(TextContent("fixture:success"),)),
            approvals=None,
        )
        iterator = stream.__aiter__()
        first = [await anext(iterator) for _ in range(3)]
        await adapter.interrupt(session, first[-1].turn_id)
        await stream.aclose()

        second = await drain(
            adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:second_turn"),)),
                approvals=None,
            )
        )
    finally:
        await adapter.close()

    assert [event.kind for event in first] == ["session_started", "turn_started", "reasoning"]
    terminal = second[-1]
    assert terminal.kind == "turn_completed"
    assert terminal_event_to_result(terminal).final_text == "Second turn."
    assert terminal.native_payload is not None
    payload = thaw_json_value(terminal.native_payload)
    assert isinstance(payload, dict)
    assert payload["uuid"] == "result-second-1"
    assert all(event.turn_id == terminal.turn_id for event in second)


async def test_a_permission_request_on_a_retired_turns_tail_never_reaches_the_caller(
    installed_sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The tail of an interrupted turn arrives on the control channel too, not only the stream.

    The SDK dispatches `can_use_tool` from a task of its own, so a permission request Claude
    Code wrote before the interrupt reached it is delivered whichever way the message stream
    is being handled. Two windows exist for it — between the interrupt and the next turn,
    where nothing is reading, and inside the drain, where the retired tail is being discarded
    — and both belong to a turn the caller has abandoned. Putting either to the caller asks
    for consent out of context, and an `allow` would run the tool on a turn nobody is
    listening to; recording the out-of-turn defect instead makes the *next* turn raise it.
    """
    case = approval_case("handler_allow")
    monkeypatch.setattr(ClaudeSDKClient, "tail_permission", case["name"])
    asked: list[ApprovalRequest] = []

    async def approvals(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request)
        return "allow"

    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(
            tmp_path,
            policy=replace(case_policy(case), allowed_tools=(*FIXTURE_TOOLS, case["tool_name"])),
        ),
        environment=environment(tmp_path),
    )
    try:
        stream = adapter.stream_turn(
            session,
            TurnRequest(input=(TextContent("fixture:success"),)),
            approvals=approvals,
        )
        iterator = stream.__aiter__()
        first = [await anext(iterator) for _ in range(3)]
        await adapter.interrupt(session, first[-1].turn_id)
        await stream.aclose()

        # The idle window: the request lands while no turn is being read at all.
        client = ClaudeSDKClient.instances[-1]
        idle = await client.options.can_use_tool(
            case["tool_name"], dict(case["input"]), SimpleNamespace(**case["context"])
        )
        # The drain window: `tail_permission` dispatches the second request from inside the
        # retired turn's tail, which the next turn's head is what finally reads.
        second = await drain(
            adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:second_turn"),)),
                approvals=approvals,
            )
        )
    finally:
        await adapter.close()

    assert asked == []
    assert isinstance(idle, PermissionResultDeny)
    assert idle.interrupt is True
    assert not any(
        isinstance(result, PermissionResultAllow) for result in client.permission_results
    )
    assert [event.kind for event in second] == ["turn_started", "text_delta", "turn_completed"]
    assert terminal_event_to_result(second[-1]).final_text == "Second turn."


async def test_a_turn_that_never_named_itself_releases_the_whole_session(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    """`interrupt(session, None)` means no turn was ever identified, so nothing can be kept."""
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, policy=fixture_policy()),
        environment=environment(tmp_path),
    )
    try:
        await adapter.interrupt(session, None)

        assert ClaudeSDKClient.instances[-1].disconnected is True
        with pytest.raises(SessionUnavailable):
            await drain(
                adapter.stream_turn(
                    session,
                    TurnRequest(input=(TextContent("fixture:success"),)),
                    approvals=None,
                )
            )
    finally:
        await adapter.close()


async def test_effective_permission_mode_widening_is_a_policy_mismatch_defect(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    """`system/init` echoes the mode the CLI actually runs in; a different one is a security fact."""
    ClaudeSDKClient.init_overrides = {"permissionMode": "bypassPermissions"}
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, policy=fixture_policy()),
        environment=environment(tmp_path),
    )
    try:
        with pytest.raises(ProtocolDefect) as captured:
            await drain(
                adapter.stream_turn(
                    session,
                    TurnRequest(input=(TextContent("fixture:structured_success"),)),
                    approvals=None,
                )
            )
    finally:
        await adapter.close()
    assert captured.value.code == "effective_policy_mismatch"


async def test_effective_tool_widening_is_a_policy_mismatch_defect(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    ClaudeSDKClient.init_overrides = {"tools": ["Read", "Write", "Bash"]}
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, policy=fixture_policy()),
        environment=environment(tmp_path),
    )
    try:
        with pytest.raises(ProtocolDefect) as captured:
            await drain(
                adapter.stream_turn(
                    session,
                    TurnRequest(input=(TextContent("fixture:structured_success"),)),
                    approvals=None,
                )
            )
    finally:
        await adapter.close()
    assert captured.value.code == "effective_policy_mismatch"


async def test_effective_tools_accept_the_cli_rename_and_mcp_namespaced_tools(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    """Claude Code reports the `Agent` tool as `Task` and MCP tools as `mcp__<server>__<tool>`."""
    ClaudeSDKClient.mcp_status = {"mcpServers": [{"name": "fixture", "status": "connected"}]}
    ClaudeSDKClient.init_overrides = {"tools": ["Task", "mcp__fixture__inspect"]}
    server = McpServerSpec(
        name="fixture",
        transport="streamable_http",
        url="https://example.invalid/mcp",
        required=True,
    )
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(
            tmp_path,
            policy=fixture_policy(
                allowed_tools=("Agent",),
                network="allowlist",
                network_allowlist=("example.invalid",),
            ),
            mcp_servers=(server,),
        ),
        environment=environment(tmp_path),
    )
    try:
        events = await drain(
            adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:structured_success"),)),
                approvals=None,
            )
        )
    finally:
        await adapter.close()
    assert events[-1].kind == "turn_completed"


async def test_effective_model_substitution_is_a_diagnostic(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    ClaudeSDKClient.init_overrides = {"model": "substituted-model"}
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, policy=fixture_policy()),
        environment=environment(tmp_path),
    )
    try:
        events = await drain(
            adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:structured_success"),)),
                approvals=None,
            )
        )
    finally:
        await adapter.close()

    assert any(
        isinstance(event.data, DiagnosticData) and event.data.code == "effective_model_changed"
        for event in events
    )
    assert "effective_model_changed" in terminal_event_to_result(events[-1]).diagnostics


async def test_quota_cancel_and_missing_terminal_are_not_collapsed(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    quota = await stream_fixture(tmp_path, "quota_failed", installed_sdk)
    cancelled = await stream_fixture(tmp_path, "cancelled", installed_sdk)

    assert quota[-1].kind == "turn_failed"
    assert terminal_event_to_result(quota[-1]).failure == "quota_exhausted"
    assert cancelled[-1].kind == "turn_cancelled"
    with pytest.raises(MissingTerminalEvent):
        await stream_fixture(tmp_path, "missing_terminal", installed_sdk)


def case_policy(case: dict[str, Any]) -> PermissionPolicy:
    declared = case["policy"]
    acknowledged = tuple(declared["acknowledged"])
    return PermissionPolicy(
        filesystem=declared["filesystem"],
        approval=declared["approval"],
        allowed_tools=tuple(declared["allowed_tools"]),
        unsafe_confirmation=UnsafeConfirmation(acknowledged) if acknowledged else None,
    )


def case_handler(case: dict[str, Any]) -> ApprovalHandler | None:
    answer = case["answer"]
    if answer is None:
        return None

    async def handler(_request: ApprovalRequest) -> ApprovalDecision:
        if answer == "raise":
            raise RuntimeError("credential=fixture-secret-must-not-leak")
        return cast(ApprovalDecision, answer)

    return handler


async def _run_approval(
    tmp_path: Path, case: dict[str, Any]
) -> tuple[list[AgentEvent], ClaudeSDKClient]:
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, policy=case_policy(case)),
        environment=environment(tmp_path),
    )
    try:
        events = await drain(
            adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent(f"fixture:approval_{case['name']}"),)),
                approvals=case_handler(case),
            ),
            scheduled=True,
        )
        return events, ClaudeSDKClient.instances[-1]
    finally:
        await adapter.close()


@pytest.mark.parametrize("case", APPROVAL_CASES, ids=lambda case: str(case["name"]))
async def test_declared_approval_matrix_is_answered_and_observable(
    installed_sdk: SimpleNamespace, tmp_path: Path, case: dict[str, Any]
) -> None:
    events, client = await _run_approval(tmp_path, case)

    kinds = [event.kind for event in events]
    assert kinds.count("approval_requested") == 1
    assert kinds.count("approval_answered") == 1
    # The SDK answers the permission request from a task it spawned, so the pair must still
    # be numbered in stream order and must precede the tool call it gated.
    assert kinds.index("approval_requested") < kinds.index("approval_answered")
    assert kinds.index("approval_answered") < kinds.index("tool_started")
    assert [event.seq for event in events] == list(range(1, len(events) + 1))

    result = client.permission_results[-1]
    assert isinstance(result, (PermissionResultAllow, PermissionResultDeny))
    assert result.behavior == case["expected_behavior"]
    assert getattr(result, "interrupt", False) is case["expected_interrupt"]
    assert client.interrupt_calls == (1 if case["expected_interrupt"] else 0)
    assert events[-1].kind == case["expected_terminal"]
    assert terminal_event_to_result(events[-1]).failure == case["expected_failure"]
    assert "fixture-secret" not in repr(terminal_event_to_result(events[-1]))


async def test_ask_without_handler_rejects_before_sdk_query(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(
            tmp_path,
            policy=PermissionPolicy(
                filesystem="workspace_write", approval="ask", allowed_tools=("Bash",)
            ),
        ),
        environment=environment(tmp_path),
    )
    try:
        with pytest.raises(InvalidAgentRequest, match="requires an approval handler"):
            async for _event in adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:approval_handler_allow"),)),
                approvals=None,
            ):
                pass
        assert ClaudeSDKClient.instances[-1].query_calls == []
    finally:
        await adapter.close()


async def test_sdk_max_turns_rejects_before_query_because_client_option_is_session_bound(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter, session_request(tmp_path), environment=environment(tmp_path)
    )
    client = ClaudeSDKClient.instances[-1]
    try:
        with pytest.raises(UnsupportedCapability, match="max_turns"):
            async for _event in adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:success"),), max_turns=2),
                approvals=None,
            ):
                pass
    finally:
        await adapter.close()

    assert client.query_calls == []


async def test_new_resume_and_fork_preserve_native_sdk_session_options(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    values = environment(tmp_path)
    request = session_request(tmp_path)
    ref = AgentSessionRef(
        schema_version="agent-session-ref.v1",
        backend="claude",
        transport="sdk",
        native_session_id="session-existing",
        profile_key="fixture",
        state_root_fingerprint=fingerprint_path(values["CLAUDE_CONFIG_DIR"]),
        cwd_fingerprint=fingerprint_path(request.cwd),
    )
    adapter = ClaudeSdkAdapter()
    try:
        new_session = await open_session(adapter, request, environment=values)
        resumed = await open_session(
            adapter, session_request(tmp_path, open=ResumeSession(ref)), environment=values
        )
        forked = await open_session(
            adapter, session_request(tmp_path, open=ForkSession(ref)), environment=values
        )
    finally:
        await adapter.close()

    assert new_session.ref_is_complete is False
    assert resumed.ref == ref
    assert forked.ref_is_complete is False
    assert ClaudeSDKClient.instances[-2].options.resume == "session-existing"
    assert ClaudeSDKClient.instances[-2].options.fork_session is False
    assert ClaudeSDKClient.instances[-1].options.resume == "session-existing"
    assert ClaudeSDKClient.instances[-1].options.fork_session is True


async def test_structured_output_and_native_option_are_preserved_and_validated(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    schema = parse_canonical_schema(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(
            tmp_path,
            output=JsonSchemaAgentOutput(name="answer", schema=schema),
            native=ClaudeNativeOptions(include_partial_messages=False),
        ),
        environment=environment(tmp_path),
    )
    try:
        events = [
            event
            async for event in adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:structured_failure"),)),
                approvals=None,
            )
        ]
        options = ClaudeSDKClient.instances[-1].options
    finally:
        await adapter.close()

    assert options.include_partial_messages is False
    assert options.output_format is not None
    assert options.output_format["type"] == "json_schema"
    assert terminal_event_to_result(events[-1]).failure == "output_schema_violation"


async def test_sdk_accepts_native_structured_output_that_matches_schema(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    schema = parse_canonical_schema(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, output=JsonSchemaAgentOutput(name="answer", schema=schema)),
        environment=environment(tmp_path),
    )
    try:
        events = [
            event
            async for event in adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:structured_success"),)),
                approvals=None,
            )
        ]
    finally:
        await adapter.close()

    assert terminal_event_to_result(events[-1]).structured_output == {"answer": "ok"}


async def test_policy_and_mcp_configuration_are_mapped_and_startup_is_fail_closed(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    server = McpServerSpec(
        name="fixture",
        transport="streamable_http",
        url="https://example.invalid/mcp",
        required=True,
    )
    ClaudeSDKClient.mcp_status = {
        "mcpServers": [
            {
                "name": "fixture",
                "status": "connected",
                "serverInfo": {"name": "fixture-server", "version": "1.0.0"},
                "scope": "session",
            }
        ]
    }
    policy = PermissionPolicy(
        filesystem="workspace_write",
        network="allowlist",
        network_allowlist=("example.invalid",),
        approval="ask",
        allowed_tools=("Bash",),
        denied_tools=("Read",),
    )
    adapter = ClaudeSdkAdapter()
    try:
        await open_session(
            adapter,
            session_request(tmp_path, policy=policy, mcp_servers=(server,)),
            environment=environment(tmp_path),
        )
        options = ClaudeSDKClient.instances[-1].options
    finally:
        await adapter.close()

    assert options.permission_mode == "default"
    assert options.allowed_tools == []
    assert options.disallowed_tools == [
        "Read",
        "Glob",
        "Grep",
        "Write",
        "Edit",
        "NotebookEdit",
        "WebFetch",
        "WebSearch",
    ]
    network = options.sandbox["network"]
    assert isinstance(network, dict)
    assert network["allowedDomains"] == ["example.invalid"]
    assert options.mcp_servers["fixture"] == {
        "type": "http",
        "url": "https://example.invalid/mcp",
        "headers": {},
    }

    for reported in ("failed", "pending", "needs-auth", "disabled"):
        ClaudeSDKClient.mcp_status = {
            "mcpServers": [{"name": "fixture", "status": reported, "error": "handshake refused"}]
        }
        with pytest.raises(McpUnavailable, match=f"required.*{reported}"):
            await open_session(
                ClaudeSdkAdapter(),
                session_request(tmp_path, policy=policy, mcp_servers=(server,)),
                environment=environment(tmp_path),
            )

    # A server the CLI never mentions is not available either.
    ClaudeSDKClient.mcp_status = {"mcpServers": []}
    with pytest.raises(McpUnavailable, match="unreported"):
        await open_session(
            ClaudeSdkAdapter(),
            session_request(tmp_path, policy=policy, mcp_servers=(server,)),
            environment=environment(tmp_path),
        )


async def test_mcp_status_shape_mismatch_is_a_defect_not_an_expected_error(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    """`get_server_info()` has no MCP member; reading one back would be a silent never-fires."""
    ClaudeSDKClient.mcp_status = {
        "commands": [],
        "agents": [],
        "output_style": "default",
        "models": [],
        "account": {},
        "pid": 4242,
    }
    with pytest.raises(ProtocolDefect) as captured:
        await open_session(
            ClaudeSdkAdapter(), session_request(tmp_path), environment=environment(tmp_path)
        )
    assert captured.value.code == "sdk_shape_mismatch"


async def test_unrequested_mcp_server_is_refused_even_with_no_requested_servers(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    ClaudeSDKClient.mcp_status = {"mcpServers": [{"name": "ambient", "status": "connected"}]}
    with pytest.raises(ProtocolDefect, match="unrequested"):
        await open_session(
            ClaudeSdkAdapter(), session_request(tmp_path), environment=environment(tmp_path)
        )


async def test_sdk_rejects_mcp_tool_filters_it_cannot_preserve_before_client_start(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    server = McpServerSpec(
        name="fixture",
        transport="streamable_http",
        url="https://example.invalid/mcp",
        allowed_tools=("inspect",),
    )

    with pytest.raises(UnsupportedCapability, match="MCP tool filters"):
        await open_session(
            ClaudeSdkAdapter(),
            session_request(
                tmp_path,
                policy=PermissionPolicy(
                    network="allowlist", network_allowlist=("example.invalid",)
                ),
                mcp_servers=(server,),
            ),
            environment=environment(tmp_path),
        )

    assert ClaudeSDKClient.instances == []


async def test_optional_mcp_failure_is_a_stream_diagnostic(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    server = McpServerSpec(
        name="optional",
        transport="streamable_http",
        url="https://example.invalid/mcp",
        required=False,
    )
    ClaudeSDKClient.mcp_status = {
        "mcpServers": [{"name": "optional", "status": "failed", "error": "connect ECONNREFUSED"}]
    }
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(
            tmp_path,
            policy=fixture_policy(network="allowlist", network_allowlist=("example.invalid",)),
            mcp_servers=(server,),
        ),
        environment=environment(tmp_path),
    )
    try:
        events = await drain(
            adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:success"),)),
                approvals=None,
            )
        )
    finally:
        await adapter.close()

    diagnostic = next(
        event
        for event in events
        if isinstance(event.data, DiagnosticData) and event.data.code == "optional_mcp_unavailable"
    )
    assert "connect ECONNREFUSED" in cast(DiagnosticData, diagnostic.data).message
    # Reported by the SDK at session startup, so the consumer sees it there: in the
    # session-scope window `events.SESSION_SCOPE_EVENT_KINDS` opens between `session_started`
    # and `turn_started`, not held back until the turn had opened. Both lanes of this package
    # show the backend's own ordering; neither reorders its frames to suit the grammar.
    assert [event.kind for event in events[:3]] == [
        "session_started",
        "diagnostic",
        "turn_started",
    ]
    assert diagnostic.seq == 2


async def test_sdk_retained_text_limit_becomes_one_terminal_failure(
    installed_sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(claude_module, "_MAX_FINAL_TEXT_BYTES", 4)
    events = await stream_fixture(tmp_path, "success", installed_sdk)

    assert [event.kind for event in events].count("turn_failed") == 1
    assert events[-1].kind == "turn_failed"
    assert terminal_event_to_result(events[-1]).failure == "output_limit_exceeded"


async def test_sdk_turn_setup_failure_does_not_expose_provider_text(
    installed_sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter, session_request(tmp_path), environment=environment(tmp_path)
    )
    client = ClaudeSDKClient.instances[-1]

    async def broken_query(_prompt: str, session_id: str = "default") -> None:
        del session_id
        raise CLIConnectionError("authorization=Bearer fixture-secret-must-not-leak")

    monkeypatch.setattr(client, "query", broken_query)
    try:
        with pytest.raises(SdkUnavailable) as captured_error:
            async for _event in adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:success"),)),
                approvals=None,
            ):
                pass
    finally:
        await adapter.close()

    assert "fixture-secret" not in repr(captured_error.value)
    assert captured_error.value.__cause__ is None


async def test_sdk_stream_failure_is_one_backend_failed_terminal(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, policy=fixture_policy()),
        environment=environment(tmp_path),
    )
    ClaudeSDKClient.instances[-1].stream_failure = ProcessError("claude exited with code 1")
    try:
        events = await drain(
            adapter.stream_turn(
                session,
                TurnRequest(input=(TextContent("fixture:success"),)),
                approvals=None,
            )
        )
    finally:
        await adapter.close()

    assert [event.kind for event in events].count("turn_failed") == 1
    assert terminal_event_to_result(events[-1]).failure == "backend_failed"


async def test_non_sdk_stream_exception_propagates_instead_of_becoming_a_turn_result(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    """A defect of ours must never be dressed up as a `backend_failed` turn the caller retries."""
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, policy=fixture_policy()),
        environment=environment(tmp_path),
    )
    ClaudeSDKClient.instances[-1].stream_failure = KeyError("permission/request")
    try:
        with pytest.raises(KeyError):
            await drain(
                adapter.stream_turn(
                    session,
                    TurnRequest(input=(TextContent("fixture:success"),)),
                    approvals=None,
                )
            )
    finally:
        await adapter.close()


async def test_unreapable_owned_process_is_a_cleanup_defect_not_sdk_unavailable(
    installed_sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`SdkUnavailable` means the optional extra is missing; a surviving child is a defect."""
    adapter = ClaudeSdkAdapter()
    await open_session(adapter, session_request(tmp_path), environment=environment(tmp_path))
    client = ClaudeSDKClient.instances[-1]
    spawned = client.process
    assert spawned is not None

    async def broken_disconnect() -> None:
        raise CLIConnectionError("stdin already closed")

    async def never_exits() -> int:
        await asyncio.Event().wait()
        return 0

    # The escalation itself is what is under test, so the child it escalates against has to
    # be one that does not answer: a real SIGKILL always lands, and then there is no defect.
    monkeypatch.setattr(client, "disconnect", broken_disconnect)
    monkeypatch.setattr(
        client,
        "_transport",
        SimpleNamespace(
            _process=SimpleNamespace(pid=spawned.pid, kill=lambda: None, wait=never_exits)
        ),
    )
    with pytest.raises(ProtocolDefect) as captured:
        await adapter.close()
    assert captured.value.code == "process_cleanup_failed"
    # The failure is reported, and the owned group is still signalled rather than abandoned.
    assert await spawned.wait() == -signal.SIGTERM


async def test_permission_callback_defect_reaches_the_stream(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    """The SDK swallows a callback exception into a control response, so it must be handed over."""
    case = approval_case("handler_allow")
    adapter = ClaudeSdkAdapter()
    session = await open_session(
        adapter,
        session_request(tmp_path, policy=case_policy(case)),
        environment=environment(tmp_path),
    )
    state = claude_module.ClaudeSdkAdapter._state(adapter, session)

    async def missing_handler(_request: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError("handler must not run")

    try:
        stream = adapter.stream_turn(
            session,
            TurnRequest(input=(TextContent(f"fixture:approval_{case['name']}"),)),
            approvals=missing_handler,
        )
        iterator = stream.__aiter__()
        await anext(iterator)
        state.current_approvals = None
        with pytest.raises(ProtocolDefect, match="without the handler"):
            while True:
                await asyncio.sleep(0)
                await anext(iterator)
    finally:
        await adapter.close()


async def test_session_discovery_never_reads_the_ambient_sdk_store(
    installed_sdk: SimpleNamespace, tmp_path: Path
) -> None:
    cwd = str((tmp_path / "repo").resolve())
    Path(cwd).mkdir(exist_ok=True)

    def ambient_store_access(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ambient Claude session store was accessed")

    installed_sdk.list_sessions = ambient_store_access
    installed_sdk.get_session_info = ambient_store_access
    adapter = ClaudeSdkAdapter()
    scope = AgentCapabilityScope(backend="claude", transport="sdk", auth=AUTH)
    values = environment(tmp_path)
    ref = AgentSessionRef(
        schema_version="agent-session-ref.v1",
        backend="claude",
        transport="sdk",
        native_session_id="session-discovered",
        profile_key="fixture",
        state_root_fingerprint=fingerprint_path(values["CLAUDE_CONFIG_DIR"]),
        cwd_fingerprint=fingerprint_path(cwd),
    )

    with pytest.raises(UnsupportedCapability, match="isolated state root"):
        await adapter.list_sessions(SessionQuery(scope=scope), environment=values)
    with pytest.raises(UnsupportedCapability, match="isolated state root"):
        await adapter.read_session(
            ref,
            SessionReadOptions(auth=AUTH, include_turns=False, include_items=False),
            environment=values,
        )


async def open_session(
    selected: ClaudeSdkAdapter,
    request: AgentSessionRequest,
    *,
    environment: Mapping[str, str],
) -> AgentSession:
    """Open a session the way ``AgentRuntime`` does: discover once, validate, then hand over.

    The adapter no longer rediscovers its own capability table, so a direct-adapter test that
    skipped this step would be exercising a call the runtime never makes.
    """
    capabilities = await selected.capabilities(
        AgentCapabilityScope(
            backend=request.backend, transport=request.transport, auth=request.auth
        ),
        environment=environment,
    )
    validate_session_capabilities(request, capabilities)
    return await selected.open_session(request, capabilities=capabilities, environment=environment)
