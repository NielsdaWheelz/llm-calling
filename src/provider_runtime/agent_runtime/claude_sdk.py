"""Native Claude Agent SDK adapter with a lazy optional dependency boundary."""

from __future__ import annotations

import asyncio
import importlib
import os
import re
import shutil
import sys
import warnings
import weakref
from collections.abc import AsyncGenerator, Iterator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

from provider_runtime.errors import sanitize_provider_text
from provider_runtime.types import Absent, Presence, Present, TokenUsage

from ._claude_launcher import OwnedProcessGroup, ensure_claude_launcher
from ._limits import OutputLimitExceeded, bounded_payload_size
from ._process import ProcessLimits, capture_process_output
from ._sandbox import bubblewrap_network_namespace_available, environment_executable
from ._structured_output import (
    OutputSchemaMismatch,
    freeze_structured_output,
    parse_structured_output,
)
from .auth import (
    credential_environment_names,
    redact_native_payload,
    state_root_from_environment,
)
from .auth import (
    freeze_native_json_object as freeze_json_object,
)
from .auth import (
    freeze_native_json_value as freeze_json_value,
)
from .errors import (
    AgentRuntimeDefect,
    AgentRuntimeError,
    ExecutableUnavailable,
    InvalidAgentRequest,
    McpUnavailable,
    MissingTerminalEvent,
    ProtocolDefect,
    SdkUnavailable,
    SessionMismatch,
    SessionUnavailable,
    UnsupportedCapability,
)
from .events import (
    AgentEvent,
    AgentFailure,
    AgentNative,
    AgentPermissionRequest,
    AgentQuotaExhausted,
    AgentTerminal,
    AgentTerminalFailure,
    AgentText,
    AgentToolUse,
    AgentUsage,
)
from .policy import PermissionPolicy, tool_is_allowed
from .sessions import (
    AgentSession,
    SessionPage,
    SessionQuery,
    SessionReadOptions,
    SessionSnapshot,
    fingerprint_path,
    validate_read_session_auth,
    validate_session_ref,
)
from .types import (
    AgentSessionRef,
    AgentSessionRequest,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    ClaudeNativeOptions,
    CredentialRef,
    ForkSession,
    FrozenJsonDict,
    JsonSchemaAgentOutput,
    NewSession,
    ResumeSession,
    TextContent,
    TurnRequest,
    thaw_json_value,
    validate_mcp_network_policy,
)

# Claude Code 2.1.220 builds `system/init` in `tAr()` as `tools: e.tools.map(o => q_n(o.name))`
# with `q_n(e) { return e === "Agent" ? "Task" : e }`, so the one internal tool named `Agent`
# is reported under its user-facing name. Every other tool is reported verbatim.
_INIT_TOOL_ALIASES: dict[str, str] = {"Agent": "Task"}
# The exact native tool names this lane accepts in `PermissionPolicy.allowed_tools`, per
# built-in family, pinned to `_SUPPORTED_CLAUDE_VERSION`: the Agent SDK ships no tool-name
# constant and `system/init` only echoes back the tools the session already asked for, so
# nothing discovers these at runtime.
#
# Read out of the installed 2.1.220 binary rather than written from memory. Each name is a
# tool-name constant carried by a tool object the default pool builder `R6()` registers:
# `zi="Read"`, `kd="Glob"`, `bd="Grep"`, `nu="Write"`, `fl="Edit"`, `LT="NotebookEdit"`,
# `ri="Bash"`. Naming `Glob`/`Grep` here is also what keeps them available: 2.1.220 sets its
# `searchToolsOptIn` flag from the `--tools`/`--allowedTools` lists and, without that opt-in,
# drops both tools from the pool in favour of `Bash` `find`/`grep`. Names the same binary
# only keeps as legacy *permission-rule* spellings — `NotebookRead`, `MultiEdit`, `BashOutput`
# and `KillShell`, the last two aliased to `TaskOutput`/`TaskStop` — register no tool and are
# deliberately absent, as is `PowerShell`, which 2.1.220 registers on Windows only.
#
# This table is behavioral policy enforcement, not a capability advertisement: it is the
# single source of both the accepted-name set and the approval-operation classification, so
# a name can never be treated as a file write on one path and an ordinary tool use on the
# other.
_BUILTIN_TOOL_NAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("file_read", ("Read", "Glob", "Grep")),
    ("file_write", ("Write", "Edit", "NotebookEdit")),
    ("command", ("Bash",)),
)
# Built-in tools 2.1.220 ships that this lane refuses outright. Both reach the network
# through Claude Code's own client rather than through the sandbox this adapter always
# engages, so a policy naming either would get network access that
# `PermissionPolicy.network` never granted.
_REFUSED_TOOL_NAMES: tuple[str, ...] = ("WebFetch", "WebSearch")
_FILE_CHANGE_TOOL_NAMES = frozenset(dict(_BUILTIN_TOOL_NAMES)["file_write"])
_COMMAND_TOOL_NAMES = frozenset(dict(_BUILTIN_TOOL_NAMES)["command"])
# Claude Code names MCP tools `mcp__<server>__<tool>` (2.1.220 `kte()`), so an MCP-provided tool
# is recognizable by prefix; the server set behind it is verified against `mcp_status` before the
# session is published.
_MCP_TOOL_PREFIX = "mcp__"
_MCP_CONNECTED_STATUS = "connected"
_SDK_PERMISSION_MODE = "default"
_INTERRUPT_TIMEOUT_SECONDS = 2.0
# How long an interrupted turn's tail may take to arrive before the session is declared
# unusable. It is the interrupt bound plus room for one in-flight tool result to land.
_DRAIN_TIMEOUT_SECONDS = 10.0
# The grace descendants of a stopped Claude Code get between SIGTERM and SIGKILL. The SDK's
# own close has already spent up to ten seconds on the leader by the time this runs, so this
# covers the orphans it left behind, not a polite shutdown of the agent itself.
_GROUP_TERMINATION_GRACE_SECONDS = 1.0
# How long the launched child may take to become its own session leader. One interpreter
# start is the real window; the rest is slack for a loaded machine.
_GROUP_ADOPTION_TIMEOUT_SECONDS = 2.0
# The SDK spawns a local Claude Code build rather than talking to a service, so the installed
# executable is as much a version surface as the SDK package is. The probe runs before any
# session exists and gets its own short bound.
_VERSION_TIMEOUT_SECONDS = 10.0
# The vetted versions are a warning threshold, not a gate: a drifted install gets exactly one
# RuntimeWarning and then answers to the behavioral `system/init` verification, which is the
# check that actually protects the policy contract.
_SUPPORTED_CLAUDE_VERSION = "2.1.220"
_CLAUDE_VERSION = re.compile(r"([0-9]+(?:\.[0-9]+){2}(?:[-+][0-9A-Za-z.-]+)?) \(Claude Code\)\Z")
_VERSION_STDOUT_LIMIT = 16 * 1024
# Claude Code 2.1.220 reports a rate/usage-limit refusal three ways and none of them is an
# `error_code` field: a `rate_limit_event` whose `rate_limit_info.status` is "rejected", an
# assistant frame whose top-level `error` is "rate_limit" (the binary maps HTTP 429 to exactly
# that literal), and `api_error_status: 429` on an `is_error` result. All three reach this
# adapter, and they are spelled once here so no reader of one branch can drift from another.
_QUOTA_HTTP_STATUS = 429
_QUOTA_ASSISTANT_ERROR = "rate_limit"
_QUOTA_RATE_LIMIT_STATUS = "rejected"
_PROCESS_LIMITS = ProcessLimits(max_stderr_bytes=64 * 1024, termination_grace_seconds=0.25)
_DISCONNECT_TIMEOUT_SECONDS = 25.0
_OPERATION_TIMEOUT_SECONDS = 30.0
_MAX_EVENT_COUNT = 100_000
_MAX_IDENTITY_MESSAGES = 64
_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_MAX_MESSAGE_ITEMS = 100_000
_MAX_TURN_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_EVENT_TEXT_BYTES = 4 * 1024 * 1024
_MAX_FINAL_TEXT_BYTES = 16 * 1024 * 1024
_MAX_DIAGNOSTICS = 256
_SUPPORTED_SDK_VERSION = "0.2.130"


async def _read_claude_code_version(
    executable: str,
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> str:
    """Probe the installed Claude Code build; drift warns, only silence fails.

    An executable that cannot be run or does not answer stays `ExecutableUnavailable` —
    there is nothing behavioral left to verify against. A build that answers with an
    unexpected version keeps running behind the `system/init` verification instead.
    """
    text = await capture_process_output(
        (executable, "--version"),
        cwd=cwd,
        environment=environment,
        limits=_PROCESS_LIMITS,
        startup_timeout_seconds=_VERSION_TIMEOUT_SECONDS,
        max_stdout_bytes=_VERSION_STDOUT_LIMIT,
        executable_label="Claude CLI",
        purpose="Claude CLI version discovery",
    )
    match = _CLAUDE_VERSION.fullmatch(text)
    if match is None:
        warnings.warn(
            f"Claude CLI answered an unrecognized version string "
            f"{sanitize_provider_text(text, limit=64)!r}; expected "
            f"{_SUPPORTED_CLAUDE_VERSION!r} — continuing behind the behavioral "
            "system/init verification",
            RuntimeWarning,
            stacklevel=2,
        )
        return text
    version = match.group(1)
    if version != _SUPPORTED_CLAUDE_VERSION:
        warnings.warn(
            f"Claude Code {version} is not the vetted {_SUPPORTED_CLAUDE_VERSION} — "
            "continuing behind the behavioral system/init verification",
            RuntimeWarning,
            stacklevel=2,
        )
    return version


async def _network_sandbox_available(*, cwd: Path, environment: Mapping[str, str]) -> bool:
    """Whether this host supports Claude's bubblewrap-and-socat network proxy."""
    if environment_executable("socat", environment) is None:
        return False
    return await bubblewrap_network_namespace_available(cwd=cwd, environment=environment)


@dataclass(slots=True)
class _ClaudeQuotaSignal:
    """One turn's accumulated evidence that Claude Code refused it for quota.

    The two earlier observations are latched because they arrive on frames that are not the
    terminal: by the time the failing `result` lands, the `rate_limit_event` or the errored
    assistant frame that explained it has already been emitted and forgotten.
    """

    observed: bool = False

    def observe_rate_limit_status(self, status: object) -> None:
        if status == _QUOTA_RATE_LIMIT_STATUS:
            self.observed = True

    def observe_assistant_error(self, error: object) -> None:
        if error == _QUOTA_ASSISTANT_ERROR:
            self.observed = True

    def is_exhausted(self, api_error_status: object) -> bool:
        """Decide one failed turn's cause from the latched evidence plus its own status."""
        return self.observed or api_error_status == _QUOTA_HTTP_STATUS


@dataclass(slots=True)
class _ClaudeSessionState:
    sdk: Any
    client: Any
    request: AgentSessionRequest
    state_root: Path
    ref: AgentSessionRef | None
    process_group: OwnedProcessGroup | None = None
    # A native turn was started and no `ResultMessage` for it has been consumed yet.
    turn_in_flight: bool = False
    # An interrupted turn's tail is still queued on the client's shared message stream and
    # must be discarded before the next turn reads it.
    drain_pending: bool = False
    message_count: int = 0
    output_bytes: int = 0
    turn_id: str | None = None
    final_text: str = ""
    final_text_bytes: int = 0
    usage: TokenUsage | None = None
    diagnostics: list[str] = field(default_factory=list)
    # What `open_session` observed about this session before any turn existed.
    # `open_session` returns a session rather than a stream, so there is no earlier event
    # this could be: the first stream emits it at its head and empties the list, so only
    # that stream carries it.
    pending_startup_diagnostics: list[str] = field(default_factory=list)
    quota: _ClaudeQuotaSignal = field(default_factory=_ClaudeQuotaSignal)
    approval_failure: bool = False
    approval_abort: bool = False
    current_policy: PermissionPolicy | None = None
    current_approvals: ApprovalHandler | None = None
    # `ToolResultBlock` carries only the call id, so the started block's name is remembered
    # here for the completion event; the map is one turn's, cleared at every turn start.
    tool_names: dict[str, str] = field(default_factory=dict)
    # Events described off the turn generator's task — the permission callback runs on a
    # task the SDK spawns itself — queued here and emitted by the generator in stream order.
    pending_events: list[AgentEvent] = field(default_factory=list)
    approval_defect: AgentRuntimeDefect | None = None


class ClaudeSdkAdapter:
    """Own Claude SDK clients while keeping the SDK import fully optional."""

    backend: Literal["claude"] = "claude"
    transport: Literal["sdk"] = "sdk"
    # Claude Code scopes its native session store by working directory, so a resumed ref
    # must match the requesting cwd. A backend fact, spelled once for the runtime.
    cwd_scopes_sessions: Literal[True] = True

    def __init__(self, *, executable: str = "claude") -> None:
        if type(executable) is not str or not executable or "\0" in executable:
            raise InvalidAgentRequest("Claude SDK executable must be a non-empty path or name")
        resolved = shutil.which(executable)
        self._executable = str(Path(resolved).resolve()) if resolved is not None else None
        self._requested_executable = executable
        self._sessions: dict[AgentSession, _ClaudeSessionState] = {}
        self._dead_sessions: weakref.WeakSet[AgentSession] = weakref.WeakSet()

    def validate_auth(self, credential: CredentialRef) -> None:
        self._require_local_auth(credential.kind)

    async def list_sessions(
        self,
        query: SessionQuery,
        *,
        environment: Mapping[str, str],
    ) -> SessionPage:
        if query.backend != self.backend or query.transport != self.transport:
            raise InvalidAgentRequest("ClaudeSdkAdapter received a different route")
        self._require_local_auth(query.auth.kind)
        del environment
        raise UnsupportedCapability(
            "Claude SDK discovery cannot be scoped to the selected isolated state root"
        )

    async def read_session(
        self,
        ref: AgentSessionRef,
        options: SessionReadOptions,
        *,
        environment: Mapping[str, str],
    ) -> SessionSnapshot:
        if ref.backend != self.backend or ref.transport != self.transport:
            raise SessionMismatch("Claude SDK cannot read a different route")
        validate_read_session_auth(ref, options)
        self._require_local_auth(options.auth.kind)
        if ref.state_root_fingerprint != fingerprint_path(
            state_root_from_environment("claude", environment)
        ):
            raise SessionMismatch("session state root does not match the supplied environment")
        raise UnsupportedCapability(
            "Claude SDK session reads cannot be scoped to the selected isolated state root"
        )

    async def open_session(
        self,
        request: AgentSessionRequest,
        *,
        environment: Mapping[str, str],
    ) -> AgentSession:
        if request.backend != self.backend or request.transport != self.transport:
            raise InvalidAgentRequest("ClaudeSdkAdapter received a different route")
        self._require_local_auth(request.auth.kind)
        self._validate_policy_mapping(request.policy)
        validate_mcp_network_policy(request.mcp_servers, request.policy)
        sdk = self._load_sdk()
        state_root = state_root_from_environment("claude", environment)
        ref: AgentSessionRef | None = None
        resume: str | None = None
        fork = False
        if isinstance(request.open, ResumeSession | ForkSession):
            validate_session_ref(
                request.open.ref,
                backend="claude",
                transport="sdk",
                profile_key=request.auth.profile_key,
                state_root_fingerprint=fingerprint_path(state_root),
                cwd=request.cwd,
                cwd_scopes_sessions=True,
            )
            resume = request.open.ref.native_session_id
            fork = isinstance(request.open, ForkSession)
            ref = None if fork else request.open.ref
        elif not isinstance(request.open, NewSession):
            raise InvalidAgentRequest("unknown Claude SDK session operation")

        executable = self._require_executable()
        # There is no capability table. The version gates are one warning each plus the
        # behavioral `system/init` verification below, and the only hard host requirement —
        # the bubblewrap/socat proxy behind `network: allowlist` — fails closed here,
        # before any SDK startup.
        await _read_claude_code_version(executable, cwd=state_root, environment=environment)
        if request.policy.network == "allowlist" and not await _network_sandbox_available(
            cwd=state_root, environment=environment
        ):
            raise UnsupportedCapability(
                "Claude network allowlist requires the bubblewrap/socat sandbox on this host"
            )

        holder: list[_ClaudeSessionState] = []

        async def can_use_tool(name: str, tool_input: dict[str, Any], context: Any) -> Any:
            if not holder:
                raise ProtocolDefect("Claude approval callback ran before session initialization")
            return await self._handle_permission(holder[0], name, tool_input, context)

        # Everything that can still reject the request runs before the launcher is written,
        # so a rejected session never leaves a file behind.
        system_prompt = self._optional_instruction(request.system)
        mcp_servers = self._mcp_servers(request)
        output_format = self._output_format(request.output)
        thinking = self._thinking(request)
        launcher = ensure_claude_launcher(state_root, executable, interpreter=sys.executable)
        accepted_tools = tuple(name for _family, names in _BUILTIN_TOOL_NAMES for name in names)
        denied_tools = tuple(
            dict.fromkeys(
                (
                    *request.policy.denied_tools,
                    *(name for name in accepted_tools if name not in request.policy.allowed_tools),
                    *_REFUSED_TOOL_NAMES,
                )
            )
        )
        options = sdk.ClaudeAgentOptions(
            tools=list(request.policy.allowed_tools),
            allowed_tools=[],
            disallowed_tools=list(denied_tools),
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            strict_mcp_config=True,
            permission_mode=_SDK_PERMISSION_MODE,
            resume=resume,
            model=request.model,
            output_format=output_format,
            cwd=request.cwd,
            add_dirs=list(request.additional_dirs),
            # Not the `claude` executable itself: the SDK spawns `cli_path` without
            # `start_new_session`, so the launcher is what makes the child a process-group
            # leader this runtime can reap whole. See `_claude_launcher`.
            cli_path=str(launcher),
            env=self._sdk_environment(environment),
            can_use_tool=can_use_tool,
            include_partial_messages=self._include_partial(request.native),
            fork_session=fork,
            setting_sources=[],
            sandbox=self._sandbox(request.policy),
            thinking=thinking,
            effort=request.reasoning.effort if request.reasoning is not None else None,
        )
        client = None
        group: OwnedProcessGroup | None = None
        try:
            try:
                client = sdk.ClaudeSDKClient(options=options)
                state = _ClaudeSessionState(
                    sdk=sdk,
                    client=client,
                    request=request,
                    state_root=state_root,
                    ref=ref,
                )
                holder.append(state)
                async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                    await client.connect()
                    # Fail the open rather than run a session whose descendants this
                    # runtime could not terminate: an unowned group is exactly the state
                    # the launcher exists to prevent, and it is never safe to signal.
                    group = await OwnedProcessGroup.adopt(
                        self._owned_process(client).pid,
                        timeout_seconds=_GROUP_ADOPTION_TIMEOUT_SECONDS,
                    )
                    state.process_group = group
                    # `get_server_info()` only replays the CLI's `initialize` control response,
                    # which carries commands/agents/output styles and has no MCP member at all.
                    # `get_mcp_status()` issues the CLI's live `mcp_status` control request
                    # (Claude Code 2.1.220: `case "mcp_status": ... response:{mcpServers:Yr()}`),
                    # so it is the only surface that can answer the fail-closed startup check
                    # before the session is published.
                    mcp_status = await client.get_mcp_status()
                state.pending_startup_diagnostics.extend(
                    self._mcp_startup_diagnostics(request, mcp_status)
                )
            except FileNotFoundError:
                raise ExecutableUnavailable(
                    "Claude SDK executable disappeared before Claude Code started"
                ) from None
            except (TimeoutError, ImportError):
                raise SdkUnavailable("claude-agent-sdk could not start Claude Code") from None
            except sdk.ClaudeSDKError:
                raise SdkUnavailable("claude-agent-sdk failed during session startup") from None
        except BaseException:
            if client is not None:
                await self._disconnect_client(sdk, client, group)
            raise
        session = AgentSession(ref)
        self._sessions[session] = state
        return session

    async def stream_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None,
    ) -> AsyncGenerator[AgentEvent, None]:
        state = self._state(session)
        if request.policy is not None:
            raise UnsupportedCapability(
                "Claude SDK cannot reconfigure policy on a persistent client"
            )
        policy = state.request.policy
        self._validate_policy_mapping(policy)
        if policy.approval == "ask" and approvals is None:
            raise InvalidAgentRequest("approval='ask' requires an approval handler")
        if (
            request.system is not None
            or request.developer is not None
            or request.reasoning is not None
            or request.mcp_servers is not None
            or request.max_turns is not None
            or request.output is not None
            or request.native is not None
        ):
            raise UnsupportedCapability(
                "Claude SDK instructions, reasoning, MCP, max_turns, output, and native options "
                "are unavailable as per-turn overrides"
            )
        if request.model is not None:
            raise UnsupportedCapability(
                "Claude SDK model changes persist and are not per-turn overrides"
            )

        state.message_count = 0
        state.output_bytes = 0
        state.turn_id = None
        state.final_text = ""
        state.final_text_bytes = 0
        state.usage = None
        state.diagnostics[:] = state.pending_startup_diagnostics
        state.quota = _ClaudeQuotaSignal()
        state.approval_failure = False
        state.approval_abort = False
        state.current_policy = policy
        state.current_approvals = approvals
        state.tool_names.clear()
        state.pending_events.clear()
        state.approval_defect = None
        if state.drain_pending:
            await self._drain_retired_turn(state)
        prompt = self._turn_text(request)
        try:
            await state.client.query(
                prompt,
                session_id=state.ref.native_session_id if state.ref is not None else "default",
            )
        except state.sdk.ClaudeSDKError:
            # No turn identity exists yet, so there is no stream to terminate with a value:
            # the caller never saw an event and the turn is not in flight.
            await self._stop_client(state)
            raise SdkUnavailable("Claude SDK could not start the turn") from None
        state.turn_in_flight = True

        source = state.client.receive_response().__aiter__()
        buffered: list[object] = []
        try:
            while state.turn_id is None or state.ref is None:
                try:
                    message = await anext(source)
                except StopAsyncIteration:
                    raise MissingTerminalEvent() from None
                except state.sdk.ClaudeSDKError:
                    raise SdkUnavailable("Claude SDK failed before stream identity") from None
                if len(buffered) >= _MAX_IDENTITY_MESSAGES:
                    raise ProtocolDefect(
                        "Claude SDK did not establish stream identity within the message limit",
                        code="missing_stream_identity",
                    )
                try:
                    self._account_message(state, message)
                except OutputLimitExceeded:
                    raise ProtocolDefect(
                        "Claude SDK exceeded output limits before stream identity",
                        code="output_limit_before_identity",
                    ) from None
                buffered.append(message)
                self._learn_identity(state, message)
        except Exception:
            # Cancellation is deliberately not handled here: the runtime interrupts the turn
            # through `interrupt()` and the session stays usable, while every failure that
            # reaches this point means the stream never became usable at all.
            await self._stop_client(state, force_disconnect=True)
            raise
        ref = state.ref
        if ref is None or state.turn_id is None:
            raise ProtocolDefect("Claude SDK did not establish stream identity")
        # The runtime defects on any event from a session whose ref is incomplete, so the
        # learned identity is published before the first yield.
        if not session.ref_is_complete:
            session.complete_ref(ref)

        # Session-scope MCP startup facts, observed by `open_session` before any turn
        # existed: they ride at the head of the first stream as native frames and were
        # already seeded into this turn's terminal diagnostics above.
        for diagnostic in state.pending_startup_diagnostics:
            yield AgentNative(
                native_type="mcp_status",
                payload=redact_native_payload({"diagnostic": diagnostic}),
            )
        state.pending_startup_diagnostics.clear()

        terminal_seen = False
        try:
            for message in buffered:
                for event in self._message_events(state, message):
                    yield event
                    if isinstance(event, AgentTerminal):
                        terminal_seen = True
                        break
                if terminal_seen:
                    return
            async for message in source:
                self._account_message(state, message)
                self._learn_identity(state, message)
                for event in self._message_events(state, message):
                    yield event
                    if isinstance(event, AgentTerminal):
                        terminal_seen = True
                        break
                if terminal_seen:
                    return
        except OutputLimitExceeded:
            await self._stop_client(state, force_disconnect=True)
            yield AgentTerminal(
                status="failed",
                failure=AgentFailure("output_limit_exceeded"),
                final_text=state.final_text,
                session_ref=ref,
                usage=self._usage_presence(state),
                diagnostics=tuple(dict.fromkeys(state.diagnostics)),
            )
            return
        except asyncio.CancelledError:
            try:
                await self._stop_client(state)
            except AgentRuntimeDefect as error:
                # justify-ignore-error: the caller is being cancelled and must receive its
                # CancelledError, so a teardown failure cannot replace it. Record it as a turn
                # diagnostic instead of discarding it silently; the session is already marked
                # dead by `_stop_client`, so nothing later can consume a stale client.
                self._record_diagnostic(state, f"Claude SDK teardown failed: {error.code}")
            raise
        except (AgentRuntimeError, AgentRuntimeDefect):
            await self._stop_client(state, force_disconnect=True)
            raise
        except state.sdk.ClaudeSDKError:
            # Only the SDK's own error hierarchy is a backend failure. Anything else reaching
            # here is our defect and must keep propagating rather than becoming a product state.
            await self._stop_client(state, force_disconnect=True)
            yield AgentTerminal(
                status="failed",
                failure=AgentFailure("backend_failed"),
                final_text=state.final_text,
                session_ref=ref,
                usage=self._usage_presence(state),
                diagnostics=("Claude SDK stream failed",),
            )
            return
        except Exception:
            # Not a modelled failure: tear the client down and let the defect propagate.
            await self._stop_client(state, force_disconnect=True)
            raise
        if not terminal_seen:
            await self._stop_client(state, force_disconnect=True)
            raise MissingTerminalEvent()

    async def interrupt(self, session: AgentSession) -> None:
        if session in self._dead_sessions:
            return
        state = self._state(session)
        if state.turn_id is None:
            # No native turn was ever named, so nothing identifies the frames still coming
            # and no later turn could tell them apart: the whole session is released.
            await self._stop_client(state, force_disconnect=True)
            return
        await self._stop_client(state)

    async def close_session(self, session: AgentSession) -> None:
        """Idempotently disconnect one SDK client and its owned process group."""
        if session in self._dead_sessions:
            return
        state = self._sessions.pop(session, None)
        if state is None:
            raise InvalidAgentRequest("session is not owned by this Claude adapter")
        self._dead_sessions.add(session)
        await self._disconnect_client(state.sdk, state.client, state.process_group)

    async def close(self) -> None:
        sessions = tuple(self._sessions)
        results = await asyncio.gather(
            *(self.close_session(session) for session in sessions),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            # Teardown owns the "no surviving owned process" invariant, so a close that could
            # not establish it is a defect, not a caller-handleable error.
            raise ProtocolDefect(
                "Claude SDK session teardown did not complete",
                code="process_cleanup_failed",
            ) from None

    async def _drain_retired_turn(self, state: _ClaudeSessionState) -> None:
        """Discard an interrupted turn's tail before this turn's stream can inherit it.

        `ClaudeSDKClient.receive_response()` is a thin filter over the one shared
        `receive_messages()` stream (0.2.130 `client.py:571`), so the abandoned turn's
        remaining messages — including its own `ResultMessage` — are what the next
        `receive_response()` reads, and the next turn would terminate on the previous turn's
        result.

        The drain cannot run inside `interrupt()`, where the protocol documents the
        obligation: the runtime issues that interrupt from `_watch_turn_stop` while the
        abandoned turn's `anext` is still pending, and a second consumer of the same memory
        stream would race it for messages. Here the abandoned generator is closed and
        nothing else is reading, which is the protocol's other sanctioned option —
        discarding the frames on arrival.
        """
        source = state.client.receive_response()
        try:
            async with asyncio.timeout(_DRAIN_TIMEOUT_SECONDS):
                async for message in source:
                    if isinstance(message, state.sdk.ResultMessage):
                        state.drain_pending = False
                        state.turn_in_flight = False
                        return
        except (TimeoutError, state.sdk.ClaudeSDKError):
            # justify-ignore-error: the tail did not arrive, so the frames still queued can
            # no longer be placed in any turn. Which of the two ways that happened does not
            # change the answer, and the session is released either way below.
            pass
        finally:
            await source.aclose()
        await self._stop_client(state, force_disconnect=True)
        raise SessionUnavailable("the interrupted Claude turn did not release its message stream")

    def _message_events(self, state: _ClaudeSessionState, message: object) -> Iterator[AgentEvent]:
        """Emit one native message's events, plus anything the permission callback queued.

        The permission callback runs on a task the SDK spawns itself
        (`Query._spawn_control_request_handler`) while this generator's consumer is
        suspended at a yield, so the callback only queues descriptions. Draining the queue
        at every yield boundary keeps an approval event ahead of the tool event it gated.
        """
        yield from self._drain_pending(state)
        for event in self._decode_message(state, message):
            yield from self._drain_pending(state)
            yield event
        yield from self._drain_pending(state)

    def _drain_pending(self, state: _ClaudeSessionState) -> list[AgentEvent]:
        """Release everything the permission callback queued, and surface any defect it hit."""
        defect = state.approval_defect
        if defect is not None:
            state.approval_defect = None
            raise defect
        queued = list(state.pending_events)
        state.pending_events.clear()
        return queued

    def _decode_message(self, state: _ClaudeSessionState, message: object) -> Iterator[AgentEvent]:
        sdk = state.sdk
        if isinstance(message, sdk.SystemMessage):
            subtype = getattr(message, "subtype", None)
            if not isinstance(subtype, str) or not subtype:
                raise ProtocolDefect("Claude SystemMessage carried no subtype")
            if subtype == "init":
                # The behavioral capability probe: `system/init` is the only frame that
                # reports the backend's *effective* configuration.
                self._verify_initialization(state, message)
            yield self._native(f"SystemMessage:{subtype}", message)
            return
        if isinstance(message, sdk.StreamEvent):
            event = getattr(message, "event", None)
            if not isinstance(event, dict):
                raise ProtocolDefect("Claude StreamEvent.event was not an object")
            delta = event.get("delta")
            if event.get("type") == "content_block_delta" and isinstance(delta, dict):
                if delta.get("type") == "text_delta":
                    text = delta.get("text")
                    if not isinstance(text, str):
                        raise ProtocolDefect("Claude text delta was malformed")
                    self._append_final_text(state, text)
                    yield AgentText(text)
                    return
                if delta.get("type") == "thinking_delta":
                    thinking = delta.get("thinking")
                    if not isinstance(thinking, str):
                        raise ProtocolDefect("Claude thinking delta was malformed")
                    # Thinking has no first-class kind; it travels as a bounded native
                    # frame, but the byte bound still applies before anything is retained.
                    self._check_event_text(thinking)
                    yield self._native("StreamEvent:thinking_delta", message)
                    return
            yield self._native("StreamEvent", message)
            return
        if isinstance(message, sdk.AssistantMessage):
            state.quota.observe_assistant_error(getattr(message, "error", None))
            usage = getattr(message, "usage", None)
            if isinstance(usage, dict):
                state.usage = self._token_usage(usage)
                yield AgentUsage(state.usage)
            partial = self._include_partial(state.request.native)
            content = getattr(message, "content", None)
            if not isinstance(content, (tuple, list)):
                raise ProtocolDefect("Claude assistant content was not an array")
            for block in content:
                if isinstance(block, sdk.ToolUseBlock):
                    arguments = getattr(block, "input", None)
                    if not isinstance(arguments, dict):
                        raise ProtocolDefect("Claude tool input was malformed")
                    tool_call_id = self._required_attr(block, "id")
                    name = self._required_attr(block, "name")
                    state.tool_names[tool_call_id] = name
                    yield AgentToolUse(
                        tool_call_id=tool_call_id,
                        name=name,
                        phase="started",
                        payload=freeze_json_object(arguments),
                    )
                elif isinstance(block, sdk.TextBlock) and not partial:
                    # With partial messages on (the default), the final blocks duplicate
                    # the deltas already streamed, so they are only decoded when off.
                    text = getattr(block, "text", None)
                    if not isinstance(text, str):
                        raise ProtocolDefect("Claude text block was malformed")
                    self._append_final_text(state, text)
                    yield AgentText(text)
                elif isinstance(block, sdk.ThinkingBlock) and not partial:
                    thinking = getattr(block, "thinking", None)
                    if not isinstance(thinking, str):
                        raise ProtocolDefect("Claude thinking block was malformed")
                    self._check_event_text(thinking)
                    yield self._native("AssistantMessage:ThinkingBlock", block)
                elif not isinstance(block, (sdk.TextBlock, sdk.ThinkingBlock)):
                    yield self._native(f"AssistantMessage:{type(block).__name__}", block)
            return
        if isinstance(message, sdk.UserMessage):
            content = getattr(message, "content", ())
            if not isinstance(content, (str, tuple, list)):
                raise ProtocolDefect("Claude user content was malformed")
            blocks = content if isinstance(content, (tuple, list)) else ()
            for block in blocks:
                if isinstance(block, sdk.ToolResultBlock):
                    tool_call_id = self._required_attr(block, "tool_use_id")
                    name = state.tool_names.get(tool_call_id)
                    if name is None:
                        raise ProtocolDefect("Claude tool result arrived before its tool use")
                    yield AgentToolUse(
                        tool_call_id=tool_call_id,
                        name=name,
                        phase="completed",
                        payload=freeze_json_value(getattr(block, "content", None)),
                        succeeded=getattr(block, "is_error", None) is not True,
                    )
                else:
                    yield self._native(f"UserMessage:{type(block).__name__}", block)
            return
        if isinstance(message, sdk.RateLimitEvent):
            info = getattr(message, "rate_limit_info", None)
            state.quota.observe_rate_limit_status(getattr(info, "status", None))
            yield self._native("RateLimitEvent", message)
            return
        if isinstance(message, sdk.ResultMessage):
            # The terminal is the owned summary; the frame itself still travels so the
            # backend-reported facts (cost, per-model usage, denials) are not lost.
            yield self._native("ResultMessage", message)
            yield self._terminal(state, message)
            return
        # claude-agent-sdk 0.2.130 closes `Message` over six classes and drops every
        # unrecognized wire type inside `parse_message`, so nothing reaches this arm today.
        # It is kept as the fail-open landing site for a future SDK that widens that union.
        yield self._native(type(message).__name__, message)

    def _verify_initialization(self, state: _ClaudeSessionState, message: object) -> None:
        """Check the backend's *effective* configuration against the one that was requested.

        `system/init` is the only frame that reports it. Claude Code 2.1.220 builds it in
        `tAr()` as `{cwd, session_id, tools, mcp_servers, model, permissionMode, ...}`. A
        widened directory, permission mode, or tool set is a security fact and defects; a
        substituted model is the backend's own prerogative and becomes a terminal
        diagnostic beside the init frame's native payload.
        """
        data = getattr(message, "data", None)
        if not isinstance(data, Mapping):
            raise ProtocolDefect("Claude system/init carried no object payload")
        if data.get("cwd") != state.request.cwd:
            raise ProtocolDefect(
                "Claude SDK reported a different effective working directory",
                code="effective_cwd_mismatch",
            )
        reported_mode = data.get("permissionMode")
        if reported_mode is not None and reported_mode != _SDK_PERMISSION_MODE:
            raise ProtocolDefect(
                "Claude SDK reported a different effective permission mode",
                code="effective_policy_mismatch",
            )
        self._require_effective_tools(state, data)
        reported_model = data.get("model")
        if not isinstance(reported_model, str) or not reported_model:
            raise ProtocolDefect("Claude SDK reported a malformed effective model")
        requested_model = state.request.model
        if requested_model is not None and reported_model != requested_model:
            self._record_diagnostic(state, "effective_model_changed")

    @staticmethod
    def _require_effective_tools(state: _ClaudeSessionState, data: Mapping[str, object]) -> None:
        reported = data.get("tools")
        if not isinstance(reported, list) or any(
            not isinstance(tool, str) or not tool for tool in reported
        ):
            raise ProtocolDefect("Claude SDK did not report an exact string tool list")
        if len(reported) != len(set(reported)):
            raise ProtocolDefect("Claude SDK reported a duplicate effective tool")
        requested = {
            _INIT_TOOL_ALIASES.get(name, name) for name in state.request.policy.allowed_tools
        }
        unexpected = {name for name in reported if name not in requested}
        if isinstance(state.request.output, JsonSchemaAgentOutput):
            unexpected.discard("StructuredOutput")
        if state.request.mcp_servers:
            unexpected = {name for name in unexpected if not name.startswith(_MCP_TOOL_PREFIX)}
        if unexpected:
            raise ProtocolDefect(
                "Claude SDK reported a wider effective tool set than the policy allows: "
                + ", ".join(sorted(unexpected)),
                code="effective_policy_mismatch",
            )

    def _terminal(self, state: _ClaudeSessionState, message: object) -> AgentTerminal:
        # The native turn is over the moment its result is consumed, so nothing is left on
        # the shared stream for a later turn to inherit.
        state.turn_in_flight = False
        ref = state.ref
        if ref is None:
            raise ProtocolDefect("Claude terminal arrived before stream identity")
        result_usage = getattr(message, "usage", None)
        if isinstance(result_usage, dict):
            state.usage = self._token_usage(result_usage)
        usage = self._usage_presence(state)
        result = getattr(message, "result", None)
        final_text = result if isinstance(result, str) else state.final_text
        self._check_final_text(final_text)
        diagnostics = tuple(dict.fromkeys(state.diagnostics))
        terminal_reason = getattr(message, "terminal_reason", None)
        if state.approval_failure:
            return AgentTerminal(
                status="failed",
                failure=AgentFailure("approval_unanswered"),
                final_text=final_text,
                session_ref=ref,
                usage=usage,
                diagnostics=("approval handler failed",),
            )
        if state.approval_abort or terminal_reason in ("aborted_streaming", "aborted_tools"):
            return AgentTerminal(
                status="cancelled",
                failure=None,
                final_text=final_text,
                session_ref=ref,
                usage=usage,
                diagnostics=diagnostics,
            )
        if getattr(message, "is_error", False):
            failure: AgentTerminalFailure
            if state.quota.is_exhausted(getattr(message, "api_error_status", None)):
                failure = AgentQuotaExhausted()
            elif getattr(message, "subtype", None) == "error_max_structured_output_retries":
                failure = AgentFailure("output_schema_violation")
            else:
                failure = AgentFailure("backend_failed")
            errors = getattr(message, "errors", None)
            if isinstance(errors, list):
                if len(errors) > _MAX_DIAGNOSTICS:
                    raise OutputLimitExceeded(_MAX_DIAGNOSTICS)
                sanitized = (
                    sanitize_provider_text(item) for item in errors if isinstance(item, str)
                )
                diagnostics = tuple(
                    dict.fromkeys([*diagnostics, *(text for text in sanitized if text)])
                )
                if len(diagnostics) > _MAX_DIAGNOSTICS:
                    raise OutputLimitExceeded(_MAX_DIAGNOSTICS)
            return AgentTerminal(
                status="failed",
                failure=failure,
                final_text=final_text,
                session_ref=ref,
                usage=usage,
                diagnostics=diagnostics,
            )
        structured: FrozenJsonDict | None = None
        if isinstance(state.request.output, JsonSchemaAgentOutput):
            # The backend enforces the schema natively; this boundary only strict-parses
            # and freezes the answer, and a miss is an expected model-output failure.
            raw = getattr(message, "structured_output", None)
            try:
                structured = (
                    freeze_structured_output(raw)
                    if raw is not None
                    else parse_structured_output(final_text)
                )
            except OutputSchemaMismatch:
                return AgentTerminal(
                    status="failed",
                    failure=AgentFailure("output_schema_violation"),
                    final_text=final_text,
                    session_ref=ref,
                    usage=usage,
                    diagnostics=diagnostics,
                )
        return AgentTerminal(
            status="succeeded",
            failure=None,
            final_text=final_text,
            session_ref=ref,
            usage=usage,
            structured_output=structured,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _usage_presence(state: _ClaudeSessionState) -> Presence[TokenUsage]:
        return Absent() if state.usage is None else Present(state.usage)

    @classmethod
    def _token_usage(cls, usage: Mapping[str, object]) -> TokenUsage:
        """Normalize one native usage dict into the provider lane's noun.

        Claude's wire `input_tokens` EXCLUDES the cache components it reports beside it;
        the lane invariant is the cache-INCLUSIVE prompt total (see
        `TokenUsage.from_components`), so both components are folded in at ingress. The
        total is derived, and Claude reports no reasoning count.
        """
        input_tokens = cls._usage_count(usage, "input_tokens") or 0
        output_tokens = cls._usage_count(usage, "output_tokens") or 0
        cache_read = cls._usage_count(usage, "cache_read_input_tokens")
        cache_write = cls._usage_count(usage, "cache_creation_input_tokens")
        return TokenUsage.from_components(
            input_tokens=input_tokens + (cache_read or 0) + (cache_write or 0),
            output_tokens=output_tokens,
            total_tokens=Absent(),
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Absent() if cache_read is None else Present(cache_read),
            cache_write_input_tokens=Absent() if cache_write is None else Present(cache_write),
        )

    @staticmethod
    def _usage_count(usage: Mapping[str, object], key: str) -> int | None:
        value = usage.get(key)
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ProtocolDefect(f"Claude usage {key} was not a non-negative integer count")
        return value

    @staticmethod
    def _record_diagnostic(state: _ClaudeSessionState, message: str) -> None:
        """Append one deduplicated turn diagnostic within the stream's diagnostic bound."""
        if message in state.diagnostics or len(state.diagnostics) >= _MAX_DIAGNOSTICS:
            return
        state.diagnostics.append(message)

    async def _handle_permission(
        self, state: _ClaudeSessionState, name: str, tool_input: dict[str, Any], context: Any
    ) -> Any:
        """Answer one native permission request from the task the SDK dispatches it on.

        `Query._spawn_control_request_handler` runs this concurrently with the turn
        generator, so it only queues its event and lets `_message_events` emit it. A defect
        detected here would otherwise be swallowed by the SDK's control-request error
        response, so it is handed to the generator too.
        """
        try:
            return await self._decide_permission(state, name, tool_input, context)
        except AgentRuntimeDefect as defect:
            if state.approval_defect is None:
                state.approval_defect = defect
            return state.sdk.PermissionResultDeny(
                message="permission denied by caller policy",
                interrupt=True,
            )

    async def _decide_permission(
        self, state: _ClaudeSessionState, name: str, tool_input: dict[str, Any], context: Any
    ) -> Any:
        if state.drain_pending:
            # An interrupted turn's tail reaches the control channel as well as the message
            # stream: Claude Code writes a permission request before the interrupt reaches it,
            # and `Query._spawn_control_request_handler` delivers it whether or not anyone is
            # reading. It belongs to a turn the caller already abandoned, so putting it to the
            # caller would ask for consent out of context and an "allow" would run the tool on
            # a retired turn. Refusing here is also what keeps the out-of-turn defect below
            # from being recorded against the *next* turn, which is the only turn left to
            # raise it.
            return state.sdk.PermissionResultDeny(
                message="permission denied because the turn was interrupted",
                interrupt=True,
            )
        policy = state.current_policy
        if policy is None or state.turn_id is None:
            raise ProtocolDefect("Claude permission callback ran outside an active turn")
        operation = self._operation(name)
        blocked_path = getattr(context, "blocked_path", None)
        native = redact_native_payload(
            {
                "tool_name": name,
                "input": tool_input,
                "tool_use_id": getattr(context, "tool_use_id", None),
                "title": getattr(context, "title", None),
                "display_name": getattr(context, "display_name", None),
                "description": getattr(context, "description", None),
                "blocked_path": blocked_path,
            }
        )
        summary = next(
            value
            for value in (
                getattr(context, "title", None),
                getattr(context, "description", None),
                f"Claude requested permission to use {name}",
            )
            if isinstance(value, str) and value
        )
        request = ApprovalRequest(
            operation=operation,
            summary=sanitize_provider_text(summary, limit=2_000),
            tool_name=name if operation == "tool_use" else None,
            native_payload=native,
        )
        handler_failed = False
        if policy.filesystem == "read_only" and operation in ("command", "file_change"):
            decision: ApprovalDecision = "deny"
        elif not tool_is_allowed(policy, name):
            decision = "deny"
        elif isinstance(blocked_path, str) and blocked_path:
            # Claude Code only sets `blocked_path` when the request already left the allowed
            # working directories (2.1.220 path validation: "Claude Code may only ... the
            # allowed working directories for this session"). `filesystem: full_access` is
            # refused outright for this lane, so no acknowledgement covers that reach and the
            # escape is denied whatever `approval` says.
            decision = "deny"
        elif policy.approval == "deny":
            decision = "deny"
        elif policy.approval == "allow":
            decision = "allow"
        else:
            approvals = state.current_approvals
            if approvals is None:
                raise ProtocolDefect(
                    "Claude approval callback ran without the handler stream_turn requires"
                )
            try:
                decision = await approvals(request)
            except Exception:
                decision = "deny"
                handler_failed = True
            if decision not in ("allow", "deny", "abort"):
                decision = "deny"
                handler_failed = True
        # One auditable event per native request, carrying the decision that was made.
        self._queue_pending(state, AgentPermissionRequest(request=request, decision=decision))
        if handler_failed:
            state.approval_failure = True
        elif decision == "abort":
            state.approval_abort = True
        if decision == "allow":
            return state.sdk.PermissionResultAllow(updated_input=tool_input)
        return state.sdk.PermissionResultDeny(
            message="permission denied by caller policy",
            interrupt=handler_failed or decision == "abort",
        )

    @staticmethod
    def _queue_pending(state: _ClaudeSessionState, event: AgentEvent) -> None:
        if len(state.pending_events) >= _MAX_EVENT_COUNT:
            raise ProtocolDefect(
                "Claude queued more approval events than one turn may emit",
                code="output_limit_before_delivery",
            )
        state.pending_events.append(event)

    def _learn_identity(self, state: _ClaudeSessionState, message: object) -> None:
        session_id: object | None = getattr(message, "session_id", None)
        if isinstance(message, state.sdk.SystemMessage):
            data = getattr(message, "data", None)
            if isinstance(data, Mapping):
                session_id = data.get("session_id")
        if state.ref is None and isinstance(session_id, str) and session_id:
            state.ref = self._make_ref(
                session_id,
                state.request.auth.profile_key,
                state.state_root,
                state.request.cwd,
            )
        elif (
            state.ref is not None
            and isinstance(session_id, str)
            and session_id
            and session_id != state.ref.native_session_id
        ):
            raise ProtocolDefect("Claude SDK stream changed session identity")
        # The native message uuid names the turn internally only: it gates the permission
        # callback's active-turn check and interrupt's soft/hard choice. No public turn id
        # exists.
        uuid = getattr(message, "uuid", None)
        if state.turn_id is None and isinstance(uuid, str) and uuid:
            state.turn_id = uuid

    @staticmethod
    def _account_message(state: _ClaudeSessionState, message: object) -> None:
        size = bounded_payload_size(
            message,
            _MAX_MESSAGE_BYTES,
            max_items=_MAX_MESSAGE_ITEMS,
        )
        if state.output_bytes + size > _MAX_TURN_OUTPUT_BYTES:
            raise OutputLimitExceeded(_MAX_TURN_OUTPUT_BYTES)
        state.message_count += 1
        if state.message_count > _MAX_EVENT_COUNT:
            raise OutputLimitExceeded(_MAX_EVENT_COUNT)
        state.output_bytes += size

    @staticmethod
    def _check_event_text(text: str) -> int:
        if len(text) > _MAX_EVENT_TEXT_BYTES:
            raise OutputLimitExceeded(_MAX_EVENT_TEXT_BYTES)
        size = len(text.encode("utf-8"))
        if size > _MAX_EVENT_TEXT_BYTES:
            raise OutputLimitExceeded(_MAX_EVENT_TEXT_BYTES)
        return size

    @classmethod
    def _append_final_text(cls, state: _ClaudeSessionState, text: str) -> None:
        size = cls._check_event_text(text)
        if state.final_text_bytes + size > _MAX_FINAL_TEXT_BYTES:
            raise OutputLimitExceeded(_MAX_FINAL_TEXT_BYTES)
        state.final_text += text
        state.final_text_bytes += size

    @staticmethod
    def _check_final_text(text: str) -> None:
        if len(text) > _MAX_FINAL_TEXT_BYTES:
            raise OutputLimitExceeded(_MAX_FINAL_TEXT_BYTES)
        if len(text.encode("utf-8")) > _MAX_FINAL_TEXT_BYTES:
            raise OutputLimitExceeded(_MAX_FINAL_TEXT_BYTES)

    async def _stop_client(
        self, state: _ClaudeSessionState, *, force_disconnect: bool = False
    ) -> None:
        interrupted = False
        try:
            async with asyncio.timeout(_INTERRUPT_TIMEOUT_SECONDS):
                await state.client.interrupt()
            interrupted = True
        except (state.sdk.ClaudeSDKError, TimeoutError) as error:
            # justify-ignore-error: an interrupt that the CLI refused, never answered, or could
            # not receive tells us nothing beyond "the soft stop did not land", and the only
            # correct response is the hard disconnect below, which runs unconditionally once
            # `interrupted` stays False. The cause is kept as a turn diagnostic so a refused
            # interrupt is never indistinguishable from a clean one. An `AttributeError` or
            # `TypeError` from an SDK signature change is deliberately *not* absorbed here.
            self._record_diagnostic(state, f"Claude SDK interrupt failed: {type(error).__name__}")
        if interrupted and not force_disconnect:
            # The session survives a soft interrupt, so the interrupted turn's remaining
            # messages stay queued on the client's one shared stream. They are discarded at
            # the head of the next turn (`_drain_retired_turn`), which is the only point at
            # which nothing else is reading that stream.
            state.drain_pending = state.turn_in_flight
            return
        try:
            await self._disconnect_client(state.sdk, state.client, state.process_group)
        finally:
            for session, candidate in tuple(self._sessions.items()):
                if candidate is state:
                    self._sessions.pop(session, None)
                    self._dead_sessions.add(session)
                    break

    @staticmethod
    def _owned_process(client: Any) -> Any:
        """The anyio process the SDK spawned for one client.

        The SDK exposes no accessor for it, and both process-group ownership and the kill
        escalation need the pid, so the reach into `client._transport._process` happens here
        once and is checked once. A shape change in the pinned SDK is a defect, not a
        silent loss of the only handle this runtime has on the child.
        """
        transport = getattr(client, "_transport", None)
        process = getattr(transport, "_process", None)
        if process is None or not isinstance(getattr(process, "pid", None), int):
            raise ProtocolDefect(
                "claude-agent-sdk did not expose the process it spawned",
                code="sdk_shape_mismatch",
            )
        return process

    @classmethod
    async def _disconnect_client(
        cls, sdk: Any, client: Any, group: OwnedProcessGroup | None
    ) -> None:
        """Release one client, then make sure nothing it started is still running.

        The SDK's own `close()` escalates terminate/kill on the single CLI pid, which leaves
        whatever Claude Code spawned — a Bash-tool command, a stdio MCP server — running
        against the workspace. `group` is the session the launcher gave that child, so
        signalling it reaches those descendants and, because the launcher made the child a
        session leader, can reach nothing else.
        """
        try:
            try:
                async with asyncio.timeout(_DISCONNECT_TIMEOUT_SECONDS):
                    await client.disconnect()
            except (sdk.ClaudeSDKError, TimeoutError, OSError):
                # justify-ignore-error: these are the modelled ways the SDK's own teardown
                # can fail (refused control write, our disconnect bound, a broken pipe).
                # None of them establishes the invariant that matters here, and the kill
                # escalation below both establishes it and reports its own outcome, so
                # discarding this cause loses nothing. Anything outside this set is SDK
                # drift and keeps propagating.
                await cls._kill_owned_process(client)
        finally:
            if group is not None:
                await group.terminate(grace_seconds=_GROUP_TERMINATION_GRACE_SECONDS)

    @classmethod
    async def _kill_owned_process(cls, client: Any) -> None:
        process = cls._owned_process(client)
        try:
            process.kill()
            async with asyncio.timeout(_INTERRUPT_TIMEOUT_SECONDS):
                await process.wait()
        except ProcessLookupError:
            # justify-ignore-error: an already-reaped child is the desired final state.
            return
        except (OSError, TimeoutError):
            raise ProtocolDefect(
                "Claude SDK owned process did not reap within the cleanup bound",
                code="process_cleanup_failed",
            ) from None

    def _state(self, session: AgentSession) -> _ClaudeSessionState:
        if session in self._dead_sessions:
            raise SessionUnavailable("Claude SDK session is no longer live")
        try:
            return self._sessions[session]
        except KeyError as error:
            raise InvalidAgentRequest("session is not owned by this Claude adapter") from error

    @classmethod
    def _load_sdk(cls) -> ModuleType:
        if os.environ.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
            raise UnsupportedCapability("ambient Claude SDK version-check bypass is forbidden")
        try:
            sdk = importlib.import_module("claude_agent_sdk")
        except ModuleNotFoundError as error:
            raise SdkUnavailable(
                "claude-agent-sdk is unavailable; install the 'claude-sdk' extra"
            ) from error
        version = cls._sdk_version(sdk)
        if version != _SUPPORTED_SDK_VERSION:
            warnings.warn(
                f"claude-agent-sdk {version} is not the vetted {_SUPPORTED_SDK_VERSION} — "
                "continuing behind the behavioral system/init verification",
                RuntimeWarning,
                stacklevel=2,
            )
        return sdk

    def _require_executable(self) -> str:
        if self._executable is None:
            raise ExecutableUnavailable(
                f"Claude SDK executable {self._requested_executable!r} is unavailable"
            )
        return self._executable

    @staticmethod
    def _sdk_version(sdk: ModuleType) -> str:
        version = getattr(sdk, "__version__", None)
        if isinstance(version, str) and version:
            return version
        try:
            from importlib.metadata import version as distribution_version

            return distribution_version("claude-agent-sdk")
        except Exception:
            return "unknown"

    @staticmethod
    def _require_local_auth(kind: str) -> None:
        # Retained security kernel: this lane forwards no API-key credential on any path.
        if kind != "local_account":
            raise UnsupportedCapability(
                "Claude Agent SDK cannot report its effective authentication identity"
            )

    @staticmethod
    def _sdk_environment(environment: Mapping[str, str]) -> dict[str, str]:
        """Build the child environment the SDK will overlay onto `os.environ`.

        Because the SDK merges rather than replaces, an ambient name is only kept out of the
        child by being present-and-empty, which is why every unauthorized name is blanked.
        `environment` is complete on its own — `auth.build_child_environment` owns PATH, HOME,
        the locale, and TMPDIR outright — so this method only has to suppress what the merge
        would otherwise let through.
        """
        result = {name: "" for name in os.environ if name not in environment}
        result.update(environment)
        for name in credential_environment_names("claude"):
            if name not in environment:
                result[name] = ""
        return result

    @staticmethod
    def _sandbox(policy: PermissionPolicy) -> dict[str, object]:
        network: dict[str, object] = {
            "allowedDomains": list(policy.network_allowlist),
            "deniedDomains": [],
            "allowManagedDomainsOnly": policy.network == "disabled",
            "allowUnixSockets": [],
            "allowAllUnixSockets": False,
            "allowLocalBinding": False,
            "allowMachLookup": [],
        }
        return {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": False,
            "allowUnsandboxedCommands": False,
            "network": network,
        }

    @staticmethod
    def _validate_policy_mapping(policy: PermissionPolicy) -> None:
        """Enforce the backend facts this transport owns, failing closed before any work."""
        if policy.filesystem == "full_access":
            raise UnsupportedCapability("Claude SDK full filesystem access is not fail-closed")
        if policy.network == "unrestricted":
            raise UnsupportedCapability("Claude SDK unrestricted network is not fail-closed")
        if any(any(marker in name for marker in "*?[]") for name in policy.allowed_tools):
            raise UnsupportedCapability("Claude SDK tool availability requires exact tool names")
        if any(any(marker in name for marker in "*?[]") for name in policy.denied_tools):
            raise UnsupportedCapability("Claude SDK tool denials require exact tool names")
        if any(name in _REFUSED_TOOL_NAMES for name in policy.allowed_tools):
            raise UnsupportedCapability(
                "Claude WebFetch/WebSearch bypass sandbox network restrictions"
            )

    @staticmethod
    def _include_partial(native: object) -> bool:
        if isinstance(native, ClaudeNativeOptions) and native.include_partial_messages is not None:
            return native.include_partial_messages
        return True

    @staticmethod
    def _optional_instruction(parts: tuple[object, ...]) -> str | None:
        if not parts:
            return None
        if any(not isinstance(part, TextContent) for part in parts):
            raise UnsupportedCapability("Claude system instructions support text only")
        return "\n\n".join(part.text for part in parts if isinstance(part, TextContent))

    @staticmethod
    def _turn_text(request: TurnRequest) -> str:
        if any(not isinstance(part, TextContent) for part in request.input):
            raise UnsupportedCapability("Claude SDK turn input supports text only")
        return "\n\n".join(part.text for part in request.input if isinstance(part, TextContent))

    @staticmethod
    def _mcp_servers(request: AgentSessionRequest) -> dict[str, object]:
        result: dict[str, object] = {}
        for server in request.mcp_servers:
            if server.transport != "streamable_http":
                # `validate_mcp_network_policy` only admits a stdio server under full
                # filesystem and unrestricted network policy, both of which
                # `_validate_policy_mapping` refuses for this lane, so this is unreachable
                # through the runtime. It fails loudly rather than silently emitting a
                # config shape nobody checked, should either gate ever move.
                raise UnsupportedCapability(
                    "Claude SDK MCP configuration supports streamable HTTP servers only"
                )
            if server.allowed_tools or server.denied_tools:
                raise UnsupportedCapability(
                    "Claude SDK cannot preserve per-server MCP tool filters"
                )
            if server.environment_refs or server.header_refs:
                raise UnsupportedCapability(
                    "Claude SDK MCP configuration cannot preserve credential references"
                )
            result[server.name] = {"type": "http", "url": server.url, "headers": {}}
        return result

    @staticmethod
    def _mcp_startup_diagnostics(request: AgentSessionRequest, status: object) -> list[str]:
        """Classify the CLI's `mcp_status` answer before the session becomes usable.

        Claude Code 2.1.220 answers the `mcp_status` control request with
        `{mcpServers: [{name, status, serverInfo?, error?, config?, scope?, tools?}]}` where
        `status` is one of `connected|failed|needs-auth|pending|disabled`; only `connected`
        means the server is usable, so `pending` is treated as not-yet-available like the rest.
        """
        raw_statuses = status.get("mcpServers") if isinstance(status, Mapping) else None
        if not isinstance(raw_statuses, list):
            raise ProtocolDefect(
                "Claude SDK MCP status response had no server list",
                code="sdk_shape_mismatch",
            )
        statuses: dict[str, str] = {}
        errors: dict[str, str] = {}
        for raw in raw_statuses:
            if not isinstance(raw, Mapping):
                raise ProtocolDefect("Claude SDK MCP status entry was malformed")
            name = raw.get("name")
            state = raw.get("status")
            if not isinstance(name, str) or not name or not isinstance(state, str) or not state:
                raise ProtocolDefect("Claude SDK MCP status identity was malformed")
            if name in statuses:
                raise ProtocolDefect("Claude SDK reported a duplicate MCP server identity")
            statuses[name] = state
            detail = raw.get("error")
            if isinstance(detail, str) and detail:
                errors[name] = sanitize_provider_text(detail, limit=2_000)
        requested_names = {server.name for server in request.mcp_servers}
        if not set(statuses).issubset(requested_names):
            raise ProtocolDefect("Claude SDK reported an unrequested MCP server")
        diagnostics: list[str] = []
        for server in request.mcp_servers:
            reported = statuses.get(server.name, "unreported")
            if reported == _MCP_CONNECTED_STATUS:
                continue
            detail = errors.get(server.name)
            suffix = f": {detail}" if detail is not None else ""
            if server.required:
                raise McpUnavailable(
                    f"required Claude MCP server {server.name!r} is {reported}{suffix}"
                )
            diagnostics.append(f"optional Claude MCP server {server.name!r} is {reported}{suffix}")
        return diagnostics

    @staticmethod
    def _output_format(output: object) -> dict[str, object] | None:
        if not isinstance(output, JsonSchemaAgentOutput):
            return None
        # The caller's plain JSON Schema passes through to the SDK's native option; the
        # backend enforces it, this lane never interprets it.
        return {"type": "json_schema", "schema": thaw_json_value(output.schema)}

    @staticmethod
    def _thinking(request: AgentSessionRequest) -> dict[str, object] | None:
        reasoning = request.reasoning
        if reasoning is None:
            return None
        if reasoning.thinking_budget is not None:
            return {"type": "enabled", "budget_tokens": reasoning.thinking_budget}
        return {"type": "adaptive"}

    @staticmethod
    def _operation(name: str) -> Literal["command", "file_change", "tool_use"]:
        """Classify one tool for the approval summary and the read-only refusal.

        Read from `_BUILTIN_TOOL_NAMES`, because the accepted-name set and this
        classification are the same fact: `_decide_permission` denies
        `command`/`file_change` outright under `filesystem: read_only`, so a file-write
        name this method called an ordinary tool use would be a write allowed under a
        read-only policy.
        """
        if name in _COMMAND_TOOL_NAMES:
            return "command"
        if name in _FILE_CHANGE_TOOL_NAMES:
            return "file_change"
        return "tool_use"

    @classmethod
    def _native(cls, native_type: str, value: object) -> AgentNative:
        """One native frame with no first-class kind, recursively redacted and bounded."""
        return AgentNative(
            native_type=native_type,
            payload=redact_native_payload(cls._object_payload(value)),
        )

    @staticmethod
    def _object_payload(value: object) -> dict[str, object]:
        if is_dataclass(value) and not isinstance(value, type):
            payload = cast(dict[str, object], asdict(value))
        elif hasattr(value, "__dict__"):
            payload = cast(dict[str, object], dict(vars(value)))
        else:
            payload = cast(dict[str, object], {"type": type(value).__name__})
        if not isinstance(payload, dict):
            raise ProtocolDefect("Claude SDK message could not be represented as an object")
        return payload

    @staticmethod
    def _required_attr(value: object, name: str) -> str:
        result = getattr(value, name, None)
        if not isinstance(result, str) or not result:
            raise ProtocolDefect(f"Claude SDK value had no {name}")
        return result

    @staticmethod
    def _make_ref(
        native_session_id: str, profile_key: str, state_root: Path, cwd: str
    ) -> AgentSessionRef:
        return AgentSessionRef(
            schema_version="agent-session-ref.v1",
            backend="claude",
            transport="sdk",
            native_session_id=native_session_id,
            profile_key=profile_key,
            state_root_fingerprint=fingerprint_path(state_root),
            cwd_fingerprint=fingerprint_path(cwd),
        )


__all__ = ["ClaudeSdkAdapter"]
