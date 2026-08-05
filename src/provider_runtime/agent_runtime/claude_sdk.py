"""Native Claude Agent SDK adapter with a lazy optional dependency boundary."""

from __future__ import annotations

import asyncio
import importlib
import os
import re
import shutil
import sys
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

from provider_runtime.errors import sanitize_provider_text
from provider_runtime.schema import to_json_schema

from ._claude_launcher import OwnedProcessGroup, ensure_claude_launcher
from ._limits import OutputLimitExceeded, bounded_payload_size
from ._process import ProcessLimits, capture_process_output
from ._sandbox import bubblewrap_network_namespace_available, environment_executable
from ._structured_output import (
    OutputSchemaMismatch,
    parse_structured_output,
    validate_structured_output,
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
from .capabilities import (
    AgentCapabilities,
    AgentCapabilityScope,
    BuiltinToolFamily,
    validate_mcp_network_policy,
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
    AgentEventData,
    AgentEventKind,
    ApprovalAnsweredData,
    ApprovalRequestedData,
    DiagnosticData,
    FileChangeData,
    ReasoningData,
    SessionStartedData,
    TextDeltaData,
    ToolCompletedData,
    ToolStartedData,
    TurnCancelledData,
    TurnCompletedData,
    TurnFailedData,
    TurnStartedData,
    UnknownData,
    UsageData,
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
    AgentOutputSpec,
    AgentSessionRef,
    AgentSessionRequest,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    ClaudeNativeOptions,
    CredentialRef,
    ForkSession,
    FrozenJsonDict,
    JsonObject,
    JsonSchemaAgentOutput,
    NewSession,
    ResumeSession,
    TextContent,
    TurnRequest,
)

_KNOWN_FIELDS: dict[str, tuple[str, ...]] = {
    "SystemMessage": ("subtype", "data"),
    "StreamEvent": ("uuid", "session_id", "parent_tool_use_id", "event"),
    "AssistantMessage": (
        "model",
        "session_id",
        "message_id",
        "uuid",
        "stop_reason",
        "error",
        "usage",
        "content",
    ),
    "UserMessage": ("uuid", "parent_tool_use_id", "tool_use_result", "content"),
    "RateLimitEvent": ("uuid", "session_id", "rate_limit_info"),
    # Every field `parse_message` populates on a 0.2.130 `ResultMessage`. The allowlist is
    # exhaustive on purpose: it is an allowlist, so a field left out here is a field the
    # terminal event silently drops, and the spec's "retaining native event data" is exactly
    # what a caller needs the terminal payload for. `permission_denials` and `model_usage`
    # carry no auth material — the sensitive-key scrub in `redact_native_payload` still runs
    # over both — and `total_cost_usd` is the backend's own number, never a runtime-derived
    # one, so reporting it does not put this lane into costing.
    "ResultMessage": (
        "subtype",
        "duration_ms",
        "duration_api_ms",
        "is_error",
        "num_turns",
        "session_id",
        "stop_reason",
        "total_cost_usd",
        "usage",
        "result",
        "structured_output",
        "model_usage",
        "permission_denials",
        "deferred_tool_use",
        "terminal_reason",
        "errors",
        "api_error_status",
        "uuid",
    ),
    "permission/request": (
        "tool_name",
        "input",
        "tool_use_id",
        "title",
        "display_name",
        "description",
        "blocked_path",
    ),
}
# Claude Code 2.1.220 builds `system/init` in `tAr()` as `tools: e.tools.map(o => q_n(o.name))`
# with `q_n(e) { return e === "Agent" ? "Task" : e }`, so the one internal tool named `Agent`
# is reported under its user-facing name. Every other tool is reported verbatim.
_INIT_TOOL_ALIASES: dict[str, str] = {"Agent": "Task"}
# The exact native tool names this lane accepts in `PermissionPolicy.allowed_tools`, per
# built-in family, pinned to `_SUPPORTED_CLAUDE_VERSION` for the same reason
# `_INIT_TOOL_ALIASES` is: the Agent SDK ships no tool-name constant and `system/init` only
# echoes back the tools the session already asked for, so nothing discovers these at runtime.
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
# This table is the single source of both the advertised families and the approval-operation
# classification below, so a name can never be advertised as a file write while the approval
# path treats it as an ordinary tool use.
_BUILTIN_TOOL_NAMES: tuple[tuple[BuiltinToolFamily, tuple[str, ...]], ...] = (
    ("file_read", ("Read", "Glob", "Grep")),
    ("file_write", ("Write", "Edit", "NotebookEdit")),
    ("command", ("Bash",)),
)
# Built-in tools 2.1.220 ships that this lane refuses outright, so no capability of it may
# advertise them. Both reach the network through Claude Code's own client rather than through
# the sandbox this adapter always engages, so a policy naming either would get network access
# that `PermissionPolicy.network` never granted.
_REFUSED_TOOL_NAMES: tuple[str, ...] = ("WebFetch", "WebSearch")
_FILE_CHANGE_TOOL_NAMES = frozenset(dict(_BUILTIN_TOOL_NAMES)["file_write"])
_COMMAND_TOOL_NAMES = frozenset(dict(_BUILTIN_TOOL_NAMES)["command"])
# Claude Code names MCP tools `mcp__<server>__<tool>` (2.1.220 `kte()`), so an MCP-provided tool
# is recognizable by prefix; the server set behind it is verified against `mcp_status` before the
# session is published.
_MCP_TOOL_PREFIX = "mcp__"
_MCP_CONNECTED_STATUS = "connected"
_SDK_PERMISSION_MODE = "default"
_DIAGNOSTIC_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
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
# Pinned against the installed Claude Code build; discovery never assumes a version.
_SUPPORTED_CLAUDE_VERSION = "2.1.220"
_CLAUDE_VERSION = re.compile(r"([0-9]+(?:\.[0-9]+){2}(?:[-+][0-9A-Za-z.-]+)?) \(Claude Code\)\Z")
_VERSION_STDOUT_LIMIT = 16 * 1024
_CLAUDE_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
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
    """Return the installed Claude Code version, refusing anything but the vetted build."""
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
        raise ExecutableUnavailable("Claude CLI returned an unrecognized version string")
    version = match.group(1)
    if version != _SUPPORTED_CLAUDE_VERSION:
        raise ExecutableUnavailable("Claude CLI version is not vetted")
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


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    """One described-but-unnumbered event.

    Sequence numbers belong to the turn generator alone (see `ClaudeSdkAdapter._event`), so any
    code that runs off that task — notably the permission callback, which the SDK dispatches from
    a task it spawns itself — describes its events with this type and lets the generator number
    them in stream order.
    """

    kind: AgentEventKind
    data: AgentEventData
    native_type: str | None = None
    native_payload: JsonObject | None = None


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
    session_started_emitted: bool = False
    seq: int = 0
    message_count: int = 0
    output_bytes: int = 0
    turn_number: int = 0
    turn_id: str | None = None
    final_text: str = ""
    final_text_bytes: int = 0
    usage: FrozenJsonDict | None = None
    current_output: AgentOutputSpec | None = None
    diagnostics: list[str] = field(default_factory=list)
    # What `open_session` observed about this session before any turn existed. `open_session`
    # returns a session rather than a stream, so there is no earlier event this could be:
    # `stream_turn` emits it in the session-scope window right after `session_started`, which
    # is where the SDK reported it. It is emptied there, so only the first stream carries it.
    pending_startup_diagnostics: list[str] = field(default_factory=list)
    quota: _ClaudeQuotaSignal = field(default_factory=_ClaudeQuotaSignal)
    approval_failure: bool = False
    approval_abort: bool = False
    current_policy: PermissionPolicy | None = None
    current_approvals: ApprovalHandler | None = None
    pending_events: list[_PendingEvent] = field(default_factory=list)
    approval_defect: AgentRuntimeDefect | None = None


class ClaudeSdkAdapter:
    """Own Claude SDK clients while keeping the SDK import fully optional."""

    backend: Literal["claude"] = "claude"
    transport: Literal["sdk"] = "sdk"

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

    async def capabilities(
        self,
        scope: AgentCapabilityScope,
        *,
        environment: Mapping[str, str],
    ) -> AgentCapabilities:
        self._require_scope(scope)
        self._require_local_auth(scope.auth.kind)
        sdk = self._load_sdk()
        sdk_version = self._sdk_version(sdk)
        if sdk_version != _SUPPORTED_SDK_VERSION:
            raise SdkUnavailable("claude-agent-sdk version is not vetted")
        state_root = state_root_from_environment("claude", environment)
        executable_version = await _read_claude_code_version(
            self._require_executable(),
            cwd=state_root,
            environment=environment,
        )
        network_sandbox = await _network_sandbox_available(cwd=state_root, environment=environment)
        return AgentCapabilities(
            scope=scope,
            executable_version=executable_version,
            sdk_version=sdk_version,
            session_operations=("new", "resume", "fork"),
            discovery_operations=(),
            models=None,
            reasoning_efforts=_CLAUDE_REASONING_EFFORTS,
            reasoning_summaries=(),
            thinking_budget=True,
            content_kinds=("text",),
            attachment_kinds=(),
            session_instruction_roles=("system",),
            turn_instruction_roles=(),
            streaming=True,
            cancellation=True,
            structured_output=True,
            native_output_schema=True,
            # Reported from the one accepted-name table, so the families a caller is told
            # about are exactly the families it can name a tool from. `web_fetch`/`web_search`
            # are absent even though 2.1.220 ships both: `_validate_policy_mapping` refuses
            # their tools, and the session's tool set *is* `policy.allowed_tools`, so no
            # request this adapter accepts can ever reach them.
            builtin_tool_families=tuple(family for family, _names in _BUILTIN_TOOL_NAMES),
            builtin_tool_names=_BUILTIN_TOOL_NAMES,
            tool_controls=True,
            mcp_transports=("streamable_http",),
            mcp_auth_forms=(),
            filesystem_modes=("read_only", "workspace_write"),
            network_modes=("disabled", "allowlist") if network_sandbox else ("disabled",),
            network_allowlist=network_sandbox,
            approval_modes=("deny", "ask", "allow"),
            additional_dirs=True,
            max_turns=False,
            timeouts=True,
            turn_overrides=(),
            persistent_turn_overrides=(),
            native_extension_version="claude-agent-sdk.python",
            native_option_names=("include_partial_messages",),
            cwd_scopes_sessions=True,
            reports_auth_identity=False,
            # Claude Code's `system/init` carries no effort member, so the clamp the
            # adapter applies stays unobservable to the caller.
            reports_effective_effort=False,
        )

    async def list_sessions(
        self,
        query: SessionQuery,
        *,
        environment: Mapping[str, str],
    ) -> SessionPage:
        self._require_scope(query.scope)
        self._require_local_auth(query.scope.auth.kind)
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
        capabilities: AgentCapabilities,
        environment: Mapping[str, str],
    ) -> AgentSession:
        if request.backend != self.backend or request.transport != self.transport:
            raise InvalidAgentRequest("ClaudeSdkAdapter received a different route")
        self._require_scope(capabilities.scope)
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
                AgentCapabilityScope(backend="claude", transport="sdk", auth=request.auth),
                state_root_fingerprint=fingerprint_path(state_root),
                cwd=request.cwd,
                cwd_scopes_sessions=True,
            )
            resume = request.open.ref.native_session_id
            fork = isinstance(request.open, ForkSession)
            ref = None if fork else request.open.ref
        elif not isinstance(request.open, NewSession):
            raise InvalidAgentRequest("unknown Claude SDK session operation")

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
        executable = self._require_executable()
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

        state.turn_number += 1
        state.seq = 0
        state.message_count = 0
        state.output_bytes = 0
        state.turn_id = None
        state.final_text = ""
        state.final_text_bytes = 0
        state.usage = None
        state.current_output = state.request.output
        state.diagnostics[:] = state.pending_startup_diagnostics
        state.quota = _ClaudeQuotaSignal()
        state.approval_failure = False
        state.approval_abort = False
        state.current_policy = policy
        state.current_approvals = approvals
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
            # the caller never saw `turn_started` and the turn is not in flight.
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
        if state.ref is None or state.turn_id is None:
            raise ProtocolDefect("Claude SDK did not establish stream identity")

        if not state.session_started_emitted:
            state.session_started_emitted = True
            yield self._event(state, _PendingEvent("session_started", SessionStartedData()))
        # Session-scope, and reported by the SDK at session startup: `open_session` observed
        # them from `mcp_status` before any turn existed, so they are emitted at the first
        # point in the session's first stream that can carry an event — after
        # `session_started`, ahead of `turn_started`. `events.SESSION_SCOPE_EVENT_KINDS`
        # makes that window legal for exactly this reason, and holding them back until the
        # turn had opened would be this lane reordering the backend's own frames — the thing
        # the Codex lane was refused permission to do.
        for diagnostic in state.pending_startup_diagnostics:
            yield self._event(
                state,
                _PendingEvent(
                    "diagnostic",
                    DiagnosticData(code="optional_mcp_unavailable", message=diagnostic),
                ),
            )
        state.pending_startup_diagnostics.clear()
        yield self._event(
            state, _PendingEvent("turn_started", TurnStartedData(), native_type="sdk/query")
        )

        terminal_seen = False
        try:
            for message in buffered:
                async for event in self._message_events(state, message):
                    yield event
                    terminal_seen = event.kind in (
                        "turn_completed",
                        "turn_failed",
                        "turn_cancelled",
                    )
                if terminal_seen:
                    return
            async for message in source:
                self._account_message(state, message)
                self._learn_identity(state, message)
                async for event in self._message_events(state, message):
                    yield event
                    terminal_seen = event.kind in (
                        "turn_completed",
                        "turn_failed",
                        "turn_cancelled",
                    )
                if terminal_seen:
                    return
        except OutputLimitExceeded:
            await self._stop_client(state, force_disconnect=True)
            yield self._event(
                state,
                _PendingEvent(
                    "turn_failed",
                    TurnFailedData(
                        failure="output_limit_exceeded",
                        final_text=state.final_text,
                        usage=state.usage,
                        diagnostics=tuple(dict.fromkeys(state.diagnostics)),
                    ),
                ),
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
            yield self._event(
                state,
                _PendingEvent(
                    "turn_failed",
                    TurnFailedData(
                        failure="backend_failed",
                        final_text=state.final_text,
                        usage=state.usage,
                        diagnostics=("Claude SDK stream failed",),
                    ),
                ),
            )
            return
        except Exception:
            # Not a modelled failure: tear the client down and let the defect propagate.
            await self._stop_client(state, force_disconnect=True)
            raise
        if not terminal_seen:
            await self._stop_client(state, force_disconnect=True)
            raise MissingTerminalEvent()

    async def interrupt(self, session: AgentSession, turn_id: str | None) -> None:
        if session in self._dead_sessions:
            return
        state = self._state(session)
        if turn_id is None:
            # No native turn was ever named, so nothing identifies the frames still coming
            # and no later turn could tell them apart: the whole session is released.
            await self._stop_client(state, force_disconnect=True)
            return
        if state.turn_id != turn_id:
            raise InvalidAgentRequest("interrupt turn id does not match the active Claude turn")
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

        The drain cannot run inside `interrupt()`, where the port documents the obligation:
        the runtime issues that interrupt from `_watch_turn_stop` while the abandoned turn's
        `anext` is still pending, and a second consumer of the same memory stream would race
        it for messages. Here the abandoned generator is closed and nothing else is reading,
        which is the port's other sanctioned option — discarding the frames on arrival.
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

    async def _message_events(
        self, state: _ClaudeSessionState, message: object
    ) -> AsyncIterator[AgentEvent]:
        """Number one native message's events, plus anything the permission callback queued.

        This generator is the single serialization point for `state.seq`: `_decode_message`
        only *describes* events and the permission callback — which the SDK dispatches from a
        task it spawns itself (`Query._spawn_control_request_handler`) while this generator is
        suspended — only queues descriptions. Draining the queue at every yield boundary keeps
        the emitted sequence gap-free and keeps an approval pair ahead of the event it gated.
        """
        for pending in self._drain_pending(state):
            yield pending
        async for described in self._decode_message(state, message):
            for pending in self._drain_pending(state):
                yield pending
            yield self._event(state, described)
        for pending in self._drain_pending(state):
            yield pending

    def _drain_pending(self, state: _ClaudeSessionState) -> list[AgentEvent]:
        """Number everything the permission callback queued, and surface any defect it hit."""
        defect = state.approval_defect
        if defect is not None:
            state.approval_defect = None
            raise defect
        queued = tuple(state.pending_events)
        state.pending_events.clear()
        return [self._event(state, pending) for pending in queued]

    async def _decode_message(
        self, state: _ClaudeSessionState, message: object
    ) -> AsyncIterator[_PendingEvent]:
        sdk = state.sdk
        native_type = type(message).__name__
        native_payload = self._native_payload(native_type, message)
        if isinstance(message, sdk.SystemMessage):
            for pending in self._system_events(state, message, native_type, native_payload):
                yield pending
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
                    yield _PendingEvent(
                        "text_delta",
                        TextDeltaData(text),
                        native_type=native_type,
                        native_payload=native_payload,
                    )
                    return
                if delta.get("type") == "thinking_delta":
                    thinking = delta.get("thinking")
                    if not isinstance(thinking, str):
                        raise ProtocolDefect("Claude thinking delta was malformed")
                    self._check_event_text(thinking)
                    yield _PendingEvent(
                        "reasoning",
                        ReasoningData(thinking, visibility="full"),
                        native_type=native_type,
                        native_payload=native_payload,
                    )
                    return
            yield _PendingEvent(
                "unknown",
                UnknownData(),
                native_type=native_type,
                native_payload=native_payload,
            )
            return
        if isinstance(message, sdk.AssistantMessage):
            state.quota.observe_assistant_error(getattr(message, "error", None))
            usage = getattr(message, "usage", None)
            if isinstance(usage, dict):
                frozen_usage = freeze_json_object(usage)
                state.usage = frozen_usage
                yield _PendingEvent(
                    "usage",
                    UsageData(frozen_usage),
                    native_type=native_type,
                    native_payload=native_payload,
                )
            partial = self._include_partial(state.request.native)
            content = getattr(message, "content", None)
            if not isinstance(content, (tuple, list)):
                raise ProtocolDefect("Claude assistant content was not an array")
            for block in content:
                if isinstance(block, sdk.ToolUseBlock):
                    arguments = getattr(block, "input", None)
                    if not isinstance(arguments, dict):
                        raise ProtocolDefect("Claude tool input was malformed")
                    yield _PendingEvent(
                        "tool_started",
                        ToolStartedData(
                            tool_call_id=self._required_attr(block, "id"),
                            name=self._required_attr(block, "name"),
                            arguments=freeze_json_object(arguments),
                        ),
                        native_type=native_type,
                        native_payload=native_payload,
                    )
                elif isinstance(block, sdk.TextBlock) and not partial:
                    text = getattr(block, "text", None)
                    if not isinstance(text, str):
                        raise ProtocolDefect("Claude text block was malformed")
                    self._append_final_text(state, text)
                    yield _PendingEvent(
                        "text_delta",
                        TextDeltaData(text),
                        native_type=native_type,
                        native_payload=native_payload,
                    )
                elif isinstance(block, sdk.ThinkingBlock) and not partial:
                    thinking = getattr(block, "thinking", None)
                    if not isinstance(thinking, str):
                        raise ProtocolDefect("Claude thinking block was malformed")
                    self._check_event_text(thinking)
                    yield _PendingEvent(
                        "reasoning",
                        ReasoningData(thinking, visibility="full"),
                        native_type=native_type,
                        native_payload=native_payload,
                    )
                elif not isinstance(block, (sdk.TextBlock, sdk.ThinkingBlock)):
                    yield _PendingEvent(
                        "unknown",
                        UnknownData(),
                        native_type=f"AssistantMessage:{type(block).__name__}",
                        native_payload=redact_native_payload(
                            self._object_payload(block), allowed_fields=None
                        ),
                    )
            return
        if isinstance(message, sdk.UserMessage):
            content = getattr(message, "content", ())
            if not isinstance(content, (str, tuple, list)):
                raise ProtocolDefect("Claude user content was malformed")
            blocks = content if isinstance(content, (tuple, list)) else ()
            for block in blocks:
                if isinstance(block, sdk.ToolResultBlock):
                    output = freeze_json_value(getattr(block, "content", None))
                    yield _PendingEvent(
                        "tool_completed",
                        ToolCompletedData(
                            tool_call_id=self._required_attr(block, "tool_use_id"),
                            output=output,
                            succeeded=getattr(block, "is_error", None) is not True,
                        ),
                        native_type=native_type,
                        native_payload=native_payload,
                    )
                else:
                    yield _PendingEvent(
                        "unknown",
                        UnknownData(),
                        native_type=f"UserMessage:{type(block).__name__}",
                        native_payload=redact_native_payload(
                            self._object_payload(block), allowed_fields=None
                        ),
                    )
            file_event = self._file_result_event(message, native_type, native_payload)
            if file_event is not None:
                yield file_event
            return
        if isinstance(message, sdk.RateLimitEvent):
            info = getattr(message, "rate_limit_info", None)
            status = getattr(info, "status", None)
            state.quota.observe_rate_limit_status(status)
            status_message = (
                sanitize_provider_text(status) if isinstance(status, str) else "unknown"
            )
            yield _PendingEvent(
                "diagnostic",
                DiagnosticData(
                    code="claude_rate_limit",
                    message=f"Claude rate limit status: {status_message}",
                    detail=native_payload,
                ),
                native_type=native_type,
                native_payload=native_payload,
            )
            return
        if isinstance(message, sdk.ResultMessage):
            yield self._result_event(state, message, native_type, native_payload)
            return
        # claude-agent-sdk 0.2.130 closes `Message` over six classes and drops every
        # unrecognized wire type inside `parse_message`, so nothing reaches this arm today.
        # It is kept as the fail-open landing site for a future SDK that widens that union.
        yield _PendingEvent(
            "unknown",
            UnknownData(),
            native_type=native_type,
            native_payload=redact_native_payload(
                self._object_payload(message), allowed_fields=None
            ),
        )

    def _system_events(
        self,
        state: _ClaudeSessionState,
        message: object,
        native_type: str,
        native_payload: FrozenJsonDict,
    ) -> list[_PendingEvent]:
        """Route every `system` frame, including the typed subclasses and `init`."""
        subtype = getattr(message, "subtype", None)
        if not isinstance(subtype, str) or not subtype:
            raise ProtocolDefect("Claude SystemMessage carried no subtype")
        if subtype == "init":
            return self._initialization_events(state, message, native_payload)
        if type(message) is not state.sdk.SystemMessage:
            # A typed `SystemMessage` subclass (0.2.130 ships TaskStarted/TaskProgress/
            # TaskNotification/TaskUpdated/MirrorError/HookEvent). These report work the turn
            # performed or failed to perform, so they are diagnostics, not unknown frames.
            code = subtype if _DIAGNOSTIC_CODE.fullmatch(subtype) else "system_event"
            return [
                _PendingEvent(
                    "diagnostic",
                    DiagnosticData(
                        code=f"claude_{code}",
                        message=f"Claude reported a {sanitize_provider_text(subtype, limit=64)} "
                        "system message",
                        detail=native_payload,
                    ),
                    native_type=f"SystemMessage:{subtype}",
                    native_payload=native_payload,
                )
            ]
        return [
            _PendingEvent(
                "unknown",
                UnknownData(),
                native_type=f"SystemMessage:{subtype}",
                native_payload=native_payload,
            )
        ]

    def _initialization_events(
        self,
        state: _ClaudeSessionState,
        message: object,
        native_payload: FrozenJsonDict,
    ) -> list[_PendingEvent]:
        """Check the backend's *effective* configuration against the one that was requested.

        `system/init` is the only frame that reports it. Claude Code 2.1.220 builds it in
        `tAr()` as `{cwd, session_id, tools, mcp_servers, model, permissionMode, ...}`.
        """
        data = getattr(message, "data", None)
        if not isinstance(data, Mapping):
            raise ProtocolDefect("Claude system/init carried no object payload")
        reported_cwd = data.get("cwd")
        if reported_cwd != state.request.cwd:
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
        if requested_model is None or reported_model == requested_model:
            return []
        self._record_diagnostic(state, "effective_model_changed")
        return [
            _PendingEvent(
                "diagnostic",
                DiagnosticData(
                    code="effective_model_changed",
                    message="Claude SDK reported a different effective model",
                ),
                native_type="SystemMessage:init",
                native_payload=native_payload,
            )
        ]

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

    def _result_event(
        self,
        state: _ClaudeSessionState,
        message: object,
        native_type: str,
        native_payload: FrozenJsonDict,
    ) -> _PendingEvent:
        # The native turn is over the moment its result is consumed, so nothing is left on
        # the shared stream for a later turn to inherit.
        state.turn_in_flight = False
        usage_value = getattr(message, "usage", None)
        usage = freeze_json_object(usage_value) if isinstance(usage_value, dict) else state.usage
        result = getattr(message, "result", None)
        final_text = result if isinstance(result, str) else state.final_text
        self._check_final_text(final_text)
        diagnostics = tuple(dict.fromkeys(state.diagnostics))
        terminal_reason = getattr(message, "terminal_reason", None)
        data: AgentEventData
        if state.approval_failure:
            data = TurnFailedData(
                failure="approval_unanswered",
                final_text=final_text,
                usage=usage,
                diagnostics=("approval handler failed",),
            )
            kind: AgentEventKind = "turn_failed"
        elif state.approval_abort or terminal_reason in ("aborted_streaming", "aborted_tools"):
            data = TurnCancelledData(
                final_text=final_text,
                usage=usage,
                diagnostics=diagnostics,
            )
            kind = "turn_cancelled"
        elif getattr(message, "is_error", False):
            subtype = getattr(message, "subtype", None)
            if state.quota.is_exhausted(getattr(message, "api_error_status", None)):
                failure = "quota_exhausted"
            elif subtype == "error_max_structured_output_retries":
                failure = "output_schema_violation"
            else:
                failure = "backend_failed"
            errors = getattr(message, "errors", None)
            if isinstance(errors, list):
                if len(errors) > _MAX_DIAGNOSTICS:
                    raise OutputLimitExceeded(_MAX_DIAGNOSTICS)
                diagnostics = tuple(
                    dict.fromkeys(
                        [
                            *diagnostics,
                            *(
                                sanitize_provider_text(item)
                                for item in errors
                                if isinstance(item, str) and item
                            ),
                        ]
                    )
                )
                if len(diagnostics) > _MAX_DIAGNOSTICS:
                    raise OutputLimitExceeded(_MAX_DIAGNOSTICS)
            data = TurnFailedData(
                failure=failure,
                final_text=final_text,
                usage=usage,
                diagnostics=diagnostics,
            )
            kind = "turn_failed"
        elif isinstance(state.current_output, JsonSchemaAgentOutput):
            raw_structured = getattr(message, "structured_output", None)
            try:
                structured = (
                    validate_structured_output(raw_structured, state.current_output.schema)
                    if raw_structured is not None
                    else parse_structured_output(final_text, state.current_output.schema)
                )
            except OutputSchemaMismatch:
                data = TurnFailedData(
                    failure="output_schema_violation",
                    final_text=final_text,
                    usage=usage,
                    diagnostics=diagnostics,
                )
                kind = "turn_failed"
            else:
                data = TurnCompletedData(
                    final_text=final_text,
                    structured_output=structured,
                    usage=usage,
                    diagnostics=diagnostics,
                )
                kind = "turn_completed"
        else:
            data = TurnCompletedData(
                final_text=final_text,
                usage=usage,
                diagnostics=diagnostics,
            )
            kind = "turn_completed"
        return _PendingEvent(
            kind,
            data,
            native_type=native_type,
            native_payload=native_payload,
        )

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

        `Query._spawn_control_request_handler` runs this concurrently with the turn generator,
        so it must not touch `state.seq`: it queues unnumbered descriptions and lets
        `_message_events` number them. A defect detected here would otherwise be swallowed by
        the SDK's control-request error response, so it is handed to the generator too.
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
            },
            allowed_fields=_KNOWN_FIELDS["permission/request"],
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
        summary = sanitize_provider_text(summary, limit=2_000)
        request = ApprovalRequest(
            operation=operation,
            summary=summary,
            tool_name=name if operation == "tool_use" else None,
            native_payload=native,
        )
        self._queue_pending(
            state,
            _PendingEvent(
                "approval_requested",
                ApprovalRequestedData(request),
                native_type="permission/request",
                native_payload=native,
            ),
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
        self._queue_pending(
            state,
            _PendingEvent(
                "approval_answered",
                ApprovalAnsweredData(decision),
                native_type="permission/response",
                native_payload=freeze_json_object({"decision": decision}),
            ),
        )
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
    def _queue_pending(state: _ClaudeSessionState, pending: _PendingEvent) -> None:
        if len(state.pending_events) >= _MAX_EVENT_COUNT:
            raise ProtocolDefect(
                "Claude queued more approval events than one turn may emit",
                code="output_limit_before_delivery",
            )
        state.pending_events.append(pending)

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

    def _event(self, state: _ClaudeSessionState, pending: _PendingEvent) -> AgentEvent:
        """Assign one stream sequence number.

        The only caller is the turn generator (`stream_turn` and `_message_events`), which is
        what keeps `seq` gap-free while the SDK dispatches permission callbacks concurrently.
        """
        if state.ref is None or state.turn_id is None:
            raise ProtocolDefect("Claude event arrived before stream identity")
        terminal = pending.kind in ("turn_completed", "turn_failed", "turn_cancelled")
        if state.seq >= _MAX_EVENT_COUNT or (not terminal and state.seq == _MAX_EVENT_COUNT - 1):
            raise OutputLimitExceeded(_MAX_EVENT_COUNT)
        state.seq += 1
        return AgentEvent(
            schema_version="agent-event.v1",
            seq=state.seq,
            backend="claude",
            transport="sdk",
            session_ref=state.ref,
            turn_id=state.turn_id,
            kind=pending.kind,
            data=pending.data,
            native_type=pending.native_type,
            native_payload=pending.native_payload,
        )

    def _state(self, session: AgentSession) -> _ClaudeSessionState:
        if session in self._dead_sessions:
            raise SessionUnavailable("Claude SDK session is no longer live")
        try:
            return self._sessions[session]
        except KeyError as error:
            raise InvalidAgentRequest("session is not owned by this Claude adapter") from error

    @staticmethod
    def _load_sdk() -> ModuleType:
        if os.environ.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
            raise UnsupportedCapability("ambient Claude SDK version-check bypass is forbidden")
        try:
            return importlib.import_module("claude_agent_sdk")
        except ModuleNotFoundError as error:
            raise SdkUnavailable(
                "claude-agent-sdk is unavailable; install the 'claude-sdk' extra"
            ) from error

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
    def _require_scope(scope: AgentCapabilityScope) -> None:
        if scope.backend != "claude" or scope.transport != "sdk":
            raise InvalidAgentRequest("Claude SDK received a different capability scope")

    @staticmethod
    def _require_local_auth(kind: str) -> None:
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
                # `capabilities()` advertises `streamable_http` only, so this is unreachable
                # through the runtime. It fails loudly rather than silently emitting a config
                # shape nobody checked, should that advertisement ever widen.
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
        return {
            "type": "json_schema",
            "schema": to_json_schema(output.schema, inline_defs=False, include_annotations=True),
        }

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

        Read from the same table `capabilities()` publishes, because the two classifications
        are the same classification: `_decide_permission` denies `command`/`file_change`
        outright under `filesystem: read_only`, so a name advertised as a file write that
        this method called an ordinary tool use would be a write allowed under a read-only
        policy.
        """
        if name in _COMMAND_TOOL_NAMES:
            return "command"
        if name in _FILE_CHANGE_TOOL_NAMES:
            return "file_change"
        return "tool_use"

    @staticmethod
    def _file_result_event(
        message: object,
        native_type: str,
        native_payload: FrozenJsonDict,
    ) -> _PendingEvent | None:
        result = getattr(message, "tool_use_result", None)
        if not isinstance(result, Mapping):
            return None
        change_type = result.get("type")
        mapped: Literal["created", "modified", "deleted"] | None
        if change_type == "create":
            mapped = "created"
        elif change_type == "update":
            mapped = "modified"
        elif change_type == "delete":
            mapped = "deleted"
        else:
            mapped = None
        path = result.get("filePath")
        if mapped is None or not isinstance(path, str):
            return None
        return _PendingEvent(
            "file_change",
            # The SDK surfaces a file change through the tool *result*, so the write has
            # already happened by the time this message arrives.
            FileChangeData(path=path, change=mapped, status="applied"),
            native_type=native_type,
            native_payload=native_payload,
        )

    @staticmethod
    def _native_payload(native_type: str, message: object) -> FrozenJsonDict:
        payload = ClaudeSdkAdapter._object_payload(message)
        return redact_native_payload(payload, allowed_fields=_KNOWN_FIELDS.get(native_type))

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
