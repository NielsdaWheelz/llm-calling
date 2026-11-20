"""Strict managed cognition on the shared Codex protocol."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import weakref
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any, Literal, cast
from uuid import uuid4

from provider_runtime.errors import sanitize_provider_text
from provider_runtime.types import Absent, Presence, Present, TokenUsage, canonical_json_bytes

from ._limits import (
    _MAX_DIAGNOSTICS,
    _MAX_EVENT_COUNT,
    _MAX_EVENT_TEXT_BYTES,
    _MAX_FINAL_TEXT_BYTES,
    _MAX_MESSAGE_BYTES,
    _MAX_MESSAGE_ITEMS,
    _MAX_TURN_OUTPUT_BYTES,
    _OPERATION_TIMEOUT_SECONDS,
    OutputLimitExceeded,
    bounded_payload_size,
)
from .auth import (
    freeze_native_json_object,
    freeze_native_json_value,
    redact_native_payload,
)
from .codex_app_server import (
    CodexAppServerClient,
    CodexAppServerConfig,
    CodexAppServerResponseError,
    CodexConnectionUnavailable,
    CodexOutputLimit,
    CodexRawQueueLimit,
    CodexRequestId,
    CodexServerRequest,
    NativeRequestFit,
    _buffer_observation,
    codex_wire_request_fit,
)
from .codex_containment import _startup_flag, _validate_codex_containment_config
from .codex_endpoint import CodexEndpoint, fingerprint_codex_endpoint
from .errors import (
    ConflictingSubmissionEvidence,
    CredentialRejected,
    CredentialUnavailable,
    ExecutableUnavailable,
    InvalidAgentRequest,
    McpUnavailable,
    ProtocolDefect,
    SdkUnavailable,
    SessionMismatch,
    SessionUnavailable,
    TurnNotStarted,
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
from .model_catalog import AgentModelCatalog, read_codex_model_catalog
from .policy import PermissionPolicy
from .sessions import (
    AgentSession,
    SessionMetadata,
    SessionPage,
    SessionQuery,
    SessionReadOptions,
    SessionSnapshot,
    SessionSummary,
    fingerprint_path,
    validate_read_session_auth,
    validate_session_ref,
)
from .turn import (
    AgentAccepted,
    AgentAttempt,
    AgentCloseResult,
    AgentControlReceipt,
    AgentInputRecorded,
    AgentMessage,
    AgentNotSubmitted,
    AgentSubmission,
    AgentSubmissionRejected,
    AgentToolCall,
    AgentToolReply,
    AgentTurn,
    AgentTurnControls,
    AgentTurnRef,
    AgentUncertain,
    LocalStopEvidence,
    NativeTerminalEvidence,
)
from .types import (
    AgentOutputSpec,
    AgentSessionRef,
    AgentSessionRequest,
    ApprovalHandler,
    ApprovalRequest,
    CodexNativeOptions,
    CodexSandboxControls,
    ContentPart,
    CredentialRef,
    ForkSession,
    ImageContent,
    JsonObject,
    JsonSchemaAgentOutput,
    JsonValue,
    NewSession,
    ResumeSession,
    TextContent,
    TurnRequest,
    _ResolvedCodexSessionRequest,
    freeze_json_object,
    freeze_json_value,
    thaw_json_value,
    validate_mcp_network_policy,
)

_REQUIRED_MCP_STARTUP_FAILURE = "required MCP servers failed to initialize"


def codex_native_request_fit(
    text: str,
    *,
    output: AgentOutputSpec,
    reasoning: str,
    input_id: str,
    operation: Literal["start", "steer"] = "start",
) -> NativeRequestFit:
    """Plan with the fixed-width native identities and largest legal RPC id."""
    identity = "00000000-0000-0000-0000-000000000000"
    parts = (TextContent(text),)
    if operation == "start":
        submitted = CodexSdkAdapter._turn_request(
            thread_id=identity,
            input=parts,
            approval_mode="deny_all",
            reasoning=reasoning,
            input_id=input_id,
            output=output,
            contained=True,
        )
        return codex_wire_request_fit(cast(dict[str, object], thaw_json_value(submitted)))
    if operation == "steer":
        params = CodexSdkAdapter._steer_params(identity, identity, input_id, parts)
        return codex_wire_request_fit({"method": "turn/steer", "params": params})
    raise InvalidAgentRequest("unknown Codex fit operation")


def _stock_size_rejection(error: Exception, method: str) -> tuple[int, int] | None:
    if (
        not isinstance(error, CodexAppServerResponseError)
        or error.method != method
        or error.code != -32602
    ):
        return None
    data = error.data
    if not isinstance(data, dict) or data.get("input_error_code") != "input_too_large":
        return None
    actual, maximum = data.get("actual_chars"), data.get("max_chars")
    if type(actual) is not int or type(maximum) is not int or not actual > maximum > 0:
        return None
    return actual, maximum


# The configurable execution, integration, and local-context features in the
# Codex runtime. Keep this as one closed vendor mapping behind
# `CodexNativeOptions.builtin_tools`; callers must not depend on Codex feature names.
_DISABLED_BUILTIN_FEATURES = (
    "apply_patch_streaming_events",
    "apps",
    "artifact",
    "auth_elicitation",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "code_mode",
    "code_mode_host",
    "code_mode_only",
    "computer_use",
    "chronicle",
    "current_time_reminder",
    "default_mode_request_user_input",
    "deferred_executor",
    "enable_fanout",
    "enable_mcp_apps",
    "exec_permission_approvals",
    "goals",
    "guardian_approval",
    "hooks",
    "image_generation",
    "in_app_browser",
    "incremental_tools",
    "memories",
    "mentions_v2",
    "multi_agent",
    "multi_agent_v2",
    "non_prefixed_mcp_tool_names",
    "plugins",
    "plugin_sharing",
    "remote_plugin",
    "request_permissions_tool",
    "rollout_budget",
    "shell_snapshot",
    "shell_tool",
    "shell_zsh_fork",
    "sleep_tool",
    "skill_mcp_dependency_install",
    "standalone_web_search",
    "send_message_to_user_async",
    "terminal_visualization_instructions",
    "token_budget",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "unified_exec",
    "unified_exec_zsh_fork",
    "view_image",
    "web_search_cached",
    "web_search_request",
    "workspace_dependencies",
)

_TURN_SCOPED_METHODS = frozenset(
    {
        "error",
        "turn/started",
        "turn/completed",
        "rawResponseItem/completed",
        "rawResponse/completed",
        "thread/tokenUsage/updated",
        "item/started",
        "item/completed",
        "item/agentMessage/delta",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/summaryPartAdded",
        "item/reasoning/textDelta",
        "item/plan/delta",
        "item/commandExecution/outputDelta",
        "item/commandExecution/terminalInteraction",
        "item/autoApprovalReview/started",
        "item/autoApprovalReview/completed",
        "item/fileChange/patchUpdated",
        "item/fileChange/outputDelta",
        "item/mcpToolCall/progress",
        "turn/diff/updated",
        "turn/plan/updated",
        "hook/started",
        "hook/completed",
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
        "item/permissions/requestApproval",
        "item/tool/call",
        "item/tool/requestUserInput",
    }
)

_INERT_NATIVE_METHODS = frozenset(
    {
        "turn/started",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/summaryPartAdded",
        "item/reasoning/textDelta",
        "item/plan/delta",
        "turn/plan/updated",
        "thread/status/changed",
        "thread/settings/updated",
        "account/rateLimits/updated",
        "model/rerouted",
        "model/safetyBuffering/updated",
        "model/verification",
        "turn/moderationMetadata",
        "thread/compacted",
        "thread/goal/cleared",
        "thread/goal/updated",
        "thread/name/updated",
        "thread/readState/changed",
        "serverRequest/resolved",
        "mcpServer/startupStatus/updated",
        "remoteControl/status/changed",
        "deprecationNotice",
    }
)

_THREAD_BOUND_INERT_METHODS = frozenset(
    {
        "thread/compacted",
        "thread/goal/cleared",
        "thread/goal/updated",
        "thread/name/updated",
        "thread/readState/changed",
        "thread/status/changed",
        "thread/settings/updated",
    }
)

_FORBIDDEN_SESSION_NOTIFICATIONS = frozenset(
    {
        "thread/archived",
        "thread/closed",
        "thread/deleted",
        "thread/unarchived",
    }
)

_INERT_ITEM_TYPES = frozenset(
    {
        "userMessage",
        "agentMessage",
        "plan",
        "reasoning",
        "enteredReviewMode",
        "exitedReviewMode",
        "contextCompaction",
    }
)

_PRETURN_INERT_METHODS = frozenset(
    {
        "account/updated",
        "account/rateLimits/updated",
        "configWarning",
        "deprecationNotice",
        "warning",
        "thread/started",
        "thread/compacted",
        "thread/goal/cleared",
        "thread/goal/updated",
        "thread/name/updated",
        "thread/readState/changed",
        "thread/status/changed",
        "thread/settings/updated",
        "mcpServer/startupStatus/updated",
        "remoteControl/status/changed",
    }
)

_AUTHORITY_ITEM_TYPES = frozenset(
    {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "collabAgentToolCall",
        "subAgentActivity",
        "webSearch",
        "imageView",
        "sleep",
        "imageGeneration",
        "functionCallOutput",
        "hookPrompt",
    }
)

type FileChangeStatus = Literal["in_progress", "applied", "failed", "declined"]
type AgentMessagePhase = Literal["commentary", "final_answer"] | None
_PATCH_APPLY_STATUS: dict[str, FileChangeStatus] = {
    "inProgress": "in_progress",
    "completed": "applied",
    "failed": "failed",
    "declined": "declined",
}


@dataclass(frozen=True, slots=True)
class _CompletedAgentMessage:
    """One authoritative SDK item, retained in native completion order."""

    item_id: str
    text: str
    phase: AgentMessagePhase


def _presence_delta(
    current: Presence[int], baseline: Presence[int], *, field_name: str
) -> Presence[int]:
    match current, baseline:
        case Present(value=current_count), Present(value=baseline_count):
            if current_count < baseline_count:
                raise ProtocolDefect(
                    f"Codex cumulative usage {field_name} decreased within one session"
                )
            return Present(current_count - baseline_count)
        case Absent(), Absent():
            return Absent()
        case _:
            raise ProtocolDefect(f"Codex cumulative usage {field_name} changed field presence")


def _usage_delta(current: TokenUsage, baseline: TokenUsage) -> TokenUsage:
    """Subtract two validated cumulative snapshots without inventing counters."""
    input_tokens = current.input_tokens - baseline.input_tokens
    output_tokens = current.output_tokens - baseline.output_tokens
    total_tokens = current.total_tokens - baseline.total_tokens
    if input_tokens < 0 or output_tokens < 0 or total_tokens < 0:
        raise ProtocolDefect("Codex cumulative usage decreased within one session")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=_presence_delta(
            current.reasoning_tokens,
            baseline.reasoning_tokens,
            field_name="reasoning_tokens",
        ),
        cache_read_input_tokens=_presence_delta(
            current.cache_read_input_tokens,
            baseline.cache_read_input_tokens,
            field_name="cache_read_input_tokens",
        ),
        cache_write_input_tokens=_presence_delta(
            current.cache_write_input_tokens,
            baseline.cache_write_input_tokens,
            field_name="cache_write_input_tokens",
        ),
    )


def _sum_reported(left: Presence[int], right: Presence[int]) -> Presence[int]:
    if isinstance(left, Present) and isinstance(right, Present):
        return Present(left.value + right.value)
    return Absent()


@dataclass(slots=True)
class _CodexUsageAccounting:
    """Raw thread counters and the attributable sum for the current invocation."""

    baseline_known: bool
    cumulative: TokenUsage | None = None
    turn_requires_rebase: bool = False
    local_usage: TokenUsage | None = None
    turn_open: bool = False

    def begin_turn(self) -> None:
        if self.turn_open:
            raise ProtocolDefect("Codex usage accounting began overlapping turns")
        self.turn_open = True
        self.turn_requires_rebase = not self.baseline_known
        self.baseline_known = False
        self.local_usage = None

    def observe(self, current: TokenUsage, context_full: bool) -> TokenUsage | None:
        if not self.turn_open:
            raise ProtocolDefect("Codex usage snapshot arrived outside its turn")
        if context_full:
            # Stock replaces every raw counter. Preserve usage already attributed,
            # and subtract this synthetic boundary from the next paid snapshot.
            self.cumulative = current
            self.baseline_known = not self.turn_requires_rebase
            self.turn_requires_rebase = False
            return None
        previous = self.cumulative
        delta = current if previous is None else _usage_delta(current, previous)
        self.cumulative = current
        if self.turn_requires_rebase:
            self.turn_requires_rebase = False
            return None
        if current == previous:
            return None
        prefix = self.local_usage
        if prefix is not None:
            delta = TokenUsage(
                input_tokens=prefix.input_tokens + delta.input_tokens,
                output_tokens=prefix.output_tokens + delta.output_tokens,
                total_tokens=prefix.total_tokens + delta.total_tokens,
                reasoning_tokens=_sum_reported(prefix.reasoning_tokens, delta.reasoning_tokens),
                cache_read_input_tokens=_sum_reported(
                    prefix.cache_read_input_tokens, delta.cache_read_input_tokens
                ),
                cache_write_input_tokens=_sum_reported(
                    prefix.cache_write_input_tokens, delta.cache_write_input_tokens
                ),
            )
        self.local_usage = delta
        self.baseline_known = True
        return delta

    def finish_turn(self) -> Presence[TokenUsage]:
        if not self.turn_open:
            raise ProtocolDefect("Codex usage accounting finalized a turn more than once")
        self.turn_open = False
        return Absent() if self.local_usage is None else Present(self.local_usage)

    def abandon_turn(self) -> None:
        if not self.turn_open:
            return
        self.turn_open = False
        # Runtime cancellation can stop consumption before Codex's final usage frame.
        # Preserve usage already emitted, but do not reuse a possibly partial boundary.
        self.baseline_known = False


@dataclass(slots=True)
class _CodexSessionState:
    client: Any
    thread: Any
    request: _ResolvedCodexSessionRequest
    ref: AgentSessionRef
    usage_accounting: _CodexUsageAccounting
    turn: Any | None = None
    turn_id: str | None = None
    message_count: int = 0
    output_bytes: int = 0
    streamed_text_bytes: int = 0
    completed_agent_messages: list[_CompletedAgentMessage] = field(default_factory=list)
    completed_item_ids: set[str] = field(default_factory=set)
    started_item_types: dict[str, str] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    active_mcp_calls: dict[str, tuple[str, str]] = field(default_factory=dict)
    active_tool_calls: dict[str, str] = field(default_factory=dict)
    custom_item_calls: dict[str, str] = field(default_factory=dict)
    unpublished_calls: dict[str, AgentToolUse] = field(default_factory=dict)
    server_request_ids: set[tuple[type[object], object]] = field(default_factory=set)
    authority_seen: bool = False
    quota_exhausted: bool = False
    user_message_id: str | None = None
    attempt: AgentAttempt | None = None
    controlled: bool = False
    handle: _CodexAgentTurn | None = None


class CodexSdkAdapter:
    """Own Codex app-server clients and normalize their complete message stream."""

    backend: Literal["codex"] = "codex"
    transport: Literal["sdk"] = "sdk"
    # Codex threads resume from any directory; the ref keeps cwd as provenance only.
    cwd_scopes_sessions: Literal[False] = False

    def __init__(
        self,
        *,
        endpoints: Mapping[str, CodexEndpoint],
        sandbox_controls: CodexSandboxControls | None = None,
    ) -> None:
        if sandbox_controls is not None and not isinstance(sandbox_controls, CodexSandboxControls):
            raise InvalidAgentRequest(
                "sandbox_controls must be CodexSandboxControls when configured"
            )
        self._endpoints = dict(endpoints)
        self._sandbox_controls = sandbox_controls
        self._sessions: dict[AgentSession, _CodexSessionState] = {}
        self._dead_sessions: weakref.WeakSet[AgentSession] = weakref.WeakSet()
        self._clients: set[Any] = set()

    def validate_auth(self, credential: CredentialRef) -> None:
        self._require_local_auth(credential.kind)

    async def model_catalog(self, *, environment: Mapping[str, str]) -> AgentModelCatalog:
        client = await self._open_client(environment=environment)
        try:
            await self._verify_auth(client)
            return await self._call(
                read_codex_model_catalog(client), operation="model catalog", failure="executable"
            )
        finally:
            await self._close_client(client)

    async def list_sessions(
        self,
        query: SessionQuery,
        *,
        environment: Mapping[str, str],
    ) -> SessionPage:
        if query.backend != self.backend or query.transport != self.transport:
            raise InvalidAgentRequest("CodexSdkAdapter received a different route")
        self._require_local_auth(query.auth.kind)
        client = await self._open_client(environment=environment)
        try:
            await self._verify_auth(client)
            response = await self._call(
                client.thread_list(archived=None, cursor=query.cursor, limit=query.limit),
                operation="thread listing",
                failure="executable",
            )
            payload = self._mapping(response, "Codex SDK thread_list response")
            data = payload.get("data")
            if not isinstance(data, list):
                raise ProtocolDefect("Codex SDK thread_list data was not an array")
            endpoint = self._endpoint(environment)
            sessions = tuple(
                self._session_summary(
                    self._mapping(item, "Codex SDK thread_list thread"),
                    profile_key=query.auth.profile_key,
                    endpoint=endpoint,
                )
                for item in data
            )
            cursor = payload.get("nextCursor")
            if cursor is not None and (not isinstance(cursor, str) or not cursor):
                raise ProtocolDefect("Codex SDK thread_list cursor was malformed")
            return SessionPage(sessions=sessions, continuation_cursor=cursor)
        finally:
            await self._close_client(client)

    async def read_session(
        self,
        ref: AgentSessionRef,
        options: SessionReadOptions,
        *,
        environment: Mapping[str, str],
    ) -> SessionSnapshot:
        if ref.backend != self.backend or ref.transport != self.transport:
            raise SessionMismatch("Codex SDK cannot read a different route")
        validate_read_session_auth(ref, options)
        self._require_local_auth(options.auth.kind)
        if ref.state_root_fingerprint != fingerprint_codex_endpoint(self._endpoint(environment)):
            raise SessionMismatch("session state root does not match the supplied environment")
        client = await self._open_client(environment=environment)
        try:
            await self._verify_auth(client)
            response = await self._call(
                client.request(
                    "thread/read", {"threadId": ref.native_session_id, "includeTurns": False}
                ),
                operation="thread read",
                failure="session",
            )
            payload = self._mapping(response, "Codex SDK thread read response")
            native_thread = self._mapping(payload.get("thread"), "Codex SDK thread read thread")
            if native_thread.get("id") != ref.native_session_id:
                raise ProtocolDefect("Codex SDK thread read changed the native identity")
            return SessionSnapshot(
                ref=ref,
                metadata=SessionMetadata(name=self._optional_string(native_thread.get("name"))),
            )
        finally:
            await self._close_client(client)

    async def open_session(
        self,
        request: AgentSessionRequest,
        *,
        environment: Mapping[str, str],
    ) -> AgentSession:
        if not isinstance(request, _ResolvedCodexSessionRequest):
            raise InvalidAgentRequest("CodexSdkAdapter requires a catalog-resolved request")
        self._require_local_auth(request.auth.kind)
        self._validate_policy_mapping(request.policy)
        validate_mcp_network_policy(request.mcp_servers, request.policy)
        self._validate_mcp_filters(request)
        endpoint = self._endpoint(environment)
        native = request.native
        contained = isinstance(native, CodexNativeOptions) and native.builtin_tools == "disabled"
        internal = isinstance(native, CodexNativeOptions) and native.archive_internal
        if internal and not isinstance(request.open, NewSession):
            raise UnsupportedCapability(
                "internal Codex cognition requires a fresh ephemeral session"
            )
        if contained:
            self._validate_strict_native_containment(request)
        client = await self._open_client(
            environment=environment,
            experimental_api=contained or bool(request.tools),
            callbacks=bool(request.tools),
        )
        try:
            await self._verify_auth(client)
            config = self._codex_config(request)
            startup = self._mapping(
                await self._call(
                    client.request("config/read", {"includeLayers": False, "cwd": request.cwd}),
                    operation="native host startup configuration",
                    failure="session",
                ),
                "Codex host startup configuration",
            )
            if contained or request.tools:
                _validate_codex_containment_config(startup, client.metadata)
            service_tier = _startup_flag(startup, "service_tier")
            if service_tier is not None:
                if not isinstance(service_tier, str) or not service_tier:
                    raise ProtocolDefect("Codex host declared an invalid startup service tier")
                request = replace(request, startup_service_tier=service_tier)
            effective = self._mapping(startup.get("config"), "Codex effective configuration")
            inherited = self._mapping(effective.get("mcp_servers"), "Codex inherited MCP servers")
            # Native config merges tables recursively; an empty table retains
            # inherited servers. Declared servers alone replace their disabled entries.
            declared = self._mapping(config["mcp_servers"], "Codex declared MCP servers")
            config["mcp_servers"] = {
                **{name: {"enabled": False} for name in inherited},
                **declared,
            }
            kwargs: dict[str, object] = {
                "approval_mode": self._approval_mode(request.policy),
                "config": config,
                "cwd": request.cwd,
                "sandbox": self._sandbox(request.policy),
            }
            kwargs["model"] = request.dispatch_model
            if internal:
                from .archive import INTERNAL_THREAD_SOURCE

                kwargs["thread_source"] = INTERNAL_THREAD_SOURCE
                kwargs["ephemeral"] = True
            if isinstance(request.open, NewSession) and contained:
                kwargs["environments"] = []
                kwargs["experimentalRawEvents"] = True
            if request.tools:
                kwargs["dynamicTools"] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": thaw_json_value(freeze_json_object(tool.parameters)),
                    }
                    for tool in request.tools
                ]
            if request.system:
                # `baseInstructions` is the public App Server system-role field on thread
                # start/resume/fork and *replaces* Codex's built-in base prompt rather than
                # appending to it, which is what a caller asking for session system
                # instructions is asking for.
                kwargs["base_instructions"] = self._text_only(
                    request.system, "Codex system instructions"
                )
            if request.developer:
                kwargs["developer_instructions"] = self._text_only(
                    request.developer, "Codex developer instructions"
                )

            if isinstance(request.open, NewSession):
                thread = await self._call(
                    client.thread_start(**kwargs), operation="thread start", failure="session"
                )
            elif isinstance(request.open, ResumeSession):
                self._validate_open_ref(request, request.open.ref, environment)
                thread = await self._call(
                    client.thread_resume(request.open.ref.native_session_id, **kwargs),
                    operation="thread resume",
                    failure="session",
                )
            elif isinstance(request.open, ForkSession):
                self._validate_open_ref(request, request.open.ref, environment)
                thread = await self._call(
                    client.thread_fork(request.open.ref.native_session_id, **kwargs),
                    operation="thread fork",
                    failure="session",
                )
            else:
                raise InvalidAgentRequest("unknown Codex session operation")

            self._validate_thread_settings(
                request,
                {
                    **thread.settings,
                    "effort": thread.settings.get("reasoningEffort"),
                    "sandboxPolicy": thread.settings.get("sandbox"),
                },
            )
            native_session_id = getattr(thread, "id", None)
            if not isinstance(native_session_id, str) or not native_session_id:
                raise ProtocolDefect("Codex SDK returned no thread id")
            if isinstance(request.open, ResumeSession) and (
                native_session_id != request.open.ref.native_session_id
            ):
                raise SessionUnavailable("Codex SDK resumed a different native thread")
            if isinstance(request.open, ForkSession) and (
                native_session_id == request.open.ref.native_session_id
            ):
                raise ProtocolDefect("Codex SDK fork did not mint a new thread id")
            restored_usage = self._restored_usage_baseline(client, native_session_id, request)
            if isinstance(request.open, NewSession) and restored_usage is not None:
                raise ProtocolDefect("Codex new thread replayed historical usage")

            ref = self._make_ref(
                native_session_id=native_session_id,
                profile_key=request.auth.profile_key,
                endpoint=endpoint,
                cwd=request.cwd,
            )
            session = AgentSession(ref)
            self._sessions[session] = _CodexSessionState(
                client=client,
                thread=thread,
                request=request,
                ref=ref,
                usage_accounting=_CodexUsageAccounting(
                    baseline_known=isinstance(request.open, NewSession)
                    or restored_usage is not None,
                    cumulative=restored_usage,
                ),
            )
            return session
        except BaseException:
            await self._close_client(client)
            raise

    def prepare_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        attempt_id: str,
        input_id: str,
        controls: AgentTurnControls,
        release: Callable[[], None],
        validate_input: Callable[[tuple[ContentPart, ...]], None] | None = None,
        controlled: bool = True,
    ) -> AgentTurn:
        state = self._state(session)
        if request.policy is not None:
            raise UnsupportedCapability("Codex cannot reconfigure policy on a started thread")
        if not state.client.usable:
            raise SessionUnavailable("Codex session connection is no longer usable")
        if type(input_id) is not str or not input_id:
            raise InvalidAgentRequest("prepared input_id must be non-empty")
        accounting = state.usage_accounting
        restored = self._restored_usage_baseline(
            state.client, state.ref.native_session_id, state.request, accounting.cumulative
        )
        if restored is not None:
            if state.attempt is None and isinstance(state.request.open, NewSession):
                raise ProtocolDefect("Codex new thread replayed historical usage")
            accounting.cumulative = restored
            accounting.baseline_known = True
        submitted = self._turn_request(
            thread_id=state.ref.native_session_id,
            input=request.input,
            approval_mode=self._approval_mode(state.request.policy),
            reasoning=state.request.native_reasoning,
            input_id=input_id,
            output=state.request.output,
            contained=self._strict_native_containment(state),
        )
        codex_wire_request_fit(cast(dict[str, object], thaw_json_value(submitted))).require_fit()
        attempt = AgentAttempt(
            attempt_id, hashlib.sha256(canonical_json_bytes(submitted)).hexdigest()
        )
        state.turn = None
        state.turn_id = None
        state.user_message_id = input_id
        state.attempt = attempt
        state.controlled = controlled
        state.message_count = 0
        state.output_bytes = 0
        state.streamed_text_bytes = 0
        state.completed_agent_messages.clear()
        state.completed_item_ids.clear()
        state.started_item_types.clear()
        state.usage_accounting.begin_turn()
        state.diagnostics.clear()
        state.active_mcp_calls.clear()
        state.active_tool_calls.clear()
        state.custom_item_calls.clear()
        state.unpublished_calls.clear()
        state.server_request_ids.clear()
        state.authority_seen = False
        state.quota_exhausted = False
        handle = _CodexAgentTurn(
            self, session, state, submitted, request, controls, release, validate_input
        )
        state.handle = handle
        return handle

    def session_usable(self, session: AgentSession) -> bool:
        state = self._sessions.get(session)
        return (
            state is not None
            and state.client.usable
            and (
                state.handle is None
                or state.handle._closed
                and (
                    state.handle.terminal is not None
                    or isinstance(state.handle.submission, AgentNotSubmitted)
                )
            )
        )

    def turn_submission(self, session: AgentSession) -> AgentSubmission | None:
        state = self._sessions.get(session)
        return None if state is None or state.handle is None else state.handle.submission

    async def stream_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None,
    ) -> AsyncGenerator[AgentEvent, None]:
        state = self._state(session)
        if approvals is not None:
            raise UnsupportedCapability("Codex does not expose caller approval callbacks")
        if state.request.tools:
            raise UnsupportedCapability("declared callbacks require prepare_turn, not stream_turn")
        handle = self.prepare_turn(
            session,
            request,
            attempt_id=str(uuid4()),
            input_id=str(uuid4()),
            controls=AgentTurnControls(_OPERATION_TIMEOUT_SECONDS, 16, 1_048_576),
            release=lambda: None,
            controlled=False,
        )
        try:
            submission = await handle.submit()
            if isinstance(submission, AgentSubmissionRejected):
                raise InvalidAgentRequest(
                    f"Codex refused input of {submission.actual_chars}/{submission.max_chars} characters"
                )
            if isinstance(submission, AgentNotSubmitted):
                raise SessionUnavailable(submission.reason)
            async for event in handle.events():
                yield event
        finally:
            await handle.close()

    async def interrupt(self, session: AgentSession) -> None:
        if session in self._dead_sessions:
            return
        state = self._state(session)
        if state.handle is not None:
            await state.handle.interrupt()
            return
        if state.turn is None:
            # Block-and-stop: a turn that never became identifiable releases the session.
            await self._destroy_session(session, state)
            return
        await self._call(state.turn.interrupt(), operation="turn interrupt", failure="session")

    def _interrupted_final_text(self, session: AgentSession) -> str:
        """Project only an already-completed eligible message on runtime interruption."""
        return self._selected_final_text(self._state(session), required=False)

    async def close_session(self, session: AgentSession) -> None:
        """Idempotently release one App Server client while preserving sibling sessions."""
        if session in self._dead_sessions:
            return
        state = self._sessions.pop(session, None)
        if state is None:
            raise InvalidAgentRequest("session is not owned by this Codex adapter")
        self._dead_sessions.add(session)
        await self._close_client(state.client)

    async def close(self) -> None:
        sessions = tuple(self._sessions)
        session_results = await asyncio.gather(
            *(self.close_session(session) for session in sessions), return_exceptions=True
        )
        clients = tuple(self._clients)
        results = await asyncio.gather(
            *(self._close_client(client) for client in clients), return_exceptions=True
        )
        if any(isinstance(result, BaseException) for result in (*session_results, *results)):
            raise ProtocolDefect(
                "Codex App Server client teardown did not complete", code="sdk_teardown_failed"
            )

    async def _open_client(
        self,
        *,
        environment: Mapping[str, str],
        experimental_api: bool = False,
        callbacks: bool = False,
    ) -> CodexAppServerClient:
        client = CodexAppServerClient(
            CodexAppServerConfig(
                endpoint=self._endpoint(environment),
                experimental_api=experimental_api,
                callbacks=callbacks,
            )
        )
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await client.__aenter__()
        except BaseException:
            await client.close()
            raise
        self._clients.add(client)
        return client

    def _endpoint(self, environment: Mapping[str, str]) -> CodexEndpoint:
        endpoint = self._endpoints.get(environment.get("CODEX_PROFILE", ""))
        if endpoint is None:
            raise CredentialUnavailable("Codex profile has no configured external endpoint")
        return endpoint

    def _restored_usage_baseline(
        self,
        client: Any,
        native_session_id: str,
        request: _ResolvedCodexSessionRequest,
        previous: TokenUsage | None = None,
    ) -> TokenUsage | None:
        """Consume and validate app-server replay emitted before resume/fork responds."""
        take_pending = getattr(client, "take_pending_messages", None)
        if not callable(take_pending):
            raise SdkUnavailable(
                "Codex app-server transport does not expose its pending message queue"
            )
        notifications = cast(tuple[object, ...], take_pending())

        snapshot_count = 0
        restored = None
        replay_bytes = 0
        for notification in notifications:
            if isinstance(notification, CodexServerRequest):
                raise ProtocolDefect("Codex server request arrived outside an active turn")
            method = getattr(notification, "method", None)
            if method in _PRETURN_INERT_METHODS:
                raw_params = getattr(notification, "params", None)
                if raw_params is None:
                    payload = getattr(notification, "payload", None)
                    raw_params = getattr(payload, "params", payload)
                params = self._mapping(raw_params, f"Codex app-server {method} notification")
                try:
                    replay_bytes += bounded_payload_size(
                        params,
                        _MAX_MESSAGE_BYTES,
                        max_items=_MAX_MESSAGE_ITEMS,
                    )
                except OutputLimitExceeded:
                    raise ProtocolDefect(
                        "Codex app-server pre-turn payload exceeded its ingress bound"
                    ) from None
                if replay_bytes > _MAX_TURN_OUTPUT_BYTES:
                    raise ProtocolDefect(
                        "Codex app-server pre-turn replay exceeded its ingress bound"
                    )
                thread = params.get("thread")
                observed_id = (
                    thread.get("id") if isinstance(thread, Mapping) else params.get("threadId")
                )
                if method in _THREAD_BOUND_INERT_METHODS and observed_id != native_session_id:
                    raise ProtocolDefect("Codex replay omitted or changed thread identity")
                if observed_id is not None and observed_id != native_session_id:
                    raise ProtocolDefect("Codex replay changed thread identity")
                if method == "thread/settings/updated":
                    self._validate_thread_settings(
                        request,
                        self._mapping(params.get("threadSettings"), "native thread settings"),
                    )
                continue
            if method != "thread/tokenUsage/updated":
                raise ProtocolDefect(f"Codex emitted unexpected pre-turn notification {method}")
            raw_params = getattr(notification, "params", None)
            if raw_params is None:
                payload = getattr(notification, "payload", None)
                raw_params = getattr(payload, "params", payload)
            bounded_params = raw_params
            try:
                replay_bytes += bounded_payload_size(
                    bounded_params,
                    _MAX_MESSAGE_BYTES,
                    max_items=_MAX_MESSAGE_ITEMS,
                )
            except OutputLimitExceeded:
                raise ProtocolDefect(
                    "Codex SDK restored usage payload exceeded its ingress bound"
                ) from None
            if snapshot_count >= _MAX_EVENT_COUNT or replay_bytes > _MAX_TURN_OUTPUT_BYTES:
                raise ProtocolDefect("Codex SDK restored usage replay exceeded its ingress bound")
            params = self._mapping(raw_params, "Codex app-server restored token usage notification")
            if params.get("threadId") != native_session_id:
                continue
            self._non_empty_string(params, "turnId", "restored token usage notification")
            restored, context_full = self._decode_token_usage(params)
            if previous is not None and not context_full:
                _usage_delta(restored, previous)
            previous = restored
            snapshot_count += 1
        return restored

    async def _close_client(self, client: Any) -> None:
        self._clients.discard(client)
        await client.close()

    async def _destroy_session(self, session: AgentSession, state: _CodexSessionState) -> None:
        if self._sessions.get(session) is not state:
            if session in self._dead_sessions:
                return
            raise ProtocolDefect("Codex session state changed before teardown")
        await self.close_session(session)

    async def _verify_auth(self, client: Any) -> None:
        response = await self._call(
            client.account(), operation="account discovery", failure="credential"
        )
        payload = self._mapping(response, "Codex SDK account response")
        account = payload.get("account")
        if account is None:
            raise CredentialUnavailable("Codex has no authenticated local account")
        account_payload = self._mapping(account, "Codex SDK account")
        if account_payload.get("type") != "chatgpt":
            raise CredentialRejected("Codex local_account requires ChatGPT subscription auth")

    async def _call(
        self,
        awaitable: Any,
        *,
        operation: str,
        failure: Literal["credential", "executable", "session"],
    ) -> Any:
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                return await awaitable
        except TimeoutError:
            if failure == "credential":
                raise CredentialUnavailable(f"Codex SDK {operation} timed out") from None
            if failure == "session":
                raise SessionUnavailable(f"Codex SDK {operation} timed out") from None
            raise ExecutableUnavailable(f"Codex SDK {operation} timed out") from None
        except (
            CredentialRejected,
            CredentialUnavailable,
            ExecutableUnavailable,
            ProtocolDefect,
            SessionUnavailable,
        ):
            raise
        except Exception as error:
            message = sanitize_provider_text(str(error))
            if _REQUIRED_MCP_STARTUP_FAILURE in message:
                raise McpUnavailable("a required MCP server did not initialize") from None
            if failure == "credential":
                raise CredentialUnavailable(f"Codex SDK {operation} failed: {message}") from None
            if failure == "session":
                raise SessionUnavailable(f"Codex SDK {operation} failed: {message}") from None
            raise ExecutableUnavailable(f"Codex SDK {operation} failed: {message}") from None

    def _notification_events(
        self, state: _CodexSessionState, notification: object
    ) -> tuple[AgentEvent, ...]:
        if isinstance(notification, CodexServerRequest):
            return (self._server_request_event(state, notification),)
        method = getattr(notification, "method", None)
        if not isinstance(method, str) or not method:
            raise ProtocolDefect("Codex app-server notification had no method")
        direct_params = getattr(notification, "params", None)
        payload = getattr(notification, "payload", None)
        if direct_params is None and payload is None:
            raise ProtocolDefect("Codex app-server notification had no params")
        raw_params = (
            direct_params if direct_params is not None else getattr(payload, "params", payload)
        )
        bounded_params = getattr(raw_params, "root", raw_params)
        try:
            size = bounded_payload_size(
                bounded_params,
                _MAX_MESSAGE_BYTES,
                max_items=_MAX_MESSAGE_ITEMS,
            )
        except OutputLimitExceeded:
            self._append_diagnostic(
                state,
                f"{method} native payload exceeded its ingress bound",
            )
            raise
        if not state.controlled:
            state.message_count += 1
            if state.message_count > _MAX_EVENT_COUNT:
                raise OutputLimitExceeded(_MAX_EVENT_COUNT)
            if state.output_bytes + size > _MAX_TURN_OUTPUT_BYTES:
                raise OutputLimitExceeded(_MAX_TURN_OUTPUT_BYTES)
            state.output_bytes += size
        params = self._mapping(raw_params, f"Codex app-server {method} notification")
        self._validate_notification_identity(state, method, params)
        if self._strict_native_containment(state) and method in ("item/started", "item/completed"):
            item = self._mapping(params.get("item"), "Codex item")
            if item.get("type") == "userMessage" and (
                state.handle is None or not state.handle.accepts_input(item.get("clientId"))
            ):
                raise ProtocolDefect("Codex cognition observed foreign user input")
        if method == "turn/completed":
            # The native completion frame travels first; the owned terminal is last.
            return (
                AgentNative(native_type=method, payload=redact_native_payload(params)),
                self._turn_terminal(state, params),
            )
        event = self._notification_event(state, method, params)
        return () if event is None else (event,)

    def _server_request_event(
        self,
        state: _CodexSessionState,
        request: CodexServerRequest,
    ) -> AgentEvent:
        try:
            size = bounded_payload_size(
                request.params,
                _MAX_MESSAGE_BYTES,
                max_items=_MAX_MESSAGE_ITEMS,
            )
        except OutputLimitExceeded:
            raise ProtocolDefect("Codex server request exceeded its ingress bound") from None
        if not state.controlled:
            state.message_count += 1
            if state.message_count > _MAX_EVENT_COUNT:
                raise OutputLimitExceeded(_MAX_EVENT_COUNT)
            if state.output_bytes + size > _MAX_TURN_OUTPUT_BYTES:
                raise OutputLimitExceeded(_MAX_TURN_OUTPUT_BYTES)
            state.output_bytes += size
        identity = (type(request.request_id), request.request_id)
        if request.method == "item/tool/call" and state.controlled and state.request.tools:
            self._validate_notification_identity(state, request.method, request.params)
            state.server_request_ids.add(identity)
            if state.handle is None:
                raise ProtocolDefect("controlled callback has no owned turn handle")
            return state.handle.tool_call(request)
        if identity in state.server_request_ids:
            raise ProtocolDefect("Codex server request identity repeated within a turn")
        state.server_request_ids.add(identity)
        params = request.params
        if request.method in ("execCommandApproval", "applyPatchApproval"):
            if params.get("conversationId") != state.ref.native_session_id:
                raise ProtocolDefect("legacy Codex approval changed thread identity")
            self._non_empty_string(params, "callId", request.method)
        elif request.method == "mcpServer/elicitation/request":
            if params.get("threadId") != state.ref.native_session_id:
                raise ProtocolDefect("MCP elicitation changed thread identity")
            turn_id = params.get("turnId")
            if turn_id is not None and turn_id != state.turn_id:
                raise ProtocolDefect("MCP elicitation changed turn identity")
        else:
            self._validate_notification_identity(state, request.method, params)
        state.authority_seen = True
        if request.method == "currentTime/read":
            if params.get("threadId") != state.ref.native_session_id:
                raise ProtocolDefect("current-time request changed thread identity")
            return AgentToolUse(
                tool_call_id=f"server-request:{request.request_id}",
                name="currentTime/read",
                phase="completed",
                payload=redact_native_payload(params),
                succeeded=False,
            )
        if request.kind == "tool":
            call_id = self._non_empty_string(params, "callId", request.method)
            tool = self._non_empty_string(params, "tool", request.method)
            expected = state.active_tool_calls.get(call_id)
            name = self._dynamic_tool_name(params, tool)
            if expected != name:
                raise ProtocolDefect("dynamic tool request did not match its started item")
            return AgentToolUse(
                tool_call_id=call_id,
                name=name,
                phase="updated",
                payload=redact_native_payload(params),
            )

        operation: Literal["command", "file_change", "tool_use"]
        tool_name: str | None = None
        if request.method in (
            "item/commandExecution/requestApproval",
            "execCommandApproval",
        ):
            operation = "command"
        elif request.method in ("item/fileChange/requestApproval", "applyPatchApproval"):
            operation = "file_change"
        else:
            operation = "tool_use"
            tool_name = request.method
        if request.method in (
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
            "item/tool/requestUserInput",
        ):
            item_id = self._non_empty_string(params, "itemId", request.method)
            if operation in ("command", "file_change"):
                name = "commandExecution" if operation == "command" else "fileChange"
                self._require_active_tool(state, item_id, name, request.method)
        return AgentPermissionRequest(
            request=ApprovalRequest(
                operation=operation,
                summary=sanitize_provider_text(
                    f"Codex denied {request.method}",
                    limit=2_000,
                ),
                tool_name=tool_name,
                native_payload=redact_native_payload(params),
            ),
            decision="deny",
        )

    def _notification_event(
        self,
        state: _CodexSessionState,
        method: str,
        params: Mapping[str, object],
    ) -> AgentEvent | None:
        if method == "rawResponse/completed":
            return None
        if method == "rawResponseItem/completed":
            item = self._mapping(params.get("item"), "raw Responses item")
            kind = self._non_empty_string(item, "type", method)
            if kind in ("message", "reasoning"):
                return None
            if kind == "configuration_update":
                if item != {"type": kind, "reasoning": {"effort": state.request.native_reasoning}}:
                    raise ProtocolDefect(
                        "Codex raw configuration changed the submitted reasoning policy"
                    )
                return None
            if kind == "function_call_output":
                held = state.unpublished_calls.pop(
                    self._non_empty_string(item, "call_id", method), None
                )
                if held is None or item.get("output") == f"unsupported call: {held.name}":
                    return None
                state.authority_seen = True
                return held
            if kind in ("function_call", "custom_tool_call"):
                name = self._non_empty_string(item, "name", method)
                namespace = item.get("namespace")
                if namespace not in (None, "functions"):
                    if type(namespace) is not str or not namespace:
                        raise ProtocolDefect("raw Responses tool namespace is malformed")
                    name = f"{namespace}/{name}"
                if kind == "function_call" and name in {tool.name for tool in state.request.tools}:
                    return None
                event = AgentToolUse(
                    tool_call_id=self._non_empty_string(item, "call_id", method),
                    name=name,
                    phase="started",
                    payload=redact_native_payload(item),
                )
                if kind == "function_call":
                    # A function the session never published: Codex answers the model
                    # itself and runs nothing. That is a model error, not authority,
                    # but only Codex's exact refusal of this call proves it.
                    if len(state.unpublished_calls) >= _MAX_MESSAGE_ITEMS:
                        raise ProtocolDefect("unpublished Codex calls exceeded their finite bound")
                    state.unpublished_calls[event.tool_call_id] = event
                    return None
                state.authority_seen = True
                return event
            raise ProtocolDefect(f"Codex raw Responses item has unknown type {kind}")
        if method == "thread/started":
            thread = self._mapping(params.get("thread"), "thread/started thread")
            if self._string(thread, "id", method) != state.ref.native_session_id:
                raise ProtocolDefect("Codex event changed thread identity")
            return None
        if method == "turn/started":
            return AgentNative(native_type=method, payload=redact_native_payload(params))
        if method == "item/agentMessage/delta":
            self._non_empty_string(params, "itemId", method)
            delta = self._string(params, "delta", method)
            self._count_streamed_text(state, delta)
            return AgentText(delta)
        if method in (
            "item/reasoning/summaryTextDelta",
            "item/reasoning/summaryPartAdded",
            "item/reasoning/textDelta",
            "item/plan/delta",
            "turn/plan/updated",
        ):
            return AgentNative(native_type=method, payload=redact_native_payload(params))
        if method in ("item/commandExecution/outputDelta", "item/fileChange/outputDelta"):
            item_id = self._string(params, "itemId", method)
            name = (
                "commandExecution"
                if method == "item/commandExecution/outputDelta"
                else "fileChange"
            )
            self._require_active_tool(state, item_id, name, method)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name=name,
                phase="updated",
                payload=freeze_native_json_object(
                    {"output_delta": self._string(params, "delta", method)}
                ),
            )
        if method in (
            "item/autoApprovalReview/started",
            "item/autoApprovalReview/completed",
        ):
            if state.request.policy.approval == "deny" and not self._strict_native_containment(
                state
            ):
                raise ProtocolDefect("Codex requested undeclared automatic approval authority")
            review_id = self._non_empty_string(params, "reviewId", method)
            state.authority_seen = True
            if method.endswith("/started"):
                self._start_tool(state, review_id, "autoApprovalReview")
                return AgentToolUse(
                    tool_call_id=review_id,
                    name="autoApprovalReview",
                    phase="started",
                    payload=redact_native_payload(params),
                )
            review = self._mapping(params.get("review"), f"{method} review")
            status = review.get("status")
            if status not in ("approved", "denied", "timedOut", "aborted"):
                raise ProtocolDefect("completed Codex auto-approval review had an invalid status")
            self._complete_tool(state, review_id, "autoApprovalReview", method)
            return AgentToolUse(
                tool_call_id=review_id,
                name="autoApprovalReview",
                phase="completed",
                payload=redact_native_payload(params),
                succeeded=status == "approved",
            )
        if method == "item/mcpToolCall/progress":
            item_id = self._string(params, "itemId", method)
            identity = state.active_mcp_calls.get(item_id)
            if identity is None:
                raise ProtocolDefect("MCP tool progress arrived before its start")
            server, tool = identity
            self._require_active_tool(state, item_id, f"{server}/{tool}", method)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name=f"{server}/{tool}",
                phase="updated",
                payload=freeze_native_json_object(
                    {"message": self._string(params, "message", method)}
                ),
            )
        if method == "item/commandExecution/terminalInteraction":
            item_id = self._non_empty_string(params, "itemId", method)
            self._require_active_tool(state, item_id, "commandExecution", method)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name="commandExecution",
                phase="updated",
                payload=redact_native_payload(params),
            )
        if method == "item/started":
            return self._item_started(state, params, method)
        if method == "item/completed":
            return self._item_completed(state, params, method)
        if method == "item/fileChange/patchUpdated":
            item_id = self._string(params, "itemId", method)
            self._require_active_tool(state, item_id, "fileChange", method)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name="fileChange",
                phase="updated",
                payload=freeze_native_json_object({"changes": params.get("changes")}),
            )
        if method == "thread/tokenUsage/updated":
            cumulative, context_full = self._token_usage(params)
            usage = state.usage_accounting.observe(cumulative, context_full)
            return None if usage is None else AgentUsage(usage)
        if method == "error":
            error = self._mapping(params.get("error"), "error notification")
            message = error.get("message")
            normalized = (
                sanitize_provider_text(message)
                if isinstance(message, str) and message
                else "Codex error"
            )
            self._append_diagnostic(state, normalized)
            if self._is_quota_error(error):
                state.quota_exhausted = True
            # Retries the backend performs itself are visible here too (willRetry);
            # the bounded native frame is their only representation.
            return AgentNative(native_type=method, payload=redact_native_payload(params))
        if method in ("configWarning", "warning", "guardianWarning"):
            message = params.get("summary", params.get("message"))
            if not isinstance(message, str) or not message:
                raise ProtocolDefect(f"{method} carried no message")
            if (
                method == "guardianWarning"
                and params.get("threadId") != state.ref.native_session_id
            ):
                raise ProtocolDefect("guardianWarning changed or omitted its thread identity")
            self._append_diagnostic(state, sanitize_provider_text(message))
            return AgentNative(native_type=method, payload=redact_native_payload(params))
        if method == "serverRequest/resolved":
            request_id = params.get("requestId")
            if type(request_id) not in (str, int) or request_id == "":
                raise ProtocolDefect("serverRequest/resolved had a malformed identity")
            identity = (type(request_id), request_id)
            if identity not in state.server_request_ids and not state.controlled:
                raise ProtocolDefect("serverRequest/resolved did not match a server request")
            state.server_request_ids.discard(identity)
            return AgentNative(native_type=method, payload=redact_native_payload(params))
        if method == "turn/diff/updated":
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=f"{state.turn_id}:file-diff",
                name="fileChange",
                phase="updated",
                payload=redact_native_payload(params),
            )
        if method in ("hook/started", "hook/completed"):
            run = self._mapping(params.get("run"), f"{method} run")
            run_id = self._non_empty_string(run, "id", method)
            status = self._non_empty_string(run, "status", method)
            state.authority_seen = True
            if method == "hook/started":
                if status != "running":
                    raise ProtocolDefect("started Codex hook had an invalid status")
                self._start_tool(state, run_id, "hook")
                return AgentToolUse(
                    tool_call_id=run_id,
                    name="hook",
                    phase="started",
                    payload=redact_native_payload(params),
                )
            if status not in ("completed", "failed", "blocked", "stopped"):
                raise ProtocolDefect("completed Codex hook had an invalid status")
            self._complete_tool(state, run_id, "hook", method)
            return AgentToolUse(
                tool_call_id=run_id,
                name="hook",
                phase="completed",
                payload=redact_native_payload(params),
                succeeded=status == "completed",
            )
        if method in ("command/exec/outputDelta", "process/outputDelta", "process/exited"):
            item_id = params.get("processId", params.get("processHandle"))
            if not isinstance(item_id, str) or not item_id:
                raise ProtocolDefect(f"{method} carried no process identity")
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name="processExecution",
                phase="completed" if method == "process/exited" else "updated",
                payload=redact_native_payload(params),
                succeeded=params.get("exitCode") == 0 if method == "process/exited" else None,
            )
        if method in _FORBIDDEN_SESSION_NOTIFICATIONS:
            raise ProtocolDefect(f"Codex app-server changed session lifecycle via {method}")
        if method == "thread/settings/updated":
            self._validate_thread_settings(
                state.request, self._mapping(params.get("threadSettings"), "native thread settings")
            )
        if method == "model/rerouted":
            raise ProtocolDefect("Codex rerouted the selected native model")
        if method in _INERT_NATIVE_METHODS:
            return AgentNative(native_type=method, payload=redact_native_payload(params))
        raise ProtocolDefect(f"Codex app-server emitted unknown notification {method}")

    def _item_started(
        self,
        state: _CodexSessionState,
        params: Mapping[str, object],
        method: str,
    ) -> AgentEvent | None:
        item = self._mapping(params.get("item"), "item/started item")
        item_type = item.get("type")
        if not isinstance(item_type, str) or not item_type:
            raise ProtocolDefect("item/started item had no type")
        if item_type == "custom_tool_call_output":
            raise ProtocolDefect("custom tool output started as an independent item")
        item_id = self._non_empty_string(item, "id", method)
        if item_id in state.started_item_types or (
            not state.controlled and item_id in state.completed_item_ids
        ):
            raise ProtocolDefect("Codex item identity started more than once")
        if len(state.started_item_types) >= _MAX_MESSAGE_ITEMS:
            raise ProtocolDefect("Codex active item count exceeded its finite bound")
        state.started_item_types[item_id] = item_type
        if item_type in _INERT_ITEM_TYPES:
            return None
        if item_type == "commandExecution":
            self._start_tool(state, item_id, "commandExecution")
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name="commandExecution",
                phase="started",
                payload=freeze_native_json_object(
                    {"command": item.get("command"), "cwd": item.get("cwd")}
                ),
            )
        if item_type == "mcpToolCall":
            server = self._string(item, "server", method)
            tool = self._string(item, "tool", method)
            requested = {spec.name: spec for spec in state.request.mcp_servers}
            spec = requested.get(server)
            if spec is None:
                raise ProtocolDefect("MCP tool call used an unconfigured server")
            if (spec.allowed_tools and tool not in spec.allowed_tools) or tool in spec.denied_tools:
                raise ProtocolDefect("MCP tool call violated its exact tool policy")
            if item_id in state.active_mcp_calls:
                raise ProtocolDefect("MCP tool call started more than once")
            state.active_mcp_calls[item_id] = (server, tool)
            self._start_tool(state, item_id, f"{server}/{tool}")
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name=f"{server}/{tool}",
                phase="started",
                payload=freeze_native_json_value(item.get("arguments")),
            )
        if item_type == "fileChange":
            self._start_tool(state, item_id, "fileChange")
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name="fileChange",
                phase="started",
                payload=freeze_native_json_object(
                    {
                        "changes": item.get("changes"),
                        "status": self._patch_status(item.get("status"), method),
                    }
                ),
            )
        if item_type == "custom_tool_call":
            call_id = self._non_empty_string(item, "call_id", method)
            name = self._non_empty_string(item, "name", method)
            state.custom_item_calls[item_id] = call_id
            self._start_tool(state, call_id, name)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=call_id,
                name=name,
                phase="started",
                payload=redact_native_payload(item),
            )
        if item_type == "dynamicToolCall":
            tool = self._non_empty_string(item, "tool", method)
            name = self._dynamic_tool_name(item, tool)
            self._start_tool(state, item_id, name)
            if state.controlled and state.request.tools:
                if state.handle is None:
                    raise ProtocolDefect("declared tool item has no controlled turn")
                state.handle.tool_started(item_id, name, item.get("arguments"))
                return None
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name=name,
                phase="started",
                payload=redact_native_payload(item),
            )
        if item_type == "webSearch" and not self._strict_native_containment(state):
            native = state.request.native
            if not isinstance(native, CodexNativeOptions) or native.web_search is not True:
                raise ProtocolDefect("Codex used disabled native web search")
        if item_type in _AUTHORITY_ITEM_TYPES:
            self._start_tool(state, item_id, item_type)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name=item_type,
                phase="started",
                payload=redact_native_payload(item),
            )
        raise ProtocolDefect(f"{method} carried unknown item type {item_type}")

    def _item_completed(
        self,
        state: _CodexSessionState,
        params: Mapping[str, object],
        method: str,
    ) -> AgentEvent | None:
        item = self._mapping(params.get("item"), "item/completed item")
        item_type = item.get("type")
        if not isinstance(item_type, str) or not item_type:
            raise ProtocolDefect("item/completed item had no type")
        if item_type == "custom_tool_call_output":
            call_id = self._non_empty_string(item, "call_id", method)
            completion_id = f"custom-output:{call_id}"
            if not state.controlled:
                if completion_id in state.completed_item_ids:
                    raise ProtocolDefect("Codex custom tool output completed more than once")
                state.completed_item_ids.add(completion_id)
            name = state.active_tool_calls.pop(call_id, None)
            if name is None:
                raise ProtocolDefect("custom tool output completed before its call started")
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=call_id,
                name=name,
                phase="completed",
                payload=redact_native_payload(item),
                succeeded=True,
            )
        item_id = self._non_empty_string(item, "id", method)
        if not state.controlled:
            if item_id in state.completed_item_ids:
                raise ProtocolDefect("Codex item identity completed more than once")
            state.completed_item_ids.add(item_id)
        started_type = state.started_item_types.pop(item_id, None)
        if (
            item_type not in _INERT_ITEM_TYPES
            and item_type not in _AUTHORITY_ITEM_TYPES
            and item_type not in ("custom_tool_call",)
        ):
            raise ProtocolDefect(f"{method} carried unknown item type {item_type}")
        if started_type is not None and started_type != item_type:
            raise ProtocolDefect("Codex item changed type during its lifecycle")
        if item_type in _INERT_ITEM_TYPES:
            if item_type == "agentMessage":
                self._record_completed_agent_message(state, item_id, item, method)
                if state.controlled:
                    if state.turn_id is None:
                        raise ProtocolDefect("completed message omitted its native turn")
                    phase = item.get("phase")
                    return AgentMessage(
                        AgentTurnRef(state.ref, state.turn_id),
                        item_id,
                        "unknown"
                        if phase is None
                        else cast(Literal["commentary", "final_answer"], phase),
                        self._string(item, "text", method),
                    )
            if item_type == "userMessage" and state.controlled:
                if state.turn_id is None or state.handle is None:
                    raise ProtocolDefect("recorded input omitted its controlled native turn")
                input_id = self._non_empty_string(item, "clientId", method)
                return AgentInputRecorded(AgentTurnRef(state.ref, state.turn_id), input_id, item_id)
            return None
        if started_type is None:
            raise ProtocolDefect("Codex authority item completed before its start")
        if item_type == "commandExecution":
            self._complete_tool(state, item_id, "commandExecution", method)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name="commandExecution",
                phase="completed",
                payload=freeze_native_json_value(item.get("aggregatedOutput")),
                succeeded=item.get("status") == "completed",
            )
        if item_type == "mcpToolCall":
            identity = state.active_mcp_calls.pop(item_id, None)
            if identity is None:
                raise ProtocolDefect("MCP tool call completed before its start")
            if (
                self._string(item, "server", method),
                self._string(item, "tool", method),
            ) != identity:
                raise ProtocolDefect("MCP tool call changed server or tool identity")
            status = item.get("status")
            if status not in ("completed", "failed"):
                raise ProtocolDefect("completed MCP tool call had an impossible status")
            server, tool = identity
            self._complete_tool(state, item_id, f"{server}/{tool}", method)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name=f"{server}/{tool}",
                phase="completed",
                payload=freeze_native_json_value(
                    item.get("result") if status == "completed" else item.get("error")
                ),
                succeeded=status == "completed",
            )
        if item_type == "fileChange":
            # A declined or failed patch is a completed tool action that did not apply.
            status = self._patch_status(item.get("status"), method)
            self._complete_tool(state, item_id, "fileChange", method)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=item_id,
                name="fileChange",
                phase="completed",
                payload=freeze_native_json_object({"changes": item.get("changes")}),
                succeeded=status == "applied",
            )
        if item_type == "custom_tool_call":
            call_id = self._non_empty_string(item, "call_id", method)
            if state.custom_item_calls.pop(item_id, None) != call_id:
                raise ProtocolDefect("custom tool call changed its call identity")
            name = self._non_empty_string(item, "name", method)
            self._require_active_tool(state, call_id, name, method)
            state.authority_seen = True
            return AgentToolUse(
                tool_call_id=call_id,
                name=name,
                phase="updated",
                payload=redact_native_payload(item),
            )
        if item_type == "dynamicToolCall":
            tool = self._non_empty_string(item, "tool", method)
            name = self._dynamic_tool_name(item, tool)
        else:
            name = item_type
        self._complete_tool(state, item_id, name, method)
        if item_type == "dynamicToolCall" and state.controlled and state.request.tools:
            if state.handle is None:
                raise ProtocolDefect("completed declared tool has no owned handle")
            state.handle.tool_completed(item_id, item.get("status"))
            return None
        state.authority_seen = True
        status = item.get("status")
        succeeded = status not in ("failed", "declined")
        return AgentToolUse(
            tool_call_id=item_id,
            name=name,
            phase="completed",
            payload=redact_native_payload(item),
            succeeded=succeeded,
        )

    def _turn_terminal(
        self,
        state: _CodexSessionState,
        params: Mapping[str, object],
    ) -> AgentTerminal:
        turn = self._mapping(params.get("turn"), "turn/completed turn")
        status = turn.get("status")
        aborted_items = state.controlled and status in ("interrupted", "failed")
        if state.active_mcp_calls:
            raise ProtocolDefect("turn completed with active MCP tool calls")
        if state.started_item_types and not aborted_items:
            raise ProtocolDefect("turn completed with unfinished Codex item lifecycles")
        if state.active_tool_calls and not aborted_items:
            raise ProtocolDefect("turn completed with active Codex authority items")
        if state.server_request_ids and not aborted_items:
            raise ProtocolDefect("turn completed with unresolved Codex server requests")
        if state.unpublished_calls:
            # No refusal arrived, so nothing proves an unpublished call did not run.
            state.authority_seen = True
        if state.authority_seen and self._strict_native_containment(state):
            raise ProtocolDefect("turn completed after forbidden Codex native authority activity")
        final_text = self._selected_final_text(state, required=status == "completed")
        usage = state.usage_accounting.finish_turn()
        diagnostics = tuple(state.diagnostics)
        if state.attempt is None or state.turn_id is None:
            raise ProtocolDefect("Codex native terminal omitted its submitted attempt")
        evidence = NativeTerminalEvidence(
            state.attempt,
            AgentTurnRef(state.ref, state.turn_id),
            "codex-turn-completed.v1",
        )
        if status == "completed":
            return AgentTerminal(
                status="succeeded",
                failure=None,
                final_text=final_text,
                session_ref=state.ref,
                evidence=evidence,
                usage=usage,
                diagnostics=diagnostics,
            )
        if status == "interrupted":
            return AgentTerminal(
                status="cancelled",
                failure=None,
                final_text=final_text,
                session_ref=state.ref,
                evidence=evidence,
                usage=usage,
                diagnostics=diagnostics,
            )
        if status == "failed":
            failure: AgentTerminalFailure = (
                AgentQuotaExhausted()
                if state.quota_exhausted or self._is_quota_error(turn.get("error"))
                else AgentFailure("backend_failed")
            )
            return AgentTerminal(
                status="failed",
                failure=failure,
                final_text=final_text,
                session_ref=state.ref,
                evidence=evidence,
                usage=usage,
                diagnostics=diagnostics,
            )
        raise ProtocolDefect("turn/completed carried an impossible status")

    def _token_usage(self, params: Mapping[str, object]) -> tuple[TokenUsage, bool]:
        """Decode a live turn update, independently witnessed in the live matrix."""
        return self._decode_token_usage(params)

    def _decode_token_usage(self, params: Mapping[str, object]) -> tuple[TokenUsage, bool]:
        """Return the cumulative counters and whether stock reset them to a full window."""
        token_usage = self._mapping(params.get("tokenUsage"), "token usage")
        total = self._usage_member(token_usage, "total")
        last = self._usage_member(token_usage, "last")
        window = token_usage.get("modelContextWindow")
        if self._zero_components(last):
            if (
                type(window) is int
                and window > 0
                and total.total_tokens == window
                and last.total_tokens <= window
                and self._zero_components(total)
            ):
                return total, True
            # Native recompute_token_usage replaces last with an estimated
            # context size, which may exceed cumulative consumption.
            if last.total_tokens > 0:
                return total, False
        self._validate_last_usage(last, total)
        return total, False

    @staticmethod
    def _zero_components(usage: TokenUsage) -> bool:
        return (
            usage.input_tokens == usage.output_tokens == 0
            and usage.cache_read_input_tokens == Present(0)
            and usage.cache_write_input_tokens == Present(0)
            and usage.reasoning_tokens == Present(0)
        )

    def _usage_member(self, token_usage: Mapping[str, object], member_name: str) -> TokenUsage:
        member = self._mapping(token_usage.get(member_name), f"token usage {member_name}")
        total_tokens = self._usage_presence(member, member_name, "totalTokens")
        if isinstance(total_tokens, Absent):
            raise ProtocolDefect(f"tokenUsage.{member_name}.totalTokens was missing")
        try:
            usage = TokenUsage.from_components(
                input_tokens=self._usage_count(member, member_name, "inputTokens"),
                output_tokens=self._usage_count(member, member_name, "outputTokens"),
                total_tokens=total_tokens,
                reasoning_tokens=self._usage_presence(member, member_name, "reasoningOutputTokens"),
                cache_read_input_tokens=self._usage_presence(
                    member, member_name, "cachedInputTokens"
                ),
                cache_write_input_tokens=self._usage_presence(
                    member, member_name, "cacheWriteInputTokens"
                ),
            )
        except ValueError:
            raise ProtocolDefect("thread/tokenUsage/updated carried negative counts") from None
        if (
            self._presence_value(usage.reasoning_tokens) > usage.output_tokens
            or self._presence_value(usage.cache_read_input_tokens)
            + self._presence_value(usage.cache_write_input_tokens)
            > usage.input_tokens
        ):
            raise ProtocolDefect(
                f"tokenUsage.{member_name} component counts were internally inconsistent"
            )
        return usage

    @staticmethod
    def _usage_count(member: Mapping[str, object], member_name: str, key: str) -> int:
        value = member.get(key)
        if value is None:
            raise ProtocolDefect(f"tokenUsage.{member_name}.{key} was missing")
        if type(value) is not int:
            raise ProtocolDefect(f"tokenUsage.{member_name}.{key} was not an integer")
        return value

    @staticmethod
    def _usage_presence(member: Mapping[str, object], member_name: str, key: str) -> Presence[int]:
        value = member.get(key)
        if value is None:
            return Absent()
        if type(value) is not int:
            raise ProtocolDefect(f"tokenUsage.{member_name}.{key} was not an integer")
        return Present(value)

    @staticmethod
    def _presence_value(value: Presence[int]) -> int:
        return value.value if isinstance(value, Present) else 0

    @staticmethod
    def _start_tool(state: _CodexSessionState, item_id: str, name: str) -> None:
        if item_id in state.active_tool_calls:
            raise ProtocolDefect("Codex authority identity started more than once")
        if len(state.active_tool_calls) >= _MAX_MESSAGE_ITEMS:
            raise ProtocolDefect("Codex active authority count exceeded its finite bound")
        state.active_tool_calls[item_id] = name

    @staticmethod
    def _require_active_tool(
        state: _CodexSessionState,
        item_id: str,
        name: str,
        method: str,
    ) -> None:
        if state.active_tool_calls.get(item_id) != name:
            raise ProtocolDefect(f"{method} did not match an active Codex authority item")

    @classmethod
    def _complete_tool(
        cls,
        state: _CodexSessionState,
        item_id: str,
        name: str,
        method: str,
    ) -> None:
        cls._require_active_tool(state, item_id, name, method)
        del state.active_tool_calls[item_id]

    @staticmethod
    def _dynamic_tool_name(params: Mapping[str, object], tool: str) -> str:
        namespace = params.get("namespace")
        if namespace is None:
            return tool
        if not isinstance(namespace, str) or not namespace:
            raise ProtocolDefect("dynamic tool namespace was malformed")
        return f"{namespace}/{tool}"

    @staticmethod
    def _strict_native_containment(state: _CodexSessionState) -> bool:
        native = state.request.native
        return isinstance(native, CodexNativeOptions) and native.builtin_tools == "disabled"

    @classmethod
    def _validate_last_usage(cls, last: TokenUsage, total: TokenUsage) -> None:
        if (
            last.input_tokens > total.input_tokens
            or last.output_tokens > total.output_tokens
            or last.total_tokens > total.total_tokens
        ):
            raise ProtocolDefect("tokenUsage.last exceeded tokenUsage.total")
        for last_value, total_value in (
            (last.reasoning_tokens, total.reasoning_tokens),
            (last.cache_read_input_tokens, total.cache_read_input_tokens),
            (last.cache_write_input_tokens, total.cache_write_input_tokens),
        ):
            if isinstance(last_value, Present):
                if not isinstance(total_value, Present) or last_value.value > total_value.value:
                    raise ProtocolDefect("tokenUsage.last exceeded tokenUsage.total")

    def _validate_notification_identity(
        self, state: _CodexSessionState, method: str, params: Mapping[str, object]
    ) -> None:
        safe_method = sanitize_provider_text(method, limit=200)
        thread_id = params.get("threadId")
        if thread_id is not None and thread_id != state.ref.native_session_id:
            raise ProtocolDefect(f"Codex {safe_method} event changed thread identity")
        if method in _THREAD_BOUND_INERT_METHODS | _FORBIDDEN_SESSION_NOTIFICATIONS:
            if thread_id != state.ref.native_session_id:
                raise ProtocolDefect(f"Codex {safe_method} event omitted its thread identity")
        turn_id = params.get("turnId")
        turn = params.get("turn")
        if turn_id is None and isinstance(turn, Mapping):
            turn_id = turn.get("id")
        if method == "thread/tokenUsage/updated":
            if thread_id != state.ref.native_session_id:
                raise ProtocolDefect(
                    "Codex thread/tokenUsage/updated event omitted its thread identity"
                )
            if isinstance(turn_id, str) and turn_id and turn_id == state.turn_id:
                return
            # A resume response and its historical cumulative-usage replay are separate
            # app-server messages. The reader can observe the response first and the
            # replay only after the next turn has started. A single stale, non-empty
            # turn id is an unambiguous baseline only while this resumed session still
            # requires its first rebase; it is never charged to the active invocation.
            if isinstance(turn_id, str) and turn_id and state.usage_accounting.turn_requires_rebase:
                return
            if turn_id is None:
                raise ProtocolDefect(
                    "Codex thread/tokenUsage/updated event omitted its turn identity"
                )
            raise ProtocolDefect("Codex thread/tokenUsage/updated event changed its turn identity")
        if method in _TURN_SCOPED_METHODS:
            if thread_id != state.ref.native_session_id:
                raise ProtocolDefect(f"Codex {safe_method} event omitted its thread identity")
            if turn_id != state.turn_id:
                raise ProtocolDefect(
                    f"Codex {safe_method} event changed or omitted its turn identity"
                )

    def _validate_thread_settings(
        self, request: _ResolvedCodexSessionRequest, settings: Mapping[str, object]
    ) -> None:
        if (
            settings.get("model") != request.dispatch_model
            or settings.get("modelProvider") != "openai"
            or settings.get("cwd") != request.cwd
        ):
            raise ProtocolDefect("Codex native settings changed the selected model or cwd")
        if settings.get("effort") != request.native_reasoning:
            raise ProtocolDefect("Codex native settings changed the selected reasoning effort")
        if (
            request.startup_service_tier is not None
            and settings.get("serviceTier") != request.startup_service_tier
        ):
            raise ProtocolDefect("Codex native settings changed the host startup service tier")
        approval = "never" if request.policy.approval == "deny" else "on-request"
        if settings.get("approvalPolicy") != approval or (
            request.policy.approval == "provider_review"
            and settings.get("approvalsReviewer") != "auto_review"
        ):
            raise ProtocolDefect("Codex native settings changed the approval policy")
        sandbox = self._mapping(settings.get("sandboxPolicy"), "native sandbox policy")
        expected: dict[str, object]
        if request.policy.filesystem == "read_only":
            expected = {"type": "readOnly", "networkAccess": False}
        elif request.policy.filesystem == "full_access":
            expected = {"type": "dangerFullAccess"}
        else:
            controls = self._sandbox_controls
            expected = {
                "type": "workspaceWrite",
                # Stock's public projection leaves the workspace cwd implicit.
                "writableRoots": [path for path in request.additional_dirs if path != request.cwd],
                "networkAccess": request.policy.network == "unrestricted",
                "excludeTmpdirEnvVar": controls.exclude_tmpdir_env_var if controls else False,
                "excludeSlashTmp": controls.exclude_slash_tmp if controls else False,
            }
        if sandbox != expected or any(
            type(sandbox.get(key)) is not type(value) for key, value in expected.items()
        ):
            raise ProtocolDefect("Codex native settings changed the sandbox policy")

    def _state(self, session: AgentSession) -> _CodexSessionState:
        if session in self._dead_sessions:
            raise SessionUnavailable("Codex SDK session is no longer live")
        try:
            return self._sessions[session]
        except KeyError as error:
            raise InvalidAgentRequest("session is not owned by this Codex adapter") from error

    def _validate_open_ref(
        self,
        request: AgentSessionRequest,
        ref: AgentSessionRef,
        environment: Mapping[str, str],
    ) -> None:
        validate_session_ref(
            ref,
            backend=self.backend,
            transport=self.transport,
            profile_key=request.auth.profile_key,
            state_root_fingerprint=fingerprint_codex_endpoint(self._endpoint(environment)),
            cwd=request.cwd,
            cwd_scopes_sessions=self.cwd_scopes_sessions,
        )

    def _session_summary(
        self,
        thread: Mapping[str, object],
        *,
        profile_key: str,
        endpoint: CodexEndpoint,
    ) -> SessionSummary:
        return SessionSummary(
            ref=self._make_ref(
                native_session_id=self._string(thread, "id", "thread_list"),
                profile_key=profile_key,
                endpoint=endpoint,
                cwd=self._string(thread, "cwd", "thread_list"),
            ),
            metadata=SessionMetadata(name=self._optional_string(thread.get("name"))),
        )

    @staticmethod
    def _make_ref(
        *, native_session_id: str, profile_key: str, endpoint: CodexEndpoint, cwd: str
    ) -> AgentSessionRef:
        return AgentSessionRef(
            schema_version="agent-session-ref.v1",
            backend="codex",
            transport="sdk",
            native_session_id=native_session_id,
            profile_key=profile_key,
            state_root_fingerprint=fingerprint_codex_endpoint(endpoint),
            cwd_fingerprint=fingerprint_path(cwd),
        )

    @staticmethod
    def _require_local_auth(kind: str) -> None:
        if kind != "local_account":
            raise UnsupportedCapability("Codex SDK agent sessions require local ChatGPT auth")

    @staticmethod
    def _approval_mode(policy: PermissionPolicy) -> str:
        if policy.approval == "deny":
            return "deny_all"
        if policy.approval == "provider_review":
            return "auto_review"
        raise UnsupportedCapability(
            "Codex SDK supports deny or provider_review approvals, not caller ask/allow"
        )

    @staticmethod
    def _sandbox(policy: PermissionPolicy) -> str:
        return {
            "read_only": "read-only",
            "workspace_write": "workspace-write",
            "full_access": "danger-full-access",
        }[policy.filesystem]

    @staticmethod
    def _validate_policy_mapping(policy: PermissionPolicy) -> None:
        if policy.network == "allowlist":
            raise UnsupportedCapability("Codex SDK has no typed network allowlist mapping")
        if policy.filesystem == "full_access" and policy.network != "unrestricted":
            raise UnsupportedCapability(
                "Codex full_access cannot preserve restricted network policy"
            )
        if policy.filesystem == "read_only" and policy.network != "disabled":
            # The typed network toggle belongs to sandbox_workspace_write; no
            # corresponding read-only toggle can preserve this request's policy.
            raise UnsupportedCapability(
                "Codex read_only sandbox has no network toggle; use workspace_write"
            )
        if policy.approval not in ("deny", "provider_review"):
            raise UnsupportedCapability("Codex SDK approval mode is unsupported")
        if policy.allowed_tools != ("*",) or policy.denied_tools:
            raise UnsupportedCapability(
                "Codex SDK has no typed built-in tool filters; explicitly allow '*'"
            )

    @staticmethod
    def _validate_mcp_filters(request: AgentSessionRequest) -> None:
        for server in request.mcp_servers:
            if any(
                any(marker in tool for marker in "*?[")
                for tool in (*server.allowed_tools, *server.denied_tools)
            ):
                raise UnsupportedCapability("Codex SDK MCP filters require exact tool names")

    @staticmethod
    def _validate_strict_native_containment(request: AgentSessionRequest) -> None:
        policy = request.policy
        if policy.filesystem != "read_only" or policy.network != "disabled":
            raise UnsupportedCapability(
                "Codex disabled built-ins require read-only filesystem and disabled network"
            )
        if policy.approval != "deny":
            raise UnsupportedCapability(
                "Codex disabled built-ins require unconditional approval denial"
            )
        if policy.environment:
            raise UnsupportedCapability(
                "Codex disabled built-ins require an empty copied environment"
            )
        if request.mcp_servers:
            raise UnsupportedCapability("Codex disabled built-ins do not permit MCP servers")
        if request.additional_dirs:
            raise UnsupportedCapability(
                "Codex disabled built-ins do not permit additional filesystem roots"
            )

    @staticmethod
    def _text_only(parts: tuple[object, ...], context: str) -> str:
        if any(not isinstance(part, TextContent) for part in parts):
            raise UnsupportedCapability(f"{context} supports text only")
        return "\n\n".join(part.text for part in parts if isinstance(part, TextContent))

    @staticmethod
    def _turn_request(
        *,
        thread_id: str,
        input: tuple[ContentPart, ...],
        approval_mode: str,
        reasoning: str | None,
        input_id: str,
        output: AgentOutputSpec,
        contained: bool,
    ) -> JsonObject:
        params: dict[str, object] = {
            "threadId": thread_id,
            "input": [CodexSdkAdapter._codex_input(part) for part in input],
            "approval_mode": approval_mode,
            "effort": reasoning,
            "clientUserMessageId": input_id,
        }
        if contained:
            params["environments"] = []
        if isinstance(output, JsonSchemaAgentOutput):
            params["output_schema"] = thaw_json_value(output.schema)
        return freeze_json_object(
            {"method": "turn/start", "params": CodexAppServerClient._thread_params(params)}
        )

    @staticmethod
    def _steer_params(
        thread_id: str, turn_id: str | None, input_id: str, parts: tuple[ContentPart, ...]
    ) -> dict[str, object]:
        return {
            "threadId": thread_id,
            "expectedTurnId": turn_id,
            "clientUserMessageId": input_id,
            "input": [CodexSdkAdapter._codex_input(part) for part in parts],
        }

    @staticmethod
    def _codex_input(part: object) -> object:
        if isinstance(part, TextContent):
            return {"type": "text", "text": part.text}
        if isinstance(part, ImageContent):
            # Existence, declared size, and containment under the authorized roots are
            # `AgentRuntime._validate_content_files`'s single check of every turn input, so
            # this only translates the part the SDK accepts.
            return {"type": "localImage", "path": part.path}
        raise UnsupportedCapability("Codex SDK input supports text and local images")

    def _codex_config(self, request: _ResolvedCodexSessionRequest) -> dict[str, object]:
        config: dict[str, object] = {
            "mcp_servers": {},
            "model_reasoning_effort": request.native_reasoning,
            "web_search": "disabled",
            "shell_environment_policy": {"inherit": "core", "exclude": []},
        }
        native = request.native
        if isinstance(native, CodexNativeOptions) and native.web_search is not None:
            config["web_search"] = "live" if native.web_search else "disabled"
        if isinstance(native, CodexNativeOptions) and native.builtin_tools == "disabled":
            config.update(
                {
                    "agents": {"enabled": False},
                    "apps": {"_default": {"enabled": False}},
                    "features": {name: False for name in _DISABLED_BUILTIN_FEATURES},
                    "include_apps_instructions": False,
                    "include_collaboration_mode_instructions": False,
                    "include_environment_context": False,
                    "include_permissions_instructions": False,
                    "skills": {
                        "bundled": {"enabled": False},
                        "include_instructions": False,
                    },
                    "tools": {
                        "experimental_request_user_input": {"enabled": False},
                        "update_plan": {"enabled": False},
                    },
                }
            )
        if request.policy.filesystem == "workspace_write":
            workspace: dict[str, object] = {
                "writable_roots": [request.cwd, *request.additional_dirs],
                "network_access": request.policy.network == "unrestricted",
                "exclude_slash_tmp": False,
                "exclude_tmpdir_env_var": False,
            }
            if self._sandbox_controls is not None:
                workspace.update(
                    exclude_slash_tmp=self._sandbox_controls.exclude_slash_tmp,
                    exclude_tmpdir_env_var=self._sandbox_controls.exclude_tmpdir_env_var,
                )
            config["sandbox_workspace_write"] = workspace
        if request.mcp_servers:
            servers: dict[str, object] = {}
            for server in request.mcp_servers:
                if server.transport == "stdio":
                    entry: dict[str, object] = {
                        "command": server.command,
                        "args": list(server.args),
                    }
                else:
                    entry = {"url": server.url}
                entry["required"] = server.required
                if server.allowed_tools:
                    entry["enabled_tools"] = list(server.allowed_tools)
                if server.denied_tools:
                    entry["disabled_tools"] = list(server.denied_tools)
                servers[server.name] = entry
            config["mcp_servers"] = servers
        return config

    @staticmethod
    def _count_streamed_text(state: _CodexSessionState, text: str) -> None:
        size = len(text.encode("utf-8"))
        if size > _MAX_EVENT_TEXT_BYTES:
            raise OutputLimitExceeded(_MAX_EVENT_TEXT_BYTES)
        if not state.controlled:
            if state.streamed_text_bytes + size > _MAX_FINAL_TEXT_BYTES:
                raise OutputLimitExceeded(_MAX_FINAL_TEXT_BYTES)
            state.streamed_text_bytes += size

    def _record_completed_agent_message(
        self,
        state: _CodexSessionState,
        item_id: str,
        item: Mapping[str, object],
        method: str,
    ) -> None:
        text = self._string(item, "text", method)
        phase = item.get("phase")
        if phase not in (None, "commentary", "final_answer"):
            raise ProtocolDefect("Codex completed agent message carried an unknown phase")
        if phase in (None, "final_answer") and len(text.encode("utf-8")) > _MAX_FINAL_TEXT_BYTES:
            raise OutputLimitExceeded(_MAX_FINAL_TEXT_BYTES)
        if state.controlled:
            if phase == "commentary":
                return
            state.completed_agent_messages[:] = [
                message for message in state.completed_agent_messages if message.phase != phase
            ]
        state.completed_agent_messages.append(
            _CompletedAgentMessage(
                item_id=item_id,
                text=text,
                phase=cast(AgentMessagePhase, phase),
            )
        )

    @staticmethod
    def _selected_final_text(state: _CodexSessionState, *, required: bool) -> str:
        """Select the last final answer, else the last unknown-phase message."""
        last_unknown: _CompletedAgentMessage | None = None
        for message in reversed(state.completed_agent_messages):
            if message.phase == "final_answer":
                return message.text
            if message.phase is None and last_unknown is None:
                last_unknown = message
        if last_unknown is not None:
            return last_unknown.text
        if required:
            raise ProtocolDefect("Codex turn completed without an eligible completed agent message")
        return ""

    @staticmethod
    def _append_diagnostic(state: _CodexSessionState, message: str) -> None:
        """Keep recent native diagnostics; bounded observation fails past the limit."""
        if message in state.diagnostics:
            return
        if len(state.diagnostics) >= _MAX_DIAGNOSTICS:
            if state.controlled:
                state.diagnostics.pop(0)
                state.diagnostics.append(message)
                return
            raise OutputLimitExceeded(_MAX_DIAGNOSTICS)
        state.diagnostics.append(message)

    @staticmethod
    def _patch_status(value: object, method: str) -> FileChangeStatus:
        if not isinstance(value, str) or value not in _PATCH_APPLY_STATUS:
            raise ProtocolDefect(f"{method} file change carried an impossible status")
        return _PATCH_APPLY_STATUS[value]

    @staticmethod
    def _is_quota_error(value: object) -> bool:
        if not isinstance(value, Mapping):
            return False
        info = value.get("codexErrorInfo")
        return info in ("usageLimitExceeded", "sessionBudgetExceeded")

    @staticmethod
    def _is_quota_error_text(value: str) -> bool:
        lowered = value.lower()
        return "usage limit" in lowered or "quota" in lowered or "rate limit" in lowered

    @classmethod
    def _mapping(cls, value: object, context: str) -> dict[str, object]:
        dumped = cls._dump(value)
        if not isinstance(dumped, Mapping) or any(not isinstance(key, str) for key in dumped):
            raise ProtocolDefect(f"{context} was not an object")
        return dict(dumped)

    @classmethod
    def _dump(cls, value: object) -> object:
        if isinstance(value, Mapping | list | tuple | str | int | float | bool) or value is None:
            return value
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json", by_alias=True, exclude_none=True)
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(cast(Any, value))
        root = getattr(value, "root", None)
        if root is not None:
            return cls._dump(root)
        return value

    @staticmethod
    def _string(value: Mapping[str, object], key: str, context: str) -> str:
        result = value.get(key)
        if not isinstance(result, str):
            raise ProtocolDefect(f"{context}.{key} was not a string")
        return result

    @staticmethod
    def _non_empty_string(value: Mapping[str, object], key: str, context: str) -> str:
        result = value.get(key)
        if not isinstance(result, str) or not result:
            raise ProtocolDefect(f"{context}.{key} was not a non-empty identity string")
        return result

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return value if isinstance(value, str) and value else None


@dataclass(slots=True)
class _ActiveCall:
    name: str
    arguments: JsonValue
    size: int
    reply_written: bool = False


@dataclass(slots=True)
class _PendingReply:
    call: AgentToolCall
    request_id: CodexRequestId
    attempted: bool = False


class _CodexAgentTurn:
    """One exclusive native turn; its reader never waits on host callbacks."""

    def __init__(
        self,
        adapter: CodexSdkAdapter,
        session: AgentSession,
        state: _CodexSessionState,
        submitted: JsonObject,
        request: TurnRequest,
        controls: AgentTurnControls,
        release: Callable[[], None],
        validate_input: Callable[[tuple[ContentPart, ...]], None] | None,
    ) -> None:
        if state.attempt is None or state.user_message_id is None:
            raise ProtocolDefect("prepared turn omitted its immutable attempt/input identity")
        self._adapter = adapter
        self._session = session
        self._state = state
        self._attempt = state.attempt
        state.client.buffer_attempt_id = self._attempt.attempt_id
        state.client.on_raw_queue_loss = self._raw_queue_loss
        self._submitted_request = submitted
        self._request = request
        self._controls = controls
        self._release = release
        self._validate_input = validate_input
        self._submission: AgentSubmission | None = None
        self._refusal: AgentSubmissionRejected | None = None
        self._terminal: AgentTerminal | None = None
        self._error: BaseException | None = None
        self._started = False
        self._writer_entered = False
        self._revoked = False
        self._released = False
        self._closed = False
        self._consumer_open = False
        self._input_ids = {state.user_message_id}
        self._pending_input_bytes = len(state.user_message_id.encode("utf-8"))
        self._calls: dict[str, _ActiveCall] = {}
        self._reply_tokens: dict[str, _PendingReply] = {}
        self._pending_call_bytes = 0
        self._events: asyncio.Queue[tuple[AgentEvent, int, bool] | None] = asyncio.Queue()
        self._space: asyncio.Event | None = None
        self._last_drain = 0.0
        self._reported_wait = False
        self._stalled = False
        self._read_timeout: asyncio.Timeout | None = None
        self._fence_until: float | None = None
        self._receipt_count = self._receipt_bytes = 0
        self._omitted_count = self._omitted_bytes = 0
        self._queued_bytes = 0
        self._accepted: asyncio.Future[AgentSubmission] = asyncio.get_running_loop().create_future()
        self._finished = asyncio.Event()
        self._sender: asyncio.Task[None] | None = None
        self._reader: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[AgentCloseResult] | None = None

    @property
    def attempt(self) -> AgentAttempt:
        return self._attempt

    @property
    def submission(self) -> AgentSubmission | None:
        return self._submission

    @property
    def terminal(self) -> AgentTerminal | None:
        return self._terminal

    @property
    def submitted_request(self) -> JsonObject:
        return self._submitted_request

    async def submit(self) -> AgentSubmission:
        if self._started:
            raise InvalidAgentRequest("a prepared turn can be submitted exactly once")
        self._started = True
        if self._revoked or self._closed:
            self._submission = AgentNotSubmitted(self.attempt, "revoked before submission")
            self._finish()
            self._release_slot()
            return self._submission
        self._reader = asyncio.create_task(self._read())
        self._sender = asyncio.create_task(self._send())
        try:
            return await asyncio.shield(self._accepted)
        except asyncio.CancelledError:
            self.revoke()
            await self.close()
            raise

    def _enter_writer(self) -> None:
        self._require_authority()
        self._writer_entered = True
        self._submission = AgentUncertain(self.attempt, None, "native writer entered")

    def _raw_queue_loss(self, error: CodexRawQueueLimit) -> None:
        if not isinstance(self._error, ConflictingSubmissionEvidence):
            self._error = error
        self.revoke()

    def _start_error(self, error: CodexAppServerResponseError) -> None:
        if not self._writer_entered:
            return
        self.revoke()
        limits = _stock_size_rejection(error, "turn/start")
        if limits is None:
            return
        self._refusal = AgentSubmissionRejected(self.attempt, "input_too_large", -32602, *limits)
        if self._state.turn_id is not None:
            self._error = ConflictingSubmissionEvidence()

    async def _send(self) -> None:
        params = self._submitted_request["params"]
        if not isinstance(params, Mapping):
            raise ProtocolDefect("prepared request lost its immutable params")
        try:
            response = await self._state.client.request(
                "turn/start",
                thaw_json_value(freeze_json_object(params)),
                on_write=self._enter_writer,
                on_error=self._start_error,
                timeout_seconds=self._controls.rpc_seconds,
            )
            payload = self._adapter._mapping(response, "turn/start response")
            turn = self._adapter._mapping(payload.get("turn"), "turn/start turn")
            self._accept(self._adapter._non_empty_string(turn, "id", "turn/start response"))
        except asyncio.CancelledError:
            if not self._writer_entered and not isinstance(self._submission, AgentAccepted):
                self._submission = AgentNotSubmitted(
                    self.attempt,
                    sanitize_provider_text(str(self._error))
                    if self._error is not None
                    else "writer was cancelled before entry",
                )
                if not self._accepted.done():
                    self._accepted.set_result(self._submission)
            raise
        except Exception as error:
            self.revoke()
            if self._refusal is None:
                if self._error is None:
                    self._error = error
                if not isinstance(self._submission, AgentAccepted):
                    reason = sanitize_provider_text(str(error)) or type(error).__name__
                    self._submission = (
                        AgentUncertain(self.attempt, self._turn_ref(), reason)
                        if self._writer_entered
                        else AgentNotSubmitted(self.attempt, reason)
                    )
            await self._state.client.close()

    def _accept(self, native_turn_id: str) -> None:
        if self._state.turn_id is not None and self._state.turn_id != native_turn_id:
            raise ProtocolDefect("native acknowledgment changed the prepared turn identity")
        if not self._writer_entered:
            raise ProtocolDefect("native turn evidence preceded its possible writer")
        self._state.turn_id = native_turn_id
        self._submission = AgentAccepted(
            self.attempt, AgentTurnRef(self._state.ref, native_turn_id)
        )
        if not self._accepted.done() and not isinstance(self._error, ConflictingSubmissionEvidence):
            self._accepted.set_result(self._submission)

    async def _read(self) -> None:
        try:
            async with asyncio.timeout(self._request.timeout_seconds) as timeout:
                self._read_timeout = timeout
                if self._fence_until is not None:
                    self._shorten_deadline(self._fence_until)
                loop = asyncio.get_running_loop()
                while True:
                    deadline = timeout.when()
                    if deadline is not None and loop.time() >= deadline:
                        raise TimeoutError
                    message = await self._state.client.next_message()
                    params = message.params
                    thread_id = params.get("threadId")
                    native = params.get("turnId")
                    turn = params.get("turn")
                    if native is None and isinstance(turn, Mapping):
                        native = turn.get("id")
                    # The first usage notification on resume can belong to the prior turn.
                    # It establishes only its raw counter boundary.
                    usage_replay = (
                        message.method == "thread/tokenUsage/updated"
                        and self._state.usage_accounting.turn_requires_rebase
                    )
                    if (
                        self._state.turn_id is None
                        and message.method in _TURN_SCOPED_METHODS
                        and not usage_replay
                    ):
                        if (
                            thread_id != self._state.ref.native_session_id
                            or type(native) is not str
                            or not native
                        ):
                            raise ProtocolDefect(
                                "native turn evidence omitted exact thread/turn identity"
                            )
                        self._state.turn_id = native
                        if self._refusal is not None:
                            self._error = ConflictingSubmissionEvidence()
                            self.revoke()
                    values = self._adapter._notification_events(self._state, message)
                    native_terminal = next(
                        (value for value in values if isinstance(value, AgentTerminal)), None
                    )
                    if native_terminal is not None:
                        evidence = native_terminal.evidence
                        if not isinstance(evidence, NativeTerminalEvidence) or not isinstance(
                            evidence.native_ref, AgentTurnRef
                        ):
                            raise ProtocolDefect("Codex native seal has invalid provenance")
                        self._accept(evidence.native_ref.native_turn_id)
                        self._terminal = native_terminal
                        return
                    if (
                        message.method == "turn/started"
                        or isinstance(message, CodexServerRequest)
                        and message.method == "item/tool/call"
                    ):
                        if self._state.turn_id is None:
                            raise ProtocolDefect("native acceptance omitted turn identity")
                        self._accept(self._state.turn_id)
                    for value in values:
                        await self._publish(value)
        except TimeoutError:
            if not self._writer_entered:
                self._error = TurnNotStarted("turn_timeout")
                return
            submission = self._submission
            if (
                self._refusal is not None
                or self._terminal is not None
                or isinstance(self._error, CodexRawQueueLimit)
            ):
                return
            reason = "consumer_stalled" if self._stalled else "turn_timeout"
            if not isinstance(submission, AgentAccepted | AgentUncertain):
                submission = AgentUncertain(self.attempt, self._turn_ref(), "native turn deadline")
                self._submission = submission
            self._terminal = AgentTerminal(
                status="failed",
                failure=AgentFailure("turn_timeout"),
                final_text=self._adapter._selected_final_text(self._state, required=False),
                session_ref=self._state.ref,
                evidence=LocalStopEvidence(submission, reason),
                usage=self._state.usage_accounting.finish_turn(),
                diagnostics=tuple(self._state.diagnostics),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._refusal is not None and isinstance(error, CodexConnectionUnavailable):
                # Only a completely examined prefix can certify non-acceptance.
                if self._state.turn_id is None and self._error is None:
                    self._submission = self._refusal
                return
            if self._terminal is None or isinstance(error, CodexOutputLimit):
                if self._error is None:
                    self._error = error
                if not isinstance(self._submission, AgentAccepted | AgentSubmissionRejected):
                    reason = sanitize_provider_text(str(self._error)) or type(self._error).__name__
                    self._submission = (
                        AgentUncertain(self.attempt, self._turn_ref(), reason)
                        if self._writer_entered
                        else None
                    )
        finally:
            self._read_timeout = None
            if self._writer_entered and not self._accepted.done() and self._sender is not None:
                await asyncio.gather(self._sender, return_exceptions=True)
            if not self._writer_entered:
                self.revoke()
                sender = self._sender
                if sender is not None and not sender.done():
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
                if not isinstance(self._submission, AgentNotSubmitted):
                    self._submission = AgentNotSubmitted(
                        self.attempt, "all deferred writers quiesced before entry"
                    )
            if not self._accepted.done() and isinstance(self._error, ConflictingSubmissionEvidence):
                self._accepted.set_exception(self._error)
            if not self._accepted.done():
                if self._submission is None:
                    self._submission = AgentNotSubmitted(self.attempt, "writer did not enter")
                self._accepted.set_result(self._submission)
            self._finish()

    async def _publish(self, event: AgentEvent) -> None:
        if isinstance(event, AgentInputRecorded) and not self.accepts_input(event.input_id):
            raise ProtocolDefect("native user item has no prepared input delivery")
        try:
            size = bounded_payload_size(event, _MAX_MESSAGE_BYTES, max_items=_MAX_MESSAGE_ITEMS)
        except OutputLimitExceeded as error:
            raise CodexOutputLimit(
                _buffer_observation(
                    "converted_event",
                    "limit",
                    error.unit,
                    0,
                    error.incoming,
                    error.limit,
                    attempt_id=self.attempt.attempt_id,
                )
            ) from None
        if not self._revoked and self._queued_bytes + size > _MAX_MESSAGE_BYTES:
            loop = asyncio.get_running_loop()
            started = self._last_drain = loop.time()
            held = self._queued_bytes
            self._space = asyncio.Event()
            self._state.client.buffer_decoder_waiting = True
            try:
                while not self._revoked and self._queued_bytes + size > _MAX_MESSAGE_BYTES:
                    self._space.clear()
                    deadline = self._last_drain + self._controls.rpc_seconds
                    try:
                        async with asyncio.timeout_at(deadline):
                            await self._space.wait()
                    except TimeoutError:
                        if self._revoked:
                            break
                        if self._last_drain + self._controls.rpc_seconds <= loop.time():
                            _buffer_observation(
                                "converted_fifo",
                                "stalled",
                                "bytes",
                                held,
                                size,
                                _MAX_MESSAGE_BYTES,
                                attempt_id=self.attempt.attempt_id,
                                wait_ms=(loop.time() - started) * 1000,
                            )
                            self._stalled = True
                            self.revoke()
                if not self._revoked and not self._reported_wait:
                    _buffer_observation(
                        "converted_fifo",
                        "drained",
                        "bytes",
                        held,
                        size,
                        _MAX_MESSAGE_BYTES,
                        attempt_id=self.attempt.attempt_id,
                        wait_ms=(loop.time() - started) * 1000,
                    )
                    self._reported_wait = True
            finally:
                self._space = None
                self._state.client.buffer_decoder_waiting = False
        if self._revoked and isinstance(event, AgentToolCall):
            return
        reserved = self._queued_bytes + size > _MAX_MESSAGE_BYTES
        if reserved:
            if not isinstance(event, AgentInputRecorded):
                self._omitted_count += 1
                self._omitted_bytes += size
                return
            if self._receipt_count >= _MAX_MESSAGE_ITEMS:
                held, incoming, limit, unit = self._receipt_count, 1, _MAX_MESSAGE_ITEMS, "records"
            else:
                held, incoming, limit, unit = self._receipt_bytes, size, _MAX_MESSAGE_BYTES, "bytes"
            if held + incoming > limit:
                raise ProtocolDefect(
                    _buffer_observation(
                        "fenced_input_receipts",
                        "lineage_unresolved",
                        unit,
                        held,
                        incoming,
                        limit,
                        attempt_id=self.attempt.attempt_id,
                    )
                )
        if isinstance(event, AgentInputRecorded):
            # Receipt admission and input retirement are one operation. A full
            # reserve leaves the delivery unresolved, including its original id.
            self._input_ids.remove(event.input_id)
            self._pending_input_bytes -= len(event.input_id.encode("utf-8"))
        if reserved:
            self._receipt_count += 1
            self._receipt_bytes += size
        else:
            self._queued_bytes += size
        self._events.put_nowait((event, size, reserved))

    async def check_submission(self, *, for_effect: bool = False) -> None:
        if self._terminal is not None and self._sender is not None:
            await asyncio.shield(asyncio.gather(self._sender, return_exceptions=True))
        if self._refusal is not None and self._reader is not None:
            await asyncio.shield(asyncio.gather(self._reader, return_exceptions=True))
        if isinstance(self._error, ConflictingSubmissionEvidence | CodexRawQueueLimit):
            raise self._error
        if for_effect:
            self._require_authority()

    async def events(self) -> AsyncGenerator[AgentEvent, None]:
        if self._consumer_open:
            raise InvalidAgentRequest("a controlled turn has one event consumer")
        self._consumer_open = True
        while (item := await self._events.get()) is not None:
            value, size, reserved = item
            if reserved:
                self._receipt_count -= 1
                self._receipt_bytes -= size
            else:
                self._queued_bytes -= size
                if size and self._space is not None:
                    self._last_drain = asyncio.get_running_loop().time()
                    self._space.set()
            if not self._revoked or not isinstance(value, AgentToolCall):
                yield value
        await self.check_submission()
        if isinstance(self._error, CodexOutputLimit):
            raise self._error
        if self._terminal is not None:
            yield self._terminal
        elif self._error is not None:
            raise self._error

    def accepts_input(self, input_id: object) -> bool:
        return type(input_id) is str and input_id in self._input_ids

    def tool_started(self, call_id: str, name: str, arguments: object) -> None:
        if name not in {tool.name for tool in self._state.request.tools}:
            raise ProtocolDefect("native callback named an undeclared tool")
        if call_id in self._calls:
            raise ProtocolDefect("native tool item started more than once")
        frozen = freeze_json_value(arguments)
        size = len(canonical_json_bytes(frozen))
        if (
            len(self._calls) >= self._controls.pending_calls
            or self._pending_call_bytes + size > self._controls.pending_call_bytes
        ):
            raise ProtocolDefect("native pending callback buffer exceeded its finite bound")
        self._calls[call_id] = _ActiveCall(name, frozen, size)
        self._pending_call_bytes += size

    def tool_call(self, request: CodexServerRequest) -> AgentToolCall:
        params = request.params
        call_id = self._adapter._non_empty_string(params, "callId", request.method)
        tool = self._adapter._non_empty_string(params, "tool", request.method)
        name = self._adapter._dynamic_tool_name(params, tool)
        original = self._calls.get(call_id)
        if original is None:
            raise ProtocolDefect("native callback preceded its declared tool item")
        arguments = freeze_json_value(params.get("arguments"))
        if (original.name, original.arguments) != (name, arguments):
            raise ProtocolDefect("native callback changed its item name or arguments")
        turn = self._turn_ref()
        if turn is None:
            raise ProtocolDefect("native callback omitted its owned turn")
        if len(self._reply_tokens) >= self._controls.pending_calls:
            raise ProtocolDefect("native pending callback requests exceeded their finite bound")
        token = str(uuid4())
        call = AgentToolCall(turn, call_id, token, name, original.arguments)
        self._reply_tokens[token] = _PendingReply(call, request.request_id)
        return call

    def tool_completed(self, call_id: str, status: object) -> None:
        if status not in ("completed", "failed"):
            raise ProtocolDefect("native dynamic item completed with an impossible status")
        original = self._calls.pop(call_id, None)
        if original is None:
            raise ProtocolDefect("native callback item completed without its original proposal")
        tokens = [
            token
            for token, pending in self._reply_tokens.items()
            if pending.call.call_id == call_id
        ]
        if status == "completed" and not original.reply_written:
            raise ProtocolDefect("native callback succeeded without an owned delivered reply")
        self._pending_call_bytes -= original.size
        for token in tokens:
            pending = self._reply_tokens.pop(token)
            identity = (type(pending.request_id), pending.request_id)
            self._state.server_request_ids.discard(identity)
            self._state.client.complete_callback(pending.request_id)

    async def reply(self, call: AgentToolCall, result: AgentToolReply) -> None:
        self._require_authority()
        pending = self._reply_tokens.get(call.reply_token)
        if pending is None or pending.call != call or pending.attempted:
            raise ProtocolDefect("native reply does not identify a fresh exact owned callback")
        if not isinstance(result, AgentToolReply):
            raise InvalidAgentRequest("reply requires AgentToolReply")
        pending.attempted = True

        def enter() -> None:
            self._require_authority()
            original = self._calls.get(call.call_id)
            if original is None:
                raise ProtocolDefect("native callback reply outlived its owned item")
            original.reply_written = True

        async with asyncio.timeout(self._controls.rpc_seconds):
            await self._state.client.respond(
                pending.request_id,
                {
                    "contentItems": [{"type": "inputText", "text": result.text}],
                    "success": result.success,
                },
                on_write=enter,
            )
        self._state.server_request_ids.discard((type(pending.request_id), pending.request_id))
        self._reply_tokens.pop(call.reply_token, None)

    async def steer(self, *, input_id: str, input: tuple[ContentPart, ...]) -> AgentControlReceipt:
        validated = TurnRequest(input=input)
        if type(input_id) is not str or not input_id or input_id in self._input_ids:
            raise InvalidAgentRequest("steering requires a fresh stable input delivery id")
        if self._validate_input is not None:
            self._validate_input(validated.input)
        params = self._adapter._steer_params(
            self._state.ref.native_session_id, self._state.turn_id, input_id, validated.input
        )
        return await self._control("steer", input_id, params)

    async def interrupt(self) -> AgentControlReceipt:
        self.revoke()
        return await self._control(
            "interrupt",
            None,
            {"threadId": self._state.ref.native_session_id, "turnId": self._state.turn_id},
        )

    async def _control(
        self,
        operation: Literal["steer", "interrupt"],
        input_id: str | None,
        params: Mapping[str, object],
    ) -> AgentControlReceipt:
        request_id = str(uuid4())
        turn = self._turn_ref()
        if (
            turn is None
            or self._closed
            or (
                self._finished.is_set()
                and (
                    operation == "steer"
                    or self._terminal is not None
                    and isinstance(self._terminal.evidence, NativeTerminalEvidence)
                )
            )
            or (operation == "steer" and self._revoked)
        ):
            return AgentControlReceipt(
                operation,
                request_id,
                turn,
                "not_sent",
                input_id,
                "turn has no live control authority",
            )
        entered = False

        def enter() -> None:
            nonlocal entered
            if operation == "steer":
                self._require_authority()
                if input_id is None or input_id in self._input_ids:
                    raise InvalidAgentRequest("steering input id already entered its native writer")
                size = len(input_id.encode("utf-8"))
                if (
                    len(self._input_ids) >= _MAX_MESSAGE_ITEMS
                    or self._pending_input_bytes + size > _MAX_MESSAGE_BYTES
                ):
                    raise ProtocolDefect(
                        "native pending input identities exceeded their finite bound"
                    )
                self._input_ids.add(input_id)
                self._pending_input_bytes += size
            elif self._closed or self._terminal is not None:
                raise SessionUnavailable("native interrupt target already ended")
            entered = True

        try:
            response = await self._state.client.request(
                f"turn/{operation}",
                params,
                on_write=enter,
                timeout_seconds=self._controls.rpc_seconds,
            )
            payload = self._adapter._mapping(response, f"turn/{operation} response")
            if operation == "steer" and payload.get("turnId") != turn.native_turn_id:
                raise ProtocolDefect("native steering acknowledgment changed the expected turn")
            if operation == "interrupt" and payload:
                raise ProtocolDefect("native interrupt acknowledgment was not empty")
            return AgentControlReceipt(
                operation,
                request_id,
                turn,
                "accepted",
                input_id,
                "rpc accepted; native termination/input incorporation is separate",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            reason = sanitize_provider_text(str(error)) or type(error).__name__
            limits = _stock_size_rejection(error, "turn/steer")
            if limits is not None and operation == "steer":
                fit = codex_wire_request_fit({"method": "turn/steer", "params": params})
                reason = (
                    f"input_too_large: fitted {fit.actual_chars}/{fit.max_chars} characters, "
                    f"stock {limits[0]}/{limits[1]}"
                )
                logging.getLogger(__name__).error("turn/steer %s", reason)
            disposition = "unknown" if entered else "not_sent"
            if isinstance(error, CodexAppServerResponseError) and error.code in (
                -32600,
                -32601,
                -32602,
            ):
                disposition = "rejected"
            return AgentControlReceipt(operation, request_id, turn, disposition, input_id, reason)

    def _turn_ref(self) -> AgentTurnRef | None:
        submission = self._submission
        if isinstance(submission, AgentAccepted):
            return submission.turn
        if isinstance(submission, AgentUncertain) and submission.turn is not None:
            return submission.turn
        if self._closed or isinstance(submission, AgentNotSubmitted):
            return None
        return (
            None
            if self._state.turn_id is None
            else AgentTurnRef(self._state.ref, self._state.turn_id)
        )

    def _require_authority(self) -> None:
        if self._revoked or self._closed or self._finished.is_set():
            raise SessionUnavailable("controlled turn authority has been revoked or ended")

    def _shorten_deadline(self, deadline: float) -> None:
        timer = self._read_timeout
        if timer is not None and not timer.expired():
            current = timer.when()
            if current is None or deadline < current:
                timer.reschedule(deadline)

    def revoke(self) -> None:
        if self._revoked:
            return
        self._revoked = True
        self._fence_until = asyncio.get_running_loop().time() + self._controls.rpc_seconds
        self._shorten_deadline(self._fence_until)
        if self._space is not None:
            self._space.set()

    def _finish(self) -> None:
        if not self._finished.is_set():
            self._finished.set()
            self._events.put_nowait(None)
            unresolved = len(self._input_ids) if self._state.controlled and self._revoked else 0
            if self._omitted_count or unresolved:
                logging.getLogger(__name__).info(
                    "Codex fenced turn attempt=%r omitted_events=%d omitted_bytes=%d unresolved_inputs=%d",
                    sanitize_provider_text(self.attempt.attempt_id, limit=96),
                    self._omitted_count,
                    self._omitted_bytes,
                    unresolved,
                )

    def _release_slot(self) -> None:
        if not self._released:
            self._released = True
            self._state.client.buffer_attempt_id = None
            self._state.client.on_raw_queue_loss = None
            self._release()

    async def close(self) -> AgentCloseResult:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        return await asyncio.shield(self._close_task)

    async def _close(self) -> AgentCloseResult:
        self.revoke()
        diagnostics: list[str] = []
        assert self._fence_until is not None
        deadline = min(self._fence_until, asyncio.get_running_loop().time() + 2)
        self._shorten_deadline(deadline)
        sender = self._sender
        if sender is not None and not sender.done():
            if self._writer_entered:
                # A correlated refusal can still qualify or contradict a raw seal.
                try:
                    async with asyncio.timeout_at(deadline):
                        await asyncio.shield(sender)
                except TimeoutError:
                    pass
            if not sender.done():
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
        reader = self._reader
        if reader is not None and not reader.done():
            try:
                async with asyncio.timeout_at(deadline):
                    await asyncio.shield(reader)
            except TimeoutError:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                diagnostics.append("native reader did not reach a terminal before local close")
        needs_discard = (
            not self._state.client.usable
            or bool(
                self._state.started_item_types
                or self._state.active_tool_calls
                or self._state.server_request_ids
            )
            or self._writer_entered
            and (self._terminal is None or isinstance(self._terminal.evidence, LocalStopEvidence))
        )
        if needs_discard:
            if (
                self._state.client.usable
                and self._state.turn_id is not None
                and (
                    self._terminal is None or isinstance(self._terminal.evidence, LocalStopEvidence)
                )
            ):
                try:
                    async with asyncio.timeout(min(2.0, self._controls.rpc_seconds)):
                        receipt = await self.interrupt()
                    diagnostics.append(
                        f"native interrupt rpc {receipt.disposition}; remote finality remains separate"
                    )
                except Exception as error:
                    diagnostics.append(sanitize_provider_text(str(error)) or type(error).__name__)
            try:
                async with asyncio.timeout(min(2.0, self._controls.rpc_seconds)):
                    await self._adapter.close_session(self._session)
            except Exception as error:
                diagnostics.append(sanitize_provider_text(str(error)) or type(error).__name__)
        if not self._writer_entered and not isinstance(
            self._submission, AgentAccepted | AgentNotSubmitted
        ):
            self._submission = AgentNotSubmitted(
                self.attempt, "all deferred writers quiesced before entry"
            )
        self._closed = True
        self._state.usage_accounting.abandon_turn()
        self._finish()
        self._release_slot()
        return AgentCloseResult(True, tuple(diagnostics[:_MAX_DIAGNOSTICS]))


__all__ = ["CodexSdkAdapter"]
