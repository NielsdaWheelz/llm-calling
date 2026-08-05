"""Native Codex Python SDK adapter with a lazy optional dependency boundary."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import os
import re
import sys
import weakref
from collections.abc import AsyncGenerator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

from provider_runtime.errors import sanitize_provider_text
from provider_runtime.schema import to_json_schema

from ._codex_launcher import ensure_codex_launcher
from ._limits import OutputLimitExceeded, bounded_payload_size
from ._sandbox import bubblewrap_network_namespace_available
from ._structured_output import OutputSchemaMismatch, parse_structured_output
from .auth import (
    freeze_native_json_object as freeze_json_object,
)
from .auth import (
    freeze_native_json_value as freeze_json_value,
)
from .auth import (
    mcp_header_environment_name,
    redact_native_payload,
    state_root_from_environment,
)
from .capabilities import AgentCapabilities, AgentCapabilityScope, validate_mcp_network_policy
from .errors import (
    CredentialRejected,
    CredentialUnavailable,
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
    DiagnosticData,
    FileChangeData,
    NativeRetryObservedData,
    ReasoningData,
    SessionStartedData,
    TextDeltaData,
    ToolCompletedData,
    ToolStartedData,
    ToolUpdatedData,
    TurnCancelledData,
    TurnCompletedData,
    TurnFailedData,
    TurnStartedData,
    UnknownData,
    UsageData,
)
from .policy import PermissionPolicy
from .sessions import (
    AgentSession,
    CodexSessionFilters,
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
from .types import (
    AgentOutputSpec,
    AgentSessionRef,
    AgentSessionRequest,
    ApprovalHandler,
    CodexNativeOptions,
    CredentialRef,
    ForkSession,
    FrozenJsonDict,
    ImageContent,
    JsonObject,
    JsonSchemaAgentOutput,
    NewSession,
    ResumeSession,
    TextContent,
    TurnRequest,
)

_MCP_REFERENCE_FORMS = ("environment_reference", "header_reference")

_SUPPORTED_SDK_VERSION = "0.144.4"
_SUPPORTED_RUNTIME_VERSION = "0.144.4"
_RUNTIME_VERSION_PREFIX = re.compile(
    r"(?P<version>[0-9]+(?:\.[0-9]+){2}(?:[-+][0-9A-Za-z.-]+)?)(?:\s|$)"
)
_OPERATION_TIMEOUT_SECONDS = 30.0
_MAX_EVENT_COUNT = 100_000
_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_MAX_MESSAGE_ITEMS = 100_000
_MAX_TURN_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_EVENT_TEXT_BYTES = 4 * 1024 * 1024
_MAX_FINAL_TEXT_BYTES = 16 * 1024 * 1024
_MAX_DIAGNOSTICS = 256
_REQUIRED_MCP_STARTUP_FAILURE = "required MCP servers failed to initialize"

_KNOWN_FIELDS: dict[str, tuple[str, ...]] = {
    "thread/started": ("thread",),
    "turn/started": ("threadId", "turn"),
    "turn/completed": ("threadId", "turn"),
    "item/started": ("threadId", "turnId", "item"),
    "item/completed": ("threadId", "turnId", "item"),
    "item/agentMessage/delta": ("threadId", "turnId", "itemId", "delta"),
    "item/reasoning/summaryTextDelta": (
        "threadId",
        "turnId",
        "itemId",
        "summaryIndex",
        "delta",
    ),
    "item/reasoning/textDelta": ("threadId", "turnId", "itemId", "contentIndex", "delta"),
    "item/commandExecution/outputDelta": ("threadId", "turnId", "itemId", "delta"),
    "item/fileChange/patchUpdated": ("threadId", "turnId", "itemId", "changes"),
    "item/fileChange/outputDelta": ("threadId", "turnId", "itemId", "delta"),
    "item/mcpToolCall/progress": ("threadId", "turnId", "itemId", "message"),
    "mcpServer/startupStatus/updated": ("name", "status", "error", "failureReason", "threadId"),
    "thread/tokenUsage/updated": ("threadId", "turnId", "tokenUsage"),
    "error": ("threadId", "turnId", "willRetry", "error"),
    "configWarning": ("summary", "details", "path", "range"),
    "warning": ("message", "threadId"),
    "account/rateLimits/updated": ("rateLimits",),
}

_TURN_SCOPED_METHODS = frozenset(
    {
        "error",
        "turn/started",
        "turn/completed",
        "thread/tokenUsage/updated",
        "item/started",
        "item/completed",
        "item/agentMessage/delta",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/textDelta",
        "item/commandExecution/outputDelta",
        "item/fileChange/patchUpdated",
        "item/fileChange/outputDelta",
        "item/mcpToolCall/progress",
    }
)

type FileChangeStatus = Literal["in_progress", "applied", "failed", "declined"]
_PATCH_APPLY_STATUS: dict[str, FileChangeStatus] = {
    "inProgress": "in_progress",
    "completed": "applied",
    "failed": "failed",
    "declined": "declined",
}


@dataclass(slots=True)
class _CodexSessionState:
    sdk: ModuleType
    client: Any
    thread: Any
    request: AgentSessionRequest
    ref: AgentSessionRef
    sdk_version: str
    executable_version: str
    session_started_emitted: bool = False
    turn: Any | None = None
    turn_id: str | None = None
    seq: int = 0
    message_count: int = 0
    output_bytes: int = 0
    final_text: str = ""
    final_text_bytes: int = 0
    usage: FrozenJsonDict | None = None
    current_output: AgentOutputSpec | None = None
    diagnostics: list[str] = field(default_factory=list)
    retry_count: int = 0
    active_mcp_calls: dict[str, tuple[str, str]] = field(default_factory=dict)
    quota_exhausted: bool = False


class CodexSdkAdapter:
    """Own official Codex SDK clients and normalize their public notification stream."""

    backend: Literal["codex"] = "codex"
    transport: Literal["sdk"] = "sdk"

    def __init__(self) -> None:
        self._sessions: dict[AgentSession, _CodexSessionState] = {}
        self._dead_sessions: weakref.WeakSet[AgentSession] = weakref.WeakSet()
        self._clients: set[Any] = set()

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
        sdk, client = await self._open_client(cwd=None, environment=environment)
        try:
            await self._verify_auth(client)
            models, efforts, model_efforts = self._decode_models(
                await self._call(client.models(), operation="model discovery", failure="executable")
            )
            executable_version = self._executable_version(client)
            state_root = state_root_from_environment("codex", environment)
            restricted_write = await bubblewrap_network_namespace_available(
                cwd=state_root, environment=environment
            )
            return AgentCapabilities(
                scope=scope,
                executable_version=executable_version,
                sdk_version=self._sdk_version(sdk),
                session_operations=("new", "resume", "fork"),
                discovery_operations=("list", "read"),
                models=models,
                reasoning_efforts=efforts,
                model_reasoning_efforts=model_efforts,
                reasoning_summaries=("none", "auto", "concise", "detailed"),
                content_kinds=("text", "image"),
                attachment_kinds=("image",),
                session_instruction_roles=("developer",),
                turn_instruction_roles=(),
                streaming=True,
                cancellation=True,
                structured_output=True,
                native_output_schema=True,
                builtin_tool_families=("file_read", "file_write", "command", "web_search"),
                tool_controls=False,
                mcp_transports=("stdio", "streamable_http"),
                mcp_auth_forms=_MCP_REFERENCE_FORMS,
                filesystem_modes=(
                    ("read_only", "workspace_write", "full_access")
                    if restricted_write
                    else ("read_only", "full_access")
                ),
                network_modes=("disabled", "unrestricted"),
                approval_modes=("deny", "provider_review"),
                additional_dirs=True,
                max_turns=False,
                timeouts=True,
                turn_overrides=(),
                persistent_turn_overrides=(),
                native_extension_version="openai-codex.v1",
                native_option_names=("web_search",),
                cwd_scopes_sessions=False,
                reports_auth_identity=False,
                reports_effective_effort=False,
                session_name_metadata=True,
                session_archive_metadata=False,
                session_tag_metadata=False,
            )
        finally:
            await self._close_client(client)

    async def list_sessions(
        self,
        query: SessionQuery,
        *,
        environment: Mapping[str, str],
    ) -> SessionPage:
        self._require_scope(query.scope)
        self._require_local_auth(query.scope.auth.kind)
        _sdk, client = await self._open_client(cwd=None, environment=environment)
        try:
            await self._verify_auth(client)
            archived = (
                query.native.archived if isinstance(query.native, CodexSessionFilters) else None
            )
            response = await self._call(
                client.thread_list(archived=archived, cursor=query.cursor, limit=query.limit),
                operation="thread listing",
                failure="executable",
            )
            payload = self._mapping(response, "Codex SDK thread_list response")
            data = payload.get("data")
            if not isinstance(data, list):
                raise ProtocolDefect("Codex SDK thread_list data was not an array")
            state_root = state_root_from_environment("codex", environment)
            sessions = tuple(
                self._session_summary(
                    self._mapping(item, "Codex SDK thread_list thread"),
                    profile_key=query.scope.auth.profile_key,
                    state_root=state_root,
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
        if ref.state_root_fingerprint != fingerprint_path(
            state_root_from_environment("codex", environment)
        ):
            raise SessionMismatch("session state root does not match the supplied environment")
        if options.cursor is not None or options.include_turns or options.include_items:
            raise UnsupportedCapability("Codex SDK adapter currently supports metadata-only reads")
        _sdk, client = await self._open_client(cwd=None, environment=environment)
        try:
            await self._verify_auth(client)
            thread = await self._call(
                client.thread_resume(ref.native_session_id),
                operation="thread resume for read",
                failure="session",
            )
            response = await self._call(
                thread.read(include_turns=False), operation="thread read", failure="session"
            )
            payload = self._mapping(response, "Codex SDK thread read response")
            native_thread = self._mapping(payload.get("thread"), "Codex SDK thread read thread")
            if native_thread.get("id") != ref.native_session_id:
                raise ProtocolDefect("Codex SDK thread read changed the native identity")
            return SessionSnapshot(
                ref=ref,
                metadata=SessionMetadata(name=self._optional_string(native_thread.get("name"))),
                items=(),
                continuation_cursor=None,
            )
        finally:
            await self._close_client(client)

    async def open_session(
        self,
        request: AgentSessionRequest,
        *,
        capabilities: AgentCapabilities,
        environment: Mapping[str, str],
    ) -> AgentSession:
        if request.backend != self.backend or request.transport != self.transport:
            raise InvalidAgentRequest("CodexSdkAdapter received a different route")
        self._require_scope(capabilities.scope)
        self._require_local_auth(request.auth.kind)
        self._validate_policy_mapping(request.policy)
        validate_mcp_network_policy(request.mcp_servers, request.policy)
        self._validate_mcp_filters(request)
        if (
            isinstance(request.native, CodexNativeOptions)
            and request.native.web_search is True
            and request.policy.network != "unrestricted"
        ):
            raise UnsupportedCapability("Codex web search requires unrestricted network policy")

        sdk, client = await self._open_client(cwd=request.cwd, environment=environment)
        try:
            await self._verify_auth(client)
            kwargs: dict[str, object] = {
                "approval_mode": self._approval_mode(sdk, request.policy),
                "config": self._codex_config(request),
                "cwd": request.cwd,
                "sandbox": self._sandbox(sdk, request.policy),
            }
            if request.model is not None:
                kwargs["model"] = request.model
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

            ref = self._make_ref(
                native_session_id=native_session_id,
                profile_key=request.auth.profile_key,
                state_root=state_root_from_environment("codex", environment),
                cwd=request.cwd,
            )
            session = AgentSession(ref)
            self._sessions[session] = _CodexSessionState(
                sdk=sdk,
                client=client,
                thread=thread,
                request=request,
                ref=ref,
                sdk_version=self._sdk_version(sdk),
                executable_version=self._executable_version(client),
            )
            return session
        except BaseException:
            await self._close_client(client)
            raise

    async def stream_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None,
    ) -> AsyncGenerator[AgentEvent, None]:
        state = self._state(session)
        if approvals is not None:
            raise UnsupportedCapability("Codex SDK does not expose caller approval callbacks")
        if any(
            value is not None
            for value in (request.model, request.reasoning, request.policy, request.output)
        ):
            raise UnsupportedCapability("Codex SDK turn overrides persist and are disabled")
        if request.system is not None or request.developer is not None:
            raise UnsupportedCapability("Codex SDK instructions are session-scoped")
        if request.mcp_servers is not None:
            raise UnsupportedCapability("Codex SDK MCP configuration is session-scoped")
        if request.max_turns is not None:
            raise UnsupportedCapability("Codex SDK does not support max_turns")
        if request.native is not None:
            raise UnsupportedCapability("Codex SDK native options are session-scoped")

        policy = state.request.policy
        inputs = [
            self._codex_input(state.sdk, part, state.request, policy) for part in request.input
        ]
        kwargs: dict[str, object] = {
            "approval_mode": self._approval_mode(state.sdk, policy),
        }
        reasoning = state.request.reasoning
        if reasoning is not None:
            kwargs["effort"] = reasoning.effort
            if reasoning.summary is not None:
                kwargs["summary"] = reasoning.summary
        output = state.request.output
        if isinstance(output, JsonSchemaAgentOutput):
            kwargs["output_schema"] = to_json_schema(
                output.schema, inline_defs=False, include_annotations=True
            )

        turn = await self._call(
            state.thread.turn(inputs, **kwargs), operation="turn start", failure="session"
        )
        turn_id = getattr(turn, "id", None)
        if not isinstance(turn_id, str) or not turn_id:
            await self._destroy_session(session, state)
            raise ProtocolDefect("Codex SDK returned no turn id")
        state.turn = turn
        state.turn_id = turn_id
        state.seq = 0
        state.message_count = 0
        state.output_bytes = 0
        state.final_text = ""
        state.final_text_bytes = 0
        state.usage = None
        state.current_output = state.request.output
        state.diagnostics.clear()
        state.retry_count = 0
        state.active_mcp_calls.clear()
        state.quota_exhausted = False

        if not state.session_started_emitted:
            state.session_started_emitted = True
            yield self._event(
                state,
                "session_started",
                SessionStartedData(),
                native_type="thread/start",
                native_payload=redact_native_payload(
                    {
                        "id": state.ref.native_session_id,
                        "cwd": state.request.cwd,
                        "model": state.request.model,
                        "sdkVersion": state.sdk_version,
                        "cliVersion": state.executable_version,
                        "approvalMode": state.request.policy.approval,
                        "sandbox": state.request.policy.filesystem,
                    },
                    allowed_fields=(
                        "id",
                        "cwd",
                        "model",
                        "sdkVersion",
                        "cliVersion",
                        "approvalMode",
                        "sandbox",
                    ),
                ),
            )

        terminal_seen = False
        try:
            stream = turn.stream()
            async for notification in stream:
                for event in self._notification_events(state, notification):
                    yield event
                    if event.kind in ("turn_completed", "turn_failed", "turn_cancelled"):
                        terminal_seen = True
                        state.turn = None
                        return
        except OutputLimitExceeded:
            try:
                await turn.interrupt()
            finally:
                await self._destroy_session(session, state)
            yield self._event(
                state,
                "turn_failed",
                TurnFailedData(
                    failure="output_limit_exceeded",
                    final_text=state.final_text,
                    usage=state.usage,
                    diagnostics=tuple(state.diagnostics),
                ),
            )
            return
        except ProtocolDefect:
            await self._destroy_session(session, state)
            raise
        except (TypeError, ValueError, RecursionError):
            await self._destroy_session(session, state)
            raise ProtocolDefect("Codex SDK emitted an invalid event payload") from None
        except Exception as error:
            message = sanitize_provider_text(str(error)) or "Codex SDK turn failed"
            self._append_diagnostic(state, message)
            await self._destroy_session(session, state)
            yield self._event(
                state,
                "turn_failed",
                TurnFailedData(
                    failure="quota_exhausted"
                    if self._is_quota_error_text(message)
                    else "backend_failed",
                    final_text=state.final_text,
                    usage=state.usage,
                    diagnostics=tuple(state.diagnostics),
                ),
            )
            return
        if not terminal_seen:
            await self._destroy_session(session, state)
            raise MissingTerminalEvent()

    async def interrupt(self, session: AgentSession, turn_id: str | None) -> None:
        if session in self._dead_sessions:
            return
        state = self._state(session)
        if turn_id is None:
            await self._destroy_session(session, state)
            return
        if state.turn_id != turn_id or state.turn is None:
            raise InvalidAgentRequest("interrupt turn id does not match the active Codex turn")
        await self._call(state.turn.interrupt(), operation="turn interrupt", failure="session")

    async def close_session(self, session: AgentSession) -> None:
        """Idempotently release one SDK client while preserving sibling sessions."""
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
                "Codex SDK client teardown did not complete", code="sdk_teardown_failed"
            )

    async def _open_client(
        self, *, cwd: str | None, environment: Mapping[str, str]
    ) -> tuple[ModuleType, Any]:
        sdk = self._load_sdk()
        sdk_version = self._sdk_version(sdk)
        if sdk_version != _SUPPORTED_SDK_VERSION:
            raise SdkUnavailable(
                f"Codex SDK {_SUPPORTED_SDK_VERSION} is required; found {sdk_version}"
            )
        runtime = self._load_runtime_package()
        runtime_version = self._runtime_version(runtime)
        if runtime_version != _SUPPORTED_RUNTIME_VERSION:
            raise SdkUnavailable(
                f"Codex bundled runtime {_SUPPORTED_RUNTIME_VERSION} is required; "
                f"found {runtime_version}"
            )
        try:
            child_environment = dict(environment)
            bundled_path_dir = runtime.bundled_path_dir()
            if bundled_path_dir is not None:
                path_dir = Path(bundled_path_dir).resolve(strict=True)
                if not path_dir.is_dir():
                    raise ExecutableUnavailable("the bundled Codex PATH directory is invalid")
                existing_path = child_environment.get("PATH", "")
                entries = tuple(entry for entry in existing_path.split(os.pathsep) if entry)
                child_environment["PATH"] = os.pathsep.join(
                    (str(path_dir), *(entry for entry in entries if entry != str(path_dir)))
                )
            bundled_executable = Path(runtime.bundled_codex_path()).resolve(strict=True)
            state_root = state_root_from_environment("codex", environment)
            launcher = ensure_codex_launcher(
                state_root,
                bundled_executable,
                tuple(child_environment),
                interpreter=sys.executable,
            )
            config = sdk.CodexConfig(
                codex_bin=str(launcher),
                cwd=cwd,
                env=child_environment,
                client_name="provider_runtime",
                client_title="provider-runtime",
                client_version="0.1.0",
            )
            client = sdk.AsyncCodex(config)
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await client.__aenter__()
            executable_version = self._executable_version(client)
            if executable_version != _SUPPORTED_RUNTIME_VERSION:
                await client.close()
                raise SdkUnavailable(
                    f"Codex runtime {_SUPPORTED_RUNTIME_VERSION} is required; "
                    f"found {executable_version}"
                )
        except TimeoutError:
            raise ExecutableUnavailable("Codex SDK initialization timed out") from None
        except (SdkUnavailable, CredentialUnavailable):
            raise
        except Exception as error:
            raise ExecutableUnavailable(
                f"Codex SDK initialization failed: {sanitize_provider_text(str(error))}"
            ) from None
        self._clients.add(client)
        return sdk, client

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
        method = getattr(notification, "method", None)
        if not isinstance(method, str) or not method:
            raise ProtocolDefect("Codex SDK notification had no method")
        payload = getattr(notification, "payload", None)
        if payload is None:
            raise ProtocolDefect("Codex SDK notification had no payload")
        raw_params = getattr(payload, "params", payload)
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
        state.message_count += 1
        if state.message_count > _MAX_EVENT_COUNT:
            raise OutputLimitExceeded(_MAX_EVENT_COUNT)
        if state.output_bytes + size > _MAX_TURN_OUTPUT_BYTES:
            raise OutputLimitExceeded(_MAX_TURN_OUTPUT_BYTES)
        state.output_bytes += size
        params = self._mapping(raw_params, f"Codex SDK {method} notification")
        self._validate_notification_identity(state, method, params)
        native = redact_native_payload(params, allowed_fields=_KNOWN_FIELDS.get(method))
        if method in ("item/started", "item/completed"):
            item = self._mapping(params.get("item"), f"{method} item")
            if item.get("type") == "fileChange":
                return self._file_change_events(
                    state,
                    item.get("changes"),
                    self._patch_status(item.get("status"), method),
                    method,
                    native,
                )
        if method == "item/fileChange/patchUpdated":
            return self._file_change_events(
                state, params.get("changes"), "in_progress", method, native
            )
        event = self._notification_event(state, method, params, native)
        return () if event is None else (event,)

    def _notification_event(
        self,
        state: _CodexSessionState,
        method: str,
        params: Mapping[str, object],
        native: FrozenJsonDict,
    ) -> AgentEvent | None:
        if method == "thread/started":
            thread = self._mapping(params.get("thread"), "thread/started thread")
            if self._string(thread, "id", method) != state.ref.native_session_id:
                raise ProtocolDefect("Codex event changed thread identity")
            return None
        if method == "turn/started":
            return self._event(
                state, "turn_started", TurnStartedData(), native_type=method, native_payload=native
            )
        if method == "item/agentMessage/delta":
            delta = self._string(params, "delta", method)
            self._append_final_text(state, delta)
            return self._event(
                state,
                "text_delta",
                TextDeltaData(delta),
                native_type=method,
                native_payload=native,
            )
        if method in ("item/reasoning/summaryTextDelta", "item/reasoning/textDelta"):
            return self._event(
                state,
                "reasoning",
                ReasoningData(
                    self._string(params, "delta", method),
                    visibility="summary" if method.endswith("summaryTextDelta") else "full",
                ),
                native_type=method,
                native_payload=native,
            )
        if method in ("item/commandExecution/outputDelta", "item/fileChange/outputDelta"):
            return self._event(
                state,
                "tool_updated",
                ToolUpdatedData(
                    tool_call_id=self._string(params, "itemId", method),
                    update=freeze_json_object(
                        {"output_delta": self._string(params, "delta", method)}
                    ),
                ),
                native_type=method,
                native_payload=native,
            )
        if method == "item/mcpToolCall/progress":
            item_id = self._string(params, "itemId", method)
            if item_id not in state.active_mcp_calls:
                raise ProtocolDefect("MCP tool progress arrived before its start")
            return self._event(
                state,
                "tool_updated",
                ToolUpdatedData(
                    tool_call_id=item_id,
                    update=freeze_json_object({"message": self._string(params, "message", method)}),
                ),
                native_type=method,
                native_payload=native,
            )
        if method == "item/started":
            return self._item_started(state, params, method, native)
        if method == "item/completed":
            return self._item_completed(state, params, method, native)
        if method == "thread/tokenUsage/updated":
            usage = freeze_json_object(self._mapping(params.get("tokenUsage"), "token usage"))
            state.usage = usage
            return self._event(
                state, "usage", UsageData(usage), native_type=method, native_payload=native
            )
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
            if params.get("willRetry") is True:
                state.retry_count += 1
                return self._event(
                    state,
                    "native_retry_observed",
                    NativeRetryObservedData(state.retry_count),
                    native_type=method,
                    native_payload=native,
                )
            return self._event(
                state,
                "diagnostic",
                DiagnosticData(code="codex_error", message=normalized, detail=native),
                native_type=method,
                native_payload=native,
            )
        if method == "turn/completed":
            return self._turn_completed(state, params, method, native)
        if method in ("configWarning", "warning"):
            message = params.get("summary", params.get("message"))
            if not isinstance(message, str) or not message:
                raise ProtocolDefect(f"{method} carried no message")
            return self._event(
                state,
                "diagnostic",
                DiagnosticData(
                    code="codex_config_warning" if method == "configWarning" else "codex_warning",
                    message=sanitize_provider_text(message),
                    detail=native,
                ),
                native_type=method,
                native_payload=native,
            )
        if method in ("thread/status/changed", "mcpServer/startupStatus/updated"):
            return None
        return self._event(
            state,
            "unknown",
            UnknownData(),
            native_type=sanitize_provider_text(method, limit=128),
            native_payload=native,
        )

    def _item_started(
        self,
        state: _CodexSessionState,
        params: Mapping[str, object],
        method: str,
        native: FrozenJsonDict,
    ) -> AgentEvent | None:
        item = self._mapping(params.get("item"), "item/started item")
        item_type = item.get("type")
        if item_type == "commandExecution":
            return self._event(
                state,
                "tool_started",
                ToolStartedData(
                    tool_call_id=self._string(item, "id", method),
                    name="commandExecution",
                    arguments=freeze_json_object(
                        {"command": item.get("command"), "cwd": item.get("cwd")}
                    ),
                ),
                native_type=method,
                native_payload=native,
            )
        if item_type == "mcpToolCall":
            server = self._string(item, "server", method)
            tool = self._string(item, "tool", method)
            item_id = self._string(item, "id", method)
            requested = {spec.name: spec for spec in state.request.mcp_servers}
            spec = requested.get(server)
            if spec is None:
                raise ProtocolDefect("MCP tool call used an unconfigured server")
            if (spec.allowed_tools and tool not in spec.allowed_tools) or tool in spec.denied_tools:
                raise ProtocolDefect("MCP tool call violated its exact tool policy")
            if item_id in state.active_mcp_calls:
                raise ProtocolDefect("MCP tool call started more than once")
            arguments = item.get("arguments")
            normalized = (
                freeze_json_object(arguments)
                if isinstance(arguments, Mapping)
                else freeze_json_object({"value": freeze_json_value(arguments)})
            )
            state.active_mcp_calls[item_id] = (server, tool)
            return self._event(
                state,
                "tool_started",
                ToolStartedData(
                    tool_call_id=item_id,
                    name=f"{server}/{tool}",
                    arguments=normalized,
                ),
                native_type=method,
                native_payload=native,
            )
        if item_type in ("reasoning", "agentMessage", "fileChange"):
            return None
        return self._event(
            state,
            "unknown",
            UnknownData(),
            native_type=f"{method}:unknownItem",
            native_payload=native,
        )

    def _item_completed(
        self,
        state: _CodexSessionState,
        params: Mapping[str, object],
        method: str,
        native: FrozenJsonDict,
    ) -> AgentEvent | None:
        item = self._mapping(params.get("item"), "item/completed item")
        item_type = item.get("type")
        if item_type == "commandExecution":
            return self._event(
                state,
                "tool_completed",
                ToolCompletedData(
                    tool_call_id=self._string(item, "id", method),
                    output=freeze_json_value(item.get("aggregatedOutput")),
                    succeeded=item.get("status") == "completed",
                ),
                native_type=method,
                native_payload=native,
            )
        if item_type == "mcpToolCall":
            item_id = self._string(item, "id", method)
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
            return self._event(
                state,
                "tool_completed",
                ToolCompletedData(
                    tool_call_id=item_id,
                    output=freeze_json_value(
                        item.get("result") if status == "completed" else item.get("error")
                    ),
                    succeeded=status == "completed",
                ),
                native_type=method,
                native_payload=native,
            )
        if item_type in ("reasoning", "agentMessage", "fileChange"):
            return None
        return self._event(
            state,
            "unknown",
            UnknownData(),
            native_type=f"{method}:unknownItem",
            native_payload=native,
        )

    def _turn_completed(
        self,
        state: _CodexSessionState,
        params: Mapping[str, object],
        method: str,
        native: FrozenJsonDict,
    ) -> AgentEvent:
        if state.active_mcp_calls:
            raise ProtocolDefect("turn completed with active MCP tool calls")
        turn = self._mapping(params.get("turn"), "turn/completed turn")
        status = turn.get("status")
        if status == "completed":
            if isinstance(state.current_output, JsonSchemaAgentOutput):
                try:
                    structured = parse_structured_output(
                        state.final_text, state.current_output.schema
                    )
                except OutputSchemaMismatch:
                    return self._event(
                        state,
                        "turn_failed",
                        TurnFailedData(
                            failure="output_schema_violation",
                            final_text=state.final_text,
                            usage=state.usage,
                            diagnostics=tuple(state.diagnostics),
                        ),
                        native_type=method,
                        native_payload=native,
                    )
            else:
                structured = None
            return self._event(
                state,
                "turn_completed",
                TurnCompletedData(
                    final_text=state.final_text,
                    structured_output=structured,
                    usage=state.usage,
                    diagnostics=tuple(state.diagnostics),
                ),
                native_type=method,
                native_payload=native,
            )
        if status == "interrupted":
            return self._event(
                state,
                "turn_cancelled",
                TurnCancelledData(
                    final_text=state.final_text,
                    usage=state.usage,
                    diagnostics=tuple(state.diagnostics),
                ),
                native_type=method,
                native_payload=native,
            )
        if status == "failed":
            error = turn.get("error")
            failure = (
                "quota_exhausted"
                if state.quota_exhausted or self._is_quota_error(error)
                else "backend_failed"
            )
            return self._event(
                state,
                "turn_failed",
                TurnFailedData(
                    failure=failure,
                    final_text=state.final_text,
                    usage=state.usage,
                    diagnostics=tuple(state.diagnostics),
                ),
                native_type=method,
                native_payload=native,
            )
        raise ProtocolDefect("turn/completed carried an impossible status")

    def _file_change_events(
        self,
        state: _CodexSessionState,
        raw_changes: object,
        status: FileChangeStatus,
        method: str,
        native: FrozenJsonDict,
    ) -> tuple[AgentEvent, ...]:
        if not isinstance(raw_changes, list):
            raise ProtocolDefect(f"{method} changes were malformed")
        events: list[AgentEvent] = []
        for raw_change in raw_changes:
            change = self._mapping(raw_change, f"{method} change")
            kind = self._mapping(change.get("kind"), f"{method} change kind")
            kind_value = kind.get("type")
            mapped: Literal["created", "modified", "deleted"]
            if kind_value == "add":
                mapped = "created"
            elif kind_value == "delete":
                mapped = "deleted"
            elif kind_value == "update":
                mapped = "modified"
            else:
                raise ProtocolDefect(f"{method} change kind was malformed")
            diff = change.get("diff")
            if not isinstance(diff, str):
                raise ProtocolDefect(f"{method} change diff was malformed")
            events.append(
                self._event(
                    state,
                    "file_change",
                    FileChangeData(
                        path=self._string(change, "path", method),
                        change=mapped,
                        status=status,
                        diff=diff,
                    ),
                    native_type=method,
                    native_payload=native,
                )
            )
        return tuple(events)

    def _validate_notification_identity(
        self, state: _CodexSessionState, method: str, params: Mapping[str, object]
    ) -> None:
        thread_id = params.get("threadId")
        if thread_id is not None and thread_id != state.ref.native_session_id:
            raise ProtocolDefect("Codex event changed thread identity")
        turn_id = params.get("turnId")
        turn = params.get("turn")
        if turn_id is None and isinstance(turn, Mapping):
            turn_id = turn.get("id")
        if method in _TURN_SCOPED_METHODS:
            if thread_id != state.ref.native_session_id:
                raise ProtocolDefect("Codex event omitted its thread identity")
            if turn_id != state.turn_id:
                raise ProtocolDefect("Codex event changed or omitted its turn identity")

    def _event(
        self,
        state: _CodexSessionState,
        kind: AgentEventKind,
        data: AgentEventData,
        *,
        native_type: str | None = None,
        native_payload: JsonObject | None = None,
    ) -> AgentEvent:
        if state.turn_id is None:
            raise ProtocolDefect("Codex event arrived before a turn id")
        terminal = kind in ("turn_completed", "turn_failed", "turn_cancelled")
        if state.seq >= _MAX_EVENT_COUNT or (not terminal and state.seq == _MAX_EVENT_COUNT - 1):
            raise OutputLimitExceeded(_MAX_EVENT_COUNT)
        state.seq += 1
        return AgentEvent(
            schema_version="agent-event.v1",
            seq=state.seq,
            backend="codex",
            transport="sdk",
            session_ref=state.ref,
            turn_id=state.turn_id,
            kind=kind,
            data=data,
            native_type=native_type,
            native_payload=native_payload,
        )

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
            AgentCapabilityScope(backend="codex", transport="sdk", auth=request.auth),
            state_root_fingerprint=fingerprint_path(
                state_root_from_environment("codex", environment)
            ),
            cwd=request.cwd,
            cwd_scopes_sessions=False,
        )

    def _session_summary(
        self,
        thread: Mapping[str, object],
        *,
        profile_key: str,
        state_root: Path,
    ) -> SessionSummary:
        return SessionSummary(
            ref=self._make_ref(
                self._string(thread, "id", "thread_list"),
                profile_key,
                state_root,
                self._string(thread, "cwd", "thread_list"),
            ),
            metadata=SessionMetadata(name=self._optional_string(thread.get("name"))),
        )

    @staticmethod
    def _make_ref(
        native_session_id: str, profile_key: str, state_root: Path, cwd: str
    ) -> AgentSessionRef:
        return AgentSessionRef(
            schema_version="agent-session-ref.v1",
            backend="codex",
            transport="sdk",
            native_session_id=native_session_id,
            profile_key=profile_key,
            state_root_fingerprint=fingerprint_path(state_root),
            cwd_fingerprint=fingerprint_path(cwd),
        )

    @staticmethod
    def _load_sdk() -> ModuleType:
        try:
            return importlib.import_module("openai_codex")
        except ModuleNotFoundError as error:
            if error.name == "openai_codex":
                raise SdkUnavailable(
                    "Codex SDK is not installed; install the 'codex-sdk' extra"
                ) from error
            raise

    @staticmethod
    def _load_runtime_package() -> ModuleType:
        try:
            return importlib.import_module("codex_cli_bin")
        except ModuleNotFoundError as error:
            if error.name == "codex_cli_bin":
                raise SdkUnavailable(
                    "Codex bundled runtime is not installed; install the 'codex-sdk' extra"
                ) from error
            raise

    @staticmethod
    def _runtime_version(runtime: ModuleType) -> str:
        for name in ("bundled_codex_path", "bundled_path_dir"):
            if not callable(getattr(runtime, name, None)):
                raise SdkUnavailable(f"Codex bundled runtime is missing public {name}")
        version = getattr(runtime, "__version__", None)
        if version is None:
            package_name = getattr(runtime, "PACKAGE_NAME", None)
            if not isinstance(package_name, str) or not package_name:
                raise SdkUnavailable("Codex bundled runtime does not report its package name")
            try:
                version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError as error:
                raise SdkUnavailable("Codex bundled runtime does not report a version") from error
        if not isinstance(version, str) or not version:
            raise SdkUnavailable("Codex bundled runtime does not report a version")
        return version

    @staticmethod
    def _sdk_version(sdk: ModuleType) -> str:
        version = getattr(sdk, "__version__", None)
        if not isinstance(version, str) or not version:
            raise SdkUnavailable("Codex SDK does not report a version")
        for name in (
            "AsyncCodex",
            "CodexConfig",
            "ApprovalMode",
            "Sandbox",
            "TextInput",
            "LocalImageInput",
        ):
            if not hasattr(sdk, name):
                raise SdkUnavailable(f"Codex SDK is missing public {name}")
        return version

    @classmethod
    def _executable_version(cls, client: Any) -> str:
        metadata = cls._mapping(client.metadata, "Codex SDK initialize metadata")
        server = cls._mapping(metadata.get("serverInfo"), "Codex SDK server metadata")
        version = server.get("version")
        if not isinstance(version, str) or not version:
            raise ProtocolDefect("Codex SDK server metadata had no version")
        match = _RUNTIME_VERSION_PREFIX.match(version)
        if match is None:
            raise ProtocolDefect("Codex SDK server metadata had an invalid version")
        return match.group("version")

    @staticmethod
    def _require_scope(scope: AgentCapabilityScope) -> None:
        if scope.backend != "codex" or scope.transport != "sdk":
            raise InvalidAgentRequest("Codex SDK received a different capability scope")

    @staticmethod
    def _require_local_auth(kind: str) -> None:
        if kind != "local_account":
            raise UnsupportedCapability("Codex SDK agent sessions require local ChatGPT auth")

    @staticmethod
    def _approval_mode(sdk: ModuleType, policy: PermissionPolicy) -> object:
        if policy.approval == "deny":
            return sdk.ApprovalMode.deny_all
        if policy.approval == "provider_review":
            return sdk.ApprovalMode.auto_review
        raise UnsupportedCapability(
            "Codex SDK supports deny or provider_review approvals, not caller ask/allow"
        )

    @staticmethod
    def _sandbox(sdk: ModuleType, policy: PermissionPolicy) -> object:
        return {
            "read_only": sdk.Sandbox.read_only,
            "workspace_write": sdk.Sandbox.workspace_write,
            "full_access": sdk.Sandbox.full_access,
        }[policy.filesystem]

    @staticmethod
    def _validate_policy_mapping(policy: PermissionPolicy) -> None:
        if policy.network == "allowlist":
            raise UnsupportedCapability("Codex SDK has no typed network allowlist mapping")
        if policy.filesystem == "full_access" and policy.network != "unrestricted":
            raise UnsupportedCapability(
                "Codex full_access cannot preserve restricted network policy"
            )
        if policy.filesystem != "full_access" and policy.network != "disabled":
            raise UnsupportedCapability(
                "Codex SDK sandbox presets cannot enable network without full_access"
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
    def _text_only(parts: tuple[object, ...], context: str) -> str:
        if any(not isinstance(part, TextContent) for part in parts):
            raise UnsupportedCapability(f"{context} supports text only")
        return "\n\n".join(part.text for part in parts if isinstance(part, TextContent))

    @staticmethod
    def _codex_input(
        sdk: ModuleType,
        part: object,
        request: AgentSessionRequest,
        policy: PermissionPolicy,
    ) -> object:
        if isinstance(part, TextContent):
            return sdk.TextInput(text=part.text)
        if isinstance(part, ImageContent):
            path = Path(part.path).resolve()
            roots = tuple(Path(root).resolve() for root in (request.cwd, *request.additional_dirs))
            if policy.filesystem != "full_access" and not any(
                path.is_relative_to(root) for root in roots
            ):
                raise InvalidAgentRequest("image attachment is outside authorized workspace roots")
            if not path.is_file() or path.stat().st_size != part.size_bytes:
                raise InvalidAgentRequest("image attachment path or declared size is invalid")
            return sdk.LocalImageInput(path=str(path))
        raise UnsupportedCapability("Codex SDK input supports text and local images")

    def _codex_config(self, request: AgentSessionRequest) -> dict[str, object]:
        config: dict[str, object] = {
            "mcp_servers": {},
            "web_search": "disabled",
            "shell_environment_policy": {"inherit": "core", "exclude": []},
        }
        native = request.native
        if isinstance(native, CodexNativeOptions) and native.web_search is not None:
            config["web_search"] = "live" if native.web_search else "disabled"
        if request.policy.filesystem == "workspace_write":
            config["sandbox_workspace_write"] = {
                "writable_roots": [request.cwd, *request.additional_dirs],
                "network_access": False,
            }
        if request.mcp_servers:
            servers: dict[str, object] = {}
            for server in request.mcp_servers:
                if server.transport == "stdio":
                    entry: dict[str, object] = {
                        "command": server.command,
                        "args": list(server.args),
                    }
                    if server.environment_refs:
                        entry["env_vars"] = [
                            reference.name for reference in server.environment_refs
                        ]
                else:
                    entry = {"url": server.url}
                    headers = {
                        reference.name: mcp_header_environment_name("codex", reference.source)
                        for reference in server.header_refs
                    }
                    if headers:
                        entry["env_http_headers"] = headers
                entry["required"] = server.required
                if server.allowed_tools:
                    entry["enabled_tools"] = list(server.allowed_tools)
                if server.denied_tools:
                    entry["disabled_tools"] = list(server.denied_tools)
                servers[server.name] = entry
            config["mcp_servers"] = servers
            excluded = sorted(
                {
                    reference.name
                    for server in request.mcp_servers
                    for reference in server.environment_refs
                }
                | {
                    mcp_header_environment_name("codex", reference.source)
                    for server in request.mcp_servers
                    for reference in server.header_refs
                }
            )
            config["shell_environment_policy"] = {"inherit": "core", "exclude": excluded}
        return config

    @classmethod
    def _decode_models(
        cls, response: object
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
        payload = cls._mapping(response, "Codex SDK model response")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ProtocolDefect("Codex SDK model response data was not an array")
        models: list[str] = []
        mapped: list[tuple[str, tuple[str, ...]]] = []
        all_efforts: list[str] = []
        for raw in data:
            model = cls._mapping(raw, "Codex SDK model")
            name = model.get("model")
            efforts = model.get("supportedReasoningEfforts")
            if not isinstance(name, str) or not name or not isinstance(efforts, list):
                raise ProtocolDefect("Codex SDK model was malformed")
            if name in models:
                raise ProtocolDefect("Codex SDK repeated a model identity")
            model_efforts: list[str] = []
            for raw_effort in efforts:
                effort = cls._mapping(raw_effort, "Codex SDK model effort").get("reasoningEffort")
                if not isinstance(effort, str) or not effort or effort in model_efforts:
                    raise ProtocolDefect("Codex SDK model effort was malformed")
                model_efforts.append(effort)
                if effort not in all_efforts:
                    all_efforts.append(effort)
            models.append(name)
            mapped.append((name, tuple(model_efforts)))
        return tuple(models), tuple(all_efforts), tuple(mapped)

    @staticmethod
    def _append_final_text(state: _CodexSessionState, text: str) -> None:
        size = len(text.encode("utf-8"))
        if size > _MAX_EVENT_TEXT_BYTES:
            raise OutputLimitExceeded(_MAX_EVENT_TEXT_BYTES)
        if state.final_text_bytes + size > _MAX_FINAL_TEXT_BYTES:
            raise OutputLimitExceeded(_MAX_FINAL_TEXT_BYTES)
        state.final_text += text
        state.final_text_bytes += size

    @staticmethod
    def _append_diagnostic(state: _CodexSessionState, message: str) -> None:
        if message in state.diagnostics:
            return
        if len(state.diagnostics) >= _MAX_DIAGNOSTICS:
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
    def _optional_string(value: object) -> str | None:
        return value if isinstance(value, str) and value else None


__all__ = ["CodexSdkAdapter"]
