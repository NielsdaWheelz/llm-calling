"""Native Codex Python SDK adapter with a lazy optional dependency boundary.

The adapter owns official ``openai-codex`` clients and normalizes their public
notification stream into the closed six-kind event vocabulary. Validation is
behavioral: there is no capability table, so every backend fact this transport
cannot enforce is refused at ``open_session``/``stream_turn`` before billable
work, and version drift is a warning backed by the behavioral probe
(``client.account()`` plus the initialize metadata), never a hard failure.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import os
import re
import sys
import warnings
import weakref
from collections.abc import AsyncGenerator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

from provider_runtime.errors import sanitize_provider_text
from provider_runtime.types import Absent, Presence, Present, TokenUsage

from ._codex_launcher import ensure_codex_launcher
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
from ._sandbox import bubblewrap_network_namespace_available
from ._structured_output import OutputSchemaMismatch, parse_structured_output
from .auth import (
    freeze_native_json_object,
    freeze_native_json_value,
    mcp_header_environment_name,
    redact_native_payload,
    state_root_from_environment,
)
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
    AgentFailure,
    AgentNative,
    AgentQuotaExhausted,
    AgentTerminal,
    AgentTerminalFailure,
    AgentText,
    AgentToolUse,
    AgentUsage,
)
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
from .types import (
    AgentSessionRef,
    AgentSessionRequest,
    ApprovalHandler,
    CodexNativeOptions,
    CredentialRef,
    ForkSession,
    ImageContent,
    JsonSchemaAgentOutput,
    NewSession,
    ResumeSession,
    TextContent,
    TurnRequest,
    thaw_json_value,
    validate_mcp_network_policy,
)

# The one version the adapter was certified against. Drift from it is reported as a
# RuntimeWarning and the behavioral probe decides fitness; only a missing module or a
# missing public surface remains a hard `SdkUnavailable`.
_CERTIFIED_SDK_VERSION = "0.144.4"
_RUNTIME_VERSION_PREFIX = re.compile(
    r"(?P<version>[0-9]+(?:\.[0-9]+){2}(?:[-+][0-9A-Za-z.-]+)?)(?:\s|$)"
)
_REQUIRED_MCP_STARTUP_FAILURE = "required MCP servers failed to initialize"

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
    turn: Any | None = None
    turn_id: str | None = None
    message_count: int = 0
    output_bytes: int = 0
    final_text: str = ""
    final_text_bytes: int = 0
    usage: TokenUsage | None = None
    diagnostics: list[str] = field(default_factory=list)
    active_mcp_calls: dict[str, tuple[str, str]] = field(default_factory=dict)
    quota_exhausted: bool = False


class CodexSdkAdapter:
    """Own official Codex SDK clients and normalize their public notification stream."""

    backend: Literal["codex"] = "codex"
    transport: Literal["sdk"] = "sdk"
    # Codex threads resume from any directory; the ref keeps cwd as provenance only.
    cwd_scopes_sessions: Literal[False] = False

    def __init__(self) -> None:
        self._sessions: dict[AgentSession, _CodexSessionState] = {}
        self._dead_sessions: weakref.WeakSet[AgentSession] = weakref.WeakSet()
        self._clients: set[Any] = set()

    def validate_auth(self, credential: CredentialRef) -> None:
        self._require_local_auth(credential.kind)

    async def list_sessions(
        self,
        query: SessionQuery,
        *,
        environment: Mapping[str, str],
    ) -> SessionPage:
        if query.backend != self.backend or query.transport != self.transport:
            raise InvalidAgentRequest("CodexSdkAdapter received a different route")
        self._require_local_auth(query.auth.kind)
        _sdk, client = await self._open_client(cwd=None, environment=environment)
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
            state_root = state_root_from_environment("codex", environment)
            sessions = tuple(
                self._session_summary(
                    self._mapping(item, "Codex SDK thread_list thread"),
                    profile_key=query.auth.profile_key,
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
            )
        finally:
            await self._close_client(client)

    async def open_session(
        self,
        request: AgentSessionRequest,
        *,
        environment: Mapping[str, str],
    ) -> AgentSession:
        if request.backend != self.backend or request.transport != self.transport:
            raise InvalidAgentRequest("CodexSdkAdapter received a different route")
        self._require_local_auth(request.auth.kind)
        self._validate_policy_mapping(request.policy)
        validate_mcp_network_policy(request.mcp_servers, request.policy)
        self._validate_mcp_filters(request)
        state_root = state_root_from_environment("codex", environment)
        if request.policy.filesystem == "workspace_write":
            # Restricted writes ride on bubblewrap network namespaces; a host without
            # them gets a fail-closed refusal before any thread is started.
            if not await bubblewrap_network_namespace_available(
                cwd=state_root, environment=environment
            ):
                raise UnsupportedCapability(
                    "Codex workspace_write requires bubblewrap network namespaces on this host"
                )

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
            if request.system:
                # `base_instructions` is the SDK's system-role channel on thread
                # start/resume/fork (0.144.4 api.py:135) and *replaces* Codex's built-in base
                # prompt rather than appending to it, which is what a caller asking for
                # session system instructions is asking for.
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
                state_root=state_root,
                cwd=request.cwd,
            )
            session = AgentSession(ref)
            self._sessions[session] = _CodexSessionState(
                sdk=sdk,
                client=client,
                thread=thread,
                request=request,
                ref=ref,
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
        if request.policy is not None:
            raise UnsupportedCapability("Codex SDK cannot reconfigure policy on a started thread")

        policy = state.request.policy
        inputs = [self._codex_input(state.sdk, part) for part in request.input]
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
            # The schema is a plain frozen JSON mapping; the backend enforces it natively.
            kwargs["output_schema"] = thaw_json_value(output.schema)

        turn = await self._call(
            state.thread.turn(inputs, **kwargs), operation="turn start", failure="session"
        )
        turn_id = getattr(turn, "id", None)
        if not isinstance(turn_id, str) or not turn_id:
            await self._destroy_session(session, state)
            raise ProtocolDefect("Codex SDK returned no turn id")
        state.turn = turn
        state.turn_id = turn_id
        state.message_count = 0
        state.output_bytes = 0
        state.final_text = ""
        state.final_text_bytes = 0
        state.usage = None
        state.diagnostics.clear()
        state.active_mcp_calls.clear()
        state.quota_exhausted = False

        terminal_seen = False
        try:
            stream = turn.stream()
            async for notification in stream:
                for event in self._notification_events(state, notification):
                    yield event
                    if isinstance(event, AgentTerminal):
                        terminal_seen = True
                        state.turn = None
                        return
        except OutputLimitExceeded:
            try:
                await turn.interrupt()
            finally:
                await self._destroy_session(session, state)
            yield AgentTerminal(
                status="failed",
                failure=AgentFailure("output_limit_exceeded"),
                final_text=state.final_text,
                session_ref=state.ref,
                usage=self._terminal_usage(state),
                diagnostics=tuple(state.diagnostics),
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
            yield AgentTerminal(
                status="failed",
                failure=AgentQuotaExhausted()
                if self._is_quota_error_text(message)
                else AgentFailure("backend_failed"),
                final_text=state.final_text,
                session_ref=state.ref,
                usage=self._terminal_usage(state),
                diagnostics=tuple(state.diagnostics),
            )
            return
        if not terminal_seen:
            await self._destroy_session(session, state)
            raise MissingTerminalEvent()

    async def interrupt(self, session: AgentSession) -> None:
        if session in self._dead_sessions:
            return
        state = self._state(session)
        if state.turn is None:
            # Block-and-stop: a turn that never became identifiable releases the session.
            await self._destroy_session(session, state)
            return
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
        if sdk_version != _CERTIFIED_SDK_VERSION:
            warnings.warn(
                f"Codex SDK {sdk_version} differs from the certified "
                f"{_CERTIFIED_SDK_VERSION}; continuing on the behavioral probe",
                RuntimeWarning,
                stacklevel=2,
            )
        runtime = self._load_runtime_package()
        runtime_version = self._runtime_version(runtime)
        if runtime_version != sdk_version:
            warnings.warn(
                f"Codex bundled runtime {runtime_version} differs from SDK "
                f"{sdk_version}; continuing on the behavioral probe",
                RuntimeWarning,
                stacklevel=2,
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
            if executable_version != runtime_version:
                warnings.warn(
                    f"Codex server reported version {executable_version}, the bundled "
                    f"runtime is {runtime_version}; continuing on the behavioral probe",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except TimeoutError:
            raise ExecutableUnavailable("Codex SDK initialization timed out") from None
        except CredentialUnavailable:
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
        if method == "turn/completed":
            # The native completion frame travels first; the owned terminal is last.
            return (
                AgentNative(native_type=method, payload=redact_native_payload(params)),
                self._turn_terminal(state, params),
            )
        event = self._notification_event(state, method, params)
        return () if event is None else (event,)

    def _notification_event(
        self,
        state: _CodexSessionState,
        method: str,
        params: Mapping[str, object],
    ) -> AgentEvent | None:
        if method == "thread/started":
            thread = self._mapping(params.get("thread"), "thread/started thread")
            if self._string(thread, "id", method) != state.ref.native_session_id:
                raise ProtocolDefect("Codex event changed thread identity")
            return None
        if method == "turn/started":
            return AgentNative(native_type=method, payload=redact_native_payload(params))
        if method == "item/agentMessage/delta":
            delta = self._string(params, "delta", method)
            self._append_final_text(state, delta)
            return AgentText(delta)
        if method in ("item/reasoning/summaryTextDelta", "item/reasoning/textDelta"):
            return AgentNative(native_type=method, payload=redact_native_payload(params))
        if method in ("item/commandExecution/outputDelta", "item/fileChange/outputDelta"):
            return AgentToolUse(
                tool_call_id=self._string(params, "itemId", method),
                name="commandExecution"
                if method == "item/commandExecution/outputDelta"
                else "fileChange",
                phase="updated",
                payload=freeze_native_json_object(
                    {"output_delta": self._string(params, "delta", method)}
                ),
            )
        if method == "item/mcpToolCall/progress":
            item_id = self._string(params, "itemId", method)
            identity = state.active_mcp_calls.get(item_id)
            if identity is None:
                raise ProtocolDefect("MCP tool progress arrived before its start")
            server, tool = identity
            return AgentToolUse(
                tool_call_id=item_id,
                name=f"{server}/{tool}",
                phase="updated",
                payload=freeze_native_json_object(
                    {"message": self._string(params, "message", method)}
                ),
            )
        if method == "item/started":
            return self._item_started(state, params, method)
        if method == "item/completed":
            return self._item_completed(state, params, method)
        if method == "item/fileChange/patchUpdated":
            return AgentToolUse(
                tool_call_id=self._string(params, "itemId", method),
                name="fileChange",
                phase="updated",
                payload=freeze_native_json_object({"changes": params.get("changes")}),
            )
        if method == "thread/tokenUsage/updated":
            usage = self._token_usage(params)
            state.usage = usage
            return AgentUsage(usage)
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
        if method in ("configWarning", "warning"):
            message = params.get("summary", params.get("message"))
            if not isinstance(message, str) or not message:
                raise ProtocolDefect(f"{method} carried no message")
            self._append_diagnostic(state, sanitize_provider_text(message))
            return AgentNative(native_type=method, payload=redact_native_payload(params))
        return AgentNative(
            native_type=sanitize_provider_text(method, limit=128),
            payload=redact_native_payload(params),
        )

    def _item_started(
        self,
        state: _CodexSessionState,
        params: Mapping[str, object],
        method: str,
    ) -> AgentEvent | None:
        item = self._mapping(params.get("item"), "item/started item")
        item_type = item.get("type")
        if item_type == "commandExecution":
            return AgentToolUse(
                tool_call_id=self._string(item, "id", method),
                name="commandExecution",
                phase="started",
                payload=freeze_native_json_object(
                    {"command": item.get("command"), "cwd": item.get("cwd")}
                ),
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
            state.active_mcp_calls[item_id] = (server, tool)
            return AgentToolUse(
                tool_call_id=item_id,
                name=f"{server}/{tool}",
                phase="started",
                payload=freeze_native_json_value(item.get("arguments")),
            )
        if item_type == "fileChange":
            return AgentToolUse(
                tool_call_id=self._string(item, "id", method),
                name="fileChange",
                phase="started",
                payload=freeze_native_json_object(
                    {
                        "changes": item.get("changes"),
                        "status": self._patch_status(item.get("status"), method),
                    }
                ),
            )
        if item_type in ("reasoning", "agentMessage"):
            return None
        return AgentNative(
            native_type=f"{method}:unknownItem",
            payload=redact_native_payload(params),
        )

    def _item_completed(
        self,
        state: _CodexSessionState,
        params: Mapping[str, object],
        method: str,
    ) -> AgentEvent | None:
        item = self._mapping(params.get("item"), "item/completed item")
        item_type = item.get("type")
        if item_type == "commandExecution":
            return AgentToolUse(
                tool_call_id=self._string(item, "id", method),
                name="commandExecution",
                phase="completed",
                payload=freeze_native_json_value(item.get("aggregatedOutput")),
                succeeded=item.get("status") == "completed",
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
            server, tool = identity
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
            return AgentToolUse(
                tool_call_id=self._string(item, "id", method),
                name="fileChange",
                phase="completed",
                payload=freeze_native_json_object({"changes": item.get("changes")}),
                succeeded=status == "applied",
            )
        if item_type in ("reasoning", "agentMessage"):
            return None
        return AgentNative(
            native_type=f"{method}:unknownItem",
            payload=redact_native_payload(params),
        )

    def _turn_terminal(
        self,
        state: _CodexSessionState,
        params: Mapping[str, object],
    ) -> AgentTerminal:
        if state.active_mcp_calls:
            raise ProtocolDefect("turn completed with active MCP tool calls")
        turn = self._mapping(params.get("turn"), "turn/completed turn")
        status = turn.get("status")
        usage = self._terminal_usage(state)
        diagnostics = tuple(state.diagnostics)
        if status == "completed":
            structured = None
            if isinstance(state.request.output, JsonSchemaAgentOutput):
                try:
                    # Strict parse and freeze only; the backend enforced the schema natively.
                    structured = parse_structured_output(state.final_text)
                except OutputSchemaMismatch:
                    return AgentTerminal(
                        status="failed",
                        failure=AgentFailure("output_schema_violation"),
                        final_text=state.final_text,
                        session_ref=state.ref,
                        usage=usage,
                        diagnostics=diagnostics,
                    )
            return AgentTerminal(
                status="succeeded",
                failure=None,
                final_text=state.final_text,
                session_ref=state.ref,
                structured_output=structured,
                usage=usage,
                diagnostics=diagnostics,
            )
        if status == "interrupted":
            return AgentTerminal(
                status="cancelled",
                failure=None,
                final_text=state.final_text,
                session_ref=state.ref,
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
                final_text=state.final_text,
                session_ref=state.ref,
                usage=usage,
                diagnostics=diagnostics,
            )
        raise ProtocolDefect("turn/completed carried an impossible status")

    def _token_usage(self, params: Mapping[str, object]) -> TokenUsage:
        """Normalize the ``tokenUsage.total`` member into the provider lane's noun.

        ``inputTokens`` is already cache-inclusive (OpenAI wire semantics), so it maps
        straight onto ``TokenUsage.input_tokens`` without re-adding cache components.
        """
        token_usage = self._mapping(params.get("tokenUsage"), "token usage")
        total = self._mapping(token_usage.get("total"), "token usage total")
        try:
            return TokenUsage.from_components(
                input_tokens=self._usage_count(total, "inputTokens"),
                output_tokens=self._usage_count(total, "outputTokens"),
                total_tokens=self._usage_presence(total, "totalTokens"),
                reasoning_tokens=self._usage_presence(total, "reasoningOutputTokens"),
                cache_read_input_tokens=self._usage_presence(total, "cachedInputTokens"),
                cache_write_input_tokens=self._usage_presence(total, "cacheWriteInputTokens"),
            )
        except ValueError:
            raise ProtocolDefect("thread/tokenUsage/updated carried negative counts") from None

    @staticmethod
    def _usage_count(member: Mapping[str, object], key: str) -> int:
        value = member.get(key)
        if value is None:
            return 0
        if type(value) is not int:
            raise ProtocolDefect(f"tokenUsage.total.{key} was not an integer")
        return value

    @staticmethod
    def _usage_presence(member: Mapping[str, object], key: str) -> Presence[int]:
        value = member.get(key)
        if value is None:
            return Absent()
        if type(value) is not int:
            raise ProtocolDefect(f"tokenUsage.total.{key} was not an integer")
        return Present(value)

    @staticmethod
    def _terminal_usage(state: _CodexSessionState) -> Presence[TokenUsage]:
        return Absent() if state.usage is None else Present(state.usage)

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
            state_root_fingerprint=fingerprint_path(
                state_root_from_environment("codex", environment)
            ),
            cwd=request.cwd,
            cwd_scopes_sessions=self.cwd_scopes_sessions,
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
                native_session_id=self._string(thread, "id", "thread_list"),
                profile_key=profile_key,
                state_root=state_root,
                cwd=self._string(thread, "cwd", "thread_list"),
            ),
            metadata=SessionMetadata(name=self._optional_string(thread.get("name"))),
        )

    @staticmethod
    def _make_ref(
        *, native_session_id: str, profile_key: str, state_root: Path, cwd: str
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
        if policy.filesystem == "read_only" and policy.network != "disabled":
            # `thread_start` selects the sandbox by mode name and carries the rest in `config`
            # (0.144.4 api.py:372-384), and that config's only network toggle is
            # `sandbox_workspace_write.network_access` (generated/v2_all.py:7877-7878, 3511-3518).
            # There is no read-only counterpart section, so a read-only sandbox stays offline.
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
    def _text_only(parts: tuple[object, ...], context: str) -> str:
        if any(not isinstance(part, TextContent) for part in parts):
            raise UnsupportedCapability(f"{context} supports text only")
        return "\n\n".join(part.text for part in parts if isinstance(part, TextContent))

    @staticmethod
    def _codex_input(sdk: ModuleType, part: object) -> object:
        if isinstance(part, TextContent):
            return sdk.TextInput(text=part.text)
        if isinstance(part, ImageContent):
            # Existence, declared size, and containment under the authorized roots are
            # `AgentRuntime._validate_content_files`'s single check of every turn input, so
            # this only translates the part the SDK accepts.
            return sdk.LocalImageInput(path=part.path)
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
                "network_access": request.policy.network == "unrestricted",
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
        """Append one deduplicated turn diagnostic, failing the turn past the bound.

        These come from the turn's own `error`/`warning` notifications, so they are part of
        the turn's output budget like its text and its frames: exceeding the bound ends the
        turn as `output_limit_exceeded` through `stream_turn`'s existing handler rather than
        reporting a silently truncated diagnostic record. Claude's `_record_diagnostic`
        drops instead, because there the callers are teardown paths with an outcome of their
        own that must not be replaced by a bound.
        """
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
