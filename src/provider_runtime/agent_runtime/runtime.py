"""Async lifecycle, closed adapter dispatch, and terminal turn ownership."""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import stat
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, assert_never, runtime_checkable

from provider_runtime.types import Absent, CancelSignal, Presence, Present, TokenUsage

from .auth import (
    AuthEnvironmentRequest,
    build_child_environment,
    child_home_directory,
    credential_environment_names,
    is_runtime_owned_environment_name,
    mcp_header_environment_name,
    resolve_state_root,
)
from .errors import (
    CredentialUnavailable,
    InvalidAgentRequest,
    McpConfigurationError,
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
    AgentTerminal,
    AgentText,
    AgentUsage,
    validate_event_stream,
)
from .policy import PermissionPolicy, narrow_policy
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
    AGENT_ROUTES,
    AgentSessionRef,
    AgentSessionRequest,
    AgentTransport,
    ApprovalHandler,
    Backend,
    ContentPart,
    CredentialRef,
    FileContent,
    ForkSession,
    ImageContent,
    McpServerSpec,
    ResumeSession,
    TextContent,
    TurnRequest,
    validate_mcp_network_policy,
)

type SecretResolver = Callable[[str], Awaitable[str]]
type Route = tuple[Backend, AgentTransport]
type StopCause = Literal["cancelled", "timed_out"]

# Cleanup budgets. `_TURN_CLEANUP_TIMEOUT_SECONDS` bounds one interruption call; the close
# budgets are a separate pair (drain the consumer, then settle after cancellation).
_TURN_CLEANUP_TIMEOUT_SECONDS = 2.0
_CLOSE_DRAIN_TIMEOUT_SECONDS = 10.0
_CLOSE_SETTLE_TIMEOUT_SECONDS = 2.0
_STATE_ROOT_MODE = 0o700


@dataclass(frozen=True, slots=True)
class AgentRuntimeConfig:
    state_root_base: Path
    claude_executable: str = "claude"
    max_turn_seconds: float = 3_600.0
    secret_resolver: SecretResolver | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state_root_base, Path):
            raise InvalidAgentRequest("state_root_base must be a Path")
        # Lexical only, exactly as `types._require_absolute_path` validates caller paths:
        # requiring `resolve() == self` would reject every base reached through a symlink,
        # including the `Path("~/.state").expanduser()` of a distro where /home -> /var/home.
        # Symlink resolution is the pre-launch job of `_environment`/`_secure_state_root`.
        text = str(self.state_root_base)
        if not self.state_root_base.is_absolute() or os.path.normpath(text) != text:
            raise InvalidAgentRequest("state_root_base must be a normalized absolute path")
        if not self.state_root_base.is_dir():
            raise InvalidAgentRequest("state_root_base must be an existing directory")
        if (
            type(self.claude_executable) is not str
            or not self.claude_executable
            or "\0" in self.claude_executable
        ):
            raise InvalidAgentRequest("claude_executable must be a non-empty executable name")
        if (
            type(self.max_turn_seconds) not in (int, float)
            or not math.isfinite(self.max_turn_seconds)
            or self.max_turn_seconds <= 0
        ):
            raise InvalidAgentRequest("max_turn_seconds must be positive and finite")
        if self.secret_resolver is not None and not callable(self.secret_resolver):
            raise InvalidAgentRequest("secret_resolver must be callable when configured")


class AgentAdapter(Protocol):
    @property
    def backend(self) -> Backend: ...

    @property
    def transport(self) -> AgentTransport: ...

    @property
    def cwd_scopes_sessions(self) -> bool:
        """Whether this backend's native sessions are directory-scoped (a backend fact)."""
        ...

    def validate_auth(self, credential: CredentialRef) -> None: ...

    async def list_sessions(
        self, query: SessionQuery, *, environment: Mapping[str, str]
    ) -> SessionPage: ...

    async def read_session(
        self,
        ref: AgentSessionRef,
        options: SessionReadOptions,
        *,
        environment: Mapping[str, str],
    ) -> SessionSnapshot: ...

    async def open_session(
        self,
        request: AgentSessionRequest,
        *,
        environment: Mapping[str, str],
    ) -> AgentSession:
        """Open one native session, validating the request behaviorally.

        There is no capability table: the adapter enforces the backend facts it
        owns (policy mapping, content kinds, native options) and fails closed
        before any billable work when the request asks for something the
        transport cannot enforce.
        """
        ...

    def stream_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None,
    ) -> AsyncGenerator[AgentEvent, None]: ...

    async def interrupt(self, session: AgentSession) -> None:
        """Interrupt the session's active turn natively and keep it fit for the next turn.

        The runtime abandons the turn's event stream as soon as this returns, so
        an adapter whose native transport outlives a single turn owns the
        interrupted turn's leftover frames: it must drain or discard them before
        the next turn reads the stream. An adapter whose turn never became
        identifiable releases the whole session instead.
        """
        ...

    async def close_session(self, session: AgentSession) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class _InterruptedFinalTextAdapter(Protocol):
    """Optional provider-owned projection for a runtime-synthesized terminal."""

    def _interrupted_final_text(self, session: AgentSession) -> str: ...


@dataclass(slots=True)
class _SessionBinding:
    adapter: AgentAdapter
    request: AgentSessionRequest
    state_root_fingerprint: str
    ref_validated: bool = False
    turn_active: bool = False
    turn_driver: asyncio.Task[None] | None = field(default=None, repr=False)
    consumer_terminal_received: asyncio.Event | None = field(default=None, repr=False)
    turn_error: BaseException | None = field(default=None, repr=False)
    forced_terminal: AgentEvent | None = field(default=None, repr=False)
    invalidated: bool = False
    interrupt_requested: bool = False
    close_task: asyncio.Task[tuple[BaseException, ...]] | None = field(default=None, repr=False)


class AgentRuntime:
    def __init__(
        self,
        config: AgentRuntimeConfig,
        *,
        adapters: Iterable[AgentAdapter] | None = None,
    ) -> None:
        if not isinstance(config, AgentRuntimeConfig):
            raise InvalidAgentRequest("AgentRuntime requires AgentRuntimeConfig")
        selected = tuple(adapters) if adapters is not None else _default_adapters(config)
        routes: dict[Route, AgentAdapter] = {}
        for adapter in selected:
            route = (adapter.backend, adapter.transport)
            if route not in AGENT_ROUTES:
                raise InvalidAgentRequest(f"adapter has unsupported route {route!r}")
            if route in routes:
                raise InvalidAgentRequest(f"duplicate adapter route {route!r}")
            routes[route] = adapter
        self._config = config
        self._adapters = routes
        self._sessions: dict[AgentSession, _SessionBinding] = {}
        self._closed_sessions: weakref.WeakSet[AgentSession] = weakref.WeakSet()
        self._closed = False
        self._close_task: asyncio.Task[tuple[BaseException, ...]] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._operation_tasks: set[asyncio.Future[Any]] = set()
        self._pending_cleanups: set[asyncio.Future[None]] = set()

    @property
    def config(self) -> AgentRuntimeConfig:
        return self._config

    async def __aenter__(self) -> AgentRuntime:
        self._require_open()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def list_sessions(self, query: SessionQuery) -> SessionPage:
        return await self._run_owned_operation(lambda: self._list_sessions_owned(query))

    async def _list_sessions_owned(self, query: SessionQuery) -> SessionPage:
        self._require_open()
        adapter = self._adapter(query.backend, query.transport)
        adapter.validate_auth(query.auth)
        environment, _ = await self._environment(query.backend, query.auth, ())
        return await adapter.list_sessions(query, environment=environment)

    async def read_session(
        self, ref: AgentSessionRef, options: SessionReadOptions
    ) -> SessionSnapshot:
        return await self._run_owned_operation(lambda: self._read_session_owned(ref, options))

    async def _read_session_owned(
        self, ref: AgentSessionRef, options: SessionReadOptions
    ) -> SessionSnapshot:
        self._require_open()
        validate_read_session_auth(ref, options)
        adapter = self._adapter(ref.backend, ref.transport)
        adapter.validate_auth(options.auth)
        environment, state_root = await self._environment(ref.backend, options.auth, ())
        validate_session_ref(
            ref,
            backend=ref.backend,
            transport=ref.transport,
            profile_key=options.auth.profile_key,
            state_root_fingerprint=fingerprint_path(state_root),
            cwd="/",
            cwd_scopes_sessions=False,
        )
        return await adapter.read_session(ref, options, environment=environment)

    async def open_session(self, request: AgentSessionRequest) -> AgentSession:
        return await self._run_owned_operation(lambda: self._open_session_locked(request))

    async def _run_owned_operation(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        async with self._lifecycle_lock:
            self._require_open()
            task = asyncio.ensure_future(operation())
            self._operation_tasks.add(task)
            task.add_done_callback(self._operation_tasks.discard)
        try:
            result = await asyncio.shield(task)
            self._require_open()
            return result
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _open_session_locked(self, request: AgentSessionRequest) -> AgentSession:
        self._validate_directories(request)
        self._validate_content_files(
            request.system + request.developer,
            request,
            request.policy,
        )
        validate_mcp_network_policy(request.mcp_servers, request.policy)
        adapter = self._adapter(request.backend, request.transport)
        adapter.validate_auth(request.auth)
        environment, state_root = await self._environment(
            request.backend, request.auth, request.policy.environment
        )
        state_root_fingerprint = fingerprint_path(state_root)
        if isinstance(request.open, ResumeSession | ForkSession):
            validate_session_ref(
                request.open.ref,
                backend=request.backend,
                transport=request.transport,
                profile_key=request.auth.profile_key,
                state_root_fingerprint=state_root_fingerprint,
                cwd=request.cwd,
                cwd_scopes_sessions=adapter.cwd_scopes_sessions,
            )
        environment.update(
            await self._mcp_environment(
                request.backend,
                request.mcp_servers,
                protected_environment=frozenset(environment),
            )
        )
        request = self._resolve_mcp_commands(request)
        session = await adapter.open_session(request, environment=environment)
        self._require_open()
        if not isinstance(session, AgentSession) or session in self._sessions:
            raise ProtocolDefect(
                "adapter returned an invalid or reused AgentSession",
                code="invalid_adapter_session",
            )
        binding = _SessionBinding(
            adapter=adapter,
            request=request,
            state_root_fingerprint=state_root_fingerprint,
        )
        if session.ref_is_complete:
            self._validate_adapter_ref(session.ref, binding, adapter.cwd_scopes_sessions)
            binding.ref_validated = True
        self._sessions[session] = binding
        return session

    async def run_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None = None,
        cancel: CancelSignal | None = None,
    ) -> AgentTerminal:
        terminal: AgentEvent | None = None
        async for event in self.stream_turn(session, request, approvals=approvals, cancel=cancel):
            terminal = event
        if not isinstance(terminal, AgentTerminal):
            raise ProtocolDefect(
                "validated stream returned no terminal",
                code="missing_terminal_projection",
            )
        return terminal

    async def close_session(self, session: AgentSession) -> None:
        """Release one owned native session without affecting sibling sessions.

        Closing is idempotent for a handle owned by this runtime. Concurrent callers share
        one cleanup task; a handle from another runtime remains a scope error.
        """
        if not isinstance(session, AgentSession):
            raise InvalidAgentRequest("close_session requires AgentSession")
        runtime_closing = False
        async with self._lifecycle_lock:
            binding = self._sessions.get(session)
            if binding is None:
                if session in self._closed_sessions:
                    return
                raise SessionMismatch("AgentSession does not belong to this AgentRuntime")
            if self._closed:
                runtime_closing = True
                task = None
            else:
                session.begin_close()
                binding.invalidated = True
                if binding.consumer_terminal_received is not None:
                    binding.consumer_terminal_received.set()
                if binding.close_task is None:
                    binding.close_task = asyncio.create_task(
                        self._close_session_owned(session, binding)
                    )
                    self._operation_tasks.add(binding.close_task)
                    binding.close_task.add_done_callback(self._operation_tasks.discard)
                task = binding.close_task
        if runtime_closing:
            await self.close()
            return
        if task is None:
            raise ProtocolDefect("session close task was not created", code="session_close_missing")
        try:
            cleanup_errors = await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise
        if cleanup_errors:
            raise ProtocolDefect(
                "agent session cleanup failed",
                code="agent_session_cleanup_failed",
            )

    async def _close_session_owned(
        self, session: AgentSession, binding: _SessionBinding
    ) -> tuple[BaseException, ...]:
        cleanup_errors: list[BaseException] = []
        if binding.turn_active:
            result = await self._bounded_cleanup_call(
                self._interrupt_once(binding, session),
                "agent session interruption exceeded its cleanup bound",
                cancel_when_late=True,
            )
            if result is not None:
                cleanup_errors.append(result)
        driver = binding.turn_driver
        if driver is not None and not driver.done():
            driver.cancel()
            done, _ = await asyncio.wait(
                {driver}, timeout=self._bounded(_TURN_CLEANUP_TIMEOUT_SECONDS)
            )
            if driver not in done:
                cleanup_errors.append(
                    ProtocolDefect(
                        "agent turn driver did not settle during session close",
                        code="agent_session_cleanup_failed",
                    )
                )
        result = await self._bounded_cleanup_call(
            binding.adapter.close_session(session),
            "agent session close exceeded its cleanup bound",
            cancel_when_late=False,
        )
        if result is not None:
            cleanup_errors.append(result)
        if binding.turn_active:
            binding.turn_active = False
            session.end_turn()
        async with self._lifecycle_lock:
            if self._sessions.get(session) is binding:
                self._sessions.pop(session)
                self._closed_sessions.add(session)
        return tuple(cleanup_errors)

    async def stream_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None = None,
        cancel: CancelSignal | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        async with self._lifecycle_lock:
            self._require_open()
            binding = self._sessions.get(session)
            if binding is None:
                if session in self._closed_sessions:
                    raise SessionUnavailable("AgentSession has been closed")
                raise SessionMismatch("AgentSession does not belong to this AgentRuntime")
            if binding.invalidated:
                raise SessionUnavailable("AgentSession native client is no longer live")
            policy = binding.request.policy
            if request.policy is not None:
                # Narrowing-only: a per-turn patch may only ever reduce what the session's
                # policy already allows. The adapter still decides whether it can apply the
                # narrowed policy at all.
                policy = narrow_policy(policy, request.policy)
                # MCP is session-scoped, so a narrowed turn policy must still admit the
                # servers the session opened with.
                validate_mcp_network_policy(binding.request.mcp_servers, policy)
            if policy.approval == "ask" and approvals is None:
                raise InvalidAgentRequest("approval ask policy requires an approval handler")
            if cancel is not None and cancel.is_set():
                raise TurnNotStarted("cancelled")
            self._validate_content_files(request.input, binding.request, policy)

            session.begin_turn()
            binding.turn_active = True
            binding.interrupt_requested = False
            binding.turn_error = None
            binding.forced_terminal = None
            queue: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=256)
            terminal_received = asyncio.Event()
            binding.consumer_terminal_received = terminal_received
            driver = asyncio.create_task(
                self._drive_turn(
                    binding,
                    session,
                    request,
                    approvals=approvals,
                    cancel=cancel,
                    queue=queue,
                    terminal_received=terminal_received,
                )
            )
            binding.turn_driver = driver
        try:
            while True:
                if queue.empty() and driver.done():
                    if binding.forced_terminal is not None:
                        event = binding.forced_terminal
                        binding.forced_terminal = None
                        yield event
                    if binding.turn_error is not None:
                        raise binding.turn_error
                    return
                next_event = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {next_event, driver}, return_when=asyncio.FIRST_COMPLETED
                )
                if next_event not in done:
                    next_event.cancel()
                    await asyncio.gather(next_event, return_exceptions=True)
                    continue
                event = next_event.result()
                if isinstance(event, AgentTerminal):
                    terminal_received.set()
                yield event
        finally:
            terminal_received.set()
            if not driver.done():
                driver.cancel()
            try:
                async with asyncio.timeout(_TURN_CLEANUP_TIMEOUT_SECONDS):
                    await asyncio.shield(driver)
            except TimeoutError:
                binding.invalidated = True
                try:
                    await binding.adapter.close()
                except BaseException:
                    binding.turn_error = ProtocolDefect(
                        "agent adapter hard-close failed",
                        code="agent_turn_cleanup_failed",
                    )
                for candidate in self._sessions.values():
                    if candidate.adapter is binding.adapter:
                        candidate.invalidated = True
                driver.cancel()
                try:
                    async with asyncio.timeout(_TURN_CLEANUP_TIMEOUT_SECONDS):
                        await asyncio.shield(driver)
                except BaseException:
                    binding.turn_error = ProtocolDefect(
                        "agent turn cleanup exceeded its hard bound",
                        code="agent_turn_cleanup_failed",
                    )
            except asyncio.CancelledError:
                pass
            except BaseException:
                if binding.turn_error is None:
                    binding.turn_error = ProtocolDefect(
                        "agent turn driver failed during cleanup",
                        code="agent_turn_cleanup_failed",
                    )
            if binding.turn_error is not None and not isinstance(
                binding.turn_error, asyncio.CancelledError
            ):
                raise binding.turn_error

    async def _drive_turn(
        self,
        binding: _SessionBinding,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None,
        cancel: CancelSignal | None,
        queue: asyncio.Queue[AgentEvent],
        terminal_received: asyncio.Event,
    ) -> None:
        source = binding.adapter.stream_turn(session, request, approvals=approvals)
        interrupted: AsyncGenerator[AgentEvent, None] | None = None
        primary_error: BaseException | None = None
        last_enqueued: AgentEvent | None = None
        text_parts: list[str] = []
        usage: Presence[TokenUsage] = Absent()
        terminal_enqueued = False
        hard_teardown = False
        try:
            timeout_seconds = self._config.max_turn_seconds
            if request.timeout_seconds is not None:
                timeout_seconds = min(timeout_seconds, request.timeout_seconds)
            interrupted = self._interruptible_stream(
                binding,
                session,
                source,
                cancel=cancel,
                timeout_seconds=timeout_seconds,
            )
            async for event in validate_event_stream(interrupted):
                self._observe_session_event(binding, session, event)
                if isinstance(event, AgentText):
                    text_parts.append(event.text)
                elif isinstance(event, AgentUsage):
                    usage = Present(event.usage)
                await queue.put(event)
                last_enqueued = event
                if isinstance(event, AgentTerminal):
                    terminal_enqueued = True
                    await terminal_received.wait()
        except BaseException as error:
            primary_error = error
            hard_teardown = isinstance(error, ProtocolDefect)
            if binding.invalidated:
                if last_enqueued is None:
                    binding.turn_error = TurnNotStarted("cancelled")
                elif not terminal_enqueued:
                    binding.forced_terminal = self._synthetic_terminal(
                        session,
                        "cancelled",
                        final_text="".join(text_parts),
                        usage=usage,
                    )
            else:
                binding.turn_error = error
        finally:
            cleanup_errors: list[BaseException] = []
            if interrupted is not None:
                try:
                    await interrupted.aclose()
                except BaseException as error:
                    cleanup_errors.append(error)
            try:
                await source.aclose()
            except BaseException as error:
                cleanup_errors.append(error)
            was_active = binding.turn_active
            binding.turn_active = False
            if was_active:
                session.end_turn()
            if cleanup_errors or hard_teardown:
                binding.invalidated = True
                try:
                    await binding.adapter.close()
                except BaseException as error:
                    cleanup_errors.append(error)
                for candidate in self._sessions.values():
                    if candidate.adapter is binding.adapter:
                        candidate.invalidated = True
                if cleanup_errors:
                    binding.turn_error = ProtocolDefect(
                        "agent turn cleanup failed",
                        code="agent_turn_cleanup_failed",
                    )
            elif primary_error is None:
                binding.turn_error = None

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._close_task is None:
                self._closed = True
                self._close_task = asyncio.create_task(self._close_owned(asyncio.current_task()))
            close_task = self._close_task
        try:
            cleanup_errors = await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await asyncio.gather(close_task, return_exceptions=True)
            raise
        except Exception:
            raise ProtocolDefect(
                "agent runtime cleanup failed",
                code="agent_runtime_cleanup_failed",
            ) from None
        for error in cleanup_errors:
            if isinstance(error, asyncio.CancelledError):
                raise error
        if cleanup_errors:
            raise ProtocolDefect(
                "agent runtime cleanup failed",
                code="agent_runtime_cleanup_failed",
            )

    async def _close_owned(self, caller: asyncio.Task[Any] | None) -> tuple[BaseException, ...]:
        operations = tuple(self._operation_tasks)
        cleanup_errors: list[BaseException] = []
        if operations:
            _, pending = await asyncio.wait(
                operations,
                timeout=self._bounded(_CLOSE_DRAIN_TIMEOUT_SECONDS),
            )
            for task in pending:
                task.cancel()
            if pending:
                _, still_pending = await asyncio.wait(
                    pending,
                    timeout=self._bounded(_CLOSE_SETTLE_TIMEOUT_SECONDS),
                )
                if still_pending:
                    cleanup_errors.append(
                        ProtocolDefect(
                            "runtime operation did not settle after cancellation",
                            code="agent_runtime_cleanup_failed",
                        )
                    )
        cleanup_errors.extend(await self._close_owned_locked(caller))
        return tuple(cleanup_errors)

    async def _close_owned_locked(
        self, caller: asyncio.Task[Any] | None
    ) -> tuple[BaseException, ...]:
        active_tasks = {
            binding.turn_driver
            for binding in self._sessions.values()
            if binding.turn_driver is not None and binding.turn_driver is not caller
        }
        for binding in self._sessions.values():
            binding.invalidated = True
            if binding.consumer_terminal_received is not None:
                binding.consumer_terminal_received.set()
        active = [
            self._bounded_cleanup_call(
                self._interrupt_once(binding, session),
                "agent interruption exceeded its cleanup bound",
                cancel_when_late=True,
            )
            for session, binding in self._sessions.items()
            if binding.turn_active
        ]
        cleanup_errors: list[BaseException] = []
        if active:
            interrupt_results = await asyncio.gather(*active, return_exceptions=True)
            cleanup_errors.extend(
                result for result in interrupt_results if isinstance(result, BaseException)
            )
        if active_tasks:
            _, pending = await asyncio.wait(
                active_tasks,
                timeout=self._bounded(_CLOSE_DRAIN_TIMEOUT_SECONDS),
            )
            for task in pending:
                task.cancel()
            if pending:
                _, still_pending = await asyncio.wait(
                    pending,
                    timeout=self._bounded(_CLOSE_SETTLE_TIMEOUT_SECONDS),
                )
                if still_pending:
                    cleanup_errors.append(
                        ProtocolDefect(
                            "agent turn driver did not settle after cancellation",
                            code="agent_turn_cleanup_failed",
                        )
                    )
        for session, binding in self._sessions.items():
            binding.turn_driver = None
            if binding.turn_active:
                cleanup_errors.append(
                    ProtocolDefect(
                        "agent turn driver did not release its session",
                        code="agent_turn_cleanup_failed",
                    )
                )
                binding.turn_active = False
                session.end_turn()
        close_results = await asyncio.gather(
            *(
                self._bounded_cleanup_call(
                    adapter.close(),
                    "agent adapter close exceeded its cleanup bound",
                    cancel_when_late=False,
                )
                for adapter in self._adapters.values()
            ),
            return_exceptions=True,
        )
        cleanup_errors.extend(
            result for result in close_results if isinstance(result, BaseException)
        )
        self._closed_sessions.update(self._sessions)
        self._sessions.clear()
        return tuple(cleanup_errors)

    def _bounded(self, budget: float) -> float:
        return min(self._config.max_turn_seconds, budget)

    async def _bounded_cleanup_call(
        self, operation: Awaitable[None], message: str, *, cancel_when_late: bool
    ) -> BaseException | None:
        """Bound how long close waits, not how long teardown is allowed to run.

        `cancel_when_late=False` is mandatory for adapter teardown: an adapter's `close()`
        is what terminates and reaps the owned process groups, so cancelling it mid-loop
        would orphan every child it had not reached yet. Lateness is reported as a defect
        and the call is retained to completion instead.
        """
        task = asyncio.ensure_future(operation)
        try:
            done, _ = await asyncio.wait(
                {task}, timeout=self._bounded(_TURN_CLEANUP_TIMEOUT_SECONDS)
            )
        except BaseException:
            self._abandon_cleanup_task(task, cancel=cancel_when_late)
            raise
        if task not in done:
            self._abandon_cleanup_task(task, cancel=cancel_when_late)
            return ProtocolDefect(message, code="agent_runtime_cleanup_failed")
        try:
            task.result()
        except BaseException as error:
            return error
        return None

    def _abandon_cleanup_task(self, task: asyncio.Future[None], *, cancel: bool) -> None:
        if cancel:
            task.cancel()
        else:
            # asyncio keeps only a weak reference to running tasks; hold one so a teardown
            # that outran its reporting deadline still finishes reaping its process groups.
            self._pending_cleanups.add(task)
            task.add_done_callback(self._pending_cleanups.discard)
        task.add_done_callback(self._consume_cleanup_result)

    @staticmethod
    def _consume_cleanup_result(task: asyncio.Future[None]) -> None:
        try:
            task.result()
        except BaseException:
            pass

    def _adapter(self, backend: Backend, transport: AgentTransport) -> AgentAdapter:
        adapter = self._adapters.get((backend, transport))
        if adapter is None:
            raise SdkUnavailable(f"no SDK adapter is registered for {backend!r}")
        return adapter

    async def _environment(
        self,
        backend: Backend,
        credential: CredentialRef,
        allowed_environment: tuple[str, ...],
    ) -> tuple[dict[str, str], Path]:
        try:
            state_root = resolve_state_root(self._config.state_root_base, backend, credential)
        except OSError:
            raise CredentialUnavailable("profile state root could not be resolved") from None
        environment = build_child_environment(
            AuthEnvironmentRequest(
                backend=backend,
                credential=credential,
                inherited_environment=dict(os.environ),
                allowed_environment=allowed_environment,
                state_root=state_root,
            )
        )
        self._secure_state_root(state_root)
        return environment, state_root

    def _secure_state_root(self, state_root: Path) -> None:
        """Create every component the runtime owns at 0700; `mkdir(parents=True)` does not."""
        # `resolve_state_root` builds the state root under the *resolved* base, so containment
        # has to be decided against the resolved base too: a base reached through a symlink
        # would otherwise contain none of its own components and nothing would be created.
        try:
            base = self._config.state_root_base.resolve()
            base_mode = stat.S_IMODE(base.stat().st_mode)
        except OSError:
            raise CredentialUnavailable("profile state root base could not be inspected") from None
        if base_mode & 0o022:
            raise InvalidAgentRequest("state_root_base must not be group- or world-writable")
        owned = [
            component
            for component in (
                *reversed(state_root.parents),
                state_root,
                child_home_directory(state_root),
            )
            if component != base and component.is_relative_to(base)
        ]
        try:
            for component in owned:
                component.mkdir(mode=_STATE_ROOT_MODE, exist_ok=True)
                if component.resolve() != component:
                    raise InvalidAgentRequest("profile state root became a symlink alias")
                component.chmod(_STATE_ROOT_MODE)
                if stat.S_IMODE(component.stat().st_mode) != _STATE_ROOT_MODE:
                    raise InvalidAgentRequest("profile state root is not privately permissioned")
        except OSError:
            raise CredentialUnavailable("profile state root could not be secured") from None

    async def _mcp_environment(
        self,
        backend: Backend,
        servers: tuple[McpServerSpec, ...],
        *,
        protected_environment: frozenset[str],
    ) -> dict[str, str]:
        environment: dict[str, str] = {}
        destinations: dict[str, CredentialRef] = {}
        values: dict[CredentialRef, str] = {}
        assignments: list[tuple[str, CredentialRef]] = []
        reserved = frozenset(
            (*credential_environment_names("codex"), *credential_environment_names("claude"))
        )
        for server in servers:
            for reference in server.environment_refs:
                if reference.name in reserved:
                    raise McpConfigurationError(
                        "MCP environment destination is a provider credential variable"
                    )
                if reference.name in protected_environment or is_runtime_owned_environment_name(
                    reference.name
                ):
                    raise McpConfigurationError(
                        "MCP destination cannot overwrite the agent process environment"
                    )
                self._claim_mcp_destination(destinations, reference.name, reference.source)
                assignments.append((reference.name, reference.source))
            for reference in server.header_refs:
                alias = mcp_header_environment_name(backend, reference.source)
                self._claim_mcp_destination(destinations, alias, reference.source)
                assignments.append((alias, reference.source))
        for destination, source in assignments:
            environment[destination] = await self._mcp_credential_value(source, values)
        return environment

    @staticmethod
    def _resolve_mcp_commands(request: AgentSessionRequest) -> AgentSessionRequest:
        resolved_servers: list[McpServerSpec] = []
        changed = False
        for server in request.mcp_servers:
            if server.transport != "stdio":
                resolved_servers.append(server)
                continue
            command = server.command
            if command is None:
                raise ProtocolDefect("validated stdio MCP server has no command")
            resolved = shutil.which(command)
            if resolved is None:
                raise McpConfigurationError(f"MCP executable {command!r} is unavailable")
            absolute = str(Path(resolved).resolve())
            resolved_servers.append(replace(server, command=absolute))
            changed = changed or absolute != command
        if not changed:
            return request
        return replace(request, mcp_servers=tuple(resolved_servers))

    async def _mcp_credential_value(
        self,
        source: CredentialRef,
        values: dict[CredentialRef, str],
    ) -> str:
        cached = values.get(source)
        if cached is not None:
            return cached
        if source.kind == "local_account" or source.name is None:
            raise McpConfigurationError("MCP credentials require a named source")
        if source.kind == "api_key_environment":
            value = os.environ.get(source.name)
            if not value:
                raise CredentialUnavailable("MCP credential environment source is unavailable")
        else:
            resolver = self._config.secret_resolver
            if resolver is None:
                raise UnsupportedCapability(
                    "MCP secret_reference requires an AgentRuntimeConfig.secret_resolver"
                )
            try:
                value = await resolver(source.name)
            except asyncio.CancelledError:
                raise
            except CredentialUnavailable:
                raise
            except Exception:
                raise CredentialUnavailable("MCP secret_reference resolution failed") from None
            if type(value) is not str or not value:
                raise CredentialUnavailable("MCP secret_reference resolution was empty")
        values[source] = value
        return value

    @staticmethod
    def _claim_mcp_destination(
        destinations: dict[str, CredentialRef],
        destination: str,
        source: CredentialRef,
    ) -> None:
        existing = destinations.get(destination)
        if existing is not None and existing != source:
            raise McpConfigurationError(
                "MCP credential destination is claimed by different sources"
            )
        destinations[destination] = source

    async def _interruptible_stream(
        self,
        binding: _SessionBinding,
        session: AgentSession,
        source: AsyncGenerator[AgentEvent, None],
        *,
        cancel: CancelSignal | None,
        timeout_seconds: float,
    ) -> AsyncGenerator[AgentEvent, None]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        started = False
        text_parts: list[str] = []
        usage: Presence[TokenUsage] = Absent()
        next_task: asyncio.Future[AgentEvent] | None = None
        stop_task = asyncio.create_task(
            self._watch_turn_stop(
                binding,
                session,
                cancel=cancel,
                deadline=deadline,
            )
        )
        terminal_seen = False
        try:
            while True:
                next_task = asyncio.ensure_future(anext(source))
                done, _ = await asyncio.wait(
                    {next_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if stop_task not in done and next_task in done:
                    try:
                        event = next_task.result()
                    except StopAsyncIteration:
                        return
                    started = True
                    if isinstance(event, AgentText):
                        text_parts.append(event.text)
                    elif isinstance(event, AgentUsage):
                        usage = Present(event.usage)
                    terminal_seen = isinstance(event, AgentTerminal)
                    if terminal_seen:
                        await self._settle({stop_task})
                    yield event
                    if terminal_seen:
                        return
                    continue
                cause = stop_task.result()
                next_task.cancel()
                await self._settle({next_task})
                if not started:
                    await source.aclose()
                    raise TurnNotStarted("turn_timeout" if cause == "timed_out" else "cancelled")
                await source.aclose()
                final_text = "".join(text_parts)
                if isinstance(binding.adapter, _InterruptedFinalTextAdapter):
                    final_text = binding.adapter._interrupted_final_text(session)
                yield self._synthetic_terminal(
                    session,
                    cause,
                    final_text=final_text,
                    usage=usage,
                )
                return
        finally:
            cleanup_errors: list[BaseException] = []
            try:
                if not terminal_seen:
                    await self._interrupt_once(binding, session)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                if next_task is not None and not next_task.done():
                    await self._settle((next_task,))
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                await self._settle((stop_task,))
            except BaseException as error:
                cleanup_errors.append(error)
            if cleanup_errors:
                if isinstance(cleanup_errors[0], asyncio.CancelledError):
                    raise cleanup_errors[0]
                raise ProtocolDefect(
                    "agent turn interruption cleanup failed",
                    code="agent_turn_cleanup_failed",
                ) from None

    async def _watch_turn_stop(
        self,
        binding: _SessionBinding,
        session: AgentSession,
        *,
        cancel: CancelSignal | None,
        deadline: float,
    ) -> StopCause:
        loop = asyncio.get_running_loop()
        cancel_task = asyncio.create_task(cancel.wait()) if cancel is not None else None
        timeout_task = asyncio.create_task(asyncio.sleep(max(0.0, deadline - loop.time())))
        tasks: set[asyncio.Future[Any]] = {timeout_task}
        if cancel_task is not None:
            tasks.add(cancel_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            cause: StopCause = "cancelled" if cancel_task in done else "timed_out"
            await self._interrupt_once(binding, session)
            return cause
        finally:
            await self._settle(tasks)

    @staticmethod
    async def _interrupt_once(binding: _SessionBinding, session: AgentSession) -> None:
        if binding.interrupt_requested:
            return
        binding.interrupt_requested = True
        try:
            await binding.adapter.interrupt(session)
        except BaseException:
            binding.interrupt_requested = False
            raise

    @staticmethod
    def _synthetic_terminal(
        session: AgentSession,
        cause: StopCause,
        *,
        final_text: str,
        usage: Presence[TokenUsage],
    ) -> AgentTerminal:
        match cause:
            case "cancelled":
                return AgentTerminal(
                    status="cancelled",
                    failure=None,
                    final_text=final_text,
                    session_ref=session.ref,
                    usage=usage,
                )
            case "timed_out":
                return AgentTerminal(
                    status="failed",
                    failure=AgentFailure("turn_timeout"),
                    final_text=final_text,
                    session_ref=session.ref,
                    usage=usage,
                )
            case _:
                assert_never(cause)

    def _observe_session_event(
        self, binding: _SessionBinding, session: AgentSession, event: AgentEvent
    ) -> None:
        """An adapter must complete and keep one consistent session ref before it streams."""
        if not session.ref_is_complete:
            raise ProtocolDefect(
                "adapter streamed an event before completing the session ref",
                code="incomplete_session_ref",
            )
        if not binding.ref_validated:
            self._validate_adapter_ref(session.ref, binding, binding.adapter.cwd_scopes_sessions)
            binding.ref_validated = True
        if isinstance(event, AgentTerminal) and event.session_ref != session.ref:
            raise ProtocolDefect(
                "terminal session ref does not match AgentSession.ref",
                code="session_ref_changed",
            )

    @staticmethod
    def _validate_adapter_ref(
        ref: AgentSessionRef,
        binding: _SessionBinding,
        cwd_scopes_sessions: bool,
    ) -> None:
        request = binding.request
        if (
            ref.backend != request.backend
            or ref.transport != request.transport
            or ref.profile_key != request.auth.profile_key
            or ref.state_root_fingerprint != binding.state_root_fingerprint
            or (cwd_scopes_sessions and ref.cwd_fingerprint != fingerprint_path(request.cwd))
        ):
            raise ProtocolDefect(
                "adapter returned a session ref inconsistent with its request",
                code="adapter_session_ref_mismatch",
            )

    @staticmethod
    def _resolve_authorized(path: str, field: str) -> Path:
        """Resolve one caller path before launch; types.py validated it lexically only."""
        try:
            return Path(path).resolve()
        except OSError:
            raise InvalidAgentRequest(f"{field} could not be resolved: {path!r}") from None

    @staticmethod
    def _validate_directories(request: AgentSessionRequest) -> None:
        for path in (request.cwd, *request.additional_dirs):
            resolved = AgentRuntime._resolve_authorized(path, "agent directory")
            try:
                valid = resolved.is_dir()
            except OSError:
                valid = False
            if not valid:
                raise InvalidAgentRequest(f"agent directory does not exist: {path!r}")

    @staticmethod
    def _validate_content_files(
        parts: tuple[ContentPart, ...],
        request: AgentSessionRequest,
        policy: PermissionPolicy,
    ) -> None:
        roots = tuple(
            AgentRuntime._resolve_authorized(path, "agent directory")
            for path in (request.cwd, *request.additional_dirs)
        )
        for part in parts:
            match part:
                case TextContent():
                    continue
                case ImageContent() | FileContent():
                    pass
                case _:
                    assert_never(part)
            # Containment is decided on resolved paths on both sides, so neither a symlinked
            # root nor a symlinked attachment can widen the authorized set.
            path = AgentRuntime._resolve_authorized(part.path, "content file")
            try:
                stats = path.stat()
            except OSError:
                raise InvalidAgentRequest(f"content file does not exist: {part.path!r}") from None
            try:
                valid = path.is_file() and stats.st_size == part.size_bytes
            except OSError:
                valid = False
            if not valid:
                raise InvalidAgentRequest(
                    f"content file size/type does not match declaration: {part.path!r}"
                )
            if policy.filesystem != "full_access" and not any(
                path.is_relative_to(root) for root in roots
            ):
                raise InvalidAgentRequest(
                    f"content file is outside the authorized directories: {part.path!r}"
                )

    def _require_open(self) -> None:
        if self._closed:
            raise ProtocolDefect("AgentRuntime is closed", code="agent_runtime_closed")

    @staticmethod
    async def _settle(tasks: Collection[asyncio.Future[Any]]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _default_adapters(config: AgentRuntimeConfig) -> tuple[AgentAdapter, ...]:
    from .claude_sdk import ClaudeSdkAdapter
    from .codex_sdk import CodexSdkAdapter

    return (
        CodexSdkAdapter(),
        ClaudeSdkAdapter(executable=_resolve_executable(config.claude_executable)),
    )


def _resolve_executable(configured: str) -> str:
    resolved = shutil.which(configured)
    if resolved is None:
        return configured
    path = Path(resolved).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        return configured
    return str(path)


__all__ = [
    "AgentAdapter",
    "AgentRuntime",
    "AgentRuntimeConfig",
    "SecretResolver",
]
