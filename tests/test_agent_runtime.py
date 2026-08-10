from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path

import pytest

from provider_runtime.agent_runtime.errors import (
    ConcurrentTurn,
    CredentialUnavailable,
    ExecutableUnavailable,
    InvalidAgentRequest,
    McpConfigurationError,
    ProtocolDefect,
    SessionMismatch,
    SessionUnavailable,
    TurnNotStarted,
    UnsupportedCapability,
)
from provider_runtime.agent_runtime.events import (
    AgentEvent,
    AgentFailure,
    AgentPermissionRequest,
    AgentQuotaExhausted,
    AgentTerminal,
    AgentText,
    AgentUsage,
)
from provider_runtime.agent_runtime.policy import (
    PermissionPolicy,
    PermissionPolicyPatch,
    UnsafeConfirmation,
)
from provider_runtime.agent_runtime.runtime import AgentRuntime, AgentRuntimeConfig
from provider_runtime.agent_runtime.sessions import (
    AgentSession,
    SessionMetadata,
    SessionPage,
    SessionQuery,
    SessionReadOptions,
    SessionSnapshot,
    fingerprint_path,
)
from provider_runtime.agent_runtime.types import (
    AgentSessionRef,
    AgentSessionRequest,
    AgentTransport,
    ApprovalHandler,
    ApprovalRequest,
    Backend,
    CredentialRef,
    EnvironmentReference,
    FileContent,
    HeaderReference,
    McpServerSpec,
    NewSession,
    TextContent,
    TurnRequest,
)
from provider_runtime.types import Absent, Present, TokenUsage

pytestmark = pytest.mark.anyio


def usage() -> TokenUsage:
    return TokenUsage.from_components(
        input_tokens=7,
        output_tokens=2,
        total_tokens=Absent(),
        reasoning_tokens=Absent(),
        cache_read_input_tokens=Absent(),
        cache_write_input_tokens=Absent(),
    )


def session_ref(
    backend: Backend = "codex",
    transport: AgentTransport = "sdk",
    profile: str = "personal",
    *,
    state_root: Path | None = None,
    cwd: Path | None = None,
) -> AgentSessionRef:
    return AgentSessionRef(
        schema_version="agent-session-ref.v1",
        backend=backend,
        transport=transport,
        native_session_id="native-session-1",
        profile_key=profile,
        state_root_fingerprint=fingerprint_path(state_root) if state_root else "a" * 64,
        cwd_fingerprint=fingerprint_path(cwd) if cwd else "b" * 64,
    )


def request(
    tmp_path: Path,
    *,
    backend: Backend = "codex",
    transport: AgentTransport = "sdk",
) -> AgentSessionRequest:
    return AgentSessionRequest(
        backend=backend,
        transport=transport,
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=PermissionPolicy(),
    )


def turn(*, timeout_seconds: float | None = None) -> TurnRequest:
    return TurnRequest(input=(TextContent("hello"),), timeout_seconds=timeout_seconds)


def stdio_mcp_policy() -> PermissionPolicy:
    return PermissionPolicy(
        filesystem="full_access",
        network="unrestricted",
        unsafe_confirmation=UnsafeConfirmation(("filesystem_full_access", "network_unrestricted")),
    )


class Cancel:
    """Minimal CancelSignal double."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()


class ScriptedAdapter:
    cwd_scopes_sessions = False

    def __init__(
        self,
        *,
        backend: Backend = "codex",
        transport: AgentTransport = "sdk",
        lazy_ref: bool = False,
        incomplete_ref: bool = False,
        hang: bool = False,
        fail_open: bool = False,
        reject_auth: bool = False,
        bad_ref: bool = False,
        hang_before_first: bool = False,
        partial_before_hang: bool = False,
        request_approval: bool = False,
        quota_failure: bool = False,
        fail_close: bool = False,
        fail_interrupt: bool = False,
        hang_interrupt: bool = False,
        hang_close: bool = False,
    ) -> None:
        self._backend: Backend = backend
        self._transport: AgentTransport = transport
        self.lazy_ref = lazy_ref
        self.incomplete_ref = incomplete_ref
        self.hang = hang
        self.fail_open = fail_open
        self.reject_auth = reject_auth
        self.bad_ref = bad_ref
        self.hang_before_first = hang_before_first
        self.partial_before_hang = partial_before_hang
        self.request_approval = request_approval
        self.quota_failure = quota_failure
        self.fail_close = fail_close
        self.fail_interrupt = fail_interrupt
        self.hang_interrupt = hang_interrupt
        self.hang_close = hang_close
        self.auth_calls = 0
        self.open_calls = 0
        self.read_calls = 0
        self.interrupt_calls = 0
        self.close_session_calls = 0
        self.close_calls = 0
        self.stream_calls = 0
        self.last_environment: dict[str, str] = {}
        self.stream_entered = asyncio.Event()
        self.first_event_emitted = asyncio.Event()
        self.interrupt_release = asyncio.Event()
        self.close_release = asyncio.Event()
        self.release = asyncio.Event()
        # `stall` models a backend that produces nothing more until torn down. interrupt()
        # deliberately does NOT open it: a native interrupt stops the backend, it does not
        # make the stalled stream complete normally.
        self.stall = asyncio.Event()
        self.requests: dict[AgentSession, AgentSessionRequest] = {}
        self.refs: dict[AgentSession, AgentSessionRef] = {}
        self.closed_sessions: set[AgentSession] = set()

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def transport(self) -> AgentTransport:
        return self._transport

    def validate_auth(self, credential: CredentialRef) -> None:
        self.auth_calls += 1
        if self.reject_auth:
            raise UnsupportedCapability(f"unsupported scripted auth kind {credential.kind!r}")

    async def list_sessions(
        self, query: SessionQuery, *, environment: Mapping[str, str]
    ) -> SessionPage:
        self.last_environment = dict(environment)
        return SessionPage(sessions=())

    async def read_session(
        self,
        ref: AgentSessionRef,
        options: SessionReadOptions,
        *,
        environment: Mapping[str, str],
    ) -> SessionSnapshot:
        self.read_calls += 1
        self.last_environment = dict(environment)
        return SessionSnapshot(ref=ref, metadata=SessionMetadata())

    async def open_session(
        self,
        request: AgentSessionRequest,
        *,
        environment: Mapping[str, str],
    ) -> AgentSession:
        self.open_calls += 1
        self.last_environment = dict(environment)
        if self.fail_open:
            raise ExecutableUnavailable("scripted selected adapter is unavailable")
        state_root_name = "CODEX_HOME" if self.backend == "codex" else "CLAUDE_CONFIG_DIR"
        ref = session_ref(
            self.backend,
            self.transport,
            request.auth.profile_key,
            state_root=Path(environment[state_root_name]),
            cwd=Path(request.cwd),
        )
        if self.bad_ref:
            ref = session_ref(self.backend, self.transport, request.auth.profile_key)
        session = AgentSession(None if self.lazy_ref else ref)
        self.requests[session] = request
        self.refs[session] = ref
        return session

    async def stream_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.stream_calls += 1
        self.stream_entered.set()
        if self.hang_before_first:
            await self.release.wait()
        session_request = self.requests[session]
        ref = self.refs[session]
        if not session.ref_is_complete and not self.incomplete_ref:
            session.complete_ref(ref)
        if self.request_approval:
            approval = ApprovalRequest(operation="command", summary="run command")
            if approvals is None:
                raise AssertionError("scripted approval requires a handler")
            try:
                decision = await approvals(approval)
            except Exception:
                yield AgentPermissionRequest(approval, "deny")
                self.first_event_emitted.set()
                await self.interrupt(session)
                yield AgentTerminal(
                    status="failed",
                    failure=AgentFailure("approval_unanswered"),
                    final_text="",
                    session_ref=ref,
                    diagnostics=("approval handler failed",),
                )
                return
            yield AgentPermissionRequest(approval, decision)
            self.first_event_emitted.set()
            await self.release.wait()
        if self.partial_before_hang:
            yield AgentText("partial")
            self.first_event_emitted.set()
            yield AgentUsage(usage())
        if self.hang:
            await self.stall.wait()
        if self.quota_failure:
            yield AgentTerminal(
                status="failed",
                failure=AgentQuotaExhausted(),
                final_text="",
                session_ref=ref,
            )
            return
        yield AgentText(session_request.cwd)
        self.first_event_emitted.set()
        yield AgentTerminal(
            status="succeeded",
            failure=None,
            final_text="done",
            session_ref=ref,
        )

    async def interrupt(self, session: AgentSession) -> None:
        self.interrupt_calls += 1
        if self.fail_interrupt:
            raise ProtocolDefect("scripted interrupt failed")
        if self.hang_interrupt:
            await self._wait_ignoring_cancellation(self.interrupt_release)
        self.release.set()

    async def close_session(self, session: AgentSession) -> None:
        if session in self.closed_sessions:
            return
        if session not in self.requests:
            raise InvalidAgentRequest("session is not owned by the scripted adapter")
        self.close_session_calls += 1
        self.closed_sessions.add(session)
        self.release.set()
        self.requests.pop(session, None)
        self.refs.pop(session, None)
        if self.fail_close:
            raise RuntimeError("adapter session cleanup secret detail")

    async def close(self) -> None:
        self.close_calls += 1
        self.release.set()
        if self.hang_close:
            await self._wait_ignoring_cancellation(self.close_release)
        if self.fail_close:
            raise RuntimeError("adapter cleanup secret detail")

    @staticmethod
    async def _wait_ignoring_cancellation(release: asyncio.Event) -> None:
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue


def runtime_for(
    tmp_path: Path, *adapters: ScriptedAdapter, max_turn_seconds: float = 5.0
) -> AgentRuntime:
    return AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, max_turn_seconds=max_turn_seconds),
        adapters=adapters,
    )


async def test_runtime_opens_streams_and_projects_one_terminal_result(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        events = [event async for event in runtime.stream_turn(session, turn())]
        result = await runtime.run_turn(session, turn())
        await runtime.close_session(session)

    assert [type(event).__name__ for event in events] == ["AgentText", "AgentTerminal"], (
        f"unexpected stream shape: {events}"
    )
    assert isinstance(events[-1], AgentTerminal)
    assert result.status == "succeeded"
    assert result.final_text == "done"
    assert result.session_ref == session.ref


async def test_quota_exhaustion_is_the_named_terminal_value(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(quota_failure=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        result = await runtime.run_turn(session, turn())

    assert result.status == "failed"
    assert result.failure == AgentQuotaExhausted(), (
        f"pool exhaustion must surface as AgentQuotaExhausted, got {result.failure!r}"
    )


async def test_close_session_is_idempotent_and_preserves_sibling_sessions(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    async with runtime_for(tmp_path, adapter) as runtime:
        first = await runtime.open_session(request(tmp_path))
        second = await runtime.open_session(request(tmp_path))
        await runtime.close_session(first)
        await runtime.close_session(first)
        with pytest.raises(SessionUnavailable):
            await runtime.run_turn(first, turn())
        result = await runtime.run_turn(second, turn())

    assert adapter.close_session_calls == 1
    assert result.status == "succeeded"


async def test_close_session_interrupts_and_settles_an_active_turn(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(partial_before_hang=True, hang=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        stream = runtime.stream_turn(session, turn())
        first = await anext(stream)
        assert first == AgentText("partial")
        await runtime.close_session(session)
        await stream.aclose()

    assert adapter.interrupt_calls >= 1
    assert adapter.close_session_calls == 1


async def test_close_session_rejects_a_foreign_handle(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    async with runtime_for(tmp_path, adapter) as runtime:
        with pytest.raises(SessionMismatch):
            await runtime.close_session(AgentSession(session_ref()))


async def test_runtime_rejects_concurrent_turn_without_interleaving(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(partial_before_hang=True, hang=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        stream = runtime.stream_turn(session, turn())
        await anext(stream)
        with pytest.raises(ConcurrentTurn):
            await runtime.run_turn(session, turn())
        adapter.stall.set()
        await stream.aclose()
        await runtime.close_session(session)


async def test_cancel_interrupts_adapter_and_synthesizes_cancelled_terminal(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(partial_before_hang=True, hang=True)
    cancel = Cancel()
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        events: list[AgentEvent] = []
        stream = runtime.stream_turn(session, turn(), cancel=cancel)
        events.append(await anext(stream))
        events.append(await anext(stream))
        cancel.set()
        async for event in stream:
            events.append(event)

    terminal = events[-1]
    assert isinstance(terminal, AgentTerminal), f"stream must end in a terminal: {events}"
    assert terminal.status == "cancelled"
    assert terminal.final_text == "partial", (
        "the synthetic terminal must preserve the text the consumer already received"
    )
    assert terminal.usage == Present(usage()), (
        "the synthetic terminal must carry the latest observed usage"
    )
    assert adapter.interrupt_calls >= 1


async def test_timeout_is_a_typed_terminal_failure(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(partial_before_hang=True, hang=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        result = await runtime.run_turn(session, turn(timeout_seconds=0.05))

    assert result.status == "failed"
    assert result.failure == AgentFailure("turn_timeout")
    assert adapter.interrupt_calls >= 1


async def test_timeout_before_any_event_is_turn_not_started(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(hang_before_first=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        with pytest.raises(TurnNotStarted) as caught:
            await runtime.run_turn(session, turn(timeout_seconds=0.05))
        adapter.release.set()

    assert caught.value.reason == "turn_timeout"


async def test_pre_set_cancel_is_typed_and_rejected_before_adapter_effect(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter()
    cancel = Cancel()
    cancel.set()
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        with pytest.raises(TurnNotStarted) as caught:
            await runtime.run_turn(session, turn(), cancel=cancel)

    assert caught.value.reason == "cancelled"
    assert adapter.stream_calls == 0, "a pre-set cancel must never reach the adapter stream"


async def test_selected_adapter_failure_never_invokes_another_transport(tmp_path: Path) -> None:
    codex = ScriptedAdapter(backend="codex", fail_open=True)
    claude = ScriptedAdapter(backend="claude")
    async with runtime_for(tmp_path, codex, claude) as runtime:
        with pytest.raises(ExecutableUnavailable):
            await runtime.open_session(request(tmp_path, backend="codex"))

    assert codex.open_calls == 1
    assert claude.open_calls == 0
    assert claude.stream_calls == 0


async def test_ask_without_handler_is_rejected_before_adapter_stream(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    ask_request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=PermissionPolicy(approval="ask"),
    )
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(ask_request)
        with pytest.raises(InvalidAgentRequest, match="approval handler"):
            await runtime.run_turn(session, turn())

    assert adapter.stream_calls == 0


async def test_turn_policy_patch_may_only_narrow(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        widened = TurnRequest(
            input=(TextContent("hello"),),
            policy=PermissionPolicyPatch(filesystem="workspace_write"),
        )
        with pytest.raises(InvalidAgentRequest, match="cannot widen"):
            await runtime.run_turn(session, widened)

    assert adapter.stream_calls == 0, "a widening patch must be refused before the adapter runs"


async def test_approval_handler_exception_becomes_typed_failure(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(request_approval=True)
    ask_request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=PermissionPolicy(approval="ask"),
    )

    async def broken_handler(_request: ApprovalRequest) -> str:
        raise RuntimeError("handler crashed")

    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(ask_request)
        events = [
            event
            async for event in runtime.stream_turn(
                session,
                turn(),
                approvals=broken_handler,  # type: ignore[arg-type]
            )
        ]

    permission = events[0]
    assert isinstance(permission, AgentPermissionRequest)
    assert permission.decision == "deny"
    terminal = events[-1]
    assert isinstance(terminal, AgentTerminal)
    assert terminal.failure == AgentFailure("approval_unanswered")


async def test_missing_cwd_is_rejected_before_adapter_open(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    missing = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str((tmp_path / "absent").resolve()),
        policy=PermissionPolicy(),
    )
    async with runtime_for(tmp_path, adapter) as runtime:
        with pytest.raises(InvalidAgentRequest, match="does not exist"):
            await runtime.open_session(missing)

    assert adapter.open_calls == 0


async def test_file_size_is_statted_before_adapter_effect(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    payload = tmp_path / "input.txt"
    payload.write_text("1234")
    wrong_size = TurnRequest(
        input=(
            TextContent("hello"),
            FileContent(path=str(payload.resolve()), size_bytes=99, media_type="text/plain"),
        ),
    )
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        with pytest.raises(InvalidAgentRequest, match="size/type"):
            await runtime.run_turn(session, wrong_size)

    assert adapter.stream_calls == 0


async def test_a_symlinked_attachment_cannot_escape_the_authorized_directories(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret")
    link = workspace / "link.txt"
    link.symlink_to(secret)

    adapter = ScriptedAdapter()
    escape = TurnRequest(
        input=(
            TextContent("hello"),
            FileContent(path=str(link), size_bytes=6, media_type="text/plain"),
        ),
    )
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(workspace))
        with pytest.raises(InvalidAgentRequest, match="outside the authorized"):
            await runtime.run_turn(session, escape)


async def test_abandoned_stream_releases_session_turn_exclusion(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(partial_before_hang=True, hang=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        stream = runtime.stream_turn(session, turn())
        await anext(stream)
        adapter.stall.set()
        await stream.aclose()
        result = await runtime.run_turn(session, turn())

    assert result.status == "succeeded", (
        "abandoning a stream must release the one-active-turn exclusion"
    )


async def test_runtime_close_interrupts_active_turns_and_closes_adapters(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(partial_before_hang=True, hang=True)
    runtime = runtime_for(tmp_path, adapter)
    session = await runtime.open_session(request(tmp_path))
    stream = runtime.stream_turn(session, turn())
    await anext(stream)

    await runtime.close()

    assert adapter.interrupt_calls >= 1
    assert adapter.close_calls == 1
    with pytest.raises(ProtocolDefect, match="closed"):
        await runtime.open_session(request(tmp_path))
    await stream.aclose()


async def test_runtime_close_maps_adapter_failure_without_leaking_detail(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(fail_close=True)
    runtime = runtime_for(tmp_path, adapter)
    await runtime.open_session(request(tmp_path))

    with pytest.raises(ProtocolDefect) as caught:
        await runtime.close()

    assert "secret detail" not in str(caught.value)


async def test_runtime_close_is_bounded_when_adapter_interrupt_never_settles(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(partial_before_hang=True, hang=True, hang_interrupt=True)
    runtime = runtime_for(tmp_path, adapter, max_turn_seconds=0.2)
    session = await runtime.open_session(request(tmp_path))
    stream = runtime.stream_turn(session, turn())
    await anext(stream)

    with pytest.raises(ProtocolDefect):
        async with asyncio.timeout(30.0):
            await runtime.close()

    adapter.interrupt_release.set()
    adapter.close_release.set()
    await stream.aclose()


def test_runtime_rejects_duplicate_or_invalid_adapter_routes(tmp_path: Path) -> None:
    with pytest.raises(InvalidAgentRequest, match="duplicate adapter route"):
        AgentRuntime(
            AgentRuntimeConfig(state_root_base=tmp_path),
            adapters=(ScriptedAdapter(), ScriptedAdapter()),
        )
    with pytest.raises(InvalidAgentRequest, match="unsupported route"):
        AgentRuntime(
            AgentRuntimeConfig(state_root_base=tmp_path),
            adapters=(ScriptedAdapter(transport="wire"),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_runtime_requires_a_positive_finite_max_turn_lifetime(tmp_path: Path, value: float) -> None:
    with pytest.raises(InvalidAgentRequest, match="max_turn_seconds"):
        AgentRuntimeConfig(state_root_base=tmp_path, max_turn_seconds=value)


async def test_profile_state_root_is_created_private(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    async with runtime_for(tmp_path, adapter) as runtime:
        await runtime.open_session(request(tmp_path))

    state_root = tmp_path / "codex" / "personal"
    assert state_root.is_dir()
    assert (state_root.stat().st_mode & 0o777) == 0o700
    assert ((tmp_path / "codex").stat().st_mode & 0o777) == 0o700
    assert ((state_root / "home").stat().st_mode & 0o777) == 0o700
    assert adapter.last_environment["CODEX_HOME"] == str(state_root.resolve())


async def test_adapter_session_ref_mismatch_is_a_defect(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(bad_ref=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        with pytest.raises(ProtocolDefect, match="inconsistent"):
            await runtime.open_session(request(tmp_path))


async def test_a_lazy_ref_is_validated_when_the_adapter_completes_it(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(lazy_ref=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        assert not session.ref_is_complete
        result = await runtime.run_turn(session, turn())

    assert session.ref_is_complete
    assert result.session_ref == session.ref


async def test_an_event_before_ref_completion_is_a_defect(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(lazy_ref=True, incomplete_ref=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        session = await runtime.open_session(request(tmp_path))
        with pytest.raises(ProtocolDefect, match="completing the session ref"):
            await runtime.run_turn(session, turn())


async def test_auth_validation_precedes_environment_and_adapter_effects(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(reject_auth=True)
    async with runtime_for(tmp_path, adapter) as runtime:
        with pytest.raises(UnsupportedCapability):
            await runtime.open_session(request(tmp_path))

    assert adapter.auth_calls == 1
    assert adapter.open_calls == 0
    assert not (tmp_path / "codex").exists(), (
        "a rejected credential must not create profile state on disk"
    )


async def test_read_session_rejects_mismatched_auth_before_adapter_side_effect(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter()
    async with runtime_for(tmp_path, adapter) as runtime:
        with pytest.raises(SessionMismatch):
            await runtime.read_session(
                session_ref(profile="personal"),
                SessionReadOptions(auth=CredentialRef(kind="local_account", profile_key="other")),
            )

    assert adapter.read_calls == 0


async def test_list_sessions_builds_the_scrubbed_child_environment(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    async with runtime_for(tmp_path, adapter) as runtime:
        page = await runtime.list_sessions(
            SessionQuery(
                backend="codex",
                transport="sdk",
                auth=CredentialRef(kind="local_account", profile_key="personal"),
            )
        )

    assert page == SessionPage(sessions=())
    assert adapter.last_environment["CODEX_HOME"].endswith("codex/personal")
    assert "OPENAI_API_KEY" not in adapter.last_environment


async def test_mcp_references_are_materialized_only_at_safe_child_aliases(
    tmp_path: Path,
) -> None:
    async def resolver(name: str) -> str:
        assert name == "vault/github"
        return "resolved-secret"

    adapter = ScriptedAdapter()
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, secret_resolver=resolver),
        adapters=(adapter,),
    )
    source = CredentialRef(kind="secret_reference", profile_key="personal", name="vault/github")
    spec = McpServerSpec(
        name="github",
        transport="streamable_http",
        url="https://mcp.example.test/mcp",
        header_refs=(HeaderReference(name="Authorization", source=source),),
    )
    mcp_request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=PermissionPolicy(
            network="unrestricted",
            unsafe_confirmation=UnsafeConfirmation(("network_unrestricted",)),
        ),
        mcp_servers=(spec,),
    )
    async with runtime:
        await runtime.open_session(mcp_request)

    aliases = [
        name for name in adapter.last_environment if name.startswith("PROVIDER_RUNTIME_MCP_SECRET_")
    ]
    assert len(aliases) == 1, f"expected exactly one MCP alias, got {adapter.last_environment}"
    assert adapter.last_environment[aliases[0]] == "resolved-secret"
    assert "vault/github" not in adapter.last_environment


async def test_mcp_rejects_primary_credential_destination_before_resolution(
    tmp_path: Path,
) -> None:
    resolved: list[str] = []

    async def resolver(name: str) -> str:
        resolved.append(name)
        return "resolved"

    adapter = ScriptedAdapter()
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, secret_resolver=resolver),
        adapters=(adapter,),
    )
    source = CredentialRef(kind="secret_reference", profile_key="personal", name="vault/openai")
    spec = McpServerSpec(
        name="tool",
        transport="stdio",
        command="python3",
        environment_refs=(EnvironmentReference(name="OPENAI_API_KEY", source=source),),
    )
    bad_request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=stdio_mcp_policy(),
        mcp_servers=(spec,),
    )
    async with runtime:
        with pytest.raises(McpConfigurationError, match="credential variable"):
            await runtime.open_session(bad_request)

    assert resolved == [], "the secret must never be resolved for a refused destination"
    assert adapter.open_calls == 0


@pytest.mark.parametrize("destination", ["CODEX_HOME", "PATH", "LD_PRELOAD", "HOME"])
async def test_mcp_rejects_state_root_and_process_control_destinations(
    tmp_path: Path, destination: str
) -> None:
    adapter = ScriptedAdapter()
    source = CredentialRef(kind="api_key_environment", profile_key="personal", name="SOME_TOKEN")
    spec = McpServerSpec(
        name="tool",
        transport="stdio",
        command="python3",
        environment_refs=(EnvironmentReference(name=destination, source=source),),
    )
    bad_request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=stdio_mcp_policy(),
        mcp_servers=(spec,),
    )
    async with runtime_for(tmp_path, adapter) as runtime:
        with pytest.raises(McpConfigurationError):
            await runtime.open_session(bad_request)

    assert adapter.open_calls == 0


async def test_mcp_rejects_cross_server_destination_collision(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    first_source = CredentialRef(
        kind="api_key_environment", profile_key="personal", name="TOKEN_ONE"
    )
    second_source = CredentialRef(
        kind="api_key_environment", profile_key="personal", name="TOKEN_TWO"
    )
    servers = (
        McpServerSpec(
            name="one",
            transport="stdio",
            command="python3",
            environment_refs=(EnvironmentReference(name="SHARED", source=first_source),),
        ),
        McpServerSpec(
            name="two",
            transport="stdio",
            command="python3",
            environment_refs=(EnvironmentReference(name="SHARED", source=second_source),),
        ),
    )
    bad_request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=stdio_mcp_policy(),
        mcp_servers=servers,
    )
    async with runtime_for(tmp_path, adapter) as runtime:
        with pytest.raises(McpConfigurationError, match="different sources"):
            await runtime.open_session(bad_request)


async def test_stdio_mcp_requires_the_unsafe_policy_pair(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    spec = McpServerSpec(name="tool", transport="stdio", command="python3")
    confined_request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=PermissionPolicy(),
        mcp_servers=(spec,),
    )
    async with runtime_for(tmp_path, adapter) as runtime:
        with pytest.raises(UnsupportedCapability, match="stdio MCP"):
            await runtime.open_session(confined_request)

    assert adapter.open_calls == 0


async def test_missing_secret_resolver_is_typed_before_adapter_effect(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    source = CredentialRef(kind="secret_reference", profile_key="personal", name="vault/github")
    spec = McpServerSpec(
        name="github",
        transport="streamable_http",
        url="https://mcp.example.test/mcp",
        header_refs=(HeaderReference(name="Authorization", source=source),),
    )
    mcp_request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=PermissionPolicy(
            network="unrestricted",
            unsafe_confirmation=UnsafeConfirmation(("network_unrestricted",)),
        ),
        mcp_servers=(spec,),
    )
    async with runtime_for(tmp_path, adapter) as runtime:
        with pytest.raises(UnsupportedCapability, match="secret_resolver"):
            await runtime.open_session(mcp_request)

    assert adapter.open_calls == 0


async def test_secret_resolver_exception_is_mapped_without_leaking_detail(tmp_path: Path) -> None:
    async def resolver(_name: str) -> str:
        raise RuntimeError("vault backend stack trace with secret material")

    adapter = ScriptedAdapter()
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, secret_resolver=resolver),
        adapters=(adapter,),
    )
    source = CredentialRef(kind="secret_reference", profile_key="personal", name="vault/github")
    spec = McpServerSpec(
        name="github",
        transport="streamable_http",
        url="https://mcp.example.test/mcp",
        header_refs=(HeaderReference(name="Authorization", source=source),),
    )
    mcp_request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(tmp_path.resolve()),
        policy=PermissionPolicy(
            network="unrestricted",
            unsafe_confirmation=UnsafeConfirmation(("network_unrestricted",)),
        ),
        mcp_servers=(spec,),
    )
    async with runtime:
        with pytest.raises(CredentialUnavailable) as caught:
            await runtime.open_session(mcp_request)

    assert "stack trace" not in str(caught.value)


async def test_a_state_root_base_reached_through_a_symlink_is_usable(tmp_path: Path) -> None:
    real = tmp_path / "real-base"
    real.mkdir(mode=0o700)
    link = tmp_path / "link-base"
    link.symlink_to(real, target_is_directory=True)

    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=link), adapters=(adapter,))
    async with runtime:
        session = await runtime.open_session(request(tmp_path))
        result = await runtime.run_turn(session, turn())

    assert result.status == "succeeded"
    assert (real / "codex" / "personal").is_dir()
