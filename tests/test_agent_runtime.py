from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from provider_runtime.agent_runtime._process import ManagedProcess, ProcessLimits
from provider_runtime.agent_runtime.capabilities import (
    AgentCapabilities,
    AgentCapabilityScope,
    TurnOverride,
)
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
    TERMINAL_EVENT_KINDS,
    AgentEvent,
    ApprovalAnsweredData,
    ApprovalRequestedData,
    DiagnosticData,
    SessionStartedData,
    TextDeltaData,
    TurnCompletedData,
    TurnFailedData,
    TurnStartedData,
    UsageData,
    terminal_event_to_result,
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
    AgentTarget,
    AgentTransport,
    ApiTarget,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    Backend,
    CredentialRef,
    EnvironmentReference,
    FileContent,
    HeaderReference,
    JsonSchemaAgentOutput,
    McpServerSpec,
    NewSession,
    ReasoningSpec,
    ResumeSession,
    TextAgentOutput,
    TextContent,
    TurnRequest,
    agent_target_to_session_request,
    api_target_to_provider_target,
    freeze_json_object,
    ref_from_json,
    ref_to_json,
)
from provider_runtime.planning import plan_generate
from provider_runtime.schema import parse_canonical_schema, to_json_schema
from provider_runtime.testing import ScriptedRuntime
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    CallMeta,
    Dynamic,
    FinalizedProviderCall,
    GenerateIntent,
    GlobalScope,
    PossiblyBillable,
    Present,
    PromptBlock,
    PromptMessage,
    ProviderCredential,
    ResponsePayload,
    Stable,
    Succeeded,
    SystemMessage,
    TextOutput,
    TokenUsage,
    UserMessage,
)
from provider_runtime.types import (
    TextContent as ProviderTextContent,
)
from tests import test_agent_claude_sdk, test_agent_codex_sdk


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


class ScriptedAdapter:
    def __init__(
        self,
        *,
        backend: Backend = "codex",
        transport: AgentTransport = "sdk",
        lazy_ref: bool = False,
        hang: bool = False,
        fail_open: bool = False,
        reject_auth: bool = False,
        cancellation: bool = True,
        bad_ref: bool = False,
        emit_session_started: bool = True,
        emit_session_started_every_turn: bool = False,
        hang_before_first: bool = False,
        hang_after_session_started: bool = False,
        flood: int = 0,
        partial_before_hang: bool = False,
        request_approval: bool = False,
        fail_close: bool = False,
        block_open: bool = False,
        block_capabilities: bool = False,
        fail_interrupt: bool = False,
        hang_interrupt: bool = False,
        hang_close: bool = False,
        reports_auth_identity: bool = True,
        persistent_turn_overrides: tuple[TurnOverride, ...] = ("policy",),
    ) -> None:
        self._backend: Backend = backend
        self._transport: AgentTransport = transport
        self.lazy_ref = lazy_ref
        self.hang = hang
        self.fail_open = fail_open
        self.reject_auth = reject_auth
        self.cancellation = cancellation
        self.bad_ref = bad_ref
        self.emit_session_started = emit_session_started
        self.emit_session_started_every_turn = emit_session_started_every_turn
        self.hang_before_first = hang_before_first
        self.hang_after_session_started = hang_after_session_started
        self.flood = flood
        self.partial_before_hang = partial_before_hang
        self.request_approval = request_approval
        self.fail_close = fail_close
        self.block_open = block_open
        self.block_capabilities = block_capabilities
        self.fail_interrupt = fail_interrupt
        self.hang_interrupt = hang_interrupt
        self.hang_close = hang_close
        self.reports_auth_identity = reports_auth_identity
        self.persistent_turn_overrides = persistent_turn_overrides
        self.auth_calls = 0
        self.capability_calls = 0
        self.open_calls = 0
        self.read_calls = 0
        self.interrupt_calls = 0
        self.close_session_calls = 0
        self.close_calls = 0
        self.stream_calls = 0
        self.last_environment: dict[str, str] = {}
        self.turn_started = asyncio.Event()
        self.session_started = asyncio.Event()
        self.session_started_release = asyncio.Event()
        self.stream_entered = asyncio.Event()
        self.open_entered = asyncio.Event()
        self.open_release = asyncio.Event()
        self.capabilities_entered = asyncio.Event()
        self.capabilities_release = asyncio.Event()
        self.interrupt_release = asyncio.Event()
        self.close_release = asyncio.Event()
        self.release = asyncio.Event()
        self.requests: dict[AgentSession, AgentSessionRequest] = {}
        self.refs: dict[AgentSession, AgentSessionRef] = {}
        self.stream_counts: dict[AgentSession, int] = {}
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

    async def capabilities(
        self, scope: AgentCapabilityScope, *, environment: Mapping[str, str]
    ) -> AgentCapabilities:
        self.capability_calls += 1
        self.capabilities_entered.set()
        if self.block_capabilities:
            await self.capabilities_release.wait()
        self.last_environment = dict(environment)
        return AgentCapabilities(
            scope=scope,
            session_operations=("new", "resume", "fork"),
            discovery_operations=("list", "read"),
            models=("supported-model",),
            content_kinds=("text", "file"),
            content_roles=("user",),
            attachment_kinds=("file",),
            session_instruction_roles=("system", "developer"),
            filesystem_modes=("read_only", "full_access"),
            network_modes=("disabled", "allowlist", "unrestricted"),
            network_allowlist=True,
            approval_modes=("deny", "ask"),
            tool_controls=True,
            mcp_transports=("stdio", "streamable_http"),
            mcp_auth_forms=("environment_reference", "header_reference"),
            streaming=True,
            cancellation=self.cancellation,
            timeouts=True,
            turn_overrides=("policy",),
            persistent_turn_overrides=self.persistent_turn_overrides,
            reports_auth_identity=self.reports_auth_identity,
        )

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
        return SessionSnapshot(ref=ref, metadata=SessionMetadata(), items=())

    async def open_session(
        self,
        request: AgentSessionRequest,
        *,
        capabilities: AgentCapabilities,
        environment: Mapping[str, str],
    ) -> AgentSession:
        # The runtime hands over the table it already discovered; a double that ignored it
        # would let a regression that stopped passing it through go unnoticed.
        assert capabilities.scope.backend == request.backend
        assert capabilities.scope.transport == request.transport
        self.open_calls += 1
        self.open_entered.set()
        if self.block_open:
            await self.open_release.wait()
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
        self.stream_counts[session] = 0
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
        count = self.stream_counts[session]
        self.stream_counts[session] = count + 1
        # A native turn id is unique per turn on both shipped SDK adapters. Reusing one across
        # turns here would
        # have hidden the runtime's retired-turn guard behind a fixture artefact.
        turn_id = f"turn-{count + 1}"
        seq = 1
        if self.emit_session_started and (count == 0 or self.emit_session_started_every_turn):
            yield AgentEvent(
                "agent-event.v1",
                seq,
                self.backend,
                self.transport,
                ref,
                turn_id,
                "session_started",
                SessionStartedData(),
            )
            seq += 1
            self.session_started.set()
            if self.hang_after_session_started:
                # A dedicated gate that interrupt() never opens, so the stall really does
                # persist across the native interrupt the way a real backend's does.
                await self.session_started_release.wait()
        yield AgentEvent(
            "agent-event.v1",
            seq,
            self.backend,
            self.transport,
            ref,
            turn_id,
            "turn_started",
            TurnStartedData(),
        )
        self.turn_started.set()
        seq += 1
        if self.request_approval:
            approval = ApprovalRequest(operation="command", summary="run command")
            yield AgentEvent(
                "agent-event.v1",
                seq,
                self.backend,
                self.transport,
                ref,
                turn_id,
                "approval_requested",
                ApprovalRequestedData(approval),
            )
            seq += 1
            if approvals is None:
                raise AssertionError("scripted approval requires a handler")
            try:
                decision = await approvals(approval)
            except Exception:
                yield AgentEvent(
                    "agent-event.v1",
                    seq,
                    self.backend,
                    self.transport,
                    ref,
                    turn_id,
                    "approval_answered",
                    ApprovalAnsweredData("deny"),
                )
                seq += 1
                await self.interrupt(session, turn_id)
                yield AgentEvent(
                    "agent-event.v1",
                    seq,
                    self.backend,
                    self.transport,
                    ref,
                    turn_id,
                    "turn_failed",
                    TurnFailedData(
                        failure="approval_unanswered",
                        final_text="",
                        diagnostics=("approval handler failed",),
                    ),
                )
                return
            if decision == "abort":
                await self.interrupt(session, turn_id)
            await self.release.wait()
        if self.partial_before_hang:
            yield AgentEvent(
                "agent-event.v1",
                seq,
                self.backend,
                self.transport,
                ref,
                turn_id,
                "text_delta",
                TextDeltaData("partial"),
            )
            seq += 1
            yield AgentEvent(
                "agent-event.v1",
                seq,
                self.backend,
                self.transport,
                ref,
                turn_id,
                "usage",
                UsageData(freeze_json_object({"input_tokens": 7}, context="test usage")),
            )
            seq += 1
            yield AgentEvent(
                "agent-event.v1",
                seq,
                self.backend,
                self.transport,
                ref,
                turn_id,
                "diagnostic",
                DiagnosticData(code="native_warning", message="bounded warning"),
            )
            seq += 1
        for index in range(self.flood):
            yield AgentEvent(
                "agent-event.v1",
                seq,
                self.backend,
                self.transport,
                ref,
                turn_id,
                "text_delta",
                TextDeltaData(f"flood-{index}"),
            )
            seq += 1
        if self.hang:
            await self.release.wait()
        yield AgentEvent(
            "agent-event.v1",
            seq,
            self.backend,
            self.transport,
            ref,
            turn_id,
            "text_delta",
            TextDeltaData(session_request.cwd),
        )
        yield AgentEvent(
            "agent-event.v1",
            seq + 1,
            self.backend,
            self.transport,
            ref,
            turn_id,
            "turn_completed",
            TurnCompletedData(final_text="done"),
        )

    async def interrupt(self, session: AgentSession, turn_id: str | None) -> None:
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
        self.stream_counts.pop(session, None)
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


async def test_runtime_opens_streams_and_projects_one_terminal_result(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(lazy_ref=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    assert not session.ref_is_complete

    stream = runtime.stream_turn(session, turn())
    first = await anext(stream)
    assert first.kind == "session_started"
    assert session.ref == first.session_ref
    remaining = [item async for item in stream]

    assert [item.kind for item in [first, *remaining]] == [
        "session_started",
        "turn_started",
        "text_delta",
        "turn_completed",
    ]
    result = await runtime.run_turn(session, turn())
    assert result.status == "succeeded"
    assert result.final_text == "done"
    await runtime.close()


async def test_close_session_is_idempotent_and_preserves_sibling_sessions(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    closed = await runtime.open_session(request(tmp_path))
    sibling = await runtime.open_session(request(tmp_path))

    await asyncio.gather(runtime.close_session(closed), runtime.close_session(closed))
    await runtime.close_session(closed)

    assert adapter.close_session_calls == 1
    with pytest.raises(SessionUnavailable, match="closed"):
        await runtime.run_turn(closed, turn())
    assert (await runtime.run_turn(sibling, turn())).status == "succeeded"
    await runtime.close()


async def test_close_session_interrupts_and_settles_an_active_turn(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(hang=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    running = asyncio.create_task(runtime.run_turn(session, turn()))
    await adapter.turn_started.wait()

    await runtime.close_session(session)
    await running

    assert adapter.interrupt_calls == 1
    assert adapter.close_session_calls == 1
    with pytest.raises(SessionUnavailable, match="closed"):
        await runtime.run_turn(session, turn())
    await runtime.close()


async def test_close_session_rejects_a_foreign_handle(tmp_path: Path) -> None:
    first_adapter = ScriptedAdapter()
    second_adapter = ScriptedAdapter()
    first = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(first_adapter,))
    second = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(second_adapter,))
    session = await first.open_session(request(tmp_path))

    with pytest.raises(SessionMismatch, match="does not belong"):
        await second.close_session(session)

    await first.close()
    await second.close()


async def test_runtime_rejects_concurrent_turn_without_interleaving(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(hang=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    first = asyncio.create_task(runtime.run_turn(session, turn()))
    await adapter.turn_started.wait()

    with pytest.raises(ConcurrentTurn):
        await runtime.run_turn(session, turn())

    adapter.release.set()
    assert (await first).status == "succeeded"
    await runtime.close()


async def test_cancel_interrupts_selected_adapter_and_synthesizes_cancelled_terminal(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    cancel = asyncio.Event()
    task = asyncio.create_task(runtime.run_turn(session, turn(), cancel=cancel))
    await adapter.turn_started.wait()
    cancel.set()

    result = await task

    assert result.status == "cancelled"
    assert adapter.interrupt_calls == 1
    await runtime.close()


async def test_cancel_interrupts_native_turn_while_consumer_is_paused(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(hang=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    cancel = asyncio.Event()
    stream = runtime.stream_turn(session, turn(), cancel=cancel)

    assert (await anext(stream)).kind == "session_started"
    assert (await anext(stream)).kind == "turn_started"
    cancel.set()
    await asyncio.wait_for(adapter.release.wait(), timeout=0.1)

    assert adapter.interrupt_calls == 1
    assert [event.kind async for event in stream][-1] == "turn_cancelled"
    await runtime.close()


async def test_timeout_interrupts_selected_adapter_and_is_a_typed_terminal_failure(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))

    result = await runtime.run_turn(session, turn(timeout_seconds=0.01))

    assert result.status == "failed"
    assert result.failure == "turn_timeout"
    assert adapter.interrupt_calls == 1
    await runtime.close()


async def test_selected_adapter_failure_never_invokes_another_transport(tmp_path: Path) -> None:
    selected = ScriptedAdapter(fail_open=True)
    other = ScriptedAdapter(backend="claude", transport="sdk")
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(selected, other))

    with pytest.raises(ExecutableUnavailable):
        await runtime.open_session(request(tmp_path))

    assert selected.open_calls == 1
    assert other.capability_calls == 0
    assert other.open_calls == 0
    await runtime.close()


async def test_capability_rejection_happens_before_adapter_open(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    invalid = replace(request(tmp_path), model="unsupported-model")

    with pytest.raises(UnsupportedCapability):
        await runtime.open_session(invalid)

    assert adapter.open_calls == 0
    await runtime.close()


async def test_read_session_rejects_mismatched_auth_before_adapter_side_effect(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    with pytest.raises(SessionMismatch):
        await runtime.read_session(
            session_ref(),
            SessionReadOptions(auth=CredentialRef(kind="local_account", profile_key="other")),
        )

    assert adapter.capability_calls == 0
    assert adapter.read_calls == 0
    await runtime.close()


async def test_read_session_uses_explicit_api_key_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "selected-test-key")
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    await runtime.read_session(
        session_ref(
            profile="api",
            state_root=tmp_path / "codex" / "api",
        ),
        SessionReadOptions(
            auth=CredentialRef(kind="api_key_environment", profile_key="api", name="OPENAI_API_KEY")
        ),
    )

    assert adapter.last_environment["OPENAI_API_KEY"] == "selected-test-key"
    await runtime.close()


async def test_read_session_resolves_secret_reference_without_ambient_lookup(
    tmp_path: Path,
) -> None:
    async def resolve(name: str) -> str:
        assert name == "codex-personal"
        return "resolved-test-key"

    adapter = ScriptedAdapter()
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, secret_resolver=resolve),
        adapters=(adapter,),
    )

    await runtime.read_session(
        session_ref(
            profile="secret",
            state_root=tmp_path / "codex" / "secret",
        ),
        SessionReadOptions(
            auth=CredentialRef(kind="secret_reference", profile_key="secret", name="codex-personal")
        ),
    )

    assert adapter.last_environment["OPENAI_API_KEY"] == "resolved-test-key"
    assert "resolved-test-key" not in repr(runtime.config)
    await runtime.close()


def test_runtime_rejects_duplicate_or_invalid_adapter_routes(tmp_path: Path) -> None:
    first = ScriptedAdapter()
    with pytest.raises(InvalidAgentRequest):
        AgentRuntime(
            AgentRuntimeConfig(state_root_base=tmp_path),
            adapters=(first, ScriptedAdapter()),
        )
    with pytest.raises(InvalidAgentRequest):
        AgentRuntime(
            AgentRuntimeConfig(state_root_base=tmp_path),
            adapters=(ScriptedAdapter(backend="codex", transport=cast(AgentTransport, "wire")),),
        )


async def test_default_runtime_does_not_require_unselected_executables_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("provider_runtime.agent_runtime.runtime.shutil.which", lambda _name: None)

    runtime = AgentRuntime(
        AgentRuntimeConfig(
            state_root_base=tmp_path,
            claude_executable="definitely-missing-claude",
        )
    )

    await runtime.close()


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_runtime_requires_a_positive_finite_max_turn_lifetime(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(InvalidAgentRequest, match="max_turn_seconds"):
        AgentRuntimeConfig(state_root_base=tmp_path, max_turn_seconds=value)  # type: ignore[arg-type]


async def test_ask_without_handler_is_rejected_before_adapter_stream(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session_request = replace(
        request(tmp_path),
        policy=PermissionPolicy(approval="ask"),
    )
    session = await runtime.open_session(session_request)

    with pytest.raises(InvalidAgentRequest, match="approval handler"):
        await runtime.run_turn(session, turn())

    assert adapter.stream_calls == 0
    await runtime.close()


async def test_unsupported_cancel_is_rejected_before_adapter_stream(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(cancellation=False)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))

    with pytest.raises(UnsupportedCapability, match="cancellation"):
        await runtime.run_turn(session, turn(), cancel=asyncio.Event())

    assert adapter.stream_calls == 0
    await runtime.close()


async def test_turn_policy_patch_is_validated_before_adapter_stream(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    widening = replace(turn(), policy=PermissionPolicyPatch(filesystem="workspace_write"))

    with pytest.raises(InvalidAgentRequest, match="cannot widen"):
        await runtime.run_turn(session, widening)

    assert adapter.stream_calls == 0
    await runtime.close()


async def test_missing_cwd_is_rejected_before_adapter_capability_effect(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    missing = replace(request(tmp_path), cwd=str((tmp_path / "missing").resolve()))

    with pytest.raises(InvalidAgentRequest, match="does not exist"):
        await runtime.open_session(missing)

    assert adapter.capability_calls == 0
    assert adapter.open_calls == 0
    await runtime.close()


async def test_file_size_is_statted_before_adapter_capability_effect(tmp_path: Path) -> None:
    attachment = tmp_path / "input.txt"
    attachment.write_text("four")
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    invalid = replace(
        request(tmp_path),
        system=(FileContent(str(attachment.resolve()), 3, "text/plain"),),
    )

    with pytest.raises(InvalidAgentRequest, match="size/type"):
        await runtime.open_session(invalid)

    assert adapter.capability_calls == 0
    await runtime.close()


async def test_abandoned_stream_releases_session_turn_exclusion(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    stream = runtime.stream_turn(session, turn())

    assert (await anext(stream)).kind == "session_started"
    await stream.aclose()

    assert adapter.interrupt_calls == 1
    assert (await runtime.run_turn(session, turn())).status == "succeeded"
    await runtime.close()


async def test_failed_abandonment_interrupt_poison_closes_the_native_route(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang=True, fail_interrupt=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    stream = runtime.stream_turn(session, turn())
    assert (await anext(stream)).kind == "session_started"

    with pytest.raises(ProtocolDefect, match="cleanup"):
        await stream.aclose()
    with pytest.raises(SessionUnavailable):
        await runtime.run_turn(session, turn())
    assert adapter.close_calls == 1
    await runtime.close()
    assert adapter.close_calls == 2


async def test_runtime_close_owns_a_stream_paused_at_a_public_yield(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    stream = runtime.stream_turn(session, turn())

    assert (await anext(stream)).kind == "session_started"
    await runtime.close()

    remaining = [event async for event in stream]
    assert remaining[-1].kind in ("turn_completed", "turn_failed", "turn_cancelled")
    assert adapter.interrupt_calls == 1


async def test_runtime_close_never_cancels_the_consumer_task_paused_after_yield(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    stream = runtime.stream_turn(session, turn())
    observed = asyncio.Event()
    unrelated = asyncio.Event()

    async def consumer() -> AgentEvent:
        event = await anext(stream)
        observed.set()
        await unrelated.wait()
        return event

    task = asyncio.create_task(consumer())
    await observed.wait()
    await runtime.close()

    assert not task.done()
    unrelated.set()
    assert (await task).kind == "session_started"
    assert [event async for event in stream][-1].kind in (
        "turn_completed",
        "turn_failed",
        "turn_cancelled",
    )


async def test_close_waits_for_registered_capability_operation_before_adapter_close(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(block_capabilities=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    operation = asyncio.create_task(
        runtime.capabilities(
            AgentCapabilityScope(
                backend="codex",
                transport="sdk",
                auth=CredentialRef(kind="local_account", profile_key="personal"),
            )
        )
    )
    await adapter.capabilities_entered.wait()
    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)

    assert not closing.done()
    assert adapter.close_calls == 0
    adapter.capabilities_release.set()
    with pytest.raises(ProtocolDefect, match="closed"):
        await operation
    await closing
    assert adapter.close_calls == 1


async def test_external_task_cancellation_interrupts_and_releases_session(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(hang=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    task = asyncio.create_task(runtime.run_turn(session, turn()))
    await adapter.turn_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert adapter.interrupt_calls == 1
    assert (await runtime.run_turn(session, turn())).status == "succeeded"
    await runtime.close()


async def test_external_cancellation_before_native_identity_propagates_cancellation(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang_before_first=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    task = asyncio.create_task(runtime.run_turn(session, turn()))
    await adapter.stream_entered.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert adapter.interrupt_calls == 1
    assert (await runtime.run_turn(session, turn())).status == "succeeded"
    await runtime.close()


async def test_pre_set_cancel_is_typed_and_rejected_before_adapter_effect(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang_before_first=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    cancel = asyncio.Event()
    cancel.set()

    with pytest.raises(TurnNotStarted) as exc_info:
        await runtime.run_turn(session, turn(), cancel=cancel)

    assert exc_info.value.code == "turn_not_started"
    assert exc_info.value.reason == "cancelled"
    assert adapter.interrupt_calls == 0
    assert not adapter.stream_entered.is_set()
    await runtime.close()


async def test_missing_secret_resolver_is_typed_before_adapter_effect(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    ref = session_ref(
        profile="secret",
        state_root=tmp_path / "codex" / "secret",
    )

    with pytest.raises(UnsupportedCapability, match="secret_resolver"):
        await runtime.read_session(
            ref,
            SessionReadOptions(
                auth=CredentialRef(
                    kind="secret_reference", profile_key="secret", name="codex-secret"
                )
            ),
        )

    assert adapter.capability_calls == 0
    await runtime.close()


async def test_empty_resolved_secret_is_typed_before_adapter_effect(tmp_path: Path) -> None:
    async def resolve(_name: str) -> str:
        return ""

    adapter = ScriptedAdapter()
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, secret_resolver=resolve),
        adapters=(adapter,),
    )

    with pytest.raises(CredentialUnavailable):
        await runtime.read_session(
            session_ref(
                profile="secret",
                state_root=tmp_path / "codex" / "secret",
            ),
            SessionReadOptions(
                auth=CredentialRef(
                    kind="secret_reference", profile_key="secret", name="codex-secret"
                )
            ),
        )

    assert adapter.capability_calls == 0
    await runtime.close()


async def test_secret_resolver_exception_is_mapped_without_leaking_detail(tmp_path: Path) -> None:
    async def resolve(_name: str) -> str:
        raise RuntimeError("vault detail with secret value")

    adapter = ScriptedAdapter()
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, secret_resolver=resolve),
        adapters=(adapter,),
    )

    with pytest.raises(CredentialUnavailable) as exc_info:
        await runtime.capabilities(
            AgentCapabilityScope(
                backend="codex",
                transport="sdk",
                auth=CredentialRef(
                    kind="secret_reference", profile_key="secret", name="codex-secret"
                ),
            )
        )

    assert "vault detail" not in str(exc_info.value)
    assert adapter.capability_calls == 0
    await runtime.close()


async def test_profile_state_root_is_created_private(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    await runtime.capabilities(
        AgentCapabilityScope(
            backend="codex",
            transport="sdk",
            auth=CredentialRef(kind="local_account", profile_key="private"),
        )
    )

    assert (tmp_path / "codex" / "private").stat().st_mode & 0o777 == 0o700
    await runtime.close()


async def test_adapter_session_ref_mismatch_is_a_defect(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(bad_ref=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    with pytest.raises(ProtocolDefect, match="session ref inconsistent") as exc_info:
        await runtime.open_session(request(tmp_path))

    assert exc_info.value.code == "adapter_session_ref_mismatch"
    await runtime.close()


async def test_auth_validation_precedes_environment_and_adapter_effects(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(reject_auth=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    rejected = replace(
        request(tmp_path),
        auth=CredentialRef(
            kind="api_key_environment", profile_key="rejected", name="OPENAI_API_KEY"
        ),
    )

    with pytest.raises(UnsupportedCapability, match="scripted auth"):
        await runtime.open_session(rejected)

    assert adapter.auth_calls == 1
    assert adapter.capability_calls == 0
    assert adapter.open_calls == 0
    assert not (tmp_path / "codex" / "rejected").exists()
    await runtime.close()


async def test_first_stream_requires_session_started_even_for_eager_ref(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(emit_session_started=False)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))

    with pytest.raises(ProtocolDefect) as exc_info:
        await runtime.run_turn(session, turn())

    assert exc_info.value.code == "missing_session_started"
    await runtime.close()


async def test_session_started_cannot_repeat_on_a_later_stream(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(emit_session_started_every_turn=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))

    assert (await runtime.run_turn(session, turn())).status == "succeeded"
    with pytest.raises(ProtocolDefect) as exc_info:
        await runtime.run_turn(session, turn())

    assert exc_info.value.code == "duplicate_session_started"
    await runtime.close()


async def test_runtime_default_deadline_applies_when_turn_omits_timeout(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(hang=True)
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, max_turn_seconds=0.01),
        adapters=(adapter,),
    )
    session = await runtime.open_session(request(tmp_path))

    result = await runtime.run_turn(session, turn())

    assert result.status == "failed"
    assert result.failure == "turn_timeout"
    assert adapter.interrupt_calls == 1
    await runtime.close()


async def test_timeout_before_native_identity_is_a_typed_expected_error(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang_before_first=True)
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, max_turn_seconds=0.01),
        adapters=(adapter,),
    )
    session = await runtime.open_session(request(tmp_path))

    with pytest.raises(TurnNotStarted) as exc_info:
        await runtime.run_turn(session, turn())

    assert exc_info.value.code == "turn_not_started"
    assert exc_info.value.reason == "turn_timeout"
    assert adapter.interrupt_calls == 1
    await runtime.close()


async def test_runtime_close_interrupts_an_active_turn_before_closing_adapter(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang_before_first=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    task = asyncio.create_task(runtime.run_turn(session, turn()))
    await adapter.stream_entered.wait()

    await runtime.close()

    assert adapter.interrupt_calls == 1
    assert adapter.close_calls == 1
    assert task.done()
    assert (await task).status == "succeeded"


async def test_runtime_close_maps_adapter_failure_without_leaking_detail(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(fail_close=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    with pytest.raises(ProtocolDefect) as exc_info:
        await runtime.close()

    assert exc_info.value.code == "agent_runtime_cleanup_failed"
    assert "secret detail" not in str(exc_info.value)
    assert adapter.close_calls == 1


async def test_runtime_close_is_bounded_when_adapter_interrupt_never_settles(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang=True, hang_interrupt=True)
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, max_turn_seconds=0.01),
        adapters=(adapter,),
    )
    session = await runtime.open_session(request(tmp_path))
    turn_task = asyncio.create_task(runtime.run_turn(session, turn()))
    await adapter.turn_started.wait()

    try:
        with pytest.raises(ProtocolDefect, match="cleanup") as exc_info:
            await asyncio.wait_for(runtime.close(), timeout=0.5)
        assert exc_info.value.code == "agent_runtime_cleanup_failed"
    finally:
        adapter.interrupt_release.set()
        adapter.release.set()
        await asyncio.gather(turn_task, return_exceptions=True)


async def test_runtime_close_is_bounded_when_adapter_close_never_settles(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang_close=True)
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, max_turn_seconds=0.01),
        adapters=(adapter,),
    )

    try:
        with pytest.raises(ProtocolDefect, match="cleanup") as exc_info:
            await asyncio.wait_for(runtime.close(), timeout=0.5)
        assert exc_info.value.code == "agent_runtime_cleanup_failed"
    finally:
        adapter.close_release.set()
        await asyncio.sleep(0)


async def test_close_cannot_publish_a_session_racing_with_open(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(block_open=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    opening = asyncio.create_task(runtime.open_session(request(tmp_path)))
    await adapter.open_entered.wait()

    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    adapter.open_release.set()

    with pytest.raises(ProtocolDefect) as exc_info:
        await opening
    await closing
    assert exc_info.value.code == "agent_runtime_closed"
    assert adapter.close_calls == 1


async def test_timeout_terminal_preserves_partial_text_latest_usage_and_diagnostics(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang=True, partial_before_hang=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))

    result = await runtime.run_turn(session, turn(timeout_seconds=0.01))

    assert result.final_text == "partial"
    assert result.usage == {"input_tokens": 7}
    assert result.diagnostics == ("bounded warning",)
    await runtime.close()


async def test_approval_handler_exception_interrupts_and_becomes_typed_failure(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(request_approval=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(
        replace(request(tmp_path), policy=PermissionPolicy(approval="ask"))
    )

    async def fail(_request: ApprovalRequest) -> ApprovalDecision:
        raise RuntimeError("handler secret detail")

    result = await runtime.run_turn(session, turn(), approvals=fail)

    assert result.status == "failed"
    assert result.failure == "approval_unanswered"
    assert adapter.interrupt_calls == 1
    assert "handler secret detail" not in repr(result)
    await runtime.close()


async def test_persistent_policy_override_is_the_next_turns_validation_base(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(
        replace(request(tmp_path), policy=PermissionPolicy(allowed_tools=("one", "two")))
    )

    assert (
        await runtime.run_turn(
            session,
            replace(turn(), policy=PermissionPolicyPatch(allowed_tools=("one",))),
        )
    ).status == "succeeded"
    with pytest.raises(InvalidAgentRequest, match="exact subset"):
        await runtime.run_turn(
            session,
            replace(turn(), policy=PermissionPolicyPatch(allowed_tools=("two",))),
        )
    await runtime.close()


async def test_a_non_persistent_override_leaves_the_next_turns_validation_base_alone(
    tmp_path: Path,
) -> None:
    """`persistent_turn_overrides` is a claim about the backend, honoured literally both ways.

    This is the exact mirror of the test above, and the pair is the whole contract: the only
    thing that decides whether one turn's override survives into the next is whether the
    adapter said it would. Neither shipped adapter reports any turn override at all — both
    refuse them outright — so a double is the only place this half of the runtime rule can be
    observed, and without it a runtime that quietly carried every override forward would
    still report `succeeded` for both turns.
    """
    adapter = ScriptedAdapter(persistent_turn_overrides=())
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(
        replace(request(tmp_path), policy=PermissionPolicy(allowed_tools=("one", "two")))
    )

    first = await runtime.run_turn(
        session, replace(turn(), policy=PermissionPolicyPatch(allowed_tools=("one",)))
    )
    # Against the session policy this is a valid narrowing; against a persisted `("one",)` it
    # would not be. It succeeds precisely because the first turn's patch did not stick.
    second = await runtime.run_turn(
        session, replace(turn(), policy=PermissionPolicyPatch(allowed_tools=("two",)))
    )

    assert (first.status, second.status) == ("succeeded", "succeeded")
    await runtime.close()


async def test_mcp_references_are_materialized_only_at_safe_child_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_MCP_TOKEN", "mcp-value")
    source = CredentialRef(kind="api_key_environment", profile_key="mcp", name="SOURCE_MCP_TOKEN")
    servers = (
        McpServerSpec(
            name="stdio",
            transport="stdio",
            command=sys.executable,
            environment_refs=(EnvironmentReference(name="DEST_TOKEN", source=source),),
        ),
        McpServerSpec(
            name="http",
            transport="streamable_http",
            url="https://example.test/mcp",
            header_refs=(HeaderReference(name="Authorization", source=source),),
        ),
    )
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    await runtime.open_session(
        replace(
            request(tmp_path),
            mcp_servers=servers,
            policy=stdio_mcp_policy(),
        )
    )

    aliases = [
        name for name in adapter.last_environment if name.startswith("PROVIDER_RUNTIME_MCP_SECRET_")
    ]
    assert adapter.last_environment["DEST_TOKEN"] == "mcp-value"
    assert adapter.last_environment[aliases[0]] == "mcp-value"
    assert "SOURCE_MCP_TOKEN" not in adapter.last_environment
    assert len(aliases) == 1
    await runtime.close()


async def test_mcp_rejects_primary_credential_destination_before_resolution(
    tmp_path: Path,
) -> None:
    calls = 0

    async def resolve(_name: str) -> str:
        nonlocal calls
        calls += 1
        return "secret"

    source = CredentialRef(kind="secret_reference", profile_key="mcp", name="mcp-secret")
    server = McpServerSpec(
        name="stdio",
        transport="stdio",
        command=sys.executable,
        environment_refs=(EnvironmentReference(name="OPENAI_API_KEY", source=source),),
    )
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, secret_resolver=resolve), adapters=(adapter,)
    )

    with pytest.raises(McpConfigurationError, match="provider credential"):
        await runtime.open_session(
            replace(request(tmp_path), mcp_servers=(server,), policy=stdio_mcp_policy())
        )

    assert calls == 0
    assert adapter.open_calls == 0
    await runtime.close()


@pytest.mark.parametrize(
    "destination",
    ("CODEX_HOME", "CLAUDE_CONFIG_DIR", "PATH", "PYTHONPATH", "LD_PRELOAD"),
)
async def test_mcp_rejects_state_root_and_process_control_destinations_before_resolution(
    tmp_path: Path,
    destination: str,
) -> None:
    calls = 0

    async def resolve(_name: str) -> str:
        nonlocal calls
        calls += 1
        return "secret"

    source = CredentialRef(kind="secret_reference", profile_key="mcp", name="mcp-secret")
    server = McpServerSpec(
        name="stdio",
        transport="stdio",
        command=sys.executable,
        environment_refs=(EnvironmentReference(name=destination, source=source),),
    )
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, secret_resolver=resolve), adapters=(adapter,)
    )

    with pytest.raises(McpConfigurationError, match="process environment"):
        await runtime.open_session(
            replace(request(tmp_path), mcp_servers=(server,), policy=stdio_mcp_policy())
        )

    assert calls == 0
    assert adapter.open_calls == 0
    await runtime.close()


async def test_mcp_rejects_cross_server_destination_collision(tmp_path: Path) -> None:
    first = CredentialRef(kind="api_key_environment", profile_key="mcp", name="FIRST_TOKEN")
    second = CredentialRef(kind="api_key_environment", profile_key="mcp", name="SECOND_TOKEN")
    servers = (
        McpServerSpec(
            name="first",
            transport="stdio",
            command=sys.executable,
            environment_refs=(EnvironmentReference(name="SHARED_TOKEN", source=first),),
        ),
        McpServerSpec(
            name="second",
            transport="stdio",
            command=sys.executable,
            environment_refs=(EnvironmentReference(name="SHARED_TOKEN", source=second),),
        ),
    )
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    with pytest.raises(McpConfigurationError, match="different sources"):
        await runtime.open_session(
            replace(request(tmp_path), mcp_servers=servers, policy=stdio_mcp_policy())
        )

    assert adapter.open_calls == 0
    await runtime.close()


async def test_default_runtime_does_not_require_unselected_executables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "provider_runtime.agent_runtime.runtime.shutil.which",
        lambda _configured: None,
    )

    runtime = AgentRuntime(
        AgentRuntimeConfig(
            state_root_base=tmp_path,
            claude_executable="missing-claude-for-test",
        )
    )

    await runtime.close()


async def test_cancel_between_session_started_and_turn_started_is_a_terminal_value(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang_after_session_started=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    cancel = asyncio.Event()
    task = asyncio.create_task(runtime.run_turn(session, turn(), cancel=cancel))
    await adapter.session_started.wait()
    cancel.set()

    result = await task

    assert result.status == "cancelled"
    assert result.turn_id == "turn-1"
    assert adapter.interrupt_calls == 1
    assert adapter.close_calls == 0
    await runtime.close()


async def test_timeout_between_session_started_and_turn_started_is_a_terminal_value(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(hang_after_session_started=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))

    result = await runtime.run_turn(session, turn(timeout_seconds=0.01))

    assert result.status == "failed"
    assert result.failure == "turn_timeout"
    assert result.turn_id == "turn-1"
    assert adapter.interrupt_calls == 1
    assert adapter.close_calls == 0
    await runtime.close()


async def test_forced_terminal_never_skips_an_event_the_consumer_did_not_receive(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter(flood=400, hang=True)
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, max_turn_seconds=0.2), adapters=(adapter,)
    )
    session = await runtime.open_session(request(tmp_path))
    stream = runtime.stream_turn(session, turn())

    received = [await anext(stream)]
    await asyncio.sleep(0.05)
    await runtime.close()
    received.extend([event async for event in stream])

    assert [event.seq for event in received] == list(range(1, len(received) + 1))
    assert received[-1].kind == "turn_cancelled"


async def test_symlinked_working_directory_and_attachments_are_accepted(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    attachment = real / "input.txt"
    attachment.write_text("four")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    session = await runtime.open_session(
        replace(
            request(tmp_path),
            cwd=str(link),
            system=(FileContent(str(link / "input.txt"), 4, "text/plain"),),
        )
    )

    assert (await runtime.run_turn(session, turn())).status == "succeeded"
    await runtime.close()


async def test_a_symlinked_attachment_cannot_escape_the_authorized_directories(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("four")
    (workspace / "alias.txt").symlink_to(secret)
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    with pytest.raises(InvalidAgentRequest, match="outside the authorized directories"):
        await runtime.open_session(
            replace(
                request(tmp_path),
                cwd=str(workspace),
                system=(FileContent(str(workspace / "alias.txt"), 4, "text/plain"),),
            )
        )

    assert adapter.open_calls == 0
    await runtime.close()


async def test_named_auth_is_rejected_when_the_transport_reports_no_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "selected-test-key")
    adapter = ScriptedAdapter(reports_auth_identity=False)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    named = replace(
        request(tmp_path),
        auth=CredentialRef(kind="api_key_environment", profile_key="api", name="OPENAI_API_KEY"),
    )

    with pytest.raises(UnsupportedCapability, match="reports its effective"):
        await runtime.open_session(named)

    assert adapter.open_calls == 0
    await runtime.close()


async def test_every_runtime_owned_state_root_component_is_private(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))

    await runtime.capabilities(
        AgentCapabilityScope(
            backend="codex",
            transport="sdk",
            auth=CredentialRef(kind="local_account", profile_key="private"),
        )
    )

    assert (tmp_path / "codex").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "codex" / "private").stat().st_mode & 0o777 == 0o700
    await runtime.close()


@pytest.mark.parametrize(
    "destination",
    (
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "SSL_CERT_FILE",
        "NODE_EXTRA_CA_CERTS",
        "REQUESTS_CA_BUNDLE",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "AWS_SECRET_ACCESS_KEY",
        "CODEX_API_BASE",
    ),
)
async def test_mcp_rejects_auth_proxy_and_tls_destinations_before_resolution(
    tmp_path: Path, destination: str
) -> None:
    calls = 0

    async def resolve(_name: str) -> str:
        nonlocal calls
        calls += 1
        return "secret"

    source = CredentialRef(kind="secret_reference", profile_key="mcp", name="mcp-secret")
    server = McpServerSpec(
        name="stdio",
        transport="stdio",
        command=sys.executable,
        environment_refs=(EnvironmentReference(name=destination, source=source),),
    )
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, secret_resolver=resolve), adapters=(adapter,)
    )

    with pytest.raises(McpConfigurationError, match="process environment"):
        await runtime.open_session(
            replace(request(tmp_path), mcp_servers=(server,), policy=stdio_mcp_policy())
        )

    assert calls == 0
    assert adapter.open_calls == 0
    await runtime.close()


# Announces readiness so the test only terminates the group once SIG_IGN is actually installed.
_SIGTERM_IGNORING = (
    "import signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "sys.stdout.write('ready\\n'); sys.stdout.flush(); time.sleep(30)"
)


class ProcessOwningAdapter(ScriptedAdapter):
    """An adapter whose close() tears its owned process groups down one after another."""

    def __init__(self, processes: tuple[ManagedProcess, ...]) -> None:
        super().__init__()
        self.processes = processes
        self.close_finished = False

    async def close(self) -> None:
        for process in self.processes:
            await process.close()
        self.close_finished = True
        await super().close()


async def test_runtime_close_never_orphans_a_process_group_that_outran_its_report_bound(
    tmp_path: Path,
) -> None:
    limits = ProcessLimits(max_stderr_bytes=1024, termination_grace_seconds=0.2)
    spawned: list[ManagedProcess] = []
    for _ in range(3):
        process = await ManagedProcess.spawn(
            (sys.executable, "-c", _SIGTERM_IGNORING),
            cwd=tmp_path,
            environment={},
            limits=limits,
        )
        assert await process.stdout.readline() == b"ready\n"
        spawned.append(process)
    processes = tuple(spawned)
    adapter = ProcessOwningAdapter(processes)
    runtime = AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, max_turn_seconds=0.05), adapters=(adapter,)
    )

    with pytest.raises(ProtocolDefect, match="cleanup") as exc_info:
        await runtime.close()

    assert exc_info.value.code == "agent_runtime_cleanup_failed"
    async with asyncio.timeout(10):
        while not adapter.close_finished:
            await asyncio.sleep(0.01)
    assert all(process.returncode is not None for process in processes), (
        "runtime.close() left an owned agent process group alive"
    )


class _ReplayingAdapter:
    """An adapter that leaks an interrupted turn's frames into the next turn's stream.

    This is what a transport whose native connection outlives one turn does when it neither
    drains the interrupted turn nor discards its leftovers: a persistent SDK client can hand
    the next turn the previous one's tail.
    """

    backend: Backend = "codex"
    transport: AgentTransport = "sdk"

    def __init__(self, ref: AgentSessionRef) -> None:
        self._ref = ref
        self.stream_calls = 0
        self.interrupt_calls = 0
        self.blocked = asyncio.Event()

    def validate_auth(self, credential: CredentialRef) -> None:
        del credential

    async def capabilities(
        self, scope: AgentCapabilityScope, *, environment: Mapping[str, str]
    ) -> AgentCapabilities:
        del environment
        return AgentCapabilities(scope=scope, streaming=True, cancellation=True, timeouts=True)

    async def list_sessions(
        self, query: SessionQuery, *, environment: Mapping[str, str]
    ) -> SessionPage:
        del query, environment
        return SessionPage(sessions=())

    async def read_session(
        self,
        ref: AgentSessionRef,
        options: SessionReadOptions,
        *,
        environment: Mapping[str, str],
    ) -> SessionSnapshot:
        del options, environment
        return SessionSnapshot(ref=ref, metadata=SessionMetadata(), items=())

    async def open_session(
        self,
        request: AgentSessionRequest,
        *,
        capabilities: AgentCapabilities,
        environment: Mapping[str, str],
    ) -> AgentSession:
        del request, capabilities, environment
        return AgentSession(self._ref)

    def _event(self, seq: int, turn_id: str, kind: str, data: object) -> AgentEvent:
        return AgentEvent(
            "agent-event.v1",
            seq,
            self.backend,
            self.transport,
            self._ref,
            turn_id,
            kind,  # type: ignore[arg-type]
            data,  # type: ignore[arg-type]
        )

    async def stream_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        approvals: ApprovalHandler | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        del session, request, approvals
        self.stream_calls += 1
        if self.stream_calls == 1:
            yield self._event(1, "native-turn-1", "session_started", SessionStartedData())
            yield self._event(2, "native-turn-1", "turn_started", TurnStartedData())
            # A real backend keeps the stream open until it answers the interrupt; the
            # runtime is what supplies the cancellation terminal.
            await self.blocked.wait()
            return
        # The leftovers of the interrupted turn, replayed under its original turn id: a
        # grammatically perfect stream that belongs to a turn the caller already abandoned.
        yield self._event(1, "native-turn-1", "turn_started", TurnStartedData())
        yield self._event(2, "native-turn-1", "text_delta", TextDeltaData("late"))
        yield self._event(3, "native-turn-1", "turn_completed", TurnCompletedData("late"))

    async def interrupt(self, session: AgentSession, turn_id: str | None) -> None:
        del session, turn_id
        self.interrupt_calls += 1

    async def close(self) -> None:
        self.blocked.set()

    async def close_session(self, session: AgentSession) -> None:
        del session
        self.blocked.set()


async def test_a_later_turn_may_never_replay_an_interrupted_turn_s_native_frames(
    tmp_path: Path,
) -> None:
    """`AgentAdapter.interrupt` owns the drain; the runtime refuses to trust it silently.

    Without this the leftover frames become the next turn's stream: its `turn_completed`
    would be projected into an `AgentResult` for a turn the backend never ran.
    """
    state_root = tmp_path / "codex" / "personal"
    ref = session_ref(state_root=state_root, cwd=tmp_path)
    adapter = _ReplayingAdapter(ref)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))

    cancel = asyncio.Event()
    async for event in runtime.stream_turn(session, turn(), cancel=cancel):
        if event.kind == "turn_started":
            cancel.set()
    assert adapter.interrupt_calls == 1

    with pytest.raises(ProtocolDefect) as captured:
        await runtime.run_turn(session, turn())
    assert captured.value.code == "retired_turn_replayed"
    await runtime.close()


async def test_a_state_root_base_reached_through_a_symlink_is_usable(tmp_path: Path) -> None:
    """`Path("~/.state").expanduser()` is the ordinary caller spelling.

    On a distro where /home is a symlink to /var/home it does not equal its own `resolve()`,
    and rejecting it defeats the very user story the caller-path fix was made for.
    """
    real = tmp_path / "real-state"
    real.mkdir(mode=0o700)
    link = tmp_path / "linked-state"
    link.symlink_to(real, target_is_directory=True)
    assert link.resolve() != link

    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=link), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    assert (await runtime.run_turn(session, turn())).status == "succeeded"
    await runtime.close()

    # Containment still holds against the resolved base, so the profile root really is created.
    assert (real / "codex" / "personal").is_dir()
    assert adapter.last_environment["CODEX_HOME"] == str(real / "codex" / "personal")


async def test_opening_a_session_discovers_capabilities_exactly_once(tmp_path: Path) -> None:
    """The runtime already validated the request against the table it hands the adapter.

    Rediscovery is not free on either lane: each `capabilities()` spawns a version probe and
    reads the account, so an adapter that repeated it doubles the work per `open_session`.
    """
    adapter = ScriptedAdapter()
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    await runtime.open_session(request(tmp_path))

    assert adapter.capability_calls == 1
    await runtime.close()


async def test_the_completed_ref_is_readable_from_the_first_event_a_consumer_receives(
    tmp_path: Path,
) -> None:
    """`ref_is_complete` and `ref` are usable together without racing the runtime.

    The ref is completed while the runtime is still holding `session_started`, strictly before
    that event is queued, so a consumer that persists on its first event never observes the
    incomplete state.
    """
    adapter = ScriptedAdapter(lazy_ref=True)
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path), adapters=(adapter,))
    session = await runtime.open_session(request(tmp_path))
    assert not session.ref_is_complete
    with pytest.raises(ProtocolDefect) as captured:
        _ = session.ref
    assert captured.value.code == "incomplete_session_ref"

    observed: list[bool] = []
    async for event in runtime.stream_turn(session, turn()):
        observed.append(session.ref_is_complete)
        if len(observed) == 1:
            assert event.kind == "session_started"
            assert session.ref == event.session_ref
    assert all(observed)
    await runtime.close()


# ------------------------------------------------------------------------------------------
# End-to-end: the public runtime over the shipped adapters and their captured doubles.
#
# Every test above this line drives a stub adapter. That proves the runtime's own contract and
# nothing at all about the two transports it dispatches to — which is how four dead-on-arrival
# adapters once shipped green. The tests below run `AgentRuntime` through the real adapter
# modules against the fixture doubles retraced from captured backend traffic, so dispatch,
# capability discovery, the auth environment, native framing, event normalization, and terminal
# projection are exercised together through the public port.


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "agent_runtime"
FAKE_CLAUDE_CODE = FIXTURE_ROOT / "claude" / "fake_claude_code.py"
ANSWER_OUTPUT = JsonSchemaAgentOutput(
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


@dataclass(frozen=True)
class Lane:
    """One shipped route bound to the double that replays its captured backend traffic."""

    backend: Backend
    transport: AgentTransport
    executable: Path | None
    policy: PermissionPolicy
    model: str | None
    # Three prompts: two turns on one live session, then one turn after the session is resumed
    # from its persisted ref. Both doubles select the scenario they replay from the prompt.
    prompts: tuple[str, str, str]
    texts: tuple[str, str, str]
    expected_kinds: frozenset[str]
    output: JsonSchemaAgentOutput | None = None


LANES = (
    Lane(
        backend="codex",
        transport="sdk",
        executable=None,
        # `tool_controls=False` with built-in tool families: the sentinel is the only spelling
        # this transport can honestly accept.
        policy=PermissionPolicy(allowed_tools=("*",)),
        model="native-model",
        # `structured_success` rather than the richer `success` corpus because this lane is
        # also the table's only `JsonSchemaAgentOutput` route, and a schema-validated terminal
        # needs a final text that is the JSON document. The full captured wire order — the
        # session-scope notifications before `turn/started`, the reasoning/tool/file-change
        # content after it — is driven end to end by
        # `test_the_captured_codex_wire_order_streams_end_to_end_through_the_public_runtime`.
        prompts=(
            "fixture:structured_success",
            "fixture:structured_success",
            "fixture:structured_success",
        ),
        texts=('{"answer":"ok"}', '{"answer":"ok"}', '{"answer":"ok"}'),
        expected_kinds=frozenset({"session_started", "turn_started", "text_delta"}),
        output=ANSWER_OUTPUT,
    ),
    Lane(
        backend="claude",
        transport="sdk",
        executable=FAKE_CLAUDE_CODE,
        policy=PermissionPolicy(allowed_tools=("Read", "Write")),
        model="native-model",
        prompts=("fixture:success", "fixture:second_turn", "fixture:success"),
        texts=("Inspection complete.", "Second turn.", "Inspection complete."),
        expected_kinds=frozenset(
            {
                "session_started",
                "turn_started",
                "reasoning",
                "tool_started",
                "tool_completed",
                "file_change",
                "text_delta",
                "diagnostic",
            }
        ),
    ),
)
LANE_IDS = [f"{lane.backend}-{lane.transport}" for lane in LANES]
CODEX_SDK_LANE, CLAUDE_SDK_LANE = LANES


@pytest.fixture
def installed_agent_sdks(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install the adapter-owned SDK doubles behind their real import boundaries."""
    claude = test_agent_claude_sdk.fake_sdk()
    codex = test_agent_codex_sdk.fake_sdk()
    codex_runtime = test_agent_codex_sdk.fake_runtime_package()
    original = importlib.import_module

    def load(name: str, package: str | None = None) -> ModuleType:
        if name == "claude_agent_sdk":
            return cast(ModuleType, claude)
        if name == "openai_codex":
            return codex
        if name == "codex_cli_bin":
            return codex_runtime
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", load)
    return codex


def lane_runtime(tmp_path: Path, lane: Lane) -> AgentRuntime:
    """Build the runtime a caller builds: a private state base plus the selected executables.

    Adapters are left to `AgentRuntimeConfig`, so the closed routing table and executable
    resolution are part of what these tests cover.
    """
    base = tmp_path / "state"
    base.mkdir(mode=0o700, exist_ok=True)
    claude_executable = lane.executable if lane.backend == "claude" else FAKE_CLAUDE_CODE
    assert claude_executable is not None
    return AgentRuntime(
        AgentRuntimeConfig(
            state_root_base=base,
            claude_executable=str(claude_executable),
        )
    )


def lane_request(tmp_path: Path, lane: Lane, **changes: object) -> AgentSessionRequest:
    cwd = tmp_path / "repo"
    cwd.mkdir(exist_ok=True)
    value = AgentSessionRequest(
        backend=lane.backend,
        transport=lane.transport,
        auth=CredentialRef(kind="local_account", profile_key="personal"),
        open=NewSession(),
        cwd=str(cwd.resolve()),
        policy=lane.policy,
        model=lane.model,
    )
    if lane.output is not None:
        value = replace(value, output=lane.output)
    return replace(value, **changes)


def text_turn(prompt: str) -> TurnRequest:
    return TurnRequest(input=(TextContent(prompt),))


def assert_one_valid_stream(events: list[AgentEvent], lane: Lane, ref: AgentSessionRef) -> None:
    kinds = [event.kind for event in events]
    assert [event.seq for event in events] == list(range(1, len(events) + 1)), kinds
    assert [kind for kind in kinds if kind in TERMINAL_EVENT_KINDS] == [kinds[-1]], kinds
    assert all(event.backend == lane.backend for event in events)
    assert all(event.transport == lane.transport for event in events)
    assert all(event.session_ref == ref for event in events), kinds
    assert len({event.turn_id for event in events}) == 1, kinds


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
async def test_every_shipped_transport_streams_a_full_turn_through_the_public_runtime(
    lane: Lane, tmp_path: Path, installed_agent_sdks: ModuleType
) -> None:
    """AgentRuntime -> real adapter -> captured double -> full event stream -> AgentResult."""
    async with lane_runtime(tmp_path, lane) as runtime:
        session = await runtime.open_session(lane_request(tmp_path, lane))
        events = [event async for event in runtime.stream_turn(session, text_turn(lane.prompts[0]))]
        result = terminal_event_to_result(events[-1])
        ref = session.ref

    kinds = [event.kind for event in events]
    assert kinds[0] == "session_started", kinds
    assert kinds[-1] == "turn_completed", kinds
    assert set(kinds) >= lane.expected_kinds, kinds
    assert_one_valid_stream(events, lane, ref)
    assert result.status == "succeeded"
    assert result.failure is None
    assert result.final_text == lane.texts[0]
    assert result.session_ref == ref
    assert ref.backend == lane.backend
    assert ref.transport == lane.transport
    assert ref.profile_key == "personal"
    assert ref.native_session_id


async def test_the_codex_sdk_notification_stream_is_normalized_through_the_public_runtime(
    tmp_path: Path, installed_agent_sdks: ModuleType
) -> None:
    """The real adapter consumes the SDK boundary and the runtime validates its full stream."""
    lane = CODEX_SDK_LANE
    async with lane_runtime(tmp_path, lane) as runtime:
        session = await runtime.open_session(lane_request(tmp_path, lane, output=TextAgentOutput()))
        events = [event async for event in runtime.stream_turn(session, text_turn("rich"))]
        ref = session.ref

    kinds = [event.kind for event in events]
    assert_one_valid_stream(events, lane, ref)
    assert kinds[0] == "session_started", kinds
    assert kinds[-1] == "turn_completed", kinds
    assert set(kinds) >= {
        "turn_started",
        "reasoning",
        "tool_started",
        "tool_updated",
        "tool_completed",
        "file_change",
        "text_delta",
        "usage",
    }, kinds

    result = terminal_event_to_result(events[-1])
    assert result.status == "succeeded"
    assert result.failure is None
    assert result.final_text == "Inspection complete."
    assert result.session_ref == ref
    assert result.usage is not None


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
async def test_every_shipped_transport_completes_two_turns_and_a_resumed_third(
    lane: Lane, tmp_path: Path, installed_agent_sdks: ModuleType
) -> None:
    """The product is a long-lived chat, so one turn proves almost none of the state.

    A second turn on the live session is what proves turn identity advances, that `seq`
    restarts at 1 per returned stream, that `session_started` stays a once-per-session event,
    and that the one-active-turn exclusion is actually released at the terminal instead of
    leaking. Resuming from the persisted ref then proves the same session survives the
    process/connection that opened it.
    """
    async with lane_runtime(tmp_path, lane) as runtime:
        session = await runtime.open_session(lane_request(tmp_path, lane))
        first = [event async for event in runtime.stream_turn(session, text_turn(lane.prompts[0]))]
        second = [event async for event in runtime.stream_turn(session, text_turn(lane.prompts[1]))]
        ref = session.ref
        # Consumers persist the ref whole and never reassemble it field by field.
        persisted = ref_from_json(ref_to_json(ref))
        resumed = await runtime.open_session(
            lane_request(tmp_path, lane, open=ResumeSession(persisted))
        )
        third = [event async for event in runtime.stream_turn(resumed, text_turn(lane.prompts[2]))]
        resumed_ref = resumed.ref

    assert_one_valid_stream(first, lane, ref)
    assert_one_valid_stream(second, lane, ref)
    assert_one_valid_stream(third, lane, resumed_ref)
    assert first[0].kind == "session_started"
    assert "session_started" not in [event.kind for event in second]
    assert third[0].kind == "session_started"
    assert first[0].turn_id != second[0].turn_id
    assert [terminal_event_to_result(stream[-1]).status for stream in (first, second, third)] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert [
        terminal_event_to_result(stream[-1]).final_text for stream in (first, second, third)
    ] == list(lane.texts)
    assert persisted == ref
    assert resumed_ref.native_session_id == ref.native_session_id
    assert resumed is not session


async def test_codex_session_defaults_are_reapplied_to_every_sdk_turn(
    tmp_path: Path, installed_agent_sdks: ModuleType
) -> None:
    """Session-owned effort, summary, approval, and schema reach every SDK turn."""
    lane = CODEX_SDK_LANE
    session_request = lane_request(
        tmp_path, lane, reasoning=ReasoningSpec(effort="low", summary="concise")
    )

    async with lane_runtime(tmp_path, lane) as runtime:
        session = await runtime.open_session(session_request)
        first = await runtime.run_turn(session, text_turn(lane.prompts[0]))
        second = await runtime.run_turn(session, text_turn(lane.prompts[1]))

    assert (first.status, second.status) == ("succeeded", "succeeded")
    turns = cast(
        list[object],
        cast(dict[str, object], installed_agent_sdks.state)["turn_calls"],  # type: ignore[attr-defined]
    )
    assert len(turns) == 2, turns
    defaults = [cast(dict[str, object], cast(dict[str, object], turn)["kwargs"]) for turn in turns]
    assert defaults[0] == defaults[1], defaults
    assert defaults[0]["effort"] == "low"
    assert defaults[0]["summary"] == "concise"
    assert defaults[0]["approval_mode"] == "deny_all"
    assert lane.output is not None
    assert defaults[0]["output_schema"] == to_json_schema(
        lane.output.schema, inline_defs=False, include_annotations=True
    )


@pytest.mark.parametrize("lane", LANES, ids=LANE_IDS)
async def test_no_shipped_transport_accepts_a_turn_override_it_did_not_advertise(
    lane: Lane, tmp_path: Path, installed_agent_sdks: ModuleType
) -> None:
    """Both survivors report `turn_overrides=()`, and both must mean it on the wire.

    An empty claim is the easiest one to leave unenforced: nothing downstream reads it, so an
    adapter that quietly forwarded a per-turn model would still report `succeeded`. The
    runtime gate is refused before the adapter is reached, and the adapter refuses it again on
    its own — which is what keeps the claim true for a caller who supplies adapters directly.

    The override carries the lane's *own* session model rather than an invented name. An
    unknown name is refused by the separate enumerated-model check on any lane that publishes
    a model list, which would let this test keep passing with the override gate removed
    entirely. What is under test is that the per-turn field is refused at all, not its value.
    """
    async with lane_runtime(tmp_path, lane) as runtime:
        scope = AgentCapabilityScope(
            backend=lane.backend,
            transport=lane.transport,
            auth=CredentialRef(kind="local_account", profile_key="personal"),
        )
        capabilities = await runtime.capabilities(scope)
        session = await runtime.open_session(lane_request(tmp_path, lane))
        assert lane.model is not None
        with pytest.raises(UnsupportedCapability, match="model"):
            await runtime.run_turn(session, replace(text_turn(lane.prompts[0]), model=lane.model))
        # The session is not poisoned by the refusal: the next plain turn still runs.
        plain = await runtime.run_turn(session, text_turn(lane.prompts[1]))

    assert capabilities.turn_overrides == ()
    assert capabilities.persistent_turn_overrides == ()
    assert plain.status == "succeeded"


async def test_read_session_reaches_codex_sdk_discovery_through_the_public_port(
    tmp_path: Path, installed_agent_sdks: ModuleType
) -> None:
    """`require_discovery_operation(capabilities, "read")` exercised through `AgentRuntime`.

    The Codex SDK is the only shipped transport that reports `read`, so this is the only path
    on which the runtime's discovery gate can be observed opening rather than closing.
    """
    lane = CODEX_SDK_LANE
    auth = CredentialRef(kind="local_account", profile_key="personal")
    async with lane_runtime(tmp_path, lane) as runtime:
        session = await runtime.open_session(lane_request(tmp_path, lane))
        assert (await runtime.run_turn(session, text_turn(lane.prompts[0]))).status == "succeeded"
        snapshot = await runtime.read_session(
            session.ref,
            SessionReadOptions(auth=auth, include_turns=False, include_items=False),
        )

    assert snapshot.ref == session.ref
    assert snapshot.metadata.name == "Fixture thread"
    assert snapshot.items == ()
    assert snapshot.continuation_cursor is None


async def test_read_session_is_refused_on_a_transport_that_reports_no_discovery(
    tmp_path: Path, installed_agent_sdks: ModuleType
) -> None:
    """The same gate, closed: Claude SDK reports `discovery_operations=()` and stays honest."""
    lane = CLAUDE_SDK_LANE
    auth = CredentialRef(kind="local_account", profile_key="personal")
    async with lane_runtime(tmp_path, lane) as runtime:
        session = await runtime.open_session(lane_request(tmp_path, lane))
        assert (await runtime.run_turn(session, text_turn(lane.prompts[0]))).status == "succeeded"
        with pytest.raises(UnsupportedCapability, match="read"):
            await runtime.read_session(
                session.ref,
                SessionReadOptions(auth=auth, include_turns=False, include_items=False),
            )


API_CHAT: tuple[tuple[str, str], ...] = (
    ("system", "You are the retention assistant."),
    ("user", "Summarize the retention policy."),
    ("assistant", "Records are kept for seven years."),
    ("user", "Now change the retention constant in the repository."),
)
HANDOFF_SUMMARY = "Context so far: the chat settled on a seven-year retention rule."


def api_chat_message(role: str, text: str) -> PromptMessage:
    if role == "system":
        return SystemMessage(
            blocks=(PromptBlock(text=text, stability=Stable(scope=GlobalScope())),)
        )
    if role == "user":
        return UserMessage(blocks=(PromptBlock(text=text, stability=Dynamic()),))
    return AssistantMessage(text=text, tool_calls=(), continuation=Absent())


def api_chat_outcome(text: str) -> Succeeded:
    return Succeeded(
        meta=CallMeta(
            provider="openai",
            model="gpt-5.6-sol",
            provider_request_id=Present("req-handoff-1"),
            upstream_provider=Absent(),
            usage=Present(
                TokenUsage.from_components(
                    input_tokens=12,
                    output_tokens=6,
                    total_tokens=Present(18),
                    reasoning_tokens=Absent(),
                    cache_read_input_tokens=Absent(),
                    cache_write_input_tokens=Absent(),
                )
            ),
            attempt_trace=(),
            billability=PossiblyBillable(),
        ),
        response=ResponsePayload(
            content=ProviderTextContent(text=text, tool_calls=()), continuation=Absent()
        ),
    )


async def test_an_api_chat_hands_off_to_a_new_agent_session_with_bounded_selected_history(
    tmp_path: Path, installed_agent_sdks: ModuleType
) -> None:
    """The one acceptance criterion that spans both lanes, driven through the shared algebra.

    A chat that started on an `ApiTarget` moves to an `AgentTarget`: the caller maps each
    target through the shipped helpers, then seeds a brand-new native session with the bounded
    history it chose. The SDK boundary observes a new thread and exactly the selected excerpt.
    """
    caller_chat_id = "chat-0001"

    api = ApiTarget(
        provider="openai",
        model="gpt-5.6-sol",
        credential=CredentialRef(
            kind="api_key_environment", profile_key="personal", name="OPENAI_API_KEY"
        ),
    )
    plan = plan_generate(
        GenerateIntent(
            target=api_target_to_provider_target(api),
            messages=tuple(api_chat_message(role, text) for role, text in API_CHAT),
            max_output_tokens=256,
            reasoning="low",
            tools=(),
            tool_choice="auto",
            output=TextOutput(),
        )
    )
    assert isinstance(plan, FinalizedProviderCall)
    provider_runtime_double = ScriptedRuntime(
        generate_outcomes=(api_chat_outcome("Which constant should I change?"),)
    )
    outcome = await provider_runtime_double.generate(
        plan,
        credential=ProviderCredential(provider="openai", key="sk-test-not-a-real-key-1234567890"),
    )
    assert isinstance(outcome, Succeeded)

    # The consumer — not the library — selects the bounded history that crosses the boundary.
    lane = CODEX_SDK_LANE
    handoff_prompt = "\n".join(
        (lane.prompts[0], HANDOFF_SUMMARY, API_CHAT[-1][1], outcome.response.content.text)
    )
    target = AgentTarget(request=lane_request(tmp_path, lane, open=NewSession()))
    async with lane_runtime(tmp_path, lane) as runtime:
        session = await runtime.open_session(agent_target_to_session_request(target))
        result = await runtime.run_turn(session, text_turn(handoff_prompt))
        ref = session.ref

    assert result.status == "succeeded"
    state = cast(dict[str, object], installed_agent_sdks.state)  # type: ignore[attr-defined]
    calls = cast(list[tuple[str, dict[str, object]]], state["calls"])
    assert [name for name, _payload in calls].count("start") == 1
    assert not any(name in ("resume", "fork") for name, _payload in calls)
    assert isinstance(target.request.open, NewSession)
    turns = cast(list[dict[str, object]], state["turn_calls"])
    assert len(turns) == 1, turns
    inputs = cast(list[object], turns[0]["inputs"])
    assert [getattr(item, "text", None) for item in inputs] == [handoff_prompt]
    for _role, earlier in API_CHAT[:3]:
        assert earlier not in handoff_prompt, handoff_prompt
    # The native session is new, and the caller's own chat id is not it.
    assert ref.native_session_id != caller_chat_id
    assert ref.native_session_id == result.session_ref.native_session_id
    assert ref.backend == "codex"
    assert ref.transport == "sdk"
