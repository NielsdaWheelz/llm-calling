"""Public no-network doubles make consumer agent tests deterministic."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from provider_runtime.agent_runtime.capabilities import (
    AgentCapabilities,
    AgentCapabilityScope,
)
from provider_runtime.agent_runtime.events import (
    AgentEvent,
    AgentEventData,
    AgentEventKind,
    SessionStartedData,
    TextDeltaData,
    TurnCompletedData,
    TurnStartedData,
)
from provider_runtime.agent_runtime.policy import PermissionPolicy
from provider_runtime.agent_runtime.sessions import AgentSession
from provider_runtime.agent_runtime.testing import (
    NoNetworkAgentRuntime,
    ScriptedAgentRuntime,
)
from provider_runtime.agent_runtime.types import (
    AgentSessionRef,
    AgentSessionRequest,
    CredentialRef,
    NewSession,
    TextAgentOutput,
    TextContent,
    TurnRequest,
)

pytestmark = pytest.mark.anyio


def _auth() -> CredentialRef:
    return CredentialRef(kind="local_account", profile_key="test-profile")


def _scope() -> AgentCapabilityScope:
    return AgentCapabilityScope(backend="codex", transport="sdk", auth=_auth())


def _ref() -> AgentSessionRef:
    return AgentSessionRef(
        schema_version="agent-session-ref.v1",
        backend="codex",
        transport="sdk",
        native_session_id="thread-test",
        profile_key="test-profile",
        state_root_fingerprint="1" * 64,
        cwd_fingerprint="2" * 64,
    )


def _request(tmp_path: Path) -> AgentSessionRequest:
    return AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=_auth(),
        open=NewSession(),
        model=None,
        reasoning=None,
        system=(),
        developer=(),
        cwd=str(tmp_path),
        additional_dirs=(),
        policy=PermissionPolicy(),
        mcp_servers=(),
        output=TextAgentOutput(),
        native=None,
    )


def _turn() -> TurnRequest:
    return TurnRequest(input=(TextContent("hello"),))


def _event(seq: int, kind: AgentEventKind, data: AgentEventData) -> AgentEvent:
    return AgentEvent(
        schema_version="agent-event.v1",
        seq=seq,
        backend="codex",
        transport="sdk",
        session_ref=_ref(),
        turn_id="turn-test",
        kind=kind,
        data=data,
    )


def _success_script() -> tuple[AgentEvent, ...]:
    return (
        _event(1, "session_started", SessionStartedData()),
        _event(2, "turn_started", TurnStartedData()),
        _event(3, "text_delta", TextDeltaData("done")),
        _event(4, "turn_completed", TurnCompletedData(final_text="done")),
    )


def _later_success_script() -> tuple[AgentEvent, ...]:
    return (
        _event(1, "turn_started", TurnStartedData()),
        _event(2, "text_delta", TextDeltaData("done")),
        _event(3, "turn_completed", TurnCompletedData(final_text="done")),
    )


async def test_no_network_double_fails_loudly_without_leaking_auth_name() -> None:
    runtime = NoNetworkAgentRuntime()

    with pytest.raises(AssertionError, match="codex/sdk") as caught:
        await runtime.capabilities(_scope())

    assert "test-profile" not in str(caught.value)


async def test_scripted_runtime_replays_values_and_records_reference_safe_calls(
    tmp_path: Path,
) -> None:
    capabilities = AgentCapabilities(scope=_scope())
    session = AgentSession(_ref())
    runtime = ScriptedAgentRuntime(
        capabilities=(capabilities,),
        sessions=(session,),
        stream_scripts=(_success_script(), _later_success_script()),
    )

    assert await runtime.capabilities(_scope()) is capabilities
    assert await runtime.open_session(_request(tmp_path)) is session
    result = await runtime.run_turn(session, _turn())
    streamed = [event async for event in runtime.stream_turn(session, _turn())]

    assert result.status == "succeeded"
    assert result.final_text == "done"
    assert streamed == list(_later_success_script())
    assert [call.operation for call in runtime.calls] == [
        "capabilities",
        "open_session",
        "run_turn",
        "stream_turn",
    ]
    assert all(not call.approvals_supplied for call in runtime.calls)
    assert all(not call.cancel_supplied for call in runtime.calls)


@pytest.mark.parametrize(
    "script, message",
    [
        ((), "must not be empty"),
        (
            (
                _event(1, "turn_started", TurnStartedData()),
                _event(3, "turn_completed", TurnCompletedData(final_text="")),
            ),
            "gap-free",
        ),
        (
            (
                _event(1, "turn_started", TurnStartedData()),
                _event(2, "turn_completed", TurnCompletedData(final_text="")),
                _event(3, "text_delta", TextDeltaData("late")),
            ),
            "events after terminal",
        ),
        ((_event(1, "text_delta", TextDeltaData("early")),), "requires turn_started"),
        ((_event(1, "turn_started", TurnStartedData()),), "must end with a terminal"),
    ],
)
async def test_scripted_runtime_rejects_invalid_streams(
    script: tuple[AgentEvent, ...], message: str
) -> None:
    with pytest.raises(AssertionError, match=re.escape(message)):
        ScriptedAgentRuntime(stream_scripts=(script,))


async def test_scripted_runtime_fails_on_unexpected_call() -> None:
    runtime = ScriptedAgentRuntime()

    with pytest.raises(AssertionError, match="No scripted agent-runtime capabilities"):
        await runtime.capabilities(_scope())


async def test_scripted_runtime_completes_lazy_ref_and_rejects_repeated_start() -> None:
    session = AgentSession()
    runtime = ScriptedAgentRuntime(
        stream_scripts=(_success_script(), _success_script()),
    )

    first = [event async for event in runtime.stream_turn(session, _turn())]
    assert session.ref == first[0].session_ref
    with pytest.raises(AssertionError, match="must not repeat session_started"):
        async for _event_value in runtime.stream_turn(session, _turn()):
            pass
