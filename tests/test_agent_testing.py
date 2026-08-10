"""Public no-network doubles make consumer agent tests deterministic."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from provider_runtime.agent_runtime.events import (
    AgentEvent,
    AgentTerminal,
    AgentText,
)
from provider_runtime.agent_runtime.policy import PermissionPolicy
from provider_runtime.agent_runtime.sessions import AgentSession, SessionQuery
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


def _terminal(text: str = "done") -> AgentTerminal:
    return AgentTerminal(
        status="succeeded",
        failure=None,
        final_text=text,
        session_ref=_ref(),
    )


def _success_script() -> tuple[AgentEvent, ...]:
    return (AgentText("done"), _terminal())


async def test_no_network_double_fails_loudly_without_leaking_auth_name() -> None:
    runtime = NoNetworkAgentRuntime()

    with pytest.raises(AssertionError, match="codex/sdk") as caught:
        await runtime.list_sessions(SessionQuery(backend="codex", transport="sdk", auth=_auth()))

    assert "test-profile" not in str(caught.value)


async def test_scripted_runtime_replays_values_and_records_reference_safe_calls(
    tmp_path: Path,
) -> None:
    session = AgentSession(_ref())
    runtime = ScriptedAgentRuntime(
        sessions=(session,),
        stream_scripts=(_success_script(), _success_script()),
    )

    assert await runtime.open_session(_request(tmp_path)) is session
    result = await runtime.run_turn(session, _turn())
    streamed = [event async for event in runtime.stream_turn(session, _turn())]

    assert result.status == "succeeded"
    assert result.final_text == "done"
    assert streamed == list(_success_script())
    assert [call.operation for call in runtime.calls] == [
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
            (_terminal(), AgentText("late")),
            "events after terminal",
        ),
        ((AgentText("early"),), "must end with AgentTerminal"),
    ],
)
async def test_scripted_runtime_rejects_invalid_streams(
    script: tuple[AgentEvent, ...], message: str
) -> None:
    with pytest.raises(AssertionError, match=re.escape(message)):
        ScriptedAgentRuntime(stream_scripts=(script,))


async def test_scripted_runtime_fails_on_unexpected_call() -> None:
    runtime = ScriptedAgentRuntime()

    with pytest.raises(AssertionError, match="No scripted agent-runtime stream_turn"):
        await runtime.run_turn(AgentSession(_ref()), _turn())


async def test_scripted_runtime_completes_lazy_ref_from_the_terminal() -> None:
    session = AgentSession()
    runtime = ScriptedAgentRuntime(stream_scripts=(_success_script(),))

    events = [event async for event in runtime.stream_turn(session, _turn())]

    assert session.ref == _ref(), "the scripted stream must complete the session ref"
    terminal = events[-1]
    assert isinstance(terminal, AgentTerminal)
    assert terminal.session_ref == session.ref


async def test_scripted_runtime_rejects_a_turn_while_one_is_active() -> None:
    session = AgentSession(_ref())
    runtime = ScriptedAgentRuntime(stream_scripts=(_success_script(), _success_script()))

    stream = runtime.stream_turn(session, _turn())
    first = await anext(stream)
    assert first == AgentText("done")
    from provider_runtime.agent_runtime.errors import ConcurrentTurn

    with pytest.raises(ConcurrentTurn):
        await runtime.run_turn(session, _turn())
    await stream.aclose()
