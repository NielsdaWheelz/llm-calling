from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from provider_runtime.agent_runtime.errors import MissingTerminalEvent, ProtocolDefect
from provider_runtime.agent_runtime.events import (
    SESSION_SCOPE_EVENT_KINDS,
    TERMINAL_EVENT_KINDS,
    AgentEvent,
    AgentEventData,
    AgentEventKind,
    ApprovalAnsweredData,
    ApprovalRequestedData,
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
    terminal_event_to_result,
    validate_event_stream,
)
from provider_runtime.agent_runtime.types import (
    AgentSessionRef,
    ApprovalRequest,
    FrozenJsonDict,
    freeze_json_object,
)


def ref() -> AgentSessionRef:
    return AgentSessionRef(
        schema_version="agent-session-ref.v1",
        backend="codex",
        transport="sdk",
        native_session_id="thread-1",
        profile_key="personal",
        state_root_fingerprint="a" * 64,
        cwd_fingerprint="b" * 64,
    )


def event(seq: int, kind: AgentEventKind, data: AgentEventData) -> AgentEvent:
    return AgentEvent(
        schema_version="agent-event.v1",
        seq=seq,
        backend="codex",
        transport="sdk",
        session_ref=ref(),
        turn_id="turn-1",
        kind=kind,
        data=data,
    )


async def source(*events: AgentEvent) -> AsyncIterator[AgentEvent]:
    for item in events:
        yield item


async def collect(*events: AgentEvent) -> list[AgentEvent]:
    return [item async for item in validate_event_stream(source(*events))]


def completed(seq: int = 4) -> AgentEvent:
    return event(
        seq,
        "turn_completed",
        TurnCompletedData(
            final_text="hello world",
            usage=freeze_json_object({"input_tokens": 2, "output_tokens": 3}),
        ),
    )


def test_agent_event_is_frozen_versioned_and_kind_data_correlated() -> None:
    value = event(1, "turn_started", TurnStartedData())

    with pytest.raises(FrozenInstanceError):
        value.seq = 2  # type: ignore[misc]
    with pytest.raises(ProtocolDefect, match="schema_version"):
        AgentEvent(
            schema_version="agent-event.v2",  # type: ignore[arg-type]
            seq=1,
            backend="codex",
            transport="sdk",
            session_ref=ref(),
            turn_id="turn-1",
            kind="turn_started",
            data=TurnStartedData(),
        )
    with pytest.raises(ProtocolDefect, match="kind and data"):
        event(1, "turn_started", TextDeltaData("wrong variant"))
    with pytest.raises(TypeError):
        TextDeltaData(text="hello", ignored=True)  # type: ignore[call-arg]
    with pytest.raises(ProtocolDefect, match="finite"):
        TurnCompletedData(final_text="", structured_output=float("nan"))
    forged = FrozenJsonDict()
    with pytest.raises(TypeError):
        dict.__setitem__(forged, "self", forged)  # type: ignore[arg-type]
    assert TurnCompletedData(final_text="", structured_output=forged).structured_output == {}


async def test_event_stream_is_gap_free_and_has_one_final_terminal() -> None:
    events = await collect(
        event(1, "session_started", SessionStartedData()),
        event(2, "turn_started", TurnStartedData()),
        event(3, "text_delta", TextDeltaData("hello world")),
        completed(),
    )

    assert [item.seq for item in events] == [1, 2, 3, 4]
    assert events[-1].kind == "turn_completed"


@pytest.mark.parametrize(
    "events",
    [
        (event(2, "turn_started", TurnStartedData()), completed(3)),
        (event(1, "turn_started", TurnStartedData()), completed(3)),
        (event(1, "text_delta", TextDeltaData("early")), completed(2)),
        (
            event(1, "turn_started", TurnStartedData()),
            completed(2),
            event(3, "diagnostic", DiagnosticData(code="late", message="late")),
        ),
        (event(1, "turn_started", TurnStartedData()), completed(2), completed(3)),
    ],
)
async def test_event_stream_rejects_sequence_and_terminal_invariant_breaks(
    events: tuple[AgentEvent, ...],
) -> None:
    with pytest.raises(ProtocolDefect):
        await collect(*events)


async def test_event_stream_eof_without_terminal_is_a_named_defect() -> None:
    with pytest.raises(MissingTerminalEvent):
        await collect(
            event(1, "turn_started", TurnStartedData()),
            event(2, "text_delta", TextDeltaData("partial")),
        )


def test_completed_terminal_projects_bijectively_to_success_result() -> None:
    terminal = completed()
    result = terminal_event_to_result(terminal)

    assert result.status == "succeeded"
    assert result.failure is None
    assert result.final_text == "hello world"
    assert result.usage == {"input_tokens": 2, "output_tokens": 3}
    assert result.session_ref == terminal.session_ref
    assert result.turn_id == terminal.turn_id


@pytest.mark.parametrize(
    ("kind", "data", "status", "failure"),
    [
        (
            "turn_failed",
            TurnFailedData(failure="quota_exhausted", diagnostics=("quota window exhausted",)),
            "failed",
            "quota_exhausted",
        ),
        (
            "turn_cancelled",
            TurnCancelledData(final_text="partial"),
            "cancelled",
            None,
        ),
    ],
)
def test_failure_and_cancellation_terminals_project_without_collapsing_causes(
    kind: AgentEventKind,
    data: AgentEventData,
    status: str,
    failure: str | None,
) -> None:
    result = terminal_event_to_result(event(2, kind, data))

    assert result.status == status
    assert result.failure == failure


def test_terminal_projection_rejects_nonterminal_events() -> None:
    with pytest.raises(ProtocolDefect):
        terminal_event_to_result(event(2, "text_delta", TextDeltaData("not terminal")))


def test_terminal_projection_preserves_redacted_native_terminal_data() -> None:
    terminal = AgentEvent(
        schema_version="agent-event.v1",
        seq=2,
        backend="codex",
        transport="sdk",
        session_ref=ref(),
        turn_id="turn-1",
        kind="turn_failed",
        data=TurnFailedData(failure="backend_failed"),
        native_type="turn/failed",
        native_payload=freeze_json_object({"safe_code": "native_failure"}),
    )

    result = terminal_event_to_result(terminal)

    assert result.terminal_native_type == "turn/failed"
    assert result.terminal_native_payload == {"safe_code": "native_failure"}


@pytest.mark.parametrize(
    "terminal",
    [
        event(2, "turn_cancelled", TurnCancelledData(final_text="")),
        event(2, "turn_failed", TurnFailedData(failure="turn_timeout")),
    ],
)
async def test_a_terminal_is_legal_between_session_started_and_turn_started(
    terminal: AgentEvent,
) -> None:
    """Cancellation and the runtime timeout land in that window and must stay values."""
    events = await collect(event(1, "session_started", SessionStartedData()), terminal)

    assert [item.kind for item in events] == ["session_started", terminal.kind]


def test_a_file_change_must_name_the_outcome_the_backend_reported() -> None:
    """A declined patch and an applied one describe the same paths and the same diff.

    Codex's `PatchApplyStatus` is `inProgress | completed | failed | declined`, so without a
    status the two are indistinguishable and an adapter can only suppress both. The field is
    required precisely so a call site cannot reintroduce that ambiguity by omitting it.
    """
    declined = FileChangeData(path="/repo/a.py", change="modified", status="declined")
    applied = FileChangeData(path="/repo/a.py", change="modified", status="applied")
    assert declined != applied

    with pytest.raises(TypeError):
        FileChangeData(path="/repo/a.py", change="modified")  # type: ignore[call-arg]
    with pytest.raises(ProtocolDefect, match="status"):
        FileChangeData(
            path="/repo/a.py",
            change="modified",
            status="completed",  # type: ignore[arg-type]
        )
    for status in ("in_progress", "applied", "failed", "declined"):
        assert FileChangeData(path="/repo/a.py", change="created", status=status).status == status


async def test_a_session_scope_diagnostic_is_legal_before_turn_started() -> None:
    """A diagnostic that describes the session is not turn content and must not be a defect.

    An SDK may emit configuration and remote-control warnings while the session is opening;
    they arrive before `turn_started` and are handed to the
    consumer as the first frames of the session's first stream. The stream contract owes them
    a place: `turn_started` orders *turn* content, and a session-scope diagnostic is not that.
    """
    events = await collect(
        event(1, "session_started", SessionStartedData()),
        event(2, "diagnostic", DiagnosticData(code="codex_config_warning", message="no sandbox")),
        event(3, "turn_started", TurnStartedData()),
        completed(4),
    )

    assert [item.kind for item in events] == [
        "session_started",
        "diagnostic",
        "turn_started",
        "turn_completed",
    ]


async def test_an_unknown_session_scope_event_is_legal_before_turn_started() -> None:
    """An unrecognized notification has no declared scope, so it cannot be called turn content.

    A vendor SDK's session-scoped notification set can be wider than the members the adapter
    names. Additive notifications normalize to `unknown` and may arrive while the session is
    opening; refusing them here would turn compatible SDK evolution into a hard failure.
    """
    events = await collect(
        event(1, "session_started", SessionStartedData()),
        event(2, "unknown", UnknownData()),
        event(3, "turn_started", TurnStartedData()),
        completed(4),
    )

    assert [item.kind for item in events] == [
        "session_started",
        "unknown",
        "turn_started",
        "turn_completed",
    ]


async def test_a_session_scope_event_may_open_a_later_turns_stream() -> None:
    """`session_started` is a once-per-session event, so later streams open on turn content.

    A persistent SDK client can receive a session-scope notification between turns and hand it
    to the consumer as the first frame of the *next* turn's stream, before its `turn_started`.
    """
    events = await collect(
        event(1, "diagnostic", DiagnosticData(code="codex_rate_limits", message="95% used")),
        event(2, "turn_started", TurnStartedData()),
        completed(3),
    )

    assert [item.kind for item in events] == ["diagnostic", "turn_started", "turn_completed"]


@pytest.mark.parametrize(
    ("kind", "data"),
    [
        ("text_delta", TextDeltaData("early")),
        ("reasoning", ReasoningData("early", "summary")),
        ("tool_started", ToolStartedData(tool_call_id="call-1", name="commandExecution")),
        (
            "tool_updated",
            ToolUpdatedData(tool_call_id="call-1", update=freeze_json_object({"output": "x"})),
        ),
        ("tool_completed", ToolCompletedData(tool_call_id="call-1", output=None, succeeded=True)),
        (
            "approval_requested",
            ApprovalRequestedData(ApprovalRequest(operation="command", summary="run")),
        ),
        ("approval_answered", ApprovalAnsweredData("deny")),
        ("file_change", FileChangeData(path="/repo/a.py", change="modified", status="applied")),
        ("usage", UsageData(freeze_json_object({"input_tokens": 1}))),
        ("native_retry_observed", NativeRetryObservedData(1)),
    ],
)
async def test_every_turn_scoped_kind_still_requires_turn_started(
    kind: AgentEventKind, data: AgentEventData
) -> None:
    """The widening is exactly two kinds, and this is what stops it growing by accident.

    Each kind below names a subject that only exists inside a turn — the turn's output, its
    tool work, its approvals, its per-turn usage report, its retry count — so a backend that
    reported one before opening a turn is misframing its own protocol. Parametrizing the
    complement of `SESSION_SCOPE_EVENT_KINDS` rather than sampling it means a future kind
    quietly added to the session-scope set fails here instead of silently loosening the
    grammar that four repair rounds spent restoring.
    """
    assert kind not in SESSION_SCOPE_EVENT_KINDS
    with pytest.raises(ProtocolDefect, match="turn content preceded turn_started") as captured:
        await collect(
            event(1, "session_started", SessionStartedData()),
            event(2, kind, data),
            completed(3),
        )

    assert captured.value.code == "missing_turn_started"


def test_the_session_scope_and_terminal_sets_partition_the_event_kinds() -> None:
    """The grammar is a partition of the declared kinds, with no member in two places.

    `AgentEventKind` is the spec's list; the three sets the validator branches on must cover
    it exactly once each, or a kind exists that the stream contract says nothing about.
    """
    declared = set(get_args(AgentEventKind.__value__))
    opening = {"session_started", "turn_started"}

    assert declared == opening | set(SESSION_SCOPE_EVENT_KINDS) | set(TERMINAL_EVENT_KINDS) | {
        "text_delta",
        "reasoning",
        "tool_started",
        "tool_updated",
        "tool_completed",
        "approval_requested",
        "approval_answered",
        "file_change",
        "usage",
        "native_retry_observed",
    }
    assert not set(SESSION_SCOPE_EVENT_KINDS) & set(TERMINAL_EVENT_KINDS)
    assert not opening & (set(SESSION_SCOPE_EVENT_KINDS) | set(TERMINAL_EVENT_KINDS))
