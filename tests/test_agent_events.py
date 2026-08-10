from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError

import pytest

from provider_runtime.agent_runtime.errors import MissingTerminalEvent, ProtocolDefect
from provider_runtime.agent_runtime.events import (
    AGENT_EVENT_KINDS,
    AgentEvent,
    AgentFailure,
    AgentNative,
    AgentPermissionRequest,
    AgentQuotaExhausted,
    AgentTerminal,
    AgentText,
    AgentToolUse,
    AgentUsage,
    validate_event_stream,
)
from provider_runtime.agent_runtime.types import (
    AgentSessionRef,
    ApprovalRequest,
    freeze_json_object,
)
from provider_runtime.types import Absent, Present, TokenUsage


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


def usage() -> TokenUsage:
    return TokenUsage.from_components(
        input_tokens=10,
        output_tokens=5,
        total_tokens=Absent(),
        reasoning_tokens=Absent(),
        cache_read_input_tokens=Present(4),
        cache_write_input_tokens=Absent(),
    )


def terminal() -> AgentTerminal:
    return AgentTerminal(
        status="succeeded",
        failure=None,
        final_text="hello world",
        session_ref=ref(),
        usage=Present(usage()),
    )


async def source(*events: AgentEvent) -> AsyncIterator[AgentEvent]:
    for item in events:
        yield item


async def collect(*events: AgentEvent) -> list[AgentEvent]:
    return [item async for item in validate_event_stream(source(*events))]


def test_the_event_union_is_exactly_six_kinds() -> None:
    assert AGENT_EVENT_KINDS == (
        AgentText,
        AgentToolUse,
        AgentUsage,
        AgentPermissionRequest,
        AgentNative,
        AgentTerminal,
    ), f"the closed event vocabulary changed: {AGENT_EVENT_KINDS}"


def test_events_are_frozen_and_validated() -> None:
    text = AgentText("hello")
    with pytest.raises(FrozenInstanceError):
        text.text = "other"  # type: ignore[misc]
    with pytest.raises(ProtocolDefect, match="text"):
        AgentText(None)  # type: ignore[arg-type]
    with pytest.raises(ProtocolDefect, match="native_type"):
        AgentNative("", freeze_json_object({}))
    with pytest.raises(ProtocolDefect, match="frozen JSON"):
        AgentNative("frame", {"raw": "dict"})  # type: ignore[arg-type]
    with pytest.raises(ProtocolDefect, match="TokenUsage"):
        AgentUsage({"input_tokens": 1})  # type: ignore[arg-type]
    with pytest.raises(ProtocolDefect, match="decision"):
        AgentPermissionRequest(
            ApprovalRequest(operation="command", summary="run ls"),
            "maybe",  # type: ignore[arg-type]
        )


def test_tool_use_completion_requires_an_outcome() -> None:
    started = AgentToolUse("call-1", "commandExecution", "started")
    assert started.succeeded is None
    with pytest.raises(ProtocolDefect, match="succeeded"):
        AgentToolUse("call-1", "commandExecution", "completed")
    with pytest.raises(ProtocolDefect, match="only valid on a completed"):
        AgentToolUse("call-1", "commandExecution", "started", succeeded=True)
    completed = AgentToolUse(
        "call-1",
        "commandExecution",
        "completed",
        payload=freeze_json_object({"output": "done"}),
        succeeded=True,
    )
    assert completed.succeeded is True


def test_terminal_failures_are_typed_values() -> None:
    quota = AgentTerminal(
        status="failed",
        failure=AgentQuotaExhausted(),
        final_text="",
        session_ref=ref(),
    )
    assert quota.failure == AgentQuotaExhausted(), (
        "pool exhaustion must be the named AgentQuotaExhausted value"
    )
    failed = AgentTerminal(
        status="failed",
        failure=AgentFailure("backend_failed"),
        final_text="",
        session_ref=ref(),
    )
    assert failed.failure == AgentFailure("backend_failed")
    with pytest.raises(ProtocolDefect, match="typed failure"):
        AgentTerminal(status="failed", failure=None, final_text="", session_ref=ref())
    with pytest.raises(ProtocolDefect, match="only a failed"):
        AgentTerminal(
            status="succeeded",
            failure=AgentFailure("backend_failed"),
            final_text="",
            session_ref=ref(),
        )
    with pytest.raises(ProtocolDefect, match="cause"):
        AgentFailure("quota_exhausted")  # type: ignore[arg-type]
    with pytest.raises(ProtocolDefect, match="duplicates"):
        AgentTerminal(
            status="succeeded",
            failure=None,
            final_text="",
            session_ref=ref(),
            diagnostics=("same", "same"),
        )


async def test_event_stream_releases_events_and_ends_with_one_terminal() -> None:
    events = await collect(
        AgentText("hello "),
        AgentUsage(usage()),
        terminal(),
    )

    assert [type(item).__name__ for item in events] == [
        "AgentText",
        "AgentUsage",
        "AgentTerminal",
    ], f"stream order changed: {events}"


async def test_stream_without_terminal_is_a_missing_terminal_defect() -> None:
    with pytest.raises(MissingTerminalEvent):
        await collect(AgentText("hello"))


async def test_events_after_terminal_are_a_defect_instead_of_a_terminal() -> None:
    with pytest.raises(ProtocolDefect, match="after its terminal") as exc_info:
        await collect(terminal(), AgentText("late"))
    assert exc_info.value.code == "event_after_terminal"


async def test_foreign_values_are_rejected_from_the_stream() -> None:
    with pytest.raises(ProtocolDefect, match="closed event union"):
        await collect("not an event")  # type: ignore[arg-type]
