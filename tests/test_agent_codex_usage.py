"""Cognition accounting through an external Codex WebSocket protocol peer."""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from websockets.asyncio.server import ServerConnection, unix_serve

from provider_runtime.agent_runtime import (
    AgentRuntime,
    AgentRuntimeConfig,
    AgentTerminal,
    AgentUsage,
    CodexCatalogSessionRequest,
    CodexNativeOptions,
    CredentialRef,
    NewSession,
    PermissionPolicy,
    ProtocolDefect,
    ResumeSession,
    TextContent,
    TurnRequest,
)
from provider_runtime.types import Present, TokenUsage

THREAD = "synthetic-thread"
TURN = TurnRequest(input=(TextContent("synthetic input"),))


def counts(input_tokens: int, output_tokens: int) -> dict[str, object]:
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + output_tokens,
        "cachedInputTokens": 0,
        "cacheWriteInputTokens": 0,
        "reasoningOutputTokens": 0,
    }


class UsagePeer:
    def __init__(self, socket: Path) -> None:
        self.socket = socket
        self.turns = 0
        self.snapshots: list[dict[str, object]] = []
        self.restored: dict[str, object] | None = None

    async def handle(self, connection: ServerConnection) -> None:
        async for frame in connection:
            message = json.loads(frame)
            method = message["method"]
            if method == "initialized":
                continue
            if method == "initialize":
                result = {"userAgent": "synthetic-codex"}
            elif method == "account/read":
                result = {"account": {"type": "chatgpt"}}
            elif method == "model/list":
                result = {
                    "data": [
                        {
                            "id": "fixture",
                            "model": "fixture",
                            "displayName": "Fixture",
                            "hidden": False,
                            "inputModalities": ["text"],
                            "supportedReasoningEfforts": [
                                {"reasoningEffort": "high", "description": "High"},
                            ],
                            "defaultReasoningEffort": "high",
                        }
                    ],
                    "nextCursor": None,
                }
            elif method in ("thread/start", "thread/resume"):
                if method == "thread/resume":
                    assert message["params"]["threadId"] == THREAD
                    assert self.restored is not None
                    await connection.send(
                        json.dumps(
                            {
                                "method": "thread/tokenUsage/updated",
                                "params": {
                                    "threadId": THREAD,
                                    "turnId": str(self.turns),
                                    "tokenUsage": self.restored,
                                },
                            }
                        )
                    )
                result = {"thread": {"id": THREAD}}
            elif method == "turn/start":
                self.turns += 1
                result = {"turn": {"id": str(self.turns)}}
            else:
                raise AssertionError(f"unexpected fixture method {method}")
            await connection.send(json.dumps({"id": message["id"], "result": result}))
            if method != "turn/start":
                continue
            scope = {"threadId": THREAD, "turnId": str(self.turns)}
            for snapshot in self.snapshots:
                await connection.send(
                    json.dumps(
                        {
                            "method": "thread/tokenUsage/updated",
                            "params": {**scope, "tokenUsage": snapshot},
                        }
                    )
                )
            await connection.send(
                json.dumps(
                    {
                        "method": "item/completed",
                        "params": {
                            **scope,
                            "item": {
                                "type": "agentMessage",
                                "id": "answer",
                                "phase": "final_answer",
                                "text": "synthetic answer",
                            },
                        },
                    }
                )
            )
            await connection.send(
                json.dumps(
                    {
                        "method": "turn/completed",
                        "params": {
                            "threadId": THREAD,
                            "turn": {
                                "id": str(self.turns),
                                "status": "completed",
                                "items": [],
                            },
                        },
                    }
                )
            )


@pytest.fixture
async def peer() -> AsyncIterator[UsagePeer]:
    # Keep the real Unix socket within macOS's sockaddr_un path bound.
    with tempfile.TemporaryDirectory(prefix="codex-usage-", dir="/tmp") as directory:
        value = UsagePeer(Path(directory) / "peer.sock")
        async with await unix_serve(value.handle, str(value.socket)):
            yield value


async def request(runtime: AgentRuntime, cwd: Path) -> CodexCatalogSessionRequest:
    auth = CredentialRef("local_account", "fixture")
    catalog = await runtime.model_catalog("codex", auth)
    (model,) = catalog.models
    return CodexCatalogSessionRequest(
        auth=auth,
        cwd=str(cwd),
        open=NewSession(),
        policy=PermissionPolicy(allowed_tools=("*",)),
        native=CodexNativeOptions(builtin_tools="disabled"),
        model_key=model.key,
        reasoning="high",
        agent_definition_revision=catalog.definition_revision,
        row_fingerprint=model.row_fingerprint,
    )


def expected_usage(input_tokens: int, output_tokens: int) -> TokenUsage:
    return TokenUsage(
        input_tokens,
        output_tokens,
        input_tokens + output_tokens,
        Present(0),
        Present(0),
        Present(0),
    )


async def test_compaction_estimates_do_not_charge_continuing_or_restored_turns(
    tmp_path: Path,
    peer: UsagePeer,
) -> None:
    # Native 0.154 core/src/session/mod.rs::recompute_token_usage leaves billing
    # total unchanged; last has zero components and a context estimate. The estimate
    # need not fit the billing total (which can even still be zero).
    estimate = {**counts(0, 0), "totalTokens": 900}
    peer.snapshots = [
        {"total": counts(0, 0), "last": estimate},
        {"total": counts(100, 20), "last": counts(100, 20)},
        {"total": counts(100, 20), "last": estimate},
        {"total": counts(150, 30), "last": counts(50, 10)},
    ]
    config = AgentRuntimeConfig(tmp_path, {"fixture": peer.socket})
    async with AgentRuntime(config) as runtime:
        session_request = await request(runtime, tmp_path)
        session = await runtime.open_session(session_request)
        events = [event async for event in runtime.stream_turn(session, TURN)]
        assert [event.usage for event in events if isinstance(event, AgentUsage)] == [
            expected_usage(0, 0),
            expected_usage(100, 20),
            expected_usage(150, 30),
        ]
        terminal = events[-1]
        assert isinstance(terminal, AgentTerminal) and terminal.status == "succeeded"
        assert terminal.usage == Present(expected_usage(150, 30))
        peer.snapshots = [
            {"total": counts(150, 30), "last": estimate},
            {"total": counts(175, 37), "last": counts(25, 7)},
        ]
        events = [event async for event in runtime.stream_turn(session, TURN)]
        assert [event.usage for event in events if isinstance(event, AgentUsage)] == [
            expected_usage(25, 7),
        ]
        terminal = events[-1]
        assert isinstance(terminal, AgentTerminal) and terminal.status == "succeeded"
        assert terminal.usage == Present(expected_usage(25, 7))
        ref = session.ref
        await runtime.close_session(session)
    peer.restored = {"total": counts(175, 37), "last": estimate}
    peer.snapshots = [{"total": counts(190, 40), "last": counts(15, 3)}]
    async with AgentRuntime(config) as runtime:
        session = await runtime.open_session(replace(session_request, open=ResumeSession(ref)))
        terminal = await runtime.run_turn(session, TURN)
        assert terminal.status == "succeeded" and terminal.session_ref == ref
        assert terminal.usage == Present(expected_usage(15, 3))
        await runtime.close_session(session)
    assert peer.turns == 3 and peer.socket.is_socket()


@pytest.mark.parametrize(
    ("member", "key", "value"),
    [
        ("total", "totalTokens", 901),
        ("total", "inputTokens", 0),
        ("last", "inputTokens", 1),
        ("last", "outputTokens", 1),
        ("last", "cachedInputTokens", 1),
        ("last", "cacheWriteInputTokens", 1),
        ("last", "reasoningOutputTokens", 1),
        ("last", "inputTokens", None),
        ("last", "reasoningOutputTokens", None),
        ("last", "totalTokens", None),
        ("last", "totalTokens", -1),
        ("last", "inputTokens", False),
        ("last", "totalTokens", True),
    ],
)
async def test_estimate_shape_does_not_excuse_malformed_usage(
    tmp_path: Path,
    peer: UsagePeer,
    member: str,
    key: str,
    value: object,
) -> None:
    snapshot = {"total": counts(100, 20), "last": {**counts(0, 0), "totalTokens": 900}}
    snapshot[member][key] = value
    peer.snapshots = [snapshot]
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"fixture": peer.socket})) as runtime:
        session = await runtime.open_session(await request(runtime, tmp_path))
        with pytest.raises(ProtocolDefect):
            await runtime.run_turn(session, TURN)


async def test_compaction_does_not_reset_cumulative_billing(
    tmp_path: Path,
    peer: UsagePeer,
) -> None:
    peer.snapshots = [
        {"total": counts(100, 20), "last": counts(100, 20)},
        {"total": counts(90, 20), "last": {**counts(0, 0), "totalTokens": 900}},
    ]
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"fixture": peer.socket})) as runtime:
        session = await runtime.open_session(await request(runtime, tmp_path))
        events = []
        with pytest.raises(ProtocolDefect):
            async for event in runtime.stream_turn(session, TURN):
                events.append(event)
        assert [event.usage for event in events if isinstance(event, AgentUsage)] == [
            expected_usage(100, 20),
        ]
        assert not any(isinstance(event, AgentTerminal) for event in events)
