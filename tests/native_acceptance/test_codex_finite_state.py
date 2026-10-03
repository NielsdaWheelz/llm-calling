"""Temporary N012 loopback proof; no actual provider/model/research claim."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from test_provider_turn import Peer, session
from websockets.asyncio.server import ServerConnection, unix_serve

from provider_runtime.agent_runtime import (
    AgentAccepted,
    AgentFailure,
    AgentInputRecorded,
    AgentMessage,
    AgentNative,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentTerminal,
    AgentText,
    AgentToolCall,
    AgentToolReply,
    AgentTurnControls,
    LocalStopEvidence,
    ProtocolDefect,
    TextContent,
    TurnRequest,
    UnsupportedCapability,
)
from provider_runtime.agent_runtime.codex_sdk import _CodexAgentTurn
from provider_runtime.types import canonical_json_bytes, freeze_json_value


class FinitePeer(Peer):
    async def handle(self, connection: ServerConnection) -> None:
        self.connections.append(connection)
        async for wire in connection:
            request = json.loads(wire)
            method = request.get("method")
            if method is None:
                self.received.put_nowait(request)
                continue
            if method == "initialized":
                continue
            if method == "initialize":
                result = {"userAgent": self.user_agent}
            elif method == "account/read":
                result = {"account": {"type": "chatgpt"}}
            elif method == "config/read":
                result = self.host_configuration
            elif method == "model/list":
                result = {
                    "data": [
                        {
                            "id": "finite-model",
                            "model": "finite-native-model",
                            "displayName": "Finite",
                            "hidden": False,
                            "inputModalities": ["text"],
                            "supportedReasoningEfforts": [
                                {"reasoningEffort": "high", "description": "High"}
                            ],
                            "defaultReasoningEffort": "high",
                        }
                    ],
                    "nextCursor": None,
                }
            elif method == "thread/start":
                result = {"thread": {"id": "finite-thread"}}
            elif method == "turn/start":
                result = {"turn": {"id": "finite-turn"}}
            elif method == "turn/steer":
                result = {"turnId": "finite-turn"}
            elif method == "turn/interrupt":
                result = {}
            else:
                raise AssertionError(f"unexpected finite-state fixture request {method}")
            await connection.send(json.dumps({"id": request["id"], "result": result}))
            if method in ("turn/start", "turn/steer") and self.mode != "hold_inputs":
                await self.send(
                    "item/completed",
                    item={
                        "id": f"user-{request['id']}",
                        "type": "userMessage",
                        "clientId": request["params"]["clientUserMessageId"],
                    },
                )

    async def send(self, method: str, *, request_id: str | None = None, **params: object) -> None:
        await self.connections[-1].send(
            json.dumps(
                {
                    "method": method,
                    **({"id": request_id} if request_id is not None else {}),
                    "params": {"threadId": "finite-thread", "turnId": "finite-turn", **params},
                }
            )
        )

    async def start_call(self, call_id: str, request_id: str, *, query: str = "source") -> None:
        await self.send(
            "item/started",
            item={
                "id": call_id,
                "type": "dynamicToolCall",
                "tool": "research__read",
                "arguments": {"query": query},
            },
        )
        await self.request_call(call_id, request_id, query=query)

    async def request_call(self, call_id: str, request_id: str, *, query: str = "source") -> None:
        await self.send(
            "item/tool/call",
            request_id=request_id,
            callId=call_id,
            tool="research__read",
            arguments={"query": query},
        )

    async def complete_call(self, call_id: str, *, status: str = "completed") -> None:
        await self.send(
            "item/completed",
            item={
                "id": call_id,
                "type": "dynamicToolCall",
                "tool": "research__read",
                "arguments": {"query": "source"},
                "status": status,
            },
        )

    async def message(
        self, item_id: str, *, phase: str | None = "commentary", text: str = "progress"
    ) -> None:
        item = {"id": item_id, "type": "agentMessage", "phase": phase, "text": text}
        await self.send("item/started", item=item)
        await self.send("item/agentMessage/delta", itemId=item_id, delta=text)
        await self.send("item/completed", item=item)


@pytest.fixture
async def finite_peer() -> AsyncIterator[FinitePeer]:
    with tempfile.TemporaryDirectory(prefix="finite-codex-", dir="/tmp") as directory:
        peer = FinitePeer(Path(directory) / "peer.sock")
        async with await unix_serve(peer.handle, str(peer.socket)):
            yield peer


def prepared(runtime: AgentRuntime, owner):
    turn = runtime.prepare_turn(
        owner,
        TurnRequest(input=(TextContent("read and report"),)),
        attempt_id="finite-attempt",
        input_id="finite-input",
        controls=AgentTurnControls(rpc_seconds=2, pending_calls=4, pending_call_bytes=8192),
    )
    assert isinstance(turn, _CodexAgentTurn)
    return turn


async def next_of(stream, kind):
    while True:
        value = await asyncio.wait_for(anext(stream), 3)
        if isinstance(value, kind):
            return value
        assert isinstance(value, AgentText | AgentNative), type(value)


class Journal:
    """The host retains exact completed identities on disk, outside transport state."""

    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE calls (id TEXT PRIMARY KEY, proposal BLOB, reply TEXT)")
        self.db.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, payload BLOB)")

    def call(self, event: AgentToolCall) -> AgentToolReply:
        proposal = canonical_json_bytes(
            freeze_json_value({"name": event.name, "arguments": event.arguments})
        )
        row = self.db.execute(
            "SELECT proposal, reply FROM calls WHERE id = ?", (event.call_id,)
        ).fetchone()
        if row is not None:
            if row[0] != proposal:
                raise ValueError("changed durable callback identity")
            return AgentToolReply(text=row[1], success=True)
        reply = "recorded source"
        with self.db:
            self.db.execute("INSERT INTO calls VALUES (?, ?, ?)", (event.call_id, proposal, reply))
        return AgentToolReply(text=reply, success=True)

    def message(self, event: AgentMessage) -> None:
        payload = canonical_json_bytes(
            freeze_json_value({"phase": event.phase, "text": event.text})
        )
        row = self.db.execute(
            "SELECT payload FROM messages WHERE id = ?", (event.message_id,)
        ).fetchone()
        if row is not None:
            if row[0] != payload:
                raise ValueError("changed durable message identity")
            return
        with self.db:
            self.db.execute("INSERT INTO messages VALUES (?, ?)", (event.message_id, payload))


async def test_n012_long_turn_retires_completed_state_but_host_keeps_exact_replay(
    finite_peer: FinitePeer, tmp_path: Path
) -> None:
    journal = Journal(tmp_path / "host.sqlite")
    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"loopback": finite_peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path, tools=True)
        turn = prepared(runtime, owner)
        assert isinstance(await turn.submit(), AgentAccepted)
        stream = turn.events()
        assert isinstance(await anext(stream), AgentInputRecorded)
        for index in range(300):
            call_id = f"call-{index}"
            await finite_peer.start_call(call_id, f"request-{index}")
            call = await next_of(stream, AgentToolCall)
            if index % 3:
                await turn.reply(call, journal.call(call))
                assert (await finite_peer.received.get())["id"] == f"request-{index}"
            await finite_peer.complete_call(
                call_id, status="failed" if index % 3 == 0 else "completed"
            )
            await finite_peer.send("warning", message=f"diagnostic-{index}")
            await finite_peer.message(
                f"message-{index}", phase=(None, "commentary", "final_answer")[index % 3]
            )
            event = await next_of(stream, AgentMessage)
            journal.message(event)
            await turn.steer(input_id=f"input-{index}", input=(TextContent("continue"),))
            assert isinstance(await next_of(stream, AgentInputRecorded), AgentInputRecorded)
        state = turn._state
        assert not state.completed_item_ids, "controlled provider retained completed item history"
        assert not getattr(state, "ended_callback_request_ids", ()), (
            "completed callbacks retained transport tombstones"
        )
        assert (
            not state.started_item_types
            and not state.active_tool_calls
            and not state.custom_item_calls
        )
        assert not state.server_request_ids and not state.client._server_request_ids
        assert not state.client._dynamic_requests and not turn._calls and not turn._reply_tokens
        assert not turn._input_ids, "recorded inputs retained transport history"
        assert state.message_count == state.output_bytes == state.streamed_text_bytes == 0
        assert len(state.completed_agent_messages) <= 2 and len(state.diagnostics) <= 256
        assert journal.db.execute("SELECT count(*) FROM messages").fetchone()[0] == 300

        # The same completed native identity is safe only if its durable proposal is exact.
        await finite_peer.start_call("call-1", "exact-replay")
        replay = await next_of(stream, AgentToolCall)
        await turn.reply(replay, journal.call(replay))
        assert (await finite_peer.received.get())["id"] == "exact-replay"
        await finite_peer.complete_call("call-1")
        await finite_peer.message("message-1")
        journal.message(await next_of(stream, AgentMessage))
        assert journal.db.execute("SELECT count(*) FROM calls").fetchone()[0] == 200
        assert journal.db.execute("SELECT count(*) FROM messages").fetchone()[0] == 300
        await finite_peer.start_call("call-1", "changed-replay", query="changed source")
        changed = await next_of(stream, AgentToolCall)
        with pytest.raises(ValueError, match="changed durable callback identity"):
            journal.call(changed)
        await finite_peer.complete_call("call-1", status="failed")
        await finite_peer.message("message-1", text="changed progress")
        with pytest.raises(ValueError, match="changed durable message identity"):
            journal.message(await next_of(stream, AgentMessage))
        await finite_peer.message("final", phase="final_answer", text='{"answer":"ok"}')
        assert isinstance(await next_of(stream, AgentMessage), AgentMessage)
        await finite_peer.send("turn/completed", turn={"id": "finite-turn", "status": "completed"})
        result = await anext(stream)
        assert isinstance(result, AgentTerminal) and result.status == "succeeded"
        assert (await turn.close()).local_closed
    journal.db.close()


async def test_n012_prepared_observation_preserves_bounded_completed_identity(
    finite_peer: FinitePeer, tmp_path: Path
) -> None:
    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"loopback": finite_peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path)
        turn = runtime.prepare_observed_turn(
            owner,
            TurnRequest(input=(TextContent("one bounded answer"),)),
            attempt_id="observed-attempt",
            input_id="observed-input",
            controls=AgentTurnControls(2, 4, 8192),
        )
        assert isinstance(await turn.submit(), AgentAccepted)
        stream = turn.events()
        await finite_peer.message("same-answer", phase="final_answer", text='{"answer":"ok"}')
        await finite_peer.message("same-answer", phase="final_answer", text='{"answer":"changed"}')
        with pytest.raises(ProtocolDefect, match="more than once"):
            _ = [event async for event in stream]
        assert turn.terminal is None
        await turn.close()


async def test_n012_prepared_observation_rejects_declared_callbacks_before_submission(
    finite_peer: FinitePeer, tmp_path: Path
) -> None:
    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"loopback": finite_peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path, tools=True)
        with pytest.raises(UnsupportedCapability, match="declared callbacks"):
            runtime.prepare_observed_turn(
                owner,
                TurnRequest(input=(TextContent("must not submit"),)),
                attempt_id="observed-callback-attempt",
                input_id="observed-callback-input",
                controls=AgentTurnControls(2, 4, 8192),
            )
        assert runtime.session_usable(owner)


@pytest.mark.parametrize("change", ("request_bytes", "item_restart", "live_proposal"))
async def test_n012_live_callback_identity_stays_strict(
    finite_peer: FinitePeer, tmp_path: Path, change: str
) -> None:
    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"loopback": finite_peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path, tools=True)
        turn = prepared(runtime, owner)
        assert isinstance(await turn.submit(), AgentAccepted)
        stream = turn.events()
        assert isinstance(await anext(stream), AgentInputRecorded)
        await finite_peer.start_call("active", "active-request")
        assert isinstance(await next_of(stream, AgentToolCall), AgentToolCall)
        if change == "item_restart":
            await finite_peer.start_call("active", "duplicate-request")
        else:
            await finite_peer.request_call(
                "active",
                "active-request" if change == "request_bytes" else "changed-request",
                query="changed",
            )
        with pytest.raises(ProtocolDefect):
            await anext(stream)
        assert turn.terminal is None and finite_peer.received.empty()
        await turn.close()


async def test_n012_live_duplicate_requests_use_finite_pending_capacity(
    finite_peer: FinitePeer, tmp_path: Path
) -> None:
    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"loopback": finite_peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path, tools=True)
        turn = prepared(runtime, owner)
        assert isinstance(await turn.submit(), AgentAccepted)
        stream = turn.events()
        assert isinstance(await anext(stream), AgentInputRecorded)
        await finite_peer.start_call("active", "request-0")
        first = await next_of(stream, AgentToolCall)
        for index in range(1, 4):
            await finite_peer.request_call("active", f"request-{index}")
            duplicate = await next_of(stream, AgentToolCall)
            assert duplicate.call_id == first.call_id and duplicate.arguments == first.arguments
        await finite_peer.request_call("active", "over-capacity")
        with pytest.raises(ProtocolDefect, match="pending"):
            await anext(stream)
        assert turn.terminal is None and len(turn._reply_tokens) == 4
        await turn.close()


async def test_n012_delivered_replies_do_not_consume_future_pending_capacity(
    finite_peer: FinitePeer, tmp_path: Path
) -> None:
    journal = Journal(tmp_path / "reply-replay.sqlite")
    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"loopback": finite_peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path, tools=True)
        turn = prepared(runtime, owner)
        assert isinstance(await turn.submit(), AgentAccepted)
        stream = turn.events()
        assert isinstance(await anext(stream), AgentInputRecorded)
        await finite_peer.start_call("one-active-item", "request-0")
        for index in range(20):
            if index:
                await finite_peer.request_call("one-active-item", f"request-{index}")
            call = await next_of(stream, AgentToolCall)
            await turn.reply(call, journal.call(call))
            assert (await finite_peer.received.get())["id"] == f"request-{index}"
            assert not turn._reply_tokens, (
                "delivered callback request retained pending reply history"
            )
            assert not turn._state.server_request_ids and not turn._state.client._dynamic_requests
        assert journal.db.execute("SELECT count(*) FROM calls").fetchone()[0] == 1
        await finite_peer.complete_call("one-active-item")
        await finite_peer.message("final", phase="final_answer", text='{"answer":"ok"}')
        assert isinstance(await next_of(stream, AgentMessage), AgentMessage)
        await finite_peer.send("turn/completed", turn={"id": "finite-turn", "status": "completed"})
        assert isinstance(await anext(stream), AgentTerminal)
        await turn.close()
    journal.db.close()


async def test_n012_unrecorded_inputs_remain_authoritative_and_bounded(
    finite_peer: FinitePeer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from provider_runtime.agent_runtime import codex_sdk

    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"loopback": finite_peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path, tools=True)
        turn = prepared(runtime, owner)
        assert isinstance(await turn.submit(), AgentAccepted)
        stream = turn.events()
        assert isinstance(await anext(stream), AgentInputRecorded)
        finite_peer.mode = "hold_inputs"
        monkeypatch.setattr(codex_sdk, "_MAX_MESSAGE_BYTES", 2048)
        input_ids = [f"{index}:".ljust(512, "x") for index in range(4)]
        for input_id in input_ids:
            receipt = await turn.steer(input_id=input_id, input=(TextContent("continue"),))
            assert receipt.disposition == "accepted"
        blocked = await turn.steer(input_id="over-capacity", input=(TextContent("continue"),))
        assert blocked.disposition == "not_sent" and "pending input" in blocked.reason
        assert turn._input_ids == set(input_ids) and turn._pending_input_bytes == 2048
        for index, input_id in enumerate(input_ids):
            await finite_peer.send(
                "item/completed",
                item={
                    "id": f"recorded-{index}",
                    "type": "userMessage",
                    "clientId": input_id,
                },
            )
            event = await next_of(stream, AgentInputRecorded)
            assert event.input_id == input_id
        assert not turn._input_ids and turn._pending_input_bytes == 0
        await finite_peer.send(
            "item/completed",
            item={
                "id": "foreign-replay",
                "type": "userMessage",
                "clientId": input_ids[0],
            },
        )
        with pytest.raises(ProtocolDefect, match="foreign user input"):
            await anext(stream)
        assert turn.terminal is None
        await turn.close()


@pytest.mark.parametrize("observed", (False, True))
async def test_n012_observed_cumulative_limit_is_not_a_native_main_limit(
    finite_peer: FinitePeer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, observed: bool
) -> None:
    from provider_runtime.agent_runtime import codex_sdk
    from provider_runtime.agent_runtime._limits import OutputLimitExceeded

    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"loopback": finite_peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path)
        prepare = runtime.prepare_observed_turn if observed else runtime.prepare_turn
        turn = prepare(
            owner,
            TurnRequest(input=(TextContent("report progress"),)),
            attempt_id="limits-attempt",
            input_id="limits-input",
            controls=AgentTurnControls(2, 4, 8192),
        )
        assert isinstance(turn, _CodexAgentTurn)
        assert isinstance(await turn.submit(), AgentAccepted)
        stream = turn.events()
        monkeypatch.setattr(codex_sdk, "_MAX_EVENT_COUNT", 10)
        for index in range(4):
            await finite_peer.message(f"message-{index}")
        await finite_peer.message("final", phase="final_answer", text='{"answer":"ok"}')
        await finite_peer.send("turn/completed", turn={"id": "finite-turn", "status": "completed"})
        if observed:
            with pytest.raises(OutputLimitExceeded):
                _ = [event async for event in stream]
            assert turn.terminal is None
        else:
            events = [event async for event in stream]
            assert isinstance(events[-1], AgentTerminal) and events[-1].status == "succeeded"
            assert turn._state.message_count == 0
        await turn.close()


@pytest.mark.parametrize("observed", (False, True))
async def test_n012_runtime_turn_deadline_applies_only_to_observation(
    finite_peer: FinitePeer, tmp_path: Path, observed: bool
) -> None:
    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"loopback": finite_peer.socket}, max_turn_seconds=0.03)
    ) as runtime:
        owner = await session(runtime, tmp_path)
        prepare = runtime.prepare_observed_turn if observed else runtime.prepare_turn
        turn = prepare(
            owner,
            TurnRequest(input=(TextContent("wait for answer"),)),
            attempt_id="timeout-attempt",
            input_id="timeout-input",
            controls=AgentTurnControls(2, 4, 8192),
        )
        assert isinstance(await turn.submit(), AgentAccepted)
        stream = turn.events()
        if observed:
            terminal = await asyncio.wait_for(anext(stream), 0.3)
            assert isinstance(terminal, AgentTerminal) and isinstance(
                terminal.failure, AgentFailure
            )
            assert terminal.failure.cause == "turn_timeout"
            assert isinstance(terminal.evidence, LocalStopEvidence)
        else:
            assert isinstance(await anext(stream), AgentInputRecorded)
            await asyncio.sleep(0.06)
            assert turn.terminal is None
            await finite_peer.message("final", phase="final_answer", text='{"answer":"ok"}')
            assert isinstance(await next_of(stream, AgentMessage), AgentMessage)
            await finite_peer.send(
                "turn/completed", turn={"id": "finite-turn", "status": "completed"}
            )
            assert isinstance(await anext(stream), AgentTerminal)
        await turn.close()
