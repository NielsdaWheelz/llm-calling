"""Temporary N001-N005 protocol integration; never actual research qualification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from websockets.asyncio.server import ServerConnection, unix_serve

from provider_runtime.agent_runtime import (
    AgentAccepted,
    AgentInputRecorded,
    AgentMessage,
    AgentNotSubmitted,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentTerminal,
    AgentToolCall,
    AgentToolReply,
    AgentToolUse,
    AgentTurnControls,
    AgentUncertain,
    CodexCatalogSessionRequest,
    CodexNativeOptions,
    CredentialRef,
    InvalidAgentRequest,
    JsonSchemaAgentOutput,
    NativeTerminalEvidence,
    NewSession,
    OutputSchemaMismatch,
    PermissionPolicy,
    ProtocolDefect,
    SessionUnavailable,
    TextContent,
    TurnNotStarted,
    TurnRequest,
    UnsupportedCapability,
    decode_agent_output,
    terminal_from_json,
    terminal_to_json,
)
from provider_runtime.agent_runtime.codex_app_server import (
    CodexAppServerResponseError,
    CodexConnectionUnavailable,
)
from provider_runtime.agent_runtime.codex_sdk import _CodexAgentTurn
from provider_runtime.types import CanonicalTool, canonical_json_bytes, thaw_json_value


class Peer:
    def __init__(self, socket: Path) -> None:
        self.socket = socket
        self.answer = '{"answer":"ok"}'
        self.user_agent = "loopback-provider/0.160.0 (controlled peer)"
        self.tail: str | None = None
        self.starts = 0
        self.mode = "normal"
        self.received: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.turn_entered = asyncio.Event()
        self.start_request: dict[str, object] | None = None
        self.session_requests: list[dict[str, object]] = []
        self.initializations: list[dict[str, object]] = []
        self._sessions = 0
        self.connections: list[ServerConnection] = []
        self._callback_start: dict[str, object] | None = None
        self.host_configuration = {
            "config": {
                "model_catalog_json": "/host/codex-contained-models-0.160.0-b8b588f4b03c8e08fdb7e2994c03578d9b94bbc01665b7adb6490bd234b3cf54.json"
            },
            "origins": {
                "model_catalog_json": {"name": {"type": "sessionFlags"}, "version": "fixture"}
            },
        }

    async def handle(self, connection: ServerConnection) -> None:
        self.connections.append(connection)
        thread_id = ""
        scope: dict[str, object] = {}
        async for frame in connection:
            request = json.loads(frame)
            if "method" not in request:
                self.received.put_nowait(request)
                assert self._callback_start is not None
                await connection.send(
                    json.dumps(
                        {
                            "id": self._callback_start["id"],
                            "result": {"turn": {"id": "turn-loopback"}},
                        }
                    )
                )
                await connection.send(
                    json.dumps(
                        {
                            "method": "serverRequest/resolved",
                            "params": {"threadId": thread_id, "requestId": "callback-1"},
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
                                    "id": "call-1",
                                    "type": "dynamicToolCall",
                                    "tool": "research__read",
                                    "arguments": {"query": "known source"},
                                    "status": "completed",
                                },
                            },
                        }
                    )
                )
                await self.finish(connection, scope)
                continue
            method = request["method"]
            if method == "initialized":
                continue
            if method == "initialize":
                self.initializations.append(request)
                result = {"userAgent": self.user_agent}
            elif method == "account/read":
                result = {"account": {"type": "chatgpt"}}
            elif method == "config/read":
                result = self.host_configuration
            elif method == "model/list":
                result = {
                    "data": [
                        {
                            "id": "loopback-model",
                            "model": "loopback-native-model",
                            "displayName": "Loopback",
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
                self.session_requests.append(request)
                self._sessions += 1
                thread_id = f"thread-loopback-{self._sessions}"
                result = {"thread": {"id": thread_id}}
            elif method == "turn/start":
                self.starts += 1
                self.start_request = request
                self.turn_entered.set()
                scope = {"threadId": thread_id, "turnId": "turn-loopback"}
                if self.mode == "drop":
                    await connection.close()
                    return
                if self.mode == "reject":
                    await connection.send(
                        json.dumps(
                            {
                                "id": request["id"],
                                "error": {"code": -32600, "message": "loopback generic rejection"},
                            }
                        )
                    )
                    continue
                if self.mode == "callback_before_ack":
                    self._callback_start = request
                    await connection.send(
                        json.dumps(
                            {
                                "method": "item/started",
                                "params": {
                                    **scope,
                                    "item": {
                                        "id": "call-1",
                                        "type": "dynamicToolCall",
                                        "tool": "research__read",
                                        "arguments": {"query": "known source"},
                                        "status": "inProgress",
                                    },
                                },
                            }
                        )
                    )
                    await connection.send(
                        json.dumps(
                            {
                                "id": "callback-1",
                                "method": "item/tool/call",
                                "params": {
                                    **scope,
                                    "callId": "call-1",
                                    "tool": "research__read",
                                    "arguments": {"query": "known source"},
                                },
                            }
                        )
                    )
                    continue
                result = {"turn": {"id": "turn-loopback"}}
            elif method == "turn/steer":
                self.received.put_nowait(request)
                result = {"turnId": "turn-loopback"}
                await connection.send(json.dumps({"id": request["id"], "result": result}))
                await connection.send(
                    json.dumps(
                        {
                            "method": "item/completed",
                            "params": {
                                **scope,
                                "item": {
                                    "id": "input-recorded",
                                    "type": "userMessage",
                                    "clientId": request["params"]["clientUserMessageId"],
                                },
                            },
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
                                    "id": "progress",
                                    "type": "agentMessage",
                                    "phase": "commentary",
                                    "text": "the source supports a useful partial answer",
                                },
                            },
                        }
                    )
                )
                continue
            elif method == "turn/interrupt":
                result = {}
            else:
                raise AssertionError(f"unexpected loopback method: {method}")
            await connection.send(json.dumps({"id": request["id"], "result": result}))
            if method != "turn/start":
                if method == "turn/interrupt":
                    await self.finish(connection, scope, status="interrupted")
                continue
            if self.mode == "hold":
                continue
            if self.mode == "raw_native":
                await connection.send(
                    json.dumps(
                        {
                            "method": "turn/started",
                            "params": {
                                "threadId": thread_id,
                                "turn": {"id": "turn-loopback", "status": "inProgress"},
                            },
                        }
                    )
                )
                await connection.send(
                    json.dumps(
                        {
                            "method": "rawResponseItem/completed",
                            "params": {
                                **scope,
                                "item": {
                                    "type": "custom_tool_call",
                                    "name": "exec",
                                    "call_id": "raw-native-1",
                                    "input": "text(await tools.clock__curr_time({}));",
                                },
                            },
                        }
                    )
                )
            await self.finish(connection, scope)

    async def finish(
        self, connection: ServerConnection, scope: dict[str, object], *, status: str = "completed"
    ) -> None:
        thread_id = scope["threadId"]
        await connection.send(
            json.dumps(
                {
                    "method": "item/completed",
                    "params": {
                        **scope,
                        "item": {
                            "id": "answer-loopback",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": self.answer,
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
                        "threadId": thread_id,
                        "turn": {"id": "turn-loopback", "status": status},
                    },
                }
            )
        )
        if self.tail == "malformed":
            await connection.send("not-json")
        if self.tail is not None:
            await connection.close()
            return


@pytest.fixture
async def peer() -> AsyncIterator[Peer]:
    with tempfile.TemporaryDirectory(prefix="native-provider-", dir="/tmp") as directory:
        fixture = Peer(Path(directory) / "peer.sock")
        async with await unix_serve(fixture.handle, str(fixture.socket)):
            yield fixture


async def session(runtime: AgentRuntime, cwd: Path, *, tools: bool = False):
    auth = CredentialRef("local_account", "loopback")
    catalog = await runtime.model_catalog(backend="codex", transport="sdk", auth=auth)
    row = catalog.models[0]
    return await runtime.open_session(
        CodexCatalogSessionRequest(
            auth=auth,
            open=NewSession(),
            cwd=str(cwd.resolve()),
            policy=PermissionPolicy(allowed_tools=("*",)),
            model_key=row.key,
            reasoning="high",
            agent_definition_revision=catalog.definition_revision,
            row_fingerprint=row.row_fingerprint,
            native=CodexNativeOptions(builtin_tools="disabled", web_search=False),
            tools=(
                CanonicalTool(
                    "research__read",
                    "read an available source",
                    {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                ),
            )
            if tools
            else (),
            output=JsonSchemaAgentOutput(
                name="answer",
                schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            ),
        )
    )


async def terminal(runtime: AgentRuntime, handle) -> AgentTerminal:
    events = [
        event
        async for event in runtime.stream_turn(
            handle, TurnRequest(input=(TextContent("return the answer"),))
        )
    ]
    assert isinstance(events[-1], AgentTerminal)
    return events[-1]


@pytest.mark.parametrize(
    "violation", ("absent", "wrong_catalog", "relative", "user_origin", "old_native")
)
async def test_n014_rejects_unqualified_stock_host_before_session_io(
    peer: Peer, tmp_path: Path, violation: str
) -> None:
    if violation == "absent":
        peer.host_configuration["config"].pop("model_catalog_json")
    elif violation == "wrong_catalog":
        peer.host_configuration["config"]["model_catalog_json"] = "/host/unrestricted-models.json"
    elif violation == "relative":
        peer.host_configuration["config"]["model_catalog_json"] = "relative/models.json"
    elif violation == "user_origin":
        peer.host_configuration["origins"]["model_catalog_json"]["name"]["type"] = "user"
    else:
        peer.user_agent = "provider-runtime/0.159.2 (old stock host)"
    async with AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, codex_endpoints={"loopback": peer.socket})
    ) as runtime:
        with pytest.raises(UnsupportedCapability, match="startup.*catalog"):
            await session(runtime, tmp_path, tools=True)
    assert peer.session_requests == [] and peer.starts == 0


async def test_n014_raw_custom_exec_is_authority_not_a_permitted_callback(
    peer: Peer, tmp_path: Path
) -> None:
    peer.mode = "raw_native"
    async with AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, codex_endpoints={"loopback": peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path, tools=True)
        turn = runtime.prepare_turn(
            owner,
            TurnRequest(input=(TextContent("attempt native clock"),)),
            attempt_id="raw-authority",
            input_id="raw-authority-input",
            controls=AgentTurnControls(5, 4, 4096),
        )
        assert isinstance(await turn.submit(), AgentAccepted)
        observed = []
        with pytest.raises(ProtocolDefect):
            async for event in turn.events():
                observed.append(event)
        authority = [event for event in observed if isinstance(event, AgentToolUse)]
        assert len(authority) == 1
        assert authority[0].name == "exec" and authority[0].phase == "started"
        assert turn.terminal is None
        assert (await turn.close()).local_closed


@pytest.mark.parametrize("declared_tools", (False, True))
async def test_n014_contained_request_overrides_model_and_inherited_native_authority(
    peer: Peer, tmp_path: Path, declared_tools: bool
) -> None:
    async with AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, codex_endpoints={"loopback": peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path, tools=declared_tools)
        turn = runtime.prepare_turn(
            owner,
            TurnRequest(input=(TextContent("verify the contained route"),)),
            attempt_id="contained-config-attempt",
            input_id="contained-config-input",
            controls=AgentTurnControls(rpc_seconds=1, pending_calls=1, pending_call_bytes=1024),
        )
        prepared_request = thaw_json_value(turn.submitted_request)
        assert isinstance(prepared_request, dict)
        params = prepared_request["params"]
        assert isinstance(params, dict)
        assert params["environments"] == []
        assert (await turn.close()).local_closed
    assert peer.starts == 0
    initialized = peer.initializations[-1]["params"]
    assert isinstance(initialized, dict)
    assert initialized["capabilities"] == {"experimentalApi": True}
    params = peer.session_requests[0]["params"]
    assert isinstance(params, dict)
    assert params["environments"] == []
    config = params["config"]
    assert isinstance(config, dict)
    assert config["agents"] == {"enabled": False}
    tools = config["tools"]
    assert isinstance(tools, dict)
    assert tools["update_plan"] == {"enabled": False}
    features = config["features"]
    assert isinstance(features, dict)
    for feature in ("sleep_tool", "view_image", "send_message_to_user_async"):
        assert features[feature] is False


@pytest.mark.parametrize(
    "unsupported",
    ("oneOf", "allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else"),
)
async def test_n001_unsupported_output_schema_is_rejected_before_native_io(
    peer: Peer, tmp_path: Path, unsupported: str
) -> None:
    output = JsonSchemaAgentOutput(
        name="unsupported",
        schema={
            "type": "object",
            "properties": {"result": {unsupported: [{"type": "string"}, {"type": "null"}]}},
            "required": ["result"],
            "additionalProperties": False,
        },
    )
    with pytest.raises(InvalidAgentRequest, match=unsupported):
        CodexCatalogSessionRequest(
            auth=CredentialRef("local_account", "test"),
            open=NewSession(),
            cwd=str(tmp_path),
            policy=PermissionPolicy(allowed_tools=("*",)),
            model_key="loopback-model",
            reasoning="high",
            agent_definition_revision="test",
            row_fingerprint="0" * 64,
            output=output,
            native=CodexNativeOptions(builtin_tools="disabled", web_search=False),
        )
    assert peer.connections == []
    assert peer.starts == 0


async def test_n001_schema_keyword_property_names_are_not_schema_keywords(
    peer: Peer, tmp_path: Path
) -> None:
    output = JsonSchemaAgentOutput(
        name="literal_property",
        schema={
            "type": "object",
            "properties": {"oneOf": {"type": "string", "enum": ["oneOf"]}},
            "required": ["oneOf"],
            "additionalProperties": False,
        },
    )
    request = CodexCatalogSessionRequest(
        auth=CredentialRef("local_account", "test"),
        open=NewSession(),
        cwd=str(tmp_path),
        policy=PermissionPolicy(allowed_tools=("*",)),
        model_key="loopback-model",
        reasoning="high",
        agent_definition_revision="test",
        row_fingerprint="0" * 64,
        output=output,
    )
    assert request.output == output
    assert peer.connections == []


@pytest.mark.parametrize(
    "schema, message",
    (
        ({"type": "array", "items": {"type": "string"}}, "root"),
        ({"type": "object", "anyOf": [{"type": "object"}]}, "anyOf"),
        ({"type": "object", "properties": {}}, "additionalProperties"),
        (
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "additionalProperties": False,
            },
            "required",
        ),
        (
            {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
                "$defs": {"answer": {"oneOf": [{"type": "string"}, {"type": "null"}]}},
            },
            "oneOf",
        ),
    ),
)
async def test_n001_established_native_schema_constraints_are_local(
    peer: Peer, tmp_path: Path, schema: dict[str, object], message: str
) -> None:
    with pytest.raises(InvalidAgentRequest, match=message):
        CodexCatalogSessionRequest(
            auth=CredentialRef("local_account", "test"),
            open=NewSession(),
            cwd=str(tmp_path),
            policy=PermissionPolicy(allowed_tools=("*",)),
            model_key="loopback-model",
            reasoning="high",
            agent_definition_revision="test",
            row_fingerprint="0" * 64,
            output=JsonSchemaAgentOutput(name="invalid", schema=schema),
        )
    assert peer.connections == []
    assert peer.starts == 0


async def test_n003_native_success_survives_invalid_product_json(
    peer: Peer, tmp_path: Path
) -> None:
    peer.answer = "not json"
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        observed = await terminal(runtime, await session(runtime, tmp_path))
        assert observed.status == "succeeded"
        assert observed.failure is None
        assert observed.final_text == "not json"
        assert peer.starts == 1


@pytest.mark.parametrize("tail", ["disconnect", "malformed"])
async def test_n004_native_terminal_survives_post_terminal_failure(
    peer: Peer, tmp_path: Path, tail: str
) -> None:
    peer.tail = tail
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        observed = await terminal(runtime, await session(runtime, tmp_path))
        assert observed.status == "succeeded"
        assert observed.final_text == '{"answer":"ok"}'
        assert peer.starts == 1


def prepare(runtime: AgentRuntime, owner):
    turn = runtime.prepare_turn(
        owner,
        TurnRequest(input=(TextContent("return the answer"),)),
        attempt_id="attempt-owned",
        input_id="input-owned",
        controls=AgentTurnControls(rpc_seconds=1, pending_calls=2, pending_call_bytes=8192),
    )
    assert isinstance(turn, _CodexAgentTurn)
    return turn


async def test_n001_prepare_reserves_freezes_and_revoked_submission_is_proven(
    peer: Peer, tmp_path: Path
) -> None:
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        owner = await session(runtime, tmp_path)
        assert runtime.session_usable(owner)
        turn = prepare(runtime, owner)
        assert peer.starts == 0
        assert turn.submission is None
        assert not runtime.session_usable(owner)
        assert (
            turn.attempt.request_digest
            == hashlib.sha256(canonical_json_bytes(turn.submitted_request)).hexdigest()
        )
        params = thaw_json_value(turn.submitted_request["params"])
        assert isinstance(params, dict)
        assert params["approvalPolicy"] == "never"
        turn.revoke()
        assert isinstance(await turn.submit(), AgentNotSubmitted)
        assert peer.starts == 0
        assert [event async for event in turn.events()] == []
        assert (await turn.close()).local_closed
        assert runtime.session_usable(owner)
        with pytest.raises(InvalidAgentRequest, match="exactly once"):
            await turn.submit()


@pytest.mark.parametrize("mode", ["drop", "reject"])
async def test_n002_possible_writer_never_becomes_proven_non_submission(
    peer: Peer, tmp_path: Path, mode: str
) -> None:
    peer.mode = mode
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        owner = await session(runtime, tmp_path)
        turn = prepare(runtime, owner)
        submission = await asyncio.wait_for(turn.submit(), 3)
        assert isinstance(submission, AgentUncertain)
        assert peer.starts == 1
        with pytest.raises((CodexConnectionUnavailable, CodexAppServerResponseError)):
            _ = [event async for event in turn.events()]
        assert turn.terminal is None
        await turn.close()
        assert isinstance(turn.submission, AgentUncertain)
        assert not runtime.session_usable(owner)


async def test_n001_reader_loss_quiesces_the_deferred_writer_before_negative_proof(
    peer: Peer, tmp_path: Path
) -> None:
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        owner = await session(runtime, tmp_path)
        turn = prepare(runtime, owner)
        lock = turn._state.client._write_lock
        await lock.acquire()
        try:
            pending = asyncio.create_task(turn.submit())
            await asyncio.sleep(0)
            assert turn.submission is None
            await peer.connections[-1].send("not-json")
            proof = await asyncio.wait_for(pending, 3)
            assert isinstance(proof, AgentNotSubmitted)
            assert "malformed JSON" in proof.reason
        finally:
            lock.release()
        with pytest.raises(ProtocolDefect, match="malformed JSON"):
            _ = [event async for event in turn.events()]
        await turn.close()
        assert peer.starts == 0


async def test_n001_deadline_before_writer_entry_preserves_original_safe_failure(
    peer: Peer, tmp_path: Path
) -> None:
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        owner = await session(runtime, tmp_path)
        turn = runtime.prepare_turn(
            owner,
            TurnRequest(input=(TextContent("return the answer"),), timeout_seconds=0.02),
            attempt_id="deadline-before-entry",
            input_id="deadline-before-entry-input",
            controls=AgentTurnControls(rpc_seconds=1, pending_calls=2, pending_call_bytes=8192),
        )
        assert isinstance(turn, _CodexAgentTurn)
        lock = turn._state.client._write_lock
        await lock.acquire()
        try:
            proof = await asyncio.wait_for(turn.submit(), 3)
            assert isinstance(proof, AgentNotSubmitted)
            assert "timed out" in proof.reason
        finally:
            lock.release()
        with pytest.raises(TurnNotStarted):
            _ = [event async for event in turn.events()]
        await turn.close()
        assert peer.starts == 0
        assert runtime.session_usable(owner)


async def test_n009_callback_before_start_ack_executes_exact_reply_then_seals(
    peer: Peer, tmp_path: Path
) -> None:
    peer.mode = "callback_before_ack"
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        owner = await session(runtime, tmp_path, tools=True)
        turn = prepare(runtime, owner)
        submission = await asyncio.wait_for(turn.submit(), 3)
        assert isinstance(submission, AgentAccepted)
        assert peer.received.empty()
        events = turn.events()
        call = None
        async for event in events:
            if isinstance(event, AgentToolCall):
                call = event
                break
        assert isinstance(call, AgentToolCall)
        assert call.name == "research__read"
        assert call.arguments == {"query": "known source"}
        assert call.turn == submission.turn
        await turn.reply(call, AgentToolReply(text="recorded source result", success=True))
        reply = await peer.received.get()
        assert reply["id"] == "callback-1"
        assert reply["result"] == {
            "contentItems": [{"type": "inputText", "text": "recorded source result"}],
            "success": True,
        }
        tail = [event async for event in events]
        assert isinstance(tail[-1], AgentTerminal)
        terminal = tail[-1]
        assert isinstance(terminal.evidence, NativeTerminalEvidence)
        assert terminal.evidence.attempt == turn.attempt
        assert turn.terminal is terminal
        assert terminal_from_json(terminal_to_json(terminal)) == terminal
        assert peer.starts == 1
        with pytest.raises(SessionUnavailable, match="revoked or ended"):
            await turn.reply(call, AgentToolReply(text="duplicate", success=True))
        await turn.close()
        assert runtime.session_usable(owner)


async def test_n015_interrupted_native_seal_can_abort_a_pending_callback(
    peer: Peer, tmp_path: Path
) -> None:
    peer.mode = "callback_before_ack"
    async with AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, codex_endpoints={"loopback": peer.socket})
    ) as runtime:
        owner = await session(runtime, tmp_path, tools=True)
        turn = runtime.prepare_turn(
            owner,
            TurnRequest(input=(TextContent("read the source"),)),
            attempt_id="aborted-pending-callback",
            input_id="first-input",
            controls=AgentTurnControls(rpc_seconds=1, pending_calls=2, pending_call_bytes=32768),
        )
        assert isinstance(await turn.submit(), AgentAccepted)
        events = turn.events()
        call = await anext(events)
        assert isinstance(call, AgentToolCall)
        turn.revoke()
        assert (await turn.interrupt()).disposition == "accepted"
        result = [event async for event in events]
        assert isinstance(result[-1], AgentTerminal)
        assert result[-1].status == "cancelled"
        assert isinstance(result[-1].evidence, NativeTerminalEvidence)
        assert turn.terminal is result[-1]
        assert (await turn.close()).local_closed
        assert not runtime.session_usable(owner)
        with pytest.raises(SessionUnavailable):
            await turn.reply(call, AgentToolReply(text="late result", success=True))


async def test_n013_steer_ack_recorded_input_and_progress_are_distinct(
    peer: Peer, tmp_path: Path
) -> None:
    peer.mode = "hold"
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        owner = await session(runtime, tmp_path)
        turn = prepare(runtime, owner)
        submission = await turn.submit()
        assert isinstance(submission, AgentAccepted)
        receipt = await turn.steer(input_id="new-request", input=(TextContent("followup"),))
        assert receipt.disposition == "accepted"
        events = turn.events()
        recorded = await anext(events)
        progress = await anext(events)
        assert isinstance(recorded, AgentInputRecorded)
        assert recorded.input_id == "new-request"
        assert isinstance(progress, AgentMessage)
        assert progress.phase == "commentary"
        interrupted = await turn.interrupt()
        assert interrupted.disposition == "accepted"
        tail = [event async for event in events]
        assert isinstance(tail[-1], AgentTerminal)
        assert tail[-1].status == "cancelled"
        assert isinstance(tail[-1].evidence, NativeTerminalEvidence)
        await turn.close()


async def test_n015_abandoned_handle_does_not_close_sibling_connection(
    peer: Peer, tmp_path: Path
) -> None:
    peer.mode = "hold"
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        first = await session(runtime, tmp_path)
        second = await session(runtime, tmp_path)
        first_turn = prepare(runtime, first)
        second_turn = prepare(runtime, second)
        first_submission = await first_turn.submit()
        second_submission = await second_turn.submit()
        assert isinstance(first_submission, AgentAccepted)
        assert isinstance(second_submission, AgentAccepted)
        assert first_submission.turn.session_ref != second_submission.turn.session_ref
        first_turn.revoke()
        await first_turn.close()
        receipt = await second_turn.steer(
            input_id="sibling-input", input=(TextContent("continue"),)
        )
        assert receipt.disposition == "accepted"
        await second_turn.interrupt()
        sibling_events = [event async for event in second_turn.events()]
        assert any(isinstance(event, AgentInputRecorded) for event in sibling_events)
        assert isinstance(sibling_events[-1], AgentTerminal)
        assert sibling_events[-1].status == "cancelled"
        await second_turn.close()


async def test_n003_decoder_rejects_product_json_after_original_proof(
    peer: Peer, tmp_path: Path
) -> None:
    peer.answer = "not json"
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"loopback": peer.socket})) as runtime:
        owner = await session(runtime, tmp_path)
        turn = prepare(runtime, owner)
        await turn.submit()
        events = [event async for event in turn.events()]
        terminal = events[-1]
        assert isinstance(terminal, AgentTerminal)
        saved = terminal_to_json(terminal)
        with pytest.raises(OutputSchemaMismatch):
            decode_agent_output(
                JsonSchemaAgentOutput(name="answer", schema={"type": "object"}), terminal
            )
        restored = terminal_from_json(saved)
        assert restored.status == "succeeded"
        assert restored.final_text == "not json"
        assert restored.evidence == terminal.evidence
        await turn.close()
