"""Public shared-Codex behavior at the external protocol boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("websockets.asyncio.server")

from websockets.asyncio.server import ServerConnection, unix_serve

from provider_runtime.agent_runtime import (
    AgentPermissionRequest,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentSessionRequest,
    AgentTerminal,
    AgentText,
    CodexNativeOptions,
    CredentialRef,
    CredentialUnavailable,
    HeaderReference,
    McpServerSpec,
    NewSession,
    PermissionPolicy,
    ProtocolDefect,
    SessionQuery,
    TextContent,
    TurnRequest,
    UnsafeConfirmation,
    UnsupportedCapability,
)
from provider_runtime.agent_runtime.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerConfig,
    CodexNotification,
)
from provider_runtime.agent_runtime.codex_control import (
    CodexBounded,
    CodexControlError,
    CodexCreateRequest,
    CodexFinished,
    CodexInterrupted,
    CodexListRequest,
    CodexPromptRequest,
    CodexStale,
    CodexSteer,
    CodexSubmit,
    CodexThreadTarget,
    CodexTurnTarget,
    CodexUnknown,
)

THREAD = "01992818-9220-714c-9c91-e39d3f006e64"
TURN = "01992818-9221-714c-9c91-e39d3f006e64"


class ProtocolPeer:
    """An independent external App Server protocol peer, never a provider."""

    def __init__(self, socket: Path) -> None:
        self.socket = socket
        self.messages: list[dict[str, object]] = []
        self.closed_connections = 0
        self.version = "0.153.4"
        self.emit_managed = False
        self.emit_approval = False
        self.foreign_noise = False
        self.turn_status = "inProgress"
        self.drop_method: str | None = None
        self.close_code = 1000
        self.errors: dict[str, str] = {}
        self.error_code = -32600
        self.answer = "synthetic answer"
        self.answer_phase = "final_answer"
        self.older_items: list[dict[str, object]] = []
        self.interrupt_settles = True
        self.status = "idle"
        self.intervene = False
        self.name = "synthetic worker"
        self.raw_reply: str | None = None
        self.between_turns = False
        self.scripted: dict[str, object] | None = None
        self.user_message_id: str | None = None
        self.hold_method: str | None = None
        self.received = asyncio.Event()
        self.release = asyncio.Event()
        self.startup_message: dict[str, object] = {
            "method": "account/updated",
            "params": {"authMode": None, "planType": None},
            "emittedAtMs": 1234,
        }

    async def handle(self, connection: ServerConnection) -> None:
        try:
            async for frame in connection:
                message = json.loads(frame)
                self.messages.append(message)
                method = message.get("method")
                if method is None:
                    continue
                if method == "initialized":
                    continue
                if method == self.hold_method:
                    self.received.set()
                    await self.release.wait()
                    return
                if method == "thread/list" and self.raw_reply is not None:
                    await connection.send(self.raw_reply)
                    continue
                if method == self.drop_method:
                    await connection.close(code=self.close_code)
                    return
                if method in self.errors:
                    await connection.send(
                        json.dumps(
                            {
                                "id": message["id"],
                                "error": {
                                    "code": self.error_code,
                                    "message": self.errors[method],
                                },
                            }
                        )
                    )
                    continue
                if method == "initialize":
                    await connection.send(json.dumps(self.startup_message))
                    result = {"userAgent": f"codex_cli_rs/{self.version} (Linux synthetic; x86_64)"}
                elif method == "account/read":
                    result = {"account": {"type": "chatgpt"}}
                elif method == "thread/list":
                    result = {"data": [self.thread()], "nextCursor": None}
                elif method == "thread/start":
                    result = {"thread": self.thread()}
                elif method == "thread/read":
                    result = {"thread": self.thread()}
                elif method == "thread/turns/list":
                    view = message["params"].get("itemsView", "summary")
                    result = {
                        "data": [
                            {
                                "id": TURN,
                                "status": self.turn_status,
                                "itemsView": view,
                                "items": [] if view == "notLoaded" else self.turn_items(),
                            }
                        ],
                        "nextCursor": None,
                    }
                elif method == "thread/items/list":
                    params = message["params"]
                    assert params["threadId"] == THREAD and params["turnId"] == TURN
                    assert params["sortDirection"] == "desc"
                    assert params.get("cursor") is None
                    items = self.turn_items()
                    limit = params["limit"]
                    result = {
                        "data": [
                            {"turnId": TURN, "item": item} for item in reversed(items[-limit:])
                        ],
                        "nextCursor": "older-items" if len(items) > limit else None,
                    }
                elif method == "thread/unsubscribe":
                    result = {"status": "unsubscribed"}
                elif method == "turn/start":
                    self.user_message_id = message["params"].get("clientUserMessageId")
                    result = {"turn": {"id": TURN, "status": "inProgress", "items": []}}
                elif method == "turn/steer":
                    result = {"turnId": TURN}
                elif method == "turn/interrupt":
                    if self.interrupt_settles:
                        self.turn_status = "interrupted"
                    result = {}
                else:
                    raise AssertionError(f"unexpected fixture request: {method}")
                await connection.send(json.dumps({"id": message["id"], "result": result}))
                if method == "turn/start":
                    await self.turn_events(connection)
        finally:
            self.closed_connections += 1

    def thread(self) -> dict[str, object]:
        return {
            "id": THREAD,
            "cwd": "/workspace",
            "name": self.name,
            "source": "appServer",
            "status": {"type": self.status},
            "turns": [],
        }

    def turn_items(self) -> list[dict[str, object]]:
        return [
            *self.older_items,
            {
                "id": "answer",
                "type": "agentMessage",
                "text": self.answer,
                "phase": self.answer_phase,
            },
        ]

    async def turn_events(self, connection: ServerConnection) -> None:
        scope = {"threadId": THREAD, "turnId": TURN}
        if self.foreign_noise:
            await connection.send(
                json.dumps(
                    {
                        "method": "item/tool/call",
                        "id": "foreign",
                        "params": {**scope, "threadId": "01992818-9222-714c-9c91-e39d3f006e64"},
                    }
                )
            )
            await connection.send(
                json.dumps(
                    {
                        "method": "turn/completed",
                        "params": {
                            "threadId": "01992818-9222-714c-9c91-e39d3f006e64",
                            "turn": {"id": TURN, "status": "completed"},
                        },
                    }
                )
            )
        if self.emit_approval:
            if self.emit_managed:
                await connection.send(
                    json.dumps(
                        {
                            "method": "item/started",
                            "params": {
                                **scope,
                                "item": {
                                    "id": "command",
                                    "type": "commandExecution",
                                    "command": "synthetic",
                                    "cwd": "/workspace",
                                    "status": "inProgress",
                                },
                            },
                        }
                    )
                )
            await connection.send(
                json.dumps(
                    {
                        "method": "item/commandExecution/requestApproval",
                        "id": "approval",
                        "params": {**scope, "itemId": "command", "command": "synthetic"},
                    }
                )
            )
        if self.emit_managed:
            await connection.send(
                json.dumps(
                    {
                        "method": "item/completed",
                        "params": {
                            **scope,
                            "item": {
                                "id": "own-user-input",
                                "type": "userMessage",
                                "clientId": self.user_message_id,
                                "content": [],
                            },
                        },
                    }
                )
            )
            if self.intervene:
                await connection.send(
                    json.dumps(
                        {
                            "method": "item/completed",
                            "params": {
                                **scope,
                                "item": {
                                    "id": "other-user-message",
                                    "type": "userMessage",
                                    "clientId": "another-client",
                                    "content": [],
                                },
                            },
                        }
                    )
                )
            if self.scripted is not None:
                scripted_events = self.scripted["events"]
                assert isinstance(scripted_events, list)
                for event in scripted_events:
                    item = {
                        key: value for key, value in event.items() if key in ("id", "text", "phase")
                    }
                    item["type"] = "agentMessage"
                    await connection.send(
                        json.dumps({"method": "item/completed", "params": {**scope, "item": item}})
                    )
                await connection.send(
                    json.dumps(
                        {
                            "method": "turn/completed",
                            "params": {
                                "threadId": THREAD,
                                "turn": {
                                    "id": TURN,
                                    "status": self.scripted["status"],
                                    "items": [],
                                },
                            },
                        }
                    )
                )
                return
            for method, params in [
                (
                    "turn/started",
                    {"threadId": THREAD, "turn": {"id": TURN, "status": "inProgress", "items": []}},
                ),
                ("item/agentMessage/delta", {**scope, "itemId": "answer", "delta": self.answer}),
                (
                    "item/completed",
                    {
                        **scope,
                        "item": {
                            "id": "answer",
                            "type": "agentMessage",
                            "text": self.answer,
                            "phase": "final_answer",
                        },
                    },
                ),
                (
                    "turn/completed",
                    {"threadId": THREAD, "turn": {"id": TURN, "status": "completed", "items": []}},
                ),
            ]:
                await connection.send(json.dumps({"method": method, "params": params}))
            if self.between_turns:
                await connection.send(
                    json.dumps(
                        {
                            "method": "turn/started",
                            "params": {
                                "threadId": THREAD,
                                "turn": {
                                    "id": "01992818-9229-714c-9c91-e39d3f006e64",
                                    "status": "inProgress",
                                    "items": [],
                                },
                            },
                        }
                    )
                )


@pytest.fixture
async def peer(tmp_path: Path) -> AsyncIterator[ProtocolPeer]:
    # macOS sockaddr_un is limited to 104 bytes; pytest's ordinary path is longer.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="codex-peer-", dir="/tmp") as directory:
        value = ProtocolPeer(Path(directory) / "peer.sock")
        async with await unix_serve(value.handle, str(value.socket)):
            yield value


async def test_timestamped_startup_notification_preserves_correlated_protocol(
    peer: ProtocolPeer,
) -> None:
    async with CodexAppServerClient(CodexAppServerConfig(peer.socket)) as client:
        assert await client.next_message() == CodexNotification(
            "account/updated", {"authMode": None, "planType": None}
        )
        assert await client.account() == {"account": {"type": "chatgpt"}}


@pytest.mark.parametrize(
    "fields",
    [
        {"emittedAtMs": True},
        {"emittedAtMs": 1.5},
        {"emittedAtMs": "1234"},
        {"emittedAtMs": -(1 << 63) - 1},
        {"emittedAtMs": 1 << 63},
        {"trace": {}},
        {"unknown": None},
        {"id": 1},
    ],
)
async def test_observer_rejects_malformed_startup_envelopes(
    peer: ProtocolPeer, fields: dict[str, object]
) -> None:
    peer.startup_message.update(fields)
    with pytest.raises(ProtocolDefect):
        async with CodexAppServerClient(
            CodexAppServerConfig(peer.socket, request_policy="observe_only")
        ):
            pytest.fail("malformed startup envelope was accepted")


async def test_public_control_creates_unsubscribes_then_submits_and_closes_only_connections(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    async with AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, codex_endpoints={"lab": peer.socket})
    ) as runtime:
        page = await runtime.codex.list(CodexListRequest("lab"))
        assert page.threads[0].target.thread_handle == THREAD
        target = await runtime.codex.create(CodexCreateRequest("lab", tmp_path))
        assert peer.messages[-1]["method"] == "thread/unsubscribe"
        turn = await runtime.codex.prompt(
            CodexPromptRequest(target, CodexSubmit("synthetic input"))
        )
        assert turn.thread == target
        assert turn.turn_handle == TURN
    methods = [message.get("method") for message in peer.messages]
    assert methods.count("thread/start") == 1
    assert methods.count("turn/start") == 1
    assert "thread/resume" not in methods
    assert tuple(tmp_path.iterdir()) == ()
    # The endpoint remains usable after client shutdown; no process/server teardown.
    async with AgentRuntime(
        AgentRuntimeConfig(state_root_base=tmp_path, codex_endpoints={"lab": peer.socket})
    ) as runtime:
        assert (await runtime.codex.list(CodexListRequest("lab"))).threads


async def test_submit_steer_interrupt_and_read_report_native_facts_without_retry(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    target = CodexThreadTarget("lab", THREAD)
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        accepted = await runtime.codex.prompt(CodexPromptRequest(target, CodexSubmit("synthetic")))
        steered = await runtime.codex.prompt(
            CodexPromptRequest(target, CodexSteer(TURN, "synthetic"))
        )
        assert accepted == steered
        read = await runtime.codex.read(target)
        assert read.turn is not None and read.turn.target == accepted
        assert read.last_answer == "synthetic answer"
        assert isinstance(await runtime.codex.interrupt(accepted), CodexInterrupted)
        peer.turn_status = "completed"
        assert await runtime.codex.interrupt(accepted) == CodexFinished("completed")
        stale = CodexTurnTarget(target, "01992818-9229-714c-9c91-e39d3f006e64")
        assert isinstance(await runtime.codex.interrupt(stale), CodexStale)
        peer.turn_status = "inProgress"
        peer.interrupt_settles = False
        assert isinstance(await runtime.codex.interrupt(accepted), CodexUnknown)
        peer.answer = "x" * (65 * 1024)
        bounded = await runtime.codex.read(target)
        assert bounded.last_answer is None and isinstance(bounded.coverage, CodexBounded)
    methods = [m.get("method") for m in peer.messages]
    assert methods.count("turn/start") == 1
    assert methods.count("turn/steer") == 1
    assert methods.count("turn/interrupt") == 2
    steer = next(m for m in peer.messages if m.get("method") == "turn/steer")
    assert isinstance(steer["params"], dict)
    assert steer["params"]["expectedTurnId"] == TURN


async def test_worker_requests_remain_unanswered_and_lost_mutations_are_unknown(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.emit_approval = True
    target = CodexThreadTarget("lab", THREAD)
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        await runtime.codex.prompt(CodexPromptRequest(target, CodexSubmit("synthetic")))
        peer.drop_method = "turn/start"
        with pytest.raises(CodexControlError) as failure:
            await runtime.codex.prompt(CodexPromptRequest(target, CodexSubmit("synthetic")))
        assert failure.value.dispatch == "Unknown" and failure.value.known_thread == target
        peer.drop_method = "thread/unsubscribe"
        with pytest.raises(CodexControlError) as partial:
            await runtime.codex.create(CodexCreateRequest("lab", tmp_path))
        assert partial.value.dispatch == "Unknown" and partial.value.known_thread == target
    assert not [message for message in peer.messages if message.get("id") == "approval"]
    assert sum(message.get("method") == "turn/start" for message in peer.messages) == 2


@pytest.mark.parametrize(
    "message,code",
    [
        ("expected active turn id old but found new", "stale"),
        ("no active turn to steer", "stale"),
        ("thread not found", "missing"),
        ("cannot steer a review turn", "busy"),
        ("usage limit exceeded", "quota"),
    ],
)
async def test_native_rejections_remain_known_and_do_not_dispatch_again(
    tmp_path: Path, peer: ProtocolPeer, message: str, code: str
) -> None:
    peer.errors["turn/steer"] = message
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        with pytest.raises(CodexControlError) as failure:
            await runtime.codex.prompt(
                CodexPromptRequest(CodexThreadTarget("lab", THREAD), CodexSteer(TURN, "synthetic"))
            )
        assert (failure.value.code, failure.value.dispatch) == (code, "Rejected")
    assert sum(m.get("method") == "turn/steer" for m in peer.messages) == 1


async def test_managed_cognition_filters_foreign_events_and_remains_protected(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.emit_managed = peer.foreign_noise = True
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        session = await runtime.open_session(
            AgentSessionRequest(
                backend="codex",
                transport="sdk",
                auth=CredentialRef("local_account", "lab"),
                cwd=str(tmp_path),
                open=NewSession(),
                policy=PermissionPolicy(allowed_tools=("*",)),
                native=CodexNativeOptions(builtin_tools="disabled"),
            )
        )
        target = CodexThreadTarget("lab", session.ref.native_session_id)
        with pytest.raises(CodexControlError, match="unauthorized"):
            await runtime.codex.prompt(CodexPromptRequest(target, CodexSubmit("synthetic")))
        events = [
            event
            async for event in runtime.stream_turn(
                session, TurnRequest(input=(TextContent("synthetic"),))
            )
        ]
        assert any(
            isinstance(event, AgentText) and event.text == "synthetic answer" for event in events
        )
        assert isinstance(events[-1], AgentTerminal) and events[-1].status == "succeeded"
        assert not [message for message in peer.messages if message.get("id") == "foreign"]
        await runtime.close_session(session)
        assert (await runtime.codex.list(CodexListRequest("lab"))).threads


async def test_managed_native_approval_is_denied_and_poisons_the_session(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.emit_managed = peer.emit_approval = True
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        session = await runtime.open_session(
            AgentSessionRequest(
                backend="codex",
                transport="sdk",
                auth=CredentialRef("local_account", "lab"),
                cwd=str(tmp_path),
                open=NewSession(),
                policy=PermissionPolicy(allowed_tools=("*",)),
                native=CodexNativeOptions(builtin_tools="disabled"),
            )
        )
        events = []
        with pytest.raises(ProtocolDefect) as failure:
            async for event in runtime.stream_turn(
                session, TurnRequest(input=(TextContent("synthetic"),))
            ):
                events.append(event)
        assert any(isinstance(event, AgentPermissionRequest) for event in events), str(
            failure.value
        )
        assert not any(isinstance(event, AgentTerminal) for event in events)
    assert next(m for m in peer.messages if m.get("id") == "approval")["result"] == {
        "decision": "decline"
    }


async def test_observed_foreign_user_input_invalidates_cognition_before_terminal(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.emit_managed = peer.intervene = True
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        session = await runtime.open_session(
            AgentSessionRequest(
                backend="codex",
                transport="sdk",
                auth=CredentialRef("local_account", "lab"),
                cwd=str(tmp_path),
                open=NewSession(),
                policy=PermissionPolicy(allowed_tools=("*",)),
                native=CodexNativeOptions(builtin_tools="disabled"),
            )
        )
        with pytest.raises(ProtocolDefect, match="foreign user input"):
            await runtime.run_turn(session, TurnRequest(input=(TextContent("synthetic"),)))


async def test_oversized_metadata_has_honest_list_error_and_bounded_read(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.name = "x" * (65 * 1024)
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        with pytest.raises(CodexControlError) as failure:
            await runtime.codex.list(CodexListRequest("lab"))
        assert (failure.value.code, failure.value.dispatch) == ("output_limit", "NotSent")
        read = await runtime.codex.read(CodexThreadTarget("lab", THREAD))
        assert read.thread.name is None and isinstance(read.coverage, CodexBounded)


async def test_bounded_native_item_page_preserves_latest_final_and_interrupt_is_metadata_only(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.older_items = [
        {
            "id": "large-output",
            "type": "commandExecution",
            "aggregatedOutput": "x" * (5 * 1024 * 1024),
        },
        *(
            {
                "id": f"answer-{index}",
                "type": "agentMessage",
                "text": f"final-{index}",
                "phase": "final_answer",
            }
            for index in range(60)
        ),
    ]
    peer.answer_phase = "commentary"
    target = CodexThreadTarget("lab", THREAD)
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        read = await runtime.codex.read(target)
        assert read.last_answer == "final-59"
        assert isinstance(read.coverage, CodexBounded)
        peer.answer = "x" * (5 * 1024 * 1024)
        bounded = await runtime.codex.read(target)
        assert bounded.last_answer is None and isinstance(bounded.coverage, CodexBounded)
        assert bounded.turn is not None and bounded.turn.target == CodexTurnTarget(target, TURN)
        before = len(peer.messages)
        assert isinstance(
            await runtime.codex.interrupt(CodexTurnTarget(target, TURN)), CodexInterrupted
        )
        assert not any(
            message.get("method") == "thread/items/list" for message in peer.messages[before:]
        )
    pages = [message for message in peer.messages if message.get("method") == "thread/items/list"]
    assert len(pages) == 2
    for message in pages:
        assert isinstance(message["params"], dict) and message["params"]["limit"] == 50
    turns = [message for message in peer.messages if message.get("method") == "thread/turns/list"]
    for message in turns:
        assert isinstance(message["params"], dict) and message["params"]["itemsView"] == "notLoaded"


@pytest.mark.parametrize("mutation", [False, True])
async def test_oversized_native_frame_reports_output_limit_without_replaying_mutation(
    tmp_path: Path, peer: ProtocolPeer, mutation: bool
) -> None:
    peer.name = "x" * (5 * 1024 * 1024)
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        with pytest.raises(CodexControlError) as failure:
            if mutation:
                await runtime.codex.create(CodexCreateRequest("lab", tmp_path))
            else:
                await runtime.codex.list(CodexListRequest("lab"))
        assert (failure.value.code, failure.value.dispatch) == (
            "output_limit",
            "Unknown" if mutation else "NotSent",
        )
    method = "thread/start" if mutation else "thread/list"
    assert sum(message.get("method") == method for message in peer.messages) == 1


async def test_oversized_managed_frame_remains_a_fatal_protocol_defect(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.emit_managed = True
    peer.answer = "x" * (5 * 1024 * 1024)
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        session = await runtime.open_session(
            AgentSessionRequest(
                backend="codex",
                transport="sdk",
                auth=CredentialRef("local_account", "lab"),
                cwd=str(tmp_path),
                open=NewSession(),
                policy=PermissionPolicy(allowed_tools=("*",)),
                native=CodexNativeOptions(builtin_tools="disabled"),
            )
        )
        observed = []
        with pytest.raises(ProtocolDefect):
            async for event in runtime.stream_turn(
                session, TurnRequest(input=(TextContent("synthetic"),))
            ):
                observed.append(event)
        assert not any(isinstance(event, AgentTerminal) for event in observed)


async def test_peer_initiated_size_close_is_not_a_locally_observed_output_limit(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.drop_method = "thread/list"
    peer.close_code = 1009
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        with pytest.raises(CodexControlError) as failure:
            await runtime.codex.list(CodexListRequest("lab"))
        assert (failure.value.code, failure.value.dispatch) == ("unavailable", "NotSent")


async def test_structurally_oversized_native_page_retains_bounded_thread_metadata(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.older_items = [
        {
            "id": "many-parts",
            "type": "userMessage",
            "content": [{"type": "text", "text": "synthetic"} for _ in range(30_000)],
        }
    ]
    assert len(json.dumps(peer.turn_items()).encode()) < 4 * 1024 * 1024
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        result = await runtime.codex.read(CodexThreadTarget("lab", THREAD))
        assert result.turn is not None and result.turn.target.turn_handle == TURN
        assert result.last_answer is None and isinstance(result.coverage, CodexBounded)


@pytest.mark.parametrize(
    "method,detail,bounded",
    [
        ("thread/items/list", "thread/items/list is not supported yet", True),
        ("thread/items/list", "unknown method", False),
        ("thread/turns/list", "thread/items/list is not supported yet", False),
    ],
)
async def test_only_exact_native_item_paging_refusal_preserves_bounded_metadata(
    tmp_path: Path, peer: ProtocolPeer, method: str, detail: str, bounded: bool
) -> None:
    peer.errors[method] = detail
    peer.error_code = -32601
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        target = CodexThreadTarget("lab", THREAD)
        if bounded:
            result = await runtime.codex.read(target)
            assert result.turn is not None and result.turn.target.turn_handle == TURN
            assert result.last_answer is None and isinstance(result.coverage, CodexBounded)
        else:
            with pytest.raises(ProtocolDefect):
                await runtime.codex.read(target)
    methods = [message.get("method") for message in peer.messages]
    assert methods.count("thread/items/list") <= 1
    assert methods.count("thread/turns/list") == 1


@pytest.mark.parametrize(
    "frame",
    [
        '{"id":1,"id":2,"result":{}}',
        '{"id":99999,"result":{}}',
        '{"id":3,"result":NaN}',
        '{"id":3,"result":{},"error":{}}',
    ],
)
async def test_malformed_correlated_protocol_is_a_defect_not_empty_inventory(
    tmp_path: Path, peer: ProtocolPeer, frame: str
) -> None:
    peer.raw_reply = frame
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        with pytest.raises(ProtocolDefect):
            await runtime.codex.list(CodexListRequest("lab"))


async def test_pin_failure_and_one_unavailable_profile_do_not_fall_back(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    async with AgentRuntime(
        AgentRuntimeConfig(
            tmp_path, {"lab": peer.socket, "offline": peer.socket.parent / "absent.sock"}
        )
    ) as runtime:
        with pytest.raises(CodexControlError) as failure:
            await runtime.codex.list(CodexListRequest("offline"))
        assert failure.value.code == "unavailable"
        assert peer.messages == []
        assert (await runtime.codex.list(CodexListRequest("lab"))).threads
        peer.version = "0.144.4"
        with pytest.raises(ProtocolDefect, match="pin"):
            await runtime.codex.list(CodexListRequest("lab"))


async def test_cancelled_create_disconnects_without_resending_or_server_cleanup(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.hold_method = "thread/start"
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        operation = asyncio.create_task(runtime.codex.create(CodexCreateRequest("lab", tmp_path)))
        await asyncio.wait_for(peer.received.wait(), 2)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        peer.release.set()
        assert (await runtime.codex.list(CodexListRequest("lab"))).threads
    assert sum(m.get("method") == "thread/start" for m in peer.messages) == 1


async def test_shared_codex_rejects_client_secret_environment_before_resolution(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    resolved = []

    async def resolver(name: str) -> str:
        resolved.append(name)
        return "synthetic-secret"

    async with AgentRuntime(
        AgentRuntimeConfig(tmp_path, {"lab": peer.socket}, secret_resolver=resolver)
    ) as runtime:
        with pytest.raises(UnsupportedCapability, match="client environment"):
            await runtime.open_session(
                AgentSessionRequest(
                    backend="codex",
                    transport="sdk",
                    auth=CredentialRef("local_account", "lab"),
                    cwd=str(tmp_path),
                    open=NewSession(),
                    policy=PermissionPolicy(
                        allowed_tools=("*",),
                        filesystem="full_access",
                        network="unrestricted",
                        unsafe_confirmation=UnsafeConfirmation(
                            ("filesystem_full_access", "network_unrestricted")
                        ),
                    ),
                    mcp_servers=(
                        McpServerSpec(
                            name="external",
                            transport="streamable_http",
                            url="https://synthetic.invalid/mcp",
                            header_refs=(
                                HeaderReference(
                                    name="Authorization",
                                    source=CredentialRef(
                                        "secret_reference", "lab", "synthetic-reference"
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            )
    assert resolved == [] and peer.messages == []


async def test_queued_intervention_rejects_next_managed_submit_before_dispatch(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    peer.emit_managed = peer.between_turns = True
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        session = await runtime.open_session(
            AgentSessionRequest(
                backend="codex",
                transport="sdk",
                auth=CredentialRef("local_account", "lab"),
                cwd=str(tmp_path),
                open=NewSession(),
                policy=PermissionPolicy(allowed_tools=("*",)),
                native=CodexNativeOptions(builtin_tools="disabled"),
            )
        )
        assert (
            await runtime.run_turn(session, TurnRequest(input=(TextContent("synthetic"),)))
        ).status == "succeeded"
        with pytest.raises(ProtocolDefect, match="pre-turn"):
            await runtime.run_turn(session, TurnRequest(input=(TextContent("synthetic"),)))
    assert sum(m.get("method") == "turn/start" for m in peer.messages) == 1


@pytest.mark.parametrize(
    "case,expected",
    [
        ("final_then_commentary", "authoritative final"),
        ("multiple_final_answers", "newest final"),
        ("last_unknown_fallback", "compatible unknown"),
        ("completed_beats_deltas", "authoritative completed text"),
        ("commentary_only", None),
        ("duplicate_identity", None),
        ("malformed_identity", None),
    ],
)
async def test_completed_message_projection_preserves_the_existing_structured_boundary(
    tmp_path: Path, peer: ProtocolPeer, case: str, expected: str | None
) -> None:
    peer.emit_managed = True
    cases = json.loads(
        (
            Path(__file__).parent / "fixtures/agent_runtime/codex/assistant_message_cases.json"
        ).read_text()
    )
    peer.scripted = cases[case]
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        session = await runtime.open_session(
            AgentSessionRequest(
                backend="codex",
                transport="sdk",
                auth=CredentialRef("local_account", "lab"),
                cwd=str(tmp_path),
                open=NewSession(),
                policy=PermissionPolicy(allowed_tools=("*",)),
                native=CodexNativeOptions(builtin_tools="disabled"),
            )
        )
        if expected is None:
            with pytest.raises(ProtocolDefect):
                await runtime.run_turn(session, TurnRequest(input=(TextContent("synthetic"),)))
        else:
            result = await runtime.run_turn(session, TurnRequest(input=(TextContent("synthetic"),)))
            assert result.final_text == expected


async def test_unconfigured_codex_profile_never_enrolls_a_private_home(tmp_path: Path) -> None:
    runtime = AgentRuntime(AgentRuntimeConfig(state_root_base=tmp_path))
    try:
        with pytest.raises(CredentialUnavailable):
            await runtime.list_sessions(
                SessionQuery(
                    backend="codex",
                    transport="sdk",
                    auth=CredentialRef(kind="local_account", profile_key="unconfigured"),
                )
            )
    finally:
        await runtime.close()
        assert tuple(tmp_path.iterdir()) == (), "Codex attachment must not enroll a private home"


async def test_create_delegates_cwd_existence_to_the_external_server(
    tmp_path: Path, peer: ProtocolPeer
) -> None:
    server_cwd = tmp_path / "exists-only-in-server-filesystem"
    assert not server_cwd.exists()
    async with AgentRuntime(AgentRuntimeConfig(tmp_path, {"lab": peer.socket})) as runtime:
        target = await runtime.codex.create(CodexCreateRequest("lab", server_cwd))
        assert target == CodexThreadTarget("lab", THREAD)
    created = [message for message in peer.messages if message.get("method") == "thread/start"]
    assert len(created) == 1
    params = created[0]["params"]
    assert isinstance(params, dict) and params["cwd"] == str(server_cwd)
    assert any(message.get("method") == "thread/unsubscribe" for message in peer.messages)
