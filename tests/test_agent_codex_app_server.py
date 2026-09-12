"""Shared transport contracts at a real local WebSocket boundary."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest
from websockets.asyncio.server import ServerConnection, unix_serve
from websockets.exceptions import ConnectionClosed

from provider_runtime.agent_runtime import ProtocolDefect
from provider_runtime.agent_runtime.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerConfig,
    CodexAppServerResponseError,
    CodexConnectionUnavailable,
    CodexNotification,
    CodexServerRequest,
)

_CASES = cast(
    dict[str, dict[str, object]],
    json.loads(
        (
            Path(__file__).parent / "fixtures/agent_runtime/codex/app_server_protocol_cases.json"
        ).read_text(encoding="utf-8")
    ),
)
_THREAD = "thread-synthetic"
type Transport = tuple[CodexAppServerClient, ServerConnection, asyncio.Queue[dict[str, object]]]


@pytest.fixture
async def transport() -> AsyncIterator[Transport]:
    incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    connected: asyncio.Future[ServerConnection] = asyncio.get_running_loop().create_future()

    async def handle(connection: ServerConnection) -> None:
        connected.set_result(connection)
        async for wire in connection:
            message = json.loads(wire)
            method = message.get("method")
            if method == "initialize":
                assert message["params"]["capabilities"] == {"experimentalApi": False}
                result = {"userAgent": "codex_cli_rs/0.153.4 (Linux synthetic; x86_64)"}
            elif method == "thread/start":
                result = {"thread": {"id": _THREAD}}
            elif method == "initialized":
                continue
            else:
                incoming.put_nowait(message)
                continue
            await connection.send(json.dumps({"id": message["id"], "result": result}))

    with tempfile.TemporaryDirectory(prefix="codex-wire-", dir="/tmp") as directory:
        socket = Path(directory) / "peer.sock"
        async with await unix_serve(handle, str(socket)):
            async with CodexAppServerClient(CodexAppServerConfig(socket)) as client:
                connection = await connected
                await client.thread_start(cwd="/synthetic")
                yield client, connection, incoming
            assert socket.is_socket(), "disconnect must leave the external listener alive"


@pytest.mark.parametrize(
    ("case", "response", "kind"),
    [
        ("command_approval", {"decision": "decline"}, "permission"),
        ("file_approval", {"decision": "decline"}, "permission"),
        ("permission_approval", {"permissions": {}, "scope": "turn"}, "permission"),
        ("dynamic_tool_request", {"contentItems": [], "success": False}, "tool"),
        ("user_input_request", {"answers": {}}, "permission"),
        ("mcp_elicitation_request", {"action": "decline", "content": None}, "permission"),
    ],
)
async def test_native_authority_denials_remain_ordered_and_observable(
    transport: Transport, case: str, response: dict[str, object], kind: str
) -> None:
    client, connection, incoming = transport
    request = _CASES[case]
    await connection.send(json.dumps(_CASES["notification"]))
    await connection.send(json.dumps(request))
    first = await client.next_message()
    assert isinstance(first, CodexNotification) and first.method == "turn/started"
    authority = await client.next_message()
    assert isinstance(authority, CodexServerRequest) and authority.kind == kind
    assert await incoming.get() == {"id": request["id"], "result": response}
    await connection.send(json.dumps(request))
    with pytest.raises(ProtocolDefect, match="repeated"):
        await client.next_message()


@pytest.mark.parametrize(
    "case",
    ["unknown_request", "auth_refresh_request", "attestation_request", "current_time_request"],
)
async def test_unsupported_authority_has_no_successful_reply_or_terminal(
    transport: Transport, case: str
) -> None:
    client, connection, incoming = transport
    request = _CASES[case]
    params = cast(dict[str, object], request["params"])
    await connection.send(json.dumps({**request, "params": {**params, "threadId": _THREAD}}))
    response = await incoming.get()
    assert response["id"] == request["id"] and "error" in response and "result" not in response
    if case == "current_time_request":
        assert isinstance(await client.next_message(), CodexServerRequest)
    with pytest.raises(ProtocolDefect):
        await client.next_message()


@pytest.mark.parametrize(
    "wire",
    [
        '{"id":1,"id":1,"result":{}}',
        '{"id":false,"result":{}}',
        '{"id":999,"result":{}}',
        '{"id":1,"result":{},"error":{"code":1,"message":"both"}}',
        '{"method":"turn/started","params":{},"unexpected":true}',
        "[]",
        "not-json",
    ],
)
async def test_malformed_or_uncorrelated_frames_fail_closed(
    transport: Transport, wire: str
) -> None:
    client, connection, _incoming = transport
    await connection.send(wire)
    with pytest.raises(ProtocolDefect):
        await client.next_message()


@pytest.mark.parametrize("case", ["legacy_command_approval", "legacy_file_approval"])
async def test_server_authority_without_managed_thread_identity_fails_closed(
    transport: Transport, case: str
) -> None:
    client, connection, incoming = transport
    await connection.send(json.dumps(_CASES[case]))
    with pytest.raises(ProtocolDefect, match="managed thread identity"):
        await client.next_message()
    assert incoming.empty()


async def test_response_correlation_survives_reordering_and_cancelled_callers(
    transport: Transport,
) -> None:
    client, connection, incoming = transport
    abandoned = asyncio.create_task(client.request("thread/read", {"threadId": "abandoned"}))
    first = await incoming.get()
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    pending = asyncio.create_task(client.request("thread/read", {"threadId": "selected"}))
    second = await incoming.get()
    assert first["id"] != second["id"]
    await connection.send(json.dumps({"id": second["id"], "result": {"selected": True}}))
    await connection.send(json.dumps({"id": first["id"], "result": {"selected": False}}))
    assert await pending == {"selected": True}
    rejected = asyncio.create_task(client.request("thread/read", {"threadId": "missing"}))
    third = await incoming.get()
    await connection.send(
        json.dumps({"id": third["id"], "error": _CASES["response_error"]["error"]})
    )
    with pytest.raises(CodexAppServerResponseError):
        await rejected


async def test_external_disconnect_fails_pending_calls_and_event_stream(
    transport: Transport,
) -> None:
    client, connection, incoming = transport
    pending = asyncio.create_task(client.request("thread/read", {"threadId": _THREAD}))
    await incoming.get()
    await connection.close()
    with pytest.raises(CodexConnectionUnavailable):
        await pending
    with pytest.raises(CodexConnectionUnavailable):
        await client.next_message()


@pytest.mark.parametrize("overflow", ["bytes", "count"])
async def test_paused_consumer_overflow_preserves_authority_then_rejects_terminal(
    transport: Transport, overflow: str
) -> None:
    client, connection, incoming = transport
    pending = asyncio.create_task(client.request("thread/read", {"threadId": _THREAD}))
    request = await incoming.get()
    await connection.send(json.dumps(_CASES["command_approval"]))
    await incoming.get()
    await connection.send(
        json.dumps(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": _THREAD,
                    "turn": {"id": "turn-synthetic", "status": "completed"},
                },
            }
        )
    )
    notification = json.dumps(
        {
            "method": "item/agentMessage/delta",
            "params": {
                "threadId": _THREAD,
                "delta": "x" * (1024 * 1024) if overflow == "bytes" else "x",
            },
        }
    )

    async def flood() -> None:
        try:
            for _ in range(65 if overflow == "bytes" else 100_000):
                await connection.send(notification)
            await connection.send(json.dumps({"id": request["id"], "result": {}}))
        except ConnectionClosed:
            pass

    producer = asyncio.create_task(flood())
    try:
        with pytest.raises(ProtocolDefect, match="queue exceeded"):
            await pending
        assert isinstance(await client.next_message(), CodexServerRequest)
        with pytest.raises(ProtocolDefect, match="queue exceeded"):
            await client.next_message()
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
