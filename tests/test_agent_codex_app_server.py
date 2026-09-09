from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from provider_runtime.agent_runtime import ProtocolDefect
from provider_runtime.agent_runtime import codex_app_server as app_server_module
from provider_runtime.agent_runtime.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerConfig,
    CodexAppServerResponseError,
    CodexNotification,
    CodexServerRequest,
)

_CASES = cast(
    dict[str, object],
    json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "agent_runtime"
            / "codex"
            / "app_server_protocol_cases.json"
        ).read_text(encoding="utf-8")
    ),
)


class FakeManagedProcess:
    def __init__(
        self,
        responder: Callable[[dict[str, object]], Awaitable[dict[str, object] | None]],
    ) -> None:
        self.stdout = asyncio.StreamReader()
        self.sent: list[dict[str, object]] = []
        self.closed = False
        self.returncode: int | None = None
        self._responder = responder

    async def send(self, data: bytes) -> None:
        message = cast(dict[str, object], json.loads(data))
        self.sent.append(message)
        response = await self._responder(message)
        if response is not None:
            self.feed(response)

    def feed(self, message: Mapping[str, object]) -> None:
        self.stdout.feed_data(
            json.dumps(dict(message), separators=(",", ":")).encode("utf-8") + b"\n"
        )

    def feed_raw(self, data: bytes) -> None:
        self.stdout.feed_data(data)

    def die(self) -> None:
        self.returncode = 23
        self.stdout.feed_eof()

    async def close(self) -> None:
        self.closed = True
        self.returncode = -15
        self.stdout.feed_eof()


async def open_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responder: Callable[[dict[str, object]], Awaitable[dict[str, object] | None]] | None = None,
) -> tuple[CodexAppServerClient, FakeManagedProcess, dict[str, object]]:
    metadata = {
        "serverInfo": {"name": "codex", "version": "0.144.4 (synthetic)"},
        "userAgent": "codex/0.144.4",
    }

    async def default_responder(message: dict[str, object]) -> dict[str, object] | None:
        if message.get("method") == "initialize":
            return {"id": message["id"], "result": metadata}
        return None

    process = FakeManagedProcess(responder or default_responder)
    spawned: dict[str, object] = {}

    async def spawn(
        _cls: type[object],
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        limits: object,
        stdin: str = "pipe",
    ) -> FakeManagedProcess:
        spawned.update(
            argv=tuple(argv),
            cwd=cwd,
            environment=dict(environment),
            limits=limits,
            stdin=stdin,
        )
        return process

    monkeypatch.setattr(app_server_module.ManagedProcess, "spawn", classmethod(spawn))
    client = CodexAppServerClient(
        CodexAppServerConfig(
            executable=Path("/synthetic/codex"),
            cwd=Path("/private/synthetic"),
            environment={"CODEX_HOME": "/private/state", "PATH": "/usr/bin"},
            config_overrides=('forced_login_method="chatgpt"',),
        )
    )
    await client.__aenter__()
    return client, process, spawned


async def test_initialize_and_response_correlation_use_the_owned_stdio_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def responder(message: dict[str, object]) -> dict[str, object] | None:
        method = message.get("method")
        if method == "initialize":
            return {
                "id": message["id"],
                "result": {"serverInfo": {"name": "codex", "version": "0.144.4 (synthetic)"}},
            }
        if method == "thread/start":
            return {**cast(dict[str, object], _CASES["response_success"]), "id": message["id"]}
        if method == "thread/read":
            error = cast(dict[str, object], _CASES["response_error"])["error"]
            return {"id": message["id"], "error": error}
        return None

    client, process, spawned = await open_client(monkeypatch, responder=responder)
    try:
        result = await client.request("thread/start", {"cwd": "/private/synthetic"})
        assert result == {"thread": {"id": "thread-synthetic"}}
        with pytest.raises(CodexAppServerResponseError, match="thread/read request failed"):
            await client.request("thread/read", {"threadId": "missing"})
    finally:
        await client.close()

    assert spawned["argv"] == (
        "/synthetic/codex",
        "--config",
        'forced_login_method="chatgpt"',
        "app-server",
        "--listen",
        "stdio://",
        "--strict-config",
    )
    assert spawned["environment"] == {
        "CODEX_HOME": "/private/state",
        "PATH": "/usr/bin",
    }, "the direct child environment is replaced, never merged"
    initialize, initialized = process.sent[:2]
    assert initialize["method"] == "initialize"
    assert initialize["params"] == {
        "capabilities": {"experimentalApi": False},
        "clientInfo": {
            "name": "provider_runtime",
            "title": "provider-runtime",
            "version": "0.1.0",
        },
    }
    assert initialized == {"method": "initialized"}
    assert process.closed


@pytest.mark.parametrize(
    ("case_name", "expected_response", "request_kind"),
    (
        ("command_approval", {"decision": "decline"}, "permission"),
        ("file_approval", {"decision": "decline"}, "permission"),
        ("permission_approval", {"permissions": {}, "scope": "turn"}, "permission"),
        ("legacy_command_approval", {"decision": "denied"}, "permission"),
        ("legacy_file_approval", {"decision": "denied"}, "permission"),
        ("dynamic_tool_request", {"contentItems": [], "success": False}, "tool"),
        ("user_input_request", {"answers": {}}, "permission"),
        ("mcp_elicitation_request", {"action": "decline", "content": None}, "permission"),
    ),
)
async def test_known_server_requests_receive_only_explicit_denials_and_remain_observable(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    expected_response: dict[str, object],
    request_kind: str,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    request = cast(dict[str, object], _CASES[case_name])
    process.feed(request)

    observed = await client.next_message()
    assert isinstance(observed, CodexServerRequest)
    assert observed.method == request["method"]
    assert observed.request_id == request["id"]
    assert observed.kind == request_kind
    await asyncio.sleep(0)
    assert process.sent[-1] == {"id": request["id"], "result": expected_response}
    assert process.sent[-1]["result"] != {}, "the transport has no default empty response"
    await client.close()


async def test_notifications_remain_ordered_with_server_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    process.feed(cast(dict[str, object], _CASES["notification"]))
    process.feed(cast(dict[str, object], _CASES["command_approval"]))

    notification = await client.next_message()
    request = await client.next_message()
    assert isinstance(notification, CodexNotification)
    assert notification.method == "turn/started"
    assert isinstance(request, CodexServerRequest)
    assert request.method == "item/commandExecution/requestApproval"
    await client.close()


async def test_unknown_server_requests_receive_an_error_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    request = cast(dict[str, object], _CASES["unknown_request"])
    process.feed(request)

    with pytest.raises(ProtocolDefect, match="unknown server request"):
        await client.next_message()
    await asyncio.sleep(0)
    assert process.sent[-1] == {
        "id": request["id"],
        "error": {"code": -32601, "message": "unsupported server request"},
    }
    await client.close()


@pytest.mark.parametrize(
    ("case_name", "message"),
    (
        ("auth_refresh_request", "credential callbacks are unsupported"),
        ("attestation_request", "attestation callbacks are unsupported"),
    ),
)
async def test_documented_credential_callbacks_receive_errors_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    message: str,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    request = cast(dict[str, object], _CASES[case_name])
    process.feed(request)

    with pytest.raises(ProtocolDefect, match="forbidden callback"):
        await client.next_message()
    await asyncio.sleep(0)
    assert process.sent[-1] == {
        "id": request["id"],
        "error": {"code": -32601, "message": message},
    }
    await client.close()


async def test_experimental_host_time_request_is_observable_then_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    request = cast(dict[str, object], _CASES["current_time_request"])
    process.feed(request)

    observed = await client.next_message()
    assert isinstance(observed, CodexServerRequest)
    assert observed.kind == "tool"
    assert observed.method == "currentTime/read"
    with pytest.raises(ProtocolDefect, match="forbidden authority"):
        await client.next_message()
    assert process.sent[-1] == {
        "id": request["id"],
        "error": {"code": -32601, "message": "host time callbacks are unsupported"},
    }
    await client.close()


@pytest.mark.parametrize(
    "wire",
    (
        b'{"id":1,"id":1,"result":{}}\n',
        b'{"id":false,"result":{}}\n',
        b'{"id":999,"result":{}}\n',
        b'{"id":1,"result":{},"error":{"code":1,"message":"both"}}\n',
        b'{"method":"turn/started","params":{},"unexpected":true}\n',
        b"[]\n",
        b"not-json\n",
    ),
)
async def test_malformed_or_uncorrelated_messages_are_protocol_defects(
    monkeypatch: pytest.MonkeyPatch, wire: bytes
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    process.feed_raw(wire)
    with pytest.raises(ProtocolDefect):
        await client.next_message()
    await client.close()


async def test_process_death_fails_pending_requests_and_the_event_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    pending = asyncio.create_task(client.request("thread/list", {}))
    await asyncio.sleep(0)
    process.die()

    with pytest.raises(ProtocolDefect, match="exited before clean shutdown"):
        await pending
    with pytest.raises(ProtocolDefect, match="exited before clean shutdown"):
        await client.next_message()
    await client.close()


async def test_process_death_with_a_partial_json_line_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    process.feed_raw(b'{"method":"turn/started"')
    process.die()

    with pytest.raises(ProtocolDefect, match="partial JSON message"):
        await client.next_message()
    await client.close()


async def test_reordered_responses_remain_correlated_to_their_exact_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    first = asyncio.create_task(client.request("thread/read", {"threadId": "one"}))
    second = asyncio.create_task(client.request("thread/read", {"threadId": "two"}))
    await asyncio.sleep(0)
    requests = [message for message in process.sent if message.get("method") == "thread/read"]
    assert len(requests) == 2
    process.feed({"id": requests[1]["id"], "result": {"which": "two"}})
    process.feed({"id": requests[0]["id"], "result": {"which": "one"}})

    assert await first == {"which": "one"}
    assert await second == {"which": "two"}
    await client.close()


async def test_cancelled_request_identity_is_not_reused_or_misdelivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    cancelled = asyncio.create_task(client.request("thread/read", {"threadId": "cancelled"}))
    await asyncio.sleep(0)
    request_id = process.sent[-1]["id"]
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    process.feed({"id": request_id, "result": {"late": True}})
    await asyncio.sleep(0)

    live = asyncio.create_task(client.request("thread/read", {"threadId": "live"}))
    await asyncio.sleep(0)
    assert process.sent[-1]["id"] != request_id
    process.feed({"id": process.sent[-1]["id"], "result": {"live": True}})
    assert await live == {"live": True}
    await client.close()


async def test_duplicate_server_request_identity_is_a_protocol_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    request = cast(dict[str, object], _CASES["command_approval"])
    process.feed(request)
    assert isinstance(await client.next_message(), CodexServerRequest)
    process.feed(request)
    with pytest.raises(ProtocolDefect, match="repeated a server request identity"):
        await client.next_message()
    await client.close()


async def test_clean_shutdown_settles_reader_and_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    await client.close()
    await client.close()
    assert process.closed


async def test_turn_interrupt_uses_the_correlated_public_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def responder(message: dict[str, object]) -> dict[str, object] | None:
        method = message.get("method")
        if method == "initialize":
            return {
                "id": message["id"],
                "result": {"userAgent": "provider_runtime/0.144.4 (synthetic)"},
            }
        if method == "thread/start":
            return {"id": message["id"], "result": {"thread": {"id": "thread-one"}}}
        if method == "turn/start":
            return {"id": message["id"], "result": {"turn": {"id": "turn-one"}}}
        if method == "turn/interrupt":
            return {"id": message["id"], "result": {}}
        return None

    client, process, _spawned = await open_client(monkeypatch, responder=responder)
    thread = await client.thread_start(
        approval_mode="deny_all",
        config={},
        cwd="/private/synthetic",
        sandbox="read-only",
    )
    turn = await thread.turn([{"type": "text", "text": "synthetic"}])
    assert await turn.interrupt() == {}
    assert process.sent[-1] == {
        "id": process.sent[-1]["id"],
        "method": "turn/interrupt",
        "params": {"threadId": "thread-one", "turnId": "turn-one"},
    }
    await client.close()


async def test_resume_preserves_pre_response_notifications_for_owned_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_ref: list[FakeManagedProcess] = []

    async def responder(message: dict[str, object]) -> dict[str, object] | None:
        method = message.get("method")
        if method == "initialize":
            return {
                "id": message["id"],
                "result": {"userAgent": "provider_runtime/0.144.4 (synthetic)"},
            }
        if method == "thread/resume":
            process_ref[0].feed(
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "thread-resumed",
                        "turnId": "turn-restored",
                        "tokenUsage": {
                            "last": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                            "total": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                        },
                    },
                }
            )
            return {
                "id": message["id"],
                "result": {"thread": {"id": "thread-resumed"}},
            }
        return None

    client, process, _spawned = await open_client(monkeypatch, responder=responder)
    process_ref.append(process)
    thread = await client.thread_resume("thread-resumed")
    pending = client.take_pending_messages()
    assert thread.id == "thread-resumed"
    assert len(pending) == 1
    assert isinstance(pending[0], CodexNotification)
    assert pending[0].method == "thread/tokenUsage/updated"
    await client.close()


@pytest.mark.parametrize(
    "error",
    (
        {"code": -1, "message": "bad", "unexpected": True},
        {"code": "bad", "message": "bad"},
    ),
)
async def test_malformed_error_response_immediately_fails_its_pending_request(
    monkeypatch: pytest.MonkeyPatch, error: dict[str, object]
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    pending = asyncio.create_task(client.request("thread/read", {"threadId": "one"}))
    await asyncio.sleep(0)
    process.feed({"id": process.sent[-1]["id"], "error": error})
    try:
        async with asyncio.timeout(1):
            with pytest.raises(ProtocolDefect, match="error response"):
                await pending
        assert process.closed, "a malformed response must terminate its provider child"
    finally:
        await client.close()


async def test_request_timeout_fails_the_transport_and_stops_its_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    monkeypatch.setattr(app_server_module, "_OPERATION_TIMEOUT_SECONDS", 0.01)
    try:
        with pytest.raises(ProtocolDefect, match="request timed out"):
            await client.request("thread/read", {"threadId": "one"})
        assert process.closed, "a timed-out RPC cannot leave an active provider transport"
        sent = len(process.sent)
        with pytest.raises(ProtocolDefect, match="request timed out"):
            await client.request("thread/read", {"threadId": "two"})
        assert len(process.sent) == sent, "no request may follow an uncertain timed-out RPC"
    finally:
        await client.close()


@pytest.mark.parametrize("overflow", ("bytes", "count"))
async def test_paused_consumer_overflow_preserves_authority_then_rejects_terminal(
    monkeypatch: pytest.MonkeyPatch, overflow: str
) -> None:
    client, process, _spawned = await open_client(monkeypatch)
    pending = asyncio.create_task(client.request("thread/read", {"threadId": "one"}))
    await asyncio.sleep(0)
    request_id = process.sent[-1]["id"]
    process.feed(cast(dict[str, object], _CASES["command_approval"]))
    process.feed({"method": "turn/completed", "params": {"turn": {"id": "turn-one"}}})
    count = 65 if overflow == "bytes" else 100_000
    notification = {
        "method": "item/agentMessage/delta",
        "params": {"delta": "x" * (1024 * 1024) if overflow == "bytes" else "x"},
    }
    for _ in range(count):
        process.feed(notification)
    process.feed({"id": request_id, "result": {}})
    try:
        with pytest.raises(ProtocolDefect, match="pending message queue exceeded"):
            await pending
        assert process.closed, "overflow must reap the child even while the consumer is paused"
        assert isinstance(await client.next_message(), CodexServerRequest)
        with pytest.raises(ProtocolDefect, match="pending message queue exceeded"):
            await client.next_message()
    finally:
        await client.close()
