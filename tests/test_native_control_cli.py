"""Native control through its process boundary; providers are synthetic peers."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from websockets.asyncio.server import unix_serve

from tests.test_agent_codex_control import THREAD, TURN, ProtocolPeer


class ControlPeer(ProtocolPeer):
    def __init__(self, socket: Path) -> None:
        super().__init__(socket)
        self.status = "active"

    def thread(self) -> dict[str, object]:
        result = super().thread()
        state = "idle" if self.turn_status == "interrupted" else self.status
        result["status"] = {"type": state, **({"activeFlags": []} if state == "active" else {})}
        result["canAcceptDirectInput"] = state in ("active", "idle")
        result["historyMode"] = "paginated"
        return result


@pytest.fixture
async def peer() -> AsyncIterator[ProtocolPeer]:
    with tempfile.TemporaryDirectory(prefix="native-peer-", dir="/tmp") as directory:
        value = ControlPeer(Path(directory) / "peer.sock")
        async with await unix_serve(value.handle, str(value.socket)):
            yield value


async def invoke(
    request: object, terminate_when: Path | None = None, /, **environment: str
) -> tuple[int, dict, bytes]:
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "provider_runtime.agent_runtime.native_control_cli",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **environment},
    )
    try:
        exchange = asyncio.create_task(child.communicate(json.dumps(request).encode()))
        if terminate_when:
            async with asyncio.timeout(3):
                while not terminate_when.exists():
                    await asyncio.sleep(0.01)
            child.terminate()
        stdout, stderr = await exchange
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    assert stdout, stderr.decode()
    assert len(stdout) <= 65536
    assert child.returncode is not None
    return child.returncode, json.loads(stdout), stderr


def codex_request(peer: ProtocolPeer, operation: str, **fields: object) -> dict:
    return {
        "operation": operation,
        "provider": "Codex",
        "profileKey": "personal",
        "endpoint": "unix://" + str(peer.socket),
        "targets": [
            {
                "sessionId": THREAD,
                **({"turnId": TURN} if operation in ("interrupt", "stop") else {}),
            }
        ],
        **({"input": {"scope": "latest"}} if operation == "read" else {}),
        **fields,
    }


async def test_codex_observation_and_multiline_submission_use_same_native_session(
    peer: ProtocolPeer,
) -> None:
    peer.turn_status = "completed"
    peer.status = "idle"
    code, value, stderr = await invoke(codex_request(peer, "inspect"))
    assert (code, stderr) == (0, b"")
    row = value["result"][0]["result"]
    assert row["status"] == {"state": "idle", "source": "native"}
    assert row["sessionId"] == THREAD and row["turn"] == {"id": TURN, "state": "completed"}
    assert row["methods"] == {
        **dict.fromkeys(("read", "sendPeer", "sendUser", "stop"), "native"),
        "queueUser": "unavailable",
    }
    text = "first line\nsecond line $(do not execute)"
    _, result, _ = await invoke(
        codex_request(peer, "send", input={"text": text, "input": "user", "delivery": "direct"})
    )
    assert result == {
        "ok": True,
        "result": {
            "method": "native",
            "input": "user",
            "delivery": "direct",
            "outcome": "accepted",
            "turnId": TURN,
        },
    }
    sent = [row for row in peer.messages if row.get("method") == "turn/start"]
    assert len(sent) == 1
    assert sent[0]["params"] == {
        "threadId": THREAD,
        "input": [{"type": "text", "text": text}],
    }


async def test_lost_native_write_is_unknown_and_never_retried(peer: ProtocolPeer) -> None:
    peer.drop_method = "turn/start"
    code, result, stderr = await invoke(
        codex_request(peer, "send", input={"text": "work", "input": "user", "delivery": "direct"})
    )
    assert (code, stderr) == (1, b"")
    assert result == {"ok": False, "error": {"code": "unknown", "dispatch": "unknown"}}
    assert sum(row.get("method") == "turn/start" for row in peer.messages) == 1


async def test_unloaded_codex_keeps_history_without_claiming_live_control(
    peer: ProtocolPeer,
) -> None:
    peer.status = "notLoaded"
    _, result, _ = await invoke(codex_request(peer, "inspect"))
    row = result["result"][0]["result"]
    assert row["status"] == {"state": "unknown", "source": "unavailable"}
    assert row["methods"] == {
        "read": "native",
        "sendPeer": "unavailable",
        "sendUser": "unavailable",
        "queueUser": "unavailable",
        "stop": "unavailable",
    }
    _, history, _ = await invoke(codex_request(peer, "read"))
    assert history["result"]["text"] == peer.answer


@pytest.mark.parametrize("operation", ["interrupt", "stop"])
async def test_unloaded_codex_does_not_confirm_halt_from_historical_turn(
    peer: ProtocolPeer, operation: str
) -> None:
    peer.status, peer.turn_status = "notLoaded", "completed"
    _, result, _ = await invoke(codex_request(peer, operation))
    assert result == {"ok": False, "error": {"code": "unavailable", "dispatch": "not_sent"}}
    assert not any(row.get("method") == "turn/interrupt" for row in peer.messages)


@pytest.mark.parametrize("operation", ["interrupt", "stop"])
async def test_observed_no_turn_does_not_authorize_cancelling_a_new_turn(
    peer: ProtocolPeer, operation: str
) -> None:
    target = codex_request(peer, operation)["targets"][0]
    del target["turnId"]
    _, result, _ = await invoke(codex_request(peer, operation, targets=[target]))
    assert result == {"ok": False, "error": {"code": "stale", "dispatch": "not_sent"}}
    assert not any(row.get("method") == "turn/interrupt" for row in peer.messages)


async def test_read_retains_native_history_outside_viewport_and_bounds_encoded_output(
    peer: ProtocolPeer,
) -> None:
    peer.answer = "older native result\n" + "\x01é" * 20000
    _, result, _ = await invoke(
        codex_request(peer, "read", input={"scope": "latest", "maxBytes": 32768})
    )
    assert result["result"]["truncated"] is True
    assert len(result["result"]["text"].encode()) <= 32768
    peer.answer = "older native result\n" + "\x01é" * 2000
    _, result, _ = await invoke(
        codex_request(peer, "read", input={"scope": "latest", "maxBytes": 1024})
    )
    output = result["result"]
    assert output["source"] == "native" and output["scope"] == "latest"
    assert output["truncated"] is True
    assert len(output["text"].encode()) <= 1024
    assert output["text"]


async def test_stop_interrupts_exact_turn_without_stopping_native_server(
    peer: ProtocolPeer,
) -> None:
    code, result, _ = await invoke(codex_request(peer, "stop"))
    assert code == 0 and result == {
        "ok": True,
        "result": {"method": "native", "outcome": "interrupted", "turnId": TURN},
    }
    interrupts = [row for row in peer.messages if row.get("method") == "turn/interrupt"]
    assert len(interrupts) == 1
    assert interrupts[0]["params"] == {"threadId": THREAD, "turnId": TURN}
    assert not any("stop" in str(row.get("method")) for row in peer.messages)
    _, result, _ = await invoke(codex_request(peer, "inspect"))
    assert result["result"][0]["result"]["status"]["state"] == "idle"
    assert result["result"][0]["result"]["turn"]["state"] == "interrupted"


async def test_successor_after_interrupt_preserves_possible_effect(peer: ProtocolPeer) -> None:
    peer.turn_id_after_interrupt = "01992818-9229-714c-9c91-e39d3f006e64"
    _, result, _ = await invoke(codex_request(peer, "interrupt"))
    assert result == {
        "ok": True,
        "result": {"method": "native", "outcome": "unknown", "turnId": TURN},
    }
    assert sum(row.get("method") == "turn/interrupt" for row in peer.messages) == 1


def claude_fixture(tmp_path: Path) -> tuple[Path, Path]:
    binary = tmp_path / "bin"
    binary.mkdir()
    command = binary / "claude"
    command.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys, time\n"
        "root = pathlib.Path(os.environ['CLAUDE_CONFIG_DIR'])\n"
        "if (root / 'fixture-hang').exists():\n"
        "    (root / 'fixture-child').write_text(str(os.getpid()))\n"
        "    time.sleep(30)\n"
        "rows = json.loads((root / 'fixture-state.json').read_text())\n"
        "if sys.argv[1:] == ['agents', '--json', '--all']:\n"
        "    print(json.dumps(rows))\n"
        "elif sys.argv[1:] == ['stop', 'job-one']:\n"
        "    rows[0].update(state='stopped')\n"
        "    rows[0].pop('pid', None)\n"
        "    rows[0].pop('status', None)\n"
        "    (root / 'fixture-state.json').write_text(json.dumps(rows))\n"
        "else:\n"
        "    sys.exit(2)\n"
    )
    command.chmod(0o755)
    root = tmp_path / "profile"
    root.mkdir()
    (root / "fixture-state.json").write_text(
        json.dumps(
            [
                {
                    "cwd": "/synthetic",
                    "kind": "background",
                    "startedAt": 1000,
                    "pid": os.getpid(),
                    "sessionId": THREAD,
                    "id": "job-one",
                    "state": "blocked",
                    "status": "waiting",
                    "waitingFor": "permission prompt",
                }
            ]
        )
    )
    return binary, root


async def test_claude_observes_conversation_blocker_and_stops_only_matched_job(
    tmp_path: Path,
) -> None:
    binary, root = claude_fixture(tmp_path)
    request = {
        "provider": "Claude",
        "profileKey": "claude-personal",
        "operation": "inspect",
        "targets": [{"sessionId": THREAD}, {"sessionId": "01992818-9229-714c-9c91-e39d3f006e64"}],
    }
    env = {"PATH": str(binary), "CLAUDE_CONFIG_DIR": str(root)}
    code, result, stderr = await invoke(request, **env)
    assert (code, stderr) == (0, b"")
    found, missing = result["result"]
    assert found["result"]["status"] == {
        "state": "blocked",
        "source": "native",
    }
    assert found["result"]["sessionId"] == THREAD
    assert found["result"]["methods"]["sendPeer"] == "unavailable"
    assert found["result"]["methods"]["stop"] == "native"
    assert missing["result"]["status"] == {"state": "unknown", "source": "unavailable"}
    request.update(operation="stop", targets=[{"sessionId": THREAD}])
    _, result, _ = await invoke(request, **env)
    assert result == {"ok": True, "result": {"method": "native", "outcome": "stopped"}}


async def test_claude_native_history_is_scoped_to_selected_profile(tmp_path: Path) -> None:
    pytest.importorskip("claude_agent_sdk")
    binary, fixture_root = claude_fixture(tmp_path)
    roots = [tmp_path / "personal", tmp_path / "work"]
    for root, text in zip(
        roots, ["personal history beyond viewport", "work history beyond viewport"], strict=True
    ):
        project = root / "projects" / "-synthetic"
        project.mkdir(parents=True)
        (root / "fixture-state.json").write_text((fixture_root / "fixture-state.json").read_text())
        (project / f"{THREAD}.jsonl").write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": THREAD,
                    "parentUuid": None,
                    "sessionId": THREAD,
                    "message": {"role": "user", "content": "fixture"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "uuid": TURN,
                    "parentUuid": THREAD,
                    "sessionId": THREAD,
                    "message": {
                        "id": "native-message",
                        "role": "assistant",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
            + "\n"
        )
    request = {
        "provider": "Claude",
        "profileKey": "claude-personal",
        "operation": "read",
        "targets": [{"sessionId": THREAD}],
        "input": {"scope": "latest", "maxBytes": 16384},
    }
    results = [
        await invoke(request, PATH=str(binary), CLAUDE_CONFIG_DIR=str(root)) for root in roots
    ]
    assert "personal history" in results[0][1]["result"]["text"]
    assert "work history" not in results[0][1]["result"]["text"]
    assert "work history" in results[1][1]["result"]["text"]
    assert results[0][1]["result"]["scope"] == "latest"


@pytest.mark.parametrize("kind", ["interactive", "background"])
async def test_claude_conversation_stop_requires_native_background_job(
    tmp_path: Path, kind: str
) -> None:
    binary, root = claude_fixture(tmp_path)
    state_file = root / "fixture-state.json"
    rows = json.loads(state_file.read_text())
    rows[0]["kind"] = kind
    if kind == "interactive":
        del rows[0]["id"]
    state_file.write_text(json.dumps(rows))
    request = {
        "provider": "Claude",
        "profileKey": "personal",
        "operation": "inspect",
        "targets": [{"sessionId": THREAD}],
    }
    _, result, _ = await invoke(request, PATH=str(binary), CLAUDE_CONFIG_DIR=str(root))
    observation = result["result"][0]["result"]
    assert "terminalOwnsAgent" not in observation
    assert observation["methods"]["stop"] == ("native" if kind == "background" else "unavailable")


async def test_hung_claude_observation_times_out_and_reaps_only_its_cli(tmp_path: Path) -> None:
    binary, root = claude_fixture(tmp_path)
    (root / "fixture-hang").touch()
    request = {
        "provider": "Claude",
        "profileKey": "claude-personal",
        "operation": "inspect",
        "targets": [{"sessionId": THREAD}],
    }
    try:
        code, result, stderr = await asyncio.wait_for(
            invoke(request, PATH=str(binary), CLAUDE_CONFIG_DIR=str(root)), timeout=3
        )
        assert (code, stderr) == (0, b"")
        assert result["result"][0]["result"]["status"] == {
            "state": "unknown",
            "source": "unavailable",
        }
        with pytest.raises(ProcessLookupError):
            os.kill(int((root / "fixture-child").read_text()), 0)
    finally:
        if (root / "fixture-child").exists():
            try:
                os.kill(int((root / "fixture-child").read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


async def test_helper_termination_reaps_its_claude_child(tmp_path: Path) -> None:
    binary, root = claude_fixture(tmp_path)
    (root / "fixture-hang").touch()
    request = {
        "provider": "Claude",
        "profileKey": "claude-personal",
        "operation": "inspect",
        "targets": [{"sessionId": THREAD}],
    }
    try:
        code, result, stderr = await invoke(
            request,
            root / "fixture-child",
            PATH=str(binary),
            CLAUDE_CONFIG_DIR=str(root),
        )
        assert (code, stderr) == (1, b"")
        assert result == {"ok": False, "error": {"code": "unavailable", "dispatch": "not_sent"}}
        with pytest.raises(ProcessLookupError):
            os.kill(int((root / "fixture-child").read_text()), 0)
    finally:
        if (root / "fixture-child").exists():
            try:
                os.kill(int((root / "fixture-child").read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


async def test_invalid_requests_and_unsupported_claude_send_do_not_launch_a_provider() -> None:
    invalid = {
        "operation": "inspect",
        "provider": "Claude",
        "profileKey": "work",
        "targets": [],
        "endpoint": "unix:///wrong",
    }
    code, result, stderr = await invoke(invalid, PATH="/nonexistent")
    assert (code, stderr) == (1, b"")
    assert result == {"ok": False, "error": {"code": "rejected", "dispatch": "not_sent"}}
    invalid.update(
        operation="send",
        targets=[{"sessionId": THREAD}],
        input={"text": "hello", "input": "user", "delivery": "direct"},
    )
    del invalid["endpoint"]
    _, result, _ = await invoke(invalid, PATH="/nonexistent")
    assert result == {"ok": False, "error": {"code": "unavailable", "dispatch": "not_sent"}}

    for provider in ("Codex", "Claude"):
        for field, value in (("pid", 1), ("startIdentity", "1"), ("view", {})):
            request = {
                "operation": "inspect",
                "provider": provider,
                "profileKey": "personal",
                "targets": [{"sessionId": THREAD, field: value}],
                **({"endpoint": "unix:///nonexistent"} if provider == "Codex" else {}),
            }
            _, result, _ = await invoke(request, PATH="/nonexistent")
            assert result == {"ok": False, "error": {"code": "rejected", "dispatch": "not_sent"}}

    for input in (
        {"cwd": "/workspace"},
        *({"cwd": "/workspace", "bypassPermissions": value} for value in (None, 0, 1, "true")),
    ):
        request = {
            "operation": "create",
            "provider": "Codex",
            "profileKey": "personal",
            "endpoint": "unix:///nonexistent",
            "input": input,
        }
        _, result, _ = await invoke(request, PATH="/nonexistent")
        assert result == {"ok": False, "error": {"code": "rejected", "dispatch": "not_sent"}}
