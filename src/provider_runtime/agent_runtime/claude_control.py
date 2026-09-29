"""Observe existing Claude sessions through the CLI and public saved-message reader."""

from __future__ import annotations

import asyncio
import ctypes
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

if TYPE_CHECKING:
    from claude_agent_sdk import SessionMessage


class ClaudeControlError(Exception):
    def __init__(
        self,
        code: Literal["unsupported", "unavailable", "stale", "unknown", "history_changed"],
        dispatch: Literal["not_sent", "unknown"] = "not_sent",
    ) -> None:
        super().__init__(code)
        self.code = code
        self.dispatch = dispatch


@dataclass(frozen=True)
class ClaudeSession:
    kind: str
    session_id: str | None
    pid: int | None
    job_id: str | None
    state: str


@dataclass(slots=True)
class _AssistantReply:
    text: list[str]
    stop_reason: str | None
    anchor: str


def history_available() -> bool:
    return importlib.util.find_spec("claude_agent_sdk") is not None


async def _command(executable: str, *arguments: str, mutating: bool = False) -> bytes:
    try:
        child = await asyncio.create_subprocess_exec(
            executable,
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={**os.environ, "LC_ALL": "C"},
        )
    except OSError:
        raise ClaudeControlError("unavailable") from None
    try:
        assert child.stdout is not None
        async with asyncio.timeout(5 if mutating else 0.75):
            try:
                output = await child.stdout.readexactly(65537)
            except asyncio.IncompleteReadError as error:
                output = error.partial
            if len(output) > 65536 or await child.wait() != 0:
                raise ClaudeControlError(
                    "unknown" if mutating else "unavailable", "unknown" if mutating else "not_sent"
                )
            return output
    except TimeoutError:
        raise ClaudeControlError(
            "unknown" if mutating else "unavailable", "unknown" if mutating else "not_sent"
        ) from None
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


async def list_sessions() -> tuple[ClaudeSession, ...]:
    output = await _command("claude", "agents", "--json", "--all")
    try:
        rows = json.loads(output)
        if not isinstance(rows, list):
            raise ValueError
        result = []
        for row in rows:
            if not isinstance(row, dict) or row.get("kind") not in ("interactive", "background"):
                raise ValueError
            pid, session_id, job_id = row.get("pid"), row.get("sessionId"), row.get("id")
            if pid is not None and (type(pid) is not int or pid <= 0):
                raise ValueError
            if session_id is not None and (
                not isinstance(session_id, str) or str(UUID(session_id)) != session_id
            ):
                raise ValueError
            if job_id is not None and (
                not isinstance(job_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", job_id)
            ):
                raise ValueError
            if row["kind"] == "background" and not job_id:
                raise ValueError
            native_state, live_status = row.get("state"), row.get("status")
            if native_state is not None and native_state not in (
                "working",
                "blocked",
                "done",
                "failed",
                "stopped",
            ):
                raise ValueError
            if live_status is not None and live_status not in ("busy", "waiting", "idle"):
                raise ValueError
            state = native_state or {"busy": "working", "waiting": "blocked", "idle": "idle"}.get(
                live_status or "", "unknown"
            )
            result.append(ClaudeSession(row["kind"], session_id, pid, job_id, state))
        return tuple(result)
    except (ValueError, TypeError, UnicodeError):
        raise ClaudeControlError("unavailable") from None


def match_session(
    sessions: tuple[ClaudeSession, ...], session_id: str | None, pid: int | None
) -> ClaudeSession:
    matches = [
        session
        for session in sessions
        if (
            (session_id is None or session.session_id == session_id)
            and (pid is None or session.pid == pid)
        )
    ]
    if len(matches) != 1:
        raise ClaudeControlError("stale")
    return matches[0]


def verify_process(pid: int, start_identity: str) -> None:
    """Compare the exact kernel lifetime encoded by the host, without signals."""
    try:
        if sys.platform == "linux":
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            actual = fields[19]
        elif sys.platform == "darwin":
            # proc_bsdinfo contains 120 bytes before its two start-time uint64s.
            info = ctypes.create_string_buffer(136)
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            count = libproc.proc_pidinfo(pid, 3, 0, info, len(info))
            if count != len(info):
                raise ClaudeControlError("stale")
            seconds = int.from_bytes(info.raw[120:128], sys.byteorder)
            micros = int.from_bytes(info.raw[128:136], sys.byteorder)
            actual = str(seconds * 1_000_000 + micros)
        else:
            raise ClaudeControlError("unavailable")
    except (OSError, ValueError, IndexError):
        raise ClaudeControlError("stale") from None
    if actual != start_identity:
        raise ClaudeControlError("stale")


def _messages(session_id: str) -> list[SessionMessage]:
    try:
        from claude_agent_sdk import get_session_info, get_session_messages
    except ImportError:
        raise ClaudeControlError("unsupported") from None
    try:
        info = get_session_info(session_id)
        if info is None:
            raise ClaudeControlError("unavailable")
        messages = get_session_messages(session_id)
        if not messages:
            raise ClaudeControlError("unavailable")
    except (OSError, ValueError):
        raise ClaudeControlError("unavailable") from None
    return messages


def _assistant_groups(messages: list[SessionMessage]) -> dict[str, _AssistantReply]:
    groups: dict[str, _AssistantReply] = {}
    for message in messages:
        if message.type != "assistant":
            continue
        value = message.message
        if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"]:
            raise ClaudeControlError("unavailable")
        content = value.get("content")
        if not isinstance(content, list) or not isinstance(message.uuid, str) or not message.uuid:
            raise ClaudeControlError("unavailable")
        handle = value["id"]
        reason = value.get("stop_reason")
        if reason is not None and not isinstance(reason, str):
            raise ClaudeControlError("unavailable")
        group = groups.setdefault(handle, _AssistantReply([], reason, message.uuid))
        if group.stop_reason != reason:
            raise ClaudeControlError("unavailable")
        group.anchor = message.uuid
        for block in content:
            if not isinstance(block, dict):
                raise ClaudeControlError("unavailable")
            if block.get("type") == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ClaudeControlError("unavailable")
                if text:
                    group.text.append(text)
    return groups


def read_messages(session_id: str, scope: Literal["latest", "history"], max_bytes: int) -> dict:
    messages = _messages(session_id)
    groups = _assistant_groups(messages)
    latest = next(
        ((handle, group) for handle, group in reversed(list(groups.items())) if group.text), None
    )
    text = []
    if scope == "history":
        for message in messages:
            value = message.message
            if not isinstance(value, dict):
                raise ClaudeControlError("unavailable")
            content = value.get("content")
            if isinstance(content, str):
                text.append(f"{message.type}: {content}")
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ):
                        text.append(f"{message.type}: {block['text']}")
    elif latest:
        text = latest[1].text
    value = "\n\n".join(text)
    encoded = value.encode()
    result: dict = {
        "text": encoded[-max_bytes:].decode(errors="ignore"),
        "source": "native",
        "scope": scope,
        "truncated": len(encoded) > max_bytes,
        "outputState": "none",
    }
    if latest:
        handle, group = latest
        result["outputState"] = "finalized" if group.stop_reason == "end_turn" else "unknown"
        if group.stop_reason == "end_turn":
            result["outputId"] = handle
    return result


def results(session_id: str, cursor: str | None) -> dict:
    messages = _messages(session_id)
    groups = _assistant_groups(messages)
    start = 0
    ordered = list(groups.items())
    if cursor is not None:
        try:
            anchor = json.loads(cursor)
            if (
                not isinstance(anchor, dict)
                or set(anchor) != {"sessionId", "messageId", "uuid"}
                or anchor["sessionId"] != session_id
            ):
                raise ValueError
            matches = [
                index
                for index, (handle, group) in enumerate(ordered)
                if handle == anchor["messageId"] and group.anchor == anchor["uuid"]
            ]
            if len(matches) != 1:
                raise ValueError
            start = matches[0] + 1
        except (ValueError, TypeError):
            raise ClaudeControlError("history_changed") from None
    ids: list[str] = []
    for index in range(start, len(ordered)):
        handle, group = ordered[index]
        if group.stop_reason == "end_turn" and group.text:
            ids.append(handle)
        if len(ids) == 128 and index + 1 < len(ordered):
            return {
                "resultIds": ids,
                "nextCursor": json.dumps(
                    {"sessionId": session_id, "messageId": handle, "uuid": group.anchor},
                    separators=(",", ":"),
                ),
            }
    return {"resultIds": ids}


async def stop(session_id: str, pid: int, start_identity: str) -> dict[str, str]:
    verify_process(pid, start_identity)
    session = match_session(await list_sessions(), session_id, pid)
    if session.job_id is None:
        raise ClaudeControlError("unsupported")
    if session.state in ("done", "failed", "stopped"):
        return {"method": "native", "outcome": "finished"}
    verify_process(pid, start_identity)
    await _command("claude", "stop", session.job_id, mutating=True)
    try:
        observed = [
            row
            for row in await list_sessions()
            if row.job_id == session.job_id and row.session_id == session_id
        ]
    except ClaudeControlError:
        return {"method": "native", "outcome": "unknown"}
    if len(observed) == 1 and observed[0].state == "stopped" and observed[0].pid is None:
        return {"method": "native", "outcome": "stopped"}
    if len(observed) == 1 and observed[0].state in ("done", "failed") and observed[0].pid is None:
        return {"method": "native", "outcome": "finished"}
    return {"method": "native", "outcome": "unknown"}
