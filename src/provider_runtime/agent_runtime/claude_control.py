"""Observe existing Claude sessions through the CLI and public saved-message reader."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
from dataclasses import dataclass
from typing import Literal


class ClaudeControlError(Exception):
    def __init__(
        self,
        code: Literal["unsupported", "unavailable", "stale", "unknown"],
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
    reason: str | None


def history_available() -> bool:
    return importlib.util.find_spec("claude_agent_sdk") is not None


async def _command(*arguments: str, mutating: bool = False) -> bytes:
    try:
        child = await asyncio.create_subprocess_exec(
            "claude",
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
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
    output = await _command("agents", "--json", "--all")
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
            if session_id is not None and not isinstance(session_id, str):
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
            reason = None
            if state == "blocked":
                reason = {
                    "permission prompt": "permission",
                    "sandbox request": "permission",
                    "input needed": "input",
                    "worker request": "input",
                    "dialog open": "dialog",
                }.get(row.get("waitingFor") or "", "dialog")
            result.append(ClaudeSession(row["kind"], session_id, pid, job_id, state, reason))
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


def read_messages(session_id: str) -> str:
    try:
        from claude_agent_sdk import get_session_messages
    except ImportError:
        raise ClaudeControlError("unsupported") from None
    try:
        messages = get_session_messages(session_id)
    except (OSError, ValueError):
        raise ClaudeControlError("unavailable") from None
    text = []
    for message in messages:
        content = message.message.get("content") if isinstance(message.message, dict) else None
        if isinstance(content, str):
            text.append(f"{message.type}: {content}")
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                value = block.get("text") if block.get("type") == "text" else block.get("content")
                if isinstance(value, str):
                    text.append(f"{message.type}: {value}")
    if not text:
        raise ClaudeControlError("unavailable")
    return "\n\n".join(text)


async def stop(session_id: str | None, pid: int | None) -> dict[str, str]:
    session = match_session(await list_sessions(), session_id, pid)
    if session.job_id is None:
        raise ClaudeControlError("unsupported")
    await _command("stop", session.job_id, mutating=True)
    try:
        observed = [row for row in await list_sessions() if row.job_id == session.job_id]
    except ClaudeControlError:
        return {"agent": "unconfirmed", "reason": "unavailable"}
    if len(observed) == 1 and observed[0].state == "stopped" and observed[0].pid is None:
        return {"agent": "stopped"}
    return {"agent": "unconfirmed"}
