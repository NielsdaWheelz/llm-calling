"""One bounded JSON exchange for native operations on existing agent sessions."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Literal, Never, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from . import claude_control
from .codex_control import (
    CodexBounded,
    CodexControl,
    CodexControlError,
    CodexFinished,
    CodexInterrupted,
    CodexPromptRequest,
    CodexSubmit,
    CodexThreadRead,
    CodexThreadTarget,
    CodexTurnTarget,
)
from .errors import AgentRuntimeError, InvalidAgentRequest, ProtocolDefect


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class _Target(_Closed):
    sessionId: str | None = None
    pid: int | None = Field(default=None, gt=0)
    turnId: str | None = None

    @field_validator("sessionId", "turnId")
    @classmethod
    def canonical_uuid(cls, value: str | None) -> str | None:
        if value is not None and str(UUID(value)) != value:
            raise ValueError("invalid native identity")
        return value


class _Input(_Closed):
    text: str | None = None
    maxBytes: int = Field(default=16384, ge=1, le=32768)


class _Request(_Closed):
    operation: Literal["inspect", "read", "send", "interrupt", "stop"]
    provider: Literal["Codex", "Claude"]
    profileKey: str = Field(min_length=1, max_length=128)
    endpoint: str | None = None
    targets: list[_Target] = Field(min_length=1, max_length=128)
    input: _Input | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        if "input" in self.model_fields_set and self.input is None:
            raise ValueError("omit absent input")
        if self.operation != "inspect" and len(self.targets) != 1:
            raise ValueError("operation requires exactly one target")
        if self.provider == "Codex":
            if (
                not self.endpoint
                or not self.endpoint.startswith("unix:///")
                or "\0" in self.endpoint
            ):
                raise ValueError("codex requires an absolute unix endpoint")
            if any(target.sessionId is None for target in self.targets):
                raise ValueError("codex requires a session identity")
        elif self.endpoint is not None:
            raise ValueError("claude uses its selected profile environment")
        if any(target.sessionId is None and target.pid is None for target in self.targets):
            raise ValueError("target requires a native session or process identity")
        if self.operation == "send":
            if self.input is None or self.input.model_fields_set != {"text"}:
                raise ValueError("send requires text")
            if not self.input.text or len(self.input.text.encode("utf-8")) > 32768:
                raise ValueError("input exceeds bounds")
        elif self.operation == "read":
            if self.input is not None and self.input.model_fields_set - {"maxBytes"}:
                raise ValueError("read accepts only maxBytes")
        elif self.input is not None:
            raise ValueError("operation accepts no input")
        return self


def _error(code: str, dispatch: str = "not_sent") -> dict:
    return {"ok": False, "error": {"code": code, "dispatch": dispatch}}


def _success(result: object) -> dict:
    return {"ok": True, "result": result}


def _codex_error(error: CodexControlError) -> dict:
    if error.dispatch == "Unknown":
        return _error("unknown", "unknown")
    code = {
        "missing": "stale",
        "stale": "stale",
        "unavailable": "unavailable",
        "output_limit": "unavailable",
    }.get(error.code, "rejected")
    return _error(code)


def _codex_observation(value: CodexThreadRead) -> dict:
    status = value.thread.status
    state, reason = "unknown", None
    if status == "active":
        state = "working"
        if "waitingOnApproval" in value.thread.active_flags:
            state, reason = "blocked", "permission"
        elif "waitingOnUserInput" in value.thread.active_flags:
            state, reason = "blocked", "input"
    elif status == "systemError":
        state = "failed"
    elif status == "idle":
        state = "idle"
        if value.turn:
            state = {
                "completed": "done",
                "failed": "failed",
                "interrupted": "stopped",
                "inProgress": "working",
            }[value.turn.status]
    result: dict = {
        "status": {"state": state, "source": "unavailable" if status == "notLoaded" else "native"},
        "sessionId": value.thread.target.thread_handle,
        "methods": dict.fromkeys(("read", "send", "interrupt"), "native"),
    }
    if status == "notLoaded":
        result["methods"].update(send="terminal", interrupt="terminal")
    if reason:
        result["status"]["reason"] = reason
    if value.turn:
        result["turnId"] = value.turn.target.turn_handle
    return result


def _read_result(text: str, scope: str, max_bytes: int, truncated: bool = False) -> dict:
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        text = encoded[-max_bytes:].decode("utf-8", errors="ignore")
        truncated = True
    result = {"text": text, "source": "native", "scope": scope, "truncated": truncated}
    # JSON escaping can exceed the text-byte limit (for example control characters).
    while len(json.dumps(_success(result), ensure_ascii=False).encode("utf-8")) >= 65536:
        result["text"] = str(result["text"])[len(str(result["text"])) // 4 :]
        result["truncated"] = True
    return result


async def _codex(request: _Request) -> dict:
    assert request.endpoint is not None
    targets = tuple(
        CodexThreadTarget(request.profileKey, target.sessionId or "") for target in request.targets
    )
    control = CodexControl(
        {request.profileKey: Path(request.endpoint[7:])}, is_managed=lambda _: False
    )
    try:
        if request.operation == "inspect":
            rows = await control.inspect(targets)
            return _success(
                [
                    _codex_error(row)
                    if isinstance(row, CodexControlError)
                    else _success(_codex_observation(row))
                    for row in rows
                ]
            )
        target = targets[0]
        if request.operation == "read":
            observed = await control.read(target)
            if observed.last_answer is None:
                return _error("unavailable")
            return _success(
                _read_result(
                    observed.last_answer,
                    "latest_turn",
                    request.input.maxBytes if request.input else 16384,
                    isinstance(observed.coverage, CodexBounded),
                )
            )
        if request.operation == "send":
            assert request.input is not None and request.input.text is not None
            result = await control.prompt(
                CodexPromptRequest(target, CodexSubmit(request.input.text))
            )
            return _success(
                {"method": "native", "outcome": "accepted", "turnId": result.turn_handle}
            )
        observed = (await control.inspect((target,)))[0]
        if isinstance(observed, CodexControlError):
            return _codex_error(observed)
        if observed.thread.status == "notLoaded":
            return _error("unavailable")
        expected = request.targets[0].turnId
        if (observed.turn.target.turn_handle if observed.turn else None) != expected:
            return _error("stale")
        if observed.turn is None:
            if observed.thread.status != "idle":
                return _error("unavailable")
            return _success(
                {"agent": "idle"}
                if request.operation == "stop"
                else {"method": "native", "outcome": "finished"}
            )
        turn = CodexTurnTarget(target, observed.turn.target.turn_handle)
        result = await control.interrupt(turn)
        if isinstance(result, CodexInterrupted):
            outcome, agent = "interrupted", "interrupted"
        elif isinstance(result, CodexFinished):
            outcome, agent = "finished", "idle"
        else:
            # Stale may be observed after an accepted interrupt; it is not proof of no effect.
            outcome, agent = "unknown", "unconfirmed"
        return _success(
            {"agent": agent}
            if request.operation == "stop"
            else {"method": "native", "outcome": outcome, "turnId": turn.turn_handle}
        )
    except CodexControlError as error:
        return _codex_error(error)
    finally:
        await control.close()


async def _claude(request: _Request) -> dict:
    if request.operation in ("send", "interrupt"):
        return _error("unsupported")
    if request.operation == "inspect":
        sessions = await claude_control.list_sessions()
        history_available = claude_control.history_available()
        rows = []
        for target in request.targets:
            try:
                session = claude_control.match_session(sessions, target.sessionId, target.pid)
            except claude_control.ClaudeControlError as error:
                rows.append(_error(error.code, error.dispatch))
                continue
            result: dict = {
                "status": {"state": session.state, "source": "native"},
                "methods": {
                    "read": "native" if session.session_id and history_available else "terminal",
                    "send": "terminal",
                    "interrupt": "terminal",
                },
            }
            if session.reason:
                result["status"]["reason"] = session.reason
            if session.session_id:
                result["sessionId"] = session.session_id
            if (
                session.kind == "interactive"
                and target.pid is not None
                and session.pid == target.pid
            ):
                result["terminalOwnsAgent"] = True
            rows.append(_success(result))
        return _success(rows)
    target = request.targets[0]
    if request.operation == "read":
        if target.sessionId is None:
            return _error("unsupported")
        text = claude_control.read_messages(target.sessionId)
        return _success(
            _read_result(
                text, "recent_messages", request.input.maxBytes if request.input else 16384
            )
        )
    return _success(await claude_control.stop(target.sessionId, target.pid))


async def _run(request: _Request) -> dict:
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await (_codex(request) if request.provider == "Codex" else _claude(request))
    except claude_control.ClaudeControlError as error:
        return _error(error.code, error.dispatch)
    except InvalidAgentRequest:
        return _error("rejected")
    except asyncio.CancelledError:
        if request.operation in ("send", "interrupt", "stop"):
            return _error("unknown", "unknown")
        return _error("unavailable")
    except (AgentRuntimeError, ProtocolDefect, OSError, ValueError):
        if request.operation in ("send", "interrupt", "stop"):
            return _error("unknown", "unknown")
        return _error("unavailable")
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _reject_constant(value: str) -> Never:
    raise ValueError("non-finite JSON value")


def main() -> None:
    try:
        encoded = sys.stdin.buffer.read(65537)
        if len(encoded) > 65536:
            raise ValueError("input exceeds bounds")
        request = _Request.model_validate(
            json.loads(encoded, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        )
    except (ValueError, ValidationError, UnicodeError):
        result = _error("rejected")
    else:
        result = asyncio.run(_run(request))
    output = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(output) > 65536:
        result = _error("unavailable")
        output = json.dumps(result).encode() + b"\n"
    sys.stdout.buffer.write(output)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
