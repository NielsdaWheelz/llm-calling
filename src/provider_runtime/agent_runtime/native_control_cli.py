"""One bounded JSON exchange for native operations on existing agent sessions."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Literal, Never, Self, assert_never
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from . import claude_control
from .codex_control import (
    CodexControl,
    CodexControlError,
    CodexCreateRequest,
    CodexThreadRead,
    CodexThreadTarget,
)
from .errors import AgentRuntimeError, InvalidAgentRequest, ProtocolDefect


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    @model_validator(mode="after")
    def omit_absent(self) -> Self:
        if any(getattr(self, name) is None for name in self.model_fields_set):
            raise ValueError("omit absent fields")
        return self


class _Target(_Closed):
    sessionId: str
    turnId: str | None = None

    @field_validator("sessionId", "turnId")
    @classmethod
    def canonical_uuid(cls, value: str | None) -> str | None:
        if value is not None and str(UUID(value)) != value:
            raise ValueError("invalid native identity")
        return value


class _Input(_Closed):
    cwd: str | None = Field(default=None, min_length=1, max_length=4096)
    text: str | None = None
    input: Literal["peer", "user"] | None = None
    delivery: Literal["direct", "queue"] | None = None
    scope: Literal["latest", "history"] | None = None
    maxBytes: int = Field(default=16384, ge=1, le=32768)
    cursor: str | None = Field(default=None, min_length=1, max_length=4096)


class _Request(_Closed):
    operation: Literal["create", "inspect", "read", "send", "interrupt", "stop", "results"]
    provider: Literal["Codex", "Claude"]
    profileKey: str = Field(min_length=1, max_length=128)
    endpoint: str | None = None
    targets: list[_Target] | None = Field(default=None, min_length=1, max_length=128)
    input: _Input | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        if self.provider == "Codex":
            if (
                not self.endpoint
                or not self.endpoint.startswith("unix:///")
                or "\0" in self.endpoint
            ):
                raise ValueError("codex requires an absolute unix endpoint")
        elif self.endpoint is not None:
            raise ValueError("claude uses its selected profile environment")
        if self.operation == "create":
            if (
                self.provider != "Codex"
                or self.targets is not None
                or self.input is None
                or self.input.model_fields_set != {"cwd"}
                or self.input.cwd is None
            ):
                raise ValueError("codex create requires only cwd input")
            if not Path(self.input.cwd).is_absolute() or "\0" in self.input.cwd:
                raise ValueError("create requires an absolute cwd")
            return self
        if self.targets is None:
            raise ValueError("operation requires targets")
        if self.operation != "inspect" and len(self.targets) != 1:
            raise ValueError("operation requires exactly one target")
        for target in self.targets:
            if self.provider == "Claude" and target.turnId is not None:
                raise ValueError("claude has no turn target")
            if self.operation == "results" and target.model_fields_set != {"sessionId"}:
                raise ValueError("results requires only conversation identity")
        if self.operation == "send":
            if self.input is None or self.input.model_fields_set != {"text", "input", "delivery"}:
                raise ValueError("send requires text, input and delivery")
            if not self.input.text or len(self.input.text.encode("utf-8")) > 32768:
                raise ValueError("input exceeds bounds")
            if self.input.input == "peer" and self.input.delivery == "queue":
                raise ValueError("peer input cannot be queued")
        elif self.operation == "read":
            if (
                self.input is None
                or self.input.scope is None
                or self.input.model_fields_set - {"scope", "maxBytes"}
            ):
                raise ValueError("read requires scope and optional maxBytes")
        elif self.operation == "results":
            if self.input is not None and self.input.model_fields_set - {"cursor"}:
                raise ValueError("results accepts only cursor")
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
        "history_changed": "history_changed",
        "unavailable": "unavailable",
        "output_limit": "unavailable",
        "auth": "unavailable",
        "quota": "unavailable",
    }.get(error.code, "rejected")
    return _error(code)


def _codex_observation(read: CodexThreadRead) -> dict:
    status = read.thread.status
    match status:
        case "active":
            state = "blocked" if read.thread.active_flags else "working"
        case "systemError":
            state = "failed"
        case "idle":
            state = "idle"
        case "notLoaded":
            state = "unknown"
        case unreachable:
            assert_never(unreachable)
    available = status in ("idle", "active")
    accepts_input = available and read.thread.can_accept_direct_input is True
    result: dict = {
        "status": {"state": state, "source": "unavailable" if status == "notLoaded" else "native"},
        "sessionId": read.thread.target.thread_handle,
        "methods": {
            "read": "native" if read.history_available else "unavailable",
            "sendPeer": "native" if accepts_input else "unavailable",
            "sendUser": "native" if accepts_input else "unavailable",
            "queueUser": "unavailable",
            "stop": "native"
            if status == "idle"
            or available
            and read.turn is not None
            and read.turn.status == "inProgress"
            else "unavailable",
        },
    }
    if read.turn:
        result["turn"] = {"id": read.turn.target.turn_handle, "state": read.turn.status}
    return result


def _bound_read(result: dict) -> dict:
    # Escaping and observation metadata count toward the same helper envelope bound.
    while len(json.dumps(_success(result), ensure_ascii=False).encode("utf-8")) >= 65536:
        text = result["text"]
        if not text:
            return _error("unavailable")
        result["text"] = text[len(text) // 4 + 1 :]
        result["truncated"] = True
    return _success(result)


async def _codex(request: _Request) -> dict:
    assert request.endpoint is not None
    control = CodexControl(
        {request.profileKey: Path(request.endpoint[7:])},
        is_managed=lambda _: False,
        native_owners=True,
    )
    try:
        if request.operation == "create":
            assert request.input is not None and request.input.cwd is not None
            created = await control.create(
                CodexCreateRequest(request.profileKey, Path(request.input.cwd))
            )
            return _success({"sessionId": created.thread_handle})
        assert request.targets is not None
        targets = tuple(
            CodexThreadTarget(request.profileKey, target.sessionId or "")
            for target in request.targets
        )
        if request.operation == "results":
            page = await control.results(
                targets[0], request.input.cursor if request.input else None
            )
            return _success(
                {
                    "resultIds": list(page.result_ids),
                    **({"nextCursor": page.next_cursor} if page.next_cursor else {}),
                }
            )
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
            assert request.input is not None and request.input.scope is not None
            value = await control.read_conversation(
                target, request.input.scope, request.input.maxBytes
            )
            output = value.output
            return _bound_read(
                {
                    "text": output.text,
                    "source": "native",
                    "scope": output.scope,
                    "truncated": output.truncated,
                    "outputState": output.state,
                    **({"outputId": output.output_id} if output.output_id else {}),
                    **({"outputTurnId": output.turn_id} if output.turn_id else {}),
                    "inspection": _codex_observation(value.inspection),
                }
            )
        if request.operation == "send":
            assert (
                request.input is not None
                and request.input.text is not None
                and request.input.input is not None
                and request.input.delivery is not None
            )
            return _success(
                await control.send_conversation(
                    target, request.input.text, request.input.input, request.input.delivery
                )
            )
        if request.operation in ("interrupt", "stop"):
            return _success(await control.interrupt_conversation(target, request.targets[0].turnId))
        assert_never(request.operation)
    except CodexControlError as error:
        return _codex_error(error)
    finally:
        await control.close()


def _claude_observation(session: claude_control.ClaudeSession) -> dict:
    if session.session_id is None:
        raise claude_control.ClaudeControlError("unavailable")
    result: dict = {
        "status": {"state": session.state, "source": "native"},
        "methods": {
            "read": "native" if claude_control.history_available() else "unavailable",
            "sendPeer": "unavailable",
            "sendUser": "unavailable",
            "queueUser": "unavailable",
            "stop": "native" if session.kind == "background" and session.job_id else "unavailable",
        },
    }
    result["sessionId"] = session.session_id
    return result


def _claude_conversation_observation(
    session_id: str, sessions: tuple[claude_control.ClaudeSession, ...] | None
) -> dict:
    if sessions is not None:
        try:
            session = claude_control.match_session(sessions, session_id, None)
        except claude_control.ClaudeControlError:
            pass
        else:
            return _claude_observation(session)
    return {
        "sessionId": session_id,
        "status": {"state": "unknown", "source": "unavailable"},
        "methods": {
            "read": "native" if claude_control.history_available() else "unavailable",
            "sendPeer": "unavailable",
            "sendUser": "unavailable",
            "queueUser": "unavailable",
            "stop": "unavailable",
        },
    }


async def _claude(request: _Request) -> dict:
    assert request.targets is not None
    target = request.targets[0]
    if request.operation in ("create", "interrupt", "send"):
        return _error("unavailable")
    if request.operation == "results":
        return _success(
            claude_control.results(
                target.sessionId, request.input.cursor if request.input else None
            )
        )
    if request.operation == "stop":
        return _success(await claude_control.stop_conversation(target.sessionId))
    try:
        sessions = await claude_control.list_sessions()
    except claude_control.ClaudeControlError:
        sessions = None
    if request.operation == "inspect":
        return _success(
            [
                _success(_claude_conversation_observation(row.sessionId, sessions))
                for row in request.targets
            ]
        )
    if request.operation == "read":
        assert request.input is not None and request.input.scope is not None
        result = claude_control.read_messages(
            target.sessionId, request.input.scope, request.input.maxBytes
        )
        result["inspection"] = _claude_conversation_observation(target.sessionId, sessions)
        return _bound_read(result)
    assert_never(request.operation)


async def _run(request: _Request) -> dict:
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    mutation = request.operation in ("create", "send", "interrupt", "stop")
    try:
        async with asyncio.timeout(2 if request.operation == "inspect" else 10):
            match request.provider:
                case "Codex":
                    return await _codex(request)
                case "Claude":
                    return await _claude(request)
                case unreachable:
                    assert_never(unreachable)
    except claude_control.ClaudeControlError as error:
        return _error("unavailable" if error.code == "unsupported" else error.code, error.dispatch)
    except InvalidAgentRequest:
        return _error("rejected")
    except (
        asyncio.CancelledError,
        TimeoutError,
        AgentRuntimeError,
        ProtocolDefect,
        OSError,
        ValueError,
    ):
        return _error("unknown", "unknown") if mutation else _error("unavailable")
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
    operation = None
    try:
        encoded = sys.stdin.buffer.read(65537)
        if len(encoded) > 65536:
            raise ValueError("input exceeds bounds")
        request = _Request.model_validate(
            json.loads(encoded, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        )
    except (ValueError, ValidationError, UnicodeError, RecursionError):
        result = _error("rejected")
    else:
        operation = request.operation
        result = asyncio.run(_run(request))
    output = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(output) > 65536:
        result = (
            _error("unknown", "unknown")
            if operation in ("create", "send", "stop", "interrupt")
            else _error("unavailable")
        )
        output = json.dumps(result).encode() + b"\n"
    sys.stdout.buffer.write(output)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
