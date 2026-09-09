"""Owned Codex app-server stdio transport.

The public app-server protocol is newline-delimited JSON-RPC over stdio.  This module
owns that byte stream directly: every client response is correlated, every notification
is queued in wire order, and every server-initiated request receives one explicit denial
or closes the transport as a protocol defect.  It deliberately does not import or reach
through the Python SDK's request loop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from provider_runtime.errors import sanitize_provider_text

from ._limits import (
    _MAX_EVENT_COUNT,
    _MAX_MESSAGE_BYTES,
    _MAX_MESSAGE_ITEMS,
    _MAX_TURN_OUTPUT_BYTES,
    _OPERATION_TIMEOUT_SECONDS,
    OutputLimitExceeded,
    bounded_payload_size,
)
from ._process import ManagedProcess, ProcessLimits
from .errors import ProtocolDefect

type CodexRequestId = str | int
type CodexServerRequestKind = Literal["permission", "tool"]

_PROCESS_LIMITS = ProcessLimits(max_stderr_bytes=64 * 1024, termination_grace_seconds=0.25)
_SERVER_REQUEST_DENIALS: dict[str, tuple[CodexServerRequestKind, dict[str, object]]] = {
    "item/commandExecution/requestApproval": ("permission", {"decision": "decline"}),
    "item/fileChange/requestApproval": ("permission", {"decision": "decline"}),
    "item/permissions/requestApproval": (
        "permission",
        {"permissions": {}, "scope": "turn"},
    ),
    "execCommandApproval": ("permission", {"decision": "denied"}),
    "applyPatchApproval": ("permission", {"decision": "denied"}),
    "item/tool/call": ("tool", {"contentItems": [], "success": False}),
    "item/tool/requestUserInput": ("permission", {"answers": {}}),
    "mcpServer/elicitation/request": (
        "permission",
        {"action": "decline", "content": None},
    ),
}
# These documented callbacks cannot be satisfied without transferring credential,
# attestation, or host-clock authority into the provider process.  They have no denial
# result variant, so the only honest response is a JSON-RPC error followed by fail-stop.
_FORBIDDEN_SERVER_REQUESTS: dict[str, str] = {
    "account/chatgptAuthTokens/refresh": "credential callbacks are unsupported",
    "attestation/generate": "attestation callbacks are unsupported",
}
# ``currentTime/read`` is experimental in the certified schema and is disabled by the
# initialize capability.  If it nevertheless appears, retain its first-class tool shape
# before the connection fails; fabricating a timestamp would grant unrequested authority.
_FORBIDDEN_AUTHORITY_REQUESTS: dict[str, tuple[CodexServerRequestKind, str]] = {
    "currentTime/read": ("tool", "host time callbacks are unsupported"),
}


@dataclass(frozen=True, slots=True)
class CodexAppServerConfig:
    executable: Path
    cwd: Path
    environment: Mapping[str, str]
    config_overrides: tuple[str, ...] = ()
    client_name: str = "provider_runtime"
    client_title: str = "provider-runtime"
    client_version: str = "0.1.0"

    def __post_init__(self) -> None:
        if not self.executable.is_absolute():
            raise ValueError("Codex app-server executable must be absolute")
        if not self.cwd.is_absolute():
            raise ValueError("Codex app-server cwd must be absolute")
        if any(
            type(value) is not str or not value or "\0" in value or "\n" in value
            for value in (
                self.client_name,
                self.client_title,
                self.client_version,
                *self.config_overrides,
            )
        ):
            raise ValueError("Codex app-server configuration contains an invalid string")
        if any(
            type(key) is not str
            or type(value) is not str
            or not key
            or "=" in key
            or "\0" in key
            or "\0" in value
            for key, value in self.environment.items()
        ):
            raise ValueError("Codex app-server environment is invalid")


@dataclass(frozen=True, slots=True)
class CodexNotification:
    method: str
    params: dict[str, object]


@dataclass(frozen=True, slots=True)
class CodexServerRequest:
    request_id: CodexRequestId
    method: str
    params: dict[str, object]
    kind: CodexServerRequestKind


@dataclass(frozen=True, slots=True)
class _TransportFailure:
    error: ProtocolDefect


type CodexServerMessage = CodexNotification | CodexServerRequest


class CodexAppServerResponseError(Exception):
    """One valid, correlated JSON-RPC error response with sanitized detail."""


class CodexAppServerClient:
    """One direct, correlated app-server connection and its owned process group."""

    def __init__(self, config: CodexAppServerConfig) -> None:
        self.config = config
        self.metadata: dict[str, object] = {}
        self._process: ManagedProcess | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, tuple[str, asyncio.Future[object]]] = {}
        self._messages: asyncio.Queue[tuple[CodexServerMessage | _TransportFailure, int]] = (
            asyncio.Queue()
        )
        self._queued_bytes = 0
        self._next_request_id = 1
        self._server_request_ids: set[tuple[type[object], object]] = set()
        self._failure: ProtocolDefect | None = None
        self._closing = False
        self._closed = False

    async def __aenter__(self) -> CodexAppServerClient:
        if self._process is not None or self._closed:
            raise ProtocolDefect("Codex app-server client was started more than once")
        argv: list[str] = [str(self.config.executable)]
        for override in self.config.config_overrides:
            argv.extend(("--config", override))
        argv.extend(("app-server", "--listen", "stdio://", "--strict-config"))
        self._process = await ManagedProcess.spawn(
            argv,
            cwd=self.config.cwd,
            environment=self.config.environment,
            limits=_PROCESS_LIMITS,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        try:
            initialized = await self.request(
                "initialize",
                {
                    "capabilities": {"experimentalApi": False},
                    "clientInfo": {
                        "name": self.config.client_name,
                        "title": self.config.client_title,
                        "version": self.config.client_version,
                    },
                },
            )
            self.metadata = self._mapping(initialized, "initialize response")
            await self.notify("initialized", None)
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        await self.close()

    async def request(self, method: str, params: Mapping[str, object] | None) -> object:
        self._require_method(method)
        self._require_live()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()
        request_id = self._next_request_id
        self._next_request_id += 1
        self._pending[request_id] = (method, future)
        try:
            await self._write(
                {
                    "id": request_id,
                    "method": method,
                    **({} if params is None else {"params": params}),
                }
            )
        except BaseException:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            raise
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                return await asyncio.shield(future)
        except TimeoutError:
            error = ProtocolDefect(f"Codex app-server {method} request timed out")
            self._fail(error)
            await self.close()
            raise error from None
        except asyncio.CancelledError:
            future.add_done_callback(self._consume_future)
            raise

    async def notify(self, method: str, params: Mapping[str, object] | None) -> None:
        self._require_method(method)
        self._require_live()
        await self._write({"method": method, **({} if params is None else {"params": params})})

    async def next_message(self) -> CodexServerMessage:
        if self._failure is not None and self._messages.empty():
            raise self._failure
        message, size = await self._messages.get()
        self._queued_bytes -= size
        if isinstance(message, _TransportFailure):
            raise message.error
        if (
            self._failure is not None
            and isinstance(message, CodexNotification)
            and message.method == "turn/completed"
        ):
            raise self._failure
        return message

    def take_pending_messages(self) -> tuple[CodexServerMessage, ...]:
        values: list[CodexServerMessage] = []
        while not self._messages.empty():
            message, size = self._messages.get_nowait()
            self._queued_bytes -= size
            if isinstance(message, _TransportFailure):
                raise message.error
            values.append(message)
        if self._failure is not None:
            raise self._failure
        return tuple(values)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closing = True
        process = self._process
        if process is not None:
            await process.close()
        reader = self._reader_task
        if reader is not None:
            await asyncio.gather(reader, return_exceptions=True)
        closed = ProtocolDefect("Codex app-server connection closed")
        for _method, future in self._pending.values():
            if not future.done():
                future.set_exception(closed)
                future.add_done_callback(self._consume_future)
        self._pending.clear()

    async def account(self) -> object:
        return await self.request("account/read", {"refreshToken": False})

    async def thread_list(self, **kwargs: object) -> object:
        return await self.request("thread/list", self._camel_params(kwargs))

    async def thread_start(self, **kwargs: object) -> CodexThread:
        response = self._mapping(
            await self.request("thread/start", self._thread_params(kwargs)),
            "thread/start response",
        )
        return CodexThread(self, self._response_thread_id(response, "thread/start"))

    async def thread_resume(self, thread_id: str, **kwargs: object) -> CodexThread:
        response = self._mapping(
            await self.request(
                "thread/resume", {"threadId": thread_id, **self._thread_params(kwargs)}
            ),
            "thread/resume response",
        )
        return CodexThread(self, self._response_thread_id(response, "thread/resume"))

    async def thread_fork(self, thread_id: str, **kwargs: object) -> CodexThread:
        response = self._mapping(
            await self.request(
                "thread/fork", {"threadId": thread_id, **self._thread_params(kwargs)}
            ),
            "thread/fork response",
        )
        return CodexThread(self, self._response_thread_id(response, "thread/fork"))

    async def _reader_loop(self) -> None:
        process = self._process
        if process is None:
            return
        buffer = bytearray()
        try:
            while True:
                chunk = await process.stdout.read(8192)
                if not chunk:
                    if buffer:
                        raise ProtocolDefect("Codex app-server ended with a partial JSON message")
                    if not self._closing:
                        raise ProtocolDefect("Codex app-server exited before clean shutdown")
                    return
                buffer.extend(chunk)
                if len(buffer) > _MAX_MESSAGE_BYTES and b"\n" not in buffer:
                    raise ProtocolDefect("Codex app-server message exceeded its byte bound")
                while True:
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(buffer[:newline])
                    del buffer[: newline + 1]
                    if not line:
                        raise ProtocolDefect("Codex app-server emitted an empty JSON line")
                    if len(line) > _MAX_MESSAGE_BYTES:
                        raise ProtocolDefect("Codex app-server message exceeded its byte bound")
                    await self._route_line(line)
        except asyncio.CancelledError:
            if not self._closing:
                self._fail(ProtocolDefect("Codex app-server reader was cancelled"))
            raise
        except ProtocolDefect as error:
            self._fail(error)
        except (OSError, UnicodeError, ValueError, RecursionError):
            self._fail(ProtocolDefect("Codex app-server emitted malformed protocol data"))
        finally:
            if self._failure is not None:
                await process.close()

    async def _route_line(self, line: bytes) -> None:
        try:
            message = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise ProtocolDefect("Codex app-server emitted malformed JSON") from None
        try:
            bounded_payload_size(
                message,
                _MAX_MESSAGE_BYTES,
                max_items=_MAX_MESSAGE_ITEMS,
            )
        except OutputLimitExceeded:
            raise ProtocolDefect("Codex app-server message exceeded its structural bound") from None
        if not isinstance(message, dict):
            raise ProtocolDefect("Codex app-server message was not an object")
        if "method" in message:
            if "result" in message or "error" in message:
                raise ProtocolDefect("Codex app-server method message mixed response fields")
            await self._route_method(message, size=len(line))
            return
        self._route_response(message)

    async def _route_method(self, message: dict[str, object], *, size: int) -> None:
        allowed = {"method", "params", "id"}
        if set(message) - allowed:
            raise ProtocolDefect("Codex app-server method message had unknown fields")
        method = message.get("method")
        self._require_method(method)
        method = cast(str, method)
        params = self._mapping(message.get("params"), f"{method} params")
        if "id" not in message:
            self._enqueue_message(CodexNotification(method=method, params=params), size=size)
            return
        request_id = message["id"]
        self._require_request_id(request_id)
        identity = (type(request_id), request_id)
        if identity in self._server_request_ids:
            raise ProtocolDefect("Codex app-server repeated a server request identity")
        self._server_request_ids.add(identity)
        if len(self._server_request_ids) > _MAX_MESSAGE_ITEMS:
            raise ProtocolDefect("Codex app-server server-request count exceeded its bound")
        denial = _SERVER_REQUEST_DENIALS.get(method)
        if denial is not None:
            kind, result = denial
            await self._write({"id": request_id, "result": result})
            self._enqueue_message(
                CodexServerRequest(
                    request_id=cast(CodexRequestId, request_id),
                    method=method,
                    params=params,
                    kind=kind,
                ),
                size=size,
            )
            return
        forbidden_authority = _FORBIDDEN_AUTHORITY_REQUESTS.get(method)
        if forbidden_authority is not None:
            kind, denial_message = forbidden_authority
            await self._write(
                {
                    "id": request_id,
                    "error": {"code": -32601, "message": denial_message},
                }
            )
            self._enqueue_message(
                CodexServerRequest(
                    request_id=cast(CodexRequestId, request_id),
                    method=method,
                    params=params,
                    kind=kind,
                ),
                size=size,
            )
            raise ProtocolDefect(f"Codex app-server requested forbidden authority {method}")
        forbidden = _FORBIDDEN_SERVER_REQUESTS.get(method)
        if forbidden is not None:
            await self._write({"id": request_id, "error": {"code": -32601, "message": forbidden}})
            raise ProtocolDefect(f"Codex app-server requested forbidden callback {method}")
        await self._write(
            {
                "id": request_id,
                "error": {"code": -32601, "message": "unsupported server request"},
            }
        )
        raise ProtocolDefect(f"Codex app-server sent unknown server request {method}")

    def _route_response(self, message: dict[str, object]) -> None:
        if set(message) not in ({"id", "result"}, {"id", "error"}):
            raise ProtocolDefect("Codex app-server response shape was malformed")
        request_id = message.get("id")
        if type(request_id) is not int:
            raise ProtocolDefect("Codex app-server response identity was malformed")
        pending = self._pending.get(request_id)
        if pending is None:
            raise ProtocolDefect("Codex app-server response identity was not pending")
        method, future = pending
        if future.done():
            raise ProtocolDefect("Codex app-server response identity completed more than once")
        if "error" in message:
            error = self._mapping(message["error"], f"{method} error response")
            if set(error) - {"code", "message", "data"}:
                raise ProtocolDefect("Codex app-server error response had unknown fields")
            code = error.get("code")
            text = error.get("message")
            if type(code) is not int or not isinstance(text, str) or not text:
                raise ProtocolDefect("Codex app-server error response was malformed")
            detail = sanitize_provider_text(text, limit=512)
            del self._pending[request_id]
            future.set_exception(
                CodexAppServerResponseError(
                    f"Codex app-server {method} request failed ({code}): {detail}"
                )
            )
            return
        del self._pending[request_id]
        future.set_result(message["result"])

    async def _write(self, message: Mapping[str, object]) -> None:
        process = self._process
        if process is None:
            raise ProtocolDefect("Codex app-server process is not live")
        try:
            encoded = (
                json.dumps(
                    dict(message), separators=(",", ":"), ensure_ascii=False, allow_nan=False
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError, RecursionError):
            raise ProtocolDefect("Codex app-server outbound message was not JSON") from None
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ProtocolDefect("Codex app-server outbound message exceeded its byte bound")
        async with self._write_lock:
            try:
                await process.send(encoded)
            except (OSError, RuntimeError):
                error = ProtocolDefect("Codex app-server stdin closed unexpectedly")
                self._fail(error)
                raise error from None

    def _enqueue_message(self, message: CodexServerMessage, *, size: int) -> None:
        if (
            self._messages.qsize() >= _MAX_EVENT_COUNT
            or self._queued_bytes + size > _MAX_TURN_OUTPUT_BYTES
        ):
            raise ProtocolDefect("Codex app-server pending message queue exceeded its bound")
        self._queued_bytes += size
        self._messages.put_nowait((message, size))

    def _fail(self, error: ProtocolDefect) -> None:
        if self._failure is not None:
            return
        self._failure = error
        for _method, future in self._pending.values():
            if not future.done():
                future.set_exception(error)
                future.add_done_callback(self._consume_future)
        self._pending.clear()
        self._messages.put_nowait((_TransportFailure(error), 0))

    def _require_live(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._process is None or self._closed:
            raise ProtocolDefect("Codex app-server connection is not live")

    @staticmethod
    def _require_method(method: object) -> None:
        if not isinstance(method, str) or not method or len(method) > 256:
            raise ProtocolDefect("Codex app-server method was malformed")

    @staticmethod
    def _require_request_id(request_id: object) -> None:
        if type(request_id) not in (str, int) or request_id == "":
            raise ProtocolDefect("Codex app-server request identity was malformed")

    @staticmethod
    def _mapping(value: object, context: str) -> dict[str, object]:
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise ProtocolDefect(f"Codex app-server {context} was not an object")
        return dict(value)

    @staticmethod
    def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    @staticmethod
    def _reject_json_constant(_value: str) -> object:
        raise ValueError("non-finite JSON number")

    @staticmethod
    def _consume_future(future: asyncio.Future[object]) -> None:
        if not future.cancelled():
            future.exception()

    @staticmethod
    def _camel_params(values: Mapping[str, object]) -> dict[str, object]:
        names = {
            "include_turns": "includeTurns",
            "next_cursor": "nextCursor",
        }
        return {
            names.get(key, key): CodexAppServerClient._json_value(value)
            for key, value in values.items()
            if value is not None
        }

    @classmethod
    def _thread_params(cls, values: Mapping[str, object]) -> dict[str, object]:
        params = cls._camel_params(values)
        approval_mode = params.pop("approval_mode", None)
        if approval_mode is not None:
            normalized = getattr(approval_mode, "value", approval_mode)
            if normalized == "deny_all":
                params["approvalPolicy"] = "never"
            elif normalized == "auto_review":
                params["approvalPolicy"] = "on-request"
                params["approvalsReviewer"] = "auto_review"
            else:
                raise ProtocolDefect("Codex app-server approval mode was unsupported")
        sandbox = params.pop("sandbox", None)
        if sandbox is not None:
            params["sandbox"] = getattr(sandbox, "value", sandbox)
        names = {
            "base_instructions": "baseInstructions",
            "developer_instructions": "developerInstructions",
            "output_schema": "outputSchema",
        }
        return {names.get(key, key): value for key, value in params.items()}

    @classmethod
    def _json_value(cls, value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {key: cls._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json", by_alias=True, exclude_none=True)
        if is_dataclass(value) and not isinstance(value, type):
            return cls._json_value(asdict(cast(Any, value)))
        return value

    @classmethod
    def _response_thread_id(cls, response: Mapping[str, object], method: str) -> str:
        thread = cls._mapping(response.get("thread"), f"{method} thread")
        thread_id = thread.get("id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ProtocolDefect(f"Codex app-server {method} response had no thread id")
        return thread_id


class CodexThread:
    def __init__(self, client: CodexAppServerClient, thread_id: str) -> None:
        self._client = client
        self.id = thread_id

    async def turn(self, inputs: list[object], **kwargs: object) -> CodexTurn:
        params = {
            "threadId": self.id,
            "input": [self._client._json_value(item) for item in inputs],
            **self._client._thread_params(kwargs),
        }
        response = self._client._mapping(
            await self._client.request("turn/start", params), "turn/start response"
        )
        turn = self._client._mapping(response.get("turn"), "turn/start turn")
        turn_id = turn.get("id")
        if not isinstance(turn_id, str) or not turn_id:
            raise ProtocolDefect("Codex app-server turn/start response had no turn id")
        return CodexTurn(self._client, self.id, turn_id)

    async def read(self, *, include_turns: bool = False) -> object:
        return await self._client.request(
            "thread/read", {"threadId": self.id, "includeTurns": include_turns}
        )


class CodexTurn:
    def __init__(self, client: CodexAppServerClient, thread_id: str, turn_id: str) -> None:
        self._client = client
        self._thread_id = thread_id
        self.id = turn_id

    async def stream(self) -> AsyncGenerator[CodexServerMessage, None]:
        while True:
            message = await self._client.next_message()
            yield message
            if isinstance(message, CodexNotification) and message.method == "turn/completed":
                return

    async def interrupt(self) -> object:
        return await self._client.request(
            "turn/interrupt", {"threadId": self._thread_id, "turnId": self.id}
        )


__all__ = [
    "CodexAppServerClient",
    "CodexAppServerConfig",
    "CodexAppServerResponseError",
    "CodexNotification",
    "CodexServerMessage",
    "CodexServerRequest",
]
