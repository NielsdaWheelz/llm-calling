"""Correlated Codex JSON-RPC over WebSocket/UDS; owns only its connection."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

from provider_runtime.errors import sanitize_provider_text

from ._limits import (
    _MAX_MESSAGE_BYTES,
    _MAX_MESSAGE_ITEMS,
    _OPERATION_TIMEOUT_SECONDS,
    OutputLimitExceeded,
    bounded_payload_size,
)
from .errors import AgentRuntimeError, ProtocolDefect, SdkUnavailable

type CodexRequestId = str | int
type CodexServerRequestKind = Literal["permission", "tool"]

CODEX_VERSION = "0.153.4"
CODEX_THREAD_SOURCES = (
    "cli",
    "vscode",
    "exec",
    "appServer",
    "subAgent",
    "subAgentReview",
    "subAgentCompact",
    "subAgentThreadSpawn",
    "subAgentOther",
    "unknown",
)
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
    socket_path: Path
    request_policy: Literal["deny_owned", "observe_only"] = "deny_owned"
    client_name: str = "provider_runtime"
    client_title: str = "provider-runtime"
    client_version: str = "0.1.0"

    def __post_init__(self) -> None:
        if not self.socket_path.is_absolute():
            raise ValueError("Codex app-server socket must be absolute")
        if self.request_policy not in ("deny_owned", "observe_only"):
            raise ValueError("Codex app-server request policy is invalid")
        if any(
            type(value) is not str or not value or "\0" in value or "\n" in value
            for value in (
                self.client_name,
                self.client_title,
                self.client_version,
            )
        ):
            raise ValueError("Codex app-server configuration contains an invalid string")


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
    error: ProtocolDefect | CodexConnectionUnavailable


type CodexServerMessage = CodexNotification | CodexServerRequest


class CodexAppServerResponseError(Exception):
    """One valid, correlated JSON-RPC error response with sanitized detail."""

    def __init__(self, method: str, code: int, detail: str, data: object = None) -> None:
        super().__init__(f"Codex app-server {method} request failed ({code}): {detail}")
        self.code = code
        self.detail = detail
        self.data = data


class CodexConnectionUnavailable(AgentRuntimeError):
    def __init__(self) -> None:
        super().__init__("Codex connection is unavailable", code="codex_unavailable")


class CodexAppServerClient:
    """One direct connection; disconnecting never terminates its external server."""

    def __init__(self, config: CodexAppServerConfig) -> None:
        self.config = config
        self.metadata: dict[str, object] = {}
        self._connection: ClientConnection | None = None
        self._thread_id: str | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, tuple[str, asyncio.Future[object]]] = {}
        self._messages: asyncio.Queue[CodexServerMessage | _TransportFailure] = asyncio.Queue(128)
        self._next_request_id = 1
        self._server_request_ids: set[tuple[type[object], object]] = set()
        self._failure: ProtocolDefect | CodexConnectionUnavailable | None = None
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> CodexAppServerClient:
        try:
            from websockets.asyncio.client import unix_connect
            from websockets.exceptions import InvalidHandshake
        except ModuleNotFoundError:
            raise SdkUnavailable("shared Codex requires the websockets dependency") from None
        if self._connection is not None or self._closed:
            raise ProtocolDefect("Codex app-server client was started more than once")
        try:
            self._connection = await unix_connect(
                str(self.config.socket_path),
                uri="ws://localhost",
                open_timeout=_OPERATION_TIMEOUT_SECONDS,
                close_timeout=2,
                max_size=_MAX_MESSAGE_BYTES,
                max_queue=16,
                compression=None,
                proxy=None,
            )
        except (OSError, TimeoutError):
            raise CodexConnectionUnavailable() from None
        except InvalidHandshake:
            raise ProtocolDefect(
                "Codex socket did not provide the pinned WebSocket protocol"
            ) from None
        if self._closed:
            await self._connection.close()
            raise CodexConnectionUnavailable()
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
            user_agent = self.metadata.get("userAgent")
            if (
                not isinstance(user_agent, str)
                or not user_agent.split(" ", 1)[0].partition("/")[0]
                or user_agent.split(" ", 1)[0].partition("/")[2] != CODEX_VERSION
            ):
                raise ProtocolDefect("Codex server version does not match the qualified pin")
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
        if len(self._pending) >= 128:
            raise ProtocolDefect("Codex app-server pending request count exceeded its bound")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()
        request_id = self._next_request_id
        self._next_request_id += 1
        self._pending[request_id] = (method, future)
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                await self._write(
                    {
                        "id": request_id,
                        "method": method,
                        **({} if params is None else {"params": params}),
                    }
                )
        except TimeoutError:
            self._pending.pop(request_id, None)
            future.cancel()
            raise CodexConnectionUnavailable() from None
        except BaseException:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            raise
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                return await asyncio.shield(future)
        except TimeoutError:
            future.add_done_callback(self._consume_future)
            raise CodexConnectionUnavailable() from None
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
        message = await self._messages.get()
        if isinstance(message, _TransportFailure):
            raise message.error
        return message

    def take_pending_messages(self) -> tuple[CodexServerMessage, ...]:
        values: list[CodexServerMessage] = []
        while not self._messages.empty():
            message = self._messages.get_nowait()
            if isinstance(message, _TransportFailure):
                raise message.error
            values.append(message)
        if self._failure is not None:
            raise self._failure
        return tuple(values)

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_connection())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await asyncio.shield(self._close_task)
            raise

    async def _close_connection(self) -> None:
        self._closed = True
        self._closing = True
        connection = self._connection
        if connection is not None:
            await connection.close()
        reader = self._reader_task
        if reader is not None:
            await asyncio.gather(reader, return_exceptions=True)
        closed = CodexConnectionUnavailable()
        for _method, future in self._pending.values():
            if not future.done():
                future.set_exception(closed)
                future.add_done_callback(self._consume_future)
        self._pending.clear()

    async def account(self) -> object:
        return await self.request("account/read", {"refreshToken": False})

    async def thread_list(self, **kwargs: object) -> object:
        kwargs["sourceKinds"] = list(CODEX_THREAD_SOURCES)
        return await self.request("thread/list", self._camel_params(kwargs))

    async def thread_start(self, **kwargs: object) -> CodexThread:
        response = self._mapping(
            await self.request("thread/start", self._thread_params(kwargs)),
            "thread/start response",
        )
        return self._select_thread(self._response_thread_id(response, "thread/start"))

    async def thread_resume(self, thread_id: str, **kwargs: object) -> CodexThread:
        self._thread_id = thread_id
        response = self._mapping(
            await self.request(
                "thread/resume", {"threadId": thread_id, **self._thread_params(kwargs)}
            ),
            "thread/resume response",
        )
        return self._select_thread(self._response_thread_id(response, "thread/resume"))

    async def thread_fork(self, thread_id: str, **kwargs: object) -> CodexThread:
        response = self._mapping(
            await self.request(
                "thread/fork", {"threadId": thread_id, **self._thread_params(kwargs)}
            ),
            "thread/fork response",
        )
        return self._select_thread(self._response_thread_id(response, "thread/fork"))

    def _select_thread(self, thread_id: str) -> CodexThread:
        self._thread_id = thread_id
        retained = self.take_pending_messages()
        for message in retained:
            if self._belongs_to_thread(message.params):
                self._enqueue(message)
        return CodexThread(self, thread_id)

    def _belongs_to_thread(self, params: Mapping[str, object]) -> bool:
        thread_id = params.get("threadId")
        thread = params.get("thread")
        if thread_id is None and isinstance(thread, Mapping):
            thread_id = thread.get("id")
        return self._thread_id is None or thread_id in (None, self._thread_id)

    async def _reader_loop(self) -> None:
        from websockets.exceptions import ConnectionClosed

        connection = self._connection
        if connection is None:
            return
        try:
            async for frame in connection:
                if not isinstance(frame, str):
                    raise ProtocolDefect("Codex app-server emitted a non-text WebSocket frame")
                await self._route_line(frame.encode("utf-8"))
            if not self._closing:
                self._fail(CodexConnectionUnavailable())
        except asyncio.CancelledError:
            if not self._closing:
                self._fail(ProtocolDefect("Codex app-server reader was cancelled"))
            raise
        except ProtocolDefect as error:
            self._fail(error)
        except ConnectionClosed:
            if not self._closing:
                self._fail(CodexConnectionUnavailable())
        except (OSError, UnicodeError, ValueError, RecursionError):
            self._fail(ProtocolDefect("Codex app-server emitted malformed protocol data"))

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
            await self._route_method(message)
            return
        self._route_response(message)

    async def _route_method(self, message: dict[str, object]) -> None:
        allowed = {"method", "params", "id"}
        if set(message) - allowed:
            raise ProtocolDefect("Codex app-server method message had unknown fields")
        method = message.get("method")
        self._require_method(method)
        method = cast(str, method)
        params = self._mapping(message.get("params"), f"{method} params")
        if self.config.request_policy == "observe_only":
            # Worker requests are intentionally left pending for a native TUI. Never
            # turn a subscription race into a competing approval response.
            return
        if not self._belongs_to_thread(params):
            return
        if "id" not in message:
            self._enqueue(CodexNotification(method=method, params=params))
            return
        if self._thread_id is None or params.get("threadId") != self._thread_id:
            raise ProtocolDefect("Codex server request omitted its managed thread identity")
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
            self._enqueue(
                CodexServerRequest(
                    request_id=cast(CodexRequestId, request_id),
                    method=method,
                    params=params,
                    kind=kind,
                )
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
            self._enqueue(
                CodexServerRequest(
                    request_id=cast(CodexRequestId, request_id),
                    method=method,
                    params=params,
                    kind=kind,
                )
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
            future.set_exception(
                CodexAppServerResponseError(method, code, detail, error.get("data"))
            )
            self._pending.pop(request_id)
            return
        if method in ("thread/start", "thread/resume", "thread/fork"):
            result = self._mapping(message["result"], "thread response")
            self._thread_id = self._response_thread_id(result, method)
        future.set_result(message["result"])
        self._pending.pop(request_id)

    async def _write(self, message: Mapping[str, object]) -> None:
        from websockets.exceptions import ConnectionClosed

        connection = self._connection
        if connection is None:
            raise ProtocolDefect("Codex app-server connection is not live")
        try:
            encoded = json.dumps(
                dict(message), separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            raise ProtocolDefect("Codex app-server outbound message was not JSON") from None
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ProtocolDefect("Codex app-server outbound message exceeded its byte bound")
        async with self._write_lock:
            try:
                await connection.send(encoded.decode("utf-8"))
            except (OSError, ConnectionClosed):
                error = CodexConnectionUnavailable()
                self._fail(error)
                raise error from None

    def _enqueue(self, message: CodexServerMessage) -> None:
        try:
            self._messages.put_nowait(message)
        except asyncio.QueueFull:
            raise ProtocolDefect("Codex app-server event queue exceeded its bound") from None

    def _fail(self, error: ProtocolDefect | CodexConnectionUnavailable) -> None:
        if self._failure is not None:
            return
        self._failure = error
        for _method, future in self._pending.values():
            if not future.done():
                future.set_exception(error)
                future.add_done_callback(self._consume_future)
        self._pending.clear()
        if self._messages.full():
            self._messages.get_nowait()
        self._messages.put_nowait(_TransportFailure(error))

    def _require_live(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._connection is None or self._closed:
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
