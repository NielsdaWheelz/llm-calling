"""Private, single-turn Unix interface to the contained Antigravity SDK host."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from secrets import token_hex
from typing import Any, Literal
from uuid import UUID, uuid4

from provider_runtime.types import CanonicalTool, canonical_json_bytes

from ._limits import _MAX_EVENT_COUNT, _OPERATION_TIMEOUT_SECONDS
from .antigravity_sdk import antigravity_native_request_fit
from .errors import (
    AgentRuntimeError,
    InvalidAgentRequest,
    ProtocolDefect,
    SessionUnavailable,
    UnsupportedCapability,
)
from .events import AgentEvent, AgentTerminal, AgentToolUse, StockToolObservation
from .neon_config import AGY_CAPABILITY_REVISION, AGY_PROFILE, AGY_SDK_VERSION, _origin
from .policy import PermissionPolicy
from .sessions import AgentSession, SessionQuery, SessionReadOptions, fingerprint_path
from .terminal_codec import (
    submission_from_json,
    submission_to_json,
    terminal_from_json,
    terminal_to_json,
)
from .turn import (
    AgentAccepted,
    AgentAttempt,
    AgentCloseResult,
    AgentControlReceipt,
    AgentInputRecorded,
    AgentMessage,
    AgentNotSubmitted,
    AgentSubmission,
    AgentToolCall,
    AgentToolReply,
    AgentTurnControls,
    AgentTurnRef,
    AgentUncertain,
)
from .types import (
    AgentOutputSpec,
    AgentSessionRef,
    AgentSessionRequest,
    AntigravitySdkSessionRequest,
    ApprovalHandler,
    ContentPart,
    CredentialRef,
    JsonSchemaAgentOutput,
    TextAgentOutput,
    TextContent,
    TurnRequest,
    freeze_json_object,
    freeze_json_value,
)

AGY_AUTH = CredentialRef("host_managed", AGY_PROFILE)
AGY_WORKSPACE_POLICY = PermissionPolicy(filesystem="workspace_write", allowed_tools=("*",))
_PROTOCOL = "antigravity-sdk-host.v1"
_FRAME_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AntigravityLease:
    turn: AgentTurnRef
    bearer: str = field(repr=False)
    _joined: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False, compare=False
    )

    @property
    def joined(self) -> bool:
        return self._joined.is_set()


def _send(writer: asyncio.StreamWriter, frame: str, **payload: Any) -> None:
    data = canonical_json_bytes(freeze_json_object({"frame": frame, **payload}))
    if len(data) >= _FRAME_BYTES:
        raise ProtocolDefect("Antigravity host frame exceeded its bound")
    writer.write(data + b"\n")


async def _receive(reader: asyncio.StreamReader) -> tuple[str, dict[str, Any]]:
    line = await reader.readline()
    if not line:
        raise EOFError("Antigravity host closed")
    value = json.loads(line)
    if not isinstance(value, dict) or type(kind := value.pop("frame", None)) is not str:
        raise ProtocolDefect("Antigravity host frame omitted its kind")
    return kind, value


async def _disconnect(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(ConnectionError):
        await writer.wait_closed()


class _Session(AgentSession):
    def __init__(
        self,
        lease: AntigravityLease,
        binding: str,
        output: AgentOutputSpec,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        super().__init__(lease.turn.session_ref)
        self.lease, self.binding, self.output = lease, binding, output
        self.reader, self.writer = reader, writer


class AntigravitySdkAdapter:
    backend: Literal["antigravity"] = "antigravity"
    transport: Literal["antigravity_sdk"] = "antigravity_sdk"
    cwd_scopes_sessions = True

    def __init__(self, endpoint: Path) -> None:
        self.endpoint = endpoint
        self.scope = fingerprint_path(endpoint)

    def validate_auth(self, credential: CredentialRef) -> None:
        if credential != AGY_AUTH:
            raise InvalidAgentRequest("Antigravity requires its qualified host profile")

    async def ready(self) -> bool:
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                reader, writer = await asyncio.open_unix_connection(
                    self.endpoint, limit=_FRAME_BYTES
                )
                try:
                    _send(writer, "ping", protocol=_PROTOCOL)
                    return await _receive(reader) == ("ready", {"protocol": _PROTOCOL})
                finally:
                    await _disconnect(writer)
        except (OSError, TimeoutError, EOFError, ValueError, ProtocolDefect):
            return False

    async def open_session(
        self, request: AgentSessionRequest, *, environment: Mapping[str, str]
    ) -> AgentSession:
        if not isinstance(request, AntigravitySdkSessionRequest) or environment:
            raise InvalidAgentRequest("Antigravity requires its exact native session request")
        expected = AGY_WORKSPACE_POLICY if request.workspace_tools else PermissionPolicy()
        if (
            request.policy != expected
            or any(
                Path(path).resolve() != Path(request.cwd, ".git").resolve()
                for path in request.additional_dirs
            )
            or any(
                not isinstance(part, TextContent) for part in (*request.system, *request.developer)
            )
        ):
            raise UnsupportedCapability(
                "Antigravity workspace or instruction authority is unsupported"
            )
        writer = None
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                reader, writer = await asyncio.open_unix_connection(
                    self.endpoint, limit=_FRAME_BYTES
                )
                open_request = dict(
                    protocol=_PROTOCOL,
                    cwd=request.cwd,
                    workspace_tools=request.workspace_tools,
                    revision=request.agent_definition_revision,
                    system="\n\n".join(
                        part.text
                        for part in (*request.system, *request.developer)
                        if isinstance(part, TextContent)
                    ),
                    output=asdict(request.output),
                    tools=[asdict(tool) for tool in request.tools],
                )
                binding = sha256(canonical_json_bytes(freeze_json_object(open_request))).hexdigest()
                _send(writer, "open", **open_request)
                kind, opened = await _receive(reader)
                if kind != "opened" or set(opened) != {"root", "turn", "bearer"}:
                    raise ProtocolDefect("Antigravity host did not open an exact session")
                root, turn, bearer = opened["root"], opened["turn"], opened["bearer"]
                if (
                    str(UUID(root)) != root
                    or str(UUID(turn)) != turn
                    or len(bearer) != 64
                    or len(bytes.fromhex(bearer)) != 32
                ):
                    raise ProtocolDefect("Antigravity host returned an invalid session identity")
                ref = AgentSessionRef(
                    "agent-session-ref.v1",
                    "antigravity",
                    "antigravity_sdk",
                    root,
                    AGY_PROFILE,
                    self.scope,
                    fingerprint_path(request.cwd),
                )
                return _Session(
                    AntigravityLease(AgentTurnRef(ref, turn), bearer),
                    binding,
                    request.output,
                    reader,
                    writer,
                )
        except BaseException as error:
            if writer is not None:
                await _disconnect(writer)
            if isinstance(error, (OSError, EOFError, TimeoutError)):
                raise SessionUnavailable("Antigravity host unavailable before submission") from None
            raise

    def prepare_turn(
        self,
        session: AgentSession,
        request: TurnRequest,
        *,
        attempt_id: str,
        input_id: str,
        controls: AgentTurnControls,
        release: Callable[[], None],
        validate_input: Callable[[tuple[ContentPart, ...]], None],
        controlled: bool,
    ) -> _RemoteTurn:
        if not isinstance(session, _Session) or not controlled:
            raise UnsupportedCapability("Antigravity requires a fresh controlled turn")
        if (
            request.policy is not None
            or len(request.input) != 1
            or not isinstance(request.input[0], TextContent)
        ):
            raise UnsupportedCapability("Antigravity requires one text batch and frozen authority")
        return _RemoteTurn(session, request, attempt_id, input_id, controls, release)

    @staticmethod
    def lease(session: AgentSession) -> AntigravityLease:
        if not isinstance(session, _Session):
            raise InvalidAgentRequest("session does not belong to Antigravity")
        return session.lease

    async def model_catalog(self, *, environment: Mapping[str, str]) -> Any:
        raise UnsupportedCapability("Antigravity is bound to one exact model at release")

    async def list_sessions(self, query: SessionQuery, *, environment: Mapping[str, str]) -> Any:
        raise UnsupportedCapability("Antigravity sessions are disposable")

    async def read_session(
        self, ref: AgentSessionRef, options: SessionReadOptions, *, environment: Mapping[str, str]
    ) -> Any:
        raise UnsupportedCapability("Antigravity sessions are disposable")

    def stream_turn(
        self, session: AgentSession, request: TurnRequest, *, approvals: ApprovalHandler | None
    ) -> AsyncGenerator[AgentEvent, None]:
        raise UnsupportedCapability("Antigravity requires a prepared turn")

    async def interrupt(self, session: AgentSession) -> None:
        await self.close_session(session)

    async def close_session(self, session: AgentSession) -> None:
        if isinstance(session, _Session):
            await _disconnect(session.writer)

    async def close(self) -> None:
        pass


class _RemoteTurn:
    def __init__(
        self,
        session: _Session,
        request: TurnRequest,
        attempt_id: str,
        input_id: str,
        controls: AgentTurnControls,
        release: Callable[[], None],
    ) -> None:
        (content,) = request.input
        assert isinstance(content, TextContent)
        if not antigravity_native_request_fit(
            content.text, output=session.output, input_id=input_id
        ).fits:
            raise InvalidAgentRequest("Antigravity input does not fit before submission")
        self._session = session
        self._turn = session.lease.turn
        self._controls = controls
        self._release = release
        self.submitted_request = freeze_json_object(
            {
                "type": _PROTOCOL,
                "binding": session.binding,
                "root": self._turn.session_ref.native_session_id,
                "turn": self._turn.native_turn_id,
                "input_id": input_id,
                "text": content.text,
                "timeout": request.timeout_seconds,
                "controls": asdict(controls),
            }
        )
        self.attempt = AgentAttempt(
            attempt_id, sha256(canonical_json_bytes(self.submitted_request)).hexdigest()
        )
        self.submission: AgentSubmission | None = None
        self.terminal: AgentTerminal | None = None
        self._revoked = False
        self._stream_consumed = False
        self._driver: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[AgentEvent | BaseException | None] = asyncio.Queue()
        self._submitted: asyncio.Future[AgentSubmission] = (
            asyncio.get_running_loop().create_future()
        )
        self._closed: AgentCloseResult | None = None

    async def submit(self) -> AgentSubmission:
        if self.submission is not None:
            raise ProtocolDefect("Antigravity turn was already submitted")
        if self._revoked or self._session.writer.is_closing():
            self.submission = AgentNotSubmitted(
                self.attempt, "Antigravity was closed before sending"
            )
            return self.submission
        start = {"attempt": asdict(self.attempt), "request": self.submitted_request}
        self.submission = AgentUncertain(
            self.attempt, self._turn, "Antigravity start entered transport"
        )
        self._driver = asyncio.create_task(self._read_events())
        try:
            _send(self._session.writer, "start", **start)
            await self._session.writer.drain()
        except OSError:
            return self.submission
        return await asyncio.shield(self._submitted)

    async def _read_events(self) -> None:
        try:
            kind, payload = await _receive(self._session.reader)
            if kind != "submission":
                raise ProtocolDefect("Antigravity omitted its submission receipt")
            receipt = submission_from_json(payload)
            if receipt.attempt != self.attempt or (
                isinstance(receipt, AgentAccepted | AgentUncertain) and receipt.turn != self._turn
            ):
                raise ProtocolDefect("Antigravity submission changed its prepared identity")
            self.submission = receipt
            self._submitted.set_result(receipt)
            count = 0
            while True:
                kind, payload = await _receive(self._session.reader)
                if kind == "closed":
                    self._closed = AgentCloseResult(
                        payload["local_closed"], tuple(payload["diagnostics"])
                    )
                    if self._closed.local_closed:
                        self._session.lease._joined.set()
                    return
                if not isinstance(receipt, AgentAccepted):
                    raise ProtocolDefect("Antigravity produced an event without accepted input")
                count += 1
                if count > _MAX_EVENT_COUNT + 1:
                    raise ProtocolDefect("Antigravity host exceeded the event bound")
                event = _decode_event(kind, payload, self._turn)
                if isinstance(event, AgentTerminal):
                    if self.terminal is not None:
                        raise ProtocolDefect("Antigravity produced multiple terminal events")
                    self.terminal = event
                elif self.terminal is not None:
                    raise ProtocolDefect("Antigravity produced an event after settlement")
                self._queue.put_nowait(event)
        except (EOFError, ValueError, TypeError, OSError, ProtocolDefect, KeyError):
            self._queue.put_nowait(ProtocolDefect("Antigravity host stream was lost"))
            self._session.writer.close()
            if not self._submitted.done() and self.submission is not None:
                self._submitted.set_result(self.submission)
        finally:
            self._queue.put_nowait(None)

    async def check_submission(self, *, for_effect: bool = False) -> None:
        if not isinstance(self.submission, AgentAccepted) or (
            for_effect
            and (self._revoked or self._session.writer.is_closing() or self.terminal is not None)
        ):
            raise SessionUnavailable("Antigravity has no active accepted turn")

    async def events(self) -> AsyncIterator[AgentEvent]:
        if self._stream_consumed:
            raise ProtocolDefect("Antigravity events already have a consumer")
        self._stream_consumed = True
        while (item := await self._queue.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def reply(self, call: AgentToolCall, result: AgentToolReply) -> None:
        await self.check_submission(for_effect=True)
        if call.turn != self._turn:
            raise ProtocolDefect("Antigravity tool reply changed its turn")
        _send(
            self._session.writer,
            "reply",
            call_id=call.call_id,
            token=call.reply_token,
            name=call.name,
            arguments=call.arguments,
            text=result.text,
            success=result.success,
        )
        await self._session.writer.drain()

    async def steer(self, *, input_id: str, input: tuple[ContentPart, ...]) -> AgentControlReceipt:
        return AgentControlReceipt(
            "steer",
            str(uuid4()),
            self._turn,
            "not_sent",
            input_id,
            "Antigravity handles follow-ups in a fresh turn",
        )

    def revoke(self) -> None:
        if self._revoked:
            return
        self._revoked = True
        if self._driver is not None and not self._session.writer.is_closing():
            with suppress(OSError):
                _send(self._session.writer, "stop")

    async def interrupt(self) -> AgentControlReceipt:
        self.revoke()
        return AgentControlReceipt(
            "interrupt",
            str(uuid4()),
            self._turn,
            "not_sent" if self._driver is None else "unknown",
            None,
            "Antigravity stop settles through the host stream",
        )

    async def close(self) -> AgentCloseResult:
        self.revoke()
        try:
            if self._driver is None:
                return AgentCloseResult(True, ())
            with suppress(TimeoutError):
                async with asyncio.timeout(self._controls.rpc_seconds):
                    await asyncio.shield(self._driver)
            return self._closed or AgentCloseResult(
                False, ("Antigravity host did not confirm closure",)
            )
        finally:
            await _disconnect(self._session.writer)
            if self._driver is not None and not self._driver.done():
                self._driver.cancel()
                await asyncio.gather(self._driver, return_exceptions=True)
            self._release()


def _encode_event(writer: asyncio.StreamWriter, event: AgentEvent, turn: AgentTurnRef) -> None:
    if isinstance(event, AgentTerminal):
        _send(writer, "terminal", **terminal_to_json(event))
    elif isinstance(event, AgentInputRecorded) and event.turn == turn:
        _send(writer, "input", input_id=event.input_id, item_id=event.item_id)
    elif isinstance(event, AgentToolCall) and event.turn == turn:
        _send(
            writer,
            "call",
            call_id=event.call_id,
            token=event.reply_token,
            name=event.name,
            arguments=event.arguments,
        )
    elif isinstance(event, AgentMessage) and event.turn == turn:
        _send(writer, "message", message_id=event.message_id, phase=event.phase, text=event.text)
    elif isinstance(event, AgentToolUse):
        _send(writer, "tool", **asdict(event))
    else:
        raise ProtocolDefect("Antigravity host emitted an unsupported event")


def _decode_event(kind: str, value: dict[str, Any], turn: AgentTurnRef) -> AgentEvent:
    match kind:
        case "terminal":
            return terminal_from_json(value)
        case "input":
            return AgentInputRecorded(turn, value["input_id"], value["item_id"])
        case "call":
            return AgentToolCall(
                turn,
                value["call_id"],
                value["token"],
                value["name"],
                freeze_json_value(value["arguments"]),
            )
        case "message":
            return AgentMessage(turn, value["message_id"], value["phase"], value["text"])
        case "tool":
            stock = value["stock"]
            if stock is not None:
                stock["reports"] = tuple(tuple(pair) for pair in stock["reports"])
            return AgentToolUse(
                value["tool_call_id"],
                value["name"],
                value["phase"],
                freeze_json_value(value["payload"]),
                value["succeeded"],
                None if stock is None else StockToolObservation(**stock),
            )
        case _:
            raise ProtocolDefect("Antigravity host sent an unsupported event")


async def serve_antigravity(
    path: Path, *, state_root: Path, endpoint_url: str
) -> asyncio.AbstractServer:
    """Listen under the dedicated host user. Socket access is set by deployment."""

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await _serve_session(reader, writer, path, state_root, endpoint_url)
        except (
            OSError,
            EOFError,
            TimeoutError,
            ValueError,
            TypeError,
            KeyError,
            AgentRuntimeError,
            ProtocolDefect,
        ):
            # A broken channel conveys no submission or teardown proof.
            pass
        finally:
            await _disconnect(writer)

    server = await asyncio.start_unix_server(serve, path, limit=_FRAME_BYTES)
    path.chmod(0o660)
    return server


async def _serve_session(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    path: Path,
    state_root: Path,
    endpoint_url: str,
) -> None:
    async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
        kind, opened = await _receive(reader)
        if opened["protocol"] != _PROTOCOL:
            raise ProtocolDefect("Antigravity host protocol mismatch")
        if kind == "ping":
            _send(writer, "ready", protocol=_PROTOCOL)
            await writer.drain()
            return
        if kind != "open":
            raise ProtocolDefect("Antigravity host expected an open request")
        if opened["revision"] != AGY_CAPABILITY_REVISION:
            raise ProtocolDefect("Antigravity host capability revision mismatch")
        cwd = Path(opened["cwd"])
        if not cwd.is_absolute() or not cwd.is_dir():
            raise InvalidAgentRequest("Antigravity working directory was unavailable")
        ref = AgentSessionRef(
            "agent-session-ref.v1",
            "antigravity",
            "antigravity_sdk",
            str(uuid4()),
            AGY_PROFILE,
            fingerprint_path(path),
            fingerprint_path(cwd),
        )
        turn_ref = AgentTurnRef(ref, str(uuid4()))
        bearer = token_hex(32)
        _send(
            writer,
            "opened",
            root=ref.native_session_id,
            turn=turn_ref.native_turn_id,
            bearer=bearer,
        )
        await writer.drain()

    kind, started = await _receive(reader)
    if kind != "start":
        raise ProtocolDefect("Antigravity host expected a prepared turn")
    from .antigravity_sdk import AntigravitySdkOptions, AntigravitySdkTurn

    wire: dict[str, Any] = started["request"]
    submitted = freeze_json_object(wire)
    attempt = AgentAttempt(**started["attempt"])
    if (
        submitted["type"] != _PROTOCOL
        or submitted["binding"]
        != sha256(canonical_json_bytes(freeze_json_object(opened))).hexdigest()
        or submitted["root"] != ref.native_session_id
        or submitted["turn"] != turn_ref.native_turn_id
        or sha256(canonical_json_bytes(submitted)).hexdigest() != attempt.request_digest
    ):
        raise ProtocolDefect("Antigravity host start changed the prepared request")
    output = opened["output"]
    specification = (
        TextAgentOutput()
        if output["kind"] == "text"
        else JsonSchemaAgentOutput(output["name"], output["schema"])
    )
    turn = AntigravitySdkTurn(
        turn=turn_ref,
        attempt=attempt,
        submitted_request=submitted,
        input_id=wire["input_id"],
        request=TurnRequest(input=(TextContent(wire["text"]),), timeout_seconds=wire["timeout"]),
        controls=AgentTurnControls(**wire["controls"]),
        options=AntigravitySdkOptions(
            endpoint_url,
            bearer,
            cwd,
            state_root,
            opened["system"],
            opened["workspace_tools"],
            specification,
        ),
        tools=tuple(CanonicalTool(**tool) for tool in opened["tools"]),
    )

    async def stream() -> None:
        submission = await turn.submit()
        _send(writer, "submission", **submission_to_json(submission))
        await writer.drain()
        if isinstance(submission, AgentAccepted):
            async for event in turn.events():
                _encode_event(writer, event, turn_ref)
                await writer.drain()

    async def controls() -> None:
        while True:
            operation, value = await _receive(reader)
            match operation:
                case "stop":
                    turn.revoke()
                    await turn.interrupt()
                case "reply":
                    await turn.reply(
                        AgentToolCall(
                            turn_ref,
                            value["call_id"],
                            value["token"],
                            value["name"],
                            freeze_json_value(value["arguments"]),
                        ),
                        AgentToolReply(value["text"], value["success"]),
                    )
                case _:
                    raise ProtocolDefect("Antigravity host control was unsupported")

    tasks = {asyncio.create_task(stream()), asyncio.create_task(controls())}
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        turn.revoke()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        result = await turn.close()
    _send(writer, "closed", **asdict(result))
    await writer.drain()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=_origin)
    options = parser.parse_args()
    if not options.socket.is_absolute() or not options.state_root.is_absolute():
        parser.error("socket and state root must be absolute")
    if not options.state_root.is_dir() or version("google-antigravity") != AGY_SDK_VERSION:
        parser.error("contained state or pinned SDK is unavailable")

    async def run() -> None:
        server = await serve_antigravity(
            options.socket, state_root=options.state_root, endpoint_url=options.upstream
        )
        async with server:
            await server.serve_forever()

    asyncio.run(run())
