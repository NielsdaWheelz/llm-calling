"""Typed, profile-scoped controls for externally owned Codex threads."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from .codex_app_server import (
    CODEX_THREAD_SOURCES,
    CodexAppServerClient,
    CodexAppServerConfig,
    CodexAppServerResponseError,
    CodexConnectionUnavailable,
)
from .errors import AgentRuntimeError, InvalidAgentRequest, ProtocolDefect
from .types import CredentialRef

type CodexDispatch = Literal["NotSent", "Rejected", "Unknown"]
type CodexErrorCode = Literal[
    "invalid",
    "unauthorized",
    "missing",
    "busy",
    "stale",
    "unavailable",
    "auth",
    "quota",
    "output_limit",
]
type CodexNativeStatus = Literal["notLoaded", "idle", "active", "systemError"]
type CodexTurnStatus = Literal["inProgress", "completed", "interrupted", "failed"]


def _profile(value: str) -> None:
    CredentialRef(kind="local_account", profile_key=value)


def _handle(value: str) -> None:
    if type(value) is not str:
        raise InvalidAgentRequest("Codex handles must be full canonical UUID strings")
    try:
        valid = str(UUID(value)) == value
    except ValueError:
        valid = False
    if not valid:
        raise InvalidAgentRequest("Codex handles must be full canonical UUID strings")


def _text(value: str) -> None:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 32 * 1024:
        raise InvalidAgentRequest("Codex input must contain 1–32768 UTF-8 bytes")


@dataclass(frozen=True, slots=True)
class CodexThreadTarget:
    profile_key: str
    thread_handle: str

    def __post_init__(self) -> None:
        _profile(self.profile_key)
        _handle(self.thread_handle)


@dataclass(frozen=True, slots=True)
class CodexTurnTarget:
    thread: CodexThreadTarget
    turn_handle: str

    def __post_init__(self) -> None:
        if not isinstance(self.thread, CodexThreadTarget):
            raise InvalidAgentRequest("Codex turn requires a thread target")
        _handle(self.turn_handle)


@dataclass(frozen=True, slots=True)
class CodexListRequest:
    profile_key: str
    cursor: str | None = None

    def __post_init__(self) -> None:
        _profile(self.profile_key)
        if self.cursor is not None and (
            type(self.cursor) is not str
            or not self.cursor
            or len(self.cursor.encode("utf-8")) > 4096
        ):
            raise InvalidAgentRequest("Codex cursor must contain 1–4096 UTF-8 bytes")


@dataclass(frozen=True, slots=True)
class CodexCreateRequest:
    profile_key: str
    cwd: Path

    def __post_init__(self) -> None:
        _profile(self.profile_key)
        if (
            not isinstance(self.cwd, Path)
            or not self.cwd.is_absolute()
            or os.path.normpath(str(self.cwd)) != str(self.cwd)
            or len(str(self.cwd).encode("utf-8")) > 4096
        ):
            raise InvalidAgentRequest("Codex cwd must be a normalized absolute bounded path")


@dataclass(frozen=True, slots=True)
class CodexSubmit:
    text: str
    tag: Literal["Submit"] = field(default="Submit", init=False)

    def __post_init__(self) -> None:
        _text(self.text)


@dataclass(frozen=True, slots=True)
class CodexSteer:
    turn_handle: str
    text: str
    tag: Literal["Steer"] = field(default="Steer", init=False)

    def __post_init__(self) -> None:
        _handle(self.turn_handle)
        _text(self.text)


@dataclass(frozen=True, slots=True)
class CodexPromptRequest:
    thread: CodexThreadTarget
    input: CodexSubmit | CodexSteer

    def __post_init__(self) -> None:
        if not isinstance(self.thread, CodexThreadTarget) or not isinstance(
            self.input, (CodexSubmit, CodexSteer)
        ):
            raise InvalidAgentRequest("Codex prompt requires a thread and Submit or Steer input")


@dataclass(frozen=True, slots=True)
class CodexThreadSummary:
    target: CodexThreadTarget
    status: CodexNativeStatus
    active_flags: tuple[str, ...]
    name: str | None
    cwd: str
    source: str


@dataclass(frozen=True, slots=True)
class CodexThreadPage:
    threads: tuple[CodexThreadSummary, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class CodexTurnSnapshot:
    target: CodexTurnTarget
    status: CodexTurnStatus


@dataclass(frozen=True, slots=True)
class CodexComplete:
    tag: Literal["Complete"] = field(default="Complete", init=False)


@dataclass(frozen=True, slots=True)
class CodexBounded:
    reason: str
    tag: Literal["Bounded"] = field(default="Bounded", init=False)


@dataclass(frozen=True, slots=True)
class CodexThreadRead:
    thread: CodexThreadSummary
    turn: CodexTurnSnapshot | None
    last_answer: str | None
    coverage: CodexComplete | CodexBounded


@dataclass(frozen=True, slots=True)
class CodexInterrupted:
    tag: Literal["Interrupted"] = field(default="Interrupted", init=False)


@dataclass(frozen=True, slots=True)
class CodexFinished:
    native_status: Literal["completed", "failed"]
    tag: Literal["Finished"] = field(default="Finished", init=False)


@dataclass(frozen=True, slots=True)
class CodexStale:
    tag: Literal["Stale"] = field(default="Stale", init=False)


@dataclass(frozen=True, slots=True)
class CodexUnknown:
    tag: Literal["Unknown"] = field(default="Unknown", init=False)


type CodexInterruptOutcome = CodexInterrupted | CodexFinished | CodexStale | CodexUnknown


class CodexControlError(AgentRuntimeError):
    """A content-free expected failure; ambiguous dispatch is never a retry signal."""

    def __init__(
        self,
        code: CodexErrorCode,
        dispatch: CodexDispatch,
        known_thread: CodexThreadTarget | None = None,
    ) -> None:
        super().__init__(f"Codex control {code}", code=code)
        self.dispatch = dispatch
        self.known_thread = known_thread


class CodexControl:
    """One native control boundary; it owns connections, never servers or workers."""

    def __init__(
        self,
        endpoints: Mapping[str, Path],
        is_managed: Callable[[CodexThreadTarget], bool],
    ) -> None:
        self._endpoints = endpoints
        self._is_managed = is_managed
        self._clients: set[CodexAppServerClient] = set()
        self._closed = False

    async def list(self, request: CodexListRequest) -> CodexThreadPage:
        async with self._client(request.profile_key) as client:
            result = _object(
                await self._request(
                    client,
                    "thread/list",
                    {
                        "cursor": request.cursor,
                        "limit": 50,
                        "sourceKinds": list(CODEX_THREAD_SOURCES),
                    },
                )
            )
        data = result.get("data")
        if not isinstance(data, list) or len(data) > 50:
            raise ProtocolDefect("Codex thread list exceeded its page contract")
        cursor = result.get("nextCursor")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise ProtocolDefect("Codex thread list returned a malformed cursor")
        page = CodexThreadPage(
            tuple(_summary(request.profile_key, _object(item)) for item in data), cursor
        )
        if _output_size(page) > 60 * 1024:
            raise CodexControlError("output_limit", "NotSent")
        return page

    async def read(self, target: CodexThreadTarget) -> CodexThreadRead:
        async with self._client(target.profile_key) as client:
            return await self._read(client, target)

    async def create(self, request: CodexCreateRequest) -> CodexThreadTarget:
        try:
            cwd = request.cwd.resolve(strict=True)
        except OSError:
            raise CodexControlError("invalid", "NotSent") from None
        if cwd != request.cwd or not cwd.is_dir():
            raise CodexControlError("invalid", "NotSent")
        async with self._client(request.profile_key) as client:
            result = _object(
                await self._request(
                    client,
                    "thread/start",
                    {
                        "cwd": str(cwd),
                        "sandbox": "workspace-write",
                        "approvalPolicy": "on-request",
                        "approvalsReviewer": "user",
                        "config": {"sandbox_workspace_write": {"network_access": False}},
                    },
                    mutation=True,
                )
            )
            summary = _summary(request.profile_key, _object(result.get("thread")))
            target = summary.target
            result = _object(
                await self._request(
                    client,
                    "thread/unsubscribe",
                    {
                        "threadId": target.thread_handle,
                    },
                    mutation=True,
                    known_thread=target,
                )
            )
            if result.get("status") not in ("notLoaded", "notSubscribed", "unsubscribed"):
                raise ProtocolDefect("Codex unsubscribe returned an unknown status")
            return target

    async def prompt(self, request: CodexPromptRequest) -> CodexTurnTarget:
        self._guard(request.thread)
        params: dict[str, object] = {
            "threadId": request.thread.thread_handle,
            "input": [{"type": "text", "text": request.input.text}],
        }
        if isinstance(request.input, CodexSteer):
            params["expectedTurnId"] = request.input.turn_handle
            method = "turn/steer"
        else:
            method = "turn/start"
        async with self._client(request.thread.profile_key) as client:
            self._guard(request.thread)
            result = _object(
                await self._request(
                    client, method, params, mutation=True, known_thread=request.thread
                )
            )
        handle = (
            result.get("turnId")
            if isinstance(request.input, CodexSteer)
            else _object(result.get("turn")).get("id")
        )
        target = _turn_target(request.thread, handle)
        if (
            isinstance(request.input, CodexSteer)
            and target.turn_handle != request.input.turn_handle
        ):
            raise ProtocolDefect("Codex steer changed the expected turn identity")
        return target

    async def interrupt(self, target: CodexTurnTarget) -> CodexInterruptOutcome:
        self._guard(target.thread)
        async with self._client(target.thread.profile_key) as client:
            observed = await self._read(client, target.thread)
            settled = _interrupt_outcome(target, observed)
            if settled is not None:
                return settled
            self._guard(target.thread)
            try:
                await self._request(
                    client,
                    "turn/interrupt",
                    {
                        "threadId": target.thread.thread_handle,
                        "turnId": target.turn_handle,
                    },
                    mutation=True,
                    known_thread=target.thread,
                )
            except CodexControlError as error:
                if error.dispatch == "Unknown":
                    return CodexUnknown()
                if error.code != "stale":
                    raise
            try:
                observed = await self._read(client, target.thread)
            except CodexControlError:
                return CodexUnknown()
            return _interrupt_outcome(target, observed) or CodexUnknown()

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(*(client.close() for client in tuple(self._clients)))

    def _guard(self, target: CodexThreadTarget) -> None:
        if self._is_managed(target):
            raise CodexControlError("unauthorized", "NotSent", target)

    @asynccontextmanager
    async def _client(self, profile: str) -> AsyncIterator[CodexAppServerClient]:
        if self._closed:
            raise CodexControlError("unavailable", "NotSent")
        endpoint = self._endpoints.get(profile)
        if endpoint is None:
            raise CodexControlError("unavailable", "NotSent")
        client = CodexAppServerClient(CodexAppServerConfig(endpoint, request_policy="observe_only"))
        self._clients.add(client)
        try:
            try:
                await client.__aenter__()
            except CodexConnectionUnavailable:
                raise CodexControlError("unavailable", "NotSent") from None
            account = _object(await self._request(client, "account/read", {"refreshToken": False}))
            if (
                account.get("account") is None
                or _object(account["account"]).get("type") != "chatgpt"
            ):
                raise CodexControlError("auth", "NotSent")
            yield client
        finally:
            self._clients.discard(client)
            await client.close()

    async def _request(
        self,
        client: CodexAppServerClient,
        method: str,
        params: dict[str, object],
        *,
        mutation: bool = False,
        known_thread: CodexThreadTarget | None = None,
    ) -> object:
        if len(json.dumps(params, ensure_ascii=False).encode("utf-8")) > 64 * 1024:
            raise CodexControlError("invalid", "NotSent", known_thread)
        try:
            return await client.request(method, params)
        except CodexConnectionUnavailable:
            raise CodexControlError(
                "unavailable", "Unknown" if mutation else "NotSent", known_thread
            ) from None
        except CodexAppServerResponseError as error:
            raise CodexControlError(
                _error_code(error), "Rejected" if mutation else "NotSent", known_thread
            ) from None

    async def _read(
        self, client: CodexAppServerClient, target: CodexThreadTarget
    ) -> CodexThreadRead:
        result = _object(
            await self._request(
                client,
                "thread/read",
                {
                    "threadId": target.thread_handle,
                    "includeTurns": False,
                },
            )
        )
        summary = _summary(target.profile_key, _object(result.get("thread")))
        if summary.target != target:
            raise ProtocolDefect("Codex read changed the requested thread identity")
        metadata_bounded = _output_size(summary) > 8 * 1024
        if metadata_bounded:
            summary = replace(summary, name=None)
            if _output_size(summary) > 8 * 1024:
                raise CodexControlError("output_limit", "NotSent")
        result = _object(
            await self._request(
                client,
                "thread/turns/list",
                {
                    "threadId": target.thread_handle,
                    "limit": 1,
                    "sortDirection": "desc",
                    "itemsView": "full",
                },
            )
        )
        data = result.get("data")
        if not isinstance(data, list) or len(data) > 1:
            raise ProtocolDefect("Codex turn list exceeded its page contract")
        if not data:
            return CodexThreadRead(
                summary,
                None,
                None,
                CodexBounded("metadata exceeds output bound")
                if metadata_bounded
                else CodexComplete(),
            )
        turn = _object(data[0])
        status = turn.get("status")
        if status not in ("inProgress", "completed", "interrupted", "failed"):
            raise ProtocolDefect("Codex turn has an unknown status")
        snapshot = CodexTurnSnapshot(
            _turn_target(target, turn.get("id")), cast(CodexTurnStatus, status)
        )
        coverage: CodexComplete | CodexBounded = (
            CodexBounded("latest turn only") if result.get("nextCursor") else CodexComplete()
        )
        if metadata_bounded:
            coverage = CodexBounded("metadata exceeds output bound")
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 48 * 1024:
            return CodexThreadRead(
                summary, snapshot, None, CodexBounded("turn exceeds output bound")
            )
        if turn.get("itemsView") != "full":
            return CodexThreadRead(
                summary, snapshot, None, CodexBounded("native items are not complete")
            )
        items = turn.get("items")
        if not isinstance(items, list):
            raise ProtocolDefect("Codex turn items were not an array")
        answer = None
        unknown_phase = None
        for item in items:
            item = _object(item)
            if item.get("type") == "agentMessage":
                text = item.get("text")
                if not isinstance(text, str):
                    raise ProtocolDefect("Codex agent message has no text")
                if item.get("phase") == "final_answer":
                    answer = text
                elif item.get("phase") is None:
                    unknown_phase = text
        return CodexThreadRead(
            summary, snapshot, answer if answer is not None else unknown_phase, coverage
        )


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ProtocolDefect("Codex control response was not an object")
    return value


def _summary(profile: str, value: dict[str, object]) -> CodexThreadSummary:
    handle = value.get("id")
    try:
        target = CodexThreadTarget(profile, cast(str, handle))
    except InvalidAgentRequest:
        raise ProtocolDefect("Codex returned a malformed thread identity") from None
    status = _object(value.get("status"))
    kind = status.get("type")
    if kind not in ("notLoaded", "idle", "active", "systemError"):
        raise ProtocolDefect("Codex thread has an unknown status")
    flags = status.get("activeFlags") if kind == "active" else []
    if not isinstance(flags, list) or any(
        flag not in ("waitingOnApproval", "waitingOnUserInput") for flag in flags
    ):
        raise ProtocolDefect("Codex thread active flags are malformed")
    name, cwd, source = value.get("name"), value.get("cwd"), value.get("source")
    if name is not None and not isinstance(name, str):
        raise ProtocolDefect("Codex thread name is malformed")
    if not isinstance(cwd, str) or not os.path.isabs(cwd):
        raise ProtocolDefect("Codex thread cwd is malformed")
    if isinstance(source, dict) and "subAgent" in source:
        source = "subAgent"
    if not isinstance(source, str) or source not in CODEX_THREAD_SOURCES:
        raise ProtocolDefect("Codex thread source is malformed")
    return CodexThreadSummary(
        target, cast(CodexNativeStatus, kind), tuple(flags), name, cwd, source
    )


def _output_size(value: CodexThreadSummary | CodexThreadPage) -> int:
    return len(json.dumps(asdict(value), ensure_ascii=False).encode("utf-8"))


def _turn_target(thread: CodexThreadTarget, handle: object) -> CodexTurnTarget:
    try:
        return CodexTurnTarget(thread, cast(str, handle))
    except InvalidAgentRequest:
        raise ProtocolDefect("Codex returned a malformed turn identity") from None


def _interrupt_outcome(
    target: CodexTurnTarget, observed: CodexThreadRead
) -> CodexInterruptOutcome | None:
    turn = observed.turn
    if turn is None:
        return (
            CodexUnknown() if observed.thread.status in ("active", "systemError") else CodexStale()
        )
    if turn.target != target:
        return CodexStale()
    match turn.status:
        case "interrupted":
            return CodexInterrupted()
        case "completed" | "failed":
            return CodexFinished(turn.status)
        case "inProgress":
            return None


def _error_code(error: CodexAppServerResponseError) -> CodexErrorCode:
    info = error.data.get("codexErrorInfo") if isinstance(error.data, dict) else None
    if info in ("usageLimitExceeded", "sessionBudgetExceeded"):
        return "quota"
    if isinstance(info, dict) and "activeTurnNotSteerable" in info:
        return "busy"
    detail = error.detail.lower()
    if "expected active turn" in detail or "no active turn" in detail:
        return "stale"
    if "not found" in detail or "not loaded" in detail:
        return "missing"
    if "cannot steer" in detail or "active turn" in detail:
        return "busy"
    if "quota" in detail or "usage limit" in detail:
        return "quota"
    if "unauthorized" in detail or "permission denied" in detail:
        return "unauthorized"
    if "auth" in detail or "login" in detail:
        return "auth"
    if error.code in (-32600, -32602):
        return "invalid"
    if error.code == -32603:
        return "unavailable"
    raise ProtocolDefect("Codex returned an unclassified protocol error")
