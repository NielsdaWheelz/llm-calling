"""Typed, profile-scoped controls for externally owned Codex threads."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, assert_never, cast
from uuid import UUID, uuid4

from .codex_app_server import (
    CODEX_THREAD_SOURCES,
    CodexAppServerClient,
    CodexAppServerConfig,
    CodexAppServerResponseError,
    CodexConnectionUnavailable,
    CodexOutputLimit,
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
    "history_changed",
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
    can_accept_direct_input: bool | None = None


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
    history_available: bool = True


@dataclass(frozen=True, slots=True)
class CodexView:
    view_id: str
    revision: int

    def __post_init__(self) -> None:
        _handle(self.view_id)
        if type(self.revision) is not int or not 0 <= self.revision < 1 << 64:
            raise InvalidAgentRequest("Codex view revision must be a uint64")


@dataclass(frozen=True, slots=True)
class CodexSelectedTarget:
    profile_key: str
    pid: int
    start_identity: str
    thread: CodexThreadTarget | None = None
    view: CodexView | None = None

    def __post_init__(self) -> None:
        _profile(self.profile_key)
        if type(self.pid) is not int or self.pid <= 0:
            raise InvalidAgentRequest("Codex process pid must be positive")
        if (
            type(self.start_identity) is not str
            or not self.start_identity.isdecimal()
            or int(self.start_identity) <= 0
        ):
            raise InvalidAgentRequest("Codex process start identity must be positive")
        if self.thread is not None and self.thread.profile_key != self.profile_key:
            raise InvalidAgentRequest("Codex selected thread must use the process profile")


@dataclass(frozen=True, slots=True)
class CodexSelectedObservation:
    view: CodexView
    read: CodexThreadRead


@dataclass(frozen=True, slots=True)
class CodexOutput:
    text: str
    scope: Literal["latest", "history"]
    truncated: bool
    state: Literal["partial", "finalized", "unknown", "none"]
    output_id: str | None
    turn_id: str | None


@dataclass(frozen=True, slots=True)
class CodexSelectedRead:
    inspection: CodexSelectedObservation
    output: CodexOutput


@dataclass(frozen=True, slots=True)
class CodexResultPage:
    result_ids: tuple[str, ...]
    next_cursor: str | None


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


class _ItemPagingUnavailable(Exception):
    """Pinned native history cannot provide an item page for this thread."""


class _HistoryNotMaterialized(CodexControlError):
    def __init__(self) -> None:
        super().__init__("unavailable", "NotSent")


class CodexControl:
    """One native control boundary; it owns connections, never servers or workers."""

    def __init__(
        self,
        endpoints: Mapping[str, Path],
        is_managed: Callable[[CodexThreadTarget], bool],
        *,
        native_owners: bool = False,
    ) -> None:
        self._endpoints = endpoints
        self._is_managed = is_managed
        self._native_owners = native_owners
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
            return await self._read(client, target, include_items=True)

    async def inspect(
        self, targets: tuple[CodexThreadTarget, ...]
    ) -> tuple[CodexThreadRead | CodexControlError, ...]:
        """Read one profile's requested states without loading conversation items."""
        if not targets or any(target.profile_key != targets[0].profile_key for target in targets):
            raise InvalidAgentRequest("Codex inspection requires targets from one profile")
        observed: list[CodexThreadRead | CodexControlError] = []
        async with self._client(targets[0].profile_key) as client:
            for target in targets:
                try:
                    observed.append(await self._read(client, target, include_items=False))
                except CodexControlError as error:
                    observed.append(error)
        return tuple(observed)

    async def inspect_selected(
        self, targets: tuple[CodexSelectedTarget, ...]
    ) -> tuple[CodexSelectedObservation | CodexControlError, ...]:
        if not targets or any(target.profile_key != targets[0].profile_key for target in targets):
            raise InvalidAgentRequest("Codex inspection requires targets from one profile")
        observed: list[CodexSelectedObservation | CodexControlError] = []
        async with self._client(targets[0].profile_key, experimental_api=True) as client:
            for target in targets:
                try:
                    view, thread = await self._selected(client, target)
                    read = await self._read(client, thread, include_items=False)
                    await self._selected(client, replace(target, thread=thread, view=view))
                    observed.append(CodexSelectedObservation(view, read))
                except CodexControlError as error:
                    observed.append(error)
        return tuple(observed)

    async def read_selected(
        self, target: CodexSelectedTarget, scope: Literal["latest", "history"], max_bytes: int
    ) -> CodexSelectedRead:
        async with self._client(target.profile_key, experimental_api=True) as client:
            view, thread = await self._selected(client, target)
            read = await self._read(client, thread, include_items=False)
            output = await self._history(client, thread, scope, max_bytes)
            await self._selected(client, replace(target, thread=thread, view=view))
            return CodexSelectedRead(CodexSelectedObservation(view, read), output)

    async def send_selected(
        self,
        target: CodexSelectedTarget,
        text: str,
        input_kind: Literal["peer", "user"],
        delivery: Literal["direct", "queue"],
    ) -> dict[str, str]:
        _text(text)
        if input_kind == "peer" and delivery == "queue":
            raise InvalidAgentRequest("Codex peer input cannot be queued")
        async with self._client(target.profile_key, experimental_api=True) as client:
            view, thread = await self._selected(client, target)
            self._guard(thread)
            observed = await self._read(client, thread, include_items=False)
            if (
                observed.thread.status not in ("active", "idle")
                or observed.thread.can_accept_direct_input is not True
            ):
                raise CodexControlError("unavailable", "NotSent")
            params: dict[str, object] = {
                "threadId": thread.thread_handle,
                "expectedView": {"viewId": view.view_id, "revision": view.revision},
                "input": [{"type": "text", "text": text}] if input_kind == "user" else [],
            }
            if delivery == "queue":
                params["clientUserMessageId"] = str(uuid4())
                method = "thread/queue/add"
            else:
                method = "turn/start"
                if input_kind == "peer":
                    params["toolOutput"] = {
                        "name": "send_message",
                        "namespace": "skid",
                        "output": text,
                    }
            result = _object(
                await self._request(client, method, params, mutation=True, known_thread=thread)
            )
        receipt = {
            "method": "native",
            "input": input_kind,
            "delivery": delivery,
            "outcome": "accepted",
        }
        if delivery == "queue":
            queued = _object(result.get("queuedSubmission"))
            queue_id, message_id = queued.get("id"), queued.get("clientUserMessageId")
            if (
                not isinstance(queue_id, str)
                or not queue_id
                or message_id != params["clientUserMessageId"]
            ):
                raise CodexControlError("unavailable", "Unknown", thread)
            receipt.update(queueItemId=queue_id, clientMessageId=cast(str, message_id))
        else:
            receipt["turnId"] = _turn_target(
                thread, _object(result.get("turn")).get("id")
            ).turn_handle
        return receipt

    async def interrupt_selected(
        self, target: CodexSelectedTarget, turn_handle: str | None
    ) -> dict[str, str]:
        async with self._client(target.profile_key, experimental_api=True) as client:
            view, thread = await self._selected(client, target)
            self._guard(thread)
            observed = await self._read(client, thread, include_items=False)
            if observed.thread.status == "notLoaded":
                raise CodexControlError("unavailable", "NotSent")
            if observed.thread.status == "idle":
                await self._selected(client, replace(target, thread=thread, view=view))
                return {"method": "native", "outcome": "finished"}
            if observed.turn is None or observed.turn.status != "inProgress":
                raise CodexControlError("unavailable", "NotSent")
            if observed.turn.target.turn_handle != turn_handle:
                raise CodexControlError("stale", "NotSent")
            turn = observed.turn.target
            try:
                await self._request(
                    client,
                    "turn/interrupt",
                    {
                        "threadId": thread.thread_handle,
                        "turnId": turn.turn_handle,
                        "expectedView": {"viewId": view.view_id, "revision": view.revision},
                    },
                    mutation=True,
                    known_thread=thread,
                )
                after = await self._read(client, thread, include_items=False)
            except CodexControlError as error:
                if error.dispatch != "Unknown":
                    raise
                return {"method": "native", "outcome": "unknown"}
            settled = _interrupt_outcome(turn, after)
            outcome = (
                "interrupted"
                if isinstance(settled, CodexInterrupted)
                else "finished"
                if isinstance(settled, CodexFinished)
                else "unknown"
            )
            return {"method": "native", "outcome": outcome, "turnId": turn.turn_handle}

    async def results(self, target: CodexThreadTarget, cursor: str | None) -> CodexResultPage:
        try:
            async with self._client(target.profile_key, experimental_api=True) as client:
                result = await self._turn_page(client, target, cursor, "asc", 128, "full")
        except CodexControlError as error:
            if error.code == "missing":
                raise CodexControlError(
                    "history_changed" if cursor else "unavailable", "NotSent"
                ) from None
            raise
        if cursor is not None and not result["data"]:
            raise CodexControlError("history_changed", "NotSent")
        ids: list[str] = []
        for value in cast(list[object], result["data"]):
            turn = _object(value)
            if turn.get("itemsView") != "full":
                raise CodexControlError("unavailable", "NotSent")
            items = turn.get("items")
            if not isinstance(items, list):
                raise ProtocolDefect("Codex turn items were malformed")
            if _finalized(turn) and any(
                _object(item).get("type") == "agentMessage"
                and _object(item).get("phase") == "final_answer"
                and isinstance(_object(item).get("text"), str)
                and bool(_object(item).get("text"))
                for item in items
            ):
                handle = cast(str, turn["id"])
                if handle in ids:
                    raise ProtocolDefect("Codex result page repeated a turn identity")
                ids.append(handle)
        return CodexResultPage(tuple(ids), cast(str | None, result.get("nextCursor")))

    async def _selected(
        self, client: CodexAppServerClient, target: CodexSelectedTarget
    ) -> tuple[CodexView, CodexThreadTarget]:
        result = _object(
            await self._request(
                client,
                "tui/view/read",
                {"process": {"pid": target.pid, "startIdentity": target.start_identity}},
            )
        )
        if result.get("view") is None:
            raise CodexControlError(
                "stale" if target.view is not None else "unavailable", "NotSent"
            )
        value = _object(result["view"])
        try:
            view = CodexView(cast(str, value.get("viewId")), cast(int, value.get("revision")))
            if target.view is not None and target.view != view:
                raise CodexControlError("stale", "NotSent")
            if value.get("threadId") is None:
                raise CodexControlError("unavailable", "NotSent")
            thread = CodexThreadTarget(target.profile_key, cast(str, value["threadId"]))
        except InvalidAgentRequest:
            raise ProtocolDefect("Codex selected view was malformed") from None
        if target.thread is not None and target.thread != thread:
            raise CodexControlError("stale", "NotSent")
        return view, thread

    async def _turn_page(
        self,
        client: CodexAppServerClient,
        target: CodexThreadTarget,
        cursor: str | None,
        direction: Literal["asc", "desc"],
        limit: int,
        items_view: Literal["full", "notLoaded"],
    ) -> dict[str, object]:
        result = _object(
            await self._request(
                client,
                "thread/turns/list",
                {
                    "threadId": target.thread_handle,
                    "cursor": cursor,
                    "limit": limit,
                    "sortDirection": direction,
                    "itemsView": items_view,
                },
            )
        )
        data = result.get("data")
        if not isinstance(data, list) or len(data) > limit:
            raise ProtocolDefect("Codex turn list exceeded its page contract")
        next_cursor = result.get("nextCursor")
        if next_cursor is not None and (
            not isinstance(next_cursor, str) or not next_cursor or len(next_cursor.encode()) > 4096
        ):
            raise ProtocolDefect("Codex turn cursor was malformed")
        if next_cursor is not None and (not data or next_cursor == cursor):
            raise ProtocolDefect("Codex history page made no progress")
        return result

    async def _history(
        self,
        client: CodexAppServerClient,
        target: CodexThreadTarget,
        scope: Literal["latest", "history"],
        max_bytes: int,
    ) -> CodexOutput:
        cursor = None
        text: list[str] = []
        latest: tuple[str, dict[str, object], dict[str, object]] | None = None
        truncated = False
        while True:
            page = await self._turn_page(client, target, cursor, "desc", 8, "full")
            for raw in cast(list[object], page["data"]):
                turn = _object(raw)
                if turn.get("itemsView") != "full" or not isinstance(turn.get("items"), list):
                    raise CodexControlError("unavailable", "NotSent")
                for entry in reversed(cast(list[object], turn["items"])):
                    item = _object(entry)
                    if item.get("type") not in ("agentMessage", "userMessage"):
                        continue
                    value = item.get("text")
                    if item.get("type") == "userMessage":
                        content = item.get("content")
                        if not isinstance(content, list):
                            raise ProtocolDefect("Codex user message was malformed")
                        value = "\n".join(
                            cast(str, _object(block)["text"])
                            for block in content
                            if _object(block).get("type") == "text"
                            and isinstance(_object(block).get("text"), str)
                        )
                    if not isinstance(value, str):
                        raise ProtocolDefect("Codex message text was malformed")
                    if value and item.get("type") == "agentMessage" and latest is None:
                        latest = value, turn, item
                    if value and scope == "history":
                        text.append(
                            ("assistant: " if item.get("type") == "agentMessage" else "user: ")
                            + value
                        )
                if scope == "latest" and latest is not None:
                    break
            cursor = cast(str | None, page.get("nextCursor"))
            if scope == "latest" and latest is not None:
                break
            if len("\n\n".join(text).encode()) >= max_bytes and latest is not None:
                truncated = cursor is not None
                break
            if cursor is None:
                break
        state: Literal["partial", "finalized", "unknown", "none"] = "none"
        output_id, turn_id = None, None
        if latest is not None:
            value, turn, item = latest
            turn_id = cast(str, turn.get("id"))
            if _finalized(turn) and item.get("phase") == "final_answer":
                state, output_id = "finalized", turn_id
            elif turn.get("status") == "inProgress":
                state = "partial"
            else:
                state = "unknown"
        value = latest[0] if scope == "latest" and latest else "\n\n".join(reversed(text))
        encoded = value.encode()
        if len(encoded) > max_bytes:
            value, truncated = encoded[-max_bytes:].decode(errors="ignore"), True
        return CodexOutput(value, scope, truncated, state, output_id, turn_id)

    async def create(self, request: CodexCreateRequest) -> CodexThreadTarget:
        # The external server may have a different filesystem view or UID.
        # CodexCreateRequest validates syntax; the server owns existence checks.
        async with self._client(request.profile_key) as client:
            result = _object(
                await self._request(
                    client,
                    "thread/start",
                    {
                        "cwd": str(request.cwd),
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
            observed = await self._read(client, target.thread, include_items=False)
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
                observed = await self._read(client, target.thread, include_items=False)
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
    async def _client(
        self, profile: str, *, experimental_api: bool = False
    ) -> AsyncIterator[CodexAppServerClient]:
        if self._closed:
            raise CodexControlError("unavailable", "NotSent")
        endpoint = self._endpoints.get(profile)
        if endpoint is None:
            raise CodexControlError("unavailable", "NotSent")
        client = CodexAppServerClient(
            CodexAppServerConfig(
                endpoint, request_policy="observe_only", experimental_api=experimental_api
            )
        )
        self._clients.add(client)
        try:
            try:
                await client.__aenter__()
            except CodexConnectionUnavailable:
                raise CodexControlError("unavailable", "NotSent") from None
            except CodexAppServerResponseError:
                if not self._native_owners:
                    raise
                raise CodexControlError("unavailable", "NotSent") from None
            if not self._native_owners:
                account = _object(
                    await self._request(client, "account/read", {"refreshToken": False})
                )
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
        except CodexOutputLimit:
            raise CodexControlError(
                "output_limit", "Unknown" if mutation else "NotSent", known_thread
            ) from None
        except CodexConnectionUnavailable:
            raise CodexControlError(
                "unavailable", "Unknown" if mutation else "NotSent", known_thread
            ) from None
        except CodexAppServerResponseError as error:
            if method == "tui/view/read" and error.code == -32601:
                raise CodexControlError("unavailable", "NotSent") from None
            if (
                method == "thread/turns/list"
                and (not self._native_owners or error.code == -32600)
                and error.detail.startswith("invalid cursor:")
            ):
                raise CodexControlError("history_changed", "NotSent") from None
            if (
                self._native_owners
                and method == "thread/turns/list"
                and error.code == -32600
                and error.detail
                == f"thread {params.get('threadId')} is not materialized yet; "
                "thread/turns/list is unavailable before first user message"
            ):
                raise _HistoryNotMaterialized() from None
            if mutation and error.code == -32602 and error.detail == "TUI selection changed":
                raise CodexControlError("stale", "Rejected", known_thread) from None
            if (
                method == "turn/interrupt"
                and error.code == -32600
                and error.detail == "expected turn is not active"
            ):
                raise CodexControlError("stale", "Rejected", known_thread) from None
            if (
                method == "thread/items/list"
                and error.code == -32601
                and error.detail == "thread/items/list is not supported yet"
                and error.data is None
            ):
                raise _ItemPagingUnavailable() from None
            if self._native_owners:
                if (
                    not mutation
                    and method in ("thread/read", "thread/turns/list", "thread/items/list")
                    and error.code == -32600
                    and error.detail == f"no rollout found for thread id {params.get('threadId')}"
                ):
                    raise CodexControlError("missing", "NotSent", known_thread) from None
                raise CodexControlError(
                    "unavailable", "Unknown" if mutation else "NotSent", known_thread
                ) from None
            raise CodexControlError(
                _error_code(error), "Rejected" if mutation else "NotSent", known_thread
            ) from None

    async def _read(
        self, client: CodexAppServerClient, target: CodexThreadTarget, *, include_items: bool
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
        try:
            result = await self._turn_page(client, target, None, "desc", 1, "notLoaded")
        except _HistoryNotMaterialized:
            return CodexThreadRead(
                summary,
                None,
                None,
                CodexBounded("native history not materialized"),
                history_available=False,
            )
        data = result.get("data")
        assert isinstance(data, list)
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
        if turn.get("itemsView") != "notLoaded" or turn.get("items") != []:
            raise ProtocolDefect("Codex turn metadata unexpectedly included items")
        if not include_items:
            return CodexThreadRead(summary, snapshot, None, CodexBounded("turn metadata only"))
        coverage: CodexComplete | CodexBounded = (
            CodexBounded("latest turn only") if result.get("nextCursor") else CodexComplete()
        )
        if metadata_bounded:
            coverage = CodexBounded("metadata exceeds output bound")
        try:
            result = _object(
                await self._request(
                    client,
                    "thread/items/list",
                    {
                        "threadId": target.thread_handle,
                        "turnId": snapshot.target.turn_handle,
                        "limit": 50,
                        "sortDirection": "desc",
                    },
                )
            )
        except _ItemPagingUnavailable:
            return CodexThreadRead(
                summary, snapshot, None, CodexBounded("native item paging unavailable")
            )
        except CodexControlError as error:
            if error.code != "output_limit":
                raise
            return CodexThreadRead(
                summary, snapshot, None, CodexBounded("items exceed output bound")
            )
        items = result.get("data")
        if not isinstance(items, list) or len(items) > 50:
            raise ProtocolDefect("Codex item list exceeded its page contract")
        cursor = result.get("nextCursor")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise ProtocolDefect("Codex item list returned a malformed cursor")
        if cursor is not None:
            coverage = CodexBounded("latest turn item page only")
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 48 * 1024:
            return CodexThreadRead(
                summary, snapshot, None, CodexBounded("items exceed output bound")
            )
        answer = None
        unknown_phase = None
        for entry in items:
            entry = _object(entry)
            if entry.get("turnId") != snapshot.target.turn_handle:
                raise ProtocolDefect("Codex item page changed the requested turn identity")
            item = _object(entry.get("item"))
            if item.get("type") == "agentMessage":
                text = item.get("text")
                if not isinstance(text, str):
                    raise ProtocolDefect("Codex agent message has no text")
                if item.get("phase") == "final_answer" and answer is None:
                    answer = text
                elif item.get("phase") is None and unknown_phase is None:
                    unknown_phase = text
        if answer is None and cursor is None:
            answer = unknown_phase
        return CodexThreadRead(summary, snapshot, answer, coverage)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ProtocolDefect("Codex control response was not an object")
    return value


def _finalized(turn: dict[str, object]) -> bool:
    handle = turn.get("id")
    if not isinstance(handle, str):
        raise ProtocolDefect("Codex turn identity was malformed")
    try:
        identity = UUID(handle)
    except ValueError:
        return False
    return (
        str(identity) == handle
        and identity.version == 7
        and turn.get("status") == "completed"
        and type(turn.get("startedAt")) is int
        and type(turn.get("completedAt")) is int
    )


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
    can_accept = value.get("canAcceptDirectInput")
    if can_accept is not None and type(can_accept) is not bool:
        raise ProtocolDefect("Codex direct-input acceptance was malformed")
    return CodexThreadSummary(
        target,
        cast(CodexNativeStatus, kind),
        tuple(flags),
        name,
        cwd,
        source,
        cast(bool | None, can_accept),
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
        case unreachable:
            assert_never(unreachable)


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
