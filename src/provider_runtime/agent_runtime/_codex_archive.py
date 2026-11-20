"""Codex history through read-only app-server methods, never rollout files."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

from ._archive_codec import (
    MAX_ARCHIVE_EVENTS,
    canonical,
    confirm_missing,
    event,
    event_pairs,
    memory_attributes,
    object_value,
    select,
    string,
    text,
    timestamp,
)
from .archive import (
    INTERNAL_THREAD_SOURCE,
    MEMORY_TOOL_NAMES,
    ArchiveBoundaryUncertain,
    ArchiveCapabilities,
    ArchiveConversation,
    ArchiveError,
    ArchiveErrorCode,
    ArchiveEvent,
    ArchiveHead,
    ArchiveHome,
    ArchiveInventory,
    ArchiveMissing,
    ArchiveRead,
)
from .codex_app_server import (
    CODEX_THREAD_SOURCES,
    CodexAppServerClient,
    CodexAppServerConfig,
    CodexAppServerResponseError,
    CodexConnectionUnavailable,
    CodexOutputLimit,
)
from .errors import ProtocolDefect

_NATIVE_PAGE_ITEMS = 100


def client(home: ArchiveHome) -> CodexAppServerClient:
    assert home.codex_endpoint is not None
    return CodexAppServerClient(
        CodexAppServerConfig(
            endpoint=home.codex_endpoint, request_policy="observe_only", experimental_api=True
        )
    )


async def request(
    connection: CodexAppServerClient, method: str, params: dict[str, object]
) -> dict[str, object]:
    try:
        return object_value(await connection.request(method, params))
    except CodexAppServerResponseError as error:
        if (
            method == "thread/read"
            and params.get("includeTurns") is False
            and error.code == -32600
            and error.data is None
            and error.detail == f"thread not loaded: {params.get('threadId')}"
        ):
            raise ArchiveMissing() from None
        if error.code == -32601:
            raise ArchiveError("unsupported") from None
        raise ArchiveError("unavailable") from None
    except CodexOutputLimit:
        raise ArchiveError("event_too_large") from None
    except (CodexConnectionUnavailable, ProtocolDefect, TimeoutError):
        raise ArchiveError("unavailable") from None


async def capabilities(home: ArchiveHome) -> ArchiveCapabilities:
    try:
        async with client(home) as connection:
            version = text(connection.metadata.get("userAgent"))
    except (CodexConnectionUnavailable, ProtocolDefect):
        raise ArchiveError("unavailable") from None
    return ArchiveCapabilities(version, True, True, True, True, True)


async def read_thread(connection: CodexAppServerClient, native_id: str) -> dict[str, object]:
    response = await request(
        connection, "thread/read", {"threadId": native_id, "includeTurns": False}
    )
    return object_value(response.get("thread"))


def conversation(value: object, *, archived: bool) -> tuple[ArchiveConversation, bool]:
    native = object_value(value)
    identifier = string(native.get("id"), limit=256)
    source = native.get("source")
    parent = string(native.get("parentThreadId"), optional=True, limit=256)
    fork = string(native.get("forkedFromId"), optional=True, limit=256)
    thread_source = string(native.get("threadSource"), optional=True, limit=256)
    internal = thread_source == INTERNAL_THREAD_SOURCE
    if type(source) is str:
        if source not in ("cli", "vscode", "exec", "appServer", "unknown"):
            raise ArchiveError("unsupported")
    elif isinstance(source, dict):
        if "subAgent" in source:
            agent = source["subAgent"]
            if type(agent) is str:
                if agent not in ("review", "compact", "memory_consolidation"):
                    raise ArchiveError("unsupported")
                internal = internal or agent in ("compact", "memory_consolidation")
            elif isinstance(agent, dict) and "thread_spawn" in agent:
                spawn = object_value(agent["thread_spawn"])
                origin = string(spawn.get("parent_thread_id"), limit=256)
                if parent is not None and parent != origin:
                    raise ArchiveError("unsupported")
                parent = origin
            elif not (isinstance(agent, dict) and isinstance(agent.get("other"), str)):
                raise ArchiveError("unsupported")
        elif not isinstance(source.get("custom"), str):
            raise ArchiveError("unsupported")
    else:
        raise ArchiveError("unsupported")
    if parent is not None and fork is not None and parent != fork:
        raise ArchiveError("unsupported")
    status = object_value(native.get("status"))
    active = status.get("type") == "active"
    if status.get("type") not in ("active", "idle", "notLoaded", "systemError"):
        raise ArchiveError("unsupported")
    created = timestamp(native.get("createdAt"))
    updated = timestamp(native.get("updatedAt"))
    if created is None or updated is None:
        raise ArchiveError("unsupported")
    return ArchiveConversation(
        native_id=identifier,
        created_at=created,
        updated_at=updated,
        archived=archived,
        internal=internal,
        relation="fork" if fork is not None else "child" if parent is not None else None,
        parent_native_id=fork or parent,
        working_directory=string(native.get("cwd"), optional=True),
    ), active


async def listing(connection: CodexAppServerClient) -> tuple[list[ArchiveConversation], set[str]]:
    result: list[ArchiveConversation] = []
    seen: dict[str, bool] = {}
    active_ids: set[str] = set()
    for archived in (False, True):
        # Native all-provider and configured-provider views have complementary
        # membership. Both must complete; neither substitutes for a failed view.
        for all_providers in (True, False):
            cursor: str | None = None
            cursors: set[str] = set()
            while True:
                page = await request(
                    connection,
                    "thread/list",
                    {
                        "archived": archived,
                        "cursor": cursor,
                        "limit": _NATIVE_PAGE_ITEMS,
                        "sourceKinds": list(CODEX_THREAD_SOURCES),
                        "sortKey": "created_at",
                        "sortDirection": "asc",
                        **({"modelProviders": []} if all_providers else {}),
                    },
                )
                data = page.get("data")
                if not isinstance(data, list) or len(data) > _NATIVE_PAGE_ITEMS:
                    raise ArchiveError("unsupported")
                for value in data:
                    identifier = string(object_value(value).get("id"), limit=256)
                    if identifier in seen:
                        if seen[identifier] != archived:
                            raise ArchiveError("source_conflict")
                        continue
                    native = await read_thread(connection, identifier)
                    if native.get("id") != identifier:
                        raise ArchiveError("unsupported")
                    item, active = conversation(native, archived=archived)
                    if active:
                        active_ids.add(identifier)
                    seen[item.native_id] = archived
                    result.append(item)
                    if len(result) > MAX_ARCHIVE_EVENTS:
                        raise ArchiveError("event_too_large")
                cursor = string(page.get("nextCursor"), optional=True)
                if cursor is None:
                    break
                if cursor in cursors:
                    raise ArchiveError("unsupported")
                cursors.add(cursor)
    result.sort(key=lambda value: value.native_id)
    return result, active_ids


async def inventory(home: ArchiveHome, *, with_heads: bool) -> ArchiveInventory:
    observed_at = datetime.now(UTC)
    try:
        async with client(home) as connection:
            before, active_ids = await listing(connection)
            if not with_heads:
                return ArchiveInventory(tuple(before), None, True, observed_at)
            items: list[ArchiveConversation] = []
            for item in before:
                if item.internal:
                    items.append(item)
                    continue
                try:
                    if item.native_id in active_ids:
                        raise ArchiveError("unavailable")
                    identifier = await stable_head(connection, home, item.native_id)
                    items.append(replace(item, head_event_id=identifier))
                except ArchiveError as error:
                    items.append(replace(item, head_error=cast(ArchiveErrorCode, error.code)))
            second, active_ids = await listing(connection)
            after = {item.native_id: item for item in second}
            complete = {item.native_id for item in before} == after.keys()
            for index, original in enumerate(before):
                current = after.get(original.native_id)
                if current is None:
                    continue
                if replace(current, updated_at=original.updated_at) != original:
                    complete = False
                if not original.internal and (
                    current.updated_at != original.updated_at or original.native_id in active_ids
                ):
                    items[index] = replace(
                        items[index], head_event_id=None, head_error="unavailable"
                    )
            return ArchiveInventory(tuple(items), None, complete, datetime.now(UTC))
    except (CodexConnectionUnavailable, ProtocolDefect):
        raise ArchiveError("unavailable") from None


async def stable_head(
    connection: CodexAppServerClient, home: ArchiveHome, native_id: str
) -> str | None:
    async def terminal() -> tuple[str | None, str | None]:
        native = await read_thread(connection, native_id)
        events, complete = await history(connection, home, native_id, native)
        if not complete:
            raise ArchiveError("unavailable")
        return (events[-1].native_event_id, events[-1].native_digest) if events else (None, None)

    before = await terminal()
    after = await terminal()
    if before != after:
        raise ArchiveError("unavailable")
    return after[0]


async def unchanged_thread(
    connection: CodexAppServerClient, before: ArchiveConversation
) -> dict[str, object] | None:
    native = await read_thread(connection, before.native_id)
    current, active = conversation(native, archived=False)
    return native if current == before and not active else None


async def head(home: ArchiveHome, native_id: str) -> ArchiveHead:
    try:
        async with client(home) as connection:
            before, active = conversation(await read_thread(connection, native_id), archived=False)
            if before.native_id != native_id or before.internal:
                raise ArchiveError("unsupported")
            if active:
                raise ArchiveError("unavailable")
            identifier = await stable_head(connection, home, native_id)
            if await unchanged_thread(connection, before) is None:
                raise ArchiveError("unavailable")
            return ArchiveHead(identifier, datetime.now(UTC))
    except (CodexConnectionUnavailable, ProtocolDefect):
        raise ArchiveError("unavailable") from None


async def read(
    home: ArchiveHome, native_id: str, *, after: str | None, checkpoint: str | None
) -> ArchiveRead:
    observed_at = datetime.now(UTC)
    try:
        async with client(home) as connection:
            native = await read_thread(connection, native_id)
            events, caught_up = await history(connection, home, native_id, native)
            page = select(
                events,
                after=after,
                checkpoint=checkpoint,
                caught_up=caught_up,
                observed_at=observed_at,
            )
            if page is not None:
                return page

            before, active = conversation(native, archived=False)
            if active:
                raise ArchiveBoundaryUncertain()
            first = event_pairs(events)
            del events
            native = await unchanged_thread(connection, before)
            if native is None:
                raise ArchiveBoundaryUncertain()
            events, caught_up = await history(connection, home, native_id, native)
            if await unchanged_thread(connection, before) is None:
                raise ArchiveBoundaryUncertain()
            confirm_missing(first, events, caught_up=caught_up, checkpoint=checkpoint)
    except (CodexConnectionUnavailable, ProtocolDefect):
        raise ArchiveError("unavailable") from None


async def history(
    connection: CodexAppServerClient, home: ArchiveHome, native_id: str, native: dict[str, object]
) -> tuple[list[ArchiveEvent], bool]:
    info, _ = conversation(native, archived=False)
    if info.native_id != native_id or info.internal:
        raise ArchiveError("unsupported")
    source = native.get("source")
    agent = source.get("subAgent") if isinstance(source, dict) else None
    peer = info.relation == "child" or isinstance(agent, dict) and "thread_spawn" in agent
    mode = native.get("historyMode", "legacy")
    if mode not in ("legacy", "paginated"):
        raise ArchiveError("unsupported")
    result: list[ArchiveEvent] = []
    previous = ""
    seen: set[str] = set()
    item_count = 0

    def append_item(item: dict[str, object], turn_id: str, occurred_at: datetime | None) -> bool:
        nonlocal previous, item_count
        item_count += 1
        if item_count > MAX_ARCHIVE_EVENTS:
            raise ArchiveError("event_too_large")
        values = item_events(
            item,
            home=home,
            turn_id=turn_id,
            occurred_at=occurred_at,
            peer=peer,
            lineage=(info.relation, info.parent_native_id),
        )
        if values is None:
            return False
        for value in values:
            if mode == "legacy":
                # Legacy item-N labels are positional. A chain over consumed content
                # makes any earlier rewrite invalidate every subsequent boundary.
                value = event(
                    value.native_event_id,
                    turn_id=value.turn_id,
                    parent_id=value.native_parent_id,
                    role=value.role,
                    kind=value.kind,
                    body=value.text,
                    attributes=value.attributes,
                    occurred_at=value.occurred_at,
                    original=value.native_digest,
                    legacy_previous=previous,
                )
                previous = value.native_event_id
            if value.native_event_id in seen:
                raise ArchiveError("source_conflict")
            seen.add(value.native_event_id)
            result.append(value)
        return True

    turns = []
    cursor = None
    turn_cursors: set[str] = set()
    while True:
        page = await request(
            connection,
            "thread/turns/list",
            {
                "threadId": native_id,
                "cursor": cursor,
                # Legacy history has full-turn paging, not item paging. One
                # complete native turn is its smallest bounded response unit.
                "limit": 1 if mode == "legacy" else _NATIVE_PAGE_ITEMS,
                "itemsView": "full" if mode == "legacy" else "notLoaded",
                "sortDirection": "asc",
            },
        )
        data = page.get("data")
        if not isinstance(data, list):
            raise ArchiveError("unsupported")
        turns.extend(data)
        if len(turns) > MAX_ARCHIVE_EVENTS:
            raise ArchiveError("event_too_large")
        following = string(page.get("nextCursor"), optional=True)
        if following is None:
            break
        if following in turn_cursors:
            raise ArchiveError("unsupported")
        turn_cursors.add(following)
        cursor = following
    for raw_turn in turns:
        turn = object_value(raw_turn)
        turn_id = string(turn.get("id"), limit=256)
        status = turn.get("status")
        if status == "inProgress":
            return result, False
        if status not in ("completed", "interrupted", "failed"):
            raise ArchiveError("unsupported")
        occurred_at = timestamp(turn.get("startedAt"))
        if mode == "legacy":
            if turn.get("itemsView", "full") != "full":
                raise ArchiveError("unsupported")
            items = turn.get("items")
            if not isinstance(items, list):
                raise ArchiveError("unsupported")
            for item in items:
                if not append_item(object_value(item), turn_id, occurred_at):
                    return result, False
        else:
            async with aclosing(item_entries(home, native_id, turn_id)) as entries:
                async for entry in entries:
                    if entry.get("turnId") != turn_id:
                        raise ArchiveError("unsupported")
                    if not append_item(
                        object_value(entry.get("item")),
                        turn_id,
                        timestamp(entry.get("startedAtMs"), milliseconds=True) or occurred_at,
                    ):
                        return result, False
    return result, True


async def item_entries(
    home: ArchiveHome, native_id: str, turn_id: str
) -> AsyncGenerator[dict[str, object], None]:
    # A rejected frame closes its connection. Retry its exact native cursor on
    # a fresh connection, reducing only the page, never the ingress bound.
    limit = _NATIVE_PAGE_ITEMS
    cursor: str | None = None
    cursors: set[str] = set()
    while True:
        try:
            async with client(home) as connection:
                while True:
                    page = await request(
                        connection,
                        "thread/items/list",
                        {
                            "threadId": native_id,
                            "turnId": turn_id,
                            "cursor": cursor,
                            "limit": limit,
                            "sortDirection": "asc",
                        },
                    )
                    data = page.get("data")
                    if not isinstance(data, list) or len(data) > limit:
                        raise ArchiveError("unsupported")
                    for entry in data:
                        yield object_value(entry)
                    following = string(page.get("nextCursor"), optional=True)
                    if following is None:
                        return
                    if not data or following in cursors:
                        raise ArchiveError("unsupported")
                    cursors.add(following)
                    cursor = following
        except ArchiveError as error:
            if error.code != "event_too_large" or limit == 1:
                raise
            limit = max(1, limit // 2)
        except (CodexConnectionUnavailable, ProtocolDefect):
            raise ArchiveError("unavailable") from None


def item_events(
    item: dict[str, object],
    *,
    home: ArchiveHome,
    turn_id: str,
    occurred_at: datetime | None,
    peer: bool,
    lineage: tuple[str | None, str | None],
) -> list[ArchiveEvent] | None:
    identifier = string(item.get("id"), limit=230)
    kind = string(item.get("type"), limit=256)
    if kind == "reasoning":
        return []
    if (
        kind
        in (
            "mcpToolCall",
            "dynamicToolCall",
            "commandExecution",
            "fileChange",
            "collabAgentToolCall",
        )
        and item.get("status") == "inProgress"
    ):
        # Neither the call nor result is immutable yet. Stop before terminal-only
        # fields are validated and before any later native item is consumed.
        return None
    values: list[ArchiveEvent] = []

    def append(
        suffix: str, role, event_kind, body: str, attributes: dict[str, object], original: object
    ):
        values.append(
            event(
                identifier + suffix,
                turn_id=turn_id,
                parent_id=None,
                role=role,
                kind=event_kind,
                body=body,
                attributes=attributes,
                occurred_at=occurred_at,
                original={
                    "content": original,
                    "relation": lineage[0],
                    "parent_conversation": lineage[1],
                },
            )
        )

    if kind == "userMessage":
        content = item.get("content")
        if not isinstance(content, list):
            raise ArchiveError("unsupported")
        for index, raw in enumerate(content):
            part = object_value(raw)
            part_kind = string(part.get("type"), limit=256)
            suffix = "" if len(content) == 1 else f":{index}"
            if part_kind == "text":
                body = text(part.get("text"))
                append(
                    suffix,
                    "peer" if peer else "owner",
                    "text",
                    body,
                    {},
                    {"type": kind, "content_type": part_kind, "text": body},
                )
            elif part_kind in ("image", "localImage", "audio", "localAudio"):
                ref_key = (
                    "path"
                    if part_kind.startswith("local")
                    else "fileId"
                    if "fileId" in part
                    else "url"
                )
                native_reference = text(part.get(ref_key))
                if native_reference.startswith("data:"):
                    reference = f"native:{identifier}:{index}"
                else:
                    reference = string(native_reference)
                attrs: dict[str, object] = {
                    "reference": reference,
                    "media_type": "image/*"
                    if "Image" in part_kind or part_kind == "image"
                    else "audio/*",
                    "text_availability": "none",
                }
                append(
                    suffix,
                    "peer" if peer else "owner",
                    "attachment",
                    "",
                    attrs,
                    {"type": part_kind, ref_key: part.get(ref_key)},
                )
            elif part_kind in ("skill", "mention"):
                name = string(part.get("name"), limit=256)
                reference = string(part.get("path"))
                attrs: dict[str, object] = {
                    "context_kind": "instructions" if part_kind == "skill" else "other",
                    "native_name": name,
                    "native_source_reference": reference,
                    "body_retention": "reference_only",
                }
                append(
                    suffix,
                    "unknown",
                    "context",
                    "",
                    attrs,
                    {"type": part_kind, "name": name, "path": reference},
                )
            else:
                append(suffix, "unknown", "gap", "", {"reason": "unsupported"}, part)
    elif kind in ("agentMessage", "plan"):
        body = text(item.get("text"))
        append(
            "",
            "assistant",
            "text",
            body,
            {},
            {"type": kind, "text": body, "phase": item.get("phase")},
        )
    elif kind == "contextCompaction":
        append(
            "",
            "unknown",
            "context",
            "",
            {"context_kind": "compaction", "native_name": None, "body_retention": "reference_only"},
            {"type": kind},
        )
    elif kind == "hookPrompt":
        fragments = item.get("fragments")
        if not isinstance(fragments, list):
            raise ArchiveError("unsupported")
        projection = [
            {
                "hookRunId": string(object_value(value).get("hookRunId")),
                "text": text(object_value(value).get("text")),
            }
            for value in fragments
        ]
        append(
            "",
            "unknown",
            "context",
            "",
            {"context_kind": "other", "native_name": None, "body_retention": "reference_only"},
            {"type": kind, "fragments": projection},
        )
    elif kind in (
        "mcpToolCall",
        "dynamicToolCall",
        "commandExecution",
        "fileChange",
        "collabAgentToolCall",
        "webSearch",
        "functionCallOutput",
    ):
        call_id = identifier
        child_native_id: str | None = None
        if kind == "mcpToolCall":
            server = string(item.get("server"), limit=256)
            tool = string(item.get("tool"), limit=256)
            memory = server == home.memory_server and tool in MEMORY_TOOL_NAMES
            tool_name = tool if memory else f"{server}.{tool}"
            if "arguments" not in item:
                raise ArchiveError("unsupported")
            arguments = item["arguments"]
            raw_output = item.get("result")
            if raw_output is not None:
                native_output = object_value(raw_output)
                mcp_result: dict[str, object] = {
                    "content": content_projection(native_output.get("content"))
                }
                if native_output.get("structuredContent") is not None:
                    mcp_result["structuredContent"] = native_output["structuredContent"]
                output = mcp_result
            else:
                output = None
            raw_error = item.get("error")
            error = (
                {"message": text(object_value(raw_error).get("message"))}
                if raw_error is not None
                else None
            )
            status = item.get("status")
            projection = {
                "type": kind,
                "server": server,
                "tool": tool,
                "arguments": arguments,
                "result": output,
                "error": error,
                "status": status,
            }
            body = (
                result_text(output)
                if output is not None
                else canonical(error)
                if error is not None
                else ""
            )
        elif kind == "dynamicToolCall":
            tool_name = string(item.get("tool"), limit=256)
            memory = False
            if "arguments" not in item:
                raise ArchiveError("unsupported")
            arguments = item["arguments"]
            output = (
                content_projection(item["contentItems"])
                if item.get("contentItems") is not None
                else None
            )
            status = item.get("status")
            projection = {
                "type": kind,
                "tool": tool_name,
                "namespace": item.get("namespace"),
                "arguments": arguments,
                "contentItems": output,
                "success": item.get("success"),
                "status": status,
            }
            body = result_text({"content": output}) if output is not None else ""
        elif kind == "commandExecution":
            tool_name, memory = "shell", False
            arguments = {"command": text(item.get("command")), "cwd": string(item.get("cwd"))}
            body = (
                text(item.get("aggregatedOutput"))
                if item.get("aggregatedOutput") is not None
                else ""
            )
            status = item.get("status")
            projection = {
                "type": kind,
                **arguments,
                "aggregatedOutput": body,
                "exitCode": item.get("exitCode"),
                "status": status,
            }
            output = None
        elif kind == "fileChange":
            tool_name, memory = "apply_patch", False
            changes = item.get("changes")
            if not isinstance(changes, list):
                raise ArchiveError("unsupported")
            arguments = [
                {
                    "path": string(object_value(v).get("path")),
                    "kind": object_value(v).get("kind"),
                    "diff": text(object_value(v).get("diff")),
                }
                for v in changes
            ]
            body, status, output = "", item.get("status"), None
            projection = {"type": kind, "changes": arguments, "status": status}
        elif kind == "collabAgentToolCall":
            tool_name, memory = string(item.get("tool"), limit=256), False
            receivers = item.get("receiverThreadIds")
            states = object_value(item.get("agentsStates"))
            if not isinstance(receivers, list) or any(
                type(value) is not str or not value for value in receivers
            ):
                raise ArchiveError("unsupported")
            sender = string(item.get("senderThreadId"), limit=256)
            arguments = {
                "sender": sender,
                "receivers": receivers,
                "prompt": string(item.get("prompt"), optional=True, limit=4 * 1024 * 1024),
            }
            output = {
                key: {
                    "status": string(object_value(value).get("status")),
                    "message": object_value(value).get("message"),
                }
                for key, value in states.items()
            }
            if len(receivers) == 1:
                child_native_id = receivers[0]
            body, status = canonical(output), item.get("status")
            projection = {
                "type": kind,
                "tool": tool_name,
                "arguments": arguments,
                "states": output,
                "status": status,
            }
        elif kind == "webSearch":
            tool_name, memory = "web.search", False
            arguments = {"query": text(item.get("query")), "action": item.get("action")}
            output = item.get("results")
            body, status = canonical(output) if output is not None else "", "completed"
            projection = {"type": kind, **arguments, "results": output}
        else:
            tool_name, memory = string(item.get("name"), limit=256), False
            raw_output = item.get("output")
            output = raw_output if isinstance(raw_output, str) else content_projection(raw_output)
            body, status, arguments = result_text(output), "completed", None
            projection = {
                "type": kind,
                "name": tool_name,
                "namespace": item.get("namespace"),
                "output": output,
            }
        if status not in ("completed", "failed", "declined", "interrupted"):
            raise ArchiveError("unsupported")
        if memory:
            append(
                "",
                "tool",
                "memory_reference",
                "",
                memory_attributes(tool_name, arguments, output),
                projection,
            )
        else:
            if kind != "functionCallOutput":
                append(
                    ":call",
                    "assistant",
                    "tool_call",
                    canonical(arguments),
                    {"native_call_id": call_id, "tool_name": tool_name},
                    {**projection, "phase": "call"},
                )
            attrs = {
                "native_call_id": call_id,
                "outcome": "ok" if status == "completed" else "error",
                "omitted_characters": 0,
            }
            if child_native_id is not None:
                attrs["child_native_id"] = child_native_id
            append(":result", "tool", "tool_result", body, attrs, {**projection, "phase": "result"})
    elif kind == "imageView":
        reference = string(item.get("path"))
        append(
            "",
            "tool",
            "attachment",
            "",
            {"reference": reference, "media_type": "image/*", "text_availability": "none"},
            {"type": kind, "path": reference},
        )
    else:
        append("", "unknown", "gap", "", {"reason": "unsupported"}, item)
    return values


def result_text(value: object) -> str:
    if isinstance(value, str):
        return value
    content = value.get("content") if isinstance(value, dict) else value
    if not isinstance(content, list):
        raise ArchiveError("unsupported")
    parts: list[str] = []
    for raw in content:
        part = object_value(raw)
        kind = part.get("type")
        if kind in ("text", "inputText"):
            parts.append(text(part.get("text")))
        elif kind in ("image", "audio", "inputImage", "inputAudio"):
            continue
        elif kind == "resource":
            resource = object_value(part.get("resource"))
            if "text" in resource:
                parts.append(text(resource["text"]))
        elif kind == "resource_link":
            parts.append(string(part.get("uri")))
        else:
            raise ArchiveError("unsupported")
    if isinstance(value, dict) and value.get("structuredContent") is not None:
        parts.append(canonical(value["structuredContent"]))
    return "\n".join(parts)


def content_projection(value: object) -> list[dict[str, object]]:
    """Bind native content while excluding its unrelated transport annotations."""
    if not isinstance(value, list):
        raise ArchiveError("unsupported")
    result: list[dict[str, object]] = []
    for raw in value:
        part = object_value(raw)
        kind = string(part.get("type"), limit=256)
        projection: dict[str, object] = {"type": kind}
        if kind in ("text", "inputText"):
            projection["text"] = text(part.get("text"))
        elif kind in ("image", "audio"):
            projection["data"] = text(part.get("data"))
            projection["mimeType"] = string(part.get("mimeType"), limit=256)
        elif kind in ("inputImage", "inputAudio"):
            projection["imageUrl" if kind == "inputImage" else "audioUrl"] = string(
                part.get("imageUrl" if kind == "inputImage" else "audioUrl"), limit=4 * 1024 * 1024
            )
        elif kind == "resource":
            resource = object_value(part.get("resource"))
            projection["resource"] = {
                key: resource[key] for key in ("uri", "mimeType", "text", "blob") if key in resource
            }
        elif kind == "resource_link":
            projection["uri"] = string(part.get("uri"))
        else:
            raise ArchiveError("unsupported")
        result.append(projection)
    return result
