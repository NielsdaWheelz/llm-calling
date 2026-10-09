"""Codex history through read-only app-server methods, never rollout files."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from ._archive_codec import (
    MAX_ARCHIVE_EVENTS,
    canonical,
    event,
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
    ArchiveCapabilities,
    ArchiveConversation,
    ArchiveError,
    ArchiveEvent,
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


def client(home: ArchiveHome) -> CodexAppServerClient:
    assert home.codex_endpoint is not None
    return CodexAppServerClient(
        CodexAppServerConfig(
            socket_path=home.codex_endpoint, request_policy="observe_only", experimental_api=True
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
    if parent is not None and fork is not None:
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


async def listing(connection: CodexAppServerClient) -> tuple[list[ArchiveConversation], bool]:
    result: list[ArchiveConversation] = []
    seen: set[str] = set()
    complete = True
    for archived in (False, True):
        cursor: str | None = None
        cursors: set[str] = set()
        while True:
            page = await request(
                connection,
                "thread/list",
                {
                    "archived": archived,
                    "cursor": cursor,
                    "limit": 100,
                    "sourceKinds": list(CODEX_THREAD_SOURCES),
                    "sortKey": "created_at",
                    "sortDirection": "asc",
                },
            )
            data = page.get("data")
            if not isinstance(data, list) or len(data) > 100:
                raise ArchiveError("unsupported")
            for value in data:
                item, active = conversation(value, archived=archived)
                complete = complete and not active
                if item.native_id in seen:
                    complete = False
                    continue
                seen.add(item.native_id)
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
    return result, complete


async def inventory(home: ArchiveHome, *, with_heads: bool) -> ArchiveInventory:
    observed_at = datetime.now(UTC)
    try:
        async with client(home) as connection:
            items, quiet = await listing(connection)
            if not with_heads:
                return ArchiveInventory(tuple(items), None, True, observed_at)
            signatures: dict[str, tuple[str | None, str | None]] = {}
            for index, item in enumerate(items):
                if item.internal:
                    continue
                events, caught_up = await history(connection, home, item.native_id)
                quiet = quiet and caught_up
                head = events[-1] if events else None
                signatures[item.native_id] = (
                    (head.native_event_id, head.native_digest) if head else (None, None)
                )
                items[index] = replace(item, head_event_id=head.native_event_id if head else None)
            second, second_quiet = await listing(connection)
            stable = [replace(value, head_event_id=None) for value in items] == second
            if stable and quiet and second_quiet:
                for item in second:
                    if item.internal:
                        continue
                    events, caught_up = await history(connection, home, item.native_id)
                    head = events[-1] if events else None
                    signature = (head.native_event_id, head.native_digest) if head else (None, None)
                    stable = stable and caught_up and signatures[item.native_id] == signature
            return ArchiveInventory(
                tuple(items), None, stable and quiet and second_quiet, observed_at
            )
    except (CodexConnectionUnavailable, ProtocolDefect):
        raise ArchiveError("unavailable") from None


async def read(
    home: ArchiveHome, native_id: str, *, after: str | None, checkpoint: str | None
) -> ArchiveRead:
    observed_at = datetime.now(UTC)
    try:
        async with client(home) as connection:
            events, caught_up = await history(connection, home, native_id)
    except (CodexConnectionUnavailable, ProtocolDefect):
        raise ArchiveError("unavailable") from None
    return select(
        events, after=after, checkpoint=checkpoint, caught_up=caught_up, observed_at=observed_at
    )


async def history(
    connection: CodexAppServerClient, home: ArchiveHome, native_id: str
) -> tuple[list[ArchiveEvent], bool]:
    response = await request(
        connection, "thread/read", {"threadId": native_id, "includeTurns": False}
    )
    native = object_value(response.get("thread"))
    info, _ = conversation(native, archived=False)
    if info.native_id != native_id or info.internal:
        raise ArchiveError("unsupported")
    mode = native.get("historyMode", "legacy")
    if mode not in ("legacy", "paginated"):
        raise ArchiveError("unsupported")
    entries: list[tuple[dict[str, object], str, datetime | None]] = []
    caught_up = True
    if mode == "legacy":
        response = await request(
            connection, "thread/read", {"threadId": native_id, "includeTurns": True}
        )
        thread = object_value(response.get("thread"))
        if thread.get("id") != native_id:
            raise ArchiveError("unsupported")
        turns = thread.get("turns")
        if not isinstance(turns, list):
            raise ArchiveError("unsupported")
    else:
        turns = []
        cursor = None
        while True:
            page = await request(
                connection,
                "thread/turns/list",
                {
                    "threadId": native_id,
                    "cursor": cursor,
                    "limit": 100,
                    "itemsView": "notLoaded",
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
            if following == cursor:
                raise ArchiveError("unsupported")
            cursor = following
    for raw_turn in turns:
        turn = object_value(raw_turn)
        turn_id = string(turn.get("id"), limit=256)
        status = turn.get("status")
        if status == "inProgress":
            caught_up = False
            break
        if status not in ("completed", "interrupted", "failed"):
            raise ArchiveError("unsupported")
        occurred_at = timestamp(turn.get("startedAt"))
        if mode == "legacy":
            if turn.get("itemsView", "full") != "full":
                raise ArchiveError("unsupported")
            items = turn.get("items")
            if not isinstance(items, list):
                raise ArchiveError("unsupported")
            entries.extend((object_value(item), turn_id, occurred_at) for item in items)
        else:
            cursor = None
            while True:
                page = await request(
                    connection,
                    "thread/items/list",
                    {
                        "threadId": native_id,
                        "turnId": turn_id,
                        "cursor": cursor,
                        "limit": 100,
                        "sortDirection": "asc",
                    },
                )
                data = page.get("data")
                if not isinstance(data, list):
                    raise ArchiveError("unsupported")
                for raw_entry in data:
                    entry = object_value(raw_entry)
                    if entry.get("turnId") != turn_id:
                        raise ArchiveError("unsupported")
                    entries.append(
                        (
                            object_value(entry.get("item")),
                            turn_id,
                            timestamp(entry.get("startedAtMs"), milliseconds=True) or occurred_at,
                        )
                    )
                following = string(page.get("nextCursor"), optional=True)
                if following is None:
                    break
                if following == cursor:
                    raise ArchiveError("unsupported")
                cursor = following
        if len(entries) > MAX_ARCHIVE_EVENTS:
            raise ArchiveError("event_too_large")
    result: list[ArchiveEvent] = []
    previous = ""
    seen: set[str] = set()
    for item, turn_id, occurred_at in entries:
        values = item_events(
            item,
            home=home,
            turn_id=turn_id,
            occurred_at=occurred_at,
            peer=info.relation == "child",
            lineage=(info.relation, info.parent_native_id),
        )
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
    return result, caught_up


def item_events(
    item: dict[str, object],
    *,
    home: ArchiveHome,
    turn_id: str,
    occurred_at: datetime | None,
    peer: bool,
    lineage: tuple[str | None, str | None],
) -> list[ArchiveEvent]:
    identifier = string(item.get("id"), limit=230)
    kind = string(item.get("type"), limit=256)
    if kind == "reasoning":
        return []
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
                reference = string(part.get(ref_key))
                if reference.startswith("data:"):
                    reference = f"native:{identifier}:{index}"
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
        if status == "inProgress":
            raise ArchiveError("unsupported")
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
