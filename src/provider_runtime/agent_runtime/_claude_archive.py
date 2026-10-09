"""Claude's persisted JSONL originals, including child transcripts and branches."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from ._archive_codec import (
    MAX_ARCHIVE_EVENTS,
    bounded,
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
    MEMORY_TOOL_NAMES,
    ArchiveConversation,
    ArchiveError,
    ArchiveEvent,
    ArchiveHome,
    ArchiveInventory,
    ArchiveMissing,
    ArchiveRead,
)

# These records describe native UI/provider bookkeeping. They are not conversation
# evidence or replacement history. Unknown persisted records with UUIDs become gaps.
_BOOKKEEPING = frozenset(
    (
        "file-history-snapshot",
        "queue-operation",
        "custom-title",
        "agent-name",
        "agent-color",
        "slug",
        "tag",
        "last-prompt",
        "pr-link",
        "progress",
        "atis-latch",
        "cost-state",
        "mode",
    )
)


def files(home: ArchiveHome) -> dict[str, Path]:
    projects = home.state_root / "projects"
    if not home.state_root.is_dir():
        raise ArchiveError("unavailable")
    result: dict[str, Path] = {}
    try:
        for path in projects.rglob("*.jsonl"):
            identifier = (
                path.stem.removeprefix("agent-") if path.parent.name == "subagents" else path.stem
            )
            string(identifier, limit=256)
            if identifier in result:
                raise ArchiveError("source_conflict")
            result[identifier] = path
            if len(result) > MAX_ARCHIVE_EVENTS:
                raise ArchiveError("event_too_large")
    except OSError:
        raise ArchiveError("unavailable") from None
    return result


def marked(home: ArchiveHome, identifier: str, path: Path) -> bool:
    root_id = path.parent.parent.name if path.parent.name == "subagents" else identifier
    return (home.state_root / ".provider-runtime" / "archive-internal" / root_id).is_file()


def inventory(home: ArchiveHome, *, with_heads: bool) -> ArchiveInventory:
    observed_at = datetime.now(UTC)
    before = files(home)
    items: list[ArchiveConversation] = []
    stats: dict[str, tuple[int, int]] = {}
    complete = True
    try:
        for identifier, path in sorted(before.items()):
            metadata = path.stat()
            stats[identifier] = (metadata.st_size, metadata.st_mtime_ns)
            internal = marked(home, identifier, path)
            parent = path.parent.parent.name if path.parent.name == "subagents" else None
            created = datetime.fromtimestamp(metadata.st_ctime, UTC)
            updated = datetime.fromtimestamp(metadata.st_mtime, UTC)
            item = ArchiveConversation(
                identifier,
                created,
                updated,
                False,
                internal,
                relation="child" if parent else None,
                parent_native_id=parent,
            )
            if with_heads and not internal:
                events, caught_up, cwd, native_created = history(home, identifier, path)
                complete = complete and caught_up
                item = replace(item, working_directory=cwd, created_at=native_created or created)
                item = replace(item, head_event_id=events[-1].native_event_id if events else None)
            items.append(item)
        after = files(home)
        if with_heads:
            complete = (
                complete
                and before == after
                and all(
                    (path.stat().st_size, path.stat().st_mtime_ns) == stats[identifier]
                    for identifier, path in after.items()
                )
            )
    except OSError:
        raise ArchiveError("unavailable") from None
    return ArchiveInventory(tuple(items), None, complete if with_heads else True, observed_at)


def read(
    home: ArchiveHome, native_id: str, *, after: str | None, checkpoint: str | None
) -> ArchiveRead:
    observed_at = datetime.now(UTC)
    path = files(home).get(native_id)
    if path is None:
        raise ArchiveMissing()
    if marked(home, native_id, path):
        raise ArchiveError("unsupported")
    events, caught_up, _, _ = history(home, native_id, path)
    return select(
        events, after=after, checkpoint=checkpoint, caught_up=caught_up, observed_at=observed_at
    )


def history(
    home: ArchiveHome, identifier: str, path: Path
) -> tuple[list[ArchiveEvent], bool, str | None, datetime | None]:
    result: list[ArchiveEvent] = []
    seen: dict[str, str] = {}
    tool_names: dict[str, str] = {}
    child_ids: dict[str, str] = {}
    previous: str | None = None
    cwd: str | None = None
    created: datetime | None = None
    caught_up = True
    try:
        with path.open("rb") as stream:
            prefix = path.stat().st_size
            while stream.tell() < prefix:
                line = stream.readline(4 * 1024 * 1024 + 1)
                if len(line) > 4 * 1024 * 1024:
                    raise ArchiveError("event_too_large")
                if not line.endswith(b"\n"):
                    caught_up = False
                    break
                try:
                    native = object_value(json.loads(line))
                except (ValueError, UnicodeError):
                    raise ArchiveError("unsupported") from None
                bounded(native)
                kind = string(native.get("type"), limit=256)
                if kind in _BOOKKEEPING:
                    continue
                raw_id = native.get("uuid")
                if raw_id is None:
                    if kind in ("summary", "system-prompt", "system_prompt"):
                        # Stable source identity for complete replacement/context records
                        # that have no native UUID; ignored diagnostics never enter it.
                        import hashlib

                        raw_id = (
                            "context:"
                            + hashlib.sha256(
                                canonical(
                                    {
                                        "type": kind,
                                        "content": native.get("content"),
                                        "summary": native.get("summary"),
                                        "leafUuid": native.get("leafUuid"),
                                    }
                                ).encode("utf-8")
                            ).hexdigest()
                        )
                    else:
                        raise ArchiveError("unsupported")
                raw_id = string(raw_id, limit=230)
                parent = string(native.get("parentUuid"), optional=True, limit=256)
                native_session = string(native.get("sessionId"), optional=True, limit=256)
                if native_session is not None:
                    expected = (
                        path.parent.parent.name if path.parent.name == "subagents" else identifier
                    )
                    if native_session != expected:
                        raise ArchiveError("source_conflict")
                current_cwd = string(native.get("cwd"), optional=True)
                if cwd is None and current_cwd is not None:
                    cwd = current_cwd
                occurred_at = timestamp(native.get("timestamp"))
                if created is None and occurred_at is not None:
                    created = occurred_at
                values = row_events(
                    native,
                    raw_id=raw_id,
                    parent=parent,
                    occurred_at=occurred_at,
                    home=home,
                    peer=path.parent.name == "subagents",
                    tool_names=tool_names,
                    child_ids=child_ids,
                )
                if previous is not None and parent is not None and parent != previous and values:
                    result.append(
                        event(
                            "branch:" + raw_id,
                            turn_id=None,
                            parent_id=parent,
                            role="unknown",
                            kind="revision",
                            body="",
                            attributes={"reason": "branch"},
                            occurred_at=occurred_at,
                            original={"reason": "branch", "parent": parent, "previous": previous},
                        )
                    )
                if native.get("uuid") is not None:
                    previous = raw_id
                for value in values:
                    old = seen.get(value.native_event_id)
                    if old is not None:
                        if old != value.native_digest:
                            raise ArchiveError("source_conflict")
                        continue
                    seen[value.native_event_id] = value.native_digest
                    result.append(value)
                if len(result) > MAX_ARCHIVE_EVENTS:
                    raise ArchiveError("event_too_large")
    except OSError:
        raise ArchiveError("unavailable") from None
    return result, caught_up, cwd, created


def row_events(
    native: dict[str, object],
    *,
    raw_id: str,
    parent: str | None,
    occurred_at: datetime | None,
    home: ArchiveHome,
    peer: bool,
    tool_names: dict[str, str],
    child_ids: dict[str, str],
) -> list[ArchiveEvent]:
    kind = native["type"]
    values: list[ArchiveEvent] = []

    def append(
        suffix: str, role, event_kind, body: str, attributes: dict[str, object], original: object
    ):
        values.append(
            event(
                raw_id + suffix,
                turn_id=None,
                parent_id=parent,
                role=role,
                kind=event_kind,
                body=body,
                attributes=attributes,
                occurred_at=occurred_at,
                original=original,
            )
        )

    if kind == "attachment":
        attachment = object_value(native.get("attachment"))
        attachment_kind = string(attachment.get("type"), limit=256)
        context_fields = {
            "environment": ("environment", ("snapshot",)),
            "prompt_snapshot": ("instructions", ("systemPrompt", "contextRendering")),
            "session_context": ("environment", ("context",)),
            "model": ("other", ("identity", "text")),
            "date": ("environment", ("date",)),
            "total_tokens_reminder": ("other", ("text",)),
        }
        if attachment_kind in context_fields:
            context_kind, fields = context_fields[attachment_kind]
            if any(field not in attachment for field in fields):
                raise ArchiveError("unsupported")
            append(
                "",
                "unknown",
                "context",
                "",
                {
                    "context_kind": context_kind,
                    "native_name": attachment_kind,
                    "body_retention": "reference_only",
                },
                {"type": attachment_kind, **{field: attachment[field] for field in fields}},
            )
        else:
            append(
                "",
                "unknown",
                "gap",
                "",
                {"reason": "unsupported"},
                attachment,
            )
        return values
    if kind not in ("user", "assistant"):
        if kind in ("system", "summary", "system-prompt", "system_prompt"):
            subtype = string(native.get("subtype"), optional=True, limit=256)
            context_kind = (
                "compaction"
                if subtype == "compact_boundary" or kind == "summary"
                else "instructions"
                if kind in ("system-prompt", "system_prompt")
                else "other"
            )
            append(
                "",
                "unknown",
                "context",
                "",
                {
                    "context_kind": context_kind,
                    "native_name": subtype,
                    "body_retention": "reference_only",
                },
                {
                    "type": kind,
                    "subtype": subtype,
                    "content": native.get("content"),
                    "summary": native.get("summary"),
                    "compactMetadata": native.get("compactMetadata"),
                },
            )
        else:
            append("", "unknown", "gap", "", {"reason": "unsupported"}, native)
        return values
    message = object_value(native.get("message"))
    if message.get("role") != kind:
        raise ArchiveError("unsupported")
    content = message.get("content")
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = [object_value(block) for block in content]
    else:
        raise ArchiveError("unsupported")
    is_meta = native.get("isMeta", False)
    is_summary = native.get("isCompactSummary", False)
    if type(is_meta) is not bool or type(is_summary) is not bool:
        raise ArchiveError("unsupported")
    for index, block in enumerate(blocks):
        block_kind = string(block.get("type"), limit=256)
        if block_kind in ("thinking", "redacted_thinking"):
            continue
        suffix = "" if len(blocks) == 1 else f":{index}"
        if block_kind == "text":
            body = text(block.get("text"))
            if is_meta or is_summary:
                append(
                    suffix,
                    "unknown",
                    "context",
                    "",
                    {
                        "context_kind": "compaction" if is_summary else "other",
                        "native_name": None,
                        "body_retention": "reference_only",
                    },
                    {"type": kind, "isMeta": is_meta, "isCompactSummary": is_summary, "text": body},
                )
            else:
                append(
                    suffix,
                    "assistant" if kind == "assistant" else "peer" if peer else "owner",
                    "text",
                    body,
                    {},
                    {"type": kind, "text": body},
                )
        elif block_kind == "tool_use":
            call_id = string(block.get("id"), limit=256)
            name = string(block.get("name"), limit=256)
            if kind != "assistant" or "input" not in block:
                raise ArchiveError("unsupported")
            arguments = block["input"]
            tool_names[call_id] = name
            projection = {"type": block_kind, "id": call_id, "name": name, "input": arguments}
            memory_name = name.removeprefix(f"mcp__{home.memory_server}__")
            if (
                name == f"mcp__{home.memory_server}__{memory_name}"
                and memory_name in MEMORY_TOOL_NAMES
            ):
                append(
                    suffix,
                    "assistant",
                    "memory_reference",
                    "",
                    memory_attributes(memory_name, arguments, None),
                    projection,
                )
            else:
                append(
                    suffix,
                    "assistant",
                    "tool_call",
                    canonical(arguments),
                    {"native_call_id": call_id, "tool_name": name},
                    projection,
                )
        elif block_kind == "tool_result":
            call_id = string(block.get("tool_use_id"), limit=256)
            name = tool_names.get(call_id)
            if kind != "user" or name is None:
                raise ArchiveError("unsupported")
            error = block.get("is_error", False)
            if type(error) is not bool:
                raise ArchiveError("unsupported")
            output = block.get("content")
            receipt = native.get("toolUseResult")
            projection = {
                "type": block_kind,
                "tool_use_id": call_id,
                "name": name,
                "content": output,
                "is_error": error,
            }
            memory_name = name.removeprefix(f"mcp__{home.memory_server}__")
            if (
                name == f"mcp__{home.memory_server}__{memory_name}"
                and memory_name in MEMORY_TOOL_NAMES
            ):
                result = (
                    {"content": [{"type": "text", "text": output}]}
                    if isinstance(output, str)
                    else {"content": output}
                )
                append(
                    suffix,
                    "tool",
                    "memory_reference",
                    "",
                    memory_attributes(memory_name, None, result),
                    projection,
                )
            else:
                attributes = {
                    "native_call_id": call_id,
                    "outcome": "error" if error else "ok",
                    "omitted_characters": 0,
                }
                if name in ("Task", "Agent"):
                    if isinstance(receipt, dict) and receipt.get("agentId") is not None:
                        child_ids[call_id] = string(receipt["agentId"], limit=256)
                        projection["child_native_id"] = child_ids[call_id]
                    if call_id in child_ids:
                        attributes["child_native_id"] = child_ids[call_id]
                if isinstance(output, str):
                    body = output
                elif isinstance(output, list):
                    body = "\n".join(
                        text(object_value(value).get("text"))
                        for value in output
                        if object_value(value).get("type") == "text"
                    )
                elif output is None:
                    body = ""
                else:
                    raise ArchiveError("unsupported")
                append(suffix, "tool", "tool_result", body, attributes, projection)
        elif block_kind in ("image", "document"):
            source = object_value(block.get("source"))
            source_kind = source.get("type")
            reference = f"native:{raw_id}:{index}"
            if source_kind == "url":
                reference = string(source.get("url"))
            elif source_kind not in ("base64", "text", "content"):
                raise ArchiveError("unsupported")
            media = string(source.get("media_type"), optional=True, limit=256) or (
                "image/*" if block_kind == "image" else "application/octet-stream"
            )
            body = text(source.get("data")) if source_kind == "text" else ""
            availability = "full" if source_kind == "text" else "none"
            projection = {
                "type": block_kind,
                "source_type": source_kind,
                "reference": reference,
                "media_type": media,
                "text": body,
                "data": source.get("data"),
            }
            append(
                suffix,
                "owner" if kind == "user" and not peer else "peer" if peer else "assistant",
                "attachment",
                body,
                {"reference": reference, "media_type": media, "text_availability": availability},
                projection,
            )
        else:
            append(suffix, "unknown", "gap", "", {"reason": "unsupported"}, block)
    return values
