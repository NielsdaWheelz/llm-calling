"""Shared archive projection, cursor selection and memory echo suppression."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal, Never, cast, overload
from uuid import UUID

from ._limits import (
    _MAX_MESSAGE_BYTES,
    _MAX_MESSAGE_ITEMS,
    OutputLimitExceeded,
    bounded_payload_size,
)
from .archive import (
    ArchiveBoundaryUncertain,
    ArchiveCapture,
    ArchiveError,
    ArchiveEvent,
    ArchiveKind,
    ArchiveOccurrence,
    ArchiveRead,
    ArchiveRole,
)

MAX_ARCHIVE_EVENTS = 100_000
ARCHIVE_PAGE_EVENTS = 100

type ArchiveEntry = ArchiveEvent | ArchiveOccurrence


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ArchiveError("unsupported")
    return value


@overload
def string(value: object, *, optional: Literal[False] = False, limit: int = 4096) -> str: ...


@overload
def string(value: object, *, optional: Literal[True], limit: int = 4096) -> str | None: ...


def string(value: object, *, optional: bool = False, limit: int = 4096) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or not value or len(value.encode("utf-8")) > limit:
        raise ArchiveError("unsupported")
    return value


def text(value: object) -> str:
    if type(value) is not str:
        raise ArchiveError("unsupported")
    return value


def timestamp(value: object, *, milliseconds: bool = False) -> datetime | None:
    if value is None:
        return None
    try:
        if type(value) is int:
            return datetime.fromtimestamp(value / (1000 if milliseconds else 1), UTC)
        if type(value) is str:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if result.tzinfo is not None:
                return result.astimezone(UTC)
    except (ValueError, OverflowError, OSError):
        pass
    raise ArchiveError("unsupported")


def canonical(value: object) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, UnicodeError):
        raise ArchiveError("unsupported") from None


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def bounded(value: object, *, max_bytes: int = _MAX_MESSAGE_BYTES) -> None:
    try:
        bounded_payload_size(value, max_bytes, max_items=_MAX_MESSAGE_ITEMS)
    except (OutputLimitExceeded, UnicodeError):
        raise ArchiveError("event_too_large") from None


def event(
    native_id: str,
    *,
    turn_id: str | None,
    parent_id: str | None,
    role: ArchiveRole,
    kind: ArchiveKind,
    body: str,
    attributes: dict[str, object],
    occurred_at: datetime | None,
    original: object,
    legacy_previous: str | None = None,
    projection_max_bytes: int = _MAX_MESSAGE_BYTES,
) -> ArchiveEvent:
    projection = {
        "native_id": native_id,
        "turn_id": turn_id,
        "native_parent_id": parent_id,
        "role": role,
        "kind": kind,
        "original": original,
        "occurred_at": occurred_at.isoformat() if occurred_at else None,
    }
    bounded(projection, max_bytes=projection_max_bytes)
    digest = canonical_digest(projection)
    if legacy_previous is not None:
        native_id = "legacy:" + hashlib.sha256((legacy_previous + digest).encode()).hexdigest()
    return ArchiveEvent(
        native_id, digest, turn_id, parent_id, role, kind, body, attributes, occurred_at
    )


def select(
    events: list[ArchiveEntry],
    *,
    after: str | None,
    checkpoint: str | None,
    caught_up: bool,
    observed_at: datetime,
    capture: bool,
) -> ArchiveRead | ArchiveCapture | None:
    """Return a page, or request confirmation of a marker missing from an exhausted scan."""
    boundary = checkpoint if checkpoint is not None else after
    start = 0
    if boundary is not None:
        index = next(
            (i for i, value in enumerate(events) if value.native_event_id == boundary), None
        )
        if index is None:
            if not caught_up:
                raise ArchiveBoundaryUncertain()
            return None
        start = index if checkpoint is not None else index + 1
    result = tuple(events[start : start + ARCHIVE_PAGE_EVENTS])
    more = start + len(result) < len(events)
    following = result[-1].native_event_id if more else None
    if capture:
        return ArchiveCapture(
            cast(tuple[ArchiveOccurrence, ...], result),
            following,
            caught_up and not more,
            observed_at,
        )
    return ArchiveRead(
        cast(tuple[ArchiveEvent, ...], result), following, caught_up and not more, observed_at
    )


def event_pairs(events: list[ArchiveEntry]) -> tuple[tuple[str, str, str | None], ...]:
    return tuple(
        (
            event.native_event_id,
            event.native_digest,
            event.capture_digest if isinstance(event, ArchiveOccurrence) else None,
        )
        for event in events
    )


def confirm_missing(
    first: tuple[tuple[str, str, str | None], ...],
    events: list[ArchiveEntry],
    *,
    caught_up: bool,
    checkpoint: str | None,
) -> Never:
    if not caught_up or first != event_pairs(events):
        raise ArchiveBoundaryUncertain()
    raise ArchiveError("history_changed" if checkpoint is not None else "activation_boundary_lost")


def occurrence(
    native: ArchiveEvent,
    records: tuple[ArchiveEvent, ...] | None = None,
    evidence: object = None,
) -> ArchiveOccurrence:
    if records is None:
        records = (
            ()
            if native.kind in ("memory_reference", "tool_call", "tool_result")
            or native.role == "tool"
            else (native,)
        )
    selected = [
        [record.native_event_id, record.kind, record.role, record.attributes, record.text]
        for record in records
    ]
    digest = canonical_digest(["capture.v4", native.native_digest, [evidence, selected]])
    return ArchiveOccurrence(native.native_event_id, native.native_digest, digest, records)


def utterance(
    native: ArchiveEvent,
    *,
    conversation: str,
    role: Literal["owner", "assistant", "peer"],
    kind: Literal["question", "answer", "answer_note", "report"],
    body: str,
    source_key: str | int,
    ordinal: int,
    delivery_id: str | None = None,
    call_id: str | None = None,
    question_id: str | None = None,
    sender_id: str | None = None,
) -> ArchiveEvent:
    source = [
        conversation,
        delivery_id or native.native_event_id,
        role,
        sender_id,
        kind,
        source_key,
    ]
    identity = "delivery:" + canonical_digest(source)
    digest = canonical_digest([source, body])
    attributes = {
        "kind": kind,
        "carrier_event_id": native.native_event_id,
        "ordinal": ordinal,
        **{
            name: value
            for name, value in (
                ("delivery_id", delivery_id),
                ("call_id", call_id),
                ("question_id", question_id),
                ("sender_id", sender_id),
            )
            if value is not None
        },
    }
    return ArchiveEvent(
        identity,
        digest,
        native.turn_id,
        native.native_parent_id,
        role,
        "text",
        body,
        {"delivery": attributes},
        native.occurred_at,
    )


def memory_attributes(tool: str, arguments: object, result: object) -> dict[str, object]:
    """Only closed references survive, including malformed/failed note submissions."""
    references: list[dict[str, object]] = []
    omitted = 0
    submission: str | None = None
    if isinstance(arguments, dict):
        candidate = arguments.get("submission_id")
        if type(candidate) is str:
            try:
                if str(UUID(candidate)) == candidate:
                    submission = candidate
            except ValueError:
                pass
    # MCP structured content is authoritative when exposed. JSON text blocks are the
    # other native MCP encoding, not ordinary prose and never retained as text here.
    values: list[object] = [result]
    if isinstance(result, dict):
        if "structuredContent" in result:
            values.append(result["structuredContent"])
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and type(item.get("text")) is str
                ):
                    try:
                        values.append(json.loads(item["text"]))
                    except ValueError:
                        pass
    # Zoom exposes a single original; search/open expose ordinary record kinds.
    values.extend(
        value["record"]
        for value in tuple(values)
        if isinstance(value, dict) and isinstance(value.get("record"), dict)
    )
    for value in values:
        if not isinstance(value, dict):
            continue
        direct = closed_reference(value)
        if direct is not None and direct not in references:
            if len(references) < 100:
                references.append(direct)
            else:
                omitted += 1
        for key in ("references", "records", "parts", "results", "matches", "notes", "nodes"):
            items = value.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                ref = closed_reference(item)
                if ref is None:
                    continue
                if ref not in references:
                    if len(references) < 100:
                        references.append(ref)
                    else:
                        omitted += 1
    attributes: dict[str, object] = {
        "tool_name": tool,
        "references": references,
        "omitted_reference_count": omitted,
    }
    if tool in ("memory_save_note", "memory.save_note") and submission is not None:
        attributes["submission_id"] = submission
    return attributes


def closed_reference(value: dict[str, object]) -> dict[str, object] | None:
    kind = value.get("kind")
    store = value.get("store")
    native_id = value.get("id")
    if (
        kind
        in (
            "record",
            "text",
            "tool_call",
            "tool_result",
            "attachment",
            "context",
            "memory_reference",
            "gap",
            "revision",
            "note",
            "summary",
            "missing",
            None,
        )
        and store in ("source_record", "memory_log", "memory_summary")
        and type(native_id) is str
    ):
        try:
            identifier = str(UUID(native_id))
        except ValueError:
            return None
        if identifier == native_id:
            return {"kind": "record", "store": store, "id": identifier}
    start, count = value.get("start"), value.get("count")
    if kind in ("range", None) and type(start) is int and type(count) is int:
        if (
            0 <= start < 2**63
            and 0 < count <= 2**63 - start
            and count & (count - 1) == 0
            and start % count == 0
        ):
            return {"kind": "range", "start": start, "count": count}
    return None
