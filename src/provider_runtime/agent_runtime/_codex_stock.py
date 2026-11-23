"""Stock root observations from typed Codex notifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ._codex_deliveries import collaboration_reports, typed_agent_delivery
from ._limits import _MAX_MESSAGE_ITEMS, OutputLimitExceeded, bounded_payload_size
from .archive import ArchiveError
from .auth import freeze_native_json_object
from .events import AgentToolUse, StockToolObservation

STOCK_PHASE_BYTES = 8 * 1024 * 1024


def stock_phase(
    event: AgentToolUse,
    item: Mapping[str, object],
    recipient_id: str,
    occurred_at_ms: object,
) -> AgentToolUse:
    item_type = item.get("type")
    completed = event.phase == "completed"
    timestamp = occurred_at_ms if type(occurred_at_ms) is int else None
    if item_type == "commandExecution":
        if completed:
            output = item.get("aggregatedOutput")
            exit_code = item.get("exitCode")
            if (
                isinstance(output, str | None)
                and (exit_code is None or type(exit_code) is int)
                and item.get("status") in ("completed", "failed", "declined")
            ):
                return replace(
                    event,
                    stock=StockToolObservation(
                        occurred_at_ms=timestamp,
                        shell_output="absent" if output is None else "reported",
                        exit_code=exit_code,
                    ),
                )
        elif isinstance(item.get("command"), str) and isinstance(item.get("cwd"), str):
            return replace(event, stock=StockToolObservation(occurred_at_ms=timestamp))
    elif item_type == "fileChange":
        changes = _file_changes(item.get("changes"), include_diffs=not completed)
        if changes is not None:
            return replace(
                event,
                payload=freeze_native_json_object({"changes": changes}),
                stock=StockToolObservation(occurred_at_ms=timestamp),
            )
    elif item_type == "collabAgentToolCall":
        try:
            reports = collaboration_reports(item, recipient_id) if completed else ()
        except ArchiveError:
            return replace(
                event,
                payload=None,
                stock=StockToolObservation(unsupported=True, occurred_at_ms=timestamp),
            )
        statuses = item.get("agentsStates")
        if isinstance(statuses, Mapping):
            fields: dict[str, object] = {
                "tool": item.get("tool"),
                "sender_thread_id": item.get("senderThreadId"),
                "receiver_thread_ids": item.get("receiverThreadIds"),
            }
            if completed:
                delivered = dict(reports)
                fields["agents"] = [
                    {
                        "id": agent,
                        "status": detail.get("status"),
                        "message": delivered.get(agent),
                    }
                    for agent, detail in statuses.items()
                    if isinstance(detail, Mapping)
                ]
            else:
                for name in ("prompt", "model", "reasoningEffort"):
                    fields[name] = item.get(name)
            return replace(
                event,
                payload=freeze_native_json_object(fields),
                succeeded=item.get("status") == "completed" if completed else None,
                stock=StockToolObservation(occurred_at_ms=timestamp, reports=reports),
            )
    elif item_type not in ("imageView", "sleep", "imageGeneration", "webSearch"):
        return event
    return replace(
        event, payload=None, stock=StockToolObservation(unsupported=True, occurred_at_ms=timestamp)
    )


def observed_delivery(item: Mapping[str, object]) -> AgentToolUse | None:
    try:
        delivery = typed_agent_delivery(item, "/root")
    except ArchiveError:
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            return None
        return AgentToolUse(
            identifier,
            "agentMessageDelivery",
            "completed",
            succeeded=False,
            stock=StockToolObservation(unsupported=True),
        )
    if delivery is None:
        return None
    identifier, sender, text = delivery
    return AgentToolUse(
        identifier,
        "agentMessageDelivery",
        "completed",
        succeeded=True,
        stock=StockToolObservation(delivery_id=identifier, reports=((sender, text),)),
    )


def _file_changes(value: object, *, include_diffs: bool) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    try:
        bounded_payload_size(value, STOCK_PHASE_BYTES, max_items=_MAX_MESSAGE_ITEMS)
    except OutputLimitExceeded:
        return None
    projected: list[dict[str, object]] = []
    for change in value:
        if not isinstance(change, Mapping):
            return None
        path, diff, kind = change.get("path"), change.get("diff"), change.get("kind")
        if not isinstance(path, str) or not isinstance(diff, str) or not isinstance(kind, Mapping):
            return None
        operation = kind.get("type")
        if operation not in ("add", "delete", "update"):
            return None
        change_kind = {"type": operation}
        if operation == "update":
            move_path = kind.get("move_path")
            if not isinstance(move_path, str | None):
                return None
            change_kind["move_path"] = move_path
        fields: dict[str, object] = {"path": path, "kind": change_kind}
        if include_diffs:
            fields["diff"] = diff
        projected.append(fields)
    return projected
