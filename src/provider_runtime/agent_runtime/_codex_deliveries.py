"""Codex collaboration evidence shared by history and live observation."""

from collections.abc import Mapping

from ._archive_codec import object_value, string, text
from .archive import ArchiveError

_FAILED_NEXT_ACTION = (
    "This agent's turn failed. If you still need this agent, use the available collaboration "
    "tools to give it another task."
)
_GUARDIAN_NEXT_ACTION = (
    "Tell the user this agent stopped after repeated Guardian denials. Do not resume this "
    "agent or retry its blocked work until the user explicitly confirms that it should continue."
)


def collaboration_reports(
    item: Mapping[str, object], recipient_id: str
) -> tuple[tuple[str, str], ...]:
    # v1 wait delivers terminal statuses to its caller. Other collaboration
    # tools expose snapshots or outgoing requests, and v2 wait carries no body.
    if item.get("tool") != "wait" or item.get("status") not in ("completed", "failed"):
        return ()
    if string(item.get("senderThreadId"), limit=256) != recipient_id:
        raise ArchiveError("source_conflict")
    receivers = item.get("receiverThreadIds")
    if not isinstance(receivers, list):
        raise ArchiveError("unsupported")
    states = object_value(item.get("agentsStates"))
    reports: list[tuple[str, str]] = []
    for receiver in dict.fromkeys(string(value, limit=256) for value in receivers):
        if receiver not in states:
            continue
        state = object_value(states[receiver])
        if state.get("status") != "completed" or state.get("message") is None:
            continue
        body = text(state["message"])
        if body.strip():
            reports.append((receiver, body))
    return tuple(reports)


def typed_agent_delivery(
    item: Mapping[str, object], recipient_path: str
) -> tuple[str, str, str] | None:
    if item.get("type") != "agent_message" or item.get("recipient") != recipient_path:
        return None
    sender = string(item.get("author"), limit=256)
    if not sender.startswith(recipient_path + "/"):
        return None
    delivery_id = string(item.get("id"), optional=True, limit=256)
    if delivery_id is None:
        return None
    blocks = item.get("content")
    if not isinstance(blocks, list):
        raise ArchiveError("unsupported")
    contents = [object_value(value) for value in blocks]
    if not contents or any(block.get("type") != "input_text" for block in contents):
        return None
    body = "\n".join(text(block.get("text")) for block in contents)
    for message_type in ("FINAL_ANSWER", "MESSAGE"):
        prefix = (
            f"Message Type: {message_type}\n"
            f"Task name: {recipient_path}\nSender: {sender}\nPayload:\n"
        )
        if body.startswith(prefix):
            report = body[len(prefix) :]
            if not report.strip():
                return None
            # Stock also sends typed FINAL_ANSWER envelopes for host errors.
            # Its completion renderer supplies these texts itself.
            if message_type == "FINAL_ANSWER" and (
                report in ("Agent shut down.", "Agent was not found.")
                or report.startswith("Agent errored: ")
                and report.endswith("\n\n" + _FAILED_NEXT_ACTION)
                or report.startswith("Agent interrupted by Guardian: ")
                and report.endswith("\n\n" + _GUARDIAN_NEXT_ACTION)
            ):
                return None
            return delivery_id, sender, report
    return None
