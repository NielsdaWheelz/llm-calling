"""Conversation delivered through Claude's typed transcript companions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ._archive_codec import object_value, occurrence, string, text, utterance
from .archive import ArchiveError, ArchiveEvent, ArchiveOccurrence

_HEADER = re.compile(r"^<(task-id|tool-use-id|status)>([^<>\r\n]*)</\1>$", re.MULTILINE)


@dataclass(slots=True)
class ClaudeDeliveries:
    conversation: str
    questions: dict[str, tuple[str, list[object]]] = field(default_factory=dict)
    agents: dict[str, list[str]] = field(default_factory=dict)

    def project(
        self,
        row: dict[str, object],
        values: list[ArchiveEvent],
        tool_names: dict[str, str],
    ) -> tuple[ArchiveOccurrence, ...]:
        if row.get("type") not in ("user", "assistant"):
            return tuple(occurrence(value) for value in values)
        message = object_value(row["message"])
        content = message["content"]
        blocks = (
            [{"type": "text", "text": content}]
            if isinstance(content, str)
            else [object_value(value) for value in content]
            if isinstance(content, list)
            else []
        )
        user = row["type"] == "user"
        origin = row.get("origin") if user else None
        origin = {} if origin is None else object_value(origin)
        origin_kind = string(origin.get("kind"), optional=True, limit=256)
        prompt_source = string(row.get("promptSource"), optional=True, limit=256) if user else None
        turn_origin = string(row.get("turnOrigin"), optional=True, limit=256) if user else None
        origin_proof = [origin_kind, prompt_source, turn_origin] if user else None
        non_human = user and (
            origin_kind not in (None, "human")
            or prompt_source == "system"
            or turn_origin == "task_notification"
        )
        projected: list[ArchiveOccurrence] = []
        unique_result = sum(block["type"] == "tool_result" for block in blocks) == 1
        body_blocks = (b for b in blocks if b["type"] not in ("thinking", "redacted_thinking"))
        for value, block in zip(values, body_blocks, strict=True):
            records = None
            evidence: object = origin_proof
            if block["type"] == "tool_use" and block.get("name") == "AskUserQuestion":
                call_id = string(block["id"], limit=256)
                arguments = block.get("input")
                if isinstance(arguments, dict) and not (
                    arguments.get("answers") or arguments.get("annotations")
                ):
                    questions = arguments.get("questions")
                    if isinstance(questions, list):
                        self.questions[call_id] = (string(row.get("uuid"), limit=256), questions)
            elif block["type"] == "tool_result":
                call_id = string(block["tool_use_id"], limit=256)
                tool = tool_names[call_id]
                companion = row.get("toolUseResult")
                if (
                    unique_result
                    and isinstance(companion, dict)
                    and row.get("sessionId") == self.conversation
                ):
                    if (
                        tool == "AskUserQuestion"
                        and not non_human
                        and not (block.get("is_error") or row.get("toolDenialKind"))
                    ):
                        records, answer_proof = self.answer(row, value, call_id, companion)
                        evidence = [origin_proof, answer_proof]
                    elif tool in ("Agent", "Task"):
                        records, agent_proof = self.agent(value, call_id, companion)
                        evidence = [origin_proof, agent_proof]
                    elif tool == "SendMessage":
                        sender = string(companion.get("resumedAgentId"), optional=True, limit=256)
                        revived = string(companion.get("revivedAgentId"), optional=True, limit=256)
                        # A successful continuation may carry a new native
                        # call identity for a worker already launched here.
                        if companion.get("success") is True:
                            agent = revived or sender
                            if agent in self.agents:
                                assert agent is not None
                                if call_id not in self.agents[agent]:
                                    self.agents[agent].append(call_id)
                                evidence = [origin_proof, agent, call_id, "continued"]
            elif non_human and block["type"] in ("text", "image", "document"):
                if value.kind != "context":
                    records = ()
                if block["type"] == "text" and row.get("sessionId") == self.conversation:
                    report, report_proof = self.notice(
                        value, text(block["text"]), origin, turn_origin
                    )
                    if report is not None:
                        records = (report,)
                    evidence = [origin_proof, report_proof]
            projected.append(occurrence(value, records, evidence))
        return tuple(projected)

    def answer(
        self,
        row: dict[str, object],
        value: ArchiveEvent,
        call_id: str,
        companion: dict[str, object],
    ) -> tuple[tuple[ArchiveEvent, ...], object]:
        source = self.questions.get(call_id)
        questions, answers = companion.get("questions"), companion.get("answers")
        if source is None or not isinstance(questions, list) or not isinstance(answers, dict):
            return (), None
        assistant_id, asked = source
        source_id = string(row.get("sourceToolAssistantUUID"), optional=True, limit=256)
        if source_id is not None and source_id != assistant_id:
            raise ArchiveError("source_conflict")
        ordered_keys = [question_key(object_value(question)) for question in questions]
        if not ordered_keys or ordered_keys != [
            question_key(object_value(question)) for question in asked
        ]:
            return (), None
        if len({key for key, _ in ordered_keys}) != len(ordered_keys):
            raise ArchiveError("unsupported")
        raw_annotations = companion.get("annotations")
        annotations = {} if raw_annotations is None else object_value(raw_annotations)
        records: list[ArchiveEvent] = []
        for index, ((key, wording), question) in enumerate(
            zip(ordered_keys, questions, strict=True)
        ):
            defined = object_value(question)
            native_question_id = string(defined.get("id"), optional=True, limit=256)
            answer = answers.get(key)
            annotation = annotations.get(key)
            raw_note = None if annotation is None else object_value(annotation).get("notes")
            literal = "" if answer is None else text(answer)
            note = "" if raw_note is None else text(raw_note)
            if not (literal.strip() or note.strip()):
                continue
            selections = selected_answers(defined, literal) if literal.strip() else []
            entries: list[tuple[Literal["owner", "assistant"], _AnswerKind, str]] = [
                ("assistant", "question", wording),
                *(("owner", "answer", selection) for selection in selections),
            ]
            if note.strip():
                entries.append(("owner", "answer_note", note))
            for field_index, (role, kind, body) in enumerate(entries):
                records.append(
                    utterance(
                        value,
                        conversation=self.conversation,
                        role=role,
                        kind=kind,
                        body=body,
                        source_key=f"{native_question_id or index}:{field_index}",
                        ordinal=len(records),
                        call_id=call_id,
                        question_id=native_question_id,
                    )
                )
        return tuple(records), [assistant_id, source_id, call_id]

    def agent(
        self, value: ArchiveEvent, call_id: str, companion: dict[str, object]
    ) -> tuple[tuple[ArchiveEvent, ...], object]:
        status = string(companion.get("status"), optional=True, limit=256)
        if status not in ("async_launched", "completed", "failed"):
            return (), None
        sender = string(companion.get("agentId"), optional=True, limit=256)
        identity = [status, call_id, sender]
        if sender is None:
            return (), identity
        calls = self.agents.setdefault(sender, [])
        if call_id not in calls:
            calls.append(call_id)
        if status == "async_launched":
            return (), identity
        content = companion.get("content")
        if content is None:
            return (), identity
        if not isinstance(content, list):
            raise ArchiveError("unsupported")
        bodies = (
            text(block.get("text"))
            for block in map(object_value, content)
            if block.get("type") == "text"
        )
        records = tuple(
            self.report(value, sender, call_id, body, index)
            for index, body in enumerate(body for body in bodies if body.strip())
        )
        return records, identity

    def notice(
        self,
        value: ArchiveEvent,
        body: str,
        origin: dict[str, object],
        turn_origin: str | None,
    ) -> tuple[ArchiveEvent | None, object]:
        origin_kind = origin.get("kind")
        if origin_kind == "task-notification" or turn_origin == "task_notification":
            outer = body.rstrip()
            if not outer.startswith("<task-notification>\n") or not outer.endswith(
                "\n</task-notification>"
            ):
                return None, None
            header_text, opening, rest = outer.partition("\n<result>")
            report, closing, trailer = rest.rpartition("</result>\n")
            if (
                not opening
                or not closing
                or (
                    trailer != "</task-notification>"
                    and not (
                        trailer.startswith("<usage>")
                        and trailer.endswith("</usage>\n</task-notification>")
                    )
                )
            ):
                return None, None
            fields = _HEADER.findall(header_text)
            header = dict(fields)
            if len(fields) != len(header):
                return None, None
            sender = string(header.get("task-id"), optional=True, limit=256)
            carrier_call = string(header.get("tool-use-id"), optional=True, limit=256)
            calls = self.agents.get(sender, ()) if sender is not None else ()
            known_call = (
                calls[0]
                if calls and carrier_call is None
                else carrier_call
                if carrier_call in calls
                else None
            )
            status = header.get("status")
            producer = string(origin.get("producer"), optional=True, limit=256)
            # runId for session-task notices is the session task-run identity,
            # not the agent ID or the identity of an individual delivery.
            run_id = string(origin.get("runId"), optional=True, limit=256)
            proof = [producer, run_id, sender, carrier_call, known_call, status]
            if (
                sender is not None
                and known_call is not None
                and status in ("completed", "failed")
                and report.strip()
            ):
                return self.report(value, sender, known_call, report, 0), proof
            return None, proof
        if origin_kind == "peer":
            agent = string(origin.get("senderTaskId"), optional=True, limit=256)
            sender = agent or string(origin.get("from"), optional=True, limit=256)
            calls = self.agents.get(sender, ()) if sender is not None else ()
            call_id = calls[0] if calls else None
            delivery_id = string(origin.get("msg_id"), optional=True, limit=256)
            proof = [sender, call_id, delivery_id]
            if sender is not None and call_id is not None and isinstance(origin.get("body"), str):
                report = text(origin["body"])
                if report.strip():
                    return self.report(value, sender, call_id, report, 0, delivery_id), proof
            return None, proof
        return None, None

    def report(
        self,
        value: ArchiveEvent,
        sender: str,
        call_id: str,
        body: str,
        ordinal: int,
        delivery_id: str | None = None,
    ) -> ArchiveEvent:
        return utterance(
            value,
            conversation=self.conversation,
            role="peer",
            kind="report",
            body=body,
            ordinal=ordinal,
            source_key=ordinal,
            delivery_id=delivery_id,
            call_id=call_id,
            sender_id=sender,
        )


type _AnswerKind = Literal["question", "answer", "answer_note"]


def question_key(question: dict[str, object]) -> tuple[str, str]:
    wording = text(question.get("question"))
    return string(question.get("id"), optional=True, limit=256) or wording, wording


def selected_answers(question: dict[str, object], answer: str) -> list[str]:
    options = question.get("options")
    if question.get("multiSelect") is True and isinstance(options, list):
        labels = {text(object_value(option).get("label")) for option in options}
        selections = answer.split(", ")
        if answer not in labels and all(label in labels for label in selections):
            return selections
    return [answer]
