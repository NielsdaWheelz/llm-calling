"""The managed worker's single-writer Codex conversation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from ._limits import _MAX_FINAL_TEXT_BYTES
from ._worker_terminal import WorkerTerminal
from .codex_app_server import (
    CodexAppServerClient,
    CodexAppServerConfig,
    CodexAppServerResponseError,
    CodexNotification,
    CodexRequestId,
    CodexServerRequest,
    CodexStdioEndpoint,
    CodexThread,
    CodexTurn,
)
from .errors import ProtocolDefect
from .worker import LAUNCH_ENVIRONMENT, Outcome


class CodexWorker:
    def __init__(
        self,
        client: CodexAppServerClient,
        thread: CodexThread,
        terminal: WorkerTerminal,
        effort: str | None,
    ) -> None:
        self.client = client
        self.thread = thread
        self.terminal = terminal
        self.effort = effort
        self.starting: asyncio.Future[CodexTurn | None] | None = None
        self.unowned: str | None = None

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        executable: Path,
        cwd: Path,
        terminal: WorkerTerminal,
        model: str | None,
        effort: str | None,
        approval: str | None,
        sandbox: str | None,
    ) -> AsyncIterator[CodexWorker]:
        async with CodexAppServerClient(
            CodexAppServerConfig(
                endpoint=CodexStdioEndpoint(
                    executable,
                    {
                        key: value
                        for key, value in os.environ.items()
                        if key not in LAUNCH_ENVIRONMENT
                    },
                ),
                request_policy="ask_owned",
                experimental_api=True,
                stock_observations=True,
                client_name="skid",
                client_title="skid",
            )
        ) as client:
            config: dict[str, object] = {
                # These can start a root turn without an input from the worker.
                "features.goals": False,
                "features.wake_on_agent_messages": False,
                "features.in_app_local_automation": False,
            }
            if reply := os.getenv("SKID_REPLY_TO"):
                config["shell_environment_policy.set.SKID_REPLY_TO"] = reply
            thread = await client.thread_start(
                cwd=str(cwd),
                model=model,
                approvalPolicy=approval,
                sandbox=sandbox,
                config=config,
            )
            yield cls(client, thread, terminal, effort)

    async def run(self, text: str) -> Outcome:
        starting: asyncio.Future[CodexTurn | None] = asyncio.get_running_loop().create_future()
        self.starting = starting
        try:
            if self.busy():
                return Outcome("failed", "busy")
            marker = str(uuid4())
            try:
                response = await self.client.request(
                    "turn/start",
                    {
                        "threadId": self.thread.id,
                        "rootTurnId": marker,
                        "input": [{"type": "text", "text": text}],
                        **({"effort": self.effort} if self.effort else {}),
                    },
                )
                native = response.get("turn") if isinstance(response, dict) else None
                if (
                    not isinstance(native, dict)
                    or not isinstance(native.get("id"), str)
                    or native.get("rootTurnId") != marker
                ):
                    raise ProtocolDefect("Codex did not identify a newly submitted root turn")
                turn = CodexTurn(self.client, self.thread.id, native["id"])
            except CodexAppServerResponseError as error:
                if error.code == -32602:
                    return Outcome("failed", "provider_refused")
                raise ProtocolDefect("Codex did not establish the submitted turn") from None
            starting.set_result(turn)
            return await self._receive(turn)
        finally:
            if not starting.done():
                starting.set_result(None)
            self.starting = None

    @property
    def connected(self) -> bool:
        return self.client.usable

    def busy(self) -> bool:
        if self.connected:
            for pending in self.client.take_pending_messages():
                if isinstance(pending, CodexNotification):
                    self._observe_unowned(pending)
        return self.unowned is not None

    def _observe_unowned(self, message: CodexNotification) -> None:
        if message.method not in ("turn/started", "turn/completed"):
            return
        native = message.params.get("turn")
        if not isinstance(native, dict) or not isinstance(native.get("id"), str):
            return
        if message.method == "turn/started":
            self.unowned = native["id"]
        elif self.unowned == native["id"]:
            self.unowned = None

    async def steer(self, text: str) -> bool:
        turn = await asyncio.shield(self.starting) if self.starting is not None else None
        if turn is None:
            return False
        try:
            result = await self.client.request(
                "turn/steer",
                {
                    "threadId": self.thread.id,
                    "expectedTurnId": turn.id,
                    "input": [{"type": "text", "text": text}],
                },
            )
        except CodexAppServerResponseError:
            return False
        return isinstance(result, dict) and result.get("turnId") == turn.id

    async def interrupt(self) -> None:
        turn = await asyncio.shield(self.starting) if self.starting is not None else None
        if turn is None and self.unowned is not None:
            turn = CodexTurn(self.client, self.thread.id, self.unowned)
        if turn is not None:
            try:
                await turn.interrupt()
            except CodexAppServerResponseError:
                return

    async def _receive(self, turn: CodexTurn) -> Outcome:
        answer = ""
        changes: dict[str, object] = {}
        requests: dict[CodexRequestId, asyncio.Task[None]] = {}
        partial: dict[str, list[str]] = {}
        partial_bytes = 0
        async with asyncio.TaskGroup() as approvals:
            try:
                while True:
                    message = await self.client.next_message()
                    params = message.params
                    if params.get("threadId") != self.thread.id:
                        continue
                    if isinstance(message, CodexServerRequest):
                        if message.kind == "permission" and params.get("turnId") in (None, turn.id):
                            requests[message.request_id] = approvals.create_task(
                                self._answer(message, changes.get(str(params.get("itemId"))))
                            )
                        continue
                    if not isinstance(message, CodexNotification):
                        continue
                    if message.method == "serverRequest/resolved":
                        request = requests.pop(cast(CodexRequestId, params.get("requestId")))
                        if request is not None:
                            request.cancel()
                        continue
                    if message.method == "turn/started":
                        native = params.get("turn")
                        if isinstance(native, dict) and native.get("id") != turn.id:
                            self._observe_unowned(message)
                        continue
                    if message.method == "turn/completed":
                        native = params.get("turn")
                        if not isinstance(native, dict) or native.get("id") != turn.id:
                            self._observe_unowned(message)
                            continue
                        status = native.get("status")
                        if status == "completed":
                            return Outcome("answered", answer=answer)
                        incomplete = answer or "".join(next(reversed(partial.values()), []))
                        if status == "interrupted":
                            return Outcome("interrupted", answer=incomplete)
                        if status == "failed":
                            error = native.get("error")
                            reason = (
                                "quota_exhausted"
                                if isinstance(error, dict)
                                and error.get("codexErrorInfo") == "usageLimitExceeded"
                                else "provider_failed"
                            )
                            return Outcome("failed", reason, incomplete)
                        raise ProtocolDefect(
                            "Codex completed a turn with an unrecognized disposition"
                        )
                    if params.get("turnId") != turn.id:
                        continue
                    item = params.get("item")
                    if message.method == "item/agentMessage/delta":
                        item_id = params.get("itemId")
                        delta = params.get("delta")
                        if (
                            isinstance(item_id, str)
                            and item_id in partial
                            and isinstance(delta, str)
                        ):
                            partial_bytes += len(delta.encode("utf-8"))
                            if partial_bytes <= _MAX_FINAL_TEXT_BYTES:
                                partial[item_id].append(delta)
                        continue
                    if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                        continue
                    item_id = item["id"]
                    kind = item.get("type")
                    if message.method == "item/started":
                        if kind == "fileChange":
                            changes[item_id] = item.get("changes")
                        if kind == "agentMessage" and item.get("phase") == "final_answer":
                            partial[item_id] = []
                        if kind == "commandExecution" and isinstance(item.get("command"), str):
                            self.terminal.show(f"running · {item['command']}")
                    elif message.method == "item/completed":
                        changes.pop(item_id, None)
                        partial.pop(item_id, None)
                        if kind == "agentMessage" and isinstance(item.get("text"), str):
                            if item.get("phase") == "final_answer":
                                answer = item["text"]
                            elif item.get("phase") == "commentary":
                                self.terminal.show(item["text"])
            finally:
                for request in requests.values():
                    request.cancel()

    async def _answer(self, request: CodexServerRequest, changes: object) -> None:
        params = request.params
        match request.method:
            case "item/tool/requestUserInput":
                answers: dict[str, dict[str, list[str]]] = {}
                questions = params.get("questions")
                if not isinstance(questions, list):
                    raise ProtocolDefect("Codex sent an invalid question")
                for question in questions:
                    options = question.get("options") or []
                    labels = [item["label"] for item in options]
                    free_text = not labels or question.get("isOther") is True
                    text = question["question"]
                    if labels:
                        text += "\n" + "\n".join(
                            f"{option['label']} · {option['description']}" for option in options
                        )
                    answer = await self.terminal.ask(
                        "question",
                        text,
                        "type an answer." if free_text else "type a listed choice.",
                        lambda value, free_text=free_text, labels=labels: (
                            bool(value.strip()) and (free_text or value in labels)
                        ),
                    )
                    answers[question["id"]] = {"answers": [answer]}
                response: dict[str, Any] = {"answers": answers}
            case "item/permissions/requestApproval":
                permissions = params.get("permissions")
                allowed = await self.terminal.approve(
                    f"codex requested permissions\n{_show_operation(params, permissions=permissions)}"
                )
                response = {"permissions": permissions if allowed else {}, "scope": "turn"}
            case "mcpServer/elicitation/request":
                message = _show_operation(params)
                if params.get("mode") == "url":
                    approve = await self.terminal.approve(f"mcp request\n{message}")
                    response = {"action": "accept" if approve else "decline", "content": None}
                else:

                    def _form(value: str) -> bool:
                        if value == "deny":
                            return True
                        try:
                            return isinstance(json.loads(value), dict)
                        except ValueError:
                            return False

                    answer = await self.terminal.ask(
                        "question", message, "type a json object or deny.", _form
                    )
                    response = (
                        {"action": "decline", "content": None}
                        if answer == "deny"
                        else {"action": "accept", "content": json.loads(answer)}
                    )
            case "item/fileChange/requestApproval":
                if changes is None:
                    self.terminal.show("file change denied · codex did not show the changes")
                    allowed = False
                else:
                    allowed = await self.terminal.approve(
                        f"codex requested file changes\n{_show_operation(params, changes=changes)}"
                    )
                response = {"decision": "accept" if allowed else "decline"}
            case "item/commandExecution/requestApproval":
                allowed = await self.terminal.approve(
                    f"codex requested command access\n{_show_operation(params)}"
                )
                response = {"decision": "accept" if allowed else "decline"}
            case "execCommandApproval" | "applyPatchApproval":
                allowed = await self.terminal.approve(
                    f"codex requested approval\n{_show_operation(params)}"
                )
                response = {"decision": "approved" if allowed else "denied"}
            case _:
                raise ProtocolDefect("Codex sent an unsupported interactive operation")
        await self.client.respond(request.request_id, response)


def _show_operation(params: dict[str, object], **details: object) -> str:
    operation = {
        name: value
        for name, value in (params | details).items()
        if name
        not in ("threadId", "turnId", "itemId", "approvalId", "environmentId", "startedAtMs")
        and value is not None
    }
    return json.dumps(operation, indent=2, ensure_ascii=False)
