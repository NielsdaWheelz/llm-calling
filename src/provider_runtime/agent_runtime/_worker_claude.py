"""The managed worker's Claude SDK conversation and native prompt acknowledgements."""

from __future__ import annotations

import asyncio
import json
import logging
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from ._limits import _MAX_FINAL_TEXT_BYTES
from ._worker_terminal import WorkerTerminal
from .claude_sdk import ClaudeSdkAdapter
from .errors import ProtocolDefect, SdkUnavailable
from .worker import LAUNCH_ENVIRONMENT, Outcome


@dataclass
class _Root:
    result: asyncio.Future[Outcome]
    deadline: asyncio.Timeout
    id: str = field(default_factory=lambda: str(uuid4()))
    pending_steer: str | None = None
    steers: set[str] = field(default_factory=set)
    settled: Outcome | None = None
    partial: str = ""


class ClaudeWorker:
    def __init__(self, sdk: ModuleType, client: Any, terminal: WorkerTerminal) -> None:
        self.sdk = sdk
        self.client = client
        self.terminal = terminal
        self.root: _Root | None = None
        self.unowned: set[str] = set()
        self.synthetic = False
        self.connected = True

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        executable: Path,
        cwd: Path,
        terminal: WorkerTerminal,
        model: str | None,
        effort: str | None,
        permission: str | None,
    ) -> AsyncIterator[ClaudeWorker]:
        sdk = ClaudeSdkAdapter._load_sdk()

        # SDK permissions fire for asks. A profile's allowed rules or explicit
        # bypass still govern its normal tools.
        async def _permission(
            name: str,
            command: dict[str, Any],
            context: Any,
        ) -> Any:
            if name == "AskUserQuestion":
                return sdk.PermissionResultAllow(updated_input=await _ask(terminal, command))
            details = json.dumps(command, ensure_ascii=False, indent=2)
            heading = context.title or f"claude requested {name}"
            allowed = await terminal.approve(f"{heading}\n{details}")
            return (
                sdk.PermissionResultAllow(updated_input=command)
                if allowed
                else sdk.PermissionResultDeny(message="permission denied by the owner")
            )

        options = sdk.ClaudeAgentOptions(
            cli_path=executable,
            cwd=cwd,
            model=model,
            effort=effort,
            permission_mode=permission,
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["user", "project", "local"],
            # The SDK overlays (rather than replaces) its child environment.
            env=dict.fromkeys(LAUNCH_ENVIRONMENT, ""),
            can_use_tool=_permission,
            extra_args={"replay-user-messages": None},
            include_partial_messages=True,
            max_buffer_size=_MAX_FINAL_TEXT_BYTES + 64 * 1024,
        )
        # SDK-internal diagnostics can contain native output. The worker owns
        # the content-free failure message and the terminal owns visible output.
        sdk_logger = logging.getLogger(sdk.__name__)
        sdk_logger.addHandler(logging.NullHandler())
        sdk_logger.propagate = False
        client = sdk.ClaudeSDKClient(options)
        try:
            with warnings.catch_warnings():
                # The selected profile intentionally supplies the bypass if configured.
                warnings.simplefilter("ignore", sdk.CanUseToolShadowedWarning)
                await client.connect()
        except sdk.ClaudeSDKError:
            raise SdkUnavailable("Claude worker could not start its native session") from None
        try:
            worker = cls(sdk, client, terminal)
            async with asyncio.TaskGroup() as group:
                receiver = group.create_task(worker._receive())
                try:
                    yield worker
                finally:
                    receiver.cancel()
        finally:
            await client.disconnect()

    async def _send(self, prompt: str, text: str) -> None:
        async def _input() -> AsyncIterator[dict[str, Any]]:
            yield {
                "type": "user",
                "uuid": prompt,
                "message": {"role": "user", "content": text},
                "parent_tool_use_id": None,
            }

        try:
            await self.client.query(_input())
        except self.sdk.ClaudeSDKError:
            raise SdkUnavailable("Claude worker lost its native session") from None

    async def steer(self, text: str) -> bool:
        root = self.root
        if root is None or root.pending_steer is not None or root.settled is not None:
            return False
        prompt = str(uuid4())
        root.pending_steer = prompt
        await self._send(prompt, text)
        return True

    async def interrupt(self) -> None:
        if self.root is not None or self.busy():
            try:
                await self.client.interrupt()
            except self.sdk.ClaudeSDKError:
                raise SdkUnavailable("Claude worker lost its native session") from None

    def busy(self) -> bool:
        return bool(self.unowned) or self.synthetic

    async def run(self, text: str) -> Outcome:
        if self.busy():
            return Outcome("failed", "busy")
        try:
            async with asyncio.timeout(None) as deadline:
                root = _Root(asyncio.get_running_loop().create_future(), deadline)
                self.root = root
                await self._send(root.id, text)
                return await root.result
        except TimeoutError:
            # A steer may have entered the provider's queue. Do not open this
            # conversation for new work until that queue has a proven disposition.
            raise ProtocolDefect("Claude did not confirm the steering boundary") from None
        finally:
            self.root = None

    async def _receive(self) -> None:
        try:
            await self._read_stream()
        except Exception:
            self.terminal.show("provider connection lost")
        finally:
            self.connected = False
            root = self.root
            if root is not None and not root.result.done():
                root.result.set_exception(SdkUnavailable("Claude ended the result stream"))

    async def _read_stream(self) -> None:
        # The pinned Python SDK's typed decoder discards command lifecycle,
        # result origin and prompt echoes. One reader owns that raw SDK stream.
        query = self.client._query
        assert query is not None
        async for message in query.receive_messages():
            root = self.root
            kind = message.get("type")
            command = message.get("command_uuid")
            state = message.get("state") if kind == "command_lifecycle" else None
            if isinstance(command, str) and (
                root is None
                or command != root.id
                and command != root.pending_steer
                and command not in root.steers
            ):
                if state == "started":
                    self.unowned.add(command)
                elif state in ("completed", "cancelled", "discarded", "refused"):
                    self.unowned.discard(command)
            origin = message.get("origin")
            if isinstance(origin, dict) and origin.get("kind") != "human":
                if kind in ("stream_event", "assistant", "user", "result"):
                    self.synthetic = kind != "result"
                continue
            if root is None or root.result.done():
                continue
            outcome = self._for_root(root, message, state, command)
            if outcome is not None:
                root.result.set_result(outcome)

    def _for_root(
        self,
        root: _Root,
        message: dict[str, Any],
        state: object,
        command: object,
    ) -> Outcome | None:
        kind = message.get("type")
        echo = message.get("user_message_uuids")
        if not isinstance(echo, list):
            echo = [message.get("user_message_uuid")]
        echoes = {identity for identity in echo if isinstance(identity, str)}
        if root.pending_steer is not None and (
            root.pending_steer in echoes or (command == root.pending_steer and state == "started")
        ):
            root.steers.add(root.pending_steer)
            root.pending_steer = None
            root.settled = None
            root.partial = ""
            root.deadline.reschedule(None)
        if (
            root.pending_steer is not None
            and command == root.pending_steer
            and state in ("cancelled", "discarded", "refused")
        ):
            root.pending_steer = None
            self.terminal.show("steer discarded · claude did not start it")
            return root.settled
        if root.id not in echoes and not root.steers.intersection(echoes):
            return None
        if kind == "assistant":
            self._progress(root, message)
        if kind != "result":
            return None
        reason = message.get("terminal_reason")
        if reason in ("aborted_streaming", "aborted_tools"):
            if root.pending_steer is not None:
                self.terminal.show("steer unconfirmed · claude interrupted the turn")
            return Outcome("interrupted", answer=root.partial)
        if message.get("is_error") is True or message.get("subtype") != "success":
            failure = (
                "rate_limited" if message.get("api_error_status") == 429 else "provider_failed"
            )
            return Outcome("failed", failure, root.partial)
        answer = message.get("result")
        outcome = Outcome("answered", answer=answer if isinstance(answer, str) else "")
        if root.pending_steer is None:
            return outcome
        root.settled = outcome
        root.deadline.reschedule(asyncio.get_running_loop().time() + 30)
        return None

    def _progress(self, root: _Root, message: dict[str, Any]) -> None:
        if message.get("parent_tool_use_id") is not None:
            return
        content = message.get("message", {}).get("content", [])
        texts = [block["text"] for block in content if block.get("type") == "text"]
        if texts:
            root.partial = "\n".join(texts)
        for block in content:
            if block.get("type") == "tool_use":
                self.terminal.show(f"using · {block.get('name', 'tool')}")
        if texts and message.get("message", {}).get("stop_reason") == "tool_use":
            self.terminal.show(root.partial)


async def _ask(
    terminal: WorkerTerminal,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    answers: dict[str, str] = {}
    for question in parameters["questions"]:
        prompt = question["question"]
        options = "\n".join(
            f"{option['label']} · {option['description']}" for option in question["options"]
        )
        multiple = question.get("multiSelect") is True

        answer = await terminal.ask(
            "question",
            f"{prompt}\n{options}",
            "type choices separated by commas, or your own answer."
            if multiple
            else "type a choice or your own answer.",
            lambda value: bool(value.strip()),
        )
        answers[prompt] = answer
    return {**parameters, "answers": answers}
