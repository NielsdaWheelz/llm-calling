"""The live tmux pane for a managed worker."""

from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.patch_stdout import patch_stdout

type Interaction = Literal["approval", "question"]


@dataclass
class _Prompt:
    kind: Interaction
    message: str
    instructions: str
    accepts: Callable[[str], bool]
    answer: asyncio.Future[str]


class WorkerTerminal:
    def __init__(self, submit: Callable[[str], None], interrupt: Callable[[], None]) -> None:
        self._prompts: list[_Prompt] = []
        keys = KeyBindings()

        @keys.add("c-c", eager=True)
        def _interrupt(event: KeyPressEvent) -> None:
            event.current_buffer.reset()
            interrupt()

        @keys.add("enter", eager=True)
        def _submit(event: KeyPressEvent) -> None:
            buffer = event.current_buffer
            text = buffer.text
            if not self._prompts:
                buffer.append_to_history()
                buffer.reset()
                if text:
                    submit(text)
                return
            buffer.reset()
            prompt = self._prompts[0]
            if prompt.answer.done():
                return
            if prompt.accepts(text):
                prompt.answer.set_result(text)
            else:
                self.show(prompt.instructions)

        self._input: PromptSession[str] = PromptSession(self._label, key_bindings=keys)

    @property
    def draft(self) -> bool:
        return bool(self._input.default_buffer.text)

    @property
    def interaction(self) -> Interaction | None:
        return self._prompts[0].kind if self._prompts else None

    def _label(self) -> str:
        if not self._prompts:
            return "> "
        return "approve> " if self._prompts[0].kind == "approval" else "answer> "

    def show(self, message: str) -> None:
        print(
            "".join(
                char if char in "\n\t" or unicodedata.category(char) != "Cc" else repr(char)[1:-1]
                for char in message
            ),
            flush=True,
        )

    def _show_prompt(self) -> None:
        if self._prompts:
            prompt = self._prompts[0]
            self.show(f"{prompt.message}\n{prompt.instructions}")
        self._input.app.invalidate()

    async def ask(
        self,
        kind: Interaction,
        message: str,
        instructions: str,
        accepts: Callable[[str], bool],
    ) -> str:
        prompt = _Prompt(
            kind, message, instructions, accepts, asyncio.get_running_loop().create_future()
        )
        self._prompts.append(prompt)
        if len(self._prompts) == 1:
            # A draft written before the prompt appeared cannot become an approval.
            self._input.default_buffer.reset()
            self._show_prompt()
        try:
            return await prompt.answer
        finally:
            active = self._prompts[0] is prompt
            self._prompts.remove(prompt)
            if active:
                self._input.default_buffer.reset()
                self._show_prompt()

    async def approve(self, message: str) -> bool:
        answer = await self.ask(
            "approval", message, "type allow or deny.", lambda text: text in ("allow", "deny")
        )
        return answer == "allow"

    async def run(self) -> None:
        with patch_stdout():
            self.show("worker ready. enter to send · ctrl-c to interrupt")
            await self._input.prompt_async()
