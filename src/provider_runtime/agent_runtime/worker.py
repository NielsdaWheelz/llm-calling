"""Run a coding conversation in one tmux pane with one assignment slot."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from ._limits import _MAX_FINAL_TEXT_BYTES
from .errors import AgentRuntimeDefect, AgentRuntimeError

_BRIEF_BYTES = 32 * 1024
_WIRE_BYTES = 6 * _BRIEF_BYTES + 1024  # The private JSON hop can escape every input byte.
type JobState = Literal["answered", "failed", "interrupted", "unknown"]
type Control = Literal["steer", "interrupt"]
LAUNCH_ENVIRONMENT = ("SKID_MANAGED_SOCKET", "SKID_MANAGED_RESULTS", "SKID_MANAGED_PROVIDER")


@dataclass(frozen=True)
class Outcome:
    state: JobState
    reason: str = ""
    answer: str = ""

    def __post_init__(self) -> None:
        try:
            size = len(self.answer.encode("utf-8"))
        except UnicodeError:
            size = -1
        reason = ""
        if size < 0 or "\0" in self.answer:
            reason = "invalid_final_answer"
        elif size > _MAX_FINAL_TEXT_BYTES:
            reason = "output_limit"
        elif self.state == "answered" and not self.answer.strip():
            reason = "no_final_answer"
        if reason:
            object.__setattr__(self, "answer", "")
            if self.state == "answered":
                object.__setattr__(self, "state", "failed")
                object.__setattr__(self, "reason", reason)


class Provider(Protocol):
    @property
    def connected(self) -> bool: ...
    def busy(self) -> bool: ...
    async def run(self, text: str) -> Outcome: ...
    async def steer(self, text: str) -> bool: ...
    async def interrupt(self) -> None: ...


class Worker:
    def __init__(self, results: Path) -> None:
        from ._worker_terminal import WorkerTerminal

        self.results = results
        self.terminal = WorkerTerminal(self.input, self.interrupt)
        self.provider: Provider
        self.turn: asyncio.Task[None] | None = None
        self.controls: dict[Control, asyncio.Task[None]] = {}
        self.job: str | None = None
        self.fenced: set[str] = set()
        self.finished = asyncio.Event()
        self.closed = asyncio.Event()

    async def serve(self, path: Path, provider: Provider) -> None:
        self.provider = provider
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            try:
                path.chmod(0o600)
                async with await asyncio.start_unix_server(
                    self.request, sock=listener, limit=_WIRE_BYTES
                ) as server:
                    terminal = asyncio.create_task(self.terminal.run())
                    closed = asyncio.create_task(self.closed.wait())
                    tasks = (terminal, closed)
                    try:
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        if terminal in done:
                            terminal.result()
                    finally:
                        server.close()
                        all_tasks = [*tasks, *self.controls.values()]
                        if self.turn is not None:
                            all_tasks.append(self.turn)
                        for task in all_tasks:
                            task.cancel()
                        await asyncio.gather(*all_tasks, return_exceptions=True)
                        self.finished.set()
            finally:
                path.unlink(missing_ok=True)

    def input(self, text: str) -> None:
        state = self.enter(text)
        if state != "accepted":
            self.terminal.show(f"input refused · {state}")

    def enter(self, text: str, job: str | None = None) -> str:
        if not _valid_brief(text):
            return "invalid_input"
        if (
            not self.provider.connected
            or self.closed.is_set()
            or (job is not None and job in self.fenced)
        ):
            return "unavailable"
        if self.terminal.interaction:
            return "dialog"
        if self.controls:
            return "busy"
        if self.turn is not None:
            if job is not None:
                return "busy"
            self.controls["steer"] = asyncio.create_task(
                self._control("steer", lambda: self.provider.steer(text))
            )
            return "accepted"
        if self.provider.busy():
            return "busy"
        self.job = job
        self.finished = asyncio.Event()
        self.turn = asyncio.create_task(self._turn(text))
        return "accepted"

    def interrupt(self) -> None:
        if (
            self.provider.connected
            and "interrupt" not in self.controls
            and (self.turn is not None or self.provider.busy())
        ):
            self.controls["interrupt"] = asyncio.create_task(
                self._control("interrupt", self.provider.interrupt)
            )

    async def _control(
        self, name: Control, operation: Callable[[], Awaitable[bool | None]]
    ) -> None:
        try:
            result = await operation()
            if result is False:
                self.terminal.show("steer refused · the turn ended")
        except (AgentRuntimeError, AgentRuntimeDefect, OSError):
            self.terminal.show("provider control unconfirmed")
        finally:
            del self.controls[name]

    async def _turn(self, text: str) -> None:
        job = self.job
        try:
            self.terminal.show("working…")
            try:
                outcome = (
                    await self.provider.run(text)
                    if self.provider.connected
                    else Outcome("failed", "unavailable")
                )
            except (AgentRuntimeError, AgentRuntimeDefect, OSError):
                outcome = Outcome("unknown", "provider_lost")
                self.closed.set()
            if job is not None:
                _seal(self.results / f"{job}.result", outcome)
            if outcome.answer:
                self.terminal.show(outcome.answer)
            if outcome.state != "answered":
                self.terminal.show(
                    f"{outcome.state} · {outcome.reason}" if outcome.reason else outcome.state
                )
        except Exception:
            self.terminal.show("worker stopped · result unavailable")
            self.closed.set()
        finally:
            self.turn = None
            self.job = None
            self.finished.set()

    async def request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            async with asyncio.timeout(5):
                request = json.loads(await reader.readline())
            if not isinstance(request, dict):
                return
            match request:
                case {"op": "status"} if request.keys() == {"op"}:
                    interaction = self.terminal.interaction
                    state = (
                        "unavailable"
                        if not self.provider.connected
                        else "needs-input"
                        if interaction
                        else "working"
                        if self.turn is not None or self.controls or self.provider.busy()
                        else "idle"
                    )
                    response = {"state": state, "draft": self.terminal.draft}
                    if interaction:
                        response["interaction"] = interaction
                case {"op": "submit", "job_id": str(job), "brief": str(brief)} if (
                    request.keys() == {"op", "job_id", "brief"}
                    and _valid_uuid(job)
                    and _valid_brief(brief)
                ):
                    state = self.enter(brief, job)
                    response = (
                        {"state": "accepted"}
                        if state == "accepted"
                        else {
                            "state": "failed",
                            "reason": "busy" if state == "dialog" else state,
                        }
                    )
                case {"op": "peer", "brief": str(brief)} if request.keys() == {
                    "op",
                    "brief",
                } and _valid_brief(brief):
                    state = "draft" if self.terminal.draft else self.enter(brief)
                    response = (
                        {"state": "accepted"}
                        if state == "accepted"
                        else {
                            "state": "failed",
                            "reason": state if state in ("draft", "dialog") else "unknown",
                        }
                    )
                case {"op": "wait", "job_id": str(job)} if request.keys() == {
                    "op",
                    "job_id",
                } and _valid_uuid(job):
                    path = self.results / f"{job}.result"
                    if self.job == job:
                        finished = asyncio.create_task(self.finished.wait())
                        disconnect = asyncio.create_task(reader.read(1))
                        try:
                            await asyncio.wait(
                                (finished, disconnect), return_when=asyncio.FIRST_COMPLETED
                            )
                            if disconnect.done():
                                return
                        finally:
                            finished.cancel()
                            disconnect.cancel()
                            await asyncio.gather(finished, disconnect, return_exceptions=True)
                    if path.is_file():
                        response = {"state": "ready"}
                    else:
                        # A crash can leave a submit in another socket's receive
                        # buffer. The missing observation must fence that input.
                        self.fenced.add(job)
                        response = {"state": "missing"}
                case _:
                    return
            writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
            await writer.drain()
        except (ValueError, TimeoutError, OSError):
            return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass


def _valid_uuid(value: str) -> bool:
    try:
        return value == str(UUID(value))
    except ValueError:
        return False


def _valid_brief(brief: str) -> bool:
    return bool(brief) and "\0" not in brief and len(brief.encode("utf-8")) <= _BRIEF_BYTES


def _seal(path: Path, outcome: Outcome) -> None:
    answer = outcome.answer.encode("utf-8")
    header = {
        "state": outcome.state,
        "length": len(answer),
        "sha256": hashlib.sha256(answer).hexdigest(),
        **({"reason": outcome.reason} if outcome.reason else {}),
    }
    descriptor, name = tempfile.mkstemp(prefix=".result.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n")
            stream.write(answer)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


async def _run(options: argparse.Namespace) -> None:
    worker = Worker(options.results)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGHUP, signal.SIGTERM):
        loop.add_signal_handler(signum, worker.closed.set)
    command = options.executable or shutil.which(options.provider)
    if not command:
        raise FileNotFoundError
    if options.provider == "codex":
        from ._worker_codex import CodexWorker

        async with CodexWorker.open(
            Path(command),
            Path.cwd(),
            worker.terminal,
            options.model,
            options.effort,
            options.approval_policy,
            options.sandbox,
        ) as provider:
            await worker.serve(options.socket, provider)
    else:
        from ._worker_claude import ClaudeWorker

        async with ClaudeWorker.open(
            Path(command),
            Path.cwd(),
            worker.terminal,
            options.model,
            options.effort,
            options.permission_mode,
        ) as provider:
            await worker.serve(options.socket, provider)


def main() -> None:
    parser = argparse.ArgumentParser(description="run one managed coding conversation")
    parser.add_argument("provider", choices=("codex", "claude"))
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--executable")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--permission-mode")
    parser.add_argument("--approval-policy")
    parser.add_argument("--sandbox")
    options = parser.parse_args()
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        parser.error("attach a terminal to start a worker")
    if options.provider == "codex" and options.permission_mode:
        parser.error("--permission-mode applies to claude")
    if options.provider == "claude" and (options.sandbox or options.approval_policy):
        parser.error("--sandbox and --approval-policy apply to codex")
    try:
        asyncio.run(_run(options))
    except (EOFError, KeyboardInterrupt):
        return
    except (AgentRuntimeError, AgentRuntimeDefect, ImportError, OSError, ExceptionGroup) as error:
        print(f"worker unavailable · {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
