"""Owned argv subprocesses with bounded diagnostics and group cleanup."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ExecutableUnavailable, ProtocolDefect

type ProcessTermination = Literal["exited", "output_limit", "closed"]
type StdinMode = Literal["pipe", "devnull"]

_STDIN_MODES: dict[StdinMode, int] = {
    "pipe": asyncio.subprocess.PIPE,
    "devnull": asyncio.subprocess.DEVNULL,
}


@dataclass(frozen=True, slots=True)
class ProcessLimits:
    max_stderr_bytes: int
    termination_grace_seconds: float


@dataclass(frozen=True, slots=True)
class ProcessExit:
    returncode: int
    termination: ProcessTermination
    stderr: bytes = field(repr=False)


class ProcessGroup:
    """One process group named by its leader's pid, pinned for as long as it is owned.

    Two owners in this package hold a process group: `ManagedProcess`, which creates its own
    with `start_new_session=True`, and `_claude_launcher.OwnedProcessGroup`, which adopts the
    session the Claude launcher's `setsid()` created. Both need the same three syscall rules,
    and each of them is a way to get the group wrong, so they have one owner here:

    * pin the pid number with a pidfd, so it cannot be recycled between a check and a signal
      and `killpg` can never reach a stranger's group;
    * signal every member while treating an absent or unsignallable group as the desired
      final state rather than as a teardown failure;
    * probe membership with signal 0, which is the same rule as signalling.

    Escalation policy — how long a grace is, what is waited on, when SIGKILL follows — stays
    with each owner, because the two answer to different lifecycles.
    """

    __slots__ = ("_pid", "_pidfd")

    def __init__(self, pid: int, pidfd: int | None) -> None:
        self._pid = pid
        self._pidfd = pidfd

    @classmethod
    def pin(cls, pid: int) -> ProcessGroup:
        """Take the pidfd pin on `pid` and hold it until `release`."""
        return cls(pid, _open_pidfd(pid))

    def signal(self, number: int) -> bool:
        """Signal every member; False means the group is already empty or unreachable."""
        try:
            os.killpg(self._pid, number)
        except ProcessLookupError:
            # justify-ignore-error: an empty owned group is the desired final state.
            return False
        except PermissionError:
            # justify-ignore-error: a group we may not signal is not a group we own. The held
            # pidfd makes this unreachable on Linux, and teardown must not fail on it.
            return False
        return True

    def exists(self) -> bool:
        """Whether the kernel still has a member of this group to deliver a signal to."""
        return self.signal(0)

    def release(self) -> None:
        """Drop the pin. Idempotent, because every owner releases on more than one path."""
        pidfd = self._pidfd
        if pidfd is None:
            return
        self._pidfd = None
        os.close(pidfd)


class ManagedProcess:
    """One process group whose forked work is settled before close returns."""

    def __init__(self, process: asyncio.subprocess.Process, limits: ProcessLimits) -> None:
        self._process = process
        self._limits = limits
        # `spawn` gave this child its own session, so its pid leads the group every descendant
        # lands in; pinning that pid is what keeps the group ours until close releases it.
        self._group = ProcessGroup.pin(process.pid)
        self._stderr_limit_reached = asyncio.Event()
        self._stderr_output = bytearray()
        # The task is owned by this process and joined by wait/close on every path.
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._exit: ProcessExit | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._close_requested = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    async def spawn(
        cls,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        limits: ProcessLimits,
        stdin: StdinMode = "pipe",
    ) -> ManagedProcess:
        """Spawn one owned process group.

        The two stdin modes have one owner each. `"pipe"` belongs to a transport that writes
        to the child for the life of a process. `"devnull"` belongs to
        `capture_process_output`, which
        never writes: it hands the child EOF from the first instruction it executes, so a
        child that reads stdin before answering cannot stall a capture until its startup
        bound. Closing a pipe after the spawn would leave a window in which a fast child
        reads a partially written pipe, and leave a release step an early return can skip;
        DEVNULL has neither.
        """
        if stdin not in _STDIN_MODES:
            raise ValueError("process stdin must be 'pipe' or 'devnull'")
        command = tuple(argv)
        if not command or any(not value or "\0" in value for value in command):
            raise ValueError("process argv must contain non-empty strings without NUL bytes")
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError("process cwd must be an existing absolute directory")
        if limits.max_stderr_bytes <= 0 or limits.termination_grace_seconds <= 0:
            raise ValueError("process limits must be positive")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "=" in key
            or "\0" in key
            or "\0" in value
            for key, value in environment.items()
        ):
            raise ValueError("process environment must contain valid string names and values")
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=dict(environment),
            stdin=_STDIN_MODES[stdin],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        return cls(process, limits)

    @property
    def stdout(self) -> asyncio.StreamReader:
        stream = self._process.stdout
        if stream is None:
            raise RuntimeError("managed process stdout pipe is missing")
        return stream

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def send(self, data: bytes) -> None:
        stream = self._process.stdin
        if stream is None or stream.is_closing():
            raise RuntimeError("managed process stdin is closed")
        stream.write(data)
        await stream.drain()

    async def wait(self) -> ProcessExit:
        async with self._lifecycle_lock:
            return await self._wait_locked()

    async def _wait_locked(self) -> ProcessExit:
        if self._exit is not None:
            return self._exit

        process_task = asyncio.create_task(self._process.wait())
        limit_task = asyncio.create_task(self._stderr_limit_reached.wait())
        close_task = asyncio.create_task(self._close_requested.wait())
        tasks: set[asyncio.Future[Any]] = {process_task, limit_task, close_task}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if limit_task in done:
                termination: ProcessTermination = "output_limit"
                await self._terminate_group()
            elif close_task in done:
                termination = "closed"
                await self._terminate_group()
            else:
                termination = "exited"
                await self._terminate_remaining_group()
            await self._settle(tasks)
            stderr = await self._collect_stderr()
            returncode = await self._await_reap()
            self._exit = ProcessExit(
                returncode=returncode,
                termination=termination,
                stderr=stderr,
            )
            self._group.release()
            return self._exit
        except asyncio.CancelledError:
            await self._terminate_group()
            await self._settle(tasks)
            await self._collect_stderr()
            raise

    async def close(self) -> None:
        self._close_requested.set()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned())
        task = self._close_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise

    async def _close_owned(self) -> None:
        async with self._lifecycle_lock:
            if self._exit is not None:
                return
            await self._terminate_group()
            stderr = await self._collect_stderr()
            returncode = await self._await_reap()
            self._exit = ProcessExit(
                returncode=returncode,
                termination="closed",
                stderr=stderr,
            )
            self._group.release()

    async def _drain_stderr(self) -> bytes:
        stream = self._process.stderr
        if stream is None:
            raise RuntimeError("managed process stderr pipe is missing")
        while chunk := await stream.read(8192):
            remaining = self._limits.max_stderr_bytes - len(self._stderr_output)
            if remaining > 0:
                self._stderr_output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self._stderr_limit_reached.set()
        return bytes(self._stderr_output)

    async def _collect_stderr(self) -> bytes:
        try:
            async with asyncio.timeout(self._limits.termination_grace_seconds):
                return await self._stderr_task
        except TimeoutError:
            self._close_pipe_transport(2)
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            return bytes(self._stderr_output)

    async def _terminate_remaining_group(self) -> None:
        if self._group.exists():
            await self._terminate_group()

    async def _terminate_group(self) -> None:
        stream = self._process.stdin
        if stream is not None and not stream.is_closing():
            stream.close()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._limits.termination_grace_seconds
        self._group.signal(signal.SIGTERM)
        if self._process.returncode is None:
            try:
                async with asyncio.timeout_at(deadline):
                    await self._process.wait()
            except TimeoutError:
                pass
        if not self._group.exists():
            return
        remaining = deadline - loop.time()
        if remaining > 0:
            await asyncio.sleep(remaining)
        self._group.signal(signal.SIGKILL)
        await self._await_reap()

    async def _await_reap(self) -> int:
        try:
            async with asyncio.timeout(self._limits.termination_grace_seconds):
                return await self._process.wait()
        except TimeoutError:
            self._close_pipe_transport(0)
            self._close_pipe_transport(1)
            self._close_pipe_transport(2)
            raise ProtocolDefect(
                "owned process did not reap within the cleanup bound",
                code="process_cleanup_failed",
            ) from None

    def _close_pipe_transport(self, fd: int) -> None:
        transport = getattr(self._process, "_transport", None)
        get_pipe = getattr(transport, "get_pipe_transport", None)
        if callable(get_pipe):
            pipe = get_pipe(fd)
            if pipe is not None:
                close = getattr(pipe, "close", None)
                if callable(close):
                    close()

    async def _settle(self, tasks: set[asyncio.Future[Any]]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self._limits.termination_grace_seconds,
            )
            for task in done:
                if not task.cancelled():
                    task.exception()
            if pending:
                raise ProtocolDefect(
                    "owned process tasks did not settle within the cleanup bound",
                    code="process_cleanup_failed",
                )


async def capture_process_output(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    limits: ProcessLimits,
    startup_timeout_seconds: float,
    max_stdout_bytes: int,
    executable_label: str,
    purpose: str,
) -> str:
    """Run one short-lived discovery command and return its bounded UTF-8 stdout.

    The child is spawned on `/dev/null`: a discovery command is never written to, so the one
    correct stdin mode is fixed here rather than asked of each caller.

    Every way the capture can fail to happen — a launch that never happened, a timeout, a
    non-zero exit — is `ExecutableUnavailable`: the executable did not answer. Output that
    arrives but is unusable is a `ProtocolDefect`, because the child did answer and lied.
    """
    process: ManagedProcess | None = None
    try:
        async with asyncio.timeout(startup_timeout_seconds):
            process = await ManagedProcess.spawn(
                argv,
                cwd=cwd,
                environment=environment,
                limits=limits,
                stdin="devnull",
            )
            output = bytearray()
            while chunk := await process.stdout.read(4096):
                output.extend(chunk)
                if len(output) > max_stdout_bytes:
                    raise ProtocolDefect(
                        f"{purpose} exceeded its stdout bound",
                        code="discovery_output_limit",
                    )
            exit_status = await process.wait()
    except TimeoutError:
        if process is not None:
            await process.close()
        raise ExecutableUnavailable(f"{purpose} exceeded its startup bound") from None
    except OSError:
        # Both the spawn that never happened (`process is None`) and a pipe-level failure
        # after it land here, so the close is conditional rather than absent: a child that
        # was launched must never be left unreaped just because the read side broke.
        if process is not None:
            await process.close()
        raise ExecutableUnavailable(f"{executable_label} executable is unavailable") from None
    except BaseException:
        if process is not None:
            await process.close()
        raise
    if exit_status.returncode != 0 or exit_status.termination != "exited":
        raise ExecutableUnavailable(f"{purpose} failed")
    try:
        return bytes(output).decode("utf-8").strip()
    except UnicodeDecodeError:
        raise ProtocolDefect(f"{purpose} was not UTF-8", code="discovery_invalid_utf8") from None


def _open_pidfd(pid: int) -> int | None:
    """Pin the PID number for this process group's lifetime where the platform allows it."""
    open_pidfd = getattr(os, "pidfd_open", None)
    if open_pidfd is None:
        return None
    try:
        return open_pidfd(pid)
    except OSError:
        # justify-ignore-error: the child was already reaped, so there is no PID left to pin;
        # signalling then falls back to the group-existence probe, which is safe for an
        # empty group and, while any descendant survives, the kernel pins the PGID itself.
        return None


__all__ = [
    "ManagedProcess",
    "ProcessExit",
    "ProcessGroup",
    "ProcessLimits",
    "ProcessTermination",
    "StdinMode",
    "capture_process_output",
]
