"""Process-group ownership for the child the Claude Agent SDK spawns.

claude-agent-sdk 0.2.130 starts Claude Code with
`anyio.open_process(cmd, stdin=PIPE, ...)` and no `start_new_session`
(`_internal/transport/subprocess_cli.py:835`), so the child joins the *host* application's
process group, and the SDK's own teardown escalates terminate/kill on that one pid
(`close()`, same file). Anything Claude Code started — a Bash-tool command, a stdio MCP
server, a compiler — therefore outlives a cancelled turn, a timeout, and `close()`. The
Claude SDK exposes no subprocess flag that would let the runtime request a new process
session at the spawn call itself.

The SDK gives exactly one seam wide enough to fix that without reimplementing its argv:
`ClaudeAgentOptions.cli_path` is public API (`types.py:1886`) and is spawned verbatim as
`cmd[0]` (`subprocess_cli.py:229`, `:565`). Pointing it at the launcher this module writes
makes the child call `setsid()` and then `execv()` the real `claude`. Because `execv` does
not fork, the pid `anyio.open_process` returns *is* the new session and process-group
leader, so `os.killpg(pid, ...)` is exact: it reaches every descendant and can never reach
the host's own group. Nothing in the SDK's command construction is read or reproduced, and
the launcher stays transparent for the SDK's `[cli_path, "-v"]` version probe
(`subprocess_cli.py:1128`) because it forwards its argv unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import signal
import stat
import tempfile
from pathlib import Path

from ._process import ProcessGroup
from .errors import AgentRuntimeDefect, ExecutableUnavailable, ProtocolDefect, SdkUnavailable

_PRIVATE_MODE = 0o700
# Linux copies the shebang line into a fixed kernel buffer and silently truncates the rest,
# so an interpreter path that does not fit must fail loudly here instead of producing a
# launcher that executes something else.
_MAX_SHEBANG_BYTES = 120
_GROUP_POLL_SECONDS = 0.02
_LAUNCHER_PREFIX = "claude-launcher-"
_LAUNCHER_SOURCE = '''#!{shebang}
"""Give this process its own session, then become the real Claude Code executable.

Written by provider_runtime.agent_runtime; see _claude_launcher.py. `execv` keeps the pid,
so the process the Agent SDK is holding is the leader of the group every Claude Code
descendant lands in.
"""

import os
import sys

_EXECUTABLE = {executable}


def main() -> None:
    try:
        os.setsid()
    except OSError:
        # POSIX defines the only failure as EPERM, raised when the caller already leads a
        # process group -- which is the state this call exists to reach. The pid is the pgid
        # either way, so the caller's killpg stays exact and the launch must not abort.
        pass
    os.execv(_EXECUTABLE, [_EXECUTABLE, *sys.argv[1:]])


main()
'''


def ensure_claude_launcher(state_root: Path, executable: str, *, interpreter: str) -> Path:
    """Materialize the launcher for one resolved `claude` executable and return its path.

    The launcher deliberately does **not** live inside `state_root`: that directory is the
    child's own `CLAUDE_CONFIG_DIR`, and its `home` subdirectory is the child's `HOME`, both
    of which Claude Code writes to by design. A launcher there would be a file a sandboxed
    session could rewrite and the next launch would execute it *outside* the sandbox. Its
    parent — the runtime-owned `<state_root_base>/claude` directory that
    `AgentRuntime._secure_state_root` creates, chmods to 0700, and mode-asserts before any
    adapter runs — is named in no child environment variable and is inside no sandbox root,
    so it is the nearest place the child cannot reach. That the directory really is private
    is re-checked here rather than assumed.

    The file name is a digest of the content, so repeated calls and concurrent runtimes
    converge on one identical file instead of racing to overwrite each other's, and the
    content is verified after the rename rather than trusted from the write. For the same
    reason it is not deleted on close: another runtime may be about to exec the identical
    file, it holds nothing secret, and rewriting it on every open is what keeps it correct.
    """
    if not interpreter or "\0" in interpreter or "\n" in interpreter:
        raise ExecutableUnavailable(
            "the running Python interpreter has no usable path to build a Claude launcher"
        )
    if not executable or "\0" in executable or "\n" in executable:
        raise ExecutableUnavailable("the Claude Code executable path cannot be launched")
    shebang = f"{interpreter} -I"
    if len(f"#!{shebang}".encode()) > _MAX_SHEBANG_BYTES:
        raise ExecutableUnavailable(
            "the running Python interpreter path is too long for a launcher shebang"
        )
    directory = launcher_directory(state_root)
    _require_private_directory(directory)
    source = _LAUNCHER_SOURCE.format(shebang=shebang, executable=repr(executable)).encode("utf-8")
    path = directory / f"{_LAUNCHER_PREFIX}{hashlib.sha256(source).hexdigest()[:32]}"
    _write_private_file(path, source)
    return path


def launcher_directory(state_root: Path) -> Path:
    return state_root.parent


def _require_private_directory(directory: Path) -> None:
    try:
        status = os.stat(directory, follow_symlinks=False)
    except OSError:
        raise ExecutableUnavailable(
            "the runtime-owned directory for the Claude launcher is unavailable"
        ) from None
    if (
        not stat.S_ISDIR(status.st_mode)
        or stat.S_IMODE(status.st_mode) != _PRIVATE_MODE
        or status.st_uid != os.geteuid()
    ):
        raise AgentRuntimeDefect(
            "the runtime-owned directory for the Claude launcher is not privately permissioned",
            code="launcher_directory_unsafe",
        )


def _write_private_file(path: Path, source: bytes) -> None:
    temporary: str | None = None
    try:
        # A unique temporary name per writer: the rename is what publishes the launcher, so
        # no reader can ever see a partial one and no second writer can truncate this one.
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            # The mode `mkstemp` sets is 0600 and the umask can only narrow it; `fchmod`
            # states the exact mode the launcher must have to be executable by its owner.
            os.fchmod(descriptor, _PRIVATE_MODE)
            os.write(descriptor, source)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise ExecutableUnavailable("the Claude Code launcher could not be written") from None
    try:
        status = os.stat(path, follow_symlinks=False)
        written = path.read_bytes()
    except OSError:
        raise ExecutableUnavailable("the Claude Code launcher could not be verified") from None
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_IMODE(status.st_mode) != _PRIVATE_MODE
        or status.st_uid != os.geteuid()
        or written != source
    ):
        raise AgentRuntimeDefect(
            "the Claude Code launcher on disk is not the one this runtime wrote",
            code="launcher_tampered",
        )


class OwnedProcessGroup:
    """One launched Claude Code child and every descendant that shares its group.

    Adoption pins the pid with a pidfd before reading the group, so the number this group is
    keyed by cannot be recycled between the check and the signal — the leader is reaped by
    the SDK, not by us, so without the pin `killpg` could reach a stranger. The pin, the
    signal, and the membership probe are `_process.ProcessGroup`'s; what this class adds is
    the adoption check that the pid really leads its own group and the escalation the SDK
    lane's teardown needs.
    """

    __slots__ = ("_group", "_lock", "_released")

    def __init__(self, group: ProcessGroup) -> None:
        self._group = group
        self._released = False
        self._lock = asyncio.Lock()

    @classmethod
    async def adopt(cls, pid: int, *, timeout_seconds: float) -> OwnedProcessGroup:
        """Take ownership of the group `pid` leads, or refuse to pretend we own one.

        The wait is part of adoption rather than a caller's problem: spawning returns as
        soon as the *launcher* has been exec'd, and the launcher only reaches `setsid()`
        after its interpreter has started, so a group read immediately after the spawn can
        still see the inherited one. The pid is pinned before the first read, so nothing
        observed during the wait can be a recycled stranger.
        """
        group = ProcessGroup.pin(pid)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            try:
                leader = os.getpgid(pid)
            except ProcessLookupError:
                group.release()
                raise SdkUnavailable(
                    "Claude Code exited before its process group could be adopted"
                ) from None
            except OSError:
                group.release()
                raise ProtocolDefect(
                    "the launched Claude Code process group could not be read",
                    code="process_group_missing",
                ) from None
            if leader == pid:
                return cls(group)
            if loop.time() >= deadline:
                group.release()
                raise ProtocolDefect(
                    "the launched Claude Code child does not lead its own process group",
                    code="process_group_missing",
                )
            await asyncio.sleep(_GROUP_POLL_SECONDS)

    async def terminate(self, *, grace_seconds: float) -> None:
        """Signal the whole owned group, escalating to SIGKILL once the grace is spent.

        What this establishes is that no member of the group is still running: SIGKILL
        cannot be caught. It deliberately does not claim to have *reaped* them — every
        member except the leader is someone else's child, so only the leader's own parent
        (the SDK) and init can reap, and `kill(pgid, 0)` cannot tell a zombie from a live
        process. Waiting for an unreapable zombie to disappear would turn a clean teardown
        into a timeout on any host whose PID 1 does not reap.
        """
        async with self._lock:
            if self._released:
                return
            try:
                if not self._group.signal(signal.SIGTERM):
                    return
                loop = asyncio.get_running_loop()
                deadline = loop.time() + max(grace_seconds, 0.0)
                while loop.time() < deadline:
                    await asyncio.sleep(_GROUP_POLL_SECONDS)
                    if not self._group.exists():
                        return
                self._group.signal(signal.SIGKILL)
            finally:
                self.release()

    def release(self) -> None:
        """Drop the pin without signalling; for a group already known to be gone."""
        if self._released:
            return
        self._released = True
        self._group.release()


__all__ = ["OwnedProcessGroup", "ensure_claude_launcher", "launcher_directory"]
