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
import os
import signal
from pathlib import Path

from ._private_files import publish_launcher
from ._process import ProcessGroup
from .errors import ExecutableUnavailable, ProtocolDefect, SdkUnavailable

_LAUNCHER_LABEL = "Claude Code launcher"
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
    is re-checked by `publish_launcher` rather than assumed.

    The launcher is not deleted on close: another runtime may be about to exec the identical
    content-addressed file, it holds nothing secret, and rewriting it on every open is what
    keeps it correct.
    """
    if not executable or "\0" in executable or "\n" in executable:
        raise ExecutableUnavailable("the Claude Code executable path cannot be launched")
    return publish_launcher(
        launcher_directory(state_root),
        label=_LAUNCHER_LABEL,
        prefix=_LAUNCHER_PREFIX,
        template=_LAUNCHER_SOURCE,
        interpreter=interpreter,
        fields={"executable": repr(executable)},
    )


def launcher_directory(state_root: Path) -> Path:
    return state_root.parent


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
