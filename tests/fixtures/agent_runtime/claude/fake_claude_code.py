#!/usr/bin/env python3
"""The local Claude Code executable the SDK double actually spawns.

The Agent SDK lane's cleanup obligation is a property of a real process tree — a child that
leads its own session, and descendants that share it — so the double cannot fake it with an
object carrying a `pid`. This script is what the runtime-owned launcher `execv`s: it answers
the version probe both the adapter and the SDK run, optionally starts a SIGTERM-ignoring
descendant the way a Bash tool or a stdio MCP server would, and otherwise stays alive until
it is signalled.

`--ready-file` is written only after every descendant has installed its signal handler, so a
test that waits for it can never observe the window in which a descendant would die to the
first SIGTERM for reasons that have nothing to do with the code under test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

VERSION = "2.1.220 (Claude Code)"
# Installs the handler first and reports readiness only afterwards; anything that ignores
# SIGTERM is exactly what a group-wide escalation has to survive.
DESCENDANT = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "sys.stdout.write('ready\\n')\n"
    "sys.stdout.flush()\n"
    "time.sleep(300)\n"
)


def option(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def spawn_descendant() -> int:
    child = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", DESCENDANT],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    if child.stdout.readline().strip() != "ready":
        raise RuntimeError("descendant did not report readiness")
    return child.pid


def main() -> None:
    argv = sys.argv[1:]
    if "--version" in argv or argv == ["-v"]:
        print(os.environ.get("LLM_CALLING_FAKE_CLAUDE_VERSION", VERSION))
        return
    descendant = spawn_descendant() if "--spawn-descendant" in argv else 0
    ready = option(argv, "--ready-file")
    if ready is not None:
        Path(ready).write_text(f"{os.getpid()} {os.getpgid(0)} {descendant}\n", encoding="utf-8")
    time.sleep(300)


if __name__ == "__main__":
    main()
