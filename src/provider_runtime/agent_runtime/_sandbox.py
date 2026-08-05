"""Host capability probes shared by native agent sandboxes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from ._process import ProcessLimits, capture_process_output
from .errors import ExecutableUnavailable

_PROBE_LIMITS = ProcessLimits(max_stderr_bytes=64 * 1024, termination_grace_seconds=0.25)
_PROBE_TIMEOUT_SECONDS = 5.0


def environment_executable(name: str, environment: Mapping[str, str]) -> str | None:
    """Resolve an executable against the exact child environment, without ambient PATH."""
    for entry in environment.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


async def bubblewrap_network_namespace_available(
    *, cwd: Path, environment: Mapping[str, str]
) -> bool:
    """Return whether bubblewrap can create the namespace used by restricted sandboxes."""
    bubblewrap = environment_executable("bwrap", environment)
    if bubblewrap is None:
        return False
    try:
        await capture_process_output(
            (bubblewrap, "--ro-bind", "/", "/", "--unshare-net", "/bin/true"),
            cwd=cwd,
            environment=environment,
            limits=_PROBE_LIMITS,
            startup_timeout_seconds=_PROBE_TIMEOUT_SECONDS,
            max_stdout_bytes=1,
            executable_label="bubblewrap",
            purpose="sandbox capability discovery",
        )
    except ExecutableUnavailable:
        return False
    return True
