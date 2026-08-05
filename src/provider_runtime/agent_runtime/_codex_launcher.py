"""Environment replacement and process-group cleanup for the Codex SDK child.

``openai-codex`` 0.144.4 exposes ``CodexConfig.env`` and ``CodexConfig.codex_bin``, but the
SDK overlays ``env`` onto ``os.environ`` rather than replacing the process environment and
starts its child in the host process group.  Pointing the public ``codex_bin`` option at this
launcher closes both gaps without reproducing the SDK's argv: the launcher forwards every
argument unchanged to the matched bundled runtime, supplies only the runtime-approved
environment names, and supervises one private process group.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path

from .errors import AgentRuntimeDefect, ExecutableUnavailable

_PRIVATE_MODE = 0o700
_MAX_SHEBANG_BYTES = 120
_LAUNCHER_PREFIX = "codex-launcher-"
_LAUNCHER_SOURCE = '''#!{shebang}
"""Supervise the matched Codex runtime with a replaced environment.

Written by provider_runtime.agent_runtime; see _codex_launcher.py. The official SDK owns
the argument vector and stdio protocol. This launcher forwards both unchanged.
"""

import os
import signal
import subprocess
import sys
import time

_EXECUTABLE = {executable}
_ENVIRONMENT_NAMES = {environment_names}
_ENVIRONMENT_READY_ARGUMENT = "--provider-runtime-codex-environment-ready-v1"
_TERMINATION_GRACE_SECONDS = 0.25
_child = None


def _terminate_group(*_args) -> None:
    # The launcher is part of the group. Ignore TERM in this process, deliver it to every
    # child, allow a short grace, then kill the whole group including this supervisor. The
    # SDK observes its owned pid exit while no tool or MCP descendant can survive it.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except ProcessLookupError:
        raise SystemExit(0) from None
    time.sleep(_TERMINATION_GRACE_SECONDS)
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main() -> None:
    global _child
    clean_environment = {{
        name: os.environ[name] for name in _ENVIRONMENT_NAMES if name in os.environ
    }}
    if len(sys.argv) < 2 or sys.argv[1] != _ENVIRONMENT_READY_ARGUMENT:
        # CodexConfig.env is an overlay, so the first interpreter was itself started with the
        # ambient host environment. Immediately exec this trusted launcher again with the
        # selected environment. The SDK keeps the same pid, while the long-lived supervisor
        # and its /proc environment no longer retain ambient credentials.
        os.execve(
            sys.executable,
            [
                sys.executable,
                "-I",
                os.path.abspath(__file__),
                _ENVIRONMENT_READY_ARGUMENT,
                *sys.argv[1:],
            ],
            clean_environment,
        )
    forwarded_arguments = sys.argv[2:]
    os.setsid()
    signal.signal(signal.SIGTERM, _terminate_group)
    _child = subprocess.Popen(
        [_EXECUTABLE, *forwarded_arguments],
        env=clean_environment,
        stdin=None,
        stdout=None,
        stderr=None,
        close_fds=True,
    )
    _child.wait()
    _terminate_group()


main()
'''


def ensure_codex_launcher(
    state_root: Path,
    executable: Path,
    environment_names: tuple[str, ...],
    *,
    interpreter: str,
) -> Path:
    """Materialize one content-addressed launcher outside the child-writable profile root."""
    if not interpreter or "\0" in interpreter or "\n" in interpreter:
        raise ExecutableUnavailable(
            "the running Python interpreter has no usable path to build a Codex launcher"
        )
    executable_text = str(executable)
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
        or "\0" in executable_text
        or "\n" in executable_text
    ):
        raise ExecutableUnavailable("the bundled Codex runtime cannot be launched")
    if len(environment_names) != len(set(environment_names)) or any(
        not isinstance(name, str) or not name or "\0" in name or "\n" in name
        for name in environment_names
    ):
        raise AgentRuntimeDefect(
            "the Codex launcher environment allowlist is malformed",
            code="launcher_environment_invalid",
        )
    shebang = f"{interpreter} -I"
    if len(f"#!{shebang}".encode()) > _MAX_SHEBANG_BYTES:
        raise ExecutableUnavailable(
            "the running Python interpreter path is too long for a Codex launcher shebang"
        )

    directory = state_root.parent
    _require_private_directory(directory)
    source = _LAUNCHER_SOURCE.format(
        shebang=shebang,
        executable=repr(executable_text),
        environment_names=repr(tuple(sorted(environment_names))),
    ).encode("utf-8")
    path = directory / f"{_LAUNCHER_PREFIX}{hashlib.sha256(source).hexdigest()[:32]}"
    _write_private_file(path, source)
    return path


def _require_private_directory(directory: Path) -> None:
    try:
        status = os.stat(directory, follow_symlinks=False)
    except OSError:
        raise ExecutableUnavailable(
            "the runtime-owned directory for the Codex launcher is unavailable"
        ) from None
    if (
        not stat.S_ISDIR(status.st_mode)
        or stat.S_IMODE(status.st_mode) != _PRIVATE_MODE
        or status.st_uid != os.geteuid()
    ):
        raise AgentRuntimeDefect(
            "the runtime-owned directory for the Codex launcher is not privately permissioned",
            code="launcher_directory_unsafe",
        )


def _write_private_file(path: Path, source: bytes) -> None:
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(descriptor, _PRIVATE_MODE)
            os.write(descriptor, source)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise ExecutableUnavailable("the Codex launcher could not be written") from None
    try:
        status = os.stat(path, follow_symlinks=False)
        written = path.read_bytes()
    except OSError:
        raise ExecutableUnavailable("the Codex launcher could not be verified") from None
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_IMODE(status.st_mode) != _PRIVATE_MODE
        or status.st_uid != os.geteuid()
        or written != source
    ):
        raise AgentRuntimeDefect(
            "the Codex launcher on disk is not the one this runtime wrote",
            code="launcher_tampered",
        )


__all__ = ["ensure_codex_launcher"]
