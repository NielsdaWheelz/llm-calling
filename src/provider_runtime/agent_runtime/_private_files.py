"""One owner for publishing and tamper-checking the runtime's private launcher files.

Both launchers publish an executable the SDK will run on this runtime's behalf, so both
need the identical guarantees: the directory is a 0700 directory owned by this euid, the
file is published by atomic rename so no reader can see a partial one, and what is on disk
after the rename is byte-for-byte what was written. Keeping that sequence in one place is
what makes a change to it — an ownership rule, a mode, an extra check — apply to both.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from .errors import AgentRuntimeDefect, ExecutableUnavailable

PRIVATE_MODE = 0o700


def require_private_directory(directory: Path, *, label: str) -> None:
    try:
        status = os.stat(directory, follow_symlinks=False)
    except OSError:
        raise ExecutableUnavailable(
            f"the runtime-owned directory for the {label} is unavailable"
        ) from None
    if (
        not stat.S_ISDIR(status.st_mode)
        or stat.S_IMODE(status.st_mode) != PRIVATE_MODE
        or status.st_uid != os.geteuid()
    ):
        raise AgentRuntimeDefect(
            f"the runtime-owned directory for the {label} is not privately permissioned",
            code="launcher_directory_unsafe",
        )


def write_private_file(path: Path, source: bytes, *, label: str) -> None:
    temporary: str | None = None
    try:
        # A unique temporary name per writer: the rename is what publishes the file, so no
        # reader can ever see a partial one and no second writer can truncate this one.
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            # The mode `mkstemp` sets is 0600 and the umask can only narrow it; `fchmod`
            # states the exact mode the file must have to be executable by its owner.
            os.fchmod(descriptor, PRIVATE_MODE)
            os.write(descriptor, source)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise ExecutableUnavailable(f"the {label} could not be written") from None
    try:
        status = os.stat(path, follow_symlinks=False)
        written = path.read_bytes()
    except OSError:
        raise ExecutableUnavailable(f"the {label} could not be verified") from None
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_IMODE(status.st_mode) != PRIVATE_MODE
        or status.st_uid != os.geteuid()
        or written != source
    ):
        raise AgentRuntimeDefect(
            f"the {label} on disk is not the one this runtime wrote",
            code="launcher_tampered",
        )


__all__ = ["PRIVATE_MODE", "require_private_directory", "write_private_file"]
