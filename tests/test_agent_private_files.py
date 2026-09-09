"""The launcher publication preamble both routes share has exactly one owner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from provider_runtime.agent_runtime._claude_launcher import ensure_claude_launcher
from provider_runtime.agent_runtime._private_files import publish_launcher
from provider_runtime.agent_runtime.errors import ExecutableUnavailable


def test_claude_launcher_rejects_an_invalid_interpreter_preamble(tmp_path: Path) -> None:
    backend_root = tmp_path / "backend"
    backend_root.mkdir(mode=0o700)
    state_root = backend_root / "personal"
    state_root.mkdir(mode=0o700)
    executable = tmp_path / "runtime"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)

    def publish(interpreter: str) -> Path:
        return ensure_claude_launcher(state_root, str(executable), interpreter=interpreter)

    with pytest.raises(
        ExecutableUnavailable, match="no usable path to build a Claude Code launcher"
    ):
        publish("/usr/bin/python3\nexec /bin/sh")

    published = publish("/" + "d" * 180 + "/python3")

    assert published.parent == backend_root
    assert published.read_bytes().startswith(b"#!/bin/sh\n'''exec' ")


def test_private_launcher_executes_an_exact_long_quoted_interpreter_path(
    tmp_path: Path,
) -> None:
    """The kernel shebang limit must not constrain an immutable checkout's location."""
    backend_root = tmp_path / "backend"
    backend_root.mkdir(mode=0o700)
    interpreter_directory = tmp_path / ("nested-" + "d" * 150) / "with ' quote"
    interpreter_directory.mkdir(parents=True)
    interpreter = interpreter_directory / "python"
    interpreter.symlink_to(sys.executable)
    marker = tmp_path / "executed"
    launcher = publish_launcher(
        backend_root,
        label="test launcher",
        prefix="test-launcher-",
        template="{preamble}\nfrom pathlib import Path\nPath({marker}).write_text('ready')\n",
        interpreter=str(interpreter),
        fields={"marker": repr(str(marker))},
    )

    completed = subprocess.run(
        (str(launcher),),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == "ready"
