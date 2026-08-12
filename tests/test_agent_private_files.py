"""The launcher publication preamble both routes share has exactly one owner."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from provider_runtime.agent_runtime._claude_launcher import ensure_claude_launcher
from provider_runtime.agent_runtime._codex_launcher import ensure_codex_launcher
from provider_runtime.agent_runtime._private_files import publish_launcher
from provider_runtime.agent_runtime.errors import ExecutableUnavailable


def _routes(state_root: Path, executable: Path) -> dict[str, Callable[[str], Path]]:
    return {
        "Codex launcher": lambda interpreter: ensure_codex_launcher(
            state_root, executable, ("KEEP",), interpreter=interpreter
        ),
        "Claude Code launcher": lambda interpreter: ensure_claude_launcher(
            state_root, str(executable), interpreter=interpreter
        ),
    }


@pytest.mark.parametrize("label", ("Codex launcher", "Claude Code launcher"))
def test_both_launchers_answer_to_one_interpreter_preamble_rule(tmp_path: Path, label: str) -> None:
    """An interpreter rule that held on one route and not the other would be a hazard."""
    backend_root = tmp_path / "backend"
    backend_root.mkdir(mode=0o700)
    state_root = backend_root / "personal"
    state_root.mkdir(mode=0o700)
    executable = tmp_path / "runtime"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    publish = _routes(state_root, executable)[label]

    with pytest.raises(ExecutableUnavailable, match=f"no usable path to build a {label}"):
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
