"""The launcher publication preamble both routes share has exactly one owner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from provider_runtime.agent_runtime._claude_launcher import ensure_claude_launcher
from provider_runtime.agent_runtime._codex_launcher import ensure_codex_launcher
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
def test_both_launchers_answer_to_one_interpreter_and_shebang_rule(
    tmp_path: Path, label: str
) -> None:
    """A shebang rule that held on one route and not the other would be a silent hazard."""
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
    with pytest.raises(ExecutableUnavailable, match=f"too long for a {label} shebang"):
        publish("/" + "d" * 130 + "/python3")

    published = publish("/usr/bin/python3")

    assert published.parent == backend_root
    assert published.read_bytes().startswith(b"#!/usr/bin/python3 -I\n")
