from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from provider_runtime.agent_runtime._process import (
    ManagedProcess,
    ProcessLimits,
    capture_process_output,
)
from provider_runtime.agent_runtime.errors import ExecutableUnavailable

LIMITS = ProcessLimits(max_stderr_bytes=128, termination_grace_seconds=0.1)


async def first_frame(process: ManagedProcess) -> dict[str, object]:
    line = await process.stdout.readline()
    frame = json.loads(line)
    assert isinstance(frame, dict)
    return frame


async def test_process_uses_argv_without_shell_interpretation(tmp_path: Path) -> None:
    marker = tmp_path / "shell-expanded"
    argument = f"$(touch {marker})"
    process = await ManagedProcess.spawn(
        (
            sys.executable,
            "-c",
            "import json,sys; print(json.dumps({'argument': sys.argv[1]}), flush=True)",
            argument,
        ),
        cwd=tmp_path,
        environment={"PATH": "/usr/bin"},
        limits=LIMITS,
    )

    assert await first_frame(process) == {"argument": argument}
    exit = await process.wait()
    assert exit.termination == "exited"
    assert exit.returncode == 0
    assert not marker.exists()


async def test_process_bounds_stderr_and_terminates_on_overflow(tmp_path: Path) -> None:
    process = await ManagedProcess.spawn(
        (
            sys.executable,
            "-c",
            "import sys,time; sys.stderr.write('x' * 4096); sys.stderr.flush(); time.sleep(10)",
        ),
        cwd=tmp_path,
        environment={},
        limits=ProcessLimits(max_stderr_bytes=32, termination_grace_seconds=0.1),
    )

    exit = await process.wait()
    assert exit.termination == "output_limit"
    assert len(exit.stderr) == 32
    assert "xxxxxxxx" not in repr(exit)
    assert process.returncode is not None


async def test_close_during_wait_terminates_and_reaps_the_entire_group(tmp_path: Path) -> None:
    """Cancellation and the turn timeout both reach the process layer as close()."""
    marker = tmp_path / "orphan-ran"
    grandchild = (
        "import pathlib,sys,time; time.sleep(.25); pathlib.Path(sys.argv[1]).write_text('orphan')"
    )
    parent = (
        "import json,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
        "print(json.dumps({'grandchild_pid': child.pid}),flush=True); time.sleep(10)"
    )
    process = await ManagedProcess.spawn(
        (sys.executable, "-c", parent, grandchild, str(marker)),
        cwd=tmp_path,
        environment={},
        limits=LIMITS,
    )

    assert isinstance((await first_frame(process))["grandchild_pid"], int)
    waiter = asyncio.create_task(process.wait())
    await asyncio.sleep(0.05)
    await process.close()
    exit = await waiter
    await asyncio.sleep(0.35)

    assert exit.termination == "closed"
    assert process.returncode is not None
    assert not marker.exists(), "a terminated process left a live descendant"


async def test_cancelling_the_waiting_task_still_reaps_the_process(tmp_path: Path) -> None:
    process = await ManagedProcess.spawn(
        (sys.executable, "-c", "import time; time.sleep(10)"),
        cwd=tmp_path,
        environment={},
        limits=LIMITS,
    )
    waiter = asyncio.create_task(process.wait())
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert process.returncode is not None


async def test_process_close_is_idempotent_and_reaps(tmp_path: Path) -> None:
    process = await ManagedProcess.spawn(
        (sys.executable, "-c", "import time; time.sleep(10)"),
        cwd=tmp_path,
        environment={},
        limits=LIMITS,
    )

    await process.close()
    await process.close()

    assert process.returncode is not None


async def test_process_close_returns_early_when_term_reaps_the_group(tmp_path: Path) -> None:
    process = await ManagedProcess.spawn(
        (sys.executable, "-c", "import time; time.sleep(10)"),
        cwd=tmp_path,
        environment={},
        limits=ProcessLimits(max_stderr_bytes=128, termination_grace_seconds=1.0),
    )

    started = time.monotonic()
    await process.close()

    assert time.monotonic() - started < 0.5
    assert process.returncode is not None


async def test_process_close_wakes_and_settles_a_concurrent_wait(tmp_path: Path) -> None:
    process = await ManagedProcess.spawn(
        (sys.executable, "-c", "import time; time.sleep(10)"),
        cwd=tmp_path,
        environment={},
        limits=LIMITS,
    )
    waiter = asyncio.create_task(process.wait())
    await asyncio.sleep(0)

    await process.close()
    exit = await waiter

    assert exit.termination == "closed"
    assert process.returncode is not None


async def test_process_close_cancellation_still_reaps_before_propagating(tmp_path: Path) -> None:
    process = await ManagedProcess.spawn(
        (sys.executable, "-c", "import time; time.sleep(10)"),
        cwd=tmp_path,
        environment={},
        limits=LIMITS,
    )
    closing = asyncio.create_task(process.close())
    await asyncio.sleep(0)
    closing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await closing
    assert process.returncode is not None
    await process.close()


@pytest.mark.parametrize(
    ("argv", "cwd"),
    [
        ((), Path("/")),
        ((sys.executable,), Path("relative")),
    ],
)
async def test_process_rejects_invalid_launches_before_spawning(
    argv: tuple[str, ...], cwd: Path
) -> None:
    with pytest.raises(ValueError):
        await ManagedProcess.spawn(argv, cwd=cwd, environment={}, limits=LIMITS)


async def test_process_rejects_invalid_environment_before_spawning(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        await ManagedProcess.spawn(
            (sys.executable,),
            cwd=tmp_path,
            environment={"BAD=NAME": "value"},
            limits=LIMITS,
        )


async def test_owned_process_pins_its_pid_and_releases_the_handle_on_teardown(
    tmp_path: Path,
) -> None:
    """The group id must stay unrecyclable for the process lifetime, and not leak afterwards."""
    open_fds = len(os.listdir("/proc/self/fd"))
    for _ in range(4):
        process = await ManagedProcess.spawn(
            (sys.executable, "-c", "import time; time.sleep(10)"),
            cwd=tmp_path,
            environment={},
            limits=LIMITS,
        )
        await process.close()
        assert process.returncode is not None

    assert len(os.listdir("/proc/self/fd")) <= open_fds + 1


async def test_close_after_the_child_exited_is_safe(tmp_path: Path) -> None:
    process = await ManagedProcess.spawn(
        (sys.executable, "-c", "import sys; sys.exit(0)"),
        cwd=tmp_path,
        environment={},
        limits=LIMITS,
    )

    exit = await process.wait()
    await process.close()

    assert exit.termination == "exited"
    assert exit.returncode == 0


async def test_devnull_stdin_gives_the_child_eof_without_a_release_step(tmp_path: Path) -> None:
    """A child that reads stdin before answering would stall a capture that never writes.

    `stdin="devnull"` closes that window at the spawn itself: there is no interval in which a
    fast child can read a partially written pipe, and no release step an early return can skip.
    `capture_process_output` — the discovery-probe runner — spawns on this mode unconditionally.
    """
    reader = "import sys; sys.stdout.write(repr(sys.stdin.read())); sys.stdout.flush()"
    process = await ManagedProcess.spawn(
        (sys.executable, "-c", reader),
        cwd=tmp_path,
        environment={},
        limits=LIMITS,
        stdin="devnull",
    )
    try:
        async with asyncio.timeout(5):
            output = await process.stdout.read()
            exit = await process.wait()
    finally:
        await process.close()

    assert output == b"''"
    assert exit.termination == "exited"
    with pytest.raises(RuntimeError, match="stdin is closed"):
        await process.send(b"anything")


async def test_pipe_stdin_is_still_available_to_a_transport_that_writes_frames(
    tmp_path: Path,
) -> None:
    """Managed subprocess owners can explicitly retain a writable stdin pipe."""
    echo = "import sys; sys.stdout.write(sys.stdin.readline()); sys.stdout.flush()"
    process = await ManagedProcess.spawn(
        (sys.executable, "-c", echo),
        cwd=tmp_path,
        environment={},
        limits=LIMITS,
    )
    try:
        await process.send(b"frame\n")
        async with asyncio.timeout(5):
            assert await process.stdout.readline() == b"frame\n"
    finally:
        await process.close()


async def test_a_pipe_failure_after_the_spawn_still_reaps_the_discovery_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError raised after the launch must not leave the child running.

    The spawn-failed case and a broken read both surface as `OSError`; only the first has no
    process to release, so the mapping to `ExecutableUnavailable` cannot skip the cleanup.
    """
    spawned: list[ManagedProcess] = []
    original = ManagedProcess.spawn

    async def spawn_then_break_stdout(*args: object, **kwargs: object) -> ManagedProcess:
        process = await original(*cast(Any, args), **cast(Any, kwargs))
        spawned.append(process)

        async def broken_read(_size: int = -1) -> bytes:
            raise OSError("pipe went away")

        monkeypatch.setattr(process.stdout, "read", broken_read)
        return process

    monkeypatch.setattr(ManagedProcess, "spawn", spawn_then_break_stdout)
    with pytest.raises(ExecutableUnavailable, match="sleeper"):
        await capture_process_output(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=tmp_path,
            environment={},
            limits=LIMITS,
            startup_timeout_seconds=5.0,
            max_stdout_bytes=1024,
            executable_label="sleeper",
            purpose="pipe failure probe",
        )

    assert len(spawned) == 1
    assert spawned[0].returncode is not None, "the launched child was left running"


async def test_an_unknown_stdin_mode_is_refused_before_the_spawn(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stdin"):
        await ManagedProcess.spawn(
            (sys.executable, "-c", "pass"),
            cwd=tmp_path,
            environment={},
            limits=LIMITS,
            stdin="inherit",  # type: ignore[arg-type]
        )
