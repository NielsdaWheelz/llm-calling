"""One bounded account-quota read from an already running Codex daemon."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Literal, Never

from pydantic import BaseModel, ConfigDict, field_validator

from .codex_control import CodexControl
from .errors import AgentRuntimeError, InvalidAgentRequest, ProtocolDefect
from .types import CredentialRef


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    profileKey: str
    endpoint: str

    @field_validator("profileKey")
    @classmethod
    def credential_key(cls, value: str) -> str:
        CredentialRef(kind="local_account", profile_key=value)
        return value

    @field_validator("endpoint")
    @classmethod
    def absolute_socket(cls, value: str) -> str:
        if not value.startswith("unix:///") or "\0" in value:
            raise ValueError("an absolute unix endpoint is required")
        return value


def _failure(code: Literal["rejected", "unavailable"]) -> dict:
    return {"ok": False, "error": {"code": code}}


async def _run(request: _Request) -> dict:
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    control = CodexControl(
        {request.profileKey: Path(request.endpoint[7:])},
        is_managed=lambda _: False,
        native_owners=True,
    )
    try:
        # Leave room for the transport's two-second close under the host's
        # authoritative five-second subprocess budget.
        async with asyncio.timeout(2):
            return {"ok": True, "result": await control.usage(request.profileKey)}
    except (
        asyncio.CancelledError,
        TimeoutError,
        AgentRuntimeError,
        ProtocolDefect,
        OSError,
        ValueError,
    ):
        return _failure("unavailable")
    finally:
        try:
            await control.close()
        finally:
            loop.remove_signal_handler(signal.SIGTERM)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _reject_constant(value: str) -> Never:
    raise ValueError("non-finite JSON value")


def main() -> None:
    try:
        encoded = sys.stdin.buffer.read(65537)
        if len(encoded) > 65536:
            raise ValueError("input exceeds bounds")
        request = _Request.model_validate(
            json.loads(encoded, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        )
    except (ValueError, InvalidAgentRequest, UnicodeError, RecursionError):
        result = _failure("rejected")
    else:
        result = asyncio.run(_run(request))
    output = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(output) > 65536:
        result = _failure("unavailable")
        output = json.dumps(result).encode() + b"\n"
    sys.stdout.buffer.write(output)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
