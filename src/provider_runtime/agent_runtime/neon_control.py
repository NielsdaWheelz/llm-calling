"""Host-only Neon sampling and delivery fence control."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ._limits import _OPERATION_TIMEOUT_SECONDS


class NeonControlError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NeonFence:
    attempt_id: str
    owner_epoch: str
    root_thread_id: str
    generation: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value or len(value) > 128
            for value in asdict(self).values()
        ):
            raise NeonControlError("invalid Neon fence identity")


@dataclass(frozen=True, slots=True)
class NeonFirstSampleRefusal:
    turn_id: str
    reason: Literal["prepaid_depleted", "rate_limited"] = "prepaid_depleted"
    retry_after: float | None = None


@dataclass(frozen=True, slots=True)
class NeonChildFacts:
    thread_id: str
    parent_thread_id: str
    turn_id: str
    model_version: str
    effort: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value or len(value) > 128
            for value in asdict(self).values()
        ):
            raise NeonControlError("invalid observed Neon child")


type NeonProbeStatus = Literal["ready", "prepaid_depleted", "rate_limited", "unavailable"]


@dataclass(frozen=True, slots=True)
class NeonProbe:
    status: NeonProbeStatus
    retry_after: float | None = None


async def _read(reader: asyncio.StreamReader) -> dict:
    try:
        line = await reader.readline()
        if not line:
            raise ConnectionError("Neon control channel closed")
        value = json.loads(line)
    except (ValueError, UnicodeError):
        raise NeonControlError("invalid Neon control message") from None
    if not isinstance(value, dict):
        raise NeonControlError("invalid Neon control message")
    return value


async def _write(writer: asyncio.StreamWriter, command: dict) -> None:
    writer.write(json.dumps(command, separators=(",", ":")).encode() + b"\n")
    await writer.drain()


def _valid_retry_after(value: object) -> bool:
    return value is None or (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 31_536_000
    )


class NeonLease:
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        fence: NeonFence,
        on_delivery: Callable[[str], Awaitable[None]],
        on_child: Callable[[NeonChildFacts], Awaitable[None]] | None,
    ) -> None:
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._fence = fence
        self._refusal: NeonFirstSampleRefusal | None = None
        self._result = asyncio.create_task(self._receive(reader, on_delivery, on_child))
        self._result.add_done_callback(self._close_on_protocol_loss)

    def _close_on_protocol_loss(self, result: asyncio.Task) -> None:
        if result.cancelled() or result.exception() is not None:
            self._writer.close()

    @property
    def first_sample_refusal(self) -> NeonFirstSampleRefusal | None:
        return self._refusal

    async def _receive(
        self,
        reader: asyncio.StreamReader,
        on_delivery: Callable[[str], Awaitable[None]],
        on_child: Callable[[NeonChildFacts], Awaitable[None]] | None,
    ) -> NeonFirstSampleRefusal | None:
        identity = asdict(self._fence)
        delivered = False
        while True:
            notice = await _read(reader)
            if notice == {"type": "disarmed"}:
                return None
            if notice.get("type") == "observed_child" and delivered:
                facts = notice.get("facts")
                if not isinstance(facts, dict) or notice != {
                    "type": "observed_child",
                    "facts": facts,
                    **identity,
                }:
                    raise NeonControlError("invalid observed child fence")
                try:
                    child = NeonChildFacts(**facts)
                except TypeError:
                    raise NeonControlError("invalid observed child fields") from None
                if on_child is not None:
                    await on_child(child)
                continue
            turn_id = notice.get("turn_id")
            if type(turn_id) is not str or not turn_id or len(turn_id) > 128:
                raise NeonControlError("Neon control did not identify the native turn")
            essence = {"turn_id": turn_id, **identity}
            if notice == {"type": "may_have_delivered", **essence} and not delivered:
                await on_delivery(turn_id)
                delivered = True
                async with self._write_lock:
                    await _write(self._writer, {"type": "delivery_recorded", **essence})
                continue
            reason = notice.get("reason")
            retry_after = notice.get("retry_after")
            if (
                not delivered
                and reason in ("prepaid_depleted", "rate_limited")
                and (reason != "prepaid_depleted" or retry_after is None)
                and _valid_retry_after(retry_after)
                and notice
                == {
                    "type": "first_sample_refused",
                    "reason": reason,
                    "retry_after": retry_after,
                    **essence,
                }
            ):
                return NeonFirstSampleRefusal(turn_id, reason, retry_after)
            raise NeonControlError("Neon control fence or order did not match")

    async def close(self) -> None:
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                if not self._result.done():
                    async with self._write_lock:
                        await _write(self._writer, {"type": "disarm"})
                self._refusal = await self._result
        except (OSError, NeonControlError):
            self._refusal = None
        finally:
            self._writer.close()
            if not self._result.done():
                self._result.cancel()


@dataclass(frozen=True, slots=True)
class NeonControl:
    path: Path

    async def available(self) -> bool:
        try:
            return await self._request("available") == {"type": "available"}
        except (OSError, NeonControlError):
            return False

    async def probe(self) -> NeonProbe:
        try:
            response = await self._request("probe")
        except (OSError, NeonControlError):
            return NeonProbe("unavailable")
        status = response.get("status")
        if response.get("type") != "probe_result" or status not in (
            "ready",
            "prepaid_depleted",
            "rate_limited",
            "unavailable",
        ):
            return NeonProbe("unavailable")
        retry_after = response.get("retry_after")
        if not _valid_retry_after(retry_after):
            return NeonProbe("unavailable")
        return NeonProbe(status, retry_after)

    async def _request(self, operation: str) -> dict:
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            reader, writer = await asyncio.open_unix_connection(self.path)
            try:
                await _write(writer, {"type": operation})
                return await _read(reader)
            finally:
                writer.close()

    async def arm(
        self,
        *,
        attempt_id: str,
        owner_epoch: str,
        root_thread_id: str,
        generation: str,
        on_delivery: Callable[[str], Awaitable[None]],
        on_child: Callable[[NeonChildFacts], Awaitable[None]] | None = None,
    ) -> NeonLease:
        fence = NeonFence(attempt_id, owner_epoch, root_thread_id, generation)
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            reader, writer = await asyncio.open_unix_connection(self.path)
            try:
                await _write(writer, {"type": "arm", **asdict(fence)})
                if await _read(reader) != {"type": "armed"}:
                    raise NeonControlError("Neon adapter did not arm the requested tree")
                return NeonLease(reader, writer, fence, on_delivery, on_child)
            except BaseException:
                writer.close()
                raise
