"""Host-only Neon sampling and delivery fence control."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ._limits import _OPERATION_TIMEOUT_SECONDS
from .neon_config import AGY_MAX_INPUT_BYTES


class NeonControlError(RuntimeError):
    pass


def _is_hex256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


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
        *,
        sdk_turn_id: str | None = None,
    ) -> None:
        self._writer = writer
        self._disarm_requested = False
        self._fence = fence
        self._refusal: NeonFirstSampleRefusal | None = None
        self._sdk_turn_id = sdk_turn_id
        self._capacity: NeonProbe | None = None
        self._sdk_upstream_entries: int | None = None
        self._result = asyncio.create_task(self._receive(reader, on_delivery, on_child))
        self._result.add_done_callback(self._close_on_protocol_loss)

    def _close_on_protocol_loss(self, result: asyncio.Task) -> None:
        if result.cancelled() or result.exception() is not None:
            self._writer.close()

    @property
    def first_sample_refusal(self) -> NeonFirstSampleRefusal | None:
        return self._refusal

    @property
    def sdk_upstream_entries(self) -> int | None:
        """Count from the matching SDK fence closure, or unknown."""
        return self._sdk_upstream_entries

    @property
    def capacity(self) -> NeonProbe | None:
        """Authenticated shared funding/throttle fact, independent of replay safety."""
        return self._capacity

    async def _receive(
        self,
        reader: asyncio.StreamReader,
        on_delivery: Callable[[str], Awaitable[None]],
        on_child: Callable[[NeonChildFacts], Awaitable[None]] | None,
    ) -> NeonFirstSampleRefusal | None:
        identity = asdict(self._fence)
        delivered_turn = None
        while True:
            notice = await _read(reader)
            if notice.get("type") == "disarmed":
                entries = notice.get("upstream_entries")
                turn_id = notice.get("turn_id")
                capacity = notice.get("capacity")
                if (
                    notice
                    != {
                        "type": "disarmed",
                        **identity,
                        "turn_id": turn_id,
                        "upstream_entries": entries,
                        "capacity": capacity,
                    }
                    or turn_id is not None
                    and (type(turn_id) is not str or not 0 < len(turn_id) <= 128)
                    or delivered_turn is not None
                    and delivered_turn != turn_id
                    or capacity is not None
                    and turn_id is None
                ):
                    raise NeonControlError("invalid Neon closure evidence")
                if self._sdk_turn_id is None:
                    if entries is not None:
                        raise NeonControlError("unexpected Neon entries in stock closure")
                elif (
                    turn_id != self._sdk_turn_id
                    or type(entries) is not int
                    or entries < 0
                    or entries == 0
                    and (delivered_turn is not None or capacity is not None)
                ):
                    raise NeonControlError("invalid SDK Neon closure evidence")
                if capacity is not None:
                    if not isinstance(capacity, dict) or capacity.keys() != {
                        "status",
                        "retry_after",
                    }:
                        raise NeonControlError("invalid Neon capacity evidence")
                    status = capacity["status"]
                    retry_after = capacity["retry_after"]
                    if (
                        status not in ("prepaid_depleted", "rate_limited")
                        or not _valid_retry_after(retry_after)
                        or status == "prepaid_depleted"
                        and retry_after is not None
                    ):
                        raise NeonControlError("invalid Neon capacity evidence")
                    self._capacity = NeonProbe(**capacity)
                self._sdk_upstream_entries = entries
                return None
            if (
                self._sdk_turn_id is None
                and notice.get("type") == "observed_child"
                and delivered_turn is not None
            ):
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
            if (
                type(turn_id) is not str
                or not turn_id
                or len(turn_id) > 128
                or self._sdk_turn_id is not None
                and turn_id != self._sdk_turn_id
            ):
                raise NeonControlError("Neon control did not identify the native turn")
            essence = {"turn_id": turn_id, **identity}
            if notice == {"type": "may_have_delivered", **essence} and delivered_turn is None:
                delivered_turn = turn_id
                if not self._disarm_requested:
                    await on_delivery(turn_id)
                if not self._disarm_requested:
                    await _write(self._writer, {"type": "delivery_recorded", **essence})
                continue
            reason = notice.get("reason")
            retry_after = notice.get("retry_after")
            if (
                delivered_turn is None
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
                if self._sdk_turn_id is not None:
                    self._sdk_upstream_entries = 1
                self._capacity = NeonProbe(reason, retry_after)
                return NeonFirstSampleRefusal(turn_id, reason, retry_after)
            raise NeonControlError("Neon control fence or order did not match")

    async def close(self) -> None:
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
                self.request_disarm()
                self._refusal = await self._result
        except (OSError, TimeoutError, NeonControlError):
            self._refusal = None
            self._sdk_upstream_entries = None
            self._capacity = None
        finally:
            self._writer.close()
            if not self._result.done():
                self._result.cancel()

    def request_disarm(self) -> None:
        """Revoke sampling now; close collects the original fenced receipt."""
        if self._disarm_requested or self._result.done() or self._writer.is_closing():
            return
        self._disarm_requested = True
        try:
            self._writer.write(b'{"type":"disarm"}\n')
        except OSError:
            self._writer.close()


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
        return await self._arm(fence, {"type": "arm"}, on_delivery, on_child)

    async def arm_sdk(
        self,
        *,
        attempt_id: str,
        owner_epoch: str,
        root_thread_id: str,
        generation: str,
        turn_id: str,
        bearer: str,
        input_bytes: int,
        input_sha256: str,
        on_delivery: Callable[[str], Awaitable[None]],
    ) -> NeonLease:
        """Arm the SDK's separate, attempt-private Gemini ingress."""
        if (
            not _is_hex256(bearer)
            or not _is_hex256(input_sha256)
            or type(input_bytes) is not int
            or not 0 < input_bytes <= AGY_MAX_INPUT_BYTES
            or not isinstance(turn_id, str)
            or not 0 < len(turn_id) <= 128
        ):
            raise NeonControlError("invalid SDK Neon authority")
        fence = NeonFence(attempt_id, owner_epoch, root_thread_id, generation)
        return await self._arm(
            fence,
            {
                "type": "arm_sdk",
                "turn_id": turn_id,
                "bearer": bearer,
                "input_bytes": input_bytes,
                "input_sha256": input_sha256,
            },
            on_delivery,
            None,
        )

    async def _arm(
        self,
        fence: NeonFence,
        command: dict[str, str | int],
        on_delivery: Callable[[str], Awaitable[None]],
        on_child: Callable[[NeonChildFacts], Awaitable[None]] | None,
    ) -> NeonLease:
        async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
            reader, writer = await asyncio.open_unix_connection(self.path)
            try:
                await _write(writer, {**command, **asdict(fence)})
                if await _read(reader) != {"type": "armed"}:
                    raise NeonControlError("Neon adapter did not arm the requested tree")
                sdk_turn_id = command.get("turn_id")
                return NeonLease(
                    reader,
                    writer,
                    fence,
                    on_delivery,
                    on_child,
                    sdk_turn_id=sdk_turn_id if isinstance(sdk_turn_id, str) else None,
                )
            except BaseException:
                writer.close()
                raise
