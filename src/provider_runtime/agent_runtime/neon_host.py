"""Private stock Codex Responses adapter to the paid Gemini Developer API."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import logging
import os
import re
from collections.abc import AsyncGenerator, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from aiohttp import web
from google.genai import errors as gemini_errors
from google.genai.types import GenerateContentResponse

from provider_runtime.engines.gemini_generate import GeminiGenerateEngine, _retry_after
from provider_runtime.registry import _resolve
from provider_runtime.types import Present, ProviderCredential

from ._limits import _MAX_MESSAGE_ITEMS, _MAX_TURN_OUTPUT_BYTES, bounded_payload_size
from .neon_config import NEON_MODEL, _origin, materialize_neon_catalog
from .neon_control import (
    NeonChildFacts,
    NeonControlError,
    NeonFence,
    NeonFirstSampleRefusal,
    _read,
    _write,
)
from .neon_responses import NativeResponses, NativeSample, NeonProtocolError

_MAX_WIRE_BYTES = 32 * 1024 * 1024


@dataclass(slots=True)
class _Armed:
    fence: NeonFence
    control: asyncio.StreamWriter
    responses: NativeResponses
    incoming: set[asyncio.Task[Any]] = field(default_factory=set)
    parents: dict[str, str | None] = field(default_factory=dict)
    turns: dict[str, set[str]] = field(default_factory=dict)
    observed_children: set[str] = field(default_factory=set)
    frame_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ack: asyncio.Event = field(default_factory=asyncio.Event)
    turn: str | None = None
    opened_samples: int = 0
    refusal: NeonFirstSampleRefusal | None = None
    closed: bool = False

    def sample_thread(self, metadata: object, header_thread: str | None) -> tuple[str, set[str]]:
        if self.closed or not isinstance(metadata, dict):
            raise NeonProtocolError("Neon sample has no active stock fence")
        thread = metadata.get("thread_id")
        turn = metadata.get("turn_id")
        root_turn = metadata.get("root_turn_id")
        parent = metadata.get("parent_thread_id")
        parent_turn = metadata.get("parent_turn_id")
        if (
            not isinstance(thread, str)
            or not isinstance(turn, str)
            or thread != header_thread
            or metadata.get("request_kind") not in {"turn", "compaction"}
        ):
            raise NeonProtocolError("Neon sample has invalid authenticated native metadata")
        if thread == self.fence.root_thread_id:
            if (
                parent is not None
                or root_turn != turn
                or metadata.get("agent_name") != "/root"
                or (self.turn is not None and turn != self.turn)
            ):
                raise NeonProtocolError("Neon root turn changed")
            self.turn = turn
        else:
            if (
                self.turn is None
                or root_turn != self.turn
                or not isinstance(parent, str)
                or not isinstance(parent_turn, str)
                or parent_turn not in self.turns.get(parent, ())
                or parent == thread
                or metadata.get("subagent_kind") != "thread_spawn"
            ):
                raise NeonProtocolError("Neon child lacks trusted stock lineage")
        if thread in self.parents and self.parents[thread] != parent:
            raise NeonProtocolError("Neon native thread changed its parent")
        self.parents[thread] = parent
        self.turns.setdefault(thread, set()).add(turn)
        ancestry = {thread}
        while parent is not None:
            if parent in ancestry:
                raise NeonProtocolError("Neon lineage is not a tree")
            ancestry.add(parent)
            parent = self.parents[parent]
        return thread, ancestry

    def close(self) -> None:
        self.closed = True
        self.ack.set()
        for operation in self.incoming:
            operation.cancel()


class _Adapter:
    def __init__(
        self,
        *,
        bearer: str,
        model_version: str,
        continuation_key: bytes,
        project_id: str,
        sample: Callable[[NativeSample], AsyncGenerator[GenerateContentResponse, None]],
    ) -> None:
        self._bearer = f"Bearer {bearer}"
        self._model_version = model_version
        self._codec = NativeResponses(continuation_key, project_id + "/" + model_version)
        self._sample = sample
        self._armed: _Armed | None = None
        self._probe_lock = asyncio.Lock()

    async def _collect(self, sample: NativeSample) -> list[GenerateContentResponse]:
        chunks = []
        size = 0
        async with contextlib.aclosing(self._sample(sample)) as stream:
            async for chunk in stream:
                size += bounded_payload_size(
                    chunk.model_dump(mode="json", exclude_none=True),
                    _MAX_TURN_OUTPUT_BYTES - size,
                    max_items=_MAX_MESSAGE_ITEMS,
                )
                chunks.append(chunk)
        return chunks

    async def control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        tree = None
        try:
            command = await _read(reader)
            if command == {"type": "available"}:
                await _write(writer, {"type": "available"})
                return
            if command == {"type": "probe"}:
                async with self._probe_lock:
                    await _write(writer, await self._probe())
                return
            if command.pop("type", None) != "arm" or self._armed is not None:
                await _write(writer, {"type": "rejected"})
                return
            tree = _Armed(NeonFence(**command), writer, self._codec)
            self._armed = tree
            await _write(writer, {"type": "armed"})
            while True:
                command = await _read(reader)
                if command == {"type": "disarm"}:
                    tree.close()
                    if tree.refusal is None:
                        final = {"type": "disarmed"}
                    else:
                        final = {
                            "type": "first_sample_refused",
                            **asdict(tree.fence),
                            **asdict(tree.refusal),
                        }
                    await _write(writer, final)
                    return
                if (
                    tree.turn is None
                    or tree.ack.is_set()
                    or command
                    != {
                        "type": "delivery_recorded",
                        **asdict(tree.fence),
                        "turn_id": tree.turn,
                    }
                ):
                    raise NeonControlError("delivery acknowledgement did not match the Neon fence")
                tree.ack.set()
        except (OSError, NeonControlError, TypeError):
            return
        finally:
            if tree is not None:
                tree.close()
                self._armed = None
            writer.close()

    async def _probe(self) -> dict:
        if self._armed is not None:
            return {"type": "probe_result", "status": "unavailable"}
        request = self._codec.request(
            {
                "model": NEON_MODEL,
                "stream": True,
                "store": False,
                "tool_choice": "none",
                "reasoning": {"effort": "low"},
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Reply OK."}],
                    }
                ],
            },
            ancestors=set(),
        )
        request.config.max_output_tokens = 32
        try:
            async with contextlib.aclosing(self._sample(request)) as stream:
                first = await anext(stream, None)
            if first is not None:
                return {
                    "type": "probe_result",
                    "status": "ready"
                    if first.model_version in (None, self._model_version)
                    else "unavailable",
                }
        except gemini_errors.APIError as error:
            if _prepaid(error):
                return {"type": "probe_result", "status": "prepaid_depleted"}
            if isinstance(error.response, httpx.Response) and error.response.status_code == 429:
                retry = _retry_after(error)
                return {
                    "type": "probe_result",
                    "status": "rate_limited",
                    "retry_after": retry.value if isinstance(retry, Present) else None,
                }
        except (ValueError, httpx.HTTPError, RuntimeError):
            pass
        return {"type": "probe_result", "status": "unavailable"}

    async def responses(self, request: web.Request) -> web.StreamResponse:
        if not hmac.compare_digest(
            request.headers.get("Authorization", "").encode(), self._bearer.encode()
        ):
            return _error(401, "invalid_stock_credential")
        tree = self._armed
        if tree is None or tree.closed or tree.refusal is not None:
            return _error(403, "unarmed_stock_generation")

        try:
            raw = await request.json()
            if not isinstance(raw, dict):
                raise NeonProtocolError("native request is not an object")
            bounded_payload_size(raw, _MAX_WIRE_BYTES, max_items=_MAX_MESSAGE_ITEMS)
            metadata = json.loads(request.headers.get("x-codex-turn-metadata", ""))
            thread, ancestor_threads = tree.sample_thread(
                metadata, request.headers.get("thread-id")
            )
            sample = tree.responses.request(raw, ancestors=ancestor_threads)
            if thread == tree.fence.root_thread_id and sample.effort != "high":
                raise NeonProtocolError("Neon root effort changed")
        except NeonProtocolError as error:
            logging.getLogger(__name__).warning("stock sample refused: %s", error)
            return _error(400, "unqualified_stock_request")
        except (ValueError, TypeError, KeyError, RuntimeError):
            return _error(400, "invalid_stock_request")
        if self._armed is not tree or tree.closed or tree.refusal is not None:
            return _error(403, "unarmed_stock_generation")

        tree.opened_samples += 1
        operation = asyncio.current_task()
        assert operation is not None
        tree.incoming.add(operation)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        try:
            await response.prepare(request)
            result = asyncio.create_task(self._collect(sample))
            try:
                while True:
                    done, _ = await asyncio.wait({result}, timeout=10)
                    if done:
                        chunks = result.result()
                        break
                    await response.write(_event({"type": "heartbeat"}))
                items, usage = tree.responses.complete(
                    sample, chunks, model_version=self._model_version, thread=thread
                )
            finally:
                if not result.done():
                    result.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await result
            async with tree.frame_lock:
                if tree.closed:
                    return response
                if not tree.ack.is_set():
                    await _write(
                        tree.control,
                        {
                            "type": "may_have_delivered",
                            **asdict(tree.fence),
                            "turn_id": tree.turn,
                        },
                    )
                    await asyncio.wait_for(tree.ack.wait(), timeout=30)
                if tree.closed:
                    return response
                if thread != tree.fence.root_thread_id and thread not in tree.observed_children:
                    turn = metadata["turn_id"]
                    parent = tree.parents[thread]
                    assert parent is not None
                    await _write(
                        tree.control,
                        {
                            "type": "observed_child",
                            **asdict(tree.fence),
                            "facts": asdict(
                                NeonChildFacts(
                                    thread,
                                    parent,
                                    turn,
                                    self._model_version,
                                    sample.effort,
                                )
                            ),
                        },
                    )
                    tree.observed_children.add(thread)
            issued = {
                "id": f"resp_{uuid4().hex}",
                "model": self._model_version,
                "status": "completed",
                "output": items,
            }
            if usage is not None:
                issued["usage"] = usage
            events = [
                _event({"type": kind, "output_index": index, "item": item})
                for index, item in enumerate(items)
                for kind in ("response.output_item.added", "response.output_item.done")
            ]
            events.append(_event({"type": "response.completed", "response": issued}))
            await response.write(b"".join(events))
            return response
        except gemini_errors.APIError as error:
            refusal = (
                "prepaid_depleted"
                if _prepaid(error)
                else "rate_limited"
                if isinstance(error.response, httpx.Response)
                and error.response.status_code == 429
                and error.code == 429
                else None
            )
            if refusal is not None:
                if (
                    tree.opened_samples == 1
                    and len(tree.parents) == 1
                    and not tree.ack.is_set()
                    and thread == tree.fence.root_thread_id
                    and metadata["request_kind"] == "turn"
                    and tree.turn is not None
                ):
                    retry = _retry_after(error) if refusal == "rate_limited" else None
                    tree.refusal = NeonFirstSampleRefusal(
                        tree.turn,
                        refusal,
                        retry.value if isinstance(retry, Present) else None,
                    )
                await response.write(
                    _failure(
                        "insufficient_quota"
                        if refusal == "prepaid_depleted"
                        else "rate_limit_exceeded"
                    )
                )
            else:
                await response.write(_failure("provider_unavailable"))
            return response
        except NeonProtocolError as error:
            logging.getLogger(__name__).warning("Neon candidate withheld: %s", error)
            await response.write(_failure("unqualified_neon_response"))
            return response
        except (ValueError, httpx.HTTPError, RuntimeError):
            await response.write(_failure("invalid_neon_response"))
            return response
        finally:
            tree.incoming.discard(operation)


def _prepaid(error: gemini_errors.APIError) -> bool:
    # The fixed, enrolled prepaid endpoint documents HTTP 402 as depleted
    # balance. An in-band API error or another HTTP status never supplies that fact.
    return (
        isinstance(error.response, httpx.Response)
        and error.response.status_code == 402
        and error.code == 402
    )


def _event(data: dict) -> bytes:
    return ("data: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode()


def _failure(code: str) -> bytes:
    return _event(
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"code": code, "message": "Neon sample did not complete"},
            },
        }
    )


def _error(status: int, code: str) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": "Neon sample did not enter"}}, status=status
    )


async def _serve(args: argparse.Namespace) -> None:
    origin = urlsplit(_origin(args.origin))
    directory = Path(os.environ["CREDENTIALS_DIRECTORY"])
    api_key = (directory / "gemini-api-key").read_text().strip()
    bearer = (directory / "neon-stock-bearer").read_text().strip()
    continuation_key = (directory / "neon-continuation-key").read_bytes()
    if not api_key or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", bearer):
        raise RuntimeError("Neon adapter credentials are missing or malformed")
    if not args.project_id or not args.model_version:
        raise RuntimeError("Neon adapter requires its enrolled project and served model")
    async with httpx.AsyncClient(trust_env=False) as transport:
        engine = GeminiGenerateEngine(http_client=transport)
        credential = ProviderCredential(provider="gemini", key=api_key)
        row = _resolve(f"gemini:{NEON_MODEL}")

        def sample(request: NativeSample) -> AsyncGenerator[GenerateContentResponse, None]:
            return engine._stream_native_sample(
                row,
                credential,
                contents=request.contents,
                config=request.config,
            )

        adapter = _Adapter(
            bearer=bearer,
            model_version=args.model_version,
            continuation_key=continuation_key,
            project_id=args.project_id,
            sample=sample,
        )
        application = web.Application(client_max_size=_MAX_WIRE_BYTES)
        application.router.add_post("/v1/responses", adapter.responses)
        runner = web.AppRunner(application, access_log=None, handler_cancellation=True)
        await runner.setup()
        try:
            site = web.TCPSite(runner, origin.hostname, origin.port)
            await site.start()
            async with await asyncio.start_unix_server(
                adapter.control, args.control_socket
            ) as control:
                Path(args.control_socket).chmod(0o660)
                await control.serve_forever()
        finally:
            await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-catalog", type=Path)
    for setting in ("origin", "control-socket", "project-id", "model-version"):
        parser.add_argument(f"--{setting}")
    args = parser.parse_args()
    if args.write_catalog is not None:
        print(materialize_neon_catalog(args.write_catalog))
        return
    if not all((args.origin, args.control_socket, args.project_id, args.model_version)):
        parser.error("origin, control socket, enrolled project and model version are required")
    logging.basicConfig()
    logging.getLogger(__package__).setLevel(logging.INFO)
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
