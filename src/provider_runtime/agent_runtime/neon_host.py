"""Private native inference endpoints for the paid Gemini Developer API."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import logging
import math
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

from provider_runtime.engines._common import retry_after_seconds
from provider_runtime.engines.gemini_generate import GeminiGenerateEngine, _retry_after
from provider_runtime.registry import _resolve
from provider_runtime.types import Present, ProviderCredential

from ._limits import (
    _MAX_MESSAGE_ITEMS,
    _MAX_TURN_OUTPUT_BYTES,
    OutputLimitExceeded,
    bounded_payload_size,
)
from .neon_config import NEON_MAX_OUTPUT_TOKENS, NEON_MODEL, _origin, materialize_neon_catalog
from .neon_control import (
    NeonChildFacts,
    NeonControlError,
    NeonFence,
    NeonFirstSampleRefusal,
    NeonProbe,
    _read,
    _valid_retry_after,
    _valid_sdk_bearer,
    _write,
)
from .neon_responses import NativeResponses, NativeSample, NeonProtocolError

_MAX_WIRE_BYTES = 32 * 1024 * 1024
_GEMINI_ORIGIN = "https://generativelanguage.googleapis.com"
_GEMINI_GENERATE = f"/v1beta/models/{NEON_MODEL}:generateContent"
_GEMINI_STREAM = f"/v1beta/models/{NEON_MODEL}:streamGenerateContent"


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
    announced_delivery: bool = False
    turn: str | None = None
    opened_samples: int = 0
    refusal: NeonFirstSampleRefusal | None = None
    closed: bool = False
    sdk_bearer: str | None = field(default=None, repr=False)
    capacity: NeonProbe | None = None

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
        sdk_transport: httpx.AsyncClient,
        sdk_api_key: str,
    ) -> None:
        self._bearer = f"Bearer {bearer}"
        self._model_version = model_version
        self._codec = NativeResponses(continuation_key, project_id + "/" + model_version)
        self._sample = sample
        self._sdk_transport = sdk_transport
        self._sdk_api_key = sdk_api_key
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
            operation = command.pop("type", None)
            if operation not in ("arm", "arm_sdk") or self._armed is not None:
                await _write(writer, {"type": "rejected"})
                return
            sdk_bearer = turn_id = None
            if operation == "arm_sdk":
                sdk_bearer = command.pop("bearer", None)
                turn_id = command.pop("turn_id", None)
                if (
                    not _valid_sdk_bearer(sdk_bearer)
                    or f"Bearer {sdk_bearer}" == self._bearer
                    or not isinstance(turn_id, str)
                    or not 0 < len(turn_id) <= 128
                ):
                    await _write(writer, {"type": "rejected"})
                    return
            tree = _Armed(
                NeonFence(**command), writer, self._codec, turn=turn_id, sdk_bearer=sdk_bearer
            )
            self._armed = tree
            await _write(writer, {"type": "armed"})
            while True:
                command = await _read(reader)
                if command == {"type": "disarm"}:
                    tree.close()
                    if tree.refusal is not None:
                        final = {
                            "type": "first_sample_refused",
                            **asdict(tree.fence),
                            **asdict(tree.refusal),
                        }
                    else:
                        final = {
                            "type": "disarmed",
                            **asdict(tree.fence),
                            "turn_id": tree.turn,
                            "upstream_entries": tree.opened_samples
                            if tree.sdk_bearer is not None
                            else None,
                            "capacity": asdict(tree.capacity)
                            if tree.capacity is not None
                            else None,
                        }
                    await _write(writer, final)
                    return
                if (
                    tree.turn is None
                    or not tree.announced_delivery
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
        if tree is None or tree.sdk_bearer is not None or tree.closed or tree.refusal is not None:
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
                    tree.announced_delivery = True
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
                retry = _retry_after(error) if refusal == "rate_limited" else None
                retry_after = (
                    retry.value
                    if isinstance(retry, Present) and _valid_retry_after(retry.value)
                    else None
                )
                tree.capacity = NeonProbe(refusal, retry_after)
                if (
                    tree.opened_samples == 1
                    and len(tree.parents) == 1
                    and not tree.ack.is_set()
                    and thread == tree.fence.root_thread_id
                    and metadata["request_kind"] == "turn"
                    and tree.turn is not None
                ):
                    tree.refusal = NeonFirstSampleRefusal(tree.turn, refusal, retry_after)
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

    async def sdk_samples(self, request: web.Request) -> web.StreamResponse:
        tree = self._armed
        # The pinned SDK sends GeminiAPIEndpoint.api_key in this header.
        # That value authenticates the enclosed attempt, not an individual child.
        key = request.headers.getall("x-goog-api-key", [])
        if (
            tree is None
            or tree.sdk_bearer is None
            or len(key) != 1
            or not hmac.compare_digest(key[0].encode(), tree.sdk_bearer.encode())
        ):
            return _sdk_error(401)
        if tree.closed or tree.refusal is not None:
            return _sdk_error(403)
        if self._model_version != NEON_MODEL:
            return _sdk_error(502)

        stream = request.path == _GEMINI_STREAM
        query = (("alt", "sse"),) if stream else ()
        if tuple(request.query.items()) != query or request.content_type != "application/json":
            return _sdk_error(400)
        try:
            payload = await request.read()
            _validate_sdk_request(payload)
        except (ValueError, RecursionError, OutputLimitExceeded):
            return _sdk_error(400)
        if self._armed is not tree or tree.closed or tree.refusal is not None:
            return _sdk_error(403)
        if tree.opened_samples and not tree.ack.is_set():
            return _sdk_error(409)

        turn = tree.turn
        assert turn is not None
        tree.opened_samples += 1
        operation = asyncio.current_task()
        assert operation is not None
        tree.incoming.add(operation)
        response = None
        try:
            async with self._sdk_transport.stream(
                "POST",
                _GEMINI_ORIGIN + (_GEMINI_STREAM if stream else _GEMINI_GENERATE),
                params=query,
                content=payload,
                headers={"Content-Type": "application/json", "x-goog-api-key": self._sdk_api_key},
                timeout=600,
                follow_redirects=False,
            ) as upstream:
                received = bytearray()
                async for chunk in upstream.aiter_bytes():
                    if len(received) + len(chunk) > _MAX_TURN_OUTPUT_BYTES:
                        raise ValueError("Gemini response exceeds wire bound")
                    received.extend(chunk)
                body = bytes(received)
                if upstream.status_code in (402, 429):
                    error = _strict_json(body).get("error")
                    if not isinstance(error, dict) or error.get("code") != upstream.status_code:
                        raise ValueError("unverified Gemini capacity response")
                    retry_after = retry_after_seconds(upstream.headers)
                    reason = "prepaid_depleted" if upstream.status_code == 402 else "rate_limited"
                    retry = (
                        retry_after.value
                        if reason == "rate_limited"
                        and isinstance(retry_after, Present)
                        and _valid_retry_after(retry_after.value)
                        else None
                    )
                    tree.capacity = NeonProbe(reason, retry)
                    if tree.opened_samples == 1 and not tree.ack.is_set():
                        tree.refusal = NeonFirstSampleRefusal(turn, reason, retry)
                    else:
                        tree.incoming.discard(operation)
                        tree.close()
                    return _sdk_error(upstream.status_code)
                upstream.raise_for_status()
                content_type = "text/event-stream" if stream else "application/json"
                if (
                    upstream.headers.get("Content-Type", "").split(";", 1)[0].strip()
                    != content_type
                ):
                    raise ValueError("unqualified Gemini content type")
                _validate_sdk_response(body, stream=stream, model_version=self._model_version)

            if not tree.ack.is_set():
                tree.announced_delivery = True
                await _write(
                    tree.control,
                    {"type": "may_have_delivered", **asdict(tree.fence), "turn_id": turn},
                )
                await asyncio.wait_for(tree.ack.wait(), timeout=30)
            if tree.closed:
                return _sdk_error(403)
            response = web.StreamResponse(headers={"Content-Type": content_type})
            response.content_length = len(body)
            await response.prepare(request)
            if not tree.closed:
                await response.write(body)
                await response.write_eof()
            return response
        except asyncio.CancelledError:
            tree.incoming.discard(operation)
            tree.close()
            raise
        except (httpx.HTTPError, OSError, ValueError, RecursionError, OutputLimitExceeded):
            tree.incoming.discard(operation)
            tree.close()
            if response is not None:
                raise
            return _sdk_error(502)
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


def _strict_json(payload: bytes) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError("Gemini JSON repeats a field")
        return value

    def invalid_constant(_value: str) -> None:
        raise ValueError("Gemini JSON has a non-finite number")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Gemini JSON has a non-finite number")
        return result

    value = json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=invalid_constant,
        parse_float=finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError("Gemini JSON is not an object")
    return value


def _validate_sdk_request(payload: bytes) -> None:
    request = _strict_json(payload)
    bounded_payload_size(request, _MAX_WIRE_BYTES, max_items=_MAX_MESSAGE_ITEMS)
    generation = request.get("generationConfig")
    thinking = generation.get("thinkingConfig") if isinstance(generation, dict) else None
    session_id = request.get("sessionId")
    if (
        request.keys()
        - {
            "contents",
            "generationConfig",
            "serviceTier",
            "sessionId",
            "systemInstruction",
            "toolConfig",
            "tools",
        }
        or request.get("serviceTier") != "standard"
        or not isinstance(session_id, str)
        or not re.fullmatch(r"-?[0-9]{1,20}", session_id)
        or not isinstance(generation, dict)
        or generation.keys() != {"maxOutputTokens", "thinkingConfig"}
        or type(generation["maxOutputTokens"]) is not int
        or not 0 < generation["maxOutputTokens"] <= NEON_MAX_OUTPUT_TOKENS
        or not isinstance(thinking, dict)
        or thinking.keys() != {"thinkingLevel", "includeThoughts"}
        or thinking["thinkingLevel"] != "high"
        or thinking["includeThoughts"] is not True
    ):
        raise ValueError("unqualified SDK Gemini settings")
    contents = request.get("contents")
    if not isinstance(contents, list) or not contents:
        raise ValueError("SDK Gemini contents are required")
    for content in contents:
        if (
            not isinstance(content, dict)
            or content.keys() != {"role", "parts"}
            or content["role"] not in ("user", "model")
            or not isinstance(content["parts"], list)
            or not content["parts"]
        ):
            raise ValueError("unqualified SDK Gemini content")
        for part in content["parts"]:
            if (
                not isinstance(part, dict)
                or not part
                or part.keys()
                - {
                    "text",
                    "thought",
                    "thoughtSignature",
                    "inlineData",
                    "functionCall",
                    "functionResponse",
                }
            ):
                raise ValueError("unqualified SDK Gemini part")
            result = part.get("functionResponse")
            if result is not None and (
                not isinstance(result, dict) or result.keys() - {"name", "id", "response"}
            ):
                raise ValueError("unqualified SDK Gemini function response")
    instruction = request.get("systemInstruction")
    if instruction is not None and (
        not isinstance(instruction, dict)
        or instruction.keys() - {"role", "parts"}
        or not isinstance(instruction.get("parts"), list)
        or any(
            not isinstance(part, dict) or part.keys() != {"text"} for part in instruction["parts"]
        )
    ):
        raise ValueError("unqualified SDK Gemini system instruction")
    tools = request.get("tools", [])
    if not isinstance(tools, list) or any(
        not isinstance(tool, dict)
        or tool.keys() != {"functionDeclarations"}
        or not isinstance(tool["functionDeclarations"], list)
        for tool in tools
    ):
        raise ValueError("hosted Gemini tools are disabled")
    config = request.get("toolConfig", {})
    if not isinstance(config, dict) or config.keys() - {"functionCallingConfig"}:
        raise ValueError("unqualified SDK Gemini tool config")


def _validate_sdk_response(body: bytes, *, stream: bool, model_version: str) -> None:
    frames: list[bytes]
    if stream:
        normalized = body.replace(b"\r\n", b"\n")
        if not normalized.endswith(b"\n\n"):
            raise ValueError("incomplete Gemini event stream")
        frames = []
        for block in normalized.split(b"\n\n"):
            lines = [line for line in block.splitlines() if not line.startswith(b":")]
            if not lines:
                continue
            if any(not line.startswith(b"data:") for line in lines):
                raise ValueError("unexpected Gemini event stream field")
            frames.append(b"\n".join(line[5:].removeprefix(b" ") for line in lines))
    else:
        frames = [body]
    witnessed_model = finished = saw_parts = saw_call = False
    for frame in frames:
        value = _strict_json(frame)
        if "error" in value or ("modelVersion" in value and value["modelVersion"] != model_version):
            raise ValueError("unqualified Gemini response")
        witnessed_model |= value.get("modelVersion") == model_version
        candidates = value.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) > 1 or (finished and candidates):
            raise ValueError("conflicting Gemini candidates")
        if candidates:
            candidate = candidates[0]
            if not isinstance(candidate, dict) or candidate.get("finishReason") not in (
                None,
                "STOP",
            ):
                raise ValueError("Gemini candidate did not complete")
            finished = candidate.get("finishReason") == "STOP"
            content = candidate.get("content")
            if content is None:
                continue
            parts = content.get("parts", []) if isinstance(content, dict) else None
            if not isinstance(parts, list):
                raise ValueError("invalid Gemini parts")
            for part in parts:
                if not isinstance(part, dict) or not part:
                    raise ValueError("invalid Gemini part")
                if not saw_call and "functionCall" in part:
                    signature = part.get("thoughtSignature")
                    if not isinstance(signature, str) or not signature:
                        raise ValueError("Gemini tool call lacks its signature")
                    saw_call = True
                saw_parts = True
    if not witnessed_model or not finished or not saw_parts:
        raise ValueError("Gemini model or completion was not attested")


def _sdk_error(code: int) -> web.Response:
    status = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        402: "RESOURCE_EXHAUSTED",
        403: "PERMISSION_DENIED",
        409: "ABORTED",
        429: "RESOURCE_EXHAUSTED",
        502: "UNAVAILABLE",
    }[code]
    return web.json_response(
        {"error": {"code": code, "status": status, "message": "Neon sample unavailable"}},
        status=code,
    )


def _application(adapter: _Adapter) -> web.Application:
    application = web.Application(client_max_size=_MAX_WIRE_BYTES)
    application.router.add_post("/v1/responses", adapter.responses)
    application.router.add_post(_GEMINI_GENERATE, adapter.sdk_samples)
    application.router.add_post(_GEMINI_STREAM, adapter.sdk_samples)
    return application


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
    async with httpx.AsyncClient(
        trust_env=False, transport=httpx.AsyncHTTPTransport(retries=0, trust_env=False)
    ) as transport:
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
            sdk_transport=transport,
            sdk_api_key=api_key,
        )
        runner = web.AppRunner(_application(adapter), access_log=None, handler_cancellation=True)
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
