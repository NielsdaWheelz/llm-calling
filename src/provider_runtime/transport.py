"""HTTP/SSE transport for finalized provider requests.

Transport owns auth-header injection, HTTP dispatch, timeouts, raw SSE framing,
and verbatim status/header/error-body passthrough — nothing else (§6). It parses
no provider envelopes: codecs alone interpret bodies and SSE event payloads.

Failure contract: transport-level failures propagate as UNWRAPPED httpx
exceptions (``httpx.TimeoutException``, ``httpx.NetworkError``, and the wider
``httpx.TransportError`` family). The runtime retry boundary classifies them;
transport must not classify, wrap, or retry. HTTP error statuses are likewise
returned verbatim — never raised — for codec ``classify_error`` ingress.

Credential handling: ``_auth_header`` is the ONLY code that touches
``ProviderCredential.key``. The key is injected into request headers and must
never appear in any exception message, repr, or log introduced here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import assert_never

import httpx

from provider_runtime.types import FinalizedProviderRequest, ProviderCredential

# ---------------------------------------------------------------------------
# SSE framing (raw mechanics only; envelope parsing is a codec concern)


@dataclass(frozen=True, slots=True)
class SseEvent:
    """One framed SSE event: the ``event`` field (None when absent) and the
    joined ``data`` payload. ``data: [DONE]`` is deliberately NOT special-cased
    here — terminal sentinels are codec concerns."""

    event: str | None
    data: str


async def _sse_events(byte_stream: AsyncIterator[bytes]) -> AsyncIterator[SseEvent]:
    """Frame a raw byte stream into SSE events per the spec subset providers use.

    - ``event:``/``data:`` field lines accumulate until a blank line dispatches
      one SseEvent; other fields (``id``, ``retry``) are ignored (no consumer).
    - Comment lines starting with ":" are skipped (OpenRouter keep-alives).
    - Multiple ``data:`` lines join with "\\n"; a value's single leading space
      is stripped per spec.
    - A frame with no ``data`` field dispatches nothing (spec behavior).
    - An unterminated trailing frame at end-of-stream is discarded (spec
      behavior; codecs fail closed on a missing terminal event).
    """
    buffer = b""
    event_name: str | None = None
    data_lines: list[str] = []
    async for chunk in byte_stream:
        buffer += chunk
        while (newline_at := buffer.find(b"\n")) != -1:
            raw_line = buffer[:newline_at].removesuffix(b"\r")
            buffer = buffer[newline_at + 1 :]
            line = raw_line.decode("utf-8")
            if not line:
                if data_lines:
                    yield SseEvent(event=event_name, data="\n".join(data_lines))
                event_name = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            field_name, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field_name == "event":
                event_name = value
            elif field_name == "data":
                data_lines.append(value)


# ---------------------------------------------------------------------------
# Response values


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status: int
    headers: Mapping[str, str]  # lower-cased keys
    body: bytes


@dataclass(frozen=True, slots=True)
class TransportStreamResponse:
    status: int
    headers: Mapping[str, str]  # lower-cased keys
    events: AsyncIterator[SseEvent]
    _response: httpx.Response = field(repr=False)

    async def read_error_body(self) -> bytes:
        """Read the raw response body (non-2xx open, before iterating events).

        The runtime hands (status, headers, body) to codec.classify_error."""
        return await self._response.aread()


# ---------------------------------------------------------------------------
# Transport


def _auth_header(credential: ProviderCredential) -> tuple[str, str]:
    """The ONLY credential-touching code in the runtime."""
    match credential.provider:
        case "openai" | "moonshot" | "openrouter":
            return ("Authorization", f"Bearer {credential.key}")
        case "anthropic":
            return ("x-api-key", credential.key)
        case "gemini":
            return ("x-goog-api-key", credential.key)
        case _:
            assert_never(credential.provider)


def _request_headers(
    request: FinalizedProviderRequest, credential: ProviderCredential
) -> dict[str, str]:
    name, value = _auth_header(credential)
    return {"content-type": "application/json", **request.safe_headers, name: value}


def _lowered(headers: httpx.Headers) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


class Transport:
    """POSTs finalized request bytes verbatim; returns responses verbatim.

    No retry, no status raising, no envelope parsing. httpx exceptions
    propagate unwrapped (see module docstring)."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def send(
        self,
        request: FinalizedProviderRequest,
        credential: ProviderCredential,
        timeout_s: float,
    ) -> TransportResponse:
        response = await self._http.request(
            request.method,
            request.url,
            content=request.body,
            headers=_request_headers(request, credential),
            timeout=httpx.Timeout(timeout_s),
        )
        return TransportResponse(
            status=response.status_code,
            headers=_lowered(response.headers),
            body=response.content,
        )

    def stream(
        self,
        request: FinalizedProviderRequest,
        credential: ProviderCredential,
        timeout_s: float,
    ) -> AbstractAsyncContextManager[TransportStreamResponse]:
        return self._stream(request, credential, timeout_s)

    @asynccontextmanager
    async def _stream(
        self,
        request: FinalizedProviderRequest,
        credential: ProviderCredential,
        timeout_s: float,
    ) -> AsyncIterator[TransportStreamResponse]:
        async with self._http.stream(
            request.method,
            request.url,
            content=request.body,
            headers=_request_headers(request, credential),
            timeout=httpx.Timeout(timeout_s),
        ) as response:
            yield TransportStreamResponse(
                status=response.status_code,
                headers=_lowered(response.headers),
                events=_sse_events(response.aiter_bytes()),
                _response=response,
            )
