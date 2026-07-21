"""Tests for the HTTP/SSE transport: auth injection, verbatim passthrough,
raw SSE framing, unwrapped httpx failure propagation, credential hygiene."""

from collections.abc import AsyncIterator
from typing import get_args

import httpx
import pytest
import respx

from provider_runtime.transport import (
    SseEvent,
    Transport,
    TransportResponse,
    _sse_events,
)
from provider_runtime.types import (
    FinalizedProviderRequest,
    ProviderCredential,
    ProviderName,
    ProviderTarget,
)

URL = "https://provider.example/v1/generate"
KEY = "sk-secret-test-key-XYZ"
BODY = b'{"model": "m-1", "input": "hello"}'


def make_request() -> FinalizedProviderRequest:
    return FinalizedProviderRequest(
        target=ProviderTarget(provider="openai", model="m-1"),
        protocol="openai_responses",
        url=URL,
        method="POST",
        safe_headers={"anthropic-version": "2023-06-01"},
        body=BODY,
    )


def make_credential(provider: ProviderName = "openai") -> ProviderCredential:
    return ProviderCredential(provider=provider, key=KEY)


async def collect(events: AsyncIterator[SseEvent]) -> list[SseEvent]:
    return [event async for event in events]


class ChunkedByteStream(httpx.AsyncByteStream):
    """Byte stream with caller-controlled chunk boundaries."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def sse_route(*chunks: bytes, status: int = 200) -> respx.Route:
    return respx.post(URL).mock(
        return_value=httpx.Response(
            status,
            headers={"content-type": "text/event-stream"},
            stream=ChunkedByteStream(list(chunks)),
        )
    )


# ---------------------------------------------------------------------------
# send()


@respx.mock
async def test_send_posts_body_and_headers_verbatim() -> None:
    route = respx.post(URL).respond(200, json={"ok": True})
    async with httpx.AsyncClient() as http:
        await Transport(http).send(make_request(), make_credential(), timeout_s=5.0)
    request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == URL
    assert request.content == BODY
    assert request.headers["content-type"] == "application/json"
    assert request.headers["anthropic-version"] == "2023-06-01"


AUTH_HEADER_CASES: dict[ProviderName, tuple[str, str]] = {
    "openai": ("Authorization", f"Bearer {KEY}"),
    "moonshot": ("Authorization", f"Bearer {KEY}"),
    "openrouter": ("Authorization", f"Bearer {KEY}"),
    "anthropic": ("x-api-key", KEY),
    "gemini": ("x-goog-api-key", KEY),
}


def test_auth_header_cases_are_exhaustive_over_provider_names() -> None:
    assert set(AUTH_HEADER_CASES) == set(get_args(ProviderName.__value__))


@pytest.mark.parametrize(("provider", "expected"), AUTH_HEADER_CASES.items())
@respx.mock
async def test_send_injects_auth_header_per_provider(
    provider: ProviderName, expected: tuple[str, str]
) -> None:
    route = respx.post(URL).respond(200, json={})
    async with httpx.AsyncClient() as http:
        await Transport(http).send(make_request(), make_credential(provider), timeout_s=5.0)
    name, value = expected
    assert route.calls.last.request.headers[name] == value


@respx.mock
async def test_send_surfaces_200_verbatim() -> None:
    respx.post(URL).respond(200, json={"ok": True}, headers={"X-Request-Id": "req_123"})
    async with httpx.AsyncClient() as http:
        response = await Transport(http).send(make_request(), make_credential(), timeout_s=5.0)
    assert isinstance(response, TransportResponse)
    assert response.status == 200
    assert response.headers["x-request-id"] == "req_123"
    assert all(key == key.lower() for key in response.headers)
    assert response.body == b'{"ok":true}'


@respx.mock
async def test_send_returns_429_without_raising() -> None:
    respx.post(URL).respond(429, json={"error": "rate limited"}, headers={"Retry-After": "17"})
    async with httpx.AsyncClient() as http:
        response = await Transport(http).send(make_request(), make_credential(), timeout_s=5.0)
    assert response.status == 429
    assert response.headers["retry-after"] == "17"
    assert response.body == b'{"error":"rate limited"}'


@respx.mock
async def test_send_timeout_propagates_unwrapped() -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectTimeout("connection timed out"))
    async with httpx.AsyncClient() as http:
        with pytest.raises(httpx.TimeoutException) as exc_info:
            await Transport(http).send(make_request(), make_credential(), timeout_s=0.1)
    assert type(exc_info.value) is httpx.ConnectTimeout


@respx.mock
async def test_httpx_errors_never_carry_the_credential() -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("connection failed"))
    async with httpx.AsyncClient() as http:
        with pytest.raises(httpx.TransportError) as exc_info:
            await Transport(http).send(make_request(), make_credential(), timeout_s=5.0)
    exc = exc_info.value
    assert KEY not in str(exc)
    assert KEY not in repr(exc)
    assert all(KEY not in str(arg) for arg in exc.args)


# ---------------------------------------------------------------------------
# stream() — SSE framing


@respx.mock
async def test_stream_frames_named_events() -> None:
    sse_route(
        b'event: message_start\ndata: {"type": "message_start"}\n\n'
        b'event: content_block_delta\ndata: {"delta": {"text": "hi"}}\n\n'
    )
    async with httpx.AsyncClient() as http:
        async with Transport(http).stream(make_request(), make_credential(), 5.0) as stream:
            assert stream.status == 200
            events = await collect(stream.events)
    assert events == [
        SseEvent(event="message_start", data='{"type": "message_start"}'),
        SseEvent(event="content_block_delta", data='{"delta": {"text": "hi"}}'),
    ]


@respx.mock
async def test_stream_frames_unnamed_data_only_events() -> None:
    sse_route(b'data: {"choices": []}\n\ndata: [DONE]\n\n')
    async with httpx.AsyncClient() as http:
        async with Transport(http).stream(make_request(), make_credential(), 5.0) as stream:
            events = await collect(stream.events)
    # [DONE] passes through verbatim — terminal sentinels are codec concerns.
    assert events == [
        SseEvent(event=None, data='{"choices": []}'),
        SseEvent(event=None, data="[DONE]"),
    ]


@respx.mock
async def test_stream_skips_comment_lines() -> None:
    sse_route(b": OPENROUTER PROCESSING\n\n: OPENROUTER PROCESSING\ndata: {}\n\n")
    async with httpx.AsyncClient() as http:
        async with Transport(http).stream(make_request(), make_credential(), 5.0) as stream:
            events = await collect(stream.events)
    # A comment-only frame dispatches nothing; comments inside a frame are skipped.
    assert events == [SseEvent(event=None, data="{}")]


@respx.mock
async def test_stream_joins_multiple_data_lines_with_newline() -> None:
    sse_route(b"data: line one\ndata: line two\n\n")
    async with httpx.AsyncClient() as http:
        async with Transport(http).stream(make_request(), make_credential(), 5.0) as stream:
            events = await collect(stream.events)
    assert events == [SseEvent(event=None, data="line one\nline two")]


@respx.mock
async def test_stream_reassembles_frames_split_across_chunks() -> None:
    sse_route(
        b"event: mes",
        b"sage_start\nda",
        b'ta: {"a"',
        b': 1}\n\r\ndata: {"b": 2}\n',
        b"\n",
    )
    async with httpx.AsyncClient() as http:
        async with Transport(http).stream(make_request(), make_credential(), 5.0) as stream:
            events = await collect(stream.events)
    assert events == [
        SseEvent(event="message_start", data='{"a": 1}'),
        SseEvent(event=None, data='{"b": 2}'),
    ]


async def test_sse_event_name_does_not_leak_across_frames() -> None:
    async def feed() -> AsyncIterator[bytes]:
        yield b"event: ping\n\ndata: real\n\n"

    events = [event async for event in _sse_events(feed())]
    # A frame with no data field dispatches nothing, and its event name resets
    # at the blank line (SSE spec) — it must not attach to the next frame.
    assert events == [SseEvent(event=None, data="real")]


async def test_sse_value_space_stripping_and_unterminated_frame() -> None:
    async def feed() -> AsyncIterator[bytes]:
        yield b"data:tight\n\ndata: dropped-no-blank-line"

    events = [event async for event in _sse_events(feed())]
    # Exactly one leading space is optional after ":"; a trailing frame not
    # terminated by a blank line is discarded per spec (codecs fail closed on
    # a missing terminal).
    assert events == [SseEvent(event=None, data="tight")]


# ---------------------------------------------------------------------------
# stream() — open behavior


@respx.mock
async def test_stream_sends_auth_and_body_like_send() -> None:
    route = sse_route(b"data: {}\n\n")
    async with httpx.AsyncClient() as http:
        async with Transport(http).stream(
            make_request(), make_credential("anthropic"), 5.0
        ) as stream:
            await collect(stream.events)
    request = route.calls.last.request
    assert request.method == "POST"
    assert request.content == BODY
    assert request.headers["x-api-key"] == KEY
    assert request.headers["content-type"] == "application/json"
    assert request.headers["anthropic-version"] == "2023-06-01"


@respx.mock
async def test_stream_non_2xx_open_exposes_error_body_without_events() -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(
            429,
            headers={"Content-Type": "application/json", "Retry-After": "3"},
            stream=ChunkedByteStream([b'{"error": {"type": "rate_limit"}}']),
        )
    )
    async with httpx.AsyncClient() as http:
        async with Transport(http).stream(make_request(), make_credential(), 5.0) as stream:
            assert stream.status == 429
            assert stream.headers["retry-after"] == "3"
            assert all(key == key.lower() for key in stream.headers)
            body = await stream.read_error_body()
    assert body == b'{"error": {"type": "rate_limit"}}'


@respx.mock
async def test_stream_context_manager_closes_response() -> None:
    sse_route(b"data: {}\n\n")
    async with httpx.AsyncClient() as http:
        async with Transport(http).stream(make_request(), make_credential(), 5.0) as stream:
            underlying = stream._response
            assert not underlying.is_closed
        assert underlying.is_closed


@respx.mock
async def test_stream_context_manager_closes_response_even_when_unconsumed() -> None:
    sse_route(b"data: {}\n\ndata: never-read\n\n")
    async with httpx.AsyncClient() as http:
        async with Transport(http).stream(make_request(), make_credential(), 5.0) as stream:
            underlying = stream._response
        assert underlying.is_closed
