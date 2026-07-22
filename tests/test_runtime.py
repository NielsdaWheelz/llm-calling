"""ProviderRuntime: retry boundary, attempt traces, stream envelope, promotion.

HTTP-boundary tests drive real codec fixtures through respx so each codec
family gets at least one end-to-end pass through the runtime.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import httpx
import pytest
import respx

from provider_runtime.errors import CredentialRejected, ProtocolDefect, RuntimeDefect
from provider_runtime.runtime import ProviderRuntime, _retry_delay_s
from provider_runtime.types import (
    Absent,
    Accounting,
    Cancelled,
    ContinuationDelta,
    Failed,
    FinalAttempt,
    FinalizedProviderCall,
    FinalizedProviderRequest,
    Incomplete,
    InvalidToolArguments,
    NotDispatched,
    OpenAIExplicitPrefix,
    PossiblyBillable,
    Presence,
    Present,
    ProviderContextTooLarge,
    ProviderCredential,
    ProviderHttpUnavailable,
    ProviderName,
    ProviderProtocol,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTarget,
    ProviderTimeout,
    RetryPolicy,
    RuntimeStreamEvent,
    StreamStart,
    StructuredContent,
    Succeeded,
    TerminalEvent,
    TextContent,
    TextDelta,
    ToolCallDone,
    ToolCallStart,
    TranscriptionCall,
    TransientExhausted,
    TransportUnavailable,
)

ABSENT: Absent = Absent()
FIXTURES = Path(__file__).parent / "fixtures"

OPENAI_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
)
GEMINI_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.5-flash:streamGenerateContent?alt=sse"
)
MOONSHOT_URL = "https://api.moonshot.ai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SSE_HEADERS = {"content-type": "text/event-stream"}


def credential(provider: ProviderName = "openai") -> ProviderCredential:
    return ProviderCredential(provider=provider, key="sk-test-not-a-real-key-1234567890")


def make_call(
    *,
    provider: ProviderName = "openai",
    protocol: ProviderProtocol = "openai_responses",
    model: str = "gpt-5.6-sol",
    url: str = OPENAI_URL,
    output_kind: Literal["text", "strict_json"] = "text",
    max_attempts: int = 3,
    max_delay_s: float = 0.0,
    deadline_s: Presence[float] = ABSENT,
) -> FinalizedProviderCall:
    return FinalizedProviderCall(
        request=FinalizedProviderRequest(
            target=ProviderTarget(provider=provider, model=model),
            protocol=protocol,
            url=url,
            method="POST",
            safe_headers={},
            body=b"{}",
        ),
        accounting=Accounting(
            currency="usd",
            input_rate=1,
            output_rate=1,
            cache_write_rate=1,
            cache_read_rate=1,
            reasoning_billed_outside_output=False,
            platform_token_reservation=10,
            maximum_cost_estimate_usd_micros=10,
        ),
        requested_reasoning="low",
        effective_reasoning="low",
        native_reasoning="low",
        cache_plan=OpenAIExplicitPrefix(key="affinity", minimum_ttl="30m", breakpoints=1),
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            initial_delay_s=0.0,
            max_delay_s=max_delay_s,
            jitter_s=0.0,
            deadline_s=deadline_s,
        ),
        catalog_revision="cat-test",
        request_fingerprint="rf",
        tool_fingerprint="tf",
        schema_fingerprint="sf",
        planned_input_token_upper_bound=10,
        output_kind=output_kind,
    )


def fixture_json(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


def sse_wire(text: str) -> bytes:
    """Convert a stream fixture (event:/data: lines) into real SSE wire bytes."""
    out: list[str] = []
    pending_event: str | None = None
    for line in text.splitlines():
        if line.startswith("event:"):
            pending_event = line
        elif line.startswith("data:"):
            if pending_event is not None:
                out.append(pending_event)
                pending_event = None
            out.append(line)
            out.append("")
    return ("\n".join(out) + "\n").encode()


def sse_fixture(name: str) -> bytes:
    return sse_wire((FIXTURES / name).read_text())


async def run_generate(call: FinalizedProviderCall, cred=None):
    async with httpx.AsyncClient() as http:
        return await ProviderRuntime(http).generate(call, credential=cred or credential())


async def run_stream(
    call: FinalizedProviderCall, cred=None, cancel=None
) -> list[RuntimeStreamEvent]:
    async with httpx.AsyncClient() as http:
        runtime = ProviderRuntime(http)
        return [
            event
            async for event in runtime.stream(call, credential=cred or credential(), cancel=cancel)
        ]


def assert_contiguous_seqs(events: list[RuntimeStreamEvent]) -> None:
    assert [event.seq for event in events] == list(range(1, len(events) + 1))


def assert_single_terminal(events: list[RuntimeStreamEvent]) -> TerminalEvent:
    terminals = [event.event for event in events if isinstance(event.event, TerminalEvent)]
    assert len(terminals) == 1
    assert isinstance(events[-1].event, TerminalEvent)
    return terminals[0]


# ---------------------------------------------------------------------------
# Retry delay policy (pure)


def test_retry_after_present_is_honored_and_capped() -> None:
    policy = RetryPolicy(
        max_attempts=3, initial_delay_s=1.0, max_delay_s=8.0, jitter_s=0.0, deadline_s=Absent()
    )
    honored = _retry_delay_s(
        attempt=1, signal=ProviderRateLimit(retry_after=Present(2.5)), policy=policy
    )
    assert honored == 2.5
    capped = _retry_delay_s(
        attempt=1, signal=ProviderRateLimit(retry_after=Present(120.0)), policy=policy
    )
    assert capped == 8.0


def test_exponential_backoff_doubles_and_caps() -> None:
    policy = RetryPolicy(
        max_attempts=9, initial_delay_s=1.0, max_delay_s=6.0, jitter_s=0.0, deadline_s=Absent()
    )
    delays = [
        _retry_delay_s(attempt=attempt, signal=ProviderHttpUnavailable(), policy=policy)
        for attempt in (1, 2, 3, 4)
    ]
    assert delays == [1.0, 2.0, 4.0, 6.0]


def test_jitter_bounds() -> None:
    policy = RetryPolicy(
        max_attempts=3, initial_delay_s=1.0, max_delay_s=60.0, jitter_s=0.5, deadline_s=Absent()
    )
    for _ in range(20):
        delay = _retry_delay_s(attempt=1, signal=ProviderTimeout(), policy=policy)
        assert 1.0 <= delay <= 1.5


# ---------------------------------------------------------------------------
# generate(): outcomes, retries, traces


@respx.mock
async def test_generate_success_single_attempt_trace() -> None:
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=fixture_json("openai/success_text.json"))
    )
    outcome = await run_generate(make_call())
    assert isinstance(outcome, Succeeded)
    assert isinstance(outcome.response.content, TextContent)
    assert outcome.response.content.text == "Hello from Sol."
    trace = outcome.meta.attempt_trace
    assert len(trace) == 1
    assert trace[0].attempt == 1
    assert trace[0].signal == FinalAttempt()
    assert trace[0].status_code == Present(200)
    assert trace[0].ended_at_ms >= trace[0].started_at_ms
    assert isinstance(outcome.meta.usage, Present)
    assert outcome.meta.usage.value.total_tokens == 1252


@respx.mock
async def test_generate_retries_429_with_retry_after_then_keeps_trace() -> None:
    route = respx.post(OPENAI_URL).mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"message": "slow down", "type": "rate_limit_error"}},
            ),
            httpx.Response(200, json=fixture_json("openai/success_text.json")),
        ]
    )
    outcome = await run_generate(make_call())
    assert isinstance(outcome, Succeeded)
    assert route.call_count == 2
    trace = outcome.meta.attempt_trace
    assert len(trace) == 2
    assert trace[0].signal == ProviderRateLimit(retry_after=Present(0.0))
    assert trace[0].status_code == Present(429)
    assert trace[1].signal == FinalAttempt()
    assert trace[1].status_code == Present(200)


@respx.mock
async def test_generate_exhaustion_returns_transient_exhausted_with_full_trace() -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "boom", "type": "server_error"}}
        )
    )
    outcome = await run_generate(make_call(max_attempts=3))
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(attempts=3, cause=ProviderHttpUnavailable())
    assert route.call_count == 3
    trace = outcome.meta.attempt_trace
    assert len(trace) == 3
    assert [record.signal for record in trace[:-1]] == [
        ProviderHttpUnavailable(),
        ProviderHttpUnavailable(),
    ]
    assert trace[-1].signal == FinalAttempt()
    assert outcome.meta.billability == PossiblyBillable()
    assert outcome.meta.usage == Absent()


@respx.mock
async def test_generate_timeout_maps_to_provider_timeout() -> None:
    respx.post(OPENAI_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    outcome = await run_generate(make_call(max_attempts=2))
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(attempts=2, cause=ProviderTimeout())
    assert all(record.status_code == Absent() for record in outcome.meta.attempt_trace)


@respx.mock
async def test_generate_network_error_maps_to_transport_unavailable() -> None:
    respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("refused"))
    outcome = await run_generate(make_call(max_attempts=2))
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(attempts=2, cause=TransportUnavailable())
    # No attempt ever reached the provider: the reservation must release.
    assert outcome.meta.billability == NotDispatched()


@respx.mock
async def test_generate_context_too_large_is_terminal_without_retry() -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(400, json=fixture_json("openai/error_400_context_length.json"))
    )
    outcome = await run_generate(make_call(max_attempts=3))
    assert isinstance(outcome, Failed)
    assert outcome.failure == ProviderContextTooLarge()
    assert route.call_count == 1
    assert len(outcome.meta.attempt_trace) == 1
    assert outcome.meta.attempt_trace[0].signal == FinalAttempt()


@respx.mock
async def test_generate_credential_rejection_defect_propagates() -> None:
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(401, json=fixture_json("openai/error_401.json"))
    )
    with pytest.raises(CredentialRejected):
        await run_generate(make_call())


@respx.mock
async def test_generate_quota_exhaustion_defect_propagates() -> None:
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            429, json=fixture_json("openai/error_429_insufficient_quota.json")
        )
    )
    with pytest.raises(RuntimeDefect) as exc_info:
        await run_generate(make_call())
    assert exc_info.value.code == "quota_exhausted"


@respx.mock
async def test_generate_expected_failure_signal_folds_to_invalid_tool_arguments() -> None:
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=fixture_json("anthropic/invalid_tool_input.json"))
    )
    call = make_call(
        provider="anthropic",
        protocol="anthropic_messages",
        model="claude-fable-5",
        url=ANTHROPIC_URL,
    )
    outcome = await run_generate(call, cred=credential("anthropic"))
    assert isinstance(outcome, Failed)
    assert isinstance(outcome.failure, InvalidToolArguments)
    assert outcome.meta.provider == "anthropic"
    assert len(outcome.meta.attempt_trace) == 1
    assert outcome.meta.attempt_trace[0].signal == FinalAttempt()


@respx.mock
async def test_generate_deadline_bounds_the_retry_loop() -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "30"},
            json={"error": {"message": "slow down", "type": "rate_limit_error"}},
        )
    )
    outcome = await run_generate(
        make_call(max_attempts=5, max_delay_s=60.0, deadline_s=Present(0.05))
    )
    assert isinstance(outcome, Failed)
    assert isinstance(outcome.failure, TransientExhausted)
    assert outcome.failure.attempts == 1
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# generate(): structured promotion


@respx.mock
async def test_structured_promotion_on_strict_json_success() -> None:
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=fixture_json("openai/success_structured.json"))
    )
    outcome = await run_generate(make_call(output_kind="strict_json"))
    assert isinstance(outcome, Succeeded)
    content = outcome.response.content
    assert isinstance(content, StructuredContent)
    assert content.payload == {"verdict": "keep", "confidence": 3}
    assert content.text == '{"verdict":"keep","confidence":3}'


@respx.mock
async def test_structured_promotion_rejects_non_object_payload() -> None:
    body = fixture_json("openai/success_structured.json")
    body["output"][0]["content"][0]["text"] = "[1,2,3]"  # type: ignore[index]
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(ProtocolDefect) as exc_info:
        await run_generate(make_call(output_kind="strict_json"))
    assert exc_info.value.code == "structured_output_not_object"


@respx.mock
async def test_structured_promotion_rejects_invalid_json() -> None:
    body = fixture_json("openai/success_structured.json")
    body["output"][0]["content"][0]["text"] = "{not json"  # type: ignore[index]
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(ProtocolDefect) as exc_info:
        await run_generate(make_call(output_kind="strict_json"))
    assert exc_info.value.code == "invalid_structured_output"


@respx.mock
async def test_text_output_kind_skips_promotion() -> None:
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=fixture_json("openai/success_structured.json"))
    )
    outcome = await run_generate(make_call(output_kind="text"))
    assert isinstance(outcome, Succeeded)
    assert isinstance(outcome.response.content, TextContent)


# ---------------------------------------------------------------------------
# generate(): remaining codec families end-to-end (respx + real fixtures)


@respx.mock
async def test_generate_gemini_end_to_end() -> None:
    respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(200, json=fixture_json("gemini/success_nonstream.json"))
    )
    call = make_call(
        provider="gemini",
        protocol="gemini_generate_content",
        model="gemini-3.5-flash",
        url=GEMINI_URL,
    )
    outcome = await run_generate(call, cred=credential("gemini"))
    assert isinstance(outcome, Succeeded)
    assert isinstance(outcome.meta.usage, Present)
    assert outcome.meta.usage.value.total_tokens == 25
    assert len(outcome.meta.attempt_trace) == 1


@respx.mock
async def test_generate_moonshot_end_to_end() -> None:
    respx.post(MOONSHOT_URL).mock(
        return_value=httpx.Response(200, json=fixture_json("moonshot/success_text.json"))
    )
    call = make_call(
        provider="moonshot", protocol="moonshot_chat", model="kimi-k3", url=MOONSHOT_URL
    )
    outcome = await run_generate(call, cred=credential("moonshot"))
    assert isinstance(outcome, Succeeded)
    assert len(outcome.meta.attempt_trace) == 1


@respx.mock
async def test_generate_openrouter_end_to_end() -> None:
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200, json=fixture_json("openrouter/success_reasoning_details.json")
        )
    )
    call = make_call(
        provider="openrouter",
        protocol="openrouter_chat",
        model="moonshotai/kimi-k3-20260715",
        url=OPENROUTER_URL,
    )
    outcome = await run_generate(call, cred=credential("openrouter"))
    assert isinstance(outcome, Succeeded)
    assert isinstance(outcome.meta.upstream_provider, Present)


# ---------------------------------------------------------------------------
# stream(): happy paths per codec family


@respx.mock
async def test_stream_anthropic_end_to_end() -> None:
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            headers=SSE_HEADERS,
            content=sse_fixture("anthropic/success_stream_chunks.txt"),
        )
    )
    call = make_call(
        provider="anthropic",
        protocol="anthropic_messages",
        model="claude-opus-4-8",
        url=ANTHROPIC_URL,
    )
    events = await run_stream(call, cred=credential("anthropic"))

    assert json.loads(route.calls[0].request.content)["stream"] is True
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    text = "".join(event.event.text for event in events if isinstance(event.event, TextDelta))
    assert text == "Hello! How can I help?"
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    trace = outcome.meta.attempt_trace
    assert len(trace) == 1
    assert trace[0].signal == FinalAttempt()
    assert isinstance(outcome.meta.usage, Present)
    assert outcome.meta.usage.value.output_tokens == 8


@respx.mock
async def test_stream_openai_end_to_end_with_continuation() -> None:
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, content=sse_fixture("openai/stream_text.sse.txt")
        )
    )
    events = await run_stream(make_call())
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    assert isinstance(events[0].event, StreamStart)
    assert any(isinstance(event.event, ContinuationDelta) for event in events)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    assert isinstance(outcome.meta.usage, Present)
    assert outcome.meta.usage.value.total_tokens == 1209
    assert outcome.meta.attempt_trace[0].signal == FinalAttempt()


@respx.mock
async def test_stream_gemini_end_to_end_with_tools() -> None:
    respx.post(GEMINI_STREAM_URL).mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, content=sse_fixture("gemini/success_stream_chunks.txt")
        )
    )
    call = make_call(
        provider="gemini",
        protocol="gemini_generate_content",
        model="gemini-3.5-flash",
        url=GEMINI_URL,
    )
    events = await run_stream(call, cred=credential("gemini"))
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    assert any(isinstance(event.event, ToolCallStart) for event in events)
    assert any(isinstance(event.event, ToolCallDone) for event in events)
    assert isinstance(terminal.outcome, Succeeded)


@respx.mock
async def test_stream_moonshot_end_to_end() -> None:
    respx.post(MOONSHOT_URL).mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, content=sse_fixture("moonshot/stream_text.txt")
        )
    )
    call = make_call(
        provider="moonshot", protocol="moonshot_chat", model="kimi-k3", url=MOONSHOT_URL
    )
    events = await run_stream(call, cred=credential("moonshot"))
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    text = "".join(event.event.text for event in events if isinstance(event.event, TextDelta))
    assert text == "Tides rise."


@respx.mock
async def test_stream_openrouter_happy_end_to_end() -> None:
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, content=sse_fixture("openrouter/stream_happy.txt")
        )
    )
    call = make_call(
        provider="openrouter",
        protocol="openrouter_chat",
        model="moonshotai/kimi-k3-20260715",
        url=OPENROUTER_URL,
    )
    events = await run_stream(call, cred=credential("openrouter"))
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    assert isinstance(outcome.meta.upstream_provider, Present)


# ---------------------------------------------------------------------------
# stream(): retry boundary


@respx.mock
async def test_stream_pre_semantic_interruption_retries_and_seq_stays_monotonic() -> None:
    # Attempt 1: only a response.created frame, then the stream dies (no
    # terminal) -> pre-semantic TransientStreamError -> retried. StreamStart is
    # NOT semantic, so retry is legal; the envelope seq continues across
    # attempts.
    partial = sse_wire(
        'data: {"type":"response.created","response":{"id":"r1","model":"m","status":"in_progress"}}'
    )
    route = respx.post(OPENAI_URL).mock(
        side_effect=[
            httpx.Response(200, headers=SSE_HEADERS, content=partial),
            httpx.Response(
                200, headers=SSE_HEADERS, content=sse_fixture("openai/stream_text.sse.txt")
            ),
        ]
    )
    events = await run_stream(make_call())
    assert route.call_count == 2
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    trace = outcome.meta.attempt_trace
    assert len(trace) == 2
    assert trace[0].signal == ProviderStreamInterrupted(partial_output=False)
    assert trace[0].status_code == Present(200)
    assert trace[1].signal == FinalAttempt()
    # Both attempts yielded StreamStart; the envelope hides nothing.
    assert sum(1 for event in events if isinstance(event.event, StreamStart)) == 2


@respx.mock
async def test_stream_open_429_is_classified_and_retried() -> None:
    route = respx.post(OPENAI_URL).mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"message": "slow down", "type": "rate_limit_error"}},
            ),
            httpx.Response(
                200, headers=SSE_HEADERS, content=sse_fixture("openai/stream_text.sse.txt")
            ),
        ]
    )
    events = await run_stream(make_call())
    assert route.call_count == 2
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    trace = outcome.meta.attempt_trace
    assert trace[0].signal == ProviderRateLimit(retry_after=Present(0.0))
    assert trace[0].status_code == Present(429)


@respx.mock
async def test_stream_post_semantic_interruption_is_terminal_with_partial_output() -> None:
    truncated = sse_wire(
        "\n".join(
            [
                'data: {"type":"response.created","response":{"id":"r1","model":"m","status":"in_progress"}}',
                'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg","type":"message","role":"assistant"}}',
                'data: {"type":"response.output_text.delta","item_id":"msg","content_index":0,"delta":"Hel"}',
            ]
        )
    )
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, content=truncated)
    )
    events = await run_stream(make_call(max_attempts=3))
    assert route.call_count == 1  # no retry after semantic output
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(
        attempts=1, cause=ProviderStreamInterrupted(partial_output=True)
    )
    assert [type(event.event) for event in events[:-1]] == [StreamStart, TextDelta]


@respx.mock
async def test_stream_openrouter_inband_error_after_output_is_terminal() -> None:
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(
            200, headers=SSE_HEADERS, content=sse_fixture("openrouter/stream_inband_error.txt")
        )
    )
    call = make_call(
        provider="openrouter",
        protocol="openrouter_chat",
        model="moonshotai/kimi-k3-20260715",
        url=OPENROUTER_URL,
        max_attempts=3,
    )
    events = await run_stream(call, cred=credential("openrouter"))
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed)
    # The fixture emits a text delta before the in-band error chunk: any
    # transient after semantic output folds into partial_output=True.
    assert outcome.failure == TransientExhausted(
        attempts=1, cause=ProviderStreamInterrupted(partial_output=True)
    )


@respx.mock
async def test_stream_exhaustion_yields_failed_terminal() -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "boom", "type": "server_error"}}
        )
    )
    events = await run_stream(make_call(max_attempts=2))
    assert route.call_count == 2
    assert len(events) == 1  # nothing but the terminal was ever yielded
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(attempts=2, cause=ProviderHttpUnavailable())
    assert len(outcome.meta.attempt_trace) == 2


@respx.mock
async def test_stream_connect_error_exhaustion_is_not_dispatched() -> None:
    respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("refused"))
    events = await run_stream(make_call(max_attempts=2))
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(attempts=2, cause=TransportUnavailable())
    # No attempt ever reached the provider: the reservation must release.
    assert outcome.meta.billability == NotDispatched()


@respx.mock
async def test_stream_context_too_large_at_open_is_terminal() -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(400, json=fixture_json("openai/error_400_context_length.json"))
    )
    events = await run_stream(make_call(max_attempts=3))
    assert route.call_count == 1
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed)
    assert outcome.failure == ProviderContextTooLarge()


@respx.mock
async def test_stream_expected_failure_mid_stream_is_terminal_invalid_tool_arguments() -> None:
    bad_tool_args = sse_wire(
        "\n".join(
            [
                'data: {"id":"c1","object":"chat.completion.chunk","model":"kimi-k3","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{not"}}]}}]}',
                'data: {"id":"c1","object":"chat.completion.chunk","model":"kimi-k3","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls","usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}}]}',
                "data: [DONE]",
            ]
        )
    )
    route = respx.post(MOONSHOT_URL).mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, content=bad_tool_args)
    )
    call = make_call(
        provider="moonshot",
        protocol="moonshot_chat",
        model="kimi-k3",
        url=MOONSHOT_URL,
        max_attempts=3,
    )
    events = await run_stream(call, cred=credential("moonshot"))
    assert route.call_count == 1
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed)
    assert isinstance(outcome.failure, InvalidToolArguments)


@respx.mock
async def test_stream_strict_json_promotes_terminal_succeeded() -> None:
    structured = sse_wire(
        "\n".join(
            [
                'data: {"type":"response.created","response":{"id":"r","model":"gpt-5.6-sol","status":"in_progress"}}',
                'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg","type":"message","role":"assistant"}}',
                'data: {"type":"response.output_text.delta","item_id":"msg","content_index":0,"delta":"{\\"ok\\":true}"}',
                'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"{\\"ok\\":true}","annotations":[]}]}}',
                'data: {"type":"response.completed","response":{"id":"r","status":"completed","model":"gpt-5.6-sol","usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":8}}}',
            ]
        )
    )
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, content=structured)
    )
    events = await run_stream(make_call(output_kind="strict_json"))
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    content = outcome.response.content
    assert isinstance(content, StructuredContent)
    assert content.payload == {"ok": True}


# ---------------------------------------------------------------------------
# stream(): cancellation


class _HangingStream(httpx.AsyncByteStream):
    def __init__(self, head: bytes) -> None:
        self._head = head
        self.closed = False
        self._release = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._head
        await self._release.wait()

    async def aclose(self) -> None:
        self.closed = True
        self._release.set()


class _HangingTransport(httpx.AsyncBaseTransport):
    def __init__(self, head: bytes) -> None:
        self.byte_stream = _HangingStream(head)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=SSE_HEADERS, stream=self.byte_stream)


async def test_stream_cancellation_mid_stream_yields_cancelled_and_closes_source() -> None:
    head = sse_wire(
        "\n".join(
            [
                'data: {"type":"response.created","response":{"id":"r","model":"m","status":"in_progress"}}',
                'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg","type":"message","role":"assistant"}}',
                'data: {"type":"response.output_text.delta","item_id":"msg","content_index":0,"delta":"Hi"}',
            ]
        )
    )
    transport = _HangingTransport(head)
    cancel = asyncio.Event()
    events: list[RuntimeStreamEvent] = []
    async with httpx.AsyncClient(transport=transport) as http:
        runtime = ProviderRuntime(http)
        async for event in runtime.stream(make_call(), credential=credential(), cancel=cancel):
            events.append(event)
            if isinstance(event.event, TextDelta):
                cancel.set()

    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Cancelled)
    assert outcome.meta.billability == PossiblyBillable()
    trace = outcome.meta.attempt_trace
    assert len(trace) == 1
    assert trace[0].signal == FinalAttempt()
    assert transport.byte_stream.closed


async def test_stream_cancel_before_dispatch_is_not_dispatched() -> None:
    def deny(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request may be dispatched after pre-set cancel")

    cancel = asyncio.Event()
    cancel.set()
    async with httpx.AsyncClient(transport=httpx.MockTransport(deny)) as http:
        runtime = ProviderRuntime(http)
        events = [
            event
            async for event in runtime.stream(make_call(), credential=credential(), cancel=cancel)
        ]
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Cancelled)
    assert outcome.meta.billability == NotDispatched()


async def test_stream_external_task_cancellation_while_parked_is_clean() -> None:
    # Regression: the CONSUMING task (not the CancelSignal) is cancelled while
    # parked in _next_or_cancel's race. Must surface as CancelledError, not
    # RuntimeError from source.aclose() racing a still-running anext, and must
    # leave no next_task/cancel_task pending.
    head = sse_wire(
        "\n".join(
            [
                'data: {"type":"response.created","response":{"id":"r","model":"m","status":"in_progress"}}',
                'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg","type":"message","role":"assistant"}}',
                'data: {"type":"response.output_text.delta","item_id":"msg","content_index":0,"delta":"Hi"}',
            ]
        )
    )
    transport = _HangingTransport(head)
    cancel = asyncio.Event()  # attached, but never set — external cancel only
    got_text_delta = asyncio.Event()

    async def consume() -> None:
        async with httpx.AsyncClient(transport=transport) as http:
            runtime = ProviderRuntime(http)
            async for event in runtime.stream(make_call(), credential=credential(), cancel=cancel):
                if isinstance(event.event, TextDelta):
                    got_text_delta.set()

    tasks_before = asyncio.all_tasks()
    task = asyncio.ensure_future(consume())
    await got_text_delta.wait()
    # Let the consumer actually re-enter the generator and park inside
    # _next_or_cancel's asyncio.wait (a purely in-process handoff with no
    # real I/O, so a handful of scheduler ticks is more than sufficient).
    for _ in range(10):
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()

    leaked = asyncio.all_tasks() - tasks_before - {asyncio.current_task()}
    assert leaked == set()


# ---------------------------------------------------------------------------
# transcribe(): multipart port end-to-end


@respx.mock
async def test_transcribe_end_to_end() -> None:
    route = respx.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={
                "text": "hello world",
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            },
        )
    )
    async with httpx.AsyncClient() as http:
        response = await ProviderRuntime(http).transcribe(
            TranscriptionCall(
                model="gpt-4o-transcribe",
                filename="a.mp3",
                content=b"audio-bytes",
                media_type="audio/mpeg",
            ),
            credential=credential(),
        )
    assert response.text == "hello world"
    assert isinstance(response.usage, Present)
    assert response.usage.value.total_tokens == 6
    request = route.calls[0].request
    assert request.headers["authorization"].startswith("Bearer ")
    assert b"gpt-4o-transcribe" in request.content
    assert b"audio-bytes" in request.content


# ---------------------------------------------------------------------------
# stream(): refusal fold stays intact through the envelope


@respx.mock
async def test_stream_anthropic_refusal_terminal_is_incomplete_refused() -> None:
    refusal = sse_wire(
        "\n".join(
            [
                "event: message_start",
                'data: {"type":"message_start","message":{"id":"msg_r","type":"message","role":"assistant","content":[],"model":"claude-opus-4-8","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}',
                "event: message_delta",
                'data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null},"usage":{"output_tokens":0}}',
                "event: message_stop",
                'data: {"type":"message_stop"}',
            ]
        )
    )
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, headers=SSE_HEADERS, content=refusal)
    )
    call = make_call(
        provider="anthropic",
        protocol="anthropic_messages",
        model="claude-opus-4-8",
        url=ANTHROPIC_URL,
    )
    events = await run_stream(call, cred=credential("anthropic"))
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Incomplete)
    assert outcome.status == "refused"
    assert outcome.meta.attempt_trace[0].signal == FinalAttempt()
