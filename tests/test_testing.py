"""Runtime test doubles: NoNetworkRuntime and ScriptedRuntime behavior."""

from __future__ import annotations

import pytest

from provider_runtime.testing import CapturedRuntimeCall, NoNetworkRuntime, ScriptedRuntime
from provider_runtime.types import (
    Absent,
    Accounting,
    CallMeta,
    EmbeddingCall,
    EmbeddingResponse,
    FinalizedProviderCall,
    FinalizedProviderRequest,
    OpenAIExplicitPrefix,
    PossiblyBillable,
    Present,
    ProviderCredential,
    ProviderTarget,
    ResponsePayload,
    RetryPolicy,
    RuntimeStreamEvent,
    StreamStart,
    Succeeded,
    TerminalEvent,
    TextContent,
    TextDelta,
    TokenUsage,
    TranscriptionCall,
    TranscriptionResponse,
)

CREDENTIAL = ProviderCredential(provider="openai", key="sk-test-not-a-real-key-1234567890")


def make_call() -> FinalizedProviderCall:
    return FinalizedProviderCall(
        request=FinalizedProviderRequest(
            target=ProviderTarget(provider="openai", model="gpt-5.6-sol"),
            protocol="openai_responses",
            url="https://api.openai.com/v1/responses",
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
            max_attempts=1,
            initial_delay_s=0.0,
            max_delay_s=0.0,
            jitter_s=0.0,
            deadline_s=Absent(),
        ),
        catalog_revision="cat-test",
        request_fingerprint="rf",
        tool_fingerprint="tf",
        schema_fingerprint="sf",
        planned_input_token_upper_bound=10,
        output_kind="text",
    )


def succeeded() -> Succeeded:
    return Succeeded(
        meta=CallMeta(
            provider="openai",
            model="gpt-5.6-sol",
            provider_request_id=Present("req-1"),
            upstream_provider=Absent(),
            usage=Present(
                TokenUsage.from_components(
                    input_tokens=5,
                    output_tokens=3,
                    total_tokens=Present(8),
                    reasoning_tokens=Absent(),
                    cache_read_input_tokens=Absent(),
                    cache_write_input_tokens=Absent(),
                )
            ),
            attempt_trace=(),
            billability=PossiblyBillable(),
        ),
        response=ResponsePayload(
            content=TextContent(text="ok", tool_calls=()),
            continuation=Absent(),
        ),
    )


# ---------------------------------------------------------------------------
# NoNetworkRuntime


async def test_no_network_generate_raises() -> None:
    with pytest.raises(AssertionError, match="Unexpected provider-runtime generate"):
        await NoNetworkRuntime().generate(make_call(), credential=CREDENTIAL)


async def test_no_network_stream_raises_at_first_event() -> None:
    stream = NoNetworkRuntime().stream(make_call(), credential=CREDENTIAL)
    with pytest.raises(AssertionError, match="Unexpected provider-runtime stream"):
        await anext(stream)


async def test_no_network_embed_and_transcribe_raise() -> None:
    runtime = NoNetworkRuntime()
    with pytest.raises(AssertionError, match="Unexpected provider-runtime embed"):
        await runtime.embed(
            EmbeddingCall(model="text-embedding-3-small", inputs=("a",), dimensions=Absent()),
            credential=CREDENTIAL,
        )
    with pytest.raises(AssertionError, match="Unexpected provider-runtime transcribe"):
        await runtime.transcribe(
            TranscriptionCall(
                model="gpt-4o-transcribe", filename="a.mp3", content=b"x", media_type="audio/mpeg"
            ),
            credential=CREDENTIAL,
        )


# ---------------------------------------------------------------------------
# ScriptedRuntime


async def test_scripted_generate_returns_queued_outcome_and_captures_call() -> None:
    outcome = succeeded()
    call = make_call()
    runtime = ScriptedRuntime(generate_outcomes=[outcome])

    result = await runtime.generate(call, credential=CREDENTIAL)

    assert result is outcome
    assert runtime.calls == [
        CapturedRuntimeCall(
            operation="generate", call=call, credential_provider="openai", streamed=False
        )
    ]


async def test_captured_call_never_stores_the_key() -> None:
    runtime = ScriptedRuntime(generate_outcomes=[succeeded()])
    await runtime.generate(make_call(), credential=CREDENTIAL)
    captured = runtime.calls[0]
    assert not hasattr(captured, "key")
    assert not hasattr(captured, "credential")
    assert CREDENTIAL.key not in repr(captured)


async def test_scripted_stream_wraps_script_in_envelopes_with_seq() -> None:
    script = [StreamStart(), TextDelta(text="hi"), TerminalEvent(outcome=succeeded())]
    call = make_call()
    runtime = ScriptedRuntime(stream_scripts=[script])

    events = [event async for event in runtime.stream(call, credential=CREDENTIAL)]

    assert events == [
        RuntimeStreamEvent(seq=1, event=script[0]),
        RuntimeStreamEvent(seq=2, event=script[1]),
        RuntimeStreamEvent(seq=3, event=script[2]),
    ]
    assert runtime.calls == [
        CapturedRuntimeCall(
            operation="stream", call=call, credential_provider="openai", streamed=True
        )
    ]


def test_stream_script_must_end_with_terminal() -> None:
    with pytest.raises(AssertionError, match="must end with a TerminalEvent"):
        ScriptedRuntime(stream_scripts=[[StreamStart(), TextDelta(text="hi")]])


def test_stream_script_rejects_events_after_terminal() -> None:
    # A non-terminal trailing event fails the ends-with-terminal check; an
    # early terminal (two terminals) fails the nothing-after-terminal check.
    with pytest.raises(AssertionError, match="must end with a TerminalEvent"):
        ScriptedRuntime(
            stream_scripts=[[TerminalEvent(outcome=succeeded()), TextDelta(text="late")]]
        )
    with pytest.raises(AssertionError, match="events after its TerminalEvent"):
        ScriptedRuntime(
            stream_scripts=[
                [TerminalEvent(outcome=succeeded()), TerminalEvent(outcome=succeeded())]
            ]
        )


def test_stream_script_rejects_empty_script() -> None:
    with pytest.raises(AssertionError, match="must end with a TerminalEvent"):
        ScriptedRuntime(stream_scripts=[[]])


async def test_scripted_pop_on_empty_queue_raises() -> None:
    runtime = ScriptedRuntime()
    with pytest.raises(AssertionError, match="No scripted provider-runtime generate"):
        await runtime.generate(make_call(), credential=CREDENTIAL)
    stream = runtime.stream(make_call(), credential=CREDENTIAL)
    with pytest.raises(AssertionError, match="No scripted provider-runtime stream"):
        await anext(stream)


async def test_scripted_embed_and_transcribe_return_queued_responses() -> None:
    embed_response = EmbeddingResponse(embeddings=((1.0, 2.0),), usage=Absent())
    transcribe_response = TranscriptionResponse(text="hello", usage=Absent())
    runtime = ScriptedRuntime(
        embed_responses=[embed_response], transcribe_responses=[transcribe_response]
    )

    embed_call = EmbeddingCall(
        model="text-embedding-3-small", inputs=("a",), dimensions=Present(256)
    )
    transcribe_call = TranscriptionCall(
        model="gpt-4o-transcribe", filename="a.mp3", content=b"x", media_type="audio/mpeg"
    )
    assert await runtime.embed(embed_call, credential=CREDENTIAL) is embed_response
    assert await runtime.transcribe(transcribe_call, credential=CREDENTIAL) is transcribe_response
    assert [captured.operation for captured in runtime.calls] == ["embed", "transcribe"]
    assert [captured.streamed for captured in runtime.calls] == [False, False]
