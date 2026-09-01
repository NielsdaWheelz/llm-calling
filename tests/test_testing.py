"""Behavior of the IR-level test doubles: FakeEngine and ScriptedRuntime.

FakeEngine is the engine-seam double (spec §11) injected into ProviderRuntime;
ScriptedRuntime is the facade-level drop-in for library consumers. Both return
exactly what was scripted and record every call — assertions here cover script
consumption order, capture records, exhaustion/mismatch error quality, and the
stream envelope grammar ScriptedRuntime mirrors from the real runtime.
"""

from __future__ import annotations

import pydantic
import pytest

from provider_runtime import registry
from provider_runtime.engines import Engine, TransientAttempt
from provider_runtime.registry import _ModelRow as ModelRow
from provider_runtime.runtime import Credentials, NonGenerationCallFailed, ProviderRuntime
from provider_runtime.testing import (
    CapturedRuntimeCall,
    ChatCall,
    EngineScriptStep,
    FakeEngine,
    JsonOutCall,
    ScriptedRuntime,
)
from provider_runtime.types import (
    Absent,
    AttemptRecord,
    CallMeta,
    CodecStreamEvent,
    EmbeddingCall,
    EmbeddingResponse,
    Failed,
    FinalAttempt,
    GenerateIntent,
    InvalidStructuredOutput,
    PossiblyBillable,
    Present,
    PromptBlock,
    ProviderContextTooLarge,
    ProviderCredential,
    ProviderTarget,
    ProviderTimeout,
    ResponsePayload,
    StreamStart,
    StructuredReply,
    Succeeded,
    TerminalEvent,
    TextContent,
    TextDelta,
    TextOutput,
    UserMessage,
)

CREDENTIAL = ProviderCredential(provider="openai", key="sk-test-not-a-real-key-1234567890")

TARGET = ProviderTarget(provider="openai", model="gpt-test")

# A registry-independent row for the engine-seam tests; only the runtime
# integration test below needs a real registry row.
ROW = ModelRow(
    ref="openai:test-row",
    provider="openai",
    model_id="gpt-test",
    engine="openai_responses",
    base_url=Absent(),
    context_window=8_000,
    max_output_tokens=1_000,
    modalities=frozenset({"text"}),
    tools=True,
    streaming=True,
    structured="native",
    reasoning=Absent(),
    source_default_reasoning=Absent(),
    upgrade=Absent(),
    retirement=Absent(),
    continuation_codec="openai.v1",
    correlation="header",
    routing=Absent(),
)


def make_intent(target: ProviderTarget = TARGET) -> GenerateIntent:
    return GenerateIntent(
        target=target,
        messages=(UserMessage(blocks=(PromptBlock(text="hi"),)),),
        max_output_tokens=64,
        reasoning="none",
        tools=(),
        tool_choice="auto",
        output=TextOutput(),
    )


def engine_meta() -> CallMeta:
    """A single-attempt meta exactly as an engine constructs it."""
    return CallMeta(
        provider="openai",
        model="gpt-test",
        provider_request_id=Present("req-1"),
        upstream_provider=Absent(),
        usage=Absent(),
        attempt_trace=(
            AttemptRecord(
                attempt=1,
                signal=FinalAttempt(),
                status_code=Present(200),
                started_at_ms=0,
                ended_at_ms=1,
            ),
        ),
        billability=PossiblyBillable(),
        native_reasoning=Absent(),
        registry_revision="rev-test",
    )


def succeeded(text: str = "ok") -> Succeeded:
    return Succeeded(
        meta=engine_meta(),
        response=ResponsePayload(
            content=TextContent(text=text, tool_calls=()), continuation=Absent()
        ),
    )


def terminal_script(text: str = "ok") -> list[CodecStreamEvent]:
    return [StreamStart(), TextDelta(text=text), TerminalEvent(outcome=succeeded(text))]


def transient() -> TransientAttempt:
    return TransientAttempt(
        cause=ProviderTimeout(),
        status_code=Absent(),
        provider_request_id=Absent(),
        billability=PossiblyBillable(),
    )


class InertCancel:
    """CancelSignal stub the doubles must ignore entirely — even when set."""

    def is_set(self) -> bool:
        return True

    async def wait(self) -> bool:
        raise AssertionError("a runtime double must never await the cancel signal")


class Invoice(pydantic.BaseModel):
    total: int


class Receipt(pydantic.BaseModel):
    total: int


# ---------------------------------------------------------------------------
# FakeEngine


def test_fake_engine_satisfies_engine_protocol() -> None:
    # Static anchor only: pyright enforces this assignment's protocol
    # conformance; there is nothing meaningful left to assert at runtime.
    engine: Engine = FakeEngine([])
    del engine


async def test_fake_engine_returns_generate_outcomes_in_script_order() -> None:
    first, second = succeeded("one"), succeeded("two")
    intent = make_intent()
    engine = FakeEngine([first, second])

    assert await engine.generate(ROW, intent, CREDENTIAL) is first
    assert await engine.generate(ROW, intent, CREDENTIAL) is second
    assert engine.calls == [(ROW, intent, CREDENTIAL), (ROW, intent, CREDENTIAL)]


async def test_fake_engine_shared_script_interleaves_generate_and_stream() -> None:
    first, second = succeeded("one"), succeeded("two")
    events = terminal_script()
    script: list[EngineScriptStep] = [first, events, second]
    intent = make_intent()
    engine = FakeEngine(script)

    assert await engine.generate(ROW, intent, CREDENTIAL) is first
    assert [event async for event in engine.stream(ROW, intent, CREDENTIAL)] == events
    assert await engine.generate(ROW, intent, CREDENTIAL) is second


async def test_fake_engine_stream_yields_scripted_events_raw() -> None:
    events = terminal_script()
    intent = make_intent()
    engine = FakeEngine([events])

    collected = [event async for event in engine.stream(ROW, intent, CREDENTIAL)]

    # Raw codec events, no RuntimeStreamEvent envelope — the runtime owns it.
    assert collected == events
    assert engine.calls == [(ROW, intent, CREDENTIAL)]


async def test_fake_engine_raises_scripted_exception_from_generate() -> None:
    failure = transient()
    engine = FakeEngine([failure])

    with pytest.raises(TransientAttempt) as caught:
        await engine.generate(ROW, make_intent(), CREDENTIAL)
    assert caught.value is failure


async def test_fake_engine_stream_raises_scripted_exception_at_iteration_not_call() -> None:
    failure = transient()
    engine = FakeEngine([failure])

    # Creating the iterator must not raise: a real engine fails only once the
    # attempt runs, inside the runtime's per-attempt try block.
    stream = engine.stream(ROW, make_intent(), CREDENTIAL)
    assert engine.calls == []
    with pytest.raises(TransientAttempt) as caught:
        await anext(stream)
    assert caught.value is failure
    assert len(engine.calls) == 1


async def test_fake_engine_script_exhaustion_message_names_call_and_count() -> None:
    engine = FakeEngine([succeeded()])
    await engine.generate(ROW, make_intent(), CREDENTIAL)

    with pytest.raises(
        AssertionError,
        match=(
            r"FakeEngine script exhausted: unexpected generate call for "
            r"row 'openai:test-row' after 1 scripted steps"
        ),
    ):
        await engine.generate(ROW, make_intent(), CREDENTIAL)


async def test_fake_engine_generate_rejects_a_stream_step() -> None:
    engine = FakeEngine([terminal_script()])
    with pytest.raises(AssertionError, match="stream script but generate was called"):
        await engine.generate(ROW, make_intent(), CREDENTIAL)


async def test_fake_engine_stream_rejects_a_generate_step() -> None:
    engine = FakeEngine([succeeded()])
    with pytest.raises(AssertionError, match="generate outcome but stream was called"):
        await anext(engine.stream(ROW, make_intent(), CREDENTIAL))


async def test_fake_engine_drives_provider_runtime_generate() -> None:
    """FakeEngine is what test_runtime.py-style consumers inject at the seam."""
    row = next(candidate for candidate in registry._ROWS if candidate.provider == "openai")
    outcome = succeeded()
    engine = FakeEngine([outcome])
    runtime = ProviderRuntime(
        Credentials(openai="sk-test-not-a-real-key-000"),
        engines={
            "openai_responses": engine,
            "openai_chat": engine,
            "anthropic_messages": engine,
            "gemini_generate": engine,
        },
    )
    intent = make_intent(target=ProviderTarget(provider=row.provider, model=row.model_id))

    result = await runtime.generate(intent)

    assert isinstance(result, Succeeded)
    assert result.response is outcome.response
    seen_row, seen_intent, seen_credential = engine.calls[0]
    assert seen_row == row
    assert seen_intent is intent
    assert seen_credential.provider == "openai"


# ---------------------------------------------------------------------------
# ScriptedRuntime — generate/chat/json_out/embed


async def test_scripted_generate_returns_outcomes_in_order_and_captures() -> None:
    first, second = succeeded("one"), succeeded("two")
    intent = make_intent()
    runtime = ScriptedRuntime(generate_outcomes=[first, second])

    assert await runtime.generate(intent) is first
    assert await runtime.generate(intent) is second
    assert runtime.calls == [
        CapturedRuntimeCall(operation="generate", call=intent, credential_provider=Absent()),
        CapturedRuntimeCall(operation="generate", call=intent, credential_provider=Absent()),
    ]


async def test_scripted_runtime_ignores_cancel_even_when_set() -> None:
    outcome = succeeded()
    runtime = ScriptedRuntime(generate_outcomes=[outcome], stream_scripts=[terminal_script()])

    assert await runtime.generate(make_intent(), cancel=InertCancel()) is outcome
    events = [event async for event in runtime.stream(make_intent(), cancel=InertCancel())]
    assert len(events) == 3


async def test_scripted_chat_returns_outcome_and_captures_args_verbatim() -> None:
    outcome = succeeded()
    runtime = ScriptedRuntime(chat_outcomes=[outcome])

    assert await runtime.chat("openai:gpt-5.6-sol", user="hi") is outcome
    assert runtime.calls == [
        CapturedRuntimeCall(
            operation="chat",
            call=ChatCall(
                ref="openai:gpt-5.6-sol",
                system="",
                user="hi",
                reasoning="none",
                max_output_tokens=None,
            ),
            credential_provider=Absent(),
        )
    ]


async def test_scripted_queues_are_per_operation() -> None:
    # A scripted generate outcome must not leak into chat: the wrong-operation
    # call is a script mismatch, not a silent fallback.
    runtime = ScriptedRuntime(generate_outcomes=[succeeded()])
    with pytest.raises(AssertionError, match=r"chat call 1 \(ref 'openai:x'\)"):
        await runtime.chat("openai:x", user="hi")


async def test_scripted_json_out_returns_reply_and_captures_model_with_intent() -> None:
    reply = StructuredReply(value=Invoice(total=1), outcome=succeeded())
    intent = make_intent()
    runtime = ScriptedRuntime(json_out_results=[reply])

    assert await runtime.json_out(Invoice, intent) is reply
    assert runtime.calls == [
        CapturedRuntimeCall(
            operation="json_out",
            call=JsonOutCall(model=Invoice, intent=intent),
            credential_provider=Absent(),
        )
    ]


async def test_scripted_json_out_passes_failure_outcomes_through() -> None:
    failed = Failed(meta=engine_meta(), failure=InvalidStructuredOutput(safe_detail="nope"))
    runtime = ScriptedRuntime(json_out_results=[failed])

    assert await runtime.json_out(Invoice, make_intent()) is failed


async def test_scripted_json_out_rejects_a_reply_for_another_model() -> None:
    runtime = ScriptedRuntime(
        json_out_results=[StructuredReply(value=Receipt(total=1), outcome=succeeded())]
    )
    with pytest.raises(
        AssertionError,
        match=r"StructuredReply\[Receipt\] but the call requested Invoice",
    ):
        await runtime.json_out(Invoice, make_intent())


async def test_scripted_exhaustion_message_counts_scripted_results() -> None:
    runtime = ScriptedRuntime(generate_outcomes=[succeeded()])
    await runtime.generate(make_intent())

    with pytest.raises(
        AssertionError,
        match=r"generate call 2 \(target openai:gpt-test\) but scripted only 1",
    ):
        await runtime.generate(make_intent())


async def test_scripted_embed_returns_response_and_captures_provider_only() -> None:
    response = EmbeddingResponse(embeddings=((1.0, 2.0),), usage=Absent())
    call = EmbeddingCall(model="text-embedding-3-small", inputs=("a",), dimensions=Present(256))
    runtime = ScriptedRuntime(embed_responses=[response])

    assert await runtime.embed(call, credential=CREDENTIAL) is response
    assert runtime.calls == [
        CapturedRuntimeCall(operation="embed", call=call, credential_provider=Present("openai"))
    ]
    # The credential key never lands in the capture record.
    assert CREDENTIAL.key not in repr(runtime.calls)


async def test_scripted_embed_raises_a_scripted_non_generation_call_failed() -> None:
    # embed's expected-failure channel is a raise, not a value — unlike every
    # other operation's scripted outcomes — so this is the one method that
    # must be able to script an exception through its normal queue.
    failure = NonGenerationCallFailed(ProviderContextTooLarge())
    call = EmbeddingCall(model="text-embedding-3-small", inputs=("a",), dimensions=Absent())
    runtime = ScriptedRuntime(embed_responses=[failure])

    with pytest.raises(NonGenerationCallFailed) as exc_info:
        await runtime.embed(call, credential=CREDENTIAL)
    assert exc_info.value is failure, f"raised: {exc_info.value!r}"
    # The call is captured before the scripted raise, exactly like a real
    # engine failing mid-attempt.
    assert runtime.calls == [
        CapturedRuntimeCall(operation="embed", call=call, credential_provider=Present("openai"))
    ]


# ---------------------------------------------------------------------------
# ScriptedRuntime — stream envelope


async def test_scripted_stream_delivers_seq_stamped_envelopes_like_the_runtime() -> None:
    script = terminal_script()
    intent = make_intent()
    runtime = ScriptedRuntime(stream_scripts=[script])

    events = [event async for event in runtime.stream(intent)]

    # Consumers assert on RuntimeStreamEvent exactly as with the real runtime:
    # 1-based seq, scripted codec events verbatim.
    assert [event.seq for event in events] == [1, 2, 3]
    assert [event.event for event in events] == script
    assert runtime.calls == [
        CapturedRuntimeCall(operation="stream", call=intent, credential_provider=Absent())
    ]


async def test_scripted_stream_seq_restarts_per_stream() -> None:
    runtime = ScriptedRuntime(
        stream_scripts=[
            terminal_script("one"),
            [StreamStart(), TerminalEvent(outcome=succeeded("two"))],
        ]
    )

    first = [event async for event in runtime.stream(make_intent())]
    second = [event async for event in runtime.stream(make_intent())]

    assert [event.seq for event in first] == [1, 2, 3]
    assert [event.seq for event in second] == [1, 2]


def test_scripted_stream_script_must_end_with_a_terminal() -> None:
    with pytest.raises(AssertionError, match="must end with a TerminalEvent"):
        ScriptedRuntime(stream_scripts=[[StreamStart(), TextDelta(text="hi")]])
    with pytest.raises(AssertionError, match="must end with a TerminalEvent"):
        ScriptedRuntime(stream_scripts=[[]])


def test_scripted_stream_script_rejects_events_after_its_terminal() -> None:
    with pytest.raises(AssertionError, match="events after its TerminalEvent"):
        ScriptedRuntime(
            stream_scripts=[
                [TerminalEvent(outcome=succeeded()), TerminalEvent(outcome=succeeded())]
            ]
        )


def test_scripted_stream_exhaustion_raises_at_call_time() -> None:
    # Mirrors the real facade, which raises its call-shape defects before any
    # iteration — no awaiting required to surface the missing script.
    runtime = ScriptedRuntime()
    with pytest.raises(
        AssertionError,
        match=r"stream call 1 \(target openai:gpt-test\) but scripted only 0",
    ):
        runtime.stream(make_intent())
