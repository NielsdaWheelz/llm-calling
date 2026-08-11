"""ProviderRuntime facade behavior against a scripted FakeEngine.

The engine seam is the spec-sanctioned test boundary (spec §11): a plain
in-file double implementing the Engine protocol — no HTTP, no internal
mocking. Assertions cover the runtime's own obligations: registry resolution
and intent gates, credential lookup, the retry loop with attempt-trace
accumulation and billability folding, the stream envelope (seq stamping,
single-StreamStart grammar, mid-stream retry rules), cancellation, json_out,
the chat sugar, and the facts the facade puts on its own call span.

The embed port has no engine seam, so its span tests drive the openai SDK's
transport with respx — the HTTP boundary, still no internal mocking.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

import httpx
import pydantic
import pytest
import respx
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import (
    Link,
    NoOpTracer,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TracerProvider,
)
from opentelemetry.trace.span import INVALID_SPAN_CONTEXT
from opentelemetry.util import types as otel_types

from provider_runtime import registry
from provider_runtime.engines import TransientAttempt
from provider_runtime.errors import (
    CredentialMissing,
    CredentialRejected,
    InvalidRequest,
    NonGenerationCallFailed,
    ProtocolDefect,
)
from provider_runtime.prices import estimate_cost
from provider_runtime.registry import REGISTRY_REVISION, ModelRow
from provider_runtime.runtime import Credentials, ProviderRuntime
from provider_runtime.types import (
    Absent,
    AttemptRecord,
    Billability,
    CallMeta,
    CallOutcome,
    Cancelled,
    CancelSignal,
    CanonicalTool,
    CodecStreamEvent,
    ConfirmedNonBillable,
    ContinuationArtifact,
    ContinuationDelta,
    EmbeddingCall,
    Failed,
    FinalAttempt,
    GenerateIntent,
    ImageBlock,
    InvalidStructuredOutput,
    NotDispatched,
    OutputSpec,
    PossiblyBillable,
    Present,
    PromptBlock,
    PromptMessage,
    ProviderContextTooLarge,
    ProviderCredential,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTarget,
    Refused,
    ResponsePayload,
    RetryPolicy,
    RuntimeStreamEvent,
    StreamStart,
    StrictJsonOutput,
    StructuredContent,
    StructuredReply,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextContent,
    TextDelta,
    TextOutput,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallDone,
    ToolCallStart,
    TransientCause,
    TransientExhausted,
    TransportUnavailable,
    UsageEvent,
    UserMessage,
    presence_of,
)
from tests.test_otel import RecordingTracerProvider

TARGET = ProviderTarget(provider="openai", model="gpt-5.6-sol")
TEXT_ONLY_TARGET = ProviderTarget(provider="deepseek", model="deepseek-v4-pro")

POSSIBLY_BILLABLE = PossiblyBillable()
TEXT_OUTPUT = TextOutput()

CREDENTIALS = Credentials(openai="sk-openai-test-key-000", deepseek="sk-deepseek-test-key-000")

# A capability-poor row for the tools/streaming gates: no current registry row
# has tools=False or streaming=False, so those two gates are exercised by
# EXTENDING the row table (data, not behavior). Derived from a real row so the
# fixture cannot invent field values the registry's own invariants would
# reject. Drop this if a real capability-poor row ever lands.
LIMITED_ROW = replace(
    registry.resolve("openai:gpt-5.6-sol"),
    ref="openai:limited",
    model_id="gpt-limited",
    tools=False,
    streaming=False,
)


# ---------------------------------------------------------------------------
# FakeEngine — Engine-protocol double with per-call scripts


@dataclass
class Hang:
    """Script step: park forever, signalling arrival (cancellation tests)."""

    reached: asyncio.Event = field(default_factory=asyncio.Event)


type GenerateStep = CallOutcome | TransientAttempt | CredentialRejected | Hang
type StreamStep = CodecStreamEvent | TransientAttempt | Hang


@dataclass
class FakeEngine:
    """Scripted double; one generate step / stream script consumed per call."""

    generate_script: list[GenerateStep] = field(default_factory=list)
    stream_script: list[list[StreamStep]] = field(default_factory=list)
    generate_calls: list[tuple[ModelRow, GenerateIntent, ProviderCredential]] = field(
        default_factory=list
    )
    stream_calls: list[tuple[ModelRow, GenerateIntent, ProviderCredential]] = field(
        default_factory=list
    )
    streams_closed: int = 0

    async def generate(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> CallOutcome:
        self.generate_calls.append((row, intent, credential))
        step = self.generate_script.pop(0)
        if isinstance(step, Hang):
            step.reached.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable: a hung attempt never resumes")
        if isinstance(step, TransientAttempt | CredentialRejected):
            raise step
        return step

    async def stream(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> AsyncIterator[CodecStreamEvent]:
        self.stream_calls.append((row, intent, credential))
        script = self.stream_script.pop(0)
        try:
            for step in script:
                if isinstance(step, Hang):
                    step.reached.set()
                    await asyncio.Event().wait()
                elif isinstance(step, TransientAttempt):
                    raise step
                else:
                    yield step
        finally:
            self.streams_closed += 1


# ---------------------------------------------------------------------------
# Builders


def make_runtime(
    engine: FakeEngine,
    *,
    max_attempts: int = 3,
    credentials: Credentials = CREDENTIALS,
    tracer_provider: TracerProvider | None = None,
) -> ProviderRuntime:
    return ProviderRuntime(
        credentials,
        # Tests are the one sanctioned RetryPolicy construction site outside
        # retry.py; zero delays keep the suite instant.
        retry=RetryPolicy(
            max_attempts=max_attempts,
            initial_delay_s=0.0,
            max_delay_s=0.0,
            jitter_s=0.0,
            deadline_s=Absent(),
        ),
        engines={
            "openai_responses": engine,
            "openai_chat": engine,
            "anthropic_messages": engine,
            "gemini_generate": engine,
        },
        tracer_provider=tracer_provider,
    )


def make_intent(
    *,
    target: ProviderTarget = TARGET,
    messages: tuple[PromptMessage, ...] = (UserMessage(blocks=(PromptBlock(text="hi"),)),),
    tools: tuple[CanonicalTool, ...] = (),
    output: OutputSpec = TEXT_OUTPUT,
) -> GenerateIntent:
    return GenerateIntent(
        target=target,
        messages=messages,
        max_output_tokens=64,
        reasoning="none",
        tools=tools,
        tool_choice="auto",
        output=output,
    )


def engine_meta(
    *, billability: Billability = POSSIBLY_BILLABLE, request_id: str = "req-final"
) -> CallMeta:
    """A single-attempt meta exactly as an engine constructs it."""
    return CallMeta(
        provider="openai",
        model="gpt-5.6-sol",
        provider_request_id=Present(request_id),
        upstream_provider=Absent(),
        usage=Present(
            TokenUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                reasoning_tokens=Absent(),
                cache_read_input_tokens=Absent(),
                cache_write_input_tokens=Absent(),
            )
        ),
        attempt_trace=(
            AttemptRecord(
                attempt=1,
                signal=FinalAttempt(),
                status_code=Present(200),
                started_at_ms=0,
                ended_at_ms=1,
            ),
        ),
        billability=billability,
        # The wire fragment the engine merged, serialised — never a bare level.
        native_reasoning=Present('{"reasoning":{"effort":"none"}}'),
        registry_revision=REGISTRY_REVISION,
    )


def succeeded(text: str = "Hello.", *, meta: CallMeta | None = None) -> Succeeded:
    return Succeeded(
        meta=meta or engine_meta(),
        response=ResponsePayload(
            content=TextContent(text=text, tool_calls=()), continuation=Absent()
        ),
    )


def structured_succeeded(payload: dict[str, object]) -> Succeeded:
    return Succeeded(
        meta=engine_meta(),
        response=ResponsePayload(
            content=StructuredContent(payload=payload, text=str(payload)), continuation=Absent()
        ),
    )


def transient(
    cause: TransientCause | None = None,
    *,
    status: int | None = None,
    request_id: str | None = None,
    billability: Billability = POSSIBLY_BILLABLE,
) -> TransientAttempt:
    return TransientAttempt(
        cause=cause if cause is not None else ProviderHttpUnavailable(),
        status_code=presence_of(status),
        provider_request_id=presence_of(request_id),
        billability=billability,
    )


def happy_stream() -> list[StreamStep]:
    return [
        StreamStart(),
        TextDelta(text="Hel"),
        TextDelta(text="lo"),
        TerminalEvent(outcome=succeeded("Hello")),
    ]


async def collect(
    rt: ProviderRuntime, intent: GenerateIntent, cancel: CancelSignal | None = None
) -> list[RuntimeStreamEvent]:
    return [event async for event in rt.stream(intent, cancel=cancel)]


def assert_contiguous_seqs(events: list[RuntimeStreamEvent]) -> None:
    assert [event.seq for event in events] == list(range(1, len(events) + 1)), (
        f"seq must be 1-based and contiguous, got {[event.seq for event in events]}"
    )


def assert_single_terminal(events: list[RuntimeStreamEvent]) -> TerminalEvent:
    terminals = [event.event for event in events if isinstance(event.event, TerminalEvent)]
    assert len(terminals) == 1, f"expected exactly one terminal, got {len(terminals)}"
    assert isinstance(events[-1].event, TerminalEvent), "the terminal must be the last envelope"
    return terminals[0]


# ---------------------------------------------------------------------------
# generate(): dispatch, retries, traces, billability


async def test_generate_dispatches_resolved_row_with_credential_and_keeps_engine_trace() -> None:
    engine = FakeEngine(generate_script=[succeeded()])
    outcome = await make_runtime(engine).generate(make_intent())
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(text="Hello.", tool_calls=())
    (call,) = engine.generate_calls
    row, intent, credential = call
    assert row.ref == "openai:gpt-5.6-sol"
    assert intent == make_intent()
    assert credential == ProviderCredential(provider="openai", key="sk-openai-test-key-000")
    trace = outcome.meta.attempt_trace
    assert len(trace) == 1
    assert trace[0].attempt == 1
    assert trace[0].signal == FinalAttempt()
    assert outcome.meta.provider_request_id == Present("req-final")
    assert outcome.meta.billability == PossiblyBillable()


async def test_generate_retries_transient_then_accumulates_and_renumbers_trace() -> None:
    engine = FakeEngine(
        generate_script=[
            transient(ProviderRateLimit(retry_after=Present(0.0)), status=429, request_id="req-1"),
            succeeded(),
        ]
    )
    outcome = await make_runtime(engine).generate(make_intent())
    assert isinstance(outcome, Succeeded)
    assert len(engine.generate_calls) == 2
    trace = outcome.meta.attempt_trace
    assert len(trace) == 2
    assert trace[0].attempt == 1
    assert trace[0].signal == ProviderRateLimit(retry_after=Present(0.0))
    assert trace[0].status_code == Present(429)
    assert trace[1].attempt == 2, "the engine's single-attempt record must be renumbered"
    assert trace[1].signal == FinalAttempt()
    assert trace[1].status_code == Present(200)


async def test_generate_exhaustion_folds_full_trace_and_last_request_id() -> None:
    engine = FakeEngine(
        generate_script=[
            transient(status=500, request_id="req-a"),
            transient(status=502, request_id="req-b"),
            transient(status=503, request_id="req-c"),
        ]
    )
    outcome = await make_runtime(engine, max_attempts=3).generate(make_intent())
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(attempts=3, cause=ProviderHttpUnavailable())
    assert len(engine.generate_calls) == 3
    trace = outcome.meta.attempt_trace
    assert [record.attempt for record in trace] == [1, 2, 3]
    assert [record.signal for record in trace[:-1]] == [
        ProviderHttpUnavailable(),
        ProviderHttpUnavailable(),
    ]
    assert trace[-1].signal == FinalAttempt()
    assert trace[-1].status_code == Present(503)
    assert outcome.meta.provider_request_id == Present("req-c")
    assert outcome.meta.billability == PossiblyBillable()
    assert outcome.meta.usage == Absent()
    assert outcome.meta.provider == "openai"
    assert outcome.meta.model == "gpt-5.6-sol"
    assert outcome.meta.registry_revision == REGISTRY_REVISION


async def test_generate_billability_folds_max_across_attempts() -> None:
    # PossiblyBillable on attempt 1 outranks the final ConfirmedNonBillable.
    engine = FakeEngine(
        generate_script=[
            transient(billability=PossiblyBillable()),
            Refused(meta=engine_meta(billability=ConfirmedNonBillable()), safe_detail="nope"),
        ]
    )
    outcome = await make_runtime(engine).generate(make_intent())
    assert isinstance(outcome, Refused)
    assert outcome.meta.billability == PossiblyBillable()
    assert len(outcome.meta.attempt_trace) == 2


async def test_generate_exhaustion_of_undispatched_attempts_stays_not_dispatched() -> None:
    engine = FakeEngine(
        generate_script=[
            transient(TransportUnavailable(), billability=NotDispatched()),
            transient(TransportUnavailable(), billability=NotDispatched()),
        ]
    )
    outcome = await make_runtime(engine, max_attempts=2).generate(make_intent())
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(attempts=2, cause=TransportUnavailable())
    assert outcome.meta.billability == NotDispatched()


async def test_generate_terminal_failure_value_passes_through_without_retry() -> None:
    engine = FakeEngine(
        generate_script=[Failed(meta=engine_meta(), failure=ProviderContextTooLarge())]
    )
    outcome = await make_runtime(engine, max_attempts=3).generate(make_intent())
    assert isinstance(outcome, Failed)
    assert outcome.failure == ProviderContextTooLarge()
    assert len(engine.generate_calls) == 1
    assert len(outcome.meta.attempt_trace) == 1


async def test_generate_defect_propagates_without_retry() -> None:
    engine = FakeEngine(generate_script=[CredentialRejected(message="openai rejected the key")])
    with pytest.raises(CredentialRejected):
        await make_runtime(engine, max_attempts=3).generate(make_intent())
    assert len(engine.generate_calls) == 1


# ---------------------------------------------------------------------------
# generate(): validation and credential gates


async def test_generate_unknown_target_raises_invalid_request() -> None:
    engine = FakeEngine()
    with pytest.raises(InvalidRequest):
        await make_runtime(engine).generate(
            make_intent(target=ProviderTarget(provider="openai", model="gpt-unknown"))
        )
    assert engine.generate_calls == []


async def test_generate_tools_with_strict_output_raises_invalid_request() -> None:
    engine = FakeEngine()
    tool = CanonicalTool(name="lookup", description="", parameters={"type": "object"})
    with pytest.raises(InvalidRequest):
        await make_runtime(engine).generate(
            make_intent(tools=(tool,), output=StrictJsonOutput(name="Out", schema={}))
        )
    assert engine.generate_calls == []


async def test_generate_image_block_to_text_only_row_raises_invalid_request() -> None:
    engine = FakeEngine()
    with pytest.raises(InvalidRequest):
        await make_runtime(engine).generate(
            make_intent(
                target=TEXT_ONLY_TARGET,
                messages=(
                    UserMessage(blocks=(ImageBlock(media_type="image/png", data=b"\x89PNG"),)),
                ),
            )
        )
    assert engine.generate_calls == []


async def test_generate_tools_on_toolless_row_raises_invalid_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "ROWS", (*registry.ROWS, LIMITED_ROW))
    engine = FakeEngine()
    tool = CanonicalTool(name="lookup", description="", parameters={"type": "object"})
    with pytest.raises(InvalidRequest):
        await make_runtime(engine).generate(
            make_intent(
                target=ProviderTarget(provider="openai", model="gpt-limited"), tools=(tool,)
            )
        )
    assert engine.generate_calls == []


async def test_generate_missing_credential_raises_credential_missing() -> None:
    engine = FakeEngine(generate_script=[succeeded()])
    with pytest.raises(CredentialMissing):
        await make_runtime(engine, credentials=Credentials()).generate(make_intent())
    assert engine.generate_calls == []


def test_credentials_repr_never_contains_keys() -> None:
    rendered = repr(Credentials(openai="sk-secret-aaa", xai="xai-secret-bbb"))
    assert "sk-secret-aaa" not in rendered
    assert "xai-secret-bbb" not in rendered


# ---------------------------------------------------------------------------
# generate(): cancellation


async def test_generate_cancel_preset_returns_cancelled_not_dispatched() -> None:
    cancel = asyncio.Event()
    cancel.set()
    engine = FakeEngine(generate_script=[succeeded()])
    outcome = await make_runtime(engine).generate(make_intent(), cancel=cancel)
    assert isinstance(outcome, Cancelled)
    assert outcome.meta.billability == NotDispatched()
    assert engine.generate_calls == []
    trace = outcome.meta.attempt_trace
    assert len(trace) == 1
    assert trace[0].signal == FinalAttempt()


async def test_generate_cancel_mid_attempt_returns_cancelled_possibly_billable() -> None:
    cancel = asyncio.Event()
    hang = Hang()
    engine = FakeEngine(generate_script=[hang])

    async def trigger() -> None:
        await hang.reached.wait()
        cancel.set()

    outcome, _ = await asyncio.gather(
        make_runtime(engine).generate(make_intent(), cancel=cancel), trigger()
    )
    assert isinstance(outcome, Cancelled)
    assert outcome.meta.billability == PossiblyBillable()
    assert len(outcome.meta.attempt_trace) == 1
    assert outcome.meta.attempt_trace[0].signal == FinalAttempt()


# ---------------------------------------------------------------------------
# stream(): envelope, retry boundary, grammar


async def test_stream_happy_path_stamps_contiguous_seqs_and_rewrites_terminal_meta() -> None:
    engine = FakeEngine(stream_script=[happy_stream()])
    events = await collect(make_runtime(engine), make_intent())
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    assert [type(event.event) for event in events] == [
        StreamStart,
        TextDelta,
        TextDelta,
        TerminalEvent,
    ]
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    trace = outcome.meta.attempt_trace
    assert len(trace) == 1
    assert trace[0].signal == FinalAttempt()
    assert outcome.meta.usage == engine_meta().usage
    assert engine.streams_closed == 1


async def test_stream_retry_before_semantic_output_emits_a_single_stream_start() -> None:
    engine = FakeEngine(
        stream_script=[
            [StreamStart(), transient(ProviderStreamInterrupted(partial_output=False), status=200)],
            happy_stream(),
        ]
    )
    events = await collect(make_runtime(engine), make_intent())
    assert len(engine.stream_calls) == 2
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    starts = [event for event in events if isinstance(event.event, StreamStart)]
    assert len(starts) == 1, "a retried attempt must not re-emit a second StreamStart envelope"
    assert isinstance(events[0].event, StreamStart)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    trace = outcome.meta.attempt_trace
    assert len(trace) == 2
    assert trace[0].signal == ProviderStreamInterrupted(partial_output=False)
    assert trace[0].status_code == Present(200)
    assert trace[1].attempt == 2, "the engine's terminal record must be renumbered"
    assert trace[1].signal == FinalAttempt()


async def test_stream_retry_after_pre_start_transient_forwards_the_first_stream_start() -> None:
    engine = FakeEngine(
        stream_script=[
            [transient(ProviderRateLimit(retry_after=Present(0.0)), status=429)],
            happy_stream(),
        ]
    )
    events = await collect(make_runtime(engine), make_intent())
    assert len(engine.stream_calls) == 2
    assert_contiguous_seqs(events)
    assert isinstance(events[0].event, StreamStart)
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.attempt_trace[0].signal == ProviderRateLimit(retry_after=Present(0.0))


@pytest.mark.parametrize(
    "semantic",
    [
        pytest.param(TextDelta(text="Hel"), id="text_delta"),
        pytest.param(ToolCallStart(call_id="c1", name="lookup"), id="tool_call_start"),
        pytest.param(ToolCallDelta(call_id="c1", arguments_delta='{"q":'), id="tool_call_delta"),
        pytest.param(
            ToolCallDone(tool_call=ToolCall(id="c1", name="lookup", arguments={"q": "x"})),
            id="tool_call_done",
        ),
        pytest.param(
            ContinuationDelta(
                artifact=ContinuationArtifact(
                    target=TARGET, codec_id="openai.v1", opaque_payload={"item": "opaque"}
                )
            ),
            id="continuation_delta",
        ),
        pytest.param(
            UsageEvent(
                usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    reasoning_tokens=Absent(),
                    cache_read_input_tokens=Absent(),
                    cache_write_input_tokens=Absent(),
                )
            ),
            id="usage_event",
        ),
    ],
)
async def test_stream_post_semantic_transient_is_terminal_for_every_semantic_event(
    semantic: CodecStreamEvent,
) -> None:
    # Freeze rule: retry ONLY when zero semantic events were emitted. Every
    # non-StreamStart, non-terminal event counts — narrowing that set to text
    # and tool deltas would double-deliver usage frames and continuations.
    engine = FakeEngine(
        stream_script=[
            [StreamStart(), semantic, transient(TransportUnavailable())],
            happy_stream(),
        ]
    )
    events = await collect(make_runtime(engine, max_attempts=3), make_intent())
    assert len(engine.stream_calls) == 1, (
        f"{type(semantic).__name__} must block stream retry, but a second attempt was dispatched"
    )
    assert_contiguous_seqs(events)
    terminal = assert_single_terminal(events)
    assert [event.event for event in events[:-1]] == [StreamStart(), semantic]
    outcome = terminal.outcome
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(
        attempts=1, cause=ProviderStreamInterrupted(partial_output=True)
    )
    assert outcome.meta.billability == PossiblyBillable()
    assert outcome.meta.attempt_trace[0].signal == FinalAttempt()


async def test_stream_exhaustion_without_any_events_yields_single_failed_terminal() -> None:
    engine = FakeEngine(
        stream_script=[
            [transient(status=500, request_id="req-1")],
            [transient(status=503, request_id="req-2")],
        ]
    )
    events = await collect(make_runtime(engine, max_attempts=2), make_intent())
    assert len(engine.stream_calls) == 2
    assert len(events) == 1, "nothing but the terminal envelope may be yielded"
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed)
    assert outcome.failure == TransientExhausted(attempts=2, cause=ProviderHttpUnavailable())
    trace = outcome.meta.attempt_trace
    assert [record.signal for record in trace] == [ProviderHttpUnavailable(), FinalAttempt()]
    assert outcome.meta.provider_request_id == Present("req-2")


async def test_stream_engine_terminal_without_stream_start_passes_through() -> None:
    engine = FakeEngine(
        stream_script=[
            [TerminalEvent(outcome=Failed(meta=engine_meta(), failure=ProviderContextTooLarge()))]
        ]
    )
    events = await collect(make_runtime(engine, max_attempts=3), make_intent())
    assert len(engine.stream_calls) == 1
    assert len(events) == 1
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Failed)
    assert outcome.failure == ProviderContextTooLarge()


async def test_stream_bare_engine_exhaustion_is_a_protocol_defect() -> None:
    engine = FakeEngine(stream_script=[[StreamStart()]])
    with pytest.raises(ProtocolDefect):
        await collect(make_runtime(engine), make_intent())


async def test_stream_on_non_streaming_row_raises_invalid_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "ROWS", (*registry.ROWS, LIMITED_ROW))
    engine = FakeEngine()
    with pytest.raises(InvalidRequest):
        make_runtime(engine).stream(
            make_intent(target=ProviderTarget(provider="openai", model="gpt-limited"))
        )
    assert engine.stream_calls == []


# ---------------------------------------------------------------------------
# stream(): cancellation


async def test_stream_cancel_preset_yields_cancelled_without_dispatch() -> None:
    cancel = asyncio.Event()
    cancel.set()
    engine = FakeEngine(stream_script=[happy_stream()])
    events = await collect(make_runtime(engine), make_intent(), cancel=cancel)
    assert engine.stream_calls == []
    terminal = assert_single_terminal(events)
    outcome = terminal.outcome
    assert isinstance(outcome, Cancelled)
    assert outcome.meta.billability == NotDispatched()


async def test_stream_cancel_mid_stream_yields_cancelled_and_closes_engine_stream() -> None:
    cancel = asyncio.Event()
    engine = FakeEngine(stream_script=[[StreamStart(), TextDelta(text="Hi"), Hang()]])
    events: list[RuntimeStreamEvent] = []
    async for event in make_runtime(engine).stream(make_intent(), cancel=cancel):
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
    assert engine.streams_closed == 1


async def test_stream_external_task_cancellation_while_parked_is_clean() -> None:
    # Regression ported from the pre-cutover runtime: the CONSUMING task (not
    # the CancelSignal) is cancelled while the runtime is parked racing the
    # next engine event. Must surface as CancelledError and leak no tasks.
    cancel = asyncio.Event()  # attached, but never set — external cancel only
    hang = Hang()
    engine = FakeEngine(stream_script=[[StreamStart(), TextDelta(text="Hi"), hang]])
    got_text_delta = asyncio.Event()

    async def consume() -> None:
        async for event in make_runtime(engine).stream(make_intent(), cancel=cancel):
            if isinstance(event.event, TextDelta):
                got_text_delta.set()

    tasks_before = asyncio.all_tasks()
    task = asyncio.ensure_future(consume())
    await got_text_delta.wait()
    await hang.reached.wait()
    for _ in range(10):
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    leaked = asyncio.all_tasks() - tasks_before - {asyncio.current_task()}
    assert leaked == set()


# ---------------------------------------------------------------------------
# json_out


class Verdict(pydantic.BaseModel):
    verdict: str
    confidence: int


async def test_json_out_round_trip_returns_structured_reply() -> None:
    engine = FakeEngine(
        generate_script=[structured_succeeded({"verdict": "keep", "confidence": 3})]
    )
    reply = await make_runtime(engine).json_out(Verdict, make_intent())
    assert isinstance(reply, StructuredReply)
    assert reply.value == Verdict(verdict="keep", confidence=3)
    assert reply.outcome.meta == engine_meta()
    (call,) = engine.generate_calls
    sent = call[1]
    assert sent.output == StrictJsonOutput(name="Verdict", schema=Verdict.model_json_schema()), (
        "json_out must derive the strict schema from the pydantic model"
    )


async def test_json_out_validation_miss_returns_failed_without_payload_echo() -> None:
    engine = FakeEngine(generate_script=[structured_succeeded({"verdict": "keep-me-secret-value"})])
    outcome = await make_runtime(engine).json_out(Verdict, make_intent())
    assert isinstance(outcome, Failed)
    assert isinstance(outcome.failure, InvalidStructuredOutput)
    assert outcome.meta == engine_meta(), "the failure must carry the same terminal meta"
    detail = outcome.failure.safe_detail
    assert "Verdict" in detail
    assert "confidence" in detail
    assert "keep-me-secret-value" not in detail, "safe_detail must never echo the payload"


async def test_json_out_requires_text_output_intent() -> None:
    engine = FakeEngine()
    with pytest.raises(InvalidRequest):
        await make_runtime(engine).json_out(
            Verdict, make_intent(output=StrictJsonOutput(name="Out", schema={}))
        )
    assert engine.generate_calls == []


async def test_json_out_passes_non_success_outcomes_through() -> None:
    refused = Refused(meta=engine_meta(), safe_detail="nope")
    engine = FakeEngine(generate_script=[refused])
    outcome = await make_runtime(engine).json_out(Verdict, make_intent())
    assert isinstance(outcome, Refused)
    assert outcome.safe_detail == "nope"


# ---------------------------------------------------------------------------
# chat sugar


async def test_chat_builds_intent_from_row_defaults() -> None:
    engine = FakeEngine(generate_script=[succeeded()])
    outcome = await make_runtime(engine).chat(
        "openai:gpt-5.6-sol", system="be brief", user="hi", reasoning="high"
    )
    assert isinstance(outcome, Succeeded)
    (call,) = engine.generate_calls
    sent = call[1]
    assert sent == GenerateIntent(
        target=TARGET,
        messages=(
            SystemMessage(blocks=(PromptBlock(text="be brief"),)),
            UserMessage(blocks=(PromptBlock(text="hi"),)),
        ),
        # The contract is "no explicit cap → the resolved row's cap", not the
        # figure the row happens to carry today.
        max_output_tokens=registry.resolve("openai:gpt-5.6-sol").max_output_tokens,
        reasoning="high",
        tools=(),
        tool_choice="auto",
        output=TextOutput(),
    )


async def test_chat_without_system_omits_the_system_message_and_caps_tokens() -> None:
    engine = FakeEngine(generate_script=[succeeded()])
    await make_runtime(engine).chat("openai:gpt-5.6-sol", user="hi", max_output_tokens=64)
    (call,) = engine.generate_calls
    sent = call[1]
    assert sent.messages == (UserMessage(blocks=(PromptBlock(text="hi"),)),)
    assert sent.max_output_tokens == 64
    assert sent.reasoning == "none"


async def test_chat_unknown_ref_raises_invalid_request() -> None:
    engine = FakeEngine()
    with pytest.raises(InvalidRequest):
        await make_runtime(engine).chat("openai:no-such-ref", user="hi")
    assert engine.generate_calls == []


# ---------------------------------------------------------------------------
# spans: the call span never leaks into the consumer's context


class AmbientSpan(Span):
    """A consumer-owned span held across stream events.

    Deliberately not the api's `NonRecordingSpan`: the no-op tracer hands that
    class straight back as the span it starts, which would hide whether the
    runtime attached a span of its own.
    """

    def end(self, end_time: int | None = None) -> None:
        return None

    def get_span_context(self) -> SpanContext:
        return INVALID_SPAN_CONTEXT

    def set_attributes(self, attributes: Mapping[str, otel_types.AttributeValue]) -> None:
        return None

    def set_attribute(self, key: str, value: otel_types.AttributeValue) -> None:
        return None

    def add_event(
        self, name: str, attributes: otel_types.Attributes = None, timestamp: int | None = None
    ) -> None:
        return None

    def update_name(self, name: str) -> None:
        return None

    def is_recording(self) -> bool:
        return True

    def set_status(self, status: Status | StatusCode, description: str | None = None) -> None:
        return None

    def record_exception(
        self,
        exception: BaseException,
        attributes: otel_types.Attributes = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        return None


class ParentRecordingTracer(NoOpTracer):
    """Records the span each started span would be parented under."""

    def __init__(self) -> None:
        self.parents: list[Span] = []

    def start_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: otel_types.Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        self.parents.append(trace.get_current_span(context))
        return AmbientSpan()


async def test_stream_never_parents_consumer_spans_under_the_call_span() -> None:
    # The stream generator resumes inside the CONSUMER's context; a call span
    # left attached across yields would adopt everything the consumer traces
    # between events.
    engine = FakeEngine(stream_script=[happy_stream()])
    consumer_span = AmbientSpan()
    tracer = ParentRecordingTracer()
    events: list[RuntimeStreamEvent] = []
    with trace.use_span(consumer_span, end_on_exit=False):
        async for event in make_runtime(engine).stream(make_intent()):
            tracer.start_span("consumer-work")
            events.append(event)
    assert len(events) == 4
    assert tracer.parents == [consumer_span] * 4, (
        "a consumer span opened between stream events must keep the consumer's own parent"
    )


async def test_abandoning_a_stream_ends_it_without_context_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Abandoning mid-stream and finalizing from a DIFFERENT task is the shape
    # the event loop's async-generator finalizer uses; a span attached at the
    # first event would be detached here from a foreign context.
    engine = FakeEngine(stream_script=[happy_stream()])
    stream = make_runtime(engine).stream(make_intent())
    assert isinstance((await anext(stream)).event, StreamStart)
    assert isinstance(stream, AsyncGenerator)
    with caplog.at_level(logging.ERROR):
        await asyncio.ensure_future(stream.aclose())
    assert caplog.records == [], f"abandoning a stream logged: {caplog.text}"
    assert engine.streams_closed == 1


# ---------------------------------------------------------------------------
# spans: what the facade records on its own call span

EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
EMBED_CALL = EmbeddingCall(model="text-embedding-3-small", inputs=("alpha",), dimensions=Absent())
EMBED_CREDENTIAL = ProviderCredential(provider="openai", key="sk-openai-test-key-000")


async def test_generate_records_the_cost_estimate_on_the_call_span() -> None:
    tracer_provider = RecordingTracerProvider()
    engine = FakeEngine(generate_script=[succeeded()])
    outcome = await make_runtime(engine, tracer_provider=tracer_provider).generate(make_intent())
    priced = estimate_cost(outcome.meta)
    assert isinstance(priced, Present), "this test needs a model the price snapshot knows"
    (span,) = tracer_provider.tracer.spans
    assert span.attributes["provider_runtime.cost_estimate_usd_micros"] == (
        priced.value.amount_usd_micros
    )
    assert span.attributes["provider_runtime.attempt_count"] == 1


@pytest.mark.parametrize(
    ("response", "failure", "attempt_count"),
    [
        pytest.param(
            httpx.Response(500, json={"error": {"message": "boom", "type": "server_error"}}),
            TransientExhausted(attempts=2, cause=ProviderHttpUnavailable()),
            2,
            id="retry-exhaustion",
        ),
        pytest.param(
            httpx.Response(
                400,
                json={
                    "error": {
                        "message": "This model's maximum context length is 8192 tokens.",
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                    }
                },
            ),
            ProviderContextTooLarge(),
            1,
            id="context-overflow",
        ),
    ],
)
@respx.mock
async def test_embed_failure_arms_record_the_attempt_count_inside_an_errored_span(
    response: httpx.Response,
    failure: TransientExhausted | ProviderContextTooLarge,
    attempt_count: int,
) -> None:
    # Both arms — exhaustion and the returned context-overflow value — raise
    # INSIDE the span, so the attempt count lands on it either way and
    # `call_span` marks the span ERROR either way.
    respx.post(EMBEDDINGS_URL).mock(return_value=response)
    tracer_provider = RecordingTracerProvider()
    runtime = make_runtime(FakeEngine(), max_attempts=2, tracer_provider=tracer_provider)
    with pytest.raises(NonGenerationCallFailed) as exc_info:
        await runtime.embed(EMBED_CALL, credential=EMBED_CREDENTIAL)
    assert exc_info.value.failure == failure
    (span,) = tracer_provider.tracer.spans
    assert span.name == "embeddings text-embedding-3-small"
    assert span.attributes["provider_runtime.attempt_count"] == attempt_count
    assert span.exceptions == [exc_info.value]
    assert [status.status_code for status in span.statuses if isinstance(status, Status)] == [
        StatusCode.ERROR
    ]
    assert span.ended
