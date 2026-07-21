"""Provider runtime: the sole same-target retry owner over transport + codecs.

`ProviderRuntime` dispatches finalized provider calls through `Transport`,
routes bodies through the protocol's codec, and owns everything the codecs do
not: the retry boundary, the attempt trace, stream envelope sequencing,
cancellation, runtime-owned `Failed`/`Cancelled` terminals, and structured
promotion (codecs decode `TextContent` only; the plan-owning layer promotes to
`StructuredContent` when the plan's output arm is `strict_json`).

Retry rules (§9): retry only exactly classified `TransientCause` signals, only
before any semantic provider event, per `call.retry_policy`. Exhaustion folds
into `Failed(TransientExhausted)` preserving the normalized final signal and
the full attempt trace on meta. Defects raise. No fallback of any kind.

Attempt-trace timestamps are monotonic milliseconds (`time.monotonic`-derived)
suitable for durations only — they are not wall-clock times.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Protocol, assert_never

import httpx

from provider_runtime import anthropic, embeddings, gemini, moonshot, openai, openrouter
from provider_runtime._signals import (
    ClassifiedError,
    ExpectedFailureSignal,
    TransientStreamError,
)
from provider_runtime.catalog import CATALOG, Catalog
from provider_runtime.errors import ProtocolDefect
from provider_runtime.planning import EXTERNAL_LLM_RETRY
from provider_runtime.transport import (
    SseEvent,
    Transport,
    TransportResponse,
    _auth_header,
    _lowered,
)
from provider_runtime.types import (
    Absent,
    AttemptRecord,
    CallMeta,
    CallOutcome,
    Cancelled,
    CancelSignal,
    CodecStreamEvent,
    ContinuationDelta,
    EmbeddingCall,
    EmbeddingResponse,
    Failed,
    FinalAttempt,
    FinalizedProviderCall,
    FinalizedProviderRequest,
    NotDispatched,
    PossiblyBillable,
    Presence,
    Present,
    ProviderContextTooLarge,
    ProviderCredential,
    ProviderProtocol,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTarget,
    ProviderTimeout,
    RetryPolicy,
    RuntimeStreamEvent,
    StreamOutcome,
    StructuredContent,
    Succeeded,
    TerminalEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallDone,
    ToolCallStart,
    TranscriptionCall,
    TranscriptionResponse,
    TransientCause,
    TransientExhausted,
    TransportUnavailable,
    UsageEvent,
)

# Per-request transport timeout (httpx phase timeout: connect/read gaps, not
# total stream duration). Prior-art DEFAULT_TIMEOUT_S preserved.
_REQUEST_TIMEOUT_S: Final[float] = 45.0

# Semantic provider events (§9): once one has been yielded to the caller,
# internal retry is unsafe. StreamStart and TerminalEvent are not semantic.
_SEMANTIC_EVENTS: Final = (
    TextDelta,
    ToolCallStart,
    ToolCallDelta,
    ToolCallDone,
    ContinuationDelta,
    UsageEvent,
)


class NonGenerationCallFailed(Exception):
    """Expected-failure channel for the non-generation ports (embed/transcribe).

    Those ports return plain responses rather than `CallOutcome`, so retry
    exhaustion and provider context overflow surface as this exception carrying
    the same closed `ExpectedModelFailure` leaves `generate()` folds into
    `Failed`. Defects (credential rejection, quota exhaustion, protocol
    breakage) raise their own types as everywhere else.
    """

    def __init__(self, failure: TransientExhausted | ProviderContextTooLarge) -> None:
        super().__init__(type(failure).__name__)
        self.failure = failure


# ---------------------------------------------------------------------------
# Codec dispatch


class _Codec(Protocol):
    decode_response: Callable[[int, Mapping[str, str], bytes], CallOutcome]
    decode_stream: Callable[
        [Mapping[str, str], AsyncIterator[SseEvent]], AsyncIterator[CodecStreamEvent]
    ]
    classify_error: Callable[[int, Mapping[str, str], bytes], ClassifiedError]
    stream_request: Callable[[FinalizedProviderRequest], FinalizedProviderRequest]


def _codec_for(protocol: ProviderProtocol) -> _Codec:
    match protocol:
        case "openai_responses":
            return openai
        case "anthropic_messages":
            return anthropic
        case "gemini_generate_content":
            return gemini
        case "moonshot_chat":
            return moonshot
        case "openrouter_chat":
            return openrouter
        case _:
            assert_never(protocol)


# ---------------------------------------------------------------------------
# Retry mechanics (shared by every non-stream port; the stream has its own
# attempt loop with identical delay/deadline rules)


def _monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def _retry_delay_s(*, attempt: int, signal: TransientCause, policy: RetryPolicy) -> float:
    """Exponential `initial * 2^(attempt-1)` plus jitter, capped at max_delay.

    A Present `ProviderRateLimit.retry_after` is honored verbatim (no jitter)
    but still capped at max_delay."""
    if isinstance(signal, ProviderRateLimit) and isinstance(signal.retry_after, Present):
        return min(signal.retry_after.value, policy.max_delay_s)
    delay = policy.initial_delay_s * (2 ** (attempt - 1))
    if policy.jitter_s > 0:
        delay += random.uniform(0, policy.jitter_s)
    return min(delay, policy.max_delay_s)


def _deadline_exhausted(loop_started: float, policy: RetryPolicy, delay_s: float) -> bool:
    """Absent deadline means no wall-clock deadline at all."""
    match policy.deadline_s:
        case Present(value=deadline):
            return time.monotonic() - loop_started + delay_s > deadline
        case Absent():
            return False
        case _:
            assert_never(policy.deadline_s)


@dataclass(frozen=True, slots=True)
class _SendSucceeded:
    response: TransportResponse
    trace: tuple[AttemptRecord, ...]


@dataclass(frozen=True, slots=True)
class _SendFailed:
    failure: TransientExhausted | ProviderContextTooLarge
    trace: tuple[AttemptRecord, ...]
    # True once any attempt reached the provider (a response of any status,
    # or a transport error other than a pure pre-connect failure). Threads
    # into generate()'s Failed billability: NotDispatched when no attempt
    # ever got past connect.
    dispatched: bool


def _runtime_meta(
    target: ProviderTarget,
    trace: tuple[AttemptRecord, ...],
    billability: NotDispatched | PossiblyBillable,
) -> CallMeta:
    """Meta for runtime-constructed terminals (Failed/Cancelled).

    Request id and usage are deliberately Absent: a runtime terminal means no
    codec decoded an authoritative provider envelope for this call."""
    return CallMeta(
        provider=target.provider,
        model=target.model,
        provider_request_id=Absent(),
        upstream_provider=Absent(),
        usage=Absent(),
        attempt_trace=trace,
        billability=billability,
    )


def _with_trace(outcome: CallOutcome, trace: tuple[AttemptRecord, ...]) -> CallOutcome:
    """Rebuild a codec-decoded outcome with the runtime-owned attempt trace.

    Codecs always construct meta with attempt_trace=(); CallMeta and the
    outcome types are plain values (the no-replace negative gate covers stream
    EVENT types only)."""
    return replace(outcome, meta=replace(outcome.meta, attempt_trace=trace))


def _promote_succeeded(outcome: Succeeded, call: FinalizedProviderCall) -> Succeeded:
    """Structured promotion: the plan-owning layer's half of StrictJsonOutput.

    Codecs decode TextContent only (their decode signatures carry no plan). On
    a Succeeded outcome for a strict_json plan, the terminal text must strictly
    parse to a JSON object; anything else is a ProtocolDefect."""
    if call.output_kind != "strict_json":
        return outcome
    content = outcome.response.content
    if isinstance(content, StructuredContent):
        return outcome
    try:
        payload = json.loads(content.text)
    except json.JSONDecodeError:
        raise ProtocolDefect(
            code="invalid_structured_output",
            message=(
                f"{call.request.target.provider} strict_json output was not valid JSON "
                f"({len(content.text)} chars)"
            ),
        ) from None
    if not isinstance(payload, dict):
        raise ProtocolDefect(
            code="structured_output_not_object",
            message=(
                f"{call.request.target.provider} strict_json output parsed to "
                f"{type(payload).__name__}, not a JSON object"
            ),
        )
    return replace(
        outcome,
        response=replace(outcome.response, content=StructuredContent(payload, content.text)),
    )


async def _guarded_anext(source: AsyncIterator[CodecStreamEvent]) -> CodecStreamEvent:
    try:
        return await anext(source)
    except StopAsyncIteration:
        # Codec contract breach: decode_stream must end with a TerminalEvent or
        # raise TransientStreamError; bare exhaustion is a defect.
        raise ProtocolDefect(
            code="missing_terminal_event",
            message="codec stream ended without a terminal event or transient signal",
        ) from None


async def _next_or_cancel(
    source: AsyncIterator[CodecStreamEvent], cancel: CancelSignal | None
) -> CodecStreamEvent | None:
    """Race the next codec event against cancellation; None means cancelled.

    Cancellation-safe: if the CONSUMING task is itself cancelled while parked
    in the race, both tasks are cancelled and settled (via `asyncio.wait`,
    which never raises a task's own exception) before the external
    CancelledError is re-raised. This guarantees the codec generator is no
    longer running by the time stream()'s `finally: await source.aclose()`
    runs — otherwise aclose() raises "already running", masking the
    CancelledError."""
    if cancel is None:
        return await _guarded_anext(source)
    next_task = asyncio.ensure_future(_guarded_anext(source))
    cancel_task = asyncio.ensure_future(cancel.wait())
    tasks: set[asyncio.Task[Any]] = {next_task, cancel_task}
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.wait(tasks)  # settle; wait() never raises task exceptions
        for t in tasks:
            if not t.cancelled():
                t.exception()  # retrieve, so no 'Task exception was never retrieved'
        raise
    if next_task in done:
        cancel_task.cancel()
        await asyncio.wait({cancel_task})
        if not cancel_task.cancelled():
            cancel_task.exception()
        return next_task.result()
    next_task.cancel()
    await asyncio.wait({next_task})
    if not next_task.cancelled():
        next_task.exception()
    return None


# ---------------------------------------------------------------------------
# Runtime


class ProviderRuntime:
    def __init__(self, http: httpx.AsyncClient, catalog: Catalog = CATALOG) -> None:
        self._http = http
        self._catalog = catalog
        self._transport = Transport(http)

    # -- generate -----------------------------------------------------------

    async def generate(
        self, call: FinalizedProviderCall, *, credential: ProviderCredential
    ) -> CallOutcome:
        codec = _codec_for(call.request.protocol)
        sent = await self._send_with_retry(
            lambda: self._transport.send(call.request, credential, _REQUEST_TIMEOUT_S),
            codec.classify_error,
            call.retry_policy,
        )
        match sent:
            case _SendFailed(failure=failure, trace=trace, dispatched=dispatched):
                billability = PossiblyBillable() if dispatched else NotDispatched()
                return Failed(
                    meta=_runtime_meta(call.request.target, trace, billability),
                    failure=failure,
                )
            case _SendSucceeded(response=response, trace=trace):
                try:
                    outcome = codec.decode_response(
                        response.status, response.headers, response.body
                    )
                except ExpectedFailureSignal as signal:
                    return Failed(
                        meta=_runtime_meta(call.request.target, trace, PossiblyBillable()),
                        failure=signal.failure,
                    )
                outcome = _with_trace(outcome, trace)
                if isinstance(outcome, Succeeded):
                    outcome = _promote_succeeded(outcome, call)
                return outcome
            case _:
                assert_never(sent)

    # -- stream -------------------------------------------------------------

    async def stream(
        self,
        call: FinalizedProviderCall,
        *,
        credential: ProviderCredential,
        cancel: CancelSignal | None = None,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        codec = _codec_for(call.request.protocol)
        streaming_request = codec.stream_request(call.request)
        policy = call.retry_policy
        target = call.request.target
        trace: list[AttemptRecord] = []
        seq = 0
        dispatched = False
        loop_started = time.monotonic()

        def final_record(
            attempt: int, status: Presence[int], started_ms: int
        ) -> tuple[AttemptRecord, ...]:
            trace.append(
                AttemptRecord(
                    attempt=attempt,
                    signal=FinalAttempt(),
                    status_code=status,
                    started_at_ms=started_ms,
                    ended_at_ms=_monotonic_ms(),
                )
            )
            return tuple(trace)

        def terminal(outcome: StreamOutcome) -> RuntimeStreamEvent:
            nonlocal seq
            seq += 1
            return RuntimeStreamEvent(seq=seq, event=TerminalEvent(outcome=outcome))

        for attempt in range(1, policy.max_attempts + 1):
            started_ms = _monotonic_ms()
            if cancel is not None and cancel.is_set():
                billability = PossiblyBillable() if dispatched else NotDispatched()
                yield terminal(
                    Cancelled(
                        meta=_runtime_meta(
                            target, final_record(attempt, Absent(), started_ms), billability
                        )
                    )
                )
                return
            semantic_emitted = False
            status: Presence[int] = Absent()
            signal: TransientCause | None = None
            try:
                async with self._transport.stream(
                    streaming_request, credential, _REQUEST_TIMEOUT_S
                ) as response:
                    dispatched = True
                    status = Present(response.status)
                    if not 200 <= response.status < 300:
                        body = await response.read_error_body()
                        classified = codec.classify_error(response.status, response.headers, body)
                        if isinstance(classified, ProviderContextTooLarge):
                            yield terminal(
                                Failed(
                                    meta=_runtime_meta(
                                        target,
                                        final_record(attempt, status, started_ms),
                                        PossiblyBillable(),
                                    ),
                                    failure=classified,
                                )
                            )
                            return
                        signal = classified
                    else:
                        source = codec.decode_stream(response.headers, response.events)
                        try:
                            while True:
                                event = await _next_or_cancel(source, cancel)
                                if event is None:
                                    yield terminal(
                                        Cancelled(
                                            meta=_runtime_meta(
                                                target,
                                                final_record(attempt, status, started_ms),
                                                PossiblyBillable(),
                                            )
                                        )
                                    )
                                    return
                                if isinstance(event, TerminalEvent):
                                    stream_trace = final_record(attempt, status, started_ms)
                                    outcome: StreamOutcome = replace(
                                        event.outcome,
                                        meta=replace(
                                            event.outcome.meta, attempt_trace=stream_trace
                                        ),
                                    )
                                    if isinstance(outcome, Succeeded):
                                        outcome = _promote_succeeded(outcome, call)
                                    yield terminal(outcome)
                                    return
                                seq += 1
                                yield RuntimeStreamEvent(seq=seq, event=event)
                                if isinstance(event, _SEMANTIC_EVENTS):
                                    semantic_emitted = True
                        finally:
                            if isinstance(source, AsyncGenerator):
                                await source.aclose()
            except ExpectedFailureSignal as expected:
                # Always terminal: tool-argument parse failure follows tool
                # deltas, so semantic content was involved; no retry.
                yield terminal(
                    Failed(
                        meta=_runtime_meta(
                            target, final_record(attempt, status, started_ms), PossiblyBillable()
                        ),
                        failure=expected.failure,
                    )
                )
                return
            except TransientStreamError as stream_error:
                signal = stream_error.cause
            except httpx.TimeoutException:
                signal = ProviderTimeout()
            except httpx.TransportError:
                signal = TransportUnavailable()
            if signal is None:
                raise AssertionError("stream attempt ended without terminal or signal")
            if semantic_emitted:
                # Post-semantic-output, ANY transient is terminal with the
                # stream-interruption leaf rebuilt to carry the true flag.
                failure_trace = final_record(attempt, status, started_ms)
                yield terminal(
                    Failed(
                        meta=_runtime_meta(target, failure_trace, PossiblyBillable()),
                        failure=TransientExhausted(
                            attempts=len(failure_trace),
                            cause=ProviderStreamInterrupted(partial_output=True),
                        ),
                    )
                )
                return
            delay_s = _retry_delay_s(attempt=attempt, signal=signal, policy=policy)
            if attempt >= policy.max_attempts or _deadline_exhausted(loop_started, policy, delay_s):
                failure_trace = final_record(attempt, status, started_ms)
                billability = PossiblyBillable() if dispatched else NotDispatched()
                yield terminal(
                    Failed(
                        meta=_runtime_meta(target, failure_trace, billability),
                        failure=TransientExhausted(attempts=len(failure_trace), cause=signal),
                    )
                )
                return
            trace.append(
                AttemptRecord(
                    attempt=attempt,
                    signal=signal,
                    status_code=status,
                    started_at_ms=started_ms,
                    ended_at_ms=_monotonic_ms(),
                )
            )
            await asyncio.sleep(delay_s)
        raise AssertionError("unreachable stream retry loop exit")

    # -- non-generation ports (openai-only; central EXTERNAL_LLM_RETRY) ------

    async def embed(
        self, call: EmbeddingCall, *, credential: ProviderCredential
    ) -> EmbeddingResponse:
        request = embeddings.build_embedding_request(call)
        sent = await self._send_with_retry(
            lambda: self._transport.send(request, credential, _REQUEST_TIMEOUT_S),
            openai.classify_error,
            EXTERNAL_LLM_RETRY,
        )
        match sent:
            case _SendFailed(failure=failure):
                raise NonGenerationCallFailed(failure)
            case _SendSucceeded(response=response):
                return embeddings.parse_embedding_response(
                    response.status,
                    response.headers,
                    response.body,
                    expected_count=len(call.inputs),
                )
            case _:
                assert_never(sent)

    async def transcribe(
        self, call: TranscriptionCall, *, credential: ProviderCredential
    ) -> TranscriptionResponse:
        request = openai.build_transcription_request(
            model=call.model,
            filename=call.filename,
            audio=call.content,
            media_type=call.media_type,
        )

        async def send() -> TransportResponse:
            # Multipart is a transport-special port (TranscriptionHttpRequest);
            # auth header naming stays owned by transport._auth_header.
            name, value = _auth_header(credential)
            response = await self._http.post(
                request.url,
                data=dict(request.form_fields),
                files={"file": (request.filename, request.content, request.media_type)},
                headers={name: value},
                timeout=httpx.Timeout(_REQUEST_TIMEOUT_S),
            )
            return TransportResponse(
                status=response.status_code,
                headers=_lowered(response.headers),
                body=response.content,
            )

        sent = await self._send_with_retry(send, openai.classify_error, EXTERNAL_LLM_RETRY)
        match sent:
            case _SendFailed(failure=failure):
                raise NonGenerationCallFailed(failure)
            case _SendSucceeded(response=response):
                result = openai.parse_transcription_response(
                    response.status, response.headers, response.body
                )
                return TranscriptionResponse(text=result.text, usage=result.usage)
            case _:
                assert_never(sent)

    # -- shared non-stream retry engine --------------------------------------

    async def _send_with_retry(
        self,
        send: Callable[[], Awaitable[TransportResponse]],
        classify: Callable[[int, Mapping[str, str], bytes], ClassifiedError],
        policy: RetryPolicy,
    ) -> _SendSucceeded | _SendFailed:
        trace: list[AttemptRecord] = []
        loop_started = time.monotonic()
        # True once any attempt reaches the provider. httpx.ConnectError is a
        # pure pre-connect failure (no bytes sent) and does NOT set it; every
        # other transport error implies the connection was at least opened.
        dispatched = False
        for attempt in range(1, policy.max_attempts + 1):
            started_ms = _monotonic_ms()
            status: Presence[int] = Absent()
            signal: TransientCause
            try:
                response = await send()
            except httpx.ConnectError:
                signal = TransportUnavailable()
            except httpx.TimeoutException:
                dispatched = True
                signal = ProviderTimeout()
            except httpx.TransportError:
                dispatched = True
                signal = TransportUnavailable()
            else:
                dispatched = True
                status = Present(response.status)
                if 200 <= response.status < 300:
                    trace.append(
                        AttemptRecord(
                            attempt=attempt,
                            signal=FinalAttempt(),
                            status_code=status,
                            started_at_ms=started_ms,
                            ended_at_ms=_monotonic_ms(),
                        )
                    )
                    return _SendSucceeded(response=response, trace=tuple(trace))
                # classify_error raises defects (credential/quota/protocol)
                # which propagate; it returns only exact classified values.
                classified = classify(response.status, response.headers, response.body)
                if isinstance(classified, ProviderContextTooLarge):
                    trace.append(
                        AttemptRecord(
                            attempt=attempt,
                            signal=FinalAttempt(),
                            status_code=status,
                            started_at_ms=started_ms,
                            ended_at_ms=_monotonic_ms(),
                        )
                    )
                    return _SendFailed(
                        failure=classified, trace=tuple(trace), dispatched=dispatched
                    )
                signal = classified
            delay_s = _retry_delay_s(attempt=attempt, signal=signal, policy=policy)
            if attempt >= policy.max_attempts or _deadline_exhausted(loop_started, policy, delay_s):
                trace.append(
                    AttemptRecord(
                        attempt=attempt,
                        signal=FinalAttempt(),
                        status_code=status,
                        started_at_ms=started_ms,
                        ended_at_ms=_monotonic_ms(),
                    )
                )
                return _SendFailed(
                    failure=TransientExhausted(attempts=len(trace), cause=signal),
                    trace=tuple(trace),
                    dispatched=dispatched,
                )
            trace.append(
                AttemptRecord(
                    attempt=attempt,
                    signal=signal,
                    status_code=status,
                    started_at_ms=started_ms,
                    ended_at_ms=_monotonic_ms(),
                )
            )
            await asyncio.sleep(delay_s)
        raise AssertionError("unreachable retry loop exit")


__all__ = [
    "NonGenerationCallFailed",
    "ProviderRuntime",
]
