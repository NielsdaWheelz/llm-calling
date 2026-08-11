"""Behavior tests for `provider_runtime.otel` (opentelemetry-api only).

Two paths, per the library rule:
- no configured SDK → the api's own no-op tracer, asserted against the real
  global (never configured here);
- attribute capture → a minimal in-test TracerProvider/Tracer/Span double at
  the external boundary (the opentelemetry-sdk is deliberately not installed).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import get_args

import pytest
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    NoOpTracer,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    TracerProvider,
)
from opentelemetry.trace.span import INVALID_SPAN_CONTEXT
from opentelemetry.util import types as otel_types

from provider_runtime.otel import SEMCONV_VERSION, as_current, call_span, record_outcome
from provider_runtime.types import (
    Absent,
    AttemptRecord,
    AttemptSignal,
    Billability,
    CallMeta,
    ConfirmedNonBillable,
    CostEstimate,
    FinalAttempt,
    NotDispatched,
    PossiblyBillable,
    Presence,
    Present,
    ProviderHttpUnavailable,
    ProviderName,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTimeout,
    TokenUsage,
    TransportUnavailable,
)

# ---------------------------------------------------------------------------
# Recording doubles at the api boundary

REVISION = "2026-08-09.1"

PROVIDER: ProviderName = "anthropic"
MODEL = "claude-opus-5"

NO_COST: Presence[CostEstimate] = Absent()


class RecordedSpan(Span):
    """Minimal recording implementation of the api's abstract Span."""

    def __init__(self, name: str, kind: SpanKind, attributes: otel_types.Attributes) -> None:
        self.name = name
        self.kind = kind
        self.attributes: dict[str, object] = dict(attributes or {})
        self.events: list[tuple[str, dict[str, object]]] = []
        self.statuses: list[Status | StatusCode] = []
        self.exceptions: list[BaseException] = []
        self.ended = False

    def end(self, end_time: int | None = None) -> None:
        self.ended = True

    def get_span_context(self) -> SpanContext:
        return INVALID_SPAN_CONTEXT

    def set_attributes(self, attributes: Mapping[str, otel_types.AttributeValue]) -> None:
        self.attributes.update(attributes)

    def set_attribute(self, key: str, value: otel_types.AttributeValue) -> None:
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: otel_types.Attributes = None,
        timestamp: int | None = None,
    ) -> None:
        self.events.append((name, dict(attributes or {})))

    def update_name(self, name: str) -> None:
        self.name = name

    def is_recording(self) -> bool:
        return True

    def set_status(self, status: Status | StatusCode, description: str | None = None) -> None:
        self.statuses.append(status)

    def record_exception(
        self,
        exception: BaseException,
        attributes: otel_types.Attributes = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        self.exceptions.append(exception)


class RecordingTracer(NoOpTracer):
    """Records started spans; inherits the api's start_as_current_span driver."""

    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []

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
        span = RecordedSpan(name, kind, attributes)
        self.spans.append(span)
        return span


class RecordingTracerProvider(TracerProvider):
    def __init__(self) -> None:
        self.tracer = RecordingTracer()
        self.scopes: list[tuple[str, str | None]] = []

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
        attributes: otel_types.Attributes = None,
    ) -> Tracer:
        self.scopes.append((instrumenting_module_name, schema_url))
        return self.tracer


_POSSIBLY_BILLABLE = PossiblyBillable()
_ABSENT = Absent()


def make_meta(
    *,
    attempts: int = 1,
    billability: Billability = _POSSIBLY_BILLABLE,
    usage: Presence[TokenUsage] = _ABSENT,
    attempt_trace: tuple[AttemptRecord, ...] | None = None,
) -> CallMeta:
    recorded = attempt_trace if attempt_trace is not None else _default_trace(attempts)
    return CallMeta(
        provider="anthropic",
        model=MODEL,
        provider_request_id=Present("req_abc"),
        upstream_provider=Absent(),
        usage=usage,
        attempt_trace=recorded,
        billability=billability,
        native_reasoning=Present('{"output_config":{"effort":"high"}}'),
        registry_revision=REVISION,
    )


def _default_trace(attempts: int) -> tuple[AttemptRecord, ...]:
    return tuple(
        AttemptRecord(
            attempt=number,
            signal=FinalAttempt() if number == attempts else ProviderTimeout(),
            status_code=Absent(),
            started_at_ms=number * 10,
            ended_at_ms=number * 10 + 5,
        )
        for number in range(1, attempts + 1)
    )


START_ATTRIBUTES = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.model": MODEL,
    "provider_runtime.semconv_version": SEMCONV_VERSION,
}

TERMINAL_ATTRIBUTES = {
    "provider_runtime.attempt_count": 1,
    "provider_runtime.billability": "possibly_billable",
    "provider_runtime.registry_revision": REVISION,
}


# ---------------------------------------------------------------------------
# call_span — request side


def test_span_is_named_operation_then_request_model() -> None:
    provider = RecordingTracerProvider()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider):
        pass
    (span,) = provider.tracer.spans
    assert span.name == "chat claude-opus-5", f"semconv span name violated: {span.name!r}"
    assert span.kind == SpanKind.CLIENT


def test_call_span_sets_exactly_the_gen_ai_request_attributes() -> None:
    provider = RecordingTracerProvider()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider):
        pass
    (span,) = provider.tracer.spans
    # Exact-dict equality is the closed-set pin: no message content, no
    # continuation payloads, nothing beyond the documented attributes.
    assert span.attributes == START_ATTRIBUTES, f"unexpected attributes: {span.attributes}"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("gemini", "gcp.gemini"),
        ("moonshot", "moonshot"),
        ("openrouter", "openrouter"),
        ("deepseek", "deepseek"),
        ("xai", "x_ai"),
    ],
)
def test_provider_name_uses_the_semconv_well_known_value(
    provider: ProviderName, expected: str
) -> None:
    # semconv 1.37 registry: "If one of them applies, then the respective value
    # MUST be used" — gemini and xai are named gcp.gemini / x_ai there;
    # moonshot and openrouter are absent from the list and stay custom.
    tracer_provider = RecordingTracerProvider()
    with call_span("chat", provider=provider, model=MODEL, tracer_provider=tracer_provider):
        pass
    (span,) = tracer_provider.tracer.spans
    assert span.attributes["gen_ai.provider.name"] == expected


def test_every_provider_name_is_mapped() -> None:
    # The parametrisation above must cover the whole ProviderName literal, so a
    # new provider cannot ship with an unchecked semconv value.
    covered = {"openai", "anthropic", "gemini", "moonshot", "openrouter", "deepseek", "xai"}
    assert covered == set(get_args(ProviderName.__value__))


def test_semconv_version_is_pinned_and_recorded() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", SEMCONV_VERSION), (
        f"SEMCONV_VERSION must be a pinned release, got {SEMCONV_VERSION!r}"
    )
    provider = RecordingTracerProvider()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider):
        pass
    assert provider.scopes == [
        ("provider_runtime", f"https://opentelemetry.io/schemas/{SEMCONV_VERSION}")
    ], f"tracer scope must pin the semconv schema, got {provider.scopes}"


def test_call_span_ends_the_span_on_exit() -> None:
    provider = RecordingTracerProvider()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
        assert isinstance(span, RecordedSpan)
        assert not span.ended
    assert span.ended, "exiting call_span must end the span"


def test_call_span_records_an_escaping_exception_and_still_ends_the_span() -> None:
    provider = RecordingTracerProvider()
    failure = RuntimeError("engine blew up")
    with pytest.raises(RuntimeError):
        with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider):
            raise failure
    (span,) = provider.tracer.spans
    assert span.exceptions == [failure]
    assert [status.status_code for status in span.statuses if isinstance(status, Status)] == [
        StatusCode.ERROR
    ]
    assert span.ended


def test_call_span_ends_the_span_on_generator_exit_without_marking_it_failed() -> None:
    # An abandoned stream unwinds its `with call_span` with GeneratorExit; that
    # is not a call failure, but the span must still end deterministically.
    provider = RecordingTracerProvider()
    with pytest.raises(GeneratorExit):
        with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider):
            raise GeneratorExit
    (span,) = provider.tracer.spans
    assert span.ended
    assert span.statuses == []
    assert span.exceptions == []


# ---------------------------------------------------------------------------
# Ambient context — the call span is only current around engine interaction


def test_call_span_does_not_attach_the_span_to_the_ambient_context() -> None:
    # An async generator resumes in its CONSUMER's context: a span attached for
    # the whole call would parent consumer spans and detach from a foreign
    # context at aclose.
    provider = RecordingTracerProvider()
    outer = trace.get_current_span()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
        assert trace.get_current_span() is outer, "call_span must not become the current span"
        assert trace.get_current_span() is not span


def test_as_current_attaches_the_span_for_its_block_only() -> None:
    provider = RecordingTracerProvider()
    outer = trace.get_current_span()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
        with as_current(span):
            assert trace.get_current_span() is span
        assert trace.get_current_span() is outer


def test_as_current_neither_ends_the_span_nor_marks_an_exception() -> None:
    # Transient failures cross this scope on every retried attempt.
    provider = RecordingTracerProvider()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
        assert isinstance(span, RecordedSpan)
        with pytest.raises(RuntimeError):
            with as_current(span):
                raise RuntimeError("transient")
        assert not span.ended
        assert span.statuses == []
        assert span.exceptions == []


# ---------------------------------------------------------------------------
# record_outcome — terminal side


def test_record_outcome_sets_attempt_count_billability_and_revision() -> None:
    provider = RecordingTracerProvider()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
        record_outcome(
            span, make_meta(attempts=3, billability=PossiblyBillable()), cost_estimate=NO_COST
        )
    (recorded,) = provider.tracer.spans
    assert recorded.attributes == START_ATTRIBUTES | {
        "provider_runtime.attempt_count": 3,
        "provider_runtime.billability": "possibly_billable",
        "provider_runtime.registry_revision": REVISION,
    }, f"unexpected terminal attributes: {recorded.attributes}"


def test_each_billability_variant_has_a_distinct_tag() -> None:
    tags: dict[str, object] = {}
    for billability in (NotDispatched(), PossiblyBillable(), ConfirmedNonBillable()):
        provider = RecordingTracerProvider()
        with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
            record_outcome(span, make_meta(billability=billability), cost_estimate=NO_COST)
        tag = provider.tracer.spans[0].attributes["provider_runtime.billability"]
        tags[type(billability).__name__] = tag
    assert tags == {
        "NotDispatched": "not_dispatched",
        "PossiblyBillable": "possibly_billable",
        "ConfirmedNonBillable": "confirmed_non_billable",
    }, f"billability tags drifted: {tags}"


def test_present_cost_estimate_is_recorded_and_absent_records_nothing() -> None:
    estimate = CostEstimate(
        amount_usd_micros=1234, source="genai-prices@2026-08-01", as_of=date(2026, 8, 1)
    )
    for cost, expected in (
        (Present(estimate), {"provider_runtime.cost_estimate_usd_micros": 1234}),
        (NO_COST, {}),
    ):
        provider = RecordingTracerProvider()
        with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
            record_outcome(span, make_meta(), cost_estimate=cost)
        attributes = provider.tracer.spans[0].attributes
        assert attributes == START_ATTRIBUTES | TERMINAL_ATTRIBUTES | expected, (
            f"cost={cost!r} recorded {attributes}"
        )


def test_usage_counts_are_recorded_without_summing_cache_reads() -> None:
    usage = TokenUsage(
        input_tokens=1200,  # cache-inclusive by contract invariant
        output_tokens=40,
        total_tokens=1240,
        reasoning_tokens=Absent(),
        cache_read_input_tokens=Present(1000),
        cache_write_input_tokens=Present(50),
    )
    provider = RecordingTracerProvider()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
        record_outcome(span, make_meta(usage=Present(usage)), cost_estimate=NO_COST)
    attributes = provider.tracer.spans[0].attributes
    assert attributes == START_ATTRIBUTES | TERMINAL_ATTRIBUTES | {
        "gen_ai.usage.input_tokens": 1200,
        "gen_ai.usage.output_tokens": 40,
        "provider_runtime.usage.cache_read_input_tokens": 1000,
        "provider_runtime.usage.cache_write_input_tokens": 50,
    }, f"unexpected usage attributes: {attributes}"
    assert attributes["gen_ai.usage.input_tokens"] == 1200, (
        "input_tokens is already cache-inclusive; cache reads must never be summed in"
    )


def test_absent_usage_and_cache_components_record_no_usage_attributes() -> None:
    bare = TokenUsage(
        input_tokens=10,
        output_tokens=2,
        total_tokens=12,
        reasoning_tokens=Absent(),
        cache_read_input_tokens=Absent(),
        cache_write_input_tokens=Absent(),
    )
    for usage, expected_usage_keys in (
        (Absent(), set()),
        (Present(bare), {"gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"}),
    ):
        provider = RecordingTracerProvider()
        with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
            record_outcome(span, make_meta(usage=usage), cost_estimate=NO_COST)
        attributes = provider.tracer.spans[0].attributes
        usage_keys = {key for key in attributes if "usage" in key}
        assert usage_keys == expected_usage_keys, (
            f"usage={usage!r} must record exactly {expected_usage_keys}, got {usage_keys}"
        )


# ---------------------------------------------------------------------------
# record_outcome — the attempt trace mirrored onto the span (spec §8)


def test_every_attempt_is_mirrored_as_a_bounded_span_event() -> None:
    attempt_trace = (
        AttemptRecord(
            attempt=1,
            signal=ProviderRateLimit(retry_after=Present(2.0)),
            status_code=Present(429),
            started_at_ms=100,
            ended_at_ms=140,
        ),
        AttemptRecord(
            attempt=2,
            signal=FinalAttempt(),
            status_code=Present(200),
            started_at_ms=200,
            ended_at_ms=275,
        ),
    )
    provider = RecordingTracerProvider()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
        record_outcome(span, make_meta(attempt_trace=attempt_trace), cost_estimate=NO_COST)
    assert provider.tracer.spans[0].events == [
        (
            "provider_runtime.attempt",
            {
                "provider_runtime.attempt.number": 1,
                "provider_runtime.attempt.signal": "provider_rate_limit",
                "provider_runtime.attempt.duration_ms": 40,
                "provider_runtime.attempt.status_code": 429,
            },
        ),
        (
            "provider_runtime.attempt",
            {
                "provider_runtime.attempt.number": 2,
                "provider_runtime.attempt.signal": "final_attempt",
                "provider_runtime.attempt.duration_ms": 75,
                "provider_runtime.attempt.status_code": 200,
            },
        ),
    ], f"attempt trace not mirrored: {provider.tracer.spans[0].events}"


def test_an_attempt_without_a_status_code_records_no_status_attribute() -> None:
    provider = RecordingTracerProvider()
    with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
        record_outcome(span, make_meta(), cost_estimate=NO_COST)
    ((_, attributes),) = provider.tracer.spans[0].events
    assert "provider_runtime.attempt.status_code" not in attributes


def test_each_attempt_signal_has_a_distinct_tag() -> None:
    signals: tuple[AttemptSignal, ...] = (
        FinalAttempt(),
        ProviderRateLimit(retry_after=Absent()),
        ProviderTimeout(),
        ProviderHttpUnavailable(),
        TransportUnavailable(),
        ProviderStreamInterrupted(partial_output=True),
    )
    tags: list[object] = []
    for signal in signals:
        record = AttemptRecord(
            attempt=1, signal=signal, status_code=Absent(), started_at_ms=0, ended_at_ms=1
        )
        provider = RecordingTracerProvider()
        with call_span("chat", provider=PROVIDER, model=MODEL, tracer_provider=provider) as span:
            record_outcome(span, make_meta(attempt_trace=(record,)), cost_estimate=NO_COST)
        ((_, attributes),) = provider.tracer.spans[0].events
        tags.append(attributes["provider_runtime.attempt.signal"])
    assert tags == [
        "final_attempt",
        "provider_rate_limit",
        "provider_timeout",
        "provider_http_unavailable",
        "transport_unavailable",
        "provider_stream_interrupted",
    ], f"attempt signal tags drifted: {tags}"


# ---------------------------------------------------------------------------
# No-op path — no SDK configured (opentelemetry-sdk is not installed and this
# suite never sets a global tracer provider)


def test_without_a_configured_sdk_the_span_is_the_apis_non_recording_span() -> None:
    with call_span("chat", provider=PROVIDER, model=MODEL) as span:
        assert isinstance(span, NonRecordingSpan), (
            f"expected the api's no-op span, got {type(span).__name__}"
        )
        assert not span.is_recording()


def test_record_outcome_is_a_no_op_on_a_non_recording_span() -> None:
    usage = TokenUsage(
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        reasoning_tokens=Absent(),
        cache_read_input_tokens=Absent(),
        cache_write_input_tokens=Absent(),
    )
    with call_span("chat", provider=PROVIDER, model=MODEL) as span:
        record_outcome(
            span, make_meta(usage=Present(usage)), cost_estimate=NO_COST
        )  # must not raise
    assert not span.is_recording()
