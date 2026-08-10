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

from provider_runtime.otel import SEMCONV_VERSION, call_span, record_outcome
from provider_runtime.registry import ModelRow
from provider_runtime.types import (
    Absent,
    AttemptRecord,
    Billability,
    CallMeta,
    ConfirmedNonBillable,
    FinalAttempt,
    NotDispatched,
    PossiblyBillable,
    Presence,
    Present,
    ProviderTimeout,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Recording doubles at the api boundary

REVISION = "2026-08-09.1"

ROW = ModelRow(
    ref="anthropic:opus",
    provider="anthropic",
    model_id="claude-opus-5",
    engine="anthropic_messages",
    base_url=Absent(),
    context_window=200_000,
    max_output_tokens=32_000,
    modalities=frozenset({"text"}),
    tools=True,
    streaming=True,
    structured="native",
    reasoning=Absent(),
    continuation_codec="anthropic.v1",
    correlation="header",
    routing=Absent(),
)


class RecordedSpan(Span):
    """Minimal recording implementation of the api's abstract Span."""

    def __init__(self, name: str, kind: SpanKind, attributes: otel_types.Attributes) -> None:
        self.name = name
        self.kind = kind
        self.attributes: dict[str, object] = dict(attributes or {})
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
        pass

    def update_name(self, name: str) -> None:
        self.name = name

    def is_recording(self) -> bool:
        return True

    def set_status(self, status: Status | StatusCode, description: str | None = None) -> None:
        pass

    def record_exception(
        self,
        exception: BaseException,
        attributes: otel_types.Attributes = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        pass


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
) -> CallMeta:
    trace = tuple(
        AttemptRecord(
            attempt=number,
            signal=FinalAttempt() if number == attempts else ProviderTimeout(),
            status_code=Absent(),
            started_at_ms=number * 10,
            ended_at_ms=number * 10 + 5,
        )
        for number in range(1, attempts + 1)
    )
    return CallMeta(
        provider="anthropic",
        model="claude-opus-5",
        provider_request_id=Present("req_abc"),
        upstream_provider=Absent(),
        usage=usage,
        attempt_trace=trace,
        billability=billability,
        native_reasoning=Present("high"),
        registry_revision=REVISION,
    )


START_ATTRIBUTES = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.model": "claude-opus-5",
    "provider_runtime.semconv_version": SEMCONV_VERSION,
}


# ---------------------------------------------------------------------------
# call_span — request side


def test_span_is_named_operation_then_request_model() -> None:
    provider = RecordingTracerProvider()
    with call_span("chat", provider=ROW.provider, model=ROW.model_id, tracer_provider=provider):
        pass
    (span,) = provider.tracer.spans
    assert span.name == "chat claude-opus-5", f"semconv span name violated: {span.name!r}"
    assert span.kind == SpanKind.CLIENT


def test_call_span_sets_exactly_the_gen_ai_request_attributes() -> None:
    provider = RecordingTracerProvider()
    with call_span("chat", provider=ROW.provider, model=ROW.model_id, tracer_provider=provider):
        pass
    (span,) = provider.tracer.spans
    # Exact-dict equality is the closed-set pin: no message content, no
    # continuation payloads, nothing beyond the documented attributes.
    assert span.attributes == START_ATTRIBUTES, f"unexpected attributes: {span.attributes}"


def test_semconv_version_is_pinned_and_recorded() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", SEMCONV_VERSION), (
        f"SEMCONV_VERSION must be a pinned release, got {SEMCONV_VERSION!r}"
    )
    provider = RecordingTracerProvider()
    with call_span("chat", provider=ROW.provider, model=ROW.model_id, tracer_provider=provider):
        pass
    assert provider.scopes == [
        ("provider_runtime", f"https://opentelemetry.io/schemas/{SEMCONV_VERSION}")
    ], f"tracer scope must pin the semconv schema, got {provider.scopes}"


def test_call_span_ends_the_span_on_exit() -> None:
    provider = RecordingTracerProvider()
    with call_span("chat", provider=ROW.provider, model=ROW.model_id, tracer_provider=provider) as span:
        assert isinstance(span, RecordedSpan)
        assert not span.ended
    assert span.ended, "exiting call_span must end the span"


# ---------------------------------------------------------------------------
# record_outcome — terminal side


def test_record_outcome_sets_attempt_count_billability_and_revision() -> None:
    provider = RecordingTracerProvider()
    with call_span("chat", provider=ROW.provider, model=ROW.model_id, tracer_provider=provider) as span:
        record_outcome(span, make_meta(attempts=3, billability=PossiblyBillable()))
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
        with call_span("chat", provider=ROW.provider, model=ROW.model_id, tracer_provider=provider) as span:
            record_outcome(span, make_meta(billability=billability))
        tag = provider.tracer.spans[0].attributes["provider_runtime.billability"]
        tags[type(billability).__name__] = tag
    assert tags == {
        "NotDispatched": "not_dispatched",
        "PossiblyBillable": "possibly_billable",
        "ConfirmedNonBillable": "confirmed_non_billable",
    }, f"billability tags drifted: {tags}"


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
    with call_span("chat", provider=ROW.provider, model=ROW.model_id, tracer_provider=provider) as span:
        record_outcome(span, make_meta(usage=Present(usage)))
    attributes = provider.tracer.spans[0].attributes
    assert attributes == START_ATTRIBUTES | {
        "provider_runtime.attempt_count": 1,
        "provider_runtime.billability": "possibly_billable",
        "provider_runtime.registry_revision": REVISION,
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
        with call_span("chat", provider=ROW.provider, model=ROW.model_id, tracer_provider=provider) as span:
            record_outcome(span, make_meta(usage=usage))
        attributes = provider.tracer.spans[0].attributes
        usage_keys = {key for key in attributes if "usage" in key}
        assert usage_keys == expected_usage_keys, (
            f"usage={usage!r} must record exactly {expected_usage_keys}, got {usage_keys}"
        )


# ---------------------------------------------------------------------------
# No-op path — no SDK configured (opentelemetry-sdk is not installed and this
# suite never sets a global tracer provider)


def test_without_a_configured_sdk_the_span_is_the_apis_non_recording_span() -> None:
    with call_span("chat", provider=ROW.provider, model=ROW.model_id) as span:
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
    with call_span("chat", provider=ROW.provider, model=ROW.model_id) as span:
        record_outcome(span, make_meta(usage=Present(usage)))  # must not raise
    assert not span.is_recording()
