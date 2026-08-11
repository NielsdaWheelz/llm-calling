"""Observability over `opentelemetry-api` only — a true no-op without an SDK.

One span per facade call: `call_span` starts it with the `gen_ai.*` request
attributes and ends it on exit, and the runtime records terminal facts from
`CallMeta` via `record_outcome` — including the span-side mirror of
`CallMeta.attempt_trace`, one bounded event per attempt. Attribute names are
drawn from one pinned semconv release (`SEMCONV_VERSION`), recorded once as the
tracer scope's schema URL.

The call span is deliberately NOT attached as the ambient current span for the
whole call: the streaming port is an async generator that resumes inside its
CONSUMER's context, so an attached span would parent the consumer's own spans
under the call and would detach its token from a foreign context when an
abandoned stream is finalized. `as_current` attaches it around engine
interaction only.

NEVER on a span: message content, continuation payloads, credentials, or
exception message text (engine exceptions quote provider bodies — only the
exception TYPE name reaches the status). Cache token components are reported
raw and never summed into any total — `TokenUsage.input_tokens` is already
cache-inclusive by contract invariant.

Layering: imports from `types` only — the cost estimate is passed in, because
`prices` sits alongside this module, not below it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final, assert_never

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer, TracerProvider
from opentelemetry.util.types import AttributeValue

from provider_runtime.types import (
    Absent,
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
    TransportUnavailable,
)

# ---------------------------------------------------------------------------
# Pinned semconv scope

# gen_ai semconv release the attribute names below come from
# (`gen_ai.provider.name` superseded `gen_ai.system` in 1.36).
SEMCONV_VERSION: Final[str] = "1.37.0"

_SCOPE_NAME: Final[str] = "provider_runtime"
_SCHEMA_URL: Final[str] = f"https://opentelemetry.io/schemas/{SEMCONV_VERSION}"

# Spec §8: the attempt trace is mirrored onto the span as one event per
# attempt — signal, status code and duration, never content.
_ATTEMPT_EVENT: Final[str] = "provider_runtime.attempt"

# Resolved once at import: with no SDK configured the api hands back its own
# no-op tracer (via a proxy that upgrades itself if a provider appears later).
_TRACER: Final[Tracer] = trace.get_tracer(_SCOPE_NAME, schema_url=_SCHEMA_URL)


# ---------------------------------------------------------------------------
# Span helpers


@contextmanager
def call_span(
    operation: str,
    *,
    provider: ProviderName,
    model: str,
    tracer_provider: TracerProvider | None = None,
) -> Iterator[Span]:
    """Open — without attaching — the one client span for a facade call.

    Semconv span name "<operation> <model>". The span is started here and ended
    on exit unless the caller already ended it (the streaming port ends its
    span at the terminal event so the duration is the call's, not the
    consumer's). An `Exception` leaving the block sets an ERROR status
    described by its TYPE NAME ALONE — engine exceptions quote provider bodies
    and the span is content-free, so `record_exception`, which writes
    `str(error)` as an event attribute, is deliberately not called. An
    abandoned generator's `GeneratorExit` is not an error and only ends the
    span. Callers make the span current with `as_current` — see the module
    docstring for why it is never ambient across a stream's yields.

    `tracer_provider` is the deterministic-test seam; the production default is
    the process-global provider (no-op when none is configured).
    """
    tracer = (
        _TRACER
        if tracer_provider is None
        else tracer_provider.get_tracer(_SCOPE_NAME, schema_url=_SCHEMA_URL)
    )
    span = tracer.start_span(
        f"{operation} {model}",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": _provider_name(provider),
            "gen_ai.request.model": model,
        },
    )
    try:
        yield span
    except Exception as error:
        if span.is_recording():
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
        raise
    finally:
        # `is_recording()` is False once a span has ended, so a span the caller
        # already ended is not ended twice (the SDK logs a warning on that).
        if span.is_recording():
            span.end()


@contextmanager
def as_current(span: Span) -> Iterator[None]:
    """Make a call span current for one engine interaction; never ends it.

    Exceptions are deliberately not recorded here: a retried attempt's
    `TransientAttempt` crosses this scope on calls that go on to succeed.
    `call_span` owns the span's error status and its end.
    """
    with trace.use_span(
        span, end_on_exit=False, record_exception=False, set_status_on_exception=False
    ):
        yield


def record_outcome(span: Span, meta: CallMeta, *, cost_estimate: Presence[CostEstimate]) -> None:
    """Record terminal call facts on the span — counts and tags, never content.

    `cost_estimate` is supplied by the caller because deriving it means reading
    the price snapshot, which this module does not import.
    """
    if not span.is_recording():
        return
    span.set_attribute("provider_runtime.attempt_count", len(meta.attempt_trace))
    span.set_attribute("provider_runtime.billability", _billability_tag(meta.billability))
    span.set_attribute("provider_runtime.registry_revision", meta.registry_revision)
    if isinstance(cost_estimate, Present):
        span.set_attribute(
            "provider_runtime.cost_estimate_usd_micros", cost_estimate.value.amount_usd_micros
        )
    match meta.usage:
        case Present(value=usage):
            span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
            if isinstance(usage.cache_read_input_tokens, Present):
                span.set_attribute(
                    "provider_runtime.usage.cache_read_input_tokens",
                    usage.cache_read_input_tokens.value,
                )
            if isinstance(usage.cache_write_input_tokens, Present):
                span.set_attribute(
                    "provider_runtime.usage.cache_write_input_tokens",
                    usage.cache_write_input_tokens.value,
                )
        case Absent():
            pass
        case _:
            assert_never(meta.usage)
    for record in meta.attempt_trace:
        attributes: dict[str, AttributeValue] = {
            "provider_runtime.attempt.number": record.attempt,
            "provider_runtime.attempt.signal": _signal_tag(record.signal),
            "provider_runtime.attempt.duration_ms": record.ended_at_ms - record.started_at_ms,
        }
        if isinstance(record.status_code, Present):
            attributes["provider_runtime.attempt.status_code"] = record.status_code.value
        span.add_event(_ATTEMPT_EVENT, attributes)


def _provider_name(provider: ProviderName) -> str:
    """The semconv 1.37 well-known `gen_ai.provider.name` value for a provider.

    The registry's well-known list is MUST-level where it applies, and two of
    our providers are named differently there. Providers absent from it
    (moonshot, openrouter) are the sanctioned custom-value case and pass
    through as their registry name.
    """
    match provider:
        case "gemini":
            return "gcp.gemini"
        case "xai":
            return "x_ai"
        case "openai" | "anthropic" | "deepseek" | "moonshot" | "openrouter":
            return provider
        case _:
            assert_never(provider)


def _signal_tag(signal: AttemptSignal) -> str:
    match signal:
        case FinalAttempt():
            return "final_attempt"
        case ProviderRateLimit():
            return "provider_rate_limit"
        case ProviderTimeout():
            return "provider_timeout"
        case ProviderHttpUnavailable():
            return "provider_http_unavailable"
        case TransportUnavailable():
            return "transport_unavailable"
        case ProviderStreamInterrupted():
            return "provider_stream_interrupted"
        case _:
            assert_never(signal)


def _billability_tag(billability: Billability) -> str:
    match billability:
        case NotDispatched():
            return "not_dispatched"
        case PossiblyBillable():
            return "possibly_billable"
        case ConfirmedNonBillable():
            return "confirmed_non_billable"
        case _:
            assert_never(billability)
