"""Observability over `opentelemetry-api` only — a true no-op without an SDK.

One span per facade call: `call_span` opens it with the `gen_ai.*` request
attributes, the runtime records terminal facts from `CallMeta` via
`record_outcome`, and leaving the context ends the span. Attribute names are
drawn from one pinned semconv release (`SEMCONV_VERSION`), recorded both as
the tracer scope's schema URL and as a span attribute.

NEVER on a span: message content, continuation payloads, credentials. Cache
token components are reported raw and never summed into any total —
`TokenUsage.input_tokens` is already cache-inclusive by contract invariant.

Layering: imports from `types` only.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final, assert_never

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Tracer, TracerProvider

from provider_runtime.types import (
    Absent,
    Billability,
    CallMeta,
    ConfirmedNonBillable,
    NotDispatched,
    PossiblyBillable,
    Present,
)

# ---------------------------------------------------------------------------
# Pinned semconv scope

# gen_ai semconv release the attribute names below come from
# (`gen_ai.provider.name` superseded `gen_ai.system` in 1.36).
SEMCONV_VERSION: Final[str] = "1.37.0"

_SCOPE_NAME: Final[str] = "provider_runtime"
_SCHEMA_URL: Final[str] = f"https://opentelemetry.io/schemas/{SEMCONV_VERSION}"

# Resolved once at import: with no SDK configured the api hands back its own
# no-op tracer (via a proxy that upgrades itself if a provider appears later).
_TRACER: Final[Tracer] = trace.get_tracer(_SCOPE_NAME, schema_url=_SCHEMA_URL)


# ---------------------------------------------------------------------------
# Span helpers


@contextmanager
def call_span(
    operation: str,
    *,
    provider: str,
    model: str,
    tracer_provider: TracerProvider | None = None,
) -> Iterator[Span]:
    """Open the one client span for a facade call — semconv name "<operation> <model>".

    `tracer_provider` is the deterministic-test seam; the production default is
    the process-global provider (no-op when none is configured).
    """
    tracer = (
        _TRACER
        if tracer_provider is None
        else tracer_provider.get_tracer(_SCOPE_NAME, schema_url=_SCHEMA_URL)
    )
    with tracer.start_as_current_span(
        f"{operation} {model}",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
            "provider_runtime.semconv_version": SEMCONV_VERSION,
        },
    ) as span:
        yield span


def record_outcome(span: Span, meta: CallMeta) -> None:
    """Record terminal call facts on the span — counts and tags, never content."""
    if not span.is_recording():
        return
    span.set_attribute("provider_runtime.attempt_count", len(meta.attempt_trace))
    span.set_attribute("provider_runtime.billability", _billability_tag(meta.billability))
    span.set_attribute("provider_runtime.registry_revision", meta.registry_revision)
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
