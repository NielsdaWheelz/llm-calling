"""Value types for the provider runtime.

Layering: this module imports nothing from the package (stdlib only). Every
other module (errors, registry, engines, retry, runtime) may import from it.
The pure data lives here so imports never cycle; the registry owns model facts
and the engines own wire handling.

Style: closed types, exhaustive matches, no hidden mutation, derived fields are
constructor fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, assert_never

# ---------------------------------------------------------------------------
# Provider identity

type ProviderName = Literal[
    "openai", "anthropic", "gemini", "moonshot", "openrouter", "deepseek", "xai"
]


@dataclass(frozen=True, slots=True)
class ProviderTarget:
    provider: ProviderName
    model: str


# Closed superset of per-model reasoning levels; no "default" — provider defaults
# are registry facts, not selectable behavior.
type ReasoningLevel = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


# ---------------------------------------------------------------------------
# Owned absence


@dataclass(frozen=True, slots=True)
class Present[T]:
    value: T


@dataclass(frozen=True, slots=True)
class Absent:
    pass


type Presence[T] = Present[T] | Absent


def presence_of[T](value: T | None) -> Presence[T]:
    """Normalize a nullable boundary value into owned absence (engine ingress)."""
    return Absent() if value is None else Present(value)


# ---------------------------------------------------------------------------
# Prompt blocks


@dataclass(frozen=True, slots=True)
class PromptBlock:
    # Empty text is legal.
    text: str


@dataclass(frozen=True, slots=True)
class ImageBlock:
    media_type: str
    data: bytes = field(repr=False)


# ---------------------------------------------------------------------------
# Tools and output


@dataclass(frozen=True, slots=True)
class ToolCall:
    # arguments come from a strict JSON parse only; NO repair.
    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CanonicalTool:
    name: str
    description: str
    # Plain JSON Schema; engines apply per-provider strictness (e.g. Anthropic
    # additionalProperties: false) at encode.
    parameters: Mapping[str, object]


type ToolChoice = Literal["auto", "none"]


@dataclass(frozen=True, slots=True)
class TextOutput:
    pass


@dataclass(frozen=True, slots=True)
class StrictJsonOutput:
    name: str
    schema: Mapping[str, object]


type OutputSpec = TextOutput | StrictJsonOutput


# ---------------------------------------------------------------------------
# Messages and continuation


@dataclass(frozen=True, slots=True)
class ContinuationArtifact:
    """Opaque native replay material.

    Ephemeral, never logged/rendered; replayable only to the identical
    target+codec (engine-validated).
    """

    target: ProviderTarget
    codec_id: str
    opaque_payload: Mapping[str, object] = field(repr=False)


@dataclass(frozen=True, slots=True)
class SystemMessage:
    blocks: tuple[PromptBlock, ...]


@dataclass(frozen=True, slots=True)
class UserMessage:
    blocks: tuple[PromptBlock | ImageBlock, ...]


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    text: str
    tool_calls: tuple[ToolCall, ...]
    # Absent when the turn has no native replay material.
    continuation: Presence[ContinuationArtifact]


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    call_id: str
    output: str
    is_error: bool


type PromptMessage = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True, slots=True)
class GenerateIntent:
    target: ProviderTarget
    messages: tuple[PromptMessage, ...]
    max_output_tokens: int
    reasoning: ReasoningLevel
    tools: tuple[CanonicalTool, ...]
    tool_choice: ToolChoice
    output: OutputSpec
    # Per-engine extension passthrough, never overrides: any key the engine
    # itself maps from core intent fields raises InvalidRequest.
    provider_options: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Usage — normalized ONCE at engine ingress; raw nullable provider JSON is
# engine-private.


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: Presence[int]
    cache_read_input_tokens: Presence[int]
    cache_write_input_tokens: Presence[int]

    def __post_init__(self) -> None:
        for label, count in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if count < 0:
                raise ValueError(f"TokenUsage.{label} must be >= 0; got {count}")
        for label, maybe in (
            ("reasoning_tokens", self.reasoning_tokens),
            ("cache_read_input_tokens", self.cache_read_input_tokens),
            ("cache_write_input_tokens", self.cache_write_input_tokens),
        ):
            if isinstance(maybe, Present) and maybe.value < 0:
                raise ValueError(f"TokenUsage.{label} must be >= 0; got {maybe.value}")

    @classmethod
    def from_components(
        cls,
        *,
        input_tokens: int,
        output_tokens: int,
        total_tokens: Presence[int],
        reasoning_tokens: Presence[int],
        cache_read_input_tokens: Presence[int],
        cache_write_input_tokens: Presence[int],
    ) -> TokenUsage:
        """The one total-derivation rule shared by every engine.

        Invariant: ``TokenUsage.input_tokens`` is ALWAYS the cache-INCLUSIVE
        total prompt token count — it already contains any cache_read/
        cache_write components the provider reports separately, matching
        OpenAI/Gemini/Moonshot/OpenRouter wire semantics. An engine whose
        native wire input count excludes cache components (Anthropic) MUST
        normalize it to the inclusive total at ingress, before calling this
        constructor, so every engine conforms. `prices.estimate_cost` relies
        on this invariant to recover billable (uncached) input by subtracting
        the cache components back out.

        Where the provider reports a total, it is authoritative. Where it
        reports none, the total is derived at construction as plain
        input_tokens + output_tokens — input_tokens already includes cache, so
        adding the cache components again would double-count them.
        """
        match total_tokens:
            case Present(value=reported):
                total = reported
            case Absent():
                total = input_tokens + output_tokens
            case _:
                assert_never(total_tokens)
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            reasoning_tokens=reasoning_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
        )


# ---------------------------------------------------------------------------
# Attempt trace — on CallMeta so EVERY outcome branch retains it.


@dataclass(frozen=True, slots=True)
class FinalAttempt:
    """The attempt that terminated the call, whatever its outcome."""


type AttemptSignal = TransientCause | FinalAttempt


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    # The trace records ALL attempts incl. the terminal one (last record
    # signal=FinalAttempt); a clean single-attempt call has exactly one record.
    # attempt_count = len(trace); retry_count = len(trace) - 1. retry_after lives
    # inside ProviderRateLimit, never duplicated flat.
    attempt: int
    signal: AttemptSignal
    status_code: Presence[int]
    started_at_ms: int
    ended_at_ms: int

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError(f"AttemptRecord.attempt must be >= 1; got {self.attempt}")
        if self.ended_at_ms < self.started_at_ms:
            raise ValueError(
                "AttemptRecord.ended_at_ms must be >= started_at_ms; "
                f"got started_at_ms={self.started_at_ms}, ended_at_ms={self.ended_at_ms}"
            )


# ---------------------------------------------------------------------------
# Billability — nexus reservation settlement: NotDispatched|ConfirmedNonBillable
# → release; usage Present → commit actuals; PossiblyBillable + usage Absent →
# commit full reservation.


@dataclass(frozen=True, slots=True)
class NotDispatched:
    """No bytes reached the provider."""


@dataclass(frozen=True, slots=True)
class PossiblyBillable:
    """Dispatched; billing state unknown without reported usage."""


@dataclass(frozen=True, slots=True)
class ConfirmedNonBillable:
    """Provider-confirmed unbilled (e.g. Anthropic pre-output refusal)."""


type Billability = NotDispatched | PossiblyBillable | ConfirmedNonBillable


@dataclass(frozen=True, slots=True)
class CallMeta:
    provider: ProviderName
    model: str
    # Header-borne (anthropic request-id, openai x-request-id) or in-band;
    # Gemini: always Absent — the registry records that correlation fact.
    provider_request_id: Presence[str]
    # openrouter engine fills from response provider metadata; direct engines:
    # Absent. Feeds llm_calls.upstream_provider.
    upstream_provider: Presence[str]
    usage: Presence[TokenUsage]
    attempt_trace: tuple[AttemptRecord, ...]
    billability: Billability
    # The exact native reasoning wire value the engine sent (e.g. "high",
    # "thinkingBudget=8192"); Absent when the model has no reasoning knob.
    # Ledger-consumed.
    native_reasoning: Presence[str]
    registry_revision: str


# ---------------------------------------------------------------------------
# Cost — derived on demand by prices.estimate_cost(meta); indicative, never
# authoritative, and never stored on CallMeta.


@dataclass(frozen=True, slots=True)
class CostEstimate:
    amount_usd_micros: int
    source: str  # e.g. "genai-prices@2026-08-01"
    as_of: date

    def __post_init__(self) -> None:
        if self.amount_usd_micros < 0:
            raise ValueError(
                f"CostEstimate.amount_usd_micros must be >= 0; got {self.amount_usd_micros}"
            )


# ---------------------------------------------------------------------------
# Expected failures — closed leaves with FIXED origin/code pairs.


@dataclass(frozen=True, slots=True)
class ProviderRateLimit:
    retry_after: Presence[float]

    def __post_init__(self) -> None:
        # A finite non-negative window or nothing: NaN passes every ordering
        # test in retry.py's cap and deadline arithmetic, and no provider can
        # honestly state an infinite retry window.
        if isinstance(self.retry_after, Present) and not (
            math.isfinite(self.retry_after.value) and self.retry_after.value >= 0
        ):
            raise ValueError(
                "ProviderRateLimit.retry_after must be a finite value >= 0; "
                f"got {self.retry_after.value}"
            )


@dataclass(frozen=True, slots=True)
class ProviderTimeout:
    pass


@dataclass(frozen=True, slots=True)
class ProviderHttpUnavailable:
    pass


@dataclass(frozen=True, slots=True)
class TransportUnavailable:
    pass


@dataclass(frozen=True, slots=True)
class ProviderStreamInterrupted:
    # partial_output=False: all pre-semantic-event attempts exhausted;
    # True: semantic output made internal retry unsafe.
    partial_output: bool


type TransientCause = (
    ProviderRateLimit
    | ProviderTimeout
    | ProviderHttpUnavailable
    | TransportUnavailable
    | ProviderStreamInterrupted
)


@dataclass(frozen=True, slots=True)
class ProviderContextTooLarge:
    """The provider rejected the request as over its context window.

    The only context-overflow signal: nothing in the lane measures a prompt
    locally before dispatch, so overflow is always the provider's verdict.
    """


@dataclass(frozen=True, slots=True)
class InvalidToolArguments:
    safe_detail: str


@dataclass(frozen=True, slots=True)
class InvalidStructuredOutput:
    # The provider returned structured output that failed caller-schema
    # validation (json_out) — native and json_mode rows alike. No repair, no
    # retry.
    safe_detail: str


@dataclass(frozen=True, slots=True)
class TransientExhausted:
    # attempts == len(meta.attempt_trace)
    attempts: int
    cause: TransientCause

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"TransientExhausted.attempts must be >= 1; got {self.attempts}")


type ExpectedModelFailure = (
    ProviderContextTooLarge | InvalidToolArguments | InvalidStructuredOutput | TransientExhausted
)

# Ledger origin union; the runtime's own image is a subset — plan/budget exist
# for nexus ledger writes. Carried by every `errors.RuntimeDefect`.
type FailureOrigin = Literal[
    "intent",
    "plan",
    "budget",
    "transport",
    "provider_http",
    "provider_stream",
    "provider_response",
    "tool_arguments",
]

# Ledger code vocabulary for nexus's failure_code column. The runtime hands
# back closed failure VALUES and never maps them to a code itself: a mapping
# here would be a second, unread source of truth for the ledger's own schema.
type FailureCode = Literal[
    "rate_limited",
    "timeout",
    "provider_unavailable",
    "stream_interrupted",
    "context_too_large",
    "invalid_tool_arguments",
    "invalid_structured_output",
]


# ---------------------------------------------------------------------------
# Response payload — output arm determined by the intent's OutputSpec, never
# re-inferred.


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str
    tool_calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class StructuredContent:
    payload: Mapping[str, object]
    text: str


# tools+strict-output rejected at intent validation ⇒ no impossible state.
type ResponseContent = TextContent | StructuredContent


@dataclass(frozen=True, slots=True)
class ResponsePayload:
    content: ResponseContent
    continuation: Presence[ContinuationArtifact]


# ---------------------------------------------------------------------------
# Terminal outcomes


@dataclass(frozen=True, slots=True)
class Succeeded:
    meta: CallMeta
    response: ResponsePayload


@dataclass(frozen=True, slots=True)
class Refused:
    # NON-STREAM ONLY.
    meta: CallMeta
    safe_detail: str


@dataclass(frozen=True, slots=True)
class Incomplete:
    meta: CallMeta
    reason: Literal["max_output_tokens", "content_filter_partial"]
    # Streamed Fable refusal = incomplete+refused.
    status: Literal["provider_incomplete", "refused"]
    safe_detail: Presence[str]


@dataclass(frozen=True, slots=True)
class Cancelled:
    meta: CallMeta


@dataclass(frozen=True, slots=True)
class Failed:
    meta: CallMeta
    failure: ExpectedModelFailure


type CallOutcome = Succeeded | Refused | Incomplete | Cancelled | Failed  # non-stream generate()
# Stream terminal: Refused excluded BY CONSTRUCTION — streams end in exactly
# one of these four kinds. Anthropic engine:
# streamed HTTP-200 stop_reason=refusal ⇒ Incomplete(status="refused",
# safe_detail), discarding partial output downstream; non-streamed keeps
# distinct Refused.
type StreamOutcome = Succeeded | Incomplete | Cancelled | Failed


@dataclass(frozen=True, slots=True)
class StructuredReply[T]:
    """json_out's success arm: the typed value never travels without its metadata."""

    value: T
    outcome: Succeeded


# ---------------------------------------------------------------------------
# Stream events — engines yield payloads; the runtime owns the envelope
# (no replace(), no post-hoc stamping).


@dataclass(frozen=True, slots=True)
class StreamStart:
    pass


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    call_id: str
    arguments_delta: str


@dataclass(frozen=True, slots=True)
class ToolCallDone:
    tool_call: ToolCall


@dataclass(frozen=True, slots=True)
class ContinuationDelta:
    # AT MOST ONE per stream: emitted only after every contributing native item
    # is final and before the terminal, carrying the COMPLETE artifact; zero is
    # legal; consumers REPLACE, never append.
    artifact: ContinuationArtifact


@dataclass(frozen=True, slots=True)
class UsageEvent:
    # Progressive telemetry ONLY; never the ledger source; still a retry-blocking
    # semantic event.
    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    # outcome.meta carries the AUTHORITATIVE final call facts: the engine folds
    # all provider usage frames into one merged TokenUsage + request id +
    # upstream_provider before emission.
    outcome: StreamOutcome


type CodecStreamEvent = (
    StreamStart
    | TextDelta
    | ToolCallStart
    | ToolCallDelta
    | ToolCallDone
    | ContinuationDelta
    | UsageEvent
    | TerminalEvent
)


@dataclass(frozen=True, slots=True)
class RuntimeStreamEvent:
    # seq: 1-based, monotonic across attempts, stamped ONCE at envelope
    # construction (live consumer: chat_runs provider_event_seq_start/end).
    # Per-event attempt deliberately absent (no consumer; attempt evidence is
    # terminal-only via meta.attempt_trace).
    seq: int
    event: CodecStreamEvent

    def __post_init__(self) -> None:
        if self.seq < 1:
            raise ValueError(f"RuntimeStreamEvent.seq must be >= 1; got {self.seq}")


# ---------------------------------------------------------------------------
# Credentials and cancellation


@dataclass(frozen=True, slots=True)
class ProviderCredential:
    # Transport-only; never part of an intent. The key never appears in repr.
    provider: ProviderName
    key: str = field(repr=False)


class CancelSignal(Protocol):
    """asyncio.Event-shaped cancellation wrapper."""

    async def wait(self) -> bool: ...

    def is_set(self) -> bool: ...


# ---------------------------------------------------------------------------
# Retry policy — RetryPolicy( is constructed only in retry.py (tests excepted).


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Same-target retry budget.

    Which signals are retryable is NOT policy state: engines classify exact
    transients (TransientCause) and the runtime retries only those, only before
    any semantic provider event.
    """

    max_attempts: int
    initial_delay_s: float
    max_delay_s: float
    jitter_s: float
    deadline_s: Presence[float]

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"RetryPolicy.max_attempts must be >= 1; got {self.max_attempts}")
        if self.initial_delay_s < 0:
            raise ValueError(
                f"RetryPolicy.initial_delay_s must be >= 0; got {self.initial_delay_s}"
            )
        if self.max_delay_s < 0:
            raise ValueError(f"RetryPolicy.max_delay_s must be >= 0; got {self.max_delay_s}")
        if self.jitter_s < 0:
            raise ValueError(f"RetryPolicy.jitter_s must be >= 0; got {self.jitter_s}")
        if isinstance(self.deadline_s, Present) and self.deadline_s.value <= 0:
            raise ValueError(f"RetryPolicy.deadline_s must be > 0; got {self.deadline_s.value}")


# ---------------------------------------------------------------------------
# Non-generation port (openai-only embeddings; live Nexus consumer). The call
# names its model as a bare string: the port never resolves a registry row, so
# there is nothing for a ProviderTarget to carry.


@dataclass(frozen=True, slots=True)
class EmbeddingCall:
    model: str
    inputs: tuple[str, ...]
    dimensions: Presence[int]


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    # One vector per input, in input order (index coverage validated at decode).
    embeddings: tuple[tuple[float, ...], ...]
    usage: Presence[TokenUsage]
