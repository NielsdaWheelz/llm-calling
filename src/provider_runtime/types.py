"""Value types for the provider runtime.

Layering: this module imports nothing from the package (stdlib only). Every other
module (errors, schema, catalog, planning, codecs, transport, runtime) may import
from it. Computation modules (schema.py, planning.py) own only computation and
re-export the value types they conceptually govern; the pure data lives here so
imports never cycle.

Style: closed types, exhaustive matches, no hidden mutation, derived fields are
constructor fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, assert_never
from uuid import UUID

# ---------------------------------------------------------------------------
# Provider identity

type ProviderName = Literal["openai", "anthropic", "gemini", "moonshot", "openrouter"]

type ProviderProtocol = Literal[
    "openai_responses",
    "anthropic_messages",
    "gemini_generate_content",
    "moonshot_chat",
    "openrouter_chat",
]


@dataclass(frozen=True, slots=True)
class ProviderTarget:
    provider: ProviderName
    model: str


# Closed superset of per-model reasoning levels; no "default" — provider defaults
# are catalog facts, not selectable behavior.
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
    """Normalize a nullable boundary value into owned absence (codec ingress)."""
    return Absent() if value is None else Present(value)


# ---------------------------------------------------------------------------
# Cache scoping (§8)


@dataclass(frozen=True, slots=True)
class GlobalScope:
    pass


@dataclass(frozen=True, slots=True)
class OwnerScope:
    owner_id: UUID


@dataclass(frozen=True, slots=True)
class ConversationScope:
    conversation_id: UUID


type CacheScope = GlobalScope | OwnerScope | ConversationScope
# RequiredCache and GenerateIntent.cache are deliberately absent (a required field
# encoding zero bits). "Caching has no off state" is realized solely by the planner
# check: stable prefix NON-EMPTY + contiguous + scope-consistent. GenerateIntent.tools
# and output participate in cache affinity and inherit the narrowest contributing
# scope (§8) — they carry no stability marker.


@dataclass(frozen=True, slots=True)
class Dynamic:
    pass


@dataclass(frozen=True, slots=True)
class Stable:
    scope: CacheScope


type BlockStability = Dynamic | Stable


@dataclass(frozen=True, slots=True)
class PromptBlock:
    # Empty text is legal (dynamic blocks may be empty); the planner owns
    # stable-prefix validation.
    text: str
    stability: BlockStability


# ---------------------------------------------------------------------------
# Canonical JSON Schema subset (§5): schema.py owns the immutable schema value
# and its parser/serializer (spec §5). Imported here type-only to annotate
# CanonicalTool/StrictJsonOutput without a runtime import cycle.

if TYPE_CHECKING:
    from .schema import CanonicalJsonSchema


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
    parameters: CanonicalJsonSchema


type ToolChoice = Literal["auto", "none"]


@dataclass(frozen=True, slots=True)
class TextOutput:
    pass


@dataclass(frozen=True, slots=True)
class StrictJsonOutput:
    name: str
    schema: CanonicalJsonSchema


type OutputSpec = TextOutput | StrictJsonOutput


# ---------------------------------------------------------------------------
# Messages and continuation


@dataclass(frozen=True, slots=True)
class ContinuationArtifact:
    """Opaque native replay material.

    Ephemeral, never logged/rendered; replayable only to the identical
    target+codec (planner-validated).
    """

    target: ProviderTarget
    codec_id: str
    opaque_payload: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_payload, Mapping):
            raise TypeError(
                "ContinuationArtifact.opaque_payload must be a Mapping; "
                f"got {type(self.opaque_payload).__name__}"
            )


@dataclass(frozen=True, slots=True)
class SystemMessage:
    blocks: tuple[PromptBlock, ...]


@dataclass(frozen=True, slots=True)
class UserMessage:
    blocks: tuple[PromptBlock, ...]


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


# ---------------------------------------------------------------------------
# Usage — normalized ONCE at codec ingress; raw nullable provider JSON is
# codec-private.


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
        """The one total-derivation rule shared by every codec.

        Invariant: ``TokenUsage.input_tokens`` is ALWAYS the cache-INCLUSIVE
        total prompt token count — it already contains any cache_read/
        cache_write components the provider reports separately, matching
        OpenAI/Gemini/Moonshot/OpenRouter wire semantics. A codec whose
        native wire input count excludes cache components (Anthropic) MUST
        normalize it to the inclusive total at ingress, before calling this
        constructor, so every codec conforms. `usage.cost_from_accounting`
        relies on this invariant to recover billable (uncached) input by
        subtracting the cache components back out.

        Where the provider reports a total, it is authoritative (the
        reservation commits provider-reported totals, §9). Where it reports
        none, the total is derived at construction as plain input_tokens +
        output_tokens — input_tokens already includes cache, so adding the
        cache components again would double-count them.
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
# Attempt trace — on CallMeta so EVERY outcome branch retains it (§11 "retain
# attempt traces").


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
# commit full reservation (§9).


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
    # Gemini: always Absent — the catalog records that correlation fact.
    provider_request_id: Presence[str]
    # openrouter codec fills from response provider/openrouter_metadata; direct
    # codecs: Absent. Feeds llm_calls.upstream_provider (§11.4).
    upstream_provider: Presence[str]
    usage: Presence[TokenUsage]
    attempt_trace: tuple[AttemptRecord, ...]
    billability: Billability


# ---------------------------------------------------------------------------
# Expected failures (§9) — closed leaves with FIXED origin/code pairs.


@dataclass(frozen=True, slots=True)
class ProviderRateLimit:
    retry_after: Presence[float]


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
class IntentContextTooLarge:
    limit: int
    measured: int


@dataclass(frozen=True, slots=True)
class ProviderContextTooLarge:
    pass


@dataclass(frozen=True, slots=True)
class InvalidToolArguments:
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
    IntentContextTooLarge | ProviderContextTooLarge | InvalidToolArguments | TransientExhausted
)

# Exact §9 ledger union; the runtime's own image is a subset — plan/budget exist
# for nexus ledger writes.
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

type FailureCode = Literal[
    "rate_limited",
    "timeout",
    "provider_unavailable",
    "stream_interrupted",
    "context_too_large",
    "invalid_tool_arguments",
]

# Rationale for the fixed pairs (runtime-fixed, propagated to spec §9): intent =
# local pre-network measurement (origin=plan is reserved for planning defects);
# provider_http = context overflow is classified from the provider's HTTP error
# body at codec classify_error ingress; provider_response stays operator-side for
# malformed/unknown terminal envelopes (ProtocolDefect), which never map to
# expected failures.


def failure_origin(failure: ExpectedModelFailure) -> FailureOrigin:
    """TOTAL mapping from expected-failure leaf to its fixed ledger origin."""
    match failure:
        case IntentContextTooLarge():
            return "intent"
        case ProviderContextTooLarge():
            return "provider_http"
        case InvalidToolArguments():
            return "tool_arguments"
        case TransientExhausted(cause=cause):
            match cause:
                case ProviderRateLimit():
                    return "provider_http"
                case ProviderTimeout():
                    return "transport"
                case ProviderHttpUnavailable():
                    return "provider_http"
                case TransportUnavailable():
                    return "transport"
                case ProviderStreamInterrupted():
                    return "provider_stream"
                case _:
                    assert_never(cause)
        case _:
            assert_never(failure)


def failure_code(failure: ExpectedModelFailure) -> FailureCode:
    """TOTAL mapping from expected-failure leaf to its fixed ledger code."""
    match failure:
        case IntentContextTooLarge():
            return "context_too_large"
        case ProviderContextTooLarge():
            return "context_too_large"
        case InvalidToolArguments():
            return "invalid_tool_arguments"
        case TransientExhausted(cause=cause):
            match cause:
                case ProviderRateLimit():
                    return "rate_limited"
                case ProviderTimeout():
                    return "timeout"
                case ProviderHttpUnavailable():
                    return "provider_unavailable"
                case TransportUnavailable():
                    return "provider_unavailable"
                case ProviderStreamInterrupted():
                    return "stream_interrupted"
                case _:
                    assert_never(cause)
        case _:
            assert_never(failure)


# ---------------------------------------------------------------------------
# Response payload — output arm determined by the PLAN, never re-inferred.


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str
    tool_calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class StructuredContent:
    payload: Mapping[str, object]
    text: str


# tools+strict-output rejected at plan ⇒ no impossible state.
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
    # NON-STREAM ONLY (spec §9).
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
# Stream terminal: Refused excluded BY CONSTRUCTION — the four-kind terminal
# grammar of the built streaming cutover is preserved. Anthropic codec: streamed
# HTTP-200 stop_reason=refusal ⇒ Incomplete(status="refused", safe_detail),
# discarding partial output downstream per §1/§9; non-streamed keeps distinct
# Refused.
type StreamOutcome = Succeeded | Incomplete | Cancelled | Failed


# ---------------------------------------------------------------------------
# Stream events — codecs yield payloads; the runtime owns the envelope
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
    # semantic event (§9).
    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    # outcome.meta carries the AUTHORITATIVE final call facts: the codec folds
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
    # Transport-only; never part of a plan. The key never appears in repr.
    provider: ProviderName
    key: str = field(repr=False)


class CancelSignal(Protocol):
    """asyncio.Event-shaped cancellation wrapper."""

    async def wait(self) -> bool: ...

    def is_set(self) -> bool: ...


# ---------------------------------------------------------------------------
# Plan value types — planning.py owns only computation (plan_generate,
# compute_cache_affinity, the policy constants) and re-exports these.


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Same-target retry budget resolved by the planner from the central policy catalog.

    Which signals are retryable is NOT policy state: codecs classify exact
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


@dataclass(frozen=True, slots=True)
class DraftRequest:
    """Phase-one codec encoding: the complete native request WITHOUT the injected
    cache-affinity field.

    prefix_bytes is the codec's exact length-framed serialization of every
    cache-affecting input (stable-prefix blocks PLUS tool definitions,
    tool_choice, and output schema where the provider's cache prefix includes
    them). compute_cache_affinity consumes it — the affinity input EXCLUDES the
    injected key — then codec.finalize(draft, affinity) constructs a NEW frozen
    finalized request (prompt_cache_key for openai, session_id for openrouter,
    passthrough for others).
    """

    target: ProviderTarget
    protocol: ProviderProtocol
    url: str
    safe_headers: Mapping[str, str]
    native_reasoning: str
    provider_framing_overhead_tokens: int
    prefix_bytes: bytes
    body: bytes


@dataclass(frozen=True, slots=True)
class FinalizedProviderRequest:
    target: ProviderTarget
    protocol: ProviderProtocol
    url: str
    method: Literal["POST"]
    safe_headers: Mapping[str, str]
    body: bytes


# Cache plans (§8) — a provider-tagged union, not a generic TTL field.


@dataclass(frozen=True, slots=True)
class OpenAIExplicitPrefix:
    key: str
    minimum_ttl: Literal["30m"]
    breakpoints: int


@dataclass(frozen=True, slots=True)
class AnthropicPrefix:
    stable_breakpoint: int
    ttl: Literal["5m"]
    automatic_append_only: bool


@dataclass(frozen=True, slots=True)
class ProviderAutomaticPrefix:
    provider: Literal["gemini", "moonshot"]


@dataclass(frozen=True, slots=True)
class OpenRouterCertifiedPrefix:
    # evidence_revision is the immutable id of the paid certification artifact.
    session_id: str
    pinned_upstream: str
    evidence_revision: str


type CachePlan = (
    OpenAIExplicitPrefix | AnthropicPrefix | ProviderAutomaticPrefix | OpenRouterCertifiedPrefix
)


def cache_strategy(plan: CachePlan) -> str:
    """The §11.4 ledger strategy string — this accessor is its only owner."""
    match plan:
        case OpenAIExplicitPrefix():
            return "openai_explicit_prefix"
        case AnthropicPrefix():
            return "anthropic_prefix"
        case ProviderAutomaticPrefix():
            return "provider_automatic_prefix"
        case OpenRouterCertifiedPrefix():
            return "openrouter_certified_prefix"
        case _:
            assert_never(plan)


def cache_ttl(plan: CachePlan) -> str | None:
    """The §11.4 ledger TTL string (nullable ledger-column boundary)."""
    match plan:
        case OpenAIExplicitPrefix(minimum_ttl=ttl):
            return ttl
        case AnthropicPrefix(ttl=ttl):
            return ttl
        case ProviderAutomaticPrefix():
            return None
        case OpenRouterCertifiedPrefix():
            return None
        case _:
            assert_never(plan)


@dataclass(frozen=True, slots=True)
class Accounting:
    # Frozen at plan time so terminal costing never re-reads the catalog. Rates
    # are usd micros per million tokens.
    currency: Literal["usd"]
    input_rate: int
    output_rate: int
    cache_write_rate: int
    cache_read_rate: int
    reasoning_billed_outside_output: bool
    platform_token_reservation: int
    maximum_cost_estimate_usd_micros: int


@dataclass(frozen=True, slots=True)
class FinalizedProviderCall:
    # Fingerprints: sha256 base64url over (finalized body bytes) / (framed
    # canonical tool encodings) / (framed canonical output-schema encoding);
    # request_fingerprint computed AFTER finalize.
    request: FinalizedProviderRequest
    accounting: Accounting
    requested_reasoning: ReasoningLevel
    effective_reasoning: ReasoningLevel
    native_reasoning: str
    cache_plan: CachePlan
    retry_policy: RetryPolicy
    catalog_revision: str
    request_fingerprint: str
    tool_fingerprint: str
    schema_fingerprint: str
    planned_input_token_upper_bound: int
    # The plan's output arm, carried so the runtime can promote the decoded
    # terminal text to StructuredContent (codecs decode TextContent only; the
    # plan-owning layer owns the promotion). "strict_json" iff the intent's
    # output is StrictJsonOutput.
    output_kind: Literal["text", "strict_json"]


@dataclass(frozen=True, slots=True)
class PlanRejected:
    """The expected planner rejection channel (oversize intent) — not a defect."""

    failure: IntentContextTooLarge


# ---------------------------------------------------------------------------
# Non-generation ports (openai-only embedding/transcription; §5 keeps separate
# typed contracts). Field shapes preserve the pre-cutover types minus
# ModelRef (a bare model string suffices — the catalog rows are openai-only),
# per-call RetryPolicy (the runtime applies the central EXTERNAL_LLM_RETRY
# policy from planning), provider_request_id, and the attempts trace (both
# were consumer-less on these ports; expected failures surface via
# runtime.NonGenerationCallFailed instead).


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


@dataclass(frozen=True, slots=True)
class TranscriptionCall:
    model: str
    filename: str
    content: bytes = field(repr=False)
    media_type: str = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class TranscriptionResponse:
    text: str
    usage: Presence[TokenUsage]
