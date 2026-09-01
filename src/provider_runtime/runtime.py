"""Provider runtime facade: registry-resolved dispatch over the engines.

`ProviderRuntime` owns everything the engines do not: registry resolution and
intent gates, credential lookup, the retry loop (`retry.attempts`) with
attempt-trace accumulation and billability folding, stream envelope
sequencing, cancellation, one OTel span per facade call, and the `json_out` /
`chat` sugar. Engines make exactly ONE attempt and construct single-attempt
`CallMeta`; the runtime rewrites every terminal meta with the accumulated
trace via `dataclasses.replace`.

Retry rules (spec §8): retry only exact `TransientCause` signals
(`TransientAttempt`), only before any semantic provider event reached the
consumer. Exhaustion folds into `Failed(TransientExhausted)` preserving the
final cause and the full attempt trace on meta. Defects raise. No fallback of
any kind.

Attempt-trace timestamps are monotonic milliseconds (`time.monotonic`-derived)
suitable for durations only — they are not wall-clock times.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, assert_never
from urllib.parse import urlsplit, urlunsplit

import httpx
import pydantic
from opentelemetry.trace import TracerProvider

from provider_runtime import embeddings
from provider_runtime.engines import Engine, TransientAttempt
from provider_runtime.engines._common import monotonic_ms
from provider_runtime.engines.anthropic_messages import AnthropicMessagesEngine
from provider_runtime.engines.gemini_generate import GeminiGenerateEngine
from provider_runtime.engines.openai_chat import OpenAIChatEngine
from provider_runtime.engines.openai_responses import OpenAIResponsesEngine
from provider_runtime.errors import (
    CredentialMissing,
    InvalidRequest,
    NonGenerationCallFailed,
    ProtocolDefect,
)
from provider_runtime.otel import as_current, call_span, record_outcome
from provider_runtime.prices import estimate_cost
from provider_runtime.registry import (
    REGISTRY_REVISION,
)
from provider_runtime.registry import (
    _ModelRow as ModelRow,
)
from provider_runtime.registry import (
    _resolve as resolve,
)
from provider_runtime.registry import (
    _resolve_target as resolve_target,
)
from provider_runtime.retry import DEFAULT_RETRY, attempts
from provider_runtime.types import (
    Absent,
    AttemptRecord,
    AttemptSignal,
    Billability,
    CallMeta,
    CallOutcome,
    Cancelled,
    CancelSignal,
    CodecStreamEvent,
    ConfirmedNonBillable,
    EmbeddingCall,
    EmbeddingResponse,
    EngineId,
    Failed,
    FinalAttempt,
    GenerateIntent,
    ImageBlock,
    Incomplete,
    InvalidStructuredOutput,
    NotDispatched,
    PossiblyBillable,
    Presence,
    Present,
    PromptBlock,
    PromptMessage,
    ProviderContextTooLarge,
    ProviderCredential,
    ProviderName,
    ProviderStreamInterrupted,
    ProviderTarget,
    ReasoningLevel,
    Refused,
    RetryPolicy,
    RuntimeStreamEvent,
    StreamOutcome,
    StreamStart,
    StrictJsonOutput,
    StructuredContent,
    StructuredReply,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextOutput,
    TransientCause,
    TransientExhausted,
    UserMessage,
)


@dataclass(frozen=True, slots=True)
class Credentials:
    """Provider API keys by provider name; the lane reads zero env vars.

    Every key is repr-suppressed; `None` means the provider is not configured
    and dispatching to it raises `CredentialMissing`.
    """

    openai: str | None = field(default=None, repr=False)
    anthropic: str | None = field(default=None, repr=False)
    gemini: str | None = field(default=None, repr=False)
    moonshot: str | None = field(default=None, repr=False)
    openrouter: str | None = field(default=None, repr=False)
    deepseek: str | None = field(default=None, repr=False)
    xai: str | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Attempt bookkeeping (trace, billability fold, runtime-owned meta)


def _record(
    attempt: int, signal: AttemptSignal, status_code: Presence[int], started_ms: int
) -> AttemptRecord:
    return AttemptRecord(
        attempt=attempt,
        signal=signal,
        status_code=status_code,
        started_at_ms=started_ms,
        ended_at_ms=monotonic_ms(),
    )


def _billability_rank(billability: Billability) -> int:
    match billability:
        case NotDispatched():
            return 0
        case ConfirmedNonBillable():
            return 1
        case PossiblyBillable():
            return 2
        case _:
            assert_never(billability)


def _fold_billability(folded: Billability, observed: Billability) -> Billability:
    """NotDispatched < ConfirmedNonBillable < PossiblyBillable — keep the max."""
    return observed if _billability_rank(observed) > _billability_rank(folded) else folded


def _absorbed[Outcome: Succeeded | Refused | Incomplete | Cancelled | Failed](
    outcome: Outcome, prior: Sequence[AttemptRecord], billability: Billability
) -> Outcome:
    """Rebuild an engine outcome with the runtime-owned accumulated trace.

    Engines construct single-attempt meta (attempt=1); their records are
    renumbered onto the accumulated trace and their billability folds in.
    """
    renumbered = tuple(
        replace(record, attempt=len(prior) + offset)
        for offset, record in enumerate(outcome.meta.attempt_trace, start=1)
    )
    meta = replace(
        outcome.meta,
        attempt_trace=(*prior, *renumbered),
        billability=_fold_billability(billability, outcome.meta.billability),
    )
    return replace(outcome, meta=meta)


def _runtime_meta(
    row: ModelRow,
    trace: tuple[AttemptRecord, ...],
    billability: Billability,
    request_id: Presence[str],
) -> CallMeta:
    """Meta for runtime-constructed terminals (exhaustion, cancellation).

    Usage is deliberately Absent — a runtime terminal means no engine decoded
    an authoritative provider envelope. The request id is the last transient
    attempt's, when the provider issued one.
    """
    return CallMeta(
        provider=row.provider,
        model=row.model_id,
        provider_request_id=request_id,
        upstream_provider=Absent(),
        usage=Absent(),
        attempt_trace=trace,
        billability=billability,
        native_reasoning=Absent(),
        registry_revision=REGISTRY_REVISION,
    )


# ---------------------------------------------------------------------------
# Intent gates — runtime-owned validation against the resolved row


def _validate_intent(row: ModelRow, intent: GenerateIntent, *, streaming: bool) -> None:
    if streaming and not row.streaming:
        raise InvalidRequest(message=f"registry row {row.ref!r} does not support streaming")
    if intent.tools:
        if not row.tools:
            raise InvalidRequest(message=f"registry row {row.ref!r} does not support tools")
        if isinstance(intent.output, StrictJsonOutput):
            # types.py: tools+strict-output rejected here ⇒ no impossible
            # ResponseContent state downstream.
            raise InvalidRequest(
                message="tools and StrictJsonOutput cannot be combined in one intent"
            )
    if "image" not in row.modalities:
        for message in intent.messages:
            if isinstance(message, UserMessage) and any(
                isinstance(block, ImageBlock) for block in message.blocks
            ):
                raise InvalidRequest(
                    message=f"registry row {row.ref!r} does not accept image input"
                )


# ---------------------------------------------------------------------------
# Cancellation race


async def _guarded_next(source: AsyncIterator[CodecStreamEvent]) -> CodecStreamEvent:
    try:
        return await anext(source)
    except StopAsyncIteration:
        # Engine contract breach: a stream must end with a TerminalEvent or
        # raise TransientAttempt; bare exhaustion is a defect.
        raise ProtocolDefect(
            code="missing_terminal_event",
            message="engine stream ended without a terminal event or transient signal",
        ) from None


async def _race_cancel[T](action: Awaitable[T], cancel: CancelSignal | None) -> T | None:
    """Race one engine await against cancellation; None means cancelled.

    Cancellation-safe: if the CONSUMING task is itself cancelled while parked
    in the race, both tasks are cancelled and settled (via `asyncio.wait`,
    which never raises a task's own exception) before the external
    CancelledError is re-raised. This guarantees the engine generator is no
    longer running by the time the caller's `finally: await source.aclose()`
    runs — otherwise aclose() raises "already running", masking the
    CancelledError.
    """
    if cancel is None:
        return await action
    next_task = asyncio.ensure_future(action)
    cancel_task = asyncio.ensure_future(cancel.wait())
    tasks: set[asyncio.Task[Any]] = {next_task, cancel_task}
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.wait(tasks)  # settle; wait() never raises task exceptions
        for task in tasks:
            if not task.cancelled():
                task.exception()  # retrieve, so no 'Task exception was never retrieved'
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
# json_out helpers


def _invalid_structured_detail(
    model: type[pydantic.BaseModel], error: pydantic.ValidationError
) -> str:
    """Class name + terse location/message summary — NEVER the payload."""
    causes = "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors(include_url=False, include_input=False)[:3]
    )
    return f"{model.__name__} validation failed ({error.error_count()} errors): {causes}"


# ---------------------------------------------------------------------------
# Runtime


_PROVIDERS = frozenset(
    {"openai", "anthropic", "gemini", "moonshot", "openrouter", "deepseek", "xai"}
)


def _endpoint_overrides(
    values: Mapping[ProviderName, str],
) -> Mapping[ProviderName, str]:
    """Freeze canonical HTTPS origins supplied by the embedding application."""

    unknown = set(values) - _PROVIDERS
    if unknown:
        raise ValueError(f"endpoint overrides name unknown providers: {sorted(unknown)!r}")
    checked: dict[ProviderName, str] = {}
    for provider, value in values.items():
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as error:
            raise ValueError(f"{provider} endpoint override has an invalid port") from error
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"{provider} endpoint override must be one HTTPS origin")
        default_port = port in {None, 443}
        authority = parsed.hostname if default_port else f"{parsed.hostname}:{port}"
        if parsed.netloc != authority:
            raise ValueError(f"{provider} endpoint override is not canonical")
        checked[provider] = f"https://{authority}"
    return MappingProxyType(checked)


def _dispatch_row(
    row: ModelRow,
    overrides: Mapping[ProviderName, str],
) -> ModelRow:
    """Apply an explicit origin after registry resolution, preserving API path."""

    origin = overrides.get(row.provider)
    if origin is None:
        return row
    if isinstance(row.base_url, Present):
        path = urlsplit(row.base_url.value).path.rstrip("/")
    elif row.engine == "openai_responses":
        path = "/v1"
    else:
        path = ""
    parsed = urlsplit(origin)
    base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return replace(row, base_url=Present(base_url))


class ProviderRuntime:
    """Engines + credentials wired once; every call dispatches through a row."""

    def __init__(
        self,
        credentials: Credentials,
        *,
        retry: RetryPolicy = DEFAULT_RETRY,
        http_client: httpx.AsyncClient | None = None,
        engines: Mapping[EngineId, Engine] | None = None,
        endpoint_overrides: Mapping[ProviderName, str] | None = None,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        # `engines` is the deterministic-test seam (spec §11: facade tests run
        # against FakeEngine); the production default wires the four adapters.
        # `tracer_provider` is the standard OTel library seam: None means the
        # process-global provider (no-op until an SDK configures one).
        self._credentials = credentials
        self._retry = retry
        self._http_client = http_client
        self._tracer_provider = tracer_provider
        self._endpoint_overrides = _endpoint_overrides(endpoint_overrides or {})
        self._engines: Mapping[EngineId, Engine] = (
            engines
            if engines is not None
            else {
                "openai_responses": OpenAIResponsesEngine(http_client=http_client),
                "openai_chat": OpenAIChatEngine(http_client=http_client),
                "anthropic_messages": AnthropicMessagesEngine(http_client=http_client),
                "gemini_generate": GeminiGenerateEngine(http_client=http_client),
            }
        )

    def _credential(self, provider: ProviderName) -> ProviderCredential:
        match provider:
            case "openai":
                key = self._credentials.openai
            case "anthropic":
                key = self._credentials.anthropic
            case "gemini":
                key = self._credentials.gemini
            case "moonshot":
                key = self._credentials.moonshot
            case "openrouter":
                key = self._credentials.openrouter
            case "deepseek":
                key = self._credentials.deepseek
            case "xai":
                key = self._credentials.xai
            case _:
                assert_never(provider)
        if key is None:
            raise CredentialMissing(message=f"no {provider} credential configured")
        return ProviderCredential(provider=provider, key=key)

    # -- generate -----------------------------------------------------------

    async def generate(
        self, intent: GenerateIntent, *, cancel: CancelSignal | None = None
    ) -> CallOutcome:
        source_row = resolve_target(intent.target)
        _validate_intent(source_row, intent, streaming=False)
        row = _dispatch_row(source_row, self._endpoint_overrides)
        credential = self._credential(source_row.provider)
        engine = self._engines[source_row.engine]
        with call_span(
            "chat",
            provider=row.provider,
            model=row.model_id,
            tracer_provider=self._tracer_provider,
        ) as span:
            with as_current(span):
                outcome = await self._generate_outcome(row, intent, credential, engine, cancel)
            record_outcome(span, outcome.meta, cost_estimate=estimate_cost(outcome.meta))
            return outcome

    async def _generate_outcome(
        self,
        row: ModelRow,
        intent: GenerateIntent,
        credential: ProviderCredential,
        engine: Engine,
        cancel: CancelSignal | None,
    ) -> CallOutcome:
        trace: list[AttemptRecord] = []
        billability: Billability = NotDispatched()
        last_cause: TransientCause | None = None
        last_request_id: Presence[str] = Absent()
        tries = attempts(self._retry)
        try:
            async for handle in tries:
                started_ms = monotonic_ms()
                if cancel is not None and cancel.is_set():
                    trace.append(_record(handle.number, FinalAttempt(), Absent(), started_ms))
                    return Cancelled(
                        meta=_runtime_meta(row, tuple(trace), billability, last_request_id)
                    )
                try:
                    outcome = await _race_cancel(engine.generate(row, intent, credential), cancel)
                except TransientAttempt as failed:
                    billability = _fold_billability(billability, failed.billability)
                    last_request_id = failed.provider_request_id
                    last_cause = failed.cause
                    trace.append(
                        _record(handle.number, failed.cause, failed.status_code, started_ms)
                    )
                    handle.mark_failed(failed.cause)
                    continue
                if outcome is None:
                    # Cancelled mid-attempt: the request was already in flight,
                    # so the conservative fold is PossiblyBillable.
                    billability = _fold_billability(billability, PossiblyBillable())
                    trace.append(_record(handle.number, FinalAttempt(), Absent(), started_ms))
                    return Cancelled(
                        meta=_runtime_meta(row, tuple(trace), billability, last_request_id)
                    )
                return _absorbed(outcome, trace, billability)
        finally:
            # Deterministic close: abandoning the suspended attempt iterator
            # would leave its finalization to the event loop's GC hook.
            await tries.aclose()
        if last_cause is None:
            raise AssertionError("attempt loop exhausted without a recorded transient cause")
        # The exhausting attempt's record carries FinalAttempt; its transient
        # cause lives in TransientExhausted.cause.
        trace[-1] = replace(trace[-1], signal=FinalAttempt())
        return Failed(
            meta=_runtime_meta(row, tuple(trace), billability, last_request_id),
            failure=TransientExhausted(attempts=len(trace), cause=last_cause),
        )

    # -- stream -------------------------------------------------------------

    def stream(
        self, intent: GenerateIntent, *, cancel: CancelSignal | None = None
    ) -> AsyncIterator[RuntimeStreamEvent]:
        source_row = resolve_target(intent.target)
        _validate_intent(source_row, intent, streaming=True)
        row = _dispatch_row(source_row, self._endpoint_overrides)
        credential = self._credential(source_row.provider)
        engine = self._engines[source_row.engine]
        return self._stream_events(row, intent, credential, engine, cancel)

    async def _stream_events(
        self,
        row: ModelRow,
        intent: GenerateIntent,
        credential: ProviderCredential,
        engine: Engine,
        cancel: CancelSignal | None,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        trace: list[AttemptRecord] = []
        billability: Billability = NotDispatched()
        last_cause: TransientCause | None = None
        last_request_id: Presence[str] = Absent()
        seq = 0
        start_forwarded = False
        semantic_forwarded = False

        with call_span(
            "chat",
            provider=row.provider,
            model=row.model_id,
            tracer_provider=self._tracer_provider,
        ) as span:

            def envelope(event: CodecStreamEvent) -> RuntimeStreamEvent:
                nonlocal seq
                seq += 1
                return RuntimeStreamEvent(seq=seq, event=event)

            def terminal(outcome: StreamOutcome) -> RuntimeStreamEvent:
                record_outcome(span, outcome.meta, cost_estimate=estimate_cost(outcome.meta))
                # End here, not at generator close: the idiomatic consumer
                # breaks after the terminal event, leaving this generator to
                # the async-generator finalizer — which would stretch the span
                # to GC time, and never runs at all in a process that exits
                # without shutdown_asyncgens. `call_span`'s exit still ends an
                # abandoned or failed stream's span.
                span.end()
                return envelope(TerminalEvent(outcome=outcome))

            def cancelled_meta(attempt: int, started_ms: int) -> CallMeta:
                trace.append(_record(attempt, FinalAttempt(), Absent(), started_ms))
                return _runtime_meta(row, tuple(trace), billability, last_request_id)

            tries = attempts(self._retry)
            try:
                async for handle in tries:
                    started_ms = monotonic_ms()
                    if cancel is not None and cancel.is_set():
                        yield terminal(Cancelled(meta=cancelled_meta(handle.number, started_ms)))
                        return
                    source = engine.stream(row, intent, credential)
                    try:
                        while True:
                            with as_current(span):
                                event = await _race_cancel(_guarded_next(source), cancel)
                            if event is None:
                                # Cancelled mid-attempt: the request was
                                # already in flight — fold PossiblyBillable.
                                billability = _fold_billability(billability, PossiblyBillable())
                                yield terminal(
                                    Cancelled(meta=cancelled_meta(handle.number, started_ms))
                                )
                                return
                            match event:
                                case StreamStart():
                                    # At most ONE StreamStart crosses the
                                    # envelope, even across retried attempts.
                                    if not start_forwarded:
                                        start_forwarded = True
                                        yield envelope(event)
                                case TerminalEvent(outcome=outcome):
                                    yield terminal(_absorbed(outcome, trace, billability))
                                    return
                                case _:
                                    semantic_forwarded = True
                                    yield envelope(event)
                    except TransientAttempt as failed:
                        billability = _fold_billability(billability, failed.billability)
                        last_request_id = failed.provider_request_id
                        if semantic_forwarded:
                            # Post-semantic-output, ANY transient is terminal
                            # with the stream-interruption leaf rebuilt to
                            # carry the true flag.
                            trace.append(
                                _record(
                                    handle.number, FinalAttempt(), failed.status_code, started_ms
                                )
                            )
                            yield terminal(
                                Failed(
                                    meta=_runtime_meta(
                                        row, tuple(trace), billability, last_request_id
                                    ),
                                    failure=TransientExhausted(
                                        attempts=len(trace),
                                        cause=ProviderStreamInterrupted(partial_output=True),
                                    ),
                                )
                            )
                            return
                        last_cause = failed.cause
                        trace.append(
                            _record(handle.number, failed.cause, failed.status_code, started_ms)
                        )
                        handle.mark_failed(failed.cause)
                    finally:
                        if isinstance(source, AsyncGenerator):
                            await source.aclose()
            finally:
                # Deterministic close: abandoning the suspended attempt
                # iterator would leave its finalization to the GC hook.
                await tries.aclose()
            if last_cause is None:
                raise AssertionError("stream attempt loop exhausted without a transient cause")
            trace[-1] = replace(trace[-1], signal=FinalAttempt())
            yield terminal(
                Failed(
                    meta=_runtime_meta(row, tuple(trace), billability, last_request_id),
                    failure=TransientExhausted(attempts=len(trace), cause=last_cause),
                )
            )

    # -- json_out -----------------------------------------------------------

    @staticmethod
    def _closed_schema(schema: Mapping[str, object]) -> dict[str, object]:
        """Recursively close object nodes: native strict modes (OpenAI Responses,
        Anthropic output_config.format) reject any object without an explicit
        `additionalProperties: false`, and pydantic's generator omits it."""

        def close(node: object) -> object:
            if isinstance(node, Mapping):
                return close_mapping(node)
            if isinstance(node, list):
                return [close(item) for item in node]
            return node

        def close_mapping(node: Mapping[str, object]) -> dict[str, object]:
            closed = {key: close(value) for key, value in node.items()}
            if "properties" in closed or closed.get("type") == "object":
                closed.setdefault("additionalProperties", False)
            return closed

        return close_mapping(schema)

    async def json_out[T: pydantic.BaseModel](
        self,
        model: type[T],
        intent: GenerateIntent,
        *,
        cancel: CancelSignal | None = None,
    ) -> StructuredReply[T] | Refused | Incomplete | Cancelled | Failed:
        """Typed structured output: strict schema derived from the model.

        Validation miss → `Failed(InvalidStructuredOutput)` carrying the same
        terminal meta. No repair, no retry.
        """
        if not isinstance(intent.output, TextOutput):
            raise InvalidRequest(
                message="json_out requires an intent with TextOutput; "
                "the strict schema is derived from the pydantic model"
            )
        schema = self._closed_schema(model.model_json_schema())
        assert isinstance(schema, Mapping)
        strict_intent = replace(
            intent,
            output=StrictJsonOutput(name=model.__name__, schema=schema),
        )
        outcome = await self.generate(strict_intent, cancel=cancel)
        match outcome:
            case Succeeded(response=response):
                content = response.content
                if not isinstance(content, StructuredContent):
                    # Engine contract: a strict-JSON intent decodes to
                    # StructuredContent on every Succeeded path.
                    raise ProtocolDefect(
                        code="invalid_structured_output",
                        message="strict-JSON success decoded without structured content",
                    )
                try:
                    value = model.model_validate(content.payload)
                except pydantic.ValidationError as error:
                    return Failed(
                        meta=outcome.meta,
                        failure=InvalidStructuredOutput(
                            safe_detail=_invalid_structured_detail(model, error)
                        ),
                    )
                return StructuredReply(value=value, outcome=outcome)
            case Refused() | Incomplete() | Cancelled() | Failed():
                return outcome
            case _:
                assert_never(outcome)

    # -- chat sugar ---------------------------------------------------------

    async def chat(
        self,
        ref: str,
        *,
        system: str = "",
        user: str,
        reasoning: ReasoningLevel = "none",
        max_output_tokens: int | None = None,
    ) -> CallOutcome:
        """The 95% call site: one system/user turn against a registry ref.

        `reasoning` defaults to "none", which is callable on every row: a row
        declaring a "none" fragment sends it; a row that declares no "none"
        level sends no reasoning field at all and the provider's own default
        applies. "none" never raises. Any OTHER level a row does not declare
        does raise `InvalidRequest` — silently discarding an explicit
        non-default request is banned.
        """
        row = resolve(ref)
        user_message = UserMessage(blocks=(PromptBlock(text=user),))
        messages: tuple[PromptMessage, ...] = (
            (SystemMessage(blocks=(PromptBlock(text=system),)), user_message)
            if system
            else (user_message,)
        )
        intent = GenerateIntent(
            target=ProviderTarget(provider=row.provider, model=row.model_id),
            messages=messages,
            max_output_tokens=(
                row.max_output_tokens if max_output_tokens is None else max_output_tokens
            ),
            reasoning=reasoning,
            tools=(),
            tool_choice="auto",
            output=TextOutput(),
        )
        return await self.generate(intent)

    # -- embed (openai-only non-generation port) -----------------------------

    async def embed(
        self, call: EmbeddingCall, *, credential: ProviderCredential
    ) -> EmbeddingResponse:
        """OpenAI-only embeddings port (live Nexus consumer), same retry owner.

        Delegates single attempts to the embeddings module's seam::

            async def embed_once(
                call: EmbeddingCall,
                credential: ProviderCredential,
                *,
                http_client: httpx.AsyncClient | None,
            ) -> EmbeddingResponse | ProviderContextTooLarge

        — one openai-SDK attempt; context overflow is the one expected
        terminal failure (returned as a value), retryable trouble raises
        `TransientAttempt`, defects raise their own types. Retry exhaustion
        and context overflow surface here as `NonGenerationCallFailed`.
        """
        last_cause: TransientCause | None = None
        attempt_count = 0
        tries = attempts(self._retry)
        with call_span(
            "embeddings",
            provider="openai",
            model=call.model,
            tracer_provider=self._tracer_provider,
        ) as span:
            # Both failure arms raise INSIDE the span, so `call_span` marks it
            # ERROR either way; the finally puts the attempt count on all arms.
            try:
                async for handle in tries:
                    attempt_count = handle.number
                    try:
                        with as_current(span):
                            result = await embeddings.embed_once(
                                call, credential, http_client=self._http_client
                            )
                    except TransientAttempt as failed:
                        last_cause = failed.cause
                        handle.mark_failed(failed.cause)
                        continue
                    if isinstance(result, ProviderContextTooLarge):
                        raise NonGenerationCallFailed(result)
                    if span.is_recording() and isinstance(result.usage, Present):
                        span.set_attribute(
                            "gen_ai.usage.input_tokens", result.usage.value.input_tokens
                        )
                    return result
                if last_cause is None:
                    raise AssertionError("embed attempt loop exhausted without a transient cause")
                raise NonGenerationCallFailed(
                    TransientExhausted(attempts=attempt_count, cause=last_cause)
                )
            finally:
                await tries.aclose()
                if span.is_recording():
                    span.set_attribute("provider_runtime.attempt_count", attempt_count)


__all__ = [
    "Credentials",
    "ProviderRuntime",
]
