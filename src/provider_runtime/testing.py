"""IR-level runtime doubles for deterministic tests.

Two seams (spec §11): `FakeEngine` implements the `engines.Engine` protocol
for runtime/facade tests; `ScriptedRuntime` is a drop-in for `ProviderRuntime`
at the facade for library consumers. Both are boring on purpose: they return
exactly what was scripted — outcomes carry their own meta, nothing is stamped
or simulated — and record every call for assertion. The one runtime behavior
`ScriptedRuntime` mirrors is the stream envelope (1-based seq stamping and the
single-terminal grammar), so consumers assert on `RuntimeStreamEvent` exactly
as against the real runtime.

The dead wire layer has no double here by design: `NoNetworkRuntime` and SSE
scripting died with it. Layering: imports from `types`, `registry` and `errors`
(plus `pydantic` for the json_out bound) — the one name taken from `errors` is
`NonGenerationCallFailed`, embed's failure channel — never `runtime` itself or
the engine modules.

`ScriptedRuntime`'s captured calls never store the credential key — only the
provider name; `FakeEngine`'s capture the full `ProviderCredential` it was
called with, key included, matching the real `Engine` protocol call shape.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pydantic

from provider_runtime.errors import NonGenerationCallFailed
from provider_runtime.registry import _ModelRow as ModelRow
from provider_runtime.types import (
    Absent,
    CallOutcome,
    Cancelled,
    CancelSignal,
    CodecStreamEvent,
    EmbeddingCall,
    EmbeddingResponse,
    Failed,
    GenerateIntent,
    Incomplete,
    Presence,
    Present,
    ProviderCredential,
    ProviderName,
    ReasoningLevel,
    Refused,
    RuntimeStreamEvent,
    StructuredReply,
    TerminalEvent,
)

# ---------------------------------------------------------------------------
# FakeEngine — Engine-protocol double

# One step per engine call, whatever the next call produces: a CallOutcome
# returned by generate, a raw codec-event sequence yielded by stream (no
# envelope — the runtime owns it), or an Exception raised by either
# (TransientAttempt for retryable trouble, defects otherwise).
type EngineScriptStep = CallOutcome | Sequence[CodecStreamEvent] | Exception


class FakeEngine:
    """Scripted `Engine`: one step consumed per call, in script order.

    `calls` records every (row, intent, credential) received. Stream steps run
    entirely at iteration time — recording, consumption, and any scripted
    raise happen inside the generator — so a scripted exception surfaces
    inside the runtime's per-attempt try block, exactly like a real engine.
    """

    def __init__(self, script: Iterable[EngineScriptStep]) -> None:
        self.calls: list[tuple[ModelRow, GenerateIntent, ProviderCredential]] = []
        self._script: deque[EngineScriptStep] = deque(script)

    async def generate(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> CallOutcome:
        step = self._next("generate", row, intent, credential)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, Sequence):
            raise AssertionError(
                f"FakeEngine script step {len(self.calls)} is a stream script "
                f"but generate was called for row {row.ref!r}"
            )
        return step

    async def stream(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> AsyncIterator[CodecStreamEvent]:
        step = self._next("stream", row, intent, credential)
        if isinstance(step, Exception):
            raise step
        if not isinstance(step, Sequence):
            raise AssertionError(
                f"FakeEngine script step {len(self.calls)} is a generate outcome "
                f"but stream was called for row {row.ref!r}"
            )
        for event in step:
            yield event

    def _next(
        self,
        operation: Literal["generate", "stream"],
        row: ModelRow,
        intent: GenerateIntent,
        credential: ProviderCredential,
    ) -> EngineScriptStep:
        self.calls.append((row, intent, credential))
        if not self._script:
            raise AssertionError(
                f"FakeEngine script exhausted: unexpected {operation} call for "
                f"row {row.ref!r} after {len(self.calls) - 1} scripted steps"
            )
        return self._script.popleft()


# ---------------------------------------------------------------------------
# ScriptedRuntime — facade-level double

type RuntimeOperation = Literal["generate", "stream", "json_out", "chat", "embed"]

type JsonOutResult = StructuredReply[Any] | Refused | Incomplete | Cancelled | Failed


@dataclass(frozen=True, slots=True)
class ChatCall:
    """`chat()` arguments captured verbatim — the double builds no intent."""

    ref: str
    system: str
    user: str
    reasoning: ReasoningLevel
    # Facade-signature echo: None means "row default", recorded as passed.
    max_output_tokens: int | None


@dataclass(frozen=True, slots=True)
class JsonOutCall:
    model: type[pydantic.BaseModel]
    intent: GenerateIntent


@dataclass(frozen=True, slots=True)
class CapturedRuntimeCall:
    operation: RuntimeOperation
    call: GenerateIntent | JsonOutCall | ChatCall | EmbeddingCall
    # Present only for embed — the sole credential-bearing facade method. The
    # key is deliberately NOT captured.
    credential_provider: Presence[ProviderName]


def _validated_stream_script(script: Sequence[CodecStreamEvent]) -> tuple[CodecStreamEvent, ...]:
    events = tuple(script)
    if not events or not isinstance(events[-1], TerminalEvent):
        raise AssertionError("ScriptedRuntime stream script must end with a TerminalEvent")
    for event in events[:-1]:
        if isinstance(event, TerminalEvent):
            raise AssertionError("ScriptedRuntime stream script has events after its TerminalEvent")
    return events


async def _envelopes(script: tuple[CodecStreamEvent, ...]) -> AsyncIterator[RuntimeStreamEvent]:
    for seq, event in enumerate(script, start=1):
        yield RuntimeStreamEvent(seq=seq, event=event)


def _target_of(intent: GenerateIntent) -> str:
    return f"target {intent.target.provider}:{intent.target.model}"


class ScriptedRuntime:
    """Facade-level drop-in for `ProviderRuntime` replaying scripted results.

    Same public methods and signatures as the real runtime; each method pops
    its own queue in call order and appends a `CapturedRuntimeCall` to
    `calls`. An exhausted queue or a shape mismatch (a `StructuredReply` for
    the wrong model, a stream script violating the one-terminal grammar) is a
    loud `AssertionError`. Cancel signals are accepted for signature parity
    and ignored — scripted results are never raced against anything.

    `embed` is the one method whose expected-failure channel is a raise
    rather than a value: scripting a `NonGenerationCallFailed` there raises
    it, mirroring the real facade exactly.
    """

    def __init__(
        self,
        *,
        generate_outcomes: Iterable[CallOutcome] = (),
        stream_scripts: Iterable[Sequence[CodecStreamEvent]] = (),
        json_out_results: Iterable[JsonOutResult] = (),
        chat_outcomes: Iterable[CallOutcome] = (),
        embed_responses: Iterable[EmbeddingResponse | NonGenerationCallFailed] = (),
    ) -> None:
        self.calls: list[CapturedRuntimeCall] = []
        self._generate_outcomes = deque(generate_outcomes)
        self._stream_scripts = deque(_validated_stream_script(s) for s in stream_scripts)
        self._json_out_results = deque(json_out_results)
        self._chat_outcomes = deque(chat_outcomes)
        self._embed_responses: deque[EmbeddingResponse | NonGenerationCallFailed] = deque(
            embed_responses
        )

    async def generate(
        self, intent: GenerateIntent, *, cancel: CancelSignal | None = None
    ) -> CallOutcome:
        del cancel
        self._capture("generate", intent, Absent())
        return self._pop(self._generate_outcomes, "generate", _target_of(intent))

    def stream(
        self, intent: GenerateIntent, *, cancel: CancelSignal | None = None
    ) -> AsyncIterator[RuntimeStreamEvent]:
        # Eager like the real facade, which raises its call-shape defects
        # before any iteration: capture, pop, and fail at call time.
        del cancel
        self._capture("stream", intent, Absent())
        return _envelopes(self._pop(self._stream_scripts, "stream", _target_of(intent)))

    async def json_out[T: pydantic.BaseModel](
        self,
        model: type[T],
        intent: GenerateIntent,
        *,
        cancel: CancelSignal | None = None,
    ) -> StructuredReply[T] | Refused | Incomplete | Cancelled | Failed:
        del cancel
        self._capture("json_out", JsonOutCall(model=model, intent=intent), Absent())
        result = self._pop(
            self._json_out_results, "json_out", f"{model.__name__} for {_target_of(intent)}"
        )
        if isinstance(result, StructuredReply) and not isinstance(result.value, model):
            raise AssertionError(
                f"ScriptedRuntime json_out result is "
                f"StructuredReply[{type(result.value).__name__}] but the call "
                f"requested {model.__name__}"
            )
        return result

    async def chat(
        self,
        ref: str,
        *,
        system: str = "",
        user: str,
        reasoning: ReasoningLevel = "none",
        max_output_tokens: int | None = None,
    ) -> CallOutcome:
        self._capture(
            "chat",
            ChatCall(
                ref=ref,
                system=system,
                user=user,
                reasoning=reasoning,
                max_output_tokens=max_output_tokens,
            ),
            Absent(),
        )
        return self._pop(self._chat_outcomes, "chat", f"ref {ref!r}")

    async def embed(
        self, call: EmbeddingCall, *, credential: ProviderCredential
    ) -> EmbeddingResponse:
        self._capture("embed", call, Present(credential.provider))
        result = self._pop(self._embed_responses, "embed", f"model {call.model!r}")
        if isinstance(result, NonGenerationCallFailed):
            raise result
        return result

    def _capture(
        self,
        operation: RuntimeOperation,
        call: GenerateIntent | JsonOutCall | ChatCall | EmbeddingCall,
        credential_provider: Presence[ProviderName],
    ) -> None:
        self.calls.append(
            CapturedRuntimeCall(
                operation=operation, call=call, credential_provider=credential_provider
            )
        )

    def _pop[T](self, queue: deque[T], operation: RuntimeOperation, described: str) -> T:
        if queue:
            return queue.popleft()
        # _capture always precedes _pop, so the per-operation capture count is
        # this call's 1-based number.
        number = sum(1 for captured in self.calls if captured.operation == operation)
        raise AssertionError(
            f"ScriptedRuntime received {operation} call {number} ({described}) "
            f"but scripted only {number - 1}"
        )


__all__ = [
    "CapturedRuntimeCall",
    "ChatCall",
    "EngineScriptStep",
    "FakeEngine",
    "JsonOutCall",
    "JsonOutResult",
    "RuntimeOperation",
    "ScriptedRuntime",
]
