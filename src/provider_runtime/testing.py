"""No-network runtime doubles for application tests.

`NoNetworkRuntime` fails loudly on any provider I/O; `ScriptedRuntime` replays
queued outcomes/scripts while recording every call. Both keep the structural
interface of `ProviderRuntime`'s public methods (`generate`, `stream`,
`embed`, `transcribe`) so application code accepts either.

Captured calls never store the credential key — only the provider name.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from provider_runtime.types import (
    CallOutcome,
    CancelSignal,
    CodecStreamEvent,
    EmbeddingCall,
    EmbeddingResponse,
    FinalizedProviderCall,
    ProviderCredential,
    ProviderName,
    RuntimeStreamEvent,
    StreamStart,
    TerminalEvent,
    TranscriptionCall,
    TranscriptionResponse,
)

type RuntimeOperation = Literal["generate", "stream", "embed", "transcribe"]


@dataclass(frozen=True, slots=True)
class CapturedRuntimeCall:
    operation: RuntimeOperation
    call: FinalizedProviderCall | EmbeddingCall | TranscriptionCall
    # The key is deliberately NOT captured.
    credential_provider: ProviderName
    streamed: bool


def _unexpected(operation: RuntimeOperation, provider: str, model: str) -> str:
    return f"Unexpected provider-runtime {operation} in test: {provider}/{model}"


class NoNetworkRuntime:
    """Runtime double that fails on any provider I/O in tests."""

    async def generate(
        self, call: FinalizedProviderCall, *, credential: ProviderCredential
    ) -> CallOutcome:
        raise AssertionError(
            _unexpected("generate", call.request.target.provider, call.request.target.model)
        )

    async def stream(
        self,
        call: FinalizedProviderCall,
        *,
        credential: ProviderCredential,
        cancel: CancelSignal | None = None,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        del cancel
        raise AssertionError(
            _unexpected("stream", call.request.target.provider, call.request.target.model)
        )
        # Unreachable yield: keeps this an async generator like the real runtime.
        yield RuntimeStreamEvent(seq=1, event=StreamStart())

    async def embed(
        self, call: EmbeddingCall, *, credential: ProviderCredential
    ) -> EmbeddingResponse:
        raise AssertionError(_unexpected("embed", "openai", call.model))

    async def transcribe(
        self, call: TranscriptionCall, *, credential: ProviderCredential
    ) -> TranscriptionResponse:
        raise AssertionError(_unexpected("transcribe", "openai", call.model))


@dataclass(slots=True)
class _Scripts:
    generate: deque[CallOutcome]
    stream: deque[tuple[CodecStreamEvent, ...]]
    embed: deque[EmbeddingResponse]
    transcribe: deque[TranscriptionResponse]


def _validated_script(script: Sequence[CodecStreamEvent]) -> tuple[CodecStreamEvent, ...]:
    events = tuple(script)
    if not events or not isinstance(events[-1], TerminalEvent):
        raise AssertionError("Scripted provider-runtime stream must end with a TerminalEvent")
    for event in events[:-1]:
        if isinstance(event, TerminalEvent):
            raise AssertionError(
                "Scripted provider-runtime stream has events after its TerminalEvent"
            )
    return events


class ScriptedRuntime(NoNetworkRuntime):
    """No-network runtime with queued responses for deterministic tests.

    Stream scripts are codec-event sequences; the double wraps them in
    `RuntimeStreamEvent` envelopes (seq starting at 1 per stream) exactly like
    the real runtime, and enforces the one-terminal grammar: each script must
    end with exactly one `TerminalEvent` and contain none before it.
    """

    def __init__(
        self,
        *,
        generate_outcomes: Iterable[CallOutcome] = (),
        stream_scripts: Iterable[Sequence[CodecStreamEvent]] = (),
        embed_responses: Iterable[EmbeddingResponse] = (),
        transcribe_responses: Iterable[TranscriptionResponse] = (),
    ) -> None:
        self.calls = []
        self._scripts = _Scripts(
            generate=deque(generate_outcomes),
            stream=deque(_validated_script(script) for script in stream_scripts),
            embed=deque(embed_responses),
            transcribe=deque(transcribe_responses),
        )

    async def generate(
        self, call: FinalizedProviderCall, *, credential: ProviderCredential
    ) -> CallOutcome:
        self._capture("generate", call, credential, streamed=False)
        return _pop(self._scripts.generate, "generate")

    async def stream(
        self,
        call: FinalizedProviderCall,
        *,
        credential: ProviderCredential,
        cancel: CancelSignal | None = None,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        del cancel
        self._capture("stream", call, credential, streamed=True)
        script = _pop(self._scripts.stream, "stream")
        for seq, event in enumerate(script, start=1):
            yield RuntimeStreamEvent(seq=seq, event=event)

    async def embed(
        self, call: EmbeddingCall, *, credential: ProviderCredential
    ) -> EmbeddingResponse:
        self._capture("embed", call, credential, streamed=False)
        return _pop(self._scripts.embed, "embed")

    async def transcribe(
        self, call: TranscriptionCall, *, credential: ProviderCredential
    ) -> TranscriptionResponse:
        self._capture("transcribe", call, credential, streamed=False)
        return _pop(self._scripts.transcribe, "transcribe")

    def _capture(
        self,
        operation: RuntimeOperation,
        call: FinalizedProviderCall | EmbeddingCall | TranscriptionCall,
        credential: ProviderCredential,
        *,
        streamed: bool,
    ) -> None:
        self.calls.append(
            CapturedRuntimeCall(
                operation=operation,
                call=call,
                credential_provider=credential.provider,
                streamed=streamed,
            )
        )


def _pop[T](queue: deque[T], operation: RuntimeOperation) -> T:
    try:
        return queue.popleft()
    except IndexError:
        raise AssertionError(f"No scripted provider-runtime {operation} result queued") from None


__all__ = [
    "CapturedRuntimeCall",
    "NoNetworkRuntime",
    "RuntimeOperation",
    "ScriptedRuntime",
]
