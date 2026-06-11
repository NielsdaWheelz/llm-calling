"""No-network runtime helpers for application tests."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Literal

from provider_runtime.catalog import DEFAULT_CATALOG, ModelCapability, ModelCatalog
from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.types import (
    EmbeddingCall,
    EmbeddingResponse,
    KeyProbeResult,
    ModelCall,
    ModelChunk,
    ModelRef,
    ModelResponse,
    ProviderApiKey,
    ProviderName,
    TranscriptionCall,
    TranscriptionResponse,
)

RuntimeOperation = Literal["generate", "stream", "embed", "transcribe", "probe_key"]


@dataclass(frozen=True)
class CapturedRuntimeCall:
    operation: RuntimeOperation
    call: ModelCall | EmbeddingCall | TranscriptionCall | None
    key: ProviderApiKey
    timeout_s: float
    provider: ProviderName | None = None


class NoNetworkRuntime:
    """Runtime implementation that defects on provider I/O in tests."""

    def __init__(self, *, catalog: ModelCatalog = DEFAULT_CATALOG):
        self._catalog = catalog

    def is_provider_available(self, provider: str) -> bool:
        return self._catalog.key_probe_model(provider) is not None  # type: ignore[arg-type]

    def capabilities(self, model: ModelRef) -> ModelCapability | None:
        return self._catalog.capabilities(model)

    async def generate(
        self,
        call: ModelCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> ModelResponse:
        raise AssertionError(_unexpected_network_message("generate", call.model))

    async def stream(
        self,
        call: ModelCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> AsyncIterator[ModelChunk]:
        raise AssertionError(_unexpected_network_message("stream", call.model))
        yield ModelChunk(done=True)

    async def embed(
        self,
        call: EmbeddingCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> EmbeddingResponse:
        raise AssertionError(_unexpected_network_message("embed", call.model))

    async def transcribe(
        self,
        call: TranscriptionCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> TranscriptionResponse:
        raise AssertionError(_unexpected_network_message("transcribe", call.model))

    async def probe_key(
        self,
        *,
        provider: ProviderName,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> KeyProbeResult:
        raise AssertionError(f"Unexpected provider key probe in test: {provider}")


class ScriptedRuntime(NoNetworkRuntime):
    """No-network runtime with queued responses for deterministic tests."""

    def __init__(
        self,
        *,
        catalog: ModelCatalog = DEFAULT_CATALOG,
        generate_responses: Iterable[ModelResponse] = (),
        stream_chunks: Iterable[Iterable[ModelChunk]] = (),
        embed_responses: Iterable[EmbeddingResponse] = (),
        transcribe_responses: Iterable[TranscriptionResponse] = (),
        probe_results: Iterable[KeyProbeResult] = (),
    ):
        super().__init__(catalog=catalog)
        self.calls: list[CapturedRuntimeCall] = []
        self._generate_responses = deque(generate_responses)
        self._stream_chunks = deque(tuple(chunks) for chunks in stream_chunks)
        self._embed_responses = deque(embed_responses)
        self._transcribe_responses = deque(transcribe_responses)
        self._probe_results = deque(probe_results)

    async def generate(
        self,
        call: ModelCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> ModelResponse:
        self.calls.append(CapturedRuntimeCall("generate", call, key, timeout_s))
        return self._pop(self._generate_responses, "generate")

    async def stream(
        self,
        call: ModelCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> AsyncIterator[ModelChunk]:
        self.calls.append(CapturedRuntimeCall("stream", call, key, timeout_s))
        for chunk in self._pop(self._stream_chunks, "stream"):
            yield chunk

    async def embed(
        self,
        call: EmbeddingCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> EmbeddingResponse:
        self.calls.append(CapturedRuntimeCall("embed", call, key, timeout_s))
        return self._pop(self._embed_responses, "embed")

    async def transcribe(
        self,
        call: TranscriptionCall,
        *,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> TranscriptionResponse:
        self.calls.append(CapturedRuntimeCall("transcribe", call, key, timeout_s))
        return self._pop(self._transcribe_responses, "transcribe")

    async def probe_key(
        self,
        *,
        provider: ProviderName,
        key: ProviderApiKey,
        timeout_s: float = 45,
    ) -> KeyProbeResult:
        self.calls.append(CapturedRuntimeCall("probe_key", None, key, timeout_s, provider))
        return self._pop(self._probe_results, "probe_key")

    @staticmethod
    def _pop[T](queue: deque[T], operation: RuntimeOperation) -> T:
        try:
            return queue.popleft()
        except IndexError as exc:
            raise AssertionError(f"No scripted provider-runtime {operation} result queued") from exc


def model_not_available(provider: ProviderName, model: str) -> ModelCallError:
    return ModelCallError(
        ModelCallErrorCode.MODEL_NOT_AVAILABLE,
        f"No scripted model result queued for {provider}/{model}",
        provider=provider,
        retryable=False,
    )


def _unexpected_network_message(operation: RuntimeOperation, model: ModelRef) -> str:
    return f"Unexpected provider-runtime {operation} in test: {model.provider}/{model.model}"
