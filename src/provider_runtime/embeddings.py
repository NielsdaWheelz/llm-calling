"""OpenAI-compatible embeddings client."""

import math

import httpx

from provider_runtime.errors import ModelCallError, ModelCallErrorCode, raise_for_provider_error
from provider_runtime.types import EmbeddingCall, EmbeddingResponse, TokenUsage


class EmbeddingsClient:
    def __init__(self, client: httpx.AsyncClient, *, provider: str, base_url: str):
        self._client = client
        self._provider = provider
        self._url = f"{base_url.rstrip('/')}/embeddings"

    async def embed(
        self,
        call: EmbeddingCall,
        *,
        api_key: str,
        timeout_s: float,
    ) -> EmbeddingResponse:
        payload: dict[str, object] = {"model": call.model.model, "input": call.inputs}
        if call.dimensions is not None:
            payload["dimensions"] = call.dimensions

        response = await self._client.post(
            self._url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        await raise_for_provider_error(response, self._provider)

        data = response.json()
        rows = data.get("data")
        if not isinstance(rows, list):
            raise ModelCallError(
                ModelCallErrorCode.PROVIDER_DOWN,
                f"{self._provider} embeddings response missing data",
                provider=self._provider,
                retryable=False,
            )

        embeddings_by_index: dict[int, list[float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ModelCallError(
                    ModelCallErrorCode.PROVIDER_DOWN,
                    f"{self._provider} embeddings response contains invalid row",
                    provider=self._provider,
                    retryable=False,
                )
            index = row.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(call.inputs)
                or index in embeddings_by_index
            ):
                raise ModelCallError(
                    ModelCallErrorCode.PROVIDER_DOWN,
                    f"{self._provider} embeddings response contains invalid index",
                    provider=self._provider,
                    retryable=False,
                )
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                raise ModelCallError(
                    ModelCallErrorCode.PROVIDER_DOWN,
                    f"{self._provider} embeddings response contains invalid vector",
                    provider=self._provider,
                    retryable=False,
                )
            vector = [float(value) for value in embedding]
            if not all(math.isfinite(value) for value in vector):
                raise ModelCallError(
                    ModelCallErrorCode.PROVIDER_DOWN,
                    f"{self._provider} embeddings response contains non-finite vector value",
                    provider=self._provider,
                    retryable=False,
                )
            embeddings_by_index[index] = vector

        if set(embeddings_by_index) != set(range(len(call.inputs))):
            raise ModelCallError(
                ModelCallErrorCode.PROVIDER_DOWN,
                f"{self._provider} embeddings response has incomplete indexes",
                provider=self._provider,
                retryable=False,
            )
        embeddings = [embeddings_by_index[index] for index in range(len(call.inputs))]

        usage = None
        usage_data = data.get("usage")
        if isinstance(usage_data, dict):
            usage = TokenUsage(
                input_tokens=usage_data.get("prompt_tokens"),
                output_tokens=None,
                total_tokens=usage_data.get("total_tokens"),
                provider_usage=dict(usage_data),
            )

        return EmbeddingResponse(
            embeddings=embeddings,
            usage=usage,
            provider_request_id=response.headers.get("x-request-id") or data.get("id"),
        )
