"""OpenAI embeddings port: request building and strict response validation.

The embedding port is openai-only (catalog `EmbeddingContract` rows). The
runtime dispatches `build_embedding_request` through the shared `Transport`
(Bearer auth injected there) with the central `EXTERNAL_LLM_RETRY` policy,
classifies non-2xx through the openai codec's `classify_error`, and hands 2xx
bodies to `parse_embedding_response`.

The strict response validation is preserved verbatim-in-behavior from the
pre-cutover `EmbeddingsClient`: missing/invalid data rows, invalid, duplicate,
or uncovered indexes, non-list vectors, non-numeric or non-finite values are
all rejected — now as `ProtocolDefect` (they were
`ModelCallError(PROVIDER_DOWN)` before the defect/expected-failure split).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from provider_runtime.errors import ProtocolDefect
from provider_runtime.types import (
    Absent,
    EmbeddingCall,
    EmbeddingResponse,
    FinalizedProviderRequest,
    Presence,
    Present,
    ProviderTarget,
    TokenUsage,
    presence_of,
)

_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


def _dump_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_embedding_request(call: EmbeddingCall) -> FinalizedProviderRequest:
    """The plain JSON POST for the openai embeddings endpoint.

    Reuses `FinalizedProviderRequest` so the shared Transport (auth injection,
    timeouts, verbatim passthrough) carries it; the protocol tag is the openai
    codec's — only its `classify_error` ever sees this request's errors."""
    body: dict[str, object] = {"model": call.model, "input": list(call.inputs)}
    if isinstance(call.dimensions, Present):
        body["dimensions"] = call.dimensions.value
    return FinalizedProviderRequest(
        target=ProviderTarget(provider="openai", model=call.model),
        protocol="openai_responses",
        url=_EMBEDDINGS_URL,
        method="POST",
        safe_headers={},
        body=_dump_bytes(body),
    )


def _defect(message: str) -> ProtocolDefect:
    return ProtocolDefect(code="invalid_embedding_response", message=message)


def parse_embedding_response(
    status: int,
    headers: Mapping[str, str],
    body: bytes,
    *,
    expected_count: int,
) -> EmbeddingResponse:
    """Decode a 2xx embeddings response; non-2xx goes through classify_error."""
    del status, headers  # 2xx only; no header-borne facts consumed
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise _defect("openai embeddings response is not valid JSON") from None
    if not isinstance(data, dict):
        raise _defect("openai embeddings response is not a JSON object")

    rows = data.get("data")
    if not isinstance(rows, list):
        raise _defect("openai embeddings response missing data")

    embeddings_by_index: dict[int, tuple[float, ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise _defect("openai embeddings response contains invalid row")
        index = row.get("index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= expected_count
            or index in embeddings_by_index
        ):
            raise _defect("openai embeddings response contains invalid index")
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            raise _defect("openai embeddings response contains invalid vector")
        try:
            vector = tuple(float(value) for value in embedding)
        except (TypeError, ValueError):
            raise _defect("openai embeddings response contains non-numeric vector value") from None
        if not all(math.isfinite(value) for value in vector):
            raise _defect("openai embeddings response contains non-finite vector value")
        embeddings_by_index[index] = vector

    if set(embeddings_by_index) != set(range(expected_count)):
        raise _defect("openai embeddings response has incomplete indexes")
    embeddings = tuple(embeddings_by_index[index] for index in range(expected_count))

    usage: Presence[TokenUsage] = Absent()
    usage_data = data.get("usage")
    if isinstance(usage_data, dict):
        prompt_tokens = usage_data.get("prompt_tokens")
        total_tokens = usage_data.get("total_tokens")
        usage = Present(
            TokenUsage.from_components(
                input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else 0,
                output_tokens=0,
                total_tokens=presence_of(total_tokens if isinstance(total_tokens, int) else None),
                reasoning_tokens=Absent(),
                cache_read_input_tokens=Absent(),
                cache_write_input_tokens=Absent(),
            )
        )

    return EmbeddingResponse(embeddings=embeddings, usage=usage)


__all__ = ["build_embedding_request", "parse_embedding_response"]
