"""OpenAI embeddings port — one SDK attempt per call; the runtime owns retries.

The port is openai-only and Nexus-live (spec §13): ``ProviderRuntime.embed``
drives `embed_once` through the shared retry owner (`retry.attempts`), so this
module makes exactly ONE attempt: retryable trouble raises `TransientAttempt`,
context overflow — the one expected terminal failure — returns as a value the
runtime wraps in `NonGenerationCallFailed`, and defects raise their own types.

Wire: the SDK's own embeddings lane (``AsyncOpenAI.embeddings.create``,
``max_retries=0``, explicit timeout, injectable http_client, and the canonical
OpenAI host pinned explicitly — this port is openai-only, and the shared
`zero_env_client` keeps every ambient SDK env read out of the request).
``encoding_format`` is deliberately omitted so the SDK default applies: it
requests base64 on the wire and decodes the packed float32 payload back to
plain floats itself; a provider answering JSON float lists passes through that
decode untouched. HTTP and transport classification are the shared openai ones;
what is local here is the credential gate and the SDK-decode breakage arm.

Decode validation is preserved verbatim-in-behavior from the pre-cutover port:
missing, invalid, duplicate, or uncovered indexes, non-list vectors, and
non-numeric or non-finite values are all `ProtocolDefect` — one vector per
input, returned in input order.
"""

from __future__ import annotations

import math
from typing import Final

import httpx
import openai
from openai.types import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding

from provider_runtime.engines._openai_common import (
    CANONICAL_BASE_URL,
    terminal_http_failure,
    transient_connection,
    zero_env_client,
)
from provider_runtime.errors import InvalidRequest, ProtocolDefect
from provider_runtime.types import (
    Absent,
    EmbeddingCall,
    EmbeddingResponse,
    Presence,
    Present,
    ProviderContextTooLarge,
    ProviderCredential,
    TokenUsage,
    presence_of,
)

# Prior-art non-generation request timeout (old runtime's _REQUEST_TIMEOUT_S);
# embeddings never stream, so the generation engines' long budget is wrong here.
_TIMEOUT_S: Final[float] = 45.0


def _defect(message: str) -> ProtocolDefect:
    return ProtocolDefect(code="invalid_embedding_response", message=message)


def _int_or_none(value: object) -> int | None:
    # bool is an int subclass; token counts are never booleans.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# ---------------------------------------------------------------------------
# Decode — the SDK constructs response models leniently (no validation), so
# every field is untrusted at this boundary.


def _decode_response(
    response: CreateEmbeddingResponse, *, expected_count: int
) -> EmbeddingResponse:
    rows = response.data
    if not isinstance(rows, list):
        raise _defect("openai embeddings response missing data")
    embeddings_by_index: dict[int, tuple[float, ...]] = {}
    for row in rows:
        if not isinstance(row, Embedding):
            raise _defect("openai embeddings response contains invalid row")
        index = row.index
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= expected_count
            or index in embeddings_by_index
        ):
            raise _defect("openai embeddings response contains invalid index")
        if not isinstance(row.embedding, list):
            raise _defect("openai embeddings response contains invalid vector")
        try:
            vector = tuple(float(value) for value in row.embedding)
        except (TypeError, ValueError):
            raise _defect("openai embeddings response contains non-numeric vector value") from None
        if not all(math.isfinite(value) for value in vector):
            raise _defect("openai embeddings response contains non-finite vector value")
        embeddings_by_index[index] = vector
    if set(embeddings_by_index) != set(range(expected_count)):
        raise _defect("openai embeddings response has incomplete indexes")
    embeddings = tuple(embeddings_by_index[index] for index in range(expected_count))
    return EmbeddingResponse(embeddings=embeddings, usage=_decode_usage(response.usage))


def _decode_usage(raw: object) -> Presence[TokenUsage]:
    if not isinstance(raw, Usage):
        return Absent()
    try:
        usage = TokenUsage.from_components(
            input_tokens=_int_or_none(raw.prompt_tokens) or 0,
            output_tokens=0,
            total_tokens=presence_of(_int_or_none(raw.total_tokens)),
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Absent(),
            cache_write_input_tokens=Absent(),
        )
    except ValueError as error:
        raise _defect(
            f"openai embeddings response usage is not valid token accounting: {error}"
        ) from error
    return Present(usage)


# ---------------------------------------------------------------------------
# The runtime's single-attempt seam


async def embed_once(
    call: EmbeddingCall,
    credential: ProviderCredential,
    *,
    http_client: httpx.AsyncClient | None,
) -> EmbeddingResponse | ProviderContextTooLarge:
    """One openai-SDK embeddings attempt; `ProviderRuntime.embed` owns the loop."""
    if credential.provider != "openai":
        raise InvalidRequest(
            message=f"embeddings port is openai-only; got a {credential.provider!r} credential"
        )
    client = zero_env_client(
        api_key=credential.key,
        base_url=CANONICAL_BASE_URL,
        timeout_s=_TIMEOUT_S,
        http_client=http_client,
    )
    try:
        try:
            response = await client.embeddings.create(
                model=call.model,
                input=list(call.inputs),
                dimensions=(
                    call.dimensions.value if isinstance(call.dimensions, Present) else openai.omit
                ),
            )
        except openai.APIStatusError as error:
            return terminal_http_failure(error)
        except openai.APIConnectionError as error:
            raise transient_connection(error) from error
        except (ValueError, AttributeError, TypeError) as error:
            # The SDK's own 2xx parse trips on malformed envelopes before this
            # module's decode sees them: invalid JSON (JSONDecodeError), empty
            # data or bad base64/length (ValueError), non-object rows in its
            # base64 post-parse (AttributeError), a truthy non-iterable `data`
            # (TypeError from its `for embedding in obj.data`).
            raise _defect(f"openai embeddings response failed SDK decode: {error}") from error
    finally:
        if http_client is None:
            await client.close()
    return _decode_response(response, expected_count=len(call.inputs))


__all__ = ["embed_once"]
