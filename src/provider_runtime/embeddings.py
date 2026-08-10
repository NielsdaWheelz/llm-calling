"""OpenAI embeddings port — one SDK attempt per call; the runtime owns retries.

The port is openai-only and Nexus-live (spec §13): ``ProviderRuntime.embed``
drives `embed_once` through the shared retry owner (`retry.attempts`), so this
module makes exactly ONE attempt: retryable trouble raises `TransientAttempt`,
context overflow — the one expected terminal failure — returns as a value the
runtime wraps in `NonGenerationCallFailed`, and defects raise their own types.

Wire: the SDK's own embeddings lane (``AsyncOpenAI.embeddings.create``,
``max_retries=0``, explicit timeout, injectable http_client, default base
URL — the rows are openai-proper only). ``encoding_format`` is deliberately
omitted so the SDK default applies: it requests base64 on the wire and decodes
the packed float32 payload back to plain floats itself; a provider answering
JSON float lists passes through that decode untouched.

Decode validation is preserved verbatim-in-behavior from the pre-cutover port:
missing, invalid, duplicate, or uncovered indexes, non-list vectors, and
non-numeric or non-finite values are all `ProtocolDefect` — one vector per
input, returned in input order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

import httpx
import openai
from openai.types import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding

from provider_runtime.engines import TransientAttempt
from provider_runtime.errors import (
    CredentialRejected,
    ProtocolDefect,
    RuntimeDefect,
    safe_provider_error_body_snippet,
)
from provider_runtime.types import (
    Absent,
    Billability,
    EmbeddingCall,
    EmbeddingResponse,
    NotDispatched,
    PossiblyBillable,
    Presence,
    Present,
    ProviderContextTooLarge,
    ProviderCredential,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderTimeout,
    TokenUsage,
    TransportUnavailable,
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
# Error classification (port of the old openai codec's classify_error; mirrors
# the openai engines' module-local classifiers)


def _terminal_http_failure(error: openai.APIStatusError) -> ProviderContextTooLarge:
    """Classify a non-2xx response.

    Returns the ONE expected terminal failure (context overflow); raises
    CredentialRejected / RuntimeDefect for terminal operator conditions and
    TransientAttempt for retryable ones.
    """
    status = error.status_code
    # The SDK unwraps body["error"] before attaching it, so error.body is the
    # inner error object for openai-shaped error envelopes.
    inner = dict(error.body) if isinstance(error.body, Mapping) else None
    code = inner.get("code") if inner else None
    error_type = inner.get("type") if inner else None
    message = inner.get("message") if inner else None
    snippet = (
        safe_provider_error_body_snippet({"error": inner} if inner else None) or f"HTTP {status}"
    )

    if status in (401, 403):
        raise CredentialRejected(
            message=f"openai rejected the platform credential (HTTP {status}): {snippet}"
        )
    if status == 402 or "insufficient_quota" in (code, error_type):
        raise RuntimeDefect(
            origin="provider_http",
            code="quota_exhausted",
            message=f"openai quota/billing exhausted (HTTP {status}): {snippet}",
        )
    if code == "context_length_exceeded" or (
        isinstance(message, str) and "maximum context length" in message.lower()
    ):
        return ProviderContextTooLarge()
    if status == 429:
        raise TransientAttempt(
            cause=ProviderRateLimit(retry_after=_retry_after_seconds(error.response.headers)),
            status_code=Present(status),
            provider_request_id=presence_of(error.request_id),
            billability=PossiblyBillable(),
        )
    if status in (500, 502, 503, 504):
        raise TransientAttempt(
            cause=ProviderHttpUnavailable(),
            status_code=Present(status),
            provider_request_id=presence_of(error.request_id),
            billability=PossiblyBillable(),
        )
    raise RuntimeDefect(
        origin="provider_http",
        code="unclassified_provider_error",
        message=f"openai returned an unclassified error (HTTP {status}): {snippet}",
    )


def _transient_connection(error: openai.APIConnectionError) -> TransientAttempt:
    if isinstance(error, openai.APITimeoutError):
        return TransientAttempt(
            cause=ProviderTimeout(),
            status_code=Absent(),
            provider_request_id=Absent(),
            billability=PossiblyBillable(),
        )
    # A pure pre-connect failure means no bytes reached the provider; every
    # other transport error implies the connection was at least opened.
    billability: Billability = (
        NotDispatched() if isinstance(error.__cause__, httpx.ConnectError) else PossiblyBillable()
    )
    return TransientAttempt(
        cause=TransportUnavailable(),
        status_code=Absent(),
        provider_request_id=Absent(),
        billability=billability,
    )


def _retry_after_seconds(headers: httpx.Headers) -> Presence[float]:
    raw = headers.get("retry-after")
    if raw is None:
        return Absent()
    try:
        seconds = float(raw)
    except ValueError:
        return Absent()
    return Present(seconds) if seconds >= 0 else Absent()


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
    return Present(
        TokenUsage.from_components(
            input_tokens=_int_or_none(raw.prompt_tokens) or 0,
            output_tokens=0,
            total_tokens=presence_of(_int_or_none(raw.total_tokens)),
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Absent(),
            cache_write_input_tokens=Absent(),
        )
    )


# ---------------------------------------------------------------------------
# The runtime's single-attempt seam


async def embed_once(
    call: EmbeddingCall,
    credential: ProviderCredential,
    *,
    http_client: httpx.AsyncClient | None,
) -> EmbeddingResponse | ProviderContextTooLarge:
    """One openai-SDK embeddings attempt; `ProviderRuntime.embed` owns the loop."""
    client = openai.AsyncOpenAI(
        api_key=credential.key,
        timeout=_TIMEOUT_S,
        max_retries=0,
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
            return _terminal_http_failure(error)
        except openai.APIConnectionError as error:
            raise _transient_connection(error) from error
        except (ValueError, AttributeError) as error:
            # The SDK's own 2xx parse trips on malformed envelopes before this
            # module's decode sees them: invalid JSON (JSONDecodeError), empty
            # data or bad base64/length (ValueError), non-object rows in its
            # base64 post-parse (AttributeError).
            raise _defect(f"openai embeddings response failed SDK decode: {error}") from error
    finally:
        if http_client is None:
            await client.close()
    return _decode_response(response, expected_count=len(call.inputs))


__all__ = ["embed_once"]
