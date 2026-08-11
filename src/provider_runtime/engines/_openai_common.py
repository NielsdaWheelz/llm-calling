"""Shared openai-protocol code: the two openai-SDK engines plus the embeddings port.

Three call sites drive the same SDK over the same error envelope, so client
construction and HTTP/transport classification are one implementation here.
What differs stays with the caller: the registry row each engine resolves, the
compat engine's provider-parameterized messages and gateway shapes, the
embeddings port's credential gate and SDK-decode handling. Everything not
specific to this SDK — Retry-After parsing, cause-chain reading — lives in
`_common`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import httpx
import openai

from provider_runtime.engines import TransientAttempt
from provider_runtime.engines._common import caused_by, retry_after_seconds
from provider_runtime.errors import (
    CredentialRejected,
    RuntimeDefect,
    safe_provider_error_body_snippet,
)
from provider_runtime.types import (
    Absent,
    Billability,
    NotDispatched,
    PossiblyBillable,
    Present,
    ProviderContextTooLarge,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderTimeout,
    TransportUnavailable,
    presence_of,
)

# OpenAI proper's host, pinned by callers rather than left to the SDK's own
# fallback, so OPENAI_BASE_URL never decides where a call goes.
CANONICAL_BASE_URL: Final[str] = "https://api.openai.com/v1"


def zero_env_client(
    *, api_key: str, base_url: str, timeout_s: float, http_client: httpx.AsyncClient | None
) -> openai.AsyncOpenAI:
    """An AsyncOpenAI that dispatches on these arguments and nothing ambient.

    The SDK constructor (openai 2.53.0, `_client.py`) reads OPENAI_ORG_ID,
    OPENAI_PROJECT_ID and OPENAI_WEBHOOK_SECRET whenever the matching argument
    is omitted, and parses OPENAI_CUSTOM_HEADERS into the client's default
    headers unconditionally — that parse runs before, and merges under, any
    `default_headers` argument, so no constructor argument suppresses it.
    Passing the other three does not help either: a non-None organization or
    project is sent as a header verbatim, empty string included. All four reads
    land on exactly these four attributes; clearing them is the suppression.
    """
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_s,
        max_retries=0,
        http_client=http_client,
    )
    client.organization = None
    client.project = None
    client.webhook_secret = None
    client._custom_headers = {}
    return client


def terminal_http_failure(error: openai.APIStatusError) -> ProviderContextTooLarge:
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
            cause=ProviderRateLimit(retry_after=retry_after_seconds(error.response.headers)),
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


def transient_connection(error: openai.APIConnectionError) -> TransientAttempt:
    if isinstance(error, openai.APITimeoutError):
        # The SDK collapses every httpx timeout into one type; only the cause
        # says which. A connect timeout is a pure pre-connect failure — the
        # handshake never completed, so no request bytes reached the provider.
        return TransientAttempt(
            cause=ProviderTimeout(),
            status_code=Absent(),
            provider_request_id=Absent(),
            billability=(
                NotDispatched() if caused_by(error, httpx.ConnectTimeout) else PossiblyBillable()
            ),
        )
    # A pure pre-connect failure means no bytes reached the provider; every
    # other transport error implies the connection was at least opened.
    billability: Billability = (
        NotDispatched() if caused_by(error, httpx.ConnectError) else PossiblyBillable()
    )
    return TransientAttempt(
        cause=TransportUnavailable(),
        status_code=Absent(),
        provider_request_id=Absent(),
        billability=billability,
    )
