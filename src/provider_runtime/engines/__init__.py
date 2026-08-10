"""Engine protocol — the surface every protocol adapter implements.

ONE attempt per call: retryable trouble raises `TransientAttempt`; retries,
seq-numbering, and span emission live in `runtime.py`. Non-retryable expected
failure returns a failure-bearing `CallOutcome` with a fully populated
`CallMeta`; malformed envelopes raise `ProtocolDefect`; 401/403 raises
`CredentialRejected`.

Layering: this module imports from `types` and `registry` only. SDK imports
are confined to the sibling engine modules.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from provider_runtime.registry import ModelRow
from provider_runtime.types import (
    Billability,
    CallOutcome,
    CodecStreamEvent,
    GenerateIntent,
    Presence,
    ProviderCredential,
    TransientCause,
)


class TransientAttempt(Exception):
    """Internal signal: this attempt failed with a retryable cause. Never crosses the facade."""

    cause: TransientCause
    status_code: Presence[int]
    provider_request_id: Presence[str]
    billability: Billability

    def __init__(
        self,
        *,
        cause: TransientCause,
        status_code: Presence[int],
        provider_request_id: Presence[str],
        billability: Billability,
    ) -> None:
        # Cause leaves are closed value types with safe reprs (no provider text).
        super().__init__(f"retryable attempt failure: {cause!r}")
        self.cause = cause
        self.status_code = status_code
        self.provider_request_id = provider_request_id
        self.billability = billability


class Engine(Protocol):
    """One attempt against one provider; the runtime owns retries and the envelope."""

    async def generate(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> CallOutcome: ...

    def stream(
        self, row: ModelRow, intent: GenerateIntent, credential: ProviderCredential
    ) -> AsyncIterator[CodecStreamEvent]: ...
