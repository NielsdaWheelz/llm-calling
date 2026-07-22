"""Private codec <-> runtime seam.

Codec ``classify_error`` returns a :data:`ClassifiedError` value for exactly
classified provider error responses; everything else raises a defect
(``CredentialRejected`` for 401/403, ``RuntimeDefect`` for quota exhaustion and
other unknown non-transient provider failures, ``ProtocolDefect`` for
malformed envelopes). Transient causes feed the runtime retry boundary;
``ProviderContextTooLarge`` is terminal and folded into ``Failed`` by the
runtime, which owns the attempt trace.

``ExpectedFailureSignal`` is raised by a codec while decoding a 2xx
response/stream when it surfaces an expected model failure that is not an HTTP
error (today: strict tool-argument parse failure). The runtime catches it and
constructs ``Failed(meta, failure)``; codecs never construct ``Failed`` or
``Cancelled`` outcomes themselves.
"""

from __future__ import annotations

from provider_runtime.types import (
    InvalidToolArguments,
    ProviderContextTooLarge,
    TransientCause,
)

type ClassifiedError = TransientCause | ProviderContextTooLarge


class ExpectedFailureSignal(Exception):
    """Expected decode-time model failure; folded into ``Failed`` by the runtime."""

    def __init__(self, failure: InvalidToolArguments) -> None:
        super().__init__(failure.safe_detail)
        self.failure = failure


class TransientStreamError(Exception):
    """In-band transient stream condition raised by codec ``decode_stream``.

    Raised when the provider signals an in-band retryable/transient condition
    over an HTTP-200 stream: an OpenRouter mid-stream error chunk, or a stream
    that ends without any terminal frame (``ProviderStreamInterrupted``; codecs
    cannot reliably know whether semantic events were already consumed, so they
    pass ``partial_output=False`` and the runtime — which tracks emitted
    semantic events — rebuilds the leaf with the true flag). The runtime retry
    boundary handles it: pre-semantic-output it retries; post-semantic-output it
    folds into ``Failed(TransientExhausted(...))``.
    """

    def __init__(self, cause: TransientCause) -> None:
        super().__init__(type(cause).__name__)
        self.cause = cause


__all__ = ["ClassifiedError", "ExpectedFailureSignal", "TransientStreamError"]
