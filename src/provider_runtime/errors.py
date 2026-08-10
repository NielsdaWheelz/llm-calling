"""Runtime defect hierarchy and provider-text redaction.

Defects are broken invariants, impossible states, and contract violations —
never product control flow. Expected, modelable failures live in
`provider_runtime.types` as the closed `ExpectedModelFailure` union (returned as
values); defects raise.

Retryable trouble is runtime-internal (`TransientAttempt` in
`provider_runtime.engines`); it never crosses the facade — exhaustion surfaces
as the `Failed(TransientExhausted)` value.

Every defect carries safe context only: no prompts, provider bodies,
credentials, or hidden reasoning. Callers embed provider diagnostics only via
`safe_provider_error_body_snippet` / `sanitize_provider_text`.
"""

from __future__ import annotations

import json
import re

from provider_runtime.types import FailureOrigin


class RuntimeDefect(Exception):
    """A broken runtime invariant.

    Carries the ledger `origin`, a closed per-subclass `code`, and a
    safe-context `message`. The worker boundary reports/re-raises defects and
    records origin/code/trace operator-side; a defect never becomes a product
    failure variant.
    """

    origin: FailureOrigin
    code: str
    message: str

    def __init__(self, *, origin: FailureOrigin, code: str, message: str) -> None:
        super().__init__(message)
        self.origin = origin
        self.code = code
        self.message = message


class InvalidRequest(RuntimeDefect):
    """A request that violates the contract or the resolved registry row.

    Raised for request validation failures: a continuation bound to a
    different target/codec, a `provider_options` key colliding with a core
    intent field the engine maps itself, image blocks sent to a text-only row,
    an unknown registry ref, or tools requested on a `tools=False` row.
    (Strict output on a `structured="json_mode"` row is NOT invalid —
    json_mode plus validation handles it.)
    """

    def __init__(self, *, message: str) -> None:
        super().__init__(origin="intent", code="invalid_request", message=message)


class ProtocolDefect(RuntimeDefect):
    """A malformed provider envelope or unknown terminal provider response.

    Raised (never returned) by engine decode/classify ingress.
    """

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(origin="provider_response", code=code, message=message)


class CredentialRejected(RuntimeDefect):
    """The platform credential was rejected by the provider (HTTP 401/403).

    A defect — platform configuration is an operator fact, so rejection is
    never a product-facing failure.
    """

    def __init__(self, *, message: str) -> None:
        super().__init__(origin="provider_http", code="credential_rejected", message=message)


class CredentialMissing(RuntimeDefect):
    """No key configured for the resolved row's provider.

    Raised at dispatch, before any bytes reach the provider — like rejection,
    a missing key is an operator fact, never a product-facing failure.
    """

    def __init__(self, *, message: str) -> None:
        super().__init__(origin="transport", code="credential_missing", message=message)


# ---------------------------------------------------------------------------
# Secret redaction (patterns + 500-char bound).

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "...redacted"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"), "...redacted"),
    (
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{10,}\b", re.IGNORECASE),
        "Bearer ...redacted",
    ),
    (
        re.compile(
            r"(?i)([?&](?:api[_-]?key|key|token|secret|access[_-]?token|"
            r"refresh[_-]?token|client[_-]?secret)=)[^&#\s]+"
        ),
        r"\1...redacted",
    ),
    (
        re.compile(
            r"(?i)(\"(?:api[_-]?key|x[_-]?api[_-]?key|key|token|secret|"
            r"authorization|access[_-]?token|refresh[_-]?token|client[_-]?secret)\""
            r"\s*:\s*\")[^\"]+(\")"
        ),
        r"\1...redacted\2",
    ),
    (
        re.compile(
            r"(?i)(\\\"(?:api[_-]?key|x[_-]?api[_-]?key|key|token|secret|"
            r"authorization|access[_-]?token|refresh[_-]?token|client[_-]?secret)\\\""
            r"\s*:\s*\\\")[^\\\"]+(\\\")"
        ),
        r"\1...redacted\2",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|x[_-]?api[_-]?key|key|token|secret|"
            r"authorization|access[_-]?token|refresh[_-]?token|client[_-]?secret)"
            r"\s*[:=]\s*[A-Za-z0-9._~+/=-]{8,}"
        ),
        r"\1=...redacted",
    ),
)


def sanitize_provider_text(text: str, *, limit: int = 500) -> str:
    snippet = (text or "").strip()[:limit]
    for pattern, replacement in _SECRET_PATTERNS:
        snippet = pattern.sub(replacement, snippet)
    return snippet


def safe_provider_error_body_snippet(json_body: dict | None) -> str | None:
    summary = _provider_error_summary(json_body)
    if summary:
        return sanitize_provider_text(
            json.dumps(summary, sort_keys=True, separators=(",", ":")),
            limit=500,
        )
    return None


def _provider_error_summary(json_body: dict | None) -> dict[str, object]:
    if not json_body:
        return {}
    summary: dict[str, object] = {}
    error = json_body.get("error")
    if isinstance(error, dict):
        for key in ("message", "type", "code", "param", "status"):
            value = error.get(key)
            if isinstance(value, str | int | float | bool) or value is None:
                if value is not None:
                    summary[key] = value
        return summary
    if isinstance(error, str):
        summary["message"] = error
    for key in ("message", "error_description", "code", "status"):
        value = json_body.get(key)
        if isinstance(value, str | int | float | bool):
            summary[key] = value
    return summary
