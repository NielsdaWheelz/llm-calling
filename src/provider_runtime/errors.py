"""Runtime defect hierarchy and provider-text redaction.

Defects are broken invariants, impossible states, and contract violations —
never product control flow. Expected, modelable failures live in
`provider_runtime.types` as the closed `ExpectedModelFailure` union (returned as
values); defects raise.

Transient signals are runtime-internal (`_TransientSignal` in the retry
boundary); `classify_error` RETURNS values and never returns a defect as a value
— it raises `ProtocolDefect`.

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

    Carries the §9 ledger `origin`, a closed per-subclass `code`, and a
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


class PlanningDefect(RuntimeDefect):
    """A planner-detected invariant violation.

    Covers invalid schemas, invalid/mismatched cache scopes,
    continuation target/codec mismatch, and unsupported cache intent.

    EXPLICITLY EXCLUDES the expected oversize case: an intent measuring over the
    contract's context limit is returned as ``PlanRejected(IntentContextTooLarge)``
    — the expected planner rejection channel — never raised as a defect.
    """

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(origin="plan", code=code, message=message)


class SchemaViolation(PlanningDefect):
    """An authored schema falls outside the canonical JSON Schema subset (§5)."""

    def __init__(self, message: str) -> None:
        super().__init__(code="schema_violation", message=message)


class ProtocolDefect(RuntimeDefect):
    """A malformed provider envelope or unknown terminal provider response.

    Raised (never returned) by codec decode/classify ingress.
    """

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(origin="provider_response", code=code, message=message)


class CredentialRejected(RuntimeDefect):
    """The platform credential was rejected by the provider (HTTP 401/403).

    A defect per §9 — platform configuration is an operator fact, so rejection is
    never a product-facing failure.
    """

    def __init__(self, *, message: str) -> None:
        super().__init__(origin="provider_http", code="credential_rejected", message=message)


# ---------------------------------------------------------------------------
# Secret redaction — preserved verbatim-in-behavior from the pre-cutover
# errors.py (patterns + 500-char bound).

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


def safe_provider_error_body_snippet(
    json_body: dict | None,
    body_text: str | None,
) -> str | None:
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
