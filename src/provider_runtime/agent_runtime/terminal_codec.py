"""One closed durable encoding of normalized provider evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from provider_runtime.types import Absent, Present, TokenUsage, canonical_json_bytes

from .events import AgentFailure, AgentFailureCause, AgentQuotaExhausted, AgentTerminal
from .turn import (
    AgentAccepted,
    AgentAttempt,
    AgentNotSubmitted,
    AgentResultRef,
    AgentSubmission,
    AgentTurnRef,
    AgentUncertain,
    LocalStopEvidence,
    NativeTerminalEvidence,
    RawAgentOutput,
)
from .types import (
    freeze_json_object,
    freeze_json_value,
    ref_from_json,
    ref_to_json,
    thaw_json_value,
)


class _Document(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Attempt(_Document):
    attempt_id: str
    request_digest: str


class _NativeRef(_Document):
    kind: Literal["codex_turn", "claude_result"]
    session_ref: dict[str, object]
    native_id: str | None


class _Submission(_Document):
    kind: Literal["accepted", "uncertain", "not_submitted"]
    attempt: _Attempt
    turn: _NativeRef | None
    reason: str | None


class _NativeEvidence(_Document):
    origin: Literal["native"]
    attempt: _Attempt
    native_ref: _NativeRef
    seal_revision: Literal["codex-turn-completed.v1", "claude-result.v1"]


class _LocalEvidence(_Document):
    origin: Literal["local_stop"]
    submission: _Submission
    reason: str


class _Usage(_Document):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None
    cache_read_input_tokens: int | None
    cache_write_input_tokens: int | None


class _RawOutput(_Document):
    value: object


class _Terminal(_Document):
    schema_version: Literal["agent-terminal.v2"]
    status: Literal["succeeded", "failed", "cancelled"]
    failure: Literal["quota_exhausted"] | AgentFailureCause | None
    final_text: str
    session_ref: dict[str, object]
    evidence: _NativeEvidence | _LocalEvidence
    raw_structured_output: _RawOutput | None
    usage: _Usage | None
    diagnostics: tuple[str, ...]


def attempt_to_json(attempt: AgentAttempt) -> dict[str, str]:
    return {"attempt_id": attempt.attempt_id, "request_digest": attempt.request_digest}


def attempt_from_json(value: Mapping[str, object]) -> AgentAttempt:
    document = _Attempt.model_validate_json(canonical_json_bytes(freeze_json_object(value)))
    return AgentAttempt(document.attempt_id, document.request_digest)


def _native_ref_json(ref: AgentTurnRef | AgentResultRef) -> dict[str, object]:
    return {
        "kind": "codex_turn" if isinstance(ref, AgentTurnRef) else "claude_result",
        "session_ref": thaw_json_value(ref_to_json(ref.session_ref)),
        "native_id": ref.native_turn_id if isinstance(ref, AgentTurnRef) else ref.native_result_id,
    }


def _native_ref(value: _NativeRef) -> AgentTurnRef | AgentResultRef:
    ref = ref_from_json(value.session_ref)
    if value.kind == "codex_turn":
        if value.native_id is None:
            raise ValueError("a Codex terminal requires its actual native turn identity")
        return AgentTurnRef(ref, value.native_id)
    return AgentResultRef(ref, value.native_id)


def submission_to_json(value: AgentSubmission) -> dict[str, object]:
    if isinstance(value, AgentAccepted):
        kind, reason, turn = "accepted", None, _native_ref_json(value.turn)
    elif isinstance(value, AgentUncertain):
        kind, reason = "uncertain", value.reason
        turn = None if value.turn is None else _native_ref_json(value.turn)
    elif isinstance(value, AgentNotSubmitted):
        kind, reason, turn = "not_submitted", value.reason, None
    else:
        raise ValueError("submission_to_json requires explicit submission evidence")
    return {"kind": kind, "attempt": attempt_to_json(value.attempt), "turn": turn, "reason": reason}


def _submission(document: _Submission) -> AgentSubmission:
    attempt = AgentAttempt(document.attempt.attempt_id, document.attempt.request_digest)
    turn = None if document.turn is None else _native_ref(document.turn)
    if turn is not None and not isinstance(turn, AgentTurnRef):
        raise ValueError("a submission cannot identify a Claude result")
    if document.kind == "accepted":
        if turn is None or document.reason is not None:
            raise ValueError("accepted submission requires a native turn and no uncertain reason")
        return AgentAccepted(attempt, turn)
    if document.reason is None:
        raise ValueError("negative or unresolved submission requires its original reason")
    if document.kind == "not_submitted":
        if turn is not None:
            raise ValueError("non-submission cannot contain a native turn")
        return AgentNotSubmitted(attempt, document.reason)
    return AgentUncertain(attempt, turn, document.reason)


def submission_from_json(value: Mapping[str, object]) -> AgentSubmission:
    document = _Submission.model_validate_json(canonical_json_bytes(freeze_json_object(value)))
    return _submission(document)


def terminal_to_json(terminal: AgentTerminal) -> dict[str, object]:
    """Preserve original evidence and raw output without product decoding."""
    evidence = terminal.evidence
    if isinstance(evidence, NativeTerminalEvidence):
        encoded_evidence = {
            "origin": "native",
            "attempt": attempt_to_json(evidence.attempt),
            "native_ref": _native_ref_json(evidence.native_ref),
            "seal_revision": evidence.seal_revision,
        }
    else:
        encoded_evidence = {
            "origin": "local_stop",
            "submission": submission_to_json(evidence.submission),
            "reason": evidence.reason,
        }
    usage = terminal.usage.value if isinstance(terminal.usage, Present) else None
    value = {
        "schema_version": "agent-terminal.v2",
        "status": terminal.status,
        "failure": "quota_exhausted"
        if isinstance(terminal.failure, AgentQuotaExhausted)
        else terminal.failure.cause
        if isinstance(terminal.failure, AgentFailure)
        else None,
        "final_text": terminal.final_text,
        "session_ref": thaw_json_value(ref_to_json(terminal.session_ref)),
        "evidence": encoded_evidence,
        "raw_structured_output": None
        if terminal.raw_structured_output is None
        else {"value": thaw_json_value(terminal.raw_structured_output.value)},
        "usage": None
        if usage is None
        else {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "reasoning_tokens": usage.reasoning_tokens.value
            if isinstance(usage.reasoning_tokens, Present)
            else None,
            "cache_read_input_tokens": usage.cache_read_input_tokens.value
            if isinstance(usage.cache_read_input_tokens, Present)
            else None,
            "cache_write_input_tokens": usage.cache_write_input_tokens.value
            if isinstance(usage.cache_write_input_tokens, Present)
            else None,
        },
        "diagnostics": list(terminal.diagnostics),
    }
    return _Terminal.model_validate_json(
        canonical_json_bytes(freeze_json_object(value))
    ).model_dump(mode="json")


def terminal_from_json(value: Mapping[str, object]) -> AgentTerminal:
    """Decode only the current closed schema; absent historical provenance fails."""
    document = _Terminal.model_validate_json(canonical_json_bytes(freeze_json_object(value)))
    stored = document.evidence
    if isinstance(stored, _NativeEvidence):
        evidence = NativeTerminalEvidence(
            AgentAttempt(stored.attempt.attempt_id, stored.attempt.request_digest),
            _native_ref(stored.native_ref),
            stored.seal_revision,
        )
    else:
        submission = _submission(stored.submission)
        if not isinstance(submission, AgentAccepted | AgentUncertain):
            raise ValueError("a local stop cannot be proof of non-submission")
        evidence = LocalStopEvidence(submission, stored.reason)
    usage = document.usage
    failure = (
        AgentQuotaExhausted()
        if document.failure == "quota_exhausted"
        else AgentFailure(document.failure)
        if document.failure is not None
        else None
    )
    return AgentTerminal(
        status=document.status,
        failure=failure,
        final_text=document.final_text,
        session_ref=ref_from_json(document.session_ref),
        evidence=evidence,
        raw_structured_output=None
        if document.raw_structured_output is None
        else RawAgentOutput(freeze_json_value(document.raw_structured_output.value)),
        usage=Absent()
        if usage is None
        else Present(
            TokenUsage(
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
                Absent() if usage.reasoning_tokens is None else Present(usage.reasoning_tokens),
                Absent()
                if usage.cache_read_input_tokens is None
                else Present(usage.cache_read_input_tokens),
                Absent()
                if usage.cache_write_input_tokens is None
                else Present(usage.cache_write_input_tokens),
            )
        ),
        diagnostics=document.diagnostics,
    )


__all__ = [
    "attempt_from_json",
    "attempt_to_json",
    "submission_from_json",
    "submission_to_json",
    "terminal_from_json",
    "terminal_to_json",
]
