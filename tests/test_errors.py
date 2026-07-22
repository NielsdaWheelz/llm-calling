"""Tests for the rewritten defect hierarchy and the preserved redaction machinery."""

import pytest

from provider_runtime.errors import (
    CredentialRejected,
    PlanningDefect,
    ProtocolDefect,
    RuntimeDefect,
    SchemaViolation,
    safe_provider_error_body_snippet,
    sanitize_provider_text,
)

# ---------------------------------------------------------------------------
# Defect hierarchy

SECRET_TEXT = (
    "Bad keys sk-live-abcdefghijklmnopqrstuvwxyz1234567890, "
    "AIzaSyabcdefghijklmnopqrstuvwxyz12345, and "
    "Bearer very-secret-provider-token; "
    '"x-api-key": "cf-secret-token-12345"; '
    "https://example.invalid/?api_key=query-secret-12345"
)

SECRET_FRAGMENTS = (
    "sk-live",
    "AIza",
    "very-secret-provider-token",
    "cf-secret-token-12345",
    "query-secret-12345",
)


def test_defect_hierarchy_and_fixed_origin_code_pairs() -> None:
    schema = SchemaViolation("root must be an object schema")
    assert isinstance(schema, PlanningDefect), "SchemaViolation must sit under PlanningDefect"
    assert isinstance(schema, RuntimeDefect)
    assert isinstance(schema, Exception)
    assert schema.origin == "plan"
    assert schema.code == "schema_violation"
    assert str(schema) == "root must be an object schema"

    planning = PlanningDefect(
        code="continuation_mismatch",
        message="continuation artifact targets anthropic/claude-sonnet-5",
    )
    assert planning.origin == "plan", "every planning defect carries origin=plan"
    assert planning.code == "continuation_mismatch"

    protocol = ProtocolDefect(code="malformed_envelope", message="missing terminal frame")
    assert protocol.origin == "provider_response", (
        "protocol defects are the operator-side provider_response origin"
    )
    assert protocol.code == "malformed_envelope"

    credential = CredentialRejected(message="anthropic HTTP 401")
    assert credential.origin == "provider_http"
    assert credential.code == "credential_rejected"
    assert str(credential) == "anthropic HTTP 401"


def test_planning_defect_documents_the_plan_rejected_exclusion() -> None:
    doc = PlanningDefect.__doc__ or ""
    assert "PlanRejected" in doc, (
        "PlanningDefect's contract must state that the expected oversize case is "
        "PlanRejected, not a defect"
    )


def test_defects_raise_with_their_safe_message() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        raise ProtocolDefect(code="unknown_terminal", message="finish_reason 'weird'")
    assert exc_info.value.message == "finish_reason 'weird'"
    assert exc_info.value.args == ("finish_reason 'weird'",)


# ---------------------------------------------------------------------------
# Redaction — behavior preserved verbatim from the pre-cutover errors.py


def test_sanitize_provider_text_redacts_known_secret_shapes() -> None:
    cleaned = sanitize_provider_text(SECRET_TEXT)
    assert "Bad keys" in cleaned, "non-secret prose must survive sanitization"
    assert "...redacted" in cleaned
    for fragment in SECRET_FRAGMENTS:
        assert fragment not in cleaned, f"secret fragment {fragment!r} leaked through redaction"


def test_sanitize_provider_text_redacts_escaped_json_secret_fields() -> None:
    raw = '{"detail": "upstream said \\"api_key\\": \\"super-secret-value\\""}'
    cleaned = sanitize_provider_text(raw)
    assert "super-secret-value" not in cleaned, "escaped-JSON secret values must be redacted"


def test_sanitize_provider_text_enforces_the_500_char_bound() -> None:
    assert len(sanitize_provider_text("a" * 2000)) == 500, "default bound is 500 chars"
    assert sanitize_provider_text("abcdef", limit=3) == "abc"
    assert sanitize_provider_text("   hi   ") == "hi", "surrounding whitespace is stripped"


def test_error_body_snippet_summarizes_and_redacts_json_error_bodies() -> None:
    snippet = safe_provider_error_body_snippet(
        {
            "error": {
                "message": "Bad key sk-live-abcdefghijklmnopqrstuvwxyz1234567890 was rejected",
                "type": "invalid_request_error",
                "code": "bad_key",
            }
        },
        None,
    )
    assert snippet is not None
    assert "invalid_request_error" in snippet
    assert "bad_key" in snippet
    assert "...redacted" in snippet
    assert "sk-live" not in snippet, "secret key material leaked into the snippet"


def test_error_body_snippet_redacts_every_secret_shape() -> None:
    snippet = safe_provider_error_body_snippet({"error": {"message": SECRET_TEXT}}, None)
    assert snippet is not None
    assert "...redacted" in snippet
    for fragment in SECRET_FRAGMENTS:
        assert fragment not in snippet, f"secret fragment {fragment!r} leaked into the snippet"


def test_error_body_snippet_is_bounded_to_500_chars() -> None:
    snippet = safe_provider_error_body_snippet({"error": {"message": "x" * 2000}}, None)
    assert snippet is not None
    assert len(snippet) <= 500, f"snippet must respect the 500-char bound, got {len(snippet)}"


def test_error_body_snippet_reads_top_level_and_string_error_shapes() -> None:
    top_level = safe_provider_error_body_snippet({"message": "boom", "code": "oops"}, None)
    assert top_level is not None
    assert "boom" in top_level
    assert "oops" in top_level

    string_error = safe_provider_error_body_snippet({"error": "upstream exploded"}, None)
    assert string_error is not None
    assert "upstream exploded" in string_error


def test_plain_text_bodies_are_never_persisted_as_snippets() -> None:
    # Non-JSON provider prose may embed prompt content; it is dropped entirely.
    snippet = safe_provider_error_body_snippet(
        None,
        "Rejected prompt fragment: user private content",
    )
    assert snippet is None, "plain-text bodies must not become operator snippets"
    assert safe_provider_error_body_snippet({}, None) is None
    assert safe_provider_error_body_snippet({"unrelated": {"nested": 1}}, None) is None
