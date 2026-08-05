from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

from provider_runtime.agent_runtime.capabilities import AgentCapabilityScope
from provider_runtime.agent_runtime.errors import (
    ConcurrentTurn,
    InvalidAgentRequest,
    ProtocolDefect,
    SessionMismatch,
    SessionUnavailable,
)
from provider_runtime.agent_runtime.events import AgentEvent, TurnStartedData
from provider_runtime.agent_runtime.sessions import (
    AgentSession,
    SessionMetadata,
    SessionPage,
    SessionQuery,
    SessionReadOptions,
    SessionSnapshot,
    SessionSummary,
    fingerprint_path,
    validate_read_session_auth,
    validate_session_ref,
)
from provider_runtime.agent_runtime.types import (
    AgentSessionRef,
    CredentialRef,
    FrozenJsonDict,
)

AUTH = CredentialRef(kind="local_account", profile_key="personal")
SCOPE = AgentCapabilityScope(backend="codex", transport="sdk", auth=AUTH)


def _ref(*, cwd_fingerprint: str | None = None) -> AgentSessionRef:
    return AgentSessionRef(
        schema_version="agent-session-ref.v1",
        backend="codex",
        transport="sdk",
        native_session_id="thread-123",
        profile_key="personal",
        state_root_fingerprint="a" * 64,
        cwd_fingerprint=cwd_fingerprint or "b" * 64,
    )


def test_path_fingerprint_is_sha256_of_resolved_absolute_path(tmp_path: Path) -> None:
    path = tmp_path / "repo"
    expected = hashlib.sha256(str(path.resolve()).encode()).hexdigest()

    assert fingerprint_path(path) == expected


def test_resume_identity_rejects_backend_profile_and_state_root_mismatch() -> None:
    with pytest.raises(SessionMismatch, match="profile_key"):
        validate_session_ref(
            _ref(),
            AgentCapabilityScope(
                backend="codex",
                transport="sdk",
                auth=CredentialRef(kind="local_account", profile_key="other"),
            ),
            state_root_fingerprint="a" * 64,
            cwd="/workspace/repo",
            cwd_scopes_sessions=False,
        )
    with pytest.raises(SessionMismatch, match="state_root_fingerprint"):
        validate_session_ref(
            _ref(),
            SCOPE,
            state_root_fingerprint="c" * 64,
            cwd="/workspace/repo",
            cwd_scopes_sessions=False,
        )


def test_cwd_only_gates_backends_that_scope_sessions_by_cwd() -> None:
    validate_session_ref(
        _ref(),
        SCOPE,
        state_root_fingerprint="a" * 64,
        cwd="/workspace/repo",
        cwd_scopes_sessions=False,
    )
    with pytest.raises(SessionMismatch, match="cwd_fingerprint"):
        validate_session_ref(
            _ref(),
            SCOPE,
            state_root_fingerprint="a" * 64,
            cwd="/workspace/repo",
            cwd_scopes_sessions=True,
        )


def test_session_rejects_a_concurrent_turn_instead_of_waiting() -> None:
    session = AgentSession(_ref())

    session.begin_turn()
    with pytest.raises(ConcurrentTurn):
        session.begin_turn()
    session.end_turn()
    session.begin_turn()
    session.end_turn()


def test_new_session_ref_is_completed_exactly_once_before_persistence() -> None:
    session = AgentSession()
    assert not session.ref_is_complete
    with pytest.raises(ProtocolDefect, match="before session_started"):
        _ = session.ref

    session.complete_ref(_ref())
    assert session.ref == _ref()
    with pytest.raises(ProtocolDefect, match="more than once"):
        session.complete_ref(_ref())


def test_session_mismatch_and_native_unavailability_are_distinct_errors() -> None:
    assert not issubclass(SessionUnavailable, SessionMismatch)
    assert SessionUnavailable("native session is gone").code == "session_unavailable"


def test_session_discovery_values_are_frozen_and_paginated() -> None:
    metadata = SessionMetadata(name="Work", archived=False, tags=("repo",))
    summary = SessionSummary(ref=_ref(), metadata=metadata)
    page = SessionPage(sessions=(summary,), continuation_cursor="next")
    query = SessionQuery(scope=SCOPE, cursor=None, limit=25)
    native_source = {"items": [{"value": "one"}]}
    item = AgentEvent(
        schema_version="agent-event.v1",
        seq=1,
        backend="codex",
        transport="sdk",
        session_ref=_ref(),
        turn_id="turn-1",
        kind="turn_started",
        data=TurnStartedData(),
        native_payload=FrozenJsonDict(native_source),
    )
    native_source["items"].append({"value": "two"})
    snapshot = SessionSnapshot(
        ref=_ref(), metadata=metadata, items=(item,), continuation_cursor=None
    )

    assert page.sessions[0].metadata.name == "Work"
    assert query.limit == 25
    assert snapshot.items == (item,)
    assert item.native_payload == {"items": ({"value": "one"},)}
    with pytest.raises(InvalidAgentRequest, match="AgentEvent"):
        SessionSnapshot(
            ref=_ref(),
            metadata=metadata,
            items=cast(tuple[AgentEvent, ...], ({"kind": "raw"},)),
        )
    options = SessionReadOptions(auth=AUTH)
    validate_read_session_auth(_ref(), options)
    with pytest.raises(SessionMismatch, match="auth profile"):
        validate_read_session_auth(
            _ref(),
            SessionReadOptions(auth=CredentialRef(kind="local_account", profile_key="other")),
        )
    with pytest.raises(InvalidAgentRequest, match="positive"):
        SessionReadOptions(auth=AUTH, limit=0)
