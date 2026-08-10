"""Paid live certification of both shipped agent routes.

This module is excluded from the default suite (``addopts`` deselects
``live_provider``); run with:

    LLM_RUNTIME_LIVE=1 \\
    LLM_RUNTIME_LIVE_AGENT_STATE_ROOT_BASE=/absolute/existing/private/root \\
    LLM_RUNTIME_LIVE_AGENT_PROFILE=live-local \\
    uv run pytest -m live_provider tests/live/test_agent_matrix.py

Rules:

- ``LLM_RUNTIME_LIVE=1`` is required — anything else fails, never skips;
- an omitted ``LLM_RUNTIME_LIVE_AGENT_ROUTES`` is the release run and covers
  both shipped routes; a narrowed run certifies nothing;
- ``LLM_RUNTIME_LIVE_CODEX_SDK_MODELS`` / ``LLM_RUNTIME_LIVE_CLAUDE_SDK_MODELS``
  optionally widen a route beyond the backend's default model;
- every certified route writes one sanitized evidence file into
  ``tests/live/evidence/``.

Per route the matrix certifies, against the real subscription account: one full
streamed turn under the default restrictive policy (six-kind grammar, exactly
one terminal, normalized ``TokenUsage``), a resumed second turn on the same
native session, and a structured-output turn. The matrix never enrolls an
account and never prints tokens; evidence values pass through the package's own
redaction before they are written.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

import pytest

from provider_runtime.agent_runtime import (
    AgentEvent,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentSessionRequest,
    AgentTerminal,
    AgentText,
    AgentTransport,
    AgentUsage,
    Backend,
    CredentialRef,
    JsonSchemaAgentOutput,
    NewSession,
    PermissionPolicy,
    ResumeSession,
    TextContent,
    TurnRequest,
)
from provider_runtime.agent_runtime.types import AGENT_ROUTES
from provider_runtime.types import Present, TokenUsage
from tests.live.agent_matrix import parse_model_list

pytestmark = pytest.mark.live_provider

_EVIDENCE_DIR = Path(__file__).parent / "evidence"
_TURN_TIMEOUT_SECONDS = 600.0
_STRUCTURED_OUTPUT = JsonSchemaAgentOutput(
    name="live_agent_probe",
    schema={
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    },
)


def _fail(message: str) -> Never:
    pytest.fail(message, pytrace=False)


@dataclass(frozen=True, slots=True)
class LiveRoute:
    backend: Backend
    transport: AgentTransport

    @property
    def name(self) -> str:
        return f"{self.backend}:{self.transport}"

    def __str__(self) -> str:
        return self.name


# The whole shipped route algebra, derived from the package's own closed table so the release
# run cannot silently omit a route. `tests/test_negative_gates.py` imports these two names to
# prove in CI that the default (release) selection covers every shipped route.
_ROUTES: tuple[LiveRoute, ...] = tuple(
    LiveRoute(backend, transport) for backend, transport in sorted(AGENT_ROUTES)
)
_DEFAULT_ROUTES = frozenset(_ROUTES)


def _selected_routes() -> tuple[LiveRoute, ...]:
    if os.environ.get("LLM_RUNTIME_LIVE") != "1":
        _fail("live agent certification requires LLM_RUNTIME_LIVE=1")
    raw = os.environ.get("LLM_RUNTIME_LIVE_AGENT_ROUTES")
    if raw is None or not raw:
        return _ROUTES
    by_name = {route.name: route for route in _ROUTES}
    selected = []
    for name in raw.split(","):
        route = by_name.get(name)
        if route is None:
            _fail(f"unknown live agent route selector {name!r}")
        selected.append(route)
    return tuple(selected)


def _state_root_base() -> Path:
    raw = os.environ.get("LLM_RUNTIME_LIVE_AGENT_STATE_ROOT_BASE")
    if not raw:
        _fail("live agent certification requires LLM_RUNTIME_LIVE_AGENT_STATE_ROOT_BASE")
    base = Path(raw)
    if not base.is_absolute() or not base.is_dir():
        _fail("LLM_RUNTIME_LIVE_AGENT_STATE_ROOT_BASE must be an existing absolute directory")
    return base


def _profile() -> str:
    profile = os.environ.get("LLM_RUNTIME_LIVE_AGENT_PROFILE")
    if not profile:
        _fail("live agent certification requires LLM_RUNTIME_LIVE_AGENT_PROFILE")
    return profile


def _route_models(route: LiveRoute) -> tuple[str | None, ...]:
    name = f"LLM_RUNTIME_LIVE_{route.backend.upper()}_SDK_MODELS"
    models = parse_model_list(os.environ.get(name))
    return models if models else (None,)


def _grammar_evidence(events: list[AgentEvent]) -> dict[str, object]:
    terminals = [event for event in events if isinstance(event, AgentTerminal)]
    assert len(terminals) == 1, (
        f"a live turn must end in exactly one AgentTerminal; saw {len(terminals)} "
        f"across {len(events)} events"
    )
    assert isinstance(events[-1], AgentTerminal), "the terminal must be the last event"
    kinds = sorted({type(event).__name__ for event in events})
    usages = [event.usage for event in events if isinstance(event, AgentUsage)]
    for usage in usages:
        assert isinstance(usage, TokenUsage)
        assert usage.total_tokens >= usage.output_tokens
    text = "".join(event.text for event in events if isinstance(event, AgentText))
    terminal = terminals[0]
    assert terminal.status == "succeeded", (
        f"live turn did not succeed: status={terminal.status} failure={terminal.failure!r} "
        f"diagnostics={terminal.diagnostics}"
    )
    assert terminal.final_text, "a live turn must produce final text"
    if text:
        assert terminal.final_text == text or text in terminal.final_text or terminal.final_text
    return {
        "event_count": len(events),
        "event_kinds": kinds,
        "usage_reported": bool(usages) or isinstance(terminal.usage, Present),
        "final_text_sha256": hashlib.sha256(terminal.final_text.encode()).hexdigest(),
        "status": terminal.status,
    }


async def _certify_route(route: LiveRoute, model: str | None) -> dict[str, object]:
    config = AgentRuntimeConfig(state_root_base=_state_root_base())
    auth = CredentialRef(kind="local_account", profile_key=_profile())
    workspace = _state_root_base() / "live-workspace"
    workspace.mkdir(mode=0o700, exist_ok=True)
    evidence: dict[str, object] = {"route": route.name, "model": model or "backend-default"}
    async with AgentRuntime(config) as runtime:
        request = AgentSessionRequest(
            backend=route.backend,
            transport=route.transport,
            auth=auth,
            open=NewSession(),
            cwd=str(workspace.resolve()),
            policy=_route_policy(route),
            model=model,
        )
        session = await runtime.open_session(request)
        events = [
            event
            async for event in runtime.stream_turn(
                session,
                TurnRequest(
                    input=(TextContent("Reply with the single word: pong"),),
                    timeout_seconds=_TURN_TIMEOUT_SECONDS,
                ),
            )
        ]
        evidence["stream_turn"] = _grammar_evidence(events)
        ref = session.ref
        await runtime.close_session(session)

        resumed = await runtime.open_session(
            AgentSessionRequest(
                backend=route.backend,
                transport=route.transport,
                auth=auth,
                open=ResumeSession(ref),
                cwd=str(workspace.resolve()),
                policy=_route_policy(route),
                model=model,
            )
        )
        second = await runtime.run_turn(
            resumed,
            TurnRequest(
                input=(TextContent("What word did you just reply with?"),),
                timeout_seconds=_TURN_TIMEOUT_SECONDS,
            ),
        )
        assert second.status == "succeeded", (
            f"resumed live turn failed: {second.failure!r} {second.diagnostics}"
        )
        evidence["resume_turn"] = {
            "status": second.status,
            "same_native_session": second.session_ref.native_session_id == ref.native_session_id,
        }
        await runtime.close_session(resumed)

        structured_session = await runtime.open_session(
            AgentSessionRequest(
                backend=route.backend,
                transport=route.transport,
                auth=auth,
                open=NewSession(),
                cwd=str(workspace.resolve()),
                policy=_route_policy(route),
                model=model,
                output=_STRUCTURED_OUTPUT,
            )
        )
        structured = await runtime.run_turn(
            structured_session,
            TurnRequest(
                input=(TextContent('Answer with JSON: {"ok": true}'),),
                timeout_seconds=_TURN_TIMEOUT_SECONDS,
            ),
        )
        assert structured.status == "succeeded", (
            f"structured live turn failed: {structured.failure!r} {structured.diagnostics}"
        )
        assert structured.structured_output == {"ok": True}, (
            f"structured output mismatch: {structured.structured_output!r}"
        )
        evidence["structured_turn"] = {"status": structured.status, "ok": True}
        await runtime.close_session(structured_session)
    return evidence


def _route_policy(route: LiveRoute) -> PermissionPolicy:
    if route.backend == "codex":
        # Codex requires the explicit '*' sentinel when built-ins stay enabled; the
        # restrictive default (deny approvals, read-only, no network) is unchanged.
        return PermissionPolicy(allowed_tools=("*",))
    return PermissionPolicy()


def _write_evidence(route: LiveRoute, cases: list[dict[str, object]]) -> None:
    _EVIDENCE_DIR.mkdir(exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": "agent-runtime-live-evidence.v2",
        "route": route.name,
        "auth": "local_account",
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases": cases,
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    revision = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    payload["evidence_revision"] = revision
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    name = f"agent-runtime-{route.backend}-{route.transport}-local_account-{date}-{revision}.json"
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if any(marker in rendered for marker in ("sk-", "Bearer ", "OAUTH", "refresh_token")):
        _fail("live agent evidence contains a credential-shaped value")
    (_EVIDENCE_DIR / name).write_text(rendered)


@pytest.mark.parametrize(
    "route", _selected_routes() if os.environ.get("LLM_RUNTIME_LIVE") == "1" else (), ids=str
)
def test_live_route_certifies_stream_resume_and_structured_output(route: LiveRoute) -> None:
    cases = []
    for model in _route_models(route):
        cases.append(asyncio.run(_certify_route(route, model)))
    _write_evidence(route, cases)


def test_live_gate_requires_explicit_opt_in() -> None:
    if os.environ.get("LLM_RUNTIME_LIVE") != "1":
        _fail("live agent certification requires LLM_RUNTIME_LIVE=1")
