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
- ``LLM_RUNTIME_LIVE_CLAUDE_EXECUTABLE`` optionally pins the exact CLI binary under
  test when ambient ``claude`` resolves to a dispatch wrapper;
- every certified route writes one sanitized evidence file into
  ``tests/live/evidence/``.

Per route the matrix certifies, against the real subscription account: one full
streamed turn under the default restrictive policy (six-kind grammar, exactly
one terminal, normalized ``TokenUsage``), close/reopen/resume on the same native
session, and a structured-output turn. Codex additionally runs at least six
turns on that thread and independently witnesses the raw cumulative snapshots:
the invocation-local terminal values must sum exactly to the native cumulative
delta without counting the restored resume snapshot. The matrix never enrolls
an account or prints credentials; evidence values pass through the package's
own redaction before they are written.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never

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
from provider_runtime.agent_runtime.codex_sdk import CodexSdkAdapter
from provider_runtime.agent_runtime.types import AGENT_ROUTES
from provider_runtime.types import Absent, Presence, Present, TokenUsage
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


class _ObservedCodexSdkAdapter(CodexSdkAdapter):
    """Live-only witness for the native cumulative values before projection."""

    def __init__(self) -> None:
        super().__init__()
        self.cumulative_by_thread: dict[str, list[tuple[str, TokenUsage]]] = {}
        self.restored_by_thread: dict[str, list[TokenUsage]] = {}

    def _token_usage(self, params: Mapping[str, object]) -> TokenUsage:
        cumulative = super()._token_usage(params)
        thread_id = params.get("threadId")
        turn_id = params.get("turnId")
        assert isinstance(thread_id, str) and isinstance(turn_id, str)
        self.cumulative_by_thread.setdefault(thread_id, []).append((turn_id, cumulative))
        return cumulative

    def _restored_usage_baseline(self, client: Any, native_session_id: str) -> TokenUsage | None:
        baseline = super()._restored_usage_baseline(client, native_session_id)
        if baseline is not None:
            self.restored_by_thread.setdefault(native_session_id, []).append(baseline)
        return baseline


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


def _claude_executable() -> str:
    # Certification pins the exact CLI build under test. Ambient `claude` may be a
    # dispatch wrapper (e.g. an account router) that cannot run under the scrubbed
    # child environment; the override names the real binary.
    raw = os.environ.get("LLM_RUNTIME_LIVE_CLAUDE_EXECUTABLE")
    if raw is None:
        return "claude"
    path = Path(raw)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        _fail("LLM_RUNTIME_LIVE_CLAUDE_EXECUTABLE must be an absolute executable path")
    return raw


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


def _optional_count(value: Presence[int]) -> int | None:
    if isinstance(value, Present):
        return value.value
    assert isinstance(value, Absent)
    return None


def _usage_components(usage: TokenUsage) -> dict[str, int | None]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": _optional_count(usage.reasoning_tokens),
        "cache_read_input_tokens": _optional_count(usage.cache_read_input_tokens),
        "cache_write_input_tokens": _optional_count(usage.cache_write_input_tokens),
    }


def _sum_usage(usages: list[TokenUsage]) -> dict[str, int | None]:
    components = [_usage_components(usage) for usage in usages]
    result: dict[str, int | None] = {}
    for name in components[0]:
        values = [item[name] for item in components]
        if all(value is None for value in values):
            result[name] = None
            continue
        assert all(type(value) is int for value in values), (
            f"Codex usage field presence changed across live turn deltas: {name}"
        )
        result[name] = sum(value for value in values if isinstance(value, int))
    return result


def _usage_difference(current: TokenUsage, baseline: TokenUsage) -> dict[str, int | None]:
    current_components = _usage_components(current)
    baseline_components = _usage_components(baseline)
    result: dict[str, int | None] = {}
    for name, current_value in current_components.items():
        baseline_value = baseline_components[name]
        if current_value is None and baseline_value is None:
            result[name] = None
            continue
        assert type(current_value) is int and type(baseline_value) is int
        assert current_value >= baseline_value
        result[name] = current_value - baseline_value
    return result


def _codex_usage_evidence(
    observer: _ObservedCodexSdkAdapter,
    thread_id: str,
    terminals: list[AgentTerminal],
) -> dict[str, object]:
    assert len(terminals) >= 6
    local = []
    for terminal in terminals:
        assert isinstance(terminal.usage, Present), (
            f"Codex live turn supplied no attributable usage: {terminal!r}"
        )
        local.append(terminal.usage.value)

    snapshots = observer.cumulative_by_thread[thread_id]
    by_turn: dict[str, list[TokenUsage]] = {}
    for turn_id, snapshot in snapshots:
        by_turn.setdefault(turn_id, []).append(snapshot)
    assert len(by_turn) == len(terminals), (
        f"expected one raw cumulative group per Codex turn; got {len(by_turn)} "
        f"for {len(terminals)} turns"
    )
    groups = list(by_turn.values())
    restored = observer.restored_by_thread[thread_id]
    assert restored == [groups[0][-1]], (
        "Codex close/reopen/resume did not expose the restored cumulative baseline"
    )

    expected_turns = [_usage_components(groups[0][-1])]
    expected_turns.append(_usage_difference(groups[1][-1], restored[0]))
    expected_turns.extend(
        _usage_difference(groups[index][-1], groups[index - 1][-1])
        for index in range(2, len(groups))
    )
    assert [_usage_components(usage) for usage in local] == expected_turns, (
        "a Codex terminal did not equal its independently witnessed fixed-baseline delta"
    )

    invocation_sum = _sum_usage(local)
    cumulative_delta = _usage_components(groups[-1][-1])
    assert invocation_sum == cumulative_delta, (
        "invocation-local Codex usage did not sum to the real cumulative delta: "
        f"local={invocation_sum!r} cumulative={cumulative_delta!r}"
    )
    return {
        "turn_count": len(terminals),
        "resume_replay_observed": True,
        "resume_history_excluded": True,
        "local_sum_matches_cumulative_delta": True,
        "invocation_sum": invocation_sum,
        "cumulative_delta": cumulative_delta,
    }


async def _certify_route(route: LiveRoute, model: str | None) -> dict[str, object]:
    config = AgentRuntimeConfig(
        state_root_base=_state_root_base(), claude_executable=_claude_executable()
    )
    auth = CredentialRef(kind="local_account", profile_key=_profile())
    workspace = _state_root_base() / "live-workspace"
    workspace.mkdir(mode=0o700, exist_ok=True)
    evidence: dict[str, object] = {"route": route.name, "model": model or "backend-default"}
    observer = _ObservedCodexSdkAdapter() if route.backend == "codex" else None
    adapters = (observer,) if observer is not None else None
    async with AgentRuntime(config, adapters=adapters) as runtime:
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
        first = events[-1]
        assert isinstance(first, AgentTerminal)
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
        thread_terminals = [first, second]
        if observer is not None:
            for turn_number in range(3, 7):
                result = await runtime.run_turn(
                    resumed,
                    TurnRequest(
                        input=(
                            TextContent(
                                f"Reply with only the single digit {turn_number}, no punctuation"
                            ),
                        ),
                        timeout_seconds=_TURN_TIMEOUT_SECONDS,
                    ),
                )
                assert result.status == "succeeded", (
                    f"Codex live usage turn {turn_number} failed: "
                    f"{result.failure!r} {result.diagnostics}"
                )
                thread_terminals.append(result)
            evidence["codex_usage_accounting"] = _codex_usage_evidence(
                observer, ref.native_session_id, thread_terminals
            )
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
