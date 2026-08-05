"""Opt-in live certification for the public local-agent runtime.

A release certification is the unnarrowed run::

    LLM_RUNTIME_LIVE=1 \
    LLM_RUNTIME_LIVE_AGENT_STATE_ROOT_BASE=/absolute/existing/private/root \
    LLM_RUNTIME_LIVE_AGENT_PROFILE=live-local \
    LLM_RUNTIME_LIVE_CLAUDE_SDK_MODELS=claude-model-a,claude-model-b \
    uv run pytest -m live_provider tests/live/test_agent_matrix.py

``LLM_RUNTIME_LIVE_AGENT_STATE_ROOT_BASE`` must be an existing, resolved,
absolute directory that is neither group- nor world-writable: ``AgentRuntime``
rejects a widely writable base with ``InvalidAgentRequest``, so a shared
temporary directory such as ``/tmp`` (mode ``1777``) is not a usable base.

``LLM_RUNTIME_LIVE_AGENT_ROUTES`` may explicitly narrow the run with
comma-separated ``codex:sdk`` and ``claude:sdk`` values. An omitted
selector means the release certification set, which is both shipped lanes.
Narrowed runs are debugging aids and certify nothing.

Every selected route requires an already-created directory at
``<state-root-base>/<backend>/<profile>``. The matrix never enrolls an account.
Codex certifies every model and that model's own discovered reasoning efforts.
Claude does not enumerate models, so ``LLM_RUNTIME_LIVE_CLAUDE_SDK_MODELS`` is
required as a strict comma-separated list; every listed model is certified at
every discovered Claude reasoning effort.
The optional ``LLM_RUNTIME_LIVE_CLAUDE_EXECUTABLE`` override takes a bare name
resolved on the operator's ``PATH`` or an absolute path. Codex has no executable override:
the pinned ``openai-codex`` SDK owns and matches its bundled runtime.

Claude's preflight pins the matrix at the **real** executable, because a
certification run against a wrapper certifies the wrapper. ``AgentRuntime``
resolves that executable with ``shutil.which`` and then ``Path.resolve()``. The
preflight resolves it the same way, refuses a resolved POSIX-shell wrapper, and
runs ``<resolved> --version`` in the exact child environment
``auth.build_child_environment`` builds. Codex is certified through the public
SDK boundary instead: capability discovery must initialize the exact pinned SDK
and report its matched bundled-runtime version before any model turn is sent.

The wrapper case is not hypothetical and it is not self-diagnosing, which is why
the refusal comes first and carries the whole story. A router that dispatches on
its own ``argv[0]`` sees the *resolved* name, because ``Path.resolve()`` already
collapsed the symlink that carried the tool's name — so it dies with something
like ``Unsupported AI router command name: ai-router`` and never mentions state
roots. The refusal instead names the symlink, the resolved script, the
``LLM_RUNTIME_LIVE_<BACKEND>_EXECUTABLE`` override to set, and the unwrapped
candidate it found further along the same ``PATH``, so the fix is a copy-paste::

    LLM_RUNTIME_LIVE_CLAUDE_EXECUTABLE=/absolute/path/to/real/claude \
    ...

Only the Claude shebang is inspected, so a wrapper that is a compiled binary or
is written in a non-shell language passes this check. The Claude SDK has no
native report of the effective state root; Codex receives its isolated home in
the SDK client configuration exercised by the deterministic boundary tests.

No shipped route accepts a named API-key credential: ``claude:cli``, which was
the only one that did, was removed with both CLI lanes. Certification therefore
no longer proves that ``api_key_environment`` or ``secret_reference`` *works* on
any agent lane — it proves the opposite, that every lane refuses both kinds
before reading a secret. ``ANTHROPIC_API_KEY`` and the API/secret profile
variables are no longer read. See ``docs/agent-runtime.md`` for the recorded
deviation from the acceptance contract this represents.

Exact native tool names come from ``AgentCapabilities.builtin_tool_names`` and
from nowhere else. A gate that carried its own vendor table could not be kept in
sync with the installed executable, and it would certify its own guess rather
than the capability report a consumer actually programs against. A route that
reports ``tool_controls=True`` and publishes no names fails the probes that need
exact names, with a message naming the adapter that must publish them.

Evidence is built from an explicit safe allowlist: route names, sanitized
versions, capability booleans/counts, and certified-feature booleans. It never
contains credential values, filesystem paths, profile keys, session refs, or
model output. Selection, network-policy, and turn-limit facts separately record
that a request was exercised and whether native effective-state output proved
the behavior; a request value alone is never effective-behavior evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Never

import pytest

import provider_runtime.agent_runtime.claude_sdk as claude_sdk
import provider_runtime.agent_runtime.codex_sdk as codex_sdk
from provider_runtime import parse_canonical_schema
from provider_runtime.agent_runtime import (
    TERMINAL_EVENT_KINDS,
    AgentCapabilities,
    AgentCapabilityScope,
    AgentEvent,
    AgentOutputSpec,
    AgentResult,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentSession,
    AgentSessionRef,
    AgentSessionRequest,
    AgentTransport,
    ApprovalAnsweredData,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    Backend,
    BuiltinToolFamily,
    CredentialKind,
    CredentialRef,
    CredentialUnavailable,
    DiagnosticData,
    ForkSession,
    InvalidAgentRequest,
    JsonSchemaAgentOutput,
    McpServerSpec,
    NewSession,
    PermissionPolicy,
    PermissionPolicyPatch,
    ReasoningSpec,
    ResumeSession,
    SessionOpen,
    SessionQuery,
    SessionReadOptions,
    TextAgentOutput,
    TextContent,
    ToolCompletedData,
    TurnCompletedData,
    TurnRequest,
    UnsafeConfirmation,
    UnsupportedCapability,
)
from provider_runtime.agent_runtime.auth import (
    AuthEnvironmentRequest,
    build_child_environment,
    resolve_state_root,
)
from provider_runtime.agent_runtime.types import AGENT_ROUTES
from tests.live.agent_matrix import (
    MatrixSelectionError,
    ModelReasoningCase,
    claude_model_reasoning_cases,
    codex_model_reasoning_cases,
    parse_claude_models,
)

pytestmark = pytest.mark.live_provider

_EVIDENCE_DIR = Path(__file__).parent / "evidence"
_MCP_FIXTURE = (Path(__file__).parent / "fixtures" / "agent_runtime_mcp.py").resolve()
# The package's own terminal grammar, not a restatement of it: a certification gate that
# carried its own copy of a closed set would keep passing after the set changed underneath it.
_TERMINAL_KINDS = frozenset(TERMINAL_EVENT_KINDS)
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,79}\Z")
_EXECUTABLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\Z")
_EXECUTABLE_PATH = re.compile(r"(?:/[A-Za-z0-9._+-]+){1,32}\Z")
# The shells a per-directory router or version-manager shim is written in. A resolved agent
# executable that is one of these is not the agent: it is a program that will re-derive the
# child environment this runtime spent its whole auth layer building.
_SHELL_INTERPRETER_NAMES = frozenset(
    ("ash", "bash", "busybox", "dash", "fish", "ksh", "mksh", "sh", "zsh")
)
_VERSION_PROBE_TIMEOUT_SECONDS = 30.0
_REPORTED_OUTPUT_BOUND = 600
_STRUCTURED_OUTPUT = JsonSchemaAgentOutput(
    name="live_agent_probe",
    schema=parse_canonical_schema(
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
    ),
)


@dataclass(frozen=True, slots=True)
class LiveRoute:
    backend: Backend
    transport: AgentTransport
    required: bool

    @property
    def name(self) -> str:
        return f"{self.backend}:{self.transport}"

    @property
    def env_prefix(self) -> str:
        return f"LLM_RUNTIME_LIVE_{self.backend}_{self.transport}".upper()


@dataclass(frozen=True, slots=True)
class TurnProof:
    model_request_exercised: bool
    model_native_value_reported: bool
    model_effective_behavior_proved: bool
    model_clamp_observed: bool
    model_clamp_diagnostic_proved: bool
    reasoning_request_exercised: bool
    reasoning_native_value_reported: bool
    reasoning_effective_behavior_proved: bool
    reasoning_clamp_observed: bool
    reasoning_clamp_diagnostic_proved: bool
    strict_output_request_exercised: bool
    strict_output_effective_behavior_proved: bool
    max_turns_request_exercised: bool
    max_turns_native_count_reported: bool
    max_turns_effective_behavior_proved: bool


_ROUTES: tuple[LiveRoute, ...] = (
    LiveRoute("codex", "sdk", True),
    LiveRoute("claude", "sdk", True),
)
_ROUTE_BY_NAME = {route.name: route for route in _ROUTES}
# Every shipped lane is required, so the release set is the whole table. There is no longer a
# lane that carries a named API-key credential to add to it — see the module docstring.
_DEFAULT_ROUTES = frozenset(route for route in _ROUTES if route.required)


@dataclass(frozen=True, slots=True, repr=False)
class LiveAgentEnvironment:
    state_root_base: Path
    local_profile: str
    selected_routes: frozenset[LiveRoute]
    # For a selected Claude lane this is the absolute path the preflight resolved and vetted,
    # so the SDK spawns the exact file that was checked rather than re-resolving a name.
    # An unselected Claude lane keeps its raw selector and its adapter is never reached.
    claude_executable: str

    def select(self, route: LiveRoute) -> None:
        if route not in self.selected_routes:
            pytest.skip("agent route was not selected for this live run")

    def runtime_config(
        self,
        *,
        secret_resolver: Callable[[str], Awaitable[str]] | None = None,
    ) -> AgentRuntimeConfig:
        return AgentRuntimeConfig(
            state_root_base=self.state_root_base,
            claude_executable=self.claude_executable,
            max_turn_seconds=75.0,
            secret_resolver=secret_resolver,
        )

    def local_auth(self) -> CredentialRef:
        return CredentialRef(kind="local_account", profile_key=self.local_profile)


@pytest.fixture(scope="session")
def live_agent_environment() -> LiveAgentEnvironment:
    if os.environ.get("LLM_RUNTIME_LIVE") != "1":
        _fail("Set LLM_RUNTIME_LIVE=1 to run the live agent matrix")

    state_root_raw = os.environ.get("LLM_RUNTIME_LIVE_AGENT_STATE_ROOT_BASE")
    if not state_root_raw:
        _fail("live agent state-root base is required")
    state_root = Path(state_root_raw)
    try:
        resolved_state_root = state_root.resolve(strict=True)
    except OSError:
        _fail("live agent state-root base must already exist")
    _require(
        state_root.is_absolute()
        and resolved_state_root == state_root
        and resolved_state_root.is_dir(),
        "live agent state-root base must be an existing resolved absolute directory",
    )
    # AgentRuntime refuses a group- or world-writable base (another local uid could swap a
    # profile root for a symlink between the check and the launch). Report it here, where the
    # operator can still fix the directory, rather than as a late InvalidAgentRequest from the
    # first capability call. A shared temporary directory such as /tmp (mode 1777) fails.
    try:
        base_mode = stat.S_IMODE(resolved_state_root.stat().st_mode)
    except OSError:
        _fail("live agent state-root base could not be inspected")
    _require(
        not base_mode & 0o022,
        "live agent state-root base must not be group- or world-writable; AgentRuntime "
        "rejects such a base, so a shared directory such as /tmp cannot be used",
    )

    local_profile = _required_profile("LLM_RUNTIME_LIVE_AGENT_PROFILE")
    selected_routes = _selected_routes()
    selected_backends: set[Backend] = {route.backend for route in selected_routes}
    for backend in selected_backends:
        _require_existing_profile_root(resolved_state_root, backend, local_profile)

    claude_selector = _executable_selector("claude", "LLM_RUNTIME_LIVE_CLAUDE_EXECUTABLE")
    if "claude" in selected_backends:
        auth = CredentialRef(kind="local_account", profile_key=local_profile)
        claude_executable = _preflight_executable(
            "claude", claude_selector, resolved_state_root, local_profile, auth
        )
    else:
        # A narrowed Codex run must not demand a Claude installation it will never touch.
        claude_executable = claude_selector
    return LiveAgentEnvironment(
        state_root_base=resolved_state_root,
        local_profile=local_profile,
        selected_routes=selected_routes,
        claude_executable=claude_executable,
    )


def _executable_selector(backend: Literal["claude"], environment_name: str) -> str:
    selector = os.environ.get(environment_name, backend)
    _require(
        _EXECUTABLE_NAME.fullmatch(selector) is not None
        or _EXECUTABLE_PATH.fullmatch(selector) is not None,
        f"{environment_name} must be a bare executable name or an absolute path",
    )
    return selector


def _preflight_executable(
    backend: Literal["claude"],
    selector: str,
    state_root_base: Path,
    profile: str,
    auth: CredentialRef,
) -> str:
    """Resolve one backend's executable the way `AgentRuntime` does, then require the real thing.

    `runtime._resolve_executable` is `shutil.which` followed by `Path.resolve()`, so the child
    the runtime spawns is the final symlink target of whatever the operator's `PATH` finds
    first. Reproducing exactly that here is the point: the failure has to name the file the
    runtime would really have launched, not the name the operator typed.
    """
    found = shutil.which(selector)
    if found is None:
        _fail(
            f"the live {backend} executable {selector!r} is not resolvable on this PATH; set "
            f"LLM_RUNTIME_LIVE_{backend.upper()}_EXECUTABLE to an absolute path"
        )
    resolved = Path(found).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        _fail(
            f"the live {backend} executable {selector!r} resolves to {resolved}, which is not "
            "an executable file"
        )
    _reject_shell_wrapper(backend, selector, found, resolved)
    _require_pinned_version(backend, selector, resolved, state_root_base, profile, auth)
    return str(resolved)


def _shebang_words(path: Path) -> tuple[str, ...] | None:
    """The interpreter line of a script, `()` for a native executable, `None` if unreadable."""
    try:
        with path.open("rb") as handle:
            head = handle.read(256)
    except OSError:
        return None
    if not head.startswith(b"#!"):
        return ()
    line = head.split(b"\n", 1)[0][2:].decode("utf-8", errors="replace").strip()
    return tuple(word for word in line.split() if word)


def _shell_interpreter(words: tuple[str, ...]) -> str | None:
    """The POSIX shell a shebang names, or `None` for a native binary or any other interpreter."""
    names = [Path(word).name for word in words]
    if names and names[0] == "env":
        names = [name for name in names[1:] if not name.startswith("-")]
    if names and names[0] in _SHELL_INTERPRETER_NAMES:
        return names[0]
    return None


def _unwrapped_candidates(selector: str) -> tuple[str, ...]:
    """Every later `PATH` match for a bare `selector` that is not itself a shell script.

    `shutil.which` returns the first match and stops, which on a machine with a router is the
    wrapper — but the real install is usually still further along the same `PATH` (a `~/bin`
    router shadowing an npm or native install under `~/.local/bin` is the shape that produced
    this check). Naming the file the operator should point the override at is the difference
    between a diagnosis and a task. These are suggestions: the operator still sets the
    override, and the suggested path is re-vetted from scratch when they do.
    """
    if _EXECUTABLE_NAME.fullmatch(selector) is None:
        return ()
    candidates: list[str] = []
    for entry in os.get_exec_path():
        candidate = Path(entry) / selector
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            continue
        words = _shebang_words(resolved)
        if words is None or _shell_interpreter(words) is not None:
            continue
        target = str(resolved)
        if target not in candidates:
            candidates.append(target)
    return tuple(candidates)


def _reject_shell_wrapper(backend: Backend, selector: str, found: str, resolved: Path) -> None:
    """Refuse a resolved executable that is a shell script standing in front of the real agent.

    A script is not disqualifying by itself: an npm-installed agent's entry point is
    `#!/usr/bin/env node`, and that one is the real program. A *shell* script in this position
    is the router/shim shape, and it is disqualifying for a reason a version check cannot
    catch — such a wrapper re-derives `CODEX_HOME` / `CLAUDE_CONFIG_DIR` from its own rules and
    exports them over the ones the adapter set, so the certified session would run against the
    operator's ambient profile rather than the isolated state root this matrix exists to prove.
    It also puts a shell between the runtime and the agent in every process-group and teardown
    assertion the run makes.

    The failure has to be self-service, because the version probe that runs next cannot
    diagnose this: a router dispatching on its own `argv[0]` sees the resolved name rather than
    the symlink's, so it exits with something like `Unsupported AI router command name:
    ai-router` and says nothing about state roots at all.
    """
    words = _shebang_words(resolved)
    if words is None:
        _fail(f"the live {backend} executable at {resolved} could not be read")
    interpreter = _shell_interpreter(words)
    if interpreter is None:
        return
    candidates = _unwrapped_candidates(selector)
    suggestion = (
        f" The real {backend} program on this PATH looks like {', '.join(candidates)}."
        if candidates
        else ""
    )
    _fail(
        f"the live {backend} executable {selector!r} resolves through {found} to {resolved}, "
        f"which is a {interpreter} script rather than the {backend} program. A shell wrapper in "
        "this position typically exports its own CODEX_HOME/CLAUDE_CONFIG_DIR over the ones "
        "the adapter set, so this run would certify the wrapper's state root instead of the "
        f"runtime's isolated profile root. Point LLM_RUNTIME_LIVE_{backend.upper()}_EXECUTABLE "
        f"at the real executable.{suggestion}"
    )


def _require_pinned_version(
    backend: Literal["claude"],
    selector: str,
    resolved: Path,
    state_root_base: Path,
    profile: str,
    auth: CredentialRef,
) -> None:
    """Run `--version` in the runtime's own child environment and require the pinned build.

    The child environment is the real one `auth.build_child_environment` produces, not the
    operator's shell, so this also proves the vetted child `PATH` can reach whatever the
    install needs. An npm-installed agent is a `#!/usr/bin/env node` script and dies with
    `env: 'node': No such file or directory` here when `node` lives only under a version
    manager — a failure that would otherwise surface much later as an opaque startup error.
    """
    state_root = resolve_state_root(state_root_base, backend, profile)
    environment = build_child_environment(
        AuthEnvironmentRequest(
            backend=backend,
            credential=auth,
            inherited_environment=dict(os.environ),
            allowed_environment=(),
            state_root=state_root,
        )
    )
    try:
        completed = subprocess.run(
            (str(resolved), "--version"),
            cwd=state_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=_VERSION_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _fail(f"the live {backend} executable at {resolved} did not report a version in time")
    except OSError as error:
        _fail(f"the live {backend} executable at {resolved} could not be started: {error}")
    if completed.returncode != 0:
        _fail(
            f"the live {backend} executable at {resolved} exited "
            f"{completed.returncode} for --version inside the runtime's own child environment. "
            f"stderr: {completed.stderr[:_REPORTED_OUTPUT_BOUND]!r}"
        )
    pinned = claude_sdk._SUPPORTED_CLAUDE_VERSION
    boundary = re.compile(rf"(?<![0-9A-Za-z._+-]){re.escape(pinned)}(?![0-9A-Za-z._+-])")
    _require(
        boundary.search(completed.stdout) is not None,
        f"the live {backend} executable at {resolved} does not report the pinned version "
        f"{pinned}; the adapter refuses every other build. Reported: "
        f"{completed.stdout[:_REPORTED_OUTPUT_BOUND]!r}",
    )


def _selected_routes() -> frozenset[LiveRoute]:
    raw = os.environ.get("LLM_RUNTIME_LIVE_AGENT_ROUTES")
    if raw is None:
        return _DEFAULT_ROUTES
    names = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    _require(bool(names), "live agent route selection must not be empty")
    _require(len(names) == len(set(names)), "live agent route selection contains duplicates")
    _require(
        all(name in _ROUTE_BY_NAME for name in names),
        "live agent route selection contains an unsupported route",
    )
    return frozenset(_ROUTE_BY_NAME[name] for name in names)


def _required_profile(environment_name: str) -> str:
    value = os.environ.get(environment_name)
    if not value:
        _fail(f"{environment_name} is required")
    try:
        CredentialRef(kind="local_account", profile_key=value)
    except InvalidAgentRequest:
        _fail(f"{environment_name} must be a valid profile key")
    return value


def _require_existing_profile_root(base: Path, backend: Backend, profile: str) -> None:
    root = base / backend / profile
    try:
        resolved = root.resolve(strict=True)
    except OSError:
        _fail("every selected live agent profile root must already exist")
    _require(
        resolved == root and resolved.is_dir(),
        "every selected live agent profile root must be an existing resolved directory",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _fail(message: str) -> Never:
    pytest.fail(message, pytrace=False)


def _scope(route: LiveRoute, auth: CredentialRef) -> AgentCapabilityScope:
    return AgentCapabilityScope(
        backend=route.backend,
        transport=route.transport,
        auth=auth,
    )


def _published_native_tools(
    capabilities: AgentCapabilities,
) -> dict[BuiltinToolFamily, tuple[str, ...]]:
    """The route's own exact native tool spellings, per built-in family.

    `builtin_tool_families` is a normalized cross-backend vocabulary; `PermissionPolicy`
    allow/deny entries are exact native names; `builtin_tool_names` is the only bridge
    between them. This gate reads that bridge instead of carrying a vendor table of its own,
    for two reasons: a table checked into a test cannot be kept in sync with whatever
    executable version the route reports, and a matrix that supplies its own names certifies
    nothing about the capability report a consumer actually programs against.

    An empty table is a legitimate report on a route without `tool_controls` — Codex says
    exactly that, and never reaches here. On a route that claims exact tool controls it is a
    gap in the adapter, so it fails the run and names what has to be filled in.
    """
    published = dict(capabilities.builtin_tool_names)
    if not published:
        _fail(
            "route reports tool_controls=True but publishes no "
            "AgentCapabilities.builtin_tool_names, so this matrix has no source for the exact "
            "native tool names its policies must name. Populate builtin_tool_names in the "
            "adapter for the executable version it reports; a hard-coded table in this test "
            "would certify the test's guess instead of the shipped capability report"
        )
    return published


def _native_builtin_tools(capabilities: AgentCapabilities) -> tuple[str, ...]:
    """Every published native built-in name, deduplicated in the capability table's own order."""
    published = _published_native_tools(capabilities)
    return tuple(dict.fromkeys(name for names in published.values() for name in names))


def _safe_policy(capabilities: AgentCapabilities) -> PermissionPolicy:
    _require("read_only" in capabilities.filesystem_modes, "route lacks read-only policy")
    _require("disabled" in capabilities.network_modes, "route lacks disabled-network policy")
    _require("deny" in capabilities.approval_modes, "route lacks deny-by-default approvals")
    allowed_tools = (
        () if capabilities.tool_controls or not capabilities.builtin_tool_families else ("*",)
    )
    return PermissionPolicy(
        # The baseline turn only needs read access. Prefer restricted writes where the host
        # can actually create the provider's sandbox namespace; the dedicated workspace
        # probe below remains responsible for certifying writes when they are advertised.
        filesystem=(
            "workspace_write" if "workspace_write" in capabilities.filesystem_modes else "read_only"
        ),
        network="disabled",
        approval="deny",
        allowed_tools=allowed_tools,
    )


def _model_reasoning_cases(
    route: LiveRoute, capabilities: AgentCapabilities
) -> tuple[ModelReasoningCase, ...]:
    try:
        if route.backend == "codex":
            return codex_model_reasoning_cases(
                capabilities.models, capabilities.model_reasoning_efforts
            )
        models = parse_claude_models(os.environ.get(f"{route.env_prefix}_MODELS"))
        return claude_model_reasoning_cases(models, capabilities.reasoning_efforts)
    except MatrixSelectionError as error:
        _fail(str(error))


def _output_for(capabilities: AgentCapabilities) -> AgentOutputSpec:
    if capabilities.structured_output and capabilities.native_output_schema:
        return _STRUCTURED_OUTPUT
    return TextAgentOutput()


def _turn_prompt(output: AgentOutputSpec) -> str:
    if isinstance(output, JsonSchemaAgentOutput):
        return "Return only one JSON object matching the supplied schema, with ok set to true."
    return "Reply with exactly the single word READY."


def _session_request(
    route: LiveRoute,
    auth: CredentialRef,
    cwd: Path,
    capabilities: AgentCapabilities,
    selection: ModelReasoningCase,
    *,
    open: SessionOpen | None = None,
    policy: PermissionPolicy | None = None,
    mcp_servers: tuple[McpServerSpec, ...] = (),
    output: AgentOutputSpec | None = None,
) -> AgentSessionRequest:
    return AgentSessionRequest(
        backend=route.backend,
        transport=route.transport,
        auth=auth,
        open=open or NewSession(),
        cwd=str(cwd.resolve()),
        policy=policy or _safe_policy(capabilities),
        model=selection.model,
        reasoning=(
            ReasoningSpec(effort=selection.reasoning_effort)
            if selection.reasoning_effort is not None
            else None
        ),
        mcp_servers=mcp_servers,
        output=output or _output_for(capabilities),
    )


# `names` is an ORDERED priority list, never a set: certification evidence must be
# reproducible, and set iteration order over strings varies with hash randomization, so a
# payload carrying two accepted aliases would otherwise certify a different value per run.
def _find_native_value(value: object, names: tuple[str, ...], *, depth: int = 0) -> object:
    if depth > 4:
        return None
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        for child in value.values():
            found = _find_native_value(child, names, depth=depth + 1)
            if found is not None:
                return found
    elif isinstance(value, tuple):
        for child in value:
            found = _find_native_value(child, names, depth=depth + 1)
            if found is not None:
                return found
    return None


def _reported_selections(events: list[AgentEvent], names: tuple[str, ...]) -> tuple[str, ...]:
    reported: list[str] = []
    for event in events:
        if event.native_payload is None:
            continue
        value = _find_native_value(event.native_payload, names)
        if isinstance(value, str) and value and value not in reported:
            reported.append(value)
    return tuple(reported)


def _effective_change_diagnostic(events: list[AgentEvent], subject: str) -> bool:
    subject_markers = (subject,) if subject == "model" else ("reason", "effort", "thinking")
    for event in events:
        if not isinstance(event.data, DiagnosticData):
            continue
        code = event.data.code.lower()
        if (
            "effective" in code
            and any(marker in code for marker in subject_markers)
            and any(marker in code for marker in ("changed", "clamp", "mismatch", "substitut"))
        ):
            return True
    return False


def _selection_dimension_proof(
    events: list[AgentEvent],
    *,
    requested: str | None,
    native_names: tuple[str, ...],
    subject: str,
) -> tuple[bool, bool, bool, bool, bool]:
    request_exercised = requested is not None
    if requested is None:
        return False, False, False, False, False
    reported = _reported_selections(events, native_names)
    native_reported = bool(reported)
    clamp_observed = any(value != requested for value in reported)
    diagnostic_proved = _effective_change_diagnostic(events, subject)
    if clamp_observed:
        _require(
            diagnostic_proved,
            f"native output reported an effective {subject} change without a diagnostic",
        )
    effective_behavior_proved = native_reported and (not clamp_observed or diagnostic_proved)
    return (
        request_exercised,
        native_reported,
        effective_behavior_proved,
        clamp_observed,
        clamp_observed and diagnostic_proved,
    )


def _turn_proof(
    events: list[AgentEvent],
    request: AgentSessionRequest,
    *,
    max_turns: int | None,
    strict_output_proved: bool,
) -> TurnProof:
    model = _selection_dimension_proof(
        events,
        requested=request.model,
        native_names=("model", "model_name", "modelName"),
        subject="model",
    )
    reasoning = _selection_dimension_proof(
        events,
        requested=request.reasoning.effort if request.reasoning is not None else None,
        native_names=("effort", "reasoning_effort", "reasoningEffort"),
        subject="reasoning",
    )
    reported_turn_count: int | None = None
    if max_turns is not None:
        for event in reversed(events):
            if event.native_payload is None:
                continue
            candidate = _find_native_value(
                event.native_payload,
                ("num_turns", "turn_count", "turnCount"),
            )
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                reported_turn_count = candidate
                break
        if reported_turn_count is not None:
            _require(
                0 < reported_turn_count <= max_turns,
                "native output reported a max_turns bound violation",
            )
    return TurnProof(
        model_request_exercised=model[0],
        model_native_value_reported=model[1],
        model_effective_behavior_proved=model[2],
        model_clamp_observed=model[3],
        model_clamp_diagnostic_proved=model[4],
        reasoning_request_exercised=reasoning[0],
        reasoning_native_value_reported=reasoning[1],
        reasoning_effective_behavior_proved=reasoning[2],
        reasoning_clamp_observed=reasoning[3],
        reasoning_clamp_diagnostic_proved=reasoning[4],
        strict_output_request_exercised=isinstance(request.output, JsonSchemaAgentOutput),
        strict_output_effective_behavior_proved=strict_output_proved,
        max_turns_request_exercised=max_turns is not None,
        max_turns_native_count_reported=reported_turn_count is not None,
        max_turns_effective_behavior_proved=reported_turn_count is not None,
    )


async def _first_streamed_turn(
    runtime: AgentRuntime,
    request: AgentSessionRequest,
    capabilities: AgentCapabilities,
) -> tuple[AgentSession, AgentSessionRef, TurnProof]:
    session = await runtime.open_session(request)
    max_turns = 1 if capabilities.max_turns else None
    events: list[AgentEvent] = []
    async for event in runtime.stream_turn(
        session,
        TurnRequest(
            input=(TextContent(_turn_prompt(request.output)),),
            max_turns=max_turns,
            timeout_seconds=60.0,
        ),
    ):
        if not events:
            _require(event.kind == "session_started", "new live turn lacked session_started first")
            _require(session.ref_is_complete, "session ref was incomplete at the first event")
            _require(
                session.ref == event.session_ref, "first event did not complete the session ref"
            )
        events.append(event)
    _require(bool(events), "new live turn emitted no events")
    _require(
        tuple(event.seq for event in events) == tuple(range(1, len(events) + 1)),
        "live stream sequence was not monotonic and gap-free",
    )
    _require(
        sum(event.kind in _TERMINAL_KINDS for event in events) == 1,
        "live stream did not contain exactly one terminal event",
    )
    _require(
        events[-1].kind == "turn_completed",
        "new live turn did not complete; "
        f"terminal_kind={events[-1].kind!r}, terminal_data={events[-1].data!r}",
    )
    terminal_data = events[-1].data
    if not isinstance(terminal_data, TurnCompletedData):
        _fail("successful live terminal had the wrong data variant")
    strict_output_proved = False
    if isinstance(request.output, JsonSchemaAgentOutput):
        structured = terminal_data.structured_output
        if not isinstance(structured, Mapping):
            _fail("strict output was not a JSON object")
        _require(structured.get("ok") is True, "strict output did not satisfy its required value")
        strict_output_proved = True
    else:
        _require(
            any(event.kind == "text_delta" for event in events), "live turn did not stream text"
        )
        _require(bool(terminal_data.final_text), "live text result was empty")
    return (
        session,
        session.ref,
        _turn_proof(
            events,
            request,
            max_turns=max_turns,
            strict_output_proved=strict_output_proved,
        ),
    )


def _require_model_reasoning_case_proof(
    case: ModelReasoningCase,
    capabilities: AgentCapabilities,
    proof: TurnProof,
) -> None:
    _require(proof.model_request_exercised, "matrix turn did not exercise its model selection")
    _require(
        proof.model_effective_behavior_proved,
        "matrix turn did not prove its effective model selection",
    )
    _require(
        proof.reasoning_request_exercised == (case.reasoning_effort is not None),
        "matrix turn did not exercise its declared reasoning selection",
    )
    if case.reasoning_effort is not None and capabilities.reports_effective_effort:
        _require(
            proof.reasoning_effective_behavior_proved,
            "matrix turn did not prove its effective reasoning selection",
        )


async def _probe_model_reasoning_matrix(
    runtime: AgentRuntime,
    route: LiveRoute,
    auth: CredentialRef,
    cwd: Path,
    capabilities: AgentCapabilities,
    cases: tuple[ModelReasoningCase, ...],
    first_proof: TurnProof,
) -> bool:
    _require(bool(cases), "live model/reasoning matrix was empty")
    _require_model_reasoning_case_proof(cases[0], capabilities, first_proof)
    for case in cases[1:]:
        request = _session_request(
            route,
            auth,
            cwd,
            capabilities,
            case,
            output=TextAgentOutput(),
        )
        session, _ref, proof = await _first_streamed_turn(runtime, request, capabilities)
        try:
            _require_model_reasoning_case_proof(case, capabilities, proof)
        finally:
            await runtime.close_session(session)
    return True


def _later_turn_request(
    base: AgentSessionRequest,
    capabilities: AgentCapabilities,
) -> TurnRequest:
    return TurnRequest(
        input=(TextContent(_turn_prompt(base.output)),),
        model=base.model if "model" in capabilities.turn_overrides else None,
        reasoning=base.reasoning if "reasoning" in capabilities.turn_overrides else None,
        policy=PermissionPolicyPatch() if "policy" in capabilities.turn_overrides else None,
        timeout_seconds=60.0,
    )


def _require_success(result: AgentResult, message: str) -> None:
    _require(result.status == "succeeded", message)


async def _probe_session_operations(
    runtime: AgentRuntime,
    base: AgentSessionRequest,
    source_ref: AgentSessionRef,
    capabilities: AgentCapabilities,
) -> tuple[bool, bool]:
    if base.model is None:
        _fail("matrix base request had no model")
    selection = ModelReasoningCase(
        base.model,
        base.reasoning.effort if base.reasoning is not None else None,
    )
    resumed = False
    forked = False
    if "resume" in capabilities.session_operations:
        session = await runtime.open_session(
            _session_request(
                LiveRoute(base.backend, base.transport, False),
                base.auth,
                Path(base.cwd),
                capabilities,
                selection,
                open=ResumeSession(source_ref),
                output=base.output,
            )
        )
        result = await runtime.run_turn(session, _later_turn_request(base, capabilities))
        _require_success(result, "reported resume operation did not complete a later turn")
        _require(result.session_ref == source_ref, "resume changed native session identity")
        await runtime.close_session(session)
        resumed = True
    if "fork" in capabilities.session_operations:
        session = await runtime.open_session(
            _session_request(
                LiveRoute(base.backend, base.transport, False),
                base.auth,
                Path(base.cwd),
                capabilities,
                selection,
                open=ForkSession(source_ref),
                output=base.output,
            )
        )
        result = await runtime.run_turn(session, _later_turn_request(base, capabilities))
        _require_success(result, "reported fork operation did not complete a later turn")
        _require(result.session_ref != source_ref, "fork reused native session identity")
        await runtime.close_session(session)
        forked = True
    return resumed, forked


async def _probe_discovery(
    runtime: AgentRuntime,
    scope: AgentCapabilityScope,
    source_ref: AgentSessionRef,
    capabilities: AgentCapabilities,
) -> tuple[bool, bool]:
    listed = False
    read = False
    if "list" in capabilities.discovery_operations:
        page = await runtime.list_sessions(SessionQuery(scope=scope, limit=100))
        _require(
            any(summary.ref == source_ref for summary in page.sessions),
            "reported session listing did not contain the new session",
        )
        listed = True
    if "read" in capabilities.discovery_operations:
        include_turns = "turn_history" in capabilities.discovery_operations
        include_items = "item_history" in capabilities.discovery_operations
        _require(
            not include_items or include_turns,
            "item-history capability was reported without turn history",
        )
        snapshot = await runtime.read_session(
            source_ref,
            SessionReadOptions(
                auth=scope.auth,
                limit=100,
                include_turns=include_turns,
                include_items=include_items,
            ),
        )
        _require(snapshot.ref == source_ref, "session read changed native session identity")
        read = True
    return listed, read


async def _probe_cancellation(
    runtime: AgentRuntime,
    route: LiveRoute,
    auth: CredentialRef,
    cwd: Path,
    capabilities: AgentCapabilities,
    selection: ModelReasoningCase,
) -> bool:
    if not capabilities.cancellation:
        return False
    session = await runtime.open_session(
        _session_request(
            route,
            auth,
            cwd,
            capabilities,
            selection,
            output=TextAgentOutput(),
        )
    )
    cancel = asyncio.Event()
    kinds: list[str] = []
    async for event in runtime.stream_turn(
        session,
        TurnRequest(
            input=(
                TextContent(
                    "Write the integers from one through ten thousand, one per line, "
                    "without using tools."
                ),
            ),
            timeout_seconds=45.0,
        ),
        cancel=cancel,
    ):
        kinds.append(event.kind)
        if event.kind == "turn_started":
            cancel.set()
    await runtime.close_session(session)
    _require("turn_started" in kinds, "cancellation was not armed after turn_started")
    _require(kinds[-1:] == ["turn_cancelled"], "post-start cancellation did not terminate cleanly")
    _require(
        sum(kind in _TERMINAL_KINDS for kind in kinds) == 1,
        "cancelled stream did not contain exactly one terminal",
    )
    return True


async def _probe_mcp(
    runtime: AgentRuntime,
    route: LiveRoute,
    auth: CredentialRef,
    cwd: Path,
    capabilities: AgentCapabilities,
    selection: ModelReasoningCase,
) -> tuple[bool, bool]:
    if "stdio" not in capabilities.mcp_transports:
        return False, False
    _require(_MCP_FIXTURE.is_file(), "live MCP fixture is unavailable")
    server_name = "live_certifier"
    native_tool_name = f"mcp__{server_name}__live_probe"
    approval_expected = capabilities.tool_controls and "ask" in capabilities.approval_modes
    if capabilities.tool_controls:
        allowed_tools = (native_tool_name,)
        denied_tools = _native_builtin_tools(capabilities)
    else:
        allowed_tools = ("*",) if capabilities.builtin_tool_families else ()
        denied_tools = ()
    policy = PermissionPolicy(
        filesystem="full_access",
        network="unrestricted",
        approval="ask" if approval_expected else "deny",
        allowed_tools=allowed_tools,
        denied_tools=denied_tools,
        unsafe_confirmation=UnsafeConfirmation(("filesystem_full_access", "network_unrestricted")),
    )
    server = McpServerSpec(
        name=server_name,
        transport="stdio",
        command=str(Path(sys.executable).resolve()),
        args=(str(_MCP_FIXTURE),),
        allowed_tools=("live_probe",),
    )
    approval_calls = 0

    async def approve(_request: ApprovalRequest) -> ApprovalDecision:
        nonlocal approval_calls
        approval_calls += 1
        return "allow"

    handler: ApprovalHandler | None = approve if approval_expected else None
    session = await runtime.open_session(
        _session_request(
            route,
            auth,
            cwd,
            capabilities,
            selection,
            policy=policy,
            mcp_servers=(server,),
            output=TextAgentOutput(),
        )
    )
    events = [
        event
        async for event in runtime.stream_turn(
            session,
            TurnRequest(
                input=(
                    TextContent(
                        "Call the live_probe MCP tool exactly once, call no other tool, "
                        "then repeat the tool's exact marker and nothing else."
                    ),
                ),
                timeout_seconds=60.0,
            ),
            approvals=handler,
        )
    ]
    _require(bool(events), "reported MCP path emitted no events")
    _require(events[-1].kind == "turn_completed", "reported MCP path did not complete")
    started = [event for event in events if event.kind == "tool_started"]
    completed = [event for event in events if event.kind == "tool_completed"]
    _require(bool(started), "reported MCP path emitted no tool_started event")
    _require(bool(completed), "reported MCP path emitted no tool_completed event")
    _require(
        any(
            isinstance(event.data, ToolCompletedData) and event.data.succeeded
            for event in completed
        ),
        "reported MCP tool did not complete successfully",
    )
    terminal_data = events[-1].data
    if not isinstance(terminal_data, TurnCompletedData):
        _fail("MCP terminal data was malformed")
    _require(
        "AGENT_RUNTIME_MCP_LIVE_OK" in terminal_data.final_text,
        "live MCP marker was not returned through the agent",
    )
    approval_certified = False
    if approval_expected:
        approval_certified = _approval_roundtrip_observed(events, approval_calls)
    await runtime.close_session(session)
    return True, approval_certified


def _approval_roundtrip_observed(events: list[AgentEvent], calls: int) -> bool:
    requested = [index for index, event in enumerate(events) if event.kind == "approval_requested"]
    answered = [index for index, event in enumerate(events) if event.kind == "approval_answered"]
    _require(calls > 0, "approval handler was never called")
    _require(bool(requested) and bool(answered), "approval events were not both emitted")
    _require(requested[0] < answered[0], "approval answer preceded its request")
    _require(
        any(
            isinstance(event.data, ApprovalAnsweredData) and event.data.decision == "allow"
            for event in events
        ),
        "approval allow decision was not observed",
    )
    return True


async def _probe_workspace_write(
    runtime: AgentRuntime,
    route: LiveRoute,
    auth: CredentialRef,
    cwd: Path,
    capabilities: AgentCapabilities,
    selection: ModelReasoningCase,
) -> tuple[bool, bool, bool]:
    if "workspace_write" not in capabilities.filesystem_modes:
        return False, False, False
    approval_expected = "ask" in capabilities.approval_modes
    if capabilities.tool_controls:
        _require(
            "file_write" in capabilities.builtin_tool_families,
            "route reports workspace_write but no writable built-in tool family",
        )
        write_tools = _published_native_tools(capabilities).get("file_write", ())
        if not write_tools:
            _fail(
                "route reports the file_write family with exact tool controls but publishes no "
                "native name for it, so no policy can allow exactly the writing tool"
            )
        allowed_tools = write_tools
        denied_tools = tuple(
            tool for tool in _native_builtin_tools(capabilities) if tool not in set(write_tools)
        )
    else:
        allowed_tools = ("*",) if capabilities.builtin_tool_families else ()
        denied_tools = ()
    policy = PermissionPolicy(
        filesystem="workspace_write",
        network="disabled",
        approval="ask" if approval_expected else "deny",
        allowed_tools=allowed_tools,
        denied_tools=denied_tools,
    )
    approval_calls = 0
    marker = cwd / "workspace_write_probe.txt"
    write_prompt = (
        "Use the native file-editing tool to create the relative path "
        f"workspace_write_probe.txt under the configured working directory {cwd}. "
        "Do not use an absolute path or any other directory. The file must contain exactly "
        "WORKSPACE_WRITE_OK."
        if route.backend == "codex"
        else f"Use the native file-editing tool to create {marker} containing exactly "
        "WORKSPACE_WRITE_OK."
    )

    async def approve(_request: ApprovalRequest) -> ApprovalDecision:
        nonlocal approval_calls
        approval_calls += 1
        return "allow"

    session = await runtime.open_session(
        _session_request(
            route,
            auth,
            cwd,
            capabilities,
            selection,
            policy=policy,
            output=TextAgentOutput(),
        )
    )
    events = [
        event
        async for event in runtime.stream_turn(
            session,
            TurnRequest(
                input=(TextContent(write_prompt),),
                timeout_seconds=60.0,
            ),
            approvals=approve if approval_expected else None,
        )
    ]
    _require(bool(events), "workspace_write probe emitted no events")
    _require(events[-1].kind == "turn_completed", "workspace_write probe did not complete")
    _require(
        marker.is_file(),
        "workspace_write did not create its bounded temporary file; "
        f"event_kinds={tuple(event.kind for event in events)!r}, "
        f"file_changes={tuple(event.data for event in events if event.kind == 'file_change')!r}, "
        f"terminal_data={events[-1].data!r}",
    )
    _require(marker.stat().st_size <= 1_024, "workspace_write file exceeded its test bound")
    _require(
        "WORKSPACE_WRITE_OK" in marker.read_text(encoding="utf-8"),
        "workspace_write file did not contain its fixed marker",
    )
    approval_proved = (
        _approval_roundtrip_observed(events, approval_calls) if approval_expected else False
    )
    await runtime.close_session(session)
    return True, True, approval_proved


async def _probe_network_allowlist(
    runtime: AgentRuntime,
    route: LiveRoute,
    auth: CredentialRef,
    cwd: Path,
    capabilities: AgentCapabilities,
    selection: ModelReasoningCase,
) -> tuple[bool, bool]:
    advertised = "allowlist" in capabilities.network_modes or capabilities.network_allowlist
    if not advertised:
        return False, False
    _require(
        "allowlist" in capabilities.network_modes and capabilities.network_allowlist,
        "route reports an incomplete exact network allowlist capability",
    )
    allowed_domain = "example.com"
    if capabilities.tool_controls:
        command_tools = _published_native_tools(capabilities).get("command", ())
        _require(bool(command_tools), "network allowlist proof requires a native command tool")
        allowed_tools = command_tools
        denied_tools = tuple(
            tool for tool in _native_builtin_tools(capabilities) if tool not in command_tools
        )
    else:
        allowed_tools = ("*",) if capabilities.builtin_tool_families else ()
        denied_tools = ()
    policy = PermissionPolicy(
        filesystem="workspace_write",
        network="allowlist",
        approval="ask" if "ask" in capabilities.approval_modes else "deny",
        network_allowlist=(allowed_domain,),
        allowed_tools=allowed_tools,
        denied_tools=denied_tools,
    )
    approval_calls = 0

    async def approve(_request: ApprovalRequest) -> ApprovalDecision:
        nonlocal approval_calls
        approval_calls += 1
        return "allow"

    session = await runtime.open_session(
        _session_request(
            route,
            auth,
            cwd,
            capabilities,
            selection,
            policy=policy,
            output=TextAgentOutput(),
        )
    )
    events = [
        event
        async for event in runtime.stream_turn(
            session,
            TurnRequest(
                input=(
                    TextContent(
                        "Use the Bash tool exactly once to run this command verbatim, then report "
                        "its result: curl --fail --silent --show-error --max-time 10 "
                        "https://example.com >/dev/null && ! curl --fail --silent --show-error "
                        "--max-time 10 https://example.org >/dev/null"
                    ),
                ),
                timeout_seconds=60.0,
            ),
            approvals=approve if "ask" in capabilities.approval_modes else None,
        )
    ]
    _require(bool(events), "network allowlist probe emitted no events")
    _require(events[-1].kind == "turn_completed", "network allowlist probe did not complete")
    reported: object = None
    for event in events:
        if event.native_payload is None:
            continue
        reported = _find_native_value(
            event.native_payload,
            ("allowedDomains", "allowed_domains", "network_allowlist"),
        )
        if reported is not None:
            break
    effective_behavior_proved = False
    if reported is not None:
        _require(
            isinstance(reported, tuple) and reported == (allowed_domain,),
            "native output reported a different effective network allowlist",
        )
        effective_behavior_proved = True
    completed_tools = [event.data for event in events if isinstance(event.data, ToolCompletedData)]
    if completed_tools:
        _require(
            any(tool.succeeded for tool in completed_tools),
            f"network allowlist behavioral probe command failed; tool_results={completed_tools!r}",
        )
        effective_behavior_proved = True
    if "ask" in capabilities.approval_modes:
        _approval_roundtrip_observed(events, approval_calls)
    await runtime.close_session(session)
    return True, effective_behavior_proved


async def _probe_unsupported_named_auth(
    environment: LiveAgentEnvironment,
    route: LiveRoute,
    monkeypatch: pytest.MonkeyPatch,
) -> bool:
    """Every shipped lane must refuse both named credential kinds, before reading a secret.

    While `claude:cli` shipped, this was the negative half of a pair whose positive half
    certified that named auth worked somewhere. That lane is gone, so this is now the whole
    story for `api_key_environment` and `secret_reference` on the agent surface: a release is
    certified on the refusal, and the refusal must land as `UnsupportedCapability` without the
    secret resolver ever being called.
    """
    unique = uuid.uuid4().hex[:16]
    api_auth = CredentialRef(
        kind="api_key_environment",
        profile_key=f"preflight-api-{unique}",
        name="OPENAI_API_KEY" if route.backend == "codex" else "ANTHROPIC_API_KEY",
    )
    secret_auth = CredentialRef(
        kind="secret_reference",
        profile_key=f"preflight-secret-{unique}",
        name="live-agent-secret",
    )
    resolver_called = False

    async def forbidden_resolver(_name: str) -> str:
        nonlocal resolver_called
        resolver_called = True
        raise CredentialUnavailable("unsupported-auth preflight reached secret resolution")

    async def expect_rejection(auth: CredentialRef) -> None:
        try:
            async with AgentRuntime(
                environment.runtime_config(secret_resolver=forbidden_resolver)
            ) as runtime:
                await runtime.capabilities(_scope(route, auth))
        except UnsupportedCapability:
            return
        except Exception:
            _fail("unsupported named auth did not fail as UnsupportedCapability")
        _fail("unsupported named auth was accepted")

    api_environment_name = "OPENAI_API_KEY" if route.backend == "codex" else "ANTHROPIC_API_KEY"
    with monkeypatch.context() as isolated:
        isolated.delenv(api_environment_name, raising=False)
        await expect_rejection(api_auth)
    await expect_rejection(secret_auth)
    _require(not resolver_called, "unsupported named auth read a secret before rejection")
    return True


def _safe_version(value: str | None) -> str | None:
    if value is None:
        return None
    return value if _VERSION.fullmatch(value) is not None else "redacted"


def _capability_evidence(capabilities: AgentCapabilities) -> dict[str, object]:
    return {
        "versions": {
            "executable": _safe_version(capabilities.executable_version),
            "sdk": _safe_version(capabilities.sdk_version),
            "native_extension": _safe_version(capabilities.native_extension_version),
        },
        "session_operations": sorted(capabilities.session_operations),
        "discovery_operations": sorted(capabilities.discovery_operations),
        "models_enumerated": capabilities.models is not None,
        "model_count": len(capabilities.models) if capabilities.models is not None else None,
        "reasoning_efforts_enumerated": capabilities.reasoning_efforts is not None,
        "reasoning_effort_count": (
            len(capabilities.reasoning_efforts)
            if capabilities.reasoning_efforts is not None
            else None
        ),
        "streaming": capabilities.streaming,
        "cancellation": capabilities.cancellation,
        "structured_output": capabilities.structured_output,
        "native_output_schema": capabilities.native_output_schema,
        "tool_controls": capabilities.tool_controls,
        "stdio_mcp": "stdio" in capabilities.mcp_transports,
        "approval_ask": "ask" in capabilities.approval_modes,
        "read_only": "read_only" in capabilities.filesystem_modes,
        "workspace_write": "workspace_write" in capabilities.filesystem_modes,
        "network_disabled": "disabled" in capabilities.network_modes,
        "network_allowlist_mode": "allowlist" in capabilities.network_modes,
        "network_allowlist": capabilities.network_allowlist,
        "max_turns": capabilities.max_turns,
        "max_turns_limit": capabilities.max_turns_limit,
        "reports_auth_identity": capabilities.reports_auth_identity,
        "reports_effective_effort": capabilities.reports_effective_effort,
        "builtin_tool_families": sorted(capabilities.builtin_tool_families),
        # Which families the route publishes exact native names for. Family names come from a
        # closed vocabulary, and the count keeps the record useful without copying vendor
        # spellings into evidence.
        "builtin_tool_name_families": sorted(
            family for family, _names in capabilities.builtin_tool_names
        ),
        "builtin_tool_name_count": sum(
            len(names) for _family, names in capabilities.builtin_tool_names
        ),
        "turn_model_override": "model" in capabilities.turn_overrides,
        "turn_reasoning_override": "reasoning" in capabilities.turn_overrides,
        "persistent_model_override": "model" in capabilities.persistent_turn_overrides,
        "persistent_reasoning_override": "reasoning" in capabilities.persistent_turn_overrides,
    }


def _cwd_effective_proved(ref: AgentSessionRef, cwd: Path) -> bool:
    """Prove the native session really adopted the requested working directory.

    A constant in the evidence record certifies nothing: it cannot fail. The session
    ref's ``cwd_fingerprint`` is the one observable the backend round-trips, so compare
    it against the fingerprint of the directory this run actually asked for.
    """
    expected = hashlib.sha256(str(cwd.resolve()).encode("utf-8")).hexdigest()
    return ref.cwd_fingerprint == expected


def _write_evidence(
    route: LiveRoute,
    auth_kind: CredentialKind,
    capabilities: AgentCapabilities,
    matrix_cases: tuple[ModelReasoningCase, ...],
    features: Mapping[str, bool],
) -> None:
    version_present = bool(capabilities.executable_version or capabilities.sdk_version)
    _require(version_present, "selected live route did not report an installed version")
    required_proofs: dict[str, bool] = {
        "cwd_and_fail_closed_policy_request_exercised": True,
        "new_streamed_turn": True,
        "ref_completed_by_first_event": True,
        "complete_model_reasoning_matrix": True,
    }
    if capabilities.cancellation:
        required_proofs["cancellation_after_turn_started"] = True
    if "list" in capabilities.discovery_operations:
        required_proofs["discovery_list"] = True
    if "read" in capabilities.discovery_operations:
        required_proofs["discovery_read"] = True
    if "resume" in capabilities.session_operations:
        required_proofs["resume"] = True
    if "fork" in capabilities.session_operations:
        required_proofs["fork"] = True
    if "stdio" in capabilities.mcp_transports:
        required_proofs["mcp_stdio_tool"] = True
    if "ask" in capabilities.approval_modes:
        required_proofs["approval_roundtrip"] = True
    if "workspace_write" in capabilities.filesystem_modes:
        required_proofs["workspace_write_request_exercised"] = True
        required_proofs["workspace_write_effective_behavior_proved"] = True
    if "allowlist" in capabilities.network_modes or capabilities.network_allowlist:
        required_proofs["network_allowlist_request_exercised"] = True
        required_proofs["network_allowlist_effective_behavior_proved"] = True
    if capabilities.structured_output:
        required_proofs["strict_output_request_exercised"] = True
        required_proofs["strict_output_effective_behavior_proved"] = True
    if capabilities.max_turns:
        required_proofs["max_turns_request_exercised"] = True
        required_proofs["max_turns_effective_behavior_proved"] = True
    if features.get("model_request_exercised", False):
        required_proofs["model_effective_behavior_proved"] = True
    # Only a transport that reports the effort it actually ran with can prove the effective
    # value. Demanding the proof from one that does not (Claude Code 2.1.220's system/init
    # carries no effort member) would make certification permanently unreachable there, and a
    # requirement that can never be met is not a requirement. The capability flag is recorded
    # in the evidence, so a reader can tell "proved" from "unobservable here".
    if features.get("reasoning_request_exercised", False) and capabilities.reports_effective_effort:
        required_proofs["reasoning_effective_behavior_proved"] = True
    missing_proofs = sorted(
        name for name, required in required_proofs.items() if required and not features.get(name)
    )
    payload: dict[str, object] = {
        "schema_version": "agent-runtime-live-evidence.v1",
        "certification_status": "certified" if not missing_proofs else "observational_incomplete",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "route": route.name,
        "auth_kind": auth_kind,
        "capabilities": _capability_evidence(capabilities),
        "feature_evidence": dict(sorted(features.items())),
        "model_reasoning_matrix": {
            "case_count": len(matrix_cases),
            "model_count": len({case.model for case in matrix_cases}),
            "reasoning_effort_count": len(
                {
                    case.reasoning_effort
                    for case in matrix_cases
                    if case.reasoning_effort is not None
                }
            ),
        },
        "missing_required_proofs": missing_proofs,
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    revision = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["evidence_revision"] = revision
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    try:
        _EVIDENCE_DIR.mkdir(exist_ok=True)
        destination = _EVIDENCE_DIR / (
            "agent-runtime-"
            f"{route.backend}-{route.transport}-{auth_kind}-"
            f"{datetime.now(UTC).date().isoformat()}-{revision[:12]}.json"
        )
        destination.write_text(rendered, encoding="utf-8")
    except OSError:
        _fail("sanitized live agent evidence could not be written")
    _require(
        not missing_proofs,
        f"selected route has incomplete certification proof: {missing_proofs}",
    )


async def test_live_codex_sdk_runtime_pair(
    live_agent_environment: LiveAgentEnvironment,
) -> None:
    """Initialize the public SDK and prove its package/runtime pair without model quota."""
    route = _ROUTE_BY_NAME["codex:sdk"]
    live_agent_environment.select(route)
    async with AgentRuntime(live_agent_environment.runtime_config()) as runtime:
        capabilities = await runtime.capabilities(
            _scope(route, live_agent_environment.local_auth())
        )
    _require(
        capabilities.sdk_version == codex_sdk._SUPPORTED_SDK_VERSION,
        "the live Codex lane did not initialize the pinned public SDK",
    )
    _require(
        capabilities.executable_version == codex_sdk._SUPPORTED_SDK_VERSION,
        "the live Codex SDK did not report its matched bundled runtime",
    )


def test_release_certification_covers_every_shipped_route() -> None:
    """An omitted route selector must certify the whole shipped matrix, never a subset.

    This used to also require that some route certify a named API-key credential. No shipped
    route accepts one any more, so the requirement it replaced would be permanently
    unsatisfiable — and a gate that can never pass is not a gate. What remains enforceable is
    that every route in the algebra is both live-certifiable and certified by default, and
    that the refusal of named credentials is proved on each of them
    (`_probe_unsupported_named_auth`, asserted per route below). If a lane that carries named
    auth is ever added back, `named_auth_preflight_rejection` stops being true for it and this
    gate must grow the positive requirement again.
    """
    _require(_DEFAULT_ROUTES == frozenset(_ROUTES), "release certification omits a shipped route")
    _require(
        frozenset((route.backend, route.transport) for route in _ROUTES) == AGENT_ROUTES,
        "the live route table drifted from the package's closed routing table",
    )


@pytest.mark.parametrize("route", _ROUTES, ids=lambda route: route.name)
async def test_live_local_account_route_matrix(
    live_agent_environment: LiveAgentEnvironment,
    route: LiveRoute,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    live_agent_environment.select(route)
    auth_preflight = await _probe_unsupported_named_auth(live_agent_environment, route, monkeypatch)
    auth = live_agent_environment.local_auth()
    scope = _scope(route, auth)
    cwd = (tmp_path / "workspace").resolve()
    cwd.mkdir()

    async with AgentRuntime(live_agent_environment.runtime_config()) as runtime:
        capabilities = await runtime.capabilities(scope)
        _require(capabilities.scope == scope, "capability discovery returned a different scope")
        _require(capabilities.streaming, "selected route does not report streaming")
        matrix_cases = _model_reasoning_cases(route, capabilities)
        selection = matrix_cases[0]
        base_request = _session_request(route, auth, cwd, capabilities, selection)
        first_session, source_ref, turn_proof = await _first_streamed_turn(
            runtime, base_request, capabilities
        )
        await runtime.close_session(first_session)
        matrix_proved = await _probe_model_reasoning_matrix(
            runtime,
            route,
            auth,
            cwd,
            capabilities,
            matrix_cases,
            turn_proof,
        )
        listed, read = await _probe_discovery(runtime, scope, source_ref, capabilities)
        resumed, forked = await _probe_session_operations(
            runtime, base_request, source_ref, capabilities
        )
        mcp, mcp_approval = await _probe_mcp(runtime, route, auth, cwd, capabilities, selection)
        (
            workspace_write_request,
            workspace_write_effective,
            workspace_approval,
        ) = await _probe_workspace_write(runtime, route, auth, cwd, capabilities, selection)
        network_allowlist_request, network_allowlist_effective = await _probe_network_allowlist(
            runtime, route, auth, cwd, capabilities, selection
        )
        approval = mcp_approval or workspace_approval
        if "ask" in capabilities.approval_modes:
            _require(approval, "route reports native approvals but no round-trip was proved")
        cancelled = await _probe_cancellation(runtime, route, auth, cwd, capabilities, selection)

    _write_evidence(
        route,
        auth.kind,
        capabilities,
        matrix_cases,
        {
            "approval_roundtrip": approval,
            "cancellation_after_turn_started": cancelled,
            "complete_model_reasoning_matrix": matrix_proved,
            "cwd_and_fail_closed_policy_request_exercised": _cwd_effective_proved(source_ref, cwd),
            "discovery_list": listed,
            "discovery_read": read,
            "fork": forked,
            "max_turns_effective_behavior_proved": (turn_proof.max_turns_effective_behavior_proved),
            "max_turns_native_count_reported": turn_proof.max_turns_native_count_reported,
            "max_turns_request_exercised": turn_proof.max_turns_request_exercised,
            "mcp_stdio_tool": mcp,
            "model_clamp_diagnostic_proved": turn_proof.model_clamp_diagnostic_proved,
            "model_clamp_observed": turn_proof.model_clamp_observed,
            "model_effective_behavior_proved": turn_proof.model_effective_behavior_proved,
            "model_native_value_reported": turn_proof.model_native_value_reported,
            "model_request_exercised": turn_proof.model_request_exercised,
            "named_auth_preflight_rejection": auth_preflight,
            "network_allowlist_effective_behavior_proved": network_allowlist_effective,
            "network_allowlist_request_exercised": network_allowlist_request,
            "new_streamed_turn": True,
            "reasoning_clamp_diagnostic_proved": (turn_proof.reasoning_clamp_diagnostic_proved),
            "reasoning_clamp_observed": turn_proof.reasoning_clamp_observed,
            "reasoning_effective_behavior_proved": (turn_proof.reasoning_effective_behavior_proved),
            "reasoning_native_value_reported": turn_proof.reasoning_native_value_reported,
            "reasoning_request_exercised": turn_proof.reasoning_request_exercised,
            "ref_completed_by_first_event": True,
            "resume": resumed,
            "strict_output_effective_behavior_proved": (
                turn_proof.strict_output_effective_behavior_proved
            ),
            "strict_output_request_exercised": turn_proof.strict_output_request_exercised,
            "workspace_write_effective_behavior_proved": workspace_write_effective,
            "workspace_write_request_exercised": workspace_write_request,
        },
    )
