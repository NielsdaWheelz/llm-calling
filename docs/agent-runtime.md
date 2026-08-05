# Local agent runtime

`provider_runtime.agent_runtime` is the public runtime for subscription-backed,
local Codex and Claude Code sessions. It is deliberately separate from the HTTP
provider runtime:

```text
ProviderRuntime: one API generation intent -> one provider HTTP outcome
AgentRuntime:    one local SDK session -> streamed turns and durable session refs
```

The agent runtime does not pretend a stateful coding agent is a stateless model
call. Callers own prompts, durable chat history, budgets, orchestration, and
product policy. The runtime owns route selection, isolated local-account state,
capability validation, SDK lifecycle, normalized events, cancellation, and one
terminal outcome per started turn.

## Shipped routes

The routing algebra is closed:

```text
(codex, sdk)  -> openai-codex
(claude, sdk) -> claude-agent-sdk
```

There is no `cli` transport, direct App Server/JSON-RPC adapter, subprocess
protocol parser, or automatic fallback. The official SDK is the integration
boundary for each backend. Codex's SDK owns its matched bundled runtime; the
Claude SDK owns its protocol and this package supplies the exact vetted Claude
Code executable through the SDK's public `cli_path` option.

Unknown backend/transport pairs fail as `InvalidAgentRequest`. A missing optional
SDK fails as `SdkUnavailable`; it never selects another lane.

## Installation and pins

The base package imports neither agent SDK. Install the route or routes an
application actually uses:

```bash
uv sync --extra codex-sdk
uv sync --extra claude-sdk
uv sync --extra agent-sdks
```

The current exact pins are:

- `openai-codex==0.144.4`;
- `openai-codex-cli-bin==0.144.4` (the SDK's matched runtime package);
- `claude-agent-sdk==0.2.130`;
- Claude Code `2.1.220`.

`openai-codex` ships a matched Codex runtime, so `AgentRuntimeConfig` has no
Codex executable setting. Claude Code is still a local executable and may be
selected with `AgentRuntimeConfig.claude_executable`.

An SDK or executable version mismatch is a typed availability failure. Upgrades
therefore require a pin change, deterministic adapter tests, the clean-wheel
checks, and a new unnarrowed live certification run.

## Minimal use

```python
from pathlib import Path

from provider_runtime.agent_runtime import (
    AgentCapabilityScope,
    AgentRuntime,
    AgentRuntimeConfig,
    AgentSessionRequest,
    CredentialRef,
    NewSession,
    PermissionPolicy,
    TextContent,
    TurnRequest,
)

config = AgentRuntimeConfig(state_root_base=Path("/private/agent-state"))
auth = CredentialRef(kind="local_account", profile_key="personal")

request = AgentSessionRequest(
    backend="codex",
    transport="sdk",
    auth=auth,
    open=NewSession(),
    cwd="/absolute/workspace",
    policy=PermissionPolicy(),
)

async with AgentRuntime(config) as runtime:
    capabilities = await runtime.capabilities(
        AgentCapabilityScope(backend="codex", transport="sdk", auth=auth)
    )
    session = await runtime.open_session(request)
    result = await runtime.run_turn(
        session,
        TurnRequest(input=(TextContent("Summarize this repository."),)),
    )
    await runtime.close_session(session)
```

`open_session` always performs its own scoped
capability discovery and validation, so callers cannot bypass it with a stale
table.

Capability discovery is host-sensitive where the native executable depends on an OS
sandbox. On Linux, restricted Codex workspace writes and Claude network allowlists are
advertised only when `bubblewrap` can actually create its network namespace (and Claude's
allowlist additionally requires `socat`). Containers that deny that namespace therefore fail
closed during request validation instead of accepting a mode that will fail midway through a
turn. Codex `full_access` remains a distinct, explicitly requested unsandboxed mode.

For streamed UI or telemetry, iterate `runtime.stream_turn(...)` instead of
calling `run_turn(...)`. `run_turn` is the terminal projection of that same
stream.

## Ownership boundary

The official SDKs own:

- vendor argument construction and protocol negotiation;
- native session/thread operations;
- native notifications and tool execution;
- subscription authentication already enrolled by the user;
- provider-defined approval behavior exposed by the SDK.

This package owns:

- one closed `(backend, transport)` selection;
- an isolated state root and child environment;
- exact SDK/executable compatibility checks;
- request validation against discovered capabilities;
- normalized immutable events and strict terminal grammar;
- timeout, cancellation, output bounds, and cleanup;
- transparent environment-replacing/process-group launchers where the public
  SDK lacks those process controls;
- redaction and typed public errors;
- SDK-neutral session references and test doubles.

The adapters use public SDK surface only. Directly consuming a vendor's internal
wire protocol would duplicate lifecycle, versioning, and notification behavior
the maintained SDK already owns.

## Authentication and state isolation

Both shipped routes accept only:

```python
CredentialRef(kind="local_account", profile_key="...")
```

The user must enroll the native tool before using the package. The runtime does
not implement login, token brokering, hosted subscription proxying, or API-key
fallback. `api_key_environment` and `secret_reference` session credentials are
rejected before secret resolution.

`state_root_base` must be an existing normalized absolute directory that is not
group- or world-writable. A profile lives at:

```text
<state_root_base>/<backend>/<profile_key>
```

Runtime-created directories are mode `0700`. The child environment is rebuilt
from a fail-closed allowlist. `HOME`, `PATH`, locale, temp, `CODEX_HOME`, and
`CLAUDE_CONFIG_DIR` are runtime-owned; caller policy cannot override them.
Credential-class, provider-selection, and process-control environment names are
also rejected.

For Codex, the selected environment is passed through public `CodexConfig.env`
and the profile root through its configuration. The SDK overlays that mapping on
the host process environment rather than replacing it, so the adapter points
public `CodexConfig.codex_bin` at a content-addressed launcher. The launcher
forwards the SDK-owned arguments unchanged to the matched bundled runtime,
replaces the environment with the exact selected-name allowlist, and supervises
one private process group. Ambient API keys and unrelated variables therefore do
not reach Codex. `client.account()` must report a ChatGPT account.

For Claude, the SDK is pointed at the isolated environment and exact executable.
Its content-addressed `0700` launcher calls `setsid()` and then `execv()` so
Claude Code and descendants can be terminated as one process group. Both
launchers live in runtime-owned backend parent directories outside the child
profile, contain no credentials, and do not reconstruct SDK arguments.

`CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK` must be unset. A non-empty value would
disable the exact-version contract and therefore fails closed.

## Capabilities first

Every request is validated against `AgentCapabilities` scoped to the exact
backend, transport, and credential. Capabilities report SDK/runtime versions,
session and discovery operations, models, reasoning, content, structured
output, tools, MCP, sandbox/network/approval modes, overrides, and observable
native facts.

The current material differences are:

| Contract | Codex SDK | Claude Agent SDK |
| --- | --- | --- |
| Auth | local ChatGPT account | local Claude account |
| Sessions | new, resume, fork | new, resume, fork |
| Discovery | list, metadata-only read | none |
| Models | enumerated by SDK | open/unknown (`None`) |
| Input | text, local images | text |
| Structured output | native JSON Schema | native JSON Schema |
| Filesystem | read-only, workspace-write, full-access | read-only, workspace-write |
| Network | disabled, unrestricted | disabled, exact-host allowlist |
| Approval | deny, provider-review | deny, caller ask, allow |
| MCP | stdio and streamable HTTP; reference auth | streamable HTTP; no reference auth |
| Native tool filters | not exposed | exact accepted Claude tool names |
| Session cwd scope | no | yes |

`models=None` means the transport cannot enumerate models; it does not mean every
string is known-good. The backend may reject a misspelled Claude model.

Neither route reports the reasoning effort it actually used, so
`reports_effective_effort=False`. A requested effort can be validated against the
published vocabulary, but silent provider-side clamping remains unobservable.

## Sessions and references

`AgentSessionRequest.open` is one of `NewSession`, `ResumeSession`, or
`ForkSession`. A successful open returns an opaque `AgentSession` for this live
runtime. Persist `session.ref` only after it is complete.

`AgentSessionRef` carries:

- schema version;
- backend and transport;
- native session identifier;
- profile key;
- state-root fingerprint;
- cwd fingerprint.

Resume and fork fail closed on route, auth profile, or state-root mismatch.
Claude additionally requires the same cwd because its native sessions are
directory-scoped. Codex refs retain the cwd fingerprint as provenance but may be
resumed from another directory.

Codex implements `list_sessions` and metadata-only `read_session`. Turn/item
history flags and cursors are rejected until a native paginated history contract
is bound. Claude advertises no discovery operation.

One session permits one active turn. A concurrent turn is `ConcurrentTurn`.
`close_session(session)` idempotently interrupts any active turn, releases only
that session's SDK client and owned processes, and leaves sibling sessions
usable. A closed handle raises `SessionUnavailable` on later turns. Runtime
close performs the same bounded cleanup for every remaining session and makes
all handles unusable.

## Policy and approvals

Defaults are intentionally restrictive:

```text
filesystem: read_only
network:    disabled
approval:   deny
tools:      no built-ins allowed
environment: empty
```

`full_access`, `unrestricted`, and unconditional approval `allow` each require an
exact `UnsafeConfirmation`; unused confirmations are rejected too.

Approval modes have distinct ownership:

- `deny`: no escalation may be approved;
- `ask`: the runtime calls the caller's typed `ApprovalHandler` (Claude only);
- `provider_review`: the provider's maintained review policy decides (Codex
  `ApprovalMode.auto_review`);
- `allow`: unconditional approval, requiring unsafe confirmation (Claude only).

`ask` and `provider_review` are incomparable: a per-turn patch cannot silently
swap who reviews a request. Either may narrow to `deny`; `allow` may narrow to
either reviewer.

Per-turn `PermissionPolicyPatch` can only narrow. Filesystem/network modes move
toward less authority, allowed tools must be an exact subset, denied tools an
exact superset, copied environment names an exact subset, and network allowlist
entries an exact subset where the base policy already had an allowlist.

Codex's public SDK does not expose exact built-in allow/deny filtering, so the
route reports `tool_controls=False`. It accepts only the sentinel
`allowed_tools=("*",)` when built-ins are intentionally enabled; specific names
would claim a control the SDK cannot enforce. Claude publishes and enforces its
exact accepted native names.

## MCP

MCP configuration is session-scoped on both routes.

- Stdio MCP is accepted only with explicit `full_access` and `unrestricted`
  policy because the selected local executable is outside sandbox attestation.
- Streamable HTTP MCP requires HTTPS and must fit the network policy.
- Claude can enforce an exact hostname allowlist but accepts no credential refs.
- Codex accepts environment/header references, but its route cannot enforce an
  exact hostname allowlist; remote MCP therefore requires unrestricted network.

Secrets are resolved at the process boundary through `secret_resolver` or a
named environment source, placed only in opaque child-environment aliases, and
never copied into public values. Stdio MCP under full access is not a credential
boundary: a same-uid command can inspect peer processes. Use a dedicated OS user
or container for credentialed stdio servers.

## Structured output and native options

`JsonSchemaAgentOutput` accepts the package's canonical JSON Schema subset. The
adapter passes the schema through the SDK's public native output-schema option,
then validates the final value again before returning it. No JSON repair or
best-effort parsing occurs.

Native extension objects are versioned, backend-specific escape hatches:

- `CodexNativeOptions(web_search=...)` is session-scoped and requires
  unrestricted network when enabled;
- `ClaudeNativeOptions(include_partial_messages=...)` is session-scoped.

Unknown or wrong-backend native options fail before SDK startup.

## Event and terminal grammar

The normalized stream kinds are:

```text
session_started | turn_started | text_delta | reasoning |
tool_started | tool_updated | tool_completed |
approval_requested | approval_answered | file_change |
usage | diagnostic | unknown | native_retry_observed |
turn_completed | turn_failed | turn_cancelled
```

`diagnostic` and `unknown` are the only session-scoped kinds allowed before
`turn_started`. After a turn starts, events must preserve native order and one of
the three terminal kinds must appear exactly once and last. Missing, duplicate,
misidentified, post-terminal, or cross-turn events are `ProtocolDefect`.

`native_payload` is a recursively frozen, redacted, field-allowlisted diagnostic
view. Unknown additive SDK notifications become `unknown`; malformed known
notifications remain defects.

`run_turn` returns `AgentResult(status="succeeded" | "failed" | "cancelled")`
from the terminal event. Model/backend failures are terminal values; broken
runtime invariants raise.

## Cancellation, limits, and retries

The runtime bounds turn duration, event count, individual message size, final
text, diagnostics, and cleanup. A caller cancellation signal or timeout invokes
the SDK's native interrupt operation. Cancellation before native turn identity
raises `TurnNotStarted`; after identity it yields `turn_cancelled`.

If a stream transport fails or violates its grammar, the Codex SDK client is
discarded rather than reused with uncertain native state. Claude drains an
interrupted turn or invalidates the session before another turn can begin.

The runtime never retries a turn. Provider-native retry observations may be
reported as `native_retry_observed`, but replaying a stateful agent turn at this
layer would risk duplicate side effects.

## Error model

Expected, modelable failures derive from `AgentRuntimeError`:

```text
InvalidAgentRequest | UnsupportedCapability |
CredentialUnavailable | CredentialRejected |
ExecutableUnavailable | SdkUnavailable |
McpConfigurationError | McpUnavailable |
SessionMismatch | SessionUnavailable | ConcurrentTurn | TurnNotStarted
```

Broken adapter/runtime invariants derive from `AgentRuntimeDefect`, principally
`ProtocolDefect` and `MissingTerminalEvent`. Errors and diagnostics sanitize
provider text and must not expose tokens, raw SDK messages, or resolved secrets.

## Testing

Application tests should use `ScriptedAgentRuntime` or
`NoNetworkAgentRuntime`. Deterministic adapter tests replace only the public SDK
boundary with typed fakes; there are no checked-in Codex wire schemas, JSON-RPC
fixtures, or executable emulators.

CI proves all of these packaging shapes:

- base wheel imports neither optional SDK;
- `codex-sdk` wheel extra imports Codex only;
- `claude-sdk` wheel extra imports Claude only;
- `agent-sdks` installs both;
- a no-extras environment exercises both typed `SdkUnavailable` paths.

The paid local-account matrix is opt-in:

```bash
LLM_RUNTIME_LIVE=1 \
LLM_RUNTIME_LIVE_AGENT_STATE_ROOT_BASE=/absolute/existing/private/root \
LLM_RUNTIME_LIVE_AGENT_PROFILE=live-local \
LLM_RUNTIME_LIVE_CLAUDE_SDK_MODELS=<model-a>,<model-b> \
uv run pytest -m live_provider tests/live/test_agent_matrix.py
```

An omitted route selector is the release run and covers both shipped routes.
`LLM_RUNTIME_LIVE_AGENT_ROUTES=codex:sdk` or `claude:sdk` narrows a debugging run
and certifies nothing. The matrix never enrolls an account or prints tokens.
Codex's no-quota preflight initializes the public SDK and checks the exact
SDK/bundled-runtime pair; Claude's preflight resolves and version-checks the real
executable in the runtime-owned child environment.
Codex certifies every discovered `(model, model-supported effort)` pair. Claude
requires the complete native model list in
`LLM_RUNTIME_LIVE_CLAUDE_SDK_MODELS` and certifies its cross-product with every
reported effort. Empty entries, whitespace, and duplicate Claude models fail
before a paid turn starts. Route-wide feature probes still run once per route.

## References

- [OpenAI Codex SDK for Python](https://github.com/openai/codex/tree/main/sdk/python)
- [Codex authentication](https://developers.openai.com/codex/auth/)
- [Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)
- [Claude Agent SDK Python](https://code.claude.com/docs/en/agent-sdk/python)
- [Claude Agent SDK sessions](https://code.claude.com/docs/en/agent-sdk/sessions)
- [Claude Agent SDK permissions](https://code.claude.com/docs/en/agent-sdk/permissions)
