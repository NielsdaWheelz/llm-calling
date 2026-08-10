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
the authorization model, SDK lifecycle, normalized events, cancellation, and one
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

The extras carry bounded constraints (`openai-codex>=0.144.4,<1`,
`openai-codex-cli-bin>=0.144.4,<1`, `claude-agent-sdk>=0.2.130,<1`); the
lockfile pins the exact resolution. The vetted versions the adapters were
written against are openai-codex 0.144.4 with its matched runtime, and
claude-agent-sdk 0.2.130 with Claude Code 2.1.220.

A version that drifts from the vetted one is met with **one warning plus a
behavioral capability probe**, never a hard fail: the adapters verify what the
backend actually does (account type, effective configuration, sandbox
capability) instead of trusting a version-keyed table. A missing SDK, missing
bundled runtime, or unresponsive executable remains a typed availability
failure (`SdkUnavailable` / `ExecutableUnavailable`).

`openai-codex` ships a matched Codex runtime, so `AgentRuntimeConfig` has no
Codex executable setting. Claude Code is still a local executable and may be
selected with `AgentRuntimeConfig.claude_executable`.

## Minimal use

```python
from pathlib import Path

from provider_runtime.agent_runtime import (
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
    session = await runtime.open_session(request)
    terminal = await runtime.run_turn(
        session,
        TurnRequest(input=(TextContent("Summarize this repository."),)),
    )
    await runtime.close_session(session)
```

For streamed UI or telemetry, iterate `runtime.stream_turn(...)` instead of
calling `run_turn(...)`. `run_turn` is the terminal projection of that same
stream and returns the stream's `AgentTerminal`.

Validation is behavioral, not table-driven: `open_session` fails closed before
any billable work when the request asks for something the transport cannot
enforce (an unenforceable tool filter, a sandbox mode the host cannot provide,
an approval mode the backend does not have). On Linux, restricted Codex
workspace writes and Claude network allowlists require `bubblewrap` to create
its network namespace (Claude's allowlist additionally requires `socat`); hosts
that cannot are refused during `open_session` instead of failing midway through
a turn.

## Ownership boundary

The official SDKs own:

- vendor argument construction and protocol negotiation;
- native session/thread operations;
- native notifications and tool execution;
- subscription authentication already enrolled by the user;
- provider-defined approval behavior exposed by the SDK.

This package owns the retained security kernel and the lifecycle around it:

- one closed `(backend, transport)` selection;
- an isolated state root and child environment;
- restrictive permission defaults and narrowing-only policy changes;
- unsafe-action confirmation for model-initiated shell/filesystem/network/MCP
  actions;
- bounded, recursively redacted native event representation;
- normalized immutable events and the strict terminal grammar;
- timeout, cancellation, output bounds, and cleanup;
- transparent environment-replacing/process-group launchers where the public
  SDK lacks those process controls;
- typed public errors;
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
rejected at every route and refused structurally by the child-environment
builder; those kinds exist only as *sources* for MCP credential references.

Subscription pool exhaustion ends the turn with an `AgentTerminal` whose
failure is the `AgentQuotaExhausted` value. Block-and-stop only: the lane never
overflows onto API-rate credentials.

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

`CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK` must be unset. It is an ambient control
over the SDK's own compatibility behavior and therefore fails closed.

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

Codex implements `list_sessions` and metadata-only `read_session` (name
metadata only; no history contract is bound). Claude has no discovery
operation and answers both with `UnsupportedCapability`.

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

Per-turn `PermissionPolicyPatch` can only narrow — a session may only ever
reduce what is allowed, never widen it. Filesystem/network modes move toward
less authority, allowed tools must be an exact subset, denied tools an exact
superset, copied environment names an exact subset, and network allowlist
entries an exact subset where the base policy already had an allowlist. The
shipped SDK routes cannot reconfigure a live client's policy, so they reject
per-turn patches; the narrowing algebra still gates the request before the
adapter sees it.

Codex's public SDK does not expose exact built-in allow/deny filtering. It
accepts only the sentinel `allowed_tools=("*",)` when built-ins are
intentionally enabled; specific names would claim a control the SDK cannot
enforce. Claude validates policies against its exact accepted native tool names
and verifies the effective tool set the backend reports at session start.

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

`JsonSchemaAgentOutput` carries a plain JSON Schema mapping (pass
`model_json_schema()` where a pydantic model exists). The adapter passes the
schema through the SDK's public native output-schema option; the backend
enforces it. The final value is strict-parsed and frozen — no JSON repair, no
coercion — and a miss is the `output_schema_violation` terminal failure.

Native extension objects are versioned, backend-specific escape hatches:

- `CodexNativeOptions(web_search=...)` is session-scoped and requires
  unrestricted network when enabled;
- `ClaudeNativeOptions(include_partial_messages=...)` is session-scoped.

Unknown or wrong-backend native options fail before SDK startup.

## Event and terminal grammar

The normalized stream is exactly six kinds:

```text
AgentText              one chunk of assistant output text
AgentToolUse           tool_call_id, name, phase started|updated|completed,
                       owned payload; completed carries succeeded
AgentUsage             TokenUsage, normalized to the provider lane's noun
AgentPermissionRequest one answered unsafe-action confirmation (request + decision)
AgentNative            any native frame without a first-class kind, as a
                       bounded, recursively redacted payload
AgentTerminal          exactly-once terminal: status, typed failure value,
                       final text, structured output, usage, session ref
```

`AgentTerminal.failure` is `None`, `AgentQuotaExhausted()`, or
`AgentFailure(cause)` with causes `backend_failed`, `turn_timeout`,
`output_limit_exceeded`, `approval_unanswered`, `output_schema_violation`.
Model/backend failures are terminal values; broken runtime invariants raise
(`ProtocolDefect`, `MissingTerminalEvent`). Exactly one `AgentTerminal` ends
every started turn, last; post-terminal frames are defects.

Native detail that used to have first-class kinds — reasoning deltas, file
changes as separate events, diagnostics, retry observations, system frames —
travels as `AgentNative` with the redacted native payload.

## Cancellation, limits, and retries

The runtime bounds turn duration, event count, individual message size, final
text, diagnostics, and cleanup. A caller cancellation signal or timeout invokes
the SDK's native interrupt operation. Cancellation before the first stream
event raises `TurnNotStarted`; after it, the stream ends with a cancelled (or
`turn_timeout`-failed) `AgentTerminal` that preserves the text and usage the
consumer already received.

If a stream transport fails or violates its grammar, the Codex SDK client is
discarded rather than reused with uncertain native state. Claude drains an
interrupted turn or invalidates the session before another turn can begin.

The runtime never retries a turn. Replaying a stateful agent turn at this layer
would risk duplicate side effects.

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
uv run pytest -m live_provider tests/live/test_agent_matrix.py
```

An omitted route selector is the release run and covers both shipped routes.
`LLM_RUNTIME_LIVE_AGENT_ROUTES=codex:sdk` or `claude:sdk` narrows a debugging run
and certifies nothing. The matrix never enrolls an account or prints tokens.
Per route it certifies: one full streamed turn under the default restrictive
policy, a resumed second turn on the same native session, and a structured
output turn — asserting the six-kind grammar, the terminal shape, and
normalized `TokenUsage` on the way through.

## References

- [OpenAI Codex SDK for Python](https://github.com/openai/codex/tree/main/sdk/python)
- [Codex authentication](https://developers.openai.com/codex/auth/)
- [Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)
- [Claude Agent SDK Python](https://code.claude.com/docs/en/agent-sdk/python)
- [Claude Agent SDK sessions](https://code.claude.com/docs/en/agent-sdk/sessions)
- [Claude Agent SDK permissions](https://code.claude.com/docs/en/agent-sdk/permissions)
