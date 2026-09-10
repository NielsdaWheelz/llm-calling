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
product policy. The runtime owns route selection, Codex connection lifecycle or
Claude process lifecycle, authorization, normalized events, cancellation, and
one terminal outcome per started turn.

## Shipped routes

The routing algebra is closed:

```text
(codex, sdk)  -> host-owned Codex app-server over WebSocket on a Unix socket
(claude, sdk) -> claude-agent-sdk
```

There is no `cli` route or automatic fallback. The public route name remains
`sdk` for stored-reference compatibility. On Codex, provider-runtime owns the
documented WebSocket client protocol over an externally configured Unix socket.
It has no Python Codex SDK, bundled executable, private App Server, or
`AsyncCodex` path. Claude remains on the official SDK and receives the exact
vetted Claude Code executable through public `cli_path`.

Unknown backend/transport pairs fail as `InvalidAgentRequest`. A missing optional
SDK fails as `SdkUnavailable`; it never selects another lane.

## Installation and native version policy

The base package imports neither agent SDK. Install the route or routes an
application actually uses:

```bash
uv sync --extra claude-sdk
```

The base package directly constrains `websockets>=16,<17`; the lockfile pins its
exact resolution. The host independently installs and supervises the latest
stable Codex App Server/TUI. Native `userAgent` is string metadata, not a
version-based admission rule. The Claude extra carries
`claude-agent-sdk>=0.2.130,<1` with its exact lock resolution. A missing transport
dependency raises `SdkUnavailable`; a missing or unreachable configured Codex
endpoint is a typed credential/profile
availability failure, never a private-runtime fallback.

Initialize response shape, correlation, account routing, and authority
classification remain strict. New native behavior is not implicitly certified:
protocol drift fails closed and live qualification is separate from routine
fixture coverage. No version parser, compatibility fallback, or private server
is selected when the shared endpoint is incompatible.

Codex replays cumulative usage around `thread/resume`. The owned
transport keeps every notification in wire order, validates an explicit
pre-turn allowlist, and derives the resume/fork baseline without a private SDK
queue seam. A replay that races behind the response is accepted only as the
first baseline-only snapshot, with its exact prior-turn identity, after the next
turn starts. Any later stale identity and every missing identity fail closed.
Unknown pre-turn messages fail before a billable turn.

The transport bounds queued notifications independently of per-message and
per-turn limits: at most 100,000 events and 64 MiB of wire data. Overflow
fails pending RPCs and disconnects this client, never the shared service.
Earlier authority events remain observable; a queued terminal cannot survive
a known transport failure. Malformed responses and RPC timeouts fail the
same connection before another request can start.

`AgentRuntimeConfig.codex_endpoints` maps caller-owned opaque profile keys to
absolute Unix-socket paths. The runtime has no Codex executable setting. Claude
Code remains a local executable selected with
`AgentRuntimeConfig.claude_executable`.

## Minimal use

```python
from pathlib import Path

from provider_runtime.agent_runtime import (
    AgentRuntime,
    AgentRuntimeConfig,
    CodexCatalogSessionRequest,
    CodexNativeOptions,
    CredentialRef,
    NewSession,
    PermissionPolicy,
    TextContent,
    TurnRequest,
)

config = AgentRuntimeConfig(
    state_root_base=Path("/private/agent-state"),
    codex_endpoints={"personal": Path("/run/codex-shared-personal/app-server.sock")},
)
auth = CredentialRef(kind="local_account", profile_key="personal")

async with AgentRuntime(config) as runtime:
    catalog = await runtime.model_catalog("codex", auth)
    model = catalog.models[0]  # Application selection, never a library default.
    reasoning = model.reasoning[0]
    request = CodexCatalogSessionRequest(
        auth=auth,
        open=NewSession(),
        cwd="/absolute/workspace",
        model_key=model.key,
        reasoning=reasoning.key,
        agent_definition_revision=catalog.definition_revision,
        row_fingerprint=model.row_fingerprint,
        # Contained cognition: read-only, offline, deny-all, no MCP,
        # copied environment, or additional roots; native behavior is separately qualified.
        policy=PermissionPolicy(allowed_tools=("*",)),
        native=CodexNativeOptions(builtin_tools="disabled"),
    )
    session = await runtime.open_session(request)
    terminal = await runtime.run_turn(
        session,
        TurnRequest(input=(TextContent("Summarize this repository."),)),
    )
    await runtime.close_session(session)
```

## Model catalog and tagged requests

`AgentRuntime.model_catalog("codex", auth)` drives the authenticated public
App Server RPC `model/list` through every page and returns every visible row in native
order. `AgentModelCatalog` carries a content-derived definition revision;
each `AgentModelFacts` carries exact model/dispatch identity, ordered reasoning
facts, modalities, lifecycle/upgrade facts, and a content-derived row
fingerprint. The public App Server catalog reports neither context-window nor
max-output capacity, so both source-capacity fields are honestly `Absent` and
do not make a row unusable. Product request budgets remain caller-owned.

The session request is the closed union
`CodexCatalogSessionRequest | ClaudeNativeSessionRequest`. The Codex arm must
carry the selected catalog revision and row fingerprint; `open_session`
re-reads the catalog, rejects stale or unsupported selections, and resolves
the server-only dispatch model and reasoning wire value before billable work.
There is no free-form Codex model path or compatibility constructor. Claude's
native arm remains separate because that SDK exposes no equivalent public
catalog; asking the Claude route for one raises `UnsupportedCapability`.

For streamed UI or telemetry, iterate `runtime.stream_turn(...)` instead of
calling `run_turn(...)`. `run_turn` is the terminal projection of that same
stream and returns the stream's `AgentTerminal`.

Validation is behavioral, not table-driven: `open_session` fails closed before
any billable work when the request asks for something the transport cannot
enforce (an unenforceable tool filter, a sandbox mode the host cannot provide,
an approval mode the backend does not have). Session instructions and reasoning
follow the same rule — each is mapped to a documented provider option or refused, never
dropped: Codex takes `system` as `base_instructions` and `developer` as
`developer_instructions`; Claude has one instruction channel (`system_prompt`),
so it takes `system` and refuses `developer`, and it maps
`ReasoningSpec.summary` onto its only summary control, `thinking.display`
(`none`/`auto`), refusing the `concise`/`detailed` verbosity it has no knob for.

All of that is session-scoped, because neither provider integration can
reconfigure a live client. `TurnRequest` therefore carries exactly one turn's `input`, an optional
narrowing `policy` patch, and an optional `timeout_seconds` — there are no
per-turn instruction, model, reasoning, MCP, or output overrides to pass.

On Linux, restricted Codex workspace writes and Claude network allowlists
require `bubblewrap` to create its network namespace (Claude's allowlist
additionally requires `socat`); hosts that cannot are refused during
`open_session` instead of failing midway through a turn.

## Ownership boundary

For Codex, this package owns the complete client connection: initialize/initialized
negotiation, client request ids, response correlation, notification ordering,
server-request policy, thread/turn operations, and disconnect. The host owns
service process and account lifecycle. Only documented public App Server methods
are used. For Claude, the
official SDK continues to own the vendor protocol. Native execution itself and
already-enrolled subscription authentication remain provider behavior.

The retained security kernel owns:

- one closed `(backend, transport)` selection;
- authenticated Codex catalog discovery and exact catalog-bound selection;
- an isolated Claude state root/environment and explicit Codex endpoint map;
- restrictive permission defaults and narrowing-only policy changes;
- unsafe-action confirmation for model-initiated shell/filesystem/network/MCP
  actions;
- bounded, recursively redacted native event representation;
- normalized immutable events and the strict terminal grammar;
- timeout, cancellation, output bounds, and cleanup;
- the existing transparent launcher where the Claude SDK lacks process controls;
- typed public errors;
- SDK-neutral session references and test doubles.

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
group- or world-writable. Only a process-owning backend stores a profile there:

```text
<state_root_base>/claude/<profile_key>
```

Codex never creates a profile directory. Runtime-created Claude directories are
mode `0700`; its child environment is
rebuilt from a fail-closed allowlist. `HOME`, `PATH`, locale, temp, and
`CLAUDE_CONFIG_DIR` are runtime-owned there. Codex starts no child and consumes
no caller environment. Its configured endpoint must be an absolute Unix-socket
path and the documented `account/read` response must report a ChatGPT account.
No token-refresh or attestation callback is implemented: either request gets a
JSON-RPC error and terminates the connection.

For Claude, the SDK is pointed at the isolated environment and exact executable.
Its content-addressed `0700` launcher calls `setsid()` and then `execv()` so
Claude Code and descendants can be terminated as one process group. The
launcher lives in a runtime-owned backend parent directory outside the child
profile, contains no credentials, and does not reconstruct SDK arguments.

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

Codex has no public typed per-name built-in filter. The portable policy therefore
still requires the sentinel `allowed_tools=("*",)`. The additional
`CodexNativeOptions(builtin_tools="disabled")` posture writes the complete
supported feature-off configuration and requires read-only
filesystem, disabled network, denied approvals, empty copied environment, no
MCP, and no additional roots. Provider review is refused in this posture.
Claude continues to accept exact tool names, reject glob patterns and its two
network-reaching built-ins, and verify the reported effective set.

## MCP

MCP configuration is session-scoped on both routes.

- Stdio MCP is accepted only with explicit `full_access` and `unrestricted`
  policy because the selected local executable is outside sandbox attestation.
- Streamable HTTP MCP requires HTTPS and must fit the network policy.
- Claude can enforce an exact hostname allowlist but accepts no credential refs.
- Codex accepts environment/header references, but its route cannot enforce an
  exact hostname allowlist, so remote MCP requires unrestricted network. Use
  `workspace_write` + `unrestricted`, acknowledging `network_unrestricted`
  only: the route writes `sandbox_workspace_write.network_access = true`, which
  confines writes to `cwd`/`additional_dirs` while leaving reads and egress
  open. `read_only` + network is refused — the Codex config has a network
  toggle only under `sandbox_workspace_write`. See SECURITY.md.

Secrets are resolved at the process boundary through `secret_resolver` or a
named environment source, placed only in opaque child-environment aliases, and
never copied into public values. Stdio MCP under full access is not a credential
boundary: a same-uid command can inspect peer processes. Use a dedicated OS user
or container for credentialed stdio servers.

Applications with a canonical `llm-tools` plan use
`lower_mcp_tools(McpToolPublication(...))`. It projects exactly that frozen
plan into one authenticated HTTPS MCP server plus an immutable reverse index;
`PublishedMcpTools.observe` maps Codex `AgentToolUse` observations back to
canonical tool ids and rejects names outside the publication without retaining
their payload. The existing provider function-tool adapter and this MCP
adapter share the same exposure/alias owner.

Applications may configure `AgentRuntimeConfig.codex_sandbox` with
`CodexSandboxControls(exclude_slash_tmp, exclude_tmpdir_env_var)`. Both native
workspace-write exclusions apply on new, resumed, and forked Codex sessions.
The shared service's TMPDIR is host-owned; the private-child `child_tmpdir`
option is removed, not silently ignored or translated into another process.

## Structured output and native options

`JsonSchemaAgentOutput` carries a plain JSON Schema mapping (pass
`model_json_schema()` where a pydantic model exists). The adapter passes the
schema through the App Server's public `turn/start.outputSchema` field; the
backend enforces it. The final value is strict-parsed and frozen — no JSON
repair, no coercion — and a miss is the `output_schema_violation` terminal
failure.

Native extension objects are versioned, backend-specific escape hatches:

- `CodexNativeOptions(web_search=...)` is session-scoped and requires
  unrestricted network when enabled;
- `CodexNativeOptions(builtin_tools="disabled")` disables the execution,
  integration, and local-context feature set through explicit native controls.
  It also suppresses app, skill, environment, permission, and collaboration
  instructions plus request-user-input.
  This is a containment posture, not proof of pre-execution
  prevention: public controls do not establish that Code Mode/native `exec` is
  absent before computation. The child is credentialless, read-only, and
  offline. Its first observable authority event poisons the turn, invalidates
  the session, and makes every later terminal ineligible;
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
AgentNative            an explicitly allowlisted inert observation, as a bounded,
                       recursively redacted payload
AgentTerminal          exactly-once terminal: status, typed failure value,
                       final text, structured output, usage, session ref
```

### Authoritative assistant response

`AgentTerminal.final_text` is the provider's authoritative selected assistant
response. It is not a concatenation of all assistant-channel traffic observed
during the turn. `AgentText` remains the bounded streaming-observation surface:
it can include provider-native commentary and drafts, so consumers may display
it but must not execute it or treat its cross-item concatenation as structured
output. Both structured and unstructured terminals use the same authoritative
message selection before strict JSON parsing or downstream schema validation.

For Codex, the adapter retains each completed `agentMessage` item's identity,
text, phase, and native completion order. At terminal it scans those items in
reverse order and selects the last `phase=final_answer` item. If none exists, it
selects the last completed item whose phase is absent, matching the supported
App Server behavior. Commentary is never eligible,
even if it is individually valid JSON or arrives after the final answer. Multiple
eligible messages are not concatenated. A completed turn with no eligible item,
or a duplicate, empty, malformed, or unknown-phase completed assistant identity,
is a `ProtocolDefect`; the adapter never guesses from streamed deltas. Failed,
interrupted, output-limited, or runtime-cancelled turns expose an already-completed
eligible response when one exists and otherwise use empty final text.

### Invocation-local usage

`AgentTerminal.usage` is usage attributable only to that `stream_turn` or
`run_turn` invocation on every terminal status. It never contains usage from an
earlier native thread/session turn, even when an upstream protocol reports a
cumulative counter. An `AgentUsage` event is an invocation-to-date snapshot;
multiple `AgentUsage` events from one turn are progressive and are not values to
sum. The terminal carries the final safely attributable snapshot.

Codex reports `tokenUsage.total` cumulatively across a native thread. The Codex
adapter validates both `total` and `last`, but does not use `last` as its
accounting source because one runtime turn can make multiple model requests. It
subtracts every cumulative update from one fixed pre-turn baseline, so the last
delta aggregates all model requests made by the invocation. The baseline is:

- synthetic zero for a fresh native thread;
- the last validated cumulative end snapshot for consecutive turns; or
- the cumulative snapshot Codex replays around resume/fork for reopened and
  reconstructed processes/sessions. A replay already queued at resume is
  consumed before a new turn. If it races behind the response, exactly the first
  stale-prior-turn usage identity may establish the unknown baseline after the
  next turn starts. Either form emits no `AgentUsage`.

Input, output, total, cached-input, cache-write-input, and reasoning-output
counters are differenced independently. Optional counter presence must remain
stable across observed cumulative snapshots; absence is never treated as zero.
Codex totals and component bounds must be internally consistent.

The accounting boundary fails safe:

- malformed, negative, internally inconsistent, decreasing, reset, or
  presence-changing snapshots raise `ProtocolDefect`; counts are never clamped;
- unchanged cumulative snapshots are suppressed, including replay and duplicate
  rate-limit updates;
- a missing usage turn id, or a stale id outside that single first-rebase race,
  is `ProtocolDefect`;
- a usage update arriving late in the turn but before `turn/completed` is still
  included; `turn/completed` is the App Server's hard stream boundary, so a
  post-completion update is not attributable and is never assigned to a later
  terminal;
- if no advancing usage notification is available, terminal usage is `Absent`
  and the cumulative boundary becomes untrusted. The next observed cumulative
  snapshot is baseline-only. Usage remains `Absent` until that invocation also
  supplies a later monotonic update, or a close/reopen/resume supplies the
  restored baseline before new usage. This deliberately prefers under-reporting
  to guessing or charging history twice;
- cancellation or interruption preserves the latest invocation-local usage
  already supplied, but invalidates its cumulative end as a baseline if the
  stream ended before Codex's terminal usage boundary.

`AgentTerminal.failure` is `None`, `AgentQuotaExhausted()`, or
`AgentFailure(cause)` with causes `backend_failed`, `turn_timeout`,
`output_limit_exceeded`, `approval_unanswered`, `output_schema_violation`.
Model/backend failures are terminal values; broken runtime invariants raise
(`ProtocolDefect`, `MissingTerminalEvent`). Exactly one `AgentTerminal` ends
every valid started-turn stream, last; a defect raises without accepting a
terminal, and post-terminal frames are defects.

Known command, file, MCP/app, Web, image, hook, sub-agent, dynamic/custom tool,
process, approval-review, and Code Mode/custom-exec activity is never
`AgentNative`: it becomes `AgentToolUse` (or `AgentPermissionRequest` for
permission requests). Known approvals are answered with their explicit denial
shape before the event is exposed. Unknown server requests, unknown item types,
malformed/duplicate/reordered identities, uncorrelated responses, and protocol
drift raise `ProtocolDefect`. A future item type therefore defaults to forbidden.
`AgentNative` is reserved for the explicit bounded/redacted allowlist: turn
lifecycle, reasoning/plan observations, warnings/errors, status/rate-limit/model
metadata, resolved-request notices, and terminal native evidence.

## Cancellation, limits, and retries

The runtime bounds turn duration, JSONL message size/shape, event count, final
text, diagnostics, and cleanup. A caller cancellation signal or timeout invokes
the provider's native interrupt operation. Cancellation before the first stream
event raises `TurnNotStarted`; after it, the stream ends with a cancelled (or
`turn_timeout`-failed) `AgentTerminal` that preserves safely attributable usage.
Codex final text uses only an eligible assistant item completed before
interruption; its observed commentary/delta buffer is never promoted into
terminal text. Other adapters preserve the partial text their own terminal
contract makes authoritative.

Codex cancellation uses documented `turn/interrupt`. If its process dies or the
stream violates its grammar, the entire App Server client is discarded rather
than reused with uncertain native state. Claude drains an
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
`NoNetworkAgentRuntime`. Deterministic adapter tests use typed fakes plus
sanitized JSON-RPC fixtures for every response/request/event family. The
retained incident fixture contains only item shapes, field names, lengths, and
hashes; conversation text, prompts, credentials, and tool arguments are absent.

CI proves all of these packaging shapes:

- base wheel imports neither optional SDK;
- the base wheel installs the shared Codex WebSocket transport;
- `claude-sdk` wheel extra imports Claude only;
- a no-extras environment exercises the Claude `SdkUnavailable` path.

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
Per route it certifies: one full streamed turn under the route's restrictive
policy (the defaults, plus the `allowed_tools=("*",)` sentinel Codex requires),
a resumed second turn on the same native session, and a structured output turn
— asserting the six-kind grammar, the terminal shape, and normalized
`TokenUsage` on the way through. Codex runs four more turns after the resume.
A live-only observer independently captures the raw native cumulative values
before projection and proves all six invocation-local terminals equal their
fixed-baseline deltas and sum exactly to the final real cumulative delta; the
restored resume snapshot must appear as baseline without being charged. For
Codex it also independently records completed assistant-message phases, requires
a real commentary-plus-final structured turn, and proves the terminal selects
the final-answer item while excluding commentary from structured parsing.

The separate paid Terra containment qualification deliberately asks for native
`exec`/Code Mode and accepts exactly two outcomes: no authority surface appears,
or first-class/defect detection invalidates the session with no accepted
terminal. Both require no sentinel host effect and no credential environment:

```bash
LLM_RUNTIME_LIVE=1 \
LLM_RUNTIME_LIVE_CODEX_ENDPOINT=/run/codex-shared-personal/app-server.sock \
uv run pytest -m live_provider tests/live/test_codex_containment.py
```

## References

- [Codex App Server protocol](https://developers.openai.com/codex/app-server)
- [OpenAI Codex App Server](https://learn.chatgpt.com/docs/app-server)
- [Codex authentication](https://developers.openai.com/codex/auth/)
- [Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)
- [Claude Agent SDK Python](https://code.claude.com/docs/en/agent-sdk/python)
- [Claude Agent SDK sessions](https://code.claude.com/docs/en/agent-sdk/sessions)
- [Claude Agent SDK permissions](https://code.claude.com/docs/en/agent-sdk/permissions)
