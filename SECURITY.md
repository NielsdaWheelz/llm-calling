# Security

Report vulnerabilities privately to the maintainers. Do not open a public issue
with exploit details. Include the affected version, reproduction steps, expected
impact, and any suggested fix.

## Trust model

`provider_runtime.agent_runtime` is for one trusted user on a local machine. It
drives an already-enrolled Codex or Claude Code account through documented
vendor surfaces. Codex uses the public App Server protocol; Claude uses the
official SDK. It is not a hosted subscription proxy, multi-tenant sandbox,
login service, or token broker.

The only shipped routes are `codex:sdk` and `claude:sdk`; the Codex route name
selects the matched runtime distribution; provider-runtime owns its App Server
stdio transport. There is no fallback. Session
authentication is subscription-only: each
route rejects named API-key and secret-reference session credentials before any
secret could be resolved, and the child-environment builder refuses to forward
them structurally, so no code path exists that places an API key in an agent
child. Subscription pool exhaustion terminates the turn with the
`AgentQuotaExhausted` failure value — the lane never overflows onto API-rate
credentials.

Report any raw credential in an event, diagnostic, exception, persisted session
reference, generated file, or child command line as a vulnerability.

## State and environment isolation

Each local account profile is isolated at:

```text
<state_root_base>/<backend>/<profile_key>
```

The base must be an existing normalized absolute directory that is not group- or
world-writable. Runtime-created directories are `0700`. The child environment is
rebuilt from a fixed allowlist; the operator's environment is not inherited.
`HOME`, `PATH`, locale, temp, `CODEX_HOME`, and `CLAUDE_CONFIG_DIR` are
runtime-owned. Credential-class, provider-selection, and process-control names
cannot be copied through `PermissionPolicy.environment` or MCP aliases.
Credential-class is every provider API key the operator may have exported, not
only the two backends' own auth variables: the agent lane has no use for any of
them, so none of them reaches a child.

For Codex, provider-runtime starts the exact bundled executable as
`app-server --listen stdio:// --strict-config` with the selected environment as
a complete replacement and owns every JSONL response, notification, and server
request. The documented account response must report ChatGPT subscription auth;
ambient API keys do not reach the runtime. Credential-refresh and attestation
callbacks are refused with JSON-RPC errors and fail the session.

For Claude, a shell router or version-manager shim can overwrite
`CLAUDE_CONFIG_DIR` and defeat isolation. Point
`AgentRuntimeConfig.claude_executable` at the real executable. The runtime
resolves the path and probes the executable in the runtime-owned child
environment before any session starts. Version drift against the vetted build
is reported as one warning and met with behavioral probes (effective
configuration verification on the wire), never silently trusted; an executable
or SDK that is missing or does not answer remains a typed availability
failure.

## SDK process-group launchers

The Claude Agent SDK does not expose a `start_new_session` process option. The
runtime therefore creates one content-addressed executable launcher in the
runtime-owned `<state_root_base>/claude/` directory. The launcher calls
`setsid()` and then `execv()` through the SDK's public `cli_path` option, making
Claude Code and descendants one process group the runtime can terminate.

The launcher directory and file are `0700`, owned by the current uid, outside
the child's `HOME`, configuration root, and sandbox. The file is published by
atomic rename and byte-compared after publication. It contains no credentials
and does not recreate or inspect the SDK's arguments.

A writable launcher directory, content mismatch, or child-controlled launcher
path is a local privilege-escalation risk and should be reported.

Codex no longer needs an SDK shim: the owned App Server process is launched
directly by the existing process-group supervisor. On close or protocol defect,
the supervisor signals the group and escalates to `SIGKILL`, preventing Codex
tools or descendants from outliving the connection.

## Policy is fail-closed

Defaults are read-only filesystem, disabled network, denied approvals, no
built-in tools, and no copied environment. Full filesystem access, unrestricted
network, and unconditional approval each require an exact
`UnsafeConfirmation`; extra acknowledgements are rejected.

Codex `provider_review` delegates escalation review through the App Server's
`approvalsReviewer="auto_review"` policy. It is not equivalent to Claude's
caller-owned `ask` mode, and per-turn narrowing cannot swap one reviewer for
the other. A caller should
treat provider review as permission for the provider's maintained policy to
approve actions within the separately selected filesystem/network sandbox.

Codex exposes no typed per-name built-in filter, so `allowed_tools=("*",)` is
the only portable-policy spelling. `CodexNativeOptions(builtin_tools="disabled")`
is a separate exact-version containment posture: it sets the public 0.144.4
feature-off configuration and requires read-only filesystem, disabled network,
deny approvals, empty copied environment, no MCP, and no additional roots.
Provider review is rejected in that posture. Claude accepts exact tool names,
rejects glob patterns and the two network-reaching built-ins (`WebFetch`, `WebSearch`)
outright, and verifies the effective tool set the backend reports at session
start against the requested set. It does not reject a name that is simply
unknown to the CLI; such a name grants nothing, so the effect is a narrower
session than the caller wrote, never a wider one.

## MCP and credentials

MCP secrets are reference-only. The runtime resolves them at the process
boundary into opaque child-environment aliases; public request, event, result,
reference, log, and exception values never contain the resolved material.

Stdio MCP executes a caller-selected local program outside provider sandbox
attestation. It is accepted only under explicitly confirmed full filesystem and
unrestricted network access. Under that policy, same-uid model-generated
commands can inspect peer processes and should be assumed able to read every
stdio MCP credential. Do not use credentialed stdio MCP as a security boundary;
use a dedicated OS user or container.

The one confined remote-MCP shape is Claude streamable HTTP MCP under an exact
hostname allowlist, with no credential references supplied by this package.

Codex streamable HTTP MCP is the shape that carries reference
headers/environment. Codex cannot enforce a hostname allowlist, so remote MCP
needs unrestricted network. The shape to use is `workspace_write` +
`unrestricted`, acknowledging `network_unrestricted` only: the route writes
`sandbox_workspace_write.network_access = true`, which is the one network toggle
the Codex config carries. `full_access` + `unrestricted` remains accepted but
acknowledges filesystem authority remote MCP does not need.

`workspace_write` confines writes, not reads and not egress. Codex's
workspace-write sandbox is read-only access plus write access to the session
`cwd`, its `additional_dirs`, and by default `/tmp` and `$TMPDIR`; everything
else on the host stays readable. With `network_access = true` there is no
hostname allowlist and no port restriction, so anything the session can read it
can also send. Treat a Codex remote-MCP session as exfiltration-capable: run it
as a dedicated OS user or in a container, or use the Claude shape instead.

`read_only` + network is refused rather than approximated. The Codex config
exposes a network toggle only under `sandbox_workspace_write`, with no read-only
counterpart, so the route will not claim a read-only sandbox can reach the
network.

## Bounds, redaction, cancellation, and cleanup

SDK messages, event count, text, final output, diagnostics, turn duration, and
cleanup are bounded. Output-limit failures terminate the turn and discard
uncertain native session state. The runtime never retries a stateful turn.

Only explicitly inert native frames cross as `AgentNative`, and only after
bounded recursive redaction. Known command, file, MCP/app, Web, dynamic/custom
tool, process, hook, image, sub-agent, and approval-review activity is
`AgentToolUse`; permission requests are `AgentPermissionRequest` and are denied.
Unknown server requests, unknown item lifecycle types, malformed identities,
uncorrelated responses, and protocol drift are `ProtocolDefect` and invalidate
the session. A future item type is forbidden by default.

Code Mode/native `exec` is not proven absent before execution. The accepted
contract is containment and detection: the process is credentialless,
read-only, and offline, and its first observable authority event poisons the
turn so no later terminal can be accepted. This still permits native
computation before that event and does not make the Codex read-only sandbox a
host confidentiality boundary.

Codex cancellation uses documented `turn/interrupt`; Claude uses the SDK's
native interrupt operation. Claude drains the interrupted tail or invalidates
the client before reuse. Codex discards the entire App Server client after
protocol/transport uncertainty. Runtime close terminates active work and closes
all clients.

The public event grammar is six kinds and every valid stream requires exactly one
`AgentTerminal` after a started turn. Identity mismatches, events from retired turns, malformed
known notifications, and post-terminal frames are defects rather than tolerated
input. A protocol defect deliberately ends without a terminal because accepting
one would conceal uncertainty or forbidden authority.

## Optional dependency boundary

The base wheel imports neither `openai_codex` nor `claude_agent_sdk`. Missing
extras fail as typed `SdkUnavailable`; arbitrary import errors are not exposed.
CI installs each extra independently, installs both together, and exercises both
real absent-module paths in a no-extras environment.
