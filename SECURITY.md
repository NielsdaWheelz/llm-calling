# Security

Report vulnerabilities privately to the maintainers. Do not open a public issue
with exploit details. Include the affected version, reproduction steps, expected
impact, and any suggested fix.

## Trust model

`provider_runtime.agent_runtime` is for one trusted user on a local machine. It
drives an already-enrolled Codex or Claude Code account through the vendor's
official SDK. It is not a hosted subscription proxy, multi-tenant sandbox, login
service, or token broker.

The only shipped routes are `codex:sdk` and `claude:sdk`. There is no raw CLI or
direct protocol fallback. Session authentication is subscription-only: each
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

For Codex, the selected environment is passed with public `CodexConfig.env`, but
the SDK overlays it on `os.environ`. The adapter therefore points public
`CodexConfig.codex_bin` at a private content-addressed launcher. That launcher
forwards the SDK-owned arguments to the exact bundled runtime while replacing
the environment with the selected-name allowlist. The SDK account must report
ChatGPT subscription auth; ambient API keys do not reach the runtime.

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

Codex's launcher also calls `setsid()`, then supervises the matched bundled
runtime in that private group. The SDK holds the supervisor pid and retains
ownership of stdin/stdout and arguments. On SDK termination, the supervisor
signals the whole group and escalates to `SIGKILL`, preventing Codex tools or MCP
descendants from outliving the client. The launcher embeds only the runtime path
and allowed environment names, never their values.

## Policy is fail-closed

Defaults are read-only filesystem, disabled network, denied approvals, no
built-in tools, and no copied environment. Full filesystem access, unrestricted
network, and unconditional approval each require an exact
`UnsafeConfirmation`; extra acknowledgements are rejected.

Codex `provider_review` delegates escalation review to the official SDK's
`auto_review` policy. It is not equivalent to Claude's caller-owned `ask` mode,
and per-turn narrowing cannot swap one reviewer for the other. A caller should
treat provider review as permission for the provider's maintained policy to
approve actions within the separately selected filesystem/network sandbox.

Codex exposes no built-in tool filter through its public SDK — neither specific
names nor an off switch. The route rejects both rather than pretending either
was enforced, so `allowed_tools=("*",)` is the only spelling a Codex session
may carry: an empty tuple would assert that built-ins are disabled when the SDK
leaves them on. A Codex session is confined by its sandbox mode and approval
mode, never by its tool list. Claude accepts exact tool names, rejects glob
patterns and the two network-reaching built-ins (`WebFetch`, `WebSearch`)
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

Native frames cross the public boundary only as `AgentNative` events whose
payload passed the bounded, recursive redaction: credential-shaped keys are
dropped wherever they appear, every retained string is sanitized and
length-bounded, and a payload exceeding its depth/item/byte bounds is dropped
whole. There are no per-version field allowlists; redaction is by key shape,
not by schema table.

Cancellation uses the SDK's native interrupt operation. Claude drains the
interrupted tail or invalidates the client before reuse. Codex discards the SDK
client after protocol/transport uncertainty. Runtime close terminates active
work and closes all clients.

The public event grammar is six kinds and requires exactly one `AgentTerminal`
after a started turn. Identity mismatches, events from retired turns, malformed
known SDK notifications, and post-terminal frames are defects rather than
tolerated input.

## Optional dependency boundary

The base wheel imports neither `openai_codex` nor `claude_agent_sdk`. Missing
extras fail as typed `SdkUnavailable`; arbitrary import errors are not exposed.
CI installs each extra independently, installs both together, and exercises both
real absent-module paths in a no-extras environment.
