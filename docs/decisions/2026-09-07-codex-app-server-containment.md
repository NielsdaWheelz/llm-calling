# Codex App Server containment maintenance record

Status: accepted for the immutable maintenance line

Date: 2026-09-07

Base: `2cfed97ee5b9b8eb11103b0575eb7f29de00a0bd`

Publication: branch `maintenance/codex-app-server-containment`, tag
`codex-app-server-containment-2cfed97-v1`

## Decision

The public `(codex, sdk)` route keeps its name and session-reference schema, but
provider-runtime now starts the exact bundled Codex 0.144.4 executable as a
documented App Server `stdio://` JSONL peer. It owns every client response,
server notification, and server-initiated request. It does not use or patch the
opaque `AsyncCodex` request loop.

The certified `CodexNativeOptions(builtin_tools="disabled")` posture requires:

- `openai-codex`, `openai-codex-cli-bin`, and executable-reported version
  exactly 0.144.4;
- local ChatGPT account authentication only;
- read-only filesystem, disabled network, unconditional approval denial,
  `allowed_tools=("*",)`, empty copied environment, no MCP, and no additional
  filesystem roots;
- the existing private state root, replacement child environment, process group,
  strict structured-output parse, authoritative completed-message selection,
  invocation-local usage accounting, resume identity, and cancellation rules.

The feature-off configuration disables every configurable 0.144.4 execution,
integration, app, Web, shell, Code Mode, MCP-app, hook, skill, plugin, memory,
collaboration, environment-context, and permission-instruction surface audited
in the pinned runtime. This is not proof that native Code Mode computation is
absent before execution. The accepted contract is containment and detection:
the child has no API credentials, is offline and read-only, and the first
observable authority event poisons the turn and session before any terminal can
be accepted.

The exact pinned executable's non-experimental generated schema was audited
alongside the current public documentation: 87 client-request methods, 10
server-request methods, 68 notification methods, and 18 `ThreadItem` variants.
The production lane emits only `initialize`, `account/read`, `thread/list`,
`thread/read`, `thread/start`, `thread/resume`, `thread/fork`, `turn/start`, and
`turn/interrupt`, plus the required `initialized` notification. All ten stable
server requests are explicitly denied or forbidden. All 18 stable item variants
are explicitly inert or authority-bearing; retained custom-call variants are
handled in addition. Every other notification is either named below or reaches
the fail-closed default.

## Protocol classification contract

| Wire family | Explicit classification | Response/projection | Fail-stop rule |
|---|---|---|---|
| Client response `{id,result}` | Correlated only to one pending integer client id | Completes exactly that request | Missing, duplicate, boolean/string, completed, or unknown id is `ProtocolDefect` |
| Client response `{id,error}` | Correlated provider error | Sanitized `CodexAppServerResponseError` | Malformed code/message/data or mixed result/error is `ProtocolDefect` |
| Pre-turn notifications | `account/updated`, rate limits, config/deprecation/warning, thread start/status/name/goal metadata, legacy compaction, MCP startup, remote-control status; restored token usage separately validated | Inert, bounded replay or usage baseline | Every other pre-turn notification/request is `ProtocolDefect` |
| Turn/text observations | `turn/started`; reasoning/plan deltas; status/rate-limit/model/moderation metadata; warnings/errors; resolved-request notices; `agentMessage` delta | Explicitly inert `AgentNative`, diagnostic, or `AgentText` | Bad thread/turn/item/request identity is `ProtocolDefect` |
| Thread metadata observations | Documented name/goal updates and legacy compaction | Explicitly inert, bounded `AgentNative` with an exact thread id | Missing or changed thread id is `ProtocolDefect`; archive, unarchive, close, or delete notifications terminate the session |
| Inert item lifecycle | `userMessage`, `agentMessage`, `plan`, `reasoning`, review-mode entry/exit, context compaction | No authority event; completed agent messages feed authoritative final selection | Unknown type, duplicate id, type drift, or unfinished lifecycle is `ProtocolDefect` |
| Native authority item lifecycle | command/file/MCP/dynamic tools, Web search, image view/generation, hook prompt, collaboration/sub-agent activity, sleep, and bounded defensive legacy output items | `AgentToolUse(started/updated/completed)` with exact correlated identity | Strict posture latches authority; terminal is rejected and session invalidated |
| Retained native custom call | `custom_tool_call` start/completion plus `custom_tool_call_output` completion, including `name="exec"` | One `AgentToolUse` lifecycle keyed by `call_id` | Missing, duplicate, reordered, or changed call/item identity is `ProtocolDefect`; strict terminal cannot survive |
| Command/file/process output | command/file output deltas, terminal interaction, patch update, turn diff, process output/exit | `AgentToolUse(updated/completed)` | Output without matching active identity is `ProtocolDefect` |
| Approval review notifications | auto-approval review start/completion | `AgentToolUse` authority lifecycle | Strict terminal cannot survive; malformed review identity/status is `ProtocolDefect` |
| Command/file approvals | v2 command/file request plus legacy `execCommandApproval`/`applyPatchApproval` | Exact documented decline/denied result, then `AgentPermissionRequest(decision="deny")` | Identity mismatch or unresolved request is `ProtocolDefect` |
| Permission/user-input/MCP elicitation | permissions approval, request-user-input, MCP elicitation | Empty permission grant, empty answers, or explicit MCP decline; always `AgentPermissionRequest(decision="deny")` | Strict terminal cannot survive; malformed identity is `ProtocolDefect` |
| Dynamic custom tool request | `item/tool/call` | `{contentItems:[],success:false}` and correlated `AgentToolUse(updated)` | Must match a started dynamic item; strict terminal cannot survive |
| Experimental host-time request | `currentTime/read`, disabled at initialize | JSON-RPC error; failed first-class `AgentToolUse` is retained before transport failure | Always terminates the turn/session |
| Credential/attestation callbacks | token refresh and attestation generation | Named JSON-RPC error; no token, attestation, or default object supplied | Always terminates the connection |
| Unknown server request | Any future method carrying an id | JSON-RPC `-32601`; never `{}` | Immediate `ProtocolDefect`, process/session teardown |
| Unknown item/notification | Any type/method outside the explicit production allowlist | No benign fallback and no generic `AgentNative` | Immediate `ProtocolDefect`, process/session teardown |
| Turn completion | Native completion evidence followed by the owned terminal | Last completed `final_answer`, else last unknown-phase completed message; strict JSON parse and invocation-local usage | Active lifecycle/request or any latched strict authority rejects the terminal |

All inbound JSON must be one bounded UTF-8 object per line with unique keys and
finite JSON values. Responses, notifications, and server requests cannot mix
fields. Server request identities are type-sensitive and unique for the whole
connection. A resume usage replay may race behind the response; only one exact
stale prior-turn identity may establish the still-unknown baseline, without
being charged, before every usage frame again requires the active turn id.
There is no automatic approval and no default response body.

The larger documented App Server surface was audited. Control-plane methods the
production lane never calls—login/logout, filesystem watch/mutation, external
config import, plugin/marketplace management, realtime audio, remote-control
management, thread archive/delete/settings mutation, Windows sandbox setup,
and similar notifications—are deliberately not inert. If any arrives on this
lane, the forbidden/unknown-message rule terminates the session. Goal and name
notifications are the narrow exception: the pinned server can replay them on
resume, so they are inert only with an exact session thread identity. App calls
themselves are covered by MCP/dynamic/native tool authority items;
`app/list/updated` is not an app-call authority grant and remains forbidden as
out-of-lane protocol drift.

## Compatibility and API impact

No public class, method signature, event type, route tuple, session-ref schema,
or transport literal changed. Existing serialized Codex refs continue to use
`transport="sdk"` and resume without rotation. Claude and every provider engine
are unchanged.

One intentional behavioral narrowing applies: a request using
`builtin_tools="disabled"` that also asks for writable/full filesystem,
network, provider review, copied environment, MCP, or additional roots now fails
as `UnsupportedCapability` before Codex starts. This posture was never safely
certified for those combinations.

The `codex-sdk` extra remains because its public enums/version and its matched
bundled executable are the certified dependency surface. It no longer grants
the SDK ownership of the production request loop.

## Trade-offs and remaining limitations

- Native Code Mode is detected/contained, not proven prevented. A computation
  may happen before its first observable item/request event.
- Read-only prevents host mutation but is not a host confidentiality boundary;
  run Codex as a dedicated OS user/container where host read isolation matters.
- Fail-closed protocol drift sacrifices forward compatibility: a new harmless
  item or request causes outage until audited and explicitly classified.
- Exact 0.144.4 certification in disabled-builtins mode delays upgrades but
  prevents an unaudited feature map from silently widening authority.
- Denying a tool request can still produce provider-side model usage. The
  runtime never retries the stateful turn.
- A protocol defect intentionally produces no `AgentTerminal`; accepting one
  after authority uncertainty would recreate the incident.

## Evidence and migration

Deterministic fixtures cover success/error responses, every documented server
request outcome, inert and authority notifications/items, the sanitized retained
custom-exec shape, dynamic tools, approvals, unknown future messages, malformed
and reordered identities, cancellation, process death, resume, and clean
shutdown. Paid evidence is recorded under `tests/live/evidence/` and contains no
prompt, response, credential, native payload, tool argument, or path.

The 2026-09-08 Codex route matrix and dedicated paid Terra containment probe
passed and are retained. The default two-route live matrix was also attempted,
but the unchanged Claude route returned `backend_failed` because the available
temporary Claude profile was not authenticated; no new Claude evidence is
claimed. This maintenance tag is therefore paid-qualified for its changed Codex
lane only. A downstream release policy requiring fresh paid evidence for every
unchanged route must rerun the default matrix with an enrolled Claude profile.

`llm-agent-kernel` migration is dependency-only:

1. Pin provider-runtime to tag
   `codex-app-server-containment-2cfed97-v1` (and its immutable commit) from
   `git+ssh://git@github.com/NielsdaWheelz/llm-calling.git` and refresh the lock.
2. Keep `(backend="codex", transport="sdk")`; do not rotate persisted
   `AgentSessionRef` values.
3. For the contained Jarvis lane, construct
   `PermissionPolicy(filesystem="read_only", network="disabled",
   approval="deny", allowed_tools=("*",), environment=())` and
   `CodexNativeOptions(builtin_tools="disabled")`, with no MCP servers or
   additional directories.
4. Keep rejecting `AgentToolUse` and `AgentPermissionRequest`; do not add native
   method-name parsing or promote `AgentNative` commentary to tool input.
5. Treat `ProtocolDefect` as fail-stop: discard the runtime session and do not
   accept, synthesize, or replay a terminal for that turn.
6. Re-run the kernel's retained incident, calendar-read, session-resume, and
   cancellation suites. No kernel event vocabulary change is required.

## Sources

- [Codex App Server protocol](https://developers.openai.com/codex/app-server)
- [Codex SDK overview](https://developers.openai.com/codex/codex-sdk)
