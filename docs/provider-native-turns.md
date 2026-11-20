# prepared native turns

`AgentRuntime` owns the provider protocol. consumers own durable journals,
callback execution, effects, approvals, and recovery decisions. codex callbacks
use the configured external app-server socket; workers hold no account state.
there is one engine for contained and callback-bearing codex turns.

## host incarnation control

`provider_runtime.agent_runtime.codex_host` owns protocol revision 1 and its closed
request/response models. `resolve_codex_host(endpoint, target=None)` returns
`CodexHostResolved(target, endpoint, retirement_allowed)`, `CodexHostRetired`, or
`CodexHostUnavailable`. `retire_codex_host(endpoint, target)` returns only the latter
two. each call owns one http connection lifetime, has a five-second transport bound,
and performs no retry. consumers own their encompassing recovery deadline.

`CodexHostUnix(control_socket)` uses http over the private socket; its resolved
native endpoint is the exact sibling `native-{host_id}-{incarnation}.sock`.
`CodexHostNetwork(control_url, native_url, control_token)` pairs http/ws or https/wss;
bootstrap bearer authentication goes only to control. the resolved native endpoint
holds the host's fresh per-incarnation credential. redirect following and ambient
http proxies are disabled. credentials remain secret values; the websocket wire
model deliberately serializes its credential only for an authenticated json reply.

the client requires an exact requested target and native transport. missing or
malformed responses raise redacted `ProtocolDefect`; protocol-version mismatch and
unreachable control are distinct unavailable results. http 401/403 raises redacted
`CredentialRejected`. no response body or credential enters diagnostics.

the supervisor must make incarnation identity inseparable from its actual native
endpoint, certify retirement only after that execution can no longer submit or
continue, and reject retirement of a replacement host. the client does not infer
retirement from process/socket loss or interruption acknowledgement. one successful
resolve permits opening a fresh thread; it never authorizes reconnecting an existing
session to a newly resolved endpoint. the consumer persists identity/proof, not
transport credentials, and independently fences effects/publication. upstream
compute or billing cessation is not part of this contract.

## contract

1. select exact model and reasoning from `model_catalog`; open a
   `CodexCatalogSessionRequest` with its revision and fingerprint. declared
   callbacks may coexist with native built-ins under the explicit frozen
   permission policy. contained codex requests explicitly disable agents and select no native
   execution environments. provider containment semantics participate in the
   catalog definition revision; a config repair rotates consumer definitions.
2. `prepare_turn(session, request, *, attempt_id, input_id, controls)` performs
   pure validation and reserves the local session slot. it freezes
   `submitted_request` and its canonical sha-256 digest. it performs no provider
   i/o. the host durably arms that exact attempt before calling `submit()`.
3. `submit()` is single-use. `AgentNotSubmitted` requires proven writer
   non-entry and quiescence. writer entry latches `AgentUncertain` before a
   possible send. `AgentAccepted` requires an exact correlated native turn
   identity, including a started event or callback received before the start
   rpc acknowledgement. a correlated `turn/start` rpc error −32602 with stock's
   `input_too_large` tag and integer `actual_chars > max_chars > 0` yields
   `AgentSubmissionRejected`. its durable codec preserves those counts, code,
   rpc code and the original attempt. contradictory positive evidence revokes
   authority and raises `ConflictingSubmissionEvidence`. acceptance before the start
   acknowledgment permits immediate callbacks. if that rpc times out, the connection
   closes under the existing operation deadline.
4. `events()` has one consumer. its independent native reader continues while
   storage or a callback handler waits. messages and recorded input drain in
   order before the terminal. pending callbacks and pending transport buffers
   are bounded; controlled turns have no cumulative event or transcript quota.
   the converted fifo holds 4 mib of measured payload, including field names for
   empty values. its sole decoder can stage one validated frame's remaining
   publishable output up to another 4 mib. the validated terminal and python
   overhead are separate. each positive dequeue renews the current waiter's rpc
   idle interval. a stalled consumer fences dispatch. without a recovered seal,
   entered work returns a failed local `turn_timeout` with reason
   `consumer_stalled`. a rejected raw frame still faults the turn.
   after a fence the decoder validates the admitted stream under the earliest
   turn, stop, close or nonrenewing rpc deadline. it withholds callback writes,
   counts display output omitted under pressure, and delivers incorporation
   receipts in order. only those receipts may exceed ordinary credit, under
   the existing 100,000-record and 4 mib limits including native item ids.
   startup refusal is definite only after the bounded original prefix was
   examined. conflicting native evidence remains available without authorizing
   settlement. buffer diagnostics contain bounded identifying and numeric data.
   stream chunk count does not impose a second quota.
   stock can disconnect an upstream response and retry within the same turn.
   on its correlated `error` with `willRetry=true` and `responseStreamDisconnected`,
   assistant ids observed before the retry become ineligible. normalized starts,
   deltas, completions in any phase and explicit raw assistant message ids all
   identify display items. anonymous raw messages supply no identity. the stock
   stream carries no sampling-attempt id. a fresh normalized completion with an
   id unseen before the retry can supply final text. a retry leaves native tools,
   input receipts and other pending items intact. controlled turns allow 100,000
   current and retired display ids and 4 mib of utf-8 identity bytes. a healthy turn
   can exceed that history. a later retry then fails because earlier ids cannot
   all be proved. the provider checks each live callback identity.
   durable host journals own completed callback/input/message identities and
   reject changed replay. written replies and completed callbacks retire pending
   request state; input ids retire only after their native recording. unresolved
   writer-entered input and callback requests remain bounded and tracked.
5. `reply(call, AgentToolReply(...))` accepts only a fresh opaque token issued
   by that handle. each delivery needs its own reply. repeated call ids retain
   identical native name and arguments; durable replay belongs to the host.
6. `steer(input_id=..., input=...)` reports its rpc disposition. an accepted
   acknowledgement is not delivered-input lineage. only the actual correlated
   native user-input item emits `AgentInputRecorded`.
7. `revoke()` synchronously fences new callback replies, steering, and native
   submission. `interrupt()` requests remote stop separately; an acknowledgement
   is not proof that the native turn ended. `close()` quiesces local authority,
   returns `AgentCloseResult`, and closes only its owned session when necessary.
   it cannot undo dispatched actions or stop sibling sessions.

`session_usable(session)` checks local connection, idle ownership, and policy
state. a prior native terminal can remain authoritative on an unusable
connection. the host must cold-bootstrap when that session cannot be reused.
`TurnRequest.timeout_seconds=None` imposes no whole-turn deadline on prepared
native turns; finite rpc timeouts still bound stuck operations.

`prepare_observed_turn` accepts the same arguments and evidence contract as
`prepare_turn`, rejects declared callbacks before submission, and preserves
bounded whole-turn event/text and completed-item validation for isolated calls.
its whole-turn deadline is the lesser of `AgentRuntimeConfig.max_turn_seconds`
and any explicit request timeout; that config does not limit native preparation.
`stream_turn` selects this observational behavior. both public preparation
contracts use the same provider engine; the selection is explicit, never inferred
from output schemas or an empty tool plan.

`codex_native_request_fit(text, *, output, reasoning, input_id, operation="start"|"steer")`
lowers the actual text, output schema and native identities for planning.
`codex_wire_request_fit(submitted)` measures a lowered start or steer directly.
both return `NativeRequestFit(actual_chars, wire_bytes)` with inclusive limits
`max_chars` (1,048,576) and `max_bytes` (4 mib). stock counts unicode scalars in
input text or start's textual tool output. serialized metadata, json escapes
and nontext parts consume wire bytes. planning reserves the largest legal rpc
id. the writer checks the actual envelope before entering transport. stock's
[`RequestId::Integer(i64)`](https://github.com/openai/codex/blob/a956835d020762cb2b570053af06f643a11c0ecc/codex-rs/app-server-protocol/src/rpc.rs)
owns that bound; the client enforces it on its actual counter.

## terminal and recovery

`AgentTerminal.evidence` is required. `NativeTerminalEvidence` binds the
prepared attempt to a native turn/result and an explicit seal revision.
`LocalStopEvidence` preserves accepted or unresolved submission; it is not
native terminal recovery authority. reader-latched `turn.terminal` survives
failure after an already validated native seal. call `await turn.check_submission()`
before independently settling that latch. it checks the bounded correlated start
reply and raises on contradictory evidence, raw queue loss or exceeded output limits. retain the original
seal as evidence. call `check_submission(for_effect=True)` at callback admission
and immediately before executor entry to enforce the turn’s live writer authority.
`events()` applies the same check before yielding a terminal.

the host stores the original terminal before calling `decode_agent_output`
or applying an application schema. `raw_structured_output=None` means absent;
`RawAgentOutput(None)` means supplied json null. invalid supplied raw payload
never falls back to final text. commentary is observable, never executable
structured output.

use `terminal_to_json` / `terminal_from_json` for durable native terminal data.
the closed `agent-terminal.v2` envelope retains exact evidence, raw payload,
usage presence, failure, diagnostics, and the honest `agent-session-ref.v1`
native session reference. `attempt_to_json` / `attempt_from_json` and
`submission_to_json` / `submission_from_json` expose the same closed evidence
values. there are no legacy decoders or invented evidence defaults.

## native capability

contained codex sessions require a dedicated stock `0.171.0` host. host
operations call `materialize_codex_containment_catalog(private_directory)`
and launch the stock process with `-c model_catalog_json="<returned-path>"`.
`qualify_codex_containment_host(endpoint)` checks the public `config/read`
declaration. callers that pass `required_service_tier` also require that exact
effective tier from the host's `sessionFlags` layer in the same response.
when the host pins a tier this way, the sdk verifies it against the initial thread
settings and every native settings update. session preparation repeats the catalog check before creating a native
thread: the canonical catalog basename and `sessionFlags` startup origin must
match; the initialize user-agent build version must be exactly `0.171.0`.
per-thread catalog overrides are documented upstream no-ops and are
never used to claim containment. the trusted host owns the materialized bytes;
this declaration is not remote attestation against a hostile local operator.

the provider retains the complete official catalog from
[`rust-v0.171.0`](https://github.com/openai/codex/blob/rust-v0.171.0/codex-rs/models-manager/models.json).
source hash: `d50b6749506d97e5f334dbabcb666eb1689f423cf3e88cfabdcf3c536b873068`.
restricted hash: `552646113928209983c67c598ce80c534652b4916d1b0595d2d175ee7ee6627b`.
the contained catalog v3 sets `tool_mode=direct` and `apply_patch_tool_type=null`,
clears native clock, async-user, literature search and multi-agent tool selectors,
and removes the `ultra` preset. stock codex combines this preset with automatic
delegation and a model-specific inference effort. for 6.2 sol and astra that
effort is `xhigh`. contained sessions disable delegation. their largest
selectable inference effort is `max`. each retained source default must still
name an offered effort. the pinned catalog's upgrade records name models only.

the compiler rejects unknown tool selectors, patch types and tool modes.
the original apache license and notice accompany the asset. model inventory
updates require an explicit provider/binary/catalog upgrade.
ordinary coding sessions use a separate endpoint. agent model catalog v4 reports
the input modalities a host declares, including audio on an unrestricted host.
turns accept text and local images. the contained stock catalog lists both.

the native workspace extension retains this current catalogue and typed endpoint
contract. a workspace session may combine declared callbacks with native shell
and file operations; native observations do not become callback receipts.
initial and ongoing settings must exactly match the frozen request, and inherited
mcp servers are explicitly suppressed before declared servers are overlaid.
isolated builtin-disabled sessions retain their existing restrictions.
the original workspace consumer qualified stock `0.160.0` at provider source
`ea9f82025632926a7c7e47d6333fb9346f6ee86c`; that immutable source remains available
for its existing pin. integration with the earlier `0.165.0` library contract
does not upgrade that consumer. controlled protocol checks establish source
behavior; actual native qualification is recorded separately and is not inferred
from those checks.

the stock static catalog manager does not replace this catalog during network
refresh. catalog revision participates in the provider definition identity.
the official [`model_catalog_json` setting](https://developers.openai.com/codex/config-reference)
describes this startup seam. model listing is source metadata, not a promise
that every listed model is entitled or successfully qualified for the account.

new contained threads opt into the installed raw response event stream.
their builtin-disabled configuration sets `incremental_tools=false`: declared
callbacks stay fixed for the thread, and native tool-declaration updates do not
enter the closed raw-event grammar. responses lite still owns its fixed request
tool encoding; disabling updates does not change that provider wire format.
declared direct function calls remain host callbacks. a call to an undeclared
function that codex itself answers with exactly `unsupported call: <name>` ran
nothing: it is a model error, emits no event, and the turn continues
([decision](decisions/2026-10-13-codex-refused-unpublished-calls.md)). any
other answer, or none before turn completion, makes it a typed native
authority event that poisons terminal acceptance, as unexpected raw custom
tools always are; unknown raw item types fail closed. the reader does not
parse javascript or treat code-mode wrappers as declared callback authority.
loaded threads retain the raw-event selection; stock resume/fork exposes no
new raw-event opt-in, so the pinned startup policy remains the authority ceiling
after a cold server restart.

session tool declarations use frozen `CanonicalTool` values. native callback
names are valid request-local aliases, while canonical portable identity stays
with `llm-tools` and the kernel. undeclared authority, rerouted models, changed
selected settings, invalid identities, malformed frames, and impossible
callback lifecycles fail closed. output/context reservations are host
admission facts; the installed native protocol has no enforceable token-ceiling
field. do not claim otherwise.

codex output-schema preflight rejects actual `oneOf` and the established
unsupported composition keywords before native i/o. it requires an object
root, closed object properties, and all object fields required. it traverses
schema members, not arbitrary annotation or property-name strings. no schema
is rewritten. the restrictions follow the
[official strict-output contract](https://developers.openai.com/api/docs/guides/structured-outputs)
and the actual installed jarvis failure. this is not an exhaustive local copy
of the remote validator; undocumented restrictions remain original native
failures, not proven non-submission or automatic retry permission.


## 2026-11-18 stock input qualification

the predecessor classified a structured refusal as uncertain and prepared an over-limit
request. stock `0.160.0` with a disposable home and loopback-only model target accepted
1,048,576 mixed unicode scalars in two input items and rejected 1,048,577. the same
stock error counts were verified for steer, string tool output and array tool output.
no provider model call was needed. temporary socket probes exercised the independent
4 mib edge, proved non-entry, malformed responses, connection loss, both arrival
orders of a contradictory native seal and the bounded missing-start-ack case.

integration with main kept its stock `0.165.0` endpoint and containment policy.
on that binary, start and steer each accepted 1,048,576 mixed unicode scalars
and returned the correlated `-32602` / `input_too_large` response at 1,048,577,
with both integer counts. start returned the same refusal for a string tool
output and an `input_text` array tool output at 1,048,577. the provider classified
each original response. the disposable home used only a loopback model fixture.
