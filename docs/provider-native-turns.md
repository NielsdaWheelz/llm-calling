# prepared native turns

`AgentRuntime` owns the provider protocol. consumers own durable journals,
callback execution, effects, approvals, and recovery decisions. codex callbacks
use the configured external app-server socket; workers hold no account state.
there is one engine for contained and callback-bearing codex turns.

## contract

1. select exact model and reasoning from `model_catalog`; open a
   `CodexCatalogSessionRequest` with its revision and fingerprint. native
   built-ins and web remain disabled when portable callbacks are declared.
   contained codex requests explicitly disable agents and select no native
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
   rpc acknowledgement. generic rpc errors, timeout, and absent ids do not
   prove non-submission.
4. `events()` has one consumer. its independent native reader continues while
   storage or a callback handler waits. messages and recorded input drain in
   order before the terminal. pending callbacks and pending transport buffers
   are bounded; controlled turns have no cumulative event or transcript quota.
   completed item ids are not retained. exact live callback identity is checked;
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

`codex_native_request_fits(text, *, output, reasoning, input_id)` checks the exact contained
text-turn serialization before opening a stock `0.160.0` thread. it uses the same
serializer and byte bound as `prepare_turn`, including json escaping and output
schema. native threads use canonical uuid strings; the host supplies its input id. this
is a pure transport fit check, not token estimation or inference admission. the
transport's own json encoder measures the complete envelope with the largest legal
positive rpc id, so a fitting request fits every counter value before a connection
is opened. stock's
[`RequestId::Integer(i64)`](https://github.com/openai/codex/blob/a956835d020762cb2b570053af06f643a11c0ecc/codex-rs/app-server-protocol/src/rpc.rs)
owns that bound; the client enforces it on its actual counter.

## terminal and recovery

`AgentTerminal.evidence` is required. `NativeTerminalEvidence` binds the
prepared attempt to a native turn/result and an explicit seal revision.
`LocalStopEvidence` preserves accepted or unresolved submission; it is not
native terminal recovery authority. reader-latched `turn.terminal` survives
failure after an already validated native seal. no reconnection or second
provider submission is performed to manufacture recovery proof.

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

contained codex sessions require a dedicated stock `0.160.0` host. host
operations call `materialize_codex_containment_catalog(private_directory)`
and launch the stock process with `-c model_catalog_json="<returned-path>"`.
`qualify_codex_containment_host(socket_path)` checks the public `config/read`
declaration. session preparation repeats this check before creating a native
thread: the canonical catalog basename and `sessionFlags` startup origin must
match; the initialize user-agent build version must be exactly `0.160.0`.
per-thread catalog overrides are documented upstream no-ops and are
never used to claim containment. the trusted host owns the materialized bytes;
this declaration is not remote attestation against a hostile local operator.

the provider retains the complete official catalog from
[`rust-v0.160.0`](https://github.com/openai/codex/blob/a956835d020762cb2b570053af06f643a11c0ecc/codex-rs/models-manager/models.json).
source hash: `fd219bd9f061278275f528939f82f54d2eb97df4b25c23b022adbe48813d920b`.
restricted hash: `b8b588f4b03c8e08fdb7e2994c03578d9b94bbc01665b7adb6490bd234b3cf54`.
the compiler preserves every field except `tool_mode=direct` and removal of
the established native clock/async-user selectors. unknown selectors or tool
modes fail before materialization. the original apache license and notice
accompany the asset. this policy changes the whole endpoint: native code mode,
clock and async-user interactions are unavailable; model inventory updates
require an explicit provider/binary/catalog upgrade. exact model identifiers,
efforts, instructions, context metadata and account routing remain intact.
ordinary coding sessions use a separate endpoint.

the stock static catalog manager does not replace this catalog during network
refresh. catalog revision participates in the provider definition identity.
the official [`model_catalog_json` setting](https://developers.openai.com/codex/config-reference)
describes this startup seam. model listing is source metadata, not a promise
that every listed model is entitled or successfully qualified for the account.

new contained threads opt into the installed raw response event stream.
declared direct function calls remain host callbacks. unexpected raw custom
tools or undeclared functions become typed native authority events and poison
terminal acceptance; unknown raw item types fail closed. the reader does not
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
