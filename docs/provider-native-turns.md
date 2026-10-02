# prepared native turns

`AgentRuntime` owns the provider protocol. consumers own durable journals,
callback execution, effects, approvals, and recovery decisions. codex callbacks
use the configured external app-server socket; workers hold no account state.
there is one engine for contained and callback-bearing codex turns.

## contract

1. select exact model and reasoning from `model_catalog`; open a
   `CodexCatalogSessionRequest` with its revision and fingerprint. native
   built-ins and web remain disabled when portable callbacks are declared.
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
