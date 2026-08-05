# Agent Runtime

## Scope

This document covers agent run and model-call protocol rules.

## Rules

- The only direct model-facing tool is `run`.
- Every agent model response must call exactly one direct tool.
- Persist exactly one model-call record per completed provider call. Store only
  the newly generated assistant turn from the provider response transcript;
  never persist provider-generated tool continuation messages as model response
  history. If the provider returns no response messages, persist an explicit
  blank assistant turn so the protocol layer can emit normal no-tool-call
  protocol feedback after the model call is recorded. If a response has no
  direct tool call, has multiple direct tool calls, targets an unsupported
  direct tool, or has invalid direct tool input, persist that response, then
  append a protocol-error context event telling the model what was wrong.
- Start an idle drain only when unread context events include at least one
  waking event. Once a drain is running, keep making model calls until the idle
  request applies, a defect occurs, or the model-call limit is reached.
- Model-call sequence numbers are local to the agent. Address
  transcript/history ranges by agent id and model-call sequence; run id records
  which run produced the model call.
- Durable agent identity lives as profile metadata: title, icon, and summary.
  These fields describe the whole agent history through the stored profile
  model-call boundary plus current agent notes, not the latest running step.
- Progress presentation is only for current activity labels. It may generate
  the focus chip and step chip for progress updates; it must not name the
  agent/thread or own durable profile metadata.
- Profile generation must reuse the existing transcript/history system:
  existing profile for continuity, current agent notes, prehistory summary,
  visible history chunk summaries, and recent unchunked transcript through a
  durable model-call sequence boundary.
- Durable project or resource summary is generated from current project notes
  and current agent summaries, coalesced through a summary drain.
- Regenerate an agent profile after the first persisted model call, after an
  idle request is successfully applied, and when current agent notes no longer
  match the stored profile source checksum. The first profile may target the
  latest model call immediately; later profiles target the latest model call
  from an idle-finalized run, even if newer active-run work has already started.
  Coalesce through the profile-generation drain claim and skip work when the
  generated-through boundary plus source checksum already cover the eligible
  target boundary and current agent notes.
- Every model call has a context-event batch. The batch may be empty; store the
  missing start sequence as database null and record the latest context-event
  sequence consumed by the model call. Decoded model-call/domain values expose
  the missing start sequence through the owned absence representation.
- Store agent context events as the canonical chronological model-visible facts.
  The model-call request payload stores only the exact provider request made for
  audit/debug. Do not store a second synthetic packed transcript copy inside the
  model-call request.
- Context-event append operations only append context events. Visible progress
  is a separate producer-owned projection, appended explicitly when the
  production path is user-visible activity. Applied
  idle-until-input lifecycle events written after the final user-visible
  message are transcript facts and must not create visible progress activity
  after that final message.
- Non-waking lifecycle context events, such as `run_completed`,
  `message_shown`, and `idled_until_new_input`, are not visible progress.
  `idle_request_not_applied` is also context-only scheduling feedback; it wakes
  the agent but does not create visible progress. When a production path appends
  a mixed context-event tuple, project only the user-visible activity or failure
  facts into progress.
- Store each context event's scheduling classification when the event is
  appended. Scheduling code should read that metadata, not re-derive wake
  behavior from rendered prompt JSON.
- Do not keep protocol-error feedback as transient in-memory-only context.
  Future prompts and history compaction should see the same invalid-response /
  feedback sequence the model saw.
- Build transcript/history groups from context-event rows using each model
  call's consumed context-event range. Transcript/history source text is a
  semantic context-event stream, not provider-shaped `user`/`tool` messages.
  Reconstruct provider-shaped continuation messages only at the provider replay
  boundary, where the previous model response is available to choose the
  required user-message versus tool-result container. History chunks end only
  after a complete model-call input group.
- Use `run` for work. Inside the guest runtime, call resource functions through
  resource-scoped namespaces, helper functions through top-level helper
  namespaces, render new internal agent context through observe-value calls,
  request short waits through sleep calls, request standalone user-visible
  output through show-message calls, request final user-visible output plus
  drain idle through idle-until-input calls, and update durable agent state
  through agent-state functions.
- Inside `run`, use resource-scoped stash functions for scoped JSON stash values.
- The `run` tool itself executes as a durable replayable multi-mutation. Pure
  guest-runtime code may re-run from the beginning after a crash; all durable
  effects must cross the host-function boundary as managed operations.
- Keep guest-runtime ambient execution deterministic. Expose nondeterminism only
  through managed host functions whose results memoize in the durable frame.
- Do not enforce an ambient wall-clock timeout around guest-runtime execution. Guest compute
  is bounded by instruction, CPU, output, and syscall-count limits; host-call
  wait time is budgeted through managed timestamp entries at syscall
  boundaries.
- If a host call pushes the managed host-call time budget over its limit, return
  that call's result to the guest runtime. Refuse later host calls through the
  normal catchable host-function error path while allowing pure guest control
  flow to finish.
- Treat the guest runtime as a deterministic control-flow VM. The exposed
  standard library is curated for durability; identity, time, randomness, host
  locale, and native pointer formatting must be unavailable or routed through
  managed host functions.
- Observe-value, sleep, show-message, and idle-until-input calls are agent-run
  functions, not agent-state functions.
- Sleep is a durable blocking host call with a bounded duration. It does not
  request drain idle and does not wait for user input.
- Observe-value immediately appends one durable waking context event containing
  a rendered JSON preview. It is not buffered until the `run` succeeds. If the
  preview is compacted, a scoped read function loads the full stored value by
  id. A same-run idle request will not apply until the agent has processed the
  observation.
- Request user-visible agent messages with show-message inside `run`. It
  immediately appends a timeline message and durable message-shown context
  event.
- Apply an idle request only after the `run` tool returns. Its final message and
  run completion apply only if no unread waking context events arrived after the
  model call's consumed context-event sequence.
- If `run` fails, append `run_failed`. If that failed run had already accepted
  an idle request, also append `idle_request_not_applied`. Already executed
  host-function side effects, such as shown messages, remain durable.
- If an idle final message is not applied after a successful run, append
  `run_completed` and `idle_request_not_applied`. Do not also append
  `message_not_shown`; `idle_request_not_applied` is the durable outcome
  explaining that the idle request and final message were not applied.
- If an idle request applies after a successful run, insert the final timeline
  message, append `run_completed`, `message_shown`, and
  `idled_until_new_input`, then complete the run and release its run-drain
  claim.
- `run_completed`, `message_shown`, and `idled_until_new_input` are non-waking
  lifecycle context events. Facts such as user messages, resource
  notifications, `value_observed`, `run_failed`, `protocol_error`, and
  `idle_request_not_applied` are waking context events.
- Readable assistant text from model output is user-visible. Record it as a
  text-block progress update, then as a message timeline event, before
  validating the direct tool-call protocol.
- The prompt/tool protocol controls tool execution only; it does not gate
  visibility of emitted assistant text. Assistant text alongside a valid tool
  call is allowed. Assistant text without a valid tool call is still shown,
  then handled by the normal `protocol_error` context event.
- Missing, multiple, unsupported, or invalid direct tool calls require a
  `protocol_error` context event and execute no tool. Assistant text does not
  itself make an otherwise valid tool call invalid.
- Call idle-until-input inside `run` after successful work when all currently
  available input has been processed and final user-visible output is needed.
  Use show-message for standalone output that should be shown immediately
  without idling.
- Render prompt context as to-agent sections first, then pack those sections
  into model messages at the final prompt boundary.
- Package to-agent sections into a protocol tool-call response when the previous
  assistant response left tool calls open; otherwise package them into a user
  message.
- Send one opening user-context message when needed, then alternate assistant
  messages and protocol tool-call responses. Additional user messages are only
  valid after an assistant response with no tool calls, where there is no
  protocol tool-call response for feedback or context.
- When a running drain has no new waking context events after a tool call, append
  explicit lifecycle context events such as `run_completed` before making the next
  model call. Do not rely on blank protocol tool-call responses to carry outcome
  meaning.
- When an idle-until-input request applies, complete the run after appending
  `run_completed`, `message_shown`, and `idled_until_new_input` as protocol
  context events without visible progress activity. On the next drain with new
  input, package the newly available context events as normal model-visible
  context.
- Before each model call, include the current agent note and project/resource
  note as ambient model-call context in that order, and persist the exact
  structured ambient snapshots with the model call. Notes are current-state
  context, not chronological context events.
- Resource functions that create visible side effects should append source-agent
  activity context events when that fact should be durable context for the
  acting agent. Ordinary reads/lists/gets, implementation details, and short
  sleeps should not create extra activity context.
- Do not force required tool choice for agent model calls. It degrades reasoning
  quality and biases the model toward premature tool emission.
- Enforce the agent tool protocol with prompt instructions, validation, and
  feedback input instead of forcing required tool choice.

## Producer Guide

Use context events, progress updates, timeline events, and ambient state for
different jobs.

Return-only resource reads, lists, gets, and helper calls should just return
their result. The tool result already reaches the model for the current call;
do not duplicate that transient result into durable context or progress.

Resource host functions that create visible side effects should emit activity
context through the resource host-function context. This appends a chronological
context event and projects the same event into visible progress for the acting
agent.

Use a shared activity-emission helper when a lower-level producer is not running
inside the resource host-function context, but the fact is still visible agent
activity.

Use timeline plus context for user-visible transcript messages. The timeline
event is what the user sees; the context event is the durable protocol fact the
model sees later. Do not add progress merely because a message was shown.

Use context-only events for protocol or scheduling facts that the model must see
but that are not user-visible work.

Use direct progress updates for model-output activity that is visible as the
agent works, such as reasoning, assistant text, and direct tool calls. If the
assistant text is user-visible, record the matching timeline message as a
separate timeline event.

Do not model current-state notes as context events. Load them as ambient
model-call context and persist the exact ambient snapshot with the model call.
