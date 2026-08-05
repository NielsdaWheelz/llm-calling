# Operation Types

## Scope

This document covers managed operation categories, entry point and composition
patterns, replay identity, memoization, stable id generation, and durable
operations.

## Purpose

Mutating operations can be interrupted by retries, timeouts, and crashes. A
later attempt must finish the same logical operation without double-applying
side effects or drifting to a different result.

Managed operations make this resumable. Every effectful operation is wrapped in
an operation category that declares its replay characteristics, composes safely
with other operations, and enforces correctness invariants through explicit
context capabilities.

## Context Capabilities

Managed operation frameworks need a small set of positive capabilities:

- Any-operation scope: read-only composition is permitted.
- Mutation scope: managed mutation composition is permitted.
- Durable scope: durable multi-step composition is permitted.

Read composition gets only read capability. Mutation composition gets read plus
mutation capability. Durable workflow bodies get read, mutation, and durable
capability.

Framework internals may need a stronger internal marker that distinguishes
multi-mutation bodies from single-mutation bodies. Ordinary domain code should
not need to mention that marker directly.

Leaf bodies also need a negative capability that blocks nested managed-operation
calls and direct operation execution. Use it for single database mutations,
read-only transaction bodies, external-call leaves, and service-call leaves.

Replay paths, step-key counters, and single-mutation guards are internal runtime
state, not public scope markers.

Coordination wrappers may spend protocol-owned internal mutation steps, but they
must not silently change the wrapped user body's mutation-cardinality contract.
A single-mutation body stays single-mutation. A multi-mutation body stays
multi-mutation.

## Operation Categories

Operation categories, ordered from lightest to heaviest:

| Category | Purpose | Replayable | Edge-runnable | Side effects |
|---|---|---|---|---|
| Pure read | Deterministic, side-effect-free computation | not needed | yes | none |
| unreplayable read | Simple read | no | yes | read-only |
| unreplayable stream read | Simple read returning a stream | no | yes | read-only stream |
| unreplayable single mutation | One-shot state change with no replay guarantee | no | yes | at most 1 |
| unreplayable multi-mutation | Opaque multi-step execution with no replay key | no | yes | multiple |
| Replayable read | Read whose result can be cached for replay | yes | yes | read-only |
| Replayable single mutation | Single atomic state change | yes | yes | at most 1 |
| Replayable multi-mutation | Multi-step workflow | yes | no | multiple |
| Durable operation | Multi-step workflow with autonomous completion | yes | yes | multiple, presented as 1 |

Column definitions:

- Replayable means retrying with the same replay key produces the same
  observable result without re-applying side effects.
- Edge-runnable means the operation can be run directly at entry points.
- Side effects means how many independent side effects, crash-separable commits,
  or external calls the operation may perform. Read-only operations perform
  none.

## Category Details

### Pure Read

A pure read is a deterministic, side-effect-free computation. It always produces
the same result for the same input. It is not memoized because determinism makes
memoization unnecessary.

- Body contains pure computation and calls to other pure reads.
- Body contains no services and no I/O.

### unreplayable read

An unreplayable read may return different results each time. It is the simplest
read category.

- Plain body: read-only operations and control flow.
- Database body: read-only database queries and control flow inside the
  required read-only transaction isolation level.
- Composition body: calls to reads and control flow. It performs no direct side
  effects.

### unreplayable stream read

An unreplayable stream read returns a stream of values rather than a single
result.

- Stream setup contains read-only operations and control flow.
- The managed scope covers both setup work and the resulting stream.
- Calling the stream read opens the stream inside an existing managed flow and
  returns it as the host runtime's task, effect, or stream result.
- Edge streaming handlers flatten setup before returning the stream.

### unreplayable single mutation

An unreplayable single mutation is a one-shot state change with no automatic
recovery if interrupted.

- Database body: one database write transaction with no replay or memoization.
- External-call body: one external call with no transaction and no retry.
- Raw interop body: ordinary effectful code at an adapter boundary that still
  needs the managed-operation capabilities already present in context.
- Composition body: calls to reads and at most one mutation call.

External-call leaves are strict leaves. They expose the negative leaf
capability, so the body cannot nest managed operations.

### unreplayable multi-mutation

An unreplayable multi-mutation is an opaque execution region that may perform
multiple managed mutation calls internally but has no replay key and no
resumable replay semantics.

- Use it for adapter or runtime regions that must stay in ordinary effectful
  code but may call multiple managed mutations internally.
- Composition may call reads and any number of mutations.
- Do not let this region call replayable multi-step workflows directly. If it
  needs durable multi-step work, start a durable operation.

### Replayable Read

A replayable read returns a result that stays stable across replays. Asking the
same question twice during a retry gives the same answer.

- Body has serializable exits so the result can be cached for replay.
- Database body runs inside the required read-only transaction isolation level.
- Composition body calls reads and control flow, but performs no direct side
  effects.
- A time-of-check/time-of-use safe read performs check, action, and recheck in a
  managed shape. On recheck, the check runs again.

### Replayable Single Mutation

A replayable single mutation is a state change that completes exactly once,
even across crashes and retries.

- Database body is a single atomic database write. Results are cached for crash
  recovery. It contains database reads, writes, and control flow, but no nested
  managed-operation calls.
- Idempotent external wrapper memoizes an external or interop mutation whose
  exact replay with identical input is already correct. Local memoization stops
  the system from depending on the external provider retaining idempotency
  forever.
- Service-call wrapper crosses a service boundary, generates a stable replay
  key, and retries transport failures.
- Composition body calls reads and at most one mutation call.
- Uncertain transition wrapper wraps one unreplayable transition with crash
  detection. Success returns a confirmed outcome. Ambiguous crash recovery
  returns an unknown outcome that must be handled internally.
- Stabilization turns an uncertain transition into a deterministic outcome by
  checking whether the effect is already visible, executing if needed, then
  retrying or reconciling unknown outcomes until success or exhaustion.
- Unsafe single-mutation memoization records the final result of an
  unreplayable single mutation without crash-detection markers. Use it only
  when duplicate execution is an explicitly accepted tradeoff.

Unknown outcomes are internal coordination states, not normal product-facing
results. Retry, reconcile, or classify a terminal modeled failure internally. If
that process exhausts first, defect.

Choosing an external-mutation wrapper:

- Use plain retry-and-defect when transient provider failure should retry and no
  crash-ambiguity protocol is needed.
- Use an idempotent external wrapper when exact replay of the same external call
  is already correct.
- Use unsafe memoization when duplicate execution of one transition is an
  accepted domain tradeoff and there is no worthwhile authoritative recovery
  path.
- Use uncertain transition plus stabilization when a replayable read can
  authoritatively prove the effect after an ambiguous external transition.
- If none of those fits cleanly, the operation needs explicit domain recovery
  logic rather than a softer error surface.

Mutation naming rule:

- Strict mutation semantics are the default.
- Prefer names where success means this call actually performed a new state
  transition.
- A pre-existing effect becomes an already-complete error. Callers should catch
  that only when "already done" is a legitimate domain outcome.
- If a mutation converges on an outcome and treats an already-existing outcome
  as normal rather than error, name it with an `ensure...` verb.
- If convergent semantics are intentional, implement them explicitly at the
  domain boundary. Do not hide them behind a generic helper.

Single-step linearization serializes concurrent single-step mutations on the
same conflict key through a short-lived live lock on the shared exclusivity
keyspace.

A time-of-check/time-of-use safe mutation performs check, action, and recheck in
a replayable mutation context. The first check is memoized for stable input
across replays.

### Replayable Multi-Mutation

A replayable multi-mutation is a multi-step workflow with multiple independently
committed state changes that must be driven to completion in order.

- Use it when the work cannot honestly be one database commit. If one
  serializable-equivalent transaction is enough, keep it as a single database
  mutation.
- It is not an edge entry point by itself. Use it as a reusable subworkflow
  inside another durable flow, or wrap it in a durable operation when the edge
  needs a named recoverable workflow.
- Opaque unreplayable regions cannot call replayable multi-mutations directly.
  Start a durable operation instead.
- Deterministic interpreters or script runtimes may be modeled as replayable
  multi-mutations only when every durable effect crosses a managed host-call
  boundary and ambient runtime nondeterminism is removed or exposed through
  replayable operations.
- Resource budgets that affect interpreter control flow must be replay-stable.
  Do not let unmanaged wall-clock time decide whether later managed calls are
  reachable.

Multi-mutation result memoization records a replayable multi-mutation's final
result while leaving child effects as ordinary replayable steps. If the process
dies before the final result is memoized, replay re-runs the orchestration;
already-completed child effects recover through their own replay keys.

Multi-step linearization serializes a durable multi-step workflow on a conflict
key through the internal replay-aware exclusivity protocol. Successful
completion and typed failures release the conflict-key lease. Defects and
interruptions intentionally do not release it; for durable dead-lettered work,
the stuck lease is operator-visible containment rather than state to auto-clear.

### Durable Operation

A durable operation is a multi-step workflow that will run to completion on its
own, even if the original caller crashes.

- Declaration names a recoverable workflow.
- Implementation binds that declaration to the workflow used by foreground
  callers and worker catalogs.
- Creation may combine declaration and binding when no separate import boundary
  is useful.
- If the process crashes mid-execution, orphaned work is picked up and replayed
  to completion.
- Handles support fire-and-forget submission, foreground execution, and inline
  composition within a parent durable operation.
- Foreground durable execution cannot be called inside a normal durable
  operation body. Opaque unreplayable interop regions may use it to create a
  child recovery boundary.
- Child durable operation bodies clear any inherited opaque unreplayable marker
  before running so their internal replayable multi-step work executes as normal
  durable work.

### In-Transaction Helpers

For composing reads and writes inside a single database mutation, read-only
database query, or unreplayable database mutation body:

- Database query helper: composable read-only helper callable only inside an
  owning database operation body.
- Database mutation helper: composable mutation helper callable only inside an
  owning database mutation body.
- Family-specific raw transaction scopes are tools for infrastructure and
  adapters. Prefer named database operation bodies and database helper
  constructors for domain code.
- Keep transaction-scoped helpers narrow. They should be private
  sequence/allocation or row-shape primitives used inside a domain database
  operation boundary.
- Public domain APIs should own the transaction boundary and expose domain
  decisions rather than raw transaction plumbing.
- Do not introduce in-transaction helper APIs just to make related follow-up
  work happen in the same commit. If the follow-up is not part of the
  committed-state invariant, model it as an explicit replayable step.

## Running At Entry Points

Every API handler or entry point terminates domain operation work with one
operation runner.

### Mutation Handlers

- Build a replayable mutation, then run it through the replayable operation
  runner with a stable replay key.
- The replay key comes from the request payload. Namespace it with the operation
  name.
- The caller keeps the replay key stable across retries of the same logical
  mutation.

### unreplayable mutation handlers

- Build an unreplayable mutation, then run it through the unreplayable operation
  runner.
- Use when replay is not needed, such as maintenance, timing, or coordination
  internals.

### Read-Only Handlers

- Build a read-only operation, then run it through the query operation runner.

### Streaming Handlers

- Build a stream-producing read-only operation, then run it through the query
  operation runner.
- For invalidation-backed snapshot streams, the handler may return a raw stream
  adapter only when the stream setup performs listener wiring and each snapshot
  load terminates its own domain work with a query or unreplayable operation
  runner.
- Database invalidation streams should use a local helper that owns listener
  setup, teardown, and snapshot execution.

### Durable Operation Handlers

- Execute the durable operation through the replayable operation runner with a
  stable replay key.

## Composing Managed Operations

Inside a domain operation flow, use the managed-operation call primitive to
invoke inner operations. Never call operation runners inside an operation flow.

Public authoring can use the managed composition builder, managed call
primitive, and structured combinators such as mapping, tapping, conditional
branching, variant matching, and iteration. Unsafe composition is reserved for
raw runtime interop.

Convert a managed operation back to the host runtime only at plain task/effect
or stream interop boundaries. Otherwise keep composing in the managed operation
model and use operation runners only at entry points.

The current context determines what may be called:

- Read flows have read-only capability, so managed calls to mutations are
  rejected by the capability model.
- Mutation flows have mutation capability, so they can call reads and mutations.
- Durable flows add durable capability, enabling managed calls to replayable
  multi-mutations.
- Opaque unreplayable interop regions deliberately reject managed calls to
  replayable multi-mutations. If that region needs multi-step durable work,
  expose the work as a durable operation and execute it as a child recovery
  boundary.
- The child durable body entered by foreground durable execution clears the
  opaque marker before it runs; only the outer interop region remains opaque.

Each managed operation call gets a structural step key. Auto-generated step keys
and structured iteration together form the replay path for memoization.

Coordination runtime and framework code that implements the operation framework
itself should use internal runtime compose builders rather than public domain
operation constructors.

## Explicit Step Keys

Public managed iteration owns iteration replay identity. Each array element gets
its own deterministic child scope, so concurrent loop bodies can safely use
plain managed calls inside the iteration body without manual keys.

Explicit step keys are internal-only. Coordination kernel code uses internal
helpers when it needs named child scopes outside the public composition
combinators.

Duplicate explicit step keys within the same scope are defects.

## Replay Identity And Memoization

### Replay Path

In replayable mutation contexts, each managed operation call gets a stable
structural address. On retry, the same path must produce the same observable
result.

Replayable mutation roots are also serialized by replay key, so duplicate
replayable root executions cannot overlap and race the same logical path.

### What Gets Memoized

- Replayable read calls in a replayable mutation context are memoized so the
  same read returns the same answer on retry.
- Single-mutation calls in a replayable mutation context are memoized so replay
  reaches the same observable result without re-executing the mutation.
- Calls to unreplayable reads are never memoized. They cannot be called
  directly from replayable mutation or durable contexts. Upgrade them to
  replayable reads instead.

Replay memoization is not an optimization or a best-effort cache. It is the
durability mechanism for in-flight replayable work. Domain tables should not
duplicate memoized values solely so a workflow can recover without the
coordination replay state. Persist a value in a domain table when it is part of
that domain's storage or observation model; otherwise let the replay path own
it.

### Stable IDs

Use a replayable token or id query to generate a random token that is memoized
on replay. Use this for any id that must be stable across retries, including
provider idempotency keys, reconciliation tags, and external names that exist
only to make an in-flight operation replayable.

Replay-stable generated ids may be created before insert when a workflow needs a
future primary key to derive handles, subjects, commands, or provider metadata.
If a workflow might reuse an existing row, read the row first and generate an id
candidate only on the new-row path. The replay boundary should memoize a generic
validated id value; convert it to the specific domain id at the domain boundary.
Do not insert a domain row solely to obtain a generated id.

Durable workflow replay state may carry candidate ids, tokens, selected
configuration, generated commands, and other setup values across steps. Persist
those values in domain tables only when they are part of the domain's storage or
observation model.

## Durable Operations

Use durable operations for workflows that span multiple independent side
effects such as separate transactions or database plus external API sequences.

### Defining

Declare and bind a durable operation in one place by default.

Split declaration from implementation only when application code needs to
enqueue work without importing the workflow implementation.

There is no required filename or file count. Keep the operation in one module
unless the import boundary is useful.

Durable operation implementations return managed operation flows. If a reusable
replayable multi-mutation already exists, call it inside the implementation with
the managed-operation call primitive.

### Running

- Submit: fire-and-forget; returns after durable enqueue.
- Execute: foreground execution with crash recovery; requires an implemented
  handle.
- Inline call: execution inside a parent durable flow; requires an implemented
  handle.

### Nesting

- Foreground durable execution cannot be called inside a normal durable
  operation body.
- Opaque unreplayable interop regions may use foreground execution to create an
  explicit child recovery boundary.
- Submit and inline call both work within durable contexts for composing
  follow-up work.

### Durable Intermediate State

Durable workflows may persist intermediate state between replayable steps. That
state is part of replay recovery and operator inspection.

- Keep separate durable workflow steps in separate transactions by default.
  Do not merge a later step into an earlier single database mutation because it
  is expected to run immediately, because replay will eventually run it, or
  because the committed prefix looks incomplete without it.
- Combining writes into one database transaction requires a concrete
  observation-time domain invariant: after either write alone commits, a fresh
  reader would see a state the domain is not allowed to expose.
- Advance support, dedupe, checkpoint, and projection state in explicit
  replayable steps unless that state participates in such an invariant.
- Durable workflow correctness is judged by the state reached after successful
  replay, not by requiring every committed prefix to look like the final product
  state.
- A committed prefix may include real side effects, reservations, charges,
  ownership markers, or derived intermediate state whose matching domain
  artifact is produced by a later replayable step.
- Such prefixes are correct when replay can resume from them and complete the
  workflow without duplicating earlier side effects.
- Do not introduce generic cleanup scopes, finalizers, or transaction-callback
  APIs solely to make a multi-step durable workflow appear atomic.
- Do not create placeholder artifacts, fallback projections, or rollback
  machinery solely to make a dead-lettered prefix look complete.
- Cleanup after typed failures belongs in explicit domain compensation steps.
- Cleanup after defects belongs in explicit operator or admin repair paths.
  Normal durable replay should continue from the persisted intermediate state.
- When reviewing durable code, do not flag "step A committed before step B" as a
  bug by itself. First check whether step A is replay-safe, whether step B will
  be retried from the memoized prefix, and whether successful replay reaches the
  correct final state. Flag it only when replay can duplicate side effects, lose
  the ability to complete, or expose an invalid committed invariant that the
  domain requires at every observation point.

### Dead Letters And Ownership State

Durable-operation dead letters are loud terminal states. A dead-lettered item
means the autonomous worker stopped at an unexpected defect and must not be
silently bypassed by another worker path.

- A dead letter is a suspended durable prefix, not the intended final business
  state.
- The default operator action is to fix the defect, dependency, or data issue
  and replay the same durable operation until it reaches normal completion.
- Compensation, deletion, and manual state edits are explicit repair choices,
  not the default correctness model.
- Do not add automatic "unstick" logic that clears domain ownership state after
  a durable operation dead-letters.
- If a durable workflow claims a persistent running or owned state before doing
  multi-step work, leave that state intact when the workflow defects. The stuck
  state is part of the operator-visible failure signal.
