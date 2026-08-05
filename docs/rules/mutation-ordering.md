# Mutation Ordering

## Scope

This document covers how to order mutations when an operation spans multiple systems or module boundaries.

## Cross-System Ordering

- When a multi-step operation mutates state across multiple systems, order mutations as the reverse of the observation order.
- Setup (resource creation): write the external system first, then the local
  database. The resource becomes observable only once fully provisioned.
- Teardown (resource deletion): write the local database first, then the
  external system. The resource becomes unobservable immediately.
- A "system" is an ownership or observation boundary, not necessarily a separate process or database.
- If module A interacts with module B only through B's interface, A may treat calls into B as a separate side effect for ordering purposes.
- These ordering rules apply to rows, refs, projections, and handles that make the resource observable to callers as existing. Internal reservations, setup resources, provider-authorization rows, queue items, replay memos, idempotency tokens, and attempt-local coordination records are protocol state, not resource-observation state.
- A committed pre-publication resource row is acceptable only when it is support
  state: tenant product get/list/auth paths must not treat it as the published
  resource. See [resource-lifecycle.md](resource-lifecycle.md) for the
  resource-plus-state row shape.
- Do not create a locally observable "resource exists" row before external setup just to hold recovery identity or obtain a database-generated id. Use replayable operation memoization, stable replay-generated tokens, provider reconciliation, and [stable id generation](operation-types.md#stable-ids) where appropriate.
- Do not keep a locally observable "resource exists" row after local teardown just to help provider cleanup. The replayable delete workflow owns the provider follow-up through its replay state.
- A domain may intentionally expose provisioning or deleting state when that state is part of the product model. Do not add placeholder lifecycle state merely to hide durable intermediate prefixes.

## Primary And Support State

- Identify the primary observable state before ordering a multi-step mutation.
- The primary observable state is the row, object, or boundary that makes the domain resource visible to fresh user or agent operations.
- Support state includes billing allocations and sessions, projections, short handles, provider handles, cleanup queues, and child-module state that follows the primary resource lifecycle.
- See [resource-lifecycle.md](resource-lifecycle.md) for row-shape rules that
  distinguish primary published state from setup, reservation, provider,
  authorization, and attempt state.
- Setup may prepare support state before writing the primary observable state. The primary observable state is the final visibility step.
- Teardown writes the primary observable state first, then tears down support state.
- Before deleting primary observable state, memoize the support handles required by later cleanup steps.
- Do not create committed prefixes where fresh operations can still observe the primary resource after required support state has already been torn down.
- If support state lives in the same physical database as primary state, still order it by observation boundary rather than by storage location.

## Recursive Boundaries

- Apply mutation ordering recursively at each boundary.
- The caller orders its own state relative to calls into the child module.
- The child module orders its own local state relative to the external systems it wraps.
- Do not reach across module boundaries to mutate another module's internal tables to force ordering.
- Do not widen a caller-owned database transaction across module boundaries.
- Do not widen a database transaction merely to guarantee that a later step
  eventually happens. Commit the local step at its natural database boundary,
  then model the later step explicitly as durable follow-up work.
- Do not add transaction-callback or finalizer APIs merely to hide durable intermediate state; split the workflow into explicit replayable steps unless a committed-state domain invariant requires one transaction.
- When a workflow spans boundaries, compose the sequence as a durable operation
  plus explicit replayable multi-mutation steps. Ordering comes from the
  managed operation structure, not from piercing abstraction boundaries.
- Intermediate committed prefixes are expected in durable workflows. Evaluate
  them by whether replay reaches the correct final state; see
  [operation-types.md](operation-types.md#durable-intermediate-state).
- Projection, cache, presentation, and notification rows are usually support
  state. Keep them in the same transaction as canonical state only when
  separately observing the committed rows would make the domain state false. If
  they merely need to follow canonical state before later ordered work, use a
  separate awaited durable step.
- Do not use fire-and-forget durable work for rows whose insertion time defines
  user-visible or model-visible ordering. Fire-and-forget follow-up is for work
  whose ordering is irrelevant or already anchored to previously committed
  state.
