# Coordination

## Scope

This document covers the coordination module surface, storage backends, transient primitives, dead letter handling, and placement rules for new coordination primitives.

## Purpose

The coordination module provides multi-process coordination primitives and higher-level patterns for replayable workflows. See [../operation-types.md](../operation-types.md) for the operation type system these primitives support.

## Module Surface

Coordination module primitives fall into three categories:

### Operation Primitives

These are the primary developer-facing API. They live on the public
coordination module surface, not behind internal runtime packages.

- **Leaf primitives**: standard read/query and write/mutation boundaries for
  database work, external side effects, transport calls, and pure computation.
- **Operation constructors**: composition helpers for query, single-mutation,
  multi-mutation, stream, replayable, and unreplayable operation shapes.
- **In-transaction helpers**: helpers for query or mutation code that must run
  inside the caller-owned transaction boundary.
- **Durable operations**: declaration, implementation, creation, and domain
  catalog helpers for durable operation types.
- **Infrastructure combinators**: uncertain execution, memoization,
  stabilization, single-step linearization, multi-step linearization, and
  time-of-check/time-of-use helpers.
- **Utilities**: replay-stable token or id generation helpers.
- `Unknown` and duplicate-risk semantics belong inside coordination and operation recovery. They must not cross a product boundary unchanged.

### Transient Coordination Services

Low-level multi-process coordination is backed by transient coordination
storage. It is used internally by operation infrastructure and rarely
referenced directly by domain code.

- **Memoized execution**: run-or-return-cached keyed by coordination key. Used
  internally by mutation-frame memoization; domain code should rarely call this
  directly, but explicit-key maintenance or infrastructure code may.
- **Queue**: ordered item collection with database notification subscription.
  Used by the durable operation engine for queue-backed execution.
- **Deferred result**: write-once, read-many value with database notification
  signaling. Used by durable operations for caller-side await.
- **Heartbeat**: liveness detection built on timestamped pulse state. Supports
  start, pulse, liveness check, keepalive, and expiry await semantics.

### Internal Primitives

Behind the internal boundary. Not imported by domain code.

- **Lease**: time-bounded exclusivity row on a key. Provides raw live acquire or
  release for infrastructure locking plus replayable acquire and release steps
  and deadline-based wait queries for the internal exclusivity protocol.
- **Exclusive execution**: replay-aware internal multi-step exclusivity protocol
  on top of leases. Sticky in durable replayable mutation contexts and live in
  unreplayable mutation contexts. The protected body keeps the caller's
  operation semantics while acquire or release protocol steps run in internal
  multi-mutation regions.
- **Live lock**: attempt-scoped heartbeat lock on top of leases. Used internally
  by replay-root serialization, durable engines, memoized execution, and
  single-step linearization.
- **Memo store**: write-once key-value store backing memoized execution.
- **Pulse store**: timestamped touch state backing heartbeat semantics.
- **Runtime compose builders**: internal builders for framework code that is
  implementing operation semantics rather than domain workflows.

## Wiring Convention

Public reusable coordination APIs expose a semantic service or handle boundary
rather than exporting helper functions that inline-close another module's
self-wired internal service dependencies.

- Use self-service runtime helpers in exported APIs only at true execution or
  adapter boundaries, or in handle factories such as durable and scheduled-task
  wrappers where choosing the canonical runtime internally is the point.
- If other services may reasonably depend on a coordination or admin API, prefer exposing that API as a service and keep lower-level inspection or backend helpers behind its layer.
- For durable operations, split declaration from implementation only when app
  code needs an enqueue-only import. Bind implementations at catalog, worker,
  or adapter composition boundaries; there is no filename convention.

## Storage Backends

### Primary Storage

Coordination tables that must commit atomically with business mutations live in
primary shared storage:

- Transaction memo rows for crash-recovery memoization.
- Durable operation dead-letter rows.
- Scheduled task definition rows.

### Transient Storage

Transient coordination state with a 72h TTL horizon and maintenance pruning. This database owns transient replay caching and hot-path coordination data:

- Transient transaction memo rows for replayable mutation coordination.
- Transient memo entries.
- Transient lease entries.
- Transient queue generation and item rows.
- Transient pulse rows.

Transient storage is a database implementation detail. Runtime access to
transient tables is confined to the internal transient backend layer, and public
and domain code should depend on coordination primitive APIs rather than on the
database directly.

## Waiting In Replayable Workflows

- Durable exclusivity acquisition uses replayable lease-acquire attempts plus a replayable "wait until notified or recorded expiry" step.
- The wait step records completion, not a cached duration, so replay never sleeps based on stale wall-clock values.
- Sticky durable exclusivity does not heartbeat. It acquires once with the
  transient-state TTL, runs the protected body, and replays its explicit release
  step like any other child mutation step.
- Sticky exclusivity releases on success and typed failure. It does not release on defects or interruptions; defected durable work should remain visibly stuck until an operator inspects the dead letter or the transient TTL expires.
- This pattern is currently kept local to lease and the internal exclusivity
  protocol rather than exposed as a generic operation primitive. The correctness
  story is specific: a storage-notified coordination key and a database-backed
  deadline.
- Stabilization primitives do not need the same abstraction today. Their retry
  and reconciliation sleeps are advisory polling backoff, not
  correctness-bearing deadlines.

## Dead Letter Handling

- Durable dead-letter handling is for item-local poison and execution defects.
- Unregistered operations, payload decode failures, and execution defects are dead-lettered.
- Infrastructure defects before a valid queue item exists, and post-execution defects such as deferred completion or queue cleanup still crash the runner.
- Dead letters are operational containment and debugging, not application control flow.
- Do not persist synthetic "success" artifacts or placeholder derived state merely to avoid dead-lettering a defective durable-operation item.
- Dead-lettering does not complete the caller deferred.
- The original caller waits for retry success or coordination TTL expiry.
- TTL expiry of coordination state is a defect, not graceful degradation.
- Dead letters can be retried through the admin UI, which re-enqueues the item into the transient queue.

## Placing New Primitives

- If a coordination primitive needs persistent primary tables, put those tables
  in primary shared storage and keep the semantic API on the public
  coordination surface.
- If a primitive only uses transient storage, place it in the transient
  coordination area or the internal transient area when it is internal.
- If a public API is keyed by replay identity or by an explicit coordination
  key, place it outside the internal boundary.
- If a primitive only provides low-level non-replay-keyed coordination mechanics
  such as write-once values, lease gates, or mutual exclusion, place it behind
  the internal boundary.
- If code is a shared implementation detail of multiple primitives, place it in
  a nested internal area within the parent primitive.
- Lower-level coordination mechanics that require callers to manage their own
  consistency guarantees belong behind the internal boundary.
