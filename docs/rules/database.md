# Database

## Scope

This document covers relational database storage-shape rules, query patterns,
transaction boundaries, and database-specific conventions.

## Storage Shape

- Every table has a primary key.
- Primary keys are stable opaque `id` values.
- Use application generation when an id is needed before insert; use a database
  default when the id is only needed after insert.
- Generated ids inside replayable or durable workflows must follow
  [operation-types.md#stable-ids](operation-types.md#stable-ids).
- Do not query the database solely to generate an id when a trusted application
  generator can produce it.
- Do not expose private database ids to users.
- Durable entity tables record creation time with the database's authoritative
  clock.
- Do not add generic unstructured metadata columns unless the column is an
  intentional extension point with a clear owner.
- Database-backed functionality owns its storage shape and migrations with the
  feature or domain it serves.
- Shared persistent tables have one explicit storage owner. Keep storage shape and
  migrations with that storage owner, and keep semantic behavior with the
  domain owner.
- Keep database storage rules to storage shape and relational identity: column
  types, `NOT NULL`, primary keys, foreign keys, and true storage-owned
  uniqueness.
- Name owner foreign keys for the lifecycle and authorization owner they
  actually point to, not for a broader nearby entity.
- Application row types use the narrowest owned application type for column
  values when the language type system supports that distinction, even when the
  database stores the value in a primitive column type.
- Nullable columns remain nullable in raw table interfaces and raw query row
  shapes.
- Convert nullable database values to the owned absence representation before
  returning domain, service, replay, API, or DTO structures.
- Convert owned absence to database `NULL` only in the final insert/update
  adapter for the nullable column.

## Database Capabilities

- Optional database extensions and engine-specific capabilities must be declared
  by migrations and provisioned by environment setup. Do not rely on ambient
  database installation details.

## Relationship Shape

- Prefer normalized relationships over duplicate discriminator fields when a row's role is already determined by a one-to-one child table or foreign key relationship.
- Do not add `type`, `kind`, `purpose`, or similar columns solely to restate which child row points at a parent row. Derive that label at query/API boundaries when needed.
- Do not add generic descriptor columns for inspection value. If row meaning comes from a concrete owning relationship, model that relationship and leave generic rows meaning-free.
- Store `source` or provenance fields when they record an intentional creation path or audit fact that cannot be cleanly derived from normalized relationships.
- If a provenance field and nullable foreign keys must agree, enforce that branch consistency in application code and defect on impossible states; do not use database constraints for conditional nullability.
- For polymorphic target pointers, prefer one validated structured target value
  over nullable one-of foreign key columns when the relationship is a generic
  pointer rather than distinct relational roles.
- Use nullable one-of foreign key columns only when each branch has distinct relational meaning that should be modeled as its own relationship.
- Use an owned structured payload for embedded data when the inner fields do
  not need top-level relational identity, joins, foreign keys, uniqueness,
  indexing, or independent lifecycle.
- An embedded payload may contain a real discriminated union when that union is
  the shape of the embedded value. This is the inline equivalent of splitting
  branch-specific nullable columns into coherent branch payloads.
- Do not promote an embedded union to separate tables merely because different branches have different fields. Promote it only when a branch has distinct relational meaning, lifecycle, observation, authorization, query, or foreign-key behavior.
- Model intentional state and activity-derived state separately when they have
  different lifecycle or retention semantics. For example, saved items and
  recently viewed items are separate state, not one table with a saved flag.

## Lifecycle Shape

- See [resource-lifecycle.md](resource-lifecycle.md) for resource setup,
  reservation, publication, activation, teardown, and lifecycle row-shape
  rules.
- Database schemas should implement lifecycle shape with storage-level identity,
  nullability, and foreign-key reachability. Richer lifecycle invariants stay in
  application code plus defects.
- Lifecycle split tables still follow normal identity rules: each table owns its
  generated `id`, and links between resource, allocation, and state rows are
  explicit foreign keys rather than matching primary-key values.

## Constraints

- Do not add `CHECK` constraints, exclusion constraints, triggers, or other database-enforced business invariant machinery.
- Do not encode domain invariants in database schema, even when they are
  row-local and easy to express in a relational expression.
- Examples that still belong in application code plus defects: conditional nullability, tagged-union branch consistency, cross-column correlation, ownership rules, and lifecycle-state rules.
- If you are tempted to add a database constraint for anything richer than storage shape, primary-key identity, foreign-key reachability, or true schema-owned uniqueness, put that invariant in application code instead.

## Foreign Keys

- Cross-module foreign keys are allowed when both tables live in the same physical database and the reference records hard storage reachability to a durable shared row.
- Do not use cross-module foreign keys to encode business policy, lifecycle ownership, permissions, API boundaries, or module dependency preferences.
- Use the database's default non-cascading delete behavior.
- Do not use `ON DELETE CASCADE` or other database-level cascading operations.
- Cleanup is explicit in application code.

## Timestamps

- Store instants in an unambiguous time-zone-aware representation.
- Compare against database-stored timestamps with that database's authoritative
  clock, not a separate local clock.
- Each database is authoritative for the timestamps it stores.

## Time Intervals

- Time intervals are right-open: `[start, end)`.
- `expires_at` is the first moment of invalidity: active while `now < expires_at`, expired once `now >= expires_at`.
- Use `>` for "active" and `<=` for "expired" when checking expiry against `now()`.

## Indexes

- Do not add indexes speculatively. Add them when a query pattern on a high-volume table needs one.
- Use database uniqueness for true schema-owned keys: primary keys, real local
  alternate keys, and one-to-one resource/state link columns that define a split
  table's row shape.
- Do not use database indexes or unique constraints to encode application-level ownership, correlation, or lifecycle invariants.
- Use application code plus defects for higher-level invariants.

## Query Patterns

- Keep business rules, branching, and fallback policy in application code when reasonably possible.
- Use the database query language for database-shaped work such as set filtering, joins, ordering, aggregation, and atomic mutations.
- Design query helpers around domain meaning and operation boundaries, not round-trip minimization.
- Do not merge clean helper APIs, widen operation boundaries, or duplicate query-shaped logic solely to reduce database round trips.
- Do not treat repeated reads across separate semantic helper boundaries as a defect. Combine queries only when a correctness boundary, a measured bottleneck, or an explicit product latency/cost target requires that shape.
- Do not use upsert helpers to merge insert and update logic when the operation
  has distinct control-flow or linearization semantics.
- Use an explicit SELECT to check for an existing row, then INSERT, UPDATE, or DELETE accordingly.
- This is safe inside serializable retry transactions because concurrent
  conflicts cause a serialization failure that triggers a retry.
- The same applies to DELETE. Without an existence check, concurrent deletes can both report success, violating linearization.
- Do not use `numDeletedRows` or `numUpdatedRows` to determine control flow. That is the SELECT's job.
- Assert that row counts match expectations after a mutation as a defect catcher.
- Use exact-row mutation helpers or assertions for mutations that must affect
  exactly one row. Defect if the count is not 1.
- Use bulk mutation helpers for bulk operations and mutations where zero results
  is an expected typed error.
- Do not add existence checks just to optimize an impossible defect path.
  Trusted internal ids may rely on foreign keys, exact-row mutation assertions,
  or the service's natural read/write point to catch broken references as
  defects.
- See [concurrency.md](concurrency.md) for locking rules and [mutation-ordering.md](mutation-ordering.md) for ordering mutations across systems.

## Computed Database Values

- Static type annotations on computed expressions describe application output
  shape; they do not coerce database driver values at runtime.
- Computed database values must make their runtime shape explicit at the
  database boundary with a local helper, query cast, or schema decode.
- Use a deliberate helper for normal domain-bounded row counts that should fit
  the application's ordinary numeric type.
- If a count may exceed that range, add a deliberately named large-count helper
  instead of widening the normal count helper.

## Transaction Boundaries

- Read-only database operation boundaries run read-only transactions at the
  isolation level required by their correctness contract. Replay-safe reads must
  declare and preserve their replay behavior.
- A replayable single-mutation database boundary owns one retryable write
  transaction and its replay or memoization state. Callers do not pass replay
  handles or manually manage recovery rows.
- An unreplayable single-mutation database boundary owns one retryable write
  transaction with the same isolation behavior but no replay cache.
- Keep database transactions scoped to the single atomic database boundary they
  actually protect. If one serializable commit establishes the invariant, stop
  there.
- If the real requirement is "this committed step happened, and some later step
  must also happen eventually", keep the committed database step in its own
  transaction and model the follow-up as durable workflow orchestration rather
  than widening the transaction.
- Same physical database, same user action, same durable operation, or same
  feature area is not enough reason to share a transaction. Use one database
  transaction only when separately committed writes would make the observed
  database state invalid or false.
- Good reasons to share a transaction include allocating a sequence number and
  inserting the row that owns that sequence, replacing one published state row
  with another when readers must never observe neither or both, or committing
  multiple rows whose separate observation would contradict a committed domain
  fact.
- Bad reasons to share a transaction include making a later step happen
  eventually, keeping projection/cache/presentation state in sync by
  association, avoiding an expected durable intermediate prefix, or shortening
  code by reaching across module boundaries.
- If ordering matters but atomicity does not, use separate awaited durable
  steps. Do not use fire-and-forget work to insert ordered rows unless their
  order comes from a stable source rather than insertion time.
- Transient coordination storage stays owned by coordination infrastructure.
  Domain code should not couple to coordination backend tables.
- Low-level retrying transaction primitives belong to their database family.
  They retry serialization failures and defect when the retry budget is
  exhausted.
- Use a family-specific raw transaction primitive directly only for simple write
  boundaries or infrastructure that needs a raw transaction without
  managed-operation framing.
- Nested database transactions are not allowed.
- Do not run non-database side effects inside a database transaction; they
  cannot be rolled back on serialization retry.

## Database Helper Composition

- Query helpers compose database reads inside caller-owned read or mutation
  transaction bodies.
- Mutation helpers compose database writes inside caller-owned mutation
  transaction bodies.
- Raw helpers that need a caller-owned transaction should require an explicit
  transaction scope value from the relevant database family.
- Prefer named database operation boundaries and database helper constructors
  over exporting raw transaction-scoped domain helpers.

## Further Reading

- See [operation-types.md](operation-types.md) for managed-operation boundaries and composition rules.
- See [concurrency.md](concurrency.md) for linearization rules.
- See [mutation-ordering.md](mutation-ordering.md) for ordering mutations across systems.
- See [retries.md](retries.md) for retry-boundary rules such as in-memory state not rolling back across retries.
