# Resource Lifecycle

## Scope

This document covers how persistent rows model resource setup, reservation,
publication, activation, and teardown.

## Row Shapes

- The core lifecycle test is whether a status changes the row's meaningful
  information. If it changes which fields are valid, required, observable, or
  actionable, the table is modeling multiple row shapes and should be split.
- Persist durable facts and derive lifecycle state when the facts already imply
  it. Do not store a status whose main purpose is to explain which nullable
  fields, foreign keys, observers, invariants, or operations are meaningful.
- A row should not exist before the durable fact it represents exists. Failed
  setup before that point is an operation failure, not a lifecycle state, unless
  the attempt itself has durable audit, rate-limit, debugging, or operator
  value.
- Active, pending, complete, expired, superseded, and stopped states should be
  derived from row presence, relationships, and terminal facts when possible.
  Store the terminal fact, not a parallel status label.
- Prefer timestamps over status values for natural transitions when the row
  remains the same coherent thing. A timestamp records both that the transition
  happened and when it happened.
- Every table-local `id` is that table's own generated UUIDv7 identity. Do not
  make rows in different tables depend on matching id values.
- Use explicit foreign keys to state the relationship between lifecycle rows.
  State rows point to neutral resource or allocation rows with `resource_id` or
  `allocation_id`; callers that require a specific state point to that state
  row's `id`.
- Primary resource tables contain resources that are published and usable by
  their normal observers.
- Do not create a primary product row merely to get an id, derive a handle,
  reserve capacity, authorize setup, or make a durable workflow replayable.
- Pre-publication state belongs in replay state or in a narrow support table
  whose name states its role.
- Prefer split tables over lifecycle status columns when phase membership is a
  durable fact, query boundary, authorization boundary, external projection
  input, capacity hold, billing/session boundary, foreign-key target, or changes
  required fields, observers, or query defaults.
- Do not split merely because an enum has multiple values. Phase tables model
  different row shapes and dependencies; they are not a table-per-status
  convention.
- Use a status column only when every status is still the same kind of row,
  every non-status field keeps the same meaning across statuses, and normal
  callers intentionally reason over the full lifecycle.
- If a status changes who is allowed to observe the row or what invariants
  callers can assume, model those states as different row shapes instead of one
  status-bearing table.

## Identity And References

- Foreign keys should target the narrowest row that proves the required
  invariant. If a row needs an active relay, published endpoint, ready backing,
  or reserved allocation, it should reference that state table. Join through
  that state row to reach the underlying resource or allocation facts.
- Reference the neutral resource or allocation row only when lifecycle state is
  intentionally irrelevant or when provider/setup code needs the durable
  underlying object before publication.
- State rows have their own `id`, `created_at`, `extra`, and explicit
  `resource_id` or `allocation_id` FK. They do not share primary keys with the
  underlying row.
- Outward handles should be derived from the identity whose lifecycle they
  represent. Provider setup handles may derive from a resource id when setup
  must happen before publication; tenant-visible active handles should derive
  from the active row when publication is the required invariant.
- If an external protocol requires pre-publication setup and post-publication
  consumers to share the same name, that protocol name may derive from the
  resource row. Keep tenant CRUD handles derived from the active row.
- If stable fields are copied from one lifecycle table into another during a
  transition, the row shape is wrong. Move the shared fields to a neutral
  resource or allocation row and let state rows point at it.

## Lifecycle Terms

- A **reservation** holds capacity, names, ports, domains, money, or other scarce
  support state before a primary resource is published.
- A **registration** is pre-publication state that lets an external projection
  or callback see a resource before the primary product row is visible.
- An **allocation** is the durable hold for scarce or economic capacity whose
  facts remain meaningful across reservation, activation, history, and teardown.
- A **setup resource** is provider-facing state needed by an external callback,
  worker, or control plane before the tenant-visible resource exists.
- A **setup attempt** is operator or workflow history for trying to create a
  resource. It is not the resource.
- A **provisioning** row is in-flight setup state for a resource that is not yet
  claimable or visible through normal product paths.
- A **published resource** is visible through normal list/get/auth paths and
  satisfies the table's normal invariants.
- A **billing session** is the post-activation billing state row with billing
  cursor fields. A **running billing session** is a billing session whose
  `stopped_at` is still null.
- A **claimable backing** is an available internal inventory item with the
  provider reference required to turn it into a machine.

## Split Categories

- **Delete-on-publish support rows** are temporary rows whose purpose ends when
  the final row is created and whose fields do not remain meaningful after that
  transition. Publish by inserting the final row and deleting the support row in
  the same transaction. Support-row metadata remains support metadata. Do not
  copy durable facts from a support row into the published row; if a fact
  remains true after publication, it belongs on a neutral resource or allocation
  row instead.
- **Resource-plus-state rows** split stable resource facts from lifecycle
  visibility. Keep one resource/allocation row for fields that remain true
  across setup, reservation, publication, deletion, authorization, and teardown;
  point narrow state rows at it for the current observers or billing/session
  lifecycle. Each state row owns its own generated id and points back with an
  explicit FK.
- Database uniqueness may enforce one row per state for a given resource or
  allocation. Mutual exclusion between different state tables belongs in the
  promotion or activation mutation that deletes one state row and inserts the
  next.
- Resource-plus-state rows are preferred over copy-delete support rows
  when the stable row has a real identity while unpublished or deleting. Domain
  allocations, public port allocations, billing allocations, provider auth
  credentials, and provider setup state are resource facts. Publication,
  reservation, registration, provisioning, readiness, and billing cursor history
  are state facts.
- Delete resource-plus-state resources by removing the externally
  observable or claimable marker first, deploying or tearing down external
  projection, then deleting the underlying resource row. If teardown
  dead-letters after the state row is removed, the resource row remains as an
  explicit allocation hold until operator repair.
- If the external system can keep using scarce capacity briefly after
  unpublication, insert a release/drain state instead of deleting the resource
  row immediately. The release row should be invisible to normal product paths
  and should name the cleanup point that may reclaim the underlying resource.
- **No-split child resources** are valid when the child system row is already a
  real control-plane resource and no separate observer needs pre-publication
  support state. Create the child row before the parent product boundary, then
  publish the parent row last.
- Function names should advertise the category they operate on. Use
  `reserve...` for support rows, `publish...` for final visibility,
  `promote...` for provisioning-to-claimable inventory, and `activate...` when a
  reserved allocation receives its billing session row.

## Publication Order

- Setup flows reserve support state when needed, perform external setup or
  projection deployment, then publish the primary resource as the final
  visibility step.
- Setup flows may leave an external projection briefly aware of an unpublished
  replay-stable candidate before the primary row is inserted. This is an
  acceptable durable prefix when normal product paths cannot observe or allocate
  the candidate until the primary row exists, and replay can publish the same
  candidate to converge the systems.
- Do not publish the primary row earlier merely to avoid that external prefix.
  The worse invariant break is a locally published resource whose required
  external projection is not ready.
- Delete flows remove publication first, then tear down support or provider
  state.
- A support table is justified only when another concurrent operation, external
  callback, provider authorization flow, operator workflow, or capacity check
  must observe that state before publication.
- If no other observer needs pre-publication state, keep it in durable replay
  state and insert only the final published row.
