# Provider-Backed Machines

## Scope

This document covers tenant/account-visible managed machines, their generic
backing records, and dispatch to provider-specific backing drivers. Local
machine mirrors, filesystem APIs, command execution after machine creation, and
billing are outside this module.

## Public Contract

Common behavior:

- Tenant/account APIs use the current tenant/account context.
- Machine-handle APIs accept a sealed machine handle, unseal it to the internal
  machine id, and verify current tenant/account ownership.
- Ownership misses use the module's typed not-found error.
- Mutating APIs accept a replay or idempotency key.

Common shapes:

- Machine specs contain the provider, zone or region, machine type, and OS or
  image family.
- Machine detail and runtime info are structured provider-neutral data.
- Machine power state is the domain state, usually running or stopped.
- Catalog entries expose public spec, detail, and pricing.
- Machine refs expose the outward handle plus detail.
- Machine summaries expose the outward handle, spec, and detail.

If the same contract is exposed through multiple transports, keep the field
shapes identical. Transport-only commands may adapt the surface, but shared
semantics remain in the service.

| Command | Input | Output | Errors | Behavior |
| --- | --- | --- | --- | --- |
| List catalog | None | Public catalog entries | None | Return public spec, detail, and pricing; hide inventory and backing config. |
| Create | Spec, bootstrap command, replay key | Machine ref | Catalog entry not found | Validate catalog entry, claim or provision backing, start bootstrap work, attach tenant/account, return handle and detail. |
| Delete | Machine handle, replay key | None | Machine not found | Remove tenant/account visibility, generic rows, and provider backing. |
| Start | Machine handle, replay key | None | Machine not found; already running | Dispatch backing to running state. |
| Stop | Machine handle, replay key | None | Machine not found; already stopped | Dispatch backing to stopped state. |
| Get | Machine handle | Machine summary | Machine not found | Return handle, spec, and detail from the generic backing record. |
| List | None | Machine summaries | None | Return current tenant/account summaries ordered by machine creation time descending. |
| Get power state | Machine handle | Power state | Machine not found | Driver reads provider state and maps it to the domain state. |
| Get runtime info | Machine handle | Runtime info | Machine not found | Driver returns provider-specific runtime info as structured data. |

## Catalog

The public catalog is derived from private machine definitions.

Each definition contains:

- Public spec.
- Public detail.
- Public pricing.
- Private inventory settings, such as warm-pool size.
- Private backing configuration for the selected provider driver.

Provider-specific catalog values belong in local catalog data, not in shared
service logic. The service reads the catalog by spec, exposes only public fields,
and uses private backing config only when provisioning or claiming a backing.

## Backing Contract

A machine is the tenant/account-visible object. A backing is the provider
resource behind it.

- The backing definition is private catalog configuration.
- The backing ref is a private persisted identity pointer.
- Backing config and backing refs decode through the selected driver's schema or
  parser.
- The generic backing ref contains only the driver key and opaque driver ref.
  Durable provider metadata belongs in provider-owned storage.
- Do not create relational foreign keys from opaque driver refs into
  provider-owned storage. The selected driver owns the interpretation.

Each installed backing driver provides:

| Operation | Behavior |
| --- | --- |
| Provision backing | Create or reconcile provider resource creation and return a driver ref. |
| Delete backing | Delete or reconcile provider resource deletion. |
| Start backing | Reconcile provider resource to running state. |
| Stop backing | Reconcile provider resource to stopped state. |
| Get power state | Read provider state and map it to domain power state. |
| Get runtime info | Read provider-specific runtime info and map it to structured data. |

Dispatch backing operations by driver key. Installations may have one or many
drivers; generic service logic should not branch on provider-specific storage.

Driver errors:

```text
delete backing -> backing already deleted
start backing  -> machine not found | already running
stop backing   -> machine not found | already stopped
read operations -> machine not found
```

## Storage Model

Use generic storage for:

- Backing rows with id, spec, detail, backing state, executor/worker token, and
  nullable backing ref.
- Machine rows that point at backing rows.
- Tenant/account ownership rows that point at machine rows.

Constraints used by the service:

- At most one machine row may point at a backing.
- At most one tenant/account ownership row may point at a machine.
- Backing worker tokens are unique.
- A tenant/account-visible machine must have a present backing ref.
- A ready backing claimed for a machine must have a present backing ref and no
  existing machine row.
- Foreign keys should link only generic rows. Provider-owned rows remain behind
  the driver boundary.

## Lifecycle

Create:

```text
tenant/account request
  -> validate spec against catalog
  -> claim oldest matching unclaimed ready backing with present backing ref
  -> otherwise reserve provisioning backing, provision provider resource, wait
     for executor/worker readiness, set backing state to ready, store backing ref
  -> insert machine row
  -> start bootstrap command through executor/worker
  -> insert tenant/account ownership row
  -> return machine handle and detail
```

Create edges:

- Finalizing a backing requires an existing provisioning backing with no machine
  row and no existing backing ref.
- Tenant/account attachment is one-time.
- Insert the machine row before starting bootstrap work.
- Insert the tenant/account ownership row only after bootstrap work is accepted
  by the executor/worker.
- Create waits for bootstrap acceptance, not bootstrap completion.
- Concurrent ready-backing claims can race at the machine-to-backing uniqueness
  guard.

Delete:

```text
tenant/account request
  -> unseal handle
  -> delete matching ownership row, or machine not found
  -> read backing ref
  -> delete machine row
  -> delete backing row
  -> driver deletes provider resource
```

Delete removes service visibility before provider teardown: ownership row,
generic machine row, generic backing row, provider-owned driver row, then
provider resource.

Start/stop:

```text
handle -> tenant/account ownership -> backing ref -> driver-key dispatch
  -> driver checks provider state
  -> already desired: already-running or already-stopped error
  -> otherwise reconcile provider resource to desired power state
```

Power/runtime reads:

```text
handle -> tenant/account ownership -> backing ref -> driver-key dispatch
  -> driver reads provider state or runtime info
```

If a driver reports machine-not-found after a backing-ref check, mutating
operations recheck the machine/backing record and read operations recheck the
tenant/account machine record. If the recheck no longer proceeds, return the
not-found error.

Warm pool:

```text
maintenance tick
  -> select machine definitions with warm-pool size above zero
  -> submit ensure-warm-pool operation
  -> count unclaimed provisioning or ready backings for the spec
  -> reserve/provision only the deficit
  -> mark new backings ready without creating machine rows
```

Warm-pool capacity counts unbound provisioning or ready backings by spec and
does not require a present backing ref. Create only claims ready backings with a
present backing ref.

Durable operation names should be stable and semantic:

```text
TenantMachineCreate
TenantMachineDelete
TenantMachineStart
TenantMachineStop
MachineBackingEnsureWarmPool
```

## Provider Driver Rules

Driver key:

- Use a stable driver key to identify each provider implementation.

Driver-owned shapes:

- Backing config contains provider-specific creation fields, such as region,
  size, image, network, or storage settings.
- Backing ref contains a stable provider-owned identity pointer.
- Provider-owned storage contains provider resource ids, reconciliation keys,
  short names, tags, and any metadata needed only by that driver.
- Runtime info contains provider-specific fields mapped into structured
  provider-neutral output where possible.

Driver storage constraints:

- Provider resource ids are unique.
- Provider-generated names or short keys are unique when they are used as
  reconciliation identities.
- Provider ids must decode into the numeric/string shape expected by the driver
  before provider API use.

Behavior:

- Provision reserves a provider-compatible resource name, creates a unique
  provisioning tag or reconciliation key, creates the provider resource with
  broad ownership and provisioning markers, and waits for the provider resource
  to become active before persisting the driver ref.
- Provision rejects a fresh reconciliation key that already finds a resource,
  multiple resources for one reconciliation key, returned name mismatch, and
  just-created resource-not-found during activation polling.
- Start maps an already-running provider state to the already-running error;
  otherwise it powers on and reconciles until running.
- Stop maps an already-stopped provider state to the already-stopped error;
  otherwise it powers off and reconciles until stopped.
- Delete removes the driver row before deleting the provider resource. Missing
  driver rows are already-deleted; provider resources that disappear after a
  driver row existed require an explicit unrecoverable or defect path.
- Power state maps provider states into the domain power states. Persisted refs
  must not point at provider states that cannot be represented by the domain.
- Runtime info may return absent network fields, does not necessarily require
  the provider resource to be running, maps provider not-found to the module's
  not-found error, and retries transient provider API errors.
- Create, delete, start, and stop use operation-specific reconcile schedules.
  Provider not-found handling differs by operation.

Tag meanings:

- Broad ownership marker: identifies resources owned by the service.
- Provisioning marker: create-reconciliation identity used to find a provider
  resource after an uncertain create outcome.
