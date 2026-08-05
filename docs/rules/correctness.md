# Correctness

## Scope

This document covers system abnormality classification and repository-wide correctness invariants.

## Abnormalities

- Expected system-level abnormalities must be modeled and handled in code.
- Unexpected abnormalities indicate a broken invariant and should trigger investigation.
- Expected abnormalities include server restarts and transient service failures within the applicable retry budget.
- Unexpected abnormalities include service failures that persist beyond the applicable retry budget.
- Retry budgets define the boundary between expected transient failure and unexpected persistent failure.
- See [retries.md](retries.md) for retry policies and exhaustion handling.

## Invariants

- If concurrent execution or crash-and-replay can produce an incorrect result, it is a bug.
- Every operation must correspond to some valid sequential ordering of all concurrent operations, including across crash-and-replay.
- Every committed external side effect must be discoverable during normal recovery; volatile reads alone are not sufficient. For replayable and durable work, retained coordination replay state is recovery state. Do not duplicate memoized step inputs or outputs into domain tables solely to recover an in-flight workflow after coordination state expiry; expiry is a defect and operator repair boundary, not a normal replay path.
- Projection drift is a broken invariant, not a user-facing branch. If one subsystem still owns a local projection or reference and the authoritative subsystem says the resource is missing at a boundary where it is expected to exist, treat that as a defect and investigate rather than silently reconciling or downgrading it to `NotFound`.
- This applies equally to provider-backed resources. If we still store a local row, ref, or handle for a provider-owned object and a later provider read says the object is missing where our model expects it to exist, defect by default. Only soften that into a typed outcome when external disappearance is intentionally modeled end-to-end as part of the product behavior.
- Read-only operations that span multiple systems must handle transient inconsistency from concurrent modification.
- Prefer compile-time enforcement of correctness invariants where possible.
- See [operation-types.md](operation-types.md) for the managed-operation replay model, [concurrency.md](concurrency.md) for linearization strategy, and [mutation-ordering.md](mutation-ordering.md) for cross-system ordering.

## Untrusted Data

- See [boundaries.md](boundaries.md) for parsing, validation, and trusted-vs-untrusted rules.
