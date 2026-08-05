# Retries

## Scope

This document covers retry boundary semantics and repository-wide retry policies.

## Boundaries

- A retryable error must not cross a retry boundary unchanged.
- Retry exhaustion is a defect by default.
- Use a retry-and-defect helper unless the operation explicitly models
  persistent dependency unavailability as an intended outcome.
- Classify exhaustion by the invariant or result the owning operation is expected to establish when our code and its dependencies are operating correctly.
- Use a retry-or-exhaust helper only when retry exhaustion is itself a
  first-class modeled outcome rather than a softer restatement of "the
  dependency stayed down."
- Retry-and-defect helpers defect with a retry-exhaustion wrapper around the
  final retried error.
- Do not choose retry-or-exhaust behavior based on whether a dependency feels
  "optional", "best effort", or "non-critical" in the local code path.
- Do not vary exhaustion semantics ad hoc by dependency vendor or by feature. Classify by the owning operation contract.
- Do not surface synthetic "Unavailable", "TransientFailure", or similar middle-ground errors from a required dependency after a retry boundary. Either classify a terminal modeled failure or keep retrying internally until success or retry exhaustion defects.

## Policies

- Define retry schedules in one central policy catalog as categorical policies.
- Creating a retry schedule anywhere else requires `justify-retry-schedule`.
- Retry schedules follow the schedule-shape rules in [timing.md](timing.md).
- Infrastructure retry policies are for transient infrastructure failures.
- External-service retry policies are for transient third-party service
  failures.
- Choose the policy by the category of dependency being called, not by use case.

## Exhaustion

- Transient database failures inside the applicable retry budget are expected
  and retried.
- Persistent dependency failure beyond the applicable retry budget is an unexpected abnormality.
- If the operation is expected to succeed or establish or observe an invariant when dependencies are operating correctly, retry exhaustion is a defect and must crash loudly.
- Retry exhaustion is a modeled result only for operations whose intended
  semantics explicitly include persistent dependency unavailability as an
  outcome, not for silent reclassification of a required dependency outage.
- Repair-loop exhaustion and post-retry invariant failures are also defects unless the caller explicitly models them as typed errors.
- Retry or reconcile transient dependency failures internally until the owning
  operation can either succeed or classify a terminal modeled failure.
- If that internal retry or reconciliation loop exhausts first, defect rather than reclassifying the outcome into a softer dependency-unavailable application error.
- For durable-operation dead-letter retry and repair rules, follow
  [operation-types.md](operation-types.md#dead-letters-and-ownership-state).
- Server-side retries are bounded.
- Unbounded retry is only for client-side components whose sole job is to maintain a connection.

## Placement

- Place retry logic as close to the operation being retried as possible.
- Retry the smallest necessary unit of work.
- Do not structure retries so it is ambiguous whether a deeper layer is already retrying the same work.

## In-Memory State

- Do not mutate in-memory state across a retry boundary.
- In-memory state created outside a retry boundary and mutated inside it is not rolled back on failure.
- Create such state inside the retryable block if it must reset on each attempt.
- Otherwise, mutate it only after the retryable block returns.
