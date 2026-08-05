# Effects

## Scope

This document covers effectful work, asynchronous work, background task
lifecycle, and scoped resource values.

## Rules

- Use explicit effect, task, result, or promise-returning functions when a
  function can fail in a handleable way or performs asynchronous work.
- Prefer the host runtime's native effect or async composition patterns unless
  a boundary clearly requires another shape.
- When in doubt, prefer the explicit effectful shape over hidden side effects.

## Background Work

- Every forked or background task must have its termination propagated,
  observed, or deliberately supervised.
- Avoid bare detached tasks. Any detached task must include a justification.
- Represent concurrent background work as returned task or effect values rather
  than forking internally.
- Compose concurrent background work at the call site so concurrency and error propagation are handled together.
- In streams, bind background producers to the stream lifecycle.
- Use structured concurrency primitives for races, joins, and parallel
  execution.
- Queue-based stream operators that need a forked producer should use a local
  helper that owns producer startup, shutdown, and error propagation.
- Dynamic concurrent work that must outlive its trigger should use a supervised
  task registry.
- Prefer keyed task registries when keyed deduplication is required.
- Every fork site must document its termination-propagation mechanism.

## Scope-Bound Values

- Do not let values produced by scoped acquisition escape their scope.
- Keep scoped value creation and use within the same scope.
- Do not return scoped values, store them in mutable references or closures, or
  pass them to long-lived background tasks.
- Resource-provider and dependency-wiring scopes must cover both acquisition
  and use.
- If a provider scope wraps acquisition but not use, the value may outlive its
  resources.
- When a value depends on scoped resources, the scope must wrap both acquisition
  and use.
