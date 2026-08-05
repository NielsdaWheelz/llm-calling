# Frontend

## Scope

This document covers browser-facing frontend code, UI state, UI boundaries, and
route-owned data loading.

## State

- App-owned empty component state uses one explicit empty-state representation.
  This applies to local UI state such as selected rows, open dialogs, draft
  requests, pending actions, and loaded resources.
- This local empty-state rule is only for component state and browser/framework
  interop. It does not change the shape of decoded same-system data.
- Decoded same-system API and domain values keep their owned absence
  representation in frontend code, including inside reusable frontend models
  and view helpers.
- It is fine for local loading or empty state to wrap a decoded DTO that itself
  contains owned absence fields. Do not flatten those DTO fields into local UI
  emptiness.
- Do not use magic sentinels such as `""`, `0`, `-1`, or ad-hoc placeholder values to mean absence, idle state, or none.
- Empty draft text is not semantic absence. `""` is valid for raw input state only; semantic absence should use the owned absence representation or an explicit typed variant.
- `boolean` is only for genuine yes/no state. Do not use `false` to stand in for "no current value".
- Framework or interop absence stays at the framework boundary.
- Normalize framework or browser absence into app-owned state immediately unless there is a strong reason not to. See [boundaries.md](boundaries.md) for the general ingress rule.
- Do not use domain absence wrappers as the outer local component-state wrapper
  for loading, selected, open, or draft state.
- Prefer derived state over duplicated state.

## Variants

- Keep expected control-flow variants explicit.
- For enum casing and exhaustiveness, follow [naming.md](naming.md) and [control-flow.md](control-flow.md).
- Omission is the default. Do not add `"Default"`-style variants unless they represent real logic distinct from absence.
- Optional variant fields represent omission, not a default variant.
- Unexpected UI invariants should fail loudly.

## Boundaries

- Map domain and API errors to UI messages in one helper near the screen
  boundary. Name these helpers `*ErrorMessage` and match exhaustively on
  structured error variants.
- External strings keep external spelling.
- Product-facing operational names use the current product brand unless the boundary explicitly requires another spelling.

## Routing and Data

- Navigable frontend context belongs in the URL.
- Route-entry state belongs to the router, not component effects or ambient browser reads during execution.
- Use route-owned loading for route entry. Inside an already-valid route, use
  the standard query/loading primitive for reactive component-local queries and
  explicit async handlers or state updates for mutations and one-off actions.
