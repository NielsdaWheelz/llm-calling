# Function Parameters

## Scope

This document covers parameter shape rules.

## Rules

- Config, builder, and boundary APIs should take a single object parameter.
- Prefer collapsing multiple business fields into one named payload object rather than adding more positional parameters.
- Boundary-owned replay identity should live in a named field such as `replayKey` inside the boundary payload or options object.
- Do not thread internal replay bookkeeping through domain helper APIs;
  operation composition carries replay identity structurally.
- Do not thread already-proven scope or invariant fields through narrow local helpers just to restate an upstream check. Once a parent step has established the invariant, keep downstream helper parameters minimal unless the helper itself independently needs that field.
- Optional parameters should always live in an object parameter.
- An optional property or optional boundary key models omission, not semantic
  absence. Use the owned absence representation for owned successful absence,
  and use optional keys only when omission is the intended call or wire shape.
- Prefer shallow object shapes by default.
- Keep fields nested when they belong to a real named sub-concern, phase, or subsystem.
- Do not add user-provided naming or labeling fields to a product resource unless that field is part of the resource's real contract or behavior.
- Annotation-only names belong in notes, not in the core resource model.
- If an external provider requires a bookkeeping name or label, synthesize it inside the adapter layer rather than exposing it in the product API.
- Small pure helpers and primitives may remain positional when the arguments are obvious and tightly coupled.
