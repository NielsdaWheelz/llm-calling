# Cleanliness

## Goal

Make the codebase pristine: minimal, elegant, and one owner per concern. Remove
every line, file, abstraction, and test that does not earn its place.

## Dead Code

- Delete unreferenced functions, exports, types, constants, files, and styles,
  including code reachable only from tests.
- Delete branches for states that can no longer occur.
- Delete branches kept only for callers, payloads, routes, or storage formats
  that no longer exist.
- Delete artifacts orphaned by earlier deletions.

## Legacy And Compatibility

- Remove backward-compatibility shims, dual old/new code paths, migration-era
  branches, and silent fallbacks that keep old behavior alive.
- Where an old model and new model both work, keep only the new one.
- Rename or delete anything whose name or comment marks a finished era:
  `legacy`, `compat`, `cutover`, `bridge`, `fallback`, `migration`,
  `temporary`, or version suffixes.
- Treat a stale doc as a lead. Hunt the same dead concept through code and
  tests.

## Duplication

- Collapse repeated logic to a single owner: mutation flows, fetch and polling
  loops, state machines, pipelines, validators, normalizers, derived-state
  calculations, constants, and near-identical branches.
- If a value is sanitized, validated, or derived in more than one place, cut it
  to one.
- If two abstractions expose the same capability, keep one.
- If a registry or cache mirrors a source of truth that already exists, delete
  it.
- Leave small local formatting duplication alone. Dedupe only when it is large
  or dangerous.

## Oversized Units

- Split files that mix unrelated concerns: routing, transport parsing, business
  logic, mutation, rendering, and persistence.
- Split functions that run unrelated phases in one body.
- Split oversized local state into units matching real ownership, not generic
  buckets.

## Ownership And Layers

- One concern has one owner. If two modules can mutate or derive the same state,
  collapse to the canonical owner.
- Keep boundary and controller files as thin dispatchers.
- Put real behavior in the unit that owns it, not in a new middle layer.
- Parse transport shapes at the boundary and pass typed values inward.
- Keep raw payloads out of domain and service APIs.
- Keep business logic out of transport handlers.
- Replace cross-module imports of private helpers by moving the code to its
  owner or exposing one public function.
- Remove barrels and re-exports that hide where a symbol lives.
- Shrink every module's public surface to what is actually called.

## Indirection

- Inline one-use helpers, types, constants, and wrappers unless they hide real
  incidental complexity.
- Remove helper stacks that only rename property access.
- Remove staging variables that only move the eye.
- Remove caches that only paper over indirection.
- Remove abstractions kept only because an older one existed.
- Prefer explicit local code over reuse that does not pay for itself.
- Prefer a little duplication over a hollow generic helper.
- Remove optimization that adds complexity without a measured need.

## Services

- Decompose services around owned capabilities with deep, typed public
  contracts.
- Use pure helpers for local stateless logic.
- Create a service when there is owned behavior, state, infrastructure, or a
  dependency boundary that other code must rely on.
- For each service, own the capability end to end: state, invariants,
  persistence, retries, provider/runtime wiring, and lifecycle rules.
- Expose only a small semantic public interface: named commands/queries, one
  object parameter at boundaries, typed inputs/outputs, typed errors, and
  explicit transaction/replay semantics when relevant.
- Keep everything else internal.
- Edge adapters such as HTTP handlers, remote-call handlers, CLI commands,
  database rows, vendor SDKs, and UI
  transport may parse, validate, narrow, translate, and invoke the service.
  They must not own business rules.
- Other modules may call only the public service, handle, or API, never another
  module's tables, private helpers, SDK clients, or wiring.
- After ingress, keep values in rich owned types.
- Model expected failures explicitly.
- Defect on impossible states.
- Do not add speculative options, duplicate APIs, generic wrapper types, or
  fallback branches.
- Enforce ownership boundaries, service-private wiring, retry/mutation
  boundaries, and narrow public surfaces.
- Public services return named operations.
- Runtime layers close their own dependencies.
- Adapters translate at edges.
- Provider-specific details sit behind driver or client services.
- Use service boundaries inside one repository as much as across processes.
- Decompose around information-hiding decisions, deep modules with small
  interfaces, ports/adapters where adapters translate external technology into
  an inner application API, and services organized around business
  capabilities.

## Types

- Make illegal states unrepresentable.
- Delete downstream guards that exist only to handle states the type system can
  rule out.
- Give unions discriminants so consumers stop reconstructing them by hand.
- Classify each "nullable until later" value at the boundary as valid data, a
  typed error, or a defect.

## Error Handling

- Narrow broad catches.
- Keep catch-alls only at real boundaries, and there map errors explicitly.
- Remove best-effort swallowing that hides failure.
- Fail fast.

## Tests

- Delete tests that add no confidence.
- Delete tests that retest the same behavior across layers.
- Delete tests that restate implementation constants.
- Delete tests that assert structure, styling, or wiring instead of observable
  behavior.
- Delete tests that exist only to prove a dead format stays dead.
- Remove production seams kept only for tests: test-only exports, environment
  branches, and fake injection points.
- Replace tests that mock internal modules and inspect their wiring with
  behavior tests at the true owner.
- Assert outcomes through the public surface.
- Push data-layer assertions down to data-layer tests.
- Share setup instead of copy-pasting it.

## Naming

- Rename to describe current ownership and behavior, not history.
- Drop version suffixes.
- Remove a constant whose name says less than the literal did at its call site.

## Discipline

- Every change must lower total complexity.
- Never trade dead code for clever code.
- Confirm a symbol is truly unused before deleting it.
- Everything that remains must earn its place.
