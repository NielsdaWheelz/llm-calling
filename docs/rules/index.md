# Docs

## Role

This directory is the canonical home for repository documentation.

## Goals

- MECE organization: documents are mutually exclusive and collectively exhaustive.
- Concision
- Clear boundaries

## Docs

### Correctness and concurrency

- [correctness.md](correctness.md): abnormality classification and system invariants
- [operation-types.md](operation-types.md): managed-operation model, replay, durable workflows, and composition rules
- [concurrency.md](concurrency.md): linearization and concurrent execution
- [mutation-ordering.md](mutation-ordering.md): ordering mutations across systems and module boundaries
- [retries.md](retries.md): retry policies and exhaustion handling

### Data and types

- [boundaries.md](boundaries.md): data representation at ingress, internal, and egress edges
- [errors.md](errors.md): error and defect modeling, null classification
- [keys-and-identities.md](keys-and-identities.md): identity naming, validated types, and sealing
- [json-values.md](json-values.md): structured JSON values
- [resource-lifecycle.md](resource-lifecycle.md): resource publication, reservations, setup state, and lifecycle row shapes
- [tagged-unions.md](tagged-unions.md): tagged variants versus domain-record shapes
- [generated-text.md](generated-text.md): escaping and quoting at generated-text boundaries

### Runtime composition

- [effect.md](effect.md): effectful work, background tasks, and scoped values
- [effect-services.md](effect-services.md): services versus helpers
- [layers.md](layers.md): runtime layer kinds and wiring rules

### Code style

- [cleanliness.md](cleanliness.md): dead code, ownership, duplication, and complexity reduction
- [simplicity.md](simplicity.md): fewer code paths, no speculative surface
- [naming.md](naming.md): naming grammar for identifiers and observability
- [function-parameters.md](function-parameters.md): parameter conventions
- [control-flow.md](control-flow.md): exhaustive branching and race-safety
- [overrides.md](overrides.md): justified escape hatches and type assertions
- [conventions.md](conventions.md): small conventions (constants, generics, encodings)

### Platform

- [codebase.md](codebase.md): technology ownership, repo structure, imports, and module boundaries
- [database.md](database.md): relational schema, queries, and transactions
- [frontend.md](frontend.md): browser-facing UI state, boundaries, and route-owned data
- [testing.md](testing.md): behavior-focused testing standards and test tiers
- [timing.md](timing.md): schedules and timing constants
- [polling.md](polling.md): polling rules

### Modules

- [modules/index.md](modules/index.md): service, infrastructure-module, and feature docs

## Placement Rules

- Each rule lives in exactly one document.
- Put content in the narrowest document that fully owns it.
- Link to related docs instead of restating them.
- If two docs need the same text, the split is wrong.
- If a document covers multiple unrelated topics, split it.
- Small docs are fine when they keep ownership and boundaries sharp.
- Keep repo-wide rule docs flat until a topic clearly needs its own directory.
- Use subdirectories for service-owned, module-owned, or feature-owned docs when that keeps them separate from repo-wide rules.
- Avoid over-categorized hierarchies and umbrella docs with weak boundaries.

## Rule Shape

- Prefer unconditional rules.
- Do not write soft rules with words like `usually`, `generally`, or `normally`.
- State the unconditional rule or the explicit exception.
- Prefer narrowing scope or splitting a rule over adding exceptions.
- If a rule needs many exceptions, the rule or the document boundary is probably wrong.

## Ownership

This file defines the documentation system itself: purpose and placement rules. It does not own product or codebase rules beyond that.
