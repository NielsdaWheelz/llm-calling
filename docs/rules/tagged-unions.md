# Tagged Unions

## Scope

This document covers tagged variants versus domain-record shapes.

## Rules

- Use a conventional tag field when a value's primary meaning is one variant of
  a sum type.
- Use semantic fields such as `provider`, `status`, `kind`, `role`, or `state` when a value is primarily a domain record.
- Persisted data, API payloads, and cross-module boundary records look like domain data rather than type-system artifacts.
- When a semantic field selects the schema of a subfield within one domain
  record, prefer that semantic field over a generic tag field.
- Use a generic tag field at a boundary only when the boundary value itself is
  fundamentally a tagged union.
- Use flat payloads when one layer consumes the whole variant payload and there is no meaningful internal grouping.
- Use nesting when different layers own different parts of the value.
- Do not add a nested discriminator unless the nested value is itself a real union.
