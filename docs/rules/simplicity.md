# Simplicity

## Scope

This document covers repository-wide implementation simplicity rules.

## Rules

- When there are multiple reasonable ways to write something, prefer fewer lines and fewer characters within reason.
- Default to fewer code paths.
- Each additional code path should be justifiable.
- Do not add speculative API surface.
- Do not add optional parameters, options, or flags until a real call site needs them.
- Apply the same bias to error handling, schema validation, and branching.
- Do not add code paths for scenarios that cannot be constructed.
- Expose each capability in one primary form. Do not expose interchangeable duplicate APIs for the same capability.
- If a capability already exists in a module, prefer using it over introducing a near-duplicate.
- Do not add synthetic labels or alternate identifiers when they duplicate meaning already present in the primary spec or schema. Derive them only at the boundary that needs them.
- Do not add descriptive metadata for possible future debugging, filtering, auditing, or classification. A field needs a current owner: behavior, a required query, or a persisted domain fact that cannot be derived from concrete relationships.
