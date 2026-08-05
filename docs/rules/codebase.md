# Codebase

## Scope

This document covers technology ownership, repository-wide code organization, imports, generated files, and module boundary rules.

## Technology Choices

- Each repository has one declared primary runtime, package manager, build
  command, and test command set.
- Do not add parallel tools for the same job unless the repository is in an
  explicit migration.
- Use one declared primary persistence system for authoritative product state.
  Additional stores need a distinct ownership, consistency, or runtime role.

## Environment

- Keep the repository's environment contract in sync with every added, removed,
  or renamed environment variable.
- Every environment variable read by source code must appear in that contract.
- Each environment variable must state whether it is required or optional, and
  its default if it has one.

## Entrypoints

- Entrypoints should live in explicit entrypoint directories.
- Only entrypoints should perform startup side effects.
- Runtime-specific helpers may live in the same module as long as imports
  respect runtime safety.

## Imports

- Relative imports may cross only nearby local boundaries.
- If a relative import would need long parent traversal, use a declared
  package, module, or self-referential import instead.
- Do not re-export symbols from other modules. Import each symbol from its defining module.

## Module Files

- Do not create generic barrel files that obscure ownership or make local and
  package imports asymmetric.
- If a module has multiple files and a primary interface file, give that file
  an explicit primary-interface name.
- If a module has one file, do not nest a single file in a directory. Name the
  file after the module.

## Generated Files

- Commit generated files.
- Generated files must have an owning generation command.
- The repository check command must fail when generated files are missing,
  non-canonical, or stale.
- Generated-file validation should be deterministic and match the artifact's ownership boundary: check committed files are present, canonical, and fresh with respect to declared inputs and configuration. Do not partially rerun generation as a proxy for correctness unless the command is explicitly an audit command.
- Generated-file owners should document the declared inputs, configuration, and
  refresh command.

## Embedded Guidance

- Built-in workflow guidance lives as plain text files under one owned guidance
  directory.
- Directory names define stable guidance namespaces.
- Runtime-exposed commands, functions, or tools should derive reference
  guidance from their source definitions when possible.
- Virtual guidance ids use an explicit prefix that identifies the generated
  reference namespace.
- Authored guidance owns workflows that span multiple functions. Function
  reference guidance lives with the function definition.

## Host And Target Runtimes

- The host language owns application logic.
- Database queries, shell scripts, templates, generated programs, and other
  emitted code are target languages that run in other runtimes.
- Prefer business rules, branching, fallback policy, and domain invariants in
  the host language whenever reasonably possible.
- Use target-language code when the target runtime is the natural owner of the work, not just because the expression is shorter there.
- Keep database query languages focused on database-shaped work such as set
  filtering, joins, ordering, aggregation, and atomic mutations.
- Keep shell focused on process and OS orchestration.

## Module Boundaries

- A module is an explicit package or directory boundary.
- Reusable modules are not owned by one product, service, or entrypoint.
- Lowest-level reusable substrate modules may be used by shared and product
  modules, but they must not import semantic sibling modules, product modules,
  build tooling, or app code.
- Shared persistent tables have one storage owner. Semantic behavior still
  lives in the module that owns the concept.
- External functionality may be consumed by any module.
- Internal functionality is only for a module and its submodules.
- Default to internal unless functionality is clearly consumed externally.
- Internal code must be visibly marked by path, package, or language access
  control.
- If a tool cannot use the normal internal marker, use an equivalent marker with
  the same boundary semantics.
- Server-only code must be visibly marked.
- Browser-facing UI support code must be visibly marked.
- Module-owned browser support code may be shared by browser code and server
  code only when the imports remain runtime-safe.
- Shared browser substrate is not itself an app.
- App-level browser code stays separate from reusable browser substrate.
