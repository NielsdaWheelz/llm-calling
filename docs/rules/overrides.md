# Overrides

## Scope

This document covers justified escape hatches in source code: overrides, intentionally retained dead code, and type assertions.

## Overrides

- Every compiler, type-checker, linter, formatter, or static-analysis override
  must include a repository-standard justification token.
- The justification must explain why the override is needed and what invariant
  keeps it safe.
- Prefer a small documented override at the exact line over a broad file-level
  or project-level suppression.

## Dead Code

- Delete dead code by default.
- If an exported symbol is intentionally kept without current call sites because
  we expect to want it again soon, justify it with the repository-standard dead
  code justification token.

## Type Assertions

- Every unsafe or narrowing type assertion must include the
  repository-standard type-assertion justification token.
- The justification must explain why the assertion is safe and why a safer
  typing approach is not feasible.
- Language-level literal or const assertions that cannot change runtime
  behavior may be exempt when the repository documents that exemption.
