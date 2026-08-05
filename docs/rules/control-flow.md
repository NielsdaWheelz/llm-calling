# Control Flow

## Scope

This document covers exhaustive branching and race-safety rules.

## Exhaustiveness

- When branching on a value with a known finite set of possibilities, use exhaustive matching. That means that adding a new possibility to the producer of the value should cause a type error in the consumer until it explicitly handles that possibility. If new possibilities could possibly be added to the consumer without creating type errors, this rule has been violated.
- Good patterns:
  - Use the language or framework's exhaustive-match primitive when one exists.
  - Use a runtime unreachable-branch check for impossible branches.
  - Use a compile-time assertion for narrowed finite variants when the language supports one.
- This applies to errors as well. Do not erase finite error channels with catch-all handlers that discard or collapse distinct errors.
- Prefer tag-specific or variant-specific error handlers for finite error sets.

## Races

- Do not race work that performs a destructive or non-idempotent operation unless losing the result is acceptable.
- Race primitives usually discard or cancel losing results.
- If the losing task performed an irreversible side effect, the side effect may
  be committed while the result is lost.
- When concurrent work needs to coordinate around destructive operations, route
  signals through a single serialization point.
