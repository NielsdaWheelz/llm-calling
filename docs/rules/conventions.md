# Conventions

## Scope

This document covers small implementation conventions that do not belong to a larger topic.

## Named Constants

- Extract a value into a named constant when the name conveys information beyond what the usage site already says.
- Keep a value inline when it is inherently part of the expression.

## Generic Type Parameters

- Constrain on composite types rather than decomposing their inner type parameters.
- Prefer `<S extends SomeType>` over separate type parameters for `SomeType`'s inner parts.
- Use indexed access such as `S["Type"]` to extract constituent types.
- Introduce separate type parameters only when callers need to specify or constrain them independently.

## Opaque Encodings

- Use base58 for small opaque tokens, handles, or identifier suffixes where punctuation hurts copy/paste or debugging.
- Use base64 for binary payloads in JSON transport bodies, structured byte
  transport, and shell interop.
- Use base64url only when the byte string itself must be URL/path/form safe; document the exception with `justify-base64url-over-base64`.
