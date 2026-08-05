# Generated Text

## Scope

This document covers escaping, quoting, and representability at generated-text boundaries.

## Rules

- Generated-text helpers own escaping and quoting for the target boundary.
- If a value's malformedness is knowable locally, validate it once into an owned type before rendering.
- Do not hide local validity checks inside the renderer when the caller could have provided a stronger owned type.
- For shell token boundaries, validate dynamic text as `ShellArgumentText` before passing it to `shellQuote`.
- Keep fixed host-language values in the host language. Either validate them once into an owned type at definition time or quote them inline at each use site.
- Use shell variables for genuinely dynamic runtime values.
- Keep shell variables quoted at each use site.
