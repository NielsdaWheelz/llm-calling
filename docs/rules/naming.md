# Naming

## Scope

This document covers global naming grammar for identifiers and observability labels.
Operation verb semantics such as `ensure...`, `require...`, and `validate...` belong to the owning semantic docs, not this grammar doc. For managed operations, see [operation-types.md](operation-types.md).

## Enums

- Enums are `PascalCase` strings.

## Identifiers

- String-valued identifiers in a global namespace should use dot-delimited PascalCase.
- Service tags, error tags, and local union discriminators should use flat PascalCase with no dot.

## Observability

- Observability names use a different grammar to align with OpenTelemetry conventions.
- App-owned span names and similar nominal observability labels should use
  dot-delimited PascalCase.
- Protocol spans should follow the applicable OpenTelemetry semantic convention.
- Span and log attribute keys are field paths, not nominal labels.
- Resource, span, and log attribute keys should use lowercase dotted field
  paths.
- Resource, span, and log attribute keys must be stable. Do not derive attribute
  keys from user input, provider data, request counters, entity ids, feature flag
  names, or other unbounded runtime values.
- Dynamic or unpredictable observability data belongs in attribute values or a
  deliberately structured payload under a stable key.
- Prefer OpenTelemetry semantic-convention keys when one fits the concept.
- Custom application-specific attribute keys should live under one repository-owned
  prefix.
- Do not use camelCase attribute keys.
- Do not reuse nominal-identifier PascalCase grammar for observability attribute keys.
