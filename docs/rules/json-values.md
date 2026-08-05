# JSON Values

## Scope

This document covers structured JSON values.

## Rules

- Keep semantically structured JSON as JSON values in code and API payloads
  rather than stringifying it.
- JSON already includes JSON `null`. Do not use a nullable wrapper around JSON
  to mean absence.
- Use the owned absence representation when an owned JSON field may be absent,
  and reserve JSON `null` for a present JSON value or an intentional public JSON
  protocol.
- Use optional-key encoding only when the boundary shape intentionally omits the
  property. Optional-key encoding is not equivalent to owned absence; do not
  convert between them mechanically.
- Persist structural JSON in a database-native JSON or structured column type,
  not plain text.
- At the database boundary, represent JSON columns with a local wrapper or
  adapter so database `NULL` stays distinct from JSON `null`.
- Wrap outgoing bind values with the local JSON bind helper.
- The database dialect or adapter owns JSON bind compilation. Do not write ad
  hoc JSON casts at query call sites.
- Unwrap or schema-decode database JSON wrapper values at the query boundary
  when the distinction is no longer needed. See [boundaries.md](boundaries.md)
  for the general boundary conversion rules.
- Do not use reference or identity equality for potentially structural JSON
  values.
- Narrow to primitives first when that is the intent.
- Otherwise use a structural JSON equality helper.
- For structural equality and dedupe in collections, use local JSON equality
  and distinct-value helpers.
