# Boundaries

## Scope

This document covers how values should change shape when they cross a boundary
between owned code and external systems, between modules, or between trust
levels.

## Goal

Boundary conversion produces a representation narrow enough that downstream code
never needs to re-examine or re-validate the value. The representation itself
carries the guarantee. A validated `FooKey` cannot be an unvalidated string. A
classified absence cannot be an ambiguous `null`. Internal code works with
these narrow values and never looks back at the raw form.

## Boundary Contracts

- Boundary adapters accept only the input states their contract intentionally models.
- Unexpected boundary input states are defects, not alternate branches.
- Do not add fallback parsing, placeholder owned values, persisted invalid state, or other recovery paths for inputs we do not expect to receive.
- If an unexpected input state needs product behavior, first make it an explicit modeled adapter output or domain state, then handle that modeled value.

## Untrusted Input

Untrusted input comes from outside the system: API requests, webhooks, third-party APIs, files, user input, or generated output.

- Parse, validate, and narrow to a specific typed form where untrusted data enters. Failures are typed errors — bad external input is expected.
- The result of parsing is the narrow internal representation: a validated
  value, domain object, or internal identifier. That representation flows
  through all downstream code.
- Do not re-parse or re-validate deeper in the stack. If the type is right, the value is right.
- Parse only the single expected shape. Do not add branches for alternative shapes "just in case" — those are dead code that hides bugs.
- See [keys-and-identities.md](keys-and-identities.md) for `parseX`/`assumeX`, brands, and unsealing at end-user boundaries. See [errors.md](errors.md) for classifying `null` and absence at the boundary.

## Human Draft Input

Human-facing drafts are the ingress for typing convenience.

- Keep draft state as text and normalize presentation affordances in explicit draft-formatting helpers such as `formatXDraft`.
- Draft formatting may strip punctuation, insert display separators, or add obvious typing defaults before submission.
- Submit handlers parse the draft into the owned canonical type before crossing
  an API boundary.
- API schemas and backend parsers for values with draft formatting accept only
  the canonical wire shape. Do not duplicate client convenience normalization at
  the backend boundary.
- If another client needs the same convenience behavior, put an equivalent
  draft-formatting step at that client boundary instead of widening the backend
  parser.

## Generated Output

Generated output from an LLM or other model is untrusted until the owning ingress helper decodes it and accepts it into owned types.

- Validate objective local invariants at the generated-output ingress: schema
  shape, enum membership, identifiers, handles, and renderable asset names.
- Do not normalize valid generated prose to satisfy style preferences. Encode style preferences in the prompt and preserve accepted prose as authored.
- After accepted generated output is stored or returned through our own typed transport, treat it as trusted same-system data.
- Use `sanitize` only for helpers that intentionally remove or rewrite unsafe content. Use `accept`, `parse`, or a domain-specific conversion name for helpers that validate or brand without rewriting.

## Trusted Data

Trusted data comes from systems we control: our database, our own modules, our persisted state.

- If trusted data does not match the expected shape, that is a defect — something we wrote or stored is wrong.
- Do not silently re-normalize trusted data. If a database value is not in the right form, the write path or schema is the problem — fix it there.
- Do not add redundant validation across layers. Tighten the source or the persisted representation instead.
- Branded internal IDs and same-system refs are trusted after the boundary that accepted or minted them. If a later service observes that such a ref points at a missing row where existence is an invariant, defect at that natural observation point. Do not add defensive preflight checks solely to make impossible defect paths cheaper.

## Same-System Transport

Responses across owned transport boundaries are same-system data after the
transport decoder has accepted them.

- Decode the wire shape at the client boundary, then treat the decoded value as owned typed data.
- If a decoded same-system payload contains owned absence fields, keep those
  fields in the owned absence representation while passing the payload through
  model and view helpers. Do not convert them to `null` just because the value
  is near UI code.
- UI components should not re-validate domain invariants that the server write path or API schema is responsible for preserving.
- If decoded same-system data violates an owned type guarantee later, treat that as a defect and fix the source or schema.

## Internal Representation

- Keep values in their narrowest, richest typed form. Do not widen back to primitives or strings until the moment an external consumer requires it.
- Do not pre-convert to lossy forms into intermediate variables. If a value has meaning beyond its primitive shape, it should have a type that reflects that meaning — not flow through code as a bare `string` or `number`.
- When the system defines a structure and controls both its writer and reader,
  represent semantic absence with the owned absence representation.
- Owned domain, service, durable-operation, replay, memoized, API, and
  same-system payload schemas use the owned absence representation for semantic
  absence. Do not use raw nullable values just because the encoding can carry
  `null`.
- Raw row, third-party, browser, framework, or intentional public protocol
  schemas may use raw nullable values only when that boundary contract
  materially speaks null. Convert those nullable values before they leave the
  boundary adapter.
- Owned service, domain, durable-operation, replay, memoized, API, and
  same-system payload APIs use the owned absence representation for normal
  successful absence.
- If absence is an expected application failure, model it as a typed error
  instead of owned successful absence.
- If absence violates an invariant, defect at the observation point with a
  justified error instead of returning owned successful absence.
- Raw `null` is reserved for boundaries whose contract materially speaks null:
  nullable database columns, third-party SDK or API payloads, browser/framework
  interop, local frontend component state, intentional public JSON protocols,
  and library or service contracts we do not control.
- Normalize boundary `null` into the owned absence representation at ingress as
  soon as the value enters owned logic.
- Do not mechanically replace optional-key semantics with owned absence
  semantics. An optional key means the wire or persisted object may omit the
  property; an owned absence field means the property carries semantic absence.
  Change between them only when intentionally changing that boundary shape.

## Outgoing Conversion

- Convert to a lossy or primitive form only at the moment it is needed — inline at the consumption site.
- Convert owned absence back to `null` only in the final adapter that writes a
  nullable database column, calls a null-speaking external API, or emits an
  intentional public JSON protocol.
- Convert owned absence to an omitted key only in the final adapter for a
  boundary whose contract intentionally uses missing properties for absence.
