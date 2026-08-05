# Keys And Identities

## Scope

This document covers identity and authority naming, validated identity types,
sealing, and related naming rules.

## Id

- Meaningless private identity should use UUID-backed `*Id` values.
- `*Id` means private meaningless identity.
- Do not expose `*Id` directly at end-user boundaries.

## Key

- Meaningful identity should use `*Key` values.
- Prefer structured keys for meaningful identity.
- Do not pass raw anonymous structured data for owned meaningful identity.
- Give owned structured keys a named type and parser or schema.
- Do not replace meaningful identity with meaningless UUIDs just because it identifies something.
- Use a canonical string conversion only when a boundary genuinely requires a
  string form of a structured key.
- Do not use canonical string conversion as the default persistence format for
  structured keys.

## Spec

- `*Spec` means a structured semantic description, not private identity.
- Use `*Spec` when the full structured value is the thing that should be selected, persisted, or compared.
- If a boundary needs stable meaningful identity for a spec, derive a `*Key` from the spec rather than calling the spec an `*Id`.
- Persist the structured spec itself when instances must stay locked to their selected terms even if a live catalog later changes.

## Handle

- `*Handle` means outward opaque identity or an established domain capability handle.
- Sealed handles are the default outward form of internal identity.
- Use sealed handles for outward opaque identity across product, service,
  infrastructure, and broad web/API boundaries.
- Outward opaque identity should be named as a handle at boundary surfaces.
- Handles identify server-owned entities; handles do not authorize actions.
- Do not use short handles for authority-bearing values.
- `ShortHandle` is the named exception to sealed `*Handle`: it is a compact alias to a typed server-side target.
- Short handles are convenience aliases to server-owned entities, not authority.
- Use short handles only for compact human- or tool-friendly references that
  are expected to be seen, copied, typed, displayed, embedded in URLs, included
  in prompts, or passed through tool calls.
- Resolve short handles server-side to the expected typed target, then enforce scope and ownership.
- Do not call an outward handle `id` at broad or weakly scoped boundaries such as API payloads, transport payloads, route params, logs, or reusable service APIs.
- Strongly scoped command/function payloads may use `id` for an outward handle or
  short handle when the command name already gives the domain and `id` is the
  simplest call shape.
- Tool-facing command payloads should prefer `id` for scoped short-handle inputs
  when the function name already gives the target domain, but the schema must
  still be an entity-specific short handle.
- When a scoped `id` carries an outward handle or short handle, docs and error messages should make that explicit enough that callers do not confuse it with private database identity.

## Token And ApiKey

- `*Token` and `*ApiKey` mean outward bearer or capability strings.
- `*Secret` means outward or setup-time secret material.
- Tokens, API keys, and secrets are authority, not identity pointers.
- Use generated random credential material for authority-bearing tokens, API keys, and secrets.
- Do not use sealed entity IDs as bearer credentials.
- Use raw bootstrap capability tokens only when the token itself is the authority and must exist before persistence.
- Persist random credential material, a verifier, or both according to the credential's lifecycle contract.
- Use a domain-separated fixed-size verifier column such as `*_hash` for credential auth lookup, even when the raw credential is also stored for repeated display or operator workflows.
- Raw credential columns such as `token` or `api_key` are product state; verifier columns such as `token_hash` or `api_key_hash` are auth/index state.
- Resolve bearer credentials by verifier lookup, then load the row and enforce lifecycle and scope. When a credential is presented alongside an entity handle, verify the presented credential against the row's verifier hash. Do not make direct raw-credential comparison the authorization path.
- SHA-256 verifiers are only appropriate for high-entropy generated credentials; human-chosen or low-entropy secrets require a slow password KDF.
- Model credential lifecycle explicitly: expiry, revocation, rotation, one-time display, repeated display, and audit behavior must be product choices rather than side effects of entity identity.

## Ref

- `*Ref` is only for lower-layer references such as provider-owned or infrastructure-owned pointers.
- Do not use `*Ref` for outward opaque values.
- Do not use `*Ref` for transport wrappers.

## Specific Names

- Prefer the most specific honest domain name.
- `Id`, `Key`, `Handle`, `Token`, `ApiKey`, and `Ref` are fallback categories, not the only allowed names.
- Prefer a sharper domain term when it captures the semantics more directly than a generic suffix.
- Use the same specific name across boundaries when the concept itself is the same.
- A specific name should still respect the underlying semantics of the fallback category it replaces.

## Validated Types And Brands

- Use validated types plus parsers or schemas for canonical values whose
  malformedness is knowable locally.
- Use nominal types or brands for provenance-backed internal IDs, outward
  handles, outward tokens, and lower-layer refs.
- Outward sealed handles should extend a validated local wire-text type.
- Outward credential values should extend a validated generated-token wire-text
  type.
- Concrete short handles must use entity-specific brands and schemas at typed boundaries; generic `ShortHandle` is only for shared short-key infrastructure and truly entity-agnostic utilities.
- Use owned named types and parsers or schemas for semantic structured values
  rather than passing raw anonymous structured data.
- Add branding when the semantics need nominal distinction beyond the structure itself.
- For canonical owned values, prefer a shared `parseX` and `assumeX` pair next to the owning type.
- `parseX` may normalize once at ingress, then validate and return the canonical owned value. See [boundaries.md](boundaries.md) for the general ingress rule.
- `assumeX` requires the value to already be canonical and defects otherwise.
- Do not use `parseX` naming for helpers that convert outward opaque handles or tokens into internal identity or authority. Use names such as `unsealX` or `resolveX`.

## Sealing

- Use sealed outward values by default for outward opaque identity.
- User-controlled infrastructure is a product boundary.
- Service-to-service product API boundaries should use sealed handles rather than exposing private IDs directly.
- Admin and debug surfaces may expose raw private IDs or tokens when inspection is the point.
- Intentional short aliases to typed server-side targets, such as `ShortHandle`, are allowed only for visible human or tool ergonomics.
- Short handles must be resolved server-side to the expected target type, then authorized against the current scope.
- Internally, always use private IDs and internal references.
- Successful unseal or resolve is what converts an outward handle into the owning internal type.
- Typed web and API schemas may validate outward sealed wire text at the
  boundary, but entity-specific unseal or resolve still happens later.
- Entity-specific `unsealX` and `resolveX` helpers own classification and conversion from outward wire text into private internal types.
- Raw unsealed strings must not escape boundary helpers.
- API and web transport schemas may carry typed outward handles, but they
  should not unseal directly into private IDs during decode.
- Handler and service code owns unseal, classification, and conversion from malformed outward values into domain errors.
- If an outward opaque value wraps generated private identity, expose
  entity-specific seal and unseal helpers.
- If an outward opaque value must wrap meaningful structured identity, seal a canonical JSON string or buffer rather than inventing an ad hoc string encoding.
