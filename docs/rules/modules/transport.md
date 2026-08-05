# Transport

## Scope

This document covers transport lifecycle ownership.

## Lifecycle

- A transport layer should not control the lifecycle of application work unless that work genuinely depends on the transport being alive.
- If application work must survive transport reconnects, decouple it into a scope that outlives the transport.
- Treat the transport layer as a delivery pipe that can disconnect and reconnect without interrupting in-flight application work.

## Long-Lived Protocol Clients

- Long-lived server integrations should have one runtime service that owns the
  authenticated protocol context.
- Domain services should construct narrow clients from that shared runtime
  context instead of exposing a broad merged client.
- If a call site needs a one-shot client, the client acquisition and full use should happen inside one fresh transport scope.
- Do not expose protocol layers as reusable domain-service fields. Expose
  domain methods or a runtime-owned client factory instead.
- One-shot client helpers are only for calls where client acquisition and full
  use are wrapped together.

Protocol implementations may route responses by client id inside the protocol,
so multiple narrow clients can safely share one protocol context. Keep the
shared lifetime at the runtime boundary and keep call surfaces domain-specific.
