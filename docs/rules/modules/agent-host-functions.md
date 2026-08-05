# Agent Host Functions

## Scope

This document covers production host functions reachable from an agent run tool
and optional HTTP exposure routes.

## Host Function Families

- Host-function infrastructure lives in one agent-owned host-functions module.
- Run-tool host-function boundary types and loading live with the run-tool
  adapter.
- Guest-runtime exposure adapters live in a dedicated guest-runtime adapter
  area.
- Host-function implementations live in an owned host-function library.
- Resource-scoped functions, helper functions, agent-state functions, and
  agent-run functions stay in separate families.
- Resource-scoped functions have explicit function names and optional HTTP
  exposure through the resource-function route. Naming prefixes are conventions;
  the host-function machinery must not infer identity from a prefix.
- Helper functions are top-level host functions such as `skill.*`, `web.*`, `bytes.*`, `text.*`, and `random.*`.
- Agent-state functions are ordinary JSON-returning host calls for durable
  agent-scoped state.
- Agent-run functions are host functions with a run boundary adapter because
  their success path returns run-control state. They may return no guest value
  and may additionally record run-control requests.

## Boundary Contract

- Use one production host-function declaration primitive. Families supply
  context and boundary adapters; they do not own input decoding or registry
  identity.
- The shared host-function call method accepts JSON input. Family adapters
  decide whether the success path is ordinary JSON, run-tool control state, or
  another boundary result.
- Resource-scoped, helper, and agent-state functions use the shared JSON
  boundary through family-specific declaration helpers.
- Agent-run functions use the same host-function primitive with a run boundary
  that maps input-decode failures and intentional implementation failures to
  guest errors.
- Helper and resource-scoped function registries stay separate; aggregation is
  only for consumers that need a combined catalog.
- Every production host function declaration must provide its complete function name explicitly as `functionName`. Runtime lookup and error reporting use only the declared `functionName`.
- Every production JSON host-function declaration must expose explicit
  model-visible input, success, and declared error schemas at the declaration
  site. Use a shared empty-input schema for no-argument functions, explicit JSON
  `null` success for no result, and an explicit no-declared-failure schema for
  no declared model-visible failures.
- Family implementations fail with an internal declared-failure wrapper.
  Families may expose adapters for adapting an inner operation whose error
  channel is already a declared model-visible host-function payload.
- Omitted raw input at the guest-runtime or HTTP boundary is decoded as `{}`
  against the declared input schema. Explicit `null` and every other non-object
  input is rejected at the schema boundary.
- Host-function success and declared error payloads must encode to JSON without
  embedded NUL bytes before crossing the guest-runtime or HTTP boundary.

## Error Shape

- Host-call wrapper errors are owned boundary envelopes and use model-visible `type` fields.
- Declared host-function errors are already model-visible before they cross the host-call boundary.
- The declared-failure wrapper is internal host-function metadata; the boundary unwraps its payload and encodes only that payload through the host function's declared error schema.
- Never recursively rewrite arbitrary JSON to replace `_tag` with `type`.
- When a production error needs both host-runtime variant handling and
  model-visible JSON fields, declare it as a model-visible tagged error.
- Keep sandbox/runtime guest errors separate from host-call wrapper errors.
  Guest snippet runtime details can contain arbitrary JSON and must pass through
  unchanged.

## Guest Exposure

- Production guest-runtime exposure flows through:

```text
HostFunction -> RunToolHostFunction -> GuestHostFunctionBinding
```

- Raw guest binding definitions are for sandbox internals, tests, and examples,
  not production resource/helper/agent host-function declarations.
- Sandbox utilities that validate guest-runtime function names may keep
  runtime-specific terminology when they refer to runtime-level functions.

## Durable Semantics

- Pure guest-runtime code may replay after a crash; durable effects must cross a
  managed host-call boundary as managed operations.
- Replay-safe nondeterminism exposed to the model, such as `random.*`, is a production host function because it crosses the durable host boundary.
- Unexpected host defects must defect. Only intentional guest-level failures
  should become catchable guest errors.
- Idle requests are evaluated after `run` returns. Successful runs apply the idle request only when no unread waking context events arrived first; failed runs append `run_failed` and, when an idle request had already been accepted, `idle_request_not_applied`.
