# Codex refusal of an unpublished function

Status: accepted (owner decision, 2026-10-13)

Date: 2026-10-13

Narrows, for raw `function_call` items only, the raw-event authority rule in
[provider-native turns](../provider-native-turns.md#native-capability).

## Decision

Under `CodexNativeOptions(builtin_tools="disabled")`, a raw `function_call`
naming a function the session never declared is a model error, not native
authority, when stock Codex answers that call itself with exactly
`unsupported call: <name>`. Codex has no handler for the name and runs
nothing; the model reads the refusal and the turn continues. The provider
emits no event for the proposal or for its refusal.

The proposal is held, not trusted. The raw `function_call_output` with the same
`call_id` must carry exactly that text. Any other answer means something
handled the call: the held `AgentToolUse(started)` is emitted then and latches
authority. A proposal still unanswered at `turn/completed` latches authority
too. Native command, file, MCP, dynamic-tool, custom-call (`exec`), approval and
every other authority surface is unchanged: first-class and terminal-poisoning.

## Why

Before this, the raw proposal alone latched authority, so a model's guess at a
tool name it was never given ended the turn as `ProtocolDefect`, and consumers
that reject `AgentToolUse` ended their answers as containment violations
(nexus harness journey T4e), after the model had already received Codex's
refusal. Codex's own refusal of that exact call is the evidence that nothing
ran. The name alone is not: a stock handler can exist for a name the session
did not declare, and its answer then differs from the refusal.

## Costs

- Detection of an undeclared call that Codex did handle moves from its proposal
  to its answer or to turn completion. The native handler ran in either case.
- The match is the exact refusal text of the pinned stock binary (present in
  0.160.0). A namespaced name or reworded refusal fails closed as authority; a
  native upgrade must requalify the text.
- Hosts never see refused proposals. Recording such attempts is a consumer
  concern.

## Evidence

Deterministic protocol-peer cases in `tests/test_agent_codex_control.py` cover
the exact refusal, an unanswered proposal, a handler-answered proposal and a
custom call. Stock Codex 0.160.0 with a fake model on a sealed network (nexus
tools harness, journey T4e): both unpublished calls (`memory__search` for a
member, `nexus__delete__everything` for the owner) reached the model as Codex's
refusal and both answers completed; on the previous pin both ended
`policy_violation`. Paid live matrix and the dedicated containment probe:
NOT_RUN.
