# Sanitized Codex assistant-message corpus

`assistant_message_cases.json` retains exact synthetic assistant objects from the
R03 incident alongside adversarial phase, identity, ordering, native-item, and
terminal-status cases. The R03 strings are deliberately preserved byte-for-byte:
each is valid JSON, while their old adjacent concatenation is not.

The corpus models `item/agentMessage/delta`, `item/completed`, and
`turn/completed` shapes exposed by the pinned Codex 0.153.4 App Server. All
identities and content are synthetic.

The active WebSocket-over-UDS protocol fixtures live beside their focused tests;
the retired private-process stdio corpus is not retained.
