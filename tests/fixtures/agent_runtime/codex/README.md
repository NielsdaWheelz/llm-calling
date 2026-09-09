# Sanitized Codex assistant-message corpus

`assistant_message_cases.json` retains exact synthetic assistant objects from the
R03 incident alongside adversarial phase, identity, ordering, native-item, and
terminal-status cases. The R03 strings are deliberately preserved byte-for-byte:
each is valid JSON, while their old adjacent concatenation is not.

The corpus models `item/agentMessage/delta`, `item/completed`, and
`turn/completed` shapes exposed by the pinned `openai-codex` 0.144.4 SDK. All
identities and content are synthetic.

`app_server_protocol_cases.json` is the provider-runtime-owned stdio protocol
corpus. Its executable messages are synthetic. The `retained_incident_shape`
member contains only production record indices, item types, field-name sets,
lengths, and SHA-256 values from the retained custom-`exec` incident; it contains
no prompt, conversation text, credential, tool argument, or output content.
