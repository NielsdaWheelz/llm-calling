# kernel live qualification endpoint prerequisite

llm-agent-kernel remains pinned to provider-runtime commit
`a26f30f7a6f1a25c31b06537614e87cc518cdf49` (`llm-agent-kernel/pyproject.toml:21`).
`tests/live/test_codex_qualification.py:116` supplies a raw socket path to
`AgentRuntimeConfig.codex_endpoints`. its next provider upgrade must construct
`CodexUnixEndpoint(socket_path)` (or an explicit authenticated websocket
endpoint) before live qualification can run. existing pins are unaffected.
resolve when the live configuration uses the typed endpoint and its native
qualification passes against the upgraded pin.
