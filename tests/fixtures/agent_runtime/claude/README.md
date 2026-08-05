# Sanitized Claude Agent SDK corpus

These fixtures encode message shapes that claude-agent-sdk 0.2.130 can actually
deliver, as JSON, so tests can rebuild external-boundary doubles without
importing the optional SDK. The `system`/`init` frames are modelled on the real
Claude Code 2.1.220 frame (built by `tAr()` in the shipped binary as
`{cwd, session_id, tools, mcp_servers, model, permissionMode, slash_commands,
apiKeySource, claude_code_version, output_style, agents, skills, plugins,
capabilities, analytics_disabled, product_feedback_disabled, uuid,
fast_mode_state, fast_mode_disabled_reason}`), with `permissionMode: "default"`
because that is the mode this transport requests.

`fake_claude_code.py` is the local Claude Code executable the SDK double actually spawns
through the runtime-owned launcher. It exists because the lane's process-group ownership is
a property of a real process tree: it answers the version probe, optionally starts a
SIGTERM-ignoring descendant (the stand-in for a Bash-tool command or a stdio MCP server) and
reports readiness only once that descendant's handler is installed, so the group escalation
is tested against a child that a single polite signal cannot end.

`second_turn.json` carries the same `session_id` as `success.json` and is the second turn of
that session, which is what makes an interrupted turn's leaked tail visible: without the
drain, a turn replaying it terminates on the previous turn's `ResultMessage`. A retired
turn's tail also reaches the *control* channel, which no message corpus can express: the SDK
double dispatches a case from `approval_cases.json` there instead, because a permission
request Claude Code wrote before an interrupt landed is delivered on a task of the SDK's own
regardless of who is reading the message stream.

The JSON scenarios cover init/session identity and effective configuration,
partial text and visible thinking, tool and file-change lifecycle, per-message and terminal
usage including every member `parse_message` fills on a `ResultMessage`
(`total_cost_usd`, `model_usage`, `permission_denials`, `deferred_tool_use`
among them), an unmodelled `system` subtype carrying credential-shaped strings, a
typed `SystemMessage` subclass reporting failed background work, quota
rejection, interruption, and iterator exhaustion without a `ResultMessage`.
`approval_cases.json` drives the permission-callback matrix in
`tests/test_agent_claude_sdk.py`. IDs, paths, model names, usage, and content
are synthetic.

A wholly unrecognized top-level message type is deliberately **not** in this
corpus: `_internal/message_parser.py` returns `None` for one and the client
drops it, so no such message can reach the adapter and a fixture claiming
otherwise would manufacture coverage the real dependency cannot deliver.
