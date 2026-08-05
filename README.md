# provider-runtime

Small async Python package (`>=3.12`) for pinned provider HTTP calls and
explicit local Codex/Claude agent sessions. The provider lane uses httpx and no
provider SDKs; the agent lane uses optional, exactly pinned official Codex and
Claude SDK extras.

The package owns two distinct execution contracts:

- `provider_runtime.ProviderRuntime` turns a typed generation intent into one
  immutable finalized provider HTTP request and one terminal outcome.
- `provider_runtime.agent_runtime.AgentRuntime` controls one explicitly chosen
  local agent session — Codex SDK or Claude Agent SDK — and exposes
  streamed events plus one terminal outcome.

Callers own prompts, credential resolution, durable history, budgets,
orchestration, and product behavior. The package never coerces an agent session
into a provider `generate()` call and never falls back between targets,
backends, or transports.

The complete living contract is [docs/agent-runtime.md](docs/agent-runtime.md).

Exactly two routes ship: `(codex, sdk)` and `(claude, sdk)`. There is no raw CLI
lane, direct App Server/JSON-RPC adapter, or fallback between routes. The
official SDKs own their vendor protocols, arguments, and native sessions; this
package owns auth isolation, process-group cleanup, capability validation,
normalized events, cancellation, policy, and terminal outcomes. Both routes
require an already-enrolled `local_account`.

## Architecture

```
types.py      frozen value vocabulary (intents, outcomes, stream events, plans)
schema.py     canonical JSON-Schema subset: parse/validate/serialize, no rewriting
catalog.py    CATALOG — exact provider contracts (limits, reasoning levels,
              cache mechanism, pricing in usd micros, privacy, certification)
errors.py     RuntimeDefect hierarchy (PlanningDefect, ProtocolDefect,
              CredentialRejected, SchemaViolation) + provider-text redaction;
              defects raise, they are never a returned value
planning.py   plan_generate: intent -> FinalizedProviderCall | PlanRejected;
              cache affinity (CACHE_AFFINITY_VERSION), retry-policy constants
openai.py / anthropic.py / gemini.py / moonshot.py / openrouter.py
              codecs: encode/finalize, decode_response, decode_stream,
              classify_error, stream_request (private; not exported)
transport.py  auth-header injection + HTTP + timeouts + raw SSE framing; parses
              nothing, classifies nothing
runtime.py    ProviderRuntime: generate/stream (sole same-target retry owner),
              embed/transcribe (non-generation ports)
embeddings.py OpenAI-only embedding port: request building + strict response
              validation, dispatched by ProviderRuntime.embed through the
              shared Transport and EXTERNAL_LLM_RETRY policy
usage.py      cost_from_accounting over the plan's frozen Accounting — terminal
              costing never re-reads the catalog
testing.py    NoNetworkRuntime / ScriptedRuntime test doubles
agent_runtime/
              typed session requests, capabilities, policy, auth isolation,
              event normalization, bounded lifecycle substrate, the official
              Codex and Claude SDK adapters, and agent test doubles
```

Data flow: `GenerateIntent -> plan_generate(CATALOG) -> FinalizedProviderCall
-> ProviderRuntime.generate/stream -> CallOutcome / RuntimeStreamEvent`.

Agent flow: `AgentSessionRequest -> AgentRuntime.open_session ->
AgentRuntime.stream_turn/run_turn -> AgentEvent / AgentResult`.

## The pinned-contract philosophy

Every callable model is a checked-in `ChatModelContract` row in `CATALOG`:
exact provider/model target, protocol, context/output limits, the declared
reasoning levels with their exact native wire values, the cache mechanism and
minimum prefix, integer usd-micro pricing with source URLs and a verification
date, privacy posture, and a certification arm. The catalog is a transcription
of verified provider facts — never a place to remember them. Any row change
bumps `CATALOG_REVISION`, which is stamped into every plan and flows into the
consumer's ledger.

There is no dynamic control plane, fallback, sampling knob, response cache, or
JSON repair. Changes are reviewed, live-certified, and deployed like code.
Nexus pins this repository at an exact revision in `python/pyproject.toml` and
consumes only the sanctioned `provider_runtime.__all__` and
`provider_runtime.agent_runtime.__all__` surfaces. A new catalog or runtime
contract reaches production only through an explicit pin bump and the owning
certification gates.

The OpenRouter operator route is special: its catalog row stays
`OperatorUncertified` (representable but unplannable) until the live
certification test observes the pinned upstream (`moonshotai/int4`), routed
Kimi `low|high|max` acceptance, and a non-zero billed cache read. The test
writes the evidence artifact to `tests/live/evidence/` and prints the
`evidence_revision` to pin in the row as `OperatorCertified(...)`.

## Environment contract

These are every environment variable the package's own source reads, and there
are no others. The provider lane reads none at all: a provider credential is a
value on the typed request, never something the package looks up for you.

- `CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK` — optional, no default, and it must be
  unset (an empty value counts as unset). The Claude SDK adapter reads it before
  importing `claude_agent_sdk` and fails closed with `UnsupportedCapability`
  whenever it holds a non-empty value. The pinned-SDK guarantee is that this
  adapter only ever drives the exact vetted `claude-agent-sdk` and Claude Code
  versions; the variable is the SDK's own ambient bypass of that check, so
  honoring it would let an exported shell variable silently void the guarantee.
- Caller-named MCP credential variables — optional, with no defaults. An
  `McpServerSpec` environment or header reference may name a variable that the
  server needs; the resolved value never enters a public request, event, result,
  reference, or exception. Session credentials are different: both shipped
  routes reject `api_key_environment` and `secret_reference` before resolution.
- `PermissionPolicy.environment` names — optional, no defaults, chosen by the
  caller. Each listed name is copied from the parent environment into the child
  agent's environment and used nowhere else. Credential-class,
  provider-selection, and process-control names are rejected with
  `InvalidAgentRequest`; see [docs/agent-runtime.md](docs/agent-runtime.md).

`PATH`, `HOME`, `LANG`, `LC_ALL`, `TMPDIR`, `CODEX_HOME`, and `CLAUDE_CONFIG_DIR`
in a child agent are runtime-owned values, never inherited ones, and a caller can
neither set nor unset them.

**The child `PATH` is fixed, and Claude's launcher must be visible on it.** Every
agent this package launches gets exactly
`/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`
(`agent_runtime/auth.py`, `_CHILD_PATH`) — never the operator's, which would hand
a sandboxed child every shim, version manager, and project-local
`node_modules/.bin`. Claude installed from npm is launched through a
`#!/usr/bin/env node` entry point, so `node` must resolve from that fixed list; a
version-manager-only `node` is intentionally invisible. Codex is different: the
pinned `openai-codex` package owns its matched bundled runtime, and this package
does not resolve an ambient `codex` executable. Its trusted bundled PATH entry is
prepended to the fixed path before launch.

The `LLM_RUNTIME_LIVE*` variables below are read by the opt-in live matrices, not
by the package.

## Certification (paid live matrix)

```bash
LLM_RUNTIME_LIVE=1 uv run pytest -m live_provider tests/live/test_provider_matrix.py
```

Required environment:

- `LLM_RUNTIME_LIVE=1` — the matrix fails closed without it;
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MOONSHOT_API_KEY`
  for the direct routes;
- `OPENROUTER_API_KEY` for the operator route (the certification and
  invalid-key probes);
- optional `LLM_RUNTIME_LIVE_PROVIDERS=openai,gemini,...` narrows a local run;
  narrowed runs are debugging aids — release certification runs unfiltered.

Local agent certification is a separate opt-in matrix:

```bash
LLM_RUNTIME_LIVE=1 \
LLM_RUNTIME_LIVE_AGENT_STATE_ROOT_BASE=/absolute/existing/private/root \
LLM_RUNTIME_LIVE_AGENT_PROFILE=live-local \
LLM_RUNTIME_LIVE_CLAUDE_SDK_MODELS=<model-a>,<model-b> \
uv run pytest -m live_provider tests/live/test_agent_matrix.py
```

The state-root base must be an existing, resolved, absolute directory that is
neither group- nor world-writable, so `/tmp` (mode `1777`) cannot be used, and
each selected route needs an already-enrolled profile directory beneath it. The
unnarrowed run is the release certification and covers both shipped lanes;
`LLM_RUNTIME_LIVE_AGENT_ROUTES` narrows it for debugging and certifies nothing.
It certifies that every lane *refuses* a named API-key or secret-reference
credential, not that any accepts one — none does.

Codex model discovery is authoritative: certification calls every discovered
model at each reasoning effort that model itself reports. Claude cannot
enumerate models, so `LLM_RUNTIME_LIVE_CLAUDE_SDK_MODELS` is a required,
strict, comma-separated list with no whitespace, empty entries, or duplicates;
certification calls every listed model at every discovered Claude effort. The
broader session, policy, MCP, approval, cancellation, and discovery probes run
once per route.

Claude's preflight also insists on the real executable. `AgentRuntime` resolves
it with `shutil.which` and then `Path.resolve()`, so a `~/bin/claude` symlinked
to a per-directory router script is what actually gets spawned — and such a
router usually re-exports `CLAUDE_CONFIG_DIR`, which would void the
state-root isolation the run exists to certify. A resolved target whose shebang
names a POSIX shell is refused, and the refusal names the symlink, the resolved
script, the unwrapped candidate it found further along the same `PATH`, and the
variable to set: point `LLM_RUNTIME_LIVE_CLAUDE_EXECUTABLE` at that real binary.
Doing this
first is worth it — a router that dispatches on its own `argv[0]` sees the
resolved name, so left to itself it fails with an unrelated message about its own
arguments. `tests/live/test_agent_matrix.py` documents every variable.

It discovers the installed/authenticated transports, records their versions
and capabilities, and certifies only features they report and prove through
native or behavioral evidence. Incomplete proof remains explicitly
observational and fails certification. One probe is free: Codex capability
discovery initializes the public SDK and proves its exact package/bundled-runtime
pair without starting a turn. The matrix never runs in the default
deterministic suite and never reads or prints a raw subscription token.

Per chat target the matrix proves: one minimal call per declared reasoning
level, an above-minimum-prefix cache warm/read probe with an observed cache
read (bounded successful-call sampling for Gemini's non-guaranteed implicit
cache), strict JSON (including a required-nullable field), a streamed tool call
plus same-target continuation replay, invalid-key classification,
request-id/usage presence per contract facts, and the planner's input-token
upper bound dominating billed input on every call — plus minimal OpenAI
embedding and transcription calls. The default `uv run pytest` suite is fully
deterministic and makes no network calls (`live_provider` is deselected by
`addopts`).

## Cache-affinity versioning rule

`planning.py` solely owns `CACHE_AFFINITY_VERSION` and the length-framed
affinity formula (scope, target, protocol, canonical cache-contract bytes, and
the codec's exact native prefix bytes — computed pre-finalize so the injected
key never feeds itself). Checked-in golden vectors
(`tests/goldens/cache_affinity.json`) pin the values across processes and
workers. Any framing, scope-encoding, prefix-encoding, or cache-contract
semantic change MUST increment `CACHE_AFFINITY_VERSION` and regenerate the
golden vectors; an old affinity value is never recomputed under new rules.
OpenAI and Moonshot receive the affinity as `prompt_cache_key`, OpenRouter as
`session_id`; Anthropic and Gemini use their native prefix mechanisms and
retain the affinity for fingerprint/telemetry only.

## Development

```bash
uv sync --all-extras --all-groups
uv run pytest            # deterministic suite (unit/golden + HTTP-boundary)
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Both agent SDKs remain optional. Use `uv sync --extra codex-sdk`,
`uv sync --extra claude-sdk`, or `uv sync --extra agent-sdks`. A base install is
provider-only: selecting either agent route without its extra raises the typed
`SdkUnavailable` error. There is no CLI or raw-protocol fallback behind it.

Application tests should use `ScriptedRuntime` / `NoNetworkRuntime` from
`provider_runtime.testing`: the runtime interface without provider network
connections, with scripts that must end in exactly one terminal. Agent
consumers use the corresponding doubles from
`provider_runtime.agent_runtime.testing`; no unit test starts a provider,
real agent executable, real SDK client, MCP server, or credential flow.
