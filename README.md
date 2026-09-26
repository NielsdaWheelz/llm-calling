# provider-runtime

Async Python library (`>=3.12`): one standardized contract calling five LLM
providers — OpenAI, Anthropic, Gemini, xAI, and DeepSeek — plus two
subscription agent backends (Claude Code and Codex).
Wire handling is rented from three official SDK packages (`openai`,
`anthropic`, `google-genai`) behind four owned protocol engines; the contract,
error taxonomy, model registry, retry policy, agent security kernel, and
observability are owned here.

The package ships two execution contracts:

- `provider_runtime.ProviderRuntime` turns a typed `GenerateIntent` into one
  terminal `CallOutcome` (or one sequenced event stream), dispatched through a
  pinned registry row.
- `provider_runtime.agent_runtime.AgentRuntime` controls one explicitly chosen
  local agent session — shared Codex App Server client or Claude Agent SDK —
  and exposes normalized events plus one terminal result.

Callers own prompts, credential resolution, durable history, budgets, and
orchestration. There is no fallback between providers, models, backends, or
transports, no dynamic control plane, no response cache, no JSON repair, and
no sampling knobs. Defects raise; expected failures are values carrying full
metadata.

## Facade

```python
from provider_runtime import Credentials, ProviderRuntime, estimate_cost

rt = ProviderRuntime(credentials=Credentials(openai="...", anthropic="..."))

# the 95% call site:
out = await rt.chat("anthropic:claude-fable-5-1", system=SYS, user=question, reasoning="adaptive/high")

out = await rt.generate(intent)                 # CallOutcome
async for event in rt.stream(intent):           # RuntimeStreamEvent(seq, event)
    ...
reply = await rt.json_out(Invoice, intent)      # StructuredReply[Invoice] | Refused | ... | Failed
vectors = await rt.embed(call, credential=cred) # EmbeddingResponse (OpenAI-only port)
cost = estimate_cost(out.meta)                  # Presence[CostEstimate]
```

Credentials are values on the runtime — **the provider lane reads zero
environment variables**. Every terminal outcome, success or failure, carries a
`CallMeta`: provider, model, request id, normalized `TokenUsage` (cache read
and write included), the full attempt trace, billability, the exact native
reasoning value sent, and the registry revision.

Multi-turn: pass the returned `ContinuationArtifact` and ordered tool results
to `ContinueGeneration`; the original request and prior native turns are bound
in the artifact. Use
`provider_runtime.continuation.encode_continuation` / `decode_continuation`
when crossing a persistence boundary; the bounded canonical codec binds bytes
to the exact target and provider codec. `pending_tool_calls` reads the exact
ordered calls after restart. The artifact carries native reasoning state
(encrypted reasoning items, thinking signatures, `thoughtSignature`,
`reasoning_content`) and is replayed verbatim only to the
identical target — anything else raises `InvalidRequest`. DeepSeek
thinking-mode tool turns replay `reasoning_content`; default-auto tool turns
omit `tool_choice`, using the provider's documented default selection. A
nondefault tool choice is rejected before dispatch because the provider does
not support it in thinking mode.

`json_out` derives a strict JSON schema from a pydantic model: native strict
output on openai/anthropic/gemini/xai, JSON mode plus validation on
deepseek. A validation miss returns
`Failed(InvalidStructuredOutput)` with full `CallMeta` — no repair, no retry.

### Portable tools

Install `provider-runtime[llm-tools]` for the optional portable-tool adapter.
For each model request, pass its frozen plan and persisted revealed-target set
in `ToolPublication`, call `provider_runtime.tool_adapter.lower_tools`, then
use the returned request-scoped object's `decode_tool_call` for every provider
tool call. Canonical dotted ids remain application identity;
double-underscore aliases exist only on that provider request. Native and
Discoverable plans publish tools; HostTable plans never do. Where an engine
observes raw arguments, its ingress owns non-UTF-8 and transport-size rejection
before constructing a `ToolCall`; an SDK that exposes only parsed JSON cannot
attest raw transport bytes. The adapter separately enforces valid bounded JSON
and the selected grant's canonical `ParsedJson` input-byte ceiling.

The adapter does not choose tools from a product operation name.

## Architecture

```
types.py       the contract: frozen value vocabulary (intents, outcomes,
               stream events, usage, failures, CallMeta)
errors.py      RuntimeDefect hierarchy + provider-text redaction; defects
               raise, they are never a returned value
registry.py    private capability rows/resolution; public api_model_catalog()
continuation.py bounded canonical provider-continuation codec
retry.py       single retry owner: DEFAULT_RETRY + the attempt iterator
otel.py        one span per facade call over opentelemetry-api only
prices.py      estimate_cost(meta) over a dated provider-rate snapshot
runtime.py     ProviderRuntime: dispatch, intent gates, retry loop, stream
               envelope, cancellation, json_out/chat sugar
engines/       the four protocol adapters (Engine protocol; one attempt each)
embeddings.py  OpenAI-only embedding port on the openai SDK
testing.py     FakeEngine + ScriptedRuntime test doubles
tool_adapter.py request-scoped llm-tools lowering and canonical name decode
agent_runtime/ agent lane: authenticated model catalog, tagged session requests,
               MCP projection, security kernel, shared Codex App Server control,
               and the Claude SDK adapter
```

| Engine | SDK | Serves |
|---|---|---|
| `openai_responses` | `openai` | OpenAI proper (native Responses API) |
| `openai_chat` | `openai` (compatibility client) | DeepSeek, xAI |
| `anthropic_messages` | `anthropic` | Anthropic |
| `gemini_generate` | `google-genai` | Gemini |

SDK types never cross the contract boundary, and SDK imports are confined to
`engines/` (plus `embeddings.py`) by a negative gate. Engines make exactly one
attempt and classify errors against the shared taxonomy; the runtime owns
retries, sequence numbering, spans, and attempt-trace accumulation.

## Provider catalog and pinning

`provider_runtime.registry.api_model_catalog()` is the sole public catalog
oracle. It returns an immutable, ordered `ApiModelCatalog` containing exact
dispatch, capacity, modality, tool, streaming, structured-output, reasoning,
default, lifecycle, continuation-codec, revision, and row-fingerprint facts.
The hand-curated rows and their resolution functions are private runtime
owners; consumers select and compare only public catalog facts. Rows are
verified against provider docs, never a place to remember guesses. Any row
change bumps the catalog's `registry_revision`, which is also stamped into
every `CallMeta` and flows into the consumer's ledger.

The catalog is closed to the nine current API model identities. Reasoning keys
and labels are public; native wire fragments stay private to the library.
Unknown refs and undeclared reasoning keys fail before dispatch.

## Retry, observability, cost

**Retry** has one owner (`retry.py`): at most 3 attempts, jittered exponential
backoff, provider `retry-after` honored up to 60s, one wall-clock deadline per
call. Every SDK client runs with `max_retries=0`. Only exact transient causes
retry (rate limit, timeout, unavailability, transport failure); streams retry
only before any semantic event reached the consumer, and exhaustion folds into
`Failed(TransientExhausted)` with the full attempt trace on `CallMeta`.

**Observability** depends on `opentelemetry-api` only and is a true no-op
without a configured tracer: one span per facade call, `gen_ai.*` attributes
from a pinned semconv version, custom attributes under `provider_runtime.*`
(attempt count, billability, registry revision). Never on a span: message
content, continuation payloads, credentials.

**Cost** is a derived `CostEstimate` (usd micros, source, as-of date) computed
on demand by `estimate_cost(meta)` over a vendored snapshot of
official provider rates — indicative, never authoritative, never stored on
`CallMeta`. Unknown price tiers return `Absent`; the library never fetches.

## Agent lane

Exactly two routes ship: `(codex, app_server)` and `(claude, sdk)`. The core package
declares `websockets`; the host separately installs the latest stable Codex and
supervises its shared service. Native version metadata is diagnostic, not an
admission gate; protocol validation, subscription auth, and containment remain
fail-closed. Installing a newer version does not certify its live behavior.
The Codex route owns the documented WebSocket-over-Unix-socket App Server
client; Claude remains on its official
SDK. This package
owns the authorization model — a retained security kernel with restrictive permission
defaults, narrowing-only policy changes, unsafe-action confirmation for
model-initiated shell/filesystem/network/MCP actions, and bounded, recursively
redacted native events. Sessions require an already-enrolled subscription
account; API-key session credentials are rejected, and quota exhaustion ends
the turn with an `AgentQuotaExhausted` terminal — the lane never overflows
onto API rates. `AgentTerminal.usage` is always local to that invocation and
never replays cumulative native-session history. `AgentTerminal.final_text` is
the provider-selected assistant response, not concatenated assistant traffic;
Codex follows the App Server's last-final-answer/last-unknown rule, while
commentary remains observable but is never executable structured output. Child
environments are runtime-owned and scrubbed. Under
`CodexNativeOptions(builtin_tools="disabled")`, every known native authority
event is first-class and poisons the turn; unknown protocol messages fail closed.
This contains/detects Code Mode but does not prove it absent before execution.
The full
living contract is [docs/agent-runtime.md](docs/agent-runtime.md).

Codex selection is catalog-bound: query
`AgentRuntime.model_catalog("codex", transport="app_server", auth=auth)`, then submit a
`CodexCatalogSessionRequest` with the exact model key, reasoning key,
definition revision, and row fingerprint. The runtime re-reads and validates
those facts before opening a session and never accepts a free-form Codex model.
Claude uses the separate `ClaudeNativeSessionRequest` arm and reports model
catalog discovery as `UnsupportedCapability`.

## Existing terminal agents

`provider-runtime-control` provides one bounded JSON request/reply for
`inspect`, `read`, `send`, `interrupt`, and `stop` on existing Codex/Claude
sessions. it is a command, not a daemon; it never owns the shared server or
copies history. callers select the profile environment before starting it.
Codex uses the existing observe-only control client; Claude uses its native
agent listing/stop commands and the optional SDK saved-message reader.

this boundary is separate from isolated cognition: terminal agents retain the
host user's existing authority. the host owns tmux discovery, process identity,
terminal fallback, and final terminal closure. read coverage and uncertain
writes are explicit; absent expected Codex turn identity never selects a newer
turn for cancellation. installation must use the committed dependency lock
with the `claude-sdk` extra. Claude's public reader loads the saved file before
slicing; bounded output does not imply bounded transcript memory.

## Development

```bash
uv sync --all-extras --all-groups
uv run pytest              # deterministic suite; no network (live_provider deselected)
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Codex App Server transport is a core dependency. Claude is available through
the optional `--extra claude-sdk`; selecting it without that extra raises the
typed `SdkUnavailable`.

Application tests use `FakeEngine` / `ScriptedRuntime` from
`provider_runtime.testing` (and the doubles in
`provider_runtime.agent_runtime.testing`): the runtime interface with scripted
outcomes, no network, no SDK clients, no credential flows.

### Live matrix (paid, evidence-recorded, never CI)

The live matrix is the acceptance gate the deterministic suite cannot be: per
registry row it probes chat, streaming, a tool round trip, `json_out`, and a
continuation replay against the real providers, and writes one evidence file
per run into `tests/live/evidence/`. It never runs in CI and is mandatory
before merging any registry or engine change and before any Nexus pin bump.

```bash
LLM_RUNTIME_LIVE=1 OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... \
DEEPSEEK_API_KEY=... XAI_API_KEY=... \
uv run pytest -m live_provider tests/live/test_provider_matrix.py
```

The `LLM_RUNTIME_LIVE*` variables are read by the opt-in live matrices only,
never by the package. A missing provider key skips that provider's rows with a
recorded reason; the release run is unfiltered with all five keys set. The
agent lane has its own matrix (`tests/live/test_agent_matrix.py`) with the
same opt-in flag and evidence conventions. Its dedicated paid Codex containment
probe is `tests/live/test_codex_containment.py` and requires an explicit
test-owned Codex endpoint. Shared worker/TUI control has its own separately
approved live qualification; deterministic fixtures do not substitute for it.

The engineering rules live in [docs/rules/](docs/rules/).
