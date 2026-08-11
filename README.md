# provider-runtime

Async Python library (`>=3.12`): one standardized contract calling seven LLM
providers — OpenAI, Anthropic, Gemini, xAI (Grok), DeepSeek, Moonshot (Kimi),
OpenRouter — plus two subscription agent backends (Claude Code and Codex).
Wire handling is rented from three official SDK packages (`openai`,
`anthropic`, `google-genai`) behind four owned protocol engines; the contract,
error taxonomy, model registry, retry policy, agent security kernel, and
observability are owned here.

The package ships two execution contracts:

- `provider_runtime.ProviderRuntime` turns a typed `GenerateIntent` into one
  terminal `CallOutcome` (or one sequenced event stream), dispatched through a
  pinned registry row.
- `provider_runtime.agent_runtime.AgentRuntime` controls one explicitly chosen
  local agent session — Codex SDK or Claude Agent SDK — and exposes normalized
  events plus one terminal result.

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
out = await rt.chat("anthropic:claude-fable-5", system=SYS, user=question, reasoning="high")

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

Multi-turn: append the returned assistant text and tool calls, plus the
outcome's opaque `ContinuationArtifact`, to the next intent's messages. The
artifact carries native reasoning state (encrypted reasoning items, thinking
signatures, `thoughtSignature`, `reasoning_content`, ordered
`reasoning_details`) and is replayed verbatim, never parsed, only to the
identical target — anything else raises `InvalidRequest`. DeepSeek
thinking-mode tool turns replay `reasoning_content`; default-auto tool turns
omit `tool_choice`, using the provider's documented default selection. A
nondefault tool choice is rejected before dispatch because the provider does
not support it in thinking mode.

`json_out` derives a strict JSON schema from a pydantic model: native strict
output on openai/anthropic/gemini/xai, JSON mode plus validation on
deepseek/moonshot and the pinned OpenRouter row. A validation miss returns
`Failed(InvalidStructuredOutput)` with full `CallMeta` — no repair, no retry.

## Architecture

```
types.py       the contract: frozen value vocabulary (intents, outcomes,
               stream events, usage, failures, CallMeta)
errors.py      RuntimeDefect hierarchy + provider-text redaction; defects
               raise, they are never a returned value
registry.py    ModelRow capability table, resolve(), REGISTRY_REVISION
retry.py       single retry owner: DEFAULT_RETRY + the attempt iterator
otel.py        one span per facade call over opentelemetry-api only
prices.py      estimate_cost(meta) over the vendored genai-prices snapshot
runtime.py     ProviderRuntime: dispatch, intent gates, retry loop, stream
               envelope, cancellation, json_out/chat sugar
engines/       the four protocol adapters (Engine protocol; one attempt each)
embeddings.py  OpenAI-only embedding port on the openai SDK
testing.py     FakeEngine + ScriptedRuntime test doubles
agent_runtime/ agent lane: typed session requests, security kernel, auth
               isolation, event normalization, both official SDK adapters
```

| Engine | SDK | Serves |
|---|---|---|
| `openai_responses` | `openai` | OpenAI proper (native Responses API) |
| `openai_chat` | `openai` (compatibility client) | DeepSeek, Moonshot, xAI, OpenRouter |
| `anthropic_messages` | `anthropic` | Anthropic |
| `gemini_generate` | `google-genai` | Gemini |

SDK types never cross the contract boundary, and SDK imports are confined to
`engines/` (plus `embeddings.py`) by a negative gate. Engines make exactly one
attempt and classify errors against the shared taxonomy; the runtime owns
retries, sequence numbering, spans, and attempt-trace accumulation.

## Registry and pinning

Every callable model is a hand-curated `ModelRow` in `registry.py`: exact wire
model id, engine, context window and output cap, modalities, tool/streaming/
structured-output capability, and the exact native reasoning wire value per
declared level. Rows are contract facts, verified against provider docs —
never a place to remember guesses. Any row change bumps `REGISTRY_REVISION`,
which is stamped into every `CallMeta` and flows into the consumer's ledger.

OpenRouter is one pinned, policy-constrained target, never a substrate: every
OpenRouter row carries explicit routing pins (`only`, `order`,
`quantizations`) with fallbacks disabled, `require_parameters` on, data
collection denied, and ZDR required. There is no unpinned passthrough — an
exotic model gets a fully pinned row or it is not callable.

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
`pydantic/genai-prices` — indicative, never authoritative, never stored on
`CallMeta`. `tools/refresh_prices.py` refreshes the snapshot; the library
itself never fetches.

## Agent lane

Exactly two routes ship: `(codex, sdk)` and `(claude, sdk)`, on the pinned
optional extras `openai-codex` and `claude-agent-sdk`. The official SDKs own
their vendor protocols and native sessions; this package owns the
authorization model — a retained security kernel with restrictive permission
defaults, narrowing-only policy changes, unsafe-action confirmation for
model-initiated shell/filesystem/network/MCP actions, and bounded, recursively
redacted native events. Sessions require an already-enrolled subscription
account; API-key session credentials are rejected, and quota exhaustion ends
the turn with an `AgentQuotaExhausted` terminal — the lane never overflows
onto API rates. Child environments are runtime-owned and scrubbed. The full
living contract is [docs/agent-runtime.md](docs/agent-runtime.md).

## Development

```bash
uv sync --all-extras --all-groups
uv run pytest              # deterministic suite; no network (live_provider deselected)
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Both agent SDKs are optional extras (`--extra codex-sdk`, `--extra
claude-sdk`, `--extra agent-sdks`); a base install is provider-only and
selecting an agent route without its extra raises the typed `SdkUnavailable`.

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
MOONSHOT_API_KEY=... OPENROUTER_API_KEY=... DEEPSEEK_API_KEY=... XAI_API_KEY=... \
uv run pytest -m live_provider tests/live/test_provider_matrix.py
```

The `LLM_RUNTIME_LIVE*` variables are read by the opt-in live matrices only,
never by the package. A missing provider key skips that provider's rows with a
recorded reason; the release run is unfiltered with all seven keys set. The
agent lane has its own matrix (`tests/live/test_agent_matrix.py`) with the
same opt-in flag and evidence conventions.

The spec for the current architecture is
[docs/pivot-spec.md](docs/pivot-spec.md); the engineering rules the code is
held to live in [docs/rules/](docs/rules/).
