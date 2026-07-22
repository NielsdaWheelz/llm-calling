# provider-runtime

Small async Python package (`>=3.12`, httpx only, no provider SDKs) for
provider-level LLM calls against a pinned, live-certified contract catalog.

The package owns one thing: turning a typed generation intent into exactly one
immutable finalized provider request, executing it, and returning a closed
terminal outcome. Callers (Nexus) own prompts, credentials, HTTP client
lifecycle, persistence, budgets, and product behavior.

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
```

Data flow: `GenerateIntent -> plan_generate(CATALOG) -> FinalizedProviderCall
-> ProviderRuntime.generate/stream -> CallOutcome / RuntimeStreamEvent`.

## The pinned-contract philosophy

Every callable model is a checked-in `ChatModelContract` row in `CATALOG`:
exact provider/model target, protocol, context/output limits, the declared
reasoning levels with their exact native wire values, the cache mechanism and
minimum prefix, integer usd-micro pricing with source URLs and a verification
date, privacy posture, and a certification arm. The catalog is a transcription
of verified provider facts — never a place to remember them. Any row change
bumps `CATALOG_REVISION`, which is stamped into every plan and flows into the
consumer's ledger.

There is no dynamic control plane, no BYOK, no fallback, no sampling knobs, no
response cache, and no JSON repair. Changes are reviewed, live-certified, and
deployed like code. Nexus pins this repository at an exact revision (the Nexus
root `Makefile` owns the pin and the `make certify-llm-providers` gate) and
consumes only the package surface in `provider_runtime.__all__`; a new
catalog/contract revision reaches production only through a pin bump plus a
fresh paid certification run.

The OpenRouter operator route is special: its catalog row stays
`OperatorUncertified` (representable but unplannable) until the live
certification test observes the pinned upstream (`moonshotai/int4`), routed
Kimi `low|high|max` acceptance, and a non-zero billed cache read. The test
writes the evidence artifact to `tests/live/evidence/` and prints the
`evidence_revision` to pin in the row as `OperatorCertified(...)`.

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

Per chat target the matrix proves: one minimal call per declared reasoning
level, an above-minimum-prefix cache warm/read pair with an observed cache
read, strict JSON (including a required-nullable field), a streamed tool call
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
uv sync --all-groups
uv run pytest            # deterministic suite (unit/golden + HTTP-boundary)
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Application tests should use `ScriptedRuntime` / `NoNetworkRuntime` from
`provider_runtime.testing`: the runtime interface without provider network
connections, with scripts that must end in exactly one terminal.
