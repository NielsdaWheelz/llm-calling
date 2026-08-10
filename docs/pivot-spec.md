# provider-runtime v2 — Pivot Spec

Status: **proposed** (v2; supersedes v1 after council request-changes review)
Date: 2026-08-09
Mode: **hard cutover** — no legacy paths, no fallbacks, no backward compatibility, no shims.
Provenance: council synthesis + request-changes review to be checked in under
`docs/decisions/2026-08-09-pivot-council.md` in WP-0. Until then this document is a proposal,
not an approval record.

v1 → v2 changes: continuation state restored to the contract (blocking finding 1); OpenRouter
routing/privacy pins preserved (2); §13 is a full migration contract (3); agent security
kernel retained (4); shared terminal metadata + cache-write usage + typed structured replies
(5); work packages re-cut as vertical slices with an atomic provider-lane cutover (6); OpenAI
native lane moved to Responses API; accuracy corrections folded in throughout.

## 1. Goal

One Python library, one standardized contract, calling seven providers — OpenAI, Anthropic,
Gemini, Grok (xAI), DeepSeek, Kimi (Moonshot), OpenRouter — plus two subscription agent
backends (Claude Code via `claude-agent-sdk`, Codex via `openai-codex`). Wire handling is
**rented** from three SDK packages behind **four owned protocol adapters**; the contract,
error taxonomy, model registry, retry policy, security kernel, and observability are owned.

This is a maintainability pivot, not a rescue: the current lane is green (1,083 deterministic
tests, ruff clean, pyright `standard` clean) and its correctness/security kernel must survive
the cutover intact.

Design stance (binding):

1. Own the waist, rent the wire — SDK types never cross the contract boundary.
2. Preserve, don't reconstruct: opaque native state (reasoning signatures, encrypted items) is carried, never parsed.
3. One retry owner; `max_retries=0` on every SDK client.
4. No implicit rerouting anywhere — including inside OpenRouter (routing pinned, fallbacks disabled).
5. Defects raise; expected failures are values with full metadata (`docs/rules/errors.md`).
6. SDKs own protocol; **we own authorization** (agent lane security kernel).
7. Bounded constraints in `pyproject.toml` (library contract); exact pins live in lockfiles (this repo's for CI, the consumer's for prod).

## 2. Non-goals

- No fallback/routing/load-balancing logic; no dynamic control plane; no response cache; no JSON repair.
- No gateway dependency: OpenRouter is one pinned, policy-constrained target, never the substrate.
- No cache-affinity hashing, prefix planning, `BlockStability`/`CacheScope` annotations, or `CACHE_AFFINITY_VERSION` (owner: never changed a decision).
- No authoritative cost: cost is a derived `CostEstimate`, never a certified fact; no price certification or staleness gates. (Billability and usage metadata are **not** cost and remain load-bearing — §4.)
- No `transcribe` port in v2 (no known production consumer as of 2026-08-09; port, catalog row, and tests deleted with the lane).
- No image/audio *generation*; no Gemini explicit `cachedContent`.
- No Nexus code changes in this repo (§13 is the contract for that separate migration).
- No pruning of `docs/rules/` (git-subtree of `engineering-docs`; inapplicable rules simply don't bind).

## 3. Final state — package layout

| File | Purpose | Notes |
|---|---|---|
| `src/provider_runtime/__init__.py` | Facade + `__all__` (≤ 40 names) | rewritten |
| `src/provider_runtime/types.py` | The contract (§4) — **trimmed in place**, not replaced | keep name; Nexus continuity |
| `src/provider_runtime/errors.py` | `RuntimeDefect` hierarchy + secret redaction | ported, trimmed |
| `src/provider_runtime/registry.py` | `ModelRow` capability table + `resolve()` + `REGISTRY_REVISION` | ex-`catalog.py` contract-facts |
| `src/provider_runtime/prices.py` | `estimate_cost(meta) -> Presence[CostEstimate]` over vendored snapshot | indicative, never authoritative |
| `src/provider_runtime/prices_snapshot.json` | Vendored `pydantic/genai-prices` data | refreshed by script, manually |
| `src/provider_runtime/retry.py` | Single retry owner (`stamina`-backed) | `RetryPolicy(` in one module |
| `src/provider_runtime/otel.py` | `gen_ai.*` via **`opentelemetry-api` only**, pinned semconv version | no-op without configured SDK |
| `src/provider_runtime/runtime.py` | `ProviderRuntime`: engines + credentials, dispatch, retry, spans, stream envelope | rewritten |
| `src/provider_runtime/engines/__init__.py` | `Engine` protocol | new |
| `src/provider_runtime/engines/openai_responses.py` | `openai` SDK, native **Responses API** — OpenAI proper | new |
| `src/provider_runtime/engines/openai_chat.py` | `openai` SDK as compat client — DeepSeek, Kimi, xAI, OpenRouter | new |
| `src/provider_runtime/engines/anthropic_messages.py` | `anthropic` SDK, native Messages | new |
| `src/provider_runtime/engines/gemini_generate.py` | `google-genai` SDK, native GenerateContent | new |
| `src/provider_runtime/embeddings.py` | OpenAI-only embed port (live Nexus consumer) | rewritten on `openai` SDK |
| `src/provider_runtime/testing.py` | `FakeEngine` + IR-level `ScriptedRuntime` | rewritten |
| `src/provider_runtime/agent_runtime/…` | Agent lane with retained security kernel (§10) | shrunk in place |
| `tools/refresh_prices.py` | Pull latest genai-prices JSON into snapshot | new |
| `docs/decisions/2026-08-09-pivot-council.md` | Council synthesis + review record | provenance |

Deleted: `openai.py`, `anthropic.py`, `gemini.py`, `moonshot.py`, `openrouter.py`,
`_chat_completions_wire.py`, `_signals.py`, `transport.py`, `schema.py`, `planning.py`,
`usage.py`, `catalog.py`, `docs/agent-runtime-hard-cutover.md`, and from `types.py`:
`BlockStability`, `CacheScope`, `DraftRequest`, `FinalizedProviderRequest`, all
`*Prefix`/cache-contract types.

Terminology: "three SDK packages, four protocol adapters." The OpenAI SDK is official for
OpenAI and a *compatibility client* for DeepSeek/Kimi/xAI/OpenRouter — the spec does not call
those four "official SDK" coverage.

## 4. The contract (`types.py`) — trim, keep, extend

**Kept verbatim (already correct; Nexus-consumed):**

- `ProviderName` (extended: `+ "deepseek", "xai"`), `ProviderTarget`, `Presence`/`Present`/`Absent`.
- `ReasoningLevel = "none"|"minimal"|"low"|"medium"|"high"|"xhigh"|"max"` — full seven levels (Nexus uses `minimal`, `xhigh`).
- Messages: `SystemMessage | UserMessage | AssistantMessage | ToolResultMessage`; `AssistantMessage.continuation: Presence[ContinuationArtifact]`.
- **`ContinuationArtifact(target, codec_id, opaque_payload)`** — opaque, non-logged, replayable only to the identical target + codec (engine-validated). Carries OpenAI Responses encrypted reasoning items, Anthropic thinking signatures, Gemini `thoughtSignature`, Kimi `reasoning_content`, OpenRouter ordered `reasoning_details`. Engines never parse it.
- Tools: `CanonicalTool`, `ToolCall`, `ToolChoice`; `OutputSpec = TextOutput | StrictJsonOutput`.
- `TokenUsage` — including `cache_read_input_tokens` **and `cache_write_input_tokens`** (both `Presence[int]`; cache read ⊆ input, never additive).
- **`CallMeta`** — the shared terminal metadata on *every* outcome, success or failure: `provider, model, provider_request_id, upstream_provider, usage, attempt_trace, billability`. Extended with `native_reasoning: Presence[str]` (ledger-consumed) and `registry_revision: str`.
- `Billability = NotDispatched | PossiblyBillable | ConfirmedNonBillable` — **control-flow input for consumers** (Nexus raises `UncertainChatStep` on `PossiblyBillable`), not telemetry.
- Failure taxonomy: `TransientCause` leaves, `IntentContextTooLarge`, `ProviderContextTooLarge`, `InvalidToolArguments`, `TransientExhausted`, `FailureOrigin`/`FailureCode` — closed, fixed pairs.
- Outcomes: `CallOutcome = Succeeded | Refused | Incomplete | Cancelled | Failed`; `StreamOutcome`; `ResponseContent = TextContent | StructuredContent`.
- Stream taxonomy unchanged: `StreamStart | TextDelta | ToolCallStart | ToolCallDelta | ToolCallDone | ContinuationDelta | UsageEvent | TerminalEvent`, delivered as `RuntimeStreamEvent(seq, event)` — Nexus depends on `seq` and the envelope.
- `GenerateIntent` (name kept), minus stability annotations: `PromptBlock` becomes plain content; new optional `ImageBlock(media_type, data)` in user messages.
- `ProviderCredential`, `CancelSignal`, `RetryPolicy`.

**New:**

- `CostEstimate(amount_usd_micros: int, source: str, as_of: date)` — returned by `prices.estimate_cost(meta)`; `Absent` when unknown. Never on `CallMeta`; always derived on demand.
- `StructuredReply[T](value: T, outcome: Succeeded)` — `json_out` returns this or the failure-bearing `CallOutcome`; the typed value never travels without its metadata.
- `provider_options: Mapping[str, JSON]` on `GenerateIntent` — validated per engine: any key the engine itself maps from core intent fields → raise `InvalidRequest`. Passthrough is for *extensions*, never overrides.

**Ruling kept from v1:** request validation failures raise `InvalidRequest` (defect);
`PlanRejected` dies with `planning.py`.

## 5. Facade API

```python
rt = ProviderRuntime(credentials=Credentials(...))          # values; the lane reads zero env vars

out  = await rt.generate(intent)                            # CallOutcome
async for ev in rt.stream(intent): ...                      # AsyncIterator[RuntimeStreamEvent]
sr   = await rt.json_out(Invoice, intent)                   # StructuredReply[Invoice] | Failed | Refused | ...
vecs = await rt.embed(call, credential=...)                 # kept, Nexus-shaped
cost = estimate_cost(out.meta)                              # Presence[CostEstimate]

# sugar for the 95% call site:
out = await rt.chat("anthropic:claude-opus-5", system=SYS, user=q, reasoning="high")
```

- Multi-turn: callers append the returned `AssistantMessage` (with its `continuation`) to the
  next intent's messages. Engines replay continuation payloads; a continuation bound to a
  different target/codec → `InvalidRequest`.
- Anthropic `cache_control`: one breakpoint inferred at end of system + leading stable turns.
  No caller annotation. Other providers: caching is automatic; nothing to express.
- `json_out`: native strict schema on openai (Responses `text.format`), anthropic
  (`output_format`, GA), gemini, xai; JSON-mode + pydantic validation on deepseek/moonshot;
  validation miss → `Failed(InvalidStructuredOutput)` value with full `CallMeta`. No repair, no retry.

## 6. Engines

`Engine` protocol: `generate(row, intent, credential) -> CallOutcome` and
`stream(row, intent, credential) -> AsyncIterator[CodecStreamEvent]`. One attempt only;
retries, seq-numbering, and span emission live in `runtime.py`. All SDK clients:
`max_retries=0`, injected timeouts, injected `http_client` where supported (testability).

| Adapter | SDK | Serves | Owns |
|---|---|---|---|
| `openai_responses` | `openai` | OpenAI proper | Responses API (OpenAI's recommended lane for reasoning/tools/multi-turn); encrypted reasoning items ↔ `ContinuationArtifact`; `store: false`. |
| `openai_chat` | `openai` (compat client) | deepseek, moonshot, xai, openrouter | DeepSeek: `reasoning_content` preserved into `ContinuationArtifact`, stripped from resends. Kimi K3: model-specific reasoning controls (per registry row), `reasoning_content` continuity. OpenRouter: unified `reasoning` field; ordered `reasoning_details` ↔ continuation; **routing pins from registry row sent on every call** (§7). |
| `anthropic_messages` | `anthropic` | anthropic | `cache_control` injection; thinking signatures ↔ continuation; strict tool schemas (`additionalProperties: false`). |
| `gemini_generate` | `google-genai` | gemini | Reasoning knob is model-generation-specific: `thinkingLevel` (Gemini 3+) vs `thinkingBudget` (2.5) — chosen by registry row, not hardcoded; `thoughtSignature` ↔ continuation. |

Error classification (SDK exception → failure value or defect) lives in each engine against
the shared taxonomy. `CallMeta` is populated on every path, including failures.

## 7. Registry

`ModelRow` (hand-curated, one screen per provider):

```
ref, provider, model_id, engine, base_url,
context_window, max_output_tokens,
modalities: frozenset[Literal["text","image"]],
tools: bool, streaming: bool,
structured: Literal["native","json_mode"],
reasoning: Mapping[ReasoningLevel, JSON] | Absent,        # exact native wire values per level
continuation_codec: str,                                   # codec_id continuations bind to
correlation: Literal["header","in_band","none"],           # gemini: "none"
routing: OpenRouterRouting | Absent,
```

`OpenRouterRouting` (required on every openrouter row — ports the current
`_provider_routing` invariants): `only`, `order`, `allow_fallbacks: False` (fixed),
`require_parameters: True` (fixed), `data_collection: "deny"`, `zdr: True`,
`quantizations`. There is **no unpinned OpenRouter passthrough**: an exotic model gets a row
with explicit routing or it is not callable. This is what "no implicit rerouting" means at
the gateway layer.

`REGISTRY_REVISION` stamped into every `CallMeta` (replaces `catalog_revision`; ledger-consumed).

## 8. Retry (single owner)

Retryable: `TransientCause` leaves (`ProviderRateLimit` honoring retry-after ≤ 60s,
`ProviderTimeout`, `ProviderHttpUnavailable`, `TransportUnavailable`). Not retryable:
everything else. Max 3 attempts, jittered exponential backoff (`stamina`), one wall-clock
deadline. Exhaustion → `Failed(TransientExhausted)` with full attempt trace in `CallMeta`.
Streams: no mid-stream resumption; interruption after any emitted event →
`ProviderStreamInterrupted`, retried only when zero non-terminal events were emitted.
`attempt_trace` stays on `CallMeta` (ledger-consumed) *and* is mirrored to span attributes.

## 9. Observability

`opentelemetry-api` dependency only (never `-sdk` — library rule). One span per facade call;
`gen_ai.*` attributes with pinned semconv version recorded as `otel.scope` attribute;
custom attributes namespaced `provider_runtime.*` (cost estimate, attempt count, billability,
registry revision). Cache-read is a subset of input tokens — never summed. No message
content, no continuation payloads, ever. No-op without a configured tracer provider.

## 10. Agent lane — shrink with a retained security kernel

Keep (ported): `_process.py` process-group ownership, both launchers,
`build_child_environment` scrub, per-profile state roots, auth isolation (subscription auth
only; API-key session credentials rejected).

**Security kernel (retained, ~300 lines; replaces v1's 50-line passthrough):** restrictive
permission defaults; narrowing-only policy changes; unsafe-action confirmation for
model-initiated shell/filesystem/network/MCP actions; bounded, recursively redacted native
event representation. SDKs own protocol and native execution; this kernel owns the
authorization model (`SECURITY.md`, `docs/agent-runtime.md` threat model carried forward).
Deleted: per-version `_KNOWN_FIELDS` capability tables, the version hard-fail gates, and the
capability matrix — validation is behavioral (capability probe), not version-keyed.

- Events → 6 kinds: `AgentText, AgentToolUse, AgentUsage, AgentPermissionRequest, AgentNative(bounded, redacted), AgentTerminal`. Shares `TokenUsage`/`CallMeta` nouns.
- Quota: pool exhaustion → `AgentQuotaExhausted` terminal. The lane never enables API-rate overflow and never forwards API-key credentials. Block-and-stop only.
- Pinning: bounded constraints in `pyproject.toml` extras; exact pins in lockfiles; runtime mismatch → one warning + capability probe.
- Size: deletions measured so far ≈ 860 lines (capabilities + policy algebra) plus taxonomy/test-double/version-gate reductions; target **≤ 8k** (from 9,902), recounted at WP-A merge. The v1 "~3k" claim was unsupported and is withdrawn.

## 11. Testing

- Facade/runtime: against `FakeEngine`; IR-level `ScriptedRuntime` for consumers.
- Per adapter: conformance tests via injected `httpx` client (`respx`) asserting request shape, response/stream decode, continuation round-trip, and **fault injection** (rate limit, timeout, mid-stream cut, malformed usage) → correct failure values + `CallMeta`.
- Negative gates (mechanism ported): SDK imports confined to `engines/` (+ agent SDKs to `agent_runtime/`); `RetryPolicy(` constructed in one module; zero `os.environ` reads in the provider lane; every openrouter registry row has `allow_fallbacks=False`; continuation payloads absent from all `repr`/logs/spans; `__all__` ≤ 40; deleted module names absent.
- **Live matrix**: paid, evidence-recorded run (per-model capability probes: chat, stream, tools, json_out, continuation round-trip; both agent backends). Not in CI; **mandatory before merging any registry or adapter change and before any Nexus pin bump**. Evidence file checked into `tests/live/evidence/`.

## 12. Acceptance criteria

1. All seven providers callable; DeepSeek + Grok work (first time); multi-turn reasoning + tool continuations round-trip on openai/anthropic/gemini/moonshot/openrouter.
2. `uv run pytest` green offline; `ruff` clean; `pyright` **standard** clean (strict is a later, separate decision).
3. `CallMeta` present on every terminal outcome incl. failures; `Billability` semantics unchanged from v1 of the lane (Nexus `chat_runs` logic ports without semantic edits).
4. Negative gates green; zero imports of deleted modules.
5. Live matrix evidence recorded for the full registry.
6. `README.md` describes v2 only.
7. Provider lane ≤ ~4.5k src lines; agent lane ≤ 8k.

## 13. Nexus migration contract (executed later, in nexus-web, at pin bump)

What Nexus consumes and what v2 guarantees:

| Nexus dependency | v2 guarantee |
|---|---|
| `ReasoningLevel` incl. `minimal`/`xhigh` (`llm_profiles.py`) | name + all seven literals unchanged |
| `plan.native_reasoning`, `plan.catalog_revision` (`llm_ledger.py`) | `CallMeta.native_reasoning`, `CallMeta.registry_revision` |
| `TokenUsage` incl. cache read/write (`llm_ledger.py`) | unchanged |
| `Billability` / `PossiblyBillable` / `TransientExhausted` control flow (`chat_runs.py`) | unchanged semantics |
| `RuntimeStreamEvent(seq, …)` envelopes + event kinds (`llm_execution.py`, `chat_runs.py`, artifacts engine) | unchanged |
| `AssistantMessage.continuation` replay (`chat_runs.py`) | unchanged; codec ids change once (registry `continuation_codec`) |
| `GenerateIntent`, `CanonicalTool`, `Presence`, outcome unions | names kept; `plan_generate` call path becomes `rt.generate(intent)` |
| `EmbeddingCall` → `runtime.embed` (`semantic_chunks.py`) | port kept, same shape |
| `ScriptedRuntime` (tests), `nexus_test_control` wire-level SSE scripting | IR-level `ScriptedRuntime`/`FakeEngine`; wire scripting has no replacement (by design) |
| `CATALOG`, `ChatModelContract` imports | replaced by `registry` rows; mostly mechanical, semantic edits called out below |
| `DirectCertification` startup gate (`llm_profiles.py`) | **deleted** — `ModelRow` has no certification field; the gate is re-founded on live-matrix evidence (out-of-band), or removed |
| `contract.reasoning.levels` (`llm_profiles.py`) | `row.reasoning` mapping keys; `Absent` = model has no reasoning knob |
| `contract.output_limit`, `contract.pricing.reasoning_reserve_tokens` (`chat_runs.py`) | `row.max_output_tokens`; v2 has **no reserve-tokens fact** — Nexus owns its reserve policy locally |
| `plan.request_fingerprint` (`llm_ledger.py`) | **deleted, no replacement** — the `llm_calls.request_fingerprint` column is retired at pin bump |
| `cache_strategy`/`cache_ttl` (`llm_ledger.py` → ledger columns) | **deleted with the cache plans** — columns retired at pin bump |
| `Accounting`/`CostBreakdown`/`cost_from_accounting`/`_accounting_snapshot` (`llm_ledger.py`) | replaced by derived `estimate_cost(meta) → Presence[CostEstimate]` (single micros amount, source, as_of; no rate breakdown) — ledger cost columns re-founded on it or retired |
| `PlanningDefect`/`PlanRejected` (`llm_execution.py`) | `InvalidRequest` defect; **observable defect origin changes `"plan"` → `"intent"`** in ledger writes |
| `NonGenerationCallFailed` (`search/embedding.py`) | kept — the embed failure channel is unchanged |
| `PromptBlock(text, stability=…)` construction (`dawn_write.py`) | `PromptBlock(text)` — stability vocabulary deleted |
| Persisted `BlockStabilityState`/`*ScopeState` round-trip (`chat_run_steps.py`) | **persisted-state migration**: stored stability/scope state is dropped or ignored on read; no v2 type round-trips it |
| `to_json_schema`/`parse_canonical_schema` (`chat_runs.py`, `chat_run_steps.py`) | deleted — `CanonicalTool.parameters`/`StrictJsonOutput.schema` take plain JSON-schema mappings |
| `ProviderName` widened `+deepseek, +xai` | `llm_credentials._PLATFORM_KEY_ATTRS` must become an exhaustive match (dict indexing is not totality-checked; today it would `KeyError` at dispatch) |

Exit criterion for the pin bump (verified in nexus-web, not here): Nexus type-checks and its
contract tests pass against v2, with a live-matrix evidence file dated ≥ the v2 revision.
The persisted-state row above means the pin bump includes a data-handling decision, not
only code edits.

## 14. Decision log

| Decision | Ruling | Source |
|---|---|---|
| Wire layer | 3 SDK packages, 4 owned protocol adapters | council + review |
| OpenAI native lane | Responses API, not Chat Completions | review (OpenAI guidance) |
| Continuation state | First-class `ContinuationArtifact`, preserved from v1 lane | review finding 1 |
| OpenRouter | Registry-pinned routing/privacy on every row; no unpinned passthrough | review finding 2 |
| Cost | `CostEstimate(amount, source, as_of)`, derived, indicative | owner + review |
| Billability / `CallMeta` | Load-bearing, on every outcome | review finding 3/5; Nexus `chat_runs` |
| Cache-affinity hashing | Deleted | owner |
| Agent lane | Security kernel retained (~300 lines); capability tables deleted | review finding 4 |
| Claude credit pool | Block at limit; no overflow ever | owner |
| Live matrix | Evidence-mandatory for registry/adapter changes + pin bumps; never CI | owner pref + review, reconciled |
| Migration | In-place; hygiene + agent-lane PRs, then one atomic provider-lane PR | review finding 6 |
| Six-month no-rewrite moratorium after WP-L | Stands | council |
| `docs/rules/` | Kept intact (subtree) | repo fact |
| Pyright | `standard` (as today); strict is a separate future decision | review correction |
| OTel | `opentelemetry-api` only | review correction |
| Versioning | Bounded constraints in `pyproject.toml`; locks pin CI/consumer resolutions | review correction |

## 15. Work packages — vertical slices, each PR leaves `main` green

| WP | One PR containing | Depends on |
|---|---|---|
| **WP-0 Hygiene + provenance** | Delete `docs/agent-runtime-hard-cutover.md`; fix dangling `.dossiers/provider-facts.md` reference; check in `docs/decisions/2026-08-09-pivot-council.md`; commit this spec | — |
| **WP-A Agent kernel** | `agent_runtime/` only: security kernel extraction, event trim, `AgentQuotaExhausted`, lockfile/constraint pinning change, deletions, its tests + gates + doc updates. Self-contained: shares only stable nouns with `types.py` | WP-0 |
| **WP-P Provider lane (atomic)** | One PR, internally ordered commits: (1) deps `openai`/`anthropic`/`google-genai`/`stamina`/`pydantic`/`opentelemetry-api` + `types.py` trim; (2) engines + conformance/fault tests; (3) `runtime.py`/`registry.py`/`retry.py`/`otel.py`/`prices.py` + facade; (4) `embeddings.py`/`testing.py` rewrites; (5) deletions; (6) gates + README. The contract change is cross-cutting — coexistence slices were rejected as dual-path legacy; atomicity is the honest shape (council review, finding 6) | WP-0 |
| **WP-L Live evidence** | Run live matrix across full registry + both agent backends; check in evidence; start the six-month moratorium | WP-A, WP-P |

WP-A and WP-P are parallel (disjoint trees, shared nouns frozen in WP-0 planning).
No WP merges red or half-cut; there is never a commit on `main` where both stacks serve the
same call path.

## 16. Open questions

None blocking. Overridable judgment calls: package/type names preserved wherever semantics
are unchanged (`types.py`, `GenerateIntent`, `ReasoningLevel`, `CallMeta`); image inputs in
scope; pyright stays `standard`. Flag any of these and they flip.
