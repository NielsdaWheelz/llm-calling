# Decision record: provider-runtime v2 pivot (2026-08-09)

Record of the multi-agent council review that produced `docs/pivot-spec.md`, and the
subsequent request-changes review that produced its v2 revision.

- Process: 14-agent workflow (run `wf_c4cb721b-c60`) — 3 repo archaeologists, 5 web
  researchers, 5-member adversarial SME council, 1 chief-architect synthesis. ~922k tokens.
- Outcome: pivot the wire layer to three SDK packages behind an owned contract; in-place
  hard cutover. The synthesis and full council positions are reproduced verbatim below.
- A subsequent external request-changes review (2026-08-09) found six blocking issues in
  spec v1; all were verified against the repos and accepted (with sizing nuance on the agent
  kernel and sequencing remedied as an atomic provider-lane PR). Disposition:

| Review finding | v2 disposition |
|---|---|
| 1. IR omitted opaque continuation state | `ContinuationArtifact` retained first-class (spec §4) |
| 2. OpenRouter passthrough allowed implicit rerouting | Registry-pinned `OpenRouterRouting` on every row (§7) |
| 3. Nexus impact understated (levels, billability, ledger, seq) | Full migration contract (§13); `CallMeta`/`Billability` load-bearing (§4) |
| 4. Agent-lane passthrough deleted the authorization model | Security kernel retained ~300 lines (§10) |
| 5. Usage/outcome/accounting regressions | `CallMeta` on every outcome; cache-write kept; `StructuredReply[T]`; `CostEstimate` (§4) |
| 6. Work packages violated green-main | Vertical slices; provider lane atomic (§15) |

Accuracy corrections also accepted: OpenAI native lane = Responses API; Gemini
`thinkingLevel` vs `thinkingBudget` per model generation; Kimi K3 reasoning controls;
"three official SDKs" reworded; pyright is `standard`; `opentelemetry-api` only; bounded
constraints in `pyproject.toml`; genai-prices is indicative → `CostEstimate`; transcribe
"no known production consumer" (deletion stands).

---

## Chief-architect synthesis (verbatim)

# Provider-Runtime Pivot: Final Synthesis

## 1. Verdict

The philosophy — one contract, pinned facts, defects raise, no fallback — is sound; the substrate is wrong: ~9.5k of 14.2k provider-lane lines hand-build wire syntax (JSON bodies, SSE framing, status→retry maps) that `openai`, `anthropic`, and `google-genai` already ship and maintain against provider drift (survey:provider-lane §2, research:provider-wire). The proof is coverage, not aesthetics: after ~40k src+test lines, DeepSeek and Grok — two of seven required providers — do not exist in the catalog at all (sme:pragmatist), and the strictest artifact in the repo cites a provenance file (`.dossiers/provider-facts.md`) that does not exist (survey:docs-rules-tests). The differentiated core is real but smaller than the repo believes: the intent→plan compiler, the defect/value split, the contract-facts half of the catalog, and the negative-gate mechanism — roughly 2.5–4k lines — and everything else is deletable.

## 2. Decision Matrix

Scores 1–5, higher is better ("complexity carried" and "blast radius": higher = less complexity / smaller blast).

| Option | Contract quality | Frontier latency | Complexity carried | Blast radius | Migration cost | Total |
|---|---|---|---|---|---|---|
| **A. Keep-and-trim handrolled** | 4 — contract is good but is a *transport* waist (bytes/SSE), not semantic (sme:architect) | 1 — every provider feature is a codec ticket + recert; DeepSeek/Grok still unbuilt | 1 — 14.2k lines, dual retry loops, hand SSE framer | 5 — httpx + 5 transitives, best-in-class (sme:sre) | 5 — zero | 16 |
| **B. any-llm as wire layer** | 2 — you adopt *their* completion types as your vocabulary; their normalization on exactly the 20% you care about (caching, reasoning) | 3 — lags native SDKs by their release cadence | 4 — 50–230 lines/provider | 2 — third party between you and every credential; union of SDK transitive closures anyway (sme:sre) | 3 | 14 |
| **C. LiteLLM in-process** | 2 — `Callable[..., Awaitable[Any]]` typing, loose | 3 | 1 — 9–12k-line monolith files, 4,806 open issues; maintainers moved core to Rust (research:unified-libs) | 1 | 3 | 10 |
| **D. OpenRouter gateway for everything** | 3 — one wire, but cache continuity breaks on routing hops; provider quirks leak through anyway (research:gateways) | 5 — day-zero catalog | 5 — thinnest client | 1 — no SLA, 1.7/5 Trustpilot, misleading 401s during outages, account-403 precedent; single custodian for all traffic | 4 | 18* |
| **E. Official SDKs behind thin owned facade** (+ OpenRouter as an explicit *target* for exotics) | 5 — you own the IR; SDK types confined to `engines/` by negative gate | 4 — `pip install -U` distance from frontier for OpenAI/Anthropic/Gemini; OpenRouter target covers day-zero exotics | 4 — ~4k provider-lane lines for **seven** providers | 4 — three first-party vendors with CVE processes; keys go only to their owners | 3 — mechanical; seam already narrow (survey:provider-lane §4) | 20 |

\* D's total is inflated by axes that don't capture its disqualifier: it makes an unaccountable reseller the sole custodian of workloads you depend on daily, and voids prompt-cache economics on any routing hop — a correctness failure, not a reliability preference (sme:pragmatist).

**Winner: E, decisively.** Three official SDKs — `openai` (serving OpenAI, DeepSeek, Kimi/Moonshot, Grok, and OpenRouter via `base_url`, since all are OpenAI-Chat-Completions-shaped), `anthropic` native, `google-genai` native (never the compat shim: it drops context caching and native multimodal, per research:provider-wire) — under a thin facade whose contract you own. Two wire dialects exist, not seven; buy both from their owners.

## 3. Target Architecture

**Layers (top to bottom):**

1. **Facade** — the golden path: `chat()`, `stream()`, `json_out(Model, ...)`, each returning a frozen `Reply` (text, parsed, usage, cost_usd_micros, cache_read_tokens, model, trace_id). Cache-stable scope inferred from `system=`/`cached_prefix=`, not annotated per block — the `Stable(scope=GlobalScope())` caller tax dies (sme:dx-product). `plan()` remains as the explicit escape hatch, off the golden path.
2. **The one contract (the IR)** — one file, a Python analogue of Vercel's `LanguageModelV2` (research:frontier-meta): frozen `Request` (model target, messages, tools, output schema, effort, `provider_options: Mapping[str, JSON]` as the named non-commodity escape hatch); closed `StreamEvent` union of ~5 kinds (text_delta, tool_call, usage, terminal, provider_native); `Usage` with cache_read as a *subset* of input tokens; the kept `ExpectedModelFailure | RuntimeDefect` split, exhaustively matched via `assert_never`. Steal OpenRouter's `reasoning: {effort|tokens|exclude}` shape for the reasoning knob rather than inventing one (research:provider-wire §5). Honors `docs/rules/errors.md` (defects raise, errors are values) and boundaries.md's Absent/Present discipline.
3. **Engines** — one adapter per SDK client, 150–300 lines each, translating SDK objects ↔ IR. All clients constructed with `max_retries=0`; one retry owner above them (`RetryPolicy(` constructed only in one module — keep that negative gate), backoff via `stamina`. DeepSeek's strip-`reasoning_content`-before-resend rule lives in its adapter, nowhere else. SDK imports confined to `engines/` and enforced by the inverted negative-gate scan.
4. **Facts** — the catalog splits in two (sme:architect's ruling, adopted): **prices** become a vendored, hash-pinned snapshot of `pydantic/genai-prices` JSON with a daily CI drift-diff that fails red (sme:sre); **contract facts** (cache minimum-prefix tokens, TTL semantics, reasoning-level mappings, OpenRouter certification tri-state) stay hand-curated — the Kimi K3 `supports_implicit_caching` contradiction (catalog.py:770-806) is empirical proof no feed carries these truthfully.
5. **Cache-affinity** — kept, but **re-anchored to canonical IR-prefix serialization instead of provider wire bytes** (sme:architect). Hashing SDK-emitted bytes makes cache keys hostage to any SDK patch's dict ordering; hashing your own IR makes the policy engine-swap-immune. Golden-vector discipline retained; one-time cache-key v2 bump accepted.
6. **Observability** — `gen_ai.*` OTel spans emitted by the facade, always on: provider, model, attempt number, status, retry-after, cache-read tokens (watching the subset-not-additive double-count footgun). Free Langfuse/Braintrust/Datadog interop (research:frontier-meta).

**Subscription backends:** same repo, **separate package** (`agent_runtime`), shared nouns only (`Usage`, `Cost`, `Failure`, core stream kinds). Agents have filesystem/session state; forcing them through the provider contract is a category error (unanimous council). Keep: the `setsid()`/`killpg` process-group launchers (~450 lines — both vendor SDKs still orphan children; this bug recurs in any rewrite), a minimal `build_child_environment` scrub, per-profile state roots. Replace the exact-version runtime hard-fail: exact pin stays **in the lockfile**, the runtime gate downgrades to warning + capability probe, and a weekly re-cert job drives bumps (sme:sre's formulation — pin the artifact, don't hard-fail the process). Budget reality: Claude Agent SDK draws from a capped monthly credit pool overflowing to full API rates; Codex is the lower-risk reference lane (research:subscription-lane).

**Binding docs/rules honored:** simplicity.md, errors.md, retries.md (single policy owner), boundaries.md, and the negative-gate mechanism. The rest of `docs/rules/` (effect, layers, database, frontend, modules/) is imported from other repos and is deleted, not inherited.

## 4. The 80/20 Ledger

**KILLED (~15–16k src lines, plus proportional tests):**
- All 5 wire codecs + `_chat_completions_wire.py` — 4,492
- `transport.py` incl. SSE framer — 188
- `runtime.py`'s dual retry loops, `_next_or_cancel` cancellation race — ~500 of 682
- `schema.py` hand-rolled JSON-Schema subset — 744 (keep an ~80-line fingerprinter; Pydantic emits schemas)
- `catalog.py` hand-transcribed prices + 180-day staleness gate — ~500 of 871
- `capabilities.py` — 599; `policy.py` narrow/UnsafeConfirmation algebra — 261
- 17-kind event taxonomy → ~5 kinds; both `testing.py` doubles — ~500
- Exact-version runtime hard-fail gates; `_KNOWN_FIELDS` per-version allowlists
- `embeddings.py` unless a call site exists — 132
- `docs/agent-runtime-hard-cutover.md` (specifies transports the code gate-tests as absent — delete **today**, before it seeds a false baseline) and `docs/rules/{effect,effect-services,layers,database,frontend,modules/}`
- Fixture corpora for wire formats no longer parsed (the tests are half the 40k; a pivot that keeps them cut nothing — sme:pragmatist)

**KEPT (~3.5–4k lines):**
- Frozen types trimmed to the IR: closed unions, Absent/Present, `assert_never` exhaustiveness
- `ExpectedModelFailure` vs `RuntimeDefect` split
- `plan_generate` shrunk to validate → resolve → dispatch
- Cache-affinity hashing, re-anchored to IR + golden vectors
- Catalog contract-facts + OpenRouter certification tri-state
- Negative-gate source-scan mechanism (rewritten checks: SDK-imports-confined-to-engines, single retry owner, no sampling knobs)
- usd-micro cost on every `Reply`; secret redaction as defense-in-depth
- Agent lane: `_process.py` core + both launchers, minimal env scrub, terminal-result union
- `uv.lock` hash-pinning + `pip-audit` + Dependabot (the unsung crown jewel — sme:sre)

**BOUGHT:**
- `openai` (latest 1.x/2.x line), `anthropic` (latest 0.x/1.x line), `google-genai` (2.x) — all `max_retries=0`
- `pydantic` 2.x (schema gen + validation)
- `stamina` (backoff under the single retry owner)
- `pydantic/genai-prices` (vendored pinned JSON + CI drift check)
- `opentelemetry-sdk` for `gen_ai.*`
- `claude-agent-sdk` + `openai-codex` (lockfile-pinned, weekly re-cert)
- Deferred: `instructor` only if provider-native strict JSON fails on DeepSeek/Kimi
- **Not bought:** LiteLLM, any-llm, aisuite, LangChain, Pydantic AI, any hosted gateway

Net: provider lane ~9.5k→~4k covering **seven** providers instead of five; agent lane ~9.9k→~3k; tests shrink with the fixtures.

## 5. Hard-Cutover Migration Shape

**In-place, not fresh repo.** The seam is already narrow (`catalog.py`/`types.py`/`planning.py` never import a codec beyond five functions), so deletion is mechanical; a fresh repo buys only lost git history and re-litigation. Decisive secondary reason: two full hard cutovers shipped in 15 days (commits 11c4083, ebbe292) — the rewrite reflex is itself the risk, and in-place staged deletion is the discipline that breaks it (sme:pragmatist). Each phase lands on `main`, leaves a working library, no compat shims, no legacy paths.

1. **Phase 0 — hygiene (day one):** delete `docs/agent-runtime-hard-cutover.md` and the inapplicable `docs/rules/` subtree; delete or check in the dangling `.dossiers/provider-facts.md` reference.
2. **Phase 1 — IR:** write the contract file + facade signatures; port cache-affinity to IR-anchored hashing with new golden vectors (cache-key v2 bump, accepted once).
3. **Phase 2 — OpenAI-dialect engine:** one adapter over `openai` parameterized by base_url; wire OpenAI, DeepSeek, Kimi, Grok, OpenRouter. **This ships the two missing providers first** — the fastest visible win. Delete `openai.py`, `moonshot.py`, `openrouter.py`, `_chat_completions_wire.py`, the SSE framer, one retry loop.
4. **Phase 3 — Anthropic + Gemini engines:** native adapters; delete `anthropic.py`, `gemini.py`, `transport.py`, the second retry loop, `schema.py`.
5. **Phase 4 — facts + observability:** vendor genai-prices snapshot + CI drift job; split catalog into contract-facts overlay; add OTel spans; rewrite negative gates for the new invariants.
6. **Phase 5 — agent lane shrink:** extract to sibling package; keep launchers + env scrub; drop capability tables, policy algebra, 17-kind taxonomy; lockfile-pin + runtime-warn; script the Claude credit-pool opt-in.
7. **Phase 6 — live certification:** one paid live-matrix run across all seven providers + both agent backends; then a six-month no-rewrite moratorium (the condition sme:pragmatist attached — adopted).

## 6. Council Map

**Unanimous:** kill all five codecs and the SSE framer; three official SDKs, two dialects; Gemini native, never the compat shim; keep the defect/value split and the negative-gate *mechanism*; keep process-group launchers; lanes stay separate; no LiteLLM in-process; no gateway-for-everything; OpenRouter only as an explicit target for exotics.

**Disagreement 1 — the catalog.** Pragmatist: delete it first; its provenance file doesn't exist, its 180-day gate can't catch mid-window changes, and at n=1 wrong cost telemetry is a cheap trade. Architect/SRE: the *contract facts* (cache minimums, TTLs, certification tri-state) have no feed, and catalog.py:770-806 proves aggregator metadata lies exactly where it matters. **Ruled for the split (Architect/SRE):** prices are a commodity — buy the feed, pin the snapshot, diff daily in CI; contract facts stay hand-curated. The pragmatist is right about transcription toil, wrong that the *whole* artifact is telemetry: cache-contract facts feed the affinity planner, which is behavior, not reporting.

**Disagreement 2 — cache-affinity hashing.** Pragmatist/Futurist: it's a prediction of a cache you don't own; kill it unless it demonstrably changed a decision. Architect: keep it, but re-anchor to IR bytes so it survives engine swaps and gets *more* durable. **Ruled for the Architect, conditionally:** re-anchoring costs little (the policy and golden-vector discipline already exist), it's the only mechanism connecting prompt structure to spend, and killing it forecloses cheaply what rebuilding would cost dearly. The condition is Open Question 2 — if the owner can't show it mattered, demote it to a private diagnostic.

**Disagreement 3 — the no-fallback gate.** DX: the tree-wide "fallback" substring scan forbids even *naming* an explicit, caller-written `alt=` target; philosophy enforcing itself past usefulness. Purists: fallback is how silent cost/cache/behavior divergence creeps in; the gate is the cheapest guard against erosion. **Ruled mostly for the purists, with the DX amendment:** no automatic fallback, ever — but dual paths to one logical model may exist as *distinct, explicitly named `ProviderTarget`s* the caller selects (Architect's framing). The substring gate is rewritten to enforce "no implicit rerouting," not "the word is banned."

**(Also ruled:** OTel ships day one — SRE/DX/Futurist over Pragmatist's "no collector yet"; it's ~100 lines, it's the 11pm artifact, and retrofitting observability never happens. Agent-lane pinning: SRE's lockfile-pin + runtime-warn over both the exact-hard-fail status quo and the naive version floor.)

## 7. Philosophy

1. **Own the waist, rent the wire.** The contract is yours forever; serialization belongs to whoever fixes it at 3am — and for wire formats, that's the vendor.
2. **Differentiation lives in outputs, not input grammar.** Correct usd-micro cost and cache-hit accounting are features; making callers annotate `Stable(scope=...)` is a tax.
3. **Facts are pinned; transcription is not rigor.** A hand-typed number with a dangling provenance file is less trustworthy than a hash-pinned snapshot of a PR-driven feed with a daily drift alarm.
4. **One retry owner, zero hidden retries.** `max_retries=0` everywhere below the waist; a retry you didn't schedule is a bill you didn't authorize.
5. **No implicit rerouting — but named alternatives are intent, not fallback.** The caller may spell out a second target; the library may never invent one.
6. **Pin the artifact, don't hard-fail the process.** Lockfiles make upgrades deliberate; runtime version gates make patch releases into outages.
7. **Gates over guidelines.** A 20-line source-scan test outlives any philosophy document — but re-derive the gates from current beliefs; a gate enforcing 2024's constraint against 2026's substrate is debt with a green checkmark.
8. **The next rewrite is the enemy of this one.** Anything cheap to generate and expensive to own gets a moratorium, not a fresh repo.

## 8. Open Questions (owner-only)

1. **Monthly token spend, and what acts on cost numbers?** If spend is <$200/mo and nothing downstream consumes cost programmatically, usd-micro accounting is telemetry and can lose half its ceremony; if budgets/ledgers act on it, it keeps frozen-type status.
2. **Has cache-affinity hashing ever changed a decision or saved measurable money?** Yes → re-anchor and keep with golden vectors. No → private diagnostic function, no goldens.
3. **Does Agency (the Go runner) call this library?** If yes, the real contract is a wire/HTTP contract and this Python facade is one of two clients — that changes where the IR is canonically defined. If no, "one centralized library" means "one Python library," and the design above stands as-is.
4. **Claude Agent SDK credit pool: claimed, capped, or overflow-enabled?** Kill switch and hard cap must be explicit before the agent lane's weekly re-cert cadence starts burning quota.
5. **Acceptable frontier lag: hours or one certification cycle?** If hours, live certification becomes advisory (post-hoc) rather than a release gate for new provider features; if a cycle, keep it gating.
---

## Council positions (verbatim)

--- sme:pragmatist ---
## (a) Verdict

**The philosophy is sound. The volume is the bug — and the rewrite cadence is the real smell.**

Numbers I verified in the repo, not from the digests:

- 19,439 src lines + 21,055 test lines = **~40k lines to call 6 models** (`catalog.py` has exactly 6 `ModelContract` rows: gpt-5.6, gpt-5.6-luna/sol/terra variants, claude-fable-5/sonnet-5, gemini-3.5-flash, kimi-k3 via Moonshot and OpenRouter).
- **DeepSeek and xAI/Grok do not exist in the catalog at all.** Two of the seven required providers are unimplemented after 40k lines. That's the whole argument in one fact: the handrolled approach spent its entire budget on 5 providers' *syntax* and never reached provider coverage.
- `.dossiers/provider-facts.md`, the cited source of truth for every price in `catalog.py`, **does not exist in the repo** (confirmed with `find`). The strictest artifact in the codebase is sourced from a dangling reference.
- `git log`: `11c4083 2026-07-21 "Rewrite provider_runtime as the LLM provider hard-cutover"`, then `ebbe292 2026-08-05 "Complete native agent SDK hard cutover"`. **Two full hard cutovers in 15 days.** This will be the third.

That last point is the one I'd put on the whiteboard. The failure mode here isn't "handrolled vs. SDK," it's that a full rewrite is the owner's default response to dissatisfaction, and each rewrite is cheap to *generate* and expensive to *own*. A third cutover that lands at 8k lines instead of 19k is still a loss if there's a fourth in September.

## (b) What I'd do

Not a fresh repo. **Delete in place, on `main`, in four PRs**, each of which leaves a working library. The seam is already narrow — survey:provider-lane is right that `catalog.py`/`types.py`/`planning.py` never import a codec beyond 5 functions (planning.py:468-481, runtime.py:135-148). That means deletion is mechanical and a fresh repo buys nothing but the loss of git history and the temptation to re-litigate every decision.

1. **Adapters over three official SDKs** (`openai`, `anthropic`, `google-genai`), plus one OpenAI-compatible adapter parameterized by base_url for DeepSeek, Moonshot, xAI, and OpenRouter. Four adapters, ~250 lines each, replacing 4,492 lines of codec + `_chat_completions_wire.py`. Per research:provider-wire, there are two wire dialects, not seven; Gemini stays native because the compat shim drops context caching.
2. **Let SDK clients own retry and streaming.** Delete `transport.py` (188), the SSE framer, and both duplicate retry loops in `runtime.py` (593-676 and 399-526 — the docstring itself admits they duplicate delay/deadline rules). This is the fork survey:provider-lane flags; I take the SDK side. Cancellation-safe task racing (`runtime.py:277-314`) is exactly the code one person should not maintain.
3. **Pricing from `pydantic/genai-prices`**, vendored as a pinned JSON with a checked-in override file for rows you've personally verified. Keeps the "pinned facts" ethos; kills the transcription.
4. **Agent lane moves to its own package** (see disagreements).

Target: **~3k src / ~2.5k test**. Six providers actually reachable, including the two that don't exist today.

## (c) Three questions

1. **What have you shipped on this, and what's the monthly token spend?** If it's under ~$200/mo, the usd-micro accounting engine costs more to maintain than the tokens it counts, and the entire pricing/certification apparatus is optimizing a rounding error.
2. **When a catalog price is wrong, what actually breaks?** Name the downstream consequence. If the answer is "my own cost report is off," pricing is telemetry, not a contract — and telemetry doesn't get frozen types, certification gates, and a 180-day staleness gate.
3. **Do you want a model-calling library or a coding-agent harness?** Because you're building both and calling it one thing.

## (d) Where I disagree with the council

**vs. survey:provider-lane — "keep `catalog.py` and cache-affinity hashing, full stop."** No. The certification-gated pricing ledger is my *first* deletion, not my protected core. Its provenance file doesn't exist; its 180-day gate cannot detect a mid-window price change; it encodes a future Anthropic rate switch as a comment (catalog.py:625-629). It is 871 lines of hand-typed numbers for 6 models, competing with four maintained free feeds (research:frontier-meta). The trade-off I'm accepting: **occasionally wrong cost reporting**, in exchange for deleting the repo's highest-toil surface. I'll take that trade every day at n=1.

Same for cache-affinity hashing (planning.py:145-261), called "the single most differentiated piece of code in the repo." It's a *prediction* of a cache you don't own. Providers decide hits; your hash observes. Differentiated is not the same as valuable — port it only if the owner can point at a decision it changed.

**vs. anyone proposing Mozilla any-llm.** Tempting, and the digest is right that it matches the philosophy. But it's 2,150★, v1.0 is nine months old, one org, and it would sit *between* you and three first-party SDKs you'd otherwise call directly. For six models, that's a bus-factor dependency buying you ~800 lines. Steal its `registry.py` pattern — declarative rows for OpenAI-compatible providers — and skip the dependency. Adopt it only if provider count goes past ~12.

**vs. research:gateways' reliability framing.** Right conclusion, wrong reasoning. Trustpilot scores and a 1.7/5 rating are not engineering evidence at n=1 — you have no SLA to breach and no users to page. The real reason to stay direct is structural: **caches live where they're written** (OpenRouter's own docs), so any router hop silently voids your cache. That's a correctness argument. Keep OpenRouter as the tail for exotic models only.

**vs. research:frontier-meta on OTel `gen_ai.*`.** Don't. You have no collector, no evals platform, and one user. Emitting an unstable-spec convention (nothing marked Stable as of July 2026) into `/dev/null` is exactly the speculative surface `docs/rules/simplicity.md` forbids. Add it the day Langfuse is actually installed.

**vs. survey:agent-lane on "one centralized library."** research:subscription-lane kills the premise: since 2026-06-15 the Claude Agent SDK draws from a capped credit pool billed at **full API rates** on overflow. The subscription is no longer cheap tokens. What remains is "I want Claude Code's *harness*" — filesystem, skills, approvals, sessions — which shares nothing with stateless completion calls but a billing account. Two lanes, two packages, two release cadences (the agent SDKs ship 1-2 releases/day; your provider adapters ship monthly). Forcing one contract over both is how you got a 599-line capability table.

**On tests.** Everyone is discussing src lines. 21k of the 40k are tests. A pivot that cuts src to 3k and keeps 21k of tests has cut nothing. Delete the fixture corpora for wire formats you no longer parse.

## (e) The cut

**Keep (~3k lines):** `types.py` trimmed to the result/failure union and Absent/Present; the defect-vs-expected-failure split (this is genuinely good and free); `plan_generate` reduced to validate→resolve→dispatch; the negative-gate *mechanism* (source-scan CI tests are the cheapest governance in the repo — but rewrite the checks, don't inherit them); process-group `killpg` cleanup + the two launcher shims (~450 lines, patches real cited bugs both vendor SDKs still have); a minimal `build_child_environment`.

**Kill (~16k lines):** all 5 wire codecs + `_chat_completions_wire.py` (4,492); `transport.py` + both retry loops; `schema.py`'s handrolled JSON-Schema subset (744 → `pydantic.TypeAdapter.json_schema()`); `catalog.py`'s pricing transcription and certification tri-state; `capabilities.py` (599); `policy.py`'s UnsafeConfirmation algebra (261); exact-version-pin-and-fail gates (floor + smoke test instead); the 17-kind event taxonomy → 6 kinds; both `testing.py` doubles; `docs/rules/{effect,effect-services,layers,database,frontend}.md` and `docs/rules/modules/`; `docs/agent-runtime-hard-cutover.md` (911 lines specifying transports the code deleted — its own header says delete it).

**Buy:** `openai`, `anthropic`, `google-genai`, `pydantic` (schema gen), `stamina` (retry, where SDK retry is insufficient), `genai-prices` (pricing), `claude-agent-sdk` + `openai-codex` (separate package). Defer `instructor` until provider-native structured output actually fails you.

**The condition I'd attach:** no fourth rewrite for six months. If the answer to that is no, don't do this one either.

--- sme:architect ---
## (a) Verdict

The philosophy is right and the waist is in the wrong place.

"One standardized contract, engines swappable underneath" is exactly correct. But this repo didn't build a semantic waist — it built a *transport* waist and called it a contract. Look at the seam: `FinalizedProviderRequest{body: bytes}` and `_Codec.decode_stream(headers, AsyncIterator[SseEvent])` (runtime.py:126-148). Everything above the line is typed and closed; everything crossing the line is bytes and SSE frames. That's why the survey concludes SDK adoption "would force a real rewrite" of runtime.py: you can't insert an engine that owns its own serialization into a waist whose currency is serialized bytes.

The falsification is quantitative, not aesthetic: 9,537 lines in the provider lane and `ProviderName` is still `Literal["openai","anthropic","gemini","moonshot","openrouter"]` (types.py:24). **DeepSeek and Grok — two of seven required providers — do not exist after 14k lines.** Marginal cost per provider is ~600-1000 lines of codec. That is the number that decides this. any-llm ships DeepSeek in 79 lines and OpenRouter in 50.

The differentiated 20% is real and I will defend it — but it is *smaller and better-anchored* than the repo thinks.

## (b) My approach

Define the IR first, in one file, before touching any engine. Request / Response / a closed `StreamEvent` union / `Usage` / a `Failure` taxonomy / `provider_options: Mapping[str, JSON]` as the explicit non-commodity escape hatch (Vercel's `LanguageModelV2` + `providerOptions` split, per the frontier digest — the design is proven, it just has no Python incarnation to import).

Then three rules that make the facade actually stable:

1. **Engines sit strictly below the waist.** `openai` SDK (OpenAI, DeepSeek, Kimi, Grok, OpenRouter via base_url), `anthropic` SDK, `google-genai` native. Adapters translate SDK objects ↔ IR. ~150-300 lines each. Zero SDK types in the public surface — enforce it with the existing negative-gate mechanism inverted: SDK imports confined to `engines/`.

2. **Re-anchor cache-affinity to the IR, not to wire bytes.** This is the single most important change and nobody else on this council will say it. Today `DraftRequest.prefix_bytes` is "the codec's exact length-framed serialization" (types.py:740-747) and the golden vector pins a hash of *provider wire bytes*. Adopt any SDK and your cache keys become hostage to that SDK's serializer: a patch bump reorders a dict, your hash changes, your cache silently cold-misses, and your billing model quietly lies. Hash a canonical serialization of **my** IR prefix instead. Same policy, same golden-vector discipline, now immune to engine swaps. This makes the differentiated 20% *more* durable, not less.

3. **One retry owner, SDK retries off.** Instantiate every client with `max_retries=0` and keep a single attempt loop over an engine-agnostic `attempt()` coroutine. Kill the duplicated stream/non-stream loops (runtime.py:399-526 vs 593-676) the docstring already confesses to. Backoff from `stamina`, not hand-rolled.

## (c) Three hardest questions

1. **Which invariant do you refuse to break: byte-stable cache keys, or engine swappability?** They are in direct conflict today. I want your answer to be "swappability, and I accept a one-time cache-key v2 bump," because the alternative means never adopting an SDK.

2. **Is cost accounting *authoritative* or *observational*?** If you act on those numbers (budgets, ledgers, per-app chargeback), pinned usd-micros earn their 871 lines. If you glance at them monthly, it's a pinned snapshot of `genai-prices` and 80 lines. This one question is worth ~700 lines.

3. **How many call sites do you actually have, and does any one of them need a provider-specific knob today?** If the honest answer is "three apps: chat, tools, structured output," then the 20-dimension capability matrix and the 7-provider ambition are speculative surface, and the contract should be sized to three apps plus `provider_options`.

## (d) Where I disagree with this council

- **Against the "adopt any-llm" faction:** any-llm is a fine *engine*, a terrible *waist*. Depending on it means their completion type becomes your type vocabulary — you asked for ONE contract and you'd be adopting someone else's, including their normalization choices on precisely the 20% you care about (Anthropic cache write-vs-read split, Gemini thinking budgets, DeepSeek's `reasoning_content` strip-before-resend). Trade-off, named: ~400 lines of adapters versus surrendering the type vocabulary and the cache/usage semantics. Pay the 400. Steal their quirk list and their declarative registry pattern; don't import their contract.

- **Against the "kill the catalog, use the feeds" faction:** split it. *Prices* are a commodity — buy `genai-prices`, snapshot it, pin the snapshot's hash, refresh deliberately. *Contract facts* are not: cache minimum-prefix tokens, TTL semantics, reasoning-level mapping, certification provenance. No feed carries those at the needed precision, and catalog.py:770-806 (OpenRouter claiming `supports_implicit_caching=false` while Moonshot caches anyway) is the empirical proof that aggregator metadata is untrustworthy exactly where it matters. Keep the tri-state certification gate. Delete the hand-transcribed prices only.

- **Against the "drop the ceremony" simplifiers:** keep the negative gates and keep the stream-invariant validator. Gates are ~200 test lines that mechanically prevent the one failure mode that kills a facade — contract erosion (a fallback here, a sampling knob there, an engine type leaking into a return signature). The stream validator earns its keep because streaming violations are invisible until production. Drop `policy.py`'s narrow/patch algebra, the 599-line capability table, and the exact-version hard-fail — those defend against threat models a single trusted user doesn't have.

- **Against the gateway faction:** OpenRouter is a *target*, never a *fallback*. Dual-path to the same model means two cache-affinity domains and two accounting truths for one logical call. If both exist, they are distinct `ProviderTarget`s the caller names explicitly.

- **Against merging or unifying the lanes:** two facades, shared *nouns*. `Usage`, `Cost`, `Failure`, and the core stream-event kinds are one vocabulary; the call shapes stay separate. Agents have filesystem and session state; forcing them through `FinalizedProviderCall` is a category error.

## (e) The 80/20 cut

**Keep (~2.5k lines):** the IR/type vocabulary (Presence, closed unions, `assert_never`); the defect-vs-expected-failure split; cache-affinity *policy* re-anchored to IR bytes + golden vectors; catalog contract-facts + certification tri-state; `plan_generate`'s intent→plan compilation, shrunk; negative gates; stream-invariant validator; process-group ownership in the agent lane (`_process.py` + launchers — both vendor SDKs still leak, so this recurs in any rewrite).

**Kill (~7k lines):** all five wire codecs, `_chat_completions_wire.py`, `transport.py`'s SSE framer, `schema.py` (Pydantic emits the schema; keep an ~80-line fingerprinter), one of the two retry loops, hand-transcribed pricing, `capabilities.py`, `policy.py`, exact-version hard-fail pins (→ floor + certified-version test assertion), `embeddings.py` unless a call site exists today, and `docs/agent-runtime-hard-cutover.md` plus the entire imported `docs/rules/` subtree except simplicity/errors/retries/boundaries.

**Buy:** `openai`, `anthropic`, `google-genai`; `pydantic` for schema; `stamina` for backoff; `genai-prices` as a pinned snapshot; OTel `gen_ai.*` span emission (watch the cache-token subset double-count).

**Target:** provider lane ~4k lines covering **seven** providers instead of 9.5k covering five. That ratio — not the line count — is the case.

--- sme:sre-security ---
## (a) Verdict

The philosophy is defensible; the *justifications* are wrong, and one of the actually-valuable properties is undocumented.

From my seat the crown jewel isn't the codecs or the catalog — it's this: `uv tree --no-dev` on the base install is **httpx + 5 transitive packages** (anyio, certifi, httpcore, h11, idna), hash-locked in `uv.lock`, with `pip-audit` and Dependabot wired into CI (`.github/workflows/ci.yml:27`). Six packages sit in the process that holds seven API keys. That is a genuinely excellent blast radius and nobody in the packet names it as a thing being traded away.

What's *bad* is exactly the code that fails at 3am, not the code that builds JSON. `runtime.py` runs **two independent retry loops with duplicated deadline math** (its own docstring at :1-17 admits this), plus a bespoke cancellation-race (`_next_or_cancel`, :277-314) and a hand-rolled SSE framer (`transport.py:43-77`). Body construction is boring and testable; retry/stream state machines are where you get silent double-billing and hung tasks.

And the certification story is partly theater. `tests/test_catalog.py:3` cites `.dossiers/provider-facts.md`, **which does not exist in the repo**. Pricing is human-transcribed with a **180-day** staleness gate (`catalog.py:845-871`) — meaning a provider price change can mis-bill for up to half a year with the system reporting "certified." An untested backup that reports green is worse than no backup.

Agent lane: `claude_sdk.py:384-385` hard-fails on anything but `claude-agent-sdk==0.2.130`. Upstream is at 0.2.134 (2026-08-08) shipping 1–2 releases/day, and **0.2.129 was a security fix** (`--allowedTools` injection via skill-name wildcards). You have converted "apply a security patch" into "re-read vendor internals, re-run a paid live matrix," and converted "degraded" into "total outage." Right instinct, worst possible failure mode.

## (b) What I'd do

1. **Two dialects, three first-party SDKs.** `openai` (OpenAI + DeepSeek + Kimi + Grok + OpenRouter via `base_url`), `anthropic`, `google-genai` native (never the compat shim — it drops caching per the wire digest). Delete ~9.5k lines of codecs and the SSE framer.
2. **`max_retries=0` on every SDK client.** Non-negotiable. SDK-internal retries × your retry loop = multiplicative retry storms during precisely the incident you're retrying through. One retry owner (keep the `RetryPolicy(`-only-in-planning gate), one deadline, `Retry-After` honored.
3. **Make mid-stream failure explicit.** There is no resumption; a stream that dies at 90% re-bills the entire prefix on retry. Model it as a first-class outcome with cost attribution, don't hide it in an attempt trace.
4. **Automate the catalog data, keep the catalog schema.** Vendor `pydantic/genai-prices` or `models.dev` JSON as a pinned artifact; add a CI job that diffs pinned vs upstream daily and fails red. Same rigor as certification, ~5% of the labor, and it catches drift in a day instead of 180.
5. **Emit `gen_ai.*` OTel spans** — attempt number, provider, model, status, retry-after, cache-read tokens. This is not observability fashion; it's the artifact you read at 3am. Watch the cache-token double-count footgun the frontier digest flagged.
6. **Keep the lanes physically separate** — different credential classes, different ToS regimes, different failure modes.

## (c) Three questions for the owner

1. **Where is the hard spend cap?** Certification burns paid probes; the Anthropic Agent SDK credit pool ($100–200/mo on Max) **overflows to full API rates**, and "unchecked agentic spending" is the top OpenRouter complaint cluster. Show me the kill switch, not the budget.
2. **OpenRouter returns `401 User not found` during an infra failure (documented Feb 2026) at 3am. What do you want to happen?** "No fallback" means your app is down until you wake up. Is the real requirement *no fallback*, or *no silent fallback*? Those are different systems.
3. **Have you actually claimed the Agent SDK credit opt-in, and what breaks if that Max account gets flagged?** Anthropic banned ~1.45M accounts in H2 2025 with ~3.3% appeal success. That one account gates both your library *and* your interactive coding environment.

## (d) Where I disagree

- **Against any-llm** (the unified-libs digest's top pick): no. 2,150★ / one org / 37 issues is fine hygiene, but it inserts a third party between you and *every credential you own*, and you inherit the union of provider-SDK transitive closures **anyway**. The 50–230 lines/provider it saves is the cheap code. If you're taking the SDK dependency, take it first-party — those vendors have security contacts and CVE processes; a Mozilla side project's maintainer account is your new supply-chain SPOF.
- **Against any gateway, including Vercel AI Gateway** (gateways digest §4): a gateway is a new credential custodian and a new SPOF you don't operate and cannot page. "Zero markup" doesn't price the incident. OpenRouter at 1.7/5 Trustpilot, no SLA, Discord-only support, and a demonstrated willingness to 403 accounts is a governance risk with no recourse at n=1.
- **Against the agent-lane digest's "replace exact pin with a version floor."** Floors are how a package shipping 1–2 releases/day breaks you unattended. Correct answer: keep the exact pin *in the lockfile*, downgrade the **runtime** gate from `SdkUnavailable` to a warning + capability probe, and let Dependabot + a weekly re-cert job drive upgrades. Pin the artifact; don't hard-fail the process.
- **Partially against "the catalog is untouchable."** The *policy* (usd-micros, cache contracts, defect/value split) is the differentiator. The *data* is a hand-typed liability with a dangling provenance file. Keep the type, automate the rows, fail CI on drift.

## (e) The 80/20 cut

**Keep** (~3.5–4k lines): frozen types + presence discipline; cache-affinity hash + golden vectors; defect/value split; **negative-gate source-scan tests** (cheapest security control in the repo — keep the *mechanism* in any fresh repo); process-group `setsid`/`killpg` launchers (~450 lines; both vendor SDKs still orphan children); minimal `build_child_environment` scrub; `uv.lock` + `pip-audit` + Dependabot; secret redaction as defense-in-depth; no-fallback; single retry owner.

**Kill**: all 5 wire codecs + `_chat_completions_wire.py` + SSE framer (~9.5k); the second retry loop; certification tri-state gating; hand-transcribed prices; `capabilities.py` (599); `policy.py` UnsafeConfirmation algebra (261); both `testing.py` doubles; the 17-kind strict event taxonomy → shrink to terminal-result + usage; `docs/rules/{effect,effect-services,layers,database,frontend,modules}`; and **delete `docs/agent-runtime-hard-cutover.md` today** — it specifies transports the code gate-tests as absent and will seed a false baseline into the new repo.

**Buy**: `openai`, `anthropic`, `google-genai` (all `max_retries=0`); `genai-prices`/`models.dev` JSON vendored + CI drift check; `stamina` or one hand-written backoff for the single retry policy; OTel `gen_ai.*`. Later, only if needed: `instructor`.

**Do not buy**: LiteLLM in-process (4,806 open issues, 9–12k-line files, maintainers moved the core to Rust — that's a gateway product now), any-llm, aisuite, LangChain, Pydantic AI, or any hosted gateway.

--- sme:dx-product ---
## (a) Verdict

The philosophy is defensible; **the surface area is the actual defect.** I counted the exports: `provider_runtime.__all__` = **151 names**, `agent_runtime.__all__` = **118**. 269 public concepts for "call an LLM." The shortest happy path is a three-phase compiler the caller has to drive by hand: build `GenerateIntent` (7 fields, each a closed union) → `plan_generate(intent, CATALOG)` → match `FinalizedProviderCall | PlanRejected` → construct `httpx.AsyncClient` → `ProviderRuntime.generate(call, credential=...)` → match `Succeeded | Failed` → `cost_from_accounting(...)`. And before any of that I must annotate every prompt block with `Stable(scope=GlobalScope())` vs `Dynamic()` (`tests/test_openai.py:100-104`) — a caller tax that exists to feed an *internal* SHA-256 cache key.

That's the tell. The repo exposes its compiler IR as its API. `survey:provider-lane` is right that cache-affinity hashing is the most differentiated code here — and it's exactly the thing that should be **invisible**. Differentiation belongs in the output (correct usd-micro cost, cache-hit accounting), not in the input grammar.

Second: **zero observability out of the box.** `AttemptRecord` tuples in a returned dataclass is not observability; it's a debugger exercise. `research:frontier-meta` is right — `gen_ai.*` OTel spans are free interop with Langfuse/Braintrust/Datadog and this repo emits nothing.

## (b) My approach

Two layers, hard boundary:

```python
from llm import chat, json_out, stream

text  = await chat("anthropic:opus-5", system=SYS, user=q, effort="high")
inv   = await json_out(Invoice, "openai:gpt-5.6", user=doc)   # returns Invoice
async for delta in stream("deepseek:reasoner", user=q): ...
```

Every call returns `Reply(text, parsed, usage, cost_usd_micros, cache_read_tokens, model, trace_id)`. Cache stability is inferred (`system=` and anything passed as `cached_prefix=` is the stable scope) — not annotated per block. Structured output is `pydantic.BaseModel` in, instance out; delete `schema.py` (744 hand-rolled lines) because `research:provider-wire` #6 says Anthropic went GA Feb 2026 and OpenAI/Gemini/Grok all take strict JSON schema natively.

Layer 2 is the escape hatch for the 5% — `plan(...)` returning the finalized request — kept but *not* on the golden path. Under the hood: three official SDKs (`openai`, `anthropic`, `google-genai`), because `research:provider-wire` establishes there are two dialects, not seven. OTel spans emitted by the facade, always on.

## (c) Three hardest questions

1. **Show me the last five call sites you actually wrote in Nexus.** How many of the 269 exports do they import? If it's under 15, the other 254 are pure carrying cost — and the API should be shaped like those five sites, not like the compiler.
2. **Agency is a Go runner.** A Python library cannot be the "ONE centralized library" for it. So either there's already an HTTP boundary (meaning the real contract is a wire contract and this repo is one of two clients), or Agency doesn't use this at all and the "one contract" goal is fiction. Which is it? The answer changes the whole architecture.
3. **It's 11pm and a call failed. What do you open?** If the honest answer isn't "a Langfuse trace," then no amount of `AttemptRecord` rigor is buying you anything at the moment you need it.

## (d) Where I disagree with the council

**With the provider-lane archaeologist:** they say keep `catalog.py` "full stop, in any rewrite." I disagree on the *pricing* half. `survey:docs-rules-tests` found that `.dossiers/provider-facts.md` — the cited source of truth — **doesn't exist in the repo**. So the catalog isn't owning the truth, it's owning the transcription errors, plus a 180-day staleness gate and an Anthropic rate change encoded as a code comment (`catalog.py:625-629`). Take `pydantic/genai-prices` or `models.dev`, vendor a *pinned snapshot* JSON, diff it in CI. Same pinning discipline, none of the manual transcription. Keep hand-curation only for the certification tri-state on OpenRouter routes, which genuinely has no feed.

**With whoever argues for `any-llm` as the substrate.** It's the right philosophy (`research:unified-libs`) but the wrong bet at this size: 2,150★, v1.0 nine months old, and its `provider_options` escape hatch means per-provider quirks reappear at *my* call site anyway. I'd rather own ~400 lines of facade over three official SDKs than depend on a fourth party's IR for the thing every one of my apps runs through. Steal their registry pattern, not their package.

**With the purists on "no fallback, ever."** At 11pm when Anthropic 529s, "no fallback" means my app is down and I get to feel principled about it. I want `alt="openrouter:claude-opus-5"` as an *explicit, caller-written second argument* — visible in the signature, stamped in the outcome, never automatic. That's not hidden fallback; that's the caller expressing intent. The current gate (`tests/test_negative_gates.py:219` greps the whole tree for the substring "fallback") would forbid even naming the concept. That gate is philosophy enforcing itself past the point of usefulness.

**With the gateway skeptics:** I agree don't route everything through OpenRouter (1.7/5, no SLA, `research:gateways`). But note *why* a gateway tempts: it's the only proposal on the table that ships one place to see cost, latency and traces. If we go direct-SDK, OTel spans are non-negotiable day-one scope, not a follow-up.

## (e) The 80/20 cut

**Keep** (~1.5k lines): defect-vs-value split, reduced to two exception types and one `Reply | Failed` union; usd-micro cost on every reply; cache-affinity hashing as a private function; the agent lane's process-group `setsid`/`killpg` cleanup (~450 lines — `survey:agent-lane` proves both vendor SDKs still orphan children); the negative-gate source-scan *mechanism*; a pinned catalog snapshot.

**Kill** (~13k lines): all 5 wire codecs + `_chat_completions_wire.py` (~4.5k); `schema.py` (744); `capabilities.py` (599); `policy.py`'s narrow/unsafe-confirmation algebra (261); the 17-kind strict event taxonomy → 4 kinds (delta/tool/usage/terminal); exact-version SDK hard-pins → floor + warn; `Stable/Dynamic/PromptBlock` at the call site; 269 exports → ~25.

**Buy:** `openai` + `anthropic` + `google-genai`; `pydantic`; `genai-prices`; `opentelemetry-sdk` for `gen_ai.*`; `stamina` for retry. `instructor` only if native strict-JSON proves insufficient on DeepSeek/Kimi.

Target: ~3k provider lane, ~1.5k agent lane, one import, one verb, one `Reply`.

--- sme:futurist ---
## (a) Verdict: the architecture is right, the *substrate* is wrong

This repo is not over-engineered in the places people usually mean. `catalog.py`/`planning.py`/`types.py` encode policy nobody sells (survey:provider-lane §3) and the negative-gate tests are a genuinely good invention (survey:docs-rules-tests §1). The failure is one of **substrate choice at the wrong altitude**: ~9.5k lines of hand-rolled wire codecs and a bespoke SSE framer sit exactly where the industry has commoditized. `research:provider-wire` is decisive — two dialects, not seven; `openai-python` (31.3k★, Chat Completions "supported indefinitely") and `anthropic-sdk-python` cover them.

The load-bearing philosophical error is the "no provider SDKs" gate. It was defensible in 2024. In 2026 it is a **structural lag machine**: the day Anthropic shipped `output_format`/`strict:true` GA (2026-02-04) or moved `thinking:{budget_tokens}` → `effort=…xhigh/max`, the SDK users got it on `pip install -U` and this repo got a codec ticket plus a recertification run. The evidence is already sitting in `pyproject.toml`: `claude-agent-sdk==0.2.130` hard-pinned, hard-failing (`claude_sdk.py:384`), while upstream is at **0.2.134** (2026-08-08, ~1–2 releases/day per research:subscription-lane). The repo is four releases stale on a lane where the pin *is* the outage. That's the whole thesis in one line of TOML.

Counter-evidence I have to concede: the agent lane already "used the official SDK" and produced **11.5k lines wrapping 3.5k of adapter code** (survey:docs-rules-tests §2). SDKs don't automatically buy simplicity. But read the *cause* — capability tables, 17-kind event taxonomy, `UnsafeConfirmation` policy algebra, exact-version gates — none of that was forced by the SDK. It was self-imposed. The lesson isn't "SDKs don't help," it's "this owner will gold-plate any substrate you give him." Design the new repo to make gold-plating expensive.

## (b) What I'd do: one owned IR, three official SDKs, community-fed facts

Fresh repo. Hub-and-spoke, not point-to-point (research:frontier-meta cites arxiv 2604.09360 on the O(n²) adapter trap this repo is currently living in).

1. **Own the IR, rent the wire.** Define a Python analogue of Vercel's `LanguageModelV2` — request/response/stream-event shapes plus a `provider_options` escape hatch for the non-commodity 20% (thinking budgets, safety settings, `cache_control`). Keep `plan_generate`'s intent→plan step and the cache-affinity hash; those are the crown jewels.
2. **Three SDK clients, six adapters.** `openai-python` for OpenAI + DeepSeek + Kimi + Grok + OpenRouter (base_url swaps), `anthropic-sdk-python` native, `google-genai` native (never the compat shim — it drops context caching and native multimodal, research:provider-wire). Target 80–250 lines per adapter, per any-llm's proven shape (DeepSeek = 79 lines, OpenRouter = 50).
3. **Use SDK clients end-to-end, including their retry and streaming.** This is the fork survey:provider-lane §5.3 names, and I take the aggressive branch: delete `transport.py`, delete `runtime.py`'s dual retry loops and `_next_or_cancel`. Keep one thin attempt-trace wrapper over `stamina`.
4. **Stop hand-transcribing prices.** Vendor `pydantic/genai-prices` JSON at a pinned commit; keep your own thin overlay for cache-contract facts the feeds get wrong (the Kimi K3 `supports_implicit_caching` contradiction, catalog.py:770-806). Pinned *and* current — same ethos, one-tenth the toil.
5. **Emit `gen_ai.*` OTel spans from day one.** Free Langfuse/Braintrust/Datadog interop. Watch the cache-read double-count footgun.
6. **Agent lane: keep separate, shrink hard.** Keep the setsid/killpg launchers — both SDKs still orphan processes, that bug survives any rewrite. Replace exact-version hard-fail with a floor + warning.

**Should you use any-llm instead of your own adapters?** Evaluate it seriously (2,150★, 37 open issues, official-SDK-first, all seven providers present). But I'd take it as a *reference implementation to read and steal from*, not a dependency — because your `provider_options` passthrough and cache semantics need to be exactly yours, and any-llm's abstraction will lag native features by exactly the interval that kills you.

## (c) Three questions for the owner

1. **When Anthropic ships a new reasoning knob on a Tuesday, what is your acceptable lag — hours, or one recertification cycle?** If the answer is "hours," the pinned-catalog + live-certification model is the binding constraint, not the codecs, and no rewrite helps unless certification becomes async/advisory rather than a gate.
2. **Is byte-exact cache-affinity hashing a real economic need or an aesthetic one?** Show me a month of spend where it saved money. It's the single most differentiated thing here (planning.py:145-261) and the single hardest thing to carry through an SDK cutover. If it saved $40, kill it.
3. **How many of your apps actually consume the closed failure taxonomy exhaustively?** If the answer is "one, and I wrote it," `ExpectedModelFailure` vs `RuntimeDefect` can collapse to two exception types and you delete ~1.5k lines of ceremony.

## (d) Where I break with the council

- **Against the "hybrid: direct SDKs primary, OpenRouter for exotics" recommendation (research:gateways §3).** Half-right, but it undersells OpenRouter's *specific* asymmetric value: **day-zero catalog freshness** (GPT-5.6 and Kimi K3 both same-day, July 2026). For a futurist optimizing "adopt next month's capability the week it ships," OpenRouter is the *experimentation* lane — you try the new model there on day zero, and promote to a direct SDK once it earns a place in production. That's a lifecycle, not a fallback tier. The trade-off I'm accepting: 5.5% fee, no SLA, 1.7/5 Trustpilot, and broken cache continuity across routing hops — all fine for the try-it-today lane, disqualifying for the daily-driver lane.
- **Against anyone citing the agent lane's 11.5k lines as proof that SDK adoption doesn't simplify (survey:docs-rules-tests §2/"So what" #2).** Correlation, not causation. The wrapper is big because of self-imposed capability tables and event taxonomy, not because `claude-agent-sdk` demanded it. Don't let that number veto the provider-lane cutover, where the deletion is unambiguous.
- **Against the conservative read of "keep the catalog, it's unbuyable" (survey:provider-lane §3).** It's unbuyable at *that precision*, and the precision is the problem. A 180-day staleness gate with manual transcription and a dangling `.dossiers/provider-facts.md` provenance reference is not more trustworthy than four PR-driven public feeds — it's less current with more ceremony. Keep the *overlay*, kill the *transcription*.
- **Against reflexively keeping the negative-gate corpus intact.** Keep the mechanism; delete `test_no_fallback_policy`'s tree-wide substring scan for "fallback" — that gate now literally forbids the one pattern (SDK-native retry + provider-tier promotion) most worth adopting. A gate that enforces 2024's philosophy against 2026's substrate is technical debt with a green checkmark.

## (e) The 80/20 cut

**Keep (~4k lines):** intent→plan compiler; cache-affinity hash *if* Q2 answers yes; frozen types trimmed ~40%; usd-micro accounting; secret redaction; negative-gate *mechanism*; process-group launchers (`_process.py` + both launchers, ~450 lines); a minimal `build_child_environment`.

**Kill (~12k lines):** all five wire codecs + `_chat_completions_wire.py` + `transport.py` (~9.7k); `runtime.py`'s dual retry loops and `_next_or_cancel` (~680); `schema.py`'s hand-rolled JSON-Schema subset → Pydantic `model_json_schema()` (744); `capabilities.py` (599); `policy.py` (261); exact-version hard-fail gates; `docs/rules/{effect,effect-services,layers,database,frontend,modules}` and `docs/agent-runtime-hard-cutover.md` — delete before it seeds the new repo with a false baseline.

**Buy/rent:** `openai` + `anthropic` + `google-genai`; `stamina` for retry; `pydantic` for schema/validation; `pydantic/genai-prices` vendored-pinned for pricing; OTel `gen_ai.*` for observability; `instructor` later if structured output needs to exceed native `response_format`. **Do not buy LiteLLM in-process** — 4,806 open issues, 9–12k-line files, and its maintainers moved the core to Rust because the Python translation layer didn't hold.

Target: ~6–7k lines total, capability-current within a `pip install -U` of frontier.