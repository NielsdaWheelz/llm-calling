# native provider evidence

branch: `feature/native-agent-supervision`, declared base
`69d41d38a3d290e7ae3bde9b57556dda41e1b2f1`.

## scope and attribution

the prepared-turn engine, submission evidence, reader-latched terminal, portable
callbacks, steering, interrupt, per-session cleanup, and terminal codec are
provider-owned. kernel journals and app effects consume these contracts; a
parent app outcome is not native recovery authority.

nexus already required the newer raw generation/continuation contract at its
provider pin `6a7093f799c88c205c8d797bbb8b14c2a9980db1`. the declared provider
base lacked that contract. the committed raw implementation and corresponding
conformance cases were adopted from that exact ancestor, rather than adding a
helper that decoded obsolete artifacts. native session/catalog behavior was
not borrowed from an unqualified later native lane. raw generation uses the
current nine-model/five-provider catalog and native continuation v2.

## controlled red / green

temporary acceptance cases lived in `tests/native_acceptance/`. they were
deleted after integrated final acceptance; the receipts below retain their proof.

- first native boundary run: two failures and one pass demonstrated premature
  structured-output rewriting and lost terminal proof after disconnect.
- callback integration exposed unresolved server-request accounting after an
  actual callback reply and native dynamic-item completion; fixed at the
  provider's pending-request owner.
- a peer withholding start acknowledgement until callback reply proved early
  accepted identity and removed the submit/callback deadlock.
- reader loss while the actual transport write lock was held and a deadline
  before writer entry proved cancellation/quiescence before non-submission.
- twelve final controlled cases pass: prepared no-i/o and immutable bytes,
  safe negative proof, post-writer uncertainty, original errors, callback
  routing, steering ack versus recorded input, interrupt, terminal retention,
  invalid json, and two-session cleanup isolation.

```sh
.venv/bin/python -m pytest tests/native_acceptance/test_provider_turn.py -q
.venv/bin/pyright --pythonpath .venv/bin/python src/provider_runtime
.venv/bin/python -m pytest tests --ignore=tests/live -q
```

deterministic conformance before the experimental-protocol followup: 987 cases
pass, three explicitly skip,
one intentional invalid-sdk-fixture serializer warning. source pyright reports
zero errors. raw conformance: 482 cases pass after adopting the exact prerequisite. existing
linux-only pidfd descriptor and process checks remain explicitly platform-bound;
macos process-group checks use native process state rather than absent `/proc`.

## actual native proof

the isolated live host uses the real installed codex `0.160.0`, the actual
personal subscription profile `codex-personal`, a private host-only auth copy,
an empty credentialless worker state root, and a separate read-only cwd.
no shared daemon was stopped or attached.

the first actual run failed closed on newly installed
`thread/settings/updated`. the installed experimental json schema supplied
the exact public shape. validation now checks selected model, reasoning, cwd,
provider, approval posture, and read-only/no-network sandbox settings; the
prepared native callback and strict-json turn then passed. the earlier repeat
passed again in 4.75 seconds after reply/cancellation refinements.

the containment followup opts into the supported experimental protocol even
when no callbacks are declared, because `environments=[]` requires it. the
controlled request probe first failed that exact no-tools initialization.
163 affected cases now pass, pyright remains clean, and the actual stock
server repeated strict-json turns both with one callback and with no tools:
two cases passed in 6.55 seconds. these results do not prove absent hidden
native authority; the adversarial qualification below remains separate.

```sh
NATIVE_PROVIDER_RECEIPT=/private/tmp/native-provider-live-d23kkjjt/isolation.json \
  .venv/bin/python -m pytest tests/native_acceptance/test_installed_provider.py -q
```

receipt: `/private/tmp/native-provider-live-d23kkjjt/native-callback.json`.
it retains actual `gpt-6-luna` / `xhigh`, submitted native request, original
terminal evidence, usage, and one actual echo callback. this qualifies native
callback and strict-json composition. **it is not actual research**. nexus and
jarvis own separate actual search/read receipts. controlled peers do not
substitute for those receipts.

the live host remains task-owned for integrated app acceptance; root owns
its final cleanup. macos uid 501 is proven here. linux uid 10001 container
socket visibility and persistent production host topology require their own
qualification; these facts must not be inferred from the macos run.

## integrated schema defect

jarvis's actual native request exposed `invalid_json_schema`: `oneOf` is
forbidden. the original native failure was retained. pure codex request
preflight now rejects that keyword and the other established composition,
root/object closure, and required-field restrictions without normalization.
the schema-member traversal preserves a legitimate property named `oneOf`.
the initial focused probe failed eight unsupported-keyword cases and passed
the literal-property case. the repaired focused suite is green. restrictions
and remaining remote-validation limits are documented in the public contract.

## actual pending-callback cancellation

the isolated linux topology proof uses actual codex 0.160.0 as uid/gid 10001.
its first cancellation failed on the native reader's requirement that every
started dynamic item complete before an interrupted native turn. the actual
server instead ends the turn with the callback item unfinished. an exact
controlled regression reproduced this failure before the repair.

native interrupted/failed evidence now retains its correlated seal. successful
completion still requires resolved native items/requests; forbidden authority
remains rejected. closing a sealed turn with aborted pending items discards
only that session. it never marks host actions done or stops the shared server.
153 affected cases pass and source pyright remains clean. the actual repeat
held two independent callbacks: job-a cancelled with native evidence, while
job-b retained its own callback, replied, and completed strict json.
receipts: `/private/tmp/nexus-native-topology-zcDlUE/job-a.json` and `job-b.json`.
this is a real native topology/cancellation proof, not research.

## actual adversarial native authority

the first actual gpt-6-luna/xhigh adversarial callback-result injection emitted
native `subAgentActivity` despite disabled multi-agent feature flags. the
qualification stopped on its first authority event; the native parent was
interrupted, no valid terminal was accepted, and the task-owned execution
marker/read-only cwd remained untouched. read-only native history independently
identified the started child and its completed activity. subsequent successful
samples do not erase this failure.

exact installed upstream 0.160.0 source (`a956835d020762cb2b570053af06f643a11c0ecc`)
shows model metadata overrides feature defaults; `agents.enabled=false` is the
authoritative disable. contained requests now supply it, explicitly disable
inherited update-plan/sleep/image/user-message tools, and select no native
execution environments at thread creation and every prepared turn. no model
rewrite, custom server, or widened policy is used. the native containment
revision participates in catalog definition identity so saved consumers rotate.

the focused loopback check failed first on absent agents/environment overrides.
96 affected provider/model-catalog cases pass after the repair. original actual
red receipt and content-free native history projection are retained at
`/private/tmp/nexus-native-topology-zcDlUE/containment-baseline.json` and
`failed-thread-read.json`. actual source-overlay delegation and inherited-policy
repeats passed in `containment-delegation.json` and
`containment-inherited-final.json`. a separate forced native clock probe proved
that model-forced code mode and clock still bypass feature flags and ordinary
v2 item visibility. the supported startup model-catalog restriction and
raw-event qualification below address this defect; final frozen-consumer
integration remains separate.

the actual inherited-host probe supplied permissive sandbox/web/shell defaults,
one harmless configured MCP server, and a private harmless canary. its worker
mounted only the shared socket volume. no inherited MCP process/call marker,
native execution marker, cwd write, or private-canary leakage occurred. this
source-overlay linux aarch64 evidence is separate from frozen-lock consumer
installation and actual research qualification.

## qualified stock startup policy

the exact public `model_catalog_json` startup seam keeps the full vendor
`ModelsResponse`, changes tool mode to direct, and removes known native
clock/async-user selectors. the actual direct-mode callback is a raw
`function_call`; the original ordinary-mode host callback and clock were both
wrapped in separate raw `custom_tool_call` values named `exec`. receipts
`raw-callback-baseline.json`, `raw-clock-baseline.json`,
`raw-callback-restricted.json` and `raw-clock-restricted.json` preserve that
distinction. no javascript parsing or private-cache dependency is used.

five transport regressions first failed: four unqualified startup declarations
were admitted and a raw native exec lacked a typed authority event. all now
pass. an older native-version peer failed rejection first, then was rejected
before session creation. 119 affected cases pass including that version
check; source pyright reports zero errors. the full final deterministic suite
passes 998 cases, skips four explicit platform/live prerequisites and deselects
40 live-provider cases; its one warning is the intentional malformed-sdk-input
fixture. exact installed source shows the
public initialize user-agent prefix includes the binary's cargo version.

the product nexus host now materializes the provider-owned catalog and passes
its path as a startup CLI flag. public host qualification requires the exact
native version, canonical artifact basename and `sessionFlags` origin. a
materializer/source-preservation check proves only the two policy fields
changed; unknown selectors fail before writing. the actual final product-host
source-overlay adversarial run succeeded with exact gpt-6-luna/xhigh, one host
callback, strict json, no native authority events and no fixture effects.
receipt: `/private/tmp/nexus-native-topology-zcDlUE/containment-final-restricted.json`.
this is native containment evidence, **not research**, cross-uid jarvis access,
or final installed-consumer acceptance.

`uv build --wheel --out-dir /private/tmp/nexus-native-topology-zcDlUE/provider-dist`
packaged the complete exact source asset and original apache license/notice.
`provider-wheel-assets.json` records wheel and asset hashes. this startup
policy restricts the whole dedicated endpoint and freezes model inventory to
the pinned catalog. ordinary coding hosts remain separate.

## delivery composition

the native branch composes main `4d270abd48dd61240b22b9d741ce737fa611f67b`
with qualified native ancestor `e1498d8382f192ae664ae9790b682a8e8a8b0d38`.
main's terminal-control cli, naming/resume, profile usage and portable process
checks remain present. prepared-turn runtime, callback supervisor, terminal
codec and containment assets are byte-identical to the qualified ancestor.
the shared transport additionally resolves its socket alias before connecting.
existing consumers retain their exact immutable pins.

two temporary actual unix-peer checks first rejected an overlong unresolved
alias, then passed with the resolved physical socket. the owned connection
replied to its application callback exactly once; the observe-only connection
left that request unanswered. this is transport proof, not model research.
receipts: `/private/tmp/provider-native-merge-rendezvous-{red,green}.log`.

after deleting those probes, the final existing suite passes 979 cases, skips
the explicitly separate no-sdk node and deselects 40 paid live cases. the
isolated no-sdk environment passed its full suite including both wire probes;
the mandatory absent-sdk node executed separately without skipping. ruff,
pyright, dependency audit, source/wheel build, base-wheel isolation and
claude-extra wheel imports pass. logs use
`/private/tmp/provider-native-merge-*`; no paid provider was invoked by these
checks. the dependency lock remains identical to the qualified ancestor.

native qualification does not satisfy the raw-provider matrix gate in
[the pivot specification](pivot-spec.md#11-testing). the retained raw receipt
uses registry `2026-08-10.2`; it does not qualify current registry
`2026-09-25.2`. on 2026-10-03 the user explicitly waived that inherited
all-five api matrix/key gate for this native pr: "skip these i don't care".
the raw matrix remains NOT_RUN and unqualified; no raw api calls were launched.
this waiver permits this merge, preserves the future gate, and leaves native
sdk/consumer qualification required. native receipts and controlled peers
cannot replace raw evidence. the missing-row inventory and qualification
command remain in [the scoped disposition](issues/raw-provider-matrix-unqualified.md).
