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

temporary acceptance cases are in `tests/native_acceptance/`; they remain
until integrated final acceptance authorizes deletion.

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

final-tree deterministic conformance: 971 cases pass, three explicitly skip,
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
prepared native callback and strict-json turn then passed. the final-tree repeat
passed again in 4.75 seconds after reply/cancellation refinements.

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
