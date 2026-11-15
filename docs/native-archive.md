# native archive

`provider_runtime.agent_runtime.archive` reads retained native evidence without
resuming, subscribing, dispatching tools or opening a model turn. `ArchiveHome`
selects one provider/state root; codex additionally requires its host-owned
`CodexEndpoint` (unix or authenticated websocket). `memory_server` is the exact configured mcp server binding.

```python
home = ArchiveHome("codex", native_home, codex_endpoint)
capabilities = await archive_capabilities(home)
inventory = await archive_list(home, with_heads=True)
batch = await archive_read(home, native_id, after=activation_head,
                           from_event_id=checkpoint)
```

inventory assembly handles native pages internally. `next_page` is null; a
non-null input page is invalid. both archived and ordinary codex threads are
enumerated across all declared native source kinds. claude enumerates the native
project transcripts, including child files and empty ordinary/direct-child files.
ordinary listing does not decode claude conversation content; workflow child
identity requires the first bounded complete record. head inventory requires two
stable observations; active/mutable histories or changed membership leave it
incomplete. internal entries are marked and their conversation events are never
projected; workflow header inspection reads only the first record for identity.
codex listing supplies membership and archived state only. each listed identity
is checked against `thread/read(includeTurns=false)` metadata before internal
or fork classification; current native listings can omit those fields. this
adds one content-free metadata read per thread per inventory pass. the second
observation rereads that metadata; no cached classification proves stability.
multiple retained codex rollout versions can list the same native thread id.
within one archived scope, inventory includes that identity once and reads its
canonical metadata/history through `thread/read`; listed paths never select an
original. occurrence in both archived scopes still leaves inventory incomplete.
each archived scope requires two complete native views: `modelProviders=[]`
for all backends, and the omitted-filter configured-provider view. the current
native implementation gives those views different storage membership; their
canonical-id union preserves both. failure of either view refuses inventory,
and both are repeated in the second observation. this costs extra listing calls
and still relies on membership advertised by the native public API.
codex can report the same parent in `parentThreadId`, `forkedFromId` and
`source.subAgent.thread_spawn.parent_thread_id`. matching values describe one
fork; contradictory values fail. fork lineage and speaker roles are independent:
a spawned child's incoming instructions are peer evidence, while its authored
output remains assistant evidence.

claude discovery accepts these paths beneath `projects/<project>/`:

- `<session>.jsonl`;
- `<session>/subagents/agent-<id>.jsonl`;
- `<session>/subagents/workflows/<run>/agent-<id>.jsonl`.

the exact workflow sibling `journal.jsonl` is lifecycle bookkeeping, not a
conversation. other layouts fail explicitly. direct-child identity is the native
agent id without the filename's `agent-` prefix. workflow-child identity is
`workflow/<run>/<agent>`: native runs can reuse a direct child's agent id for a
distinct stream, even under the same original parent. both streams remain
addressable; no alias or content-based coalescing applies. native resume can move
a workflow directory into another session, so its first user record binds the original
parent through `sessionId`, with matching `agentId` and `isSidechain=true`.
history validates those message fields throughout; internal marking follows
that original parent. an empty/incomplete first workflow record is unavailable
until readable; contradictory complete metadata fails. there is no folder-parent
fallback or cached provenance.
ordinary `Agent` receipts retain their native direct-child link. a workflow's
aggregate result identifies a run rather than one child; no individual child
link is invented from it or inferred from the caller's workflow scope.

reads emit at most 100 complete normalized events. `next` is the final returned
event identity when another batch remains; reread that identity inclusively with
`from_event_id`. the checkpoint takes precedence over the exclusive activation
head. lost boundaries are explicit errors, never a new baseline. `caught_up`
requires exhaustion of the stable complete prefix observed when reading began.
native times do not decide eligibility.

the current codec contract is `native-archive.v2`. one implementation replaces
v1; there is no old-digest fallback or compatibility reader. corrected claude
`content` document projections change; unaffected projections stay exact. new
native layouts and record kinds retain that same public contract.
old corrected-variant checkpoints cannot be silently relabeled or reset. the
original v2 repair was qualified in isolation; it did not activate production
capture.

events carry `native_event_id`, `native_digest`, `turn_id`, `native_parent_id`,
`role`, `kind`, `text`, closed `attributes` and `occurred_at`. the digest binds the
capture projection before body suppression. unrelated native diagnostic fields
are ignored. legacy codex positional labels receive deterministic prefix-linked
identities: an earlier content rewrite changes later identities. paginated codex
uses its native identities and item timestamps. both modes use public turn
paging: one complete full turn per legacy page; up to 100 metadata turns per
paginated page, followed by one complete native item per item page. stock
0.160.0 and 0.169.0 reconstruct the same legacy items before slicing turns.
the digest chain spans pages unchanged;
there is no aggregate-read fallback or rollout-file reader. legacy paging costs
extra native reads and server-side history reconstruction; single-item pages
also increase native roundtrips. broader throughput work remains separate.

codex websocket ingress and pre-suppression event projections retain their
4 mib bounds. claude raw JSONL records permit 8 mib, including excluded native
bookkeeping, with the same 100,000-item and depth-32 structural limits. this
allows a larger raw/decoded allocation without enlarging a retained event or any
active cognition limit. these bounds are distinct from the library's 8 mib
encoded normalized-event ceiling. oversize is `event_too_large`, before any
returned prefix or checkpoint change; no record is silently truncated or skipped.
the bounds can still reject a native body whose reference-only projection would
fit, and do not promise unbounded source-history coverage.

codex inline `data:` image content is governed by those native bounds, rather
than the 4,096-byte external-reference bound. its complete original uri binds the
digest; the archive retains a short `native:<item-id>:<content-index>` reference,
no binary text, and text availability `none`. ordinary urls, paths and file ids
keep their existing bound. no image is decoded, fetched or copied.

claude documents with `source.type="content"` expose a string or an ordered list
of text/image blocks. strings stay exact; block text is joined in order by a
newline, preserving each string's whitespace. all-text sources, including an
empty source, have availability `full`; mixed text/images are `partial`; images
alone are `none`. these labels describe exposed native text, not recovery of an
original file or layout. image sources are base64 data with media type, or a url.
the entire supported source projection binds text, image bytes, source form and
ordering before binary suppression; diagnostic/cache metadata stays outside it.
unsupported nesting and malformed required fields fail explicitly. one document
has one event/reference, without nested-image rows, binary storage or ocr.

within a terminal codex turn, a recognized tool item still marked `inProgress`
ends the readable stable prefix. reads return preceding complete events with
`caught_up=false`, without emitting that item's call/result, validating its
terminal-only fields or consuming later semantic content. an in-progress turn
still waits as a whole. completion later emits the complete immutable item;
prior identities/digests stay unchanged. page cursors name emitted events only,
and inclusive checkpoint verification remains mandatory. a permanently mutable
tail can block indefinitely; no timeout skips it or fabricates a gap/completion.

reasoning is excluded. instructions, environment and native compaction are
reference-only context. configured memory calls/results retain only canonical
tool names and bounded record/range references; failed or malformed saves cannot
echo note prose. child outcomes retain known native child identities. unknown
event kinds become explicit gaps. broad native ancestry does not prove exact
original coverage; reads retain inherited copies. no current native mapping
fabricates an `inherited_origin` proof from matching prose or identifiers.

claude's `api-request-shape`, `api-request-blob`, `api-request` and `tools-used`
rows are native request/cache and tool-name bookkeeping, not new conversation
evidence. their assembled request bodies and tool inventories are excluded,
including rows without a message uuid. original tool calls/results remain. other
unrecognized rows still follow the existing strict identity and gap rules.
the same exclusion covers native session/UI state: `permission-mode`,
`session-owner`, `ai-title`, `file-history-delta`, `artifact-comment-monitor`,
`artifact-autoreact-ledger`, `frame-link`, `relocated`, `worktree-state` and
`bridge-session`. original message/tool evidence remains.
the native `prompt_snapshot` attachment requires `systemPrompt`, while
`contextRendering` is optional. present context binds the digest; absence and
explicit null stay distinct. both fields remain reference-only.

`fork-context-ref` is retained as empty-body context, with its exact closed
`agentId`, `parentSessionId`, `parentLastUuid` and `contextLength` reference in
`native_source_reference`. its stable identity and digest bind those fields,
excluding unrelated diagnostics. the child and parent identities must match the
transcript. this reference does not prove retained-original coverage and never
authorizes omission of inherited events.

secret normalization, the permanent tool-result head/tail policy and splitting
into bounded source parts belong to the capture/library layer, after the native
digest. this module does not alter canonical tool/action recovery receipts.

`CodexNativeOptions.archive_internal=True` requests a fresh marked ephemeral
thread and verifies both acknowledgments before any turn. it cannot resume or
fork. `ClaudeNativeOptions.archive_internal=True` chooses and records an internal
uuid before launching the sdk with that exact session id. native children inherit
the transcript parent's internal marker. the marker is an honest same-user
client contract, not protection against hostile filesystem mutation.

`ArchiveError` carries only a closed, content-free code. `ArchiveMissing` is the
`unavailable` subtype reserved for direct proof of absence; generic network or
native-method failures do not authorize an absence claim.

temporary qualification on 2026-10-09:

- installed stock codex 0.160.0: real legacy/paginated model turns, stable repeat
  reads, activation/checkpoint semantics, archived heads, restart persistence and
  post-turn absence of marked ephemeral cognition passed. the exact direct
  missing-thread response was distinct from successful reads of existing archived
  and restarted/unloaded threads.
- installed claude 2.1.289: actual persisted error/fork transcripts and recognized
  supplied context mapped successfully. successful live claude-work inference
  was waived by the owner for this run after expired oauth (401); no reauth or
  production qualification is implied.
- temporary public transcript integration verified reference-only bodies,
  malformed memory-save suppression, reasoning exclusion, additive metadata,
  consumed-field rejection, branch revisions, empty/internal inventories and
  incomplete tails. no private history was used as fixtures.
- production installation and fleet admission were not performed.

source repair checks on 2026-10-13:

- the temporary codec/history probes were red before repair (48 failures, four
  control cases passing), then 53 cases passed after repair/refactor, including
  legacy/paginated unfinished barriers and malformed later suffixes. 35 existing
  public/session/type cases also passed. unchanged document/image projections
  matched their pre-repair digest fixtures.
- changed-file format/lint and typing passed. probes use synthetic native peers
  and no account, inference, database or network. end-to-end collector/storage,
  live persistence and final frozen package qualification remain separate.
- all new repair probes are temporary and removed after final qualification;
  existing tests remain. no production activation or source commit is implied.

0.160.0 maintenance source qualification (`97c0475`) on 2026-11-15:

- 54 temporary public protocol/file cases pass, including unchanged prior event
  identities/digests, relocated workflow provenance, run-qualified child streams,
  exact context references, complementary listing membership, either-view failure,
  cursor-cycle rejection and each distinct ingress bound.
  the existing suite passes 983 cases; one sdk-absence case skips with the sdk
  installed, then passes separately against the clean base wheel. 40 paid/live
  provider tests are excluded; this repair opens no model turn.
- lint, formatting, typing, lock validation, builds and clean base/extra wheel
  imports pass. the inherited frozen environment had 22 advisories. six exact
  lock updates remove them: certifi 2026.10.19, filelock 4.2.4, msgpack 1.2.3,
  opentelemetry-api 1.45.2, pyasn1 0.6.5 and mcp 1.30.0. the optional claude sdk
  stays 0.2.130 with its mcp `<2` constraint; no runtime, model or catalogue
  upgrade is included. uv 0.13.4 regenerates lock format revision 5.
- read-only native evidence found one 4,268,702-byte excluded claude request
  snapshot. the failing codex legacy aggregate was 6,039,580 bytes over 23 turns;
  complete-turn pages peaked at 2,551,993 bytes and matched its exact ordered
  consumed projection. these observations justify the separate raw-record bound
  and turn paging. a later 100-item page was 8,821,663 bytes; its largest complete
  item was 1,155,433 bytes. single-item pages peaked at 1,155,936 bytes and matched
  the same ordered entries. no websocket or event bound was enlarged; extra
  native reads/server work are explicit above.
- native 0.169.0 listing retains multiple rollout versions for one thread id.
  canonical read metadata/history remains authoritative. explicit
  `modelProviders=[]` switched native listing to its database path and omitted
  three known identities in the observed homes; the omitted-filter view alone
  restricts backends. requiring both views preserves their union without a
  fallback or direct database/file discovery. inaccessible native lineage still
  prevents complete history qualification.
- production admission and missing native lineage are separate from codec
  support. a native missing-source-rollout error remains unavailable; no empty
  history, new baseline, inferred coverage or silent omission replaces it.

the current-runtime forward port preserves its explicit `CodexEndpoint` contract
and stock 0.165.0 catalogue. its existing suite passes 1,000 cases, with one
sdk-present skip and 61 paid/live exclusions; the same 54 archive cases pass with
only the endpoint fixture adapted. lint, format, typing, audit and builds pass.
jarvis's separately pinned 0.160.0 maintenance release does not adopt this runtime.
