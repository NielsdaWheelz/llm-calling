# native archive

`provider_runtime.agent_runtime.archive` reads retained native evidence without
resuming, subscribing, dispatching tools or opening a model turn. `ArchiveHome`
selects one provider/state root; codex additionally requires its host-owned
`CodexEndpoint` (unix or authenticated websocket). `memory_server` is the exact configured mcp server binding.

```python
home = ArchiveHome("codex", native_home, codex_endpoint)
capabilities = await archive_capabilities(home)
inventory = await archive_list(home, with_heads=True)
head = await archive_head(home, native_id)
batch = await archive_read(home, native_id, after=activation_head,
                           from_event_id=checkpoint)
```

shared memory requests the v4 capture projection.

```python
inventory = await archive_list(home, with_heads=True, capture=True)
head = await archive_head(home, native_id, capture=True)
batch = await archive_capture(home, native_id, after=activation_head,
                              from_event_id=checkpoint)
```

capture inventories mark proven workers with `worker=True` and skip their
heads. capture head and read reject an excluded worker. codex proves worker
identity with its subagent source and parent metadata, including spawns with
fork lineage. claude checks native agent and sidechain markers, including an
explicit fork-context reference. a directory can supply an inventory identity;
it cannot prove the worker marker. original evidence remains available via the
raw read surface at its original addresses.

inventory assembly handles native pages internally. `next_page` is null; a
non-null input page is invalid. both archived and ordinary codex threads are
enumerated across all declared native source kinds. claude enumerates the native
project transcripts, including child files and empty ordinary/direct-child files.
ordinary listing does not decode claude conversation content; workflow child
identity requires the first bounded complete record. head inventory requires
stable identity and attribution observations. changed membership or ambiguous
classification leaves it incomplete; failure to enumerate any required scope
raises. individual unreadable, active or moving histories retain their identity
with a content-free `head_error` and null `head_event_id`. they do not invalidate
otherwise complete membership. null head with null error proves a known empty
history. internal entries are marked and their conversation events are never
projected; workflow header inspection reads only the first record for identity.
codex listing supplies membership and archived state only. each listed identity
is checked against `thread/read(includeTurns=false)` metadata before internal
or fork classification; current native listings can omit those fields. this
adds one content-free metadata read per thread per inventory pass. the second
observation rereads that metadata; no cached classification proves stability.
multiple retained codex rollout versions can list the same native thread id.
within one archived scope, inventory includes that identity once and reads its
canonical metadata/history through `thread/read`; listed paths never select an
original. occurrence in both archived scopes fails with `source_conflict`.
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
head. a marker found in a completed prefix admits progress before an unfinished
suffix. an absent marker in an unfinished traversal is retryable. after an
exhausted miss, confirm once before parking. both traversals must be exhausted
and agree on all ordered native id/digest pairs. codex requires unchanged
conversation metadata and a non-active thread across both traversals. claude
requires the same eligible transcript with stable length and nanosecond mtime
and a complete final line. a confirmed miss returns `activation_boundary_lost`
for an activation cut or `history_changed` for a checkpoint. uncertainty never
moves the marker. `caught_up` describes delivery of the entire observed prefix
after the marker. a page limit can make it false after an exhausted traversal.

`archive_head` reads one conversation's stable head with the same rules as head
inventory. its null head proves empty; unreadable, internal, active or moving
histories raise `ArchiveError`. successful `ArchiveHead` includes `observed_at`.
consumers use this narrow read to repair a pending baseline without rereading
every other conversation. the consumer owns the new activation cut and excluded
gap; this module never resets a boundary or backfills it.

the current contract is `native-archive.v4`. the existing source identities,
digests, pending baselines and inclusive checkpoints carry through the cut.
`archive_capture` returns `occurrences`, `next`, `caught_up` and `observed_at`.
each occurrence carries `native_event_id`, `native_digest`, `capture_digest`
and ordered `records`. an empty record list accounts for an excluded occurrence
and preserves the native cursor. each capture decision uses the current
occurrence and its prefix. the new digest binds the consumed evidence and
selection before redaction. capture consumes no tool bodies, including memory
tool wrappers, downstream of the provider projection.

capture keeps speech, eligible attachments, reference-only context, revisions and
source gaps. derived utterances are `text` events with a closed `delivery`
attribute and a conversation-scoped `delivery:` identity. the attribute identifies
the carrier, kind, order and available source call, question, delivery and sender
ids. the record digest binds the evidenced speaker and literal pre-redaction
wording. a repeated native delivery with the same wording retains the same
record identity across carrier copies. a changed utterance at that identity
conflicts in admission. a separate delivery can carry the same words.

claude pairs a typed question result with the earlier ask. the delivered record
supplies the question, actual answers and human notes. previews and cancelled or
unstructured tool results do not become speech. native origin, prompt source
or turn origin can establish a non-human message independently. a task notice
needs native delivery and a corresponding earlier worker launch to carry a
report. a successful inline agent receipt can also deliver a report. queued
items and formatted markup pasted into ordinary user speech carry no worker
provenance.

codex terminal wait results can carry a report for each named, completed child.
typed async question messages retain the assistant's delivered wording. a
subsequent ordinary user answer comes through at its native address. stock
0.171 public history supplies neither
the matched blocking answer nor modern mailbox message carriers. the same
release delivers typed agent-mail items on the live root stream. the provider
helper extracts those live deliveries from explicit author, recipient and
delivery fields. external capture can miss genuine mailbox reports. both
surfaces require source evidence before interpreting a wrapper
as a report.

events carry `native_event_id`, `native_digest`, `turn_id`, `native_parent_id`,
`role`, `kind`, `text`, closed `attributes` and `occurred_at`. the digest binds the
capture projection before body suppression. unrelated native diagnostic fields
are ignored. legacy codex positional labels receive deterministic prefix-linked
identities: an earlier content rewrite changes later identities. paginated codex
uses its native identities and item timestamps. both modes use public turn
paging: one complete full turn per legacy page; up to 100 metadata turns per
paginated page, followed by forward item pages starting at 100 items. an item
page exceeding the unchanged byte or structural ingress bound emits nothing:
close that connection, halve its page size and retry the same opaque cursor on
a fresh connection. retain the reduced size for that turn. a one-item overflow
remains `event_too_large`; other errors do not reduce the page size or retry.
consume received items in forward order, stopping at the first unfinished item
before validating later bodies, including later bodies in the same received page.
stock 0.160.0 and 0.169.0 reconstruct the same legacy items before slicing turns.
the digest chain spans pages unchanged;
there is no aggregate-read fallback or rollout-file reader. each paginated turn
owns a connection, closed on completion, error or an unfinished prefix; this adds
one initialization per turn while reducing per-item roundtrips. oversize pages
add failed reads and reconnects. all reads and stable-head observations still
inspect the complete prefix: native reconstruction/normalization work and large
new backlogs remain. legacy paging costs extra native reads and server-side
history reconstruction. broader throughput work remains separate.

codex websocket ingress and pre-suppression event projections retain their
4 mib bounds. claude permits 16 mib per physical jsonl record including the
newline, and 16 mib of structural walk for a decoded record, including native
bookkeeping. both claude layouts and bookkeeping use the same raw bound. only
the pre-suppression identity projection of an ordinary claude tool result uses
the 8 mib structural budget. user attachments and memory-tool references retain
4 mib. existing 100,000-item and depth-32 bounds apply. these bounds are distinct from the library's 8 mib
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

claude's provenance tracker persists `data-flow-records` updates (a `records`
array with optional boolean `snapshot`) and `data-flow-incomplete` loss codes
(a `codes` array). array values are opaque. a root can also persist
`derived-from` with exactly one origin naming its own session through
`{"type":"claude_code_session","id":"<session>","via":"compact"}`.
exclude these complete records after applying the physical and decoded limits.
the envelope requires `type`, integer `v=1`, the transcript's `sessionId`, and
only the listed fields. child and workflow records use their native parent as
the session and require matching `agentId` on data-flow records. root records
omit `agentId`. extra fields, including `uuid` or `parentUuid`, use ordinary
identity and error rules. tracker loss leaves no archive gap. compaction
boundaries and fork references retain their own events.
the direct-child capture header scan skips these records before checking the
native sidechain or fork marker. workflow identity still requires the first
physical sidechain user record.

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
`ArchiveBoundaryUncertain` is the `unavailable` subtype for an inconclusive
missing-marker observation. callers can log it privately and retry at the same
position.

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

partial-baseline source qualification on 2026-11-15:

- 22 temporary public protocol/file cases pass on both runtime compositions.
  healthy and empty histories remain available beside missing, active, incomplete
  or moving histories. attribution/internal changes refuse complete inventory;
  contradictory identity and listing failure refuse it outright. single-head
  reads preserve the same distinctions. ordinary reads retain exact events.
  the initial run had 15 failing cases and two passing controls before repair.
- current-runtime and 0.160.0 suites pass 1,000 and 983 existing cases,
  respectively. each has one sdk-present skip; the absent-sdk case passes
  separately from its clean base wheel. lint, formatting, typing, dependency
  audit, source/wheel builds and base/extra wheel imports pass.
- only the archive contract changes. native runtime/catalogue and dependency
  pins are unchanged. these checks establish source behavior; installed capture
  and consumer migration require their separate end-to-end qualification.
- new probes are temporary and removed after qualification; existing tests stay.

adaptive item-page qualification on 2026-11-15:

- 27 temporary public websocket cases pass on both runtime compositions after
  four failing target cases and 18 passing controls. byte and structural
  overflow retries preserve the exact opaque cursor, including after an accepted
  page; rejected pages emit nothing. one-item overflow, unfinished/malformed
  suffixes, split-item checkpoints, changed digests, cursor cycles, empty
  nonterminal pages and connection cleanup preserve the closed contract.
- an isolated stock 0.169.0 server read three synthetic turns: 250 small items,
  one short item and seven one-million-character items. public resume hydrated
  this disposable fixture without a model turn. old and new reads returned
  identical ordered events and digests. item requests fell from 258 to 12;
  total requests, including initialization, fell from 261 to 23. initialization
  increased from one to nine, including five overflow reconnects. the large
  turn reduced pages from 100 through 50/25/12/6 to three items under 4 mib.
  elapsed time was 0.1405 versus 0.0899 seconds in this warm isolated fixture;
  this is not a fleet latency or load qualification.
- a read of the existing macbook synthetic verification conversation returned
  its exact 77-byte assistant marker and the same previously captured native
  digest, with an inclusive checkpoint and a complete tail. no model turn or
  profile mutation was required.
- current-runtime and compatible 0.160.0 suites pass 1,000 and 983 existing
  cases. each sdk-present skip passes separately in a clean base wheel. frozen
  lock, lint, format, typing, audit, source/wheel builds and clean base/extra
  imports pass. dependency pins, runtime catalogue, v3 wire contract and native
  event projections are unchanged.
- the new probes and synthetic fixture are removed after qualification. installed
  collector convergence remains a separate end-to-end acceptance check.
