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
project transcripts, including child files and empty files. ordinary listing
does not need to decode claude conversation bodies. head inventory requires two
stable observations; active/mutable histories or changed membership leave it
incomplete. internal entries are marked and their histories are never read.

reads emit at most 100 complete normalized events. `next` is the final returned
event identity when another batch remains; reread that identity inclusively with
`from_event_id`. the checkpoint takes precedence over the exclusive activation
head. lost boundaries are explicit errors, never a new baseline. `caught_up`
requires exhaustion of the stable complete prefix observed when reading began.
native times do not decide eligibility.

the current codec contract is `native-archive.v2`. one implementation replaces
v1; there is no old-digest fallback or compatibility reader. only corrected
claude `content` document projections change; unaffected projections stay exact.
old corrected-variant checkpoints cannot be silently relabeled or reset. the
repair delivery is source/isolated qualification only; production capture has
not been activated.

events carry `native_event_id`, `native_digest`, `turn_id`, `native_parent_id`,
`role`, `kind`, `text`, closed `attributes` and `occurred_at`. the digest binds the
capture projection before body suppression. unrelated native diagnostic fields
are ignored. legacy codex positional labels receive deterministic prefix-linked
identities: an earlier content rewrite changes later identities. paginated codex
uses its native identities and item timestamps. neither mode reads rollout files.
the existing 4 mib raw native line/websocket and structural ingress bounds remain
separate from the library's 8 mib encoded normalized-event ceiling. raw oversize
is `event_too_large`, before any returned prefix or checkpoint change. this can
reject a large native body whose eventual reference-only projection would fit;
the reader never silently truncates or skips it.

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
