# native archive

`provider_runtime.agent_runtime.archive` reads retained native evidence without
resuming, subscribing, dispatching tools or opening a model turn. `ArchiveHome`
selects one provider/state root; codex additionally requires its host-owned unix
app-server endpoint. `memory_server` is the exact configured mcp server binding.

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
