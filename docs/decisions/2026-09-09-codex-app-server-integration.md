# Codex App Server containment on the current provider contract

Status: implemented; live qualification remains separate.

The later [shared-service decision](2026-09-09-shared-codex-app-server.md)
supersedes this record's private-process, environment, SDK-package, stdio
ownership, and native-version admission. Its generation/catalog, typed selection,
and containment requirements remain binding. Verification below describes this
historical revision only.

## Problem

The shared kernel and Jarvis pinned containment commit
`4ddced3bb5487ce988858c4c6d45d2e5ee0acad9`, based on
`2cfed97ee5b9b8eb11103b0575eb7f29de00a0bd`. Provider main at
`16499e1c4783a890063f51bb16b64d48b0cdbe3b` added exact source catalogs,
explicit session choices and endpoints, and corrected initialize metadata;
it did not contain the native-authority fix. Selecting either line alone
would lose a required behavior.

## Decision

Use the documented App Server stdio transport for Codex execution, session
inspection, and authenticated model discovery. Retain the public tagged
catalog request/choice types, exact catalog validation, and endpoint APIs.
`openai-codex` supplies only its option vocabulary/version and the matched
bundled executable. Delete the superseded Codex SDK launcher and its tests.
Keep the Claude SDK lane unchanged.

Completed text, reasoning, usage, and documented metadata are observations.
Native/custom/dynamic tools, MCP calls, approvals, and hooks have first-class
normalized authority events. Unknown methods/items, invalid lifecycle identity,
and malformed JSON/RPC are protocol defects. Explicitly configured native/MCP
tools remain available; the exact-version `builtin_tools="disabled"` posture
rejects any authority before accepting a terminal and invalidates its session.
There is no second execution path or automatic fallback.

All RPC responses stay correlated to an owned pending request through complete
validation. Malformed errors and RPC timeouts fail pending work and terminate
the child. The pending event queue uses the existing 100,000-event/64-MiB limits;
overflow is a fatal protocol result, never dropped output. Previously queued
authority remains observable, while queued completion is suppressed once a
fatal transport result is known. This bounds a provider that continues emitting
while its application waits for acknowledged output persistence.

Model discovery reads all pages over that same transport. The initialize
parser retains current main's `userAgent` handling when optional `serverInfo`
is absent. The internal `read_codex_model_catalog` helper no longer takes a
Python SDK response-model argument. Public catalog rows, hashes, and session
choices retain their current contracts.

The optional `llm-tools` projection is pinned to
`9e6d155f3b64f03495911435b7cae8b8d131f9a2` and qualified with its existing
behavioral adapter tests.

## Trade-offs

- Provider protocol drift reduces availability by failing explicitly. Keeping
  an opaque SDK execution fallback would lose containment and is rejected.
- A paused consumer can exhaust a bounded transport queue. The connection stops
  and its outcome requires host recovery; memory cannot grow without bound.
- Denying server callbacks also denies credential-refresh and attestation
  authority. Applications enroll credentials outside agent execution.
- Read-only/offline containment is not a host-confidentiality sandbox. Operating
  system isolation remains the owning boundary for that requirement.
- Live/paid certification is distinct from deterministic fixture qualification.

## Verification

On current main, three target cases (native command, custom exec, and unknown
native item) failed because no `ProtocolDefect` was raised. The same gate
passed after integration. Those temporary duplicate tests were removed after
the richer lifecycle tests covered the same acceptance boundaries.

Adversarial review added five behavior cases for malformed error correlation,
RPC timeout, and paused-consumer byte/count overflow. All five first failed
with the expected missing behavior and then passed. The retained suite includes
those cases, complete native lifecycle classification, explicit MCP behavior,
usage replay, exact catalogs, and a real subprocess discovery/initialize test
that proves ambient environment exclusion. Two tests that mirrored private
classification constants and the obsolete SDK replay-surface test were removed.
Notification normalization consumes the owned typed wire values directly; it no
longer accepts a second SDK payload shape.

- Darwin baseline: `uv run pytest -q` produced 981 passed, 3 failed,
  2 skipped, 59 live cases deselected.
- Integrated Darwin before the final typed-notification simplification: `uv run pytest -q` produced 1049 passed, 2 failed,
  2 skipped, 60 live cases deselected. The remaining failures are the unchanged
  Claude process-group assumption and Linux-only `/proc/self/fd` proof.
  The obsolete Codex launcher and its failing test were deleted together.
- Linux full gate in an isolated ext4 container copy, using
  `ghcr.io/astral-sh/uv:python3.12-bookworm`, locked all-extras/all-groups sync,
  then `uv run pytest -q`: 1050 passed, 2 optional-SDK absence cases skipped,
  60 live cases deselected. Both Darwin failures pass on Linux.
- The built base wheel was installed in a fresh environment. Neither optional
  SDK nor the deleted Codex launcher was importable. Both actual
  `SdkUnavailable` absence tests ran and passed: 2 passed, no skips.
- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`,
  and `git diff --check`: passed.
- `uv build --no-sources`: passed.
- `uv run pip-audit`: no known vulnerabilities; local provider-runtime and
  git-sourced llm-tools are not available for PyPI vulnerability auditing.
- Live provider calls and the dedicated containment probe: NOT_RUN.

The dedicated live containment test uses public authenticated catalog selection
and the current tagged session request, with no hidden model/reasoning default.
