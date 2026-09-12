# Shared Codex App Server control record

Status: accepted for the Jarvis/devserver integration

Date: 2026-09-09

Supersedes for Codex: the process, private-home, environment, and shutdown
ownership and native-version admission in `2026-09-07-codex-app-server-containment.md` and
`2026-09-09-codex-app-server-integration.md`. Preserve the latter's generation
catalog, authenticated selection, endpoint routing, and containment contracts.

## Decision

The `(codex, sdk)` route attaches to a caller-configured host-owned Codex
App Server through WebSocket frames over a Unix socket. Provider-runtime owns
initialization, correlation, closed event classification, managed cognition
approval denial, native thread/turn routing, typed errors, and connection
cleanup. It does not start, enroll, signal, or kill the server and has no private
server fallback. The `sdk` route label remains only for stored-reference
compatibility.

The host installs the latest stable native release. Initialize metadata is
validated as protocol data, but `userAgent` is diagnostic identification, not
a version or capability gate. No native version constant, semver parser, or
fallback remains. Protocol shape, response correlation, subscription routing,
and managed cognition containment remain strict.

Profile keys are arbitrary opaque application choices mapped by configuration
to absolute Unix-socket paths. The library does not know Jarvis's three-profile
catalog. Authentication remains the already-enrolled local ChatGPT account at
the host service; API keys and automatic account substitution remain forbidden.

Catalog discovery uses this same shared transport and retains exact tagged
session requests, model/reasoning selection, revision and row-fingerprint
checks. Server process environment, including TMPDIR, is host-owned. Remove
the private-child `child_tmpdir` setting; retain the native per-thread sandbox
exclusion controls. No environment substitution is simulated through a client.

The public Codex control surface lists and reads native threads, creates one
prompt-free native thread and unsubscribes the control connection before
return, submits native start-or-steer input, explicitly steers an expected turn,
and interrupts through the native exact-turn precheck while observing its
outcome. Full native thread and turn handles are required. Process-local state
is disposable and closing a client never closes a native thread or shared
service.

Read fetches turn metadata without items, then at most 50 newest items from
that exact turn. Omitted pages or oversized items produce explicit `Bounded`
coverage. When the server reports item paging unsupported for a native
history, return the known metadata with `Bounded` and no answer; never retry
with full history or read private storage. Interrupt uses metadata only.
Local byte/structural overflow is `output_limit` for controls (a sent write
remains `Unknown`), and fatal `ProtocolDefect` for contained cognition. The
transport ingress ceiling is unchanged.

Managed cognition continues to answer every native approval with denial and
poisons the contained turn on authority activity. Generic worker control never
answers approval requests. Its create path unsubscribes so the later stock TUI
is the intended responding subscriber. The previously inspected Codex 0.153.4
offered no exclusive reviewer lease or cross-client TUI-ready event; neither
is claimed for a newer release without qualification.

`turn/start` is natively start-or-steer and does not disclose which occurred.
Interrupt has an App Server expected-turn precheck but no core compare-and-swap
token. The public types expose these facts. They do not simulate idle admission
with a racy read or retry an ambiguous mutation.

## Proof and trade-offs

An external WebSocket-over-UDS fixture owns routine protocol tests: routing,
unsubscription, request policy, exact-handle conflicts, unknown outcomes, and
disconnect-only cleanup. The real installed service/TUI journey is a separate live
gate and `NOT_RUN` is never a pass.

The shared service reduces private process isolation and makes account-local
clients a trust/failure/version boundary. In return it unifies manual and
programmatic native history and removes duplicate server lifecycle machinery.
Latest-stable deployment avoids frozen native releases, but does not promise
forward compatibility or transfer safety authority to a version string. An
upstream protocol change may still fail closed and need an explicit adapter
change plus qualification. Historical live evidence certifies only its recorded
build; routine alternate-version fixtures do not certify an upgraded binary.
`websockets` is a direct core dependency because the
transport must not rely on another provider's transitive dependency.
