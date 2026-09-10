# Shared Codex App Server control record

Status: accepted for the Jarvis/devserver integration

Date: 2026-09-09

Supersedes for Codex: the process, private-home, environment, and shutdown
ownership in `2026-09-07-codex-app-server-containment.md`

## Decision

The `(codex, sdk)` route attaches to a caller-configured host-owned Codex 0.153.4
App Server through WebSocket frames over a Unix socket. Provider-runtime owns
initialization, correlation, closed event classification, managed cognition
approval denial, native thread/turn routing, typed errors, and connection
cleanup. It does not start, enroll, signal, or kill the server and has no private
server fallback. The `sdk` route label remains only for stored-reference
compatibility.

Profile keys are arbitrary opaque application choices mapped by configuration
to absolute Unix-socket paths. The library does not know Jarvis's three-profile
catalog. Authentication remains the already-enrolled local ChatGPT account at
the host service; API keys and automatic account substitution remain forbidden.

The public Codex control surface lists and reads native threads, creates one
prompt-free native thread and unsubscribes the control connection before
return, submits native start-or-steer input, explicitly steers an expected turn,
and interrupts through the pinned exact-turn precheck while observing its
outcome. Full native thread and turn handles are required. Process-local state
is disposable and closing a client never closes a native thread or shared
service.

Read fetches turn metadata without items, then at most 50 newest items from
that exact turn. Omitted pages or oversized items produce explicit `Bounded`
coverage. When the pinned server reports item paging unsupported for a native
history, return the known metadata with `Bounded` and no answer; never retry
with full history or read private storage. Interrupt uses metadata only.
Local byte/structural overflow is `output_limit` for controls (a sent write
remains `Unknown`), and fatal `ProtocolDefect` for contained cognition. The
transport ingress ceiling is unchanged.

Managed cognition continues to answer every native approval with denial and
poisons the contained turn on authority activity. Generic worker control never
answers approval requests. Its create path unsubscribes so the later stock TUI
is the intended responding subscriber. Codex 0.153.4 offers no exclusive
reviewer lease or cross-client TUI-ready event; neither is claimed.

`turn/start` is natively start-or-steer and does not disclose which occurred.
Interrupt has an App Server expected-turn precheck but no core compare-and-swap
token. The public types expose these facts. They do not simulate idle admission
with a racy read or retry an ambiguous mutation.

## Proof and trade-offs

An external WebSocket-over-UDS fixture owns routine protocol tests: routing,
unsubscription, request policy, exact-handle conflicts, unknown outcomes, and
disconnect-only cleanup. The real pinned service/TUI journey is a separate live
gate and `NOT_RUN` is never a pass.

The shared service reduces private process isolation and makes account-local
clients a trust/failure/version boundary. In return it unifies manual and
programmatic native history and removes duplicate server lifecycle machinery.
Fail-closed protocol classification and exact host pinning sacrifice forward
compatibility. `websockets` is a direct core dependency because the
transport must not rely on another provider's transitive dependency.
