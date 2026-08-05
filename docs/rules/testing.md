# Testing Standards

> Normative target-state document for tests. This describes the desired steady
> state, not necessarily the current state. All new tests must conform.
> Existing tests should be migrated when touched or when a planned cleanup
> change explicitly owns them.

## 1. Philosophy

Tests exist to verify behavior, not implementation. Tests are the primary verification gate — specs give direction, but passing tests prove the code works.

- If a test breaks when internals are refactored but observable behavior is unchanged, the test is wrong.
- Tests are contracts: they document what the system promises to users and operators, not how the code is arranged internally.
- A passing test suite should mean "the product works." A failing test should mean "something meaningful regressed."
- Prefer fewer, higher-confidence tests over many shallow tests. One real user-flow E2E test is worth many brittle mocked tests.
- Use red/green TDD: write failing tests from acceptance criteria first (red), then write code to make them pass (green). This prevents both non-functional code and unnecessary code.
- When starting a session, run the existing test suite first. This reveals project scope, surfaces pre-existing failures, and establishes a testing mindset.

## 2. Scope and Definitions

This document covers backend, frontend, platform shell, and end-to-end tests.

Definitions used throughout this document:

- `behavior`: observable HTTP responses, persisted state visible through supported APIs, rendered UI state, routing/navigation outcomes, and user-visible error handling.
- `implementation`: internal helper calls, module wiring, service composition,
  query shape, component child composition details, or framework internals.
- `internal boundary`: code owned by the repo.
- `external boundary`: third-party systems or APIs outside the repo.
- `real stack`: real running app services and dependencies used in local/CI,
  not mocked equivalents.

## 3. Command Surface

The repository command list is the canonical source for repo-level commands.
This document only names the stable gates and how they are grouped.

- The static-analysis gate covers formatting, linting, and type checks.
- The dependency/security gate audits dependencies and known vulnerabilities.
- Unit, integration, component, E2E, and platform-shell test gates should be
  independently runnable.
- Verification gates should compose lower-level gates into routine and full
  verification commands.
- Type-check gates should keep their enforced baseline honest; expand them when
  newly touched modules are clean.
- Workflow or CI configuration checks should have a stable gate.
- E2E should stay a single local command; CI may shard it across jobs.

## 4. Testing Trophy

```text
           /  E2E                 \      <- real client, real server,
          /------------------------\         real data/services
         /   Component/Interface    \     <- UI or adapter behavior
        /----------------------------\
       /  Unit                       \    <- pure logic, no I/O
      /------------------------------\
     /        Static Analysis          \  <- format, lint, type checks
    /----------------------------------\
```

Note: the trophy visual is user-flow weighted. Backend or service integration
is still a required, separate tier and is not collapsed into E2E.

For apps where routing, rendering, auth, or transport behavior emerges only in
the running stack, E2E should be the largest test layer because:

1. Integration tests around framework boundaries often require mocks, which undermines confidence.
2. Modern E2E tooling is fast enough to cover high-value user journeys directly.
3. Real-stack E2E catches routing, auth forwarding, streaming, rendering, and proxy bugs other layers miss.

Service integration tests remain a separate tier because they validate API and
database behavior faster than E2E and provide better failure localization.

## 5. Test Tiers

### Tier 0: Static Analysis

Rules:

- Runs on every PR and in local verification commands.
- Treat static-analysis failures as real failures (no `|| true`).
- Include formatting, linting, type checks, dependency checks, and generated-file
  checks where they apply.
- Keep type-check surfaces honest: a small passing baseline is better than a
  broad gate full of suppressions.

### Tier 1: Unit Tests

What belongs here:

- Pure functions (text processing, offsets, normalization, crypto wrappers, serializers)
- Input and data validation
- Configuration parsing

What does not belong here:

- Database access
- Network calls
- Component rendering
- Anything that requires mocks to execute

Rules:

- No database fixtures, no file I/O, no network I/O
- No mocks by default; move the test up a tier if behavior depends on external interactions
- Keep tests fast and deterministic

### Tier 2: Component Tests

What belongs here:

- Component rendering and interaction
- Real client API usage, such as browser, mobile, terminal, or platform APIs
- Interaction-heavy UI (highlighting, selection, drag/drop, toggles)
- UI behavior and accessibility states

What does not belong here:

- Page-level flows that require real app routing and API calls (E2E)
- Tests that mock internal modules

Rules:

- Run against the real client runtime when framework or platform behavior
  matters.
- Prefer semantic queries and user-level interactions over implementation
  selectors.
- No global framework-navigation mocks in shared setup.

### Tier 3: Integration Tests

What belongs here:

- Database-backed service workflows
- Multi-step backend behavior that is faster to validate at API level than through UI

Rules:

- Use a real test database for database-backed behavior.
- Assert through API responses or public service surfaces, not raw table
  inspection, except documented schema-level exceptions.
- Mock only external boundaries (Section 7)
- Use domain- or repository-backed factories and fixtures, not raw database
  inserts in factories.

### Tier 4: E2E Tests

What belongs here:

- End-to-end user journeys
- Auth and session behavior
- Proxy behavior and token forwarding
- Streaming UX and cross-page workflows
- Multi-user permission and sharing flows

Rules:

- Run against real services.
- No mock API servers or fetch-level shortcuts.
- Use authenticated-session reuse where the E2E tool supports it.
- Seed data through app APIs or dedicated seed scripts, not ad hoc database
  writes from client tests.
- Tests must be independent and parallelizable
- CI may shard E2E to keep wall time down; local E2E remains a single command.

### Platform Shell Tests

Use platform-shell tests when a thin native or platform wrapper hosts the main
app experience.

What belongs here:

- App launch into the configured product URL or runtime entrypoint.
- Verified deep-link or app-link re-entry.
- Same-origin navigation stays inside the shell; off-origin navigation leaves to
  the platform browser or external handler.
- Back navigation and platform file-picker handoff.
- Debug/test build validity.
- Signed release build validity when release secrets are available.

Rules:

- Test the real platform activity, shell, or wrapper behavior.
- Prefer platform instrumentation tests. Use local unit tests only for already
  pure code with no platform dependency.
- Do not add wrappers, bridges, or fake navigators just to make shell code
  unit-testable.
- Mock only external boundaries the app does not own

## 6. Assertion Standards

### Service Tests: Assert Through the API

Prefer API-response assertions over raw database inspection when testing API
behavior.

```text
WRONG: asserting API behavior by querying tables directly
  response = POST /resources { name: "Test" }
  assert response.status == 201
  row = query_database("membership role for created resource")
  assert row.role == "admin"

RIGHT: assert the API contract
  response = POST /resources { name: "Test" }
  assert response.status == 201
  assert response.body.data.role == "admin"

ALSO RIGHT: follow-up API read when create response omits the field
  members = GET /resources/{resource_id}/members
  assert members contains { role: "admin" }
```

Exceptions:

- Migration and schema-level tests.
- Database-level constraint tests where the API intentionally hides the internal
  detail. Prefer repository queries over raw database text where feasible.

### Assertion Messages Should Be Rich

Include extra context in assertion messages. When a test fails, the failure message is often the only feedback an agent or developer sees. Rich messages enable faster self-correction.

```text
WEAK: no context in failure
  assert response.status == 201

BETTER: context in failure message
  assert response.status == 201
    message: "expected 201; include actual status and response body"

BEST: structured context for complex assertions
  assert data.role == "owner"
    message:
      expected_role: "owner"
      actual_role: data.role
      resource_id: resource_id
      user_id: user_id
      response: data
```

This is especially valuable for integration and E2E tests where failures can be indirect and hard to diagnose.

### UI: Assert User-Visible Behavior

```text
WRONG: implementation-coupled callback assertions
  fake_callback = make_fake_callback()
  render navigation with fake_callback
  click "Collapse"
  assert fake_callback was called

RIGHT: behavior assertion
  render navigation
  click "Collapse navigation"
  assert the navigation region is collapsed
```

Guidelines:

- Prefer role/text/label queries over selectors and test IDs when practical
- Assert visible state, navigation outcome, or response behavior
- Avoid testing library/framework internals

## 7. Mocking Policy

### Allowed Mocks (External Boundaries Only)

| Boundary | Tool / Pattern | Why |
|---|---|---|
| External model or provider APIs | HTTP-level mock at the external boundary | Third-party cost, nondeterminism, rate limits |
| External auth verification boundary | test verifier / fake verifier at boundary | Third-party dependency boundary |
| Async job dispatch boundary | mock queue enqueue helper | Verifies dispatch intent without running the worker inline |
| External object storage | mock storage client | Third-party service dependency |

### Disallowed Mocks (Internal Boundaries)

| Thing | Why |
|---|---|
| Internal service modules | Couples tests to implementation; refactors break tests without behavior regressions |
| Database sessions/queries | The database is part of the behavior contract for integration tests |
| Global framework-navigation mock | Hides routing behavior and contaminates unrelated tests |
| Module mocks for internal API helpers or proxy modules | Bypasses the codepath under test |
| Internal UI components | Stops testing real composition and behavior |
| The app's proxy or backend-for-frontend layer | Critical integration layer; must be tested real |

### Network Mock Policy

Network-level mocking is better than module-level mocking because it intercepts
at the boundary, but it is still a mock. Use it only when the boundary is
external or the test tier explicitly owns network simulation.

Instead:

- Component tests validate rendering and interaction without network-dependent page flows
- Service integration tests hit real database/services and mock only external HTTP boundaries
- E2E tests hit the real running stack

### Exceptions (Temporary and Explicit)

Short-term exceptions are allowed only when migration work is in progress and the test would otherwise be deleted or blocked. Requirements:

- The exception must be documented in the PR description or a code comment at the mock site
- The exception must be time-bounded (for example, "remove in this PR before merge" or "remove in next planned cleanup PR")
- The exception must not be hidden in global/shared test setup
- If an external boundary is only reachable through an internal accessor in current code, a temporary patch at that seam may be used during migration, but the test must still assert behavior and the exception must be called out explicitly
- Every exception entry must name the intended replacement layer/test (or the exact follow-up test to be added)

## 8. Data Setup and Fixtures

### Backend: Factories Use Domain Or Repository Models

Prefer domain- or repository-backed factory helpers over raw database inserts.

```text
WRONG: raw database factory insert
  create_test_item(owner_id, title):
    execute raw insert into item storage

RIGHT: repository-backed factory insert
  create_test_item(owner_id, title):
    item = build domain item with owner_id and title
    save item through repository or domain helper
    return item
```

Reasons:

- Uses the same validation/default paths as production code
- Reduces schema-coupled test breakage
- Produces clearer fixture code

### Fixture Placement

- Shared fixtures used in multiple files belong in the repository's shared test
  fixture area.
- Data creation helpers belong in a dedicated factory/helper area.
- Composite fixtures and scenario setup belong in a dedicated fixture area.
- Test files should contain only test-local fixtures unique to that file.

Rule:

- If the same fixture appears in multiple files, centralize it.

### E2E Seeding

- Seed through app APIs or dedicated E2E seed scripts
- Avoid direct database writes from E2E client tests
- Prefer deterministic seed inputs and idempotent setup behavior
- Prefer centralized seeding/bootstrap so all invocation paths share identical
  setup guarantees.
- Central setup may load repo environment and runtime port files to mirror the
  normal command behavior when tests are run through another entrypoint.

### E2E Determinism and Pane-Aware Assertions

- Normalize persisted state before asserting initial UI when prior state can
  affect the screen.
- E2E tests must assert user-visible contracts without implementation mocks.
- Prefer explicit action-menu interactions over styling-dependent selectors.
- Keep single-flow E2E tests focused on one behavior; use API setup for prerequisites already covered by separate UI stress/interaction tests

## 9. Test Organization

### Backend Layout

```text
backend-tests/
|- shared fixtures
|- factories
|- helpers
|- support/
|  `- ...
|- utilities/
|  `- ...
|- tests
```

Expectations:

- Mark every test file with the appropriate execution marker or tag when the
  test framework supports it.
- Keep migration/schema tests separate.

### Frontend Layout

```text
frontend/
|- test config                # unit and component test projects
|- test setup                 # minimal shared setup
|- source/
|  |- lib/                    # pure unit tests live near pure modules
|  |- components/             # component tests
|  `- app/                    # avoid page-level unit tests unless truly pure
```

Guidance:

- Pure utility tests stay near source files
- Component tests can live in one component-test directory or near components if
  consistent.
- Page-level behavior belongs in E2E unless the page unit is truly pure and isolated

### E2E Layout

```text
e2e-tests/
|- test config
|- global setup
|- runtime config
|- seed scripts
`- tests/
   |- auth setup
   `- specs
```

## 10. Markers and Naming

### Test Markers

Expected marker or tag set:

- `unit`: pure logic tests, no database or network
- `integration`: database/API-backed tests
- `slow`: tests that are materially slower than normal local feedback loops
- service-specific markers for local external service dependencies

Rules:

- No unmarked backend or service tests when markers are the routing mechanism.
- Markers describe execution requirements, not implementation details.

### Naming Conventions

Backend/service examples:

- `create resource with valid name returns created response`
- `search with no results returns empty list`

Frontend/component examples:

- `shows error message when API returns server error`
- `navigates to detail view when card is clicked`

E2E examples:

- `user creates resource and it appears in navigation`
- `user edits content and the change persists after reload`

## 11. CI and Local Commands

Target local commands after migration:

- Static check command: static checks and format checks only.
- Unit test command: fast unit tests only; no database, no component runtime,
  no E2E.
- Standard test command: non-E2E automated tests, including integration and
  component tests.
- E2E command: explicit default real-stack E2E run, used before merge and in CI.
- Live-provider command: live external-provider acceptance gate.
- Routine verify command: static checks, build, and non-E2E tests for routine
  development.
- Full verify command: routine verification plus live-provider and E2E gates.
- Platform-shell test command: instrumentation tests; requires a connected
  device, emulator, or platform runtime when applicable.
- Platform-shell verify command: static and build verification only; should not
  require a connected device.
- Release verify command: signed release artifact verification; requires release
  signing environment variables or secrets.
- Interactive E2E command: interactive E2E UI mode when the tool supports it.
- Policy-specific runtime assertion commands may exist for strict runtime
  policies such as CSP, permissions, or sandboxing.

Target CI shape:

1. Run static checks and type checks
2. Run backend and frontend tests in parallel where independent
3. Run E2E after lower layers pass
4. Upload E2E artifacts on failure (and optionally always)

## 12. What Not to Test

- Library internals
- Framework behavior already guaranteed by the framework (unless you are testing your integration with it)
- One-time migration audit assertions that are no longer part of ongoing product behavior
- Configuration introspection via implementation details (prefer runtime behavior assertions)

## 13. Migration Rules for Existing Tests

When modifying existing tests during cleanup:

1. Prefer replacement over patching brittle mocks
2. If deleting a test, identify the replacement layer (`unit`, `component`, `integration`, or `E2E`)
3. Do not add new global test mocks
4. Do not add new tests that depend on fake client runtimes for behavior that
   requires the real runtime
5. If a temporary exception is required, document it in the PR description and remove it before merge when feasible
