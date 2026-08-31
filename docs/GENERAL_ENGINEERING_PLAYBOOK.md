# General Engineering Playbook
## AI-Assisted Software Development for Production Systems

**Scope:** This playbook captures the durable engineering practices extracted from
building a production-grade agentic system with AI assistance. It covers the full
development lifecycle — from pre-development specification through architecture,
testing, observability, and release — with emphasis on the failure modes that appear
regardless of domain or technology stack.

**Intended audience:** Engineers, tech leads, and teams using AI coding assistants
(Claude, Copilot, Cursor, etc.) to build non-trivial backend or service-oriented
systems. The practices here apply to any language or framework.

**Companion document:** `ENGINEERING_PLAYBOOK.md` in this repository applies these
principles specifically to LLM agentic / RAG systems. Read this document first; the
companion adds domain-specific depth.

---

**Revision, 2026-08-26 — additions from the `hippo-pipeline` project.** Every change below
came from a defect or a near-miss in a real build, not from theory. If you keep a canonical
copy of this playbook elsewhere, these are the sections to merge:

| Section | Change | What prompted it |
|---|---|---|
| §1.3 | ADR granularity is about blast radius, not importance; reject with a measurement | 18 ADRs where 15 would have been better; a feature killed by data rather than opinion |
| **§1.7 (new)** | Decision order is itself a decision; never reserve ADR numbers | A dominating question answered second, which reversed an already-made choice when corrected |
| §2.5 | For computed outputs, derive expected values with an independent program | The session split is the weaker guard for aggregations |
| AP-01 | The parse-time variant: the decoder destroys the value before your code runs | `Decimal(str(x))` after a JSON float parse |
| AP-11 | Prevention over detection: do not build an abstraction before its caller exists | A registry that would have had only tests as callers |
| AP-13 | The maintainer's convenience path diverges from the documented one | The documented setup command was unexercised until the final check |
| **AP-19 (new)** | The document that asserts what the code does not do | A decision record promising behaviour that was never implemented, undetected for four sessions |
| **AP-20 (new)** | Quarantine that conflates a defect with an exclusion | A 15% "reject rate" for a source whose defect rate was 0.011% |
| §5.1, §5.2 | `jq` is not universal; a fifth hook pattern that *blocks* a write | A hook that fails on a missing tool trains you to ignore hooks |

---

## Table of Contents

1. [Project Foundation & Specification](#1-project-foundation--specification)
2. [AI-Assisted Development Model](#2-ai-assisted-development-model)
3. [Architecture & Design Patterns](#3-architecture--design-patterns)
4. [Production Readiness](#4-production-readiness)
5. [Claude Code Hooks Reference](#5-claude-code-hooks-reference)

---

## 1. Project Foundation & Specification

### 1.1 The Cost of Starting Without a Spec

The most common failure mode in software projects is beginning implementation before
the system's contracts are defined. The problem compounds in AI-assisted development:
an AI coding agent will produce confident, coherent code that satisfies an ambiguous
specification in the same way a junior developer would — by making assumptions that
seem reasonable locally but conflict globally.

By the time the contradictions surface, you have written tests, docs, and follow-on
features against an incorrect foundation. Rework cost is not linear; it compounds
with every layer of code written on top of a wrong assumption.

The practice that prevents this is **Spec-Driven Development (SDD)**: every structural
decision is made, written down, and conflict-checked against prior decisions *before*
the first line of implementation is written. The artifact that enforces this is a
**Project Memory Asset (PMA)** — a single, living Markdown document that is the
authoritative source of truth for the project.

**What the PMA contains:**

- The project charter: what problem this solves and for whom
- All Architecture Decision Records (ADRs) — including superseded ones, never deleted
- The canonical data contracts (API schemas, database table schemas, shared state)
- A feature log, keyed to per-feature spec files
- Open Questions with resolution status and the ADR that resolved each
- A retrospective (added as the project matures)

Every session with an AI coding assistant begins by pointing it at the PMA. Every
structural change — new ADR, updated schema, modified API contract — is written into
the PMA in the same response as the code change. No feature is done until the PMA
reflects it.

**The rule that makes this work:** the PMA is append-only for decisions. Superseded
ADRs are never deleted; they are marked `Status: Superseded` with a pointer to the
superseding ADR. The *reason* a decision was made is often more valuable than the
decision itself, and losing it means re-discovering the failure mode later.

### 1.2 Technology Selection: The Requirements-First Method

Technology choices made on familiarity or trend carry hidden cost: you discover the
technology's limitations only after the implementation is committed to it. The
requirements-first method surfaces those limitations before any code exists.

**The process:**

1. Enumerate the concrete control-flow or behavior requirements that cannot be
   negotiated away. Write them down, not as features but as system behaviors: "the
   system must be able to partially complete an operation, pause, and resume later
   with no data loss."
2. For each candidate technology, explicitly map each requirement to either
   "supported natively," "requires workaround," or "cannot be satisfied."
3. Document the mapping as an ADR. The ADR captures not just the choice but the
   requirements that drove it — making future technology re-evaluations traceable.

A table is the right format for this comparison:

| Requirement | Option A | Option B | Notes |
|---|---|---|---|
| Pause/resume mid-workflow | Requires external queue | Native durable state | Queue adds ops complexity |
| Sub-millisecond reads | Yes | No (disk-backed) | Critical if on hot path |
| Schema migration without downtime | Supported | Not supported | Blocks zero-downtime deploys |

Never select a technology to avoid one column of "no" and then route around that
"no" in code. If a requirement requires a workaround, either reconsider the
requirement or select a different technology. Workarounds accumulate; they become
the technical debt that blocks the next migration.

### 1.3 The ADR Process

Every structural decision is recorded as an **Architecture Decision Record** before
any implementation that depends on it:

```
### ADR-NNN: Title
- Context:      What problem or gap triggered this decision.
- Decision:     The concrete, binding choice made.
- Consequences: What constraints this creates for future decisions.
- Status:       Accepted | Superseded (never Deleted).
```

**What deserves an ADR:** any decision that, if changed later, would require touching
more than one file or would break an existing test. This includes: choice of
persistence layer, inter-service communication protocol, authentication mechanism,
error handling contract, external API versioning strategy, and any "we will not do X"
decisions (negative ADRs are often the most important).

**What does not deserve an ADR:** naming conventions, code style, single-file
refactors, and any decision reversible in under an hour.

**The test is about blast radius, not importance.** A decision can be central to the
system and still belong *inside* an existing ADR rather than beside it. Closely-related
decisions — three facets of one question — should share one record. Splitting them reads
as thoroughness and is not: every additional document is additional surface that has to be
kept true, and documents that are not kept true are worse than absent (AP-19). If two ADRs
would be read together every time, they are one ADR.

**A rejection backed by a measurement is worth more than one backed by reasoning**, and it
comes with its own reversal condition for free. "We are not building X because it does not
fit our model" invites the same argument in six months. "We are not building X because we
measured it and the signal was 11.9% against a runner-up of 11.8% — build it when the mode
exceeds 30%" ends the argument and tells the next person exactly what would change it.
Measure before you reject, not only before you accept.

**ADR numbering is sequential and permanent.** ADR-007 does not become ADR-007a or
get removed when superseded. It stays, marked Superseded, pointing at ADR-012 which
replaced it. Reviewers can read the full decision history without consulting git blame.

### 1.4 Contract-First API Design

For any service that exposes an HTTP API, the API schema is written before the
handler code. This is the API-layer equivalent of the ADR process: the contract is
the source of truth; the implementation is downstream of it.

**The contract-first workflow:**

1. Write the OpenAPI schema (or equivalent: JSON Schema, gRPC proto, GraphQL SDL)
   for every endpoint that will be built in this feature.
2. Generate server stubs from the schema. The stubs return `501 Not Implemented`
   until the handler is written.
3. Write integration tests against the schema *before* writing the handler. Tests
   assert request shape, response shape, and error codes.
4. Implement the handler. The schema and tests are the specification; the
   implementation is finished when all pre-written tests pass.

**Practical invariants:**

- Response schemas are additive. New fields are added with a default value; existing
  fields are never removed or retyped without a versioned endpoint (`/v2/`) or an
  explicit deprecation cycle.
- Error responses are typed, not free-form strings. A client that receives
  `{"error": "something went wrong"}` cannot distinguish a rate-limit from a
  validation failure from an internal crash. Use a consistent error envelope:
  ```json
  {"code": "VALIDATION_ERROR", "field": "email", "message": "must be a valid email"}
  ```
- Status codes are semantically correct. `200` means the operation succeeded. `202`
  means it was accepted for async processing. `400` is a client error; `500` is a
  server error. `404` means the resource does not exist; `403` means it exists but
  access is denied. Misusing these codes breaks client retry logic.

### 1.5 The Conflict Check

Before implementing any feature, run an explicit **Conflict Check**: read every
existing ADR and data contract and ask whether the new feature's spec contradicts any
of them. Do this before writing a line of code.

This sounds slow. In practice it takes 5–10 minutes and reliably catches defects that
would otherwise ship. The failure mode it prevents is the silent contradiction: a new
feature spec that technically passes its own tests but violates a contract established
three features ago, discovered only when a customer hits both features in a single
transaction.

**What to cross-reference:**

- New data schema fields against existing query patterns (does this rename break an
  existing index? does this new nullable field break an existing NOT NULL assumption?)
- New API endpoints against existing auth middleware (does this route need auth? does
  the middleware apply to it automatically, or does it require explicit opt-in?)
- New background jobs against existing transaction boundaries (does this job read
  data that another job or request mutates? is there a race condition?)
- New external service dependencies against existing failure handling (if this new
  service is unavailable, what is the graceful degradation path? is it specified?)

### 1.6 Acceptance Criteria Before Implementation

No feature is implemented before its acceptance criteria are written. This is
the core discipline of Behavior-Driven Development (BDD), and it prevents the most
common TDD failure mode: writing tests *after* the code that confirm the code does
what it already does, rather than what it *should* do.

**The practical form:** for each feature, write the Gherkin scenarios first:

```gherkin
Feature: User password reset

  Scenario: Reset link sent to verified email
    Given a user exists with email "user@example.com"
    When a reset is requested for "user@example.com"
    Then a reset email is sent to "user@example.com"
    And the reset link expires in 24 hours

  Scenario: Reset request for unknown email
    Given no user exists with email "ghost@example.com"
    When a reset is requested for "ghost@example.com"
    Then no email is sent
    And the response is identical to the success case   # security: no enumeration
```

The second scenario captures a non-obvious security requirement (no user enumeration)
that is easy to miss if you write the implementation first and then "add tests."
Writing Gherkin first forces you to think about the failure path and the security
boundary before the happy path is built.

**The Definition of Done** per feature:

1. PMA updated with any new ADR or contract change.
2. All Gherkin scenarios have passing step definitions.
3. Unit tests cover the state-transition contract for every new function.
4. `make test` passes at 100% in the deterministic tier.
5. Integration tests verify the new endpoint or service behavior end-to-end.
6. The feature's spec file in `memory/features/` is marked `Status: Done`.

### 1.7 Decision Order Is Itself a Decision

Open questions accumulate in the order they are *noticed*, which is almost never the order
in which they should be *answered*. The PMA records both, and they must be kept apart:
**IDs are permanent labels; the decision order is a separate, changing list.**

The failure this prevents is subtle and expensive. Some questions **dominate** others: the
answer to A determines what the sensible answer to B even is. Decide B first and you have
not decided B — you have decided A by accident, silently, without recording why.

A worked example. Two open questions:

- **A:** what surface do others use to extend this system?
- **B:** which engine/framework does it run on?

These look independent and are not. If A resolves to "plain functions in our language",
the zero-dependency option wins B. If A resolves to "SQL over a published artifact", an
engine that speaks SQL wins B. Answering B first means the engine picks the extension
surface — the tool choosing the architecture, which is exactly backwards. In the project
this playbook was extended from, correcting that order **reversed the answer to B**: the
evidence had leaned one way, and the leading candidate lost its decisive argument the
moment A was settled.

**How to spot a dominating question.** For each pair, ask: *does the answer to one change
what "good" means for the other?* If yes, that one goes first. Most pairs are independent
and the order is arbitrary; the few that are not are where the damage happens.

**Corollary: never reserve ADR numbers for decisions not yet made.** It is tempting to
write "OQ-07 → ADR-013" in the open-questions table. Do not. Numbers are assigned when an
ADR is *written*, and since questions are not answered in ID order the reservations will be
wrong — producing gaps, misleading cross-references, and an index that lies about what
exists. Record what each open question *blocks in the code*, not which future ADR it will
become.

**Re-ordering is a first-class event, not an embarrassment.** When the order changes,
record the change and the reason next to the table. A future reader who sees only the final
sequence learns nothing; one who sees that the sequence was corrected, and why, learns the
most valuable thing in the document.

---

## 2. AI-Assisted Development Model

### 2.1 The Iterative Development Loop

AI coding assistants are most effective when given a narrow, well-specified task with
a clear acceptance criterion. The productivity trap is giving the assistant a large,
ambiguous task and then reviewing a large, possibly wrong output. The effective loop
is tight:

```
[1] Spec (ADR + Gherkin) → [2] PyTest/Jest skeleton → [3] Implementation → [4] Test run
         ↑                                                                        |
         └──────────────── Conflict Check ────────────────────────────────────────┘
                    (surface contradictions before next feature)
```

**Step 1 — Spec:** Update the PMA with the new ADR(s) and any contract changes. Write
the Gherkin feature file. Do not ask the AI to do both the spec and the implementation
in one prompt — the spec is a decision, not a draft.

**Step 2 — Test skeleton:** Write failing tests that mirror the Gherkin scenarios.
The tests define the observable contract: what the function takes, what it returns,
which side effects it must produce. A test that cannot be written usually means the
spec is ambiguous — fix the spec, not the test.

**Step 3 — Implementation:** Give the AI the spec, the failing tests, and the existing
relevant code. The tests are the specification; the implementation is done when the
tests pass.

**Step 4 — Test run:** Run the full deterministic test suite, not just the new tests.
This is the regression check. Paste the output as confirmation; never mark a task
done without it.

### 2.2 Test Strategy: Two Tiers

Split tests into two tiers with different semantics, different tooling, and different
CI gating:

**Deterministic Tier** (fast, fully mocked, must be 100%)

- No network, no database, no external services.
- Every external dependency is mocked at the *factory or constructor boundary*,
  not at the class level.
- Correct mocking target:
  ```python
  @patch("src.services.user.get_db_client")    # ✓ mock at the factory — survives impl changes
  # not:
  @patch("psycopg.connect")                    # ✗ mock at the library — breaks on driver swap
  ```
- 100% pass rate is a hard gate — a single failure blocks merge.
- Runs in under 5 seconds. There is never a reason to skip it.

**System-Behavior Tier** (slower, real dependencies, quality gate)

- Real database, real network, real external services (or high-fidelity stubs).
- Measures behavior under realistic conditions: throughput, latency percentiles,
  failure modes, recovery paths.
- Never asserted with `==` for timing or rate-dependent results. Gated against
  a versioned baseline: "≥ recorded baseline" or "within 20% of baseline."
- Run in a dedicated CI job, not the pre-merge check.

**The tier mapping principle:** if a test requires mocking more than two collaborators,
it is likely testing implementation rather than behavior. Consider whether it belongs
in the system-behavior tier instead, where real collaborators are used.

### 2.3 Context Preservation Across Sessions

AI coding assistants start each session with no memory of prior sessions. Projects
that span multiple sessions — which is every non-trivial project — require deliberate
context management. Without it, the assistant makes decisions in session 8 that
contradict ADRs established in session 2.

**The Project Memory Asset as context injector:**

Every session prompt begins with: "Read `docs/PROJECT_MEMORY.md` in full before
making any structural change." This is not optional guidance — it is the first
instruction. The PMA is the accumulated context; reading it at session start puts
the assistant in the same state as the session that last wrote to it.

**The minimum session-opening brief for any structural task:**

```
1. Read docs/PROJECT_MEMORY.md.
2. Read memory/features/feature-NN-name.md if continuing a specific feature.
3. The specific task: "[verb] [object] per [ADR reference]."
4. Constraints: [list the contracts this task must not violate]
5. Definition of Done: [paste §1.6 or the feature-specific DoD]
6. After implementation: run `make test-local` and paste the output.
```

**The Living Memory update rule:** every session that changes an ADR, data contract,
or feature status must update the PMA in the same response as the code change. Never
defer PMA updates. A PMA that is two sessions behind is unreliable as context and
defeats its own purpose.

### 2.4 Safe AI Edits: The Regression-Prevention Protocol

Allowing an AI agent to edit shared contracts (the PMA, API schemas, database
migrations, CI configuration) introduces regression risk that does not appear in
the test that covers the immediate change. The protocol:

**1. Additive-only changes to data contracts.**

New fields on a response schema are added with a default value. New columns in a
database table are added as `NULL` or with a `DEFAULT`. Existing fields and columns
are never renamed or retyped without an explicit ADR that documents the migration
path and the blast radius (which clients or queries are affected).

**2. Conflict Check before any schema change.**

Before the AI modifies any of: API schema files, database migration files,
shared TypedDict or interface definitions, or CI configuration — it explicitly lists
every existing ADR and contract that the change might affect. If a conflict exists,
the ADR is updated first; the code change follows.

**3. Run the deterministic test suite after every edit session.**

Never end a session without running `make test` (or equivalent) and reviewing the
output. A test failure discovered at the end of a session is cheap to fix. The same
failure discovered two sessions later, after more code was built on the broken
assumption, is expensive.

**4. Mock at the factory boundary, never the implementation class.**

If a test mocks `PostgresClient` directly instead of `get_db_client()`, it will
silently break when the factory's implementation is swapped. The factory boundary
is the contract; it is the correct mock target.

**5. Prompt structure for safe codebase edits:**

```
Context: [paste the relevant ADR + data contract the function reads/writes]
Task: Implement [function name] per ADR-NNN.
Constraints:
  - Reads: [list input contracts]
  - Writes: [list output contracts]
  - Must NOT construct [ClassName] directly — use get_X() factory.
  - Must NOT change any existing ADR, schema field, or API response shape
    without first recording a new or amended ADR in docs/PROJECT_MEMORY.md.
Definition of Done: [paste §1.6]
After implementation: run `make test` and paste the full output.
```

### 2.5 Managing Long-Running Agentic Tasks

Tasks that involve more than 5 tool calls or span more than a few minutes need
explicit structure to prevent mid-task context loss.

**Create a task list before execution.** For any task with more than 3 subtasks,
declare each subtask and its acceptance criterion before beginning. This prevents
partial completion going unnoticed when the assistant approaches its context limit.

**Checkpoint-and-continue protocol.** When a session approaches its context limit
mid-task:
1. Ask the assistant to summarize: what is completed, what is pending, and what is
   the exact next action.
2. Start a fresh session with: "Continuing from [checkpoint]. Completed: [list].
   Next action: [specific step with file path and acceptance criterion]."
3. The next session reads the PMA before doing anything else.

**Verify programmatically, not by assertion.** After any code change, run the test
suite and paste the output. A response of "this should work" without a test run is
not acceptable. The test output is the acceptance criterion.

**For computed outputs, derive the expected values with a second, independent program.**
The session split above exists so that tests specify rather than confirm. For aggregations,
statistics and any other derived number, that split is the *weaker* form of the guard: a
scenario written a day earlier, by the same person who will write the implementation, can
still be written to match whatever the function is going to compute. Nothing stops it.

A number produced by a **different program** cannot be bent. Write a throwaway script that
re-reads the raw inputs and re-applies the rules without importing the implementation, and
assert against what it produces. It costs an hour, it is deleted afterwards or committed as
a spike, and it is the only technique that catches a whole aggregation being subtly,
consistently wrong — the failure mode where every test passes because every test was
written against the same misunderstanding.

Use both where you can. Where you must choose, for derived numbers, choose the independent
derivation.

**Split by concern, not by time.** When a long task involves both spec work and
implementation work, run them in separate sessions. Session 1 produces the ADR and
Gherkin. Session 2 reads the ADR, Gherkin, and PMA, then implements. Mixing spec and
implementation in one session produces code that confirms the spec rather than
testing it.

---

## 3. Architecture & Design Patterns

### 3.1 The Dependency Chokepoint Principle

Every dependency your code has on external systems — databases, APIs, file systems,
message queues — is a seam. How you manage that seam determines how testable,
swappable, and observable your system is.

The consistent practice is to route every dependency through a **factory function**
that owns construction, configuration, and fallback behavior. No module constructs
an external client directly. Every module receives its clients through the factory.

This creates three properties:
- **Testability:** tests mock the factory function, not the client class. When the
  client class changes, tests are unaffected.
- **Swappability:** changing the implementation (e.g., Postgres → DynamoDB) is a
  change inside the factory only. Call sites require zero changes.
- **Observability:** the factory is the single place to inject tracing, logging,
  timeout defaults, and retry behavior. It is never re-implemented at each call site.

### 3.2 Core Patterns

**Factory** — the sole construction path for every external dependency.

```python
# Every external dependency is obtained through a factory.
# No module ever instantiates a database client, cache client,
# or external API client directly.

def get_db_client(url: str | None = None) -> DatabaseClient:
    """Select backend based on environment. Returns Postgres in production,
    in-memory in tests. All call sites receive the same interface."""
    if url:
        return PostgresClient(url)
    return InMemoryClient()

def get_cache_client(url: str | None = None) -> CacheClient:
    if url:
        return RedisClient(url)
    return InMemoryCacheClient()
```

The correct mock target in tests:
```python
@patch("src.services.payments.get_db_client")   # ✓ mock the factory
# not:
@patch("psycopg.connect")                        # ✗ mock the library internals
```

**Repository** — storage abstraction over the persistence layer.

The Repository pattern separates the query/storage logic from the business logic.
Business logic operates against a Repository interface and never writes SQL or ORM
queries directly. This makes the business logic testable without a database and
swappable between persistence backends.

```python
class UserRepository(Protocol):
    def get(self, user_id: str) -> User | None: ...
    def save(self, user: User) -> None: ...
    def find_by_email(self, email: str) -> User | None: ...

class PostgresUserRepository:
    def __init__(self, conn): self._conn = conn
    def get(self, user_id: str) -> User | None:
        row = self._conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
        return User.from_row(row) if row else None
    # ...

class InMemoryUserRepository:
    def __init__(self): self._store: dict[str, User] = {}
    def get(self, user_id: str) -> User | None:
        return self._store.get(user_id)
    # ...

def get_user_repository(db_url: str | None = None) -> UserRepository:
    if db_url:
        return PostgresUserRepository(psycopg.connect(db_url))
    return InMemoryUserRepository()
```

Business logic receives the repository through the factory, never by constructing
either implementation directly.

**Strategy** — runtime selection between interchangeable implementations.

The Strategy pattern is the correct way to handle feature flags, provider switches,
and environment-specific behavior. Each strategy implements the same interface; the
caller never knows which strategy is active.

```python
class NotificationSender(Protocol):
    def send(self, to: str, subject: str, body: str) -> None: ...

class SmtpSender:
    def send(self, to: str, subject: str, body: str) -> None:
        # real SMTP delivery
        ...

class LoggingSender:
    """Dev/test strategy — logs instead of sending."""
    def send(self, to: str, subject: str, body: str) -> None:
        logger.info(f"[MOCK EMAIL] to={to} subject={subject}")

def get_notification_sender() -> NotificationSender:
    if os.environ.get("EMAIL_ENABLED") == "true":
        return SmtpSender(host=os.environ["SMTP_HOST"])
    return LoggingSender()
```

**Adapter** — interface translation at a dependency boundary.

When an external library or legacy service returns data in a shape your code was not
written against, introduce an Adapter rather than updating every call site or
polluting business logic with conversion code.

```python
class LegacyPaymentAdapter:
    """Adapts the legacy payment API's dict response to the domain PaymentResult type.

    The legacy client returns {'statusCode': int, 'transactionRef': str, 'errorMsg': str | None}.
    Every call site was written against PaymentResult(success, transaction_id, error).
    The adapter translates once at the boundary; no call site requires a change.
    """

    def __init__(self, legacy_client): self._client = legacy_client

    def charge(self, amount: int, card_token: str) -> PaymentResult:
        raw = self._client.processCharge(amount, card_token)   # legacy API
        return PaymentResult(
            success=raw["statusCode"] == 0,
            transaction_id=raw.get("transactionRef"),
            error=raw.get("errorMsg"),
        )
```

**Gateway** — a chokepoint for all calls to a specific external system.

A Gateway is a module (not a class) that owns every function that calls a specific
external system. No other module imports the underlying SDK or makes HTTP calls
directly. The Gateway injects auth, tracing, timeouts, and retry behavior once.

```python
# src/gateway/payment_gateway.py

def charge_card(amount_cents: int, card_token: str, idempotency_key: str) -> PaymentResult:
    """Sole entry point for all payment API calls. Injects auth, timeout,
    idempotency header, and trace ID. Call sites never import stripe directly."""
    client = _get_client()
    with trace_span("payment.charge"):
        response = client.charges.create(
            amount=amount_cents,
            currency="usd",
            source=card_token,
            idempotency_key=idempotency_key,
        )
    return PaymentResult.from_stripe(response)
```

Enforce the Gateway boundary with a lint check — a grep or AST rule in CI that
rejects any direct `import stripe` (or equivalent) outside the Gateway module.

### 3.3 Service Integration Patterns

**Prefer async/queue for side-effecting cross-service calls.**

Synchronous HTTP calls between services for non-critical side effects (sending a
notification, updating a secondary index, triggering a workflow) couple the caller's
latency to the callee's availability. If the notification service is slow, the user's
checkout slows. If the notification service is down, the checkout fails.

The correct pattern for non-critical side effects: publish an event to a queue
(Kafka, SQS, Redis Streams) and let the consumer handle it asynchronously. The
producer's response time is decoupled from the consumer's processing time.

**Idempotency keys on every mutating operation.**

Any operation that creates or modifies a resource must accept an idempotency key.
If the request is retried (network timeout, client retry), the second call returns
the same result as the first without creating a duplicate resource. The idempotency
key is typically a client-generated UUID, passed as a header or body field, stored
server-side and used to short-circuit duplicate requests.

Without idempotency keys: a network timeout on a payment creates a charge; the
client retries; the user is charged twice. This is the most expensive missing pattern
in any service that handles money, inventory, or any resource with uniqueness
constraints.

**Don't block the thread on I/O.**

In async runtimes (Node.js, Python asyncio, Go goroutines), I/O-bound operations
must not run synchronously on the event loop thread. A blocking DB call on the event
loop blocks all other requests.

Python pattern:
```python
# Synchronous library call in an async endpoint
@app.post("/orders")
async def create_order(req: OrderRequest) -> OrderResponse:
    # Wrong: blocks the event loop
    result = db_client.insert(req.to_row())

    # Correct: offload to a thread pool
    result = await asyncio.to_thread(db_client.insert, req.to_row())
    return OrderResponse.from_row(result)
```

Add a timeout to every I/O call:
```python
result = await asyncio.wait_for(
    asyncio.to_thread(db_client.insert, req.to_row()),
    timeout=float(os.environ.get("DB_TIMEOUT_SECONDS", "10")),
)
```

### 3.4 Database Migration Strategy

Database migrations are the highest-risk operation in a running system. The practices
here prevent the failure modes that appear when migrations are treated as an
afterthought.

**All migrations must be reversible.** Every migration file includes both an `up`
(apply) and a `down` (revert). This is not optional hygiene — it is what makes
rollback possible. A migration without a `down` means a failed deploy cannot be
rolled back without manual DBA intervention.

**Column renames are three-phase operations.** Renaming `user_email` to `email` is
not a single migration. It is three migrations over two deployments:
1. Migration 1 (deploy N): Add `email` column with the same default. Write to both
   `user_email` and `email`.
2. Deploy N+1: All reads use `email`. All writes go to both columns.
3. Migration 2 (deploy N+2): Drop `user_email`. The old column is gone only after
   all consumers have migrated to the new name.

Skipping steps produces a broken deploy window: the moment the rename migration runs,
any old code still reading `user_email` fails.

**Schema migrations and data migrations are separate files.** Schema changes (ADD
COLUMN, DROP INDEX, CREATE TABLE) should never be in the same migration file as data
transformations (UPDATE users SET ...). Schema migrations are fast and lock-minimal.
Data migrations on large tables take minutes and hold locks. Running them together
means a long lock window on a production table.

**Never run `CREATE INDEX` inside a transaction block.** Postgres's `CREATE INDEX
CONCURRENTLY` avoids table locks but cannot run inside a transaction. Any migration
runner that wraps every migration in a transaction will fail on this statement. Either
configure the runner to disable transaction wrapping for this specific migration, or
run the index creation manually outside the migration framework.

---

## 4. Production Readiness

### 4.1 Observability: Logging, Tracing, Metrics

**Structured logging over plain text.** Every log line is a JSON object with
consistent fields: `timestamp`, `level`, `service`, `trace_id`, `message`, and any
event-specific fields. Plain-text logs are human-readable but machine-unsearchable.
A log line like `"Payment failed for user 1234"` cannot be aggregated, filtered by
user ID, or joined with a trace without a regex. The equivalent structured log:

```json
{"timestamp": "2026-07-10T14:22:01Z", "level": "error", "service": "payment-api",
 "trace_id": "abc123", "user_id": "1234", "event": "payment_failed",
 "reason": "insufficient_funds", "amount_cents": 4999}
```

**Distributed tracing with a `trace_id` on every request.** Every inbound HTTP
request generates a `trace_id` (UUID or W3C Trace Context format). The `trace_id` is
propagated to every downstream call: database queries, external API calls, queue
messages. Every log line includes the `trace_id`. This makes it possible to reconstruct
a single request's full execution path across multiple services from a single ID.

```python
@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
    with trace_context(trace_id):
        response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response
```

**Three metrics every service must export:**

- **Request rate** — requests per second per endpoint, by status code. Alertable:
  sudden drop to zero or sudden spike above normal.
- **Error rate** — fraction of requests returning 5xx. Alertable: exceeds 1% for 5
  minutes (tune per SLO).
- **Latency** — p50, p95, p99 per endpoint. Alertable: p99 exceeds SLO threshold.

These three — combined with the `trace_id` to drill into specific failures — cover
the majority of production incidents.

### 4.2 Security Checkpoints in the Development Loop

Security review is not a gate at the end of a project; it is a checkpoint at the
start of every feature that involves authentication, authorization, external input,
or data exposure.

**Input validation at the API boundary, not the business logic layer.** Validate
request shape, field types, and field constraints in the API handler or request
schema before the data enters business logic. Business logic should assume its inputs
are already validated. A validation error at the business logic layer means the
boundary let something through; a validation error at the boundary is correct behavior.

**Secrets are never environment variables in the traditional sense.** The pattern
`export SECRET_KEY=abc123` in a `.env` file that gets committed to git is how secrets
leak. The correct pattern:

- `.env.example` contains placeholder values only: `SECRET_KEY=your-key-here`. This
  file is committed to git.
- `.env` is in `.gitignore` and never committed.
- In production, secrets are injected by a secrets manager (AWS Secrets Manager,
  HashiCorp Vault, Kubernetes Secrets) at runtime, not from environment files.

**The principle of least privilege applies to service accounts, not just users.** The
database user your application connects with should have `SELECT`, `INSERT`, `UPDATE`,
`DELETE` on the tables it needs — and nothing else. It should not have `DROP TABLE`,
`CREATE INDEX`, or `ALTER TABLE`. Migrations run as a separate, elevated user that is
revoked after the migration completes.

**Dependency audits are part of CI.** Add `pip audit` (Python), `npm audit` (Node),
or `govulncheck` (Go) to the CI pipeline. Treat a high-severity CVE in a direct
dependency as a blocking issue. Treat a high-severity CVE in a transitive dependency
as a time-bounded issue (resolve within SLO, typically 7 days).

### 4.3 CI/CD Pipeline Structure

**Fast tests first, slow tests last.** The CI pipeline runs tests in order of
increasing runtime. A broken unit test should fail in under a minute, not after
waiting 20 minutes for integration tests to run. The pipeline stages:

```
[1] Lint + type check       (<10s)    — fail fast on obvious errors
[2] Unit tests (mocked)     (<60s)    — deterministic tier, 100% required
[3] Integration tests       (2–5min)  — real database, real cache, mocked external
[4] End-to-end tests        (5–15min) — full stack, deployed to staging-like env
[5] Security scan           (varies)  — dependency audit, SAST
```

**Environment parity.** The environment in which code runs in CI must be as close as
possible to production. Specifically: the same database engine (not SQLite in CI and
Postgres in production), the same environment variable names (even if values differ),
and the same secrets injection mechanism. Differences between CI and production are
how bugs survive CI and appear in production.

**Deployment is blocked by tests, not scheduled.** The rule: code does not reach
production unless all CI stages pass. There is no "we'll fix the test after deploy"
— that decision converts a CI failure into a production risk. If a failing test is
incorrect, fix the test in the same PR as the code change.

**Feature flags over long-lived branches.** Feature branches that live for more than
a week accumulate merge conflicts and diverge from main. Prefer short-lived branches
(1–3 days) with feature flags that control whether a partially-built feature is active.
This keeps main releasable at all times and lets you deploy code that is not yet
"on" for users.

### 4.4 Anti-Pattern Catalog

The following failure modes appear across teams and tech stacks. Each is documented
with its root cause and the specific fix that resolves it.

---

#### AP-01: SDK Returns a Typed Object, Not a Primitive

**Symptom:** `AttributeError: 'SomeObject' has no attribute 'strip'` or
`TypeError: expected str, got SomeObject` in production but not in tests.

**Root cause:** Tests mock the return value as a `str` (or `int`, or `dict`). The
real SDK returns a typed object. The mismatch is invisible during testing and only
appears when real packages are used.

**Fix:** Introduce an Adapter at the factory boundary that unwraps the typed object
to the interface your code expects. Change one place; no call sites require updates.

```python
class _UnwrappingClient:
    def __init__(self, real_client): self._client = real_client
    def call(self, *args, **kwargs) -> str:
        result = self._client.call(*args, **kwargs)
        return result.content if hasattr(result, "content") else str(result)
```

**Variant — the parser destroys the value before your code sees it.** The same family,
but the loss happens at *parse* time and is irreversible, so no amount of careful handling
downstream can recover it. A JSON decoder converts numbers to the platform float type
before returning them; wrapping that float in an exact decimal afterwards preserves the
error rather than removing it. A price of `0.1` becomes `0.1000000000000000055511151231257827`
and stays that way. The tell is that the bug survives code that looks obviously correct.

**Fix:** intervene at parse time, not after. Most decoders expose a hook for exactly this
(`parse_float` / `parse_int`, a custom reviver, a schema-driven reader). If yours does not,
read the raw token yourself. Write the test with a value that cannot survive a float
round-trip — `0.1` is the classic — so the naive implementation fails loudly instead of
being off in the fifteenth decimal place where nobody looks.

**Lesson:** When integrating any real SDK against a hand-written stub, validate the
actual return type of every method before writing code that calls it. Never assume
the real SDK has the same calling convention as your stub. And check *where* a value is
converted, not only *what* it converts to.

---

#### AP-02: Mocking at the Wrong Layer

**Symptom:** All tests pass after an implementation swap, but production breaks
immediately.

**Root cause:** Tests mocked the implementation class (`@patch("psycopg.connect")`)
instead of the factory function (`@patch("src.db.get_db_client")`). When the
implementation was swapped (psycopg → asyncpg), the mock no longer intercepted the
call.

**Fix:** Always mock at the factory boundary. The factory is the contract; its
internals are the implementation. Tests coupled to internals break on every internal
change.

---

#### AP-03: Transaction Semantics Differ Between Drivers

**Symptom:** `OperationalError: cannot run inside a transaction block` on specific
DDL statements.

**Root cause:** Some DDL statements (Postgres `CREATE INDEX CONCURRENTLY`, for
example) cannot run inside a transaction. Most database drivers default to
`autocommit=False`, which wraps every statement in a transaction. The statement fails
only when it hits that specific restriction.

**Fix:** For connections used to run DDL, set `autocommit=True`:
```python
conn = psycopg.connect(database_url, autocommit=True)
```
For application connections, keep `autocommit=False` (the default) to preserve
transactional safety on writes.

---

#### AP-04: Ingest and Retrieval Use Different Instances

**Symptom:** Write operation succeeds; subsequent read returns empty or stale data.

**Root cause:** The write path instantiated a storage object directly
(`store = InMemoryStore()`). The read path used a different object obtained through
the factory (`store = get_store(db_url)`). Two separate instances; no shared data.

**Fix:** All write paths and all read paths for a shared data store must go through
the same factory function. Any hardcoded instantiation bypasses this guarantee.
The factory is responsible for ensuring that calls with the same arguments return
the same backend instance.

---

#### AP-05: SDK Environment Variable Renames

**Symptom:** A feature that was working silently stops working after a dependency
upgrade. No error — just no effect.

**Root cause:** The upstream SDK renamed its environment variables between versions
(e.g., `PROVIDER_API_KEY` → `PROVIDER_SECRET`). The old name is silently ignored —
no warning, no deprecation notice in the log, no error.

**Fix:** After any dependency upgrade, check the changelog for environment variable
renames. Pin your env var names to the documentation for the installed version.
Add a startup check that validates required env vars are set before the application
accepts traffic:

```python
REQUIRED_VARS = ["DATABASE_URL", "PROVIDER_SECRET", "SMTP_HOST"]

def check_env():
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(f"Required env vars not set: {', '.join(missing)}")
```

---

#### AP-06: Expensive Resource Loaded Per Request

**Symptom:** Request latency is unexpectedly high — 2–10× what the operation itself
should take. Memory usage spikes on every request.

**Root cause:** A resource that is expensive to initialize (a database connection
pool, an ML model, a compiled regex, a cryptographic key derivation) is being
constructed on every request instead of once at startup.

**Fix:** Initialize expensive resources once at module level and reuse them:

```python
# Bad: new connection on every request
@app.post("/search")
async def search(req: SearchRequest):
    client = ElasticsearchClient(url=ES_URL)   # 150ms per request
    return client.search(req.query)

# Good: connection pool initialized once at startup
_es_client: ElasticsearchClient | None = None

@app.on_event("startup")
async def startup():
    global _es_client
    _es_client = ElasticsearchClient(url=ES_URL)

@app.post("/search")
async def search(req: SearchRequest):
    return _es_client.search(req.query)        # <1ms for connection reuse
```

The singleton is intentional when the resource is stateless (it only carries
configuration or a connection pool) and thread-safe.

---

#### AP-07: Missing Port in Service URL

**Symptom:** `Connection refused` on inter-service calls in Docker Compose or
Kubernetes, even though the service is running.

**Root cause:** `http://service-name` without a port defaults to port 80. The
service binds to a non-standard port (8080, 9000, etc.). Docker's internal DNS
resolves the service name correctly but the port default is wrong.

**Fix:** Always specify ports explicitly in service-to-service URLs:
```python
BASE_URL = os.environ.get("PAYMENT_API_URL", "http://payment-service:8080")
```

Never rely on port 80 defaults in internal service URLs. Never embed ports in
code; always read from environment variables so they can be changed without a
deploy.

---

#### AP-08: Containerized Workload Needs Host Hardware

**Symptom:** A workload that runs in 100ms on a developer's machine takes 20+ seconds
in the Docker container, or produces different outputs.

**Root cause:** The workload requires specialized hardware (GPU for ML inference,
hardware security module for cryptographic operations) that Docker cannot pass through
on macOS. Docker Desktop on macOS runs Linux in a VM; GPU drivers, Metal, and other
hardware accelerators are not available inside the VM.

**Fix:** Run the hardware-dependent process natively on the host. Have the container
call it over localhost or `host.docker.internal` (macOS/Docker Desktop). This gives
the container-based service access to the result without the container needing the
hardware itself.

```yaml
# docker-compose.yml — point to host-side process
environment:
  INFERENCE_API_URL: "http://host.docker.internal:11434"
```

---

#### AP-09: Silent Exception Swallowing in Async Handlers

**Symptom:** An async background task stops working with no error in the logs.

**Root cause:** An unhandled exception inside an async task is swallowed silently
unless the task is explicitly awaited or an exception handler is attached. Python's
`asyncio.create_task()` and `asyncio.gather()` without `return_exceptions=False`
(the default) will drop exceptions on the floor.

**Fix:** Always attach an exception handler to fire-and-forget tasks, or use a
structured concurrency pattern:

```python
# Dangerous: exception silently discarded
task = asyncio.create_task(background_job())

# Safe: exception logged and re-raised
def _handle_task_exception(task: asyncio.Task):
    if not task.cancelled() and task.exception():
        logger.error("Background task failed", exc_info=task.exception())

task = asyncio.create_task(background_job())
task.add_done_callback(_handle_task_exception)
```

---

#### AP-10: Boolean Flag Overload

**Symptom:** A function has 4+ boolean parameters. Call sites are unreadable:
`create_user(True, False, True, False)`. Adding a new option requires updating
every call site.

**Root cause:** Each new behavior variant was added as a new boolean parameter
rather than as a new function, a configuration object, or a named enum.

**Fix:** Replace boolean flag parameters with a configuration dataclass or named
enum. Call sites are self-documenting; new variants add a new enum value, not a
new parameter.

```python
# Bad
def send_email(to: str, sync: bool, retry: bool, cc_admin: bool) -> None: ...

# Good
@dataclass
class EmailOptions:
    sync: bool = False
    retry: bool = True
    cc_admin: bool = False

def send_email(to: str, options: EmailOptions = EmailOptions()) -> None: ...
```

---

#### AP-11: Parallel Abstraction Exercised Only by Tests

**Symptom:** A module or function is green in the test suite but never runs in
production. Coverage looks healthy. The real entry point calls a separate,
similar-looking implementation, so a fix applied to the tested version has no effect in
prod.

**Root cause:** Two implementations of the same flow exist. A clean, composable one was
written first — for elegance, or to satisfy an architecture note ("assemble it as a
pipeline"). The one the entry point actually calls was written later, because the real
path needed cross-cutting side effects the clean version never modeled: error reporting,
metrics, auth, request-scoped IDs. The clean abstraction survives only through its own
test and drifts into dead code. Its test is worse than no test — it manufactures
confidence in a path that ships to no one.

**Fix:** Enforce the invariant **the tested path is the production path.** When a second
orchestration of the same sequence appears, delete one. If the entry point *cannot* use
the clean abstraction because of side effects, that is the signal the abstraction was
wrong for this problem: fold the steps into the real path (keep them as pure functions so
they remain unit-testable) rather than maintaining a parallel copy. Ensure at least one
test drives the real entry point, not only the internal helpers.

```bash
# Before trusting a new abstraction, check who actually calls it.
# If the only callers are tests, it is dead on arrival — wire it in or delete it.
grep -rn "run_pipeline" src/ | grep -v "/tests/"   # (empty output = red flag)
```

**AI-assisted note:** Coding agents are especially prone to this. Asked to "assemble the
pipeline," an agent will produce a tidy composable object plus a passing test, then — when
a later task needs reporting, metrics, or idempotency — wire a *separate* real path and
leave the first as orphaned scaffolding. Both look correct in isolation. Grep for
non-test callers of any abstraction an agent introduces before accepting it.

**Prevention beats detection here.** The grep above finds this after it exists. Cheaper is
to refuse to build an abstraction before its production caller exists — if the feature that
will call it is two sessions away, the abstraction waits two sessions. "This component has
no open questions blocking it" is a true statement that hides a false one: a component
whose only possible caller is a test is blocked, whatever the dependency list says.

**Lesson:** Test coverage of a function you do not call in production is negative value:
maintenance cost plus false confidence. Connect the tested path to the shipped path, or
remove the abstraction.

---

#### AP-12: Lazy Evaluation Breaks an Exception-Timing Contract

**Symptom:** A handler wraps a call in `try/except` for a documented exception,
but the exception is never caught. In HTTP services: a success status line is
sent, then the response body fails — the client sees a 200 with broken content.

**Root cause:** The implementation is a generator (or returns a lazy iterator/
promise). Calling it executes *none* of its body — validation deferred until
first iteration, which happens after the caller's `try` block has exited (and,
for streaming responses, after headers are already on the wire).

**Fix:** Perform all validation eagerly in a plain function, then return an
inner generator for the streaming part:

```python
def open(self, key):                      # NOT a generator itself
    if not self._exists(key):
        raise ObjectNotFound(key)         # raises AT CALL TIME
    return self._iter_chunks(key)         # laziness only where intended
```

**Lesson:** *When* an exception is raised is part of the interface contract,
not an implementation detail. Document raise-at-call-time on the ABC. And
beware the masking unit test: asserting `next(f())` raises tests the
workaround, not the contract — assert that the *call* raises, without
iterating. The end-to-end contract test is the tier that catches this class.

---

#### AP-13: Works Under the Test Runner, Fails Under the Production Launcher

**Symptom:** The full test suite is green; the same code crashes at deploy or
container start — often with `ModuleNotFoundError`, missing files, or path
errors.

**Root cause:** Each entry point constructs its own execution context. Test
runners (pytest, jest) add the project root to the import path and run from
the repo; production launchers (migration CLIs, init scripts, container
entrypoints) do not. Code paths that only execute at boot — migrations,
env checks, asset loading — are never exercised by the suite, and the build
artifact may not even contain the files the suite relied on.

**Fix:** Ship the test suite inside the runtime artifact and run it there
(environment parity, §4.3). For CLI-launched code (migration runners), set the
tool's path configuration explicitly (e.g. Alembic's `prepend_sys_path`)
rather than relying on the runner's implicit behavior. Treat "container boots
cleanly from a fresh volume" as a test case in its own right.

**Second symptom, same root cause: the maintainer's convenience path.** Long-running work
accumulates shortcuts — an environment outside the project directory, cache overrides, a
handful of exported variables that make the loop faster. Every test passes, thousands of
times, through a path **no other person will ever take**. The documented setup command goes
unexercised for the entire life of the project and is discovered to be broken by the first
stranger who runs it.

**Fix:** at least once before shipping, and ideally in CI, run the documented path exactly
as written — fresh clone, no exported variables, the setup command from the README, then
the gate. It takes two minutes and it is the only test of the instructions themselves.

**Lesson:** A test suite that never runs inside the shipping artifact tests a
different program. The gap between "tests pass" and "the deploy path ran" is
where first-boot failures live — and the documented instructions are part of the
deploy path.

---

#### AP-14: Blind Dependency-Audit Fixes in Version-Locked Ecosystems

**Symptom:** After running the package manager's automated vulnerability fix
in aggressive mode, the project no longer builds; framework versions have
jumped across multiple majors in both directions.

**Root cause:** Audit auto-fixers optimize exactly one objective — making
advisories disappear — with no knowledge of framework version matrices.
Ecosystems with a version-locked core (Expo SDK ↔ React Native ↔ React,
similar in others) require coordinated versions that the fixer freely breaks.
Compounding it: most flagged CVEs sit in *dev-time tooling* that never ships
to the user, so the "fix" broke the build to remove risk that wasn't in the
threat model.

**Fix:** Restore the known-good manifest, delete the lockfile and installed
tree, reinstall, then use the ecosystem's own alignment tool (e.g.
`expo install --fix`) for version reconciliation. Gate CI on advisories in
*direct runtime* dependencies; time-box transitive and dev-tooling advisories
instead of blocking on them (§4.2).

**Lesson:** Read audit reports with a threat model — where does this code
run, and who can reach it? Never run an auto-fixer's force mode on a
version-locked ecosystem.

---

#### AP-15: One Dependency Tree, Two Resolvers, Two Different Views

**Symptom:** Tool A (the bundler) finds a package; tool B (the native linker,
a codegen step, a CLI) reports it missing — from the same installed tree. At
runtime: "module found" in one layer, "native counterpart missing" in another.

**Root cause:** The package manager produced a *valid but unusual* layout
(e.g. packages nested under a dependency instead of hoisted), typically after
a corrupted or interrupted resolution. Consumers walk the tree with different
algorithms: bundlers resolve nested paths transitively; autolinkers and
codegen tools often scan only the top level. A layout satisfying one consumer
is invisible to the other.

**Fix:** Rebuild the layout flat: delete the installed tree *and* the
lockfile (the lockfile records the broken resolution), reinstall from the
manifest. If a package must be top-level for a consumer, promote it to a
direct dependency — that is a placement guarantee, not duplication.

**Lesson:** When two tools disagree about an installed package, suspect the
tree's *shape*, not the tools. Verify with `ls` at the path each consumer
actually scans.

---

#### AP-16: Toolchain Upgrade Breaks Vendored Third-Party Code

**Symptom:** After a compiler/SDK/platform upgrade, the build fails deep
inside a dependency-of-a-dependency you have never touched. Nothing in your
code changed.

**Root cause:** New toolchains enforce stricter language or API rules; the
framework's vendored dependencies were validated against the old ones. The
upstream fix exists but only lands in a framework version several releases
away — upgrading to it mid-task is a larger change than the problem justifies.

**Fix:** Apply the narrowest possible workaround, scoped to the single
offending compilation unit (e.g. build one library against the older language
standard — never the whole project). Mark the workaround with (1) the reason,
(2) upstream issue links, (3) the explicit removal condition ("delete when on
framework ≥ X"). If the build file is generated, note that regeneration
erases the patch.

**Lesson:** Workarounds without removal conditions become permanent. Scope
determines safety: a project-wide flag "fixes" the build and quietly breaks
three other things; a single-target flag fixes the build, full stop.

---

#### AP-17: Global Hardware State With Two Uncoordinated Owners

**Symptom:** Two features each work in isolation; enabling both produces
resource-denied errors or native crashes (audio capture fails when video
plays; camera dies when screen recording starts).

**Root cause:** Both libraries configure the same process-global hardware
resource (audio session, camera pipeline, GPU context) on their own schedule.
Last writer wins; the loser's handle silently dies or the next operation
crashes. Configuration is also mode-sensitive: combinations the library
author never tested crash inside native code, beyond any try/catch.

**Fix:** Designate exactly ONE owner of the shared resource; configure every
other consumer to cooperative/passive mode (the "mix with others" family).
Stay on each library's default modes — they are the tested paths; override
one variable at a time so a crash identifies its cause. Document the
ownership contract at both call sites, cross-referenced.

**Lesson:** Process-global resources need an ownership decision, made
explicitly and written down. Two libraries "managing" one session is not
redundancy; it is a race.

---

#### AP-18: Provisional Events Fire Actions Twice

**Symptom:** A voice/gesture/streaming input triggers the same action twice,
or a refined version of the input ("rewind" → "rewind 10") stacks a second
action on top of the first instead of correcting it.

**Root cause:** Recognition pipelines emit *provisional* results before the
*final* one, and both describe the same user intent. Naive handlers treat
every event as a new command. Deduplication by command name alone then blocks
the refined final result — forcing a choice between double-firing and
ignoring the refinement.

**Fix:** Two mechanisms, both in the pure/testable layer:
1. Dedupe on a key that includes the payload (`command:args`) within a short
   window — identical repeats are dropped, refinements pass through.
2. Amendment semantics for refinements: remember the pre-action state; when a
   refined event arrives inside the window, re-apply from that remembered
   state so the net effect equals the final intent (not the sum of both).

**Lesson:** For event streams with interim results, idempotency is not
enough — you need *amendability*. Model "same utterance, better transcript"
as a correction to the previous action, never as a new one.

---

#### AP-19: The Document That Asserts What the Code Does Not Do

**Symptom:** A design document states that the system does something. It does not. The gap
survives for months and is found by accident, usually while investigating something else.
Nobody was careless — the document was simply trusted exactly as much as the code it
described.

**Root cause:** Every enforcement mechanism in this playbook reads **code**. The
architectural lint parses source. The hooks fire on file writes. CI runs tests. None of them
reads prose. §1.1 makes the project memory asset the authoritative source of truth and
gives it no mechanism for staying true — so an ADR is the one artifact in the repository
that can assert anything at all and never be contradicted.

This is §1.1's own thesis turned on itself: a rule with no enforcement is followed until the
first deadline. It applies to the documents describing the rules just as much as to the code
obeying them.

**Fix — three habits, in decreasing order of value:**

1. **An ADR that asserts testable behaviour names the test that proves it, or is reworded
   as an intention.** "The manifest records a digest of every input" is a claim and needs a
   test. "We intend the manifest to identify its inputs" is a direction and needs none. Know
   which one you are writing.
2. **Give documentation a lint.** Relative links resolve; in-page anchors match real
   headings; generated files match their generator. These are cheap checks that catch the
   small rot before it teaches readers to distrust the document. A generated file plus a
   drift check makes staleness impossible rather than unlikely.
3. **Verify claims about the environment by running them.** A negative result usually has
   several possible causes and no diagnostic value on its own. `docker info` fails
   identically whether a daemon is absent or merely not started; writing "there is no
   daemon" into a decision record turns an unverified guess into an institutional fact.

**Lesson:** Documentation that lies is worse than documentation that is missing, because it
is believed. Every claim in a design document is either enforced, dated, or decaying — and
you do not get to choose which without doing something about it.

---

#### AP-20: Quarantine That Conflates a Defect With an Exclusion

**Symptom:** A pipeline reports that it rejected 15% of its input. Investigation shows the
source system is fine: almost all of those records were **valid and simply not in scope**.
The handful of genuine defects — 0.011% — are buried under records that were never a
problem. Either the team learns to ignore the reject metric, or somebody escalates to an
upstream team that has done nothing wrong.

**Root cause:** One sink for everything that "didn't make it through", merging two facts
that mean opposite things:

| | Meaning | Who acts | Should it alert? |
|---|---|---|---|
| **Defect** | The record is malformed. Something upstream is broken | The source system's owner | Yes |
| **Exclusion** | The record is well-formed and outside our scope | Nobody, usually | No — but the *count* is a business signal |

Merging them makes the defect rate unreadable in both directions: real defects are hidden
by volume, and a legitimate scope filter looks like an incident.

**Fix:** two sinks, two counters, two thresholds. A quality gate fires on the defect rate
only; an exclusion is never a failure condition. Both counts appear in the run summary,
because "which sources are sending us records we have no configuration for" is a real
question with real value — usually an onboarding lead rather than a bug.

**Generalises past data pipelines.** Any filtering stage has this shape: a request rejected
as malformed and a request rejected as unauthorised are not the same event; a message that
failed to parse and a message correctly routed elsewhere are not the same event. Wherever
code decides "this one does not continue", ask whether it is answering *broken* or *not
mine*, and count them apart.

---

### 4.5 Numeric Thresholds That Require Empirical Tuning

The following parameters are commonly set to "reasonable" defaults and left untouched.
They should be treated as placeholders until measured against real traffic:

| Parameter | Common default | What to measure |
|---|---|---|
| Request timeout | 30s | p99 latency of the slowest legitimate request |
| DB connection pool size | 10 | Peak concurrent DB operations under load |
| Retry max attempts | 3 | Downstream failure duration distribution |
| Rate limit per key | 100 req/min | Peak legitimate usage per user/tenant |
| Cache TTL | 5 min | Data freshness requirement vs. cache hit rate |
| Pagination page size | 20 | Median result set size vs. client rendering time |

The mechanism for each should be implemented and tested before the threshold is
finalized. The number is provisional until measured.

---

## 5. Claude Code Hooks Reference

Claude Code hooks are shell commands that fire automatically on tool events. They are
defined in `.claude/settings.json` and run in the project directory every time Claude
writes or edits a file. They do not require the developer to remember to run anything
— enforcement becomes a property of the tool, not a discipline reminder.

**Important:** Hooks run only in **Claude Code CLI** sessions (`claude` command in a
terminal). They do not run in Cowork, claude.ai, or other interfaces.

### 5.1 The PostToolUse Pattern

The most useful hook type is `PostToolUse` with a `Write|Edit` matcher: it fires after
every file creation or edit. Combined with a `file_path` filter in the command, you
get targeted automation — run the right check against the right file, instantly, every
time Claude touches it.

The basic `settings.json` shape:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path // .tool_response.filePath' | { read -r f; case \"$f\" in <GLOB_PATTERN>) <YOUR_COMMAND> ;; esac; }",
            "statusMessage": "<Message shown while hook runs>"
          }
        ]
      }
    ]
  }
}
```

The `jq` pipeline extracts the file path from the tool event JSON. The `case` block
filters by glob pattern so the command only runs on relevant files. Multiple hook
entries in the array run in sequence after every write.

**A note on `jq`.** It is the clearest way to write these examples and it is not installed
everywhere — a hook that fails because a tool is missing is worse than no hook, because it
trains you to ignore hook output. If the projects you work on cannot assume `jq`, extract
the path with the interpreter the project already depends on and keep the rest identical:

```bash
python3 -c "import json,sys; e=json.load(sys.stdin); \
print((e.get('tool_input') or {}).get('file_path') \
      or (e.get('tool_response') or {}).get('filePath') or '')"
```

Whatever the extractor, it must never fail loudly on an event it does not understand.
Print an empty line and exit zero; the `case` block will then match nothing.

### 5.2 The Generic Hook Patterns

**Pattern 1 — Architectural lint on source file save**

Run a project-specific structural constraint check (forbidden imports, layer boundary
violations, naming convention enforcement) whenever any source file is edited:

```json
{
  "type": "command",
  "command": "jq -r '.tool_input.file_path // .tool_response.filePath' | { read -r f; case \"$f\" in */src/*.py|src/*.py) bash scripts/lint_architecture.sh \"$f\" ;; esac; }",
  "statusMessage": "Checking architectural constraints..."
}
```

Replace `scripts/lint_architecture.sh` with whatever enforces your project's structural
rules. In Sentinel, this runs `lint_gateway_usage.sh` to enforce ADR-006 (no direct
provider SDK imports outside `src/gateway/`). In another project it might check that
repository classes don't import HTTP clients, or that config modules don't import
business logic.

**Pattern 2 — Auto-run matching unit test on implementation file save**

Automatically run the unit test that covers a changed file, using the project's
`src/X.py → tests/test_X.py` naming convention:

```json
{
  "type": "command",
  "command": "jq -r '.tool_input.file_path // .tool_response.filePath' | { read -r f; base=\"$(basename \"$f\" .py)\"; case \"$f\" in */src/<MODULE>/*.py) if [ \"$base\" != \"__init__\" ] && [ -f \"tests/<MODULE>/test_${base}.py\" ]; then python3 -m unittest \"tests.<MODULE>.test_${base}\"; fi ;; esac; }",
  "statusMessage": "Running matching unit test..."
}
```

Substitute `<MODULE>` with the package path that maps between your implementation and
test directories. Adapt for other languages: replace `python3 -m unittest` with
`jest --testPathPattern`, `go test ./...`, `cargo test`, etc.

This pattern is the highest-leverage hook in the set. Every edit to an implementation
file immediately reveals whether its unit tests still pass, without any manual step.

**Pattern 3 — YAML/JSON config syntax validation on save**

Catch syntax errors in configuration files immediately on save, before they surface
as cryptic runtime failures:

```json
{
  "type": "command",
  "command": "jq -r '.tool_input.file_path // .tool_response.filePath' | { read -r f; case \"$f\" in *.yml|*.yaml) python3 -c \"import sys, yaml; yaml.safe_load(open(sys.argv[1]))\" \"$f\" && echo \"YAML OK: $f\" || echo \"YAML INVALID: $f\" ;; *.json) python3 -c \"import sys, json; json.load(open(sys.argv[1]))\" \"$f\" && echo \"JSON OK: $f\" || echo \"JSON INVALID: $f\" ;; esac; }",
  "statusMessage": "Validating config syntax..."
}
```

Replace the file glob with the specific path prefix you want to watch (e.g.
`infra/*.yml`, `config/*.json`). Broadening it to all YAML/JSON is usually fine —
syntax validation is fast and always useful.

**Pattern 4 — Config consistency contract test on paired file change**

When two files must stay in sync (e.g. a config file and an environment variable
declaration, or an API schema and a mock fixture), run a contract test automatically
whenever either is modified:

```json
{
  "type": "command",
  "command": "jq -r '.tool_input.file_path // .tool_response.filePath' | { read -r f; case \"$f\" in */<CONFIG_FILE_A>|*/<CONFIG_FILE_B>) <TEST_COMMAND> ;; esac; }",
  "statusMessage": "Checking config consistency..."
}
```

Concrete examples of paired file relationships worth protecting with this hook:
- `docker-compose.yml` ↔ `.env.example` (all services have their required vars)
- `openapi.yaml` ↔ `tests/test_api_contract.py` (schema matches test fixtures)
- `Makefile` ↔ `README.md` (documented commands actually exist)
- a generated file ↔ its generator (regenerate to a temp file and diff)

**Pattern 5 — Block a write before it happens (`PreToolUse`)**

The four patterns above all report *after* the fact, which §5.4 notes is the limitation of
`PostToolUse`. For the small number of paths where a write must be **prevented** rather than
noticed, `PreToolUse` is the exception worth using:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "python3 scripts/hooks/protect.py",
            "statusMessage": "Checking protected paths..."
          }
        ]
      }
    ]
  }
}
```

The script reads the event on stdin and **exits 2 to block**, writing its reason to stderr,
where the assistant will read it. Exit 0 allows the write. Never block on an event that
failed to parse — an unparseable event should allow, or a malformed payload becomes an
outage.

Worth protecting this way: test fixtures and golden files that results are judged against,
generated files that should only ever be produced by their generator, vendored code, and
lockfiles. The distinguishing question is whether a well-meaning edit does damage that a
later report cannot undo. Keep the list short — a gate that fires on ordinary work gets
disabled, and then nothing is protected.

### 5.3 Generic settings.json Template

Copy `.claude/generic-settings.json` from this repo and customise for a new project.
The file contains the four `PostToolUse` patterns with placeholder paths and commands that
document what to replace. (It predates Pattern 5; add the `PreToolUse` block from §5.2 by
hand if the project has paths worth protecting.)

See the file at `.claude/generic-settings.json` for the full template. Rename it to
`.claude/settings.json` in the target project after filling in the project-specific
paths and commands.

### 5.4 What Hooks Cannot Do

- They cannot modify the file that triggered them mid-write (the tool call is already
  complete by the time the hook fires).
- They cannot block a write retroactively; they can only surface a failure after the
  fact. Use `PreToolUse` hooks if you need to gate a write (e.g. prevent writes to
  certain paths).
- They do not run in parallel — hooks in the same array run sequentially. Keep
  individual hooks fast (< 2s) to avoid visible lag.
- They run in the project directory, not in a sandboxed environment. A hook that
  mutates the filesystem can cause unintended side effects. Keep hooks read-only
  (lint, test, validate) rather than write-oriented.

---

## Appendix: Reusable Prompt Templates

### A.1 Session-Opening Prompt (New Feature)

```
Read docs/PROJECT_MEMORY.md in full.
Read memory/features/feature-NN-name.md.

Task: Implement [function/service/endpoint name] per ADR-NNN.

Input contracts (what this component reads): [list types and sources]
Output contracts (what this component writes/returns): [list types and destinations]

Constraints:
  - Obtain [DatabaseClient | CacheClient | ExternalApiClient] via get_X() factory only.
  - Do not modify any existing ADR, API schema, or database column without first
    recording the change as a new or amended ADR in docs/PROJECT_MEMORY.md.
  - Mock target in tests: @patch("src.[module].[path].get_X")
  - The function signature is: def [name]([typed args]) -> [return type]:

Definition of Done:
  1. PMA updated in the same response as the code.
  2. All Gherkin scenarios implemented and passing.
  3. Unit test covers the contract (input → output + side effects).
  4. `make test` output pasted (must be N/N passing, 0 errors).
```

### A.2 Session-Opening Prompt (Bug Fix)

```
Read docs/PROJECT_MEMORY.md §[ADR section] and §[Open Questions section].

Bug: [symptom] — [full error message or stack trace]

Before proposing a fix:
  1. Identify which ADR's contract the bug violates, if any.
  2. Identify whether the fix changes an API contract, data schema, or external
     interface. If it does, record a new or amended ADR first.
  3. If the fix does not change a contract, implement it directly.

After fix: run `make test` and paste the full output. Confirm the specific
  error from the bug report no longer appears.
```

### A.3 Pre-Deploy Checklist

```
[ ] 1. make test — all passing at 100% in deterministic tier
[ ] 2. make lint — no violations (gateway, import, type checks)
[ ] 3. Database migrations reviewed:
       [ ] Each migration has a down() function
       [ ] No column renames in a single migration
       [ ] No data migrations mixed with schema migrations
       [ ] No CREATE INDEX CONCURRENTLY inside a transaction
[ ] 4. New environment variables:
       [ ] Added to .env.example with placeholder value
       [ ] Added to REQUIRED_VARS in check_env script
       [ ] Documented in relevant ADR
[ ] 5. API contract changes:
       [ ] Additive only (new fields with defaults)
       [ ] Versioned endpoint if breaking change is unavoidable
       [ ] Client consumers identified and notified
[ ] 6. Rollback plan documented:
       [ ] Migration down() tested in staging
       [ ] Feature flag available if incremental rollout needed
       [ ] On-call engineer aware of the deploy window
[ ] 7. Observability verified in staging:
       [ ] New endpoints appear in trace dashboard
       [ ] Error rate baseline recorded before deploy
       [ ] Alerts configured for new SLO thresholds
```

### A.4 ADR Template

```markdown
### ADR-NNN: [Title]

**Date:** YYYY-MM-DD
**Status:** Accepted

**Context:**
[What problem or gap triggered this decision. What options were considered.]

**Decision:**
[The concrete, binding choice. Written in the present tense: "We use X for Y."]

**Consequences:**
[What this decision constrains. What future decisions must not contradict it.
What migrations would be required if this decision is reversed.]

**Alternatives considered:**
[Why the alternatives were rejected. This is the section that prevents the
same debate from happening again in 6 months.]
```
