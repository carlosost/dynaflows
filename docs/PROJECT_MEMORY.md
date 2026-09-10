# PROJECT_MEMORY.md — `dynaflows`

**Status:** Seed document. Written before any implementation, per §1.1 of `GENERAL_ENGINEERING_PLAYBOOK.md`.
**Last updated:** 2026-09-10 (rev 3)
**Rule:** append-only for decisions. Superseded ADRs are marked `Superseded`, never deleted.

> **This file is the single source of truth for the architecture.**
>
> One other document describes it: a published HTML page, *Dynaflows Architecture*, at
> `https://claude.ai/code/artifact/c31aa5fc-4349-423e-b8cc-7066d435def4`. It is a **frozen snapshot
> of 2026-08-30** — dated, banner-marked, and deliberately not maintained. It carries the node-map
> diagram and the narrative argument for each decision, which is why it was kept; it is not updated,
> which is why it cannot be trusted against this file.
>
> **Do not edit it, do not sync it, and do not treat anything on it as current.** If a future session
> finds it and is tempted to bring it up to date, the answer is no: two hand-maintained copies of one
> architecture is AP-19 with extra steps. If it ever needs to be current, it gets *generated* from
> this file with a drift check (AP-19 habit 2) — and that generator waits until it has a real reader
> (AP-11).

---

## 0. Project Charter

**Problem.** Claude Code's "dynamic workflows" decompose a large task into parallel sub-agents,
verify their output adversarially, and synthesize a single answer — with checkpointed resume and a
confirmation gate before the fan-out runs. The mechanism is closed. There is no local, inspectable,
cost-controlled implementation an engineer can run against their own model budget and their own
knowledge base.

**What we are building.** `dynaflows` — a Python CLI that reconstructs those orchestration patterns on
LangGraph, one pattern at a time, with full LangSmith tracing, OpenRouter model selection per node,
a human gate before expensive work, and the engineer's own Markdown knowledge base injected as
runtime context.

**Who it is for.** A single senior engineer running high-stakes analysis tasks (audits, migration
planning, design stress-testing) on their own codebase, who needs to see exactly which model made
which decision and what it cost.

**Explicit non-goals.**
- Not a coding agent. `dynaflows` does not edit files in a target repository in Phases 0–5.
- Not a hosted service. Single-process CLI, local SQLite, no server.
- Not a Claude Code replacement. It is a study instrument that happens to be useful.

---

## 1. Architecture Decision Records

**Review status.** ADR-001 (static topology, dynamism in plan data), ADR-003 (routing layer deferred
to Phase 3 because the worker catalogue dominates its taxonomy) and ADR-004 (Phase 1's evaluator is a
structural gate, not an LLM judge) were **reviewed and accepted by Carlos on 2026-09-10**. They were
written as overrides of the originally requested node chain, so the acceptance is recorded here
rather than left implicit — a superseding ADR now has to argue against an accepted decision, not
against an unreviewed proposal.

### ADR-001: The dynamism is in the plan data, not in the graph topology

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
Claude Code's dynamic workflows work by having the model *write orchestration code at runtime* — it
emits a script that spawns N sub-agents with a topology chosen per task. LangGraph does not work this
way. A `StateGraph` is declared and `compile()`d before execution; nodes and edges are fixed from that
point. Reproducing runtime-authored orchestration on LangGraph would require generating Python and
executing it, which is an arbitrary-code-execution surface, not a workflow feature.

Three dimensions of a compiled LangGraph *are* runtime-variable, and together they cover the
observable behaviour of every pattern in the two source articles:

| Runtime-variable dimension | LangGraph mechanism | Which patterns need it |
|---|---|---|
| Fan-out **width** (N decided by data) | `Send` from a conditional edge | Fan-out-and-Synthesize, Adversarial Verification, Generate-and-Filter, Tournament |
| **Path selection** through declared edges | conditional edges | Classify-and-Act |
| **Iteration count** (data-dependent stop) | cycle + stop predicate | Loop Until Done |

**Decision.**
The graph topology is static and declared in source. Everything the Planner "decides" is expressed as
**typed plan data** — a list of tasks, each naming a capability from a pre-registered worker catalog —
not as generated topology. Worker behaviour varies by prompt, model tier, tool grant and injected
context; it does not vary by node identity.

**Consequences.**
- Adding a genuinely new *kind* of work is a code change (register a capability), not a prompt change.
  This is the cost we accept in exchange for a graph that can be tested, traced and reasoned about.
- The worker catalog becomes a load-bearing contract. It is versioned; `plan_hash` in the trace
  includes the catalog version, so a plan can never be compared across incompatible catalogs.
- Any future request to "let the planner invent a new node type" must supersede this ADR and bring a
  sandbox design with it.

**Alternatives considered.**
*Generate-and-exec Python orchestration.* Rejected. This turns the project from a workflow study into
a sandboxing study; the sandbox would be the hardest and least interesting component in the system,
and getting it 95% right is the same as getting it wrong.
*A dynamic sub-graph builder that compiles a fresh graph per run.* Rejected for Phase 1–5: it defeats
checkpoint compatibility (a resumed run must attach to the same compiled graph) and makes traces
non-comparable between runs. Revisit only if the anchor "we need per-run topology" is ever backed by
a task we could not express as plan data.

---

### ADR-002: Phase 1 implements Fan-out-and-Synthesize, and nothing else

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
The articles describe six patterns: Classify-and-Act, Fan-out-and-Synthesize, Adversarial
Verification, Generate-and-Filter, Tournament, Loop Until Done. We build one at a time (project
constraint). The question is which one carries the foundation.

Mapped against the structural machinery each pattern requires:

| Pattern | Send fan-out | State reducers | Concurrency control | Partial-failure handling | Checkpoint pressure | Cycle + stop predicate |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **Fan-out-and-Synthesize** | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Adversarial Verification | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Generate-and-Filter | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Tournament | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Loop Until Done | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Classify-and-Act | — | — | — | — | — | — |

Five of six patterns are topological variants of one spine: *a variable-width parallel map, followed
by a barrier, followed by a reduce.* Adversarial Verification is that spine with the previous stage's
output as its map input. Generate-and-Filter is that spine with a filtering reducer. Tournament is
that spine with a pairwise reduce run to a fixpoint. Loop Until Done is that spine inside a cycle.

Classify-and-Act is the outlier: it is a conditional edge and a prompt. It exercises none of the
machinery, and building it first would produce a CLI that looks like it works while none of
requirements 6 (resiliency under concurrency) or 7 (state management under interruption) had been
tested by anything.

**Decision.**
Phase 1 = **Prompt Enhancer → HITL gate → Planner → HITL gate → parallel Workers (`Send`) →
Structural Evaluator → Synthesizer.** Fan-out-and-Synthesize is the Phase 1 pattern. All other
patterns are deferred to Phases 2–5 and are additive to this spine.

**Consequences.**
- The hard failure modes (429 under concurrency, one branch failing among sixteen, malformed JSON at
  scale, checkpoint size growth) are met in Phase 1 rather than retrofitted. This is deliberate: the
  resiliency layer designed under real fan-out pressure is a different and better layer than one
  designed against a single sequential call.
- Phases 2–5 are cheap because the spine already exists. Phase 5 (Generate-and-Filter, Tournament)
  should land in days, not weeks.

**Alternatives considered.**
*Classify-and-Act first, to get a working CLI faster.* Rejected — see the table. It buys a demo and
defers every real risk.

---

### ADR-003: The Routing Layer is deferred to Phase 3, because the worker catalog dominates it

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
The requested workflow shape is Enhancer → **Router** → Planner → Workers → Evaluator → Synthesizer.
Playbook §1.7: some open questions *dominate* others — the answer to A determines what a sensible
answer to B even is, and deciding B first decides A silently.

Here:
- **A:** what capabilities do workers have? (the worker catalog)
- **B:** what classes does the router discriminate between?

These look independent and are not. A router built first must invent its own taxonomy of request
types, and that taxonomy then *becomes* the worker contract by default — the classifier picking the
architecture, which is backwards. Build the catalog first and the router's classes fall out of it
as a projection, testable against a fixed set.

**Decision.**
Phase 1 ships no router. The edge from the plan gate to fan-out is unconditional. The Routing Layer
is Phase 3, after the worker catalog has been shaped by two phases of real use.

**Consequences.**
- Phase 1 runs every request through the full planner path, including trivial ones. This is a known,
  accepted cost — measured in Phase 1 traces, and it produces the data that tells us what the router's
  classes should be.
- Reversal condition: implement the router when traces show ≥30% of runs producing a plan of one task
  (a plan of one is a request that never needed a planner).

---

### ADR-004: Phase 1's evaluator is a structural gate, not an LLM judge

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
The requested shape includes a "Global Evaluator" before synthesis. An LLM judge grading its own
pipeline's output, with no rubric grounded in anything outside the pipeline, is decoration: it
produces a score nobody acts on and adds a top-tier call per run. Worse, playbook AP-19 — a document
(or a metric) that asserts quality the system does not actually have is worse than none, because it
is believed.

**Decision.**
Phase 1's `evaluate` node is a **deterministic structural gate**, no LLM call:
1. Every plan task has exactly one `WorkerResult`.
2. Every result is schema-valid or explicitly `status="failed"`.
3. `failed_count / task_count` is below `EVAL_MAX_FAILURE_RATE` (default 0.34 — placeholder, §4.5).
4. No result is empty-but-ok.

Failing the gate routes to `synthesize` with a `degraded=True` flag that the synthesizer must
surface in its output, not to a silent retry.

The LLM judge arrives in **Phase 2**, as the Adversarial Verification pattern, where it has a real
rubric and a per-task target — and where it comes from a different model family than the worker it
judges (ADR-006).

**Consequences.**
- Phase 1's evaluator is fast, free and fully deterministic — it can be unit-tested in the
  deterministic tier with no mocking of an LLM at all.
- The name `evaluate` is reserved now and its contract widens in Phase 2. Widening is additive:
  `EvaluationReport` gains fields with defaults, per playbook §2.4.

---

### ADR-005: Two human gates; the plan gate is the load-bearing one

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
The requirement specifies a HITL gate on the *enhanced prompt*. That gate is correct and cheap, but
it is not the gate that protects the budget. Approving a well-worded prompt does not tell you that
the planner is about to spawn nineteen workers against the wrong twelve files. Claude Code itself
gates on *what is about to run*, not on the phrasing of the request.

**Decision.**
Two `interrupt()` points:

| Gate | Node | Payload shown | Default behaviour |
|---|---|---|---|
| **G1 — prompt** | `approve_prompt` | original vs enhanced prompt, diff-highlighted | always interrupts; `--yes-prompt` skips |
| **G2 — plan** | `approve_plan` | task list, per-task capability + model tier + anchors, estimated token & USD cost | interrupts when `estimated_cost_usd > HITL_PLAN_THRESHOLD_USD` (default 0.50) or `--review-plan` |

Both gates accept `approve` / `edit` / `reject`. `edit` returns the human's replacement text (G1) or
an edited task list (G2) and re-enters the same node's downstream path without re-running the LLM.

**Consequences.**
- The `interrupt()` re-execution semantics force a node split (ADR-007). This is not optional.
- G2 rejections are the highest-value training signal the system produces and are captured as
  LangSmith feedback (ADR-009).

---

### ADR-006: Model tier is assigned by fan-out multiplier, and the verifier is family-diverse

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
The intuitive allocation — cheap planner, good workers — is backwards. Cost scales with the fan-out
multiplier; quality risk scales with blast radius. The planner runs once per run and its output
determines the cost and correctness of N worker calls. The worker runs N times and its output is
checked by an evaluator.

**Decision — the cost-asymmetry rule.** Model tier is assigned by *calls-per-run multiplier*, not by
how important the node feels:

| Node | Calls/run | Dominant requirement | Tier |
|---|---|---|---|
| `enhance_prompt` | 1 | instruction-following, latency, strict JSON | **Small** |
| `plan` | 1 | decomposition, schema adherence, judgement | **Frontier** — the one place we do not economize |
| `route` (Phase 3) | 1 | enum classification, latency | **Small** |
| `worker` | **N** | throughput × context; dominates total cost | **Mid**, cost-optimized |
| `verify` (Phase 2) | **N** | adversarial reading | **Mid-high, different family from `worker`** |
| `synthesize` | 1 | long-context aggregation, prose quality | **Mid-high, long context** |

**Decision — the diversity rule.** The verifier must not be the same model family as the worker it
verifies. A same-family verifier shares training data, tokenization and failure modes with the thing
it is checking, and rubber-stamps. The synthesis/verification barrier only works as a structural
guard against self-preferential bias if the two sides are structurally different.

**Decision — capability-homogeneous fallback chains.** Every model in a tier's fallback chain must
support `response_format: {type: "json_schema"}`. A price-ordered chain that degrades to a model
without structured-output support converts a rate-limit into a parse failure — it turns a recoverable
error into a wrong answer. Enforced two ways: `provider.require_parameters: true` on every request,
and a startup capability probe (`dynaflows doctor`) against
`/api/v1/models?supported_parameters=structured_outputs`.

**Decision — model ids are pinned, never floating.** No `:latest`, no auto-router aliases. A silent
upstream model swap invalidates every LangSmith baseline with no error and no log line — AP-05 in its
purest form. Model ids live in `config/models.toml`, one file, versioned, hashed into `plan_hash`.

**Consequences.**
- Tier→model assignment is configuration, not code. Swapping a tier is a TOML edit plus a
  `dynaflows doctor` run.
- The concrete model ids in `models.toml` are **placeholders until benchmarked** (§4.5). Phase 1's
  DoD includes recording a baseline of cost/latency/schema-failure-rate per tier on a fixed
  10-request evaluation set.

**Amendment, 2026-09-10 (Phase 0). `config/models.toml` ships with empty chains, on purpose.**

Phase 0 intended to seed each tier with candidate ids. It does not, and the reason is worth keeping.
The provider catalogue could not be reached from the build environment, which left exactly two
options: write ids from memory, or write none. Ids from memory is AP-05 in its purest form — a
renamed or retired id is ignored silently, with no error and no log line, and it invalidates every
baseline recorded against it. A config that is *honestly empty* fails loudly; a config that is
*confidently wrong* does not.

So the tier→model decision moved from a static guess to a command:

- `dynaflows models` lists the live catalogue, already filtered to endpoints that support
  `response_format: json_schema`.
- `dynaflows models --suggest` prints a pasteable TOML block with prices and context lengths.
- `dynaflows doctor` re-validates every configured id against that live list and fails if any has
  disappeared or lost structured-output support.

This makes OQ-01 answerable by running something rather than by asserting something, which is the
§1.3 rule (reject — and choose — with a measurement) applied to configuration. `doctor` reports an
unpopulated registry as **WARN, not FAIL**: "not configured yet" and "misconfigured" need different
actions and must not share a counter (AP-20).

Cost order is explicitly *not* the recommendation `--suggest` makes. Tier assignment follows the
fan-out multiplier, and the verifier's family diversity is a constraint price cannot express; both
are stated in the tool's own output so the next person does not read the cheapest column as advice.

**Amendment 2, same day, after the first real `--suggest` output was pasted into a config.**

Three defects surfaced the moment a human used the command as intended. All three are recorded
because each is a different failure of the same kind — a rule that had never been exercised.

1. **`--suggest` recommended the planner by descending price.** Most expensive is a legacy-premium
   heuristic, not a capability signal, and it proposed a chain whose primary cost $150/$600 per
   million tokens with an 8k-context fallback behind it. Nothing in the provider catalogue measures
   reasoning quality, so the command now **refuses to choose the frontier tier**: it prints
   candidates commented out and says why the decision is not derivable from this data. It also
   excludes `:free` endpoints from the worker tier — those are rate-limited hard, and the worker
   tier is the one that runs N times, so "free" there buys 429s rather than savings. That is OQ-03
   arriving early, from the direction nobody was watching.

2. **The family-diversity rule only checked the verifier's *preferred* model.** A verifier that fell
   back into the worker's family passed silently — precisely when the fallback mattered. The rule now
   compares whole chains.

3. **Context length was not treated as a capability.** ADR-006 required capability-homogeneous
   chains and checked only structured-output support, so an 8k fallback behind a 200k primary was
   legal. It is now checked — with a caveat worth keeping: the first implementation used a *ratio*
   of the primary's context, which made a deliberately large primary flag every ordinary fallback.
   A gate that fires on ordinary work gets disabled, and then nothing is protected (§5.2, Pattern 5).
   It is now an absolute `min_context_tokens` floor, **defaulting to 0 = disabled**, which §4.5
   prescribes directly: implement and test the mechanism, treat the threshold as provisional until
   real traffic sets it. `doctor` reports the disabled state as WARN, never OK — a skipped check must
   not read as a passing one.

---

### ADR-007: Every `interrupt()` lives alone in a pure node

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
LangGraph resumes an interrupted graph by **re-executing the interrupted node from its first line** —
it does not resume at the `interrupt()` call. Any work performed before the `interrupt()` in that node
runs again on every resume. If the enhancer's LLM call and its approval gate share a node, every
resume pays for a second enhancement, and (worse) may show the human a *different* enhanced prompt
than the one they were approving.

**Decision.**
`interrupt()` appears only in nodes that are otherwise pure: they read state, format a payload, call
`interrupt()`, and write the human's answer back. All LLM calls and all side effects happen in a
preceding node whose output is checkpointed before the gate is reached.

```
enhance_prompt   (LLM call, writes state)  →  approve_prompt   (interrupt only)
plan             (LLM call, writes state)  →  approve_plan     (interrupt only)
```

**Consequences.**
- Re-execution of a gate node is free and idempotent by construction.
- This rule is testable and therefore enforced: an AST lint in CI rejects any node function that both
  calls `interrupt` and performs I/O. Per AP-19, this ADR names the check that proves it —
  `tests/architecture/test_interrupt_node_purity.py`.

---

### ADR-008: State carries references; artifacts live in a content-addressed run store

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
LangGraph checkpoints the full state at every superstep. With sixteen parallel workers writing raw
model output into state, each superstep serializes and writes every worker's full text to SQLite —
repeatedly, since later supersteps still carry it. Checkpoint size grows with N × output length,
resume gets slower as the run progresses, and SQLite's single-writer model turns the fan-in into a
serialized write storm.

**Decision.**
Worker nodes write a **structured summary plus an `ArtifactRef`** to state. Raw output goes to
`.dynaflows/runs/<run_id>/artifacts/<sha256>.md`. State carries
`ArtifactRef(sha, path, kind, tokens, preview)` where `preview` is capped at 280 characters.

The checkpointer is `AsyncSqliteSaver` at `.dynaflows/state.db` with `PRAGMA journal_mode=WAL`.

**Consequences.**
- Checkpoint size is bounded by plan size, not by output size.
- The synthesizer must *load* artifacts by ref rather than read them from state. That load is
  budgeted and traced like any other context assembly.
- The run store is the natural place for the "what did this run actually produce" CLI command, and it
  survives `state.db` deletion.
- **Threshold to measure (§4.5):** checkpoint row size at p95, and fan-in write latency at N=4, 8, 16,
  32. WAL + serialized writes is assumed adequate at our scale and is unproven until measured.

---

### ADR-009: Playbook retrieval is anchor lookup first, BM25 second, and never embeddings in Phase 1

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
The runtime knowledge base is `GENERAL_ENGINEERING_PLAYBOOK.md` (~1,700 lines, ~60 leaf sections) plus
an optional Obsidian vault. The reflex is a vector store. Applied to this corpus, that reflex is wrong
on three counts: the corpus is small enough that semantic recall is not the bottleneck; it is
**authored with stable human-readable identifiers** (`AP-01`…`AP-20`, `§1.1`…`§5.4`, `ADR-NNN`) that a
model can name directly; and an embedding index introduces a staleness failure mode plus
nondeterminism into every trace, in a system whose entire value proposition is traceability.

**Decision.**
Three-layer mechanism, in order of priority:

1. **Anchor lookup (primary, deterministic, zero-LLM).** The Planner's output schema requires, per
   task, `playbook_anchors: list[str]`. The planner chooses them from a **catalog** injected into its
   prompt — heading paths + anchor ids + one-line summaries for all sections, ~1.5k tokens, *not* the
   document. One Frontier-tier call therefore routes context for all N workers, and each worker gets an
   exact, reproducible slice. `PlaybookRepository.by_anchor(ids) -> list[Chunk]`.
2. **BM25 search (fallback).** SQLite **FTS5** virtual table over the same chunk rows, ranked with the
   built-in `bm25()`. Zero new dependencies — SQLite is already in the stack for checkpointing.
   Used when the planner names no anchors, and to expand a thin anchor set.
3. **Deterministic packing.** `pack(chunks, budget_tokens) -> ContextPack`: dedupe by chunk id, order
   anchor-hits before search-hits, fill to budget, truncate at paragraph boundaries, always prepend the
   full heading path (`4. Production Readiness > 4.4 Anti-Pattern Catalog > AP-11: …`) so a truncated
   chunk still says what it is. `ContextPack` records `included_ids` and `dropped_ids`, and both go
   into the LangSmith trace — so "what did this node actually see" is answerable from the trace alone.

**Chunking.** Markdown is parsed to an AST (`markdown-it-py`), chunked on **heading structure**, never
on fixed token windows. One chunk per leaf section. Obsidian `[[wikilinks]]` and frontmatter tags are
extracted into an adjacency table for optional 1-hop expansion.

**Index integrity.** `sources(path, file_sha256)` is stored alongside the chunks. `dynaflows index --check`
recomputes and exits non-zero on drift. Wired as a CI step and as a `PostToolUse` hook — this is
playbook AP-19's "generated file plus a drift check makes staleness impossible rather than unlikely."

**Consequences.**
- Retrieval is reproducible: the same plan produces the same context bytes. Two runs that differ can
  be diffed.
- The planner acquires a second responsibility (context routing) and its prompt grows by the catalog.
  Accepted: it is one call at Frontier tier, and it is the cheapest place in the system to make this
  decision well.
- **Negative decision with a reversal condition** (§1.3 — reject with a measurement): no embeddings,
  no vector store. Revisit when either the corpus exceeds **~2,000 chunks**, or trace data shows the
  **anchor-hit rate below 60%** of retrievals — meaning the planner cannot name what it needs and
  genuine semantic recall is the gap.

---

### ADR-010: All provider traffic goes through one Gateway module, enforced by lint

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
Playbook §3.1/§3.2: every external dependency is routed through a chokepoint that owns construction,
configuration, tracing, timeouts, retry and fallback. Concurrency limiting in particular cannot live
at the node level: LangGraph's `max_concurrency` bounds *branch* parallelism, but a single node may
make several provider calls, and the provider rate-limits *requests per API key across the process*.
Two different questions need two different limits.

**Decision.**
`src/dynaflows/gateway/openrouter.py` is the sole importer of the HTTP/LLM client. It owns, in order:

1. **Budget check** — per-run USD and token ceilings. Exceeding raises `BudgetExceeded`, which halts
   the graph to a *resumable checkpoint*, not a crash.
2. **Global semaphore** — `asyncio.Semaphore(OPENROUTER_MAX_CONCURRENT)`, per API key, process-global.
   Layered under LangGraph's `max_concurrency`, which stays as the branch-level limit.
3. **Timeout** — `asyncio.wait_for` on every call (§3.3). No unbounded provider call exists.
4. **Retry** — `tenacity`, exponential backoff with full jitter on 429/5xx/timeout, honoring
   `Retry-After`. Max 3 attempts (placeholder, §4.5).
5. **Tier fallback** — on exhausted retries, or 402/model-unavailable, advance to the next
   capability-homogeneous model in the chain. `fallback_depth` is stamped on the trace; a rising
   fallback rate is a leading indicator of a quality regression and is alertable.
6. **Circuit breaker** — K consecutive failures on a model id marks it unhealthy for T seconds and
   skips it in chains.
7. **JSON repair ladder** — (a) `response_format: json_schema` + `require_parameters: true`;
   (b) on `ValidationError`, exactly one repair call at **Small** tier with the validation error
   appended, because repairing JSON is a formatting task, not a reasoning task; (c) on second failure,
   return `status="failed"` — **never raise into the graph.**

**Decision — failures and exclusions are counted apart (AP-20).** Two sinks, two counters, two
thresholds:

| Counter | Meaning | Alertable |
|---|---|---|
| `schema_failure` | The model produced output we could not parse. Something is wrong. | Yes |
| `task_skipped` | The task was well-formed and deliberately not executed (budget cap, circuit breaker, human edit removed it). | No — but the count is a signal |

Merging these would make the failure rate unreadable in both directions.

**Consequences.**
- A worker raising an exception inside a `Send` branch aborts the entire superstep. Therefore: worker
  nodes **must not raise**; they return `WorkerResult(status=ok|degraded|failed)`. This is a hard
  contract, and it is what makes partial failure survivable.
- Enforced by an architectural lint in CI and as a `PostToolUse` hook (playbook §5.2 Pattern 1):
  any import of the provider SDK or `httpx` outside `src/dynaflows/gateway/` fails the build. The check
  that proves this ADR is `tests/architecture/test_gateway_boundary.py` (AP-19 habit 1).

---

### ADR-011: LangSmith is the primary telemetry surface; local JSONL joins to it by run id

**Date:** 2026-08-30
**Status:** Accepted

**Context.**
Requirement 2 asks for full traceability of every step, LLM call and state transition. Playbook §4.1
asks for structured logs with a propagated `trace_id`, and three metrics (rate, error rate, latency).
These are the same requirement viewed from two sides.

**Decision.**
`LANGSMITH_TRACING=true` from the first commit, before any workflow node exists. Every node receives a
`RunnableConfig` carrying run-scoped metadata:

```
run_id, thread_id, plan_hash, catalog_version, node, task_id,
model_id, tier, fallback_depth, attempt, playbook_anchors,
context_pack_tokens, context_dropped_ids
```

Gateway calls that are not LangChain runnables are wrapped with `@traceable` so no provider call is
invisible. Local structured logs go to `.dynaflows/runs/<run_id>/run.jsonl` with `trace_id` set to the
LangSmith run id, making local logs joinable to the hosted trace.

Playbook §4.1's three metrics, mapped: **rate** → calls per run per tier; **error rate** →
`schema_failure_rate` and `fallback_rate` (not merged, per ADR-010); **latency** → p50/p95/p99 per
node type.

**Decision — the feedback loop.** Every G2 plan rejection or edit writes LangSmith feedback
(`key="plan_accepted"`, `score=0`, `correction=<the human's edited plan>`). Normal use therefore
produces a labelled planner evaluation dataset at zero marginal cost. This is the single highest-value
observability decision in the project.

**Consequences.**
- The system is unusable without network access to LangSmith in Phase 1. Accepted; a local-only
  tracing backend is an explicit open question (OQ-04), not a Phase 1 requirement.
- Metadata propagation is a contract: a node that drops the config it was handed silently breaks the
  trace. Covered by `tests/architecture/test_config_propagation.py`.

---

### ADR-012: Python 3.14, and the dependency floor is verified by resolution, not by assumption

**Date:** 2026-08-31
**Status:** Accepted

**Context.**
The scaffold arrived pinned to `requires-python = ">=3.14"` via `.python-version`. A runtime pin has
wide blast radius — it decides which wheels exist, which CI images are usable, and whether a
contributor can build the project at all — so it belongs in an ADR (§1.3). The risk with a very new
interpreter is not resolution but **installation**: a tree can resolve and then fail to build because
a transitive dependency has no wheel for the interpreter/arch combination.

Playbook AP-19 habit 3 says to verify claims about the environment by *running* them, because a
negative result has several possible causes and no diagnostic value on its own. So this was run, not
reasoned about.

**Verification performed (2026-08-31, aarch64 Linux, uv 0.12.3).** `uv lock` + `uv sync` on the full
intended tree, then an import and capability probe:

| Package | Resolved | Notes |
|---|---|---|
| `langgraph` | 1.2.11 | `StateGraph`, `Send`, `interrupt`, `Command` all import |
| `langgraph-checkpoint-sqlite` | 3.1.1 | `AsyncSqliteSaver` importable — ADR-008's checkpointer exists |
| `langchain-core` | 1.6.1 | |
| `langchain-openai` | 1.6.0 | OpenRouter is reached through its OpenAI-compatible endpoint |
| `langsmith` | 0.11.2 | |
| `pydantic` | 2.13.5 | |
| `tenacity` | 9.1.4 | ADR-010 retry layer |
| `rich` | 15.0.0 | |
| `typer` | 0.27.2 | |
| `markdown-it-py` | 4.2.0 | ADR-009 chunker |
| `aiosqlite` | 0.22.1 | |
| stdlib `sqlite3` | 3.53.1 | **FTS5 compiled in; `bm25()` available** — ADR-009's retrieval layer needs no new dependency |

56 packages resolved, full sync succeeded, zero source builds required.

**Decision.**
Python 3.14 stands. `.python-version` and `requires-python = ">=3.14"` are kept as the scaffold set
them. All direct dependencies are pinned to exact versions in `uv.lock`, which is committed.

**Consequences.**
- CI must use the same interpreter and the committed lockfile, not a fresh resolve (§4.3 environment
  parity). A CI job that resolves independently is testing a different program.
- The probe above ran on **aarch64 Linux**, not on the target macOS arm64 host. Wheel availability is
  usually identical across those two for this tree, but "usually" is not "verified" — the
  `dynaflows doctor` command re-runs the import probe on whatever host it is executed on, so the claim
  is checked where it matters rather than asserted here.
- Reversal condition: if a dependency the project actually needs turns out to have no 3.14 wheel, the
  floor drops to 3.13 in an amended ADR — it does not get worked around with a source build.

**Alternatives considered.**
*Drop to 3.12 for safety.* Rejected on evidence: the tree installs clean on 3.14, so the caution would
cost the newer interpreter for nothing. This is §1.3's rule — reject with a measurement, and the
measurement is above.

---

## 2. Data Contracts

Written before implementation (§1.4, contract-first). These are the canonical shapes; changes are
additive-only (§2.4) and require an amended ADR.

### 2.1 Graph state

```python
# src/dynaflows/contracts/state.py
from typing import Annotated, Literal, TypedDict
from operator import add


class WorkflowState(TypedDict):
    # --- identity -------------------------------------------------------
    run_id: str
    thread_id: str

    # --- stage 1: enhancement ------------------------------------------
    raw_prompt: str
    enhanced_prompt: str | None
    prompt_gate: GateOutcome | None  # approve | edit | reject

    # --- stage 2: planning ---------------------------------------------
    plan: Plan | None
    plan_hash: str | None
    plan_gate: GateOutcome | None

    # --- stage 3: fan-out ----------------------------------------------
    # Reducer is REQUIRED: parallel Send branches write here concurrently.
    results: Annotated[list[WorkerResult], add]

    # --- stage 4: evaluation / synthesis --------------------------------
    evaluation: EvaluationReport | None
    degraded: bool
    synthesis: ArtifactRef | None

    # --- accounting -----------------------------------------------------
    cost: Annotated[CostLedger, merge_cost]  # custom reducer, monotonic
```

**Invariant.** Any key written by more than one concurrent branch carries a reducer. A key without
one raises `InvalidUpdateError` at runtime under fan-out. `tests/architecture/test_state_reducers.py`
asserts this over the annotated type at import time — it cannot be forgotten.

### 2.2 Plan

```python
class PlanTask(BaseModel):
    task_id: str  # stable, plan-local; used as trace key
    capability: CapabilityId  # MUST exist in the worker catalog
    objective: str  # what this worker must produce
    inputs: list[str]  # file paths / refs, resolved before dispatch
    playbook_anchors: list[str]  # ADR-009 primary retrieval path
    tier_override: Tier | None = None
    depends_on: list[str] = []  # Phase 1: MUST be empty (see OQ-02)


class Plan(BaseModel):
    tasks: list[PlanTask]  # 1..MAX_FANOUT
    rationale: str  # shown at gate G2
    estimated_tokens: int
    estimated_cost_usd: float
```

### 2.3 Worker result

```python
class WorkerResult(BaseModel):
    task_id: str
    status: Literal["ok", "degraded", "failed"]
    summary: str  # <= 1200 chars, goes into state
    artifact: ArtifactRef | None  # raw output, ADR-008
    model_id: str
    tier: Tier
    fallback_depth: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: ErrorEnvelope | None  # typed, never a bare string (§1.4)


class ErrorEnvelope(BaseModel):
    code: Literal[
        "RATE_LIMIT", "TIMEOUT", "SCHEMA_INVALID", "BUDGET_EXCEEDED", "MODEL_UNAVAILABLE", "UNKNOWN"
    ]
    message: str
    attempts: int
```

### 2.4 Playbook index (SQLite, `.dynaflows/playbook.db`)

```sql
CREATE TABLE sources (
    path        TEXT PRIMARY KEY,
    file_sha256 TEXT NOT NULL,
    indexed_at  TEXT NOT NULL
);

CREATE TABLE chunks (
    id           TEXT PRIMARY KEY,     -- sha256(path + heading_path)
    source_path  TEXT NOT NULL REFERENCES sources(path),
    heading_path TEXT NOT NULL,        -- "4. Production Readiness > 4.4 … > AP-11: …"
    anchors      TEXT NOT NULL,        -- JSON array: ["AP-11", "§4.4"]
    tokens       INTEGER NOT NULL,
    body         TEXT NOT NULL
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    heading_path, anchors, body,
    content='chunks', content_rowid='rowid',
    tokenize='porter unicode61'
);

-- DEFERRED — not created in Phase 1. See §3 OQ-05 and the AP-11 note below.
-- CREATE TABLE links (       -- Obsidian [[wikilink]] adjacency
--     from_chunk TEXT NOT NULL REFERENCES chunks(id),
--     to_target  TEXT NOT NULL
-- );
```

**AP-11 note.** The `links` table and 1-hop wikilink expansion have **no caller in Phase 1** — the
Obsidian vault is not a Phase 1 input, and OQ-05 is unresolved. Building the table now would produce
a structure exercised only by its own test. It is deferred until the feature that reads it exists.
The chunker still *extracts* wikilinks (that is parsing, and it is free); it just does not persist an
adjacency structure nothing queries.

### 2.5 Repository interface (§3.2, Protocol + factory)

```python
class PlaybookRepository(Protocol):
    def by_anchor(self, anchors: list[str]) -> list[Chunk]: ...
    def search(self, query: str, k: int) -> list[Chunk]: ...
    def catalog(self) -> str: ...  # heading paths + anchors, for the planner prompt
    def source_drift(self) -> list[str]: ...  # paths whose sha changed since indexing


def get_playbook_repository(db_path: Path | None = None) -> PlaybookRepository:
    """Sole construction path. Returns SqlitePlaybookRepository in production,
    InMemoryPlaybookRepository in the deterministic tier."""
```

**Mock target in tests:** `@patch("src.dynaflows.nodes.planner.get_playbook_repository")` — the factory,
never `sqlite3.connect` (§2.2, AP-02).

---

## 3. Open Questions

**IDs are permanent labels. The decision order below is a separate, changing list** (§1.7).
**No ADR numbers are reserved.** Each question records what it *blocks in the code*.

| ID | Question | Blocks | Status |
|---|---|---|---|
| OQ-01 | Which concrete model ids fill each tier in `models.toml`? | `config/models.toml`; the Phase 1 cost baseline | Open — requires measurement, not opinion |
| OQ-02 | Does a plan need intra-plan task dependencies (`depends_on`), or is a flat map sufficient? | `PlanTask.depends_on`; whether fan-out is one superstep or a scheduler | Open — **deliberately deferred**, see order note |
| OQ-03 | What is `MAX_FANOUT`, and does it derive from the rate limit or the checkpoint write cost? | `Send` dispatch; the G2 cost estimate | Open — measure both, take the lower |
| OQ-04 | Is an offline/local tracing backend required, or is LangSmith a hard dependency? | `gateway/telemetry.py` abstraction — or its absence | Open — **do not build the abstraction until an answer exists** (AP-11) |
| OQ-05 | Does the Obsidian vault need frontmatter-tag filtering, or are wikilinks + FTS5 enough? | `links` table usage; `PlaybookRepository.search` signature | Open |
| OQ-06 | Should `dynaflows` ever write to the target repository, or stay read-only? | The tool-grant model for workers; the entire sandboxing question | Open — a "no" here is a negative ADR worth writing explicitly |
| OQ-07 | Do worker provider calls need an idempotency key, or is checkpoint granularity sufficient to prevent double-billing on crash-resume? | `gateway.call()` signature; whether `WorkerResult` is written before or after the superstep commits | Open — §3.3 requires idempotency on mutating operations, and a paid LLM call is a mutating operation |

**Decision order (current, and it has already been corrected once):**

`OQ-07 → OQ-01 → OQ-03 → OQ-02 → OQ-06 → OQ-05 → OQ-04`

*Re-ordering note (2026-08-30).* The initial order put OQ-01 (model ids) first, because it feels like
the foundational choice. It is not: **OQ-03 dominates it.** `MAX_FANOUT` determines the requests-per-
minute envelope, which determines which providers are viable at the worker tier at all — a model that
is excellent and rate-limited to 20 rpm is not a worker-tier candidate at N=32 no matter how good it
is. Choosing models first would have silently decided the fan-out ceiling by picking a rate limit.
Recording the correction here rather than only the final order, per §1.7.

Similarly, OQ-06 must precede OQ-05: whether workers can write determines what "context" even means,
and therefore what the retrieval layer is for.

*Third re-ordering (2026-09-10, after Phase 0 shipped a gateway design).* **OQ-03 is demoted; it
does not dominate OQ-01 as strongly as claimed.** The original argument was that `MAX_FANOUT` sets
the requests-per-minute envelope and therefore decides which models are viable at the worker tier.
That is only true without a concurrency limiter. ADR-010 puts a process-global semaphore in the
gateway, so a wide fan-out **queues** rather than exceeding the rate limit — plan width and request
concurrency are separate knobs. Rate limits still constrain the *semaphore*, and still rule out
`:free` endpoints for a tier that runs N times, so the coupling is real; it is just not dominating.
The correction is recorded rather than quietly applied, because a re-ordering justified by an
argument that turned out to be weak is exactly the thing a future reader needs to see.

*Second re-ordering (2026-08-31, found during the pre-implementation verification pass).* **OQ-07
leads.** Playbook §3.3 requires an idempotency key on every mutating operation, and a billed provider
call is a mutating operation — retrying one after a crash charges twice. The answer determines the
`gateway.call()` signature, which every node depends on, and it determines whether `WorkerResult` is
written before or after the superstep commits. Deciding `MAX_FANOUT` first would have set the
crash-exposure window without ever asking whether re-execution is safe.

---

## 4. Feature Log

| Feature | Spec file | Phase | Status |
|---|---|---|---|
| F-00 Foundation, doctor, model registry | `memory/features/feature-00-foundation.md` | 0 | **Done** (2026-09-10) |
| F-01 Playbook indexer + repository | `memory/features/feature-01-playbook-index.md` | 1 | Not started |
| F-02 Prompt enhancer + gate G1 | `memory/features/feature-02-enhancer.md` | 1 | Not started |
| F-03 Planner + gate G2 | `memory/features/feature-03-planner.md` | 1 | Not started |
| F-04 Fan-out workers + resiliency | `memory/features/feature-04-fanout.md` | 1 | Not started |
| F-05 Structural evaluator + synthesizer | `memory/features/feature-05-synthesis.md` | 1 | Not started |
| F-06 Rich terminal UX + resume | `memory/features/feature-06-cli-ux.md` | 1 | Not started |
| F-07 Adversarial verification | — | 2 | Not started |
| F-08 Routing layer | — | 3 | Not started |
| F-09 Loop Until Done | — | 4 | Not started |
| F-10 Generate-and-Filter, Tournament | — | 5 | Not started |

---

## 5. Phase 1 Acceptance Criteria (Gherkin)

Written before implementation (§1.6). These are the observable contract.

```gherkin
Feature: Human gate on the enhanced prompt

  Scenario: The human accepts the enhanced prompt
    Given a run started with the raw prompt "audit auth in this service"
    When the enhancer produces an enhanced prompt
    Then the graph interrupts before the planner
    And the CLI displays the original and the enhanced prompt
    When the human approves
    Then the planner runs exactly once
    And the enhancer LLM is called exactly once across the whole run

  Scenario: Resuming a gate does not re-run the enhancer
    Given a run interrupted at the prompt gate
    When the process is killed and the run is resumed by thread id
    Then the enhanced prompt shown is byte-identical to the one shown before
    And the enhancer LLM call count for the run is still exactly one

  Scenario: The human edits the enhanced prompt
    Given a run interrupted at the prompt gate
    When the human submits an edited prompt
    Then the planner receives the edited text, not the model's text
    And the enhancer is not called again

  Scenario: The human rejects
    Given a run interrupted at the prompt gate
    When the human rejects
    Then the graph terminates with no planner call and no worker call
    And the run's total cost equals the enhancer's cost alone
```

```gherkin
Feature: Fan-out survives partial failure

  Scenario: One worker fails among many
    Given an approved plan with 8 tasks
    And the provider returns unparseable output for task 3 on both attempts
    When the fan-out completes
    Then 7 results have status "ok"
    And task 3 has status "failed" with code "SCHEMA_INVALID"
    And the graph reaches the synthesizer                 # no aborted superstep
    And the synthesis explicitly states that task 3 produced no result

  Scenario: Rate limiting does not fail the run
    Given an approved plan with 16 tasks
    And the provider returns 429 for the first 6 requests
    When the fan-out completes
    Then all 16 results have status "ok" or "degraded"
    And the schema_failure counter is 0                   # a 429 is not a schema failure
    And the task_skipped counter is 0                     # nor an exclusion — AP-20

  Scenario: Budget ceiling halts to a resumable state
    Given a per-run budget of 0.10 USD
    And an approved plan whose 12th task would exceed it
    When the fan-out reaches the ceiling
    Then the run halts with code "BUDGET_EXCEEDED"
    And the run is resumable by thread id after the budget is raised
    And no completed worker result is lost
```

```gherkin
Feature: Playbook context is exact and reproducible

  Scenario: The planner names an anchor and the worker receives that section
    Given a plan task with playbook_anchors ["AP-11"]
    When the worker assembles its context
    Then the context contains the AP-11 section body
    And the context contains the heading path "4.4 Anti-Pattern Catalog > AP-11"
    And the context does not contain any other anti-pattern section

  Scenario: Context assembly is deterministic
    Given the same plan and the same playbook file
    When the run is executed twice
    Then the context bytes handed to each worker are identical
    And the ContextPack included_ids recorded in the trace are identical

  Scenario: The index detects source drift
    Given a playbook indexed at sha A
    When the playbook file is edited
    Then "dynaflows index --check" exits non-zero
    And names the drifted path
```

---

## 6. Definition of Done (per feature)

Inherited verbatim from playbook §1.6, with two project-specific additions:

1. PMA updated with any new ADR or contract change, **in the same response as the code**.
2. All Gherkin scenarios have passing step definitions.
3. Unit tests cover the state-transition contract for every new node.
4. `make test` passes 100% in the deterministic tier (no network, no provider, no SQLite file I/O
   outside tmp).
5. Integration tests verify the behaviour end-to-end against a recorded-cassette provider.
6. The feature's spec file is marked `Status: Done`.
7. **(project-specific)** Every new node appears in a LangSmith trace with complete run-scoped
   metadata. A node that runs untraced is not done.
8. **(project-specific)** `make lint-architecture` passes: no provider import outside the gateway, no
   I/O in an interrupt node, no unreduced concurrent state key.

---

## 7. Known Gaps in This Document

Listed explicitly, per AP-19 — a design document that quietly asserts more than it has is worse than
one that names its holes.

- Every numeric threshold in this document (`EVAL_MAX_FAILURE_RATE`, `HITL_PLAN_THRESHOLD_USD`,
  `MAX_FANOUT`, retry counts, context budgets) is a **placeholder**, not a measurement (§4.5).
- ADR-006's tier assignments are reasoning, not benchmark results. OQ-01 is the measurement.
- ADR-008's claim that WAL + serialized writes is adequate at our fan-out is **unverified**. It is an
  intention, not a tested property, until the Phase 1 checkpoint-pressure measurement exists.
- The `dynaflows doctor` **network** checks — OpenRouter auth, the live capability probe, LangSmith
  connectivity, and the schema-enforced handshake — have **never been executed**. They were written
  against the documented API shapes, not against a response. No credentials existed in the build
  environment. Per AP-19 habit 3 they stay unverified until someone runs `make doctor` with real
  keys, and that run is the true Phase 0 exit gate. The offline checks (`--offline`) have been run
  and do work.
- ADR-008's checkpointer path is untested against a real fan-out; only that SQLite is writable and
  has FTS5 is confirmed.
- **Idempotency is unresolved (OQ-07).** ADR-007 makes *gate* re-execution safe. It says nothing about
  a `worker` node re-executing after a crash and re-issuing a billed provider call. §3.3 says this
  needs an idempotency key; this document does not yet specify one. Named here rather than left to be
  discovered by a duplicated bill.
- ADR-004's `EVAL_MAX_FAILURE_RATE` default of 0.34 is arbitrary. It should be derived from what a
  synthesis can usefully degrade to, which is a measurement nobody has taken.
- ADR-012's probe ran on aarch64 Linux, not on the macOS arm64 host this project is developed on.
  Closed only when `dynaflows doctor` has run there.

**Learned on 2026-09-10, second pass — three bugs, one root cause: data crossing a boundary that
interprets it.** `models --suggest` printed through Rich, which read the TOML header `[tiers.small]`
as a style tag and swallowed it, and turned a model id ending `:free` into an emoji because `:free:`
is Rich's emoji shorthand — so the pasted block was invalid TOML and the doctor's own output showed a
corrupted model id. Rich also hard-wraps to the terminal, which would have split an id across lines
on a narrow one. Separately, `doctor`'s environment check called `check_env()` with no argument and
fell through to `os.environ`, ignoring the settings it was handed, and `get_settings()` wrote the
`.env` file into `os.environ` as a side effect — so one call anywhere in a test session leaked a
developer's real credentials into every later test, which is exactly how it was found.

The general rule now applied in both places: **anything that came from outside is data, and must be
passed as data.** Model ids go through Rich as `Text`; pasteable output is printed with markup
disabled and soft wrapping; `parse_dotenv` is pure and composition happens in a visible `resolve_env`;
`Settings` carries `missing_required` computed from the same mapping every other field was read from.

**Learned on 2026-09-10, and worth more than the fixes:** a deterministic-tier test asserted that
the shipped `config/models.toml` was *unpopulated*. It passed continuously and then failed the first
time the project was used correctly, because populating the registry is the intended next step. A
test that encodes a transient state as a permanent invariant punishes progress; the assertion now
checks that the config's `version` string and its chains tell the same story, which is invariant.
The general form: before asserting a fact about a file, ask whether that file is designed to change.

**Closed by verification on 2026-09-10** (Phase 0):
- *"the deterministic tier can be 100% green with no network and no provider"* — 37 tests, 0.09s.
- *"ADR-010's gateway boundary is enforceable by lint"* — `scripts/lint_architecture.py`, and
  `tests/architecture/test_gateway_boundary.py` proves the linter actually catches a violation
  rather than merely passing on clean code.
- *"SQLite FTS5 + bm25 is present"* — now re-checked by `doctor` on whatever machine runs it, not
  only in a one-off probe.

**Closed by verification on 2026-08-31** (moved out of the gap list, kept for the record):
- *"`AsyncSqliteSaver` exists and is importable"* — was an assumption in ADR-008. Now checked.
- *"SQLite ships FTS5 with `bm25()`, so ADR-009 needs no new dependency"* — was the load-bearing claim
  of the whole retrieval design, and was unverified when written. Now checked: `sqlite3` 3.53.1,
  `CREATE VIRTUAL TABLE … USING fts5` succeeds.
- *"The intended dependency tree installs on Python 3.14"* — see ADR-012.
