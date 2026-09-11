# PROJECT_MEMORY.md — `dynaflows`

**Status:** Seed document. Written before any implementation, per §1.1 of `GENERAL_ENGINEERING_PLAYBOOK.md`.
**Last updated:** 2026-09-11 (rev 16)
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

**Amendment, 2026-09-11 (step 1.5). G2 always interrupts in Phase 1; the cost threshold is deferred.**

The table below says G2 interrupts when `estimated_cost_usd > HITL_PLAN_THRESHOLD_USD`. Implementing
that needs a price per model, which needs the provider catalogue, which is a network round trip the
gate should not take — and a **fabricated** price would make the ledger confidently wrong, which is
worse than visibly incomplete (the same reasoning as `PriceBook` in ADR-010).

So Phase 1 gates every plan, with `--yes-plan` to skip. The gate shows the task list, the anchors
each worker will be given, and an estimated **token** count, which is derivable without a network
call. The USD threshold returns in step 1.8 alongside the price data the measurement pass produces.

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

**Implemented 2026-09-10 (step 1.4), and the check this ADR names now exists.**
`scripts/lint_architecture.py` rejects any function that calls `interrupt()` and either `await`s or
calls a known I/O helper; `tests/architecture/test_interrupt_node_purity.py` drives a violating
fixture for each shape, so the rule can fail rather than only pass. The `await` ban is the
load-bearing half — every I/O path in this codebase is async — and the rule exempts nobody, including
the gateway package that ADR-010 does exempt.

The behavioural proof is `test_resuming_does_not_re_run_the_enhancer`: the enhancer call count stays
at 1 across a kill and resume, and the text that proceeds is byte-identical to the text the human was
shown.

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

**Amendment, 2026-09-10 (step 1.3). Checkpointed types must be registered, or resume dies on an
upgrade.**

Running the skeleton printed, to stderr and only to stderr:

> Deserializing unregistered type `dynaflows.contracts.state.GateOutcome` from checkpoint. **This
> will be blocked in a future version.**

That is AP-05 in its purest form — works today, silently stops working after a dependency upgrade,
no error until it is too late — and the casualty is this ADR's entire promise. Not "some runs fail":
**every checkpoint ever written becomes unreadable**, so no interrupted run can ever be resumed.

Fixed by registering our types with the checkpoint serialiser, and two decisions inside the fix are
worth keeping:

- Registered as **class references**, not `(module, name)` strings. A string list can name a class
  that was renamed or deleted and nothing notices; an import fails on the spot.
- Proven under `LANGGRAPH_STRICT_MSGPACK=true`, which makes the future behaviour the present. A test
  round-trips a full state through a real checkpoint with strict mode on, and an architecture test
  asserts the allowlist covers every Pydantic model and enum defined in the contracts modules — so a
  new state type that nobody registers fails the suite now rather than an upgrade later.

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

**Amendment 2, 2026-09-10 — the catalogue is ~4,000 tokens, not ~1,500. The ADR was wrong.**

This ADR asserted the planner catalogue would be "roughly 1.5k tokens, *not* the document". Built and
measured, it is **3,956** — and 6,040 before two redundant columns were removed. AP-19 says a design
document that asserts what the code does not do is worse than one that is missing, because it is
believed; the number is corrected here rather than the code contorted to reach a figure invented
before anything existed.

Measured on 2026-09-10, 84 chunks:

| Source | Chunks | Catalogue tokens |
|---|---:|---:|
| `GENERAL_ENGINEERING_PLAYBOOK.md` | 52 | ~2,497 |
| `PROJECT_MEMORY.md` | 32 | ~1,459 |
| **Total** | **84** | **~3,956** |

Two economies were applied first, and both are worth keeping as rules: a catalogue row lists only
*formal* anchors (`AP-`, `ADR-`, `§`), because a slug restates the heading printed two columns over
and billing twice for one fact is pure waste; and the heading path is trimmed to its last two levels,
because the document title and top-level part are constant across most rows and carry no signal.

**Is ~4k acceptable?** At frontier-tier pricing this is well under a tenth of a cent per planner
call, and 84 lines is scannable. The argument for the catalogue was never that it is small in
absolute terms — it is that it is *bounded and structural* where the corpus is not. That still holds.

**The threshold to watch** (§4.5): the catalogue grows with the PMA, which is append-only by design
and already a third of it. When it becomes a real cost, the lever is to exclude `PROJECT_MEMORY.md`
from the runtime corpus — it is the document the *developer* reads, and the planner may not need
every ADR to route a task. That is a decision to make with data, not now.

**Amendment, 2026-09-10 — the Phase 1 corpus is `docs/` only (resolves OQ-05).**

There is no Obsidian vault to index yet. The wikilink adjacency table and frontmatter-tag filtering
are therefore **not built**: they would have no caller, which is the AP-11 shape this project has
already been bitten by once. The chunker still *extracts* wikilinks — that is parsing, and it is
free — but nothing persists or queries an adjacency structure.

Scope is recorded inside this ADR rather than as a separate record, per §1.3: which corpus is indexed
and how it is indexed would be read together every time, so they are one decision.

**Reversal condition:** a notes tree the user actually wants queried at runtime. At that point the
questions to answer are whether those notes use `[[wikilinks]]` and tags *meaningfully* — an
adjacency table over notes that barely link to each other is dead weight, and FTS5 alone would serve
better.

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

**Amendment, 2026-09-10 (step 1.2 implementation). Two corrections and one addition.**

1. **Retry is hand-rolled; `tenacity` is removed from the dependencies.** The ladder needs three
   things at once — honour `Retry-After`, full jitter, and an *injected sleeper* so the deterministic
   tier never actually waits. That is about twenty-five lines written directly and a wrapper fight
   otherwise. `tenacity` was declared in Phase 0 and never imported; a dependency nothing uses is the
   same drift AP-19 describes, so it is gone rather than left as a claim.

2. **A fatal error aborts the chain instead of walking it.** The layer order above says "on exhausted
   retries, advance to the next model", which is right for a rate limit and wrong for a 401. An
   authentication failure, a config error or an exceeded budget is not about *this* model: falling
   through repeats it once per chain entry and reports the last one instead of the real one. With a
   three-model chain in a twelve-way fan-out that is 36 doomed requests before anything says "401".
   `_FATAL = {AUTH_FAILED, CONFIG_INVALID, BUDGET_EXCEEDED}` short-circuits. `UNKNOWN` deliberately
   does *not*: an unrecognised failure might be model-specific, so falling through is the safer
   default. **This was found by a test, not by reasoning** — the assertion was written first and the
   code failed it.

3. **A schema failure does not open the circuit breaker and does not fall through.** The model
   answered; it answered badly. Treating bad output as "the model is down" would take a healthy model
   out of rotation for the cooldown, and layer 8's repair is the right response instead.

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

### ADR-013: Billed provider calls go through a content-addressed response cache

**Date:** 2026-09-10
**Status:** Accepted — resolves OQ-07

**Context.**
LangGraph writes checkpoints at superstep boundaries, not inside nodes. A worker reads state,
assembles context, calls the provider — 10 to 60 seconds, and money — and returns a result. The
checkpoint recording that result is written *after* the node returns. If the process dies in
between (Ctrl-C, sleep, OOM, a crash in a sibling branch), the provider may have completed and
billed the call while nothing recorded it; on resume LangGraph re-runs the node from its first line
(ADR-007) and calls again. At N=16, a crash three-quarters through re-issues about twelve calls.

Playbook §3.3 requires an idempotency key on every mutating operation. A billed LLM call is mutating
in the sense that matters: an irreversible external effect, and a client that cannot distinguish
"already did this" from "haven't done this."

Three options were considered.

| Option | Mechanism | Assessment |
|---|---|---|
| Provider-side `Idempotency-Key` header | Ask OpenRouter to dedupe | OpenRouter proxies many upstreams; support is per-provider and not a guarantee. Free to send, cannot be relied on. |
| **Local content-addressed cache** | Hash the call, check before, write immediately after | ~50 lines. Shrinks the double-pay window from a whole node to the gap between HTTP response and local INSERT. |
| Nothing | Accept double billing | At the mid tier a 4k-in/1k-out call is ≈$0.0003; a crashed 16-worker run wastes half a cent. |

**Decision.**
The gateway keeps a content-addressed response cache in `.dynaflows/calls.db`.

**The honest justification is not crash-safety.** Half a cent does not buy a subsystem. The cache is
built because it makes the system *developable*: tuning the synthesizer prompt without re-paying and
re-waiting for sixteen worker calls, comparing two evaluator rubrics against fixed worker outputs,
replayable system-tier tests. The measurement that decides it is not "how often will I crash" but
"how many times will I re-run a plan while tuning the nodes downstream of the workers" — dozens, and
most of them on the day something is wrong. Crash-safety is a side effect, and the weaker argument.

Binding details:

1. **Separate database from the checkpointer.** `calls.db`, not `state.db`. Different lifecycle:
   clearing the cache to force fresh calls must never destroy resumable runs.
2. **The key** is `sha256` over a canonical serialization of `model_id`, the full message list, the
   JSON schema, the sampling parameters, `extra_body`, and an explicit `variant: int`. It must NOT
   include `run_id`, `task_id`, `thread_id` or any timestamp, or nothing ever hits.
3. **`variant` is how sampling diversity is requested.** Caching a sampled response and replaying it
   is deliberate — determinism on replay is the feature. Phase 5's Tournament and Generate-and-Filter
   want different outputs from an identical prompt, and they get them by incrementing `variant`, not
   by disabling the cache.
4. **The raw provider response is stored, not a parsed form.** Storing a derived representation
   invites the AP-01 family of defects, where the stored form has already lost something the caller
   needs. Parse on read.
5. **Only validated successes are cached.** A 429, a timeout, or a schema failure that survived the
   repair attempt is never written. Caching a transient error would make it permanent — the worst
   available outcome and an easy mistake.
6. **The write happens in the gateway**, immediately after the response is parsed and validated,
   before returning to the node.
7. **A hit is visible or the cost numbers lie.** Every hit is stamped `cache_hit=true` in the
   LangSmith metadata, and the ledger keeps two counters — `usd_spent` and `usd_avoided` — never
   merged. This is AP-20's shape again: money not spent and money spent are different facts and a
   single number answers neither question.
8. **`--no-cache` exists**, and so does `dynaflows cache clear`.

**Consequences.**
- Two identical worker calls in one plan collapse to one provider call. For workers this is correct
  (same task, same context, same answer). Where it is not wanted, `variant` is the mechanism.
- **Known limitation, recorded rather than solved:** a provider can change the model behind a pinned
  id without changing the id — AP-05's exact shape — and the cache will then serve responses from
  the old behaviour indefinitely. Ids are pinned and `doctor` validates their existence, but neither
  detects a silent behavioural change. `dynaflows cache clear` is the remedy, and it is manual.
- Cache growth is unbounded. No eviction is implemented, because no measurement exists to size one
  (§4.5). Revisit when `calls.db` becomes inconvenient, which is a fact that will announce itself.
- N workers writing cache rows at fan-in put the same pressure on SQLite's single writer as the
  checkpointer does. WAL mode; short transactions; no open transaction across a provider call.

**Alternatives considered.**
*Cache in `state.db` alongside checkpoints.* Rejected: it couples two lifecycles that need to be
cleared independently.
*Skip the cache and rely on the provider header.* Rejected: unverifiable per-provider behaviour is
not a foundation, and it delivers none of the iteration-speed benefit that actually justifies this.

---

### ADR-014: Fan-out width and request concurrency are separate limits

**Date:** 2026-09-10
**Status:** Accepted — resolves OQ-03

**Context.**
OQ-03 was originally framed as "what is `MAX_FANOUT`, and does it derive from the rate limit or the
checkpoint write cost?" The framing contained an error, now corrected: with ADR-010's process-global
semaphore in the gateway, a plan of 32 tasks against a semaphore of 6 runs six in flight and queues
twenty-six. **The rate limit binds the semaphore, not the plan width.** They are different questions.

What actually bounds fan-out width, lowest ceiling first:

| Ceiling | Nature | Status |
|---|---|---|
| Plan quality | A planner emitting 40 tasks for a 6-task problem produces shallow, overlapping tasks | The one that actually bites; not technical |
| Wall clock vs. gate G2 | A 40-minute run turns the human gate into a rubber stamp before walking away | Judgement |
| Checkpoint write throughput | N branches writing pending-writes rows through SQLite's single writer (ADR-008) | **Unmeasured** |
| Cost | At worker-tier prices, N=64 is ≈2 cents | Not binding |

**Decision.**

- `MAX_FANOUT = 12` — the upper bound on `Plan.tasks`, enforced by the schema and stated in the
  planner's prompt.
- `OPENROUTER_MAX_CONCURRENT = 6` — the gateway semaphore, per API key, process-global.
- LangGraph's own `max_concurrency` is set equal to `MAX_FANOUT`, i.e. the graph does not limit.
  The gateway is the single limiter, because the provider rate-limits requests per key across the
  process and only the gateway sees all of them (ADR-010).

Both numbers are **placeholders, not measurements** (§4.5). Twelve is chosen to be wide enough to
exercise every fan-out failure mode — partial failure, real 429s, checkpoint pressure — while
keeping a debug cycle short enough that someone will actually run it twice.

**When a plan exceeds the bound, it is rejected, never truncated.** One re-plan attempt with the
violation stated in the prompt; if the second plan also exceeds, the run stops at G2 and shows the
human. Silently dropping tasks would produce a synthesis that is incomplete without saying so, which
is AP-20's confusion in another costume — a task dropped for capacity and a task that failed are
different facts and must not share a fate.

**Consequences.**
- Rate limits still decide *which models* are viable at the worker tier: an endpoint capped at 20 rpm
  delivers 20 requests per minute regardless of the semaphore, so a 12-task plan spends most of a
  minute waiting. This is the concrete reason `:free` endpoints are excluded from the mid tier.
- Interaction with ADR-013: once a crashed run resumes from cache for free, a wider fan-out becomes
  much less expensive to get wrong. Revisit `MAX_FANOUT` upward only after the cache exists.
- **The measurement that replaces these numbers** is Phase 1 step 1.8: p95 checkpoint write latency
  at fan-in for N = 4, 8, 16, 32, and the observed 429 rate at semaphore 2, 4, 6, 8. Until that runs,
  this ADR states an intention, not a tested property.

---

### ADR-015: LangSmith is a hard dependency; the tracing backend is not abstracted

**Date:** 2026-09-10
**Status:** Accepted (negative) — resolves OQ-04

**Context.**
OQ-04 asked whether an offline or local tracing backend is required. Behind it sat a tempting piece
of architecture: a `TracingBackend` protocol with a LangSmith implementation and a local JSONL one,
selected by a factory. It is the kind of abstraction that looks like good design in the abstract.

It has no caller. Nothing in this project needs a second backend, and AP-11 is explicit that an
abstraction whose only exercise is its own test is negative value — maintenance cost plus false
confidence. The prevention form of that anti-pattern is the relevant one here: *refuse to build the
abstraction before its production caller exists.*

There is also a substantive reason, not just a scheduling one. Requirement 2 of this project is that
every step, call and routing decision is traceable. Traceability is not a feature bolted onto the
workflow — it is the reason the workflow is worth building rather than just prompting a model. A
degraded local-only tracing mode would be a second code path that has to be kept true (AP-19) while
delivering a worse version of the product's central claim.

**Decision.**
LangSmith is a hard runtime dependency. `dynaflows doctor` fails without it. There is no
`TracingBackend` protocol, no factory, and no local-only fallback mode. `gateway/telemetry.py` talks
to LangSmith directly.

**Consequences.**
- A run requires network egress to `api.smith.langchain.com`. Prompts and outputs leave the machine.
  This is stated here so it is a known property rather than a discovery.
- The local `run.jsonl` (ADR-011) stays, but as a *join key* to the hosted trace, not as a substitute
  for it. It is not a fallback and must not grow into one by accident.
- **Reversal condition:** a concrete run that must happen without third-party egress — an
  air-gapped environment, or a client whose contract forbids it. Not "we might want it someday."
  When that arrives, this ADR is superseded and the abstraction is built with a caller in hand.

---

### ADR-016: Workers are read-only; dynaflows never writes to a target repository

**Date:** 2026-09-10
**Status:** Accepted (negative) — resolves OQ-06

**Context.**
The charter has said since day one that this is not a coding agent. That statement lived in prose,
with nothing enforcing it — which is AP-19's exact failure shape: a document asserting behaviour the
code does not guarantee, believed until someone checks.

The decision has to be made *before* step 1.6, not after. If a worker may write, the tool grant needs
a sandbox design, a path allow-list, and a rollback story, and all three have to exist before the
first worker node is written. If a worker may not write, the tool grant is trivial. Deciding this
after the worker exists means retrofitting a security boundary around code that was not built for
one, which is the expensive order.

**Decision.**
Workers read and reason. They never create, modify or delete a file in any directory other than
`.dynaflows/` (the run store and databases this tool owns). No worker receives a write tool, a shell,
or a network client other than the gateway.

Outputs reach the user through the synthesizer and the run store. If a user wants a produced artifact
in their repository, they copy it there themselves.

**Consequences.**
- Sandboxing is out of scope for Phases 0–5 entirely. This is the single largest scope reduction in
  the project and the reason Phase 1 is a matter of days rather than weeks.
- The worker tool-grant model in step 1.6 is: the playbook repository (read), the run store (write,
  inside `.dynaflows/` only), and the gateway. Nothing else.
- This ADR is enforceable and therefore must be enforced: step 1.6 adds an architectural lint
  rejecting filesystem-write calls outside `src/dynaflows/store/`, alongside the ADR-010 gateway
  check. An ADR asserting testable behaviour names the test that proves it (AP-19 habit 1);
  `tests/architecture/test_worker_write_boundary.py` is that test, and it is written in step 1.6
  rather than now, because its subject does not exist yet.
- **Amended by ADR-017 (2026-09-11):** "the playbook repository (read)" was incomplete — a worker
  also needs the files the plan names. ADR-017 resolves them at dispatch rather than granting a
  read tool, so the grant above still holds verbatim: a worker receives text, never a tool.
- **Reversal condition:** a task that cannot be expressed as "read, reason, report." Reversing this
  requires a sandbox ADR first, not a code change first.

---

### ADR-017: The plan's `inputs` are resolved once at dispatch, not by the worker

**Date:** 2026-09-11
**Status:** Accepted — amends ADR-016's consequences

**Context.**
ADR-016 grants a worker "the playbook repository (read), the run store (write), and the gateway.
Nothing else." The `analyse` capability, written the same day, says a worker will "read the named
inputs and the retrieved playbook sections." Those two sentences contradict each other, and the
contradiction was invisible until a live plan emitted `inputs: ["src/dynaflows/gateway",
"src/dynaflows/contracts"]` — files no worker could open. Either `inputs` is real or it is
decoration, and a decorative field the planner spends frontier tokens populating is worse than no
field at all.

This is AP-19 caught by execution rather than by review: two documents, each internally consistent,
asserting incompatible things about code that did not exist yet.

**Decision.**
The **graph** resolves `inputs` once, at dispatch, and folds the resulting text into each worker's
context pack alongside the playbook chunks. Workers receive text and nothing else.

**Why not a per-worker read tool.** It is the obvious design and it is worse on four counts: the
path allow-list, symlink-escape check, secrets deny-list and binary/size guard would each have to
hold across N parallel branches instead of one; the token cost per worker becomes unbounded, which
defeats the pack budget that already exists; a read tool is a tool, so ADR-016's grant would need
amending rather than merely clarifying; and the architectural lint that proves ADR-016 gets harder
to write, because "no worker touches the filesystem" is a far easier property to enforce than "every
worker touches the filesystem only within these bounds."

**Consequences.**
- ADR-016's tool grant is **unchanged**. A worker still receives no file tool, no shell and no
  network client. This ADR narrows where a read may happen; it does not add a capability.
- Source text and playbook text compete for one budget, ordered: playbook anchors first (the planner
  chose them deliberately), then source. A tight budget therefore drops source before it drops the
  rules the source is being judged against. `ContextPack.dropped_ids` records what was lost, so a
  worker's blind spot is visible in the trace rather than inferred (ADR-011).
- The read boundary is `src/dynaflows/store/`, which is also the only package permitted to write —
  so ONE lint rule covers both directions of ADR-016 and this ADR.
- A path outside the project root, a symlink leaving it, a file matching the secrets deny-list, and a
  file over the size cap are four different refusals and are reported as four different reasons
  (AP-20). A worker that saw nothing must say why it saw nothing.
- **Reversal condition:** a task that cannot name its inputs in advance — one that must follow a
  reference discovered mid-analysis. That is a real limitation of this design and the honest trigger
  for revisiting it. Reversing requires a tool-grant ADR first, not a code change first.

---

### ADR-018: The planner is shown a source catalogue, and a plan naming a path that does not exist is rejected

**Date:** 2026-09-11
**Status:** Accepted

**Context.**
Live run `w1` produced five tasks, no `inputs`, and one fabricated audit citing four files that do
not exist. The proximate cause was not the worker. **The planner has never been shown what the
repository contains.** It is given the playbook catalogue — 84 lines, 3,956 measured tokens — and
nothing at all about the code it is planning against, so `inputs` has always been guesswork. On
thread `p1` it guessed `src/dynaflows/gateway` and was right by luck; on `w1` it declined to guess
and wrote "verify the package exists" into all five objectives instead, which is a rational response
to being asked to plan blind.

ADR-009 already established the shape of the answer for the playbook: one frontier call reads a
compact catalogue and routes context for all N workers. The code side had no equivalent, and
ADR-017's dispatch-time resolution was therefore machinery with nothing to resolve.

**Decision.**
Two halves, and neither works alone.

1. **The planner is shown a source catalogue**: one line per file — path, size, and the first line of
   its module docstring or first markdown heading — built by the same pruning rules that govern
   reading (`store/sources.py`'s skip list, secret deny-list and text-suffix filter), so the planner
   cannot name a file the resolver would then refuse.
2. **A plan naming a path absent from that catalogue is a violation**, handled by ADR-014's existing
   one-re-plan-then-halt path, with a correction that names the offending path. An `analyse` task
   with no inputs at all is also a violation while `analyse` is the only registered capability,
   because "read the named inputs" with no inputs named is a task with nothing to read.

**Why not a search tool for the planner.** It is the richer design and it costs a tool-calling path
through the gateway, a second place where the filesystem is touched, and a planner whose token
spend is no longer bounded before the call is made. The catalogue is one render, one budget, one
lint exemption — and if it proves insufficient the reversal is additive rather than a rewrite.

**Why the existence check matters more than the catalogue.** The catalogue makes good plans likely;
the check makes bad plans *impossible to execute silently*. A planner that hallucinates a path with
the catalogue in front of it is a real failure mode, and without the check it produces exactly the
`w1` outcome again — refusals at dispatch, a hedging worker, and a run that looks like it ran.

**Consequences.**
- The planner prompt grows. The catalogue is budgeted and **truncation is stated in the prompt**: a
  planner shown a partial tree it believes is complete will confidently plan against files it cannot
  see, which is AP-19 relocated into a prompt. Its size is a §4.5 placeholder until step 1.8.
- Building the catalogue reads every candidate file's first bytes. That is a filesystem traversal at
  plan time, inside `store/`, covered by lint rule 3.
- `PlanTask.inputs` stops being advisory. Requiring it is a **narrowing** and the reversal condition
  is explicit: the first capability that legitimately needs no source — "summarise the playbook's
  position on retries" is the obvious one — removes the empty-inputs rule, not the existence check.
- This does NOT make worker output trustworthy. A worker handed `auth.py` can still cite a line
  number that is not in it. Phase 2's adversarial verification is the designed answer and nothing
  here should be read as a substitute.

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

### 2.5 Response cache (SQLite, `.dynaflows/calls.db`) — ADR-013

```sql
CREATE TABLE calls (
    key           TEXT PRIMARY KEY,  -- sha256(model_id|messages|schema|params|extra_body|variant)
    model_id      TEXT NOT NULL,     -- as requested
    served_by     TEXT,              -- as actually routed; OpenRouter may differ
    variant       INTEGER NOT NULL DEFAULT 0,
    raw_response  TEXT NOT NULL,     -- the provider's JSON, verbatim. Parse on read (ADR-013.4)
    tokens_in     INTEGER NOT NULL,
    tokens_out    INTEGER NOT NULL,
    cost_usd      REAL NOT NULL,     -- what the ORIGINAL call cost; a hit adds this to usd_avoided
    created_at    TEXT NOT NULL
);
```

Only validated successes are written (ADR-013.5). `PRAGMA journal_mode=WAL`.

### 2.6 Cost ledger — two counters, never merged

```python
class CostLedger(BaseModel):
    usd_spent: float = 0.0       # money that left the account
    usd_avoided: float = 0.0     # money a cache hit did not spend
    tokens_in: int = 0
    tokens_out: int = 0
    calls_made: int = 0
    calls_cached: int = 0
```

A single "cost" number answers neither "what did this run cost" nor "what would it have cost cold",
and the G2 estimate needs the second one. AP-20: two facts, two counters.

### 2.7 Repository interface (§3.2, Protocol + factory)

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
| OQ-02 | Does a plan need intra-plan task dependencies (`depends_on`), or is a flat map sufficient? | `PlanTask.depends_on`; whether fan-out is one superstep or a scheduler | Open — **deliberately deferred.** Phase 1 is a pure map and `depends_on` is contract-bound to empty, so nothing is blocked. Reversal condition: the first plan where the planner wants task B to consume task A's output. Deciding it in the abstract would be guessing at a shape no real task has demanded. |
| OQ-03 | What is `MAX_FANOUT`, and does it derive from the rate limit or the checkpoint write cost? | `Send` dispatch; the G2 cost estimate | **Resolved 2026-09-10 → ADR-014.** The framing was wrong: the rate limit binds the semaphore, not the plan width. 12 and 6, both provisional. |
| OQ-04 | Is an offline/local tracing backend required, or is LangSmith a hard dependency? | `gateway/telemetry.py` abstraction — or its absence | **Resolved 2026-09-10 → ADR-015.** Hard dependency; no abstraction. |
| OQ-05 | Does the Obsidian vault need frontmatter-tag filtering, or are wikilinks + FTS5 enough? | `links` table usage; `PlaybookRepository.search` signature | **Resolved 2026-09-10 → ADR-009 amendment.** No vault yet; `docs/` only, adjacency deferred. |
| OQ-06 | Should `dynaflows` ever write to the target repository, or stay read-only? | The tool-grant model for workers; the entire sandboxing question | **Resolved 2026-09-10 → ADR-016.** Read-only. Sandboxing out of scope. |
| OQ-07 | Do worker provider calls need an idempotency key, or is checkpoint granularity sufficient to prevent double-billing on crash-resume? | `gateway.call()` signature; whether `WorkerResult` is written before or after the superstep commits | **Resolved 2026-09-10 → ADR-013.** Content-addressed cache in the gateway. Note the accepted reasoning is iteration speed, not crash cost. |

**Decision order (current, and it has already been corrected once):**

`OQ-07 ✓ → OQ-01 → OQ-03 ✓ → OQ-02 → OQ-06 ✓ → OQ-05 ✓ → OQ-04 ✓`

**Remaining: OQ-01 and OQ-02 — and neither is answerable by deciding.** OQ-01 needs the step 1.8
measurement; OQ-02 needs a real task that a flat map cannot express. Both have reversal conditions
recorded above. Nothing in Phase 1 is blocked by either.

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
| F-01 Playbook indexer + repository | `memory/features/feature-01-playbook-index.md` | 1 (step 1.1) | **Done** (2026-09-10) |
| F-02 Prompt enhancer + gate G1 | `memory/features/feature-02-enhancer.md` | 1 (step 1.4) | **Done** (2026-09-10) |
| F-03 Planner + gate G2 | `memory/features/feature-03-planner.md` | 1 (step 1.5) | **Done** (2026-09-11) |
| F-12 Graph skeleton, state contract, resume | `memory/features/feature-12-graph-skeleton.md` | 1 (step 1.3) | **Done** (2026-09-10) |
| F-04 Fan-out workers + resiliency | `memory/features/feature-04-fanout.md` | 1 | Not started |
| F-11 Gateway response cache (ADR-013) | `memory/features/feature-11-call-cache.md` | 1 (step 1.2) | **Done** (2026-09-10) |
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
Feature: A billed call is paid for once            # ADR-013

  Scenario: A crash between the provider response and the checkpoint
    Given an approved plan with 8 tasks
    And the process is killed after task 5's provider response arrives
    When the run is resumed by thread id
    Then task 5 is served from the cache
    And the provider receives 7 calls in total across both attempts
    And the ledger shows 7 in usd_spent and task 5's cost in usd_avoided

  Scenario: Re-running an unchanged plan costs nothing
    Given a completed run
    When the same plan is executed again with the same inputs
    Then the provider receives 0 calls
    And every result is marked cache_hit in its trace
    And usd_spent for the second run is 0.0

  Scenario: A changed downstream prompt does not re-pay for workers
    Given a completed run
    When only the synthesizer prompt is changed and the run is repeated
    Then every worker result is served from the cache
    And exactly one provider call is made

  Scenario: Sampling diversity is requested explicitly
    Given a task executed with variant 0
    When the same task is executed with variant 1
    Then the provider receives a second call
    And the two results are stored under different keys

  Scenario: A transient failure is never cached
    Given the provider returns 429 for a task on every attempt
    When the task fails
    Then nothing is written to calls.db for that key
    And a later run of the same task calls the provider again

  Scenario: A schema failure is never cached
    Given the provider returns unparseable output twice for a task
    Then nothing is written to calls.db for that key


Feature: Fan-out width is bounded and never silently truncated   # ADR-014

  Scenario: A plan within the bound runs as written
    Given a planner that emits 12 tasks
    Then the plan is accepted
    And 12 Send branches are dispatched
    And no more than 6 provider requests are in flight at any moment

  Scenario: An oversized plan is re-planned, not trimmed
    Given a planner that emits 20 tasks
    When the plan is validated
    Then the planner is called once more with the violation stated
    And no task is dropped from the first plan without the human seeing it

  Scenario: A persistently oversized plan stops at the human gate
    Given a planner that emits 20 tasks on both attempts
    Then the run halts at gate G2
    And the CLI shows the human all 20 tasks and the bound that was exceeded
    And no worker call is made

  Scenario: Capacity and failure are different facts
    Given a run in which one task failed and one task was never dispatched
    Then the failure counter is 1
    And the skipped counter is 1
    And the synthesis names both, separately


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

### Step 1.6 — the fan-out, and what it is allowed to touch (2026-09-11)

`Send` now dispatches one real worker per plan task. Each worker gets a context pack assembled at
dispatch (ADR-017), makes one MID-tier call, writes its report to the run store and returns a
`WorkerResult` delta. It never raises: a gateway failure, an unexpected exception from below the
gateway, a missing dependency and an unwritable disk are four different results, not four ways to
lose the run.

New: `src/dynaflows/store/` (run store + input resolution), `graph/prompts.WorkerReport`,
`graph/deps.store_from`/`source_root_from`, `WorkflowState.task`, and lint rule 3.

**ADR-017 exists because a live run found a contradiction two documents could not.** ADR-016's tool
grant and the `analyse` capability's description were written the same day and disagreed about
whether a worker may read the files a plan names. Nothing caught it until a real plan emitted
`inputs: ["src/dynaflows/gateway"]` — files no worker could open. Both documents were internally
consistent; only running the thing exposed it.

**`status` has three values and they are separable on purpose.** A model that reports insufficient
context, a context pack that dropped sections, and a refused input all produce `degraded` rather than
`ok`, because ADR-004's evaluator cannot distinguish "went fine" from "went fine as far as it could
tell" if both say `ok`. An unwritable store is also `degraded`, not `failed`: the analysis was paid
for and the summary is still worth synthesising.

**The linter fired on correct code the first time it ran** — `replace` matched `dataclasses.replace`
four times. Fixed by dropping `replace` and `copy` from the watch list and saying so in the source: a
gate that fires on ordinary work is a gate people switch off (playbook §5.2, Pattern 5), and the
resulting hole in the rule is named rather than left to be found.

**Two test-code fixes that were the same bug as a production one.** The graph `configurable` was
copied into two test modules; when the worker gained two dependencies, both copies needed the same
edit and neither could fail if only one got it. It is now one `graph_config`. Collapsing it, however,
also flattened a real difference — g1 auto-approves `plan`, g2 auto-approves `prompt` — and every g1
test silently sailed past its own subject until the suite caught it. The shared part is shared; the
gate default is not. The same reasoning applies to `cli._graph_config`, which replaces the two copies
that let `run` ship without a gateway.

**Still owed from 1.5:** ADR-011's feedback write on a G2 rejection (`key="plan_accepted"`,
`score=0`, `correction=<the edited plan>`). It needs the LangSmith run id surfaced from the invoker
and is not in this step.

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
- ADR-013 and ADR-014 are **decisions, not implementations**. Nothing in `src/` reads or writes
  `calls.db`, and no code enforces `MAX_FANOUT`. Both land in Phase 1 step 1.2. Until then these two
  ADRs are intentions, and this line is here so nobody reads them as descriptions.
- ADR-014's two numbers (12 and 6) have no measurement behind them at all. Step 1.8 is where they
  stop being guesses.
- `min_context_tokens` is now **32,000**, derived rather than guessed: the largest realistic prompt
  today is the synthesizer's at ~12,008 tokens (12 worker summaries per ADR-014 plus an 8,000-token
  artifact budget), and the planner's is ~5,456 (the measured 3,956-token catalogue plus prompt and
  schema). Doubled for headroom and rounded. **It is a lower bound the system provably exceeds, not
  the true requirement** — step 1.8 replaces it with LangSmith's real per-call token counts. Verified
  locally against the context lengths `--suggest` records as comments in `config/models.toml`: zero
  violations across all four chains.
- `pack()`'s token count is an ESTIMATE (`CHARS_PER_TOKEN_ESTIMATE = 3.6`), not a tokenizer. The
  tier's model is configurable and OpenRouter fronts many providers, so the tokenizer that will count
  these characters is unknown at pack time. It leans high on purpose. Calibration path: LangSmith
  reports real token counts per call, so step 1.8 compares estimate to actual and replaces the
  constant with a measured one.
- **CLOSED 2026-09-11: `'NoneType' object is not iterable`. Root cause was an SDK rename (AP-05).**
  `ChatOpenAI` renames a constructor `max_tokens` to `max_completion_tokens` unconditionally
  (`langchain_openai/chat_models/base.py`, `_default_params` and `_get_request_payload`). ADR-006
  sends `provider.require_parameters: true`, OpenRouter does not recognise that parameter name for
  these endpoints, and the routing funnel drops every candidate at `Filter by Parameters`. Two
  models, two different symptoms for the same fault: `nex-agi/nex-n2.5-mini:free` returned a clean
  404 naming the failed routing step; `openai/gpt-5.6-luna-pro` returned **HTTP 200 with
  `choices: null`**, which the openai SDK's `parse_chat_completion` iterated, producing a bare
  `TypeError` that named neither the parameter nor the rename. Fix: `max_tokens` now travels in
  `extra_body`, which is merged into the request body verbatim, so the name chosen is the name sent.
  **The 402 fix introduced this.** Before `default_max_tokens = 4096`, no `max_tokens` was ever set,
  so nothing was renamed and nothing was filtered — a fix whose blast radius was one line wider than
  the reasoning behind it (§1.3).
  What this cost, recorded because the cost is the lesson:
  - **Two wrong hypotheses, both plausible, both expensive.** First: the model is a reasoner and
    4096 max_tokens left nothing for the answer. Second: `include_raw=True`, added in the
    immediately preceding commit, was the trigger — a strong temporal correlation that was pure
    coincidence. Neither survived a diagnostic. Correlation with the last commit is the most
    seductive false lead there is, and it was wrong twice in a row here.
  - **`probe` could not find it because it tested a different call.** It differed from the failing
    call in three ways at once. `diagnose` was built to differ in none, and answered in one run.
  - **261 tests could not find it because every one of them stops at the invoker seam.** The fakes
    see `max_tokens`; the wire sees `max_completion_tokens`. `tests/architecture/test_wire_payload.py`
    now asserts the exact top-level payload key set and was confirmed to fail against the old code
    before being kept. It is designed to fail on an SDK upgrade: that failure is the notification.
- **CLOSED 2026-09-11: the cost ledger reported `$0.0000 spent` after a paid call.** `PriceBook` is
  constructed empty in every code path in `src/` — nothing ever populated it — so
  `PriceBook.cost()` returned 0.0 for every model and 0.0 formatted as `$0.0000`. The first live
  end-to-end run spent $0.00501886 on the planner and reported nothing. No error, no failing test,
  no warning; the only symptom was a number that looked like good news. **This is the third wrong
  number this project has shipped** (zero tokens, vacuous `passed=True`, zero cost) and all three
  had the same shape: a value that was structurally valid, semantically false, and never asserted.
  Fixes, in order of importance:
  - The cost is now **read, not computed**. OpenRouter states what a call cost in its `usage` block
    and LangChain passes it through to `response_metadata["token_usage"]` whole. That figure already
    accounts for provider markup, the cached-prompt discount and which upstream actually served the
    request — none of which a local price table can know. The price book survives only as a fallback
    for providers that report nothing.
  - An unknown cost is `None`, never `0.0`. `CostLedger.calls_unpriced` counts them (AP-20: a call
    nobody could price and a call that cost nothing are different facts), and the run report says
    the total is a **lower bound** and why. `RawResponse.cost_usd`, `CallResult.cost_usd` and
    `CachedCall.cost_usd` are all optional, so the unknown survives a cache round trip instead of
    becoming a confident zero on the next run.
  - `calls.cost_usd` was `NOT NULL`. `CREATE TABLE IF NOT EXISTS` says nothing about an existing
    table, so a pre-existing `calls.db` would have raised `IntegrityError` at fan-in — inside a
    worker, where ADR-010 says nothing may raise. `connect_cache` now rebuilds the table, copying
    rows rather than dropping them.
- **CLOSED 2026-09-11: a run passed while a worker fabricated an audit.** Live run `w1`, the first
  fan-out against real models. Five workers, `0 ok, 0 failed, passed=True`. What actually happened:
  - The planner emitted **no `inputs` at all**, so ADR-017's resolution had nothing to resolve and
    every worker received playbook chunks and no code. Four correctly reported "no package named
    `gateway` was found". **The fifth invented `gateway/logger.py`, `gateway/middleware.py`,
    `gateway/handlers.py` and `gateway/metrics.py`** — none of which exist anywhere in the repo —
    with line numbers ("Lines 70–80") and six findings, one rated HIGH severity for sensitive-data
    leakage. It also set `context_was_sufficient=False`: it admitted it lacked context *and*
    fabricated findings anyway. **A tool that invents an audit is worse than one that produces
    nothing**, and nothing in the pipeline objected.
  - `EvaluationReport` had counters for ok, failed and empty but **not for `degraded`**, which had
    been introduced in the same commit. Five degraded results summed to zero of everything,
    `reasons` came out empty, and `passed` was therefore True. Third occurrence of this exact shape
    after the zero-token ledger and the zero-cost ledger: **a fact with no counter becomes silence,
    and silence reads as good news.**
  - Fixes: `degraded_count`; ADR-004 **rule 2** — a run where no task succeeded cleanly does not
    pass, and a partly degraded run passes but says so; `EvaluationReport.render()`, because the old
    status line could only say ok and failed and so described a five-degraded run as "0 ok, 0
    failed"; and an explicit no-source instruction in the worker prompt, since "audit the gateway
    package" reads to a model like permission to describe what such a package usually contains.
  - `accounted_for` asserts the status counters partition the results, so a future status with no
    counter fails a test instead of vanishing. Its first version summed `empty_count` too and
    double-counted — caught by the test written beside it, which is the only reason it is not a
    fourth wrong number.
- **CLOSED 2026-09-11 by ADR-018: the planner named files blind.** It is now shown a source
  catalogue — one line per file, path plus size plus the first line of the module docstring or
  markdown heading — and a plan naming a path absent from that catalogue is a violation routed
  through ADR-014's one-re-plan-then-halt path. An `analyse` task with no inputs is also a
  violation, since "read the named inputs" with none named is a task with nothing to read.
  **Measured**: the catalogue is ~1,914 tokens for 77 files, and the planner's system prompt grew
  from ~4,280 to ~6,309 tokens. Both are real numbers, not estimates, and they are the first two
  entries in step 1.8's measurement pass that did not have to wait for it.
  Two things found while building it:
  - The first catalogue of this repository came out **60% `.pytest_cache` and `.ruff_cache`**. The
    skip rule was a deny-list, and a deny-list only knows the tools that existed when it was
    written — the next tool ships its own cache directory. It is now an allow-list: every dotted
    directory is skipped except `.github`/`.gitlab`/`.circleci`. This fixed `read_sources` too,
    which would otherwise have fed ruff cache hashes into a worker prompt. 90 files became 77.
  - `source_root` was hardcoded to `settings.project_root`, so **dynaflows could only ever analyse
    its own repository**. `run` and `resume` now take `--root`. That was not a test convenience; the
    tests merely made it visible.
- **CLOSED 2026-09-11: `merge_cost` silently dropped `calls_unpriced`.** The field was added to
  `CostLedger` two commits earlier and never added to the reducer, so it was discarded at the first
  superstep — meaning the "the total is a LOWER BOUND" warning built to surface unpriced calls could
  not fire in any run with more than one node, which is every run. **Fourth occurrence of the same
  pattern** after zero tokens, zero cost and `degraded` without a counter: a new fact and a
  hand-written aggregation list that nobody updated. `merge_cost` now enumerates
  `dataclasses.fields()`, and `tests/architecture/test_reducers_cover_their_contracts.py` asserts
  both the property (every field survives a merge) and the shape (the reducer is not a hand-written
  list), so the next field added fails a test rather than vanishing.
- **CLOSED 2026-09-11: a zero-byte report counted as output.** Live run `w2` produced a worker whose
  findings file was empty; it still received a valid `ArtifactRef`, so `produced_something` returned
  True, `empty_count` stayed 0 and the task reported `ok`. A reference to nothing is nothing:
  `produced_something` now requires a non-blank summary or an artifact with tokens, and a worker
  whose `findings` is blank is `degraded` regardless of what it says about its context.
- **STILL OPEN: the worker output contract does not require evidence.** Run `w2` was the first with
  real code in front of the workers, and three of four returned a single sentence — "no issues were
  found" — despite a prompt demanding file, line, impact, severity and evidence per finding.
  `WorkerReport.findings` is free text, so "I found nothing" and "I looked and here is what I
  checked" are indistinguishable, and neither can be verified. Making findings a structured list
  with required citations is the candidate fix and is a design change, not a patch.
- **The analysed root is not recorded in the checkpoint.** `resume` accepts `--root` and defaults to
  the project root, so resuming a run that used `--root` without passing it again resolves inputs
  somewhere else entirely and the workers quietly analyse the wrong files. The plan is checkpointed;
  the tree it was planned against is not. Recording it in `WorkflowState` is the fix and is owed.
- **ADR-018's empty-inputs rule is a narrowing and will eventually fire on legitimate work.** The
  first capability that needs no source — "summarise the playbook's position on retries" is the
  obvious one — makes it wrong. The reversal is explicit in the ADR: remove the empty-inputs rule,
  keep the existence check.
- **STILL OPEN: nothing verifies that a finding cites something real.** The no-source prompt is an
  instruction, not a guarantee, and it does not cover the harder case — a worker that WAS given
  `auth.py` and cites a line number that is not in it. Phase 2's adversarial verification is the
  designed answer; until it exists, worker output is unverified by construction and this document
  should not imply otherwise.
- **`WORKER_CONTEXT_BUDGET = 6_000`, `MAX_FILE_BYTES = 200_000` and `MAX_FILES_PER_INPUT = 25` are
  placeholders** (§4.5), sized so twelve workers stay inside the 32,000-token floor `doctor`
  enforces. None is measured. Step 1.8 owes all three, and they now sit alongside `MAX_FANOUT`, the
  semaphore, the retry count, `CHARS_PER_TOKEN_ESTIMATE`, `min_context_tokens` and
  `EVAL_MAX_FAILURE_RATE` on that list.
- **The source reader's secrets deny-list is a name-and-suffix match, not a scanner.** It stops
  `.env`, `*.pem` and friends. It does NOT stop a credential hard-coded in a `.py` file the plan
  named, and nothing here should be read as a claim that it does. ADR-016 removed the need for a
  sandbox; it did not remove the need to think about what leaves the machine.
- **ADR-017's dispatch-time read cannot serve a task that discovers what it needs mid-analysis.**
  That is the stated reversal condition and it is a real limitation, not a hypothetical one: a task
  like "follow the auth flow wherever it goes" is not expressible under this design.
- **Worker behaviour has never run against a live model.** 308 deterministic tests drive the node
  with a fake that always returns a well-formed `WorkerReport`. Every prior step in this project has
  had at least one defect that only a live run found, and there is no reason to expect this one to
  be different.
- **STILL OPEN: the USD budget ceiling has never been able to fire.** ADR-013's `Budget.max_usd` was
  compared against a total that was structurally always 0.0, so only the token ceiling was ever a
  real guard. It should work now that costs are real, but "should" is the word that produced this
  entry — it stays open until a run is deliberately driven into the USD ceiling and stops.
- `gateway/invoker.py`'s error classifier has now met exactly ONE real failure (402) and was wrong
  about it. 429, 401, 500 and timeout remain unverified against live traffic — and the 402 is the
  reason to treat that as a real gap rather than a formality.
- **Superseded 2026-09-11:** ~~the classifier is unverified against a live provider~~ The status-code
  and class-name mapping was written from documentation, not from observed exceptions, and the
  consequences of getting it wrong are not cosmetic: a 401 classified as retryable burns the whole
  chain, a 429 classified as fatal ends a run that would have succeeded. Eighteen tests drive it with
  fakes; only real traffic closes it.
- The cache's ADR-013 limitation is now live code: a provider changing the model behind a pinned id
  serves stale responses indefinitely. `dynaflows cache --clear` is the remedy and it is manual.
- `MAX_FANOUT = 12` is now enforced by `Plan`'s schema, and `PlanTask.depends_on` is rejected
  outright while OQ-02 is open — a dependency that is silently ignored produces a plan running in the
  wrong order with no error anywhere.
- `FAN_OUT_KEYS` binds the reducer annotations to a declared list, and the pair of architecture tests
  keeps them in step. **It does not detect a NEW fan-out node writing a key nobody declared** — only
  LangGraph's runtime `InvalidUpdateError` catches that, which is why a real six-way fan-out runs in
  the deterministic tier rather than only the annotations being inspected. Recorded so the check is
  not mistaken for complete.
- **ADR-011's feedback loop is NOT implemented.** The ADR calls a G2 rejection writing LangSmith
  feedback "the single highest-value observability decision in the project", and step 1.5 shipped the
  gate without it. The blocker is real — the gateway does not yet surface the LangSmith run id that
  `create_feedback` needs — but the consequence is that every rejection is currently a training
  signal thrown away. It belongs in step 1.6, where the invoker is being touched anyway.
- The worker catalogue registers exactly **one** capability, `analyse`. That is AP-11 applied
  honestly rather than a placeholder: `verify` arrives in Phase 2 with the node that performs it.
  A twelve-way fan-out of `analyse` tasks with different objectives, inputs and anchors is still
  Fan-out-and-Synthesize.
- ADR-016 is currently unenforced. The lint and `tests/architecture/test_worker_write_boundary.py`
  it names arrive in step 1.6, with their subject. Until then it is a rule with nothing reading code,
  which is precisely the state AP-19 says not to mistake for a guarantee.
- The `frontier` tier has a chain of one, so ADR-010's tier-fallback layer is inert for the tier
  where failure is most expensive — the planner runs before the fan-out, so a 429 there ends the run
  before any work happens. A second entry from a different family is a config change, not a code one.
- ADR-006 checks capability homogeneity in two dimensions (structured output, context length) and not
  a third: **latency class**. `--suggest` now excludes `:batch` endpoints from every tier and says so
  in its output, but `doctor` does **not** reject a `:batch` id pasted in by hand. That enforcement is
  deliberately absent: no chain has ever contained one, so the rule would have no subject (AP-11).
  Add it the first time one appears.
- What a `:batch` endpoint actually does behind this gateway is **unverified**. It either blocks far
  longer than a human at a gate will tolerate, or it times out and burns three retries; which one is
  unknown, because checking costs real credits and the answer changes nothing — both are unusable
  here. Recorded as unverified rather than asserted (AP-19 habit 3).
- **Idempotency is unresolved (OQ-07).** ADR-007 makes *gate* re-execution safe. It says nothing about
  a `worker` node re-executing after a crash and re-issuing a billed provider call. §3.3 says this
  needs an idempotency key; this document does not yet specify one. Named here rather than left to be
  discovered by a duplicated bill.
- ADR-004's `EVAL_MAX_FAILURE_RATE` default of 0.34 is arbitrary. It should be derived from what a
  synthesis can usefully degrade to, which is a measurement nobody has taken.
- ADR-012's probe ran on aarch64 Linux, not on the macOS arm64 host this project is developed on.
  Closed only when `dynaflows doctor` has run there.

**Learned on 2026-09-11, third pass — the first real provider failure, and the ladder got it wrong
three ways.** A 402 from OpenRouter ("you requested up to 65536 tokens, but can only afford 33333")
hit the planner. §7 had said for two steps that the error classifier was *unverified against live
traffic* and that "only real traffic closes it". This is what real traffic found:

1. **402 was classified `MODEL_UNAVAILABLE`, which is retryable.** So a request that could never
   succeed was sent three times before moving on. The distinction that was missing: a model that is
   *down* may come back, so retrying is reasonable; a request you cannot *afford* costs exactly the
   same on every attempt. `INSUFFICIENT_CREDIT` is now its own code — not retryable, but still not
   fatal, because the next entry in the chain may well be affordable, which is the whole reason a
   chain has more than one model.

2. **The root cause was ours: we never set `max_tokens`.** Providers price a request as prompt plus
   the **full output allowance**, so an unset ceiling reserves the model's maximum — 65,536 here —
   and the account was refused for a request that would have used a fraction of it. The gateway now
   caps every request it sends, and nodes that know their own answer size declare it. A call site
   that forgets can no longer reserve the maximum.

3. **The typed error escaped as a 200-line traceback.** §1.4 types errors precisely so a caller can
   act on them, and the one genuinely useful line — the provider's own `remedy_hint`, naming exactly
   what to change — was buried at the bottom under LangChain internals. Typed errors now render as a
   message plus the remedy, with the resume command, and exit 3.

The shared lesson: **the taxonomy was wrong in a way no fake could have shown.** Eighteen tests drove
the classifier with invented exceptions and all of them passed, because they asserted the mapping I
had written rather than the mapping the provider needed. A test can only check the rule you thought
of; live traffic is what supplies the rule you did not.

**Learned on 2026-09-11, second pass — two things the ledger was quietly not counting.**
A live run at gate G1 exposed both, and neither had a failing test because both reported a
*plausible* number rather than an error.

1. **Every structured call recorded 0 tokens and $0.00.** `with_structured_output()` returns only
   the parsed model and discards the `AIMessage` carrying `usage_metadata`, so the invoker never saw
   it. The ledger, ADR-010's budget ceiling and the G2 estimate were all counting nothing — and
   counting nothing looks exactly like a cheap run. Fixed with `include_raw=True`, which returns the
   raw message alongside the parsed object. Zeros are honest when a provider sends none; they were
   not honest when we threw the message away.
2. **ADR-004's first rule was specified and never implemented.** "Every plan task has exactly one
   `WorkerResult`" is rule 1 of the evaluator, and without it five planned tasks with zero results
   reported `passed=True` — a vacuous pass, because nothing came back to fail. A missing result is
   not a silent success: it means a branch never ran or never returned, which is strictly worse than
   one that failed and said so. The report now carries `reasons`, so the gap is named rather than
   implied by a boolean.

The general shape, and it is the third time this project has hit it: **a wrong number is harder to
notice than an error.** Both of these passed every test, rendered fine, and would have been believed.

**Learned on 2026-09-11 — the first human at a live gate got stuck in it.**
Gate G1 worked on its first real run: a good rewrite, two substantive assumptions declared, one
small-tier call. Then the human chose `edit`, had no `$EDITOR` set, and hit a fallback that was wrong
twice over — a single-line prompt for a five-line brief, with no default, so an empty Enter re-asked
forever. There was no way out of the gate except Ctrl-C.

**A gate the user cannot leave is worse than no gate.** ADR-005's whole argument is that stopping
here is cheap; a stop you cannot exit converts that into the most expensive kind of interruption.

Three rules came out of it, and they generalise past this one prompt:

1. **Every interactive branch needs a way out that is not Ctrl-C.** Empty input now means "keep the
   text", because "I changed my mind about editing" is the likeliest reason someone submits nothing.
2. **A fallback is a code path, and untested code paths are where this project keeps finding bugs.**
   The editor path had no test at all — it was the only branch of `_ask_gate` nothing exercised.
3. **Defaults should assume the user is not you.** With no `$EDITOR` set, the editor search now
   prefers `nano` over `vim`/`vi`: someone who never configured one is unlikely to be a vi user, and
   dropping them into modal editing unannounced is its own trap.

What held up under the same failure, and is worth keeping: Ctrl-C lost nothing. The thread was
checkpointed at the gate, `resume` re-presented the identical text with the assumptions intact, and
the enhancer call count stayed at 1 — ADR-007 and ADR-008 paying off on the first run that tested
them for real.

**Learned on 2026-09-10, fourth pass — 204 green tests and the command was still broken.**
`dynaflows run` built its config WITHOUT a gateway, so every real run failed inside `enhance_prompt`
with CONFIG_INVALID while the whole suite stayed green. Two causes, and the second is the one worth
keeping:

1. The graph was tested and the *wiring that feeds it* was not. Unit tests constructed their own
   config and passed a fake gateway directly, so they exercised the node and never the command. The
   fix is `tests/unit/test_cli_run.py`, which drives the real Typer command with `get_gateway`
   patched at the factory boundary.
2. The edit that should have added the gateway was a string replacement that **silently did not
   match** — a formatter had reshaped the target line first. Other replacements in the same batch
   asserted their match; this one did not, so it failed quietly and looked like success. Any
   mechanical edit that is not asserted is a change that might not have happened.

**Learned on 2026-09-10, third pass — the CI step that could never have passed.** The workflow ran
`dynaflows doctor --offline` and required exit 0, on a runner that has no credentials and never will.
`doctor` was correct; the assertion was impossible. It went unnoticed because CI had not run against
a commit that reached that step until the repository had a remote.

The general form is worth more than the fix: **a check whose expected result was never derived from
the environment it runs in is not a check.** CI's environment differs from a developer's in exactly
one important way here — no secrets — and the step was written as if that difference did not exist.
The corrected step asserts the *failure* instead, which is a real test: it proves the checks execute
against the installed package, that the exit code is honest, and it breaks loudly if `doctor` ever
starts passing without credentials.

Also corrected: `actions/checkout` and `astral-sh/setup-uv` were pinned to majors that GitHub had
already moved off Node 20 for. The available majors were checked with `git ls-remote --tags` rather
than recalled (AP-19 habit 3), and both are now on v7.

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

**Closed by verification on 2026-09-10, with real credentials** (the network checks that had never
executed in the project's life, per the note above — this is what closed them):

| Check | Result |
|---|---|
| `openrouter` | authenticated; **338** endpoints support structured outputs |
| `langsmith` | reachable, project `dynaflows` |
| `tier capability` | every configured model in all four chains honours `json_schema` |
| `handshake` | structured output honoured, **through the full ADR-010 ladder** |

The handshake is the one that matters most. It is not a bare HTTP call: it runs registry → cache →
breaker → semaphore → timeout → invoker → provider → parse, so the whole chokepoint is exercised
end to end by `make doctor`. `probe_structured` was rewritten in step 1.2 to go through the gateway
precisely so that this check tests the shipped path (AP-11) rather than a parallel one.

**What this does NOT close:** every response so far has been a success. The error classifier in
`gateway/invoker.py` — which decides whether a failure retries, falls through or aborts — has still
never seen a real 429, 401 or timeout.

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
