# dynaflows

A Python CLI that reconstructs the orchestration patterns behind Claude Code's
"dynamic workflows" on LangGraph — one pattern at a time, with every routing
decision, token and fallback visible in a trace.

**Status: Phase 0.** Foundation and environment verification only. There is no
workflow yet — no graph, no nodes, no provider calls beyond a single handshake.

---

## Read this first

[`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md) is the source of truth: the
charter, every architecture decision (ADR-001…012), the data contracts, the
open questions, and the phase plan. It is written before the code, and it is
updated in the same commit as the code.

[`docs/GENERAL_ENGINEERING_PLAYBOOK.md`](docs/GENERAL_ENGINEERING_PLAYBOOK.md)
is the standard this project is held to. It is also the retrieval corpus the
tool itself will index at runtime, from Phase 1 onward.

## Requirements

- Python **3.14** (`.python-version`) — pinned and verified, see ADR-012
- [`uv`](https://docs.astral.sh/uv/)
- An OpenRouter API key and a LangSmith API key

## Setup

```sh
git clone https://github.com/carlosost/dynaflows.git
cd dynaflows
make install
cp .env.example .env      # then fill in your keys
make doctor
```

`make doctor` is the Phase 0 gate. It checks, cheapest first:

| Check | What it proves |
|---|---|
| `environment` | Required variables are set (playbook AP-05) |
| `sqlite` | SQLite is writable **and ships FTS5 with `bm25()`** — the premise of ADR-009 |
| `models.toml` | The tier registry parses, and the ADR-006 family-diversity rule holds |
| `langsmith` | Tracing is configured and the API answers |
| `openrouter` | The key authenticates, and how many endpoints support structured outputs |
| `tier capability` | Every configured model still exists and still honours `json_schema` |
| `handshake` | One real, traced, schema-enforced call — the whole path, end to end |

`make doctor` will report `models.toml` as **unpopulated** on a fresh clone.
That is correct and deliberate — see below.

## Choosing models

`config/models.toml` ships with empty tier chains. No model ids are written
there by guesswork: a renamed or retired provider id fails the way AP-05
describes — silently, with no error and no log line — and it invalidates every
LangSmith baseline recorded against it.

```sh
uv run dynaflows models              # the live structured-output catalogue
uv run dynaflows models --suggest    # a TOML block to paste, with prices
```

Cost order is a starting point, not a recommendation. ADR-006 assigns tiers by
**calls-per-run multiplier** — spend at the planner, which runs once and decides
the cost of N worker calls; economize at the worker, which runs N times — and
requires the verifier to come from a different model family than the workers it
checks. Neither rule can be derived from price.

## Development

```sh
make check              # everything CI runs, in CI's order
make test               # deterministic tier only — must be 100%
make lint-architecture  # ADR-010: no provider SDK outside the gateway
```

Tests are split into two tiers (playbook §2.2). The **deterministic** tier
touches no network, no provider and no filesystem outside `tmp`; it is a hard
gate. The **system** tier uses real credentials and is never a pre-merge gate.

Mock at the factory boundary — `get_settings`, `get_model_registry` — never at
the library (`@patch("sqlite3.connect")` is the AP-02 mistake).

## Layout

```
config/models.toml            tier → model chains (ADR-006)
docs/PROJECT_MEMORY.md        source of truth
scripts/lint_architecture.py  ADR-010 enforcement
src/dynaflows/
  cli.py                      doctor, models
  doctor.py                   the Phase 0 gate
  settings.py                 env + the AP-05 startup check
  contracts/                  Tier, ErrorEnvelope
  gateway/                    the ONLY place a provider SDK is imported
tests/unit/                   deterministic tier
tests/architecture/           ADR enforcement tests
```

## License

MIT. See [LICENSE](LICENSE).
