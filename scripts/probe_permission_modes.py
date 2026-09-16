"""Which --permission-mode completes a non-interactive agent run? ADR-025.

`claude --help` lists six modes and documents the non-interactive behaviour of
none of them. The failure this resolves is not an error: a mode that prompts
with no terminal to prompt into HANGS, and a hung run is indistinguishable
from a slow one until the timeout fires. That is the most expensive way to be
wrong about a constant, so it is measured rather than chosen (§4.5).

Each mode gets a throwaway worktree and one task that needs BOTH an edit and a
shell command, because those are granted separately and a mode that accepts
edits may still stop at the first `Bash`.

    uv run python scripts/probe_permission_modes.py

Costs a handful of small agent turns against your Claude subscription. Every
worktree it cuts is removed before it exits, including on Ctrl-C.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynaflows.executor import workspace as ws  # noqa: E402
from dynaflows.executor.agent import AgentOptions, ClaudeCodeAgent  # noqa: E402

# `manual` is omitted: it is the mode whose entire purpose is to ask, so
# running it would buy a timeout we can already predict. `plan` is omitted for
# the opposite reason -- it is documented not to edit.
# `dontAsk` is dropped after round one: it did not hang, it completed having
# done nothing. A mode that silently no-ops is worse than one that blocks,
# because the run reports success to a human who then looks for a diff.
MODES = ("acceptEdits", "auto", "bypassPermissions")

# Two tasks, because the first round measured the wrong thing. `git rev-parse`
# is a read-only git command and is very likely auto-allowed, so "the mode
# completed" meant "the mode did not need to ask" -- not "the mode never
# asks". The second task needs a shell command no allowlist would cover.
# A mode that passes the first and hangs on the second is the worst outcome
# available: an INTERMITTENT hang, which looks like a slow change on the runs
# where it happens and like nothing at all on the runs where it does not.
TASKS = (
    (
        "benign",
        "Run `git rev-parse --short HEAD` and write its output into a new file "
        "named PROBE.txt in the repository root. Do nothing else.",
        "PROBE.txt",
    ),
    (
        "arbitrary",
        "Run this exact shell command and nothing else: "
        "python3 -c \"open('PROBE2.txt','w').write('ok')\"",
        "PROBE2.txt",
    ),
)
TIMEOUT_S = 120


def main() -> int:
    repo = Path.cwd()
    home = repo / ".dynaflows"
    rows: list[tuple[str, str, str, str]] = []

    for index, mode in enumerate(MODES):
        for label, task, expected in TASKS:
            run_id = f"probe-{index}-{label}-{int(time.time())}"
            space = None
            try:
                space = ws.create(repo, home, run_id)
                agent = ClaudeCodeAgent(AgentOptions(permission_mode=mode, timeout_s=TIMEOUT_S))
                print(f"  {mode:<20} {label:<10} running…", flush=True)
                result = agent.run(space, task)

                wrote = (space.path / expected).is_file()
                if result.unavailable:
                    verdict = "agent unavailable"
                elif result.timed_out:
                    verdict = "HUNG (prompted)"
                elif not result.usable:
                    verdict = f"failed (exit {result.exit_code})"
                elif wrote:
                    verdict = "OK"
                else:
                    verdict = "SILENT NO-OP"

                rows.append(
                    (
                        f"{mode} / {label}",
                        verdict,
                        f"{result.duration_s:.0f}s",
                        "-" if result.cost_usd is None else f"${result.cost_usd:.4f}",
                    )
                )
                if verdict not in ("OK",):
                    print(f"    result : {result.result_text[:400] or '(none)'}", flush=True)
            except Exception as error:  # noqa: BLE001 - a probe reports, it does not crash
                rows.append((f"{mode} / {label}", f"error: {error}", "-", "-"))
            finally:
                if space is not None:
                    ws.discard(space)

    width = max(len(m) for m, *_ in rows)
    print("\n  mode" + " " * (width - 2) + "verdict                        time     cost")
    for mode, verdict, taken, cost in rows:
        print(f"  {mode:<{width + 2}}{verdict:<31}{taken:<9}{cost}")
    print(
        "\nA mode is only usable if BOTH its rows are OK. One OK and one HUNG is\n"
        "an intermittent hang, which is worse than a consistent one.\n"
        "Costs are one sample each -- read them as an order of magnitude, not a\n"
        "ranking (the ADR-006 correction: one sample per model measured noise)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
