"""Writing a brief and its gate to disk, so a run survives the terminal.

The A/B test that motivated this lost its first run to a shell redirect: the
brief went into one file, the gate into another, and both landed in `/tmp`
where neither the author nor a reviewer could find them. A command whose
entire output is text a human will paste somewhere else should not depend on
the human getting the redirect right.

Two files per run, deliberately:

  <thread>.brief.txt   the brief, verbatim, nothing else -- this is what gets
                       pasted, and anything added to it would be pasted too
  <thread>.md          the record: the original request, what the model made
                       of it, what it said the request was about, what it
                       assumed, what the human decided, and what it cost

The record renders the gate's CONTENT, not the gate's appearance. Capturing
the rendered panels would tie the file to whatever `COLUMNS` happened to be:
the same run recorded at 200 columns and at 80 produces two files that diff
against each other line for line while saying the same thing. A record whose
bytes depend on the terminal that made it cannot be compared across runs, and
comparing runs is the only reason to keep it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["BRIEF_DIR_NAME", "BriefRecord", "record_markdown", "save_brief"]

# A constant rather than a setting. A second place to configure is a second
# place for two runs to disagree about where their output went, which is the
# failure this module exists to end.
BRIEF_DIR_NAME = ".ai"


@dataclass(frozen=True, slots=True)
class BriefRecord:
    """Everything worth keeping about one trip through G1.

    `cost` arrives already formatted. `cli._money` owns the rule that an
    unpriced call makes the total a lower bound; restating that rule here
    would give the project two formatters to keep in step. One formats, this
    one records.
    """

    thread_id: str
    run_id: str
    raw_prompt: str
    brief: str
    decision: str
    source_root: str
    cost: str
    model: str = ""
    intent: str = ""
    relevant_paths: list[str] = field(default_factory=list)
    invented_paths: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _section(title: str, items: list[str], empty: str) -> list[str]:
    """A heading that is always present, so its absence is never ambiguous.

    An empty list and a missing section look identical in a file and mean
    different things: "the model named no paths" versus "this record predates
    path checking". AP-20 -- two facts, two texts.
    """
    if not items:
        return [f"## {title}", "", f"_{empty}_", ""]
    return [f"## {title}", "", *[f"- `{item}`" for item in items], ""]


def record_markdown(record: BriefRecord) -> str:
    """The record file, as text. Pure: same record in, same bytes out."""
    lines: list[str] = [
        f"# {record.thread_id}",
        "",
        f"- **when** {record.at.isoformat(timespec='seconds')}",
        f"- **run** `{record.run_id}`",
        f"- **decision** {record.decision}",
        f"- **written by** `{record.model or 'not recorded'}`",
        f"- **read as a** {record.intent or 'not recorded'}",
        f"- **about** `{record.source_root}`",
        f"- **cost** {record.cost}",
        "",
        "## you asked",
        "",
        record.raw_prompt.strip(),
        "",
        "## the brief",
        "",
        record.brief.strip(),
        "",
    ]
    lines += _section("it reads this as being about", record.relevant_paths, "no paths named")
    if record.invented_paths:
        # Only when it happened. An empty "invented" section on every record
        # would train the reader to skip the heading, and this is the one
        # heading that has to be read when it has content.
        lines += _section("paths it named that do not exist", record.invented_paths, "none")
    lines += _section("assumed on your behalf", record.assumptions, "no assumptions declared")
    return "\n".join(lines).rstrip() + "\n"


def save_brief(directory: Path, record: BriefRecord) -> tuple[Path, Path]:
    """Write both files. Returns (brief, record), in that order.

    The brief is written with a trailing newline and nothing else -- no
    header, no fence, no provenance comment. It is meant to be pasted, and a
    provenance line at the top of a prompt is a line the receiving model has
    to decide what to do with.
    """
    directory.mkdir(parents=True, exist_ok=True)
    brief_path = directory / f"{record.thread_id}.brief.txt"
    record_path = directory / f"{record.thread_id}.md"
    brief_path.write_text(record.brief.strip() + "\n", encoding="utf-8")
    record_path.write_text(record_markdown(record), encoding="utf-8")
    return brief_path, record_path
