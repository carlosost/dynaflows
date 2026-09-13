"""What a saved brief has to contain, and what it must not.

The first A/B run was lost to a shell redirect. These tests exist so the
replacement -- saving both files itself -- cannot quietly regress into the
same shape of loss: a brief with furniture in it, or a record that cannot be
compared to the next one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynaflows.store.briefs import BriefRecord, record_markdown, save_brief

pytestmark = pytest.mark.deterministic


def a_record(**overrides: object) -> BriefRecord:
    base: dict[str, object] = {
        "thread_id": "brief-0b7a3a65",
        "run_id": "abc123",
        "raw_prompt": "can I reuse anything here for a RAG?",
        "brief": "Assess whether the retrieval stack can be reused. Do not implement.",
        "decision": "approve",
        "source_root": "/Users/x/Projects/dynaflows",
        "cost": "$0.0000 spent, $0.0000 avoided by cache, 1 call(s)",
        "relevant_paths": ["src/dynaflows/playbook/pack.py"],
        "invented_paths": [],
        "assumptions": ["Markdown corpus."],
        "at": datetime(2026, 9, 13, 14, 30, tzinfo=UTC),
    }
    return BriefRecord(**{**base, **overrides})  # type: ignore[arg-type]


def test_the_brief_file_is_the_brief_and_nothing_else(tmp_path: Path) -> None:
    """It is pasted into another model. A provenance header would be pasted
    too, and the receiving model would have to decide what to do with it."""
    record = a_record()

    brief_path, _ = save_brief(tmp_path, record)

    assert brief_path.read_text() == record.brief + "\n"


def test_the_record_carries_the_original_request(tmp_path: Path) -> None:
    """A brief with no record of what produced it is unusable a week later:
    the whole point of keeping it is comparing the rewrite to the request."""
    _, record_path = save_brief(tmp_path, a_record())

    assert "can I reuse anything here for a RAG?" in record_path.read_text()


def test_the_record_does_not_depend_on_the_terminal(tmp_path: Path) -> None:
    """Capturing the rendered Rich panels would embed COLUMNS in the file, so
    the same run recorded on two terminals would diff line for line."""
    text = record_markdown(a_record())

    assert "│" not in text
    assert "╭" not in text
    assert max(len(line) for line in text.splitlines()) < 200


def test_the_record_is_deterministic() -> None:
    """Same record in, same bytes out -- or two runs cannot be compared."""
    assert record_markdown(a_record()) == record_markdown(a_record())


def test_an_empty_path_list_says_so_rather_than_vanishing() -> None:
    """AP-20. A missing section and an empty one read identically in a file
    and mean different things: nothing named, versus not recorded."""
    text = record_markdown(a_record(relevant_paths=[]))

    assert "it reads this as being about" in text
    assert "no paths named" in text


def test_invented_paths_appear_only_when_there_are_some() -> None:
    """This is the one heading that must be read when it has content, and a
    heading printed empty on every record is a heading nobody reads."""
    clean = record_markdown(a_record())
    dirty = record_markdown(a_record(invented_paths=["src/dynaflows/rag/store.py"]))

    assert "do not exist" not in clean
    assert "src/dynaflows/rag/store.py" in dirty


def test_a_rejected_brief_is_still_recorded(tmp_path: Path) -> None:
    """A rejection is the most informative run there is -- it is the model
    being wrong, with a human's verdict attached. Discarding it keeps only
    the successes, which is the wrong half."""
    _, record_path = save_brief(tmp_path, a_record(decision="reject"))

    assert "**decision** reject" in record_path.read_text()


def test_the_directory_is_created(tmp_path: Path) -> None:
    _, record_path = save_brief(tmp_path / "deep" / ".ai", a_record())

    assert record_path.exists()


def test_two_runs_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """Thread ids are the filename. If they collided, an A/B test would
    silently compare a run against itself."""
    first, _ = save_brief(tmp_path, a_record(thread_id="brief-aaa"))
    second, _ = save_brief(tmp_path, a_record(thread_id="brief-bbb"))

    assert first != second
    assert {p.name for p in tmp_path.iterdir()} == {
        "brief-aaa.brief.txt",
        "brief-aaa.md",
        "brief-bbb.brief.txt",
        "brief-bbb.md",
    }
