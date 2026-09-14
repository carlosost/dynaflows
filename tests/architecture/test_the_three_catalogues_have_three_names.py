"""Three different lists used to share one word, and it cost two readers.

  - the CAPABILITY list      (analyse, answer)      -> hashed into plan_hash
  - the SOURCE FILE list     (every file in a repo) -> budgeted at 6,000 tokens
  - the PLAYBOOK SECTION list (every indexed chunk) -> not budgeted at all

All three were "catalogue". Within one week, two independent readers of this
codebase borrowed one's constant for another's argument:

  - `catalog()`'s docstring divided 48.9 tokens/chunk by the SOURCE FILE
    budget and reported a ceiling of "about 10 documents". Two unrelated
    numbers.
  - An outside review asserted, with a line citation, that changing the
    SOURCE FILE summariser required bumping the CAPABILITY version because it
    is hashed into plan_hash. It is not; the capability list is.

Neither reader was careless. The names were. This test keeps them apart.
"""

from __future__ import annotations

import pathlib

import pytest

pytestmark = pytest.mark.deterministic

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "dynaflows"

# `gateway/probe.py` is deliberately exempt: its `catalogue` is the PROVIDER's
# model catalogue, which is that industry's own word for the thing and is not
# confusable with a list of files or sections. A rule that renames it would be
# a rule firing on correct work.
_EXEMPT = {"gateway/probe.py", "gateway/registry.py", "cli.py"}


def _sources() -> list[pathlib.Path]:
    return [p for p in _SRC.rglob("*.py") if "calibration/fixtures" not in p.as_posix()]


def test_no_identifier_is_merely_called_catalogue() -> None:
    """Prose may still say "catalogue" where it is unambiguous. An IDENTIFIER
    may not: an identifier is what gets imported into another module, where
    the sentence explaining which one it is does not travel with it."""
    offenders: list[str] = []
    for path in _sources():
        relative = path.relative_to(_SRC).as_posix()
        if relative in _EXEMPT:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            for bad in ("CATALOGUE", "catalog_line", "build_catalogue", "SourceCatalogue"):
                if bad in code:
                    offenders.append(f"{relative}:{number}: {bad}")
    assert not offenders, "name the list, not the genre:\n" + "\n".join(offenders)


def test_the_three_lists_are_importable_under_three_distinct_names() -> None:
    from dynaflows.graph.capabilities import CAPABILITIES_VERSION
    from dynaflows.playbook.repository import SqlitePlaybookRepository
    from dynaflows.store.source_map import SOURCE_MAP_BUDGET_TOKENS

    assert CAPABILITIES_VERSION
    assert SOURCE_MAP_BUDGET_TOKENS
    assert hasattr(SqlitePlaybookRepository, "section_map")


def test_only_the_capability_list_is_in_plan_hash() -> None:
    """The claim that went wrong. `plan_hash` carries the CAPABILITY version
    and nothing else catalogue-shaped -- editing the source-file summariser or
    the section map does not require a version bump, and the key says which."""
    import inspect

    from dynaflows.graph import planner

    body = inspect.getsource(planner.plan_hash)

    assert '"capabilities_version": CAPABILITIES_VERSION' in body
    assert "source_map" not in body
    assert "section_map" not in body
