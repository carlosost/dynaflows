"""What the enhancer is allowed to hand a human at G1.

The rules live in ENHANCER_SYSTEM too, but a prompt is a request and not a
guarantee. The live run that prompted these enforced neither: it named eleven
paths, three of them test modules and one of them `docs/PROJECT_MEMORY.md`.
"""

from __future__ import annotations

import pytest

from dynaflows.graph.nodes import _MAX_RELEVANT_PATHS, _useful_paths
from dynaflows.graph.prompts import ENHANCER_SYSTEM
from dynaflows.store import is_test_path

pytestmark = pytest.mark.deterministic


def test_test_modules_are_dropped() -> None:
    """A test module says where a thing is ASSERTED, never where it lives --
    the least useful entry in a list whose job is to say where to look."""
    kept = _useful_paths(
        [
            "src/dynaflows/playbook/store.py",
            "tests/unit/test_playbook_store.py",
            "src/dynaflows/playbook/repository.py",
        ]
    )

    assert kept == [
        "src/dynaflows/playbook/store.py",
        "src/dynaflows/playbook/repository.py",
    ]


def test_a_list_of_only_tests_survives() -> None:
    """Dropping everything would say less than saying what it found. If every
    path is a test the model probably understood the subject correctly."""
    paths = ["tests/unit/test_a.py", "tests/unit/test_b.py"]

    assert _useful_paths(paths) == paths


def test_the_list_is_capped() -> None:
    """Eleven paths is not a stronger answer than six, it is a weaker one."""
    kept = _useful_paths([f"src/dynaflows/m{i}.py" for i in range(11)])

    assert len(kept) == _MAX_RELEVANT_PATHS


def test_order_is_preserved() -> None:
    """The model put its best guess first. Re-ranking here would be this
    function inventing an opinion it does not have."""
    paths = ["src/z.py", "src/a.py", "src/m.py"]

    assert _useful_paths(paths) == paths


@pytest.mark.parametrize(
    "path",
    ["tests/unit/test_x.py", "tests/conftest.py", "src/pkg/tests/helpers.py", "test_x.py"],
)
def test_test_paths_are_recognised(path: str) -> None:
    assert is_test_path(path)


@pytest.mark.parametrize("path", ["src/dynaflows/cli.py", "docs/PROJECT_MEMORY.md"])
def test_ordinary_paths_are_not_mistaken_for_tests(path: str) -> None:
    assert not is_test_path(path)


def test_the_prompt_forbids_third_person_narration() -> None:
    """The p2 brief opened "The user wants to understand how the playbook
    search works" -- a description of a request, which whoever receives it has
    to translate back into a request before starting."""
    assert "third person" in ENHANCER_SYSTEM
    assert "Start with a verb." in ENHANCER_SYSTEM


def test_the_prompt_demands_executed_verification() -> None:
    """The finding that decided the A/B test: what separated a good answer
    from a bad one was whether the thread RAN the code. One answer claimed a
    file did not parse, with a line number and a verbatim quote, and it parsed
    fine -- on the interpreter the project actually uses."""
    assert "interpreter version" in ENHANCER_SYSTEM
    assert "reading it is not verifying it" in ENHANCER_SYSTEM
