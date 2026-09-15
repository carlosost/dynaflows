"""What the enhancer is allowed to hand a human at G1.

The rules live in ENHANCER_SYSTEM too, but a prompt is a request and not a
guarantee. The live run that prompted these enforced neither: it named eleven
paths, three of them test modules and one of them `docs/PROJECT_MEMORY.md`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dynaflows.graph.nodes import _MAX_RELEVANT_PATHS, _settled_intent, _useful_paths
from dynaflows.graph.prompts import (
    ENHANCER_SYSTEM,
    STANDING_REQUIREMENTS,
    EnhancedPrompt,
    asks_a_question,
    compose_brief,
)
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


@pytest.mark.parametrize(
    "narration",
    [
        "I'll examine the playbook search implementation to understand how it works.",
        "I need to read the playbook indexing files to understand what would change.",
        "Let me start by reading the key files.",
        "We'll trace the Markdown assumptions through the pipeline.",
        "First, I will look at the repository module.",
    ],
)
def test_agent_narration_is_rejected_at_the_contract(narration: str) -> None:
    """Both retest briefs came back as role-play -- the model announcing what
    it was about to do instead of writing a brief. The prompt already forbade
    the first person and both models ignored it, which is the ordinary outcome
    for a weak model and a long instruction.

    A rule that lives only in a prompt is a request. This one is checkable by
    inspection, so it is checked: the schema rejects it, which buys a repair
    call naming the actual problem and then the next model in the chain.
    """
    with pytest.raises(ValidationError):
        EnhancedPrompt(enhanced=narration)


@pytest.mark.parametrize(
    "brief",
    [
        "Explain how the playbook search works and whether it scales to a few thousand documents.",
        "Identify what would have to change to index something other than Markdown.",
        "Investigate the login bug and name the failing path.",
        "Trace the retrieval pipeline from chunker to repository.",
    ],
)
def test_a_real_brief_is_accepted(brief: str) -> None:
    """The other half, and the more important one. A check that fires on
    ordinary work is a check people disable (playbook 5.2, Pattern 5), and
    "Investigate..." begins with a verb that a sloppier pattern would catch."""
    assert EnhancedPrompt(enhanced=brief).enhanced == brief


def test_the_standing_requirements_are_appended_not_generated() -> None:
    """The drift this fixes.

    The verification requirement was a paragraph in ENHANCER_SYSTEM telling
    the model to write it "in your own words". It is the SAME SENTENCE every
    time, so that paid tokens for a constant and then depended on a weak
    model's instruction-following for whether a standing policy appeared at
    all -- and in both live runs it appeared nowhere.

    The enhancer's job is to sharpen THIS request. Executor policy is a
    constant and belongs in the program.
    """
    assert "interpreter version" not in ENHANCER_SYSTEM
    assert "interpreter version" in STANDING_REQUIREMENTS

    composed = compose_brief("Explain how retrieval works.")

    assert composed.startswith("Explain how retrieval works.")
    assert STANDING_REQUIREMENTS in composed


def test_the_prompt_forbids_the_first_person_mechanically() -> None:
    """Belt and braces: the schema is the enforcement, but a model told
    plainly gets it right more often, and the two must not disagree."""
    assert "IMPERATIVE MOOD" in ENHANCER_SYSTEM
    assert "There is no first person in a brief." in ENHANCER_SYSTEM


def test_the_standing_requirements_scope_what_execution_may_touch() -> None:
    """The first live run of the verification requirement broke the repo.

    "Verify by RUNNING it" is an instruction to execute, and it was appended
    to every brief with no statement of where, or what execution was allowed
    to touch. The executor built a benchmark harness, ran `git status` inside
    the repository, and left a `.git/index.lock` behind that would have failed
    the next commit.

    A demand to execute that does not bound execution is half an instruction.
    """
    for clause in (
        "scratch directory outside the repository",
        "take no lock on it",
        "`git status` takes one",
        "Every number states how it was obtained",
    ):
        assert clause in STANDING_REQUIREMENTS, clause


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("how does the playbook search work", "question"),
        ("what would I have to change to index something other than markdown", "question"),
        ("add support for .txt files", "work"),
    ],
)
def test_intent_is_carried_rather_than_inferred(request_text: str, expected: str) -> None:
    """The contract exists; whether a model fills it well is measured live.

    What this pins is that the field EXISTS and reaches the planner, because
    the failure it fixes was the planner inferring intent from wording:
    "what would I have to change here" came back as "Modify the pipeline so
    that...", a work order for a question. ADR-024 gave the worker the
    answer/analyse division and left the enhancer guessing.
    """
    assert EnhancedPrompt(enhanced=f"Explain {request_text}.", intent=expected).intent == expected


def test_the_default_intent_is_question() -> None:
    """Asymmetric costs. Answering a question when work was wanted costs one
    turn. Doing work when a question was asked costs a diff nobody asked for,
    against code the user was still deciding about."""
    assert EnhancedPrompt(enhanced="Explain how retrieval works.").intent == "question"


def test_the_planner_is_told_which_it_is() -> None:
    from dynaflows.graph.prompts import PLANNER_SYSTEM

    assert "QUESTION or asked for" in PLANNER_SYSTEM
    assert "is a question about a change" in PLANNER_SYSTEM


@pytest.mark.parametrize(
    "narration",
    [
        "Looking at the playbook index mechanism to understand how stale results are avoided.",
        "Examining the retrieval stack for the answer.",
        "Reviewing how drift detection works in this repository.",
        "Analyzing the chunker to determine what would change.",
        "Starting with the store module.",
    ],
)
def test_a_gerund_opening_is_rejected_too(narration: str) -> None:
    """The first pattern caught first-person narration and the very next live
    run produced the other shape: "Looking at the playbook index mechanism..."
    -- no first person anywhere, and the same failure. A participle names an
    activity in progress; a brief names a task to be done."""
    with pytest.raises(ValidationError):
        EnhancedPrompt(enhanced=narration)


@pytest.mark.parametrize(
    "brief",
    [
        "Check whether the FTS index agrees with the chunk table, then report.",
        "Read the retrieval layer and explain the anchor lookup.",
        "Trace the drift check from store.drift into index_corpus.",
        "Review the chunker and name every Markdown assumption.",
    ],
)
def test_the_imperative_of_those_same_verbs_is_accepted(brief: str) -> None:
    """The half that matters more. "Check" and "Read" are how a brief
    legitimately opens; only "Checking" and "Reading" are the failure. A gate
    that fires on ordinary work is a gate people disable (playbook 5.2)."""
    assert EnhancedPrompt(enhanced=brief).enhanced == brief


@pytest.mark.parametrize(
    ("raw", "interrogative"),
    [
        ("how does the playbook index avoid returning stale results", True),
        ("what would I have to change here to index something other than markdown", True),
        ("why does resume analyse the wrong tree", True),
        ("can I reuse anything here for a RAG?", True),
        ("would it scale to a few thousand documents?", True),
        ("add support for .txt files", False),
        ("fix the login bug", False),
        ("refactor the gateway to use a protocol", False),
        ("the login page 500s on a bad token", False),
        # Polite imperatives. Interrogative grammar, a request for work.
        ("can you add support for .txt files", False),
        ("could you fix the failing test", False),
        ("please fix the failing test", False),
    ],
)
def test_the_question_form_is_decided_by_grammar(raw: str, interrogative: bool) -> None:
    assert asks_a_question(raw) is interrogative


def test_a_question_the_model_called_work_is_corrected() -> None:
    """The first live `run`. A 2.6B model read "how does the playbook index
    avoid returning stale results after a file changes" as `work`; approving
    that would have sent the planner to audit the retrieval code for defects
    instead of explaining it, at frontier prices."""
    settled = _settled_intent("how does the playbook index avoid stale results", "work")

    assert settled == "question"


def test_the_override_runs_one_way_only() -> None:
    """Grammar that says nothing leaves the model's judgement alone. "Add
    support for .txt" is not interrogative and neither is a bug report, and
    forcing those to `question` would break the case the field exists for."""
    assert _settled_intent("add support for .txt files", "work") == "work"
    assert _settled_intent("fix the login bug", "work") == "work"
    # And it never drags a question towards work.
    assert _settled_intent("how does retrieval work", "question") == "question"
