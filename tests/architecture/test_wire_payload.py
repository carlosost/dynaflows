"""What the SDK actually puts on the wire, asserted.

Every other test in this repo drives the gateway with a fake invoker, so every
one of them is blind to anything the provider SDK does BELOW that seam. A live
run found the consequence: `ChatOpenAI` renames `max_tokens` to
`max_completion_tokens` unconditionally, OpenRouter's `require_parameters`
filter does not recognise that name for these endpoints, and every candidate
endpoint was dropped -- reported as a 404 "Filter by Parameters" for one model
and, for another, a 200 whose `choices` was null, which reached the user as
"'NoneType' object is not iterable".

That is AP-05 in its purest form: a rename between our call site and the
request, silent on both sides. The only test that can see it is one that reads
the payload. It uses a private LangChain method on purpose -- the alternative
is not testing the wire at all, and the wire is where the bug was.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dynaflows.contracts.calls import CallRequest
from dynaflows.contracts.tiers import Tier
from dynaflows.gateway.invoker import build_chat_client
from dynaflows.graph.prompts import PlanDraft
from dynaflows.settings import get_settings

_MESSAGES = [("system", "S"), ("human", "U")]


def _payload(tmp_path: Path, **request_kwargs: Any) -> dict[str, Any]:
    settings = get_settings({"OPENROUTER_API_KEY": "k"}, root=tmp_path)
    request = CallRequest(tier=Tier.FRONTIER, prompt="U", system="S", **request_kwargs)
    chat = build_chat_client(settings, "vendor/model", request)
    return dict(chat._get_request_payload(_MESSAGES))  # noqa: SLF001


def test_max_tokens_reaches_the_wire_under_that_name(tmp_path: Path) -> None:
    payload = _payload(tmp_path, max_tokens=4096, schema=PlanDraft)

    # The bug: LangChain moves a constructor max_tokens to this name.
    assert "max_completion_tokens" not in payload, (
        "LangChain renamed max_tokens; OpenRouter's require_parameters filter "
        "will drop every endpoint. Keep it in extra_body."
    )
    assert payload["extra_body"]["max_tokens"] == 4096


def test_require_parameters_survives_alongside_the_token_cap(tmp_path: Path) -> None:
    """ADR-006 and the token cap share one extra_body. Writing the second must
    not overwrite the first -- they are set in the same dict literal, which is
    exactly the kind of edit that silently drops a key."""
    payload = _payload(tmp_path, max_tokens=1024)

    assert payload["extra_body"]["provider"] == {"require_parameters": True}
    assert payload["extra_body"]["max_tokens"] == 1024


def test_no_token_cap_sends_no_token_parameter(tmp_path: Path) -> None:
    payload = _payload(tmp_path)

    assert "max_tokens" not in payload["extra_body"]
    assert "max_completion_tokens" not in payload
    assert "max_tokens" not in payload


@pytest.mark.parametrize("temperature", [None, 0.2])
def test_the_wire_keys_are_the_ones_we_chose(tmp_path: Path, temperature: float | None) -> None:
    """An exact key set, not a subset.

    A subset assertion passes when the SDK starts sending something new, and
    "something new" is precisely what `require_parameters` punishes. This test
    is meant to fail on an SDK upgrade: that failure is the notification.
    """
    payload = _payload(tmp_path, max_tokens=512, temperature=temperature)

    expected = {"model", "stream", "extra_body"}
    if temperature is not None:
        expected.add("temperature")
    assert set(payload) - {"messages"} == expected
