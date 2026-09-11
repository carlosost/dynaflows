"""A reproduction that differs from production proves nothing.

These tests exist because the first version of `langchain_response_format`
returned a pydantic CLASS -- LangChain hands the class straight to the openai
SDK, which does the real conversion -- so the diagnostic would have serialised
a `ModelMetaclass` and "reproduced" a call the product never makes. That is the
failure mode worth a test: not that the helper errors, but that it quietly
sends something else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dynaflows.gateway import invoker as invoker_module
from dynaflows.gateway.invoker import langchain_response_format, raw_completion
from dynaflows.graph.prompts import EnhancedPrompt, PlanDraft
from dynaflows.settings import get_settings


@pytest.mark.parametrize("schema", [EnhancedPrompt, PlanDraft])
def test_response_format_is_wire_shaped_not_a_class(schema: type) -> None:
    block = langchain_response_format(schema)

    assert isinstance(block, dict)
    assert block["type"] == "json_schema"
    inner = block["json_schema"]
    assert inner["strict"] is True
    assert inner["name"] == schema.__name__
    # Must be JSON, not a class hiding inside a dict.
    import json

    json.dumps(block)


def test_nested_schema_keeps_its_defs_strict() -> None:
    """PlanDraft is the schema that failed live; the nested object is the
    difference from the enhancer's flat one, so assert its strict form
    explicitly rather than trusting the round trip."""
    inner = langchain_response_format(PlanDraft)["json_schema"]["schema"]
    task = inner["$defs"]["PlannedTask"]

    assert task["additionalProperties"] is False
    # OpenAI strict mode requires EVERY property in `required`, including the
    # ones pydantic defaulted.
    assert set(task["required"]) == set(task["properties"])
    assert inner["additionalProperties"] is False


def test_raw_completion_sends_system_first_and_the_block_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sent: dict[str, Any] = {}

    class _Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"choices": []}

    def _post(url: str, **kwargs: Any) -> _Response:
        sent.update(kwargs["json"])
        return _Response()

    monkeypatch.setattr(invoker_module.httpx, "post", _post)

    block = langchain_response_format(PlanDraft)
    raw_completion(
        get_settings({"OPENROUTER_API_KEY": "k"}, root=tmp_path),
        "vendor/model",
        system="SYS",
        prompt="USER",
        max_tokens=512,
        response_format=block,
    )

    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"] == "SYS"
    assert sent["response_format"] == block
    assert sent["max_tokens"] == 512
    # ADR-006: the reproduction must be routed the same way the product is.
    assert sent["provider"] == {"require_parameters": True}
