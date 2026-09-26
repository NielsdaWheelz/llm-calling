"""Temporary acceptance proof for native structured output with tools."""

from __future__ import annotations

from dataclasses import replace

import pytest

from provider_runtime.errors import InvalidRequest
from provider_runtime.registry import api_model_catalog
from provider_runtime.types import CanonicalTool, StrictJsonOutput, StructuredContent, Succeeded
from tests.test_runtime import FakeEngine, make_intent, make_runtime, structured_succeeded


@pytest.mark.parametrize("model", ("gpt-6-sol", "gpt-6-luna"))
async def test_catalog_fact_admits_strict_output_with_frozen_tools(model: str) -> None:
    row = next(item for item in api_model_catalog().models if item.model_ref == f"openai:{model}")
    assert row.structured_with_tools is True
    tool = CanonicalTool(name="lookup", description="", parameters={"type": "object"})
    engine = FakeEngine(generate_script=[structured_succeeded({"ok": True})])
    intent = replace(
        make_intent(),
        target=replace(make_intent().target, model=model),
        tools=(tool,),
        output=StrictJsonOutput(name="Out", schema={"type": "object"}),
    )
    outcome = await make_runtime(engine).generate(intent)
    assert isinstance(outcome, Succeeded)
    assert isinstance(outcome.response.content, StructuredContent)
    assert len(engine.generate_calls) == 1


async def test_unproven_model_still_rejects_combination_before_dispatch() -> None:
    row = next(
        item for item in api_model_catalog().models if item.model_ref == "openai:gpt-6-astra"
    )
    assert row.structured_with_tools is False
    tool = CanonicalTool(name="lookup", description="", parameters={"type": "object"})
    engine = FakeEngine()
    intent = replace(
        make_intent(),
        target=replace(make_intent().target, model="gpt-6-astra"),
        tools=(tool,),
        output=StrictJsonOutput(name="Out", schema={"type": "object"}),
    )
    with pytest.raises(InvalidRequest):
        await make_runtime(engine).generate(intent)
    assert not engine.generate_calls
