"""Immutable current-model facts from API declarations and authenticated Codex discovery."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

import provider_runtime.registry as registry
from provider_runtime.agent_runtime.errors import ProtocolDefect
from provider_runtime.agent_runtime.model_catalog import (
    AGENT_BACKEND_CONTRACT_REVISION,
    AgentUpgradeFacts,
    UpgradeTargetUnresolved,
    read_codex_model_catalog,
)
from provider_runtime.registry import GPT6_MODEL_IDS, api_model_catalog
from provider_runtime.types import Absent, Present


class _GenericClient:
    def __init__(self, pages: list[Mapping[str, object]]) -> None:
        self.pages = list(pages)
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    async def request(self, method: str, params: Mapping[str, object]) -> object:
        self.calls.append((method, dict(params)))
        if not self.pages:
            raise AssertionError("model reader requested an unscripted page")
        return self.pages.pop(0)


def _row(model_id: str, *, upgrade: str | None = None) -> dict[str, object]:
    return {
        "id": model_id,
        "model": model_id,
        "displayName": model_id.upper(),
        "hidden": False,
        "inputModalities": ["text", "image"],
        "supportedReasoningEfforts": [
            {"reasoningEffort": effort, "description": effort}
            for effort in ("low", "medium", "high", "xhigh", "max")
        ],
        "defaultReasoningEffort": "high",
        "upgrade": upgrade,
        # Not in the pinned public SDK contract; these must not leak into facts.
        "contextWindow": 999_999,
        "maxOutputTokens": 111_111,
    }


def _complete_codex_rows() -> list[dict[str, object]]:
    return [_row(model_id) for model_id in GPT6_MODEL_IDS]


@pytest.mark.parametrize("missing_model", GPT6_MODEL_IDS)
async def test_codex_catalog_rejects_missing_gpt6_model(missing_model: str) -> None:
    rows = [row for row in _complete_codex_rows() if row["id"] != missing_model]
    with pytest.raises(ProtocolDefect, match="omitted an approved GPT-6 model"):
        await read_codex_model_catalog(_GenericClient([{"data": rows}]))


@pytest.mark.parametrize("missing_effort", ("low", "medium", "high", "xhigh", "max"))
async def test_codex_catalog_rejects_missing_effort(missing_effort: str) -> None:
    rows = _complete_codex_rows()
    options = rows[0]["supportedReasoningEfforts"]
    assert isinstance(options, list)
    rows[0]["supportedReasoningEfforts"] = [
        item
        for item in options
        if isinstance(item, dict) and item.get("reasoningEffort") != missing_effort
    ]
    with pytest.raises(ProtocolDefect, match="reasoning effort|source default"):
        await read_codex_model_catalog(_GenericClient([{"data": rows}]))


async def test_codex_catalog_reads_every_page_and_normalizes_only_public_facts() -> None:
    rows = _complete_codex_rows()
    rows[0]["upgrade"] = GPT6_MODEL_IDS[1]
    hidden = {**_row("gpt-5.6-sol"), "hidden": True}
    client = _GenericClient(
        [
            {"data": [rows[0], hidden], "nextCursor": "page-2", "revision": "native-revision"},
            {"data": rows[1:], "nextCursor": None, "revision": "native-revision"},
        ]
    )
    catalog = await read_codex_model_catalog(
        client, now=lambda: datetime(2026, 9, 25, 12, tzinfo=UTC)
    )
    assert client.calls == [
        ("model/list", {"includeHidden": False, "cursor": None}),
        ("model/list", {"includeHidden": False, "cursor": "page-2"}),
    ]
    assert catalog.backend_contract_revision == AGENT_BACKEND_CONTRACT_REVISION
    assert catalog.supports_frozen_mcp_tools is True
    assert catalog.native_revision == Present("native-revision")
    assert tuple(row.key for row in catalog.models) == GPT6_MODEL_IDS
    astra = catalog.models[0]
    assert astra.dispatch_model == GPT6_MODEL_IDS[0]
    assert astra.source_context_window == Absent()
    assert astra.source_max_output_tokens == Absent()
    assert astra.source_default_reasoning == Present("high")
    assert astra.upgrade == Present(AgentUpgradeFacts(target_key=GPT6_MODEL_IDS[1]))
    assert tuple(item.key for item in astra.reasoning) == ("low", "medium", "high", "xhigh", "max")
    assert catalog.diagnostics == ()


async def test_codex_definition_hash_excludes_observation_provenance() -> None:
    rows = _complete_codex_rows()
    first = await read_codex_model_catalog(
        _GenericClient([{"data": rows, "revision": "native-a"}]),
        now=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )
    second = await read_codex_model_catalog(
        _GenericClient([{"data": rows, "revision": "native-b"}]),
        now=lambda: datetime(2026, 9, 25, tzinfo=UTC),
    )
    changed_rows = _complete_codex_rows()
    changed_rows[0]["displayName"] = "a changed label"
    changed = await read_codex_model_catalog(_GenericClient([{"data": changed_rows}]))
    assert first.definition_revision == second.definition_revision
    assert first.models[0].row_fingerprint == second.models[0].row_fingerprint
    assert first.definition_revision != changed.definition_revision


@pytest.mark.parametrize(
    "pages, message",
    [
        (
            [
                {"data": _complete_codex_rows(), "nextCursor": "again"},
                {"data": [], "nextCursor": "again"},
            ],
            "repeated",
        ),
        ({"data": _complete_codex_rows() + [_row(GPT6_MODEL_IDS[0])]}, "duplicate model ids"),
        (
            {
                "data": [{**_row(GPT6_MODEL_IDS[0]), "defaultReasoningEffort": "minimal"}]
                + _complete_codex_rows()[1:]
            },
            "source default|reasoning effort",
        ),
        (
            {
                "data": [
                    {
                        key: value
                        for key, value in _row(GPT6_MODEL_IDS[0]).items()
                        if key != "hidden"
                    }
                ]
                + _complete_codex_rows()[1:]
            },
            "boolean hidden fact",
        ),
    ],
)
async def test_codex_catalog_rejects_repeated_or_incomplete_source_facts(
    pages: list[Mapping[str, object]] | Mapping[str, object], message: str
) -> None:
    page_list = pages if isinstance(pages, list) else [pages]
    with pytest.raises(ProtocolDefect, match=message):
        await read_codex_model_catalog(_GenericClient(page_list))


async def test_codex_catalog_refuses_more_than_64_pages_without_truncation() -> None:
    pages: list[Mapping[str, object]] = [
        {"data": _complete_codex_rows() if index == 0 else [], "nextCursor": f"p-{index + 1}"}
        for index in range(64)
    ]
    client = _GenericClient(pages)
    with pytest.raises(ProtocolDefect, match="exceeded 64 pages"):
        await read_codex_model_catalog(client)
    assert len(client.calls) == 64


async def test_codex_upgrade_diagnostic_is_typed_and_never_guesses() -> None:
    rows = _complete_codex_rows()
    rows[0]["upgrade"] = "missing"
    catalog = await read_codex_model_catalog(_GenericClient([{"data": rows}]))
    assert catalog.models[0].upgrade == Absent()
    assert catalog.diagnostics == (
        UpgradeTargetUnresolved(model_key=GPT6_MODEL_IDS[0], native_target="missing"),
    )


def test_api_catalog_is_exact_immutable_and_the_registry_has_no_public_rows() -> None:
    first = api_model_catalog()
    second = api_model_catalog()
    assert first == second
    assert first.backend_contract_revision == "provider-runtime.api-model-catalog.v3"
    assert first.registry_revision == registry.REGISTRY_REVISION
    assert len(first.definition_revision) == 64
    assert tuple(row.model_ref for row in first.models) == (
        "openai:gpt-6-astra",
        "openai:gpt-6-sol",
        "openai:gpt-6-luna",
        "anthropic:claude-fable-5-1",
        "anthropic:claude-opus-5-5",
        "anthropic:claude-sonnet-5",
        "gemini:gemini-3.8-flash",
        "deepseek:deepseek-flash",
        "xai:grok-4.7",
    )
    assert len({row.row_fingerprint for row in first.models}) == len(first.models)
    assert all(len(row.row_fingerprint) == 64 for row in first.models)
    assert all(row.label and all(item.label for item in row.reasoning) for row in first.models)
    defaults = {row.model_ref: row.source_default_reasoning for row in first.models}
    assert defaults["openai:gpt-6-astra"] == Absent()
    assert defaults["openai:gpt-6-sol"] == Present("standard/medium")
    assert defaults["anthropic:claude-opus-5-5"] == Present("adaptive/medium")
    for row in first.models:
        assert row.reasoning
        if isinstance(row.source_default_reasoning, Present):
            assert row.source_default_reasoning.value in tuple(fact.key for fact in row.reasoning)

    assert registry.__all__ == ["GPT6_MODEL_IDS", "api_model_catalog"]
    for deleted in ("ROWS", "ModelRow", "OpenRouterRouting", "resolve", "resolve_target"):
        assert not hasattr(registry, deleted), f"private registry owner leaked as {deleted}"
