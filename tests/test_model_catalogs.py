"""Conformance for both immutable source-model catalog ownership boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import pytest

import provider_runtime.registry as registry
from provider_runtime.agent_runtime.codex_containment import (
    CODEX_CONTAINMENT_CATALOG_SHA256,
    CODEX_CONTAINMENT_SOURCE_SHA256,
    CODEX_CONTAINMENT_VERSION,
    materialize_codex_containment_catalog,
)
from provider_runtime.agent_runtime.errors import ProtocolDefect
from provider_runtime.agent_runtime.model_catalog import (
    AGENT_BACKEND_CONTRACT_REVISION,
    AgentUpgradeFacts,
    UpgradeTargetAmbiguous,
    UpgradeTargetUnresolved,
    read_codex_model_catalog,
)
from provider_runtime.registry import api_model_catalog
from provider_runtime.types import Absent, Present


class _GenericClient:
    def __init__(self, pages: list[Mapping[str, object]]) -> None:
        self.pages = list(pages)
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    async def request(
        self,
        method: str,
        params: Mapping[str, object],
    ) -> object:
        self.calls.append((method, dict(params)))
        if not self.pages:
            raise AssertionError("model reader requested an unscripted page")
        return self.pages.pop(0)


def _row(
    key: str,
    dispatch: str,
    *,
    hidden: bool = False,
    upgrade: str | None = None,
    default: str = "high",
) -> dict[str, object]:
    return {
        "id": key,
        "model": dispatch,
        "displayName": key.replace("-", " ").title(),
        "hidden": hidden,
        "inputModalities": ["text", "image"],
        "supportedReasoningEfforts": [
            {"reasoningEffort": "low", "description": "Low"},
            {"reasoningEffort": "high", "description": "High"},
        ],
        "defaultReasoningEffort": default,
        "upgrade": upgrade,
        # Deliberately ignored: these are not fields in the pinned public SDK.
        "contextWindow": 999_999,
        "maxOutputTokens": 111_111,
    }


async def test_codex_catalog_reads_every_page_and_normalizes_only_public_facts() -> None:
    client = _GenericClient(
        [
            {
                "data": [
                    _row("old", "gpt-6-sol", upgrade="gpt-6.1-sol"),
                    _row("hidden", "hidden-wire", hidden=True),
                ],
                "nextCursor": "page-2",
                "revision": "native-provenance-only",
            },
            {
                "data": [_row("new", "gpt-6.1-sol")],
                "nextCursor": None,
                "revision": "native-provenance-only",
            },
        ]
    )

    catalog = await read_codex_model_catalog(
        client,
        now=lambda: datetime(2026, 8, 31, 12, tzinfo=UTC),
    )

    assert client.calls == [
        (
            "model/list",
            {"includeHidden": False, "cursor": None},
        ),
        (
            "model/list",
            {"includeHidden": False, "cursor": "page-2"},
        ),
    ]
    assert catalog.backend_contract_revision == AGENT_BACKEND_CONTRACT_REVISION
    assert catalog.native_revision == Present("native-provenance-only")
    assert tuple(row.key for row in catalog.models) == ("old", "new")
    old = catalog.models[0]
    assert old.dispatch_model == "gpt-6-sol"
    assert old.source_context_window == Absent()
    assert old.source_max_output_tokens == Absent()
    assert old.source_default_reasoning == Present("high")
    assert old.upgrade == Present(AgentUpgradeFacts(target_key="new"))
    assert catalog.diagnostics == ()


async def test_codex_definition_hash_excludes_observation_provenance() -> None:
    first = await read_codex_model_catalog(
        _GenericClient([{"data": [_row("one", "gpt-6.1-sol")], "revision": "native-a"}]),
        now=lambda: datetime(2026, 8, 30, tzinfo=UTC),
    )
    second = await read_codex_model_catalog(
        _GenericClient([{"data": [_row("one", "gpt-6.1-sol")], "revision": "native-b"}]),
        now=lambda: datetime(2026, 8, 31, tzinfo=UTC),
    )
    changed = await read_codex_model_catalog(
        _GenericClient([{"data": [_row("one", "gpt-6.1-terra")]}]),
    )

    assert first.definition_revision == second.definition_revision
    assert first.models[0].row_fingerprint == second.models[0].row_fingerprint
    assert first.definition_revision != changed.definition_revision


@pytest.mark.parametrize(
    "pages, message",
    [
        (
            [
                {"data": [_row("one", "gpt-6.1-sol")], "nextCursor": "again"},
                {"data": [_row("two", "gpt-6-luna")], "nextCursor": "again"},
            ],
            "repeated",
        ),
        (
            [{"data": [_row("one", "gpt-6.1-sol"), _row("one", "gpt-6-luna")]}],
            "duplicate model ids",
        ),
        (
            [{"data": [_row("one", "gpt-6.1-sol", default="max")]}],
            "source default",
        ),
        (
            [
                {
                    "data": [
                        {
                            key: value
                            for key, value in _row("one", "gpt-6.1-sol").items()
                            if key != "hidden"
                        }
                    ]
                }
            ],
            "boolean hidden fact",
        ),
    ],
)
async def test_codex_catalog_rejects_repeated_or_incomplete_source_facts(
    pages: list[Mapping[str, object]], message: str
) -> None:
    with pytest.raises(ProtocolDefect, match=message):
        await read_codex_model_catalog(_GenericClient(pages))


async def test_codex_catalog_refuses_more_than_64_pages_without_truncation() -> None:
    pages: list[Mapping[str, object]] = [
        {"data": [_row(f"model-{index}", f"wire-{index}")], "nextCursor": f"p-{index + 1}"}
        for index in range(64)
    ]
    client = _GenericClient(pages)

    with pytest.raises(ProtocolDefect, match="exceeded 64 pages"):
        await read_codex_model_catalog(client)
    assert len(client.calls) == 64


async def test_codex_upgrade_diagnostics_are_typed_and_never_guess() -> None:
    unresolved = await read_codex_model_catalog(
        _GenericClient([{"data": [_row("one", "gpt-6.1-sol", upgrade="missing")]}]),
    )
    assert unresolved.models[0].upgrade == Absent()
    assert unresolved.diagnostics == (
        UpgradeTargetUnresolved(model_key="one", native_target="missing"),
    )

    ambiguous = await read_codex_model_catalog(
        _GenericClient(
            [
                {
                    "data": [
                        _row("source", "gpt-6-astra", upgrade="gpt-6.1-sol"),
                        _row("gpt-6.1-sol", "gpt-6-sol"),
                        _row("three", "gpt-6.1-sol"),
                    ]
                }
            ]
        ),
    )
    assert ambiguous.models[0].upgrade == Absent()
    assert ambiguous.diagnostics == (
        UpgradeTargetAmbiguous(model_key="source", native_target="gpt-6.1-sol"),
    )


def test_contained_catalog_disables_native_patch_and_preserves_all_other_source_facts(
    tmp_path: Path,
) -> None:
    source = (
        files("provider_runtime.agent_runtime")
        .joinpath(f"_codex_models_{CODEX_CONTAINMENT_VERSION.replace('.', '_')}.json")
        .read_bytes()
    )
    assert hashlib.sha256(source).hexdigest() == CODEX_CONTAINMENT_SOURCE_SHA256
    materialized = materialize_codex_containment_catalog(tmp_path).read_bytes()
    assert hashlib.sha256(materialized).hexdigest() == CODEX_CONTAINMENT_CATALOG_SHA256
    raw = json.loads(source)
    contained = json.loads(materialized)
    assert len(contained["models"]) == len(raw["models"])
    for original, restricted in zip(raw["models"], contained["models"], strict=True):
        assert restricted == {
            **original,
            "tool_mode": "direct",
            "experimental_supported_tools": [],
            "apply_patch_tool_type": None,
        }


async def test_codex_publication_is_explicit_and_rejects_unclassified_dispatch() -> None:
    catalog = await read_codex_model_catalog(
        _GenericClient([{"data": [_row("old", "gpt-6-sol"), _row("current", "gpt-6.1-sol")]}]),
    )
    assert catalog.backend_contract_revision == "provider-runtime.agent-model-catalog.v3"
    assert [(row.dispatch_model, row.offered) for row in catalog.models] == [
        ("gpt-6-sol", False),
        ("gpt-6.1-sol", True),
    ]
    with pytest.raises(ProtocolDefect, match="unclassified"):
        await read_codex_model_catalog(
            _GenericClient([{"data": [_row("new", "unreviewed-model")]}]),
        )


def test_api_catalog_is_exact_immutable_and_the_registry_has_no_public_rows() -> None:
    first = api_model_catalog()
    second = api_model_catalog()
    assert first == second
    assert first.backend_contract_revision == "provider-runtime.api-model-catalog.v5"
    assert first.registry_revision == registry.REGISTRY_REVISION
    assert len(first.definition_revision) == 64
    assert tuple(row.model_ref for row in first.models) == (
        "openai:gpt-6-astra",
        "openai:gpt-6.1-sol",
        "openai:gpt-6.1-terra",
        "openai:gpt-6-luna",
        "anthropic:claude-fable-5-1",
        "anthropic:claude-opus-5-5",
        "anthropic:claude-sonnet-5-5",
        "anthropic:claude-haiku-5-5",
        "gemini:gemini-4-argon",
        "gemini:gemini-3.8-flash",
        "gemini:gemini-3.5-flash-lite",
        "gemini:gemini-3.1-pro-preview",
        "deepseek:deepseek-flash",
        "deepseek:deepseek-v4-pro",
        "xai:grok-4.7",
        "xai:grok-build-0.1",
    )
    assert len({row.row_fingerprint for row in first.models}) == len(first.models)
    assert all(row.offered is True for row in first.models)
    assert all(len(row.row_fingerprint) == 64 for row in first.models)
    assert all(row.label and all(item.label for item in row.reasoning) for row in first.models)
    defaults = {row.model_ref: row.source_default_reasoning for row in first.models}
    assert defaults["openai:gpt-6-astra"] == Absent()
    assert defaults["openai:gpt-6.1-sol"] == Present("standard/medium")
    assert defaults["anthropic:claude-opus-5-5"] == Present("adaptive/medium")
    for row in first.models:
        assert row.reasoning
        if isinstance(row.source_default_reasoning, Present):
            assert row.source_default_reasoning.value in tuple(fact.key for fact in row.reasoning)

    assert registry.__all__ == ["GPT6_MODEL_IDS", "api_model_catalog"]
    for deleted in ("ROWS", "ModelRow", "OpenRouterRouting", "resolve", "resolve_target"):
        assert not hasattr(registry, deleted), f"private registry owner leaked as {deleted}"
