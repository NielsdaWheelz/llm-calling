"""Conformance for both immutable source-model catalog ownership boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

import provider_runtime.registry as registry
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


class _ResponseModel:
    """Identity token proving the reader uses the generated response model argument."""


class _GenericClient:
    def __init__(self, pages: list[Mapping[str, object]]) -> None:
        self.pages = list(pages)
        self.calls: list[tuple[str, Mapping[str, object], type[Any]]] = []

    async def request(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        response_model: type[Any],
    ) -> object:
        self.calls.append((method, dict(params), response_model))
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
                    _row("old", "old-wire", upgrade="new-wire"),
                    _row("hidden", "hidden-wire", hidden=True),
                ],
                "nextCursor": "page-2",
                "revision": "native-provenance-only",
            },
            {
                "data": [_row("new", "new-wire")],
                "nextCursor": None,
                "revision": "native-provenance-only",
            },
        ]
    )

    catalog = await read_codex_model_catalog(
        client,
        _ResponseModel,
        now=lambda: datetime(2026, 8, 31, 12, tzinfo=UTC),
    )

    assert client.calls == [
        (
            "model/list",
            {"includeHidden": False, "cursor": None},
            _ResponseModel,
        ),
        (
            "model/list",
            {"includeHidden": False, "cursor": "page-2"},
            _ResponseModel,
        ),
    ]
    assert catalog.backend_contract_revision == AGENT_BACKEND_CONTRACT_REVISION
    assert catalog.native_revision == Present("native-provenance-only")
    assert tuple(row.key for row in catalog.models) == ("old", "new")
    old = catalog.models[0]
    assert old.dispatch_model == "old-wire"
    assert old.source_context_window == Absent()
    assert old.source_max_output_tokens == Absent()
    assert old.source_default_reasoning == Present("high")
    assert old.upgrade == Present(AgentUpgradeFacts(target_key="new"))
    assert catalog.diagnostics == ()


async def test_codex_definition_hash_excludes_observation_provenance() -> None:
    first = await read_codex_model_catalog(
        _GenericClient([{"data": [_row("one", "one-wire")], "revision": "native-a"}]),
        _ResponseModel,
        now=lambda: datetime(2026, 8, 30, tzinfo=UTC),
    )
    second = await read_codex_model_catalog(
        _GenericClient([{"data": [_row("one", "one-wire")], "revision": "native-b"}]),
        _ResponseModel,
        now=lambda: datetime(2026, 8, 31, tzinfo=UTC),
    )
    changed = await read_codex_model_catalog(
        _GenericClient([{"data": [_row("one", "changed-wire")]}]),
        _ResponseModel,
    )

    assert first.definition_revision == second.definition_revision
    assert first.models[0].row_fingerprint == second.models[0].row_fingerprint
    assert first.definition_revision != changed.definition_revision


@pytest.mark.parametrize(
    "pages, message",
    [
        (
            [
                {"data": [_row("one", "one-wire")], "nextCursor": "again"},
                {"data": [_row("two", "two-wire")], "nextCursor": "again"},
            ],
            "repeated",
        ),
        (
            [{"data": [_row("one", "one-wire"), _row("one", "two-wire")]}],
            "duplicate model ids",
        ),
        (
            [{"data": [_row("one", "one-wire", default="max")]}],
            "source default",
        ),
        (
            [
                {
                    "data": [
                        {
                            key: value
                            for key, value in _row("one", "one-wire").items()
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
        await read_codex_model_catalog(_GenericClient(pages), _ResponseModel)


async def test_codex_catalog_refuses_more_than_64_pages_without_truncation() -> None:
    pages: list[Mapping[str, object]] = [
        {"data": [_row(f"model-{index}", f"wire-{index}")], "nextCursor": f"p-{index + 1}"}
        for index in range(64)
    ]
    client = _GenericClient(pages)

    with pytest.raises(ProtocolDefect, match="exceeded 64 pages"):
        await read_codex_model_catalog(client, _ResponseModel)
    assert len(client.calls) == 64


async def test_codex_upgrade_diagnostics_are_typed_and_never_guess() -> None:
    unresolved = await read_codex_model_catalog(
        _GenericClient([{"data": [_row("one", "one-wire", upgrade="missing")]}]),
        _ResponseModel,
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
                        _row("source", "source-wire", upgrade="shared"),
                        _row("shared", "wire-two"),
                        _row("three", "shared"),
                    ]
                }
            ]
        ),
        _ResponseModel,
    )
    assert ambiguous.models[0].upgrade == Absent()
    assert ambiguous.diagnostics == (
        UpgradeTargetAmbiguous(model_key="source", native_target="shared"),
    )


def test_api_catalog_is_exact_immutable_and_the_registry_has_no_public_rows() -> None:
    first = api_model_catalog()
    second = api_model_catalog()

    assert first == second
    assert first.backend_contract_revision == "provider-runtime.api-model-catalog.v1"
    assert first.registry_revision == registry.REGISTRY_REVISION
    assert len(first.definition_revision) == 64
    assert tuple(row.model_ref for row in first.models) == (
        "openai:gpt-5.6-sol",
        "openai:gpt-5.6-terra",
        "openai:gpt-5.6-luna",
        "anthropic:claude-sonnet-5",
        "anthropic:claude-fable-5",
        "gemini:gemini-3.5-flash",
        "moonshot:kimi-k3",
        "openrouter:kimi-k3",
        "deepseek:deepseek-v4-pro",
        "deepseek:deepseek-v4-flash",
        "xai:grok-4.5",
    )
    assert len({row.row_fingerprint for row in first.models}) == len(first.models)
    assert all(len(row.row_fingerprint) == 64 for row in first.models)
    defaults = {row.model_ref: row.source_default_reasoning for row in first.models}
    assert defaults["openrouter:kimi-k3"] == Absent()
    assert defaults["moonshot:kimi-k3"] == Present("max")
    for row in first.models:
        assert row.reasoning
        if isinstance(row.source_default_reasoning, Present):
            assert row.source_default_reasoning.value in tuple(fact.key for fact in row.reasoning)

    assert registry.__all__ == ["api_model_catalog"]
    for deleted in ("ROWS", "ModelRow", "OpenRouterRouting", "resolve", "resolve_target"):
        assert not hasattr(registry, deleted), f"private registry owner leaked as {deleted}"
