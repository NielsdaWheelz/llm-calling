"""Public package surface: everything in ``__all__`` is real, sorted, and
importable; codecs and deleted legacy modules are NOT part of the surface."""

from __future__ import annotations

import importlib

import pytest

import provider_runtime


def test_every_all_name_is_importable_from_package_root() -> None:
    missing = [name for name in provider_runtime.__all__ if not hasattr(provider_runtime, name)]
    assert not missing, f"__all__ names missing from the package root: {missing}"


def test_all_is_sorted_and_unique() -> None:
    assert provider_runtime.__all__ == sorted(provider_runtime.__all__)
    assert len(provider_runtime.__all__) == len(set(provider_runtime.__all__))


def test_core_surface_names_are_exported() -> None:
    for name in (
        # catalog
        "CATALOG",
        "CATALOG_REVISION",
        "Catalog",
        "ChatModelContract",
        "DirectCertification",
        "OperatorCertified",
        "OperatorUncertified",
        "check_catalog_freshness",
        # planning
        "CACHE_AFFINITY_VERSION",
        "EXTERNAL_LLM_RETRY",
        "OPENROUTER_SINGLE_ATTEMPT",
        "canonical_cache_contract_bytes",
        "compute_cache_affinity",
        "plan_generate",
        # runtime
        "NonGenerationCallFailed",
        "ProviderRuntime",
        # transport
        "SseEvent",
        "Transport",
        "TransportResponse",
        "TransportStreamResponse",
        # usage
        "CostBreakdown",
        "cost_from_accounting",
        # testing
        "CapturedRuntimeCall",
        "NoNetworkRuntime",
        "ScriptedRuntime",
        # non-generation ports
        "EmbeddingCall",
        "EmbeddingResponse",
        "TranscriptionCall",
        "TranscriptionResponse",
    ):
        assert name in provider_runtime.__all__, f"{name} missing from __all__"


def test_codec_and_private_modules_are_not_exported() -> None:
    for name in (
        "openai",
        "anthropic",
        "gemini",
        "moonshot",
        "openrouter",
        "embeddings",
        "_chat_completions_wire",
        "_signals",
    ):
        assert name not in provider_runtime.__all__, f"{name} must stay unexported"


def test_router_module_is_still_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("provider_runtime.router")


@pytest.mark.parametrize(
    "deleted_module",
    [
        "lowering",
        "tool_schema",
        "structured_output",
        "tool_arguments",
        "_adapter_runtime",
        "openai_compatible",
        "cloudflare",
        "endpoints",
        "_artifact_validation",
    ],
)
def test_deleted_legacy_modules_do_not_import(deleted_module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"provider_runtime.{deleted_module}")
