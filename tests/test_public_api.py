import importlib

import pytest

import provider_runtime
from provider_runtime.runtime import ModelRuntime as PublicModelRuntime


def test_top_level_exports_public_runtime_facade() -> None:
    assert provider_runtime.ModelRuntime is PublicModelRuntime


def test_router_module_is_not_public_api() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("provider_runtime.router")


def test_target_surface_is_exported() -> None:
    for name in (
        "DEFAULT_CATALOG",
        "ModelCatalog",
        "ModelCapability",
        "PromptCacheCapability",
        "Pricing",
        "ProviderApiKey",
        "ProviderApiKeySource",
        "ProviderBaseUrls",
        "RetryAttempt",
        "RetryAttemptStatus",
        "GenerateRequestPlan",
        "lower_generate_request",
        "NoNetworkRuntime",
        "ScriptedRuntime",
        "CostBreakdown",
        "estimate_cost",
    ):
        assert hasattr(provider_runtime, name)
