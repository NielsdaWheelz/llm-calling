"""Public package surface: a small facade (≤ 40 names), every export real and
importable, the deleted planner/catalog/transport surface gone from the root.

The full contract vocabulary stays importable from ``provider_runtime.types``;
test doubles from ``provider_runtime.testing``; registry rows from
``provider_runtime.registry``.
"""

from __future__ import annotations

import provider_runtime


def test_all_has_at_most_forty_names() -> None:
    assert len(provider_runtime.__all__) <= 40, (
        f"the facade is capped at 40 names, got {len(provider_runtime.__all__)}"
    )


def test_all_is_sorted_and_unique() -> None:
    assert provider_runtime.__all__ == sorted(provider_runtime.__all__)
    assert len(provider_runtime.__all__) == len(set(provider_runtime.__all__))


def test_every_all_name_is_importable_from_package_root() -> None:
    missing = [name for name in provider_runtime.__all__ if not hasattr(provider_runtime, name)]
    assert not missing, f"__all__ names missing from the package root: {missing}"


def test_high_traffic_surface_is_exported() -> None:
    for name in (
        # runtime facade
        "Credentials",
        "ProviderRuntime",
        "NonGenerationCallFailed",
        # derived cost
        "estimate_cost",
        # intent side
        "GenerateIntent",
        "ProviderTarget",
        "PromptBlock",
        "ImageBlock",
        "SystemMessage",
        "UserMessage",
        "AssistantMessage",
        "ToolResultMessage",
        "CanonicalTool",
        "TextOutput",
        # outcomes
        "CallOutcome",
        "Succeeded",
        "Refused",
        "Incomplete",
        "Cancelled",
        "Failed",
        "CallMeta",
        "TokenUsage",
        "StructuredReply",
        "TextContent",
        "StructuredContent",
        # stream envelope
        "RuntimeStreamEvent",
        "StreamStart",
        "TextDelta",
        "TerminalEvent",
        # embed port
        "EmbeddingCall",
        "EmbeddingResponse",
        "ProviderCredential",
        # owned absence
        "Present",
        "Absent",
    ):
        assert name in provider_runtime.__all__, f"{name} missing from __all__"


def test_deleted_surface_is_absent_from_the_package_root() -> None:
    for name in (
        # catalog / certification
        "CATALOG",
        "CATALOG_REVISION",
        "Catalog",
        "ChatModelContract",
        "DirectCertification",
        "OperatorCertified",
        "OperatorUncertified",
        "check_catalog_freshness",
        # planner / cache plans
        "CACHE_AFFINITY_VERSION",
        "EXTERNAL_LLM_RETRY",
        "OPENROUTER_SINGLE_ATTEMPT",
        "plan_generate",
        "PlanRejected",
        "CachePlan",
        "BlockStability",
        "CacheScope",
        "Dynamic",
        "Stable",
        "DraftRequest",
        "FinalizedProviderCall",
        "FinalizedProviderRequest",
        # transport
        "SseEvent",
        "Transport",
        "TransportResponse",
        "TransportStreamResponse",
        # accounting
        "Accounting",
        "CostBreakdown",
        "cost_from_accounting",
        # transcription (deleted port)
        "TranscriptionCall",
        "TranscriptionResponse",
        # testing doubles live in provider_runtime.testing, not the root
        "ScriptedRuntime",
        "NoNetworkRuntime",
        "CapturedRuntimeCall",
    ):
        assert name not in provider_runtime.__all__, f"{name} must stay unexported"
        assert not hasattr(provider_runtime, name), f"{name} must be gone from the package root"
