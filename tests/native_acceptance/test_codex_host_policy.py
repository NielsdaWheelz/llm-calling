"""Temporary exact vendor-preservation and host-policy upgrade fences."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import pytest

from provider_runtime.agent_runtime import (
    CODEX_CONTAINMENT_CATALOG_SHA256,
    CODEX_CONTAINMENT_SOURCE_SHA256,
    UnsupportedCapability,
    materialize_codex_containment_catalog,
)
from provider_runtime.agent_runtime.codex_containment import _restrict_catalog


def test_n014_host_catalog_preserves_every_vendor_field_except_explicit_tool_policy(
    tmp_path: Path,
) -> None:
    original = (
        files("provider_runtime.agent_runtime").joinpath("_codex_models_0_160_0.json").read_bytes()
    )
    assert hashlib.sha256(original).hexdigest() == CODEX_CONTAINMENT_SOURCE_SHA256
    path = materialize_codex_containment_catalog(tmp_path)
    restricted = path.read_bytes()
    assert hashlib.sha256(restricted).hexdigest() == CODEX_CONTAINMENT_CATALOG_SHA256
    source = json.loads(original)
    output = json.loads(restricted)
    assert len(source["models"]) == len(output["models"])
    for before, after in zip(source["models"], output["models"], strict=True):
        assert after["tool_mode"] == "direct"
        assert after["experimental_supported_tools"] == []
        assert {
            key: value
            for key, value in before.items()
            if key not in {"tool_mode", "experimental_supported_tools"}
        } == {
            key: value
            for key, value in after.items()
            if key not in {"tool_mode", "experimental_supported_tools"}
        }
    luna = next(model for model in output["models"] if model["slug"] == "gpt-6-luna")
    assert any(row["effort"] == "xhigh" for row in luna["supported_reasoning_levels"])


@pytest.mark.parametrize("change", ("unknown_selector", "unknown_tool_mode"))
def test_n014_unknown_vendor_tool_policy_cannot_be_silently_qualified(change: str) -> None:
    source = json.loads(
        files("provider_runtime.agent_runtime").joinpath("_codex_models_0_160_0.json").read_bytes()
    )
    if change == "unknown_selector":
        source["models"][0]["experimental_supported_tools"].append("new_native_network_authority")
    else:
        source["models"][0]["tool_mode"] = "future_inherited_tools"
    with pytest.raises(UnsupportedCapability, match="unqualified"):
        _restrict_catalog(source)
