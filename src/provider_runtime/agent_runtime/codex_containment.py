"""Pinned stock host policy for contained Codex sessions."""

from __future__ import annotations

import asyncio
import json
from importlib.resources import files
from pathlib import Path

from provider_runtime.types import canonical_json_bytes, freeze_json_value

from ._limits import _OPERATION_TIMEOUT_SECONDS
from .codex_app_server import CodexAppServerClient, CodexAppServerConfig
from .codex_endpoint import CodexEndpoint
from .errors import UnsupportedCapability

CODEX_CONTAINMENT_VERSION = "0.171.0"
CODEX_CONTAINMENT_SOURCE_SHA256 = "d50b6749506d97e5f334dbabcb666eb1689f423cf3e88cfabdcf3c536b873068"
CODEX_CONTAINMENT_CATALOG_SHA256 = (
    "552646113928209983c67c598ce80c534652b4916d1b0595d2d175ee7ee6627b"
)
CODEX_CONTAINMENT_CATALOG_REVISION = (
    f"codex-contained-catalog.v3:{CODEX_CONTAINMENT_CATALOG_SHA256}"
)
CODEX_CONTAINMENT_CATALOG_FILENAME = (
    f"codex-contained-models-{CODEX_CONTAINMENT_VERSION}-{CODEX_CONTAINMENT_CATALOG_SHA256}.json"
)


def _restrict_catalog(catalog: object) -> bytes:
    if not isinstance(catalog, dict) or set(catalog) != {"models"}:
        raise UnsupportedCapability("Codex source catalog has an unqualified shape")
    models = catalog["models"]
    if not isinstance(models, list) or not models:
        raise UnsupportedCapability("Codex source catalog must contain models")
    for model in models:
        if not isinstance(model, dict):
            raise UnsupportedCapability("Codex source model metadata is malformed")
        mode = model.get("tool_mode")
        if mode is not None and mode not in ("direct", "code_mode", "code_mode_only"):
            raise UnsupportedCapability("Codex source model has an unqualified tool mode")
        selectors = model.get("experimental_supported_tools")
        if not isinstance(selectors, list) or any(
            type(value) is not str
            or value
            not in (
                "clock",
                "literature_search_query",
                "multi_agent_v2_in_code_mode",
                "send_user_message_async",
                "request_user_input_async",
                "send_message_to_user_async",
            )
            for value in selectors
        ):
            raise UnsupportedCapability("Codex source model has unqualified native tool selectors")
        if model.get("apply_patch_tool_type") not in (None, "freeform"):
            raise UnsupportedCapability("Codex source model has an unqualified native patch tool")
        # Ultra combines a lower inference effort with automatic delegation.
        # Builtin-disabled sessions cannot execute that preset.
        levels = model.get("supported_reasoning_levels")
        if not isinstance(levels, list) or any(
            not isinstance(row, dict) or type(row.get("effort")) is not str for row in levels
        ):
            raise UnsupportedCapability("Codex source model has unqualified reasoning presets")
        levels = [row for row in levels if row["effort"] != "ultra"]
        if model.get("default_reasoning_level") not in {row["effort"] for row in levels}:
            raise UnsupportedCapability("Codex contained model lacks its default reasoning effort")
        model["supported_reasoning_levels"] = levels
        model["apply_patch_tool_type"] = None
        model["tool_mode"] = "direct"
        model["experimental_supported_tools"] = []
    return canonical_json_bytes(freeze_json_value(catalog))


def materialize_codex_containment_catalog(directory: Path) -> Path:
    """Write the host startup catalog; never touch auth or an existing process."""
    if not directory.is_absolute():
        raise ValueError("Codex host catalog directory must be absolute")
    source = files(__package__).joinpath("_codex_models_0_171_0.json").read_bytes()
    payload = _restrict_catalog(json.loads(source))
    path = directory / CODEX_CONTAINMENT_CATALOG_FILENAME
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _startup_flag(response: object, key: str) -> object:
    if not isinstance(response, dict):
        return None
    config = response.get("config")
    origins = response.get("origins")
    origin = origins.get(key) if isinstance(origins, dict) else None
    name = origin.get("name") if isinstance(origin, dict) else None
    if isinstance(config, dict) and isinstance(name, dict) and name.get("type") == "sessionFlags":
        return config.get(key)
    return None


def _validate_codex_containment_config(
    response: object, metadata: dict[str, object], required_service_tier: str | None = None
) -> Path:
    user_agent = metadata.get("userAgent")
    if (
        not isinstance(user_agent, str)
        or user_agent.partition(" ")[0].rpartition("/")[2] != CODEX_CONTAINMENT_VERSION
    ):
        raise UnsupportedCapability(
            "Codex host startup model catalog requires qualified native version "
            f"{CODEX_CONTAINMENT_VERSION}"
        )
    path = _startup_flag(response, "model_catalog_json")
    catalog = Path(path) if isinstance(path, str) else None
    if (
        catalog is None
        or not catalog.is_absolute()
        or catalog.name != CODEX_CONTAINMENT_CATALOG_FILENAME
    ):
        raise UnsupportedCapability(
            "Codex contained sessions require the pinned host startup model catalog"
        )
    if (
        required_service_tier is not None
        and _startup_flag(response, "service_tier") != required_service_tier
    ):
        raise UnsupportedCapability(
            "Codex host must set the required service tier with a startup flag"
        )
    return catalog


async def qualify_codex_containment_host(
    endpoint: CodexEndpoint, *, required_service_tier: str | None = None
) -> Path:
    """Verify the public startup declaration; the trusted host owns file bytes."""
    async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
        async with CodexAppServerClient(
            CodexAppServerConfig(
                endpoint=endpoint, request_policy="observe_only", experimental_api=True
            )
        ) as client:
            return _validate_codex_containment_config(
                await client.request("config/read", {"includeLayers": False}),
                client.metadata,
                required_service_tier,
            )


__all__ = [
    "CODEX_CONTAINMENT_CATALOG_FILENAME",
    "CODEX_CONTAINMENT_CATALOG_REVISION",
    "CODEX_CONTAINMENT_CATALOG_SHA256",
    "CODEX_CONTAINMENT_SOURCE_SHA256",
    "CODEX_CONTAINMENT_VERSION",
    "materialize_codex_containment_catalog",
    "qualify_codex_containment_host",
]
