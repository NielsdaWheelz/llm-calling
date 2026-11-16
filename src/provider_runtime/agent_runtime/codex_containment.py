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

CODEX_CONTAINMENT_VERSION = "0.165.0"
CODEX_CONTAINMENT_SOURCE_SHA256 = "1384625c0b6b3e67d3c60da29becea2b017f2c283fc1e2d05b8edafb4698eb10"
CODEX_CONTAINMENT_CATALOG_SHA256 = (
    "3bb7176bbd2aa37652d94eeca7a2917867350b5ed0a4ecaa684c809d2d55fb72"
)
CODEX_CONTAINMENT_CATALOG_REVISION = (
    f"codex-contained-catalog.v2:{CODEX_CONTAINMENT_CATALOG_SHA256}"
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
                "send_user_message_async",
                "request_user_input_async",
                "send_message_to_user_async",
            )
            for value in selectors
        ):
            raise UnsupportedCapability("Codex source model has unqualified native tool selectors")
        if model.get("apply_patch_tool_type") not in (None, "freeform"):
            raise UnsupportedCapability("Codex source model has an unqualified native patch tool")
        model["apply_patch_tool_type"] = None
        model["tool_mode"] = "direct"
        model["experimental_supported_tools"] = []
    return canonical_json_bytes(freeze_json_value(catalog))


def materialize_codex_containment_catalog(directory: Path) -> Path:
    """Write the host startup catalog; never touch auth or an existing process."""
    if not directory.is_absolute():
        raise ValueError("Codex host catalog directory must be absolute")
    source = files(__package__).joinpath("_codex_models_0_165_0.json").read_bytes()
    payload = _restrict_catalog(json.loads(source))
    path = directory / CODEX_CONTAINMENT_CATALOG_FILENAME
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _validate_codex_containment_config(response: object, metadata: dict[str, object]) -> Path:
    user_agent = metadata.get("userAgent")
    if (
        not isinstance(user_agent, str)
        or user_agent.partition(" ")[0].rpartition("/")[2] != CODEX_CONTAINMENT_VERSION
    ):
        raise UnsupportedCapability(
            "Codex host startup model catalog requires qualified native version "
            f"{CODEX_CONTAINMENT_VERSION}"
        )
    if isinstance(response, dict):
        config = response.get("config")
        origins = response.get("origins")
        if isinstance(config, dict) and isinstance(origins, dict):
            path = config.get("model_catalog_json")
            origin = origins.get("model_catalog_json")
            if isinstance(path, str) and isinstance(origin, dict):
                name = origin.get("name")
                catalog = Path(path)
                if (
                    catalog.is_absolute()
                    and catalog.name == CODEX_CONTAINMENT_CATALOG_FILENAME
                    and isinstance(name, dict)
                    and name.get("type") == "sessionFlags"
                ):
                    return catalog
    raise UnsupportedCapability(
        "Codex contained sessions require the pinned host startup model catalog"
    )


async def qualify_codex_containment_host(endpoint: CodexEndpoint) -> Path:
    """Verify the public startup declaration; the trusted host owns file bytes."""
    async with asyncio.timeout(_OPERATION_TIMEOUT_SECONDS):
        async with CodexAppServerClient(
            CodexAppServerConfig(
                endpoint=endpoint, request_policy="observe_only", experimental_api=True
            )
        ) as client:
            return _validate_codex_containment_config(
                await client.request("config/read", {"includeLayers": False}), client.metadata
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
