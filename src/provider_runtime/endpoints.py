"""Canonical HTTPS origins shared by configuration and provider dispatch.

This module validates configuration without importing transports or SDKs.
Origins replace only the host; each provider adapter owns its API path.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from urllib.parse import urlsplit

from provider_runtime.types import ProviderName

_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "deepseek", "xai"})


def normalize_endpoint_overrides(
    values: Mapping[ProviderName, str] | None,
) -> Mapping[ProviderName, str]:
    """Freeze canonical HTTPS origins; None keeps every provider's default host."""
    values = values or {}
    unknown = set(values) - _PROVIDERS
    if unknown:
        raise ValueError(f"endpoint overrides name unknown providers: {sorted(unknown)!r}")
    checked: dict[ProviderName, str] = {}
    for provider, value in values.items():
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as error:
            raise ValueError(f"{provider} endpoint override has an invalid port") from error
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"{provider} endpoint override must be one HTTPS origin")
        default_port = port in {None, 443}
        authority = parsed.hostname if default_port else f"{parsed.hostname}:{port}"
        if parsed.netloc != authority:
            raise ValueError(f"{provider} endpoint override is not canonical")
        checked[provider] = f"https://{authority}"
    return MappingProxyType(checked)


__all__ = ["normalize_endpoint_overrides"]
