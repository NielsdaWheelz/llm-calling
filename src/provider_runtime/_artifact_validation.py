"""Validation helpers for opaque provider artifact replay."""

from __future__ import annotations

from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.types import ProviderArtifact, ProviderArtifactPurpose, ProviderName


def validated_provider_artifacts(
    artifacts: tuple[ProviderArtifact, ...],
    *,
    provider: ProviderName,
    model: str,
    purpose: ProviderArtifactPurpose,
) -> tuple[ProviderArtifact, ...]:
    """Return artifacts only when replay metadata exactly matches this adapter call."""
    for artifact in artifacts:
        if (
            artifact.provider == provider
            and artifact.model == model
            and artifact.purpose == purpose
        ):
            continue
        raise ModelCallError(
            ModelCallErrorCode.BAD_REQUEST,
            (
                f"{provider} artifact replay requires {provider}/{model}/{purpose}; "
                f"got {artifact.provider}/{artifact.model}/{artifact.purpose}"
            ),
            provider=provider,
            retryable=False,
        )
    return artifacts
