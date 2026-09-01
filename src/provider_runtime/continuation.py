"""Canonical public codec for bounded provider continuation artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping

from provider_runtime.errors import InvalidRequest
from provider_runtime.types import (
    ContinuationArtifact,
    ProviderTarget,
    canonical_json_bytes,
    freeze_json_object,
)

_SCHEMA_VERSION = "provider-continuation.v1"
_FIELDS = frozenset({"schema_version", "target", "codec_id", "opaque_payload"})
_TARGET_FIELDS = frozenset({"provider", "model"})
_MAX_ENCODED_BYTES = 16 * 1024 * 1024 + 1024


def encode_continuation(artifact: ContinuationArtifact) -> bytes:
    """Encode one artifact into its canonical, target-bound JSON envelope."""
    if not isinstance(artifact, ContinuationArtifact):
        raise InvalidRequest(message="encode_continuation requires ContinuationArtifact")
    envelope = freeze_json_object(
        {
            "schema_version": _SCHEMA_VERSION,
            "target": {
                "provider": artifact.target.provider,
                "model": artifact.target.model,
            },
            "codec_id": artifact.codec_id,
            "opaque_payload": artifact.opaque_payload,
        },
        context="continuation envelope",
    )
    encoded = canonical_json_bytes(envelope)
    if len(encoded) > _MAX_ENCODED_BYTES:
        raise InvalidRequest(message="encoded continuation exceeds its 16 MiB payload bound")
    return encoded


def decode_continuation(
    encoded: bytes,
    target: ProviderTarget,
    codec_id: str,
) -> ContinuationArtifact:
    """Decode canonical bytes and require the caller's exact target and codec."""
    if not isinstance(encoded, bytes):
        raise InvalidRequest(message="continuation encoding must be bytes")
    if len(encoded) > _MAX_ENCODED_BYTES:
        raise InvalidRequest(message="encoded continuation exceeds its 16 MiB payload bound")
    if not isinstance(target, ProviderTarget):
        raise InvalidRequest(message="continuation target must be ProviderTarget")
    if type(codec_id) is not str or not codec_id:
        raise InvalidRequest(message="continuation codec_id must be a non-empty string")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise InvalidRequest(message="continuation encoding is not valid JSON") from None
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise InvalidRequest(message="continuation envelope has an invalid field set")
    if value.get("schema_version") != _SCHEMA_VERSION:
        raise InvalidRequest(message="continuation schema version is unsupported")
    encoded_target = value.get("target")
    if not isinstance(encoded_target, Mapping) or set(encoded_target) != _TARGET_FIELDS:
        raise InvalidRequest(message="continuation target is malformed")
    if (
        encoded_target.get("provider") != target.provider
        or encoded_target.get("model") != target.model
    ):
        raise InvalidRequest(message="continuation target does not match the requested target")
    if value.get("codec_id") != codec_id:
        raise InvalidRequest(message="continuation codec does not match the requested codec")
    payload = value.get("opaque_payload")
    if not isinstance(payload, Mapping):
        raise InvalidRequest(message="continuation opaque_payload must be a JSON object")
    try:
        artifact = ContinuationArtifact(
            target=target,
            codec_id=codec_id,
            opaque_payload=payload,
        )
    except (TypeError, ValueError):
        raise InvalidRequest(
            message="continuation payload is outside the bounded JSON domain"
        ) from None
    if encode_continuation(artifact) != encoded:
        raise InvalidRequest(message="continuation encoding is not canonical")
    return artifact


__all__ = ["decode_continuation", "encode_continuation"]
