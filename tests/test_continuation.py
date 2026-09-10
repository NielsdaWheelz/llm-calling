"""Canonical continuation ownership, immutability, and binding conformance."""

from __future__ import annotations

import json
from typing import cast

import pytest

from provider_runtime.continuation import decode_continuation, encode_continuation
from provider_runtime.errors import InvalidRequest
from provider_runtime.types import (
    ContinuationArtifact,
    FrozenJsonDict,
    JsonValueError,
    ProviderTarget,
    thaw_json_value,
)

TARGET = ProviderTarget(provider="openai", model="gpt-5.6-sol")


def test_continuation_round_trip_is_canonical_bound_and_recursively_immutable() -> None:
    source = {"items": [{"kind": "reasoning", "opaque": "sealed"}], "flag": True}
    artifact = ContinuationArtifact(
        target=TARGET,
        codec_id="openai.v1",
        opaque_payload=source,
    )
    source["items"].append({"kind": "late"})  # type: ignore[union-attr]

    encoded = encode_continuation(artifact)
    decoded = decode_continuation(encoded, TARGET, "openai.v1")

    assert encode_continuation(decoded) == encoded
    assert thaw_json_value(decoded.opaque_payload) == {
        "flag": True,
        "items": [{"kind": "reasoning", "opaque": "sealed"}],
    }
    items = cast(tuple[FrozenJsonDict, ...], decoded.opaque_payload["items"])
    with pytest.raises(TypeError):
        items[0]["opaque"] = "changed"  # pyright: ignore[reportIndexIssue]


def test_continuation_decode_rejects_target_codec_and_noncanonical_bytes() -> None:
    artifact = ContinuationArtifact(TARGET, "openai.v1", {"opaque": "value"})
    encoded = encode_continuation(artifact)

    with pytest.raises(InvalidRequest, match="target does not match"):
        decode_continuation(
            encoded,
            ProviderTarget(provider="openai", model="gpt-5.6-terra"),
            "openai.v1",
        )
    with pytest.raises(InvalidRequest, match="codec does not match"):
        decode_continuation(encoded, TARGET, "openai.v2")

    noncanonical = json.dumps(json.loads(encoded), indent=2).encode()
    with pytest.raises(InvalidRequest, match="not canonical"):
        decode_continuation(noncanonical, TARGET, "openai.v1")


@pytest.mark.parametrize(
    "value",
    (
        b"not-json",
        b"[]",
        b'{"schema_version":"provider-continuation.v1"}',
        b'{"codec_id":"openai.v1","opaque_payload":{"x":NaN},'
        b'"schema_version":"provider-continuation.v1",'
        b'"target":{"model":"gpt-5.6-sol","provider":"openai"}}',
    ),
)
def test_continuation_decode_rejects_malformed_or_non_json_domain_values(value: bytes) -> None:
    with pytest.raises(InvalidRequest):
        decode_continuation(value, TARGET, "openai.v1")


def test_continuation_payload_enforces_the_16_mib_canonical_bound() -> None:
    with pytest.raises(ValueError, match="16 MiB"):
        ContinuationArtifact(
            TARGET,
            "openai.v1",
            {"oversized": "x" * (16 * 1024 * 1024)},
        )
    with pytest.raises(JsonValueError, match="JSON-safe"):
        ContinuationArtifact(TARGET, "openai.v1", {"bad": object()})


def test_continuation_public_codec_has_no_implicit_target_or_codec_fallback() -> None:
    artifact = ContinuationArtifact(TARGET, "openai.v1", {})
    with pytest.raises(TypeError):
        decode_continuation(encode_continuation(artifact))  # type: ignore[call-arg]
