"""OpenAI embeddings port: request golden, strict validation, runtime retries."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

import provider_runtime.runtime as runtime_module
from provider_runtime.embeddings import build_embedding_request, parse_embedding_response
from provider_runtime.errors import CredentialRejected, ProtocolDefect
from provider_runtime.runtime import NonGenerationCallFailed, ProviderRuntime
from provider_runtime.types import (
    Absent,
    EmbeddingCall,
    Presence,
    Present,
    ProviderCredential,
    ProviderHttpUnavailable,
    RetryPolicy,
    TransientExhausted,
)

ABSENT: Absent = Absent()
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
CREDENTIAL = ProviderCredential(provider="openai", key="sk-test-not-a-real-key-1234567890")

FAST_RETRY = RetryPolicy(
    max_attempts=3, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0, deadline_s=Absent()
)


def call(
    *, inputs: tuple[str, ...] = ("alpha", "beta"), dimensions: Presence[int] = ABSENT
) -> EmbeddingCall:
    return EmbeddingCall(model="text-embedding-3-small", inputs=inputs, dimensions=dimensions)


def success_body(*, out_of_order: bool = False) -> dict[str, object]:
    rows = [
        {"index": 0, "embedding": [0.1, 0.2]},
        {"index": 1, "embedding": [0.3, 0.4]},
    ]
    if out_of_order:
        rows.reverse()
    return {
        "object": "list",
        "data": rows,
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }


def parse(body: object, *, expected_count: int = 2):
    return parse_embedding_response(
        200, {}, json.dumps(body).encode(), expected_count=expected_count
    )


# ---------------------------------------------------------------------------
# build_embedding_request


def test_build_request_golden() -> None:
    request = build_embedding_request(call())
    assert request.url == EMBEDDINGS_URL
    assert request.method == "POST"
    assert request.target.provider == "openai"
    assert request.safe_headers == {}
    assert json.loads(request.body) == {
        "model": "text-embedding-3-small",
        "input": ["alpha", "beta"],
    }


def test_build_request_includes_present_dimensions() -> None:
    request = build_embedding_request(call(dimensions=Present(256)))
    assert json.loads(request.body) == {
        "model": "text-embedding-3-small",
        "input": ["alpha", "beta"],
        "dimensions": 256,
    }


# ---------------------------------------------------------------------------
# parse_embedding_response — strict validation preserved from the old client


def test_parse_success_reorders_by_index_and_folds_usage() -> None:
    response = parse(success_body(out_of_order=True))
    assert response.embeddings == ((0.1, 0.2), (0.3, 0.4))
    assert isinstance(response.usage, Present)
    assert response.usage.value.input_tokens == 7
    assert response.usage.value.total_tokens == 7
    assert response.usage.value.output_tokens == 0


def test_parse_without_usage_is_absent() -> None:
    body = success_body()
    del body["usage"]
    assert parse(body).usage == Absent()


def test_parse_rejects_non_json_and_non_object() -> None:
    with pytest.raises(ProtocolDefect):
        parse_embedding_response(200, {}, b"not json", expected_count=1)
    with pytest.raises(ProtocolDefect):
        parse([1, 2, 3])


def test_parse_rejects_missing_data() -> None:
    with pytest.raises(ProtocolDefect, match="missing data"):
        parse({"object": "list"})


def test_parse_rejects_invalid_row() -> None:
    with pytest.raises(ProtocolDefect, match="invalid row"):
        parse({"data": ["nope", "nope"]})


@pytest.mark.parametrize(
    "index",
    [None, "0", True, -1, 2],
    ids=["missing", "string", "bool", "negative", "out-of-range"],
)
def test_parse_rejects_invalid_indexes(index: object) -> None:
    with pytest.raises(ProtocolDefect, match="invalid index"):
        parse({"data": [{"index": index, "embedding": [0.1]}]})


def test_parse_rejects_duplicate_index() -> None:
    with pytest.raises(ProtocolDefect, match="invalid index"):
        parse(
            {
                "data": [
                    {"index": 0, "embedding": [0.1]},
                    {"index": 0, "embedding": [0.2]},
                ]
            }
        )


def test_parse_rejects_invalid_vector() -> None:
    with pytest.raises(ProtocolDefect, match="invalid vector"):
        parse({"data": [{"index": 0, "embedding": "zap"}, {"index": 1, "embedding": [0.1]}]})


def test_parse_rejects_non_numeric_vector_value() -> None:
    with pytest.raises(ProtocolDefect, match="non-numeric vector value"):
        parse({"data": [{"index": 0, "embedding": [{}]}, {"index": 1, "embedding": [0.1]}]})


def test_parse_rejects_non_finite_vector_value() -> None:
    body = json.dumps(
        {"data": [{"index": 0, "embedding": [float("nan")]}]},
    ).encode()
    with pytest.raises(ProtocolDefect, match="non-finite"):
        parse_embedding_response(200, {}, body, expected_count=1)


def test_parse_rejects_incomplete_index_coverage() -> None:
    with pytest.raises(ProtocolDefect, match="incomplete indexes"):
        parse({"data": [{"index": 0, "embedding": [0.1]}]})


# ---------------------------------------------------------------------------
# End-to-end through ProviderRuntime.embed (openai classify_error + retries)


@pytest.fixture
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module, "EXTERNAL_LLM_RETRY", FAST_RETRY)


@respx.mock
async def test_embed_end_to_end_success(fast_retry: None) -> None:
    route = respx.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=success_body()))
    async with httpx.AsyncClient() as http:
        response = await ProviderRuntime(http).embed(call(), credential=CREDENTIAL)
    assert response.embeddings == ((0.1, 0.2), (0.3, 0.4))
    assert route.call_count == 1
    sent = json.loads(route.calls[0].request.content)
    assert sent == {"model": "text-embedding-3-small", "input": ["alpha", "beta"]}
    assert route.calls[0].request.headers["authorization"] == f"Bearer {CREDENTIAL.key}"


@respx.mock
async def test_embed_retries_transient_then_succeeds(fast_retry: None) -> None:
    route = respx.post(EMBEDDINGS_URL).mock(
        side_effect=[
            httpx.Response(500, json={"error": {"message": "boom", "type": "server_error"}}),
            httpx.Response(200, json=success_body()),
        ]
    )
    async with httpx.AsyncClient() as http:
        response = await ProviderRuntime(http).embed(call(), credential=CREDENTIAL)
    assert route.call_count == 2
    assert len(response.embeddings) == 2


@respx.mock
async def test_embed_exhaustion_raises_expected_port_failure(fast_retry: None) -> None:
    route = respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "boom", "type": "server_error"}}
        )
    )
    async with httpx.AsyncClient() as http:
        with pytest.raises(NonGenerationCallFailed) as exc_info:
            await ProviderRuntime(http).embed(call(), credential=CREDENTIAL)
    assert route.call_count == FAST_RETRY.max_attempts
    assert exc_info.value.failure == TransientExhausted(
        attempts=FAST_RETRY.max_attempts, cause=ProviderHttpUnavailable()
    )


@respx.mock
async def test_embed_credential_rejection_is_a_defect(fast_retry: None) -> None:
    respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            401, json={"error": {"message": "bad key", "type": "invalid_request_error"}}
        )
    )
    async with httpx.AsyncClient() as http:
        with pytest.raises(CredentialRejected):
            await ProviderRuntime(http).embed(call(), credential=CREDENTIAL)
