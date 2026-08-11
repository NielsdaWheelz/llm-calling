"""OpenAI embeddings port conformance + fault injection (HTTP boundary via respx).

Every test drives the Nexus-facing seam — ``ProviderRuntime.embed`` — with
respx intercepting the openai SDK's own httpx transport; no internal mocking.
Request tests assert the EXACT body the SDK puts on the wire (the SDK default
requests ``encoding_format=base64`` and decodes the packed float32 payload
itself, so wire vectors here are base64 strings and expected values are
float32-exact).
"""

from __future__ import annotations

import json
from array import array
from base64 import b64encode

import httpx
import pytest
import respx

from provider_runtime.errors import (
    CredentialRejected,
    InvalidRequest,
    ProtocolDefect,
    RuntimeDefect,
)
from provider_runtime.runtime import Credentials, NonGenerationCallFailed, ProviderRuntime
from provider_runtime.types import (
    Absent,
    EmbeddingCall,
    Presence,
    Present,
    ProviderContextTooLarge,
    ProviderCredential,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderTimeout,
    RetryPolicy,
    TokenUsage,
    TransientCause,
    TransientExhausted,
    TransportUnavailable,
)

ABSENT: Absent = Absent()
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
CREDENTIAL = ProviderCredential(provider="openai", key="sk-test-not-a-real-key-1234567890")

# Every request-affecting environment variable the openai SDK reads on its own.
POISON_ENV = {
    "OPENAI_BASE_URL": "https://poisoned.invalid/v1",
    "OPENAI_ORG_ID": "org-poison",
    "OPENAI_PROJECT_ID": "proj-poison",
    "OPENAI_WEBHOOK_SECRET": "whsec-poison",
    "OPENAI_CUSTOM_HEADERS": "X-Poison: pwned",
}

# Tests are the one sanctioned RetryPolicy( construction site outside retry.py.
FAST_RETRY = RetryPolicy(
    max_attempts=3, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0, deadline_s=Absent()
)

# Exactly representable in float32, so base64 round-trips compare equal.
VECTOR_0 = (0.5, -1.25)
VECTOR_1 = (2.0, 0.75)


def runtime() -> ProviderRuntime:
    return ProviderRuntime(Credentials(), retry=FAST_RETRY)


def call(
    *, inputs: tuple[str, ...] = ("alpha", "beta"), dimensions: Presence[int] = ABSENT
) -> EmbeddingCall:
    return EmbeddingCall(model="text-embedding-3-small", inputs=inputs, dimensions=dimensions)


def b64_floats(*values: float) -> str:
    """The wire form the SDK's default encoding_format=base64 asks for: packed float32."""
    return b64encode(array("f", values).tobytes()).decode("ascii")


def success_body(*, out_of_order: bool = False) -> dict[str, object]:
    rows: list[object] = [
        {"object": "embedding", "index": 0, "embedding": b64_floats(*VECTOR_0)},
        {"object": "embedding", "index": 1, "embedding": b64_floats(*VECTOR_1)},
    ]
    if out_of_order:
        rows.reverse()
    return {
        "object": "list",
        "data": rows,
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }


def data_body(data: object) -> dict[str, object]:
    body = success_body()
    body["data"] = data
    return body


def error_response(
    status: int, message: str, *, code: str | None = None, headers: dict[str, str] | None = None
) -> httpx.Response:
    error: dict[str, object] = {"message": message, "type": "invalid_request_error"}
    if code is not None:
        error["code"] = code
    return httpx.Response(status, headers=headers or {}, json={"error": error})


# ---------------------------------------------------------------------------
# Request conformance


@respx.mock
async def test_embed_sends_exact_request_body_and_bearer_auth() -> None:
    route = respx.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=success_body()))
    response = await runtime().embed(call(), credential=CREDENTIAL)
    assert route.call_count == 1, f"expected exactly one dispatch; got {route.call_count}"
    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {CREDENTIAL.key}", (
        f"authorization header: {request.headers.get('authorization')!r}"
    )
    body = json.loads(request.content)
    assert body == {
        "model": "text-embedding-3-small",
        "input": ["alpha", "beta"],
        "encoding_format": "base64",  # SDK default: packed float32 on the wire
    }, f"request body: {body!r}"
    assert response.embeddings == (VECTOR_0, VECTOR_1), f"embeddings: {response.embeddings!r}"


@respx.mock
async def test_embed_sends_dimensions_when_present() -> None:
    route = respx.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=success_body()))
    await runtime().embed(call(dimensions=Present(256)), credential=CREDENTIAL)
    body = json.loads(route.calls.last.request.content)
    assert body == {
        "model": "text-embedding-3-small",
        "input": ["alpha", "beta"],
        "dimensions": 256,
        "encoding_format": "base64",
    }, f"request body: {body!r}"


@respx.mock
async def test_embed_suppresses_every_ambient_sdk_env_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The openai SDK reads these at client construction whenever the matching
    argument is omitted, and OPENAI_CUSTOM_HEADERS injects arbitrary headers
    into every request. None of them may touch the wire."""
    for name, value in POISON_ENV.items():
        monkeypatch.setenv(name, value)
    route = respx.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=success_body()))
    await runtime().embed(call(), credential=CREDENTIAL)
    assert route.call_count == 1, "request must hit the canonical OpenAI host"
    headers = route.calls.last.request.headers
    assert "x-poison" not in headers, f"OPENAI_CUSTOM_HEADERS reached the wire: {headers!r}"
    assert "openai-organization" not in headers, f"OPENAI_ORG_ID reached the wire: {headers!r}"
    assert "openai-project" not in headers, f"OPENAI_PROJECT_ID reached the wire: {headers!r}"


@respx.mock
async def test_embed_rejects_a_non_openai_credential() -> None:
    # This port is openai-only: any other credential's key must never reach
    # the openai wire.
    route = respx.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=success_body()))
    other = ProviderCredential(provider="anthropic", key="sk-ant-not-a-real-key")
    with pytest.raises(InvalidRequest, match="openai-only"):
        await runtime().embed(call(), credential=other)
    assert route.call_count == 0, (
        f"the credential must never reach the wire; got {route.call_count} dispatches"
    )


# ---------------------------------------------------------------------------
# Decode — order, vector forms, usage


@respx.mock
async def test_embed_reorders_vectors_by_index_into_input_order() -> None:
    respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=success_body(out_of_order=True))
    )
    response = await runtime().embed(call(), credential=CREDENTIAL)
    assert response.embeddings == (VECTOR_0, VECTOR_1), f"embeddings: {response.embeddings!r}"


@respx.mock
async def test_embed_accepts_plain_float_list_vectors() -> None:
    # A provider answering JSON float lists (despite the base64 request) passes
    # through the SDK's decode untouched and must still validate and order.
    respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200,
            json=data_body(
                [
                    {"object": "embedding", "index": 1, "embedding": list(VECTOR_1)},
                    {"object": "embedding", "index": 0, "embedding": list(VECTOR_0)},
                ]
            ),
        )
    )
    response = await runtime().embed(call(), credential=CREDENTIAL)
    assert response.embeddings == (VECTOR_0, VECTOR_1), f"embeddings: {response.embeddings!r}"


@respx.mock
async def test_embed_maps_usage_tokens() -> None:
    respx.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=success_body()))
    response = await runtime().embed(call(), credential=CREDENTIAL)
    assert response.usage == Present(
        TokenUsage(
            input_tokens=7,
            output_tokens=0,
            total_tokens=7,
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Absent(),
            cache_write_input_tokens=Absent(),
        )
    ), f"usage: {response.usage!r}"


@respx.mock
async def test_embed_without_usage_is_absent() -> None:
    body = success_body()
    del body["usage"]
    respx.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=body))
    response = await runtime().embed(call(), credential=CREDENTIAL)
    assert response.usage == Absent(), f"usage: {response.usage!r}"


@respx.mock
async def test_embed_rejects_negative_usage_counts() -> None:
    body = success_body()
    body["usage"] = {"prompt_tokens": -1, "total_tokens": 7}
    route = respx.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(ProtocolDefect, match="token accounting"):
        await runtime().embed(call(), credential=CREDENTIAL)
    assert route.call_count == 1, f"defects must not retry; got {route.call_count} dispatches"


# ---------------------------------------------------------------------------
# Decode — strict index/vector validation (malformed envelope ⇒ ProtocolDefect)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"object": "embedding", "embedding": b64_floats(0.5)}, id="missing"),
        pytest.param(
            {"object": "embedding", "index": "0", "embedding": b64_floats(0.5)}, id="string"
        ),
        pytest.param(
            {"object": "embedding", "index": True, "embedding": b64_floats(0.5)}, id="bool"
        ),
        pytest.param(
            {"object": "embedding", "index": -1, "embedding": b64_floats(0.5)}, id="negative"
        ),
        pytest.param(
            {"object": "embedding", "index": 2, "embedding": b64_floats(0.5)}, id="out-of-range"
        ),
    ],
)
@respx.mock
async def test_embed_rejects_invalid_indexes(row: dict[str, object]) -> None:
    respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200,
            json=data_body(
                [row, {"object": "embedding", "index": 1, "embedding": b64_floats(0.5)}]
            ),
        )
    )
    with pytest.raises(ProtocolDefect, match="invalid index"):
        await runtime().embed(call(), credential=CREDENTIAL)


@respx.mock
async def test_embed_rejects_duplicate_index() -> None:
    respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200,
            json=data_body(
                [
                    {"object": "embedding", "index": 0, "embedding": b64_floats(0.5)},
                    {"object": "embedding", "index": 0, "embedding": b64_floats(0.25)},
                ]
            ),
        )
    )
    with pytest.raises(ProtocolDefect, match="invalid index"):
        await runtime().embed(call(), credential=CREDENTIAL)


@respx.mock
async def test_embed_rejects_incomplete_index_coverage() -> None:
    route = respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200, json=data_body([{"object": "embedding", "index": 0, "embedding": b64_floats(0.5)}])
        )
    )
    with pytest.raises(ProtocolDefect, match="incomplete indexes"):
        await runtime().embed(call(), credential=CREDENTIAL)
    assert route.call_count == 1, f"defects must not retry; got {route.call_count} dispatches"


@respx.mock
async def test_embed_rejects_non_list_vector() -> None:
    respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200,
            json=data_body(
                [
                    {"object": "embedding", "index": 0, "embedding": {"x": 1}},
                    {"object": "embedding", "index": 1, "embedding": b64_floats(0.5)},
                ]
            ),
        )
    )
    with pytest.raises(ProtocolDefect, match="invalid vector"):
        await runtime().embed(call(), credential=CREDENTIAL)


@respx.mock
async def test_embed_rejects_non_numeric_vector_value() -> None:
    respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200,
            json=data_body(
                [
                    {"object": "embedding", "index": 0, "embedding": [{}]},
                    {"object": "embedding", "index": 1, "embedding": b64_floats(0.5)},
                ]
            ),
        )
    )
    with pytest.raises(ProtocolDefect, match="non-numeric vector value"):
        await runtime().embed(call(), credential=CREDENTIAL)


@respx.mock
async def test_embed_rejects_non_finite_vector_value() -> None:
    respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200,
            json=data_body(
                [
                    {"object": "embedding", "index": 0, "embedding": b64_floats(float("nan"))},
                    {"object": "embedding", "index": 1, "embedding": b64_floats(0.5)},
                ]
            ),
        )
    )
    with pytest.raises(ProtocolDefect, match="non-finite vector value"):
        await runtime().embed(call(), credential=CREDENTIAL)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([], id="empty-data"),
        pytest.param(["nope", "nope"], id="non-object-row"),
        pytest.param(
            [
                {"object": "embedding", "index": 0, "embedding": "zap"},
                {"object": "embedding", "index": 1, "embedding": b64_floats(0.5)},
            ],
            id="malformed-base64",
        ),
        # A truthy non-iterable `data` (not merely falsy/empty) trips the
        # SDK's own `for embedding in obj.data` with a TypeError before this
        # module's decode ever sees it.
        pytest.param(5, id="truthy-non-list-int"),
        pytest.param(True, id="truthy-non-list-bool"),
    ],
)
@respx.mock
async def test_embed_sdk_decode_breakage_is_a_protocol_defect(data: object) -> None:
    # These envelopes trip the SDK's own base64 post-parse or its own
    # `for embedding in obj.data` before our decode ever sees them; the
    # breakage must still surface as a ProtocolDefect.
    route = respx.post(EMBEDDINGS_URL).mock(return_value=httpx.Response(200, json=data_body(data)))
    with pytest.raises(ProtocolDefect, match="failed SDK decode"):
        await runtime().embed(call(), credential=CREDENTIAL)
    assert route.call_count == 1, f"defects must not retry; got {route.call_count} dispatches"


# ---------------------------------------------------------------------------
# Fault injection — retries through the real runtime loop


@pytest.mark.parametrize(
    "first_response",
    [
        pytest.param(
            error_response(429, "slow down", headers={"retry-after": "0"}), id="rate-limit"
        ),
        pytest.param(error_response(503, "overloaded"), id="unavailable"),
    ],
)
@respx.mock
async def test_embed_retries_transient_then_succeeds(first_response: httpx.Response) -> None:
    route = respx.post(EMBEDDINGS_URL).mock(
        side_effect=[first_response, httpx.Response(200, json=success_body())]
    )
    response = await runtime().embed(call(), credential=CREDENTIAL)
    assert route.call_count == 2, f"expected one retry; got {route.call_count} dispatches"
    assert response.embeddings == (VECTOR_0, VECTOR_1), f"embeddings: {response.embeddings!r}"


@pytest.mark.parametrize(
    ("fault", "expected_cause"),
    [
        pytest.param(
            error_response(429, "slow down"),
            ProviderRateLimit(retry_after=Absent()),
            id="rate-limit",
        ),
        pytest.param(
            error_response(500, "boom"),
            ProviderHttpUnavailable(),
            id="unavailable",
        ),
        pytest.param(
            httpx.ReadTimeout("read timed out"),
            ProviderTimeout(),
            id="timeout",
        ),
        pytest.param(
            httpx.ConnectError("no route to host"),
            TransportUnavailable(),
            id="connect-error",
        ),
    ],
)
@respx.mock
async def test_embed_exhaustion_raises_expected_port_failure(
    fault: httpx.Response | Exception, expected_cause: TransientCause
) -> None:
    route = respx.post(EMBEDDINGS_URL)
    if isinstance(fault, httpx.Response):
        route.mock(return_value=fault)
    else:
        route.mock(side_effect=fault)
    with pytest.raises(NonGenerationCallFailed) as exc_info:
        await runtime().embed(call(), credential=CREDENTIAL)
    assert route.call_count == FAST_RETRY.max_attempts, (
        f"expected {FAST_RETRY.max_attempts} attempts; got {route.call_count}"
    )
    assert exc_info.value.failure == TransientExhausted(
        attempts=FAST_RETRY.max_attempts, cause=expected_cause
    ), f"failure: {exc_info.value.failure!r}"


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            error_response(
                400,
                "This model's maximum context length is 8192 tokens.",
                code="context_length_exceeded",
            ),
            id="by-code",
        ),
        pytest.param(
            error_response(400, "Maximum context length exceeded for this request."),
            id="by-message",
        ),
    ],
)
@respx.mock
async def test_embed_context_overflow_raises_expected_port_failure(
    response: httpx.Response,
) -> None:
    route = respx.post(EMBEDDINGS_URL).mock(return_value=response)
    with pytest.raises(NonGenerationCallFailed) as exc_info:
        await runtime().embed(call(), credential=CREDENTIAL)
    assert exc_info.value.failure == ProviderContextTooLarge(), (
        f"failure: {exc_info.value.failure!r}"
    )
    assert route.call_count == 1, f"expected values must not retry; got {route.call_count}"


@pytest.mark.parametrize("status", [401, 403])
@respx.mock
async def test_embed_credential_rejection_is_a_defect(status: int) -> None:
    route = respx.post(EMBEDDINGS_URL).mock(return_value=error_response(status, "bad key"))
    with pytest.raises(CredentialRejected, match=str(status)):
        await runtime().embed(call(), credential=CREDENTIAL)
    assert route.call_count == 1, f"defects must not retry; got {route.call_count} dispatches"


@respx.mock
async def test_embed_unclassified_error_is_a_defect() -> None:
    respx.post(EMBEDDINGS_URL).mock(
        return_value=error_response(400, "unknown parameter: 'frobnicate'")
    )
    with pytest.raises(RuntimeDefect) as exc_info:
        await runtime().embed(call(), credential=CREDENTIAL)
    assert exc_info.value.code == "unclassified_provider_error", f"code: {exc_info.value.code!r}"


@respx.mock
async def test_embed_quota_exhausted_is_a_defect() -> None:
    route = respx.post(EMBEDDINGS_URL).mock(return_value=error_response(402, "insufficient funds"))
    with pytest.raises(RuntimeDefect) as exc_info:
        await runtime().embed(call(), credential=CREDENTIAL)
    assert exc_info.value.code == "quota_exhausted", f"code: {exc_info.value.code!r}"
    assert route.call_count == 1, f"defects must not retry; got {route.call_count} dispatches"
