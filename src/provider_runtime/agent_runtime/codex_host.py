"""Incarnation-bound Codex endpoints and exact host retirement observations.

The external supervisor owns process lifetime and authenticates these facts.
Resolving never reconnects a native session; each result carries the endpoint
for only its own incarnation. HTTP loss cannot establish retirement. Callers own
recovery deadlines and retries; this module makes one request and retains no state.
Protocol types import without loading the HTTP transport, which belongs to requests.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_serializer,
)

from .codex_endpoint import CodexEndpoint, CodexUnixEndpoint, CodexWebSocketEndpoint
from .errors import CredentialRejected, ProtocolDefect

CODEX_HOST_PROTOCOL_REVISION = 1
type _ProtocolRevision = Annotated[int, Field(ge=1, le=1)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CodexHostTarget(_Model):
    host_id: UUID
    incarnation: Annotated[int, Field(gt=0)]


class CodexHostResolveRequest(_Model):
    protocol_revision: _ProtocolRevision
    target: CodexHostTarget | None


class CodexHostRetireRequest(_Model):
    protocol_revision: _ProtocolRevision
    target: CodexHostTarget


class CodexHostUnixTransport(_Model):
    kind: Literal["unix"] = "unix"
    socket_name: str


class CodexHostWebSocketTransport(_Model):
    kind: Literal["websocket"] = "websocket"
    token: SecretStr

    @field_serializer("token", when_used="json")
    def _wire_token(self, value: SecretStr) -> str:
        # Only the authenticated control response carries the actual credential.
        return value.get_secret_value()


class CodexHostActive(_Model):
    protocol_revision: _ProtocolRevision = CODEX_HOST_PROTOCOL_REVISION
    status: Literal["active"] = "active"
    target: CodexHostTarget
    transport: Annotated[
        CodexHostUnixTransport | CodexHostWebSocketTransport, Field(discriminator="kind")
    ]
    retirement_allowed: bool


class CodexHostRetired(_Model):
    protocol_revision: _ProtocolRevision = CODEX_HOST_PROTOCOL_REVISION
    status: Literal["retired"] = "retired"
    target: CodexHostTarget


type CodexHostUnavailableReason = Literal[
    "starting",
    "retiring",
    "unreachable",
    "unknown_host",
    "unknown_incarnation",
    "retirement_disabled",
    "invalid_state",
    "protocol_mismatch",
]


class CodexHostUnavailable(_Model):
    protocol_revision: _ProtocolRevision = CODEX_HOST_PROTOCOL_REVISION
    status: Literal["unavailable"] = "unavailable"
    reason: CodexHostUnavailableReason


type CodexHostResponse = Annotated[
    CodexHostActive | CodexHostRetired | CodexHostUnavailable, Field(discriminator="status")
]
_RESPONSE = TypeAdapter(CodexHostResponse)


@dataclass(frozen=True, slots=True)
class CodexHostUnix:
    control_socket: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_socket", CodexUnixEndpoint(self.control_socket).path)


@dataclass(frozen=True, slots=True)
class CodexHostNetwork:
    control_url: str
    native_url: str
    control_token: SecretStr = field(repr=False)

    def __post_init__(self) -> None:
        native = CodexWebSocketEndpoint(self.native_url, self.control_token)
        try:
            parsed = urlsplit(self.control_url)
            if not (
                (parsed.scheme, urlsplit(native.url).scheme) in {("http", "ws"), ("https", "wss")}
                and parsed.hostname
                and parsed.path in ("", "/")
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and parsed.port != 0
                and not any(
                    character.isspace() or ord(character) < 32 for character in self.control_url
                )
            ):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(
                "Codex host requires paired HTTP/WS origins without credentials"
            ) from None
        object.__setattr__(
            self, "control_url", urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        )
        object.__setattr__(self, "native_url", native.url)


type CodexHostEndpoint = CodexHostUnix | CodexHostNetwork


@dataclass(frozen=True, slots=True)
class CodexHostResolved:
    target: CodexHostTarget
    endpoint: CodexEndpoint
    retirement_allowed: bool


async def _request(
    endpoint: CodexHostEndpoint,
    path: Literal["resolve", "retire"],
    request: CodexHostResolveRequest | CodexHostRetireRequest,
) -> CodexHostResponse:
    import httpx

    if isinstance(endpoint, CodexHostUnix):
        transport = httpx.AsyncHTTPTransport(uds=str(endpoint.control_socket))
        url, headers = f"http://codex-host/{path}", {}
    else:
        transport = None
        url = f"{endpoint.control_url}/{path}"
        headers = {"Authorization": f"Bearer {endpoint.control_token.get_secret_value()}"}
    try:
        async with (
            asyncio.timeout(5.0),
            httpx.AsyncClient(
                transport=transport, trust_env=False, follow_redirects=False, timeout=None
            ) as client,
        ):
            response = await client.post(url, headers=headers, json=request.model_dump(mode="json"))
    except (httpx.TransportError, TimeoutError):
        return CodexHostUnavailable(reason="unreachable")
    if response.status_code in {401, 403}:
        raise CredentialRejected("Codex host control rejected its credential")
    if response.status_code >= 500:
        return CodexHostUnavailable(reason="unreachable")
    if response.status_code != 200:
        raise ProtocolDefect(f"Codex host control returned HTTP {response.status_code}")
    try:
        document = response.json()
    except ValueError:
        raise ProtocolDefect("Codex host control returned invalid JSON") from None
    if not isinstance(document, dict) or type(document.get("protocol_revision")) is not int:
        raise ProtocolDefect("Codex host control omitted its protocol revision")
    if document["protocol_revision"] != CODEX_HOST_PROTOCOL_REVISION:
        return CodexHostUnavailable(reason="protocol_mismatch")
    try:
        result = _RESPONSE.validate_json(response.content)
    except ValidationError:
        raise ProtocolDefect("Codex host control returned an invalid response") from None
    if isinstance(result, (CodexHostActive, CodexHostRetired)):
        if request.target is not None and result.target != request.target:
            raise ProtocolDefect("Codex host control changed the requested target")
        if request.target is None and isinstance(result, CodexHostRetired):
            raise ProtocolDefect("Codex host control retired an unspecified target")
    return result


async def resolve_codex_host(
    endpoint: CodexHostEndpoint, target: CodexHostTarget | None = None
) -> CodexHostResolved | CodexHostRetired | CodexHostUnavailable:
    result = await _request(
        endpoint,
        "resolve",
        CodexHostResolveRequest(protocol_revision=CODEX_HOST_PROTOCOL_REVISION, target=target),
    )
    if not isinstance(result, CodexHostActive):
        return result
    if isinstance(endpoint, CodexHostUnix):
        name = f"native-{result.target.host_id}-{result.target.incarnation}.sock"
        if (
            not isinstance(result.transport, CodexHostUnixTransport)
            or result.transport.socket_name != name
        ):
            raise ProtocolDefect("Codex host control returned a different incarnation socket")
        native: CodexEndpoint = CodexUnixEndpoint(endpoint.control_socket.parent / name)
    else:
        if not isinstance(result.transport, CodexHostWebSocketTransport):
            raise ProtocolDefect("Codex host control changed the native transport")
        try:
            native = CodexWebSocketEndpoint(endpoint.native_url, result.transport.token)
        except ValueError:
            raise ProtocolDefect(
                "Codex host control returned an invalid native credential"
            ) from None
    return CodexHostResolved(result.target, native, result.retirement_allowed)


async def retire_codex_host(
    endpoint: CodexHostEndpoint, target: CodexHostTarget
) -> CodexHostRetired | CodexHostUnavailable:
    result = await _request(
        endpoint,
        "retire",
        CodexHostRetireRequest(protocol_revision=CODEX_HOST_PROTOCOL_REVISION, target=target),
    )
    if isinstance(result, CodexHostActive):
        raise ProtocolDefect("Codex host retirement returned an active incarnation")
    return result
