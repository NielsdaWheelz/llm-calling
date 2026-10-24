"""Explicit Codex listeners, separate from client state and working directories.

WebSocket credentials authorize a connection, not a session identity. Endpoints
never place secrets in URLs or representations; fingerprints contain only the
transport and address. Neither transport owns or restarts the external daemon.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import SecretStr


@dataclass(frozen=True, slots=True)
class CodexUnixEndpoint:
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute() or "\0" in str(self.path):
            raise ValueError("Codex Unix endpoint requires an absolute path")
        object.__setattr__(self, "path", self.path.resolve())


@dataclass(frozen=True, slots=True)
class CodexWebSocketEndpoint:
    url: str
    bearer_token: SecretStr = field(repr=False)

    def __post_init__(self) -> None:
        try:
            parsed = urlsplit(self.url)
            valid = (
                parsed.scheme in ("ws", "wss")
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and parsed.port != 0
                and not any(character.isspace() or ord(character) < 32 for character in self.url)
            )
            if not valid:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(
                "Codex WebSocket endpoint requires a ws/wss URL without credentials"
            ) from None
        host = parsed.hostname
        assert host is not None
        host = f"[{host.lower()}]" if ":" in host else host.encode("idna").decode("ascii").lower()
        default_port = 443 if parsed.scheme == "wss" else 80
        address = host if parsed.port in (None, default_port) else f"{host}:{parsed.port}"
        object.__setattr__(
            self, "url", urlunsplit((parsed.scheme, address, parsed.path or "/", "", ""))
        )
        if not isinstance(self.bearer_token, SecretStr):
            raise ValueError("Codex WebSocket bearer token must be a secret")
        token = self.bearer_token.get_secret_value()
        if not token or any(ord(character) < 33 or ord(character) > 126 for character in token):
            raise ValueError("Codex WebSocket bearer token must be nonempty printable ASCII")


type CodexEndpoint = CodexUnixEndpoint | CodexWebSocketEndpoint


def fingerprint_codex_endpoint(endpoint: CodexEndpoint) -> str:
    """Session scope follows the configured listener, independent of credentials."""
    if isinstance(endpoint, CodexUnixEndpoint):
        address = f"unix:{endpoint.path}"
    else:
        address = f"websocket:{endpoint.url}"
    return hashlib.sha256(address.encode("utf-8")).hexdigest()
