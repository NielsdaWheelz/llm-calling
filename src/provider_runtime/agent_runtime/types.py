"""Frozen, strict, JSON-safe values for the agent-runtime lane."""

from __future__ import annotations

import math
import os.path
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

from provider_runtime.types import ProviderName, ProviderTarget

from .errors import InvalidAgentRequest, UnsupportedCapability
from .policy import PermissionPolicy, PermissionPolicyPatch

type Backend = Literal["codex", "claude"]
type AgentTransport = Literal["sdk"]
type CredentialKind = Literal["local_account", "api_key_environment", "secret_reference"]
type ApprovalDecision = Literal["allow", "deny", "abort"]

_BACKENDS: tuple[Backend, ...] = ("codex", "claude")
_TRANSPORTS: tuple[AgentTransport, ...] = ("sdk",)
# The closed routing table has exactly one owner in the package; runtime.py and the adapters
# read this name instead of re-listing. One transport per backend is not an invariant this
# table asserts — it is what the two shipped lanes happen to be. Consumers key off the pair,
# never off the backend alone.
AGENT_ROUTES: frozenset[tuple[Backend, AgentTransport]] = frozenset(
    {
        ("codex", "sdk"),
        ("claude", "sdk"),
    }
)
_PROFILE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_REFERENCE_NAME = re.compile(r"[^\x00-\x20\x7f]{1,256}\Z")
_MCP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_DEPTH = 64
_MIN_JSON_INTEGER = -(2**63)
_MAX_JSON_INTEGER = 2**63 - 1
_PROVIDERS: tuple[ProviderName, ...] = (
    "openai",
    "anthropic",
    "gemini",
    "moonshot",
    "openrouter",
)
_PROVIDER_CREDENTIAL_ENVIRONMENT: dict[ProviderName, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "moonshot": ("MOONSHOT_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


class FrozenJsonDict(Mapping[str, object]):
    """An immutable mapping containing recursively frozen JSON values."""

    __slots__ = ("__items",)

    def __init__(self, value: Mapping[str, object] | None = None) -> None:
        source = {} if value is None else value
        if not isinstance(source, Mapping):
            raise InvalidAgentRequest("frozen JSON objects require a mapping")
        frozen: dict[str, object] = {}
        active = {id(source)}
        try:
            for key, child in source.items():
                if type(key) is not str:
                    raise InvalidAgentRequest("frozen JSON object keys must be strings")
                frozen[key] = _freeze_json_value(
                    child,
                    context=f"value.{key}",
                    active=active,
                    depth=1,
                )
        finally:
            active.remove(id(source))
        self.__items = tuple(frozen.items())

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_FrozenJsonDict__items" and not hasattr(self, name):
            super().__setattr__(name, value)
            return
        raise AttributeError("FrozenJsonDict is immutable")

    def __getitem__(self, key: str) -> object:
        if not isinstance(key, str):
            raise TypeError("frozen JSON object keys must be strings")
        for candidate, value in self.__items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self.__items)

    def __len__(self) -> int:
        return len(self.__items)

    def __repr__(self) -> str:
        return f"FrozenJsonDict({dict(self.items())!r})"

    def __hash__(self) -> int:
        # Mapping supplies structural, key-order-insensitive __eq__, so the hash must be
        # order-insensitive too or equal payloads land in different buckets.
        return hash(frozenset(self.__items))


def thaw_json_value(value: object) -> object:
    """Return ordinary dict/list JSON data for serializers and wire encoders."""
    if isinstance(value, FrozenJsonDict):
        return {key: thaw_json_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(child) for child in value]
    return value


type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | tuple[JsonValue, ...] | FrozenJsonDict
type JsonObject = FrozenJsonDict


def freeze_json_value(value: object, *, context: str = "value") -> JsonValue:
    return _freeze_json_value(value, context=context, active=set(), depth=0)


def _freeze_json_value(
    value: object,
    *,
    context: str,
    active: set[int],
    depth: int,
) -> JsonValue:
    if depth > _MAX_JSON_DEPTH:
        raise InvalidAgentRequest(f"{context} exceeds the maximum JSON nesting depth")
    if value is None:
        return None
    if type(value) is bool:
        return bool(value)
    if type(value) is int:
        if not _MIN_JSON_INTEGER <= value <= _MAX_JSON_INTEGER:
            raise InvalidAgentRequest(f"{context} integers must fit in signed 64 bits")
        return int(value)
    if type(value) is str:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidAgentRequest(f"{context} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise InvalidAgentRequest(f"{context} must not contain a reference cycle")
        active.add(identity)
        try:
            frozen: dict[str, object] = {}
            for key, child in value.items():
                if type(key) is not str:
                    raise InvalidAgentRequest(f"{context} object keys must be strings")
                frozen[key] = _freeze_json_value(
                    child,
                    context=f"{context}.{key}",
                    active=active,
                    depth=depth + 1,
                )
            return FrozenJsonDict(frozen)
        finally:
            active.remove(identity)
    if isinstance(value, tuple | list):
        identity = id(value)
        if identity in active:
            raise InvalidAgentRequest(f"{context} must not contain a reference cycle")
        active.add(identity)
        try:
            return tuple(
                _freeze_json_value(
                    child,
                    context=f"{context}[{index}]",
                    active=active,
                    depth=depth + 1,
                )
                for index, child in enumerate(value)
            )
        finally:
            active.remove(identity)
    raise InvalidAgentRequest(
        f"{context} must contain JSON-safe values; got {type(value).__name__}"
    )


def freeze_json_object(value: Mapping[str, object], *, context: str = "value") -> JsonObject:
    frozen = freeze_json_value(value, context=context)
    if not isinstance(frozen, FrozenJsonDict):
        raise InvalidAgentRequest(f"{context} must be a JSON object")
    return frozen


def _require_non_empty(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise InvalidAgentRequest(f"{field_name} must be a non-empty string")
    return value


def _require_tuple(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise InvalidAgentRequest(f"{field_name} must be a tuple")
    return value


def _require_unique_strings(value: object, field_name: str) -> tuple[str, ...]:
    items = _require_tuple(value, field_name)
    if any(type(item) is not str or not item for item in items):
        raise InvalidAgentRequest(f"{field_name} entries must be non-empty strings")
    if len(items) != len(set(items)):
        raise InvalidAgentRequest(f"{field_name} must not contain duplicate entries")
    return tuple(item for item in items if isinstance(item, str))


def _require_argv(value: object, field_name: str) -> tuple[str, ...]:
    items = _require_tuple(value, field_name)
    if any(type(item) is not str or not item or "\x00" in item for item in items):
        raise InvalidAgentRequest(f"{field_name} entries must be non-empty NUL-free strings")
    return tuple(item for item in items if isinstance(item, str))


def _require_absolute_path(value: object, field_name: str) -> str:
    """Validate a path lexically only; symlink resolution is the runtime's pre-launch job."""
    path = _require_non_empty(value, field_name)
    if "\x00" in path:
        raise InvalidAgentRequest(f"{field_name} must not contain NUL bytes")
    if not os.path.isabs(path):
        raise InvalidAgentRequest(f"{field_name} must be an absolute path")
    if os.path.normpath(path) != path:
        raise InvalidAgentRequest(
            f"{field_name} must be a normalized absolute path "
            "without '.', '..', or redundant separators"
        )
    return path


@dataclass(frozen=True, slots=True)
class CredentialRef:
    kind: CredentialKind
    profile_key: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("local_account", "api_key_environment", "secret_reference"):
            raise InvalidAgentRequest(f"unknown credential kind {self.kind!r}")
        if type(self.profile_key) is not str or _PROFILE_KEY.fullmatch(self.profile_key) is None:
            raise InvalidAgentRequest("profile_key must be a safe named state scope")
        if self.kind == "local_account":
            if self.name is not None:
                raise InvalidAgentRequest("CredentialRef.name must be absent for local_account")
            return
        if self.name is None:
            raise InvalidAgentRequest(f"CredentialRef.name is required for {self.kind}")
        if self.kind == "api_key_environment":
            if type(self.name) is not str or _ENVIRONMENT_NAME.fullmatch(self.name) is None:
                raise InvalidAgentRequest("api_key_environment name must be a variable name")
        elif type(self.name) is not str or _REFERENCE_NAME.fullmatch(self.name) is None:
            raise InvalidAgentRequest("secret_reference name must be a non-empty reference")


type ReasoningSummary = Literal["none", "auto", "concise", "detailed"]


@dataclass(frozen=True, slots=True)
class ReasoningSpec:
    effort: str
    thinking_budget: int | None = None
    summary: ReasoningSummary | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.effort, "ReasoningSpec.effort")
        if self.thinking_budget is not None and (
            type(self.thinking_budget) is not int or self.thinking_budget <= 0
        ):
            raise InvalidAgentRequest("ReasoningSpec.thinking_budget must be a positive integer")
        if self.summary not in (None, "none", "auto", "concise", "detailed"):
            raise InvalidAgentRequest(f"unknown reasoning summary {self.summary!r}")


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str
    kind: Literal["text"] = field(default="text", init=False)

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise InvalidAgentRequest("TextContent.text must be a string")


@dataclass(frozen=True, slots=True)
class ImageContent:
    path: str
    size_bytes: int
    media_type: str
    kind: Literal["image"] = field(default="image", init=False)

    def __post_init__(self) -> None:
        _validate_file_content(self.path, self.size_bytes, self.media_type, "ImageContent")
        if not self.media_type.startswith("image/"):
            raise InvalidAgentRequest("ImageContent.media_type must be an image media type")


@dataclass(frozen=True, slots=True)
class FileContent:
    path: str
    size_bytes: int
    media_type: str
    kind: Literal["file"] = field(default="file", init=False)

    def __post_init__(self) -> None:
        _validate_file_content(self.path, self.size_bytes, self.media_type, "FileContent")


def _validate_file_content(
    path: object, size_bytes: object, media_type: object, owner: str
) -> None:
    _require_absolute_path(path, f"{owner}.path")
    if type(size_bytes) is not int or size_bytes < 0:
        raise InvalidAgentRequest(f"{owner}.size_bytes must be a non-negative integer")
    _require_non_empty(media_type, f"{owner}.media_type")


type ContentPart = TextContent | ImageContent | FileContent
type ContentKind = Literal["text", "image", "file"]
type AttachmentKind = Literal["image", "file"]


@dataclass(frozen=True, slots=True)
class TextAgentOutput:
    kind: Literal["text"] = field(default="text", init=False)


@dataclass(frozen=True, slots=True)
class JsonSchemaAgentOutput:
    """A plain JSON Schema passed through the SDK's native output-schema option.

    The schema is not interpreted here; it is frozen so events stay immutable and
    validated only for JSON-safety. Callers with a pydantic model pass
    ``model_json_schema()``.
    """

    name: str
    schema: Mapping[str, object]
    kind: Literal["json_schema"] = field(default="json_schema", init=False)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "JsonSchemaAgentOutput.name")
        object.__setattr__(
            self, "schema", freeze_json_object(self.schema, context="JsonSchemaAgentOutput.schema")
        )


type AgentOutputSpec = TextAgentOutput | JsonSchemaAgentOutput


@dataclass(frozen=True, slots=True)
class AgentSessionRef:
    schema_version: Literal["agent-session-ref.v1"]
    backend: Backend
    transport: AgentTransport
    native_session_id: str
    profile_key: str
    state_root_fingerprint: str
    cwd_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != "agent-session-ref.v1":
            raise InvalidAgentRequest("AgentSessionRef.schema_version must be agent-session-ref.v1")
        if (self.backend, self.transport) not in AGENT_ROUTES:
            raise InvalidAgentRequest("AgentSessionRef has an unsupported backend/transport pair")
        _require_non_empty(self.native_session_id, "AgentSessionRef.native_session_id")
        if _PROFILE_KEY.fullmatch(self.profile_key) is None:
            raise InvalidAgentRequest("AgentSessionRef.profile_key is invalid")
        for name, value in (
            ("state_root_fingerprint", self.state_root_fingerprint),
            ("cwd_fingerprint", self.cwd_fingerprint),
        ):
            if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
                raise InvalidAgentRequest(f"AgentSessionRef.{name} must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class NewSession:
    kind: Literal["new"] = field(default="new", init=False)


@dataclass(frozen=True, slots=True)
class ResumeSession:
    ref: AgentSessionRef
    kind: Literal["resume"] = field(default="resume", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, AgentSessionRef):
            raise InvalidAgentRequest("ResumeSession.ref must be an AgentSessionRef")


@dataclass(frozen=True, slots=True)
class ForkSession:
    ref: AgentSessionRef
    kind: Literal["fork"] = field(default="fork", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, AgentSessionRef):
            raise InvalidAgentRequest("ForkSession.ref must be an AgentSessionRef")


type SessionOpen = NewSession | ResumeSession | ForkSession


@dataclass(frozen=True, slots=True)
class EnvironmentReference:
    name: str
    source: CredentialRef

    def __post_init__(self) -> None:
        if type(self.name) is not str or _ENVIRONMENT_NAME.fullmatch(self.name) is None:
            raise InvalidAgentRequest("environment reference name must be a variable name")
        if not isinstance(self.source, CredentialRef):
            raise InvalidAgentRequest("EnvironmentReference.source must be CredentialRef")
        if self.source.kind == "local_account":
            raise InvalidAgentRequest("environment references require a named credential source")


@dataclass(frozen=True, slots=True)
class HeaderReference:
    name: str
    source: CredentialRef

    def __post_init__(self) -> None:
        if type(self.name) is not str or _HEADER_NAME.fullmatch(self.name) is None:
            raise InvalidAgentRequest("header reference name must be an HTTP header name")
        if not isinstance(self.source, CredentialRef):
            raise InvalidAgentRequest("HeaderReference.source must be CredentialRef")
        if self.source.kind == "local_account":
            raise InvalidAgentRequest("header references require a named credential source")


type McpTransport = Literal["stdio", "streamable_http"]


@dataclass(frozen=True, slots=True)
class McpServerSpec:
    name: str
    transport: McpTransport
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    environment_refs: tuple[EnvironmentReference, ...] = ()
    header_refs: tuple[HeaderReference, ...] = ()
    required: bool = True
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not str or _MCP_NAME.fullmatch(self.name) is None:
            raise InvalidAgentRequest("McpServerSpec.name must be a safe unique name")
        if self.transport not in ("stdio", "streamable_http"):
            raise InvalidAgentRequest(f"unknown MCP transport {self.transport!r}")
        args = _require_argv(self.args, "McpServerSpec.args")
        _require_tuple(self.environment_refs, "McpServerSpec.environment_refs")
        _require_tuple(self.header_refs, "McpServerSpec.header_refs")
        environment_refs = self.environment_refs
        header_refs = self.header_refs
        if any(not isinstance(item, EnvironmentReference) for item in environment_refs):
            raise InvalidAgentRequest("McpServerSpec.environment_refs contains an invalid value")
        if any(not isinstance(item, HeaderReference) for item in header_refs):
            raise InvalidAgentRequest("McpServerSpec.header_refs contains an invalid value")
        if len({item.name for item in environment_refs}) != len(environment_refs):
            raise InvalidAgentRequest("McpServerSpec.environment_refs has duplicate names")
        if len({item.name.lower() for item in header_refs}) != len(header_refs):
            raise InvalidAgentRequest("McpServerSpec.header_refs has duplicate names")
        _require_unique_strings(self.allowed_tools, "McpServerSpec.allowed_tools")
        _require_unique_strings(self.denied_tools, "McpServerSpec.denied_tools")
        if type(self.required) is not bool:
            raise InvalidAgentRequest("McpServerSpec.required must be bool")
        if self.transport == "stdio":
            if (
                self.command is None
                or not self.command.strip()
                or "\x00" in self.command
                or self.url is not None
            ):
                raise InvalidAgentRequest("stdio MCP requires command and forbids url")
            if header_refs:
                raise InvalidAgentRequest("stdio MCP forbids header_refs")
        else:
            parsed_url = urlsplit(self.url) if type(self.url) is str else None
            if (
                parsed_url is None
                or parsed_url.scheme != "https"
                or not parsed_url.hostname
                or parsed_url.username is not None
                or parsed_url.password is not None
                or bool(parsed_url.fragment)
                or self.command is not None
                or args
            ):
                raise InvalidAgentRequest(
                    "streamable_http MCP requires an https url and forbids command/args"
                )
            if environment_refs:
                raise InvalidAgentRequest("streamable_http MCP forbids environment_refs")


def validate_mcp_network_policy(
    servers: tuple[McpServerSpec, ...], policy: PermissionPolicy
) -> None:
    """The fail-closed coupling between MCP transports and the permission policy.

    A stdio server is a local executable outside sandbox attestation, so it runs
    only under explicitly confirmed full filesystem and unrestricted network
    access. A remote server must fit the network policy, including an exact
    hostname allowlist where the policy carries one.
    """
    for server in servers:
        if server.transport == "stdio":
            if policy.filesystem != "full_access" or policy.network != "unrestricted":
                raise UnsupportedCapability(
                    "stdio MCP requires explicit full filesystem and unrestricted network policy"
                )
            continue
        if policy.network == "disabled":
            raise UnsupportedCapability("remote MCP is outside the disabled network policy")
        if policy.network == "allowlist":
            hostname = urlsplit(server.url).hostname if type(server.url) is str else None
            if hostname is None or hostname not in policy.network_allowlist:
                raise UnsupportedCapability("remote MCP host is absent from the network allowlist")


@dataclass(frozen=True, slots=True)
class CodexNativeOptions:
    web_search: bool | None = None

    def __post_init__(self) -> None:
        if self.web_search is not None and type(self.web_search) is not bool:
            raise InvalidAgentRequest("CodexNativeOptions.web_search must be bool when present")


@dataclass(frozen=True, slots=True)
class ClaudeNativeOptions:
    include_partial_messages: bool | None = None

    def __post_init__(self) -> None:
        if (
            self.include_partial_messages is not None
            and type(self.include_partial_messages) is not bool
        ):
            raise InvalidAgentRequest(
                "ClaudeNativeOptions.include_partial_messages must be bool when present"
            )


type NativeOptions = CodexNativeOptions | ClaudeNativeOptions


@dataclass(frozen=True, slots=True)
class AgentSessionRequest:
    backend: Backend
    transport: AgentTransport
    auth: CredentialRef
    open: SessionOpen
    cwd: str
    policy: PermissionPolicy
    model: str | None = None
    reasoning: ReasoningSpec | None = None
    system: tuple[ContentPart, ...] = ()
    developer: tuple[ContentPart, ...] = ()
    additional_dirs: tuple[str, ...] = ()
    mcp_servers: tuple[McpServerSpec, ...] = ()
    output: AgentOutputSpec = TextAgentOutput()
    native: NativeOptions | None = None

    def __post_init__(self) -> None:
        if (self.backend, self.transport) not in AGENT_ROUTES:
            raise InvalidAgentRequest(
                f"unsupported backend/transport pair: {self.backend!r}/{self.transport!r}"
            )
        if not isinstance(self.auth, CredentialRef):
            raise InvalidAgentRequest("AgentSessionRequest.auth must be CredentialRef")
        if not isinstance(self.open, NewSession | ResumeSession | ForkSession):
            raise InvalidAgentRequest("AgentSessionRequest.open is invalid")
        _require_absolute_path(self.cwd, "AgentSessionRequest.cwd")
        if not isinstance(self.policy, PermissionPolicy):
            raise InvalidAgentRequest("AgentSessionRequest.policy must be PermissionPolicy")
        if self.model is not None:
            _require_non_empty(self.model, "AgentSessionRequest.model")
        if self.reasoning is not None and not isinstance(self.reasoning, ReasoningSpec):
            raise InvalidAgentRequest("AgentSessionRequest.reasoning must be ReasoningSpec")
        _validate_content(self.system, "AgentSessionRequest.system", allow_empty=True)
        _validate_content(self.developer, "AgentSessionRequest.developer", allow_empty=True)
        _require_tuple(self.additional_dirs, "additional_dirs")
        for index, path in enumerate(self.additional_dirs):
            _require_absolute_path(path, f"AgentSessionRequest.additional_dirs[{index}]")
        if len(self.additional_dirs) != len(set(self.additional_dirs)):
            raise InvalidAgentRequest("AgentSessionRequest.additional_dirs contains duplicates")
        _require_tuple(self.mcp_servers, "AgentSessionRequest.mcp_servers")
        servers = self.mcp_servers
        if any(not isinstance(server, McpServerSpec) for server in servers):
            raise InvalidAgentRequest("AgentSessionRequest.mcp_servers contains an invalid value")
        if len({server.name for server in servers}) != len(servers):
            raise InvalidAgentRequest("AgentSessionRequest.mcp_servers contains duplicate names")
        if not isinstance(self.output, TextAgentOutput | JsonSchemaAgentOutput):
            raise InvalidAgentRequest("AgentSessionRequest.output is invalid")
        if self.native is not None:
            if self.backend == "codex" and not isinstance(self.native, CodexNativeOptions):
                raise InvalidAgentRequest("codex requests require CodexNativeOptions")
            if self.backend == "claude" and not isinstance(self.native, ClaudeNativeOptions):
                raise InvalidAgentRequest("claude requests require ClaudeNativeOptions")
            if (
                isinstance(self.native, CodexNativeOptions)
                and self.native.web_search is True
                and self.policy.network != "unrestricted"
            ):
                raise InvalidAgentRequest(
                    "CodexNativeOptions.web_search requires unrestricted network policy"
                )


def _validate_content(value: object, field_name: str, *, allow_empty: bool) -> None:
    parts = _require_tuple(value, field_name)
    if not allow_empty and not parts:
        raise InvalidAgentRequest(f"{field_name} must be non-empty")
    if any(not isinstance(part, TextContent | ImageContent | FileContent) for part in parts):
        raise InvalidAgentRequest(f"{field_name} contains an invalid content part")


@dataclass(frozen=True, slots=True)
class TurnRequest:
    input: tuple[ContentPart, ...]
    system: tuple[ContentPart, ...] | None = None
    developer: tuple[ContentPart, ...] | None = None
    model: str | None = None
    reasoning: ReasoningSpec | None = None
    policy: PermissionPolicyPatch | None = None
    mcp_servers: tuple[McpServerSpec, ...] | None = None
    max_turns: int | None = None
    timeout_seconds: float | None = None
    output: AgentOutputSpec | None = None
    native: NativeOptions | None = None

    def __post_init__(self) -> None:
        _validate_content(self.input, "TurnRequest.input", allow_empty=False)
        if self.system is not None:
            _validate_content(self.system, "TurnRequest.system", allow_empty=True)
        if self.developer is not None:
            _validate_content(self.developer, "TurnRequest.developer", allow_empty=True)
        if self.model is not None:
            _require_non_empty(self.model, "TurnRequest.model")
        if self.reasoning is not None and not isinstance(self.reasoning, ReasoningSpec):
            raise InvalidAgentRequest("TurnRequest.reasoning must be ReasoningSpec")
        if self.policy is not None and not isinstance(self.policy, PermissionPolicyPatch):
            raise InvalidAgentRequest("TurnRequest.policy must be PermissionPolicyPatch")
        if self.mcp_servers is not None:
            _require_tuple(self.mcp_servers, "TurnRequest.mcp_servers")
            servers = self.mcp_servers
            if any(not isinstance(server, McpServerSpec) for server in servers):
                raise InvalidAgentRequest("TurnRequest.mcp_servers contains an invalid value")
            if len({server.name for server in servers}) != len(servers):
                raise InvalidAgentRequest("TurnRequest.mcp_servers contains duplicate names")
        if self.max_turns is not None and (type(self.max_turns) is not int or self.max_turns <= 0):
            raise InvalidAgentRequest("TurnRequest.max_turns must be a positive integer")
        if self.timeout_seconds is not None and (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise InvalidAgentRequest("TurnRequest.timeout_seconds must be positive and finite")
        if self.output is not None and not isinstance(
            self.output, TextAgentOutput | JsonSchemaAgentOutput
        ):
            raise InvalidAgentRequest("TurnRequest.output is invalid")
        if self.native is not None and not isinstance(
            self.native, CodexNativeOptions | ClaudeNativeOptions
        ):
            raise InvalidAgentRequest("TurnRequest.native is invalid")


@dataclass(frozen=True, slots=True)
class ApiTarget:
    provider: ProviderName
    model: str
    credential: CredentialRef
    kind: Literal["api"] = field(default="api", init=False)

    def __post_init__(self) -> None:
        if self.provider not in _PROVIDERS:
            raise InvalidAgentRequest(f"unknown API provider {self.provider!r}")
        _require_non_empty(self.model, "ApiTarget.model")
        if not isinstance(self.credential, CredentialRef):
            raise InvalidAgentRequest("ApiTarget.credential must be CredentialRef")
        if self.credential.kind == "local_account":
            raise InvalidAgentRequest("API targets require a named API credential reference")
        if (
            self.credential.kind == "api_key_environment"
            and self.credential.name not in _PROVIDER_CREDENTIAL_ENVIRONMENT[self.provider]
        ):
            raise InvalidAgentRequest(
                f"credential environment is not valid for provider {self.provider!r}"
            )


@dataclass(frozen=True, slots=True)
class AgentTarget:
    request: AgentSessionRequest
    kind: Literal["agent"] = field(default="agent", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, AgentSessionRequest):
            raise InvalidAgentRequest("AgentTarget.request must be AgentSessionRequest")


type CallTarget = ApiTarget | AgentTarget


def api_target_to_provider_target(target: ApiTarget) -> ProviderTarget:
    if not isinstance(target, ApiTarget):
        raise InvalidAgentRequest("api_target_to_provider_target requires ApiTarget")
    return ProviderTarget(provider=target.provider, model=target.model)


def agent_target_to_session_request(target: AgentTarget) -> AgentSessionRequest:
    if not isinstance(target, AgentTarget):
        raise InvalidAgentRequest("agent_target_to_session_request requires AgentTarget")
    return target.request


def ref_to_json(ref: AgentSessionRef) -> JsonObject:
    if not isinstance(ref, AgentSessionRef):
        raise InvalidAgentRequest("ref_to_json requires AgentSessionRef")
    return FrozenJsonDict(
        {
            "schema_version": ref.schema_version,
            "backend": ref.backend,
            "transport": ref.transport,
            "native_session_id": ref.native_session_id,
            "profile_key": ref.profile_key,
            "state_root_fingerprint": ref.state_root_fingerprint,
            "cwd_fingerprint": ref.cwd_fingerprint,
        }
    )


_REF_FIELDS = frozenset(
    {
        "schema_version",
        "backend",
        "transport",
        "native_session_id",
        "profile_key",
        "state_root_fingerprint",
        "cwd_fingerprint",
    }
)


def ref_from_json(value: Mapping[str, object]) -> AgentSessionRef:
    if not isinstance(value, Mapping):
        raise InvalidAgentRequest("agent session ref JSON must be an object")
    keys = set(value)
    unknown = sorted(keys - _REF_FIELDS)
    missing = sorted(_REF_FIELDS - keys)
    if unknown:
        raise InvalidAgentRequest(f"agent session ref JSON has unknown fields: {unknown}")
    if missing:
        raise InvalidAgentRequest(f"agent session ref JSON is missing fields: {missing}")
    if value["schema_version"] != "agent-session-ref.v1":
        raise InvalidAgentRequest("agent session ref schema_version is unsupported")
    backend_value = value["backend"]
    transport_value = value["transport"]
    string_fields = {
        name: value[name]
        for name in (
            "native_session_id",
            "profile_key",
            "state_root_fingerprint",
            "cwd_fingerprint",
        )
    }
    if backend_value not in _BACKENDS or type(backend_value) is not str:
        raise InvalidAgentRequest("agent session ref backend is invalid")
    if transport_value not in _TRANSPORTS or type(transport_value) is not str:
        raise InvalidAgentRequest("agent session ref transport is invalid")
    if any(type(item) is not str for item in string_fields.values()):
        raise InvalidAgentRequest("agent session ref JSON fields have invalid types")
    backend: Backend = backend_value
    transport: AgentTransport = transport_value
    return AgentSessionRef(
        schema_version="agent-session-ref.v1",
        backend=backend,
        transport=transport,
        native_session_id=str(string_fields["native_session_id"]),
        profile_key=str(string_fields["profile_key"]),
        state_root_fingerprint=str(string_fields["state_root_fingerprint"]),
        cwd_fingerprint=str(string_fields["cwd_fingerprint"]),
    )


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    operation: Literal["command", "file_change", "tool_use"]
    summary: str
    tool_name: str | None = None
    native_payload: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.operation not in ("command", "file_change", "tool_use"):
            raise InvalidAgentRequest(f"unknown approval operation {self.operation!r}")
        _require_non_empty(self.summary, "ApprovalRequest.summary")
        if self.operation == "tool_use":
            _require_non_empty(self.tool_name, "ApprovalRequest.tool_name")
        elif self.tool_name is not None:
            raise InvalidAgentRequest("tool_name is only valid for tool_use approvals")
        if self.native_payload is not None and not isinstance(self.native_payload, FrozenJsonDict):
            raise InvalidAgentRequest("ApprovalRequest.native_payload must be frozen JSON")
        if self.native_payload is not None:
            require_frozen_json(self.native_payload, "ApprovalRequest.native_payload")


type ApprovalHandler = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


def require_frozen_json(value: object, field_name: str) -> None:
    """Assert a value is already recursively frozen JSON, raising `InvalidAgentRequest` if not.

    types.py owns the frozen-JSON representation, so it owns this check; callers that need a
    different error class wrap the raise at their own boundary.
    """
    _require_frozen_json(value, field_name, active=set(), depth=0)


def _require_frozen_json(
    value: object,
    field_name: str,
    *,
    active: set[int],
    depth: int,
) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise InvalidAgentRequest(f"{field_name} exceeds the maximum JSON nesting depth")
    if value is None or type(value) in (bool, int, float, str):
        freeze_json_value(value, context=field_name)
        return
    if isinstance(value, FrozenJsonDict):
        identity = id(value)
        if identity in active:
            raise InvalidAgentRequest(f"{field_name} must not contain a reference cycle")
        active.add(identity)
        try:
            for key, child in value.items():
                _require_frozen_json(
                    child,
                    f"{field_name}.{key}",
                    active=active,
                    depth=depth + 1,
                )
        finally:
            active.remove(identity)
        return
    if isinstance(value, tuple):
        identity = id(value)
        if identity in active:
            raise InvalidAgentRequest(f"{field_name} must not contain a reference cycle")
        active.add(identity)
        try:
            for index, child in enumerate(value):
                _require_frozen_json(
                    child,
                    f"{field_name}[{index}]",
                    active=active,
                    depth=depth + 1,
                )
        finally:
            active.remove(identity)
        return
    raise InvalidAgentRequest(f"{field_name} must be recursively frozen JSON")
