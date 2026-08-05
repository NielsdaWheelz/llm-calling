"""Discovered agent capabilities and fail-closed request validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never
from urllib.parse import urlsplit

from .errors import AgentRuntimeDefect, InvalidAgentRequest, UnsupportedCapability
from .policy import (
    ApprovalMode,
    FilesystemMode,
    NetworkMode,
    PermissionPolicy,
    PermissionPolicyPatch,
    narrow_policy,
)
from .types import (
    AGENT_ROUTES,
    AgentSessionRequest,
    AgentTransport,
    AttachmentKind,
    Backend,
    ClaudeNativeOptions,
    CodexNativeOptions,
    ContentKind,
    ContentPart,
    CredentialRef,
    FileContent,
    ForkSession,
    ImageContent,
    JsonSchemaAgentOutput,
    McpServerSpec,
    McpTransport,
    NewSession,
    ReasoningSpec,
    ResumeSession,
    TextContent,
    TurnRequest,
    native_option_names,
)

type SessionOperation = Literal["new", "resume", "fork"]
# One cross-backend vocabulary, deliberately not vendor tool names. A caller programming
# against this package branches on what a backend's built-ins can *do* — "can this transport
# edit files for me", "can it reach the web" — and cannot be asked to know that Codex spells
# file editing `apply_patch` while Claude Code spells it `Write`/`Edit`. Per-vendor spellings
# stay where they belong: in `PermissionPolicy.allowed_tools`, which is native by contract —
# and, so that a caller has somewhere to read them, in `AgentCapabilities.builtin_tool_names`.
type BuiltinToolFamily = Literal[
    "file_read",
    "file_write",
    "command",
    "web_fetch",
    "web_search",
]
type DiscoveryOperation = Literal["list", "read", "turn_history", "item_history"]
type InstructionRole = Literal["system", "developer", "user"]
type McpAuthForm = Literal["environment_reference", "header_reference"]
type TurnOverride = Literal[
    "system", "developer", "model", "reasoning", "policy", "mcp_servers", "output", "native"
]

_SESSION_OPERATIONS: tuple[SessionOperation, ...] = ("new", "resume", "fork")
_DISCOVERY_OPERATIONS: tuple[DiscoveryOperation, ...] = (
    "list",
    "read",
    "turn_history",
    "item_history",
)
_ROLES: tuple[InstructionRole, ...] = ("system", "developer", "user")
_CONTENT_KINDS: tuple[ContentKind, ...] = ("text", "image", "file")
_ATTACHMENT_KINDS: tuple[AttachmentKind, ...] = ("image", "file")
_BUILTIN_TOOL_FAMILIES: tuple[BuiltinToolFamily, ...] = (
    "file_read",
    "file_write",
    "command",
    "web_fetch",
    "web_search",
)
_MCP_TRANSPORTS: tuple[McpTransport, ...] = ("stdio", "streamable_http")
_MCP_AUTH_FORMS: tuple[McpAuthForm, ...] = ("environment_reference", "header_reference")
_FILESYSTEM_MODES: tuple[FilesystemMode, ...] = ("read_only", "workspace_write", "full_access")
_NETWORK_MODES: tuple[NetworkMode, ...] = ("disabled", "allowlist", "unrestricted")
_APPROVAL_MODES: tuple[ApprovalMode, ...] = ("deny", "ask", "provider_review", "allow")
_OVERRIDES: tuple[TurnOverride, ...] = (
    "system",
    "developer",
    "model",
    "reasoning",
    "policy",
    "mcp_servers",
    "output",
    "native",
)


def _invalid_report(message: str) -> AgentRuntimeDefect:
    """A capability table our own adapter built is broken; never a product-facing limitation."""
    return AgentRuntimeDefect(message, code="invalid_capability_report")


def _strict_unique_tuple(
    value: object, field_name: str, *, allowed: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise _invalid_report(f"{field_name} must be a tuple")
    if any(type(item) is not str or not item for item in value):
        raise _invalid_report(f"{field_name} entries must be non-empty strings")
    if len(value) != len(set(value)):
        raise _invalid_report(f"{field_name} contains duplicate entries")
    if allowed is not None and any(item not in allowed for item in value):
        raise _invalid_report(f"{field_name} contains an unknown value")
    return value


def _positive_optional_int(value: int | None, field_name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise _invalid_report(f"{field_name} must be a positive integer when present")


def _positive_optional_float(value: float | None, field_name: str) -> None:
    if value is not None and (type(value) not in (int, float) or value <= 0):
        raise _invalid_report(f"{field_name} must be positive when present")


@dataclass(frozen=True, slots=True)
class AgentCapabilityScope:
    backend: Backend
    transport: AgentTransport
    auth: CredentialRef

    def __post_init__(self) -> None:
        if (self.backend, self.transport) not in AGENT_ROUTES:
            raise InvalidAgentRequest("capability scope has an unsupported backend/transport pair")
        if not isinstance(self.auth, CredentialRef):
            raise InvalidAgentRequest("capability scope auth must be CredentialRef")


@dataclass(frozen=True, slots=True)
class AgentCapabilities:
    scope: AgentCapabilityScope
    executable_version: str | None = None
    sdk_version: str | None = None
    session_operations: tuple[SessionOperation, ...] = ("new",)
    discovery_operations: tuple[DiscoveryOperation, ...] = ()
    models: tuple[str, ...] | None = None
    reasoning_efforts: tuple[str, ...] | None = None
    model_reasoning_efforts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    reasoning_summaries: tuple[str, ...] = ()
    thinking_budget: bool = False
    content_kinds: tuple[ContentKind, ...] = ("text",)
    content_roles: tuple[InstructionRole, ...] = ("user",)
    attachment_kinds: tuple[AttachmentKind, ...] = ()
    session_instruction_roles: tuple[InstructionRole, ...] = ()
    turn_instruction_roles: tuple[InstructionRole, ...] = ()
    streaming: bool = True
    cancellation: bool = True
    structured_output: bool = False
    native_output_schema: bool = False
    builtin_tool_families: tuple[BuiltinToolFamily, ...] = ()
    # The exact native tool names this transport accepts in `PermissionPolicy.allowed_tools`
    # and `denied_tools`, grouped by the family each name belongs to — the one bridge from the
    # normalized `builtin_tool_families` vocabulary to the vendor spellings a policy is written
    # in. Neither backend enumerates these at discovery time (the Claude Agent SDK ships no
    # tool-name constant, and Claude Code's `system/init` reports only the tools the session
    # already asked for), so an adapter fills this from a table pinned to the executable
    # version it reports, exactly as it pins any other version-specific vocabulary.
    #
    # It carries *accepted* names, not merely existing ones: a name an adapter would refuse in
    # a policy does not belong here even if the backend ships the tool. An empty tuple is the
    # honest report that this transport does not publish its native tool names; a caller must
    # then take them from the backend's own documentation for `executable_version`. Names are
    # only meaningful where the transport honours them, so a non-empty table requires
    # `tool_controls`.
    builtin_tool_names: tuple[tuple[BuiltinToolFamily, tuple[str, ...]], ...] = ()
    tool_controls: bool = False
    mcp_transports: tuple[McpTransport, ...] = ()
    mcp_auth_forms: tuple[McpAuthForm, ...] = ()
    filesystem_modes: tuple[FilesystemMode, ...] = ("read_only",)
    network_modes: tuple[NetworkMode, ...] = ("disabled",)
    network_allowlist: bool = False
    approval_modes: tuple[ApprovalMode, ...] = ("deny",)
    additional_dirs: bool = False
    max_turns: bool = False
    max_turns_limit: int | None = None
    timeouts: bool = True
    max_timeout_seconds: float | None = None
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    turn_overrides: tuple[TurnOverride, ...] = ()
    persistent_turn_overrides: tuple[TurnOverride, ...] = ()
    native_extension_version: str | None = None
    native_option_names: tuple[str, ...] = ()
    cwd_scopes_sessions: bool = False
    reports_auth_identity: bool = False
    # Whether the transport reports the effort it actually ran with. Where it does not, the
    # spec's requested-vs-effective diagnostic is unobservable and a caller must treat a
    # silently clamped effort as possible; saying so here beats leaving a permanently silent
    # branch in the adapter that reads like a live check.
    reports_effective_effort: bool = False
    # Which provider-native session metadata dimensions the discovery ports can actually fill,
    # so a consumer can tell "this backend has no session names" from "this session is unnamed".
    session_name_metadata: bool = False
    session_archive_metadata: bool = False
    session_tag_metadata: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scope, AgentCapabilityScope):
            raise _invalid_report("AgentCapabilities.scope must be AgentCapabilityScope")
        for name in ("executable_version", "sdk_version", "native_extension_version"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise _invalid_report(f"{name} must be a non-empty string when present")
        _strict_unique_tuple(
            self.session_operations, "session_operations", allowed=_SESSION_OPERATIONS
        )
        _strict_unique_tuple(
            self.discovery_operations, "discovery_operations", allowed=_DISCOVERY_OPERATIONS
        )
        # Provider-defined vocabularies stay open; every finite one is pinned to its own union.
        for name in (
            "models",
            "reasoning_efforts",
            "reasoning_summaries",
            "native_option_names",
        ):
            value = getattr(self, name)
            if value is not None:
                _strict_unique_tuple(value, name)
        for name, allowed in (
            ("builtin_tool_families", _BUILTIN_TOOL_FAMILIES),
            ("content_kinds", _CONTENT_KINDS),
            ("content_roles", _ROLES),
            ("attachment_kinds", _ATTACHMENT_KINDS),
            ("mcp_transports", _MCP_TRANSPORTS),
            ("mcp_auth_forms", _MCP_AUTH_FORMS),
            ("filesystem_modes", _FILESYSTEM_MODES),
            ("network_modes", _NETWORK_MODES),
            ("approval_modes", _APPROVAL_MODES),
        ):
            _strict_unique_tuple(getattr(self, name), name, allowed=allowed)
        if not isinstance(self.model_reasoning_efforts, tuple):
            raise _invalid_report("model_reasoning_efforts must be a tuple")
        mapped_models: list[str] = []
        for entry in self.model_reasoning_efforts:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise _invalid_report("model_reasoning_efforts entries must be pairs")
            model, efforts = entry
            if type(model) is not str or not model:
                raise _invalid_report("model_reasoning_efforts model names must be strings")
            _strict_unique_tuple(efforts, f"model_reasoning_efforts[{model!r}]")
            mapped_models.append(model)
        if len(mapped_models) != len(set(mapped_models)):
            raise _invalid_report("model_reasoning_efforts contains duplicate models")
        if self.models is None and mapped_models:
            raise _invalid_report("model_reasoning_efforts requires enumerated models")
        if self.models is not None and any(model not in self.models for model in mapped_models):
            raise _invalid_report("model_reasoning_efforts contains an unknown model")
        if not isinstance(self.builtin_tool_names, tuple):
            raise _invalid_report("builtin_tool_names must be a tuple")
        named_families: list[str] = []
        for entry in self.builtin_tool_names:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise _invalid_report("builtin_tool_names entries must be pairs")
            family, names = entry
            if family not in self.builtin_tool_families:
                raise _invalid_report("builtin_tool_names names an unreported tool family")
            _strict_unique_tuple(names, f"builtin_tool_names[{family!r}]")
            if not names:
                raise _invalid_report(
                    f"builtin_tool_names[{family!r}] must list at least one native tool name"
                )
            named_families.append(family)
        if len(named_families) != len(set(named_families)):
            raise _invalid_report("builtin_tool_names contains duplicate families")
        if named_families and not self.tool_controls:
            raise _invalid_report("builtin_tool_names requires tool allow/deny controls")
        _strict_unique_tuple(
            self.session_instruction_roles, "session_instruction_roles", allowed=_ROLES
        )
        _strict_unique_tuple(self.turn_instruction_roles, "turn_instruction_roles", allowed=_ROLES)
        _strict_unique_tuple(self.turn_overrides, "turn_overrides", allowed=_OVERRIDES)
        persistent = _strict_unique_tuple(
            self.persistent_turn_overrides,
            "persistent_turn_overrides",
            allowed=_OVERRIDES,
        )
        if not set(persistent).issubset(self.turn_overrides):
            raise _invalid_report("persistent_turn_overrides must be a subset of turn_overrides")
        for name in (
            "thinking_budget",
            "streaming",
            "cancellation",
            "structured_output",
            "native_output_schema",
            "tool_controls",
            "network_allowlist",
            "additional_dirs",
            "max_turns",
            "timeouts",
            "cwd_scopes_sessions",
            "reports_auth_identity",
            "reports_effective_effort",
            "session_name_metadata",
            "session_archive_metadata",
            "session_tag_metadata",
        ):
            if type(getattr(self, name)) is not bool:
                raise _invalid_report(f"{name} must be bool")
        _positive_optional_int(self.max_turns_limit, "max_turns_limit")
        _positive_optional_float(self.max_timeout_seconds, "max_timeout_seconds")
        _positive_optional_int(self.max_context_tokens, "max_context_tokens")
        _positive_optional_int(self.max_output_tokens, "max_output_tokens")
        # `validate_mcp_network_policy` only lets a stdio MCP server run under
        # `full_access` + `unrestricted`, because a stdio server is a local executable the
        # sandbox cannot mediate. A table advertising stdio without both of those modes would
        # advertise a transport no policy it also accepts can ever reach, so state the
        # coupling here rather than let a caller discover it as a late UnsupportedCapability.
        if "stdio" in self.mcp_transports and not (
            "full_access" in self.filesystem_modes and "unrestricted" in self.network_modes
        ):
            raise _invalid_report(
                "stdio MCP requires filesystem full_access and unrestricted network support"
            )
        if self.max_turns_limit is not None and not self.max_turns:
            raise _invalid_report("max_turns_limit requires max_turns support")
        if self.max_timeout_seconds is not None and not self.timeouts:
            raise _invalid_report("max_timeout_seconds requires timeout support")


def require_discovery_operation(
    capabilities: AgentCapabilities, operation: DiscoveryOperation
) -> None:
    if operation not in capabilities.discovery_operations:
        raise UnsupportedCapability(f"session discovery operation {operation!r} is unsupported")


def _session_operation(request: AgentSessionRequest) -> SessionOperation:
    match request.open:
        case NewSession():
            return "new"
        case ResumeSession():
            return "resume"
        case ForkSession():
            return "fork"
        case _:
            assert_never(request.open)


def _validate_scope(request: AgentSessionRequest, capabilities: AgentCapabilities) -> None:
    scope = capabilities.scope
    if (
        request.backend != scope.backend
        or request.transport != scope.transport
        or request.auth != scope.auth
    ):
        raise UnsupportedCapability("request does not match the discovered capability scope")


def _validate_content_kinds(
    parts: tuple[ContentPart, ...], capabilities: AgentCapabilities
) -> None:
    for part in parts:
        match part:
            case TextContent():
                attachment = False
            case ImageContent() | FileContent():
                attachment = True
            case _:
                assert_never(part)
        if part.kind not in capabilities.content_kinds:
            raise UnsupportedCapability(f"content kind {part.kind!r} is unsupported")
        if attachment and part.kind not in capabilities.attachment_kinds:
            raise UnsupportedCapability(f"attachment kind {part.kind!r} is unsupported")


def _validate_mcp(servers: tuple[McpServerSpec, ...], capabilities: AgentCapabilities) -> None:
    for server in servers:
        if server.transport not in capabilities.mcp_transports:
            raise UnsupportedCapability(
                f"MCP transport {server.transport!r} is unsupported for {server.name!r}"
            )
        if server.environment_refs and "environment_reference" not in capabilities.mcp_auth_forms:
            raise UnsupportedCapability("MCP environment references are unsupported")
        if server.header_refs and "header_reference" not in capabilities.mcp_auth_forms:
            raise UnsupportedCapability("MCP header references are unsupported")


def validate_mcp_network_policy(
    servers: tuple[McpServerSpec, ...], policy: PermissionPolicy
) -> None:
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
            hostname = urlsplit(server.url).hostname if server.url is not None else None
            if hostname is None or hostname not in policy.network_allowlist:
                raise UnsupportedCapability("remote MCP host is absent from the network allowlist")


def _validate_policy(request: AgentSessionRequest, capabilities: AgentCapabilities) -> None:
    _validate_permission_policy(request.policy, capabilities)


def _validate_permission_policy(policy: PermissionPolicy, capabilities: AgentCapabilities) -> None:
    if policy.filesystem not in capabilities.filesystem_modes:
        raise UnsupportedCapability(f"filesystem mode {policy.filesystem!r} is unsupported")
    if policy.network not in capabilities.network_modes:
        raise UnsupportedCapability(f"network mode {policy.network!r} is unsupported")
    if policy.network == "allowlist" and not capabilities.network_allowlist:
        raise UnsupportedCapability("exact network allowlist entries are unsupported")
    if policy.approval not in capabilities.approval_modes:
        raise UnsupportedCapability(f"approval mode {policy.approval!r} is unsupported")
    if not capabilities.tool_controls:
        if capabilities.builtin_tool_families:
            if policy.allowed_tools != ("*",) or policy.denied_tools:
                raise UnsupportedCapability(
                    "transport cannot disable or filter its built-in tools; "
                    "an explicit unrestricted '*' sentinel is required"
                )
        elif policy.allowed_tools or policy.denied_tools:
            raise UnsupportedCapability("tool allow/deny controls are unsupported")


def _validate_native(
    native: CodexNativeOptions | ClaudeNativeOptions | None,
    capabilities: AgentCapabilities,
) -> None:
    if native is None:
        return
    requested = native_option_names(native)
    unsupported = sorted(set(requested) - set(capabilities.native_option_names))
    if unsupported:
        raise UnsupportedCapability(f"native option names are unsupported: {unsupported}")
    if requested and capabilities.native_extension_version is None:
        raise UnsupportedCapability("native options require a discovered extension version")


def validate_auth_capability(credential: CredentialRef, capabilities: AgentCapabilities) -> None:
    """A transport that cannot report its effective identity may only use the local account."""
    if credential.kind != "local_account" and not capabilities.reports_auth_identity:
        raise UnsupportedCapability(
            f"{credential.kind} auth requires a transport that reports its effective "
            "authentication identity"
        )


def _validate_model_and_effort(
    model: str | None, reasoning: ReasoningSpec | None, capabilities: AgentCapabilities
) -> None:
    """One owner for the (model, reasoning) capability rules on both the session and turn path."""
    if model is not None and capabilities.models is not None and model not in capabilities.models:
        raise UnsupportedCapability(f"model {model!r} is unsupported")
    if reasoning is None:
        return
    effort_map = dict(capabilities.model_reasoning_efforts)
    if model is not None and effort_map:
        if model not in effort_map or reasoning.effort not in effort_map[model]:
            raise UnsupportedCapability(
                f"reasoning effort {reasoning.effort!r} is unsupported for model {model!r}"
            )
    elif (
        capabilities.reasoning_efforts is not None
        and reasoning.effort not in capabilities.reasoning_efforts
    ):
        raise UnsupportedCapability(f"reasoning effort {reasoning.effort!r} is unsupported")
    if reasoning.thinking_budget is not None and not capabilities.thinking_budget:
        raise UnsupportedCapability("native thinking budgets are unsupported")
    if reasoning.summary is not None and reasoning.summary not in capabilities.reasoning_summaries:
        raise UnsupportedCapability(f"reasoning summary {reasoning.summary!r} is unsupported")


def validate_session_capabilities(
    request: AgentSessionRequest, capabilities: AgentCapabilities
) -> None:
    _validate_scope(request, capabilities)
    validate_auth_capability(request.auth, capabilities)
    operation = _session_operation(request)
    if operation not in capabilities.session_operations:
        raise UnsupportedCapability(f"session operation {operation!r} is unsupported")
    if not capabilities.streaming:
        raise UnsupportedCapability("streaming is required by the agent runtime contract")
    _validate_model_and_effort(request.model, request.reasoning, capabilities)
    _validate_content_kinds(request.system + request.developer, capabilities)
    if "user" not in capabilities.content_roles:
        raise UnsupportedCapability("user content is unsupported")
    if request.system and "system" not in capabilities.session_instruction_roles:
        raise UnsupportedCapability("session-scoped system content is unsupported")
    if request.developer and "developer" not in capabilities.session_instruction_roles:
        raise UnsupportedCapability("session-scoped developer content is unsupported")
    if isinstance(request.output, JsonSchemaAgentOutput) and not (
        capabilities.structured_output and capabilities.native_output_schema
    ):
        raise UnsupportedCapability("structured output with a native output schema is unsupported")
    _validate_policy(request, capabilities)
    if request.additional_dirs and not capabilities.additional_dirs:
        raise UnsupportedCapability("additional working directories are unsupported")
    _validate_mcp(request.mcp_servers, capabilities)
    validate_mcp_network_policy(request.mcp_servers, request.policy)
    _validate_native(request.native, capabilities)


def _require_override(name: TurnOverride, capabilities: AgentCapabilities) -> None:
    if name not in capabilities.turn_overrides:
        raise UnsupportedCapability(f"per-turn {name} override is unsupported")


def validate_turn_capabilities(
    request: TurnRequest,
    capabilities: AgentCapabilities,
    *,
    session_model: str | None,
) -> None:
    """Validate one turn against the capability table, including the model it will actually run.

    `session_model` is the session's own model, which a turn that overrides only the reasoning
    effort keeps. Without it the effort would be checked against the union of every model's
    efforts instead of against the model the turn really uses, and an effort the session's
    model does not support would pass here and be rejected natively mid-turn.
    """
    if "user" not in capabilities.content_roles:
        raise UnsupportedCapability("user content is unsupported")
    _validate_content_kinds(request.input, capabilities)
    if request.system is not None:
        _require_override("system", capabilities)
        if "system" not in capabilities.turn_instruction_roles:
            raise UnsupportedCapability("per-turn system content is unsupported")
        _validate_content_kinds(request.system, capabilities)
    if request.developer is not None:
        _require_override("developer", capabilities)
        if "developer" not in capabilities.turn_instruction_roles:
            raise UnsupportedCapability("per-turn developer content is unsupported")
        _validate_content_kinds(request.developer, capabilities)
    if request.model is not None:
        _require_override("model", capabilities)
    if request.reasoning is not None:
        _require_override("reasoning", capabilities)
    _validate_model_and_effort(
        request.model if request.model is not None else session_model,
        request.reasoning,
        capabilities,
    )
    if request.policy is not None:
        _require_override("policy", capabilities)
    if request.mcp_servers is not None:
        _require_override("mcp_servers", capabilities)
        _validate_mcp(request.mcp_servers, capabilities)
    if request.output is not None:
        _require_override("output", capabilities)
        if isinstance(request.output, JsonSchemaAgentOutput) and not (
            capabilities.structured_output and capabilities.native_output_schema
        ):
            raise UnsupportedCapability(
                "structured output with a native output schema is unsupported"
            )
    if request.native is not None:
        _require_override("native", capabilities)
        if capabilities.scope.backend == "codex" and not isinstance(
            request.native, CodexNativeOptions
        ):
            raise UnsupportedCapability("codex turns require CodexNativeOptions")
        if capabilities.scope.backend == "claude" and not isinstance(
            request.native, ClaudeNativeOptions
        ):
            raise UnsupportedCapability("claude turns require ClaudeNativeOptions")
        _validate_native(request.native, capabilities)
    if request.max_turns is not None:
        if not capabilities.max_turns:
            raise UnsupportedCapability("max_turns is unsupported")
        if (
            capabilities.max_turns_limit is not None
            and request.max_turns > capabilities.max_turns_limit
        ):
            raise UnsupportedCapability("max_turns exceeds the discovered limit")
    if request.timeout_seconds is not None:
        if not capabilities.timeouts:
            raise UnsupportedCapability("runtime timeouts are unsupported")
        if (
            capabilities.max_timeout_seconds is not None
            and request.timeout_seconds > capabilities.max_timeout_seconds
        ):
            raise UnsupportedCapability("timeout exceeds the discovered limit")


def validate_turn_policy(
    session_policy: PermissionPolicy,
    patch: PermissionPolicyPatch,
    capabilities: AgentCapabilities,
) -> PermissionPolicy:
    effective = narrow_policy(session_policy, patch)
    _validate_permission_policy(effective, capabilities)
    return effective
