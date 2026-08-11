"""Immutable permission policy and exact-set per-turn narrowing."""

from __future__ import annotations

import fnmatch
import ipaddress
import re
from dataclasses import dataclass
from typing import Literal, cast

from ._validation import ENVIRONMENT_NAME, require_unique_strings
from .errors import InvalidAgentRequest

type FilesystemMode = Literal["read_only", "workspace_write", "full_access"]
type NetworkMode = Literal["disabled", "allowlist", "unrestricted"]
type ApprovalMode = Literal["deny", "ask", "provider_review", "allow"]
type UnsafeDimension = Literal["filesystem_full_access", "network_unrestricted", "approval_allow"]

_FILESYSTEM_ORDER: tuple[FilesystemMode, ...] = (
    "read_only",
    "workspace_write",
    "full_access",
)
_NETWORK_ORDER: tuple[NetworkMode, ...] = ("disabled", "allowlist", "unrestricted")
_APPROVAL_MODES: tuple[ApprovalMode, ...] = ("deny", "ask", "provider_review", "allow")
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_UNSAFE_DIMENSIONS: tuple[UnsafeDimension, ...] = (
    "filesystem_full_access",
    "network_unrestricted",
    "approval_allow",
)


def _validate_network_host(host: str) -> None:
    if len(host) > 253 or host != host.lower() or host.endswith(".") or "." not in host:
        raise InvalidAgentRequest(
            "network_allowlist entries must be canonical lowercase DNS hostnames"
        )
    try:
        host.encode("ascii")
    except UnicodeEncodeError:
        raise InvalidAgentRequest(
            "network_allowlist entries must use canonical ASCII IDNA hostnames"
        ) from None
    if any(_DNS_LABEL.fullmatch(label) is None for label in host.split(".")):
        raise InvalidAgentRequest("network_allowlist entries contain an invalid DNS label")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise InvalidAgentRequest("network_allowlist entries must not be IP addresses")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise InvalidAgentRequest("network_allowlist entries must not target local hostnames")


@dataclass(frozen=True, slots=True)
class UnsafeConfirmation:
    acknowledged: tuple[UnsafeDimension, ...]

    def __post_init__(self) -> None:
        acknowledged = require_unique_strings(self.acknowledged, "acknowledged")
        if any(item not in _UNSAFE_DIMENSIONS for item in acknowledged):
            raise InvalidAgentRequest("acknowledged contains an unknown unsafe dimension")


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    filesystem: FilesystemMode = "read_only"
    network: NetworkMode = "disabled"
    approval: ApprovalMode = "deny"
    network_allowlist: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    unsafe_confirmation: UnsafeConfirmation | None = None

    def __post_init__(self) -> None:
        if self.filesystem not in _FILESYSTEM_ORDER:
            raise InvalidAgentRequest(f"unknown filesystem mode {self.filesystem!r}")
        if self.network not in _NETWORK_ORDER:
            raise InvalidAgentRequest(f"unknown network mode {self.network!r}")
        if self.approval not in _APPROVAL_MODES:
            raise InvalidAgentRequest(f"unknown approval mode {self.approval!r}")
        require_unique_strings(self.allowed_tools, "allowed_tools")
        require_unique_strings(self.denied_tools, "denied_tools")
        network_allowlist = require_unique_strings(self.network_allowlist, "network_allowlist")
        for entry in network_allowlist:
            _validate_network_host(entry)
        if self.network == "allowlist" and not network_allowlist:
            raise InvalidAgentRequest("network allowlist mode requires network_allowlist entries")
        if self.network != "allowlist" and network_allowlist:
            raise InvalidAgentRequest("network_allowlist is only valid in allowlist mode")
        environment = require_unique_strings(self.environment, "environment")
        if any(ENVIRONMENT_NAME.fullmatch(name) is None for name in environment):
            raise InvalidAgentRequest("environment entries must be valid variable names")

        required: set[UnsafeDimension] = set()
        if self.filesystem == "full_access":
            required.add("filesystem_full_access")
        if self.network == "unrestricted":
            required.add("network_unrestricted")
        if self.approval == "allow":
            required.add("approval_allow")
        actual = (
            set(self.unsafe_confirmation.acknowledged)
            if self.unsafe_confirmation is not None
            else set()
        )
        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)
            detail = f"missing={missing}, extra={extra}"
            raise InvalidAgentRequest(
                "unsafe_confirmation must acknowledge exactly every widened dimension: " + detail
            )


@dataclass(frozen=True, slots=True)
class PermissionPolicyPatch:
    filesystem: FilesystemMode | None = None
    network: NetworkMode | None = None
    approval: ApprovalMode | None = None
    network_allowlist: tuple[str, ...] | None = None
    allowed_tools: tuple[str, ...] | None = None
    denied_tools: tuple[str, ...] | None = None
    environment: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.filesystem is not None and self.filesystem not in _FILESYSTEM_ORDER:
            raise InvalidAgentRequest(f"unknown filesystem mode {self.filesystem!r}")
        if self.network is not None and self.network not in _NETWORK_ORDER:
            raise InvalidAgentRequest(f"unknown network mode {self.network!r}")
        if self.approval is not None and self.approval not in _APPROVAL_MODES:
            raise InvalidAgentRequest(f"unknown approval mode {self.approval!r}")
        for field in (
            "network_allowlist",
            "allowed_tools",
            "denied_tools",
            "environment",
        ):
            value = getattr(self, field)
            if value is not None:
                normalized = require_unique_strings(value, field)
                if field == "environment" and any(
                    ENVIRONMENT_NAME.fullmatch(name) is None for name in normalized
                ):
                    raise InvalidAgentRequest("environment entries must be valid variable names")
                if field == "network_allowlist":
                    for entry in normalized:
                        _validate_network_host(entry)


def _require_not_wider[T](*, field: str, base: T, requested: T | None, order: tuple[T, ...]) -> T:
    if requested is None:
        return base
    if order.index(requested) > order.index(base):
        raise InvalidAgentRequest(f"per-turn policy cannot widen {field}")
    return requested


def _narrow_approval(base: ApprovalMode, requested: ApprovalMode | None) -> ApprovalMode:
    """Narrow approval authority without pretending review mechanisms are ordered.

    Caller review and provider review can each deny an escalation and can each approve one;
    neither is a restriction of the other.  Unconditional allow may be narrowed to either
    reviewer, while every mode may be narrowed to deny.
    """
    if requested is None or requested == base:
        return base
    if requested == "deny":
        return requested
    if base == "allow" and requested in ("ask", "provider_review"):
        return requested
    raise InvalidAgentRequest("per-turn policy cannot widen or change approval reviewer")


def narrow_policy(base: PermissionPolicy, patch: PermissionPolicyPatch) -> PermissionPolicy:
    """Apply a restrictive patch without interpreting pattern containment."""
    allowed = base.allowed_tools if patch.allowed_tools is None else patch.allowed_tools
    denied = base.denied_tools if patch.denied_tools is None else patch.denied_tools
    environment = base.environment if patch.environment is None else patch.environment
    if not set(allowed).issubset(base.allowed_tools):
        raise InvalidAgentRequest("allowed_tools patch must be an exact subset of session policy")
    if not set(denied).issuperset(base.denied_tools):
        raise InvalidAgentRequest("denied_tools patch must be an exact superset of session policy")
    if not set(environment).issubset(base.environment):
        raise InvalidAgentRequest("environment patch must be an exact subset of session policy")

    filesystem = cast(
        FilesystemMode,
        _require_not_wider(
            field="filesystem",
            base=base.filesystem,
            requested=patch.filesystem,
            order=_FILESYSTEM_ORDER,
        ),
    )
    network = cast(
        NetworkMode,
        _require_not_wider(
            field="network", base=base.network, requested=patch.network, order=_NETWORK_ORDER
        ),
    )
    approval = _narrow_approval(base.approval, patch.approval)
    if network == "allowlist":
        network_allowlist = (
            base.network_allowlist if patch.network_allowlist is None else patch.network_allowlist
        )
        # An `unrestricted` base permits every host, so stepping down to `allowlist` may name any
        # host; only an `allowlist` base carries a set the patch has to stay inside.
        if base.network == "allowlist" and not set(network_allowlist).issubset(
            base.network_allowlist
        ):
            raise InvalidAgentRequest(
                "network_allowlist patch must be an exact subset of session policy"
            )
        if not network_allowlist:
            raise InvalidAgentRequest("allowlist policy patch must retain at least one entry")
    else:
        if patch.network_allowlist:
            raise InvalidAgentRequest(
                "network_allowlist patch must be empty when the result is not allowlist mode"
            )
        network_allowlist = ()
    required: list[UnsafeDimension] = []
    if filesystem == "full_access":
        required.append("filesystem_full_access")
    if network == "unrestricted":
        required.append("network_unrestricted")
    if approval == "allow":
        required.append("approval_allow")
    confirmation = UnsafeConfirmation(tuple(required)) if required else None
    return PermissionPolicy(
        filesystem=filesystem,
        network=network,
        approval=approval,
        network_allowlist=network_allowlist,
        allowed_tools=allowed,
        denied_tools=denied,
        environment=environment,
        unsafe_confirmation=confirmation,
    )


def tool_is_allowed(policy: PermissionPolicy, name: str) -> bool:
    """Deny wins; an empty allowlist permits no built-in tools."""
    if any(fnmatch.fnmatchcase(name, pattern) for pattern in policy.denied_tools):
        return False
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in policy.allowed_tools)
