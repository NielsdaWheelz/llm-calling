from __future__ import annotations

from typing import Any, cast

import pytest

from provider_runtime.agent_runtime.errors import InvalidAgentRequest
from provider_runtime.agent_runtime.policy import (
    PermissionPolicy,
    PermissionPolicyPatch,
    UnsafeConfirmation,
    UnsafeDimension,
    narrow_policy,
    tool_is_allowed,
)


def test_policy_defaults_fail_closed() -> None:
    policy = PermissionPolicy()

    assert policy.filesystem == "read_only"
    assert policy.network == "disabled"
    assert policy.approval == "deny"
    assert not tool_is_allowed(policy, "shell")


@pytest.mark.parametrize(
    ("field", "value", "dimension"),
    (
        ("filesystem", "full_access", "filesystem_full_access"),
        ("network", "unrestricted", "network_unrestricted"),
        ("approval", "allow", "approval_allow"),
    ),
)
def test_each_unsafe_dimension_requires_exact_confirmation(
    field: str, value: str, dimension: str
) -> None:
    invalid: dict[str, Any] = {field: value}
    with pytest.raises(InvalidAgentRequest, match=dimension):
        PermissionPolicy(**invalid)

    confirmed: dict[str, Any] = {
        field: value,
        "unsafe_confirmation": UnsafeConfirmation(acknowledged=(cast(UnsafeDimension, dimension),)),
    }
    policy = PermissionPolicy(**confirmed)
    assert getattr(policy, field) == value


def test_confirmation_cannot_be_reused_for_unrequested_access() -> None:
    with pytest.raises(InvalidAgentRequest, match="exactly"):
        PermissionPolicy(unsafe_confirmation=UnsafeConfirmation(acknowledged=("approval_allow",)))


def test_policy_patch_narrows_by_enum_and_exact_set_operations() -> None:
    base = PermissionPolicy(
        filesystem="workspace_write",
        network="allowlist",
        approval="ask",
        network_allowlist=("example.test", "api.example.test"),
        allowed_tools=("shell:*", "read"),
        denied_tools=("shell:rm",),
        environment=("LANG", "PATH"),
    )
    narrowed = narrow_policy(
        base,
        PermissionPolicyPatch(
            filesystem="read_only",
            network="disabled",
            approval="deny",
            allowed_tools=("read",),
            denied_tools=("shell:rm", "read:private"),
            environment=("LANG",),
        ),
    )

    assert narrowed.allowed_tools == ("read",)
    assert narrowed.network_allowlist == ()
    assert narrowed.denied_tools == ("shell:rm", "read:private")
    assert narrowed.environment == ("LANG",)
    with pytest.raises(InvalidAgentRequest, match="cannot widen filesystem"):
        narrow_policy(narrowed, PermissionPolicyPatch(filesystem="workspace_write"))
    with pytest.raises(InvalidAgentRequest, match="exact subset"):
        narrow_policy(base, PermissionPolicyPatch(allowed_tools=("write",)))
    with pytest.raises(InvalidAgentRequest, match="exact superset"):
        narrow_policy(base, PermissionPolicyPatch(denied_tools=()))


def test_provider_review_and_caller_review_are_incomparable_narrowing_modes() -> None:
    provider_review = PermissionPolicy(approval="provider_review")
    caller_review = PermissionPolicy(approval="ask")

    assert narrow_policy(provider_review, PermissionPolicyPatch(approval="deny")).approval == "deny"
    assert narrow_policy(caller_review, PermissionPolicyPatch(approval="deny")).approval == "deny"
    with pytest.raises(InvalidAgentRequest, match="change approval reviewer"):
        narrow_policy(provider_review, PermissionPolicyPatch(approval="ask"))
    with pytest.raises(InvalidAgentRequest, match="change approval reviewer"):
        narrow_policy(caller_review, PermissionPolicyPatch(approval="provider_review"))

    allow = PermissionPolicy(
        approval="allow",
        unsafe_confirmation=UnsafeConfirmation(acknowledged=("approval_allow",)),
    )
    assert (
        narrow_policy(allow, PermissionPolicyPatch(approval="provider_review")).approval
        == "provider_review"
    )


def test_network_allowlist_is_explicit_and_narrows_by_exact_entries() -> None:
    with pytest.raises(InvalidAgentRequest, match="requires"):
        PermissionPolicy(network="allowlist")
    base = PermissionPolicy(network="allowlist", network_allowlist=("a.example", "b.example"))

    narrowed = narrow_policy(base, PermissionPolicyPatch(network_allowlist=("a.example",)))
    assert narrowed.network_allowlist == ("a.example",)
    with pytest.raises(InvalidAgentRequest, match="exact subset"):
        narrow_policy(base, PermissionPolicyPatch(network_allowlist=("new.example",)))


@pytest.mark.parametrize(
    "host",
    ("*", "Example.test", "example.test.", "127.0.0.1", "localhost", "service.local"),
)
def test_network_allowlist_rejects_ambiguous_or_local_host_syntax(host: str) -> None:
    with pytest.raises(InvalidAgentRequest, match="network_allowlist"):
        PermissionPolicy(network="allowlist", network_allowlist=(host,))


def test_deny_wins_when_tool_patterns_overlap() -> None:
    policy = PermissionPolicy(
        allowed_tools=("shell:*",),
        denied_tools=("shell:rm",),
    )

    assert tool_is_allowed(policy, "shell:cat")
    assert not tool_is_allowed(policy, "shell:rm")


def test_unrestricted_network_can_be_narrowed_to_an_explicit_allowlist() -> None:
    base = PermissionPolicy(
        network="unrestricted",
        unsafe_confirmation=UnsafeConfirmation(("network_unrestricted",)),
    )

    narrowed = narrow_policy(
        base,
        PermissionPolicyPatch(network="allowlist", network_allowlist=("api.example.test",)),
    )

    assert narrowed.network == "allowlist"
    assert narrowed.network_allowlist == ("api.example.test",)
    assert narrowed.unsafe_confirmation is None


def test_an_allowlist_base_still_only_narrows_by_removing_entries() -> None:
    base = PermissionPolicy(
        network="allowlist", network_allowlist=("api.example.test", "mcp.example.test")
    )

    assert narrow_policy(
        base, PermissionPolicyPatch(network_allowlist=("api.example.test",))
    ).network_allowlist == ("api.example.test",)
    with pytest.raises(InvalidAgentRequest, match="exact subset"):
        narrow_policy(base, PermissionPolicyPatch(network_allowlist=("other.example.test",)))
    assert narrow_policy(base, PermissionPolicyPatch(network="disabled")).network == "disabled"
