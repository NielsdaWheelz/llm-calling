from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from provider_runtime.agent_runtime.capabilities import (
    AgentCapabilities,
    AgentCapabilityScope,
    require_discovery_operation,
    validate_auth_capability,
    validate_session_capabilities,
    validate_turn_capabilities,
    validate_turn_policy,
)
from provider_runtime.agent_runtime.errors import (
    AgentRuntimeDefect,
    InvalidAgentRequest,
    UnsupportedCapability,
)
from provider_runtime.agent_runtime.policy import (
    PermissionPolicy,
    PermissionPolicyPatch,
    UnsafeConfirmation,
)
from provider_runtime.agent_runtime.types import (
    AgentSessionRequest,
    CodexNativeOptions,
    CredentialRef,
    EnvironmentReference,
    FileContent,
    JsonSchemaAgentOutput,
    McpServerSpec,
    NewSession,
    ReasoningSpec,
    TextContent,
    TurnRequest,
)
from provider_runtime.schema import parse_canonical_schema

AUTH = CredentialRef(kind="local_account", profile_key="personal")
SCOPE = AgentCapabilityScope(backend="codex", transport="sdk", auth=AUTH)


def _capabilities(**changes: Any) -> AgentCapabilities:
    values: dict[str, Any] = {
        "scope": SCOPE,
        "session_operations": ("new", "resume", "fork"),
        "models": ("gpt-native",),
        "reasoning_efforts": ("high",),
        "content_kinds": ("text",),
        "filesystem_modes": ("read_only",),
        "network_modes": ("disabled",),
        "approval_modes": ("deny",),
    }
    values.update(changes)
    return AgentCapabilities(**values)


def _request(**changes: Any) -> AgentSessionRequest:
    values: dict[str, Any] = {
        "backend": "codex",
        "transport": "sdk",
        "auth": AUTH,
        "open": NewSession(),
        "cwd": "/workspace/repo",
        "policy": PermissionPolicy(),
        "model": "gpt-native",
        "reasoning": ReasoningSpec(effort="high"),
    }
    values.update(changes)
    return AgentSessionRequest(**values)


def test_discovered_values_are_validated_before_launch() -> None:
    validate_session_capabilities(_request(), _capabilities())

    with pytest.raises(UnsupportedCapability, match="model"):
        validate_session_capabilities(_request(model="other"), _capabilities())
    with pytest.raises(UnsupportedCapability, match="reasoning effort"):
        validate_session_capabilities(
            _request(reasoning=ReasoningSpec(effort="max")), _capabilities()
        )


def test_undiscoverable_model_values_pass_through() -> None:
    validate_session_capabilities(
        _request(model="installed-native-value"),
        _capabilities(models=None, reasoning_efforts=None),
    )


def test_structured_output_and_instruction_roles_are_capability_checked() -> None:
    schema = parse_canonical_schema(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    with pytest.raises(UnsupportedCapability, match="structured output"):
        validate_session_capabilities(
            _request(output=JsonSchemaAgentOutput(name="answer", schema=schema)),
            _capabilities(),
        )
    with pytest.raises(UnsupportedCapability, match="system"):
        validate_session_capabilities(_request(system=(TextContent("system"),)), _capabilities())


def test_turn_overrides_and_max_turns_are_capability_checked() -> None:
    turn = TurnRequest(input=(TextContent("hello"),), model="gpt-native", max_turns=2)
    with pytest.raises(UnsupportedCapability, match="model override"):
        validate_turn_capabilities(turn, _capabilities(), session_model=None)

    validate_turn_capabilities(
        turn,
        _capabilities(turn_overrides=("model",), max_turns=True),
        session_model=None,
    )


def test_capability_scope_must_match_request_scope() -> None:
    other = AgentCapabilityScope(
        backend="claude", transport="sdk", auth=CredentialRef(kind="local_account", profile_key="x")
    )
    with pytest.raises(UnsupportedCapability, match="scope"):
        validate_session_capabilities(_request(), AgentCapabilities(scope=other))
    with pytest.raises(InvalidAgentRequest, match="backend/transport"):
        AgentCapabilityScope(backend="codex", transport="wire", auth=AUTH)  # type: ignore[arg-type]


def test_discovery_constraints_and_override_persistence_are_explicit_facts() -> None:
    capabilities = _capabilities(
        discovery_operations=("list", "read", "turn_history", "item_history"),
        max_turns=True,
        max_turns_limit=8,
        max_timeout_seconds=120.0,
        max_context_tokens=200_000,
        max_output_tokens=16_000,
        turn_overrides=("model", "reasoning"),
        persistent_turn_overrides=("model",),
        native_extension_version="codex.v1",
        native_option_names=("web_search",),
        cwd_scopes_sessions=True,
        reports_auth_identity=True,
    )

    require_discovery_operation(capabilities, "read")
    assert capabilities.max_turns_limit == 8
    assert capabilities.max_timeout_seconds == 120.0
    assert capabilities.persistent_turn_overrides == ("model",)
    with pytest.raises(UnsupportedCapability, match="discovery"):
        require_discovery_operation(_capabilities(), "list")
    # A capability table our own adapter builds is our invariant, so a broken one is a defect,
    # never a product-facing "this route does not support it".
    with pytest.raises(AgentRuntimeDefect, match="subset"):
        _capabilities(persistent_turn_overrides=("model",))
    with pytest.raises(AgentRuntimeDefect, match="unknown value"):
        _capabilities(filesystem_modes=("readonly",))
    with pytest.raises(AgentRuntimeDefect, match="duplicate"):
        _capabilities(models=("gpt-native", "gpt-native"))


def test_attachments_mcp_auth_tool_controls_and_native_options_are_checked() -> None:
    source = CredentialRef(kind="secret_reference", profile_key="personal", name="mcp-secret")
    mcp = McpServerSpec(
        name="local",
        transport="stdio",
        command="mcp-local",
        environment_refs=(EnvironmentReference(name="TOKEN", source=source),),
    )
    request = _request(
        developer=(FileContent("/tmp/context.txt", 3, "text/plain"),),
        mcp_servers=(mcp,),
        native=CodexNativeOptions(web_search=False),
        policy=PermissionPolicy(
            filesystem="full_access",
            network="unrestricted",
            allowed_tools=("read",),
            unsafe_confirmation=UnsafeConfirmation(
                ("filesystem_full_access", "network_unrestricted")
            ),
        ),
    )
    supported = _capabilities(
        content_kinds=("text", "file"),
        attachment_kinds=("file",),
        session_instruction_roles=("developer",),
        mcp_transports=("stdio",),
        mcp_auth_forms=("environment_reference",),
        tool_controls=True,
        filesystem_modes=("read_only", "full_access"),
        network_modes=("disabled", "unrestricted"),
        native_extension_version="codex.v1",
        native_option_names=("web_search",),
    )

    validate_session_capabilities(request, supported)
    with pytest.raises(UnsupportedCapability, match="attachment"):
        validate_session_capabilities(request, _capabilities(content_kinds=("text", "file")))
    with pytest.raises(UnsupportedCapability, match="native option"):
        validate_session_capabilities(
            _request(native=CodexNativeOptions(web_search=False)), _capabilities()
        )


def test_remote_mcp_must_be_inside_the_effective_network_policy() -> None:
    remote = McpServerSpec(
        name="remote",
        transport="streamable_http",
        url="https://mcp.example.test/rpc",
    )
    capabilities = _capabilities(
        mcp_transports=("streamable_http",),
        network_modes=("disabled", "allowlist"),
        network_allowlist=True,
    )

    with pytest.raises(UnsupportedCapability, match="network policy"):
        validate_session_capabilities(
            _request(mcp_servers=(remote,), policy=PermissionPolicy()), capabilities
        )
    with pytest.raises(UnsupportedCapability, match="network allowlist"):
        validate_session_capabilities(
            _request(
                mcp_servers=(remote,),
                policy=PermissionPolicy(
                    network="allowlist", network_allowlist=("other.example.test",)
                ),
            ),
            capabilities,
        )
    validate_session_capabilities(
        _request(
            mcp_servers=(remote,),
            policy=PermissionPolicy(network="allowlist", network_allowlist=("mcp.example.test",)),
        ),
        capabilities,
    )


def test_turn_limits_use_discovered_numeric_constraints() -> None:
    turn = TurnRequest(input=(TextContent("hello"),), max_turns=9, timeout_seconds=121)
    capabilities = _capabilities(
        max_turns=True,
        max_turns_limit=8,
        timeouts=True,
        max_timeout_seconds=120,
    )
    with pytest.raises(UnsupportedCapability, match="max_turns exceeds"):
        validate_turn_capabilities(turn, capabilities, session_model=None)

    with pytest.raises(UnsupportedCapability, match="timeout exceeds"):
        validate_turn_capabilities(
            TurnRequest(input=(TextContent("hello"),), timeout_seconds=121),
            capabilities,
            session_model=None,
        )


def test_user_role_reasoning_summary_and_additional_dirs_are_checked() -> None:
    with pytest.raises(UnsupportedCapability, match="user content"):
        validate_turn_capabilities(
            TurnRequest(input=(TextContent("hello"),)),
            _capabilities(content_roles=()),
            session_model=None,
        )
    with pytest.raises(UnsupportedCapability, match="reasoning summary"):
        validate_turn_capabilities(
            TurnRequest(
                input=(TextContent("hello"),),
                reasoning=ReasoningSpec(effort="high", summary="concise"),
            ),
            _capabilities(turn_overrides=("reasoning",)),
            session_model=None,
        )
    validate_turn_capabilities(
        TurnRequest(
            input=(TextContent("hello"),),
            reasoning=ReasoningSpec(effort="high", summary="concise"),
        ),
        _capabilities(turn_overrides=("reasoning",), reasoning_summaries=("concise",)),
        session_model=None,
    )
    request = _request(additional_dirs=("/workspace/shared",))
    with pytest.raises(UnsupportedCapability, match="additional working"):
        validate_session_capabilities(request, _capabilities())
    validate_session_capabilities(request, _capabilities(additional_dirs=True))


def test_effective_turn_policy_is_narrowed_then_capability_checked() -> None:
    base = PermissionPolicy(
        network="allowlist", network_allowlist=("api.example.test", "mcp.example.test")
    )
    capabilities = _capabilities(network_modes=("disabled", "allowlist"), network_allowlist=True)

    effective = validate_turn_policy(
        base,
        PermissionPolicyPatch(network_allowlist=("api.example.test",)),
        capabilities,
    )
    assert effective.network_allowlist == ("api.example.test",)
    with pytest.raises(UnsupportedCapability, match="exact network allowlist"):
        validate_turn_policy(
            base, PermissionPolicyPatch(), _capabilities(network_modes=("allowlist",))
        )


def test_transport_without_tool_controls_requires_explicit_unrestricted_sentinel() -> None:
    uncontrolled = _capabilities(
        builtin_tool_families=("command", "file_write"),
        tool_controls=False,
    )

    with pytest.raises(UnsupportedCapability, match="cannot disable or filter"):
        validate_session_capabilities(_request(), uncontrolled)

    validate_session_capabilities(
        _request(policy=PermissionPolicy(allowed_tools=("*",))),
        uncontrolled,
    )

    with pytest.raises(UnsupportedCapability, match="cannot disable or filter"):
        validate_session_capabilities(
            _request(
                policy=PermissionPolicy(
                    allowed_tools=("*",),
                    denied_tools=("command",),
                )
            ),
            uncontrolled,
        )

    no_builtin_tools = replace(uncontrolled, builtin_tool_families=())
    validate_session_capabilities(_request(), no_builtin_tools)


def test_builtin_tool_names_carry_the_exact_names_a_policy_is_written_in() -> None:
    """A route with tool controls can publish the native spellings its policy demands.

    `builtin_tool_families` is a normalized cross-backend vocabulary; `allowed_tools` and
    `denied_tools` are native by contract. Without this bridge a consumer reading
    `tool_controls=True` has no discoverable source for the names it must actually send,
    and every one of them ends up hard-coding a vendor table (as the live release matrix
    had to). Reading the table and writing the policy from it must therefore validate.
    """
    capabilities = _capabilities(
        builtin_tool_families=("file_read", "command"),
        builtin_tool_names=(("file_read", ("Read", "Glob")), ("command", ("Bash",))),
        tool_controls=True,
    )

    published = dict(capabilities.builtin_tool_names)
    assert published["file_read"] == ("Read", "Glob")

    validate_session_capabilities(
        _request(policy=PermissionPolicy(allowed_tools=published["file_read"])),
        capabilities,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param(
            {"builtin_tool_names": [("file_read", ("Read",))]}, "must be a tuple", id="list"
        ),
        pytest.param({"builtin_tool_names": (("file_read",),)}, "must be pairs", id="not-a-pair"),
        pytest.param(
            {"builtin_tool_names": (("web_search", ("WebSearch",)),)},
            "unreported tool family",
            id="family-not-reported",
        ),
        pytest.param(
            {"builtin_tool_names": (("file_reed", ("Read",)),)},
            "unreported tool family",
            id="family-outside-the-vocabulary",
        ),
        pytest.param(
            {"builtin_tool_names": (("file_read", ("Read", "Read")),)},
            "duplicate entries",
            id="duplicate-name",
        ),
        pytest.param(
            {"builtin_tool_names": (("file_read", ()),)},
            "at least one native tool name",
            id="empty-name-list",
        ),
        pytest.param(
            {"builtin_tool_names": (("file_read", ("Read",)), ("file_read", ("Glob",)))},
            "duplicate families",
            id="duplicate-family",
        ),
        pytest.param(
            {"builtin_tool_names": (("file_read", ("Read",)),), "tool_controls": False},
            "requires tool allow/deny controls",
            id="names-without-controls",
        ),
    ],
)
def test_a_native_tool_name_table_the_transport_cannot_honour_is_a_defect(
    changes: dict[str, Any], message: str
) -> None:
    values: dict[str, Any] = {
        "builtin_tool_families": ("file_read", "command"),
        "tool_controls": True,
    }
    values.update(changes)
    with pytest.raises(AgentRuntimeDefect, match=message):
        _capabilities(**values)


def test_per_turn_model_and_effort_are_checked_against_the_same_enumeration() -> None:
    capabilities = _capabilities(
        models=("gpt-x", "gpt-y"),
        reasoning_efforts=("low", "high"),
        model_reasoning_efforts=(("gpt-x", ("low",)), ("gpt-y", ("low", "high"))),
        turn_overrides=("model", "reasoning"),
    )
    request = _request(model="gpt-x", reasoning=ReasoningSpec(effort="high"))
    valid = TurnRequest(
        input=(TextContent("hello"),), model="gpt-y", reasoning=ReasoningSpec(effort="high")
    )
    invalid = TurnRequest(
        input=(TextContent("hello"),), model="gpt-x", reasoning=ReasoningSpec(effort="high")
    )

    with pytest.raises(UnsupportedCapability, match="unsupported for model 'gpt-x'"):
        validate_session_capabilities(request, capabilities)
    validate_turn_capabilities(valid, capabilities, session_model=None)
    with pytest.raises(UnsupportedCapability, match="unsupported for model 'gpt-x'"):
        validate_turn_capabilities(invalid, capabilities, session_model=None)


def test_named_auth_requires_a_transport_that_reports_its_effective_identity() -> None:
    named = CredentialRef(kind="api_key_environment", profile_key="api", name="OPENAI_API_KEY")
    scope = AgentCapabilityScope(backend="codex", transport="sdk", auth=named)
    request = _request(auth=named)

    with pytest.raises(UnsupportedCapability, match="reports its effective"):
        validate_session_capabilities(request, _capabilities(scope=scope))
    validate_session_capabilities(request, _capabilities(scope=scope, reports_auth_identity=True))
    validate_auth_capability(AUTH, _capabilities())


def test_session_metadata_dimensions_are_reported_capability_facts() -> None:
    capabilities = _capabilities(session_name_metadata=True)

    assert capabilities.session_name_metadata is True
    assert capabilities.session_archive_metadata is False
    assert capabilities.session_tag_metadata is False
    with pytest.raises(AgentRuntimeDefect, match="session_name_metadata must be bool"):
        _capabilities(session_name_metadata="yes")


def test_builtin_tool_families_are_one_cross_backend_vocabulary() -> None:
    """Vendor tool names make the field unprogrammable: nothing can branch on it portably.

    A caller asking "can this transport edit files for me" cannot be expected to know that
    Codex spells that `apply_patch` and Claude Code spells it `Write`/`Edit`.
    """
    for family in ("file_read", "file_write", "command", "web_fetch", "web_search"):
        assert _capabilities(builtin_tool_families=(family,)).builtin_tool_families == (family,)
    for vendor_name in ("Read", "Bash", "shell", "file", "file_change", "WebSearch"):
        with pytest.raises(AgentRuntimeDefect, match="builtin_tool_families"):
            _capabilities(builtin_tool_families=(vendor_name,))


def test_stdio_mcp_support_is_reported_with_the_policy_it_requires() -> None:
    """`validate_mcp_network_policy` only admits stdio under full_access + unrestricted.

    A table advertising stdio without both modes advertises a transport that no policy the
    same table accepts can ever reach, which a caller can only discover by trying it.
    """
    with pytest.raises(AgentRuntimeDefect, match="stdio MCP requires"):
        _capabilities(mcp_transports=("stdio",))
    with pytest.raises(AgentRuntimeDefect, match="stdio MCP requires"):
        _capabilities(
            mcp_transports=("stdio",),
            filesystem_modes=("read_only", "full_access"),
        )
    reachable = _capabilities(
        mcp_transports=("stdio",),
        filesystem_modes=("read_only", "full_access"),
        network_modes=("disabled", "unrestricted"),
    )
    assert reachable.mcp_transports == ("stdio",)


def test_an_effort_only_turn_override_is_checked_against_the_session_model() -> None:
    """A turn that overrides only the effort still runs on the session's model.

    Checking such a turn against the union of every model's efforts accepts an effort the
    session's own model does not support, and the rejection then arrives natively mid-turn.
    """
    capabilities = _capabilities(
        models=("narrow-model", "wide-model"),
        reasoning_efforts=("low", "high"),
        model_reasoning_efforts=(("narrow-model", ("low",)), ("wide-model", ("low", "high"))),
        turn_overrides=("model", "reasoning"),
    )
    effort_only = TurnRequest(input=(TextContent("hello"),), reasoning=ReasoningSpec(effort="high"))

    validate_turn_capabilities(effort_only, capabilities, session_model="wide-model")
    with pytest.raises(UnsupportedCapability, match="unsupported for model 'narrow-model'"):
        validate_turn_capabilities(effort_only, capabilities, session_model="narrow-model")
    # An explicit per-turn model still wins over the session's.
    validate_turn_capabilities(
        TurnRequest(
            input=(TextContent("hello"),),
            model="wide-model",
            reasoning=ReasoningSpec(effort="high"),
        ),
        capabilities,
        session_model="narrow-model",
    )


def test_reports_effective_effort_is_part_of_the_capability_table() -> None:
    """Where a backend never reports the effort it ran with, the clamp is unobservable."""
    assert _capabilities().reports_effective_effort is False
    assert _capabilities(reports_effective_effort=True).reports_effective_effort is True
    with pytest.raises(AgentRuntimeDefect, match="reports_effective_effort must be bool"):
        _capabilities(reports_effective_effort="yes")
