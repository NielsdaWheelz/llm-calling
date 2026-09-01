from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
from llm_tools import (
    TOOL_FAMILY,
    WEB_READ_SPEC,
    WEB_SEARCH_SPEC,
    CapabilityProfile,
    Discoverable,
    FrozenToolPlan,
    HostTable,
    Native,
    ProfileId,
    RunLimits,
    ToolCatalog,
    ToolGrant,
    ToolId,
    ToolPlan,
    web_family,
)

import provider_runtime.tool_adapter as tool_adapter
from provider_runtime.agent_runtime.events import AgentToolUse
from provider_runtime.agent_runtime.tool_projection import (
    CanonicalMcpToolObservation,
    McpToolPublication,
    RejectedMcpToolObservation,
    lower_mcp_tools,
)
from provider_runtime.agent_runtime.types import CredentialRef
from provider_runtime.engines.anthropic_messages import _encode_request as encode_anthropic
from provider_runtime.engines.gemini_generate import _encode as encode_gemini
from provider_runtime.engines.openai_chat import _encode as encode_openai_chat
from provider_runtime.engines.openai_responses import _encode_request as encode_openai_responses
from provider_runtime.registry import _resolve as resolve
from provider_runtime.tool_adapter import (
    CanonicalToolCall,
    RejectedToolArguments,
    RejectedToolCall,
    ToolPublication,
    lower_tools,
)
from provider_runtime.types import (
    GenerateIntent,
    ProviderTarget,
    TextOutput,
    ToolCall,
    freeze_json_object,
)

_ANNOTATIONS = frozenset({"description", "examples", "title"})


def _catalog() -> ToolCatalog:
    return ToolCatalog.compose((web_family(), TOOL_FAMILY))


def _run_limits() -> RunLimits:
    return RunLimits(
        max_calls=8,
        max_external_attempts=8,
        max_input_bytes=64 * 1024,
        max_output_bytes=1024 * 1024,
        max_in_flight=2,
        max_elapsed_seconds=60,
    )


def _plan(
    exposure: Native | Discoverable | HostTable,
    *,
    web_search_max_input_bytes: int | None = None,
) -> FrozenToolPlan:
    catalog = _catalog()
    search_limits = (
        None
        if web_search_max_input_bytes is None
        else WEB_SEARCH_SPEC.limits.tightened(max_input_bytes=web_search_max_input_bytes)
    )
    grants = (
        ToolGrant(ToolId("tool.search"), None),
        ToolGrant(ToolId("tool.read"), None),
        ToolGrant(WEB_SEARCH_SPEC.id, search_limits),
        ToolGrant(WEB_READ_SPEC.id, None),
    )
    profile = CapabilityProfile(ProfileId("provider-test"), grants, _run_limits()).freeze(catalog)
    return ToolPlan(profile.id, exposure).freeze(catalog, profile)


def _semantic_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _semantic_projection(child)
            for key, child in value.items()
            if key not in _ANNOTATIONS
        }
    if isinstance(value, list | tuple):
        return [_semantic_projection(child) for child in value]
    return value


def _intent(row_ref: str, tools: tuple):
    row = resolve(row_ref)
    return row, GenerateIntent(
        target=ProviderTarget(provider=row.provider, model=row.model_id),
        messages=(),
        max_output_tokens=64,
        reasoning="none",
        tools=tools,
        tool_choice="auto",
        output=TextOutput(),
    )


def test_frozen_plans_lower_to_exact_immutable_native_tools_and_round_trip_names() -> None:
    assert tool_adapter.__all__ == [
        "CanonicalToolCall",
        "InvalidArgumentsReason",
        "PublishedTools",
        "RejectedToolArguments",
        "RejectedToolCall",
        "ToolCallResolution",
        "ToolPublication",
        "lower_tools",
    ]
    native = lower_tools(ToolPublication(plan=_plan(Native()), revealed_targets=()))

    assert tuple(tool.name for tool in native.tools) == (
        "tool__search",
        "tool__read",
        "web__search",
        "web__read",
    )
    assert native.tools[2].description == WEB_SEARCH_SPEC.documentation.text
    assert _semantic_projection(native.tools[2].parameters) == WEB_SEARCH_SPEC.input_schema.semantic

    assert isinstance(native.tools[2].parameters, dict)
    with pytest.raises(TypeError):
        native.tools[2].parameters["type"] = "array"
    properties = native.tools[2].parameters["properties"]
    assert isinstance(properties, dict)
    with pytest.raises(TypeError):
        properties["query"] = {}
    assert copy.deepcopy(native.tools[2].parameters) is native.tools[2].parameters

    decoded = native.decode_tool_call(
        ToolCall(
            id="provider-call-1",
            name="web__search",
            arguments={"nested": {"values": [1, {"present_null": None}]}, "query": "cats"},
        )
    )
    assert isinstance(decoded, CanonicalToolCall)
    assert decoded.provider_call_id == "provider-call-1"
    assert decoded.tool_id == WEB_SEARCH_SPEC.id
    assert json.loads(json.dumps(decoded.arguments)) == {
        "nested": {"values": [1, {"present_null": None}]},
        "query": "cats",
    }
    assert not hasattr(decoded, "wire_name")
    assert isinstance(decoded.arguments, dict)
    with pytest.raises(TypeError):
        decoded.arguments["query"] = "changed"

    rejected = native.decode_tool_call(
        ToolCall(id="provider-call-2", name="§" * 100, arguments={"secret": "discarded"})
    )
    assert isinstance(rejected, RejectedToolCall)
    assert rejected.provider_call_id == "provider-call-2"
    assert len(rejected.raw_name.encode("utf-8")) <= 64
    assert not hasattr(rejected, "arguments")

    invalid_arguments = native.decode_tool_call(
        ToolCall(id="provider-call-3", name="web__search", arguments={"value": float("nan")})
    )
    assert invalid_arguments == RejectedToolArguments(
        provider_call_id="provider-call-3",
        tool_id=WEB_SEARCH_SPEC.id,
        reason="InvalidJson",
    )
    assert not hasattr(invalid_arguments, "arguments")

    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(65):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    too_deep = native.decode_tool_call(
        ToolCall(id="provider-call-depth", name="web__search", arguments=nested)
    )
    assert too_deep == RejectedToolArguments(
        provider_call_id="provider-call-depth",
        tool_id=WEB_SEARCH_SPEC.id,
        reason="InvalidJson",
    )

    bounded_call = ToolCall(
        id="provider-call-4",
        name="web__search",
        arguments={"freshness_days": None, "query": "x" * 200},
    )
    assert isinstance(native.decode_tool_call(bounded_call), CanonicalToolCall)
    tightened = lower_tools(
        ToolPublication(
            plan=_plan(Native(), web_search_max_input_bytes=128),
            revealed_targets=(),
        )
    )
    oversized_arguments = tightened.decode_tool_call(bounded_call)
    assert oversized_arguments == RejectedToolArguments(
        provider_call_id="provider-call-4",
        tool_id=WEB_SEARCH_SPEC.id,
        reason="InputTooLarge",
    )
    assert not hasattr(oversized_arguments, "arguments")

    discoverable = lower_tools(
        ToolPublication(
            plan=_plan(
                Discoverable(
                    targets=(WEB_SEARCH_SPEC.id, WEB_READ_SPEC.id),
                    max_target_tools_published=1,
                )
            ),
            revealed_targets=(WEB_SEARCH_SPEC.id, WEB_READ_SPEC.id),
        )
    )
    assert tuple(tool.name for tool in discoverable.tools) == (
        "tool__search",
        "tool__read",
        "web__read",
    )
    with pytest.raises(ValueError, match="unique"):
        ToolPublication(
            plan=_plan(
                Discoverable(
                    targets=(WEB_SEARCH_SPEC.id, WEB_READ_SPEC.id),
                    max_target_tools_published=1,
                )
            ),
            revealed_targets=(WEB_SEARCH_SPEC.id, WEB_SEARCH_SPEC.id),
        )
    with pytest.raises(ValueError, match="outside"):
        lower_tools(
            ToolPublication(
                plan=_plan(
                    Discoverable(
                        targets=(WEB_SEARCH_SPEC.id, WEB_READ_SPEC.id),
                        max_target_tools_published=1,
                    )
                ),
                revealed_targets=(ToolId("other.unknown"),),
            )
        )

    with pytest.raises(ValueError, match="HostTable"):
        lower_tools(ToolPublication(plan=_plan(HostTable()), revealed_targets=()))


def test_provider_aliases_fail_closed_on_length_grammar_and_collision() -> None:
    plan = _plan(Native())
    duplicate = plan.profile.ordered_grants[0]
    duplicate_profile = replace(
        plan.profile,
        ordered_grants=(duplicate, duplicate),
    )
    with pytest.raises(ValueError, match="collision"):
        lower_tools(
            ToolPublication(
                plan=replace(plan, profile=duplicate_profile),
                revealed_targets=(),
            )
        )

    long_id = ToolId(f"web.{('a' * 61)}")
    long_spec = replace(WEB_SEARCH_SPEC, id=long_id)
    long_binding = replace(plan.catalog_view.binding(WEB_SEARCH_SPEC.id), spec=long_spec)
    long_grant = replace(
        plan.profile.ordered_grants[2],
        id=long_id,
        tool_contract_revision=long_spec.tool_contract_revision,
    )
    long_view = replace(
        plan.catalog_view,
        _specs={long_id: long_spec},
        _bindings={long_id: long_binding},
    )
    long_profile = replace(
        plan.profile,
        grants={long_id: long_grant},
        ordered_grants=(long_grant,),
    )
    with pytest.raises(ValueError, match="64"):
        lower_tools(
            ToolPublication(
                plan=replace(plan, profile=long_profile, catalog_view=long_view),
                revealed_targets=(),
            )
        )

    malformed_id = str.__new__(ToolId, "web.bad-name")
    malformed_spec = replace(WEB_SEARCH_SPEC, id=malformed_id)
    malformed_binding = replace(
        plan.catalog_view.binding(WEB_SEARCH_SPEC.id),
        spec=malformed_spec,
    )
    malformed_grant = replace(
        plan.profile.ordered_grants[2],
        id=malformed_id,
        tool_contract_revision=malformed_spec.tool_contract_revision,
    )
    malformed_view = replace(
        plan.catalog_view,
        _specs={malformed_id: malformed_spec},
        _bindings={malformed_id: malformed_binding},
    )
    malformed_profile = replace(
        plan.profile,
        grants={malformed_id: malformed_grant},
        ordered_grants=(malformed_grant,),
    )
    with pytest.raises(ValueError, match="canonical tool id"):
        lower_tools(
            ToolPublication(
                plan=replace(plan, profile=malformed_profile, catalog_view=malformed_view),
                revealed_targets=(),
            )
        )


def test_all_engine_encoders_preserve_the_frozen_portable_schema_semantics() -> None:
    published = lower_tools(ToolPublication(plan=_plan(Native()), revealed_targets=()))
    tool = published.tools[2]
    expected = WEB_SEARCH_SPEC.input_schema.semantic

    openai_row, openai_intent = _intent("openai:gpt-5.6-sol", (tool,))
    openai_tools = encode_openai_responses(openai_row, openai_intent).params["tools"]
    assert isinstance(openai_tools, list)
    openai = openai_tools[0]
    assert isinstance(openai, Mapping)

    anthropic_row, anthropic_intent = _intent("anthropic:claude-sonnet-5", (tool,))
    anthropic_tools = encode_anthropic(anthropic_row, anthropic_intent).params["tools"]
    assert isinstance(anthropic_tools, list)
    anthropic = anthropic_tools[0]
    assert isinstance(anthropic, Mapping)

    chat_row, chat_intent = _intent("moonshot:kimi-k3", (tool,))
    chat_tools = encode_openai_chat("moonshot", chat_row, chat_intent).body["tools"]
    assert isinstance(chat_tools, list)
    chat_definition = chat_tools[0]
    assert isinstance(chat_definition, Mapping)
    chat_function = chat_definition["function"]
    assert isinstance(chat_function, Mapping)

    gemini_row, gemini_intent = _intent("gemini:gemini-3.5-flash", (tool,))
    gemini_config = encode_gemini(gemini_row, gemini_intent).config
    assert gemini_config.tools is not None
    declarations = getattr(gemini_config.tools[0], "function_declarations", None)
    assert declarations is not None
    gemini = declarations[0]

    for engine, name, description, schema in (
        (
            "anthropic_messages",
            anthropic["name"],
            anthropic["description"],
            anthropic["input_schema"],
        ),
        (
            "gemini_generate",
            gemini.name,
            gemini.description,
            gemini.parameters_json_schema,
        ),
        (
            "openai_chat",
            chat_function["name"],
            chat_function["description"],
            chat_function["parameters"],
        ),
        (
            "openai_responses",
            openai["name"],
            openai["description"],
            openai["parameters"],
        ),
    ):
        assert name == "web__search", f"{engine} changed the provider-safe name: {name!r}"
        assert description == WEB_SEARCH_SPEC.documentation.text, (
            f"{engine} changed the frozen tool documentation: {description!r}"
        )
        wire_schema = json.loads(json.dumps(schema))
        assert _semantic_projection(wire_schema) == expected, (
            f"{engine} changed the portable semantic input schema: {schema!r}"
        )


def test_one_frozen_plan_lowers_to_exact_mcp_publication_and_observation() -> None:
    plan = _plan(Native())
    bearer = CredentialRef(
        kind="secret_reference",
        profile_key="personal",
        name="run-scoped-mcp-bearer",
    )
    published = lower_mcp_tools(
        McpToolPublication(
            plan=plan,
            server_name="nexus",
            url="https://nexus.example.test/private/mcp",
            bearer=bearer,
        )
    )

    assert published.server.name == "nexus"
    assert published.server.transport == "streamable_http"
    assert published.server.required is True
    assert published.server.allowed_tools == (
        "tool__search",
        "tool__read",
        "web__search",
        "web__read",
    )
    assert published.server.denied_tools == ()
    assert published.server.header_refs[0].name == "Authorization"
    assert published.server.header_refs[0].source is bearer

    observed = published.observe(
        AgentToolUse(
            tool_call_id="call-1",
            name="nexus/web__search",
            phase="started",
            payload=freeze_json_object({"query": "cats"}),
        )
    )
    assert observed == CanonicalMcpToolObservation(
        tool_call_id="call-1",
        tool_id=WEB_SEARCH_SPEC.id,
        phase="started",
        payload=freeze_json_object({"query": "cats"}),
        succeeded=None,
    )

    rejected = published.observe(
        AgentToolUse(
            tool_call_id="call-2",
            name="other/private_tool",
            phase="completed",
            payload=freeze_json_object({"secret": "discarded with the observation"}),
            succeeded=False,
        )
    )
    assert isinstance(rejected, RejectedMcpToolObservation)
    assert rejected.tool_call_id == "call-2"
    assert not hasattr(rejected, "payload")


def test_mcp_projection_uses_the_same_exposure_owner_as_function_publication() -> None:
    plan = _plan(
        Discoverable(
            targets=(WEB_SEARCH_SPEC.id, WEB_READ_SPEC.id),
            max_target_tools_published=1,
        )
    )
    revealed = (WEB_SEARCH_SPEC.id, WEB_READ_SPEC.id)
    function_names = tuple(
        tool.name
        for tool in lower_tools(ToolPublication(plan=plan, revealed_targets=revealed)).tools
    )
    mcp = lower_mcp_tools(
        McpToolPublication(
            plan=plan,
            server_name="nexus",
            url="https://nexus.example.test/private/mcp",
            bearer=CredentialRef(
                kind="secret_reference",
                profile_key="personal",
                name="run-bearer",
            ),
            revealed_targets=revealed,
        )
    )

    assert mcp.server.allowed_tools == function_names
    with pytest.raises(ValueError, match="unique"):
        McpToolPublication(
            plan=plan,
            server_name="nexus",
            url="https://nexus.example.test/private/mcp",
            bearer=CredentialRef(
                kind="secret_reference",
                profile_key="personal",
                name="run-bearer",
            ),
            revealed_targets=(WEB_SEARCH_SPEC.id, WEB_SEARCH_SPEC.id),
        )
    with pytest.raises(ValueError, match="HostTable"):
        lower_mcp_tools(
            McpToolPublication(
                plan=_plan(HostTable()),
                server_name="nexus",
                url="https://nexus.example.test/private/mcp",
                bearer=CredentialRef(
                    kind="secret_reference",
                    profile_key="personal",
                    name="run-bearer",
                ),
            )
        )
