from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast, get_args

import pytest

from provider_runtime.agent_runtime import types as types_module
from provider_runtime.agent_runtime.errors import InvalidAgentRequest
from provider_runtime.agent_runtime.events import AGENT_FAILURE_CAUSES, AgentFailureCause
from provider_runtime.agent_runtime.policy import PermissionPolicy
from provider_runtime.agent_runtime.types import (
    AGENT_ROUTES,
    AgentSessionRef,
    AgentSessionRequest,
    AgentTransport,
    ApprovalRequest,
    Backend,
    ClaudeNativeOptions,
    CodexNativeOptions,
    CredentialRef,
    EnvironmentReference,
    FileContent,
    ForkSession,
    FrozenJsonDict,
    HeaderReference,
    ImageContent,
    JsonSchemaAgentOutput,
    McpServerSpec,
    NewSession,
    ResumeSession,
    TextContent,
    TurnRequest,
    freeze_json_value,
    ref_from_json,
    ref_to_json,
    thaw_json_value,
)


def _credential() -> CredentialRef:
    return CredentialRef(kind="local_account", profile_key="personal")


def _ref() -> AgentSessionRef:
    return AgentSessionRef(
        schema_version="agent-session-ref.v1",
        backend="codex",
        transport="sdk",
        native_session_id="thread-123",
        profile_key="personal",
        state_root_fingerprint="a" * 64,
        cwd_fingerprint="b" * 64,
    )


def _request() -> AgentSessionRequest:
    return AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=_credential(),
        open=NewSession(),
        cwd="/workspace/repo",
        policy=PermissionPolicy(),
    )


def test_credential_references_never_accept_values_or_ambiguous_sources() -> None:
    assert _credential().name is None
    assert (
        CredentialRef(kind="api_key_environment", profile_key="api", name="OPENAI_API_KEY").name
        == "OPENAI_API_KEY"
    )

    with pytest.raises(InvalidAgentRequest, match="name must be absent"):
        CredentialRef(kind="local_account", profile_key="personal", name="secret-value")
    with pytest.raises(InvalidAgentRequest, match="name is required"):
        CredentialRef(kind="secret_reference", profile_key="vault")
    with pytest.raises(TypeError):
        CredentialRef(kind="local_account", profile_key="personal", value="secret")  # type: ignore[call-arg]


def test_agent_routes_are_closed_and_request_collections_are_owned_tuples() -> None:
    request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=_credential(),
        open=NewSession(),
        cwd="/workspace/repo",
        policy=PermissionPolicy(),
        system=(TextContent("system"),),
        additional_dirs=("/workspace/shared",),
    )

    assert request.system == (TextContent("system"),)
    assert request.additional_dirs == ("/workspace/shared",)
    with pytest.raises(FrozenInstanceError):
        request.cwd = "/other"  # type: ignore[misc]
    with pytest.raises(InvalidAgentRequest, match="unsupported backend/transport pair"):
        AgentSessionRequest(
            backend="codex",
            transport="wire",  # type: ignore[arg-type]
            auth=_credential(),
            open=NewSession(),
            cwd="/workspace/repo",
            policy=PermissionPolicy(),
        )
    with pytest.raises(InvalidAgentRequest, match="absolute"):
        AgentSessionRequest(
            backend="codex",
            transport="sdk",
            auth=_credential(),
            open=NewSession(),
            cwd="relative/repo",
            policy=PermissionPolicy(),
        )
    with pytest.raises(InvalidAgentRequest, match="additional_dirs contains duplicates"):
        AgentSessionRequest(
            backend="codex",
            transport="sdk",
            auth=_credential(),
            open=NewSession(),
            cwd="/workspace/repo",
            policy=PermissionPolicy(),
            additional_dirs=("/workspace/shared", "/workspace/shared"),
        )


def test_frozen_json_backing_storage_cannot_be_mutated_or_bypass_integer_bounds() -> None:
    value = FrozenJsonDict({"safe": 1})

    with pytest.raises(AttributeError):
        value._data = {"safe": 2}  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        value._FrozenJsonDict__items = (  # pyright: ignore[reportAttributeAccessIssue]
            ("safe", 2),
        )
    assert value == {"safe": 1}
    with pytest.raises(InvalidAgentRequest, match="signed 64 bits"):
        freeze_json_value(10**10000)


@pytest.mark.parametrize(
    ("backend", "transport"),
    (
        ("codex", "sdk"),
        ("claude", "sdk"),
    ),
)
def test_every_declared_backend_transport_pair_is_constructible(
    backend: Backend, transport: AgentTransport
) -> None:
    request = AgentSessionRequest(
        backend=backend,
        transport=transport,
        auth=_credential(),
        open=NewSession(),
        cwd="/workspace/repo",
        policy=PermissionPolicy(),
    )

    assert (request.backend, request.transport) == (backend, transport)


def test_file_content_and_turn_limits_reject_invalid_local_values() -> None:
    assert FileContent(path="/tmp/input.txt", size_bytes=4, media_type="text/plain").kind == "file"
    assert ImageContent(path="/tmp/input.png", size_bytes=4, media_type="image/png").kind == "image"
    with pytest.raises(InvalidAgentRequest, match="absolute"):
        FileContent(path="input.txt", size_bytes=4, media_type="text/plain")
    with pytest.raises(InvalidAgentRequest, match="positive"):
        TurnRequest(input=(TextContent("hello"),), timeout_seconds=0)
    with pytest.raises(InvalidAgentRequest, match="non-empty"):
        TurnRequest(input=())


def test_native_options_are_typed_preserved_and_backend_paired() -> None:
    native = CodexNativeOptions(web_search=False, builtin_tools="disabled")
    request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=_credential(),
        open=NewSession(),
        cwd="/workspace/repo",
        policy=PermissionPolicy(),
        native=native,
    )

    assert request.native is native
    with pytest.raises(InvalidAgentRequest, match="CodexNativeOptions"):
        AgentSessionRequest(
            backend="codex",
            transport="sdk",
            auth=_credential(),
            open=NewSession(),
            cwd="/workspace/repo",
            policy=PermissionPolicy(),
            native=ClaudeNativeOptions(include_partial_messages=True),
        )
    with pytest.raises(TypeError):
        CodexNativeOptions(untyped_option=True)  # type: ignore[call-arg]
    with pytest.raises(InvalidAgentRequest, match="builtin_tools"):
        CodexNativeOptions(builtin_tools="enabled")  # type: ignore[arg-type]
    with pytest.raises(InvalidAgentRequest, match="forbids web search"):
        CodexNativeOptions(web_search=True, builtin_tools="disabled")
    with pytest.raises(InvalidAgentRequest, match="unrestricted network"):
        AgentSessionRequest(
            backend="codex",
            transport="sdk",
            auth=_credential(),
            open=NewSession(),
            cwd="/workspace/repo",
            policy=PermissionPolicy(),
            native=CodexNativeOptions(web_search=True),
        )


def test_mcp_servers_accept_references_only_and_enforce_transport_shape() -> None:
    source = CredentialRef(kind="secret_reference", profile_key="personal", name="github-token")
    spec = McpServerSpec(
        name="github",
        transport="stdio",
        command="mcp-github",
        args=("--readonly",),
        environment_refs=(EnvironmentReference(name="GITHUB_TOKEN", source=source),),
    )
    assert spec.url is None

    with pytest.raises(InvalidAgentRequest, match="command and forbids url"):
        McpServerSpec(name="bad", transport="stdio", url="https://example.test/mcp")
    with pytest.raises(InvalidAgentRequest, match="command and forbids url"):
        McpServerSpec(name="bad-command", transport="stdio", command="mcp\x00server")
    with pytest.raises(InvalidAgentRequest, match="https"):
        McpServerSpec(name="bad", transport="streamable_http", url="http://example.test/mcp")
    with pytest.raises(InvalidAgentRequest, match="https"):
        McpServerSpec(
            name="bad-auth",
            transport="streamable_http",
            url="https://user:password@example.test/mcp#fragment",
        )
    with pytest.raises(InvalidAgentRequest):
        EnvironmentReference(name="TOKEN", source="raw-token")  # type: ignore[arg-type]

    http = McpServerSpec(
        name="remote",
        transport="streamable_http",
        url="https://example.test/mcp",
        header_refs=(HeaderReference(name="Authorization", source=source),),
    )
    assert http.header_refs[0].source is source
    assert McpServerSpec(
        name="repeat-args",
        transport="stdio",
        command="mcp",
        args=("--header", "x", "--header", "y"),
    ).args == ("--header", "x", "--header", "y")
    with pytest.raises(InvalidAgentRequest, match="duplicate"):
        McpServerSpec(
            name="duplicate-tools",
            transport="stdio",
            command="mcp",
            allowed_tools=("read", "read"),
        )
    with pytest.raises(InvalidAgentRequest, match="duplicate names"):
        AgentSessionRequest(
            backend="codex",
            transport="sdk",
            auth=_credential(),
            open=NewSession(),
            cwd="/workspace/repo",
            policy=PermissionPolicy(),
            mcp_servers=(spec, spec),
        )


def test_ref_json_is_strict_versioned_and_round_trips_as_plain_json() -> None:
    ref = _ref()
    encoded = ref_to_json(ref)

    assert ref_from_json(encoded) == ref
    assert json.loads(json.dumps(thaw_json_value(encoded)))["native_session_id"] == "thread-123"
    with pytest.raises(InvalidAgentRequest, match="schema_version"):
        ref_from_json({**encoded, "schema_version": "agent-session-ref.v2"})
    with pytest.raises(InvalidAgentRequest, match="unknown fields"):
        ref_from_json({**encoded, "raw_path": "/secret/repo"})


def test_json_values_are_recursively_immutable_finite_and_serializable() -> None:
    frozen = freeze_json_value({"outer": [{"token": "redacted"}], "count": 2})

    assert json.loads(json.dumps(thaw_json_value(frozen))) == {
        "outer": [{"token": "redacted"}],
        "count": 2,
    }
    with pytest.raises(TypeError):
        frozen["count"] = 3  # type: ignore[index]
    with pytest.raises(InvalidAgentRequest, match="finite"):
        freeze_json_value(float("nan"))
    with pytest.raises(InvalidAgentRequest, match="JSON-safe"):
        freeze_json_value({"bad": object()})


@pytest.mark.parametrize("root_kind", ["object", "array"])
def test_json_reference_cycles_are_typed_request_errors(root_kind: str) -> None:
    if root_kind == "object":
        value: object = {}
        assert isinstance(value, dict)
        value["self"] = value
    else:
        value = []
        assert isinstance(value, list)
        value.append(value)

    with pytest.raises(InvalidAgentRequest, match="reference cycle"):
        freeze_json_value(value)


def test_json_nesting_is_bounded_with_a_typed_request_error() -> None:
    value: object = "leaf"
    for _ in range(66):
        value = [value]

    with pytest.raises(InvalidAgentRequest, match="maximum JSON nesting depth"):
        freeze_json_value(value)


def test_freezing_a_nested_object_visits_every_node_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freezing is linear in the payload's size, not exponential in its depth.

    A nested object used to be frozen by the recursive walk and then frozen again by the
    `FrozenJsonDict` it was handed to, so the work doubled with every level: this 25-deep
    payload cost 2**25 visits. The counter aborts rather than hangs if that owner splits
    in two again.
    """
    depth = 25
    expected = 2 * depth + 2
    original = types_module._freeze_json_value
    visits = 0

    def counted(value: object, **keywords: Any) -> object:
        nonlocal visits
        visits += 1
        if visits > expected:
            raise AssertionError("the freezer re-walked descendants it had already frozen")
        return original(value, **keywords)

    monkeypatch.setattr(types_module, "_freeze_json_value", counted)
    payload: dict[str, object] = {"leaf": "bottom"}
    for _ in range(depth):
        payload = {"child": payload, "leaf": "level"}

    frozen = types_module.freeze_json_value(payload)

    assert visits == expected
    assert thaw_json_value(frozen) == payload


def test_direct_frozen_json_construction_cannot_smuggle_mutable_or_nonfinite_values() -> None:
    source = {"items": [{"value": "one"}]}
    frozen = FrozenJsonDict(source)
    source["items"].append({"value": "two"})

    assert frozen == {"items": ({"value": "one"},)}
    nested = cast(tuple[FrozenJsonDict, ...], frozen["items"])
    with pytest.raises(TypeError):
        nested[0]["value"] = "changed"  # pyright: ignore[reportIndexIssue]
    with pytest.raises(InvalidAgentRequest, match="finite"):
        FrozenJsonDict({"bad": float("nan")})
    with pytest.raises(InvalidAgentRequest, match="JSON-safe"):
        FrozenJsonDict({"bad": object()})

    approval = ApprovalRequest(operation="command", summary="run checks", native_payload=frozen)
    assert approval.native_payload == frozen


def test_session_open_variants_keep_the_complete_ref() -> None:
    ref = _ref()
    assert ResumeSession(ref).ref is ref
    assert ForkSession(ref).ref is ref


def test_equal_frozen_json_objects_hash_equal_regardless_of_key_order() -> None:
    first = FrozenJsonDict({"a": 1, "b": {"c": ("x",)}})
    second = FrozenJsonDict({"b": {"c": ("x",)}, "a": 1})

    assert first == second
    assert hash(first) == hash(second)
    assert second in {first}
    assert len({first, second}) == 1


def test_absolute_paths_are_validated_lexically_without_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    request = AgentSessionRequest(
        backend="codex",
        transport="sdk",
        auth=_credential(),
        open=NewSession(),
        cwd=str(link),
        policy=PermissionPolicy(),
    )

    assert request.cwd == str(link)
    for rejected in ("/workspace/../etc", "/workspace/./repo", "/workspace//repo", "/workspace/"):
        with pytest.raises(InvalidAgentRequest, match="normalized absolute path"):
            AgentSessionRequest(
                backend="codex",
                transport="sdk",
                auth=_credential(),
                open=NewSession(),
                cwd=rejected,
                policy=PermissionPolicy(),
            )


def test_closed_route_and_failure_vocabularies_have_exactly_one_owner() -> None:
    assert set(AGENT_FAILURE_CAUSES) == set(get_args(AgentFailureCause.__value__))
    assert len(AGENT_FAILURE_CAUSES) == len(set(AGENT_FAILURE_CAUSES))
    assert AGENT_ROUTES == frozenset({("codex", "sdk"), ("claude", "sdk")})
    # A transport the type admits but no route reaches is exactly the stale state that let
    # unreachable branches survive here before: every declared member must be routable, and
    # every routed member must be declared.
    assert {transport for _backend, transport in AGENT_ROUTES} == set(
        get_args(AgentTransport.__value__)
    )
    assert {backend for backend, _transport in AGENT_ROUTES} == set(get_args(Backend.__value__))


def test_json_schema_output_freezes_a_plain_json_schema_mapping() -> None:
    source: dict[str, object] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    output = JsonSchemaAgentOutput(name="Answer", schema=source)
    source["properties"] = {}

    assert isinstance(output.schema, FrozenJsonDict), (
        "the schema must be frozen at construction so events stay immutable"
    )
    assert thaw_json_value(output.schema) == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    with pytest.raises(InvalidAgentRequest, match="JSON object"):
        JsonSchemaAgentOutput(name="Answer", schema="not a schema")  # type: ignore[arg-type]
    with pytest.raises(InvalidAgentRequest, match="non-empty"):
        JsonSchemaAgentOutput(name="", schema={})
