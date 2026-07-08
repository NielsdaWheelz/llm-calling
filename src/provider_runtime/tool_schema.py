"""Provider-specific JSON Schema normalization for tool and output schemas."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

from provider_runtime.catalog import ModelCapability
from provider_runtime.types import ModelCall, ToolSpec

OPENAI_STRICT_SCHEMA_PROVIDERS = {"openai", "openrouter", "cloudflare"}
_COMPOSITION_KEYS = ("oneOf", "allOf", "not", "if", "then", "else")


@dataclass(frozen=True)
class SchemaIssue:
    path: str
    message: str


class StrictSchemaError(ValueError):
    def __init__(self, issues: list[SchemaIssue]):
        self.issues = tuple(issues)
        message = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        super().__init__(message)


def normalize_model_call_schemas(
    call: ModelCall,
    capabilities: ModelCapability,
) -> ModelCall:
    """Normalize schemas for providers that require OpenAI strict JSON Schema."""
    if capabilities.provider not in OPENAI_STRICT_SCHEMA_PROVIDERS:
        return call

    tools = tuple(normalize_tool_spec_for_provider(tool, capabilities) for tool in call.tools)
    structured_output = call.structured_output
    if structured_output is not None and structured_output.strict:
        structured_output = replace(
            structured_output,
            schema=normalize_openai_strict_schema(structured_output.schema),
        )
    if tools == call.tools and structured_output == call.structured_output:
        return call
    return replace(call, tools=tools, structured_output=structured_output)


def normalize_tool_spec_for_provider(
    tool: ToolSpec,
    capabilities: ModelCapability,
) -> ToolSpec:
    if capabilities.provider not in OPENAI_STRICT_SCHEMA_PROVIDERS or not tool.strict:
        return tool
    return replace(tool, parameters=normalize_openai_strict_schema(tool.parameters))


def normalize_openai_strict_schema(schema: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(schema)
    issues = validate_openai_strict_schema(normalized)
    if issues:
        raise StrictSchemaError(issues)
    _normalize_schema_node(normalized, "$", root=True)
    issues = validate_openai_strict_schema(normalized)
    if issues:
        raise StrictSchemaError(issues)
    return normalized


def validate_openai_strict_schema(schema: dict[str, object]) -> list[SchemaIssue]:
    issues: list[SchemaIssue] = []
    if not isinstance(schema, dict):
        return [SchemaIssue("$", "schema must be an object")]
    _collect_unstrictifiable_issues(schema, "$", issues, root=True)
    return issues


def _normalize_schema_node(node: Any, path: str, *, root: bool = False) -> None:
    if not isinstance(node, dict):
        return

    nullable = node.pop("nullable", None) is True
    if nullable:
        _add_null_type(node)

    for key in _COMPOSITION_KEYS:
        if key in node:
            return
    if "anyOf" in node and isinstance(node["anyOf"], list):
        for index, branch in enumerate(node["anyOf"]):
            _normalize_schema_node(branch, f"{path}.anyOf[{index}]")

    for defs_key in ("$defs", "definitions"):
        defs = node.get(defs_key)
        if isinstance(defs, dict):
            for name, definition in defs.items():
                _normalize_schema_node(definition, f"{path}.{defs_key}.{name}")

    properties = node.get("properties")
    is_object = _is_object_schema(node)
    original_required = _required_set(node, path)
    if is_object:
        node["additionalProperties"] = False
        if isinstance(properties, dict):
            node["required"] = list(properties.keys())
            for name, prop_schema in properties.items():
                prop_path = f"{path}.properties.{name}"
                if name not in original_required:
                    _add_null_type(prop_schema)
                _normalize_schema_node(prop_schema, prop_path)
        else:
            node.setdefault("required", [])

    items = node.get("items")
    if isinstance(items, dict):
        _normalize_schema_node(items, f"{path}.items")


def _collect_unstrictifiable_issues(
    node: Any,
    path: str,
    issues: list[SchemaIssue],
    *,
    root: bool = False,
) -> None:
    if not isinstance(node, dict):
        return

    if root and not _is_object_schema(node):
        issues.append(SchemaIssue(path, "root schema must be an object"))
    if root and "anyOf" in node:
        issues.append(SchemaIssue(path, "root schema must be an object, not anyOf"))
    for key in _COMPOSITION_KEYS:
        if key in node:
            issues.append(SchemaIssue(f"{path}.{key}", f"{key} is not supported in strict schemas"))
    if "patternProperties" in node:
        issues.append(SchemaIssue(f"{path}.patternProperties", "patternProperties is not supported"))

    additional = node.get("additionalProperties")
    if isinstance(additional, dict):
        issues.append(
            SchemaIssue(
                f"{path}.additionalProperties",
                "map-like additionalProperties schemas cannot be strictified",
            )
        )
    elif additional is True:
        issues.append(
            SchemaIssue(
                f"{path}.additionalProperties",
                "additionalProperties=true cannot be strictified",
            )
        )
    elif additional not in (None, False):
        issues.append(
            SchemaIssue(
                f"{path}.additionalProperties",
                "additionalProperties must be false or omitted",
            )
        )

    required = node.get("required")
    if required is not None and (
        not isinstance(required, list) or any(not isinstance(item, str) for item in required)
    ):
        issues.append(SchemaIssue(f"{path}.required", "required must be an array of strings"))

    items = node.get("items")
    if isinstance(items, list):
        issues.append(SchemaIssue(f"{path}.items", "tuple validation cannot be strictified"))

    if "anyOf" in node:
        any_of = node["anyOf"]
        if not isinstance(any_of, list):
            issues.append(SchemaIssue(f"{path}.anyOf", "anyOf must be an array"))
        else:
            for index, branch in enumerate(any_of):
                _collect_unstrictifiable_issues(branch, f"{path}.anyOf[{index}]", issues)

    properties = node.get("properties")
    if isinstance(properties, dict):
        for name, prop_schema in properties.items():
            _collect_unstrictifiable_issues(prop_schema, f"{path}.properties.{name}", issues)

    if isinstance(items, dict):
        _collect_unstrictifiable_issues(items, f"{path}.items", issues)

    for defs_key in ("$defs", "definitions"):
        defs = node.get(defs_key)
        if isinstance(defs, dict):
            for name, definition in defs.items():
                _collect_unstrictifiable_issues(definition, f"{path}.{defs_key}.{name}", issues)


def _is_object_schema(node: dict[str, Any]) -> bool:
    schema_type = node.get("type")
    if schema_type == "object":
        return True
    if isinstance(schema_type, list) and "object" in schema_type:
        return True
    return "properties" in node


def _required_set(node: dict[str, Any], path: str) -> set[str]:
    required = node.get("required")
    if required is None:
        return set()
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        raise StrictSchemaError([SchemaIssue(f"{path}.required", "required must be an array of strings")])
    return set(required)


def _add_null_type(node: Any) -> None:
    if not isinstance(node, dict):
        return
    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        if not any(isinstance(branch, dict) and branch.get("type") == "null" for branch in any_of):
            any_of.append({"type": "null"})
        return
    schema_type = node.get("type")
    if schema_type is None:
        if "properties" in node:
            node["type"] = ["object", "null"]
            return
        if "items" in node:
            node["type"] = ["array", "null"]
            return
        node["type"] = ["null"]
        return
    if isinstance(schema_type, str):
        if schema_type != "null":
            node["type"] = [schema_type, "null"]
        return
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            node["type"] = [*schema_type, "null"]
