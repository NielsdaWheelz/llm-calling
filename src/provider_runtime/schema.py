"""Canonical JSON-Schema subset (spec §5).

Immutable schema values, a pure parser/validator, a deterministic serializer,
and a canonical byte encoding used for schema fingerprints and cache-affinity
framing. This module owns the node values; ``provider_runtime.types``
references :class:`CanonicalJsonSchema` type-only (CanonicalTool /
StrictJsonOutput) and the package surface re-exports everything from here.

The subset uses JSON Schema 2020-12 meanings but admits only this closed
structural subset:

- the document root is an object schema and may carry one root ``$defs`` map;
- an object node has ``type: "object"``, finite ``properties``, ``required``
  equal to exactly every property name, and explicit
  ``additionalProperties: false``;
- an array node has ``type: "array"`` and one homogeneous ``items`` schema;
- a scalar node has ``type`` in string/number/integer/boolean and may have one
  non-empty, type-compatible ``enum``;
- ``{"type": "null"}`` is the null node, legal only inside a nullable union;
- a semantically optional value is expressed only as
  ``anyOf: [<non-null schema>, {"type": "null"}]``; no other union is valid;
- ``$ref`` may target only an acyclic ``#/$defs/<name>`` definition and may
  have no sibling keys;
- ``title`` and ``description`` are the only annotations, permitted on object,
  array, scalar, and nullable-union nodes.

Everything else raises :class:`SchemaViolation` carrying a JSON-pointer-ish
path. The parser never rewrites trusted input; the only accepted freedom is
the nullable-union branch order (either authored order parses to the same
order-normalized value) and the ``required`` list order (set-equality against
the property names).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, assert_never, cast

from .errors import SchemaViolation
from .types import Absent, Presence, Present

__all__ = [
    "ArrayNode",
    "CanonicalJsonSchema",
    "EnumValue",
    "Node",
    "NullNode",
    "NullableUnion",
    "ObjectNode",
    "Ref",
    "ScalarNode",
    "ScalarType",
    "canonical_schema_bytes",
    "parse_canonical_schema",
    "to_json_schema",
]

type ScalarType = Literal["string", "number", "integer", "boolean"]
type EnumValue = str | int | float | bool
type Node = ObjectNode | ArrayNode | ScalarNode | NullNode | NullableUnion | Ref

_SCALAR_TYPES: tuple[ScalarType, ...] = ("string", "number", "integer", "boolean")
# `Absent` is an empty frozen value, so one shared instance is every instance; the public
# node constructors read their unauthored-annotation default from here.
_ABSENT: Absent = Absent()

_NULL_ONLY_IN_UNION = (
    'the null schema {"type": "null"} is legal only as one branch of a nullable anyOf union'
)


def _validate_enum_values(
    scalar_type: ScalarType, values: tuple[object, ...], context: str
) -> tuple[EnumValue, ...]:
    """Enforce non-empty, type-compatible enum values for ``scalar_type``.

    Type compatibility with the single declared scalar type is also the
    homogeneity rule: ``string``/``integer``/``boolean`` admit exactly ``str``/
    ``int``/``bool`` values, and ``number`` admits ``int`` and ``float``
    (both are the one JSON number type). ``bool`` never satisfies ``integer``
    or ``number``; non-finite floats are rejected (not representable in JSON).
    """
    if len(values) == 0:
        raise SchemaViolation(f"{context}: enum must be non-empty")
    for index, value in enumerate(values):
        match scalar_type:
            case "string":
                compatible = type(value) is str
            case "integer":
                compatible = type(value) is int
            case "boolean":
                compatible = type(value) is bool
            case "number":
                if type(value) is float and not math.isfinite(value):
                    raise SchemaViolation(
                        f"{context}: enum value at index {index} ({value!r}) is a non-finite"
                        " number and cannot be represented in JSON"
                    )
                compatible = type(value) is int or type(value) is float
            case _:
                assert_never(scalar_type)
        if not compatible:
            raise SchemaViolation(
                f"{context}: enum value at index {index} ({value!r}, JSON-incompatible type"
                f" {type(value).__name__!r}) does not match scalar type {scalar_type!r}"
            )
    return cast(tuple[EnumValue, ...], values)  # each element proven compatible above


def _frozen_node_map(value: Mapping[str, Node], field: str) -> Mapping[str, Node]:
    """Snapshot a caller-supplied node mapping into a read-only view of our own copy.

    Both mapping-valued fields in this module are validated once, at construction, and
    then read again much later — a fingerprint at plan time, a wire document at turn
    time. Storing the caller's live dict would leave the accepted value aliased to a
    mapping the caller can still mutate, so a schema :func:`_validate_ref_graph` already
    accepted could grow an unvalidated property (or an unresolvable ``$ref``) before it
    reaches a provider. Copy first, then expose the copy read-only.
    """
    if not isinstance(value, Mapping):
        raise SchemaViolation(
            f"{field}: must be a mapping of name to schema node, got {type(value).__name__}"
        )
    snapshot: dict[str, Node] = {}
    for key, node in value.items():
        if type(key) is not str:
            raise SchemaViolation(f"{field}: mapping keys must be strings, got {key!r}")
        snapshot[key] = node
    return MappingProxyType(snapshot)


@dataclass(frozen=True, slots=True)
class NullNode:
    """The null schema ``{"type": "null"}``; only ever a nullable-union branch."""


@dataclass(frozen=True, slots=True)
class Ref:
    """A ``#/$defs/<name>`` reference. No annotations: ``$ref`` admits no siblings."""

    name: str


@dataclass(frozen=True, slots=True)
class ScalarNode:
    type: ScalarType
    enum: Presence[tuple[EnumValue, ...]] = Absent()
    title: Presence[str] = Absent()
    description: Presence[str] = Absent()

    def __post_init__(self) -> None:
        if isinstance(self.enum, Present):
            _validate_enum_values(self.type, self.enum.value, f"ScalarNode(type={self.type!r})")


@dataclass(frozen=True, slots=True)
class _ObjectNodeFields:
    """Field carrier for :class:`ObjectNode`; never constructed directly.

    A frozen dataclass cannot replace a field value in ``__post_init__`` without the
    attribute-assignment bypass ``tests/test_negative_gates.py`` bans as hidden mutation.
    Splitting the fields out lets the public class own the conversion in its own
    ``__init__`` and the stdlib-generated one do the assignment, with no bypass anywhere
    in our own source.
    """

    properties: Mapping[str, Node]
    title: Presence[str] = _ABSENT
    description: Presence[str] = _ABSENT


class ObjectNode(_ObjectNodeFields):
    """Closed object: ``required`` == all property names and
    ``additionalProperties: false`` are implied by construction, not stored.

    ``properties`` is snapshotted at construction (see :func:`_frozen_node_map`): the
    stored mapping is this class's own read-only copy, never the caller's live dict.
    """

    __slots__ = ()

    def __init__(
        self,
        properties: Mapping[str, Node],
        title: Presence[str] = _ABSENT,
        description: Presence[str] = _ABSENT,
    ) -> None:
        super().__init__(_frozen_node_map(properties, "ObjectNode.properties"), title, description)


@dataclass(frozen=True, slots=True)
class ArrayNode:
    items: Node
    title: Presence[str] = Absent()
    description: Presence[str] = Absent()


@dataclass(frozen=True, slots=True)
class NullableUnion:
    """``anyOf: [non_null, {"type": "null"}]``. The non-null branch may not be
    the null node or another union (directly, or via ``$ref`` — the resolved
    form is enforced by :class:`CanonicalJsonSchema`)."""

    non_null: Node
    title: Presence[str] = Absent()
    description: Presence[str] = Absent()

    def __post_init__(self) -> None:
        if isinstance(self.non_null, NullNode | NullableUnion):
            raise SchemaViolation(
                "NullableUnion.non_null must be a non-null, non-union node; got"
                f" {type(self.non_null).__name__}"
            )


def _walk(node: Node, path: str) -> Iterator[tuple[str, Node]]:
    yield path, node
    match node:
        case ObjectNode(properties=properties):
            for name, child in properties.items():
                yield from _walk(child, f"{path}/properties/{name}")
        case ArrayNode(items=items):
            yield from _walk(items, f"{path}/items")
        case NullableUnion(non_null=non_null):
            yield from _walk(non_null, f"{path}/anyOf/0")
        case ScalarNode() | NullNode() | Ref():
            pass
        case _:
            assert_never(node)


def _validate_ref_graph(root: ObjectNode, defs: Mapping[str, Node]) -> None:
    """Enforce: every ref resolves, no misplaced null node, the def graph is
    acyclic, and no nullable union's non-null branch resolves (through ref
    chains) to null/union."""
    subtrees: list[tuple[str, Node]] = [("#", root)]
    subtrees.extend((f"#/$defs/{name}", node) for name, node in defs.items())

    for prefix, subtree in subtrees:
        for path, node in _walk(subtree, prefix):
            if isinstance(node, NullNode):
                # NullableUnion never stores its null branch, so any walked
                # NullNode sits at a property/items/def position — illegal.
                raise SchemaViolation(f"{path}: {_NULL_ONLY_IN_UNION}")
            if isinstance(node, Ref) and node.name not in defs:
                raise SchemaViolation(
                    f"{path}: $ref targets undefined definition '#/$defs/{node.name}'"
                )

    edges: dict[str, tuple[str, ...]] = {
        name: tuple(n.name for _, n in _walk(node, "") if isinstance(n, Ref))
        for name, node in defs.items()
    }
    state: dict[str, int] = dict.fromkeys(edges, 0)  # 0 unvisited, 1 in progress, 2 done

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if state[name] == 2:
            return
        if state[name] == 1:
            cycle = trail[trail.index(name) :] + (name,)
            rendered = " -> ".join(f"#/$defs/{n}" for n in cycle)
            raise SchemaViolation(f"#/$defs/{name}: recursive $ref cycle: {rendered}")
        state[name] = 1
        for target in edges[name]:
            visit(target, trail + (name,))
        state[name] = 2

    for name in edges:
        visit(name, ())

    for prefix, subtree in subtrees:
        for path, node in _walk(subtree, prefix):
            if not isinstance(node, NullableUnion):
                continue
            resolved = node.non_null
            hops: list[str] = []
            while isinstance(resolved, Ref):
                hops.append(resolved.name)
                resolved = defs[resolved.name]
            if isinstance(resolved, NullNode | NullableUnion):
                chain = " -> ".join(f"#/$defs/{n}" for n in hops)
                raise SchemaViolation(
                    f"{path}: the non-null branch of a nullable union resolves to"
                    f" {type(resolved).__name__} via {chain}; unions may not nest and null is"
                    " legal only as the union's null branch"
                )


@dataclass(frozen=True, slots=True)
class _CanonicalJsonSchemaFields:
    """Field carrier for :class:`CanonicalJsonSchema`; see :class:`_ObjectNodeFields`."""

    root: ObjectNode
    defs: Mapping[str, Node]

    def __post_init__(self) -> None:
        _validate_ref_graph(self.root, self.defs)


class CanonicalJsonSchema(_CanonicalJsonSchemaFields):
    """A validated schema document: an object root plus its ``$defs`` map.

    ``defs`` is snapshotted at construction, so the ref graph ``__post_init__`` accepts is
    the graph every later reader — fingerprint, serializer, adapter — sees.
    """

    __slots__ = ()

    def __init__(self, root: ObjectNode, defs: Mapping[str, Node]) -> None:
        super().__init__(root, _frozen_node_map(defs, "CanonicalJsonSchema.defs"))


# --- parser -----------------------------------------------------------------

_UNION_HINT = 'only the nullable union anyOf: [<non-null schema>, {"type": "null"}] is supported'
_CONSTRAINT_HINT = (
    "value constraints live in the surface's domain validator after decode, not in the schema"
)
_KEYWORD_HINTS: Mapping[str, str] = {
    "nullable": 'express nullability as anyOf: [<schema>, {"type": "null"}]',
    "default": "defaults are not part of the canonical subset",
    "oneOf": _UNION_HINT,
    "allOf": _UNION_HINT,
    "not": _UNION_HINT,
    "if": _UNION_HINT,
    "then": _UNION_HINT,
    "else": _UNION_HINT,
    "$defs": "'$defs' is legal only at the document root",
    "definitions": "use a single root '$defs' map",
    "patternProperties": "objects are closed: fixed properties only",
    "propertyNames": "objects are closed: fixed properties only",
    "unevaluatedProperties": "objects are closed: fixed properties only",
    "prefixItems": "tuple arrays are not allowed; arrays take one homogeneous 'items' schema",
    "additionalItems": "tuple arrays are not allowed; arrays take one homogeneous 'items' schema",
    "contains": _CONSTRAINT_HINT,
    "minItems": _CONSTRAINT_HINT,
    "maxItems": _CONSTRAINT_HINT,
    "uniqueItems": _CONSTRAINT_HINT,
    "minLength": _CONSTRAINT_HINT,
    "maxLength": _CONSTRAINT_HINT,
    "pattern": _CONSTRAINT_HINT,
    "format": _CONSTRAINT_HINT,
    "minimum": _CONSTRAINT_HINT,
    "maximum": _CONSTRAINT_HINT,
    "exclusiveMinimum": _CONSTRAINT_HINT,
    "exclusiveMaximum": _CONSTRAINT_HINT,
    "multipleOf": _CONSTRAINT_HINT,
    "minProperties": _CONSTRAINT_HINT,
    "maxProperties": _CONSTRAINT_HINT,
    "dependentRequired": _CONSTRAINT_HINT,
    "dependentSchemas": _CONSTRAINT_HINT,
    "const": "single-value constraints are not in the subset; use a one-element 'enum'",
}


def _reject_keyword(path: str, key: str) -> SchemaViolation:
    hint = _KEYWORD_HINTS.get(key)
    suffix = f" ({hint})" if hint is not None else " (outside the canonical subset)"
    return SchemaViolation(f"{path}: unsupported keyword {key!r}{suffix}")


def _require_string_keys(value: Mapping[str, object], path: str) -> None:
    for key in value:
        if not isinstance(key, str):
            raise SchemaViolation(f"{path}: mapping keys must be strings, got {key!r}")


def _check_allowed_keys(value: Mapping[str, object], allowed: frozenset[str], path: str) -> None:
    for key in value:
        if key not in allowed:
            raise _reject_keyword(path, key)


def _as_sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return None


def _parse_annotation(value: Mapping[str, object], key: str, path: str) -> Presence[str]:
    if key not in value:
        return Absent()
    raw = value[key]
    if not isinstance(raw, str):
        raise SchemaViolation(f"{path}/{key}: annotation must be a string, got {raw!r}")
    return Present(raw)


def _is_null_literal(entry: object) -> bool:
    return isinstance(entry, Mapping) and set(entry.keys()) == {"type"} and entry["type"] == "null"


_OBJECT_KEYS = frozenset(
    {"type", "properties", "required", "additionalProperties", "title", "description"}
)
_ARRAY_KEYS = frozenset({"type", "items", "title", "description"})
_SCALAR_KEYS = frozenset({"type", "enum", "title", "description"})
_UNION_KEYS = frozenset({"anyOf", "title", "description"})


def _parse_object(
    value: Mapping[str, object], path: str, def_names: frozenset[str], allowed: frozenset[str]
) -> ObjectNode:
    _check_allowed_keys(value, allowed, path)
    if "properties" not in value:
        raise SchemaViolation(f"{path}: object schema requires 'properties'")
    properties_raw = value["properties"]
    if not isinstance(properties_raw, Mapping):
        raise SchemaViolation(
            f"{path}/properties: must be a mapping of property name to schema,"
            f" got {type(properties_raw).__name__}"
        )
    _require_string_keys(properties_raw, f"{path}/properties")

    if "required" not in value:
        raise SchemaViolation(
            f"{path}: object schema must list 'required' with exactly every property name"
        )
    required_raw = _as_sequence(value["required"])
    if required_raw is None:
        raise SchemaViolation(f"{path}/required: must be an array of property names")
    required: list[str] = []
    for entry in required_raw:
        if not isinstance(entry, str):
            raise SchemaViolation(f"{path}/required: entries must be strings, got {entry!r}")
        required.append(entry)
    if len(required) != len(set(required)):
        duplicates = sorted({name for name in required if required.count(name) > 1})
        raise SchemaViolation(f"{path}/required: duplicate entries {duplicates}")
    property_names = {str(name) for name in properties_raw}
    if set(required) != property_names:
        missing = sorted(property_names - set(required))
        extra = sorted(set(required) - property_names)
        raise SchemaViolation(
            f"{path}/required: must equal exactly every property name"
            f" (missing {missing}, extra {extra})"
        )

    if "additionalProperties" not in value:
        raise SchemaViolation(
            f"{path}: object schema must set 'additionalProperties' to false explicitly"
        )
    additional = value["additionalProperties"]
    if additional is not False:
        if isinstance(additional, Mapping):
            raise SchemaViolation(
                f"{path}/additionalProperties: schema-valued additionalProperties is not"
                " allowed; objects are closed (must be exactly false)"
            )
        raise SchemaViolation(
            f"{path}/additionalProperties: must be exactly false, got {additional!r}"
        )

    properties: dict[str, Node] = {}
    for name, child in properties_raw.items():
        properties[str(name)] = _parse_node(child, f"{path}/properties/{name}", def_names)
    return ObjectNode(
        properties=properties,
        title=_parse_annotation(value, "title", path),
        description=_parse_annotation(value, "description", path),
    )


def _parse_array(value: Mapping[str, object], path: str, def_names: frozenset[str]) -> ArrayNode:
    _check_allowed_keys(value, _ARRAY_KEYS, path)
    if "items" not in value:
        raise SchemaViolation(f"{path}: array schema requires one homogeneous 'items' schema")
    items_raw = value["items"]
    if _as_sequence(items_raw) is not None:
        raise SchemaViolation(
            f"{path}/items: tuple 'items' are not allowed; arrays take one homogeneous schema"
        )
    return ArrayNode(
        items=_parse_node(items_raw, f"{path}/items", def_names),
        title=_parse_annotation(value, "title", path),
        description=_parse_annotation(value, "description", path),
    )


def _parse_scalar(value: Mapping[str, object], scalar_type: ScalarType, path: str) -> ScalarNode:
    _check_allowed_keys(value, _SCALAR_KEYS, path)
    enum: Presence[tuple[EnumValue, ...]] = Absent()
    if "enum" in value:
        enum_raw = _as_sequence(value["enum"])
        if enum_raw is None:
            raise SchemaViolation(f"{path}/enum: must be an array of values")
        enum = Present(_validate_enum_values(scalar_type, tuple(enum_raw), f"{path}/enum"))
    return ScalarNode(
        type=scalar_type,
        enum=enum,
        title=_parse_annotation(value, "title", path),
        description=_parse_annotation(value, "description", path),
    )


def _parse_union(
    value: Mapping[str, object], path: str, def_names: frozenset[str]
) -> NullableUnion:
    _check_allowed_keys(value, _UNION_KEYS, path)
    entries = _as_sequence(value["anyOf"])
    if entries is None:
        raise SchemaViolation(f"{path}/anyOf: must be an array of schemas")
    if len(entries) != 2:
        raise SchemaViolation(
            f"{path}/anyOf: {_UNION_HINT}; got {len(entries)} entries instead of 2"
        )
    null_flags = [_is_null_literal(entry) for entry in entries]
    if null_flags.count(True) == 0:
        raise SchemaViolation(f"{path}/anyOf: arbitrary anyOf is not allowed; {_UNION_HINT}")
    if null_flags.count(True) == 2:
        raise SchemaViolation(
            f"{path}/anyOf: nullable union must pair exactly one non-null schema with the"
            " null schema"
        )
    index = null_flags.index(False)
    non_null = _parse_node(entries[index], f"{path}/anyOf/{index}", def_names)
    if isinstance(non_null, NullableUnion):
        # Pre-checked (rather than relying on the constructor) so the
        # violation carries the JSON path of the offending branch.
        raise SchemaViolation(f"{path}/anyOf/{index}: nullable unions may not nest")
    return NullableUnion(
        non_null=non_null,
        title=_parse_annotation(value, "title", path),
        description=_parse_annotation(value, "description", path),
    )


def _parse_node(
    value: object,
    path: str,
    def_names: frozenset[str],
    *,
    root_extra_keys: frozenset[str] = frozenset(),
) -> Node:
    if not isinstance(value, Mapping):
        raise SchemaViolation(f"{path}: schema must be a JSON object, got {type(value).__name__}")
    _require_string_keys(value, path)

    if "$ref" in value:
        siblings = sorted(set(value.keys()) - {"$ref"})
        if siblings:
            raise SchemaViolation(f"{path}: $ref must have no sibling keys, found {siblings}")
        ref_raw = value["$ref"]
        if not isinstance(ref_raw, str):
            raise SchemaViolation(f"{path}/$ref: must be a string, got {ref_raw!r}")
        prefix = "#/$defs/"
        name = ref_raw.removeprefix(prefix)
        if name == ref_raw or not name or "/" in name or "~" in name:
            raise SchemaViolation(
                f"{path}/$ref: only local '#/$defs/<name>' references are allowed, got {ref_raw!r}"
            )
        if name not in def_names:
            raise SchemaViolation(f"{path}/$ref: targets undefined definition '#/$defs/{name}'")
        return Ref(name=name)

    if "anyOf" in value:
        return _parse_union(value, path, def_names)

    if "type" not in value:
        raise SchemaViolation(
            f"{path}: schema node must declare 'type', '$ref', or a nullable 'anyOf'"
        )
    type_raw = value["type"]
    if _as_sequence(type_raw) is not None:
        raise SchemaViolation(
            f"{path}/type: union types are not allowed; express nullability as"
            f' anyOf: [<schema>, {{"type": "null"}}], got {type_raw!r}'
        )
    if not isinstance(type_raw, str):
        raise SchemaViolation(f"{path}/type: must be a string, got {type_raw!r}")
    if type_raw == "object":
        return _parse_object(value, path, def_names, _OBJECT_KEYS | root_extra_keys)
    if type_raw == "array":
        return _parse_array(value, path, def_names)
    if type_raw in _SCALAR_TYPES:
        return _parse_scalar(value, cast(ScalarType, type_raw), path)
    if type_raw == "null":
        if set(value.keys()) != {"type"}:
            raise SchemaViolation(
                f'{path}: the null schema must be exactly {{"type": "null"}} with no other'
                " keys (annotations are not allowed on the null node)"
            )
        raise SchemaViolation(f"{path}: {_NULL_ONLY_IN_UNION}")
    raise SchemaViolation(f"{path}/type: unsupported type {type_raw!r}")


def parse_canonical_schema(raw: Mapping[str, object]) -> CanonicalJsonSchema:
    """Parse and validate ``raw`` into an immutable :class:`CanonicalJsonSchema`.

    Pure: never mutates or rewrites ``raw``. Raises :class:`SchemaViolation`
    with a JSON-pointer-ish path on the first subset violation.
    """
    if not isinstance(raw, Mapping):
        raise SchemaViolation(f"#: schema document must be a JSON object, got {type(raw).__name__}")
    _require_string_keys(raw, "#")

    defs: dict[str, Node] = {}
    def_names: frozenset[str] = frozenset()
    if "$defs" in raw:
        defs_raw = raw["$defs"]
        if not isinstance(defs_raw, Mapping):
            raise SchemaViolation(
                f"#/$defs: must be a mapping of definition name to schema,"
                f" got {type(defs_raw).__name__}"
            )
        _require_string_keys(defs_raw, "#/$defs")
        for name in defs_raw:
            name_str = str(name)
            if not name_str or "/" in name_str or "~" in name_str:
                raise SchemaViolation(
                    f"#/$defs: definition name {name_str!r} must be non-empty and must not"
                    " contain '/' or '~'"
                )
        def_names = frozenset(str(name) for name in defs_raw)
        for name, def_raw in defs_raw.items():
            # _parse_node never returns NullNode (a bare {"type": "null"} raises
            # with the only-inside-anyOf message), so defs cannot hold one.
            defs[str(name)] = _parse_node(def_raw, f"#/$defs/{name}", def_names)

    if raw.get("type") != "object":
        # Root-shape gate before node dispatch so a $ref/anyOf/scalar/typeless
        # root reports the root rule, not a nested-node message.
        raise SchemaViolation('#: document root must be an object schema (type: "object")')
    root = _parse_node(raw, "#", def_names, root_extra_keys=frozenset({"$defs"}))
    if not isinstance(root, ObjectNode):  # unreachable: type=="object" parses to ObjectNode
        raise SchemaViolation('#: document root must be an object schema (type: "object")')
    return CanonicalJsonSchema(root=root, defs=defs)


# --- serializer -------------------------------------------------------------


def _node_to_json(
    node: Node,
    inline: Mapping[str, Node] | None,
    include_annotations: bool,
    sort_required: bool,
) -> dict[str, object]:
    out: dict[str, object]
    match node:
        case ObjectNode(properties=properties):
            names = list(properties.keys())
            out = {
                "type": "object",
                "properties": {
                    name: _node_to_json(child, inline, include_annotations, sort_required)
                    for name, child in properties.items()
                },
                "required": sorted(names) if sort_required else names,
                "additionalProperties": False,
            }
        case ArrayNode(items=items):
            out = {
                "type": "array",
                "items": _node_to_json(items, inline, include_annotations, sort_required),
            }
        case ScalarNode(type=scalar_type, enum=enum):
            out = {"type": scalar_type}
            if isinstance(enum, Present):
                out["enum"] = list(enum.value)
        case NullNode():
            return {"type": "null"}
        case NullableUnion(non_null=non_null):
            out = {
                "anyOf": [
                    _node_to_json(non_null, inline, include_annotations, sort_required),
                    {"type": "null"},
                ]
            }
        case Ref(name=name):
            if inline is None:
                return {"$ref": f"#/$defs/{name}"}
            return _node_to_json(inline[name], inline, include_annotations, sort_required)
        case _:
            assert_never(node)
    if include_annotations and not isinstance(node, NullNode | Ref):
        if isinstance(node.title, Present):
            out["title"] = node.title.value
        if isinstance(node.description, Present):
            out["description"] = node.description.value
    return out


def _to_json_schema(
    schema: CanonicalJsonSchema,
    *,
    inline_defs: bool,
    include_annotations: bool,
    sort_required: bool,
) -> dict[str, object]:
    inline = schema.defs if inline_defs else None
    doc = _node_to_json(schema.root, inline, include_annotations, sort_required)
    if not inline_defs and schema.defs:
        doc["$defs"] = {
            name: _node_to_json(node, None, include_annotations, sort_required)
            for name, node in schema.defs.items()
        }
    return doc


def to_json_schema(
    schema: CanonicalJsonSchema, *, inline_defs: bool, include_annotations: bool
) -> dict[str, object]:
    """Serialize back to a JSON Schema 2020-12 document (deterministically).

    Object nodes emit ``properties``, ``required`` (every property name, in
    property order), and ``additionalProperties: false``; nullable unions emit
    ``anyOf: [<non-null>, {"type": "null"}]`` in that normalized order. With
    ``inline_defs`` each ``$ref`` is replaced by its (recursively inlined)
    definition and ``$defs`` is omitted; otherwise refs serialize as
    ``#/$defs/<name>`` and a non-empty ``$defs`` map is emitted at the root.
    Round-trip law: ``parse_canonical_schema(to_json_schema(s,
    inline_defs=False, include_annotations=True)) == s``.
    """
    return _to_json_schema(
        schema,
        inline_defs=inline_defs,
        include_annotations=include_annotations,
        sort_required=False,
    )


def canonical_schema_bytes(schema: CanonicalJsonSchema) -> bytes:
    """Deterministic canonical encoding for fingerprints and cache-affinity framing.

    Encoding: the full serialized form (``inline_defs=False``,
    ``include_annotations=True`` — annotations reach providers, so they are
    fingerprint-relevant) dumped as JSON with sorted object keys,
    ``separators=(",", ":")``, ``ensure_ascii=False``, UTF-8 encoded. To make
    the bytes a function of the schema *value* (dataclass equality ignores
    mapping insertion order), ``required`` arrays are emitted sorted here —
    unlike :func:`to_json_schema`, which preserves property order. Semantic
    order (``enum`` values, ``anyOf`` normalized branch order, array items)
    is preserved. Equal schemas therefore yield identical bytes across
    processes; any structural or annotation difference changes the bytes.
    """
    doc = _to_json_schema(schema, inline_defs=False, include_annotations=True, sort_required=True)
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
