"""Accept/reject matrix, round-trip, and determinism tests for the §5 canonical
JSON-Schema subset (provider_runtime.schema)."""

import copy
import json
from collections.abc import Mapping
from typing import Literal, cast

import pytest

import provider_runtime as package_root
import provider_runtime.schema as schema_module
from provider_runtime.errors import SchemaViolation
from provider_runtime.schema import (
    ArrayNode,
    CanonicalJsonSchema,
    NullableUnion,
    NullNode,
    ObjectNode,
    Ref,
    ScalarNode,
    canonical_schema_bytes,
    parse_canonical_schema,
    to_json_schema,
)
from provider_runtime.types import Absent, Presence, Present

_ABSENT = Absent()


def scalar(
    scalar_type: Literal["string", "number", "integer", "boolean"],
    enum: Presence[tuple[str | int | float | bool, ...]] = _ABSENT,
) -> ScalarNode:
    """A ScalarNode with unauthored annotations (node fields have no defaults)."""
    return ScalarNode(type=scalar_type, enum=enum, title=Absent(), description=Absent())


STR: dict[str, object] = {"type": "string"}
INT: dict[str, object] = {"type": "integer"}
NUM: dict[str, object] = {"type": "number"}
BOOL: dict[str, object] = {"type": "boolean"}
NULL: dict[str, object] = {"type": "null"}


def obj(props: dict[str, object] | None = None, **extra: object) -> dict[str, object]:
    """A canonical object schema over ``props`` (required = all names, closed)."""
    properties = props or {}
    doc: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    doc.update(extra)
    return doc


def nullable(inner: dict[str, object]) -> dict[str, object]:
    return {"anyOf": [inner, dict(NULL)]}


def rejects(raw: object, *expect: str) -> str:
    """Assert parse rejects ``raw`` and every fragment appears in the message."""
    with pytest.raises(SchemaViolation) as exc:
        parse_canonical_schema(cast(Mapping[str, object], raw))
    msg = str(exc.value)
    for fragment in expect:
        assert fragment in msg, (
            f"expected fragment {fragment!r} in SchemaViolation message, got: {msg}"
        )
    return msg


# --- accepts ----------------------------------------------------------------


def test_accepts_minimal_empty_object_root() -> None:
    schema = parse_canonical_schema(obj())
    assert schema.root == ObjectNode(properties={}, title=Absent(), description=Absent()), (
        "empty-properties object root must parse to an empty ObjectNode"
    )
    assert dict(schema.defs) == {}, "no $defs map must parse to empty defs"


def test_accepts_all_scalar_types() -> None:
    schema = parse_canonical_schema(obj({"s": STR, "i": INT, "n": NUM, "b": BOOL}))
    assert schema.root.properties == {
        "s": scalar("string"),
        "i": scalar("integer"),
        "n": scalar("number"),
        "b": scalar("boolean"),
    }


def test_accepts_type_compatible_enums() -> None:
    schema = parse_canonical_schema(
        obj(
            {
                "s": {"type": "string", "enum": ["a", "b"]},
                "i": {"type": "integer", "enum": [1, 2, 3]},
                "n": {"type": "number", "enum": [1, 2.5]},  # int+float: one JSON number type
                "b": {"type": "boolean", "enum": [True]},
            }
        )
    )
    n = schema.root.properties["n"]
    assert isinstance(n, ScalarNode) and n.enum == Present((1, 2.5)), (
        "number enum must admit mixed int/float values as authored"
    )


def test_accepts_nullable_union_in_either_authored_order() -> None:
    forward = parse_canonical_schema(obj({"x": {"anyOf": [dict(STR), dict(NULL)]}}))
    reversed_ = parse_canonical_schema(obj({"x": {"anyOf": [dict(NULL), dict(STR)]}}))
    assert forward == reversed_, (
        "anyOf branch order is semantically free and must parse order-normalized"
    )
    assert forward.root.properties["x"] == NullableUnion(
        non_null=scalar("string"), title=Absent(), description=Absent()
    )


def test_accepts_annotations_on_object_array_scalar_union() -> None:
    raw = obj(
        {
            "arr": {
                "type": "array",
                "items": {"type": "string", "title": "Item", "description": "one item"},
                "title": "Arr",
            },
            "opt": {"anyOf": [dict(INT), dict(NULL)], "description": "maybe int"},
        },
        title="Root",
        description="root object",
    )
    schema = parse_canonical_schema(raw)
    assert schema.root.title == Present("Root")
    assert schema.root.description == Present("root object")
    arr = schema.root.properties["arr"]
    assert isinstance(arr, ArrayNode) and arr.title == Present("Arr")
    assert isinstance(arr.items, ScalarNode) and arr.items.description == Present("one item")
    opt = schema.root.properties["opt"]
    assert isinstance(opt, NullableUnion) and opt.description == Present("maybe int")
    assert opt.title == Absent(), "unauthored annotations must parse to Absent"


def test_accepts_refs_including_chains_and_unreferenced_defs() -> None:
    raw = obj(
        {"a": {"$ref": "#/$defs/alias"}, "b": {"$ref": "#/$defs/point"}},
        **{
            "$defs": {
                "point": obj({"x": NUM, "y": NUM}),
                "alias": {"$ref": "#/$defs/point"},  # def-to-def chain, acyclic
                "unused": STR,  # unreferenced defs are legal and preserved
            }
        },
    )
    schema = parse_canonical_schema(raw)
    assert schema.root.properties["a"] == Ref(name="alias")
    assert schema.defs["alias"] == Ref(name="point")
    assert schema.defs["unused"] == scalar("string")


def test_accepts_required_in_any_order() -> None:
    raw = obj({"a": STR, "b": INT})
    raw["required"] = ["b", "a"]
    schema = parse_canonical_schema(raw)
    assert set(schema.root.properties) == {"a", "b"}, (
        "required is set-equality against property names; order is free"
    )


def test_accepts_empty_root_defs_map() -> None:
    schema = parse_canonical_schema(obj({"a": STR}, **{"$defs": {}}))
    assert dict(schema.defs) == {}


def test_parse_is_pure_and_never_mutates_input() -> None:
    raw = obj(
        {"x": {"anyOf": [dict(NULL), dict(STR)]}, "arr": {"type": "array", "items": dict(INT)}},
        **{"$defs": {"d": obj({"y": NUM})}},
    )
    snapshot = copy.deepcopy(raw)
    parse_canonical_schema(raw)
    assert raw == snapshot, "parse_canonical_schema must never rewrite trusted input"


# --- rejects: root ----------------------------------------------------------


def test_rejects_non_mapping_document() -> None:
    rejects(["not", "a", "schema"], "#:", "must be a JSON object")


def test_rejects_non_object_root() -> None:
    rejects(STR, "#:", "root must be an object schema")


def test_rejects_union_root() -> None:
    rejects(nullable(obj()), "root must be an object schema")


def test_rejects_ref_root() -> None:
    rejects({"$ref": "#/$defs/a"}, "root must be an object schema")


def test_rejects_missing_type_keyword() -> None:
    rejects(
        obj({"x": {"properties": {}, "required": [], "additionalProperties": False}}),
        "#/properties/x",
        "'type'",
    )


# --- rejects: object shape --------------------------------------------------


def test_rejects_object_missing_properties() -> None:
    rejects(
        obj({"o": {"type": "object", "required": [], "additionalProperties": False}}),
        "#/properties/o",
        "requires 'properties'",
    )


def test_rejects_omitted_required() -> None:
    inner: dict[str, object] = {
        "type": "object",
        "properties": {"a": dict(STR)},
        "additionalProperties": False,
    }
    rejects(obj({"inner": inner}), "#/properties/inner", "'required'")


def test_rejects_required_not_matching_properties() -> None:
    raw = obj({"a": STR, "b": INT})
    raw["required"] = ["a"]
    rejects(raw, "#/required", "exactly every property name", "['b']")
    raw["required"] = ["a", "b", "ghost"]
    rejects(raw, "#/required", "['ghost']")


def test_rejects_required_duplicates_and_non_strings() -> None:
    raw = obj({"a": STR})
    raw["required"] = ["a", "a"]
    rejects(raw, "#/required", "duplicate")
    raw["required"] = [1]
    rejects(raw, "#/required", "must be strings")
    raw["required"] = "a"
    rejects(raw, "#/required", "must be an array")


def test_rejects_missing_additional_properties() -> None:
    raw: dict[str, object] = {"type": "object", "properties": {}, "required": []}
    rejects(raw, "#:", "'additionalProperties'", "false explicitly")


def test_rejects_true_additional_properties() -> None:
    rejects(obj(additionalProperties=True), "#/additionalProperties", "exactly false")


def test_rejects_schema_valued_additional_properties() -> None:
    rejects(obj(additionalProperties=dict(STR)), "#/additionalProperties", "schema-valued")


def test_rejects_zero_as_additional_properties() -> None:
    # JSON false only; a falsy non-bool is not an explicit false.
    rejects(obj(additionalProperties=0), "#/additionalProperties", "exactly false")


def test_rejects_non_mapping_properties() -> None:
    raw: dict[str, object] = {
        "type": "object",
        "properties": [],
        "required": [],
        "additionalProperties": False,
    }
    rejects(raw, "#/properties", "must be a mapping")


# --- rejects: arrays --------------------------------------------------------


def test_rejects_array_without_items() -> None:
    rejects(obj({"a": {"type": "array"}}), "#/properties/a", "'items'")


def test_rejects_tuple_items() -> None:
    rejects(
        obj({"tags": {"type": "array", "items": [dict(STR), dict(INT)]}}),
        "#/properties/tags/items",
        "tuple 'items'",
    )


def test_rejects_prefix_items() -> None:
    rejects(
        obj({"a": {"type": "array", "items": dict(STR), "prefixItems": [dict(STR)]}}),
        "prefixItems",
        "tuple arrays",
    )


# --- rejects: unions and null ----------------------------------------------


def test_rejects_null_outside_union_as_property() -> None:
    rejects(obj({"x": dict(NULL)}), "#/properties/x", "only as one branch of a nullable anyOf")


def test_rejects_null_outside_union_as_items() -> None:
    rejects(obj({"a": {"type": "array", "items": dict(NULL)}}), "#/properties/a/items")


def test_rejects_null_as_definition() -> None:
    rejects(obj({"a": STR}, **{"$defs": {"n": dict(NULL)}}), "#/$defs/n")


def test_rejects_annotated_null_branch() -> None:
    raw = obj({"x": {"anyOf": [{"type": "null", "title": "nil"}, dict(NULL)]}})
    rejects(raw, "#/properties/x/anyOf/0", 'exactly {"type": "null"}')


def test_rejects_any_of_with_wrong_arity() -> None:
    rejects(obj({"x": {"anyOf": [dict(NULL)]}}), "#/properties/x/anyOf", "1 entries instead of 2")
    rejects(
        obj({"x": {"anyOf": [dict(STR), dict(INT), dict(NULL)]}}),
        "#/properties/x/anyOf",
        "3 entries instead of 2",
    )


def test_rejects_arbitrary_any_of() -> None:
    rejects(
        obj({"x": {"anyOf": [dict(STR), dict(INT)]}}), "#/properties/x/anyOf", "arbitrary anyOf"
    )


def test_rejects_double_null_any_of() -> None:
    rejects(
        obj({"x": {"anyOf": [dict(NULL), dict(NULL)]}}),
        "#/properties/x/anyOf",
        "exactly one non-null",
    )


def test_rejects_nested_nullable_union() -> None:
    raw = obj({"x": {"anyOf": [nullable(dict(STR)), dict(NULL)]}})
    rejects(raw, "#/properties/x/anyOf/0", "may not nest")


def test_rejects_union_type_list() -> None:
    rejects(obj({"x": {"type": ["string", "null"]}}), "#/properties/x/type", "union types")


def test_rejects_ref_resolving_to_union_as_union_branch() -> None:
    raw = obj(
        {"x": {"anyOf": [{"$ref": "#/$defs/u"}, dict(NULL)]}},
        **{"$defs": {"u": nullable(dict(STR))}},
    )
    rejects(raw, "resolves to NullableUnion", "#/$defs/u")


def test_rejects_non_array_any_of() -> None:
    rejects(obj({"x": {"anyOf": dict(STR)}}), "#/properties/x/anyOf", "must be an array")


# --- rejects: refs ----------------------------------------------------------


def test_rejects_external_ref() -> None:
    rejects(
        obj({"x": {"$ref": "https://example.com/schema.json"}}),
        "#/properties/x/$ref",
        "only local '#/$defs/<name>'",
    )


def test_rejects_non_defs_pointer_ref() -> None:
    rejects(obj({"x": {"$ref": "#/properties/y"}}), "only local '#/$defs/<name>'")


def test_rejects_nested_pointer_ref() -> None:
    rejects(
        obj({"x": {"$ref": "#/$defs/a/b"}}, **{"$defs": {"a": obj()}}),
        "only local '#/$defs/<name>'",
    )


def test_rejects_ref_with_siblings() -> None:
    raw = obj(
        {"x": {"$ref": "#/$defs/a", "title": "sibling"}},
        **{"$defs": {"a": obj()}},
    )
    rejects(raw, "#/properties/x", "no sibling keys", "['title']")


def test_rejects_undefined_ref() -> None:
    rejects(
        obj({"x": {"$ref": "#/$defs/missing"}}),
        "#/properties/x/$ref",
        "undefined definition '#/$defs/missing'",
    )


def test_rejects_self_referential_def() -> None:
    raw = obj(
        {"x": {"$ref": "#/$defs/a"}},
        **{"$defs": {"a": {"type": "array", "items": {"$ref": "#/$defs/a"}}}},
    )
    rejects(raw, "recursive $ref cycle", "#/$defs/a -> #/$defs/a")


def test_rejects_mutually_recursive_defs() -> None:
    raw = obj(
        {"x": {"$ref": "#/$defs/a"}},
        **{
            "$defs": {
                "a": {"type": "array", "items": {"$ref": "#/$defs/b"}},
                "b": {"type": "array", "items": {"$ref": "#/$defs/a"}},
            }
        },
    )
    msg = rejects(raw, "recursive $ref cycle")
    assert "#/$defs/a" in msg and "#/$defs/b" in msg, f"cycle members must be named: {msg}"


def test_rejects_defs_below_root() -> None:
    inner = obj({"a": STR}, **{"$defs": {"d": dict(STR)}})
    rejects(obj({"o": inner}), "#/properties/o", "'$defs'", "document root")


def test_rejects_bad_definition_names() -> None:
    rejects(obj({"a": STR}, **{"$defs": {"a/b": dict(STR)}}), "#/$defs", "'a/b'")
    rejects(obj({"a": STR}, **{"$defs": {"": dict(STR)}}), "#/$defs", "non-empty")


# --- rejects: scalars and enums ---------------------------------------------


def test_rejects_unsupported_type() -> None:
    rejects(obj({"x": {"type": "decimal"}}), "#/properties/x/type", "unsupported type 'decimal'")


def test_rejects_non_string_type() -> None:
    rejects(obj({"x": {"type": 3}}), "#/properties/x/type", "must be a string")


def test_rejects_empty_enum() -> None:
    rejects(obj({"x": {"type": "string", "enum": []}}), "#/properties/x/enum", "non-empty")


def test_rejects_non_array_enum() -> None:
    rejects(
        obj({"x": {"type": "string", "enum": "abc"}}), "#/properties/x/enum", "must be an array"
    )


def test_rejects_type_incompatible_enum() -> None:
    rejects(
        obj({"x": {"type": "string", "enum": ["a", 1]}}),
        "#/properties/x/enum",
        "index 1",
        "'string'",
    )


def test_rejects_bool_in_integer_enum() -> None:
    # bool is a Python int subclass but not a JSON integer.
    rejects(obj({"x": {"type": "integer", "enum": [1, True]}}), "index 1", "'integer'")


def test_rejects_float_in_integer_enum() -> None:
    rejects(obj({"x": {"type": "integer", "enum": [2.0]}}), "index 0", "'integer'")


def test_rejects_int_in_boolean_enum() -> None:
    rejects(obj({"x": {"type": "boolean", "enum": [1]}}), "index 0", "'boolean'")


def test_rejects_non_finite_number_enum() -> None:
    rejects(obj({"x": {"type": "number", "enum": [float("inf")]}}), "non-finite")


def test_rejects_non_string_annotation() -> None:
    rejects(obj({"x": {"type": "string", "title": 3}}), "#/properties/x/title", "must be a string")


# --- rejects: foreign keywords ----------------------------------------------


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("nullable", True),
        ("default", "x"),
        ("oneOf", [dict(STR)]),
        ("allOf", [dict(STR)]),
        ("not", dict(STR)),
        ("if", dict(STR)),
        ("format", "email"),
        ("pattern", "^a"),
        ("minLength", 1),
        ("maxLength", 9),
        ("minimum", 0),
        ("maximum", 10),
        ("const", "a"),
        ("examples", ["a"]),
        ("x-provider-extension", True),
    ],
)
def test_rejects_foreign_keyword_on_scalar(keyword: str, value: object) -> None:
    rejects(
        obj({"f": {"type": "string", keyword: value}}),
        "#/properties/f",
        f"unsupported keyword '{keyword}'",
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("patternProperties", {"^x": dict(STR)}),
        ("propertyNames", dict(STR)),
        ("unevaluatedProperties", False),
        ("minProperties", 1),
        ("maxProperties", 3),
        ("enum", ["a"]),
        ("$schema", "https://json-schema.org/draft/2020-12/schema"),
        ("$id", "https://example.com/root"),
    ],
)
def test_rejects_foreign_keyword_on_object(keyword: str, value: object) -> None:
    rejects(obj({}, **{keyword: value}), f"unsupported keyword '{keyword}'")


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("minItems", 1), ("maxItems", 4), ("uniqueItems", True), ("contains", dict(STR))],
)
def test_rejects_foreign_keyword_on_array(keyword: str, value: object) -> None:
    rejects(
        obj({"a": {"type": "array", "items": dict(STR), keyword: value}}),
        "#/properties/a",
        f"unsupported keyword '{keyword}'",
    )


def test_rejects_bare_one_of_node() -> None:
    rejects(
        obj({"x": {"oneOf": [dict(STR), dict(NULL)]}}),
        "#/properties/x",
        "'type', '$ref', or a nullable 'anyOf'",
    )


# --- ownership surface and direct-construction invariants -------------------


def test_package_surface_reexports_schema_objects() -> None:
    # schema.py is the owner surface; the package root re-exports it.
    for name in (
        "ObjectNode",
        "ArrayNode",
        "ScalarNode",
        "NullNode",
        "NullableUnion",
        "Ref",
        "CanonicalJsonSchema",
        "Node",
        "parse_canonical_schema",
        "to_json_schema",
        "canonical_schema_bytes",
    ):
        assert getattr(package_root, name) is getattr(schema_module, name), (
            f"provider_runtime.{name} must be the same object as"
            f" provider_runtime.schema.{name} (re-export, not a redefinition)"
        )


def test_nullable_union_constructor_rejects_null_inner() -> None:
    with pytest.raises(SchemaViolation, match="non-null, non-union"):
        NullableUnion(non_null=NullNode(), title=Absent(), description=Absent())


def test_nullable_union_constructor_rejects_union_inner() -> None:
    inner = NullableUnion(non_null=scalar("string"), title=Absent(), description=Absent())
    with pytest.raises(SchemaViolation, match="non-null, non-union"):
        NullableUnion(non_null=inner, title=Absent(), description=Absent())


def test_scalar_constructor_rejects_empty_enum() -> None:
    with pytest.raises(SchemaViolation, match="non-empty"):
        scalar("string", enum=Present(()))


def test_scalar_constructor_rejects_incompatible_enum() -> None:
    with pytest.raises(SchemaViolation, match="does not match scalar type 'integer'"):
        scalar("integer", enum=Present((True,)))


def test_schema_constructor_rejects_dangling_ref() -> None:
    root = ObjectNode(properties={"a": Ref(name="ghost")})
    with pytest.raises(SchemaViolation, match="undefined definition"):
        CanonicalJsonSchema(root=root, defs={})


def test_schema_constructor_rejects_cyclic_defs() -> None:
    root = ObjectNode(properties={"a": Ref(name="a")})
    with pytest.raises(SchemaViolation, match="recursive \\$ref cycle"):
        CanonicalJsonSchema(root=root, defs={"a": ArrayNode(items=Ref(name="a"))})


def test_schema_constructor_rejects_misplaced_null_node() -> None:
    root = ObjectNode(properties={"a": NullNode()})
    with pytest.raises(SchemaViolation, match="nullable anyOf"):
        CanonicalJsonSchema(root=root, defs={})


# --- round-trip -------------------------------------------------------------

ROUND_TRIP_BATTERY: list[dict[str, object]] = [
    obj(),
    obj({"s": STR, "i": INT, "n": NUM, "b": BOOL}),
    obj(
        {
            "mode": {"type": "string", "enum": ["fast", "slow"], "title": "Mode"},
            "level": {"type": "integer", "enum": [1, 2, 3]},
            "ratio": {"type": "number", "enum": [0.5, 1, 2.5]},
            "flag": {"type": "boolean", "enum": [False]},
        },
        title="Enums",
        description="every scalar enum kind",
    ),
    obj({"grid": {"type": "array", "items": {"type": "array", "items": dict(NUM)}}}),
    obj(
        {
            "opt_scalar": nullable(dict(STR)),
            "opt_array": nullable({"type": "array", "items": dict(INT)}),
            "opt_object": {"anyOf": [obj({"y": NUM}), dict(NULL)], "title": "maybe"},
        }
    ),
    obj(
        {"a": {"$ref": "#/$defs/alias"}, "b": nullable({"$ref": "#/$defs/point"})},
        **{
            "$defs": {
                "point": obj({"x": NUM, "y": NUM}, description="a point"),
                "alias": {"$ref": "#/$defs/point"},
                "unused": {"type": "string", "enum": ["z"]},
            }
        },
    ),
    obj(
        {
            "outer": obj(
                {"inner": obj({"leaf": {"type": "string", "description": "deep"}})},
                title="Outer",
            )
        }
    ),
]


@pytest.mark.parametrize("raw", ROUND_TRIP_BATTERY, ids=range(len(ROUND_TRIP_BATTERY)))
def test_round_trip_law(raw: dict[str, object]) -> None:
    schema = parse_canonical_schema(raw)
    doc = to_json_schema(schema, inline_defs=False, include_annotations=True)
    assert parse_canonical_schema(doc) == schema, (
        "parse(to_json_schema(s, inline_defs=False, include_annotations=True)) must equal s"
    )


def test_to_json_schema_emits_exact_canonical_document() -> None:
    raw = obj(
        {"name": {"type": "string", "title": "Name"}, "opt": nullable(dict(INT))},
        **{"$defs": {"d": obj({"k": STR})}},
    )
    schema = parse_canonical_schema(raw)
    assert to_json_schema(schema, inline_defs=False, include_annotations=True) == raw, (
        "a canonically-authored document must serialize back identically"
    )


def test_to_json_schema_normalizes_union_branch_order() -> None:
    schema = parse_canonical_schema(obj({"x": {"anyOf": [dict(NULL), dict(STR)]}}))
    doc = to_json_schema(schema, inline_defs=False, include_annotations=True)
    properties = cast(dict[str, object], doc["properties"])
    assert cast(dict[str, object], properties["x"])["anyOf"] == [STR, NULL], (
        "serializer must emit the normalized [non-null, null] branch order"
    )


def test_to_json_schema_required_follows_property_order() -> None:
    raw = obj({"b": STR, "a": INT})
    doc = to_json_schema(parse_canonical_schema(raw), inline_defs=False, include_annotations=True)
    assert doc["required"] == ["b", "a"], "required must list every property in property order"
    assert doc["additionalProperties"] is False


def test_inline_defs_resolves_every_ref() -> None:
    raw = obj(
        {"a": {"$ref": "#/$defs/alias"}, "b": nullable({"$ref": "#/$defs/point"})},
        **{
            "$defs": {
                "point": obj({"x": NUM}),
                "alias": {"$ref": "#/$defs/point"},
            }
        },
    )
    doc = to_json_schema(parse_canonical_schema(raw), inline_defs=True, include_annotations=True)
    dumped = json.dumps(doc)
    assert "$ref" not in dumped and "$defs" not in doc, f"inlined doc still has refs: {dumped}"
    expected = parse_canonical_schema(obj({"a": obj({"x": NUM}), "b": nullable(obj({"x": NUM}))}))
    assert parse_canonical_schema(doc) == expected, "inlining must be structure-preserving"


def test_include_annotations_false_strips_all_annotations() -> None:
    raw = obj(
        {"a": {"type": "string", "title": "T", "description": "D"}},
        title="Root",
        description="R",
        **{"$defs": {"d": obj({"k": STR}, title="Def")}},
    )
    doc = to_json_schema(parse_canonical_schema(raw), inline_defs=False, include_annotations=False)
    dumped = json.dumps(doc)
    assert '"title"' not in dumped and '"description"' not in dumped, (
        f"annotations must be omitted entirely: {dumped}"
    )
    reparsed = parse_canonical_schema(doc)
    a = reparsed.root.properties["a"]
    assert isinstance(a, ScalarNode) and a.title == Absent()


# --- canonical bytes ---------------------------------------------------------


def test_canonical_bytes_ignore_authoring_order() -> None:
    defs_forward: dict[str, object] = {"pt": obj({"x": NUM, "y": NUM})}
    raw_forward: dict[str, object] = {
        "type": "object",
        "properties": {"a": {"anyOf": [{"$ref": "#/$defs/pt"}, dict(NULL)]}, "b": dict(STR)},
        "required": ["a", "b"],
        "additionalProperties": False,
        "$defs": defs_forward,
    }
    pt_shuffled: dict[str, object] = {
        "additionalProperties": False,
        "required": ["y", "x"],
        "properties": {"y": dict(NUM), "x": dict(NUM)},
        "type": "object",
    }
    raw_shuffled: dict[str, object] = {
        "$defs": {"pt": pt_shuffled},
        "additionalProperties": False,
        "required": ["b", "a"],
        "properties": {"b": dict(STR), "a": {"anyOf": [dict(NULL), {"$ref": "#/$defs/pt"}]}},
        "type": "object",
    }
    forward = parse_canonical_schema(raw_forward)
    shuffled = parse_canonical_schema(raw_shuffled)
    assert forward == shuffled, "authoring order must not affect the parsed value"
    assert canonical_schema_bytes(forward) == canonical_schema_bytes(shuffled), (
        "equal schemas must produce identical canonical bytes regardless of authored order"
    )


def test_canonical_bytes_are_valid_sorted_compact_json() -> None:
    schema = parse_canonical_schema(obj({"a": STR}, **{"$defs": {"d": dict(INT)}}))
    raw_bytes = canonical_schema_bytes(schema)
    text = raw_bytes.decode("utf-8")
    doc = json.loads(text)
    assert text == json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False), (
        "canonical bytes must already be in sorted-key compact form"
    )


def test_canonical_bytes_distinguish_different_schemas() -> None:
    plain = parse_canonical_schema(obj({"a": STR}))
    other = parse_canonical_schema(obj({"a": INT}))
    annotated = parse_canonical_schema(obj({"a": {"type": "string", "title": "A"}}))
    assert canonical_schema_bytes(plain) != canonical_schema_bytes(other), (
        "structural differences must change the canonical bytes"
    )
    assert canonical_schema_bytes(plain) != canonical_schema_bytes(annotated), (
        "annotations reach providers, so they must be fingerprint-relevant"
    )


def test_canonical_bytes_preserve_enum_value_order() -> None:
    forward = parse_canonical_schema(obj({"a": {"type": "string", "enum": ["x", "y"]}}))
    reversed_ = parse_canonical_schema(obj({"a": {"type": "string", "enum": ["y", "x"]}}))
    assert forward != reversed_, "enum order is semantic; differently-ordered enums differ"
    assert canonical_schema_bytes(forward) != canonical_schema_bytes(reversed_)
