"""Protocol-agnostic engine code — the rules all four engines share.

What lives here is decided by the IR and the registry, not by a wire dialect:
the row's reasoning knob (which level is expressible, what goes on the wire,
which request keys the row owns), the strict-JSON decode of a terminal text,
the continuation/target compatibility rule, Retry-After parsing, and the
narrowing primitives every ingress uses. Each engine keeps only its thin
protocol-specific application of the result — where the fragment merges, how a
failure value is folded into an outcome, which SDK exception carries the cause.

Layering: imports from `types`, `errors` and `registry` only; no SDK imports
(httpx is the shared transport type, not a provider SDK).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, assert_never

import httpx

from provider_runtime.errors import InvalidRequest, RuntimeDefect
from provider_runtime.registry import ModelRow
from provider_runtime.types import (
    Absent,
    ContinuationArtifact,
    GenerateIntent,
    InvalidStructuredOutput,
    Presence,
    Present,
    ResponseContent,
    StrictJsonOutput,
    StructuredContent,
    TextContent,
    TextOutput,
    ToolCall,
)

# ---------------------------------------------------------------------------
# Narrowing primitives (nullable/extra provider JSON stays private to ingress).


def int_or_none(value: object) -> int | None:
    # bool is an int subclass; token counts are never booleans.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def mapping_or_none(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


# ---------------------------------------------------------------------------
# Attempt bookkeeping.


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def registry_invalid(row: ModelRow, detail: str) -> RuntimeDefect:
    return RuntimeDefect(
        origin="intent", code="registry_invalid", message=f"row {row.ref!r}: {detail}"
    )


# ---------------------------------------------------------------------------
# Reasoning — the whole knob, for every engine.


@dataclass(frozen=True, slots=True)
class RowReasoning:
    """What the row's reasoning knob contributes to one request."""

    # The wire fragment for the requested level, merged verbatim by the engine.
    fragment: Mapping[str, Any] = field(repr=False)
    # The fragment as compact sorted-keys JSON; Absent when nothing is sent.
    native_reasoning: Presence[str]
    # Every request key the row's knob can write, across ALL its declared
    # levels — the collision set does not depend on which level was selected.
    owned_keys: frozenset[str]


def row_reasoning(row: ModelRow, intent: GenerateIntent) -> RowReasoning:
    """Resolve the row's self-describing reasoning knob for the intent's level.

    Two rulings live here (spec §14). ``reasoning="none"`` is callable on every
    row: a row declaring a ``"none"`` fragment sends it, a row that declares
    none sends nothing and the provider's own default applies — "none" never
    raises, because it is the facade default. Any OTHER undeclared level does
    raise: silently downgrading an explicit effort request is banned.

    And ``owned_keys`` spans every declared level, not the selected one, so
    provider_options can never smuggle in a knob the row expresses elsewhere
    (deepseek's ``reasoning_effort`` under level "none", say).
    """
    match row.reasoning:
        case Absent():
            if intent.reasoning != "none":
                raise InvalidRequest(
                    message=f"row {row.ref!r} has no reasoning knob; "
                    f"level {intent.reasoning!r} is not expressible"
                )
            return RowReasoning(fragment={}, native_reasoning=Absent(), owned_keys=frozenset())
        case Present(value=levels):
            pass
        case _:
            assert_never(row.reasoning)

    fragments: dict[str, dict[str, Any]] = {}
    owned: set[str] = set()
    for level, value in levels.items():
        if not isinstance(value, Mapping):
            raise registry_invalid(row, "reasoning values must be request-fragment mappings")
        fragment: dict[str, Any] = dict(value)
        fragments[level] = fragment
        owned.update(fragment)

    selected = fragments.get(intent.reasoning)
    if selected is None:
        if intent.reasoning != "none":
            raise InvalidRequest(
                message=f"reasoning level {intent.reasoning!r} is not declared for {row.ref!r}"
            )
        return RowReasoning(fragment={}, native_reasoning=Absent(), owned_keys=frozenset(owned))
    return RowReasoning(
        fragment=selected,
        native_reasoning=Present(json.dumps(selected, sort_keys=True, separators=(",", ":"))),
        owned_keys=frozenset(owned),
    )


# ---------------------------------------------------------------------------
# Continuations + terminal content.


def validate_continuation(
    artifact: ContinuationArtifact, row: ModelRow, intent: GenerateIntent
) -> None:
    if artifact.target != intent.target or artifact.codec_id != row.continuation_codec:
        raise InvalidRequest(
            message=(
                f"continuation artifact for {artifact.target.provider}/{artifact.target.model} "
                f"(codec {artifact.codec_id!r}) cannot replay to "
                f"{intent.target.provider}/{intent.target.model} "
                f"(codec {row.continuation_codec!r})"
            )
        )


def response_content(
    intent: GenerateIntent, *, text: str, tool_calls: tuple[ToolCall, ...]
) -> ResponseContent | InvalidStructuredOutput:
    """The output arm is the intent's OutputSpec, never re-inferred from the wire.

    Strict JSON parse only; NO repair. A provider that answers a strict-JSON
    intent with unparseable output is an expected model failure — the value the
    caller folds into Failed — not a wire-protocol defect.
    """
    match intent.output:
        case TextOutput():
            return TextContent(text=text, tool_calls=tool_calls)
        case StrictJsonOutput():
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                return InvalidStructuredOutput(
                    safe_detail=f"structured output is not valid JSON "
                    f"({error.msg} at char {error.pos})"
                )
            if not isinstance(payload, dict):
                return InvalidStructuredOutput(
                    safe_detail=f"structured output parsed to {type(payload).__name__}, "
                    f"not a JSON object"
                )
            return StructuredContent(payload=payload, text=text)
        case _:
            assert_never(intent.output)


# ---------------------------------------------------------------------------
# Error-shape helpers.


def retry_after_seconds(headers: httpx.Headers) -> Presence[float]:
    """Numeric Retry-After seconds; the HTTP-date form has no consumer → Absent."""
    raw = headers.get("retry-after")
    if raw is None:
        return Absent()
    try:
        seconds = float(raw)
    except ValueError:
        return Absent()
    return Present(seconds) if seconds >= 0 else Absent()


def caused_by(error: BaseException, kind: type[BaseException]) -> bool:
    """Whether `kind` appears in the __cause__ chain — the SDKs raise their own
    exception type `from` the httpx error that actually happened, so what a
    timeout MEANS is only readable through the chain."""
    cause = error.__cause__
    while cause is not None:
        if isinstance(cause, kind):
            return True
        cause = cause.__cause__
    return False
