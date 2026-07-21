"""Private syntax-only Chat Completions wire helpers (moonshot + openrouter).

Scope discipline: this module knows Chat Completions SYNTAX only — message /
tool-definition / response_format dict shapes, response envelope field
extraction, SSE chunk primitives, strict tool-argument parsing, and raw
usage-frame folding. It contains NO provider branching and makes NO provider
decisions: each codec picks which usage location it consumes (top-level vs
``choices[0].usage`` — BOTH are surfaced here), which continuation payload it
replays, and how errors classify.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from provider_runtime._signals import ExpectedFailureSignal
from provider_runtime.errors import ProtocolDefect
from provider_runtime.schema import to_json_schema
from provider_runtime.transport import SseEvent
from provider_runtime.types import (
    Absent,
    CanonicalTool,
    InvalidToolArguments,
    Presence,
    Present,
    PromptMessage,
    Stable,
    StrictJsonOutput,
    SystemMessage,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    UserMessage,
    presence_of,
)

# ---------------------------------------------------------------------------
# Serialization (the ONE body encoding both codecs use everywhere)


def dump_body(payload: Mapping[str, object]) -> bytes:
    """Deterministic body bytes: construction key order, compact separators."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def frame(chunk: bytes) -> bytes:
    """Length-framed prefix component: ``uint64_be(len) || bytes`` (seam contract)."""
    return len(chunk).to_bytes(8, "big") + chunk


# ---------------------------------------------------------------------------
# Narrowing primitives (nullable provider JSON stays private to this seam)


def mapping_or_none(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def sequence_or_none(value: object) -> Sequence[object] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return None


def str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


# ---------------------------------------------------------------------------
# Message-array construction


def system_message(text: str) -> dict[str, object]:
    return {"role": "system", "content": text}


def user_message(text: str) -> dict[str, object]:
    return {"role": "user", "content": text}


def assistant_message(text: str, tool_calls: Sequence[ToolCall]) -> dict[str, object]:
    # Empty text alongside tool calls is a null content on this wire.
    content: str | None = text if (text or not tool_calls) else None
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [tool_call_wire(call) for call in tool_calls]
    return message


def tool_call_wire(call: ToolCall) -> dict[str, object]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                dict(call.arguments), separators=(",", ":"), ensure_ascii=False
            ),
        },
    }


def tool_result_message(call_id: str, output: str) -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": output}


def tool_definition(tool: CanonicalTool) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": to_json_schema(
                tool.parameters, inline_defs=False, include_annotations=True
            ),
        },
    }


def response_format_json_schema(output: StrictJsonOutput) -> dict[str, object]:
    """The ``chat_completions_response_format_json_schema`` dialect shape."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": output.name,
            "schema": to_json_schema(output.schema, inline_defs=False, include_annotations=True),
            "strict": True,
        },
    }


# ---------------------------------------------------------------------------
# Cache-affinity prefix bytes (seam §prefix_bytes: fixed component order)


def stable_prefix_messages(messages: tuple[PromptMessage, ...]) -> tuple[dict[str, object], ...]:
    """Leading contiguous run of Stable content, framed per native message
    dict (``{"role": ..., "content": <joined stable text>}``, mirroring
    ``system_message``/``user_message``) — so the enclosing role/message
    boundary participates in the cache-affinity projection, not bare text.

    Collection stops at the first Dynamic block or the first non-prompt-block
    message; the planner owns non-empty/contiguity/scope validation. A
    message contributes an entry only for its LEADING stable run (trailing
    Dynamic blocks in the same message never enter the projection)."""
    projected: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, SystemMessage | UserMessage):
            return tuple(projected)
        role = "system" if isinstance(message, SystemMessage) else "user"
        texts: list[str] = []
        for block in message.blocks:
            if not isinstance(block.stability, Stable):
                if texts:
                    projected.append({"role": role, "content": "".join(texts)})
                return tuple(projected)
            texts.append(block.text)
        if texts:
            projected.append({"role": role, "content": "".join(texts)})
    return tuple(projected)


def cache_prefix_bytes(
    stable_messages: Sequence[Mapping[str, object]],
    tool_definitions: Sequence[Mapping[str, object]],
    tool_choice_wire: str | None,
    output_format: Mapping[str, object] | None,
) -> bytes:
    """Length-framed concatenation in the seam's fixed order: stable messages,
    tool definitions (the Chat Completions cache prefix includes tools),
    tool_choice, output format. Fixed section tags separate the four
    component classes so a stable text byte-equal to a tool-definition dump
    can never collide across classes."""
    parts = [frame(b"stable")]
    parts.extend(frame(dump_body(message)) for message in stable_messages)
    parts.append(frame(b"tools"))
    parts.extend(frame(dump_body(definition)) for definition in tool_definitions)
    parts.append(frame(b"tool_choice"))
    if tool_choice_wire is not None:
        parts.append(frame(tool_choice_wire.encode("utf-8")))
    parts.append(frame(b"format"))
    if output_format is not None:
        parts.append(frame(dump_body(output_format)))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Response envelope extraction (non-stream)


def parse_json_object(text: str | bytes, *, what: str) -> Mapping[str, object]:
    try:
        raw = text.decode("utf-8") if isinstance(text, bytes) else text
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolDefect(code="malformed_json", message=f"{what} is not valid JSON") from exc
    if not isinstance(data, Mapping):
        raise ProtocolDefect(code="malformed_envelope", message=f"{what} is not a JSON object")
    return data


def first_choice(data: Mapping[str, object], *, what: str) -> Mapping[str, object]:
    choices = sequence_or_none(data.get("choices"))
    if choices:
        choice = mapping_or_none(choices[0])
        if choice is not None:
            return choice
    raise ProtocolDefect(code="missing_choices", message=f"{what} has no choices[0] object")


def choice_message(choice: Mapping[str, object], *, what: str) -> Mapping[str, object]:
    message = mapping_or_none(choice.get("message"))
    if message is None:
        raise ProtocolDefect(code="missing_message", message=f"{what} choice has no message object")
    return message


def message_text(message: Mapping[str, object]) -> str:
    return str_or_none(message.get("content")) or ""


def finish_reason_of(choice: Mapping[str, object]) -> str | None:
    return str_or_none(choice.get("finish_reason"))


def message_tool_calls(message: Mapping[str, object]) -> tuple[ToolCall, ...]:
    raw = message.get("tool_calls")
    if raw is None:
        return ()
    entries = sequence_or_none(raw)
    if entries is None:
        raise ProtocolDefect(code="malformed_tool_call", message="tool_calls is not an array")
    calls: list[ToolCall] = []
    for entry in entries:
        entry_mapping = mapping_or_none(entry)
        if entry_mapping is None:
            raise ProtocolDefect(
                code="malformed_tool_call", message="tool_calls entry is not an object"
            )
        calls.append(parse_tool_call(entry_mapping))
    return tuple(calls)


def parse_tool_call(entry: Mapping[str, object]) -> ToolCall:
    call_id = str_or_none(entry.get("id")) or ""
    function = mapping_or_none(entry.get("function")) or {}
    name = str_or_none(function.get("name")) or ""
    if not call_id or not name:
        raise ProtocolDefect(
            code="malformed_tool_call", message="tool call is missing id or function.name"
        )
    arguments = parse_strict_tool_arguments(
        str_or_none(function.get("arguments")) or "", tool_name=name, call_id=call_id
    )
    return ToolCall(id=call_id, name=name, arguments=arguments)


def parse_strict_tool_arguments(raw: str, *, tool_name: str, call_id: str) -> Mapping[str, object]:
    """Strict JSON-object parse of tool-call arguments — NO repair.

    An empty payload means no arguments (the provider omitted the object);
    anything else must parse as exactly one JSON object or the codec raises
    ``ExpectedFailureSignal(InvalidToolArguments)`` for the runtime to fold."""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExpectedFailureSignal(
            InvalidToolArguments(
                safe_detail=(
                    f"tool call {call_id} ({tool_name}): arguments are not valid JSON"
                    f" ({exc.msg} at char {exc.pos})"
                )
            )
        ) from exc
    if not isinstance(parsed, dict):
        raise ExpectedFailureSignal(
            InvalidToolArguments(
                safe_detail=f"tool call {call_id} ({tool_name}): arguments are not a JSON object"
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# Usage — BOTH wire locations surfaced; each codec picks which it consumes


def top_level_usage(data: Mapping[str, object]) -> Mapping[str, object] | None:
    return mapping_or_none(data.get("usage"))


def choice_usage(choice: Mapping[str, object]) -> Mapping[str, object] | None:
    return mapping_or_none(choice.get("usage"))


def fold_raw_usage(
    base: Mapping[str, object] | None, new: Mapping[str, object] | None
) -> Mapping[str, object] | None:
    """Field-wise fold of raw usage frames, preferring the most complete picture:
    later non-null fields override, null/missing fields never erase, and nested
    detail objects merge the same way."""
    if new is None:
        return base
    if base is None:
        return dict(new)
    merged = dict(base)
    for key, value in new.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            inner = dict(existing)
            inner.update({k: v for k, v in value.items() if v is not None})
            merged[key] = inner
        elif value is not None:
            merged[key] = value
    return merged


def parse_usage(raw: Mapping[str, object]) -> TokenUsage:
    prompt_details = mapping_or_none(raw.get("prompt_tokens_details")) or {}
    completion_details = mapping_or_none(raw.get("completion_tokens_details")) or {}
    # cached_tokens appears nested under prompt_tokens_details or directly on
    # the usage object (Moonshot's flat form) — both nestings surfaced.
    cache_read = int_or_none(prompt_details.get("cached_tokens"))
    if cache_read is None:
        cache_read = int_or_none(raw.get("cached_tokens"))
    return TokenUsage.from_components(
        input_tokens=int_or_none(raw.get("prompt_tokens")) or 0,
        output_tokens=int_or_none(raw.get("completion_tokens")) or 0,
        total_tokens=presence_of(int_or_none(raw.get("total_tokens"))),
        reasoning_tokens=presence_of(int_or_none(completion_details.get("reasoning_tokens"))),
        cache_read_input_tokens=presence_of(cache_read),
        cache_write_input_tokens=presence_of(int_or_none(prompt_details.get("cache_write_tokens"))),
    )


def retry_after_seconds(headers: Mapping[str, str]) -> Presence[float]:
    """Numeric Retry-After seconds; the HTTP-date form has no consumer → Absent."""
    value = headers.get("retry-after")
    if value is None:
        return Absent()
    try:
        return Present(float(value))
    except ValueError:
        return Absent()


# ---------------------------------------------------------------------------
# SSE chunk primitives


DONE_DATA = "[DONE]"


def is_done(event: SseEvent) -> bool:
    return event.data.strip() == DONE_DATA


def parse_chunk(event: SseEvent, *, what: str) -> Mapping[str, object]:
    return parse_json_object(event.data, what=what)


def chunk_choice(chunk: Mapping[str, object], *, what: str) -> Mapping[str, object] | None:
    """``choices[0]`` of a stream chunk; None for the legal empty/missing-choices
    chunk (trailing usage-only frames)."""
    raw = chunk.get("choices")
    if raw is None:
        return None
    choices = sequence_or_none(raw)
    if choices is None:
        raise ProtocolDefect(code="malformed_envelope", message=f"{what} choices is not an array")
    if not choices:
        return None
    choice = mapping_or_none(choices[0])
    if choice is None:
        raise ProtocolDefect(
            code="malformed_envelope", message=f"{what} choices[0] is not an object"
        )
    return choice


def chunk_delta(choice: Mapping[str, object]) -> Mapping[str, object]:
    return mapping_or_none(choice.get("delta")) or {}


def delta_content(delta: Mapping[str, object]) -> str:
    return str_or_none(delta.get("content")) or ""


def delta_reasoning_content(delta: Mapping[str, object]) -> str:
    return str_or_none(delta.get("reasoning_content")) or ""


def delta_tool_calls(delta: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    entries = sequence_or_none(delta.get("tool_calls"))
    if entries is None:
        return ()
    return tuple(entry for entry in entries if isinstance(entry, Mapping))


# ---------------------------------------------------------------------------
# Streamed tool-call accumulation (index-keyed, per the Chat Completions wire)


@dataclass(slots=True)
class _ToolCallSlot:
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    started: bool = False


@dataclass(frozen=True, slots=True)
class FinishedToolCall:
    tool_call: ToolCall
    # Exact wire shape rebuilt with the RAW accumulated argument string, for
    # codecs that replay the complete native assistant message.
    native: Mapping[str, object]


class ToolCallAccumulator:
    """Accumulates streamed tool-call deltas keyed by ``index``.

    ``apply`` yields ToolCallStart once id+name are known and ToolCallDelta per
    argument fragment; ``finish`` strict-parses each accumulated argument string
    (raising ``ExpectedFailureSignal(InvalidToolArguments)`` — no repair) and
    returns the completed calls in index order."""

    def __init__(self) -> None:
        self._slots: dict[int, _ToolCallSlot] = {}

    @property
    def pending(self) -> bool:
        return bool(self._slots)

    def apply(
        self, entries: Sequence[Mapping[str, object]]
    ) -> Iterator[ToolCallStart | ToolCallDelta]:
        for entry in entries:
            index = int_or_none(entry.get("index")) or 0
            slot = self._slots.setdefault(index, _ToolCallSlot())
            call_id = str_or_none(entry.get("id"))
            if call_id:
                slot.call_id = call_id
            function = mapping_or_none(entry.get("function")) or {}
            name = str_or_none(function.get("name"))
            if name:
                slot.name = name
            if slot.call_id and slot.name and not slot.started:
                slot.started = True
                yield ToolCallStart(call_id=slot.call_id, name=slot.name)
            fragment = str_or_none(function.get("arguments"))
            if fragment:
                slot.arguments += fragment
                if slot.started:
                    yield ToolCallDelta(call_id=slot.call_id, arguments_delta=fragment)

    def finish(self) -> tuple[FinishedToolCall, ...]:
        finished: list[FinishedToolCall] = []
        for index in sorted(self._slots):
            slot = self._slots[index]
            if not slot.call_id or not slot.name:
                raise ProtocolDefect(
                    code="malformed_tool_call",
                    message=f"streamed tool call at index {index} never carried id and function.name",
                )
            arguments = parse_strict_tool_arguments(
                slot.arguments, tool_name=slot.name, call_id=slot.call_id
            )
            native: Mapping[str, object] = {
                "id": slot.call_id,
                "type": "function",
                "function": {"name": slot.name, "arguments": slot.arguments},
            }
            finished.append(
                FinishedToolCall(
                    tool_call=ToolCall(id=slot.call_id, name=slot.name, arguments=arguments),
                    native=native,
                )
            )
        self._slots.clear()
        return tuple(finished)
