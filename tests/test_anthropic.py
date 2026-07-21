"""Anthropic Messages codec tests: encode goldens, finalize/stream_request,
decode_response fixtures, decode_stream sequences, and the classify_error table."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from provider_runtime import anthropic
from provider_runtime._signals import ExpectedFailureSignal, TransientStreamError
from provider_runtime.catalog import CATALOG
from provider_runtime.errors import (
    CredentialRejected,
    PlanningDefect,
    ProtocolDefect,
    RuntimeDefect,
)
from provider_runtime.schema import parse_canonical_schema
from provider_runtime.transport import SseEvent
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    CanonicalTool,
    CodecStreamEvent,
    ConfirmedNonBillable,
    ContinuationArtifact,
    ContinuationDelta,
    Dynamic,
    GenerateIntent,
    GlobalScope,
    Incomplete,
    InvalidToolArguments,
    OutputSpec,
    PossiblyBillable,
    Present,
    PromptBlock,
    PromptMessage,
    ProviderContextTooLarge,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTarget,
    ReasoningLevel,
    Refused,
    Stable,
    StreamStart,
    StrictJsonOutput,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextContent,
    TextDelta,
    TextOutput,
    ToolCall,
    ToolCallDelta,
    ToolCallDone,
    ToolCallStart,
    ToolChoice,
    ToolResultMessage,
    UsageEvent,
    UserMessage,
)

FIXTURES = Path(__file__).parent / "fixtures" / "anthropic"

FABLE = ProviderTarget(provider="anthropic", model="claude-fable-5")
SONNET = ProviderTarget(provider="anthropic", model="claude-sonnet-5")
FABLE_CONTRACT = CATALOG.chat_contract(FABLE)
SONNET_CONTRACT = CATALOG.chat_contract(SONNET)

TOOL = CanonicalTool(
    name="get_weather",
    description="Get current weather",
    parameters=parse_canonical_schema(
        {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
            "additionalProperties": False,
        }
    ),
)
TOOL_SCHEMA_WIRE = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City name"}},
    "required": ["city"],
    "additionalProperties": False,
}

STRICT_SCHEMA = parse_canonical_schema(
    {
        "type": "object",
        "properties": {
            "person": {"$ref": "#/$defs/person"},
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["person", "note"],
        "additionalProperties": False,
        "$defs": {
            "person": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            }
        },
    }
)
STRICT_OUTPUT = StrictJsonOutput(name="extraction", schema=STRICT_SCHEMA)
STRICT_SCHEMA_WIRE = {
    "type": "object",
    "properties": {
        "person": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["person", "note"],
    "additionalProperties": False,
}


def _default_messages() -> tuple[PromptMessage, ...]:
    return (
        SystemMessage(
            blocks=(PromptBlock(text="You are terse.", stability=Stable(GlobalScope())),)
        ),
        UserMessage(blocks=(PromptBlock(text="hi", stability=Dynamic()),)),
    )


def _intent(
    *,
    target: ProviderTarget = FABLE,
    messages: tuple[PromptMessage, ...] | None = None,
    max_output_tokens: int = 1024,
    reasoning: ReasoningLevel = "high",
    tools: tuple[CanonicalTool, ...] = (),
    tool_choice: ToolChoice = "auto",
    output: OutputSpec | None = None,
) -> GenerateIntent:
    return GenerateIntent(
        target=target,
        messages=_default_messages() if messages is None else messages,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        tools=tools,
        tool_choice=tool_choice,
        output=TextOutput() if output is None else output,
    )


def _body(intent: GenerateIntent) -> dict[str, object]:
    contract = FABLE_CONTRACT if intent.target == FABLE else SONNET_CONTRACT
    return json.loads(anthropic.encode(intent, contract).body)


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# ---------------------------------------------------------------------------
# encode goldens


def test_encode_plain_text_golden() -> None:
    draft = anthropic.encode(_intent(), FABLE_CONTRACT)
    assert draft.url == "https://api.anthropic.com/v1/messages"
    assert draft.protocol == "anthropic_messages"
    assert dict(draft.safe_headers) == {
        "anthropic-version": "2023-06-01",
        "accept": "application/json",
    }
    assert draft.native_reasoning == "high"
    assert draft.provider_framing_overhead_tokens == FABLE_CONTRACT.provider_framing_overhead_tokens
    assert json.loads(draft.body) == {
        "model": "claude-fable-5",
        "max_tokens": 1024,
        "system": [
            {
                "type": "text",
                "text": "You are terse.",
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "output_config": {"effort": "high"},
        "cache_control": {"type": "ephemeral"},
    }


def test_encode_never_sends_sampling_or_thinking_fields() -> None:
    for target in (FABLE, SONNET):
        body = _body(_intent(target=target, reasoning="max"))
        for forbidden in ("temperature", "top_p", "top_k", "thinking", "stream", "output_format"):
            assert forbidden not in body, forbidden


def test_encode_breakpoint_extends_into_leading_stable_user_block() -> None:
    messages: tuple[PromptMessage, ...] = (
        SystemMessage(
            blocks=(PromptBlock(text="You are terse.", stability=Stable(GlobalScope())),)
        ),
        UserMessage(
            blocks=(
                PromptBlock(text="shared corpus", stability=Stable(GlobalScope())),
                PromptBlock(text="fresh question", stability=Dynamic()),
            )
        ),
    )
    body = _body(_intent(messages=messages))
    # System block is part of the stable prefix but NOT the last stable block.
    assert body["system"] == [{"type": "text", "text": "You are terse."}]
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "shared corpus",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
                {"type": "text", "text": "fresh question"},
            ],
        }
    ]
    # Top-level automatic caching stays a byte-constant.
    assert body["cache_control"] == {"type": "ephemeral"}


def test_encode_tools_and_tool_choice_golden() -> None:
    body = _body(_intent(tools=(TOOL,), tool_choice="auto"))
    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": TOOL_SCHEMA_WIRE,
        }
    ]
    assert body["tool_choice"] == {"type": "auto"}

    none_body = _body(_intent(tools=(TOOL,), tool_choice="none"))
    assert none_body["tool_choice"] == {"type": "none"}

    no_tools = _body(_intent())
    assert "tools" not in no_tools
    assert "tool_choice" not in no_tools


def test_encode_strict_output_merges_one_output_config() -> None:
    draft = anthropic.encode(
        _intent(target=SONNET, reasoning="medium", output=STRICT_OUTPUT), SONNET_CONTRACT
    )
    body = json.loads(draft.body)
    # ONE merged object carrying BOTH effort and format — the deleted
    # forced-tool path must not resurface as a tool or a beta header.
    assert body["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": STRICT_SCHEMA_WIRE},
    }
    assert "tools" not in body
    assert "output_format" not in body
    assert "anthropic-beta" not in {key.lower() for key in draft.safe_headers}


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_encode_reasoning_levels_identity(level: ReasoningLevel) -> None:
    for target, contract in ((FABLE, FABLE_CONTRACT), (SONNET, SONNET_CONTRACT)):
        draft = anthropic.encode(_intent(target=target, reasoning=level), contract)
        body = json.loads(draft.body)
        output_config = body["output_config"]
        assert isinstance(output_config, dict)
        assert output_config["effort"] == level
        assert draft.native_reasoning == level


def test_encode_unsupported_reasoning_level_is_planning_defect() -> None:
    with pytest.raises(PlanningDefect):
        anthropic.encode(_intent(reasoning="none"), FABLE_CONTRACT)


def _thinking_payload_blocks() -> list[dict[str, object]]:
    return [
        {"type": "thinking", "thinking": "Prior reasoning.", "signature": "sig-abc"},
        {"type": "redacted_thinking", "data": "opaque-bytes"},
    ]


def _continuation_artifact() -> ContinuationArtifact:
    return ContinuationArtifact(
        target=FABLE,
        codec_id=anthropic.CODEC_ID,
        opaque_payload={"blocks": _thinking_payload_blocks()},
    )


def test_encode_continuation_replays_thinking_blocks_first_verbatim() -> None:
    messages: tuple[PromptMessage, ...] = (
        *_default_messages(),
        AssistantMessage(
            text="Checking the weather.",
            tool_calls=(ToolCall(id="toolu_01A", name="get_weather", arguments={"city": "Paris"}),),
            continuation=Present(_continuation_artifact()),
        ),
        ToolResultMessage(call_id="toolu_01A", output="12C, rain", is_error=False),
        ToolResultMessage(call_id="toolu_01B", output="boom", is_error=True),
    )
    body = _body(_intent(messages=messages, tools=(TOOL,)))
    wire_messages = body["messages"]
    assert isinstance(wire_messages, list)
    assert wire_messages[1] == {
        "role": "assistant",
        "content": [
            *_thinking_payload_blocks(),
            {"type": "text", "text": "Checking the weather."},
            {
                "type": "tool_use",
                "id": "toolu_01A",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ],
    }
    # Consecutive tool results coalesce into ONE user turn of tool_result blocks.
    assert wire_messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01A",
                "content": "12C, rain",
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01B",
                "content": "boom",
                "is_error": True,
            },
        ],
    }


def test_encode_continuation_mismatch_is_planning_defect() -> None:
    wrong_codec = ContinuationArtifact(
        target=FABLE, codec_id="openai_responses", opaque_payload={"blocks": []}
    )
    wrong_target = ContinuationArtifact(
        target=SONNET, codec_id=anthropic.CODEC_ID, opaque_payload={"blocks": []}
    )
    for artifact in (wrong_codec, wrong_target):
        messages: tuple[PromptMessage, ...] = (
            *_default_messages(),
            AssistantMessage(text="x", tool_calls=(), continuation=Present(artifact)),
        )
        with pytest.raises(PlanningDefect):
            anthropic.encode(_intent(messages=messages), FABLE_CONTRACT)


def test_encode_invalid_continuation_payload_is_planning_defect() -> None:
    artifact = ContinuationArtifact(
        target=FABLE, codec_id=anthropic.CODEC_ID, opaque_payload={"blocks": "nope"}
    )
    messages: tuple[PromptMessage, ...] = (
        *_default_messages(),
        AssistantMessage(text="x", tool_calls=(), continuation=Present(artifact)),
    )
    with pytest.raises(PlanningDefect):
        anthropic.encode(_intent(messages=messages), FABLE_CONTRACT)


def test_encode_misplaced_system_message_is_planning_defect() -> None:
    messages: tuple[PromptMessage, ...] = (
        UserMessage(blocks=(PromptBlock(text="hi", stability=Stable(GlobalScope())),)),
        SystemMessage(blocks=(PromptBlock(text="late", stability=Dynamic()),)),
    )
    with pytest.raises(PlanningDefect):
        anthropic.encode(_intent(messages=messages), FABLE_CONTRACT)


def test_encode_drops_empty_dynamic_block_from_system_and_user_content() -> None:
    # Anthropic rejects empty text blocks; an empty dynamic block (e.g. an
    # empty context slot) must vanish from the wire, not become {"type":
    # "text","text":""}.
    messages: tuple[PromptMessage, ...] = (
        SystemMessage(
            blocks=(
                PromptBlock(text="You are terse.", stability=Stable(GlobalScope())),
                PromptBlock(text="", stability=Dynamic()),
            )
        ),
        UserMessage(
            blocks=(
                PromptBlock(text="", stability=Dynamic()),
                PromptBlock(text="hi", stability=Dynamic()),
            )
        ),
    )
    body = _body(_intent(messages=messages))
    assert body["system"] == [
        {
            "type": "text",
            "text": "You are terse.",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    ]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_encode_drops_trailing_empty_stable_block_breakpoint_on_last_nonempty() -> None:
    # A trailing empty STABLE block must not extend the leading run nor
    # receive the explicit breakpoint (the aggravating empty-cache_control
    # case): the breakpoint lands on the last NON-EMPTY stable wire block.
    messages: tuple[PromptMessage, ...] = (
        SystemMessage(
            blocks=(
                PromptBlock(text="Alpha.", stability=Stable(GlobalScope())),
                PromptBlock(text="", stability=Stable(GlobalScope())),
            )
        ),
        UserMessage(blocks=(PromptBlock(text="hi", stability=Dynamic()),)),
    )
    body = _body(_intent(messages=messages))
    assert body["system"] == [
        {
            "type": "text",
            "text": "Alpha.",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    ]


def test_encode_empty_user_message_is_planning_defect() -> None:
    # blocks=() produces zero wire blocks outright.
    messages: tuple[PromptMessage, ...] = (
        SystemMessage(
            blocks=(PromptBlock(text="You are terse.", stability=Stable(GlobalScope())),)
        ),
        UserMessage(blocks=()),
    )
    with pytest.raises(PlanningDefect) as excinfo:
        anthropic.encode(_intent(messages=messages), FABLE_CONTRACT)
    assert excinfo.value.code == "empty_message_content"

    # A non-empty blocks tuple whose sole block is empty text is equally zero
    # wire blocks after filtering.
    all_empty: tuple[PromptMessage, ...] = (
        SystemMessage(
            blocks=(PromptBlock(text="You are terse.", stability=Stable(GlobalScope())),)
        ),
        UserMessage(blocks=(PromptBlock(text="", stability=Dynamic()),)),
    )
    with pytest.raises(PlanningDefect) as excinfo:
        anthropic.encode(_intent(messages=all_empty), FABLE_CONTRACT)
    assert excinfo.value.code == "empty_message_content"


def test_encode_empty_assistant_turn_is_planning_defect() -> None:
    messages: tuple[PromptMessage, ...] = (
        *_default_messages(),
        AssistantMessage(text="", tool_calls=(), continuation=Absent()),
    )
    with pytest.raises(PlanningDefect) as excinfo:
        anthropic.encode(_intent(messages=messages), FABLE_CONTRACT)
    assert excinfo.value.code == "empty_assistant_turn"


def test_encode_whitespace_only_block_survives_to_wire() -> None:
    # Whitespace-only text is wire-legal on Anthropic; only truly empty text
    # ("") is dropped.
    messages: tuple[PromptMessage, ...] = (
        SystemMessage(
            blocks=(PromptBlock(text="You are terse.", stability=Stable(GlobalScope())),)
        ),
        UserMessage(blocks=(PromptBlock(text="   ", stability=Dynamic()),)),
    )
    body = _body(_intent(messages=messages))
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "   "}]}]


def test_prefix_bytes_determinism_and_sensitivity() -> None:
    base = anthropic.encode(_intent(), FABLE_CONTRACT)
    again = anthropic.encode(_intent(), FABLE_CONTRACT)
    assert base.prefix_bytes == again.prefix_bytes
    assert base.body == again.body

    with_tools = anthropic.encode(_intent(tools=(TOOL,)), FABLE_CONTRACT)
    assert with_tools.prefix_bytes != base.prefix_bytes

    with_format = anthropic.encode(_intent(output=STRICT_OUTPUT), FABLE_CONTRACT)
    assert with_format.prefix_bytes != base.prefix_bytes

    dynamic_changed: tuple[PromptMessage, ...] = (
        SystemMessage(
            blocks=(PromptBlock(text="You are terse.", stability=Stable(GlobalScope())),)
        ),
        UserMessage(blocks=(PromptBlock(text="a different question", stability=Dynamic()),)),
    )
    changed = anthropic.encode(_intent(messages=dynamic_changed), FABLE_CONTRACT)
    assert changed.prefix_bytes == base.prefix_bytes
    assert changed.body != base.body


def test_prefix_bytes_sensitive_to_role_move_system_vs_leading_user() -> None:
    # The same stable text placed in the top-level `system` field vs. a
    # leading `messages[]` user turn must produce distinct prefix_bytes.
    stable_text = "Same stable text."
    as_system = anthropic.encode(
        _intent(
            messages=(
                SystemMessage(
                    blocks=(PromptBlock(text=stable_text, stability=Stable(GlobalScope())),)
                ),
                UserMessage(blocks=(PromptBlock(text="hi", stability=Dynamic()),)),
            )
        ),
        FABLE_CONTRACT,
    ).prefix_bytes
    as_leading_user = anthropic.encode(
        _intent(
            messages=(
                UserMessage(
                    blocks=(PromptBlock(text=stable_text, stability=Stable(GlobalScope())),)
                ),
                UserMessage(blocks=(PromptBlock(text="hi", stability=Dynamic()),)),
            )
        ),
        FABLE_CONTRACT,
    ).prefix_bytes
    assert as_system != as_leading_user


def test_prefix_bytes_sensitive_to_message_regrouping() -> None:
    # Two stable blocks in ONE system message vs. split across a system
    # message and a leading user message must produce distinct prefix_bytes.
    one_message = anthropic.encode(
        _intent(
            messages=(
                SystemMessage(
                    blocks=(
                        PromptBlock(text="Alpha.", stability=Stable(GlobalScope())),
                        PromptBlock(text="Beta.", stability=Stable(GlobalScope())),
                    )
                ),
                UserMessage(blocks=(PromptBlock(text="hi", stability=Dynamic()),)),
            )
        ),
        FABLE_CONTRACT,
    ).prefix_bytes
    two_messages = anthropic.encode(
        _intent(
            messages=(
                SystemMessage(
                    blocks=(PromptBlock(text="Alpha.", stability=Stable(GlobalScope())),)
                ),
                UserMessage(blocks=(PromptBlock(text="Beta.", stability=Stable(GlobalScope())),)),
                UserMessage(blocks=(PromptBlock(text="hi", stability=Dynamic()),)),
            )
        ),
        FABLE_CONTRACT,
    ).prefix_bytes
    assert one_message != two_messages


def test_prefix_bytes_sensitive_to_stable_text_impersonating_tool_definition() -> None:
    # A stable block whose text is byte-equal to a real tool definition's dump
    # must never collide with an intent that actually declares that tool.
    tool_entry_json = json.dumps(
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": TOOL_SCHEMA_WIRE,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    impersonating = anthropic.encode(
        _intent(
            messages=(
                SystemMessage(
                    blocks=(PromptBlock(text=tool_entry_json, stability=Stable(GlobalScope())),)
                ),
                UserMessage(blocks=(PromptBlock(text="hi", stability=Dynamic()),)),
            )
        ),
        FABLE_CONTRACT,
    ).prefix_bytes
    with_matching_tool = anthropic.encode(_intent(tools=(TOOL,)), FABLE_CONTRACT).prefix_bytes
    assert impersonating != with_matching_tool


def test_finalize_is_passthrough() -> None:
    draft = anthropic.encode(_intent(), FABLE_CONTRACT)
    one = anthropic.finalize(draft, "affinity-one")
    two = anthropic.finalize(draft, "affinity-two")
    assert one.method == "POST"
    assert one.url == draft.url
    assert one.safe_headers == draft.safe_headers
    assert one.body == draft.body
    assert one.body == two.body


def test_stream_request_injects_stream_true_deterministically() -> None:
    finalized = anthropic.finalize(anthropic.encode(_intent(), FABLE_CONTRACT), "affinity")
    streamed_one = anthropic.stream_request(finalized)
    streamed_two = anthropic.stream_request(finalized)
    assert streamed_one.body == streamed_two.body
    assert json.loads(streamed_one.body)["stream"] is True
    # New value; the non-stream request is unchanged.
    assert "stream" not in json.loads(finalized.body)
    assert streamed_one.url == finalized.url
    assert streamed_one.method == finalized.method
    assert streamed_one.safe_headers == finalized.safe_headers
    # Identical serialization: the only byte difference is the appended field.
    assert streamed_one.body == finalized.body[:-1] + b',"stream":true}'


# ---------------------------------------------------------------------------
# decode_response


def test_decode_success_text_thinking_usage_and_request_id() -> None:
    outcome = anthropic.decode_response(
        200, {"request-id": "req_header_1"}, _fixture("success_text_thinking.json")
    )
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.provider == "anthropic"
    assert outcome.meta.model == "claude-fable-5"
    # Header-borne request id preferred over the in-band message id.
    assert outcome.meta.provider_request_id == Present("req_header_1")
    assert outcome.meta.billability == PossiblyBillable()
    assert outcome.meta.upstream_provider == Absent()
    assert outcome.meta.attempt_trace == ()
    usage = outcome.meta.usage
    assert isinstance(usage, Present)
    # Wire input_tokens=100 excludes cache; the codec normalizes to the
    # cache-INCLUSIVE convention: 100 + cache_read(30) + cache_write(20).
    assert usage.value.input_tokens == 150
    assert usage.value.output_tokens == 50
    assert usage.value.cache_read_input_tokens == Present(30)
    assert usage.value.cache_write_input_tokens == Present(20)
    # Anthropic reports no total: derived as inclusive input + output.
    assert usage.value.total_tokens == 150 + 50

    content = outcome.response.content
    assert isinstance(content, TextContent)
    assert content.text == "Hello! How can I help you today?"
    assert content.tool_calls == ()

    continuation = outcome.response.continuation
    assert isinstance(continuation, Present)
    artifact = continuation.value
    assert artifact.target == FABLE
    assert artifact.codec_id == "anthropic_messages"
    assert artifact.opaque_payload["blocks"] == [
        {
            "type": "thinking",
            "thinking": "The user greeted me; a short reply suffices.",
            "signature": "EqQBCgIYAhIkey-signature-bytes",
        },
        {"type": "redacted_thinking", "data": "opaque-redacted-bytes"},
    ]


def test_decode_success_in_band_request_id_fallback() -> None:
    outcome = anthropic.decode_response(200, {}, _fixture("success_text_thinking.json"))
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.provider_request_id == Present("msg_01Fable123")


def test_decode_success_tool_use() -> None:
    outcome = anthropic.decode_response(200, {}, _fixture("success_tool_use.json"))
    assert isinstance(outcome, Succeeded)
    content = outcome.response.content
    assert isinstance(content, TextContent)
    assert content.text == "Let me check the weather."
    assert content.tool_calls == (
        ToolCall(id="toolu_01A", name="get_weather", arguments={"city": "Paris"}),
    )
    assert outcome.response.continuation == Absent()


def test_decode_normalizes_input_tokens_to_cache_inclusive() -> None:
    # Regression for the cache-exclusive/inclusive cost-accounting blocker:
    # Anthropic's wire input_tokens EXCLUDES cache reads/writes, but
    # TokenUsage.input_tokens must always be the cache-INCLUSIVE total so
    # cost_from_accounting's cache subtraction bills the uncached input
    # correctly. A cache_read that exceeds the raw wire input_tokens is the
    # sharpest case: previously it floored billable input to 0.
    body = json.dumps(
        {
            "id": "msg_cache_heavy",
            "model": "claude-fable-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 2000,
                "output_tokens": 100,
            },
        }
    ).encode()
    outcome = anthropic.decode_response(200, {}, body)
    assert isinstance(outcome, Succeeded)
    usage = outcome.meta.usage
    assert isinstance(usage, Present)
    assert usage.value.input_tokens == 2500
    assert usage.value.cache_read_input_tokens == Present(2000)
    assert usage.value.cache_write_input_tokens == Present(0)
    assert usage.value.total_tokens == 2500 + 100


def test_decode_invalid_tool_input_raises_expected_failure() -> None:
    with pytest.raises(ExpectedFailureSignal) as excinfo:
        anthropic.decode_response(200, {}, _fixture("invalid_tool_input.json"))
    failure = excinfo.value.failure
    assert isinstance(failure, InvalidToolArguments)
    assert "get_weather" in failure.safe_detail


def test_decode_strict_output_response_returns_text_content_only() -> None:
    # Cross-codec rule: decode returns TextContent only — the strict-JSON text
    # rides TextContent.text and StructuredContent construction belongs to the
    # plan-owning runtime (decode signatures carry no plan).
    outcome = anthropic.decode_response(200, {}, _fixture("success_structured.json"))
    assert isinstance(outcome, Succeeded)
    content = outcome.response.content
    assert isinstance(content, TextContent)
    assert content.text == '{"person":{"name":"Ada"},"note":null}'
    assert json.loads(content.text) == {"person": {"name": "Ada"}, "note": None}
    # Thinking still yields a continuation artifact alongside strict output.
    assert isinstance(outcome.response.continuation, Present)


def test_decode_refusal_pre_output_confirmed_non_billable() -> None:
    outcome = anthropic.decode_response(200, {}, _fixture("refusal_pre_output.json"))
    assert isinstance(outcome, Refused)
    assert outcome.safe_detail == "Declined by safety classifiers before output."
    # usage.output_tokens == 0 → provider-confirmed unbilled.
    assert outcome.meta.billability == ConfirmedNonBillable()


def test_decode_refusal_with_billed_output_possibly_billable() -> None:
    outcome = anthropic.decode_response(200, {}, _fixture("refusal_mid_output.json"))
    assert isinstance(outcome, Refused)
    assert outcome.meta.billability == PossiblyBillable()
    assert outcome.safe_detail == "provider refusal"


def test_decode_max_tokens_incomplete() -> None:
    outcome = anthropic.decode_response(200, {}, _fixture("incomplete_max_tokens.json"))
    assert isinstance(outcome, Incomplete)
    assert outcome.reason == "max_output_tokens"
    assert outcome.status == "provider_incomplete"
    assert outcome.safe_detail == Absent()
    assert outcome.meta.billability == PossiblyBillable()


def test_decode_malformed_body_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        anthropic.decode_response(200, {}, b"not json at all")
    with pytest.raises(ProtocolDefect):
        anthropic.decode_response(200, {}, b'["array"]')


def test_decode_unknown_stop_reason_is_protocol_defect() -> None:
    body = json.dumps(
        {
            "id": "msg_x",
            "model": "claude-fable-5",
            "content": [],
            "stop_reason": "pause_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()
    with pytest.raises(ProtocolDefect):
        anthropic.decode_response(200, {}, body)


def test_thinking_round_trip_decode_then_encode_replays_verbatim() -> None:
    outcome = anthropic.decode_response(200, {}, _fixture("success_text_thinking.json"))
    assert isinstance(outcome, Succeeded)
    continuation = outcome.response.continuation
    assert isinstance(continuation, Present)
    content = outcome.response.content
    assert isinstance(content, TextContent)

    messages: tuple[PromptMessage, ...] = (
        *_default_messages(),
        AssistantMessage(text=content.text, tool_calls=(), continuation=continuation),
        UserMessage(blocks=(PromptBlock(text="and now?", stability=Dynamic()),)),
    )
    body = _body(_intent(messages=messages))
    wire_messages = body["messages"]
    assert isinstance(wire_messages, list)
    assistant_turn = wire_messages[1]
    assert assistant_turn["role"] == "assistant"
    fixture_content = json.loads(_fixture("success_text_thinking.json"))["content"]
    # Thinking/redacted blocks replay VERBATIM and lead the assistant turn.
    assert assistant_turn["content"][:2] == fixture_content[:2]
    assert assistant_turn["content"][2] == {
        "type": "text",
        "text": "Hello! How can I help you today?",
    }


# ---------------------------------------------------------------------------
# decode_stream


def _sse(data: object, event: str | None = None) -> SseEvent:
    return SseEvent(event=event, data=json.dumps(data))


async def _aiter(events: Sequence[SseEvent]) -> AsyncIterator[SseEvent]:
    for event in events:
        yield event


async def _collect(
    events: Sequence[SseEvent],
    *,
    headers: dict[str, str] | None = None,
) -> list[CodecStreamEvent]:
    iterator = anthropic.decode_stream(headers if headers is not None else {}, _aiter(events))
    return [event async for event in iterator]


def _message_start(*, model: str = "claude-fable-5", output_tokens: int = 0) -> SseEvent:
    return _sse(
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream_1",
                "model": model,
                "usage": {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "output_tokens": output_tokens,
                },
            },
        }
    )


def _happy_events() -> list[SseEvent]:
    return [
        _message_start(),
        _sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }
        ),
        _sse(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think."},
            }
        ),
        _sse(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig123"},
            }
        ),
        _sse({"type": "content_block_stop", "index": 0}),
        _sse(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        _sse(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Hello"},
            }
        ),
        _sse(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": " world"},
            }
        ),
        _sse({"type": "content_block_stop", "index": 1}),
        _sse({"type": "ping"}),
        _sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 42},
            }
        ),
        _sse({"type": "message_stop"}),
    ]


async def test_stream_happy_path_sequence_and_terminal() -> None:
    events = await _collect(_happy_events(), headers={"request-id": "req_stream_h"})
    assert isinstance(events[0], StreamStart)
    assert events[1] == TextDelta(text="Hello")
    assert events[2] == TextDelta(text=" world")
    assert isinstance(events[3], UsageEvent)

    continuation_delta = events[4]
    assert isinstance(continuation_delta, ContinuationDelta)
    assert continuation_delta.artifact.codec_id == "anthropic_messages"
    assert continuation_delta.artifact.target == FABLE
    assert continuation_delta.artifact.opaque_payload["blocks"] == [
        {"type": "thinking", "thinking": "Let me think.", "signature": "sig123"}
    ]
    # Exactly one ContinuationDelta, positioned before the terminal.
    assert sum(isinstance(event, ContinuationDelta) for event in events) == 1

    terminal = events[5]
    assert isinstance(terminal, TerminalEvent)
    assert len(events) == 6
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.provider_request_id == Present("req_stream_h")
    usage = outcome.meta.usage
    assert isinstance(usage, Present)
    # Wire input_tokens=100 excludes cache; normalized to cache-inclusive.
    assert usage.value.input_tokens == 150
    assert usage.value.output_tokens == 42
    assert usage.value.cache_read_input_tokens == Present(30)
    assert usage.value.cache_write_input_tokens == Present(20)
    assert usage.value.total_tokens == 150 + 42
    content = outcome.response.content
    assert isinstance(content, TextContent)
    assert content.text == "Hello world"
    assert outcome.response.continuation == Present(continuation_delta.artifact)


async def test_stream_in_band_request_id_fallback() -> None:
    events = await _collect(_happy_events())
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert terminal.outcome.meta.provider_request_id == Present("msg_stream_1")


async def test_stream_tool_call_events() -> None:
    events = await _collect(
        [
            _message_start(),
            _sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_s1",
                        "name": "get_weather",
                        "input": {},
                    },
                }
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
                }
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'},
                }
            ),
            _sse({"type": "content_block_stop", "index": 0}),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 12},
                }
            ),
            _sse({"type": "message_stop"}),
        ]
    )
    assert events[1] == ToolCallStart(call_id="toolu_s1", name="get_weather")
    assert events[2] == ToolCallDelta(call_id="toolu_s1", arguments_delta='{"city":')
    assert events[3] == ToolCallDelta(call_id="toolu_s1", arguments_delta='"Paris"}')
    done = events[4]
    assert isinstance(done, ToolCallDone)
    assert done.tool_call == ToolCall(
        id="toolu_s1", name="get_weather", arguments={"city": "Paris"}
    )
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    content = outcome.response.content
    assert isinstance(content, TextContent)
    assert content.tool_calls == (done.tool_call,)


async def test_stream_invalid_tool_arguments_raise_expected_failure() -> None:
    with pytest.raises(ExpectedFailureSignal) as excinfo:
        await _collect(
            [
                _message_start(),
                _sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_bad",
                            "name": "get_weather",
                            "input": {},
                        },
                    }
                ),
                _sse(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "input_json_delta", "partial_json": "not json"},
                    }
                ),
                _sse({"type": "content_block_stop", "index": 0}),
            ]
        )
    assert isinstance(excinfo.value.failure, InvalidToolArguments)


async def test_stream_refusal_is_incomplete_refused_never_refused_terminal() -> None:
    events = await _collect(
        [
            _message_start(),
            _sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                }
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "partial"},
                }
            ),
            _sse({"type": "content_block_stop", "index": 0}),
            _sse(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                }
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "I can"},
                }
            ),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "refusal",
                        "stop_details": {
                            "type": "refusal",
                            "category": "cyber",
                            "explanation": "Declined for safety.",
                        },
                    },
                    "usage": {"output_tokens": 0},
                }
            ),
            _sse({"type": "message_stop"}),
        ]
    )
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    # The four-kind stream grammar: NEVER Refused on a stream.
    assert isinstance(outcome, Incomplete)
    assert outcome.status == "refused"
    assert outcome.reason == "content_filter_partial"
    assert outcome.safe_detail == Present("Declined for safety.")
    # Pre-output refusal (zero billed output tokens) → confirmed unbilled.
    assert outcome.meta.billability == ConfirmedNonBillable()
    # Partial output is invalidated: no ContinuationDelta for a refused stream.
    assert not any(isinstance(event, ContinuationDelta) for event in events)


async def test_stream_refusal_with_billed_output_possibly_billable() -> None:
    events = await _collect(
        [
            _message_start(),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "refusal"},
                    "usage": {"output_tokens": 17},
                }
            ),
            _sse({"type": "message_stop"}),
        ]
    )
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Incomplete)
    assert outcome.status == "refused"
    assert outcome.safe_detail == Present("provider refusal")
    assert outcome.meta.billability == PossiblyBillable()


async def test_stream_max_tokens_terminal() -> None:
    events = await _collect(
        [
            _message_start(),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "max_tokens"},
                    "usage": {"output_tokens": 128},
                }
            ),
            _sse({"type": "message_stop"}),
        ]
    )
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Incomplete)
    assert outcome.reason == "max_output_tokens"
    assert outcome.status == "provider_incomplete"
    assert outcome.safe_detail == Absent()


async def test_stream_strict_output_terminal_is_text_content() -> None:
    # Cross-codec rule: the stream terminal also carries TextContent only; the
    # accumulated strict-JSON text is the terminal text.
    events = await _collect(
        [
            _message_start(model="claude-sonnet-5"),
            _sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": '{"person":{"name":"Ada"},'},
                }
            ),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": '"note":null}'},
                }
            ),
            _sse({"type": "content_block_stop", "index": 0}),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 18},
                }
            ),
            _sse({"type": "message_stop"}),
        ],
    )
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    content = outcome.response.content
    assert isinstance(content, TextContent)
    assert content.text == '{"person":{"name":"Ada"},"note":null}'
    assert json.loads(content.text) == {"person": {"name": "Ada"}, "note": None}


async def test_stream_missing_message_stop_is_transient() -> None:
    events = _happy_events()[:-1]  # drop message_stop
    with pytest.raises(TransientStreamError) as excinfo:
        await _collect(events)
    assert excinfo.value.cause == ProviderStreamInterrupted(partial_output=False)


async def test_stream_overloaded_error_event_is_transient() -> None:
    with pytest.raises(TransientStreamError) as excinfo:
        await _collect(
            [
                _message_start(),
                _sse({"type": "error", "error": {"type": "overloaded_error", "message": "Busy"}}),
            ]
        )
    assert excinfo.value.cause == ProviderHttpUnavailable()


async def test_stream_non_transient_error_event_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        await _collect(
            [
                _message_start(),
                _sse(
                    {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "bad"},
                    }
                ),
            ]
        )


async def test_stream_malformed_event_data_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        await _collect([SseEvent(event="message_start", data="{not json")])


# ---------------------------------------------------------------------------
# classify_error


def _error_body(error_type: str, message: str) -> bytes:
    return json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode()


def test_classify_429_rate_limited_with_retry_after() -> None:
    classified = anthropic.classify_error(
        429, {"retry-after": "12"}, _error_body("rate_limit_error", "slow down")
    )
    assert classified == ProviderRateLimit(retry_after=Present(12.0))


def test_classify_429_without_retry_after_header() -> None:
    classified = anthropic.classify_error(429, {}, _error_body("rate_limit_error", "slow down"))
    assert classified == ProviderRateLimit(retry_after=Absent())


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529])
def test_classify_5xx_and_529_unavailable(status: int) -> None:
    classified = anthropic.classify_error(status, {}, _error_body("api_error", "boom"))
    assert classified == ProviderHttpUnavailable()


def test_classify_overloaded_body_shape_is_unavailable() -> None:
    classified = anthropic.classify_error(529, {}, _error_body("overloaded_error", "Overloaded"))
    assert classified == ProviderHttpUnavailable()


def test_classify_context_too_large() -> None:
    assert (
        anthropic.classify_error(413, {}, _error_body("request_too_large", "32MB max"))
        == ProviderContextTooLarge()
    )
    assert (
        anthropic.classify_error(
            400,
            {},
            _error_body("invalid_request_error", "prompt is too long: 1200000 tokens > 1000000"),
        )
        == ProviderContextTooLarge()
    )


@pytest.mark.parametrize("status", [401, 403])
def test_classify_credential_rejected_raises(status: int) -> None:
    with pytest.raises(CredentialRejected):
        anthropic.classify_error(
            status, {}, _error_body("authentication_error", "invalid x-api-key")
        )


def test_classify_credit_exhausted_is_quota_defect() -> None:
    with pytest.raises(RuntimeDefect) as excinfo:
        anthropic.classify_error(
            400,
            {},
            _error_body(
                "invalid_request_error",
                "Your credit balance is too low to access the Anthropic API.",
            ),
        )
    assert excinfo.value.code == "quota_exhausted"
    assert excinfo.value.origin == "provider_http"


def test_classify_other_400_is_unclassified_defect() -> None:
    with pytest.raises(RuntimeDefect) as excinfo:
        anthropic.classify_error(400, {}, _error_body("invalid_request_error", "bad field"))
    assert excinfo.value.code == "unclassified_provider_error"
    assert excinfo.value.origin == "provider_http"


def test_classify_defect_details_redact_secrets() -> None:
    with pytest.raises(RuntimeDefect) as excinfo:
        anthropic.classify_error(
            400,
            {},
            _error_body("invalid_request_error", "bad key sk-abcdefghijklmnop provided"),
        )
    assert "sk-abcdefghijklmnop" not in excinfo.value.message
    assert "...redacted" in excinfo.value.message
