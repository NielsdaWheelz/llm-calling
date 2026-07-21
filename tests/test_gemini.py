"""Tests for the rewritten Gemini generateContent codec (spec §7 Gemini row).

Coverage per the codec-seam test list plus Gemini specifics: no-cache-fields
golden, thinkingLevel per level, unstripped native JSON schema
(additionalProperties present on the wire), thoughtSignature replay round-trip,
synthesized call ids mapped back to coalesced functionResponse parts, stream
chunk accumulation, SAFETY mapping, and always-Absent request ids.
"""

import json
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest

from provider_runtime import gemini
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
    ToolCallDone,
    ToolCallStart,
    ToolChoice,
    ToolResultMessage,
    UserMessage,
)

FIXTURES = Path(__file__).parent / "fixtures" / "gemini"
TARGET = ProviderTarget(provider="gemini", model="gemini-3.5-flash")
CONTRACT = CATALOG.chat_contract(TARGET)

SYSTEM = SystemMessage(
    blocks=(PromptBlock(text="You are a librarian.", stability=Stable(scope=GlobalScope())),)
)
USER = UserMessage(blocks=(PromptBlock(text="Hello", stability=Dynamic()),))

TOOL_SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
        "library_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["query", "library_id"],
    "additionalProperties": False,
}
SEARCH_TOOL = CanonicalTool(
    name="app_search",
    description="Search the library",
    parameters=parse_canonical_schema(TOOL_SCHEMA_JSON),
)
_TEXT_OUTPUT = TextOutput()


def _intent(
    messages: Iterable[PromptMessage] = (SYSTEM, USER),
    *,
    reasoning: ReasoningLevel = "medium",
    tools: Iterable[CanonicalTool] = (),
    tool_choice: ToolChoice = "auto",
    output: OutputSpec = _TEXT_OUTPUT,
    max_output_tokens: int = 1024,
) -> GenerateIntent:
    return GenerateIntent(
        target=TARGET,
        messages=tuple(messages),
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        tools=tuple(tools),
        tool_choice=tool_choice,
        output=output,
    )


def _body(intent: GenerateIntent) -> dict:
    return json.loads(gemini.encode(intent, CONTRACT).body)


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _decode_fixture(name: str):
    return gemini.decode_response(200, {}, _fixture_bytes(name))


def _sse(data: str) -> SseEvent:
    return SseEvent(event=None, data=data)


def _fixture_stream_events() -> list[SseEvent]:
    lines = (FIXTURES / "success_stream_chunks.txt").read_text().splitlines()
    return [_sse(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]


async def _aiter(events: Iterable[SseEvent]) -> AsyncIterator[SseEvent]:
    for event in events:
        yield event


async def _collect(events: Iterable[SseEvent]):
    return [event async for event in gemini.decode_stream({}, _aiter(events))]


# ---------------------------------------------------------------------------
# encode goldens


def test_encode_plain_text_golden_and_no_cache_fields():
    draft = gemini.encode(_intent(), CONTRACT)
    assert draft.url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
    )
    assert dict(draft.safe_headers) == {}
    assert draft.native_reasoning == "medium"
    assert json.loads(draft.body) == {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "systemInstruction": {"parts": [{"text": "You are a librarian."}]},
        "generationConfig": {
            "maxOutputTokens": 1024,
            "thinkingConfig": {"thinkingLevel": "medium"},
        },
    }
    # Implicit caching only: NO cache control of any kind reaches the wire.
    assert "cache" not in draft.body.decode().lower()


@pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
def test_encode_thinking_level_per_level(level: ReasoningLevel):
    draft = gemini.encode(_intent(reasoning=level), CONTRACT)
    body = json.loads(draft.body)
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": level}
    assert draft.native_reasoning == level


def test_encode_unsupported_reasoning_level_is_planning_defect():
    with pytest.raises(PlanningDefect):
        gemini.encode(_intent(reasoning="max"), CONTRACT)


def test_encode_tools_native_json_schema_unstripped():
    body = _body(_intent(tools=(SEARCH_TOOL,)))
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "app_search",
                    "description": "Search the library",
                    # JSON-Schema-native and UNSTRIPPED: additionalProperties,
                    # required, the nullable anyOf union, and annotations all
                    # survive (the old keyword stripping is dead).
                    "parametersJsonSchema": TOOL_SCHEMA_JSON,
                }
            ]
        }
    ]
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}


def test_encode_tool_choice_none():
    body = _body(_intent(tools=(SEARCH_TOOL,), tool_choice="none"))
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}


def test_encode_without_tools_sends_no_tool_fields():
    body = _body(_intent())
    assert "tools" not in body
    assert "toolConfig" not in body


def test_encode_strict_output_response_json_schema():
    schema = parse_canonical_schema(
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "topic": {"$ref": "#/$defs/topic"},
            },
            "required": ["summary", "topic"],
            "additionalProperties": False,
            "$defs": {"topic": {"type": "string", "enum": ["tides", "moons"]}},
        }
    )
    body = _body(_intent(output=StrictJsonOutput(name="summary", schema=schema)))
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    # Defs are inlined; nothing is stripped or rewritten.
    assert config["responseJsonSchema"] == {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "topic": {"type": "string", "enum": ["tides", "moons"]},
        },
        "required": ["summary", "topic"],
        "additionalProperties": False,
    }


def test_encode_tools_with_strict_output_is_planning_defect():
    strict = StrictJsonOutput(name="x", schema=SEARCH_TOOL.parameters)
    with pytest.raises(PlanningDefect):
        gemini.encode(_intent(tools=(SEARCH_TOOL,), output=strict), CONTRACT)


# ---------------------------------------------------------------------------
# continuation replay + functionResponse coalescing


def _tooled_conversation() -> tuple[GenerateIntent, list[dict]]:
    outcome = _decode_fixture("success_nonstream_tools.json")
    assert isinstance(outcome, Succeeded)
    assert isinstance(outcome.response.content, TextContent)
    assistant = AssistantMessage(
        text=outcome.response.content.text,
        tool_calls=outcome.response.content.tool_calls,
        continuation=outcome.response.continuation,
    )
    intent = _intent(
        (
            SYSTEM,
            USER,
            assistant,
            ToolResultMessage(call_id="call_0", output="42 results", is_error=False),
            ToolResultMessage(call_id="call_1", output="boom", is_error=True),
        ),
        tools=(SEARCH_TOOL,),
    )
    fixture_parts = json.loads(_fixture_bytes("success_nonstream_tools.json"))["candidates"][0][
        "content"
    ]["parts"]
    return intent, fixture_parts


def test_continuation_replay_round_trip_verbatim_parts():
    intent, fixture_parts = _tooled_conversation()
    body = _body(intent)
    model_turn = body["contents"][1]
    # The prior model turn is replayed verbatim — every part, thoughtSignatures
    # included, never merged or rebuilt.
    assert model_turn == {"role": "model", "parts": fixture_parts}


def test_tool_results_coalesce_into_one_user_turn_by_synthesized_call_id():
    intent, _ = _tooled_conversation()
    body = _body(intent)
    assert len(body["contents"]) == 3  # user, model, coalesced tool-result turn
    assert body["contents"][2] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "app_search",
                    "response": {"output": "42 results"},
                }
            },
            {
                "functionResponse": {
                    "name": "other_tool",
                    "response": {"error": "boom"},
                }
            },
        ],
    }


def test_unknown_tool_result_call_id_is_planning_defect():
    intent, _ = _tooled_conversation()
    messages = intent.messages[:-1] + (
        ToolResultMessage(call_id="call_9", output="?", is_error=False),
    )
    with pytest.raises(PlanningDefect):
        gemini.encode(_intent(messages, tools=(SEARCH_TOOL,)), CONTRACT)


def test_tool_result_names_resolve_turn_scoped_across_colliding_synthesized_ids():
    # decode synthesizes call_0, call_1, ... per response, so a two-iteration
    # tool loop has "call_0" recur in every turn. A flat intent-wide name map
    # would let the second turn's call_0 -> read_resource overwrite the
    # first's call_0 -> app_search, mis-naming the first functionResponse.
    first_assistant = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="call_0", name="app_search", arguments={"query": "x"}),),
        continuation=Absent(),
    )
    second_assistant = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="call_0", name="read_resource", arguments={"id": "42"}),),
        continuation=Absent(),
    )
    intent = _intent(
        (
            SYSTEM,
            USER,
            first_assistant,
            ToolResultMessage(call_id="call_0", output="found 3", is_error=False),
            second_assistant,
            ToolResultMessage(call_id="call_0", output="resource body", is_error=False),
        ),
        tools=(SEARCH_TOOL,),
    )
    contents = _body(intent)["contents"]
    assert contents[2] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "app_search", "response": {"output": "found 3"}}}],
    }
    assert contents[4] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "read_resource",
                    "response": {"output": "resource body"},
                }
            }
        ],
    }


def test_tool_result_referencing_non_adjacent_turn_is_planning_defect():
    first_assistant = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="call_0", name="app_search", arguments={}),),
        continuation=Absent(),
    )
    second_assistant = AssistantMessage(text="done", tool_calls=(), continuation=Absent())
    intent = _intent(
        (
            SYSTEM,
            USER,
            first_assistant,
            second_assistant,
            ToolResultMessage(call_id="call_0", output="stale", is_error=False),
        ),
        tools=(SEARCH_TOOL,),
    )
    with pytest.raises(PlanningDefect):
        gemini.encode(intent, CONTRACT)


def test_duplicate_tool_call_id_within_turn_is_planning_defect():
    assistant = AssistantMessage(
        text="",
        tool_calls=(
            ToolCall(id="call_0", name="app_search", arguments={}),
            ToolCall(id="call_0", name="other_tool", arguments={}),
        ),
        continuation=Absent(),
    )
    with pytest.raises(PlanningDefect):
        gemini.encode(_intent((SYSTEM, USER, assistant), tools=(SEARCH_TOOL,)), CONTRACT)


def test_assistant_without_continuation_encodes_typed_fields():
    assistant = AssistantMessage(
        text="Searching.",
        tool_calls=(ToolCall(id="call_0", name="app_search", arguments={"query": "x"}),),
        continuation=Absent(),
    )
    body = _body(_intent((SYSTEM, USER, assistant), tools=(SEARCH_TOOL,)))
    assert body["contents"][1] == {
        "role": "model",
        "parts": [
            {"text": "Searching."},
            {"functionCall": {"name": "app_search", "args": {"query": "x"}}},
        ],
    }


def test_continuation_codec_mismatch_is_planning_defect():
    artifact = ContinuationArtifact(
        target=TARGET, codec_id="openai_responses", opaque_payload={"parts": []}
    )
    assistant = AssistantMessage(text="", tool_calls=(), continuation=Present(artifact))
    with pytest.raises(PlanningDefect):
        gemini.encode(_intent((SYSTEM, USER, assistant)), CONTRACT)


def test_continuation_target_mismatch_is_planning_defect():
    artifact = ContinuationArtifact(
        target=ProviderTarget(provider="gemini", model="gemini-other"),
        codec_id=gemini.CODEC_ID,
        opaque_payload={"parts": []},
    )
    assistant = AssistantMessage(text="", tool_calls=(), continuation=Present(artifact))
    with pytest.raises(PlanningDefect):
        gemini.encode(_intent((SYSTEM, USER, assistant)), CONTRACT)


def test_continuation_typed_call_mismatch_is_planning_defect():
    artifact = ContinuationArtifact(
        target=TARGET,
        codec_id=gemini.CODEC_ID,
        opaque_payload={"parts": [{"functionCall": {"name": "app_search", "args": {}}}]},
    )
    assistant = AssistantMessage(text="", tool_calls=(), continuation=Present(artifact))
    with pytest.raises(PlanningDefect):
        gemini.encode(_intent((SYSTEM, USER, assistant)), CONTRACT)


# ---------------------------------------------------------------------------
# finalize / stream_request / prefix_bytes


def test_finalize_is_passthrough():
    draft = gemini.encode(_intent(), CONTRACT)
    final = gemini.finalize(draft, "affinity-abc123")
    assert final.method == "POST"
    assert final.url == draft.url
    assert final.body == draft.body
    assert dict(final.safe_headers) == dict(draft.safe_headers)
    assert b"affinity-abc123" not in final.body


def test_stream_request_derives_alt_sse_url_only():
    final = gemini.finalize(gemini.encode(_intent(), CONTRACT), "aff")
    streamed = gemini.stream_request(final)
    assert streamed.url == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash:streamGenerateContent?alt=sse"
    )
    assert streamed.body == final.body  # byte-identical body; URL selects streaming
    assert streamed.method == "POST"
    assert final.url.endswith(":generateContent")  # original value untouched


def test_stream_request_rejects_foreign_url():
    final = gemini.finalize(gemini.encode(_intent(), CONTRACT), "aff")
    foreign = gemini.stream_request(final)  # already the stream variant
    with pytest.raises(PlanningDefect):
        gemini.stream_request(foreign)


def test_prefix_bytes_deterministic_and_sensitive():
    base = gemini.encode(_intent(), CONTRACT).prefix_bytes
    assert base == gemini.encode(_intent(), CONTRACT).prefix_bytes
    assert base != b""

    # Dynamic-block changes do NOT touch the affinity input...
    other_user = UserMessage(blocks=(PromptBlock(text="Different", stability=Dynamic()),))
    assert gemini.encode(_intent((SYSTEM, other_user)), CONTRACT).prefix_bytes == base

    # ...stable-block changes do...
    other_system = SystemMessage(
        blocks=(PromptBlock(text="You are a pirate.", stability=Stable(scope=GlobalScope())),)
    )
    assert gemini.encode(_intent((other_system, USER)), CONTRACT).prefix_bytes != base

    # ...and so do tools and the output schema (both live in the cache prefix).
    assert gemini.encode(_intent(tools=(SEARCH_TOOL,)), CONTRACT).prefix_bytes != base
    strict = StrictJsonOutput(name="x", schema=SEARCH_TOOL.parameters)
    assert gemini.encode(_intent(output=strict), CONTRACT).prefix_bytes != base


def test_prefix_bytes_sensitive_to_role_move_system_vs_leading_user():
    # The same stable text placed in systemInstruction vs. a leading user
    # content turn must produce distinct prefix_bytes.
    stable_text = "Same stable text."
    as_system = gemini.encode(
        _intent(
            (
                SystemMessage(
                    blocks=(PromptBlock(text=stable_text, stability=Stable(scope=GlobalScope())),)
                ),
                USER,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    as_leading_user = gemini.encode(
        _intent(
            (
                UserMessage(
                    blocks=(PromptBlock(text=stable_text, stability=Stable(scope=GlobalScope())),)
                ),
                USER,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    assert as_system != as_leading_user


def test_prefix_bytes_sensitive_to_message_regrouping():
    # Two stable blocks in ONE system message vs. split across a system
    # message and a leading user message must produce distinct prefix_bytes.
    one_message = gemini.encode(
        _intent(
            (
                SystemMessage(
                    blocks=(
                        PromptBlock(text="Alpha.", stability=Stable(scope=GlobalScope())),
                        PromptBlock(text="Beta.", stability=Stable(scope=GlobalScope())),
                    )
                ),
                USER,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    two_messages = gemini.encode(
        _intent(
            (
                SystemMessage(
                    blocks=(PromptBlock(text="Alpha.", stability=Stable(scope=GlobalScope())),)
                ),
                UserMessage(
                    blocks=(PromptBlock(text="Beta.", stability=Stable(scope=GlobalScope())),)
                ),
                USER,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    assert one_message != two_messages


def test_prefix_bytes_sensitive_to_stable_text_impersonating_tool_declaration():
    # A stable block whose text is byte-equal to a real tool declaration's
    # dump must never collide with an intent that actually declares that tool.
    declaration_json = json.dumps(
        {
            "name": "app_search",
            "description": "Search the library",
            "parametersJsonSchema": TOOL_SCHEMA_JSON,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    impersonating = gemini.encode(
        _intent(
            (
                SystemMessage(
                    blocks=(
                        PromptBlock(text=declaration_json, stability=Stable(scope=GlobalScope())),
                    )
                ),
                USER,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    with_matching_tool = gemini.encode(_intent(tools=(SEARCH_TOOL,)), CONTRACT).prefix_bytes
    assert impersonating != with_matching_tool


# ---------------------------------------------------------------------------
# decode_response


def test_decode_success_text_skips_thought_parts_and_folds_usage():
    outcome = _decode_fixture("success_nonstream.json")
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(
        text="Tides are caused by the Moon.", tool_calls=()
    )
    assert outcome.response.continuation == Absent()
    meta = outcome.meta
    assert meta.provider == "gemini"
    assert meta.model == "gemini-3.5-flash"
    assert meta.provider_request_id == Absent()  # ALWAYS Absent on this wire
    assert meta.upstream_provider == Absent()
    assert meta.attempt_trace == ()
    assert meta.billability == PossiblyBillable()
    assert isinstance(meta.usage, Present)
    usage = meta.usage.value
    assert usage.input_tokens == 10
    assert usage.output_tokens == 8
    assert usage.total_tokens == 25  # reported total is authoritative
    assert usage.reasoning_tokens == Present(7)
    assert usage.cache_read_input_tokens == Present(4)
    assert usage.cache_write_input_tokens == Absent()


def test_decode_tool_calls_synthesize_ids_and_capture_signatures():
    outcome = _decode_fixture("success_nonstream_tools.json")
    assert isinstance(outcome, Succeeded)
    content = outcome.response.content
    assert isinstance(content, TextContent)
    assert content.text == "Searching now."
    assert content.tool_calls == (
        ToolCall(id="call_0", name="app_search", arguments={"query": "tides", "limit": 3}),
        ToolCall(id="call_1", name="other_tool", arguments={}),
    )
    continuation = outcome.response.continuation
    assert isinstance(continuation, Present)
    artifact = continuation.value
    assert artifact.target == TARGET
    assert artifact.codec_id == "gemini_generate_content"
    fixture_parts = json.loads(_fixture_bytes("success_nonstream_tools.json"))["candidates"][0][
        "content"
    ]["parts"]
    assert artifact.opaque_payload == {"parts": fixture_parts}


def test_decode_non_object_tool_args_raises_expected_failure():
    body = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "app_search", "args": [1, 2]}}]
                    },
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3.5-flash",
        }
    ).encode()
    with pytest.raises(ExpectedFailureSignal) as excinfo:
        gemini.decode_response(200, {}, body)
    assert isinstance(excinfo.value.failure, InvalidToolArguments)


def _terminal_body(finish_reason: str) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {"content": {"parts": [{"text": "partial"}]}, "finishReason": finish_reason}
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
            "modelVersion": "gemini-3.5-flash",
        }
    ).encode()


def test_decode_max_tokens_is_incomplete():
    outcome = gemini.decode_response(200, {}, _terminal_body("MAX_TOKENS"))
    assert isinstance(outcome, Incomplete)
    assert outcome.reason == "max_output_tokens"
    assert outcome.status == "provider_incomplete"
    assert isinstance(outcome.meta.usage, Present)


@pytest.mark.parametrize("finish_reason", ["SAFETY", "PROHIBITED_CONTENT"])
def test_decode_safety_block_is_content_filter_incomplete(finish_reason: str):
    outcome = gemini.decode_response(200, {}, _terminal_body(finish_reason))
    assert isinstance(outcome, Incomplete)
    assert outcome.reason == "content_filter_partial"
    assert outcome.status == "provider_incomplete"
    assert isinstance(outcome.safe_detail, Present)
    assert finish_reason in outcome.safe_detail.value


def test_decode_prompt_block_without_candidates_is_content_filter_incomplete():
    body = json.dumps(
        {
            "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
            "usageMetadata": {"promptTokenCount": 5, "totalTokenCount": 5},
        }
    ).encode()
    outcome = gemini.decode_response(200, {}, body)
    assert isinstance(outcome, Incomplete)
    assert outcome.reason == "content_filter_partial"
    assert outcome.status == "provider_incomplete"


def test_decode_malformed_function_call_finish_reason_is_expected_failure():
    with pytest.raises(ExpectedFailureSignal) as excinfo:
        gemini.decode_response(200, {}, _terminal_body("MALFORMED_FUNCTION_CALL"))
    assert isinstance(excinfo.value.failure, InvalidToolArguments)


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b'{"candidates": []}',
        b'{"no_candidates": true}',
        json.dumps({"candidates": [{"content": {"parts": []}}]}).encode(),  # no finishReason
    ],
    ids=["malformed-json", "empty-candidates", "missing-candidates", "missing-finish-reason"],
)
def test_decode_malformed_envelopes_raise_protocol_defect(body: bytes):
    with pytest.raises(ProtocolDefect):
        gemini.decode_response(200, {}, body)


def test_decode_unknown_finish_reason_is_protocol_defect():
    with pytest.raises(ProtocolDefect):
        gemini.decode_response(200, {}, _terminal_body("SOMETHING_NEW"))


# ---------------------------------------------------------------------------
# decode_stream


async def test_stream_happy_path_accumulates_chunks():
    events = await _collect(_fixture_stream_events())
    assert events[0] == StreamStart()
    assert events[1] == TextDelta(text="Let me")
    assert events[2] == TextDelta(text=" search.")
    assert events[3] == ToolCallStart(call_id="call_0", name="app_search")
    expected_call = ToolCall(id="call_0", name="app_search", arguments={"query": "tides"})
    assert events[4] == ToolCallDone(tool_call=expected_call)
    assert isinstance(events[5], ContinuationDelta)
    terminal = events[6]
    assert isinstance(terminal, TerminalEvent)
    assert len(events) == 7

    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(
        text="Let me search.", tool_calls=(expected_call,)
    )
    # Exactly one ContinuationDelta, complete, before the terminal.
    continuation = outcome.response.continuation
    assert isinstance(continuation, Present)
    assert continuation.value == events[5].artifact
    payload_parts = continuation.value.opaque_payload["parts"]
    assert isinstance(payload_parts, list)
    assert {"text": "Let me"} in payload_parts
    assert any("thoughtSignature" in part for part in payload_parts)

    meta = outcome.meta
    assert meta.provider_request_id == Absent()
    assert isinstance(meta.usage, Present)
    assert meta.usage.value.total_tokens == 27  # folded from the final frame
    assert meta.usage.value.reasoning_tokens == Present(6)


async def test_stream_safety_terminal_is_content_filter_incomplete():
    events = await _collect(
        [
            _sse('{"candidates":[{"content":{"parts":[{"text":"par"}]},"index":0}]}'),
            _sse(
                '{"candidates":[{"content":{"parts":[]},"finishReason":"SAFETY"}],'
                '"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":1,'
                '"totalTokenCount":5}}'
            ),
        ]
    )
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert isinstance(terminal.outcome, Incomplete)
    assert terminal.outcome.reason == "content_filter_partial"
    assert terminal.outcome.status == "provider_incomplete"


async def test_stream_max_tokens_terminal():
    events = await _collect(
        [_sse('{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"MAX_TOKENS"}]}')]
    )
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert isinstance(terminal.outcome, Incomplete)
    assert terminal.outcome.reason == "max_output_tokens"


async def test_stream_missing_terminal_raises_transient_stream_error():
    with pytest.raises(TransientStreamError) as excinfo:
        await _collect([_sse('{"candidates":[{"content":{"parts":[{"text":"x"}]}}]}')])
    assert excinfo.value.cause == ProviderStreamInterrupted(partial_output=False)


async def test_stream_malformed_frame_raises_protocol_defect():
    with pytest.raises(ProtocolDefect):
        await _collect([_sse("not json")])


# ---------------------------------------------------------------------------
# classify_error


def test_classify_429_rate_limit_with_and_without_retry_after():
    body = _fixture_bytes("error_429.json")
    assert gemini.classify_error(429, {"retry-after": "2.5"}, body) == ProviderRateLimit(
        retry_after=Present(2.5)
    )
    assert gemini.classify_error(429, {}, body) == ProviderRateLimit(retry_after=Absent())


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_classify_5xx_unavailable(status: int):
    body = _fixture_bytes("error_500.json") if status == 500 else b"upstream hiccup"
    assert gemini.classify_error(status, {}, body) == ProviderHttpUnavailable()


def test_classify_rpc_status_unavailable_and_deadline():
    for rpc in ["UNAVAILABLE", "DEADLINE_EXCEEDED"]:
        body = json.dumps({"error": {"code": 503, "message": "x", "status": rpc}}).encode()
        assert gemini.classify_error(503, {}, body) == ProviderHttpUnavailable()


def test_classify_context_too_large():
    body = _fixture_bytes("error_context_too_large.json")
    assert gemini.classify_error(400, {}, body) == ProviderContextTooLarge()


def test_classify_credential_rejections():
    with pytest.raises(CredentialRejected):
        gemini.classify_error(401, {}, _fixture_bytes("error_401.json"))
    with pytest.raises(CredentialRejected):
        gemini.classify_error(
            403,
            {},
            json.dumps(
                {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}
            ).encode(),
        )
    # Gemini reports an invalid key as HTTP 400 INVALID_ARGUMENT.
    with pytest.raises(CredentialRejected):
        gemini.classify_error(
            400,
            {},
            json.dumps(
                {
                    "error": {
                        "code": 400,
                        "message": "API key not valid. Please pass a valid API key.",
                        "status": "INVALID_ARGUMENT",
                    }
                }
            ).encode(),
        )


@pytest.mark.parametrize("rpc_status", ["INVALID_ARGUMENT", "FAILED_PRECONDITION"])
def test_classify_other_4xx_is_unclassified_defect(rpc_status: str):
    body = json.dumps(
        {"error": {"code": 400, "message": "something else", "status": rpc_status}}
    ).encode()
    with pytest.raises(RuntimeDefect) as excinfo:
        gemini.classify_error(400, {}, body)
    assert not isinstance(excinfo.value, CredentialRejected | ProtocolDefect)
    assert excinfo.value.origin == "provider_http"
    assert excinfo.value.code == "unclassified_provider_error"


def test_classify_unparseable_body_where_parsing_needed_is_protocol_defect():
    with pytest.raises(ProtocolDefect):
        gemini.classify_error(404, {}, b"<html>not json</html>")


def test_classify_error_detail_redacts_secrets():
    secret = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
    body = json.dumps(
        {
            "error": {
                "code": 400,
                "message": f"Bad request for key {secret}",
                "status": "FAILED_PRECONDITION",
            }
        }
    ).encode()
    with pytest.raises(RuntimeDefect) as excinfo:
        gemini.classify_error(400, {}, body)
    assert secret not in excinfo.value.message
    assert "redacted" in excinfo.value.message
