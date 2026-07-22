"""Moonshot codec tests: encode goldens, affinity finalize, stream_request
injection determinism, prefix-bytes sensitivity, non-stream/stream decode
(usage nesting: choices[0].usage REAL shape + tolerated top-level chunk),
complete-native-message continuation replay, and the classify_error table."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from provider_runtime import moonshot
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
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallDone,
    ToolCallStart,
    ToolResultMessage,
    UsageEvent,
    UserMessage,
)

FIXTURES = Path(__file__).parent / "fixtures" / "moonshot"
TARGET = ProviderTarget(provider="moonshot", model="kimi-k3")
CONTRACT = CATALOG.chat_contract(TARGET)

TOOL_PARAMETERS_RAW: dict[str, object] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}
ANSWER_SCHEMA_RAW: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

LOOKUP_TOOL = CanonicalTool(
    name="lookup",
    description="Look things up.",
    parameters=parse_canonical_schema(TOOL_PARAMETERS_RAW),
)

TEXT_OUTPUT: OutputSpec = TextOutput()

BASE_MESSAGES: tuple[PromptMessage, ...] = (
    SystemMessage(
        blocks=(PromptBlock(text="You are terse.", stability=Stable(scope=GlobalScope())),)
    ),
    UserMessage(blocks=(PromptBlock(text="Why do tides rise?", stability=Dynamic()),)),
)


def make_intent(
    *,
    messages: tuple[PromptMessage, ...] = BASE_MESSAGES,
    reasoning: ReasoningLevel = "max",
    tools: tuple[CanonicalTool, ...] = (),
    output: OutputSpec = TEXT_OUTPUT,
) -> GenerateIntent:
    return GenerateIntent(
        target=TARGET,
        messages=messages,
        max_output_tokens=512,
        reasoning=reasoning,
        tools=tools,
        tool_choice="auto",
        output=output,
    )


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def fixture_json(name: str) -> dict[str, object]:
    return json.loads(fixture_bytes(name))


def sse_events(name: str) -> list[SseEvent]:
    lines = (FIXTURES / name).read_text().splitlines()
    return [
        SseEvent(event=None, data=line.removeprefix("data: "))
        for line in lines
        if line.startswith("data: ")
    ]


async def aiter_events(events: list[SseEvent]) -> AsyncIterator[SseEvent]:
    for event in events:
        yield event


async def drive(events: list[SseEvent]) -> list[CodecStreamEvent]:
    return [event async for event in moonshot.decode_stream({}, aiter_events(events))]


# ---------------------------------------------------------------------------
# encode goldens


def test_encode_plain_text_golden() -> None:
    draft = moonshot.encode(make_intent(), CONTRACT)
    assert json.loads(draft.body) == {
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Why do tides rise?"},
        ],
        "max_completion_tokens": 512,
        "reasoning_effort": "max",
    }
    assert draft.url == "https://api.moonshot.ai/v1/chat/completions"
    assert draft.protocol == "moonshot_chat"
    assert draft.safe_headers == {}
    assert draft.native_reasoning == "max"


@pytest.mark.parametrize("level", ["low", "high", "max"])
def test_encode_reasoning_levels_are_identity(level: ReasoningLevel) -> None:
    draft = moonshot.encode(make_intent(reasoning=level), CONTRACT)
    assert json.loads(draft.body)["reasoning_effort"] == level


def test_encode_unsupported_reasoning_level_is_planning_defect() -> None:
    with pytest.raises(PlanningDefect):
        moonshot.encode(make_intent(reasoning="medium"), CONTRACT)


def test_encode_sends_no_sampling_or_deprecated_fields() -> None:
    body = json.loads(moonshot.encode(make_intent(tools=(LOOKUP_TOOL,)), CONTRACT).body)
    for forbidden in ("max_tokens", "temperature", "top_p", "frequency_penalty", "stream"):
        assert forbidden not in body


def test_encode_tools_golden() -> None:
    body = json.loads(moonshot.encode(make_intent(tools=(LOOKUP_TOOL,)), CONTRACT).body)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look things up.",
                "parameters": TOOL_PARAMETERS_RAW,
            },
        }
    ]
    assert body["tool_choice"] == "auto"


def test_encode_strict_output_golden() -> None:
    output = StrictJsonOutput(name="answer", schema=parse_canonical_schema(ANSWER_SCHEMA_RAW))
    body = json.loads(moonshot.encode(make_intent(output=output), CONTRACT).body)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": ANSWER_SCHEMA_RAW, "strict": True},
    }


def test_encode_continuation_replays_complete_native_message() -> None:
    payload: dict[str, object] = {
        "role": "assistant",
        "content": "Prior answer.",
        "reasoning_content": "Prior thinking.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "tides"}'},
            }
        ],
    }
    artifact = ContinuationArtifact(
        target=TARGET, codec_id=moonshot.CODEC_ID, opaque_payload=payload
    )
    messages = BASE_MESSAGES + (
        AssistantMessage(
            text="Prior answer.",
            tool_calls=(ToolCall(id="call_1", name="lookup", arguments={"query": "tides"}),),
            continuation=Present(artifact),
        ),
        ToolResultMessage(call_id="call_1", output="high tide 14:02", is_error=False),
    )
    body = json.loads(moonshot.encode(make_intent(messages=messages), CONTRACT).body)
    assert body["messages"][2] == payload
    assert body["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "high tide 14:02",
    }


def test_encode_absent_continuation_encodes_typed_fields() -> None:
    messages = BASE_MESSAGES + (
        AssistantMessage(
            text="",
            tool_calls=(ToolCall(id="call_1", name="lookup", arguments={"query": "tides"}),),
            continuation=Absent(),
        ),
    )
    body = json.loads(moonshot.encode(make_intent(messages=messages), CONTRACT).body)
    assert body["messages"][2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query":"tides"}'},
            }
        ],
    }


def test_encode_rejects_mismatched_continuation() -> None:
    artifact = ContinuationArtifact(
        target=TARGET, codec_id="openrouter_chat", opaque_payload={"role": "assistant"}
    )
    messages = BASE_MESSAGES + (
        AssistantMessage(text="x", tool_calls=(), continuation=Present(artifact)),
    )
    with pytest.raises(PlanningDefect):
        moonshot.encode(make_intent(messages=messages), CONTRACT)


# ---------------------------------------------------------------------------
# finalize / stream_request


def test_finalize_injects_affinity_without_mutating_draft() -> None:
    draft = moonshot.encode(make_intent(), CONTRACT)
    draft_body = json.loads(draft.body)
    final = moonshot.finalize(draft, "affinity-xyz")
    parsed = json.loads(final.body)
    assert parsed == {**draft_body, "prompt_cache_key": "affinity-xyz"}
    assert list(parsed)[-1] == "prompt_cache_key"
    assert "prompt_cache_key" not in json.loads(draft.body)
    assert final.method == "POST"
    assert final.url == draft.url
    assert final.safe_headers == draft.safe_headers


def test_stream_request_appends_stream_fields_deterministically() -> None:
    final = moonshot.finalize(moonshot.encode(make_intent(), CONTRACT), "aff")
    streaming = moonshot.stream_request(final)
    expected_suffix = b',"stream":true,"stream_options":{"include_usage":true}}'
    assert streaming.body == final.body[:-1] + expected_suffix
    assert moonshot.stream_request(final).body == streaming.body
    assert b"stream" not in final.body  # source value untouched


# ---------------------------------------------------------------------------
# prefix_bytes


def test_prefix_bytes_deterministic_and_dynamic_insensitive() -> None:
    assert (
        moonshot.encode(make_intent(), CONTRACT).prefix_bytes
        == moonshot.encode(make_intent(), CONTRACT).prefix_bytes
    )
    other_dynamic: tuple[PromptMessage, ...] = (
        BASE_MESSAGES[0],
        UserMessage(blocks=(PromptBlock(text="Completely different.", stability=Dynamic()),)),
    )
    assert (
        moonshot.encode(make_intent(messages=other_dynamic), CONTRACT).prefix_bytes
        == moonshot.encode(make_intent(), CONTRACT).prefix_bytes
    )


def test_prefix_bytes_sensitive_to_tools_schema_and_stable_blocks() -> None:
    base = moonshot.encode(make_intent(), CONTRACT).prefix_bytes
    with_tools = moonshot.encode(make_intent(tools=(LOOKUP_TOOL,)), CONTRACT).prefix_bytes
    output = StrictJsonOutput(name="answer", schema=parse_canonical_schema(ANSWER_SCHEMA_RAW))
    with_schema = moonshot.encode(make_intent(output=output), CONTRACT).prefix_bytes
    other_stable: tuple[PromptMessage, ...] = (
        SystemMessage(
            blocks=(PromptBlock(text="You are verbose.", stability=Stable(scope=GlobalScope())),)
        ),
        BASE_MESSAGES[1],
    )
    with_other_stable = moonshot.encode(make_intent(messages=other_stable), CONTRACT).prefix_bytes
    assert len({base, with_tools, with_schema, with_other_stable}) == 4


def test_prefix_bytes_sensitive_to_role_move_system_vs_leading_user() -> None:
    # The same stable text framed as a SystemMessage vs. a leading UserMessage
    # must produce distinct prefix_bytes: role/message placement participates
    # in the projection, not bare joined text.
    stable_text = "Same stable text."
    as_system = moonshot.encode(
        make_intent(
            messages=(
                SystemMessage(
                    blocks=(PromptBlock(text=stable_text, stability=Stable(scope=GlobalScope())),)
                ),
                BASE_MESSAGES[1],
            )
        ),
        CONTRACT,
    ).prefix_bytes
    as_leading_user = moonshot.encode(
        make_intent(
            messages=(
                UserMessage(
                    blocks=(PromptBlock(text=stable_text, stability=Stable(scope=GlobalScope())),)
                ),
                BASE_MESSAGES[1],
            )
        ),
        CONTRACT,
    ).prefix_bytes
    assert as_system != as_leading_user


def test_prefix_bytes_sensitive_to_message_regrouping() -> None:
    # Two stable blocks in ONE system message vs. split across a system
    # message and a leading user message must produce distinct prefix_bytes.
    one_message = moonshot.encode(
        make_intent(
            messages=(
                SystemMessage(
                    blocks=(
                        PromptBlock(text="Alpha.", stability=Stable(scope=GlobalScope())),
                        PromptBlock(text="Beta.", stability=Stable(scope=GlobalScope())),
                    )
                ),
                BASE_MESSAGES[1],
            )
        ),
        CONTRACT,
    ).prefix_bytes
    two_messages = moonshot.encode(
        make_intent(
            messages=(
                SystemMessage(
                    blocks=(PromptBlock(text="Alpha.", stability=Stable(scope=GlobalScope())),)
                ),
                UserMessage(
                    blocks=(PromptBlock(text="Beta.", stability=Stable(scope=GlobalScope())),)
                ),
                BASE_MESSAGES[1],
            )
        ),
        CONTRACT,
    ).prefix_bytes
    assert one_message != two_messages


def test_prefix_bytes_sensitive_to_stable_text_impersonating_tool_definition() -> None:
    # A stable block whose text is byte-equal to a real tool definition's dump
    # must never collide with an intent that actually declares that tool.
    tool_entry_json = json.dumps(
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look things up.",
                "parameters": TOOL_PARAMETERS_RAW,
            },
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    impersonating = moonshot.encode(
        make_intent(
            messages=(
                SystemMessage(
                    blocks=(
                        PromptBlock(text=tool_entry_json, stability=Stable(scope=GlobalScope())),
                    )
                ),
                BASE_MESSAGES[1],
            )
        ),
        CONTRACT,
    ).prefix_bytes
    with_matching_tool = moonshot.encode(make_intent(tools=(LOOKUP_TOOL,)), CONTRACT).prefix_bytes
    assert impersonating != with_matching_tool


# ---------------------------------------------------------------------------
# decode_response


def test_decode_response_success_text() -> None:
    outcome = moonshot.decode_response(200, {}, fixture_bytes("success_text.json"))
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(
        text="Venus is the hottest planet.", tool_calls=()
    )
    assert outcome.meta.provider == "moonshot"
    assert outcome.meta.model == "kimi-k3"
    assert outcome.meta.provider_request_id == Present("chatcmpl-msh-001")
    assert outcome.meta.upstream_provider == Absent()
    assert outcome.meta.attempt_trace == ()
    assert outcome.meta.billability == PossiblyBillable()
    assert outcome.meta.usage == Present(
        TokenUsage(
            input_tokens=120,
            output_tokens=40,
            total_tokens=160,
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Present(96),
            cache_write_input_tokens=Absent(),
        )
    )


def test_decode_response_continuation_is_complete_message_verbatim() -> None:
    outcome = moonshot.decode_response(200, {}, fixture_bytes("success_text.json"))
    assert isinstance(outcome, Succeeded)
    continuation = outcome.response.continuation
    assert isinstance(continuation, Present)
    artifact = continuation.value
    assert artifact.target == TARGET
    assert artifact.codec_id == "moonshot_chat"
    fixture = fixture_json("success_text.json")
    choices = fixture["choices"]
    assert isinstance(choices, list)
    assert artifact.opaque_payload == choices[0]["message"]


def test_decode_response_round_trips_into_encode() -> None:
    outcome = moonshot.decode_response(200, {}, fixture_bytes("success_tool_calls.json"))
    assert isinstance(outcome, Succeeded)
    continuation = outcome.response.continuation
    assert isinstance(continuation, Present)
    content = outcome.response.content
    assert isinstance(content, TextContent)
    messages = BASE_MESSAGES + (
        AssistantMessage(
            text="",
            tool_calls=content.tool_calls,
            continuation=continuation,
        ),
        ToolResultMessage(call_id="call_1", output="high tide 14:02", is_error=False),
    )
    body = json.loads(moonshot.encode(make_intent(messages=messages), CONTRACT).body)
    fixture = fixture_json("success_tool_calls.json")
    choices = fixture["choices"]
    assert isinstance(choices, list)
    assert body["messages"][2] == choices[0]["message"]


def test_decode_response_tool_calls_strict_parse() -> None:
    outcome = moonshot.decode_response(200, {}, fixture_bytes("success_tool_calls.json"))
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(
        text="",
        tool_calls=(ToolCall(id="call_1", name="lookup", arguments={"query": "tides"}),),
    )


def test_decode_response_invalid_tool_arguments_signal() -> None:
    with pytest.raises(ExpectedFailureSignal) as exc_info:
        moonshot.decode_response(200, {}, fixture_bytes("invalid_tool_arguments.json"))
    failure = exc_info.value.failure
    assert isinstance(failure, InvalidToolArguments)
    assert "call_1" in failure.safe_detail


def test_decode_response_length_is_incomplete() -> None:
    outcome = moonshot.decode_response(200, {}, fixture_bytes("incomplete_length.json"))
    assert isinstance(outcome, Incomplete)
    assert outcome.reason == "max_output_tokens"
    assert outcome.status == "provider_incomplete"
    assert outcome.safe_detail == Absent()


@pytest.mark.parametrize(
    "body",
    [b"not json", b'{"choices": []}', b'{"model": "kimi-k3"}'],
)
def test_decode_response_malformed_is_protocol_defect(body: bytes) -> None:
    with pytest.raises(ProtocolDefect):
        moonshot.decode_response(200, {}, body)


def test_decode_response_unknown_finish_reason_is_protocol_defect() -> None:
    data = fixture_json("success_text.json")
    choices = data["choices"]
    assert isinstance(choices, list)
    choices[0]["finish_reason"] = "flagged"
    with pytest.raises(ProtocolDefect):
        moonshot.decode_response(200, {}, json.dumps(data).encode())


# ---------------------------------------------------------------------------
# decode_stream


async def test_stream_happy_path_events_and_usage_folding() -> None:
    events = await drive(sse_events("stream_text.txt"))
    assert isinstance(events[0], StreamStart)
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["Tides ", "rise."]

    continuations = [event for event in events if isinstance(event, ContinuationDelta)]
    assert len(continuations) == 1
    assert continuations[0].artifact.opaque_payload == {
        "role": "assistant",
        "content": "Tides rise.",
        "reasoning_content": "Thinking about tides.",
    }

    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert events[-2] is continuations[0]  # continuation precedes the terminal
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(text="Tides rise.", tool_calls=())
    assert outcome.meta.provider_request_id == Present("chatcmpl-msh-002")
    # choices[0].usage on the finish chunk folded with the tolerated trailing
    # top-level include_usage chunk (which adds reasoning detail).
    assert outcome.meta.usage == Present(
        TokenUsage(
            input_tokens=50,
            output_tokens=12,
            total_tokens=62,
            reasoning_tokens=Present(6),
            cache_read_input_tokens=Present(32),
            cache_write_input_tokens=Absent(),
        )
    )
    assert len([event for event in events if isinstance(event, UsageEvent)]) == 2


async def test_stream_tool_calls_accumulate_by_index() -> None:
    events = await drive(sse_events("stream_tool_calls.txt"))
    starts = [event for event in events if isinstance(event, ToolCallStart)]
    deltas = [event for event in events if isinstance(event, ToolCallDelta)]
    dones = [event for event in events if isinstance(event, ToolCallDone)]
    assert starts == [ToolCallStart(call_id="call_1", name="lookup")]
    assert [delta.arguments_delta for delta in deltas] == ['{"query"', ': "tides"}']
    assert dones == [
        ToolCallDone(tool_call=ToolCall(id="call_1", name="lookup", arguments={"query": "tides"}))
    ]
    continuations = [event for event in events if isinstance(event, ContinuationDelta)]
    assert len(continuations) == 1
    assert continuations[0].artifact.opaque_payload == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Need the tide tool.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "tides"}'},
            }
        ],
    }
    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert isinstance(terminal.outcome, Succeeded)


async def test_stream_missing_done_is_transient_stream_error() -> None:
    with pytest.raises(TransientStreamError) as exc_info:
        await drive(sse_events("stream_missing_done.txt"))
    assert exc_info.value.cause == ProviderStreamInterrupted(partial_output=False)


async def test_stream_done_without_finish_reason_is_transient_stream_error() -> None:
    events = [
        SseEvent(
            event=None,
            data='{"id":"c","model":"kimi-k3","choices":[{"index":0,"delta":{"content":"hi"}}]}',
        ),
        SseEvent(event=None, data="[DONE]"),
    ]
    with pytest.raises(TransientStreamError) as exc_info:
        await drive(events)
    assert exc_info.value.cause == ProviderStreamInterrupted(partial_output=False)


async def test_stream_malformed_chunk_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        await drive([SseEvent(event=None, data="not json")])


# ---------------------------------------------------------------------------
# classify_error


def test_classify_401_and_403_raise_credential_rejected() -> None:
    for status in (401, 403):
        with pytest.raises(CredentialRejected):
            moonshot.classify_error(status, {}, b'{"error": {"message": "auth"}}')


def test_classify_429_rate_limit_with_retry_after() -> None:
    classified = moonshot.classify_error(
        429, {"retry-after": "2.5"}, b'{"error": {"message": "slow down"}}'
    )
    assert classified == ProviderRateLimit(retry_after=Present(2.5))
    assert moonshot.classify_error(429, {}, b"{}") == ProviderRateLimit(retry_after=Absent())


def test_classify_429_quota_type_raises_quota_defect() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        moonshot.classify_error(429, {}, fixture_bytes("error_429_quota.json"))
    assert exc_info.value.code == "quota_exhausted"
    assert exc_info.value.origin == "provider_http"


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_classify_5xx_is_provider_unavailable(status: int) -> None:
    assert moonshot.classify_error(status, {}, b"") == ProviderHttpUnavailable()


def test_classify_context_overflow() -> None:
    classified = moonshot.classify_error(400, {}, fixture_bytes("error_context_too_large.json"))
    assert classified == ProviderContextTooLarge()


def test_classify_unknown_400_is_unclassified_defect() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        moonshot.classify_error(400, {}, b'{"error": {"message": "odd", "type": "other"}}')
    assert exc_info.value.code == "unclassified_provider_error"


def test_classify_unparseable_400_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        moonshot.classify_error(400, {}, b"<html>gateway</html>")


def test_classify_error_details_are_redacted() -> None:
    body = json.dumps(
        {"error": {"message": "bad key sk-abcdefghijk1234567890", "type": "other"}}
    ).encode()
    with pytest.raises(RuntimeDefect) as exc_info:
        moonshot.classify_error(400, {}, body)
    assert "sk-abcdefghijk1234567890" not in exc_info.value.message
