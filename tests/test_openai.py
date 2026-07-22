"""OpenAI Responses codec tests (codec-seam contract).

Encode goldens assert EXACT body dicts; decode tests feed fixture bodies and
hand-built SseEvent iterators (no HTTP mocking needed at this seam)."""

import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import cast

import pytest

from provider_runtime._signals import ExpectedFailureSignal, TransientStreamError
from provider_runtime.catalog import CATALOG, ChatModelContract
from provider_runtime.errors import (
    CredentialRejected,
    PlanningDefect,
    ProtocolDefect,
    RuntimeDefect,
)
from provider_runtime.openai import (
    CODEC_ID,
    build_transcription_request,
    classify_error,
    decode_response,
    decode_stream,
    encode,
    finalize,
    parse_transcription_response,
    stream_request,
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
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallDone,
    ToolCallStart,
    ToolChoice,
    ToolResultMessage,
    UserMessage,
)

FIXTURES = Path(__file__).parent / "fixtures" / "openai"

TARGET = ProviderTarget(provider="openai", model="gpt-5.6-sol")
CONTRACT: ChatModelContract = CATALOG.chat_contract(TARGET)

TOOL_SCHEMA_RAW: dict[str, object] = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "Search query"}},
    "required": ["query"],
    "additionalProperties": False,
}
SEARCH_TOOL = CanonicalTool(
    name="search_library",
    description="Search the library",
    parameters=parse_canonical_schema(TOOL_SCHEMA_RAW),
)

OUTPUT_SCHEMA_RAW: dict[str, object] = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}, "confidence": {"type": "integer"}},
    "required": ["verdict", "confidence"],
    "additionalProperties": False,
}
VERDICT_OUTPUT = StrictJsonOutput(name="verdict", schema=parse_canonical_schema(OUTPUT_SCHEMA_RAW))

SYSTEM_STABLE = SystemMessage(
    blocks=(PromptBlock(text="You are terse.", stability=Stable(scope=GlobalScope())),)
)
USER_DYNAMIC = UserMessage(blocks=(PromptBlock(text="hi", stability=Dynamic()),))
TEXT_OUTPUT = TextOutput()

STABLE_SYSTEM_PART_WITH_BREAKPOINT: dict[str, object] = {
    "type": "input_text",
    "text": "You are terse.",
    "prompt_cache_breakpoint": {"mode": "explicit"},
}

PRIOR_OUTPUT: tuple[dict[str, object], ...] = (
    {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "gAAAAB-enc-1"},
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "prior turn text"}],
    },
    {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "search_library",
        "arguments": '{"query":"hyperion"}',
    },
)


def make_intent(
    *,
    messages: tuple[PromptMessage, ...] = (SYSTEM_STABLE, USER_DYNAMIC),
    reasoning: ReasoningLevel = "medium",
    tools: tuple[CanonicalTool, ...] = (),
    tool_choice: ToolChoice = "auto",
    output: OutputSpec = TEXT_OUTPUT,
    max_output_tokens: int = 256,
) -> GenerateIntent:
    return GenerateIntent(
        target=TARGET,
        messages=messages,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        tools=tools,
        tool_choice=tool_choice,
        output=output,
    )


def body_of(intent: GenerateIntent) -> dict[str, object]:
    return cast(dict[str, object], json.loads(encode(intent, CONTRACT).body))


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def fixture_json(name: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(fixture_bytes(name)))


def sse(payload: Mapping[str, object]) -> SseEvent:
    return SseEvent(event=cast(str | None, payload.get("type")), data=json.dumps(payload))


def sse_fixture(name: str) -> list[SseEvent]:
    events: list[SseEvent] = []
    for line in (FIXTURES / name).read_text().splitlines():
        if line.startswith("data: "):
            data = line.removeprefix("data: ")
            payload = json.loads(data)
            events.append(SseEvent(event=payload.get("type"), data=data))
    return events


async def drain(events: list[SseEvent], headers: Mapping[str, str] | None = None) -> list[object]:
    async def gen() -> AsyncIterator[SseEvent]:
        for event in events:
            yield event

    return [event async for event in decode_stream(headers if headers is not None else {}, gen())]


# ---------------------------------------------------------------------------
# encode goldens


def test_encode_plain_text_golden() -> None:
    assert body_of(make_intent()) == {
        "model": "gpt-5.6-sol",
        "input": [
            {"role": "system", "content": [STABLE_SYSTEM_PART_WITH_BREAKPOINT]},
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ],
        "max_output_tokens": 256,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": "medium"},
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
    }


@pytest.mark.parametrize("level", ["none", "low", "medium", "high", "xhigh", "max"])
def test_encode_sends_every_reasoning_level_verbatim(level: ReasoningLevel) -> None:
    # Identity map incl. distinct xhigh/max; "none" IS sent — nothing omitted.
    draft = encode(make_intent(reasoning=level), CONTRACT)
    assert cast(dict[str, object], json.loads(draft.body))["reasoning"] == {"effort": level}
    assert draft.native_reasoning == level


def test_encode_tools_golden() -> None:
    body = body_of(make_intent(tools=(SEARCH_TOOL,), tool_choice="auto"))
    assert body["tools"] == [
        {
            "type": "function",
            "name": "search_library",
            "description": "Search the library",
            "parameters": TOOL_SCHEMA_RAW,
            "strict": True,
        }
    ]
    assert body["tool_choice"] == "auto"


def test_encode_tool_choice_none() -> None:
    body = body_of(make_intent(tools=(SEARCH_TOOL,), tool_choice="none"))
    assert body["tool_choice"] == "none"


def test_encode_without_tools_omits_tools_and_tool_choice() -> None:
    body = body_of(make_intent())
    assert "tools" not in body
    assert "tool_choice" not in body


def test_encode_strict_output_golden() -> None:
    body = body_of(make_intent(output=VERDICT_OUTPUT))
    assert body["text"] == {
        "format": {
            "type": "json_schema",
            "name": "verdict",
            "schema": OUTPUT_SCHEMA_RAW,
            "strict": True,
        }
    }


def test_encode_breakpoint_on_last_stable_prefix_block() -> None:
    # The stable prefix spans messages; the marker lands on its LAST content
    # block only, and dynamic/post-prefix stable blocks never carry it.
    intent = make_intent(
        messages=(
            SYSTEM_STABLE,
            UserMessage(
                blocks=(
                    PromptBlock(text="corpus digest", stability=Stable(scope=GlobalScope())),
                    PromptBlock(text="fresh question", stability=Dynamic()),
                    PromptBlock(text="late stable", stability=Stable(scope=GlobalScope())),
                )
            ),
        )
    )
    body = body_of(intent)
    assert body["input"] == [
        {"role": "system", "content": [{"type": "input_text", "text": "You are terse."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "corpus digest",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "input_text", "text": "fresh question"},
                {"type": "input_text", "text": "late stable"},
            ],
        },
    ]
    assert json.dumps(body).count("prompt_cache_breakpoint") == 1


def test_encode_assistant_typed_turn_without_tool_calls() -> None:
    # No tool_calls on the Absent-continuation turn: nothing to replay, so the
    # plain output_text item is emitted as-is.
    intent = make_intent(
        messages=(
            SYSTEM_STABLE,
            USER_DYNAMIC,
            AssistantMessage(text="typed reply", tool_calls=(), continuation=Absent()),
        )
    )
    body = body_of(intent)
    assert cast(list[object], body["input"])[2:] == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "typed reply"}]},
    ]


def test_encode_tool_result_pairs_with_continuation_replayed_function_call() -> None:
    # A ToolResultMessage's function_call_output is only well-formed when the
    # preceding function_call item actually entered the input — which, on this
    # wire, only the continuation artifact can supply.
    artifact = ContinuationArtifact(
        target=TARGET, codec_id=CODEC_ID, opaque_payload={"output": PRIOR_OUTPUT}
    )
    intent = make_intent(
        messages=(
            SYSTEM_STABLE,
            USER_DYNAMIC,
            AssistantMessage(
                text="",
                tool_calls=(ToolCall(id="call_1", name="search_library", arguments={}),),
                continuation=Present(artifact),
            ),
            ToolResultMessage(call_id="call_1", output="tool says hi", is_error=False),
        )
    )
    body = body_of(intent)
    assert cast(list[object], body["input"])[2:] == [
        *[dict(item) for item in PRIOR_OUTPUT],
        {"type": "function_call_output", "call_id": "call_1", "output": "tool says hi"},
    ]


def test_encode_rejects_typed_tool_calls_without_continuation() -> None:
    # Absent continuation cannot supply the ordered function_call/reasoning
    # items a tool turn needs on replay; typed tool_calls alone are not
    # enough, so this must fail loud rather than silently drop the calls.
    intent = make_intent(
        messages=(
            SYSTEM_STABLE,
            USER_DYNAMIC,
            AssistantMessage(
                text="",
                tool_calls=(ToolCall(id="call_1", name="search_library", arguments={}),),
                continuation=Absent(),
            ),
            ToolResultMessage(call_id="call_1", output="tool says hi", is_error=False),
        )
    )
    with pytest.raises(PlanningDefect) as excinfo:
        encode(intent, CONTRACT)
    assert excinfo.value.code == "continuation_required_for_tool_replay"


def test_encode_continuation_replays_output_items_verbatim_as_sole_source() -> None:
    artifact = ContinuationArtifact(
        target=TARGET, codec_id=CODEC_ID, opaque_payload={"output": PRIOR_OUTPUT}
    )
    intent = make_intent(
        messages=(
            SYSTEM_STABLE,
            USER_DYNAMIC,
            AssistantMessage(
                text="typed text that must NOT reach the wire",
                tool_calls=(ToolCall(id="call_1", name="search_library", arguments={}),),
                continuation=Present(artifact),
            ),
            ToolResultMessage(call_id="call_1", output="42 results", is_error=False),
        )
    )
    body = body_of(intent)
    assert cast(list[object], body["input"])[2:5] == [dict(item) for item in PRIOR_OUTPUT]
    assert "typed text that must NOT reach the wire" not in json.dumps(body)


@pytest.mark.parametrize(
    "artifact",
    [
        ContinuationArtifact(
            target=ProviderTarget(provider="openai", model="gpt-5.6-luna"),
            codec_id=CODEC_ID,
            opaque_payload={"output": PRIOR_OUTPUT},
        ),
        ContinuationArtifact(
            target=TARGET,
            codec_id="anthropic_messages",
            opaque_payload={"output": PRIOR_OUTPUT},
        ),
    ],
)
def test_encode_rejects_mismatched_continuation(artifact: ContinuationArtifact) -> None:
    intent = make_intent(
        messages=(
            SYSTEM_STABLE,
            AssistantMessage(text="", tool_calls=(), continuation=Present(artifact)),
        )
    )
    with pytest.raises(PlanningDefect) as excinfo:
        encode(intent, CONTRACT)
    assert excinfo.value.code == "continuation_mismatch"


def test_encode_rejects_malformed_continuation_payload() -> None:
    artifact = ContinuationArtifact(
        target=TARGET, codec_id=CODEC_ID, opaque_payload={"items": PRIOR_OUTPUT}
    )
    intent = make_intent(
        messages=(
            SYSTEM_STABLE,
            AssistantMessage(text="", tool_calls=(), continuation=Present(artifact)),
        )
    )
    with pytest.raises(PlanningDefect) as excinfo:
        encode(intent, CONTRACT)
    assert excinfo.value.code == "invalid_continuation_payload"


def test_encode_rejects_undeclared_reasoning_level() -> None:
    with pytest.raises(PlanningDefect) as excinfo:
        encode(make_intent(reasoning="minimal"), CONTRACT)
    assert excinfo.value.code == "unsupported_reasoning_level"


def test_encode_draft_facts() -> None:
    draft = encode(make_intent(), CONTRACT)
    assert draft.target == TARGET
    assert draft.protocol == "openai_responses"
    assert draft.url == "https://api.openai.com/v1/responses"
    assert draft.safe_headers == {}
    assert draft.provider_framing_overhead_tokens == CONTRACT.provider_framing_overhead_tokens
    assert "prompt_cache_key" not in json.loads(draft.body)


# ---------------------------------------------------------------------------
# finalize / stream_request


def test_finalize_injects_affinity_deterministically() -> None:
    draft = encode(make_intent(), CONTRACT)
    first = finalize(draft, "affinity-abc")
    second = finalize(draft, "affinity-abc")
    assert first.body == second.body
    assert first.method == "POST"
    assert first.url == draft.url
    parsed = cast(dict[str, object], json.loads(first.body))
    draft_body = cast(dict[str, object], json.loads(draft.body))
    assert parsed == {**draft_body, "prompt_cache_key": "affinity-abc"}
    assert list(parsed)[-1] == "prompt_cache_key"
    # The draft is never mutated.
    assert "prompt_cache_key" not in json.loads(draft.body)


def test_stream_request_adds_stream_true_as_new_value() -> None:
    final = finalize(encode(make_intent(), CONTRACT), "affinity-abc")
    streaming_first = stream_request(final)
    streaming_second = stream_request(final)
    assert streaming_first.body == streaming_second.body
    assert streaming_first.url == final.url
    assert streaming_first.method == final.method
    assert streaming_first.safe_headers == final.safe_headers
    parsed = cast(dict[str, object], json.loads(streaming_first.body))
    final_body = cast(dict[str, object], json.loads(final.body))
    assert parsed == {**final_body, "stream": True}
    # The non-stream request stays untouched (generate() sends it as-is).
    assert "stream" not in json.loads(final.body)


# ---------------------------------------------------------------------------
# prefix_bytes


def frame(component: bytes) -> bytes:
    return len(component).to_bytes(8, "big") + component


def test_prefix_bytes_exact_framing_for_stable_prefix() -> None:
    draft = encode(make_intent(), CONTRACT)
    stable_item = {"role": "system", "content": [STABLE_SYSTEM_PART_WITH_BREAKPOINT]}
    expected = b"".join(
        frame(component)
        for component in (
            b"stable",
            json.dumps(stable_item, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            b"tools",
            b"tool_choice",
            b"format",
        )
    )
    assert draft.prefix_bytes == expected


def test_prefix_bytes_deterministic_and_dynamic_insensitive() -> None:
    base = encode(make_intent(), CONTRACT).prefix_bytes
    again = encode(make_intent(), CONTRACT).prefix_bytes
    dynamic_changed = encode(
        make_intent(
            messages=(
                SYSTEM_STABLE,
                UserMessage(blocks=(PromptBlock(text="DIFFERENT", stability=Dynamic()),)),
            )
        ),
        CONTRACT,
    ).prefix_bytes
    assert base == again == dynamic_changed


def test_prefix_bytes_sensitive_to_stable_blocks_tools_and_output_schema() -> None:
    base = encode(make_intent(), CONTRACT).prefix_bytes
    stable_changed = encode(
        make_intent(
            messages=(
                SystemMessage(
                    blocks=(PromptBlock(text="Different.", stability=Stable(scope=GlobalScope())),)
                ),
                USER_DYNAMIC,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    with_tools = encode(make_intent(tools=(SEARCH_TOOL,)), CONTRACT).prefix_bytes
    with_output = encode(make_intent(output=VERDICT_OUTPUT), CONTRACT).prefix_bytes
    assert stable_changed != base
    assert with_tools != base
    assert with_output != base
    assert with_tools != with_output


def test_prefix_bytes_sensitive_to_role_move_system_vs_leading_user() -> None:
    # The same stable text framed as a SystemMessage vs. a leading UserMessage
    # must produce distinct prefix_bytes: role/message placement participates
    # in the projection, not bare content-part bytes.
    stable_text = "Same stable text."
    as_system = encode(
        make_intent(
            messages=(
                SystemMessage(
                    blocks=(PromptBlock(text=stable_text, stability=Stable(scope=GlobalScope())),)
                ),
                USER_DYNAMIC,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    as_leading_user = encode(
        make_intent(
            messages=(
                UserMessage(
                    blocks=(PromptBlock(text=stable_text, stability=Stable(scope=GlobalScope())),)
                ),
                USER_DYNAMIC,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    assert as_system != as_leading_user


def test_prefix_bytes_sensitive_to_message_regrouping() -> None:
    # Two stable blocks in ONE message vs. the same two blocks split across
    # TWO messages must produce distinct prefix_bytes.
    one_message = encode(
        make_intent(
            messages=(
                SystemMessage(
                    blocks=(
                        PromptBlock(text="Alpha.", stability=Stable(scope=GlobalScope())),
                        PromptBlock(text="Beta.", stability=Stable(scope=GlobalScope())),
                    )
                ),
                USER_DYNAMIC,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    two_messages = encode(
        make_intent(
            messages=(
                SystemMessage(
                    blocks=(PromptBlock(text="Alpha.", stability=Stable(scope=GlobalScope())),)
                ),
                UserMessage(
                    blocks=(PromptBlock(text="Beta.", stability=Stable(scope=GlobalScope())),)
                ),
                USER_DYNAMIC,
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
            "name": "search_library",
            "description": "Search the library",
            "parameters": TOOL_SCHEMA_RAW,
            "strict": True,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    impersonating = encode(
        make_intent(
            messages=(
                SystemMessage(
                    blocks=(
                        PromptBlock(text=tool_entry_json, stability=Stable(scope=GlobalScope())),
                    )
                ),
                USER_DYNAMIC,
            )
        ),
        CONTRACT,
    ).prefix_bytes
    with_matching_tool = encode(make_intent(tools=(SEARCH_TOOL,)), CONTRACT).prefix_bytes
    assert impersonating != with_matching_tool


# ---------------------------------------------------------------------------
# decode_response


def test_decode_success_text() -> None:
    outcome = decode_response(200, {"x-request-id": "req-123"}, fixture_bytes("success_text.json"))
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.provider == "openai"
    assert outcome.meta.model == "gpt-5.6-sol"
    assert outcome.meta.provider_request_id == Present("req-123")
    assert outcome.meta.upstream_provider == Absent()
    assert outcome.meta.attempt_trace == ()
    assert outcome.meta.billability == PossiblyBillable()
    assert outcome.meta.usage == Present(
        TokenUsage(
            input_tokens=1200,
            output_tokens=52,
            total_tokens=1252,
            reasoning_tokens=Present(12),
            cache_read_input_tokens=Present(1024),
            cache_write_input_tokens=Present(128),
        )
    )
    assert outcome.response.content == TextContent(text="Hello from Sol.", tool_calls=())


def test_decode_success_request_id_falls_back_to_body_id() -> None:
    outcome = decode_response(200, {}, fixture_bytes("success_text.json"))
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.provider_request_id == Present("resp_0a1b2c")


def test_decode_success_continuation_carries_complete_ordered_output() -> None:
    envelope = fixture_json("success_text.json")
    outcome = decode_response(200, {}, fixture_bytes("success_text.json"))
    assert isinstance(outcome, Succeeded)
    continuation = outcome.response.continuation
    assert isinstance(continuation, Present)
    artifact = continuation.value
    assert artifact.target == TARGET
    assert artifact.codec_id == CODEC_ID
    expected_items = tuple(cast(list[dict[str, object]], envelope["output"]))
    assert artifact.opaque_payload == {"output": expected_items}


def test_decode_success_tool_calls() -> None:
    outcome = decode_response(200, {}, fixture_bytes("success_tool_calls.json"))
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(
        text="",
        tool_calls=(
            ToolCall(id="call_abc", name="search_library", arguments={"query": "hyperion"}),
        ),
    )


def test_decode_strict_output_returns_text_only() -> None:
    # The output arm is plan-determined and decode carries no plan:
    # StructuredContent construction (strict parse) belongs to the runtime,
    # so a strict-output response decodes to the raw TEXT.
    outcome = decode_response(200, {}, fixture_bytes("success_structured.json"))
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(
        text='{"verdict":"keep","confidence":3}', tool_calls=()
    )


def test_decode_refusal_content_part_is_refused() -> None:
    outcome = decode_response(200, {}, fixture_bytes("refusal.json"))
    assert isinstance(outcome, Refused)
    assert "can't help" in outcome.safe_detail
    assert isinstance(outcome.meta.usage, Present)


def test_decode_incomplete_max_output_tokens() -> None:
    outcome = decode_response(200, {}, fixture_bytes("incomplete_max_output_tokens.json"))
    assert isinstance(outcome, Incomplete)
    assert outcome.reason == "max_output_tokens"
    assert outcome.status == "provider_incomplete"
    assert outcome.safe_detail == Present("max_output_tokens")


def test_decode_incomplete_content_filter() -> None:
    envelope = fixture_json("incomplete_max_output_tokens.json")
    envelope["incomplete_details"] = {"reason": "content_filter"}
    outcome = decode_response(200, {}, json.dumps(envelope).encode())
    assert isinstance(outcome, Incomplete)
    assert outcome.reason == "content_filter_partial"
    assert outcome.status == "provider_incomplete"


def test_decode_unknown_incomplete_reason_is_protocol_defect() -> None:
    envelope = fixture_json("incomplete_max_output_tokens.json")
    envelope["incomplete_details"] = {"reason": "mystery"}
    with pytest.raises(ProtocolDefect):
        decode_response(200, {}, json.dumps(envelope).encode())


@pytest.mark.parametrize("arguments", ["{broken", "[1,2]"])
def test_decode_invalid_tool_arguments_raise_expected_failure(arguments: str) -> None:
    envelope = fixture_json("success_tool_calls.json")
    cast(list[dict[str, object]], envelope["output"])[1]["arguments"] = arguments
    with pytest.raises(ExpectedFailureSignal) as excinfo:
        decode_response(200, {}, json.dumps(envelope).encode())
    assert isinstance(excinfo.value.failure, InvalidToolArguments)
    assert "search_library" in excinfo.value.failure.safe_detail


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[1,2,3]",
        json.dumps({"id": "r", "status": "completed", "output": []}).encode(),  # no model
        json.dumps({"id": "r", "model": "m", "status": "queued", "output": []}).encode(),
        json.dumps({"id": "r", "model": "m", "status": "completed", "output": "nope"}).encode(),
    ],
)
def test_decode_malformed_envelopes_are_protocol_defects(body: bytes) -> None:
    with pytest.raises(ProtocolDefect):
        decode_response(200, {}, body)


def test_decode_without_usage_is_absent() -> None:
    envelope = fixture_json("success_text.json")
    del envelope["usage"]
    outcome = decode_response(200, {}, json.dumps(envelope).encode())
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.usage == Absent()


# ---------------------------------------------------------------------------
# decode_stream


async def test_stream_text_happy_path() -> None:
    events = await drain(sse_fixture("stream_text.sse.txt"), {"x-request-id": "req-stream-1"})
    assert [type(event) for event in events] == [
        StreamStart,
        TextDelta,
        TextDelta,
        ContinuationDelta,
        TerminalEvent,
    ]
    assert cast(TextDelta, events[1]).text == "Hello"
    assert cast(TextDelta, events[2]).text == " world"
    outcome = cast(TerminalEvent, events[-1]).outcome
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(text="Hello world", tool_calls=())
    assert outcome.meta.model == "gpt-5.6-sol"
    assert outcome.meta.provider_request_id == Present("req-stream-1")
    assert outcome.meta.usage == Present(
        TokenUsage(
            input_tokens=1200,
            output_tokens=9,
            total_tokens=1209,
            reasoning_tokens=Present(4),
            cache_read_input_tokens=Present(1024),
            cache_write_input_tokens=Present(0),
        )
    )
    continuation_delta = cast(ContinuationDelta, events[3])
    payload_items = cast(
        tuple[dict[str, object], ...], continuation_delta.artifact.opaque_payload["output"]
    )
    assert [item["id"] for item in payload_items] == ["rs_s1", "msg_s1"]
    assert payload_items[0]["encrypted_content"] == "gAAAAB-enc-s1"
    # The terminal payload carries the SAME complete artifact.
    assert outcome.response.continuation == Present(continuation_delta.artifact)


async def test_stream_request_id_falls_back_to_in_band_id() -> None:
    events = await drain(sse_fixture("stream_text.sse.txt"))
    outcome = cast(TerminalEvent, events[-1]).outcome
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.provider_request_id == Present("resp_s1")


async def test_stream_tool_calls() -> None:
    events = await drain(sse_fixture("stream_tool_calls.sse.txt"))
    assert [type(event) for event in events] == [
        StreamStart,
        ToolCallStart,
        ToolCallDelta,
        ToolCallDelta,
        ToolCallDone,
        ContinuationDelta,
        TerminalEvent,
    ]
    assert cast(ToolCallStart, events[1]) == ToolCallStart(
        call_id="call_abc", name="search_library"
    )
    assert cast(ToolCallDelta, events[2]).arguments_delta == '{"query":'
    assert cast(ToolCallDone, events[4]).tool_call == ToolCall(
        id="call_abc", name="search_library", arguments={"query": "hyperion"}
    )
    outcome = cast(TerminalEvent, events[-1]).outcome
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(
        text="",
        tool_calls=(
            ToolCall(id="call_abc", name="search_library", arguments={"query": "hyperion"}),
        ),
    )


async def test_stream_emits_at_most_one_continuation_delta_before_terminal() -> None:
    events = await drain(sse_fixture("stream_text.sse.txt"))
    continuation_indices = [
        index for index, event in enumerate(events) if isinstance(event, ContinuationDelta)
    ]
    assert len(continuation_indices) == 1
    assert continuation_indices[0] == len(events) - 2
    assert isinstance(events[-1], TerminalEvent)


async def test_stream_refusal_is_incomplete_refused() -> None:
    events = await drain(
        [
            sse(
                {"type": "response.created", "response": {"id": "resp_r1", "model": "gpt-5.6-sol"}}
            ),
            sse({"type": "response.refusal.delta", "item_id": "msg_r1", "delta": "I can't "}),
            sse(
                {"type": "response.refusal.delta", "item_id": "msg_r1", "delta": "help with that."}
            ),
            sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": "msg_r1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "I can't help with that."}],
                    },
                }
            ),
            sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_r1",
                        "status": "completed",
                        "model": "gpt-5.6-sol",
                        "usage": {"input_tokens": 30, "output_tokens": 8, "total_tokens": 38},
                    },
                }
            ),
        ]
    )
    assert [type(event) for event in events] == [StreamStart, TerminalEvent]
    outcome = cast(TerminalEvent, events[-1]).outcome
    assert isinstance(outcome, Incomplete)
    assert outcome.status == "refused"
    assert outcome.safe_detail == Present("I can't help with that.")
    assert isinstance(outcome.meta.usage, Present)


async def test_stream_incomplete_terminal() -> None:
    events = await drain(
        [
            sse(
                {"type": "response.created", "response": {"id": "resp_i1", "model": "gpt-5.6-sol"}}
            ),
            sse({"type": "response.output_text.delta", "item_id": "msg_i1", "delta": "partial"}),
            sse(
                {
                    "type": "response.incomplete",
                    "response": {
                        "id": "resp_i1",
                        "status": "incomplete",
                        "model": "gpt-5.6-sol",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "usage": {"input_tokens": 20, "output_tokens": 64, "total_tokens": 84},
                    },
                }
            ),
        ]
    )
    assert [type(event) for event in events] == [StreamStart, TextDelta, TerminalEvent]
    outcome = cast(TerminalEvent, events[-1]).outcome
    assert isinstance(outcome, Incomplete)
    assert outcome.reason == "max_output_tokens"
    assert outcome.status == "provider_incomplete"


async def test_stream_missing_terminal_raises_transient_stream_error() -> None:
    with pytest.raises(TransientStreamError) as excinfo:
        await drain(
            [
                sse({"type": "response.created", "response": {"id": "r", "model": "gpt-5.6-sol"}}),
                sse({"type": "response.output_text.delta", "item_id": "m", "delta": "hi"}),
            ]
        )
    assert excinfo.value.cause == ProviderStreamInterrupted(partial_output=False)


async def test_stream_failed_server_error_is_transient() -> None:
    with pytest.raises(TransientStreamError) as excinfo:
        await drain(
            [
                sse(
                    {
                        "type": "response.failed",
                        "response": {
                            "id": "r",
                            "model": "gpt-5.6-sol",
                            "status": "failed",
                            "error": {"code": "server_error", "message": "boom"},
                        },
                    }
                )
            ]
        )
    assert excinfo.value.cause == ProviderHttpUnavailable()


async def test_stream_error_event_rate_limit_is_transient() -> None:
    with pytest.raises(TransientStreamError) as excinfo:
        await drain([sse({"type": "error", "code": "rate_limit_exceeded", "message": "slow down"})])
    assert excinfo.value.cause == ProviderRateLimit(retry_after=Absent())


async def test_stream_failed_unknown_code_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        await drain(
            [
                sse(
                    {
                        "type": "response.failed",
                        "response": {
                            "id": "r",
                            "model": "gpt-5.6-sol",
                            "status": "failed",
                            "error": {"code": "invalid_prompt", "message": "nope"},
                        },
                    }
                )
            ]
        )


async def test_stream_malformed_frame_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        await drain([SseEvent(event="response.created", data="{broken")])


async def test_stream_done_sentinel_and_unknown_events_are_ignored() -> None:
    frames = sse_fixture("stream_text.sse.txt")
    frames.insert(2, SseEvent(event=None, data="[DONE]"))
    frames.insert(3, sse({"type": "response.reasoning_summary_text.delta", "delta": "..."}))
    events = await drain(frames)
    assert [type(event) for event in events] == [
        StreamStart,
        TextDelta,
        TextDelta,
        ContinuationDelta,
        TerminalEvent,
    ]


async def test_stream_invalid_tool_arguments_raise_expected_failure() -> None:
    with pytest.raises(ExpectedFailureSignal):
        await drain(
            [
                sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": "fc_bad",
                            "type": "function_call",
                            "call_id": "call_bad",
                            "name": "search_library",
                            "arguments": "",
                        },
                    }
                ),
                sse(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "id": "fc_bad",
                            "type": "function_call",
                            "call_id": "call_bad",
                            "name": "search_library",
                            "arguments": "{broken",
                        },
                    }
                ),
            ]
        )


# ---------------------------------------------------------------------------
# classify_error


def test_classify_401_raises_credential_rejected_with_redaction() -> None:
    with pytest.raises(CredentialRejected) as excinfo:
        classify_error(401, {}, fixture_bytes("error_401.json"))
    assert "sk-abcDEF1234567890XYZ" not in excinfo.value.message
    assert "...redacted" in excinfo.value.message


def test_classify_403_raises_credential_rejected() -> None:
    with pytest.raises(CredentialRejected):
        classify_error(403, {}, b"")


def test_classify_429_returns_rate_limit_with_retry_after() -> None:
    assert classify_error(
        429, {"retry-after": "2.5"}, fixture_bytes("error_429.json")
    ) == ProviderRateLimit(retry_after=Present(2.5))


@pytest.mark.parametrize("headers", [{}, {"retry-after": "soon"}, {"retry-after": "-1"}])
def test_classify_429_without_usable_retry_after(headers: dict[str, str]) -> None:
    assert classify_error(429, headers, fixture_bytes("error_429.json")) == ProviderRateLimit(
        retry_after=Absent()
    )


def test_classify_429_insufficient_quota_is_quota_defect() -> None:
    with pytest.raises(RuntimeDefect) as excinfo:
        classify_error(429, {}, fixture_bytes("error_429_insufficient_quota.json"))
    assert excinfo.value.origin == "provider_http"
    assert excinfo.value.code == "quota_exhausted"


def test_classify_402_is_quota_defect() -> None:
    with pytest.raises(RuntimeDefect) as excinfo:
        classify_error(402, {}, b"")
    assert excinfo.value.code == "quota_exhausted"


def test_classify_context_length_exceeded_code() -> None:
    assert (
        classify_error(400, {}, fixture_bytes("error_400_context_length.json"))
        == ProviderContextTooLarge()
    )


def test_classify_maximum_context_length_message() -> None:
    body = json.dumps(
        {"error": {"message": "Your input exceeds this model's Maximum Context Length."}}
    ).encode()
    assert classify_error(400, {}, body) == ProviderContextTooLarge()


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_classify_5xx_unavailable(status: int) -> None:
    body = fixture_bytes("error_500.json") if status == 500 else b""
    assert classify_error(status, {}, body) == ProviderHttpUnavailable()


@pytest.mark.parametrize("status", [400, 404, 409, 501])
def test_classify_other_statuses_are_unclassified_defects(status: int) -> None:
    with pytest.raises(RuntimeDefect) as excinfo:
        classify_error(status, {}, json.dumps({"error": {"message": "weird"}}).encode())
    assert excinfo.value.code == "unclassified_provider_error"
    assert excinfo.value.origin == "provider_http"


# ---------------------------------------------------------------------------
# transcription port


def test_build_transcription_request_multipart_contract() -> None:
    request = build_transcription_request(
        model="gpt-4o-transcribe",
        filename="clip.mp3",
        audio=b"\x00\x01",
        media_type="audio/mpeg",
    )
    assert request.url == "https://api.openai.com/v1/audio/transcriptions"
    assert request.form_fields == {"model": "gpt-4o-transcribe", "response_format": "json"}
    assert request.filename == "clip.mp3"
    assert request.content == b"\x00\x01"
    assert request.media_type == "audio/mpeg"


def test_parse_transcription_response_success() -> None:
    body = json.dumps(
        {
            "text": "hello there",
            "usage": {"input_tokens": 14, "output_tokens": 3, "total_tokens": 17},
        }
    ).encode()
    result = parse_transcription_response(200, {"x-request-id": "req-t1"}, body)
    assert result.text == "hello there"
    assert result.provider_request_id == Present("req-t1")
    assert result.usage == Present(
        TokenUsage(
            input_tokens=14,
            output_tokens=3,
            total_tokens=17,
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Absent(),
            cache_write_input_tokens=Absent(),
        )
    )


def test_parse_transcription_response_derives_total_and_tolerates_missing_usage() -> None:
    derived = parse_transcription_response(
        200,
        {},
        json.dumps({"text": "x", "usage": {"input_tokens": 14, "output_tokens": 3}}).encode(),
    )
    assert isinstance(derived.usage, Present)
    assert derived.usage.value.total_tokens == 17
    bare = parse_transcription_response(200, {}, json.dumps({"text": "x"}).encode())
    assert bare.usage == Absent()
    assert bare.provider_request_id == Absent()


def test_parse_transcription_response_without_text_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        parse_transcription_response(200, {}, json.dumps({"status": "ok"}).encode())
