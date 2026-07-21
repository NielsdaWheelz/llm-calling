"""OpenRouter codec tests: provider-block golden (all routing fields),
session_id finalize injection, metadata-header presence, upstream_provider
extraction precedence, reasoning_details replay round-trip, in-band mid-stream
error handling, stream_request injection determinism, classify_error table."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from provider_runtime import openrouter
from provider_runtime._signals import TransientStreamError
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
    OutputSpec,
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
    UserMessage,
)

FIXTURES = Path(__file__).parent / "fixtures" / "openrouter"
TARGET = ProviderTarget(provider="openrouter", model="moonshotai/kimi-k3-20260715")
CONTRACT = CATALOG.chat_contract(TARGET)

EXPECTED_PROVIDER_BLOCK: dict[str, object] = {
    "only": ["moonshotai/int4"],
    "order": ["moonshotai/int4"],
    "allow_fallbacks": False,
    "require_parameters": True,
    "data_collection": "deny",
    "zdr": True,
    "quantizations": ["int4"],
}

TOOL_PARAMETERS_RAW: dict[str, object] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
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
    return [event async for event in openrouter.decode_stream({}, aiter_events(events))]


# ---------------------------------------------------------------------------
# encode goldens


def test_encode_golden_body_and_headers() -> None:
    draft = openrouter.encode(make_intent(), CONTRACT)
    assert json.loads(draft.body) == {
        "model": "moonshotai/kimi-k3-20260715",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Why do tides rise?"},
        ],
        "max_tokens": 512,
        "reasoning": {"effort": "max", "exclude": False},
        "provider": EXPECTED_PROVIDER_BLOCK,
    }
    assert draft.url == "https://openrouter.ai/api/v1/chat/completions"
    assert draft.protocol == "openrouter_chat"
    assert draft.safe_headers == {
        "X-OpenRouter-Cache": "false",
        "X-OpenRouter-Metadata": "enabled",
    }
    assert draft.native_reasoning == "max"


def test_encode_uses_routed_max_tokens_not_direct_field() -> None:
    body = json.loads(openrouter.encode(make_intent(), CONTRACT).body)
    assert body["max_tokens"] == 512
    assert "max_completion_tokens" not in body


@pytest.mark.parametrize("level", ["low", "high", "max"])
def test_encode_reasoning_effort_identity_with_exclude_false(level: ReasoningLevel) -> None:
    body = json.loads(openrouter.encode(make_intent(reasoning=level), CONTRACT).body)
    assert body["reasoning"] == {"effort": level, "exclude": False}


def test_encode_tools_and_strict_output() -> None:
    tools_body = json.loads(openrouter.encode(make_intent(tools=(LOOKUP_TOOL,)), CONTRACT).body)
    assert tools_body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look things up.",
                "parameters": TOOL_PARAMETERS_RAW,
            },
        }
    ]
    assert tools_body["tool_choice"] == "auto"
    schema_raw: dict[str, object] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    output = StrictJsonOutput(name="answer", schema=parse_canonical_schema(schema_raw))
    strict_body = json.loads(openrouter.encode(make_intent(output=output), CONTRACT).body)
    assert strict_body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": schema_raw, "strict": True},
    }


def test_encode_rejects_mismatched_continuation() -> None:
    artifact = ContinuationArtifact(
        target=TARGET, codec_id="moonshot_chat", opaque_payload={"reasoning_details": []}
    )
    messages = BASE_MESSAGES + (
        AssistantMessage(text="x", tool_calls=(), continuation=Present(artifact)),
    )
    with pytest.raises(PlanningDefect):
        openrouter.encode(make_intent(messages=messages), CONTRACT)


def test_encode_rejects_payload_without_reasoning_details() -> None:
    artifact = ContinuationArtifact(
        target=TARGET, codec_id=openrouter.CODEC_ID, opaque_payload={"role": "assistant"}
    )
    messages = BASE_MESSAGES + (
        AssistantMessage(text="x", tool_calls=(), continuation=Present(artifact)),
    )
    with pytest.raises(PlanningDefect):
        openrouter.encode(make_intent(messages=messages), CONTRACT)


# ---------------------------------------------------------------------------
# finalize / stream_request


def test_finalize_injects_session_id_deterministically() -> None:
    draft = openrouter.encode(make_intent(), CONTRACT)
    final = openrouter.finalize(draft, "affinity-abc")
    body = json.loads(final.body)
    assert body["session_id"] == "affinity-abc"
    del body["session_id"]
    assert body == json.loads(draft.body)
    assert final.body == draft.body[:-1] + b',"session_id":"affinity-abc"}'
    assert openrouter.finalize(draft, "affinity-abc").body == final.body
    assert b"session_id" not in draft.body  # draft is never mutated
    assert final.method == "POST"
    assert final.safe_headers == draft.safe_headers


def test_stream_request_adds_only_stream_true() -> None:
    final = openrouter.finalize(openrouter.encode(make_intent(), CONTRACT), "aff")
    streaming = openrouter.stream_request(final)
    assert streaming.body == final.body[:-1] + b',"stream":true}'
    body = json.loads(streaming.body)
    assert body["stream"] is True
    assert "stream_options" not in body
    assert openrouter.stream_request(final).body == streaming.body


# ---------------------------------------------------------------------------
# prefix_bytes


def test_prefix_bytes_deterministic_and_dynamic_insensitive() -> None:
    assert (
        openrouter.encode(make_intent(), CONTRACT).prefix_bytes
        == openrouter.encode(make_intent(), CONTRACT).prefix_bytes
    )
    other_dynamic: tuple[PromptMessage, ...] = (
        BASE_MESSAGES[0],
        UserMessage(blocks=(PromptBlock(text="Completely different.", stability=Dynamic()),)),
    )
    assert (
        openrouter.encode(make_intent(messages=other_dynamic), CONTRACT).prefix_bytes
        == openrouter.encode(make_intent(), CONTRACT).prefix_bytes
    )


def test_prefix_bytes_sensitive_to_tools_schema_and_stable_blocks() -> None:
    base = openrouter.encode(make_intent(), CONTRACT).prefix_bytes
    with_tools = openrouter.encode(make_intent(tools=(LOOKUP_TOOL,)), CONTRACT).prefix_bytes
    output = StrictJsonOutput(
        name="answer",
        schema=parse_canonical_schema(
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        ),
    )
    with_schema = openrouter.encode(make_intent(output=output), CONTRACT).prefix_bytes
    other_stable: tuple[PromptMessage, ...] = (
        SystemMessage(
            blocks=(PromptBlock(text="You are verbose.", stability=Stable(scope=GlobalScope())),)
        ),
        BASE_MESSAGES[1],
    )
    with_other_stable = openrouter.encode(make_intent(messages=other_stable), CONTRACT).prefix_bytes
    assert len({base, with_tools, with_schema, with_other_stable}) == 4


def test_prefix_bytes_sensitive_to_role_move_system_vs_leading_user() -> None:
    # The same stable text framed as a SystemMessage vs. a leading UserMessage
    # must produce distinct prefix_bytes: role/message placement participates
    # in the projection, not bare joined text.
    stable_text = "Same stable text."
    as_system = openrouter.encode(
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
    as_leading_user = openrouter.encode(
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
    one_message = openrouter.encode(
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
    two_messages = openrouter.encode(
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
    impersonating = openrouter.encode(
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
    with_matching_tool = openrouter.encode(make_intent(tools=(LOOKUP_TOOL,)), CONTRACT).prefix_bytes
    assert impersonating != with_matching_tool


# ---------------------------------------------------------------------------
# decode_response


def test_decode_response_success_with_metadata() -> None:
    outcome = openrouter.decode_response(200, {}, fixture_bytes("success_reasoning_details.json"))
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(text="Tides follow the moon.", tool_calls=())
    assert outcome.meta.provider == "openrouter"
    assert outcome.meta.model == "moonshotai/kimi-k3-20260715"
    assert outcome.meta.provider_request_id == Present("gen-or-001")  # generation id
    assert outcome.meta.upstream_provider == Present("moonshotai/int4")
    assert outcome.meta.usage == Present(
        TokenUsage(
            input_tokens=200,
            output_tokens=60,
            total_tokens=260,
            reasoning_tokens=Present(24),
            cache_read_input_tokens=Present(128),
            cache_write_input_tokens=Present(16),
        )
    )


def test_upstream_provider_extraction_precedence() -> None:
    data = fixture_json("success_reasoning_details.json")

    metadata_less = dict(data)
    del metadata_less["openrouter_metadata"]
    outcome = openrouter.decode_response(200, {}, json.dumps(metadata_less).encode())
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.upstream_provider == Present("Moonshot AI")

    endpoints_only = dict(data)
    endpoints_only["openrouter_metadata"] = {"endpoints": ["moonshotai/int4"]}
    outcome = openrouter.decode_response(200, {}, json.dumps(endpoints_only).encode())
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.upstream_provider == Present("moonshotai/int4")

    bare = dict(data)
    del bare["openrouter_metadata"]
    del bare["provider"]
    outcome = openrouter.decode_response(200, {}, json.dumps(bare).encode())
    assert isinstance(outcome, Succeeded)
    assert outcome.meta.upstream_provider == Absent()


def test_reasoning_details_replay_round_trip() -> None:
    outcome = openrouter.decode_response(200, {}, fixture_bytes("success_reasoning_details.json"))
    assert isinstance(outcome, Succeeded)
    continuation = outcome.response.continuation
    assert isinstance(continuation, Present)
    artifact = continuation.value
    fixture = fixture_json("success_reasoning_details.json")
    choices = fixture["choices"]
    assert isinstance(choices, list)
    expected_details = choices[0]["message"]["reasoning_details"]
    assert artifact.target == TARGET
    assert artifact.codec_id == "openrouter_chat"
    assert artifact.opaque_payload == {"reasoning_details": expected_details}

    messages = BASE_MESSAGES + (
        AssistantMessage(text="Tides follow the moon.", tool_calls=(), continuation=continuation),
        UserMessage(blocks=(PromptBlock(text="And why twice a day?", stability=Dynamic()),)),
    )
    body = json.loads(openrouter.encode(make_intent(messages=messages), CONTRACT).body)
    assert body["messages"][2] == {
        "role": "assistant",
        "content": "Tides follow the moon.",
        "reasoning_details": expected_details,  # verbatim, in sequence
    }


def test_decode_response_malformed_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        openrouter.decode_response(200, {}, b'{"choices": []}')


# ---------------------------------------------------------------------------
# decode_stream


async def test_stream_happy_path() -> None:
    events = await drive(sse_events("stream_happy.txt"))
    assert isinstance(events[0], StreamStart)
    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "Tides ",
        "follow the moon.",
    ]

    continuations = [event for event in events if isinstance(event, ContinuationDelta)]
    assert len(continuations) == 1
    payload_details = continuations[0].artifact.opaque_payload["reasoning_details"]
    assert isinstance(payload_details, list)
    assert [detail["id"] for detail in payload_details] == ["rd-1", "rd-2"]

    terminal = events[-1]
    assert isinstance(terminal, TerminalEvent)
    assert events[-2] is continuations[0]
    outcome = terminal.outcome
    assert isinstance(outcome, Succeeded)
    assert outcome.response.content == TextContent(text="Tides follow the moon.", tool_calls=())
    assert outcome.meta.provider_request_id == Present("gen-or-002")
    assert outcome.meta.upstream_provider == Present("moonshotai/int4")  # from metadata chunk
    assert outcome.meta.usage == Present(
        TokenUsage(
            input_tokens=150,
            output_tokens=30,
            total_tokens=180,
            reasoning_tokens=Present(12),
            cache_read_input_tokens=Present(100),
            cache_write_input_tokens=Absent(),
        )
    )


async def test_stream_inband_5xx_error_chunk_is_transient_unavailable() -> None:
    with pytest.raises(TransientStreamError) as exc_info:
        await drive(sse_events("stream_inband_error.txt"))
    assert exc_info.value.cause == ProviderHttpUnavailable()


async def test_stream_inband_429_error_chunk_is_rate_limit() -> None:
    events = [
        SseEvent(
            event=None,
            data='{"id":"gen-or-004","error":{"code":429,"message":"Rate limited"},'
            '"choices":[{"index":0,"delta":{},"finish_reason":"error"}]}',
        ),
    ]
    with pytest.raises(TransientStreamError) as exc_info:
        await drive(events)
    assert exc_info.value.cause == ProviderRateLimit(retry_after=Absent())


async def test_stream_finish_reason_error_without_error_object_is_transient_unavailable() -> None:
    # Off-spec finish chunk: finish_reason "error" but no top-level error
    # object — still keyed as transient on finish_reason, never a spurious
    # unknown-finish-reason ProtocolDefect or strict-parsed tool arguments.
    with pytest.raises(TransientStreamError) as exc_info:
        await drive(sse_events("stream_finish_error_no_body.txt"))
    assert exc_info.value.cause == ProviderHttpUnavailable()


async def test_stream_missing_done_is_transient_stream_error() -> None:
    events = [
        SseEvent(
            event=None,
            data='{"id":"gen-or-005","model":"moonshotai/kimi-k3-20260715",'
            '"choices":[{"index":0,"delta":{"content":"hi"}}]}',
        ),
    ]
    with pytest.raises(TransientStreamError) as exc_info:
        await drive(events)
    assert exc_info.value.cause == ProviderStreamInterrupted(partial_output=False)


# ---------------------------------------------------------------------------
# classify_error


def test_classify_401_raises_credential_rejected() -> None:
    with pytest.raises(CredentialRejected):
        openrouter.classify_error(401, {}, b'{"error": {"code": 401, "message": "key"}}')


def test_classify_403_without_moderation_metadata_raises_credential_rejected() -> None:
    with pytest.raises(CredentialRejected):
        openrouter.classify_error(403, {}, b'{"error": {"code": 403, "message": "key"}}')


def test_classify_403_moderation_flagged_is_runtime_defect() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        openrouter.classify_error(403, {}, fixture_bytes("error_moderation_403.json"))
    assert exc_info.value.code == "input_moderation_flagged"
    assert exc_info.value.origin == "provider_response"


def test_classify_402_is_quota_defect() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        openrouter.classify_error(
            402, {}, b'{"error": {"code": 402, "message": "Insufficient credits"}}'
        )
    assert exc_info.value.code == "quota_exhausted"
    assert exc_info.value.origin == "provider_http"


def test_classify_429_rate_limit_with_retry_after() -> None:
    classified = openrouter.classify_error(
        429, {"retry-after": "7"}, b'{"error": {"code": 429, "message": "Rate limited"}}'
    )
    assert classified == ProviderRateLimit(retry_after=Present(7.0))


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_classify_5xx_is_provider_unavailable(status: int) -> None:
    assert openrouter.classify_error(status, {}, b"") == ProviderHttpUnavailable()


def test_classify_context_overflow() -> None:
    classified = openrouter.classify_error(400, {}, fixture_bytes("error_context_too_large.json"))
    assert classified == ProviderContextTooLarge()


def test_classify_unknown_400_includes_sanitized_provider_code() -> None:
    with pytest.raises(RuntimeDefect) as exc_info:
        openrouter.classify_error(400, {}, fixture_bytes("error_provider_code.json"))
    assert exc_info.value.code == "unclassified_provider_error"
    assert "provider_code=invalid_request_error" in exc_info.value.message


def test_classify_unparseable_400_is_protocol_defect() -> None:
    with pytest.raises(ProtocolDefect):
        openrouter.classify_error(400, {}, b"<html>gateway</html>")


def test_classify_error_details_are_redacted() -> None:
    body = json.dumps(
        {"error": {"code": 400, "message": "bad key sk-abcdefghijk1234567890"}}
    ).encode()
    with pytest.raises(RuntimeDefect) as exc_info:
        openrouter.classify_error(400, {}, body)
    assert "sk-abcdefghijk1234567890" not in exc_info.value.message
