"""Bound, canonical continuation and complete native turn replay."""

import json
from dataclasses import replace

import pytest

from provider_runtime.continuation import (
    decode_continuation,
    encode_continuation,
    pending_tool_calls,
    preflight_generation,
    resume_generation,
    seal_generation,
)
from provider_runtime.engines.anthropic_messages import _encode_request
from provider_runtime.errors import InvalidRequest
from provider_runtime.registry import _resolve
from provider_runtime.runtime import Credentials, ProviderRuntime
from provider_runtime.types import (
    AssistantMessage,
    CanonicalTool,
    ContinuationArtifact,
    ContinueGeneration,
    GenerateIntent,
    Present,
    PromptBlock,
    ProviderTarget,
    ResponsePayload,
    TextContent,
    TextOutput,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def test_canonical_codec_requires_current_target_and_codec() -> None:
    target = ProviderTarget("openai", "gpt-6-sol")
    artifact = ContinuationArtifact(target, "openai.v2", {"private": [{"signed": "data"}]})
    encoded = encode_continuation(artifact)
    assert encode_continuation(decode_continuation(encoded, target, "openai.v2")) == encoded
    with pytest.raises(InvalidRequest, match="target"):
        decode_continuation(encoded, ProviderTarget("openai", "gpt-6-luna"), "openai.v2")
    with pytest.raises(InvalidRequest, match="codec"):
        decode_continuation(encoded, target, "openai.v1")
    with pytest.raises(InvalidRequest, match="canonical"):
        decode_continuation(json.dumps(json.loads(encoded), indent=2).encode(), target, "openai.v2")
    with pytest.raises(InvalidRequest, match="16 MiB"):
        encode_continuation(
            ContinuationArtifact(target, "openai.v2", {"large": "x" * (16 * 1024 * 1024)})
        )


def test_continuation_bound_applies_only_to_tool_capable_initial_requests() -> None:
    row = _resolve("openai:gpt-6-sol")
    target = ProviderTarget("openai", row.model_id)
    prompt = UserMessage((PromptBlock("x" * (16 * 1024 * 1024)),))
    intent = GenerateIntent(target, (prompt,), 64, "standard/medium", (), "auto", TextOutput())
    preflight_generation(row, intent)
    tool = CanonicalTool("lookup", "look up", {"type": "object"})
    preflight_generation(row, replace(intent, tools=(tool,), tool_choice="none"))
    with pytest.raises(InvalidRequest, match="cannot fit a continuation"):
        preflight_generation(row, replace(intent, tools=(tool,)))


@pytest.mark.asyncio
async def test_three_turn_claude_continuation_preserves_signed_order_and_first_result() -> None:
    row = _resolve("anthropic:claude-fable-5-1")
    target = ProviderTarget("anthropic", row.model_id)
    first = GenerateIntent(
        target,
        (UserMessage((PromptBlock("find facts"),)),),
        4096,
        "adaptive/high",
        (CanonicalTool("lookup", "look up", {"type": "object"}),),
        "auto",
        TextOutput(),
    )
    state = preflight_generation(row, first)
    native = ContinuationArtifact(
        target,
        row.continuation_codec,
        {
            "blocks": [
                {"type": "thinking", "thinking": "", "signature": "signed-empty"},
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"key": "first"}},
            ]
        },
    )
    first_response = ResponsePayload(
        TextContent("checking", (ToolCall("call-1", "lookup", {"key": "first"}),)), Present(native)
    )
    sealed = seal_generation(row, first, state, first_response)
    assert isinstance(sealed.continuation, Present)
    assert pending_tool_calls(sealed.continuation.value) == (
        ToolCall("call-1", "lookup", {"key": "first"}),
    )
    oversized = ContinueGeneration(
        sealed.continuation.value,
        (ToolResultMessage("call-1", "x" * (16 * 1024 * 1024), False),),
    )
    with pytest.raises(InvalidRequest, match="16 MiB"):
        resume_generation(row, oversized)
    with pytest.raises(InvalidRequest, match="16 MiB"):
        await ProviderRuntime(Credentials(), engines={}).generate(oversized)
    with pytest.raises(InvalidRequest, match="exact order"):
        resume_generation(
            row,
            ContinueGeneration(
                sealed.continuation.value, (ToolResultMessage("wrong", "alpha", False),)
            ),
        )
    second, state = resume_generation(
        row,
        ContinueGeneration(
            sealed.continuation.value, (ToolResultMessage("call-1", "alpha", False),)
        ),
    )
    assistant = _encode_request(row, second).params["messages"][1]
    assert [block["type"] for block in assistant["content"]] == ["thinking", "text", "tool_use"]
    assert assistant["content"][0]["signature"] == "signed-empty"
    second_native = ContinuationArtifact(
        target,
        row.continuation_codec,
        {
            "blocks": [
                {"type": "thinking", "thinking": "", "signature": "signed-two"},
                {"type": "tool_use", "id": "call-2", "name": "lookup", "input": {"key": "second"}},
            ]
        },
    )
    second_response = ResponsePayload(
        TextContent("", (ToolCall("call-2", "lookup", {"key": "second"}),)), Present(second_native)
    )
    sealed = seal_generation(row, second, state, second_response)
    assert isinstance(sealed.continuation, Present)
    assert pending_tool_calls(sealed.continuation.value) == (
        ToolCall("call-2", "lookup", {"key": "second"}),
    )
    third, _ = resume_generation(
        row,
        ContinueGeneration(
            sealed.continuation.value, (ToolResultMessage("call-2", "beta", False),)
        ),
    )
    assert isinstance(third.messages[-3], ToolResultMessage)
    assert third.messages[-3].output == "alpha"
    assert isinstance(third.messages[-2], AssistantMessage)
    assert third.messages[-2].tool_calls[0].id == "call-2"
    assert (
        _encode_request(row, third).params["messages"][3]["content"][0]["signature"] == "signed-two"
    )
