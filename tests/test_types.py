import pytest

from provider_runtime.types import (
    BinaryPart,
    ModelChunk,
    ModelMessage,
    ModelResponse,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


def test_non_terminal_chunk_cannot_include_usage() -> None:
    with pytest.raises(ValueError):
        ModelChunk(
            delta_text="x",
            done=False,
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def test_non_terminal_chunk_cannot_include_terminal_status() -> None:
    with pytest.raises(ValueError):
        ModelChunk(delta_text="x", done=False, status="incomplete")


def test_non_terminal_chunk_cannot_include_incomplete_details() -> None:
    with pytest.raises(ValueError):
        ModelChunk(
            delta_text="x",
            done=False,
            incomplete_details={"reason": "max_output_tokens"},
        )


def test_non_terminal_chunk_may_carry_provider_artifact() -> None:
    chunk = ModelChunk(provider_artifact={"type": "reasoning", "id": "rs_1"}, done=False)

    assert chunk.provider_artifact == {"type": "reasoning", "id": "rs_1"}


def test_provider_artifacts_are_excluded_from_dataclass_repr() -> None:
    artifact = {
        "type": "reasoning",
        "encrypted_content": "opaque-provider-secret",
        "signature": "sig-secret",
        "thinking": "private thought",
    }

    message = ModelMessage(role="assistant", provider_artifacts=(artifact,))
    response = ModelResponse(
        text="",
        usage=None,
        provider_request_id="req_123",
        provider_artifacts=(artifact,),
    )
    chunk = ModelChunk(provider_artifact=artifact)
    tool_call = ToolCall(
        id="call_1",
        name="lookup",
        arguments={},
        provider_metadata={"thoughtSignature": "tool-secret"},
    )

    combined = "\n".join(map(repr, (message, response, chunk, tool_call)))
    assert "opaque-provider-secret" not in combined
    assert "sig-secret" not in combined
    assert "private thought" not in combined
    assert "tool-secret" not in combined


def test_content_parts_can_carry_text_and_binary_payloads() -> None:
    message = ModelMessage(
        role="user",
        content_parts=(
            TextPart(text="inspect this"),
            BinaryPart(data=b"opaque", media_type="image/png", filename="image.png"),
        ),
    )

    assert message.content_parts[0] == TextPart(text="inspect this")
    assert isinstance(message.content_parts[1], BinaryPart)
    assert message.content_parts[1].media_type == "image/png"


def test_tool_contract_defaults_to_strict_valid_arguments() -> None:
    tool = ToolSpec(name="lookup", description="Lookup.", parameters={"type": "object"})
    call = ToolCall(id="call_1", name="lookup", arguments={})

    assert tool.strict is True
    assert call.argument_status == "valid"
