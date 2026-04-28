import pytest

from llm_calling.types import LLMChunk, LLMUsage


def test_non_terminal_chunk_cannot_include_usage() -> None:
    with pytest.raises(ValueError):
        LLMChunk(
            delta_text="x",
            done=False,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def test_non_terminal_chunk_cannot_include_terminal_status() -> None:
    with pytest.raises(ValueError):
        LLMChunk(delta_text="x", done=False, status="incomplete")


def test_non_terminal_chunk_cannot_include_incomplete_details() -> None:
    with pytest.raises(ValueError):
        LLMChunk(
            delta_text="x",
            done=False,
            incomplete_details={"reason": "max_output_tokens"},
        )
