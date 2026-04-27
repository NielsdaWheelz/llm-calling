import pytest

from llm_calling.types import LLMChunk, LLMUsage


def test_non_terminal_chunk_cannot_include_usage() -> None:
    with pytest.raises(ValueError):
        LLMChunk(
            delta_text="x",
            done=False,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
