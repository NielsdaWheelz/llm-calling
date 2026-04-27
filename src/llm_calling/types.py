"""Shared types for provider-level LLM calls."""

from dataclasses import dataclass
from typing import Literal

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "max"]
ProviderName = Literal["openai", "anthropic", "gemini", "deepseek"]


@dataclass(frozen=True)
class Turn:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class LLMRequest:
    model_name: str
    messages: list[Turn]
    max_tokens: int
    temperature: float | None = None
    reasoning_effort: ReasoningEffort = "none"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: LLMUsage | None
    provider_request_id: str | None


@dataclass(frozen=True)
class LLMChunk:
    delta_text: str
    done: bool
    usage: LLMUsage | None = None
    provider_request_id: str | None = None

    def __post_init__(self):
        if not self.done and self.usage is not None:
            raise ValueError("Non-terminal chunks (done=False) must have usage=None")
