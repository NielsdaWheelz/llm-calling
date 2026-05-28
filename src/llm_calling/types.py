"""Shared types for provider-level LLM calls."""

from dataclasses import dataclass, field
from typing import Literal

ReasoningEffort = Literal["default", "none", "minimal", "low", "medium", "high", "max"]
ProviderName = Literal["openai", "anthropic", "gemini", "deepseek"]
PromptCacheTTL = Literal["none", "5m", "1h"]
ToolChoice = Literal["auto", "none", "required"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class Turn:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    cache_ttl: PromptCacheTTL = "none"
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cached_tokens: int | None = None
    provider_usage: dict[str, object] | None = None


@dataclass(frozen=True)
class StructuredOutputSpec:
    name: str
    schema: dict[str, object]
    strict: bool = True


@dataclass(frozen=True)
class LLMRequest:
    model_name: str
    messages: list[Turn]
    max_tokens: int
    temperature: float | None = None
    reasoning_effort: ReasoningEffort = "none"
    prompt_cache_key: str | None = None
    structured_output: StructuredOutputSpec | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = "auto"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: LLMUsage | None
    provider_request_id: str | None
    status: str | None = None
    incomplete_details: dict[str, object] | None = None
    structured_output: dict[str, object] | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LLMChunk:
    delta_text: str = ""
    tool_call: ToolCall | None = None
    done: bool = False
    usage: LLMUsage | None = None
    provider_request_id: str | None = None
    status: str | None = None
    incomplete_details: dict[str, object] | None = None

    def __post_init__(self):
        if not self.done and self.usage is not None:
            raise ValueError("Non-terminal chunks (done=False) must have usage=None")
        if not self.done and self.status is not None:
            raise ValueError("Non-terminal chunks (done=False) must have status=None")
        if not self.done and self.incomplete_details is not None:
            raise ValueError("Non-terminal chunks (done=False) must have incomplete_details=None")
