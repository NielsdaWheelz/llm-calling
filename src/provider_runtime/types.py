"""Shared types for provider-level model calls."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

ReasoningEffort = Literal["default", "none", "minimal", "low", "medium", "high", "max"]
ProviderName = Literal["openai", "anthropic", "gemini", "openrouter", "cloudflare"]
PromptCacheTTL = Literal["none", "5m", "1h"]
ToolChoice = Literal["auto", "none", "required"]

# Verbatim provider payload fragment. Opaque: captured from responses and
# replayed unmodified on continuation requests, never interpreted.
ProviderArtifact = Mapping[str, object]


@dataclass(frozen=True)
class ModelRef:
    provider: ProviderName
    model: str
    route: str | None = None


@dataclass(frozen=True)
class ReasoningConfig:
    effort: ReasoningEffort = "none"
    budget_tokens: int | None = None


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    initial_delay_s: float = 0.25
    max_delay_s: float = 2.0

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("RetryPolicy.max_attempts must be >= 1")
        if self.initial_delay_s < 0:
            raise ValueError("RetryPolicy.initial_delay_s must be >= 0")
        if self.max_delay_s < 0:
            raise ValueError("RetryPolicy.max_delay_s must be >= 0")


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class BinaryPart:
    data: bytes = field(repr=False)
    media_type: str
    filename: str | None = None


ContentPart = TextPart | BinaryPart


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, object]
    strict: bool = True


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]
    argument_status: Literal["valid", "repaired"] = "valid"
    provider_metadata: Mapping[str, object] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class ModelMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    content_parts: tuple[ContentPart, ...] = ()
    cache_ttl: PromptCacheTTL = "none"
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    provider_artifacts: tuple[ProviderArtifact, ...] = field(default_factory=tuple, repr=False)


@dataclass(frozen=True)
class TokenUsage:
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
class ModelCall:
    model: ModelRef
    messages: list[ModelMessage]
    max_output_tokens: int
    temperature: float | None = None
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    prompt_cache_key: str | None = None
    structured_output: StructuredOutputSpec | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = "auto"
    retry: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: TokenUsage | None
    provider_request_id: str | None
    status: str | None = None
    incomplete_details: dict[str, object] | None = None
    structured_output: dict[str, object] | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    provider_artifacts: tuple[ProviderArtifact, ...] = field(default_factory=tuple, repr=False)


@dataclass(frozen=True)
class ModelChunk:
    delta_text: str = ""
    tool_call: ToolCall | None = None
    provider_artifact: ProviderArtifact | None = field(default=None, repr=False)
    done: bool = False
    usage: TokenUsage | None = None
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


@dataclass(frozen=True)
class EmbeddingCall:
    model: ModelRef
    inputs: list[str]
    dimensions: int | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class EmbeddingResponse:
    embeddings: list[list[float]]
    usage: TokenUsage | None
    provider_request_id: str | None


@dataclass(frozen=True)
class KeyProbeResult:
    provider: ProviderName
    model: str
    ok: bool
    error_code: str | None = None
    provider_request_id: str | None = None
