"""Shared types for provider-level model calls."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

ReasoningEffort = Literal["default", "none", "minimal", "low", "medium", "high", "max"]
ProviderName = Literal["openai", "anthropic", "gemini", "openrouter", "cloudflare"]
ProviderApiKeySource = Literal["platform", "byok", "probe", "test"]
PromptCacheTTL = Literal["none", "5m", "1h"]
ToolChoice = Literal["auto", "none", "required"]
RetryableErrorCode = Literal["rate_limit", "timeout", "provider_down"]

ProviderArtifactPurpose = Literal["reasoning", "thinking", "signature", "provider_item"]
ProviderArtifactRetention = Literal["ephemeral", "durable"]
RetryAttemptStatus = Literal["success", "retryable_error", "terminal_error", "abandoned"]


@dataclass(frozen=True)
class ProviderArtifact:
    """Opaque provider replay payload with explicit classification metadata."""

    provider: ProviderName
    model: str
    purpose: ProviderArtifactPurpose
    payload: Mapping[str, object] = field(repr=False)
    retention: ProviderArtifactRetention = "ephemeral"

    def to_provider_payload(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass(frozen=True)
class ModelRef:
    provider: ProviderName
    model: str
    route: str | None = None


@dataclass(frozen=True)
class ProviderApiKey:
    """Opaque provider credential passed across the public runtime boundary."""

    value: str = field(repr=False)
    source: ProviderApiKeySource

    def __post_init__(self):
        if not self.value:
            raise ValueError("ProviderApiKey.value must be non-empty")

    def reveal(self) -> str:
        return self.value

    def __str__(self) -> str:
        return "<provider-api-key redacted>"


@dataclass(frozen=True)
class ReasoningConfig:
    effort: ReasoningEffort = "none"
    budget_tokens: int | None = None


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    initial_delay_s: float = 0.25
    max_delay_s: float = 2.0
    deadline_s: float | None = None
    jitter_s: float = 0.0
    retryable_error_codes: tuple[RetryableErrorCode, ...] = (
        "rate_limit",
        "timeout",
        "provider_down",
    )

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("RetryPolicy.max_attempts must be >= 1")
        if self.initial_delay_s < 0:
            raise ValueError("RetryPolicy.initial_delay_s must be >= 0")
        if self.max_delay_s < 0:
            raise ValueError("RetryPolicy.max_delay_s must be >= 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("RetryPolicy.deadline_s must be > 0")
        if self.jitter_s < 0:
            raise ValueError("RetryPolicy.jitter_s must be >= 0")
        allowed = {"rate_limit", "timeout", "provider_down"}
        if any(code not in allowed for code in self.retryable_error_codes):
            raise ValueError("RetryPolicy.retryable_error_codes contains an unsupported code")


@dataclass(frozen=True)
class RetryAttempt:
    """Stable, redacted retry metadata for runtime-owned provider attempts."""

    attempt_number: int
    max_attempts: int
    status: RetryAttemptStatus
    error_code: str | None = None
    status_code: int | None = None
    retryable: bool | None = None
    retry_after_seconds: float | None = None
    delay_s: float | None = None
    provider_request_id: str | None = None
    safe_body_snippet: str | None = None
    streamed_output_started: bool = False

    def __post_init__(self):
        if self.attempt_number < 1:
            raise ValueError("RetryAttempt.attempt_number must be >= 1")
        if self.max_attempts < 1:
            raise ValueError("RetryAttempt.max_attempts must be >= 1")
        if self.attempt_number > self.max_attempts:
            raise ValueError("RetryAttempt.attempt_number must be <= max_attempts")
        if self.status == "success" and self.error_code is not None:
            raise ValueError("Successful retry attempts must not carry error_code")
        if self.delay_s is not None and self.delay_s < 0:
            raise ValueError("RetryAttempt.delay_s must be >= 0")
        if self.retry_after_seconds is not None and self.retry_after_seconds < 0:
            raise ValueError("RetryAttempt.retry_after_seconds must be >= 0")

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
            "status": self.status,
            "streamed_output_started": self.streamed_output_started,
        }
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.retryable is not None:
            payload["retryable"] = self.retryable
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        if self.delay_s is not None:
            payload["delay_s"] = self.delay_s
        if self.provider_request_id is not None:
            payload["provider_request_id"] = self.provider_request_id
        if self.safe_body_snippet is not None:
            payload["safe_body_snippet"] = self.safe_body_snippet
        return payload


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
    provider_metadata: dict[str, object] | None = field(default=None, repr=False)


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
    prompt_cache_key: str | None = field(default=None, init=False, repr=False, compare=False)
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
    attempts: tuple[RetryAttempt, ...] = field(default_factory=tuple, repr=False)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts) if self.attempts else 1

    @property
    def retry_count(self) -> int:
        return max(0, self.attempt_count - 1)

    @property
    def terminal_attempt_status(self) -> RetryAttemptStatus:
        return self.attempts[-1].status if self.attempts else "success"


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
    attempts: tuple[RetryAttempt, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self):
        if not self.done and self.usage is not None:
            raise ValueError("Non-terminal chunks (done=False) must have usage=None")
        if not self.done and self.status is not None:
            raise ValueError("Non-terminal chunks (done=False) must have status=None")
        if not self.done and self.incomplete_details is not None:
            raise ValueError("Non-terminal chunks (done=False) must have incomplete_details=None")
        if not self.done and self.attempts:
            raise ValueError("Non-terminal chunks (done=False) must have attempts=()")

    @property
    def attempt_count(self) -> int:
        return len(self.attempts) if self.attempts else 1

    @property
    def retry_count(self) -> int:
        return max(0, self.attempt_count - 1)

    @property
    def terminal_attempt_status(self) -> RetryAttemptStatus:
        return self.attempts[-1].status if self.attempts else "success"


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
    attempts: tuple[RetryAttempt, ...] = field(default_factory=tuple, repr=False)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts) if self.attempts else 1

    @property
    def retry_count(self) -> int:
        return max(0, self.attempt_count - 1)

    @property
    def terminal_attempt_status(self) -> RetryAttemptStatus:
        return self.attempts[-1].status if self.attempts else "success"


@dataclass(frozen=True)
class TranscriptionCall:
    model: ModelRef
    audio: bytes = field(repr=False)
    filename: str
    media_type: str
    retry: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class TranscriptionResponse:
    text: str
    usage: TokenUsage | None
    provider_request_id: str | None
    attempts: tuple[RetryAttempt, ...] = field(default_factory=tuple, repr=False)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts) if self.attempts else 1

    @property
    def retry_count(self) -> int:
        return max(0, self.attempt_count - 1)

    @property
    def terminal_attempt_status(self) -> RetryAttemptStatus:
        return self.attempts[-1].status if self.attempts else "success"


@dataclass(frozen=True)
class KeyProbeResult:
    provider: ProviderName
    model: str
    ok: bool
    error_code: str | None = None
    provider_request_id: str | None = None
    status: str | None = None
    usage: TokenUsage | None = None
    attempts: tuple[RetryAttempt, ...] = field(default_factory=tuple, repr=False)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts) if self.attempts else 1

    @property
    def retry_count(self) -> int:
        return max(0, self.attempt_count - 1)
