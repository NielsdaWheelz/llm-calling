"""Public provider-runtime API — the high-traffic surface only.

The facade re-exports the names the 95% call site touches: the runtime and
its credentials, the intent vocabulary, the terminal outcomes, the stream
envelope, the embed port, and derived cost estimation. The FULL contract
vocabulary stays importable from ``provider_runtime.types``; registry rows
from ``provider_runtime.registry``; test doubles from
``provider_runtime.testing``.
"""

from provider_runtime.errors import NonGenerationCallFailed
from provider_runtime.prices import estimate_cost
from provider_runtime.runtime import Credentials, ProviderRuntime
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    CallMeta,
    CallOutcome,
    Cancelled,
    CanonicalTool,
    ContinuationDelta,
    EmbeddingCall,
    EmbeddingResponse,
    Failed,
    GenerateIntent,
    ImageBlock,
    Incomplete,
    Present,
    PromptBlock,
    ProviderCredential,
    ProviderTarget,
    ReasoningLevel,
    Refused,
    RuntimeStreamEvent,
    StreamStart,
    StructuredContent,
    StructuredReply,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextContent,
    TextDelta,
    TextOutput,
    TokenUsage,
    ToolCallDelta,
    ToolCallDone,
    ToolCallStart,
    ToolResultMessage,
    UsageEvent,
    UserMessage,
)

__all__ = [
    "Absent",
    "AssistantMessage",
    "CallMeta",
    "CallOutcome",
    "Cancelled",
    "CanonicalTool",
    "ContinuationDelta",
    "Credentials",
    "EmbeddingCall",
    "EmbeddingResponse",
    "Failed",
    "GenerateIntent",
    "ImageBlock",
    "Incomplete",
    "NonGenerationCallFailed",
    "Present",
    "PromptBlock",
    "ProviderCredential",
    "ProviderRuntime",
    "ProviderTarget",
    "ReasoningLevel",
    "Refused",
    "RuntimeStreamEvent",
    "StreamStart",
    "StructuredContent",
    "StructuredReply",
    "Succeeded",
    "SystemMessage",
    "TerminalEvent",
    "TextContent",
    "TextDelta",
    "TextOutput",
    "TokenUsage",
    "ToolCallDelta",
    "ToolCallDone",
    "ToolCallStart",
    "ToolResultMessage",
    "UsageEvent",
    "UserMessage",
    "estimate_cost",
]
