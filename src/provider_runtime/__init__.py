"""Public provider-runtime API — the high-traffic surface only.

The facade re-exports the names the 95% call site touches: the runtime and
its credentials, the intent vocabulary, the terminal outcomes, the stream
envelope, the embed port, and derived cost estimation. The full contract
vocabulary stays importable from ``provider_runtime.types``; the sole public
provider-catalog query from ``provider_runtime.registry``; test doubles from
``provider_runtime.testing``. Registry rows and resolution remain private to
the runtime. The HTTP runtime is loaded only when either of its two exports is
read, so importing the independent agent-runtime package does not initialize
every HTTP provider engine.
"""

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from provider_runtime.errors import NonGenerationCallFailed
from provider_runtime.prices import estimate_cost
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

if _TYPE_CHECKING:
    from provider_runtime.runtime import Credentials, ProviderRuntime

_RUNTIME_EXPORTS = frozenset({"Credentials", "ProviderRuntime"})


def __getattr__(name: str) -> _Any:
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module("provider_runtime.runtime"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RUNTIME_EXPORTS)


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
