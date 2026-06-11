# provider-runtime

Small async Python package for provider-level model calls.

The package owns the shared runtime contract: catalog validation, high-level request lowering,
HTTP request formatting, response parsing, streaming chunks, bounded provider retries, normalized
provider errors, key probes, embeddings, no-network test fakes, and deterministic cost estimates
from explicit catalog pricing. It supports OpenAI, Anthropic, Gemini, OpenRouter,
Cloudflare/OpenAI-compatible chat, and OpenAI-compatible embeddings.

Callers own prompts, API keys, HTTP client lifecycle, logging, persistence, application
idempotency, and product behavior. Provider adapter modules are internal implementation details;
application code should use `provider_runtime.ModelRuntime`.

## Install

```bash
uv add provider-runtime
```

Python `>=3.12` is required.

## Runtime

```python
import httpx

from provider_runtime import ModelCall, ModelMessage, ModelRef, ModelRuntime, RetryPolicy

async with httpx.AsyncClient() as http:
    runtime = ModelRuntime(http)
    response = await runtime.generate(
        ModelCall(
            model=ModelRef(provider="openai", model="gpt-5.4-mini"),
            messages=[ModelMessage(role="user", content="Write one sentence.")],
            max_output_tokens=128,
            retry=RetryPolicy(max_attempts=2),
        ),
        key="sk-...",
        timeout_s=45,
    )
    print(response.text)
```

## Reasoning

`reasoning=ReasoningConfig(effort="default")` leaves OpenAI `reasoning` unset. Pass it explicitly when product
policy should use the OpenAI API default. Explicit OpenAI reasoning values map directly for
`"none"`, `"minimal"`, `"low"`, `"medium"`, and `"high"`; product `"max"` maps to OpenAI
`"xhigh"`.

OpenAI responses preserve `status`, `incomplete_details`, `provider_request_id`, and
`usage.output_tokens_details.reasoning_tokens`. A `response.incomplete` result is returned
with `status="incomplete"` instead of being marked as a successful completion.

## Streaming

```python
import httpx

from provider_runtime import ModelCall, ModelMessage, ModelRef, ModelRuntime, RetryPolicy

async with httpx.AsyncClient() as http:
    runtime = ModelRuntime(http, enable_openai=True)
    async for chunk in runtime.stream(
        ModelCall(
            model=ModelRef(provider="openai", model="gpt-5.4-mini"),
            messages=[ModelMessage(role="user", content="Stream one sentence.")],
            max_output_tokens=128,
            retry=RetryPolicy(max_attempts=2),
        ),
        key="sk-...",
    ):
        if not chunk.done:
            print(chunk.delta_text, end="")
```

Retries are bounded and limited to normalized retryable provider failures: timeouts, connection
failures, rate limits, and 5xx-class provider outages. Streaming calls retry only before any chunk
has been yielded; after a visible delta, tool call, or provider artifact escapes, the error is
returned to the caller so the application can restart the durable run under its own idempotency
rules.

## Tests

Use `ScriptedRuntime` or `NoNetworkRuntime` from `provider_runtime.testing` for application tests.
They expose the runtime interface without opening provider network connections.
