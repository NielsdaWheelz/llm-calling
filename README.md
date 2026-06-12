# provider-runtime

Small async Python package for provider-level model calls.

The package owns the shared runtime contract: catalog validation, high-level request lowering,
HTTP request formatting, response parsing, streaming chunks, bounded provider retries, normalized
provider errors, key probes, embeddings, transcription, no-network test fakes, and deterministic cost estimates
from explicit catalog pricing. It supports OpenAI, Anthropic, Gemini, OpenRouter,
Cloudflare/OpenAI-compatible chat, OpenAI-compatible embeddings, and OpenAI
audio transcription.

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

from provider_runtime import (
    ModelCall,
    ModelMessage,
    ModelRef,
    ModelRuntime,
    ProviderApiKey,
    RetryPolicy,
)

async with httpx.AsyncClient() as http:
    runtime = ModelRuntime(http)
    response = await runtime.generate(
        ModelCall(
            model=ModelRef(provider="openai", model="gpt-5.4-mini"),
            messages=[ModelMessage(role="user", content="Write one sentence.")],
            max_output_tokens=128,
            retry=RetryPolicy(max_attempts=2),
        ),
        key=ProviderApiKey("sk-...", source="platform"),
        timeout_s=45,
    )
    print(response.text)
```

## Catalog And Cost

`DEFAULT_CATALOG` is the hard provider contract. Each row declares its operation surface
(`generation`, `embeddings`, `transcription`), provider-owned model ID, supported reasoning
controls, prompt-cache shape, normalized usage claims, artifact support, and pricing provenance.
The runtime rejects operation mismatches before provider I/O.

Catalog pricing is advisory and fail-closed. A cost estimate is returned only when normalized usage
and verified catalog rates are sufficient for the selected model and input size. Rates with provider
thresholds use `Pricing.applies_up_to_input_tokens`; calls above that threshold return
`missing_pricing` rather than a flattened under-estimate. Prices without a provider source URL and
verification date are treated as absent.

## Reasoning

`reasoning=ReasoningConfig(effort="default")` is the runtime-owned safe default for the
selected catalog row. Direct OpenAI catalog rows omit `reasoning` for `default`; explicit direct
OpenAI values currently include `"none"`, `"low"`, `"medium"`, and `"high"`, while product
`"max"` maps to OpenAI `"xhigh"`.

Gemini lowering is model-specific. Gemini 2.5 models use `thinkingBudget`; Gemini 3 models use
`thinkingLevel`. The runtime maps Gemini `default` to a cost-safe visible-output setting for each
model family instead of blindly using provider defaults that can consume small responses with
thinking tokens. Unsupported thinking-off modes, such as `none` on Gemini Pro models, are rejected
by catalog validation before a request reaches the provider adapter.

OpenRouter receives explicit `reasoning` controls, including `exclude=true`, for every catalog
reasoning mode. This keeps hidden reasoning out of the response while still letting callers choose
supported reasoning effort levels.

OpenAI responses preserve `status`, `incomplete_details`, `provider_request_id`, and
`usage.output_tokens_details.reasoning_tokens`. A `response.incomplete` result is returned
with `status="incomplete"` instead of being marked as a successful completion.

## Streaming

```python
import httpx

from provider_runtime import (
    ModelCall,
    ModelMessage,
    ModelRef,
    ModelRuntime,
    ProviderApiKey,
    RetryPolicy,
)

async with httpx.AsyncClient() as http:
    runtime = ModelRuntime(http, enable_openai=True)
    async for chunk in runtime.stream(
        ModelCall(
            model=ModelRef(provider="openai", model="gpt-5.4-mini"),
            messages=[ModelMessage(role="user", content="Stream one sentence.")],
            max_output_tokens=128,
            retry=RetryPolicy(max_attempts=2),
        ),
        key=ProviderApiKey("sk-...", source="platform"),
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

Default tests exclude live provider calls. The shared live matrix is the strict
provider-runtime acceptance gate and must be run explicitly after real keys are
available:

```bash
LLM_RUNTIME_LIVE=1 uv run pytest -v -m live_provider tests/live/test_provider_matrix.py
```

By default it covers every generation row in `DEFAULT_CATALOG`, every declared reasoning effort,
OpenAI, Anthropic, Gemini, OpenRouter, Cloudflare, embeddings, and transcription.
For focused diagnosis, set `LLM_RUNTIME_LIVE_PROVIDERS=openai,anthropic`; narrowed runs are
debugging aids and do not count as acceptance proof.
Required key env vars are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `OPENROUTER_API_KEY`, and for Cloudflare both
`CLOUDFLARE_AI_API_TOKEN` and `CLOUDFLARE_AI_ACCOUNT_ID`.
