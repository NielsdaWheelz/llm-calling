# llm-calling

Small async Python package for provider-level LLM calls.

The package owns HTTP request formatting, response parsing, streaming chunks, and normalized provider
errors for OpenAI, Anthropic, Gemini, and DeepSeek. Callers own prompts, API keys, HTTP client
lifecycle, logging, persistence, retries, model policy, and product behavior.

## Install

```bash
uv add llm-calling
```

Python `>=3.12` is required.

## Direct Client

```python
import httpx

from llm_calling.openai import OpenAIClient
from llm_calling.types import LLMRequest, Turn

async with httpx.AsyncClient() as http:
    client = OpenAIClient(http)
    response = await client.generate(
        LLMRequest(
            model_name="gpt-5.4-mini",
            messages=[Turn(role="user", content="Write one sentence.")],
            max_tokens=128,
        ),
        api_key="sk-...",
        timeout_s=45,
    )
    print(response.text)
```

## Router

```python
import httpx

from llm_calling.router import LLMRouter
from llm_calling.types import LLMRequest, Turn

async with httpx.AsyncClient() as http:
    router = LLMRouter(http, enable_openai=True)
    async for chunk in router.generate_stream(
        "openai",
        LLMRequest(
            model_name="gpt-5.4-mini",
            messages=[Turn(role="user", content="Stream one sentence.")],
            max_tokens=128,
        ),
        api_key="sk-...",
    ):
        if not chunk.done:
            print(chunk.delta_text, end="")
```
