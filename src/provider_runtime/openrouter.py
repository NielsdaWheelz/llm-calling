"""OpenRouter chat client."""

import httpx

from provider_runtime.openai_compatible import OpenAICompatibleChatClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(OpenAICompatibleChatClient):
    def __init__(self, client: httpx.AsyncClient, *, base_url: str = OPENROUTER_BASE_URL):
        super().__init__(client, provider="openrouter", base_url=base_url)
