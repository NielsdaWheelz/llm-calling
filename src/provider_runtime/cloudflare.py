"""Cloudflare Workers AI OpenAI-compatible chat client."""

import httpx

from provider_runtime.openai_compatible import OpenAICompatibleChatClient

CLOUDFLARE_AI_BASE_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"


def cloudflare_ai_base_url(account_id: str) -> str:
    return CLOUDFLARE_AI_BASE_URL_TEMPLATE.format(account_id=account_id)


class CloudflareClient(OpenAICompatibleChatClient):
    def __init__(self, client: httpx.AsyncClient, *, base_url: str):
        super().__init__(client, provider="cloudflare", base_url=base_url)
