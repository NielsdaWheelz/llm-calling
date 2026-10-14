"""Public dispatch and its process-level SDK resource boundary."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

import provider_runtime
from provider_runtime import (
    Credentials,
    GenerateIntent,
    PromptBlock,
    ProviderRuntime,
    ProviderTarget,
    Succeeded,
    TerminalEvent,
    TextContent,
    TextDelta,
    TextOutput,
    UserMessage,
)
from provider_runtime.types import ProviderName
from tests.engines import test_anthropic_messages as anthropic
from tests.engines import test_gemini_generate as gemini
from tests.engines import test_openai_chat as chat
from tests.engines import test_openai_responses as responses


def test_public_embedding_call_does_not_load_unused_provider_sdks() -> None:
    # A fresh process is essential: the engine conformance tests import every SDK.
    # This resource boundary prevents an embedding-only API from retaining them.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import asyncio, base64, json, sys
                from array import array
                sys.path.insert(0, sys.argv[1])
                from provider_runtime import (
                    Absent, Credentials, EmbeddingCall, ProviderCredential, ProviderRuntime,
                )
                runtime = ProviderRuntime(Credentials())
                sdks = ('openai', 'anthropic', 'google.genai')
                assert not any(name in sys.modules for name in sdks), [
                    name for name in sdks if name in sys.modules
                ]
                import httpx
                calls = []
                def respond(request):
                    calls.append(request)
                    assert str(request.url) == 'https://api.openai.com/v1/embeddings'
                    assert json.loads(request.content) == {
                        'model': 'text-embedding-3-small', 'input': ['alpha'],
                        'encoding_format': 'base64',
                    }
                    return httpx.Response(200, json={
                        'object': 'list', 'model': 'text-embedding-3-small',
                        'data': [{'object': 'embedding', 'index': 0, 'embedding':
                            base64.b64encode(array('f', [0.5, -1.25]).tobytes()).decode()}],
                        'usage': {'prompt_tokens': 1, 'total_tokens': 1},
                    })
                async def main():
                    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                        runtime = ProviderRuntime(Credentials(), http_client=client)
                        result = await runtime.embed(
                            EmbeddingCall(model='text-embedding-3-small', inputs=('alpha',),
                                          dimensions=Absent()),
                            credential=ProviderCredential(provider='openai', key='test-key'),
                        )
                        assert result.embeddings == ((0.5, -1.25),), result
                        assert not client.is_closed, 'the caller owns its HTTP client'
                asyncio.run(main())
                assert len(calls) == 1, len(calls)
                assert 'openai' in sys.modules
                assert 'anthropic' not in sys.modules
                assert 'google.genai' not in sys.modules
                assert not any(name in sys.modules for name in (
                    'provider_runtime.engines.openai_responses',
                    'provider_runtime.engines.openai_chat',
                    'provider_runtime.engines.anthropic_messages',
                    'provider_runtime.engines.gemini_generate',
                ))
                """
            ),
            str(Path(provider_runtime.__file__).resolve().parents[1]),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("provider", "model", "reasoning", "path"),
    [
        ("openai", "gpt-6-sol", "standard/medium", "/v1/responses"),
        ("anthropic", "claude-sonnet-5", "adaptive/high", "/v1/messages"),
        ("gemini", "gemini-3.8-flash", "medium", "/v1beta/models/gemini-3.8-flash"),
        ("deepseek", "deepseek-flash", "none", "/chat/completions"),
    ],
)
async def test_default_runtime_generate_and_stream_use_the_selected_sdk(
    provider: ProviderName, model: str, reasoning: str, path: str
) -> None:
    # Reuse the conformance envelopes; intercept only the external HTTP boundary.
    if provider == "openai":
        body = responses.envelope(output=[responses.TEXT_ITEM], model=model)
        frames = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": responses.envelope(output=[], status="in_progress", model=model),
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "output_index": 0,
                "content_index": 0,
                "item_id": "msg_1",
                "delta": "hello",
            },
            {"type": "response.completed", "sequence_number": 2, "response": body},
        ]
        streamed = responses.sse_bytes(frames)
    elif provider == "anthropic":
        body = anthropic.envelope(content=[anthropic.TEXT_BLOCK], model=model)
        frames = [
            {
                "type": "message_start",
                "message": anthropic.envelope(
                    content=[], model=model, stop_reason=None, usage=anthropic.usage_body()
                ),
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            anthropic.text_delta_frame(0, "hello"),
            {"type": "content_block_stop", "index": 0},
            anthropic.message_delta_frame(stop_reason="end_turn"),
            {"type": "message_stop"},
        ]
        streamed = anthropic.sse_bytes(frames)
    elif provider == "gemini":
        body = gemini.response_body(model_version=model)
        streamed = gemini.sse_bytes(body)
    else:
        body = chat.completion_body(model=model)
        streamed = chat.sse_bytes(
            {
                "id": "resp-1",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "resp-1",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            "[DONE]",
        )
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        is_stream = len(calls) == 2
        expected_path = path + (
            (":streamGenerateContent" if is_stream else ":generateContent")
            if provider == "gemini"
            else ""
        )
        assert request.url.path == expected_path, request.url
        assert (
            request.url.host
            == {
                "openai": "api.openai.com",
                "anthropic": "api.anthropic.com",
                "gemini": "generativelanguage.googleapis.com",
                "deepseek": "api.deepseek.com",
            }[provider]
        ), request.url
        assert len(calls) <= 2, "one HTTP attempt per public call"
        if provider != "gemini":
            sent = json.loads(request.content)
            assert sent["model"] == model
            assert bool(sent.get("stream")) == is_stream
        if is_stream:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=streamed
            )
        return httpx.Response(200, json=body)

    intent = GenerateIntent(
        target=ProviderTarget(provider=provider, model=model),
        messages=(UserMessage((PromptBlock("hi"),)),),
        max_output_tokens=128,
        reasoning=reasoning,
        tools=(),
        tool_choice="auto",
        output=TextOutput(),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        runtime = ProviderRuntime(
            Credentials(
                openai="test-key", anthropic="test-key", gemini="test-key", deepseek="test-key"
            ),
            http_client=client,
        )
        outcome = await runtime.generate(intent)
        assert isinstance(outcome, Succeeded), outcome
        assert outcome.response.content == TextContent(text="hello", tool_calls=())
        assert (outcome.meta.provider, outcome.meta.model) == (provider, model)
        events = [event async for event in runtime.stream(intent)]
        assert [event.seq for event in events] == list(range(1, len(events) + 1))
        assert (
            "".join(event.event.text for event in events if isinstance(event.event, TextDelta))
            == "hello"
        )
        terminal = events[-1].event
        assert isinstance(terminal, TerminalEvent), events
        assert isinstance(terminal.outcome, Succeeded), terminal
        assert terminal.outcome.response.content == outcome.response.content
        assert len(calls) == 2
        assert not client.is_closed, "the caller owns its HTTP client"


async def test_injected_empty_engine_map_never_falls_back_to_default_dispatch() -> None:
    runtime = ProviderRuntime(Credentials(openai="test-key"), engines={})
    intent = GenerateIntent(
        target=ProviderTarget(provider="openai", model="gpt-6-sol"),
        messages=(UserMessage((PromptBlock("hi"),)),),
        max_output_tokens=128,
        reasoning="standard/medium",
        tools=(),
        tool_choice="auto",
        output=TextOutput(),
    )
    with pytest.raises(KeyError, match="openai_responses"):
        await runtime.generate(intent)
    # stream has always validated and selected the engine before iteration.
    with pytest.raises(KeyError, match="openai_responses"):
        runtime.stream(intent)
