import httpx
import pytest
import respx

from provider_runtime import ModelRuntime, ProviderApiKey, ProviderBaseUrls
from provider_runtime.errors import ModelCallError, ModelCallErrorCode
from provider_runtime.types import ModelRef, RetryPolicy, TranscriptionCall

pytestmark = pytest.mark.asyncio

KEY = ProviderApiKey("sk-test", source="test")


def call(*, retry: RetryPolicy | None = None) -> TranscriptionCall:
    return TranscriptionCall(
        model=ModelRef(provider="openai", model="gpt-4o-transcribe"),
        audio=b"RIFF...",
        filename="clip.wav",
        media_type="audio/wav",
        retry=retry or RetryPolicy(max_attempts=1),
    )


@respx.mock
async def test_runtime_transcribes_with_openai_base_url() -> None:
    route = respx.post("https://openai-proxy.test/v1/audio/transcriptions").respond(
        200,
        json={
            "text": "hello world",
            "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        },
        headers={"x-request-id": "req-transcribe-1"},
    )

    async with httpx.AsyncClient() as http:
        response = await ModelRuntime(
            http,
            base_urls=ProviderBaseUrls(openai="https://openai-proxy.test/v1"),
        ).transcribe(call(), key=KEY)

    assert route.called
    assert response.text == "hello world"
    assert response.provider_request_id == "req-transcribe-1"
    assert response.usage is not None
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3


@respx.mock
async def test_runtime_retries_retryable_transcription_errors_before_success() -> None:
    route = respx.post("https://api.openai.com/v1/audio/transcriptions").mock(
        side_effect=[
            httpx.Response(503, json={"error": {"message": "temporarily unavailable"}}),
            httpx.Response(200, json={"text": "ok"}),
        ]
    )

    async with httpx.AsyncClient() as http:
        response = await ModelRuntime(http).transcribe(
            call(retry=RetryPolicy(max_attempts=2, initial_delay_s=0)),
            key=KEY,
        )

    assert route.call_count == 2
    assert response.text == "ok"
    assert [attempt.status for attempt in response.attempts] == ["retryable_error", "success"]
    assert response.attempts[0].error_code == ModelCallErrorCode.PROVIDER_DOWN.value
    assert response.retry_count == 1


@respx.mock
async def test_terminal_parser_exceptions_do_not_retry_transcription_calls() -> None:
    route = respx.post("https://api.openai.com/v1/audio/transcriptions").mock(
        side_effect=TypeError("'NoneType' object is not subscriptable")
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await ModelRuntime(http).transcribe(
                call(retry=RetryPolicy(max_attempts=3, initial_delay_s=0)),
                key=KEY,
            )

    assert route.call_count == 1
    assert exc_info.value.retryable is False
    assert [attempt.status for attempt in exc_info.value.attempts] == ["terminal_error"]


async def test_runtime_rejects_unknown_transcription_model() -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await ModelRuntime(http).transcribe(
                TranscriptionCall(
                    model=ModelRef(provider="openai", model="not-a-transcription-model"),
                    audio=b"x",
                    filename="clip.wav",
                    media_type="audio/wav",
                ),
                key=KEY,
            )

    assert exc_info.value.error_code == ModelCallErrorCode.MODEL_NOT_AVAILABLE


async def test_runtime_rejects_unconfigured_transcription_provider() -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelCallError) as exc_info:
            await ModelRuntime(http).transcribe(
                TranscriptionCall(
                    model=ModelRef(provider="anthropic", model="claude-haiku-4-5-20251001"),
                    audio=b"x",
                    filename="clip.wav",
                    media_type="audio/wav",
                ),
                key=KEY,
            )

    assert exc_info.value.error_code == ModelCallErrorCode.MODEL_NOT_AVAILABLE
