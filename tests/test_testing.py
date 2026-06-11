import pytest

from provider_runtime import (
    EmbeddingCall,
    EmbeddingResponse,
    KeyProbeResult,
    ModelCall,
    ModelChunk,
    ModelMessage,
    ModelRef,
    ModelResponse,
    NoNetworkRuntime,
    ScriptedRuntime,
    TokenUsage,
)

pytestmark = pytest.mark.asyncio


def _call() -> ModelCall:
    return ModelCall(
        model=ModelRef(provider="openai", model="gpt-5.4-mini"),
        messages=[ModelMessage(role="user", content="hello")],
        max_output_tokens=16,
    )


async def test_no_network_runtime_defects_on_provider_io() -> None:
    with pytest.raises(AssertionError, match="Unexpected provider-runtime generate"):
        await NoNetworkRuntime().generate(_call(), key="sk-test")


async def test_scripted_runtime_returns_queued_generate_response_and_records_call() -> None:
    runtime = ScriptedRuntime(
        generate_responses=(
            ModelResponse(
                text="ok",
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                provider_request_id="req-test",
            ),
        )
    )

    response = await runtime.generate(_call(), key="sk-test", timeout_s=12)

    assert response.text == "ok"
    assert len(runtime.calls) == 1
    assert runtime.calls[0].operation == "generate"
    assert runtime.calls[0].key == "sk-test"
    assert runtime.calls[0].timeout_s == 12


async def test_scripted_runtime_returns_queued_stream_chunks() -> None:
    runtime = ScriptedRuntime(stream_chunks=((ModelChunk(delta_text="a"), ModelChunk(done=True)),))

    chunks = [chunk async for chunk in runtime.stream(_call(), key="sk-test")]

    assert [chunk.delta_text for chunk in chunks] == ["a", ""]
    assert chunks[-1].done is True
    assert runtime.calls[0].operation == "stream"


async def test_scripted_runtime_returns_queued_embeddings_and_key_probe() -> None:
    runtime = ScriptedRuntime(
        embed_responses=(
            EmbeddingResponse(
                embeddings=[[0.1, 0.2]],
                usage=TokenUsage(input_tokens=1, output_tokens=None, total_tokens=1),
                provider_request_id="req-emb",
            ),
        ),
        probe_results=(
            KeyProbeResult(
                provider="openai",
                model="gpt-5.4-mini",
                ok=True,
                provider_request_id="req-probe",
            ),
        ),
    )

    embedding = await runtime.embed(
        EmbeddingCall(
            model=ModelRef(provider="openai", model="text-embedding-3-small"), inputs=["x"]
        ),
        key="sk-test",
    )
    probe = await runtime.probe_key(provider="openai", key="sk-test")

    assert embedding.provider_request_id == "req-emb"
    assert probe.ok is True
    assert [call.operation for call in runtime.calls] == ["embed", "probe_key"]
