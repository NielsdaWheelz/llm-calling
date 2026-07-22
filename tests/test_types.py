"""Tests for the rewritten provider_runtime.types value contract."""

import dataclasses

import pytest

from provider_runtime.types import (
    Absent,
    AnthropicPrefix,
    AttemptRecord,
    CallMeta,
    Cancelled,
    ContinuationArtifact,
    Dynamic,
    Failed,
    FinalAttempt,
    GeminiAutomaticPrefix,
    GlobalScope,
    Incomplete,
    IntentContextTooLarge,
    InvalidToolArguments,
    MoonshotKeyedPrefix,
    OpenAIExplicitPrefix,
    OpenRouterCertifiedPrefix,
    PossiblyBillable,
    Present,
    PromptBlock,
    ProviderContextTooLarge,
    ProviderHttpUnavailable,
    ProviderRateLimit,
    ProviderStreamInterrupted,
    ProviderTarget,
    ProviderTimeout,
    ResponsePayload,
    RetryPolicy,
    RuntimeStreamEvent,
    Stable,
    StreamStart,
    Succeeded,
    TextContent,
    TokenUsage,
    ToolCall,
    TransientExhausted,
    TransportUnavailable,
    cache_strategy,
    cache_ttl,
    failure_code,
    failure_origin,
    presence_of,
)

# ---------------------------------------------------------------------------
# failure_origin / failure_code — exhaustive 8-leaf golden table


def _exhausted(cause) -> TransientExhausted:
    return TransientExhausted(attempts=3, cause=cause)


GOLDEN_FAILURE_TABLE = [
    pytest.param(
        _exhausted(ProviderRateLimit(retry_after=Present(2.5))),
        "provider_http",
        "rate_limited",
        id="transient-rate-limit",
    ),
    pytest.param(
        _exhausted(ProviderTimeout()),
        "transport",
        "timeout",
        id="transient-timeout",
    ),
    pytest.param(
        _exhausted(ProviderHttpUnavailable()),
        "provider_http",
        "provider_unavailable",
        id="transient-provider-http-unavailable",
    ),
    pytest.param(
        _exhausted(TransportUnavailable()),
        "transport",
        "provider_unavailable",
        id="transient-transport-unavailable",
    ),
    pytest.param(
        _exhausted(ProviderStreamInterrupted(partial_output=True)),
        "provider_stream",
        "stream_interrupted",
        id="transient-stream-interrupted",
    ),
    pytest.param(
        IntentContextTooLarge(limit=100_000, measured=140_000),
        "intent",
        "context_too_large",
        id="intent-context-too-large",
    ),
    pytest.param(
        ProviderContextTooLarge(),
        "provider_http",
        "context_too_large",
        id="provider-context-too-large",
    ),
    pytest.param(
        InvalidToolArguments(safe_detail="arguments were not strict JSON"),
        "tool_arguments",
        "invalid_tool_arguments",
        id="invalid-tool-arguments",
    ),
]


def test_golden_failure_table_covers_all_eight_leaves() -> None:
    # 3 non-transient leaves + 5 transient causes wrapped in TransientExhausted.
    assert len(GOLDEN_FAILURE_TABLE) == 8, (
        "the golden failure table must stay exhaustive over the closed "
        "ExpectedModelFailure x TransientCause leaf set (8 pairs)"
    )


@pytest.mark.parametrize(("failure", "expected_origin", "expected_code"), GOLDEN_FAILURE_TABLE)
def test_failure_origin_and_code_golden_pairs(failure, expected_origin, expected_code) -> None:
    origin = failure_origin(failure)
    code = failure_code(failure)
    assert origin == expected_origin, (
        f"failure_origin({failure!r}) must be {expected_origin!r}, got {origin!r}"
    )
    assert code == expected_code, (
        f"failure_code({failure!r}) must be {expected_code!r}, got {code!r}"
    )


# ---------------------------------------------------------------------------
# TokenUsage.from_components — the one shared total-derivation rule


def test_from_components_uses_provider_reported_total_when_present() -> None:
    usage = TokenUsage.from_components(
        input_tokens=100,
        output_tokens=50,
        total_tokens=Present(175),
        reasoning_tokens=Present(25),
        cache_read_input_tokens=Absent(),
        cache_write_input_tokens=Absent(),
    )
    assert usage.total_tokens == 175, (
        "a provider-reported total is authoritative and must never be re-derived"
    )


def test_from_components_derives_input_plus_output_when_total_absent() -> None:
    # TokenUsage.input_tokens is ALWAYS the cache-INCLUSIVE total prompt token
    # count (the codec-invariant convention) — the derived total must be
    # plain input + output, never re-adding the cache components, or a codec
    # that already folded cache into input_tokens (Anthropic, post-
    # normalization) would be double-counted.
    usage = TokenUsage.from_components(
        input_tokens=1240,  # already inclusive: 40 raw + 1000 cache_read + 200 cache_write
        output_tokens=10,
        total_tokens=Absent(),
        reasoning_tokens=Absent(),
        cache_read_input_tokens=Present(1000),
        cache_write_input_tokens=Present(200),
    )
    assert usage.total_tokens == 1240 + 10, (
        f"derived total must be input + output (1250), got {usage.total_tokens}"
    )
    assert usage.cache_read_input_tokens == Present(1000)
    assert usage.cache_write_input_tokens == Present(200)


def test_from_components_derivation_treats_absent_cache_components_as_zero() -> None:
    usage = TokenUsage.from_components(
        input_tokens=7,
        output_tokens=3,
        total_tokens=Absent(),
        reasoning_tokens=Absent(),
        cache_read_input_tokens=Absent(),
        cache_write_input_tokens=Absent(),
    )
    assert usage.total_tokens == 10, "with no cache components the total is input + output"


def test_token_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="input_tokens must be >= 0"):
        TokenUsage(
            input_tokens=-1,
            output_tokens=0,
            total_tokens=0,
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Absent(),
            cache_write_input_tokens=Absent(),
        )
    with pytest.raises(ValueError, match="cache_read_input_tokens must be >= 0"):
        TokenUsage(
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            reasoning_tokens=Absent(),
            cache_read_input_tokens=Present(-5),
            cache_write_input_tokens=Absent(),
        )


# ---------------------------------------------------------------------------
# Presence


def test_presence_of_normalizes_none_to_absent() -> None:
    assert presence_of(None) == Absent(), "None at a boundary must become owned Absent"


def test_presence_of_wraps_values_including_falsy_ones() -> None:
    assert presence_of(0) == Present(0), "falsy values are still Present"
    assert presence_of("req_abc") == Present("req_abc")


def test_presence_values_have_structural_equality() -> None:
    assert Present(3) == Present(3)
    assert Present(3) != Present(4)
    assert Absent() == Absent()
    assert Present(3) != Absent()


# ---------------------------------------------------------------------------
# AttemptRecord validation


def test_attempt_record_accepts_a_clean_single_attempt() -> None:
    record = AttemptRecord(
        attempt=1,
        signal=FinalAttempt(),
        status_code=Present(200),
        started_at_ms=1000,
        ended_at_ms=1000,
    )
    assert record.attempt == 1
    assert record.signal == FinalAttempt()


def test_attempt_record_rejects_attempt_below_one() -> None:
    with pytest.raises(ValueError, match="attempt must be >= 1"):
        AttemptRecord(
            attempt=0,
            signal=FinalAttempt(),
            status_code=Absent(),
            started_at_ms=0,
            ended_at_ms=1,
        )


def test_attempt_record_rejects_ending_before_start() -> None:
    with pytest.raises(ValueError, match="ended_at_ms must be >= started_at_ms"):
        AttemptRecord(
            attempt=1,
            signal=ProviderTimeout(),
            status_code=Absent(),
            started_at_ms=2000,
            ended_at_ms=1999,
        )


def test_transient_exhausted_rejects_attempts_below_one() -> None:
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        TransientExhausted(attempts=0, cause=ProviderTimeout())


# ---------------------------------------------------------------------------
# Frozen-ness


def _sample_meta() -> CallMeta:
    return CallMeta(
        provider="anthropic",
        model="claude-fable-5",
        provider_request_id=Present("req_123"),
        upstream_provider=Absent(),
        usage=Present(
            TokenUsage.from_components(
                input_tokens=12,
                output_tokens=4,
                total_tokens=Absent(),
                reasoning_tokens=Present(2),
                cache_read_input_tokens=Absent(),
                cache_write_input_tokens=Absent(),
            )
        ),
        attempt_trace=(
            AttemptRecord(
                attempt=1,
                signal=ProviderRateLimit(retry_after=Present(1.0)),
                status_code=Present(429),
                started_at_ms=10,
                ended_at_ms=20,
            ),
            AttemptRecord(
                attempt=2,
                signal=FinalAttempt(),
                status_code=Present(200),
                started_at_ms=30,
                ended_at_ms=90,
            ),
        ),
        billability=PossiblyBillable(),
    )


def test_value_types_are_frozen() -> None:
    meta = _sample_meta()
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.provider = "gemini"  # type: ignore
    block = PromptBlock(text="system prompt", stability=Stable(scope=GlobalScope()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        block.text = "mutated"  # type: ignore
    event = RuntimeStreamEvent(seq=1, event=StreamStart())
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.seq = 2  # type: ignore


# ---------------------------------------------------------------------------
# CallMeta / outcome construction


def test_succeeded_outcome_retains_meta_and_attempt_trace() -> None:
    meta = _sample_meta()
    outcome = Succeeded(
        meta=meta,
        response=ResponsePayload(
            content=TextContent(
                text="hello",
                tool_calls=(ToolCall(id="call_1", name="app_search", arguments={"q": "x"}),),
            ),
            continuation=Absent(),
        ),
    )
    assert outcome.meta.provider == "anthropic"
    # attempt_count = len(trace); retry_count = len(trace) - 1.
    assert len(outcome.meta.attempt_trace) == 2, "every outcome branch retains the attempt trace"
    assert outcome.meta.attempt_trace[-1].signal == FinalAttempt(), (
        "the last trace record must be the terminal attempt"
    )
    assert outcome.response.content == TextContent(
        text="hello",
        tool_calls=(ToolCall(id="call_1", name="app_search", arguments={"q": "x"}),),
    )


def test_streamed_fable_refusal_is_incomplete_with_refused_status() -> None:
    outcome = Incomplete(
        meta=_sample_meta(),
        reason="content_filter_partial",
        status="refused",
        safe_detail=Present("refused by model"),
    )
    assert outcome.status == "refused"
    assert outcome.safe_detail == Present("refused by model")


def test_failed_and_cancelled_outcomes_carry_meta() -> None:
    meta = _sample_meta()
    failure = TransientExhausted(attempts=2, cause=ProviderRateLimit(retry_after=Absent()))
    failed = Failed(meta=meta, failure=failure)
    assert failed.failure == failure
    assert Cancelled(meta=meta).meta == meta


def test_continuation_artifact_requires_a_mapping_payload() -> None:
    target = ProviderTarget(provider="openai", model="gpt-5.6-terra")
    artifact = ContinuationArtifact(
        target=target,
        codec_id="openai_responses",
        opaque_payload={"output": []},
    )
    assert artifact.codec_id == "openai_responses"
    with pytest.raises(TypeError, match="opaque_payload must be a Mapping"):
        ContinuationArtifact(
            target=target,
            codec_id="openai_responses",
            opaque_payload=[("output", [])],  # type: ignore
        )


def test_continuation_artifact_payload_never_enters_repr() -> None:
    artifact = ContinuationArtifact(
        target=ProviderTarget(provider="openai", model="gpt-5.6-terra"),
        codec_id="openai_responses",
        opaque_payload={"encrypted_content": "SECRET-REPLAY-MATERIAL"},
    )
    assert "SECRET-REPLAY-MATERIAL" not in repr(artifact), (
        "continuation payloads are ephemeral and must never be rendered"
    )


def test_runtime_stream_event_seq_is_one_based() -> None:
    assert RuntimeStreamEvent(seq=1, event=StreamStart()).seq == 1
    with pytest.raises(ValueError, match="seq must be >= 1"):
        RuntimeStreamEvent(seq=0, event=StreamStart())


# ---------------------------------------------------------------------------
# Plan value types


def test_cache_strategy_and_ttl_accessors_cover_all_five_plans() -> None:
    openai_plan = OpenAIExplicitPrefix(key="affinity", minimum_ttl="30m", breakpoints=1)
    anthropic_plan = AnthropicPrefix(stable_breakpoint=2, ttl="5m", automatic_append_only=True)
    gemini_plan = GeminiAutomaticPrefix()
    moonshot_plan = MoonshotKeyedPrefix(key="affinity")
    openrouter_plan = OpenRouterCertifiedPrefix(
        session_id="affinity",
        pinned_upstream="moonshotai/int4",
        evidence_revision="cert-2026-07",
    )

    assert cache_strategy(openai_plan) == "openai_explicit_prefix"
    assert cache_ttl(openai_plan) == "30m"

    assert cache_strategy(anthropic_plan) == "anthropic_prefix"
    assert cache_ttl(anthropic_plan) == "5m"

    assert cache_strategy(gemini_plan) == "gemini_automatic_prefix"
    assert cache_ttl(gemini_plan) is None, "automatic prefix plans declare no wire TTL"

    assert cache_strategy(moonshot_plan) == "moonshot_keyed_prefix"
    assert cache_ttl(moonshot_plan) is None

    assert cache_strategy(openrouter_plan) == "openrouter_certified_prefix"
    assert cache_ttl(openrouter_plan) is None


def test_retry_policy_validates_its_budget() -> None:
    policy = RetryPolicy(
        max_attempts=3,
        initial_delay_s=0.25,
        max_delay_s=2.0,
        jitter_s=0.1,
        deadline_s=Present(30.0),
    )
    assert policy.max_attempts == 3
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        RetryPolicy(
            max_attempts=0,
            initial_delay_s=0.0,
            max_delay_s=0.0,
            jitter_s=0.0,
            deadline_s=Absent(),
        )
    with pytest.raises(ValueError, match="deadline_s must be > 0"):
        RetryPolicy(
            max_attempts=1,
            initial_delay_s=0.0,
            max_delay_s=0.0,
            jitter_s=0.0,
            deadline_s=Present(0.0),
        )


def test_prompt_block_permits_empty_dynamic_text() -> None:
    block = PromptBlock(text="", stability=Dynamic())
    assert block.text == "", "empty dynamic blocks are legal; the planner owns prefix rules"
