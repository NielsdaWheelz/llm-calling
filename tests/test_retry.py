"""Behavior tests for the single retry owner (`provider_runtime.retry`).

The attempt iterator is driven with an injected RNG, clock, and sleep — no
real time passes and no internals are patched. Assertions are on the observable
contract: which attempt numbers are yielded and which delays are slept.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from provider_runtime.retry import DEFAULT_RETRY, attempts
from provider_runtime.types import (
    Absent,
    Present,
    ProviderRateLimit,
    ProviderTimeout,
    RetryPolicy,
    TransientCause,
)

# ---------------------------------------------------------------------------
# Deterministic harness

TIMEOUT = ProviderTimeout()


def policy(
    *,
    max_attempts: int = 3,
    initial_delay_s: float = 1.0,
    max_delay_s: float = 8.0,
    jitter_s: float = 0.0,
    deadline_s: float | None = None,
) -> RetryPolicy:
    # Tests are the one sanctioned construction site outside retry.py.
    return RetryPolicy(
        max_attempts=max_attempts,
        initial_delay_s=initial_delay_s,
        max_delay_s=max_delay_s,
        jitter_s=jitter_s,
        deadline_s=Absent() if deadline_s is None else Present(deadline_s),
    )


class ManualClock:
    """Injected monotonic clock; only sleeps and tests advance it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SleepRecorder:
    """Injected sleep: records each delay and advances the clock, never waits."""

    def __init__(self, clock: ManualClock) -> None:
        self._clock = clock
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        self._clock.now += delay


async def drive(
    retry_policy: RetryPolicy,
    *,
    fail_with: Callable[[int], TransientCause],
    clock: ManualClock,
    sleep: SleepRecorder,
    rng: random.Random | None = None,
) -> list[int]:
    """Mark every yielded attempt failed; return the attempt numbers seen."""
    numbers: list[int] = []
    async for attempt in attempts(
        retry_policy, rng=rng or random.Random(0), clock=clock, sleep=sleep
    ):
        numbers.append(attempt.number)
        attempt.mark_failed(fail_with(attempt.number))
    return numbers


# ---------------------------------------------------------------------------
# Attempt budget


async def test_yields_one_based_attempts_up_to_max_then_stops() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    numbers = await drive(
        policy(max_attempts=3), fail_with=lambda _: TIMEOUT, clock=clock, sleep=sleep
    )
    assert numbers == [1, 2, 3], f"expected attempts 1..3, got {numbers}"
    assert len(sleep.delays) == 2, (
        f"expected a sleep between attempts only (2 sleeps), got {sleep.delays}"
    )


async def test_single_attempt_policy_never_sleeps() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    numbers = await drive(
        policy(max_attempts=1), fail_with=lambda _: TIMEOUT, clock=clock, sleep=sleep
    )
    assert numbers == [1], f"expected exactly one attempt, got {numbers}"
    assert sleep.delays == [], f"single-attempt policy must not sleep, got {sleep.delays}"


async def test_finishing_inside_an_attempt_sleeps_nothing() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    seen: list[int] = []
    async for attempt in attempts(
        policy(max_attempts=3), rng=random.Random(0), clock=clock, sleep=sleep
    ):
        seen.append(attempt.number)
        break  # the runtime returns its outcome; iteration simply stops
    assert seen == [1], f"expected only the first attempt, got {seen}"
    assert sleep.delays == [], f"a finished call must not back off, got {sleep.delays}"


async def test_iterating_past_an_unmarked_attempt_is_a_defect() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    iterator = attempts(policy(max_attempts=3), rng=random.Random(0), clock=clock, sleep=sleep)
    first = await anext(iterator)
    assert first.number == 1
    with pytest.raises(AssertionError):
        await anext(iterator)  # the defect terminates the generator itself


# ---------------------------------------------------------------------------
# Backoff shape


async def test_backoff_doubles_from_initial_and_caps_at_max_delay() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    await drive(
        policy(max_attempts=5, initial_delay_s=1.0, max_delay_s=6.0),
        fail_with=lambda _: TIMEOUT,
        clock=clock,
        sleep=sleep,
    )
    assert sleep.delays == [1.0, 2.0, 4.0, 6.0], (
        f"expected jitter-free exponential capped at 6.0, got {sleep.delays}"
    )


async def test_jitter_adds_a_bounded_random_offset() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    await drive(
        policy(max_attempts=4, initial_delay_s=1.0, max_delay_s=100.0, jitter_s=0.5),
        fail_with=lambda _: TIMEOUT,
        clock=clock,
        sleep=sleep,
        rng=random.Random(7),
    )
    bases = [1.0, 2.0, 4.0]
    assert len(sleep.delays) == 3
    for base, delay in zip(bases, sleep.delays, strict=True):
        assert base <= delay <= base + 0.5, (
            f"delay {delay} outside jitter window [{base}, {base + 0.5}]"
        )
    offsets = [delay - base for base, delay in zip(bases, sleep.delays, strict=True)]
    assert sum(offsets) > 0, f"jitter was configured but never applied: {sleep.delays}"


async def test_production_default_rng_works_uninjected() -> None:
    # jitter_s=0 keeps the schedule deterministic even with the default RNG.
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    numbers: list[int] = []
    async for attempt in attempts(policy(max_attempts=2), clock=clock, sleep=sleep):
        numbers.append(attempt.number)
        attempt.mark_failed(TIMEOUT)
    assert numbers == [1, 2]
    assert sleep.delays == [1.0]


# ---------------------------------------------------------------------------
# retry_after (ProviderRateLimit)


async def test_retry_after_is_honored_verbatim_without_jitter() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    await drive(
        policy(max_attempts=2, initial_delay_s=1.0, max_delay_s=8.0, jitter_s=5.0),
        fail_with=lambda _: ProviderRateLimit(retry_after=Present(7.5)),
        clock=clock,
        sleep=sleep,
    )
    assert sleep.delays == [7.5], (
        f"retry_after must be slept verbatim (no jitter), got {sleep.delays}"
    )


async def test_retry_after_may_exceed_the_backoff_max_delay() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    await drive(
        policy(max_attempts=2, max_delay_s=8.0),
        fail_with=lambda _: ProviderRateLimit(retry_after=Present(30.0)),
        clock=clock,
        sleep=sleep,
    )
    assert sleep.delays == [30.0], (
        f"the provider's explicit retry_after overrides max_delay_s, got {sleep.delays}"
    )


async def test_retry_after_at_the_sixty_second_cap_is_honored() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    numbers = await drive(
        policy(max_attempts=2),
        fail_with=lambda _: ProviderRateLimit(retry_after=Present(60.0)),
        clock=clock,
        sleep=sleep,
    )
    assert numbers == [1, 2], f"a retry_after exactly at the cap must retry, got {numbers}"
    assert sleep.delays == [60.0], (
        f"retry_after at the cap must be slept verbatim, got {sleep.delays}"
    )


async def test_retry_after_above_sixty_seconds_exhausts_without_retrying() -> None:
    # Never retry before the provider's stated window: a retry_after above
    # the 60s cap is not honored by an early retry, so the call exhausts
    # instead of sleeping a clamped, premature delay.
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    numbers = await drive(
        policy(max_attempts=3),
        fail_with=lambda _: ProviderRateLimit(retry_after=Present(90.0)),
        clock=clock,
        sleep=sleep,
    )
    assert numbers == [1], (
        f"a retry_after above 60s must exhaust on the first failed attempt, got {numbers}"
    )
    assert sleep.delays == [], f"an above-cap retry_after must never be slept, got {sleep.delays}"


async def test_negative_retry_after_never_sleeps_negative() -> None:
    # Speculative (a mis-parsed Retry-After header): retry_after has no
    # non-negativity invariant at construction, so the loop must not turn a
    # negative value into an instant zero-backoff retry via a bare sleep().
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    await drive(
        policy(max_attempts=2),
        fail_with=lambda _: ProviderRateLimit(retry_after=Present(-5.0)),
        clock=clock,
        sleep=sleep,
    )
    assert sleep.delays == [0.0], f"a negative retry_after must clamp to 0, got {sleep.delays}"


async def test_rate_limit_without_retry_after_uses_backoff() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    await drive(
        policy(max_attempts=3, initial_delay_s=1.0),
        fail_with=lambda _: ProviderRateLimit(retry_after=Absent()),
        clock=clock,
        sleep=sleep,
    )
    assert sleep.delays == [1.0, 2.0], (
        f"absent retry_after falls back to exponential backoff, got {sleep.delays}"
    )


# ---------------------------------------------------------------------------
# Wall-clock deadline


async def test_deadline_stops_before_a_sleep_that_would_exceed_it() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    numbers = await drive(
        policy(max_attempts=5, initial_delay_s=4.0, max_delay_s=100.0, deadline_s=10.0),
        fail_with=lambda _: TIMEOUT,
        clock=clock,
        sleep=sleep,
    )
    # attempt 1: elapsed 0 + delay 4 <= 10 → sleep; attempt 2: 4 + 8 > 10 → stop.
    assert numbers == [1, 2], f"expected the deadline to stop after attempt 2, got {numbers}"
    assert sleep.delays == [4.0], f"expected one pre-deadline sleep, got {sleep.delays}"


async def test_deadline_counts_time_spent_inside_attempts() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)

    def slow_attempt(_: int) -> TransientCause:
        clock.now += 9.5  # the attempt itself consumed most of the deadline
        return TIMEOUT

    numbers = await drive(
        policy(max_attempts=3, initial_delay_s=1.0, deadline_s=10.0),
        fail_with=slow_attempt,
        clock=clock,
        sleep=sleep,
    )
    assert numbers == [1], f"9.5s elapsed + 1s delay exceeds 10s deadline, got {numbers}"
    assert sleep.delays == [], f"expected no sleep past the deadline, got {sleep.delays}"


async def test_first_attempt_always_runs_even_under_a_tiny_deadline() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    numbers = await drive(
        policy(max_attempts=3, deadline_s=0.001),
        fail_with=lambda _: TIMEOUT,
        clock=clock,
        sleep=sleep,
    )
    assert numbers == [1], f"the deadline gates sleeps, never the first attempt: {numbers}"
    assert sleep.delays == []


async def test_deadline_applies_to_retry_after_delays_too() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    numbers = await drive(
        policy(max_attempts=3, deadline_s=30.0),
        fail_with=lambda _: ProviderRateLimit(retry_after=Present(60.0)),
        clock=clock,
        sleep=sleep,
    )
    assert numbers == [1], (
        f"a 60s retry_after under a 30s deadline must exhaust immediately, got {numbers}"
    )
    assert sleep.delays == []


async def test_absent_deadline_never_stops_early() -> None:
    clock = ManualClock()
    sleep = SleepRecorder(clock)
    numbers = await drive(
        policy(max_attempts=4, initial_delay_s=100.0, max_delay_s=1000.0),
        fail_with=lambda _: TIMEOUT,
        clock=clock,
        sleep=sleep,
    )
    assert numbers == [1, 2, 3, 4], f"no deadline means the full budget runs: {numbers}"
    assert sleep.delays == [100.0, 200.0, 400.0]


# ---------------------------------------------------------------------------
# DEFAULT_RETRY


def test_default_retry_is_three_jittered_attempts_under_one_deadline() -> None:
    assert DEFAULT_RETRY.max_attempts == 3
    assert DEFAULT_RETRY.initial_delay_s > 0
    assert DEFAULT_RETRY.max_delay_s >= DEFAULT_RETRY.initial_delay_s
    assert DEFAULT_RETRY.jitter_s > 0, "the default backoff must be jittered"
    assert isinstance(DEFAULT_RETRY.deadline_s, Present), (
        "the default policy must carry one wall-clock deadline"
    )
