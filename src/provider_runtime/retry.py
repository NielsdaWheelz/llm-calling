"""Single retry owner — the default policy and the attempt iterator.

The runtime drives every retryable call through ``attempts(policy)``: one
`Attempt` handle per try, retryable failure marked on the handle
(`mark_failed`), and the iterator sleeps the backoff between tries — a
jittered exponential curve, `ProviderRateLimit.retry_after` honored up to a
60s cap, and one wall-clock deadline for the whole call. Iteration ending
without the runtime breaking out means the budget is exhausted; the runtime
folds that into `Failed(TransientExhausted)`.

Deliberately a thin explicit generator, not a stamina wrapper: stamina's
retry-after path (a backoff hook smuggling a float through tenacity call
state) recomputes the policy arithmetic inside the hook, its jitter comes from
module-global `random`, and its deadline check does not count the pending
sleep — none of it injectable for deterministic tests. The spec's requirement
is single-owner semantics, which this module is.

Layering: imports from `types` only. This module is the sole construction site
in src for the retry policy value (negative-gated; tests excepted).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Final, assert_never

from provider_runtime.types import (
    Absent,
    Present,
    ProviderRateLimit,
    RetryPolicy,
    TransientCause,
)

# ---------------------------------------------------------------------------
# Default policy

# A provider-sent retry_after is honored verbatim beyond max_delay_s (the
# provider's explicit instruction outranks our backoff curve) but never past
# this cap; the deadline still gates the resulting sleep.
_RETRY_AFTER_CAP_S: Final[float] = 60.0

# Backoff values ported from the old runtime's external-LLM policy. The 120s
# deadline is new (spec §8: one wall-clock deadline): checked pre-sleep only,
# so it never gates the final attempt — three full 45s-timeout attempts fit —
# while bounding rate-limit stalls to roughly one capped retry_after window.
DEFAULT_RETRY: Final[RetryPolicy] = RetryPolicy(
    max_attempts=3,
    initial_delay_s=1.0,
    max_delay_s=8.0,
    jitter_s=0.25,
    deadline_s=Present(120.0),
)


# ---------------------------------------------------------------------------
# Attempt loop


@dataclass(slots=True)
class Attempt:
    """One try's loop handle — deliberately mutable.

    ``async for`` cannot send into the iterator, so the runtime marks a
    retryable failure on the handle before advancing; the failure cause drives
    the next delay (retry_after).
    """

    number: int
    _failure: TransientCause | None = None

    def mark_failed(self, cause: TransientCause) -> None:
        self._failure = cause


async def attempts(
    policy: RetryPolicy,
    *,
    rng: random.Random | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[Attempt]:
    """Yield 1-based attempt handles, sleeping the policy's backoff between them.

    The runtime's contract per yielded handle: return/break on a terminal
    outcome, or `mark_failed(cause)` and iterate. Exhaustion (attempt budget or
    deadline) ends the iteration; the deadline is checked before each sleep
    (elapsed + pending delay), so the first attempt always runs and no sleep
    ever starts past the deadline.

    `rng`/`clock`/`sleep` are deterministic-test seams with production defaults.
    """
    jitter_rng = random.Random() if rng is None else rng
    started = clock()
    for number in range(1, policy.max_attempts + 1):
        attempt = Attempt(number=number)
        yield attempt
        cause = attempt._failure
        if cause is None:
            raise AssertionError(
                f"attempts() advanced past attempt {number} without mark_failed(); "
                "a finished call must stop iterating"
            )
        delay_s = _delay_s(attempt=number, cause=cause, policy=policy, rng=jitter_rng)
        if number >= policy.max_attempts or _deadline_exhausted(clock() - started, policy, delay_s):
            return
        await sleep(delay_s)


def _delay_s(
    *, attempt: int, cause: TransientCause, policy: RetryPolicy, rng: random.Random
) -> float:
    """Exponential ``initial * 2^(attempt-1)`` plus jitter, capped at max_delay.

    A Present `ProviderRateLimit.retry_after` is honored verbatim (no jitter,
    no max_delay cap) up to the 60s cap.
    """
    if isinstance(cause, ProviderRateLimit) and isinstance(cause.retry_after, Present):
        return min(cause.retry_after.value, _RETRY_AFTER_CAP_S)
    delay = policy.initial_delay_s * (2 ** (attempt - 1))
    if policy.jitter_s > 0:
        delay += rng.uniform(0, policy.jitter_s)
    return min(delay, policy.max_delay_s)


def _deadline_exhausted(elapsed_s: float, policy: RetryPolicy, delay_s: float) -> bool:
    """Absent deadline means no wall-clock deadline at all."""
    match policy.deadline_s:
        case Present(value=deadline):
            return elapsed_s + delay_s > deadline
        case Absent():
            return False
        case _:
            assert_never(policy.deadline_s)
