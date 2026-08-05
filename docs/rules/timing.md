# Timing

## Scope

This document covers timing parameters and schedule-shape rules.

## Schedules

- Retry and polling schedules should be self-bounding.
- Keep a schedule's cadence and termination behavior in the same schedule definition.
- Do not apply time or attempt limits externally with a separate timeout when the schedule itself should own them.
- `retryOrDieTag` and `retryOrExhaustTag` with `activateBoundedSchedule` are standard ways to keep bounded retry behavior explicit.

## Constants

- Timing parameters such as retry intervals, backoff caps, timeouts, and polling periods should be named constants.
- Represent timing parameters as `Duration` values or fully constructed `Schedule` objects.
- Avoid raw numbers and anonymous inline `Duration` literals in business logic.
- If a duration is only meaningful as part of one named schedule constant, it may be embedded inline inside that schedule constant.
