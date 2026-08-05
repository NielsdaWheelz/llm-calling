# Polling

## Scope

This document covers polling rules.

## Rules

- Avoid polling by default.
- Prefer push- or event-driven designs such as notifications, subscriptions, and streaming.
- If polling is unavoidable, include `justify-polling`.
- Polling cadence and termination follow the schedule-shape rules in [timing.md](timing.md).
- `justify-polling` must explain why polling is necessary and the chosen interval or backoff strategy.
