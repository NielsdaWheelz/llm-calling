# Keep-Alive

## Scope

This document covers keep-alive policies.

## Policies

- Define keep-alive policies in one central policy catalog as categorical policies.
- Creating a keep-alive policy anywhere else requires `justify-keep-alive-policy`.
- A `KeepAlivePolicy` pairs a TTL with a derived keep-alive interval of one third of that TTL.
- Server-side resources should use one standard server keep-alive policy unless
  they have a documented reason for a narrower policy.
- Keep-alive TTLs reflect shared infrastructure characteristics.
- Do not use keep-alive policies for data-retention or expiry durations.
- Data-retention and expiry durations are separate domain rules.
