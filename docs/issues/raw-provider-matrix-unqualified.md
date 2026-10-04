# raw provider merge gate

status: OPEN. native pinned-artifact qualification remains separate.

the source delivery changes main's raw registry/engines to the already adopted
`6a7093f799c88c205c8d797bbb8b14c2a9980db1` implementation. no retained
unfiltered live receipt matches its registry `2026-09-25.2`.
[readme](../../README.md#live-matrix-paid-evidence-recorded-never-ci) and
[pivot spec section 11](../pivot-spec.md#11-testing) require that proof before
merging registry/adapter changes. native codex receipts do not qualify raw api calls.

the retained `provider-runtime-2026-08-11T064753Z-2bf13e5df201.json` uses
registry `2026-08-10.2`. eight current rows are absent: openai gpt-6-astra,
gpt-6-sol and gpt-6-luna; anthropic claude-fable-5-1 and claude-opus-5-5;
gemini gemini-3.8-flash; deepseek deepseek-flash; xai grok-4.7. its
claude-sonnet-5 proof also predates the current registry/engine contract.

resolution: run the existing matrix unfiltered with all five provider keys
supplied privately, review every current row's chat, stream, tools, structured
output and continuation evidence, and retain the matching receipt. retain
unsupported-capability skips honestly; missing credentials or a filtered run
cannot satisfy the gate. no provider call was made by this investigation.

```sh
LLM_RUNTIME_LIVE=1 uv run --frozen pytest -m live_provider tests/live/test_provider_matrix.py
```
