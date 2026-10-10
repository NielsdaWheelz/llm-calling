# raw provider merge gate

## native supervision, 2026-10-03

status: WAIVED FOR THIS NATIVE PR ONLY, 2026-10-03. the current raw api
matrix remains NOT_RUN and unqualified. native sdk and consumer qualification
remain required and separate.

the user replied "skip these i don't care" to the inherited all-five api
matrix/key gate. this permits [native pr 37](https://github.com/NielsdaWheelz/llm-calling/pull/37)
without that live run; it does not mark any raw row green or weaken the gate
for future changes. no
raw api provider calls were launched for this disposition.

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

to qualify the still-unverified raw contract, run the existing matrix
unfiltered with all five provider keys
supplied privately, review every current row's chat, stream, tools, structured
output and continuation evidence, and retain the matching receipt. retain
unsupported-capability skips honestly; missing credentials or a filtered run
cannot satisfy the gate. no provider call was made by this investigation.

```sh
LLM_RUNTIME_LIVE=1 uv run --frozen pytest -m live_provider tests/live/test_provider_matrix.py
```

## universal memory, 2026-10-10 utc

the owner explicitly extended the full raw-provider matrix waiver to this memory
delivery: [provider pr 38](https://github.com/NielsdaWheelz/llm-calling/pull/38)
and the immutable memory pin adoption in
[nexus pr 537](https://github.com/NielsdaWheelz/nexus-web/pull/537).
this delivery leaves the raw registry and engines unchanged. focused transport,
archive, stock codex native and consumer integration checks passed; the integrated
provider offline suites and static/build checks passed separately. successful
live claude-work inference remains
[owner-waived after expired oauth](../native-archive.md).

the raw matrix remains not_run and unqualified. these checks do not qualify raw
provider calls. this waiver applies only to those two memory-delivery prs; the
general readme/pivot gate and the earlier native pr 37 waiver retain their scope.
no raw api calls were launched and no credentials were provisioned for this
extension. native/sdk and consumer qualification remain separate requirements.
