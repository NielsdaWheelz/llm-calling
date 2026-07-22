"""Live provider certification matrix (spec §14 paid tier; §8 operator evidence).

Fail-closed paid acceptance tests over the real providers. Excluded from the
default suite (``addopts`` deselects ``live_provider``); run with:

    LLM_RUNTIME_LIVE=1 uv run pytest -m live_provider tests/live/test_provider_matrix.py

Environment contract (preserved from the pre-cutover matrix):

- ``LLM_RUNTIME_LIVE=1`` is required — anything else fails, never skips;
- ``LLM_RUNTIME_LIVE_PROVIDERS`` optionally narrows to a comma-separated subset
  of ``openai,anthropic,gemini,moonshot,openrouter`` (release certification runs
  unfiltered);
- required keys per provider: ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
  ``GEMINI_API_KEY``, ``MOONSHOT_API_KEY``, ``OPENROUTER_API_KEY``.

Per direct chat target the matrix proves: every declared reasoning level, an
above-minimum-prefix cache warm/read probe with a reported cache read, strict
JSON with a required-nullable field, a streamed tool call + same-target
continuation replay, invalid-key classification, request-id/usage presence per
contract facts, and the §9 input-bound obligation
(``planned_input_token_upper_bound`` >= provider-billed input) on every call.

The OpenRouter operator route is OperatorUncertified in CATALOG, so the planner
refuses it by design. ``test_openrouter_certification`` is THE CERTIFIER: it
plans through a hand-built OperatorCertified copy of the catalog row (pinned to
``moonshotai/int4``), proves routed Kimi ``low|high|max`` (spec §4 recheck, both
arms: direct levels are covered by the matrix above), the exact endpoint request
pin plus its observed provider identity, and a
NON-ZERO billed cache read, then writes the §8 evidence artifact
(endpoint-metadata snapshot, probe generation ids, observed cache usage) to
``tests/live/evidence/openrouter-<date>.json`` and prints the
``evidence_revision`` to pin in the catalog row.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import os
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
import pytest

from provider_runtime import (
    CATALOG,
    CATALOG_REVISION,
    Absent,
    AssistantMessage,
    CallOutcome,
    CanonicalTool,
    ChatModelContract,
    ContinuationArtifact,
    ContinuationDelta,
    CredentialRejected,
    Dynamic,
    EmbeddingCall,
    FinalizedProviderCall,
    GenerateIntent,
    GlobalScope,
    Incomplete,
    OpenRouterCertifiedPrefix,
    OperatorCertified,
    Presence,
    Present,
    PromptBlock,
    ProviderCredential,
    ProviderName,
    ProviderRuntime,
    ReasoningLevel,
    Stable,
    StrictJsonOutput,
    StructuredContent,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextContent,
    TextDelta,
    TextOutput,
    TokenUsage,
    ToolCall,
    ToolCallDone,
    ToolResultMessage,
    TranscriptionCall,
    UserMessage,
    parse_canonical_schema,
    plan_generate,
)
from provider_runtime.catalog import (
    AnthropicPrefixContract,
    Catalog,
    GeminiAutomaticPrefixContract,
    MoonshotKeyedPrefixContract,
    OpenAIExplicitPrefixContract,
    OpenRouterPrefixContract,
)

pytestmark = pytest.mark.live_provider

_PROVIDER_ORDER: tuple[ProviderName, ...] = (
    "openai",
    "anthropic",
    "gemini",
    "moonshot",
    "openrouter",
)
_PROVIDER_ENV: dict[ProviderName, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_OPENROUTER_ROW: ChatModelContract = next(
    row for row in CATALOG.chat if row.protocol == "openrouter_chat"
)
_DIRECT_ROWS: tuple[ChatModelContract, ...] = tuple(
    row for row in CATALOG.chat if row.protocol != "openrouter_chat"
)

# The certifier plans through an OperatorCertified copy of the catalog row; the
# artifact this run writes mints the real evidence_revision for the catalog. The
# certified pin facts must match the row's cache contract or catalog construction
# rejects the copy.
assert isinstance(_OPENROUTER_ROW.cache, OpenRouterPrefixContract)
_CERTIFYING_OPENROUTER_ROW: ChatModelContract = dataclasses.replace(
    _OPENROUTER_ROW,
    certification=OperatorCertified(
        certified_pinned_upstream=_OPENROUTER_ROW.cache.pinned_upstream,
        certified_canonical_revision=_OPENROUTER_ROW.cache.canonical_revision,
        evidence_revision="certification-in-progress",
    ),
)

_EVIDENCE_DIR = Path(__file__).parent / "evidence"
_CACHE_REFERENCE_CORPUS = (
    Path(__file__).parent / "fixtures" / "family_picnic_reference.txt"
).read_text(encoding="utf-8")
_OPENROUTER_ENDPOINTS_URL = (
    f"https://openrouter.ai/api/v1/models/{_OPENROUTER_ROW.target.model}/endpoints"
)


def _row_id(row: ChatModelContract) -> str:
    return f"{row.target.provider}:{row.target.model}"


# ---------------------------------------------------------------------------
# Environment gate (fail closed, never skip-success)


@dataclasses.dataclass(frozen=True)
class LiveEnv:
    selected_providers: frozenset[ProviderName]

    def credential_for(self, provider: ProviderName) -> ProviderCredential:
        if provider not in self.selected_providers:
            pytest.skip(f"{provider} not selected by LLM_RUNTIME_LIVE_PROVIDERS")
        env_name = _PROVIDER_ENV[provider]
        key = os.environ.get(env_name)
        if not key:
            pytest.fail(f"{provider} live matrix requires env var {env_name}")
        return ProviderCredential(provider=provider, key=key)


@pytest.fixture(scope="session")
def live_env() -> LiveEnv:
    if os.environ.get("LLM_RUNTIME_LIVE") != "1":
        pytest.fail("Set LLM_RUNTIME_LIVE=1 to run the live provider matrix")
    return LiveEnv(selected_providers=_selected_providers())


def _selected_providers() -> frozenset[ProviderName]:
    raw = os.environ.get("LLM_RUNTIME_LIVE_PROVIDERS")
    if not raw:
        return frozenset(_PROVIDER_ORDER)
    requested = {name.strip().lower() for name in raw.split(",") if name.strip()}
    unknown = requested - set(_PROVIDER_ORDER)
    if unknown:
        pytest.fail(
            "Unknown LLM_RUNTIME_LIVE_PROVIDERS value(s): "
            f"{', '.join(sorted(unknown))}. Expected any of: {', '.join(_PROVIDER_ORDER)}"
        )
    return frozenset(cast(ProviderName, name) for name in requested)


# ---------------------------------------------------------------------------
# Intent/plan/call helpers


def _catalog_for(row: ChatModelContract) -> Catalog:
    if row.protocol == "openrouter_chat":
        return Catalog(chat=(row,), embeddings=(), transcriptions=())
    return CATALOG


def _plan(row: ChatModelContract, intent: GenerateIntent) -> FinalizedProviderCall:
    plan = plan_generate(intent, _catalog_for(row))
    assert isinstance(plan, FinalizedProviderCall), f"{_row_id(row)}: planner rejected {plan}"
    return plan


_TEXT_OUTPUT = TextOutput()


def _intent(
    row: ChatModelContract,
    *,
    prompt: str,
    stable_prefix: str,
    max_output_tokens: int,
    reasoning: ReasoningLevel,
    tools: tuple[CanonicalTool, ...] = (),
    output: TextOutput | StrictJsonOutput = _TEXT_OUTPUT,
    history: tuple[AssistantMessage | ToolResultMessage, ...] = (),
) -> GenerateIntent:
    return GenerateIntent(
        target=row.target,
        messages=(
            SystemMessage(
                blocks=(PromptBlock(text=stable_prefix, stability=Stable(GlobalScope())),)
            ),
            UserMessage(blocks=(PromptBlock(text=prompt, stability=Dynamic()),)),
            *history,
        ),
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        tools=tools,
        tool_choice="auto" if tools else "none",
        output=output,
    )


async def _generate(
    row: ChatModelContract, intent: GenerateIntent, credential: ProviderCredential
) -> tuple[FinalizedProviderCall, CallOutcome]:
    call = _plan(row, intent)
    async with httpx.AsyncClient() as http:
        outcome = await ProviderRuntime(http).generate(call, credential=credential)
    return call, outcome


_SHORT_STABLE_PREFIX = "You are a concise reference assistant. Answer briefly and factually."


def _cheapest_level(row: ChatModelContract) -> ReasoningLevel:
    return row.reasoning.levels[0]


def _reasoning_budget(row: ChatModelContract, level: ReasoningLevel) -> int:
    # Reasoning/thinking tokens bill inside the output budget on every current
    # provider, so budgets scale with effort; Incomplete(max_output_tokens) is
    # still an accepted terminal for level probes.
    if row.target.provider == "gemini":
        return {"minimal": 1024, "low": 2048, "medium": 8448, "high": 16512}[level]
    return {
        "none": 256,
        "minimal": 512,
        "low": 1024,
        "medium": 2048,
        "high": 4096,
        "xhigh": 8192,
        "max": 8192,
    }[level]


# ---------------------------------------------------------------------------
# Assertions shared across the matrix


def _accepted(row: ChatModelContract, outcome: CallOutcome) -> Succeeded | Incomplete:
    assert isinstance(outcome, Succeeded | Incomplete), f"{_row_id(row)}: {outcome}"
    if isinstance(outcome, Incomplete):
        assert outcome.status == "provider_incomplete", f"{_row_id(row)}: {outcome}"
        assert outcome.reason == "max_output_tokens", f"{_row_id(row)}: {outcome}"
    return outcome


def _usage_of(row: ChatModelContract, outcome: CallOutcome) -> TokenUsage:
    usage = outcome.meta.usage
    assert isinstance(usage, Present), f"{_row_id(row)}: no usage reported"
    return usage.value


def _count(maybe: Present[int] | Absent) -> int:
    return maybe.value if isinstance(maybe, Present) else 0


def _assert_input_bound(
    row: ChatModelContract, call: FinalizedProviderCall, usage: TokenUsage
) -> None:
    # §9 certification obligation: the bytes-as-tokens planner bound dominates
    # normalized billed input. Every codec's TokenUsage.input_tokens is already
    # cache-inclusive.
    billed_input = usage.input_tokens
    assert call.planned_input_token_upper_bound >= billed_input, (
        f"{_row_id(row)}: planned bound {call.planned_input_token_upper_bound} "
        f"< billed input {billed_input}"
    )


def _assert_correlation_facts(row: ChatModelContract, outcome: CallOutcome) -> None:
    meta = outcome.meta
    assert meta.provider == row.target.provider
    if row.provider_request_id_available:
        assert isinstance(meta.provider_request_id, Present), (
            f"{_row_id(row)}: contract declares a provider request id; got Absent"
        )
    else:
        # Gemini: no request correlation — the catalog records that fact.
        assert isinstance(meta.provider_request_id, Absent), (
            f"{_row_id(row)}: contract declares NO provider request id; got Present"
        )


def _assert_call_facts(
    row: ChatModelContract, call: FinalizedProviderCall, outcome: CallOutcome
) -> TokenUsage:
    usage = _usage_of(row, outcome)
    _assert_input_bound(row, call, usage)
    _assert_correlation_facts(row, outcome)
    return usage


# ---------------------------------------------------------------------------
# 1. Per-target minimal generate per declared reasoning level


@pytest.mark.parametrize(
    ("row", "level"),
    [
        pytest.param(row, level, id=f"{_row_id(row)}:{level}")
        for row in _DIRECT_ROWS
        for level in row.reasoning.levels
    ],
)
async def test_live_declared_reasoning_level(
    live_env: LiveEnv, row: ChatModelContract, level: ReasoningLevel
) -> None:
    credential = live_env.credential_for(row.target.provider)
    call, outcome = await _generate(
        row,
        _intent(
            row,
            prompt="Answer in one short sentence: what is two plus two?",
            stable_prefix=_SHORT_STABLE_PREFIX,
            max_output_tokens=_reasoning_budget(row, level),
            reasoning=level,
        ),
        credential,
    )
    terminal = _accepted(row, outcome)
    _assert_call_facts(row, call, terminal)
    assert call.native_reasoning == row.reasoning.native_mapping[level]
    if isinstance(terminal, Succeeded):
        content = terminal.response.content
        assert isinstance(content, TextContent) and content.text.strip()


# ---------------------------------------------------------------------------
# 2. Above-minimum-prefix cache warm/read pair


def _minimum_prefix_tokens(row: ChatModelContract) -> int:
    match row.cache:
        case OpenAIExplicitPrefixContract(minimum_prefix_tokens=minimum):
            return minimum
        case AnthropicPrefixContract(minimum_prefix_tokens=minimum):
            return minimum
        case GeminiAutomaticPrefixContract(minimum_prefix_tokens=Present(value=minimum)):
            return minimum
        case MoonshotKeyedPrefixContract() | OpenRouterPrefixContract():
            # Moonshot direct/routed: no documented minimum — probe well above
            # every known threshold.
            return 4096
    raise AssertionError(f"unhandled cache contract for {_row_id(row)}")


def _cache_probe_prefix(row: ChatModelContract) -> str:
    # Fresh natural-language marker per run: call one writes the cache, call two
    # must read it.
    # The appendix is intentionally ordinary product-shaped prose: semantic
    # refusal remains a failure, but the probe itself does not resemble a
    # repetitive security-evaluation payload.
    marker_value = uuid.uuid4().int
    marker = "-".join(
        (
            ("willow", "maple", "cedar", "birch")[marker_value % 4],
            ("daisy", "violet", "clover", "buttercup")[(marker_value // 4) % 4],
            ("meadow", "pond", "hill", "village")[(marker_value // 16) % 4],
            str((marker_value // 64) % 10_000),
        )
    )
    # Six UTF-8 characters per declared token minimum keeps the actual
    # provider-reported input above every current threshold without relying on
    # a provider tokenizer in the shared probe.
    minimum_chars = _minimum_prefix_tokens(row) * 6
    paragraphs = _CACHE_REFERENCE_CORPUS.split("\n\n")
    body_parts: list[str] = []
    body_length = 0
    while body_length < minimum_chars:
        paragraph = paragraphs[len(body_parts) % len(paragraphs)]
        body_parts.append(paragraph)
        body_length += len(paragraph) + 2
    body = "\n\n".join(body_parts)
    return (
        "You are reading a gentle family picnic story. Answer briefly and factually.\n"
        f"Picnic detail {marker}. Use this story as background.\n"
        f"{body}"
    )


async def _cache_warm_read_pair(
    row: ChatModelContract, credential: ProviderCredential
) -> tuple[
    tuple[FinalizedProviderCall, CallOutcome],
    tuple[FinalizedProviderCall, CallOutcome],
    TokenUsage,
]:
    prefix = _cache_probe_prefix(row)
    level = _cheapest_level(row)

    def probe_intent(prompt: str) -> GenerateIntent:
        return _intent(
            row,
            prompt=prompt,
            stable_prefix=prefix,
            max_output_tokens=_reasoning_budget(row, level),
            reasoning=level,
        )

    warm = await _generate(
        row,
        probe_intent("What foods did the family bring? Answer in one short sentence."),
        credential,
    )
    _accepted(row, warm[1])
    warm_usage = _assert_call_facts(row, warm[0], warm[1])
    assert warm[0].request_fingerprint != ""
    read_prompts = (
        "What did Mara draw? Answer in one short sentence.",
        "What shape did Theo see in the clouds? Answer in one short sentence.",
        "Where did the family shelter from the rain? Answer in one short sentence.",
    )
    # GenerateContent implicit caching is opportunistic, not a guaranteed
    # write/read primitive. Sample up to three successful distinct reads for
    # Gemini; every other cache contract remains a strict warm/read pair.
    maximum_reads = 3 if isinstance(row.cache, GeminiAutomaticPrefixContract) else 1
    read_usages: list[TokenUsage] = []
    for index, prompt in enumerate(read_prompts[:maximum_reads]):
        await asyncio.sleep(3 if index == 0 else 1)
        read = await _generate(row, probe_intent(prompt), credential)
        _accepted(row, read[1])
        read_usage = _assert_call_facts(row, read[0], read[1])
        read_usages.append(read_usage)
        if _count(read_usage.cache_read_input_tokens) > 0:
            return warm, read, read_usage
    pytest.fail(
        f"{_row_id(row)}: no cache read reported after {maximum_reads} "
        f"above-minimum-prefix reads (warm usage: {warm_usage}, read usages: {read_usages})"
    )


@pytest.mark.parametrize("row", _DIRECT_ROWS, ids=_row_id)
async def test_live_cache_warm_read_pair(live_env: LiveEnv, row: ChatModelContract) -> None:
    credential = live_env.credential_for(row.target.provider)
    await _cache_warm_read_pair(row, credential)


# ---------------------------------------------------------------------------
# 3. Strict JSON (canonical subset incl. a required-nullable field)


_STRICT_SCHEMA = parse_canonical_schema(
    {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "summary": {"type": "string", "description": "Two-word answer summary."},
            "note": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Optional remark; null when there is nothing to add.",
            },
        },
        "required": ["ok", "summary", "note"],
        "additionalProperties": False,
    }
)


@pytest.mark.parametrize("row", _DIRECT_ROWS, ids=_row_id)
async def test_live_strict_json_output(live_env: LiveEnv, row: ChatModelContract) -> None:
    credential = live_env.credential_for(row.target.provider)
    level = _cheapest_level(row)
    call, outcome = await _generate(
        row,
        _intent(
            row,
            prompt="Return ok=true, a two-word summary, and note=null.",
            stable_prefix=_SHORT_STABLE_PREFIX,
            max_output_tokens=max(_reasoning_budget(row, level), 2048),
            reasoning=level,
            output=StrictJsonOutput(name="live_matrix_result", schema=_STRICT_SCHEMA),
        ),
        credential,
    )
    assert isinstance(outcome, Succeeded), f"{_row_id(row)}: {outcome}"
    _assert_call_facts(row, call, outcome)
    content = outcome.response.content
    assert isinstance(content, StructuredContent), f"{_row_id(row)}: {content}"
    assert content.payload.get("ok") is True
    assert isinstance(content.payload.get("summary"), str)
    assert "note" in content.payload  # required-nullable: present, possibly null


# ---------------------------------------------------------------------------
# 4. Streamed tool call + same-target continuation replay


_SEARCH_TOOL = CanonicalTool(
    name="search_library",
    description="Look up a compact snippet from the library index for a query.",
    parameters=parse_canonical_schema(
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query text."}},
            "required": ["query"],
            "additionalProperties": False,
        }
    ),
)


def _tool_level(row: ChatModelContract) -> ReasoningLevel:
    # Reasoning ON gives each provider an opportunity to emit its native replay
    # material. Adaptive providers may still legally omit it for a simple turn.
    return "low" if "low" in row.reasoning.levels else _cheapest_level(row)


async def _stream_tool_turn(
    row: ChatModelContract,
    intent: GenerateIntent,
    credential: ProviderCredential,
) -> tuple[FinalizedProviderCall, str, ToolCall, Presence[ContinuationArtifact]]:
    call = _plan(row, intent)
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    artifacts: list[ContinuationArtifact] = []
    terminals: list[TerminalEvent] = []
    sequence: list[int] = []
    async with httpx.AsyncClient() as http:
        async for event in ProviderRuntime(http).stream(call, credential=credential):
            sequence.append(event.seq)
            match event.event:
                case TextDelta(text=text):
                    text_parts.append(text)
                case ToolCallDone(tool_call=tool_call):
                    tool_calls.append(tool_call)
                case ContinuationDelta(artifact=artifact):
                    artifacts.append(artifact)
                case TerminalEvent() as terminal:
                    terminals.append(terminal)
                case _:
                    pass
            if isinstance(event.event, TerminalEvent):
                break
    assert sequence == list(range(1, len(sequence) + 1)), f"{_row_id(row)}: seq gap {sequence}"
    assert len(terminals) == 1, f"{_row_id(row)}: {len(terminals)} terminal events"
    outcome = terminals[0].outcome
    terminal = _accepted(row, outcome)
    _assert_call_facts(row, call, terminal)
    assert isinstance(terminal, Succeeded), f"{_row_id(row)}: tool turn incomplete: {terminal}"
    assert tool_calls, f"{_row_id(row)}: model streamed no completed tool call"
    assert len(artifacts) <= 1, f"{_row_id(row)}: more than one ContinuationDelta"
    continuation = terminal.response.continuation
    match continuation:
        case Present(value=artifact):
            assert artifacts == [artifact], (
                f"{_row_id(row)}: terminal continuation and streamed delta disagree"
            )
        case Absent():
            assert not artifacts, (
                f"{_row_id(row)}: streamed continuation delta absent from terminal response"
            )
    return call, "".join(text_parts), tool_calls[0], continuation


@pytest.mark.parametrize("row", _DIRECT_ROWS, ids=_row_id)
async def test_live_tool_call_and_continuation(live_env: LiveEnv, row: ChatModelContract) -> None:
    credential = live_env.credential_for(row.target.provider)
    level = _tool_level(row)
    prompt = (
        "You MUST call the search_library tool exactly once, with query='resonance engine', "
        "before answering. Then summarize the tool result in one sentence."
    )
    first_intent = _intent(
        row,
        prompt=prompt,
        stable_prefix=_SHORT_STABLE_PREFIX,
        max_output_tokens=max(_reasoning_budget(row, level), 2048),
        reasoning=level,
        tools=(_SEARCH_TOOL,),
    )
    _, assistant_text, tool_call, continuation = await _stream_tool_turn(
        row, first_intent, credential
    )
    assert tool_call.name == _SEARCH_TOOL.name
    assert "query" in dict(tool_call.arguments)

    replay_intent = _intent(
        row,
        prompt=prompt,
        stable_prefix=_SHORT_STABLE_PREFIX,
        max_output_tokens=max(_reasoning_budget(row, level), 2048),
        reasoning=level,
        tools=(_SEARCH_TOOL,),
        history=(
            AssistantMessage(
                text=assistant_text,
                tool_calls=(tool_call,),
                continuation=continuation,
            ),
            ToolResultMessage(
                call_id=tool_call.id,
                output="search_library result: the resonance engine links related sources.",
                is_error=False,
            ),
        ),
    )
    call, outcome = await _generate(row, replay_intent, credential)
    assert isinstance(outcome, Succeeded), f"{_row_id(row)}: continuation replay: {outcome}"
    _assert_call_facts(row, call, outcome)
    content = outcome.response.content
    assert isinstance(content, TextContent) and content.text.strip(), (
        f"{_row_id(row)}: continuation replay produced no text"
    )


# ---------------------------------------------------------------------------
# 5. Invalid-key probe per provider


def _invalid_key_row(provider: ProviderName) -> ChatModelContract:
    if provider == "openrouter":
        return _CERTIFYING_OPENROUTER_ROW
    return next(row for row in _DIRECT_ROWS if row.target.provider == provider)


@pytest.mark.parametrize("provider", _PROVIDER_ORDER)
async def test_live_invalid_key_is_credential_rejected(
    live_env: LiveEnv, provider: ProviderName
) -> None:
    live_env.credential_for(provider)  # provider selection + real-key presence gate
    row = _invalid_key_row(provider)
    intent = _intent(
        row,
        prompt="This call must fail before model output.",
        stable_prefix=_SHORT_STABLE_PREFIX,
        max_output_tokens=16,
        reasoning=_cheapest_level(row),
    )
    call = _plan(row, intent)
    async with httpx.AsyncClient() as http:
        with pytest.raises(CredentialRejected):
            await ProviderRuntime(http).generate(
                call,
                credential=ProviderCredential(provider=provider, key="invalid-live-matrix-key"),
            )


# ---------------------------------------------------------------------------
# 6. OpenRouter operator certification (THE certifier; §8 evidence artifact)


_ROUTED_KIMI_LEVELS: tuple[ReasoningLevel, ...] = ("low", "high", "max")


def _observed_upstream(row: ChatModelContract, outcome: CallOutcome) -> str:
    upstream = outcome.meta.upstream_provider
    assert isinstance(upstream, Present), f"{_row_id(row)}: no upstream provider observed"
    return upstream.value


def _generation_id(outcome: CallOutcome) -> str | None:
    request_id = outcome.meta.provider_request_id
    return request_id.value if isinstance(request_id, Present) else None


async def _fetch_endpoint_metadata(credential: ProviderCredential) -> object:
    async with httpx.AsyncClient() as http:
        response = await http.get(
            _OPENROUTER_ENDPOINTS_URL,
            headers={"Authorization": f"Bearer {credential.key}"},
            timeout=30,
        )
    assert response.status_code == 200, (
        f"endpoint metadata fetch failed: HTTP {response.status_code}"
    )
    return response.json()


def _openrouter_endpoint_provider(
    metadata: object, *, routing_endpoint_slug: str, model: str
) -> str:
    assert isinstance(metadata, dict), "OpenRouter endpoint metadata is not an object"
    data = metadata.get("data")
    assert isinstance(data, dict), "OpenRouter endpoint metadata carries no data object"
    endpoints = data.get("endpoints")
    assert isinstance(endpoints, list), "OpenRouter endpoint metadata carries no endpoint list"
    matches = [
        endpoint
        for endpoint in endpoints
        if isinstance(endpoint, dict)
        and endpoint.get("tag") == routing_endpoint_slug
        and endpoint.get("model_variant_slug", model) == model
    ]
    assert len(matches) == 1, (
        f"expected one endpoint tagged {routing_endpoint_slug!r}; found {len(matches)}"
    )
    endpoint = matches[0]
    expected_quantization = routing_endpoint_slug.rpartition("/")[2]
    assert endpoint.get("quantization") == expected_quantization, (
        f"endpoint {routing_endpoint_slug!r} reports quantization "
        f"{endpoint.get('quantization')!r}, expected {expected_quantization!r}"
    )
    provider_name = endpoint.get("provider_name")
    assert isinstance(provider_name, str) and provider_name, (
        f"endpoint {routing_endpoint_slug!r} carries no provider_name"
    )
    return provider_name


def _assert_openrouter_request_pin(call: FinalizedProviderCall, routing_endpoint_slug: str) -> None:
    body = json.loads(call.request.body)
    variant = routing_endpoint_slug.rpartition("/")[2]
    assert body.get("provider") == {
        "only": [routing_endpoint_slug],
        "order": [routing_endpoint_slug],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "quantizations": [variant],
    }, f"OpenRouter request did not preserve the exact endpoint pin: {body.get('provider')!r}"


async def test_openrouter_certification(live_env: LiveEnv) -> None:
    row = _CERTIFYING_OPENROUTER_ROW
    credential = live_env.credential_for("openrouter")
    cache_contract = row.cache
    assert isinstance(cache_contract, OpenRouterPrefixContract)
    pinned_upstream = cache_contract.pinned_upstream
    endpoint_metadata = await _fetch_endpoint_metadata(credential)
    observed_provider_name = _openrouter_endpoint_provider(
        endpoint_metadata,
        routing_endpoint_slug=pinned_upstream,
        model=row.target.model,
    )
    # Leave a clean provider window after endpoint discovery or the preceding
    # invalid-key probe before the first paid generation.
    await asyncio.sleep(3)

    # Reasoning probes: routed Kimi low|high|max (spec §4 recheck, routed arm).
    reasoning_probes: list[dict[str, object]] = []
    for level in _ROUTED_KIMI_LEVELS:
        call, outcome = await _generate(
            row,
            _intent(
                row,
                prompt="Answer in one short sentence: what is two plus two?",
                stable_prefix=_SHORT_STABLE_PREFIX,
                max_output_tokens=_reasoning_budget(row, level),
                reasoning=level,
            ),
            credential,
        )
        terminal = _accepted(row, outcome)
        _assert_call_facts(row, call, terminal)
        _assert_openrouter_request_pin(call, pinned_upstream)
        cache_plan = call.cache_plan
        assert isinstance(cache_plan, OpenRouterCertifiedPrefix)
        assert cache_plan.pinned_upstream == pinned_upstream
        observed = _observed_upstream(row, terminal)
        assert observed == observed_provider_name, (
            f"observed provider {observed!r}, expected {observed_provider_name!r} "
            f"for endpoint {pinned_upstream!r}"
        )
        reasoning_probes.append(
            {
                "level": level,
                "generation_id": _generation_id(terminal),
                "observed_provider_name": observed,
            }
        )
        # The operator route is deliberately single-attempt. Pace independent
        # paid probes instead of turning provider 429s into runtime retries.
        await asyncio.sleep(3)

    # Billed cache read on the warm/read pair (the §8 hard gate: endpoint
    # metadata claims supports_implicit_caching=false; only this paid probe
    # settles it — zero read keeps the route uncertified).
    warm, read, read_usage = await _cache_warm_read_pair(row, credential)
    _assert_openrouter_request_pin(warm[0], pinned_upstream)
    _assert_openrouter_request_pin(read[0], pinned_upstream)
    assert _observed_upstream(row, warm[1]) == observed_provider_name
    assert _observed_upstream(row, read[1]) == observed_provider_name
    cache_read_tokens = _count(read_usage.cache_read_input_tokens)
    assert cache_read_tokens > 0

    # Evidence artifact (§8): endpoint-metadata snapshot + probe generation ids
    # + observed cache usage. The revision id is content-addressed.
    captured_on = datetime.now(UTC).date().isoformat()
    artifact_body: dict[str, object] = {
        "captured_at": datetime.now(UTC).isoformat(),
        "catalog_revision": CATALOG_REVISION,
        "target": f"{row.target.provider}/{row.target.model}",
        "routing_endpoint_slug": pinned_upstream,
        "observed_provider_name": observed_provider_name,
        "canonical_revision": cache_contract.canonical_revision,
        "endpoint_metadata": endpoint_metadata,
        "reasoning_probes": reasoning_probes,
        "cache_probe": {
            "warm_generation_id": _generation_id(warm[1]),
            "read_generation_id": _generation_id(read[1]),
            "observed_cache_read_tokens": cache_read_tokens,
            "billed_cache_read": True,
        },
    }
    digest = hashlib.sha256(
        json.dumps(artifact_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    evidence_revision = f"openrouter-{captured_on}-{digest}"
    artifact = {"evidence_revision": evidence_revision, **artifact_body}

    _EVIDENCE_DIR.mkdir(exist_ok=True)
    evidence_path = _EVIDENCE_DIR / f"openrouter-{captured_on}.json"
    evidence_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        f"\nOpenRouter certification evidence written to {evidence_path}\n"
        "Pin in the catalog row: OperatorCertified("
        f"certified_pinned_upstream={pinned_upstream!r}, "
        f"certified_canonical_revision={cache_contract.canonical_revision!r}, "
        f"evidence_revision={evidence_revision!r})"
    )


# ---------------------------------------------------------------------------
# 7. Non-generation ports (openai-only)


async def test_live_embeddings(live_env: LiveEnv) -> None:
    credential = live_env.credential_for("openai")
    embedding_row = CATALOG.embeddings[0]
    async with httpx.AsyncClient() as http:
        response = await ProviderRuntime(http).embed(
            EmbeddingCall(
                model=embedding_row.target.model,
                inputs=("nexus live embedding smoke",),
                dimensions=Absent(),
            ),
            credential=credential,
        )
    assert len(response.embeddings) == 1
    assert response.embeddings[0]
    assert all(isinstance(value, float) for value in response.embeddings[0])


def _silent_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 4_000)
    return buffer.getvalue()


async def test_live_transcription(live_env: LiveEnv) -> None:
    credential = live_env.credential_for("openai")
    transcription_row = CATALOG.transcriptions[0]
    async with httpx.AsyncClient() as http:
        response = await ProviderRuntime(http).transcribe(
            TranscriptionCall(
                model=transcription_row.target.model,
                filename="silence.wav",
                content=_silent_wav_bytes(),
                media_type="audio/wav",
            ),
            credential=credential,
        )
    assert isinstance(response.text, str)
