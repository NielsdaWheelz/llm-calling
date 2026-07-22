"""Planner tests (spec §6 seven steps, §8 affinity formula + goldens, §9 reservation).

The golden-vector suite is the cross-worker stability contract for
CACHE_AFFINITY_VERSION=3: it recomputes every checked-in vector through the
module AND through an independent reimplementation of the §8 formula, and
proves version-bump sensitivity. Never regenerate the goldens casually.
"""

import base64
import dataclasses
import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest

from provider_runtime import anthropic as anthropic_codec
from provider_runtime import moonshot as moonshot_codec
from provider_runtime import openai as openai_codec
from provider_runtime import openrouter as openrouter_codec
from provider_runtime.catalog import (
    CATALOG,
    CATALOG_REVISION,
    AnthropicPrefixContract,
    CacheContract,
    Catalog,
    GeminiAutomaticPrefixContract,
    MoonshotKeyedPrefixContract,
    OpenAIExplicitPrefixContract,
    OpenRouterPrefixContract,
    OperatorCertified,
    OperatorUncertified,
)
from provider_runtime.errors import PlanningDefect, RuntimeDefect
from provider_runtime.planning import (
    CACHE_AFFINITY_VERSION,
    EXTERNAL_LLM_RETRY,
    OPENROUTER_SINGLE_ATTEMPT,
    canonical_cache_contract_bytes,
    compute_cache_affinity,
    plan_generate,
)
from provider_runtime.schema import parse_canonical_schema
from provider_runtime.types import (
    Absent,
    AnthropicPrefix,
    AssistantMessage,
    CacheScope,
    CanonicalTool,
    ContinuationArtifact,
    ConversationScope,
    Dynamic,
    FinalizedProviderCall,
    GeminiAutomaticPrefix,
    GenerateIntent,
    GlobalScope,
    MoonshotKeyedPrefix,
    OpenAIExplicitPrefix,
    OpenRouterCertifiedPrefix,
    OwnerScope,
    PlanRejected,
    Present,
    PromptBlock,
    PromptMessage,
    ProviderTarget,
    ReasoningLevel,
    Stable,
    StrictJsonOutput,
    SystemMessage,
    TextOutput,
    ToolResultMessage,
    UserMessage,
    cache_strategy,
    cache_ttl,
)

GOLDENS = Path(__file__).parent / "goldens" / "cache_affinity.json"

OWNER_A = UUID("7d3f1c2a-9b4e-4f6d-8a1b-2c3d4e5f6a7b")
OWNER_B = UUID("0b1c2d3e-4f5a-4b6c-8d7e-9f0a1b2c3d4e")
CONV_A = UUID("1a2b3c4d-5e6f-4a1b-9c8d-7e6f5a4b3c2d")

OPENAI_TARGET = ProviderTarget(provider="openai", model="gpt-5.6-sol")
ANTHROPIC_TARGET = ProviderTarget(provider="anthropic", model="claude-sonnet-5")
GEMINI_TARGET = ProviderTarget(provider="gemini", model="gemini-3.5-flash")
MOONSHOT_TARGET = ProviderTarget(provider="moonshot", model="kimi-k3")
OPENROUTER_TARGET = ProviderTarget(provider="openrouter", model="moonshotai/kimi-k3-20260715")

TOOL_SCHEMA_RAW: dict[str, object] = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "Search query"}},
    "required": ["query"],
    "additionalProperties": False,
}
SEARCH_TOOL = CanonicalTool(
    name="search_library",
    description="Search the library",
    parameters=parse_canonical_schema(TOOL_SCHEMA_RAW),
)
OUTPUT_SCHEMA_RAW: dict[str, object] = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}
VERDICT_OUTPUT = StrictJsonOutput(name="verdict", schema=parse_canonical_schema(OUTPUT_SCHEMA_RAW))

SYSTEM_STABLE = SystemMessage(
    blocks=(PromptBlock(text="You are terse.", stability=Stable(scope=GlobalScope())),)
)
USER_DYNAMIC = UserMessage(blocks=(PromptBlock(text="hi", stability=Dynamic()),))


def intent_for(
    target: ProviderTarget,
    reasoning: ReasoningLevel,
    *,
    messages: tuple[PromptMessage, ...] = (SYSTEM_STABLE, USER_DYNAMIC),
    max_output_tokens: int = 512,
    tools: tuple[CanonicalTool, ...] = (),
    output: TextOutput | StrictJsonOutput | None = None,
) -> GenerateIntent:
    return GenerateIntent(
        target=target,
        messages=messages,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
        tools=tools,
        tool_choice="auto",
        output=TextOutput() if output is None else output,
    )


def openai_intent(**kwargs: object) -> GenerateIntent:
    return intent_for(OPENAI_TARGET, "medium", **kwargs)  # type: ignore[arg-type]


def plan_ok(intent: GenerateIntent, catalog: Catalog = CATALOG) -> FinalizedProviderCall:
    plan = plan_generate(intent, catalog)
    assert isinstance(plan, FinalizedProviderCall)
    return plan


# ---------------------------------------------------------------------------
# Golden vectors — the cross-worker affinity stability contract.


def _frame(component: bytes) -> bytes:
    return len(component).to_bytes(8, "big") + component


def _scope_from_spec(spec: dict[str, str]) -> CacheScope:
    match spec["kind"]:
        case "global":
            return GlobalScope()
        case "owner":
            return OwnerScope(owner_id=UUID(spec["id"]))
        case "conversation":
            return ConversationScope(conversation_id=UUID(spec["id"]))
        case other:
            raise AssertionError(f"unknown scope kind {other!r}")


def _contract_from_spec(spec: dict[str, object]) -> CacheContract:
    match spec["kind"]:
        case "openai_explicit":
            return OpenAIExplicitPrefixContract(
                ttl="30m",
                minimum_prefix_tokens=int(spec["minimum_prefix_tokens"]),  # type: ignore[arg-type]
            )
        case "anthropic":
            return AnthropicPrefixContract(
                ttl="5m",
                minimum_prefix_tokens=int(spec["minimum_prefix_tokens"]),  # type: ignore[arg-type]
            )
        case "gemini_automatic":
            minimum = spec.get("minimum_prefix_tokens")
            return GeminiAutomaticPrefixContract(
                minimum_prefix_tokens=Absent() if minimum is None else Present(int(minimum)),  # type: ignore[arg-type]
            )
        case "moonshot_keyed":
            minimum = spec.get("minimum_prefix_tokens")
            return MoonshotKeyedPrefixContract(
                minimum_prefix_tokens=Absent() if minimum is None else Present(int(minimum)),  # type: ignore[arg-type]
            )
        case "openrouter":
            return OpenRouterPrefixContract(
                pinned_upstream=str(spec["pinned_upstream"]),
                canonical_revision=str(spec["canonical_revision"]),
            )
        case other:
            raise AssertionError(f"unknown contract kind {other!r}")


def _vector_inputs(
    vector: dict[str, object],
) -> tuple[CacheScope, ProviderTarget, str, bytes, bytes]:
    target_spec = vector["target"]
    assert isinstance(target_spec, dict)
    prefix_components = vector["prefix_components_utf8"]
    assert isinstance(prefix_components, list)
    return (
        _scope_from_spec(vector["scope"]),  # type: ignore[arg-type]
        ProviderTarget(provider=target_spec["provider"], model=target_spec["model"]),
        str(vector["protocol"]),
        canonical_cache_contract_bytes(_contract_from_spec(vector["cache_contract"])),  # type: ignore[arg-type]
        b"".join(_frame(component.encode("utf-8")) for component in prefix_components),
    )


def _independent_affinity(
    version: int,
    scope: CacheScope,
    target: ProviderTarget,
    protocol: str,
    cache_contract_bytes: bytes,
    prefix_bytes: bytes,
) -> str:
    """A from-scratch reimplementation of the §8 formula (test-side spec)."""
    match scope:
        case GlobalScope():
            tag, scope_id = b"global", b""
        case OwnerScope(owner_id=owner_id):
            tag, scope_id = b"owner", owner_id.bytes
        case ConversationScope(conversation_id=conversation_id):
            tag, scope_id = b"conversation", conversation_id.bytes
        case other:
            raise AssertionError(other)
    preimage = (
        _frame(b"nexus-cache-affinity")
        + _frame(version.to_bytes(8, "big"))
        + _frame(tag)
        + _frame(scope_id)
        + _frame(f"{target.provider}/{target.model}".encode())
        + _frame(protocol.encode("utf-8"))
        + _frame(cache_contract_bytes)
        + _frame(prefix_bytes)
    )
    digest = hashlib.sha256(preimage).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _golden_vectors() -> list[dict[str, object]]:
    doc = json.loads(GOLDENS.read_text())
    assert doc["cache_affinity_version"] == CACHE_AFFINITY_VERSION == 3
    vectors = doc["vectors"]
    assert len(vectors) >= 12
    return vectors


class TestCacheAffinityGoldens:
    def test_recompute_matches_every_pinned_vector(self) -> None:
        for vector in _golden_vectors():
            computed = compute_cache_affinity(*_vector_inputs(vector))
            assert computed == vector["affinity"], vector["name"]

    def test_independent_formula_reimplementation_agrees(self) -> None:
        for vector in _golden_vectors():
            computed = _independent_affinity(3, *_vector_inputs(vector))
            assert computed == vector["affinity"], vector["name"]

    def test_version_bump_changes_every_vector(self) -> None:
        for vector in _golden_vectors():
            recomputed_v4 = _independent_affinity(4, *_vector_inputs(vector))
            assert recomputed_v4 != vector["affinity"], vector["name"]

    def test_pairwise_distinct(self) -> None:
        affinities = [vector["affinity"] for vector in _golden_vectors()]
        assert len(set(affinities)) == len(affinities)

    def test_unpadded_base64url_shape(self) -> None:
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        for vector in _golden_vectors():
            affinity = vector["affinity"]
            assert isinstance(affinity, str)
            assert len(affinity) == 43  # sha256 → 43 unpadded base64url chars
            assert "=" not in affinity
            assert set(affinity) <= allowed

    def test_scope_coverage(self) -> None:
        kinds = {vector["scope"]["kind"] for vector in _golden_vectors()}  # type: ignore[index]
        assert kinds == {"global", "owner", "conversation"}


class TestCanonicalCacheContractBytes:
    def test_openai_explicit(self) -> None:
        raw = canonical_cache_contract_bytes(
            OpenAIExplicitPrefixContract(ttl="30m", minimum_prefix_tokens=1024)
        )
        assert json.loads(raw) == {
            "strategy": "openai_explicit_prefix",
            "ttl": "30m",
            "minimum_prefix_tokens": 1024,
        }
        # Sorted-key compact form, byte-exact.
        assert (
            raw == b'{"minimum_prefix_tokens":1024,"strategy":"openai_explicit_prefix","ttl":"30m"}'
        )

    def test_anthropic(self) -> None:
        raw = canonical_cache_contract_bytes(
            AnthropicPrefixContract(ttl="5m", minimum_prefix_tokens=512)
        )
        assert raw == b'{"minimum_prefix_tokens":512,"strategy":"anthropic_prefix","ttl":"5m"}'

    def test_provider_cache_presence_fields_serialize_only_when_present(self) -> None:
        with_minimum = canonical_cache_contract_bytes(
            GeminiAutomaticPrefixContract(minimum_prefix_tokens=Present(4096))
        )
        assert (
            with_minimum == b'{"minimum_prefix_tokens":4096,"strategy":"gemini_automatic_prefix"}'
        )
        without_minimum = canonical_cache_contract_bytes(
            MoonshotKeyedPrefixContract(minimum_prefix_tokens=Absent())
        )
        assert without_minimum == b'{"strategy":"moonshot_keyed_prefix"}'

    def test_openrouter(self) -> None:
        raw = canonical_cache_contract_bytes(
            OpenRouterPrefixContract(
                pinned_upstream="moonshotai/int4",
                canonical_revision="moonshotai/kimi-k3-20260715",
            )
        )
        assert raw == (
            b'{"canonical_revision":"moonshotai/kimi-k3-20260715",'
            b'"pinned_upstream":"moonshotai/int4",'
            b'"strategy":"openrouter_certified_prefix"}'
        )


# ---------------------------------------------------------------------------
# Step 1+2 — validation failures (each a PlanningDefect with a distinct code;
# oversize alone is PlanRejected).


def expect_defect(intent: GenerateIntent, code: str, catalog: Catalog = CATALOG) -> None:
    with pytest.raises(PlanningDefect) as excinfo:
        plan_generate(intent, catalog)
    assert excinfo.value.code == code


class TestValidation:
    def test_unknown_target_is_runtime_defect(self) -> None:
        with pytest.raises(RuntimeDefect) as excinfo:
            plan_generate(intent_for(ProviderTarget(provider="openai", model="nope"), "medium"))
        assert excinfo.value.code == "unknown_target"

    def test_max_output_tokens_below_one(self) -> None:
        expect_defect(openai_intent(max_output_tokens=0), "invalid_max_output_tokens")

    def test_max_output_tokens_over_output_limit(self) -> None:
        limit = CATALOG.chat_contract(OPENAI_TARGET).output_limit
        expect_defect(openai_intent(max_output_tokens=limit + 1), "output_limit_exceeded")

    def test_unsupported_reasoning_level(self) -> None:
        expect_defect(intent_for(ANTHROPIC_TARGET, "none"), "unsupported_reasoning_level")

    def test_tools_with_strict_output_cross_product(self) -> None:
        expect_defect(
            openai_intent(tools=(SEARCH_TOOL,), output=VERDICT_OUTPUT),
            "tools_with_strict_output",
        )

    def test_empty_messages(self) -> None:
        expect_defect(openai_intent(messages=()), "empty_messages")


class TestStablePrefix:
    def test_no_stable_block_is_a_defect(self) -> None:
        expect_defect(openai_intent(messages=(USER_DYNAMIC,)), "empty_stable_prefix")

    def test_all_empty_text_stable_blocks_are_a_defect(self) -> None:
        empty_stable = SystemMessage(
            blocks=(
                PromptBlock(text="", stability=Stable(scope=GlobalScope())),
                PromptBlock(text="   ", stability=Stable(scope=GlobalScope())),
            )
        )
        expect_defect(openai_intent(messages=(empty_stable, USER_DYNAMIC)), "empty_stable_prefix")

    def test_stable_after_dynamic_in_same_message(self) -> None:
        broken = SystemMessage(
            blocks=(
                PromptBlock(text="stable", stability=Stable(scope=GlobalScope())),
                PromptBlock(text="dynamic", stability=Dynamic()),
                PromptBlock(text="late stable", stability=Stable(scope=GlobalScope())),
            )
        )
        expect_defect(openai_intent(messages=(broken, USER_DYNAMIC)), "stable_block_after_dynamic")

    def test_stable_after_dynamic_across_messages(self) -> None:
        late_stable = UserMessage(
            blocks=(PromptBlock(text="late", stability=Stable(scope=GlobalScope())),)
        )
        expect_defect(
            openai_intent(messages=(SYSTEM_STABLE, USER_DYNAMIC, late_stable)),
            "stable_block_after_dynamic",
        )

    def test_assistant_turn_ends_the_prefix_region(self) -> None:
        assistant = AssistantMessage(text="prior", tool_calls=(), continuation=Absent())
        late_stable = UserMessage(
            blocks=(PromptBlock(text="late", stability=Stable(scope=GlobalScope())),)
        )
        expect_defect(
            openai_intent(messages=(SYSTEM_STABLE, assistant, late_stable)),
            "stable_block_after_dynamic",
        )

    def test_tool_result_ends_the_prefix_region(self) -> None:
        tool_result = ToolResultMessage(call_id="call_1", output="{}", is_error=False)
        late_stable = UserMessage(
            blocks=(PromptBlock(text="late", stability=Stable(scope=GlobalScope())),)
        )
        expect_defect(
            openai_intent(messages=(SYSTEM_STABLE, tool_result, late_stable)),
            "stable_block_after_dynamic",
        )


def _two_scope_system(first: CacheScope, second: CacheScope) -> SystemMessage:
    return SystemMessage(
        blocks=(
            PromptBlock(text="alpha", stability=Stable(scope=first)),
            PromptBlock(text="beta", stability=Stable(scope=second)),
        )
    )


class TestScopeResolution:
    def test_two_owner_ids_rejected(self) -> None:
        messages = (
            _two_scope_system(OwnerScope(owner_id=OWNER_A), OwnerScope(owner_id=OWNER_B)),
            USER_DYNAMIC,
        )
        expect_defect(openai_intent(messages=messages), "cache_scope_mismatch")

    def test_two_conversation_ids_rejected(self) -> None:
        messages = (
            _two_scope_system(
                ConversationScope(conversation_id=CONV_A),
                ConversationScope(conversation_id=UUID("9e8d7c6b-5a4f-4e3d-8c2b-1a0f9e8d7c6b")),
            ),
            USER_DYNAMIC,
        )
        expect_defect(openai_intent(messages=messages), "cache_scope_mismatch")

    def test_owner_plus_conversation_accepted_narrowest_conversation(self) -> None:
        # Ownership cross-checking is the assembler's authority; the planner
        # only orders scopes: narrowest = Conversation(c).
        messages: tuple[PromptMessage, ...] = (
            _two_scope_system(
                OwnerScope(owner_id=OWNER_A), ConversationScope(conversation_id=CONV_A)
            ),
            USER_DYNAMIC,
        )
        intent = openai_intent(messages=messages)
        plan = plan_ok(intent)
        contract = CATALOG.chat_contract(OPENAI_TARGET)
        draft = openai_codec.encode(intent, contract)
        expected = compute_cache_affinity(
            ConversationScope(conversation_id=CONV_A),
            OPENAI_TARGET,
            contract.protocol,
            canonical_cache_contract_bytes(contract.cache),
            draft.prefix_bytes,
        )
        body = json.loads(plan.request.body)
        assert body["prompt_cache_key"] == expected

    def test_global_plus_owner_resolves_to_owner(self) -> None:
        messages: tuple[PromptMessage, ...] = (
            _two_scope_system(GlobalScope(), OwnerScope(owner_id=OWNER_A)),
            USER_DYNAMIC,
        )
        intent = openai_intent(messages=messages)
        plan = plan_ok(intent)
        contract = CATALOG.chat_contract(OPENAI_TARGET)
        draft = openai_codec.encode(intent, contract)
        expected = compute_cache_affinity(
            OwnerScope(owner_id=OWNER_A),
            OPENAI_TARGET,
            contract.protocol,
            canonical_cache_contract_bytes(contract.cache),
            draft.prefix_bytes,
        )
        assert json.loads(plan.request.body)["prompt_cache_key"] == expected


# ---------------------------------------------------------------------------
# Step 4 — continuation validation (planner-first; codecs re-check at encode).


class TestContinuationValidation:
    def _intent_with_artifact(self, artifact: ContinuationArtifact) -> GenerateIntent:
        assistant = AssistantMessage(text="prior", tool_calls=(), continuation=Present(artifact))
        return openai_intent(messages=(SYSTEM_STABLE, USER_DYNAMIC, assistant, USER_DYNAMIC))

    def test_codec_mismatch(self) -> None:
        artifact = ContinuationArtifact(
            target=OPENAI_TARGET, codec_id="anthropic_messages", opaque_payload={"output": [{}]}
        )
        expect_defect(self._intent_with_artifact(artifact), "continuation_mismatch")

    def test_target_mismatch(self) -> None:
        artifact = ContinuationArtifact(
            target=ANTHROPIC_TARGET, codec_id="openai_responses", opaque_payload={"output": [{}]}
        )
        expect_defect(self._intent_with_artifact(artifact), "continuation_mismatch")


# ---------------------------------------------------------------------------
# Steps 5+6 — operator gate, per-protocol cache plans, affinity injection.


def certified_openrouter_catalog(evidence_revision: str = "ev-test-1") -> Catalog:
    uncertified_row = CATALOG.chat_contract(OPENROUTER_TARGET)
    assert isinstance(uncertified_row.cache, OpenRouterPrefixContract)
    row = dataclasses.replace(
        uncertified_row,
        certification=OperatorCertified(
            certified_pinned_upstream=uncertified_row.cache.pinned_upstream,
            certified_canonical_revision=uncertified_row.cache.canonical_revision,
            evidence_revision=evidence_revision,
        ),
    )
    return Catalog(chat=(row,), embeddings=(), transcriptions=())


class TestOperatorGate:
    def test_catalog_openrouter_row_is_uncertified_today(self) -> None:
        # Pin the current catalog fact the gate depends on.
        row = CATALOG.chat_contract(OPENROUTER_TARGET)
        assert isinstance(row.certification, OperatorUncertified)

    def test_uncertified_operator_route_is_a_planning_defect(self) -> None:
        expect_defect(intent_for(OPENROUTER_TARGET, "max"), "operator_route_uncertified")

    def test_certified_operator_route_plans_with_evidence_threaded(self) -> None:
        catalog = certified_openrouter_catalog("ev-cert-42")
        intent = intent_for(OPENROUTER_TARGET, "max")
        plan = plan_ok(intent, catalog)
        assert isinstance(plan.cache_plan, OpenRouterCertifiedPrefix)
        assert plan.cache_plan.pinned_upstream == "moonshotai/int4"
        assert plan.cache_plan.evidence_revision == "ev-cert-42"
        body = json.loads(plan.request.body)
        assert body["session_id"] == plan.cache_plan.session_id
        assert plan.retry_policy is OPENROUTER_SINGLE_ATTEMPT


class TestCachePlans:
    def test_openai_explicit_prefix(self) -> None:
        plan = plan_ok(openai_intent())
        assert isinstance(plan.cache_plan, OpenAIExplicitPrefix)
        assert plan.cache_plan.minimum_ttl == "30m"
        assert plan.cache_plan.breakpoints == 1
        assert json.loads(plan.request.body)["prompt_cache_key"] == plan.cache_plan.key
        assert cache_strategy(plan.cache_plan) == "openai_explicit_prefix"
        assert cache_ttl(plan.cache_plan) == "30m"

    def test_anthropic_prefix_breakpoint_is_last_stable_block_index(self) -> None:
        two_stable = SystemMessage(
            blocks=(
                PromptBlock(text="alpha", stability=Stable(scope=GlobalScope())),
                PromptBlock(text="beta", stability=Stable(scope=GlobalScope())),
            )
        )
        plan = plan_ok(intent_for(ANTHROPIC_TARGET, "high", messages=(two_stable, USER_DYNAMIC)))
        assert isinstance(plan.cache_plan, AnthropicPrefix)
        assert plan.cache_plan.stable_breakpoint == 1  # 0-based index of the last stable block
        assert plan.cache_plan.ttl == "5m"
        assert plan.cache_plan.automatic_append_only is True
        assert cache_strategy(plan.cache_plan) == "anthropic_prefix"
        assert cache_ttl(plan.cache_plan) == "5m"

    def test_gemini_automatic_prefix(self) -> None:
        plan = plan_ok(intent_for(GEMINI_TARGET, "medium"))
        assert plan.cache_plan == GeminiAutomaticPrefix()
        assert cache_strategy(plan.cache_plan) == "gemini_automatic_prefix"
        assert cache_ttl(plan.cache_plan) is None

    def test_moonshot_keyed_prefix(self) -> None:
        plan = plan_ok(intent_for(MOONSHOT_TARGET, "max"))
        assert isinstance(plan.cache_plan, MoonshotKeyedPrefix)
        assert json.loads(plan.request.body)["prompt_cache_key"] == plan.cache_plan.key
        assert cache_strategy(plan.cache_plan) == "moonshot_keyed_prefix"
        assert cache_ttl(plan.cache_plan) is None

    def test_openrouter_certified_prefix_strategy(self) -> None:
        plan = plan_ok(intent_for(OPENROUTER_TARGET, "max"), certified_openrouter_catalog())
        assert cache_strategy(plan.cache_plan) == "openrouter_certified_prefix"
        assert cache_ttl(plan.cache_plan) is None


class TestAffinityInjection:
    def test_affinity_input_excludes_the_injected_openai_key(self) -> None:
        intent = openai_intent()
        contract = CATALOG.chat_contract(OPENAI_TARGET)
        plan = plan_ok(intent)
        draft = openai_codec.encode(intent, contract)
        expected_affinity = compute_cache_affinity(
            GlobalScope(),
            OPENAI_TARGET,
            contract.protocol,
            canonical_cache_contract_bytes(contract.cache),
            draft.prefix_bytes,
        )
        body = json.loads(plan.request.body)
        assert body.pop("prompt_cache_key") == expected_affinity
        # With the injected key removed, the finalized body is the draft body:
        # the affinity input cannot have contained its own output.
        assert body == json.loads(draft.body)

    def test_openrouter_body_carries_the_affinity_as_session_id(self) -> None:
        catalog = certified_openrouter_catalog()
        intent = intent_for(OPENROUTER_TARGET, "max")
        contract = catalog.chat_contract(OPENROUTER_TARGET)
        plan = plan_ok(intent, catalog)
        draft = openrouter_codec.encode(intent, contract)
        expected_affinity = compute_cache_affinity(
            GlobalScope(),
            OPENROUTER_TARGET,
            contract.protocol,
            canonical_cache_contract_bytes(contract.cache),
            draft.prefix_bytes,
        )
        body = json.loads(plan.request.body)
        assert body.pop("session_id") == expected_affinity
        assert body == json.loads(draft.body)
        assert isinstance(plan.cache_plan, OpenRouterCertifiedPrefix)
        assert plan.cache_plan.session_id == expected_affinity

    def test_moonshot_body_carries_the_affinity_as_prompt_cache_key(self) -> None:
        intent = intent_for(MOONSHOT_TARGET, "max")
        contract = CATALOG.chat_contract(MOONSHOT_TARGET)
        plan = plan_ok(intent)
        draft = moonshot_codec.encode(intent, contract)
        expected_affinity = compute_cache_affinity(
            GlobalScope(),
            MOONSHOT_TARGET,
            contract.protocol,
            canonical_cache_contract_bytes(contract.cache),
            draft.prefix_bytes,
        )
        body = json.loads(plan.request.body)
        assert body.pop("prompt_cache_key") == expected_affinity
        assert body == json.loads(draft.body)
        assert plan.cache_plan == MoonshotKeyedPrefix(key=expected_affinity)

    def test_anthropic_body_carries_no_affinity_field(self) -> None:
        plan = plan_ok(intent_for(ANTHROPIC_TARGET, "high"))
        contract = CATALOG.chat_contract(ANTHROPIC_TARGET)
        draft = anthropic_codec.encode(intent_for(ANTHROPIC_TARGET, "high"), contract)
        assert plan.request.body == draft.body  # passthrough finalize


# ---------------------------------------------------------------------------
# Step 7 — bound, PlanRejected, reservation, accounting, retry selection.


def _independent_half_up_micros(numerator: int) -> int:
    quotient, remainder = divmod(numerator, 1_000_000)
    return quotient + (1 if remainder * 2 >= 1_000_000 else 0)


class TestBoundAndAccounting:
    def test_oversize_intent_is_plan_rejected_with_exact_limit_and_measured(self) -> None:
        intent = openai_intent()
        reference = plan_ok(intent)
        small = dataclasses.replace(CATALOG.chat_contract(OPENAI_TARGET), context_limit=100)
        rejected = plan_generate(intent, Catalog(chat=(small,), embeddings=(), transcriptions=()))
        assert isinstance(rejected, PlanRejected)
        assert rejected.failure.limit == 100
        # Same intent, same encoding: the measured bound equals the reference
        # plan's planned_input_token_upper_bound (context_limit never affects
        # the body).
        assert rejected.failure.measured == reference.planned_input_token_upper_bound
        assert rejected.failure.measured > 100

    def test_bound_is_body_bytes_plus_framing_overhead(self) -> None:
        plan = plan_ok(openai_intent())
        contract = CATALOG.chat_contract(OPENAI_TARGET)
        assert plan.planned_input_token_upper_bound == (
            len(plan.request.body) + contract.provider_framing_overhead_tokens
        )

    def test_reservation_without_reasoning_reserve(self) -> None:
        plan = plan_ok(openai_intent(max_output_tokens=512))
        assert plan.accounting.platform_token_reservation == (
            plan.planned_input_token_upper_bound + 512
        )

    def test_reservation_with_reasoning_reserve_branch(self) -> None:
        base = CATALOG.chat_contract(OPENAI_TARGET)
        pricing = dataclasses.replace(
            base.pricing, reasoning_billed_outside_output=True, reasoning_reserve_tokens=5_000
        )
        catalog = Catalog(
            chat=(dataclasses.replace(base, pricing=pricing),), embeddings=(), transcriptions=()
        )
        plan = plan_ok(openai_intent(max_output_tokens=512), catalog)
        assert plan.accounting.platform_token_reservation == (
            plan.planned_input_token_upper_bound + 512 + 5_000
        )
        assert plan.accounting.reasoning_billed_outside_output is True
        # The reserve is priced at the output rate in the telemetry estimate.
        expected = _independent_half_up_micros(
            plan.planned_input_token_upper_bound * pricing.input_rate
            + (512 + 5_000) * pricing.output_rate
        )
        assert plan.accounting.maximum_cost_estimate_usd_micros == expected

    def test_maximum_cost_estimate_is_worst_case_micros(self) -> None:
        plan = plan_ok(openai_intent(max_output_tokens=512))
        pricing = CATALOG.chat_contract(OPENAI_TARGET).pricing
        expected = _independent_half_up_micros(
            plan.planned_input_token_upper_bound * pricing.input_rate + 512 * pricing.output_rate
        )
        assert plan.accounting.maximum_cost_estimate_usd_micros == expected

    def test_accounting_freezes_catalog_rates(self) -> None:
        plan = plan_ok(openai_intent())
        pricing = CATALOG.chat_contract(OPENAI_TARGET).pricing
        assert plan.accounting.currency == "usd"
        assert plan.accounting.input_rate == pricing.input_rate
        assert plan.accounting.output_rate == pricing.output_rate
        assert plan.accounting.cache_read_rate == pricing.cache_read_rate
        assert plan.accounting.cache_write_rate == pricing.cache_write_rate


class TestRetryPolicySelection:
    def test_direct_protocols_use_the_central_external_policy(self) -> None:
        for intent in (
            openai_intent(),
            intent_for(ANTHROPIC_TARGET, "high"),
            intent_for(GEMINI_TARGET, "medium"),
            intent_for(MOONSHOT_TARGET, "max"),
        ):
            assert plan_ok(intent).retry_policy is EXTERNAL_LLM_RETRY

    def test_policy_literals_pinned(self) -> None:
        # The ONLY retry-schedule literals in the package (docs/rules
        # retries.md central-catalog rule).
        assert EXTERNAL_LLM_RETRY.max_attempts == 3
        assert EXTERNAL_LLM_RETRY.initial_delay_s == 1.0
        assert EXTERNAL_LLM_RETRY.max_delay_s == 8.0
        assert EXTERNAL_LLM_RETRY.jitter_s == 0.25
        assert EXTERNAL_LLM_RETRY.deadline_s == Absent()
        assert OPENROUTER_SINGLE_ATTEMPT.max_attempts == 1
        assert OPENROUTER_SINGLE_ATTEMPT.initial_delay_s == 0.0
        assert OPENROUTER_SINGLE_ATTEMPT.max_delay_s == 0.0
        assert OPENROUTER_SINGLE_ATTEMPT.jitter_s == 0.0
        assert OPENROUTER_SINGLE_ATTEMPT.deadline_s == Absent()


# ---------------------------------------------------------------------------
# Fingerprints, determinism, and remaining plan facts.


def _b64url_sha256(payload: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")


class TestFingerprintsAndDeterminism:
    def test_plan_is_deterministic(self) -> None:
        first = plan_ok(openai_intent(tools=(SEARCH_TOOL,)))
        second = plan_ok(openai_intent(tools=(SEARCH_TOOL,)))
        assert first.request.body == second.request.body  # byte-identical
        assert first.request_fingerprint == second.request_fingerprint
        assert first.tool_fingerprint == second.tool_fingerprint
        assert first.schema_fingerprint == second.schema_fingerprint
        assert first.cache_plan == second.cache_plan

    def test_request_fingerprint_is_hash_of_finalized_body(self) -> None:
        plan = plan_ok(openai_intent())
        assert plan.request_fingerprint == _b64url_sha256(plan.request.body)

    def test_tool_fingerprint_sensitivity(self) -> None:
        without_tools = plan_ok(openai_intent())
        with_tools = plan_ok(openai_intent(tools=(SEARCH_TOOL,)))
        assert without_tools.tool_fingerprint != with_tools.tool_fingerprint
        assert without_tools.tool_fingerprint == _b64url_sha256(b"")

    def test_schema_fingerprint_sensitivity(self) -> None:
        text = plan_ok(openai_intent())
        strict = plan_ok(openai_intent(output=VERDICT_OUTPUT))
        assert text.schema_fingerprint != strict.schema_fingerprint
        # TextOutput fingerprints the empty input.
        assert text.schema_fingerprint == _b64url_sha256(b"")

    def test_output_kind(self) -> None:
        assert plan_ok(openai_intent()).output_kind == "text"
        assert plan_ok(openai_intent(output=VERDICT_OUTPUT)).output_kind == "strict_json"

    def test_reasoning_and_catalog_facts_threaded(self) -> None:
        plan = plan_ok(openai_intent())
        assert plan.requested_reasoning == "medium"
        assert plan.effective_reasoning == "medium"
        assert plan.native_reasoning == "medium"
        assert plan.catalog_revision == CATALOG_REVISION
        assert plan.request.method == "POST"
        assert plan.request.protocol == "openai_responses"
