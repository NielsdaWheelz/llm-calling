"""The sole planner: GenerateIntent -> FinalizedProviderCall | PlanRejected (spec §6).

planning.py owns ONLY computation — cache-affinity derivation (§8), the retry
policy constants, and the seven-step ``plan_generate`` — and re-exports the plan
value types that live in ``provider_runtime.types``.

Operator probes: there is deliberately NO ``plan_operator_probe`` public API.
The certification command builds its probe intents against the DIRECT moonshot
target, and the raw openrouter probe path is owned by the live tests
(``tests/live``) outside the planner. The planner's operator gate is therefore
absolute: an ``openrouter_chat`` target plans only with ``OperatorCertified``
evidence, and first certification cannot deadlock on it.

Scope ordering decision (§8): scopes form the total order
``Global ⊂ Owner ⊂ Conversation`` (broadest to narrowest). The narrowest
contributing scope wins. Two different owner ids, or two different conversation
ids, are a same-kind mismatch and a PlanningDefect. An Owner(a)+Conversation(c)
combination is ACCEPTED with narrowest=Conversation(c): cross-checking that the
conversation belongs to the stated owner is the prompt ASSEMBLER's authority
(nexus supplies already-authorized identities); the planner only orders scopes.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from typing import Final, assert_never

from provider_runtime import anthropic as _anthropic_codec
from provider_runtime import gemini as _gemini_codec
from provider_runtime import moonshot as _moonshot_codec
from provider_runtime import openai as _openai_codec
from provider_runtime import openrouter as _openrouter_codec
from provider_runtime.catalog import (
    CATALOG,
    CATALOG_REVISION,
    AnthropicPrefixContract,
    AutomaticPrefixContract,
    CacheContract,
    Catalog,
    ChatModelContract,
    OpenAIExplicitPrefixContract,
    OpenRouterPrefixContract,
    OperatorCertified,
    PricingContract,
)
from provider_runtime.errors import PlanningDefect
from provider_runtime.schema import canonical_schema_bytes
from provider_runtime.types import (
    Absent,
    Accounting,
    AnthropicPrefix,
    AssistantMessage,
    CachePlan,
    CacheScope,
    ContinuationArtifact,
    ConversationScope,
    DraftRequest,
    Dynamic,
    FinalizedProviderCall,
    FinalizedProviderRequest,
    GenerateIntent,
    GlobalScope,
    IntentContextTooLarge,
    OpenAIExplicitPrefix,
    OpenRouterCertifiedPrefix,
    OwnerScope,
    PlanRejected,
    Present,
    ProviderAutomaticPrefix,
    ProviderProtocol,
    ProviderTarget,
    RetryPolicy,
    Stable,
    StrictJsonOutput,
    SystemMessage,
    TextOutput,
    ToolResultMessage,
    UserMessage,
    cache_strategy,
    cache_ttl,
)

__all__ = [
    "CACHE_AFFINITY_VERSION",
    "EXTERNAL_LLM_RETRY",
    "OPENROUTER_SINGLE_ATTEMPT",
    "Accounting",
    "AnthropicPrefix",
    "CachePlan",
    "DraftRequest",
    "FinalizedProviderCall",
    "FinalizedProviderRequest",
    "OpenAIExplicitPrefix",
    "OpenRouterCertifiedPrefix",
    "PlanRejected",
    "ProviderAutomaticPrefix",
    "RetryPolicy",
    "cache_strategy",
    "cache_ttl",
    "canonical_cache_contract_bytes",
    "compute_cache_affinity",
    "plan_generate",
]


# ---------------------------------------------------------------------------
# Retry policies — the ONLY retry-schedule literals in the package (docs/rules
# retries.md central-catalog rule): callers cannot select or vary schedules, the
# planner resolves one of these two constants and nothing else defines delays.

# The central external-service policy.
EXTERNAL_LLM_RETRY: Final[RetryPolicy] = RetryPolicy(
    max_attempts=3,
    initial_delay_s=1.0,
    max_delay_s=8.0,
    jitter_s=0.25,
    deadline_s=Absent(),
)

# §7: the routed operator path never retries same-target (no hidden re-billing
# through the operator).
OPENROUTER_SINGLE_ATTEMPT: Final[RetryPolicy] = RetryPolicy(
    max_attempts=1,
    initial_delay_s=0.0,
    max_delay_s=0.0,
    jitter_s=0.0,
    deadline_s=Absent(),
)


# ---------------------------------------------------------------------------
# Cache affinity (§8) — planning.py solely owns CACHE_AFFINITY_VERSION and the
# binary frame encoder. Any framing, scope, prefix-encoding, or cache-contract
# semantic change MUST increment the version and update the checked-in
# cross-worker golden vectors (tests/goldens/cache_affinity.json); an old value
# is never recomputed under new rules.

CACHE_AFFINITY_VERSION: Final = 2

_AFFINITY_DOMAIN: Final = b"nexus-cache-affinity"


def _frame(component: bytes) -> bytes:
    """frame(b) = uint64_be(len(b)) || b — the §8 length-framing encoder."""
    return len(component).to_bytes(8, "big") + component


def _b64url_no_pad(digest: bytes) -> str:
    # Decision (golden-pinned): base64url WITHOUT padding. The affinity is a
    # 43-char string; '=' never appears in wire fields, ledger columns, or
    # goldens.
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _b64url_sha256(payload: bytes) -> str:
    return _b64url_no_pad(hashlib.sha256(payload).digest())


def canonical_cache_contract_bytes(contract: CacheContract) -> bytes:
    """Deterministic encoding of the closed cache strategy + parameters (§8).

    Sorted-key compact JSON of a dict carrying a ``strategy`` tag (the §11.4
    ledger strategy string) plus the variant's fields; Presence fields are
    serialized only when Present. No object ordering or locale participates.
    """
    payload: dict[str, object]
    match contract:
        case OpenAIExplicitPrefixContract(ttl=ttl, minimum_prefix_tokens=minimum):
            payload = {
                "strategy": "openai_explicit_prefix",
                "ttl": ttl,
                "minimum_prefix_tokens": minimum,
            }
        case AnthropicPrefixContract(ttl=ttl, minimum_prefix_tokens=minimum):
            payload = {
                "strategy": "anthropic_prefix",
                "ttl": ttl,
                "minimum_prefix_tokens": minimum,
            }
        case AutomaticPrefixContract(provider=provider, minimum_prefix_tokens=maybe_minimum):
            payload = {
                "strategy": "provider_automatic_prefix",
                "provider": provider,
            }
            match maybe_minimum:
                case Present(value=minimum):
                    payload["minimum_prefix_tokens"] = minimum
                case Absent():
                    pass
                case _:
                    assert_never(maybe_minimum)
        case OpenRouterPrefixContract(pinned_upstream=pinned, canonical_revision=revision):
            payload = {
                "strategy": "openrouter_certified_prefix",
                "pinned_upstream": pinned,
                "canonical_revision": revision,
            }
        case _:
            assert_never(contract)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def compute_cache_affinity(
    scope: CacheScope,
    target: ProviderTarget,
    protocol: str,
    cache_contract_bytes: bytes,
    prefix_bytes: bytes,
) -> str:
    """The §8 affinity formula, byte-exact.

    base64url(sha256(frame("nexus-cache-affinity") || frame(uint64_be(VERSION))
    || frame(scope-tag) || frame(scope-id-or-empty) || frame(exact-target) ||
    frame(protocol) || frame(cache-contract-bytes) || frame(prefix-bytes))).

    Literal/enum fields are UTF-8; scope ids are RFC 4122 network-order 16-byte
    form (``uuid.bytes``); ``global`` uses an empty id; the target frames as
    "provider/model". base64url is unpadded (see ``_b64url_no_pad``). The
    prefix bytes are the codec's DraftRequest.prefix_bytes, computed
    pre-finalize, so the affinity input excludes the injected key field by
    construction.
    """
    match scope:
        case GlobalScope():
            scope_tag, scope_id = b"global", b""
        case OwnerScope(owner_id=owner_id):
            scope_tag, scope_id = b"owner", owner_id.bytes
        case ConversationScope(conversation_id=conversation_id):
            scope_tag, scope_id = b"conversation", conversation_id.bytes
        case _:
            assert_never(scope)
    preimage = b"".join(
        (
            _frame(_AFFINITY_DOMAIN),
            _frame(CACHE_AFFINITY_VERSION.to_bytes(8, "big")),
            _frame(scope_tag),
            _frame(scope_id),
            _frame(f"{target.provider}/{target.model}".encode()),
            _frame(protocol.encode("utf-8")),
            _frame(cache_contract_bytes),
            _frame(prefix_bytes),
        )
    )
    return _b64url_sha256(preimage)


# ---------------------------------------------------------------------------
# Step 2 — intent validation (all PlanningDefect; oversize alone is the
# expected PlanRejected channel, measured in step 7 after finalize).


def _validate_limits(intent: GenerateIntent, contract: ChatModelContract) -> None:
    if not intent.messages:
        raise PlanningDefect(
            code="empty_messages",
            message="GenerateIntent.messages must be non-empty",
        )
    if intent.max_output_tokens < 1:
        raise PlanningDefect(
            code="invalid_max_output_tokens",
            message=f"max_output_tokens must be >= 1; got {intent.max_output_tokens}",
        )
    if intent.max_output_tokens > contract.output_limit:
        raise PlanningDefect(
            code="output_limit_exceeded",
            message=(
                f"max_output_tokens {intent.max_output_tokens} exceeds the "
                f"{_label(contract.target)} output limit {contract.output_limit}"
            ),
        )
    if intent.reasoning not in contract.reasoning.levels:
        raise PlanningDefect(
            code="unsupported_reasoning_level",
            message=(
                f"reasoning level {intent.reasoning!r} is not declared for "
                f"{_label(contract.target)}; declared: {list(contract.reasoning.levels)}"
            ),
        )
    if intent.tools and isinstance(intent.output, StrictJsonOutput):
        raise PlanningDefect(
            code="tools_with_strict_output",
            message=(
                "tools combined with StrictJsonOutput are an unsupported "
                "cross-product; choose tool calling or strict output"
            ),
        )


def _validate_stable_prefix(intent: GenerateIntent) -> tuple[CacheScope, int]:
    """Enforce the §8 stable-prefix rules; return (narrowest scope, breakpoint).

    Blocks are collected over SystemMessage/UserMessage in intent order. Every
    Stable block must precede every Dynamic block GLOBALLY: the stable prefix is
    one contiguous leading run across messages, and an AssistantMessage or
    ToolResultMessage ends the prefix region. At least one stable block must
    carry non-empty text (whitespace-only counts as empty — the sole realization
    of "caching has no off state" must cover real bytes). The returned
    breakpoint is the 0-based index of the LAST block of the leading stable run,
    counted over prompt blocks in order (== stable block count - 1).
    """
    prefix_open = True
    contributing: list[CacheScope] = []
    stable_blocks = 0
    has_nonempty_stable = False
    for message in intent.messages:
        match message:
            case SystemMessage(blocks=blocks) | UserMessage(blocks=blocks):
                for block in blocks:
                    match block.stability:
                        case Stable(scope=scope):
                            if not prefix_open:
                                raise PlanningDefect(
                                    code="stable_block_after_dynamic",
                                    message=(
                                        "stable blocks must form one contiguous "
                                        "leading prefix; found a Stable block after "
                                        "a Dynamic block or a conversation turn"
                                    ),
                                )
                            contributing.append(scope)
                            stable_blocks += 1
                            if block.text.strip():
                                has_nonempty_stable = True
                        case Dynamic():
                            prefix_open = False
                        case _:
                            assert_never(block.stability)
            case AssistantMessage() | ToolResultMessage():
                prefix_open = False
            case _:
                assert_never(message)
    if stable_blocks == 0:
        raise PlanningDefect(
            code="empty_stable_prefix",
            message=(
                "the stable prefix must be non-empty: caching has no off state, "
                "and no message block is marked Stable"
            ),
        )
    if not has_nonempty_stable:
        raise PlanningDefect(
            code="empty_stable_prefix",
            message=(
                "the stable prefix must contain at least one Stable block with "
                "non-empty text; all stable blocks are empty"
            ),
        )
    return _resolve_narrowest_scope(contributing), stable_blocks - 1


def _resolve_narrowest_scope(contributing: list[CacheScope]) -> CacheScope:
    """Resolve the narrowest contributing scope (Global ⊂ Owner ⊂ Conversation).

    Same-kind mismatches (two owner ids / two conversation ids) are planning
    defects. Owner(a)+Conversation(c) is accepted with narrowest=Conversation(c):
    ownership cross-checking is the assembler's authority (see module docstring).
    """
    owner_ids = {scope.owner_id for scope in contributing if isinstance(scope, OwnerScope)}
    conversation_ids = {
        scope.conversation_id for scope in contributing if isinstance(scope, ConversationScope)
    }
    if len(owner_ids) > 1:
        raise PlanningDefect(
            code="cache_scope_mismatch",
            message=(
                f"stable blocks contribute {len(owner_ids)} different owner scopes; "
                "a plan has exactly one owner identity"
            ),
        )
    if len(conversation_ids) > 1:
        raise PlanningDefect(
            code="cache_scope_mismatch",
            message=(
                f"stable blocks contribute {len(conversation_ids)} different "
                "conversation scopes; a plan has exactly one conversation identity"
            ),
        )
    if conversation_ids:
        return ConversationScope(conversation_id=next(iter(conversation_ids)))
    if owner_ids:
        return OwnerScope(owner_id=next(iter(owner_ids)))
    return GlobalScope()


# ---------------------------------------------------------------------------
# Step 3 — fingerprints (tools/schema are CanonicalJsonSchema by construction;
# nothing is re-parsed here).


def _tool_fingerprint(intent: GenerateIntent) -> str:
    # sha256 base64url of the framed concatenation of each tool's name,
    # description, and canonical schema bytes in declaration order; no tools
    # fingerprints the empty input.
    payload = b"".join(
        _frame(tool.name.encode("utf-8"))
        + _frame(tool.description.encode("utf-8"))
        + _frame(canonical_schema_bytes(tool.parameters))
        for tool in intent.tools
    )
    return _b64url_sha256(payload)


def _schema_fingerprint(intent: GenerateIntent) -> str:
    match intent.output:
        case StrictJsonOutput(schema=schema):
            return _b64url_sha256(_frame(canonical_schema_bytes(schema)))
        case TextOutput():
            # Decision: TextOutput fingerprints the EMPTY input (no frame) —
            # distinct by construction from every framed schema encoding.
            return _b64url_sha256(b"")
        case _:
            assert_never(intent.output)


# ---------------------------------------------------------------------------
# Step 4 — continuation validation. Codecs re-check at encode; the planner
# checks first so a mismatch fails with a clean PlanningDefect before any
# native encoding work.


def _validate_continuations(intent: GenerateIntent, contract: ChatModelContract) -> None:
    for message in intent.messages:
        match message:
            case AssistantMessage(continuation=Present(value=artifact)):
                _require_matching_continuation(artifact, intent, contract)
            case _:
                pass


def _require_matching_continuation(
    artifact: ContinuationArtifact, intent: GenerateIntent, contract: ChatModelContract
) -> None:
    if artifact.target != intent.target or artifact.codec_id != contract.continuation_codec:
        raise PlanningDefect(
            code="continuation_mismatch",
            message=(
                f"continuation artifact for {_label(artifact.target)} "
                f"(codec {artifact.codec_id!r}) is not replayable to "
                f"{_label(intent.target)} (codec {contract.continuation_codec!r})"
            ),
        )


# ---------------------------------------------------------------------------
# Steps 5+6 — codec dispatch (two-phase encode/finalize), affinity, cache plan.

type _EncodeFn = Callable[[GenerateIntent, ChatModelContract], DraftRequest]
type _FinalizeFn = Callable[[DraftRequest, str], FinalizedProviderRequest]


def _codec_functions(protocol: ProviderProtocol) -> tuple[_EncodeFn, _FinalizeFn]:
    match protocol:
        case "openai_responses":
            return _openai_codec.encode, _openai_codec.finalize
        case "anthropic_messages":
            return _anthropic_codec.encode, _anthropic_codec.finalize
        case "gemini_generate_content":
            return _gemini_codec.encode, _gemini_codec.finalize
        case "moonshot_chat":
            return _moonshot_codec.encode, _moonshot_codec.finalize
        case "openrouter_chat":
            return _openrouter_codec.encode, _openrouter_codec.finalize
        case _:
            assert_never(protocol)


def _require_operator_certification(contract: ChatModelContract) -> str:
    """The absolute operator gate: openrouter_chat plans only when certified.

    Returns the evidence revision threaded into OpenRouterCertifiedPrefix.
    """
    match contract.certification:
        case OperatorCertified(evidence_revision=evidence_revision):
            return evidence_revision
        case _:
            raise PlanningDefect(
                code="operator_route_uncertified",
                message=(
                    f"{_label(contract.target)}: openrouter_chat requires "
                    f"OperatorCertified paid-probe evidence; certification is "
                    f"{type(contract.certification).__name__}"
                ),
            )


def _cache_plan(
    contract: ChatModelContract,
    affinity: str,
    stable_breakpoint: int,
    evidence_revision: str | None,
) -> CachePlan:
    match contract.protocol:
        case "openai_responses":
            return OpenAIExplicitPrefix(key=affinity, minimum_ttl="30m", breakpoints=1)
        case "anthropic_messages":
            return AnthropicPrefix(
                stable_breakpoint=stable_breakpoint, ttl="5m", automatic_append_only=True
            )
        case "gemini_generate_content":
            return ProviderAutomaticPrefix(provider="gemini")
        case "moonshot_chat":
            return ProviderAutomaticPrefix(provider="moonshot")
        case "openrouter_chat":
            cache = contract.cache
            if not isinstance(cache, OpenRouterPrefixContract) or evidence_revision is None:
                raise PlanningDefect(
                    code="cache_contract_mismatch",
                    message=(
                        f"{_label(contract.target)}: openrouter_chat requires an "
                        f"OpenRouterPrefixContract cache contract with certification "
                        f"evidence; got {type(cache).__name__}"
                    ),
                )
            return OpenRouterCertifiedPrefix(
                session_id=affinity,
                pinned_upstream=cache.pinned_upstream,
                evidence_revision=evidence_revision,
            )
        case _:
            assert_never(contract.protocol)


# ---------------------------------------------------------------------------
# Step 7 — bound, reservation, accounting.


def _maximum_cost_estimate_usd_micros(
    pricing: PricingContract, bound: int, max_output_tokens: int, reasoning_reserve: int
) -> int:
    """Worst-case cost in usd micros — TELEMETRY ONLY, never a quota or a
    dollar reservation (§9).

    All planned input at the uncached input rate plus the full output budget
    and any reasoning reserve at the output rate; rates are usd micros per
    million tokens, so tokens*rate/1e6 yields micros. Decimal ROUND_HALF_UP to
    an integer micro.
    """
    micros = (
        Decimal(bound) * pricing.input_rate
        + Decimal(max_output_tokens + reasoning_reserve) * pricing.output_rate
    ) / Decimal(1_000_000)
    return int(micros.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _label(target: ProviderTarget) -> str:
    return f"{target.provider}/{target.model}"


# ---------------------------------------------------------------------------
# plan_generate — §6 seven steps in order.


def plan_generate(
    intent: GenerateIntent, catalog: Catalog = CATALOG
) -> FinalizedProviderCall | PlanRejected:
    """Plan one generation: the §6 seven steps, in order.

    1. require the exact catalog contract;
    2. validate the complete intent and capability cross-product;
    3. compile the canonical tool/schema fingerprints;
    4. validate same-target continuation artifacts;
    5. resolve provider-native reasoning and cache controls;
    6. construct and fingerprint the final native body (two-phase
       encode -> affinity -> finalize, so the affinity input excludes the
       injected key);
    7. freeze wire, retry, cache, reasoning, and accounting facts in one
       FinalizedProviderCall — or return PlanRejected(IntentContextTooLarge)
       for the one expected rejection.
    """
    # Step 1 — missing target raises RuntimeDefect(code="unknown_target").
    contract = catalog.chat_contract(intent.target)

    # Step 2.
    _validate_limits(intent, contract)
    scope, stable_breakpoint = _validate_stable_prefix(intent)

    # Step 3.
    tool_fingerprint = _tool_fingerprint(intent)
    schema_fingerprint = _schema_fingerprint(intent)

    # Step 4.
    _validate_continuations(intent, contract)

    # Steps 5+6. The operator gate fires before any encoding work.
    evidence_revision = (
        _require_operator_certification(contract)
        if contract.protocol == "openrouter_chat"
        else None
    )
    encode, finalize = _codec_functions(contract.protocol)
    draft = encode(intent, contract)
    affinity = compute_cache_affinity(
        scope,
        intent.target,
        contract.protocol,
        canonical_cache_contract_bytes(contract.cache),
        draft.prefix_bytes,
    )
    cache_plan = _cache_plan(contract, affinity, stable_breakpoint, evidence_revision)
    request = finalize(draft, affinity)

    # Step 7 — bytes-as-tokens bound (deliberately conservative,
    # certification-proved) against the context limit; body is bytes already.
    bound = len(request.body) + contract.provider_framing_overhead_tokens
    if bound > contract.context_limit:
        return PlanRejected(
            failure=IntentContextTooLarge(limit=contract.context_limit, measured=bound)
        )
    pricing = contract.pricing
    reasoning_reserve = (
        pricing.reasoning_reserve_tokens if pricing.reasoning_billed_outside_output else 0
    )
    reservation = bound + intent.max_output_tokens + reasoning_reserve
    accounting = Accounting(
        currency=pricing.currency,
        input_rate=pricing.input_rate,
        output_rate=pricing.output_rate,
        cache_write_rate=pricing.cache_write_rate,
        cache_read_rate=pricing.cache_read_rate,
        reasoning_billed_outside_output=pricing.reasoning_billed_outside_output,
        platform_token_reservation=reservation,
        maximum_cost_estimate_usd_micros=_maximum_cost_estimate_usd_micros(
            pricing, bound, intent.max_output_tokens, reasoning_reserve
        ),
    )
    retry_policy = (
        OPENROUTER_SINGLE_ATTEMPT if contract.protocol == "openrouter_chat" else EXTERNAL_LLM_RETRY
    )
    return FinalizedProviderCall(
        request=request,
        accounting=accounting,
        requested_reasoning=intent.reasoning,
        effective_reasoning=intent.reasoning,
        native_reasoning=draft.native_reasoning,
        cache_plan=cache_plan,
        retry_policy=retry_policy,
        catalog_revision=CATALOG_REVISION,
        request_fingerprint=_b64url_sha256(request.body),
        tool_fingerprint=tool_fingerprint,
        schema_fingerprint=schema_fingerprint,
        planned_input_token_upper_bound=bound,
        output_kind="strict_json" if isinstance(intent.output, StrictJsonOutput) else "text",
    )
