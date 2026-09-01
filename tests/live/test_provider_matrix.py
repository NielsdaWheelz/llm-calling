"""Paid live capability matrix over every registry row (spec §11).

Fail-closed acceptance probes against the real providers, driven through the
real ``ProviderRuntime``. Excluded from the default suite (``addopts``
deselects ``live_provider``); run with:

    LLM_RUNTIME_LIVE=1 uv run pytest -m live_provider tests/live/test_provider_matrix.py

Environment contract (read by this test module only — the library reads none):

- ``LLM_RUNTIME_LIVE=1`` is required — anything else fails, never skips;
- one key per provider: ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
  ``GEMINI_API_KEY``, ``MOONSHOT_API_KEY``, ``OPENROUTER_API_KEY``,
  ``DEEPSEEK_API_KEY``, ``XAI_API_KEY``. A missing key skips that provider's
  rows with a loud reason and the skip is recorded in evidence;
- narrowing (``-k``) is a debugging aid — release evidence runs unfiltered.

Per registry row the matrix probes: plain chat, streaming (envelope grammar +
exactly one terminal), one tool round trip, ``json_out`` (typed reply), and a
two-turn reasoning continuation replay — each probe skipping cleanly when the
row's capabilities say unsupported. DeepSeek additionally proves its
thinking-mode tool continuation: reasoning + tool call + native continuation
replay + tool result + final answer. Every run writes one evidence file into
``tests/live/evidence/`` (timestamped, per-row per-probe status + usage +
request ids + the registry revision). Evidence values pass through the
package's own redaction before they are written.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import assert_never

import pydantic
import pytest

from provider_runtime import (
    Credentials,
    ProviderRuntime,
)
from provider_runtime.errors import sanitize_provider_text
from provider_runtime.registry import (
    _ROWS as ROWS,
)
from provider_runtime.registry import (
    REGISTRY_REVISION,
)
from provider_runtime.registry import (
    _ModelRow as ModelRow,
)
from provider_runtime.types import (
    Absent,
    AssistantMessage,
    CallMeta,
    CanonicalTool,
    ContinuationDelta,
    GenerateIntent,
    Present,
    PromptBlock,
    ProviderName,
    ProviderTarget,
    ReasoningLevel,
    StreamStart,
    StructuredReply,
    Succeeded,
    SystemMessage,
    TerminalEvent,
    TextContent,
    TextDelta,
    TextOutput,
    ToolResultMessage,
    UserMessage,
)

pytestmark = pytest.mark.live_provider

_EVIDENCE_DIR = Path(__file__).parent / "evidence"
_MAX_OUTPUT_TOKENS = 4096
_SYSTEM = "You are a terse live-matrix probe. Answer exactly as instructed."

_PROVIDER_ENV: Mapping[ProviderName, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
}

# Cheapest-first probe order over the closed level vocabulary; a row declares a
# subset and each probe picks from what the row declares, never inventing one.
_LEVEL_ORDER: tuple[ReasoningLevel, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


# ---------------------------------------------------------------------------
# Selection helpers


def _require_live() -> None:
    if os.environ.get("LLM_RUNTIME_LIVE") != "1":
        pytest.fail("the live provider matrix requires LLM_RUNTIME_LIVE=1", pytrace=False)


def _credentials(provider: ProviderName, key: str) -> Credentials:
    match provider:
        case "openai":
            return Credentials(openai=key)
        case "anthropic":
            return Credentials(anthropic=key)
        case "gemini":
            return Credentials(gemini=key)
        case "moonshot":
            return Credentials(moonshot=key)
        case "openrouter":
            return Credentials(openrouter=key)
        case "deepseek":
            return Credentials(deepseek=key)
        case "xai":
            return Credentials(xai=key)
        case _:
            assert_never(provider)


def _runtime_for(row: ModelRow) -> ProviderRuntime:
    _require_live()
    variable = _PROVIDER_ENV[row.provider]
    key = os.environ.get(variable)
    if not key:
        pytest.skip(f"no {row.provider} credential: set {variable} to probe row {row.ref}")
    return ProviderRuntime(credentials=_credentials(row.provider, key))


def _declared_levels(row: ModelRow) -> tuple[ReasoningLevel, ...]:
    match row.reasoning:
        case Present(value=levels):
            return tuple(level for level in _LEVEL_ORDER if level in levels)
        case Absent():
            return ()
        case _:
            assert_never(row.reasoning)


def _chat_reasoning(row: ModelRow) -> ReasoningLevel:
    """The cheapest level the row can express ("none" when it has no knob)."""
    declared = _declared_levels(row)
    return declared[0] if declared else "none"


def _replay_reasoning(row: ModelRow) -> ReasoningLevel | None:
    """The declared level that reliably reasons; None when the row has none.

    "high" over the cheapest declared level: certified live that claude-sonnet-5
    at "low" effort completes without emitting a thinking signature, so a cheap
    level can leave nothing for the continuation probe to replay."""
    declared: tuple[ReasoningLevel, ...] = tuple(
        level for level in _declared_levels(row) if level != "none"
    )
    if not declared:
        return None
    return "high" if "high" in declared else declared[-1]


def _intent(
    row: ModelRow,
    *,
    user: str,
    reasoning: ReasoningLevel | None = None,
    tools: tuple[CanonicalTool, ...] = (),
) -> GenerateIntent:
    return GenerateIntent(
        target=ProviderTarget(provider=row.provider, model=row.model_id),
        messages=(
            SystemMessage(blocks=(PromptBlock(text=_SYSTEM),)),
            UserMessage(blocks=(PromptBlock(text=user),)),
        ),
        max_output_tokens=min(_MAX_OUTPUT_TOKENS, row.max_output_tokens),
        reasoning=reasoning if reasoning is not None else _chat_reasoning(row),
        tools=tools,
        tool_choice="auto",
        output=TextOutput(),
    )


# ---------------------------------------------------------------------------
# Evidence recording — one JSON file per run


class _EvidenceRecorder:
    """Accumulates per-row per-probe records; written once at module teardown."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, dict[str, object]]] = {}

    @contextmanager
    def probe(self, row: ModelRow, probe: str) -> Iterator[dict[str, object]]:
        record: dict[str, object] = {}
        try:
            yield record
        except BaseException as error:
            if isinstance(error, pytest.skip.Exception):
                record["status"] = "skipped"
                record["reason"] = str(error)
            else:
                record["status"] = "failed"
                record["detail"] = sanitize_provider_text(str(error))
            raise
        else:
            record["status"] = "passed"
        finally:
            self.rows.setdefault(row.ref, {})[probe] = record


def _meta_evidence(meta: CallMeta) -> dict[str, object]:
    evidence: dict[str, object] = {
        "provider_request_id": (
            meta.provider_request_id.value
            if isinstance(meta.provider_request_id, Present)
            else None
        ),
        "upstream_provider": (
            meta.upstream_provider.value if isinstance(meta.upstream_provider, Present) else None
        ),
        "attempts": len(meta.attempt_trace),
    }
    if isinstance(meta.usage, Present):
        usage = meta.usage.value
        evidence["usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
    return evidence


def _write_evidence(recorder: _EvidenceRecorder) -> None:
    # Nothing to record from a run that never went live (fail-closed runs
    # without LLM_RUNTIME_LIVE=1 produce no evidence, only failures).
    if not recorder.rows or os.environ.get("LLM_RUNTIME_LIVE") != "1":
        return
    _EVIDENCE_DIR.mkdir(exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": "provider-runtime-live-evidence.v1",
        "registry_revision": REGISTRY_REVISION,
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": recorder.rows,
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    revision = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    payload["evidence_revision"] = revision
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    name = f"provider-runtime-{stamp}-{revision}.json"
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if any(marker in rendered for marker in ("sk-", "Bearer ", "AIza")):
        pytest.fail("live provider evidence contains a credential-shaped value", pytrace=False)
    (_EVIDENCE_DIR / name).write_text(rendered)


@pytest.fixture(scope="module", autouse=True)
def evidence() -> Iterator[_EvidenceRecorder]:
    recorder = _EvidenceRecorder()
    yield recorder
    _write_evidence(recorder)


# ---------------------------------------------------------------------------
# Probes — one test per capability, parametrized over every registry row


_ROW_IDS = [row.ref for row in ROWS]
_DEEPSEEK_ROWS = tuple(row for row in ROWS if row.provider == "deepseek")
_DEEPSEEK_ROW_IDS = [row.ref for row in _DEEPSEEK_ROWS]


@pytest.mark.parametrize("row", ROWS, ids=_ROW_IDS)
async def test_chat_probe(row: ModelRow, evidence: _EvidenceRecorder) -> None:
    with evidence.probe(row, "chat") as record:
        runtime = _runtime_for(row)
        level = _chat_reasoning(row)
        outcome = await runtime.chat(
            row.ref,
            system=_SYSTEM,
            user="Reply with the single word: pong",
            reasoning=level,
            max_output_tokens=min(_MAX_OUTPUT_TOKENS, row.max_output_tokens),
        )
        assert isinstance(outcome, Succeeded), f"chat probe did not succeed: {outcome!r}"
        content = outcome.response.content
        assert isinstance(content, TextContent), f"chat probe decoded {type(content).__name__}"
        assert content.text.strip(), "chat probe produced empty text"
        assert outcome.meta.registry_revision == REGISTRY_REVISION
        record["reasoning"] = level
        record.update(_meta_evidence(outcome.meta))


@pytest.mark.parametrize("row", ROWS, ids=_ROW_IDS)
async def test_stream_probe(row: ModelRow, evidence: _EvidenceRecorder) -> None:
    with evidence.probe(row, "stream") as record:
        runtime = _runtime_for(row)
        if not row.streaming:
            pytest.skip(f"row {row.ref} declares streaming=False")
        intent = _intent(row, user="Count from 1 to 5 as digits separated by spaces.")
        events = [event async for event in runtime.stream(intent)]

        assert events, "stream produced no events"
        for index, event in enumerate(events, start=1):
            assert event.seq == index, (
                f"stream seq must be 1-based and gapless; event {index} carries seq {event.seq}"
            )
        assert isinstance(events[0].event, StreamStart), (
            f"stream must open with StreamStart; got {type(events[0].event).__name__}"
        )
        terminals = [event.event for event in events if isinstance(event.event, TerminalEvent)]
        assert len(terminals) == 1, f"stream must end in exactly one terminal; saw {len(terminals)}"
        assert isinstance(events[-1].event, TerminalEvent), "the terminal must be the last event"
        continuation_count = sum(
            1 for event in events if isinstance(event.event, ContinuationDelta)
        )
        assert continuation_count <= 1, (
            f"at most one ContinuationDelta per stream; saw {continuation_count}"
        )
        text = "".join(event.event.text for event in events if isinstance(event.event, TextDelta))
        assert text.strip(), "stream produced no text deltas"

        outcome = terminals[0].outcome
        assert isinstance(outcome, Succeeded), f"stream probe did not succeed: {outcome!r}"
        assert isinstance(outcome.meta.usage, Present), (
            "the stream terminal meta must fold provider usage"
        )
        record["event_count"] = len(events)
        record["event_kinds"] = sorted({type(event.event).__name__ for event in events})
        record.update(_meta_evidence(outcome.meta))


@pytest.mark.parametrize("row", ROWS, ids=_ROW_IDS)
async def test_tools_probe(row: ModelRow, evidence: _EvidenceRecorder) -> None:
    with evidence.probe(row, "tools") as record:
        runtime = _runtime_for(row)
        if not row.tools:
            pytest.skip(f"row {row.ref} declares tools=False")
        tool = CanonicalTool(
            name="lookup_temperature",
            description="Return the current temperature for a city, in celsius.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        )
        intent = _intent(
            row,
            user="Use the lookup_temperature tool for Paris, then state the temperature.",
            tools=(tool,),
        )
        first = await runtime.generate(intent)
        assert isinstance(first, Succeeded), f"tool turn did not succeed: {first!r}"
        content = first.response.content
        assert isinstance(content, TextContent), f"tool turn decoded {type(content).__name__}"
        assert content.tool_calls, "the model made no tool call"
        call = content.tool_calls[0]
        assert call.name == tool.name, f"unexpected tool called: {call.name!r}"

        second_intent = replace(
            intent,
            messages=(
                *intent.messages,
                AssistantMessage(
                    text=content.text,
                    tool_calls=content.tool_calls,
                    continuation=first.response.continuation,
                ),
                ToolResultMessage(call_id=call.id, output='{"temperature_c": 21}', is_error=False),
            ),
        )
        second = await runtime.generate(second_intent)
        assert isinstance(second, Succeeded), f"tool-result turn did not succeed: {second!r}"
        final = second.response.content
        assert isinstance(final, TextContent) and final.text.strip(), (
            "the tool round trip produced no final answer"
        )
        record["tool_called"] = call.name
        record.update(_meta_evidence(second.meta))


@pytest.mark.parametrize("row", _DEEPSEEK_ROWS, ids=_DEEPSEEK_ROW_IDS)
async def test_deepseek_thinking_tool_continuation_probe(
    row: ModelRow, evidence: _EvidenceRecorder
) -> None:
    with evidence.probe(row, "thinking_tool_continuation") as record:
        runtime = _runtime_for(row)
        tool = CanonicalTool(
            name="lookup_temperature",
            description="Return the current temperature for a city, in celsius.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        )
        intent = _intent(
            row,
            user=(
                "Call lookup_temperature exactly once for Paris. After receiving the tool result, "
                "give the final temperature in one terse sentence."
            ),
            reasoning="high",
            tools=(tool,),
        )
        first = await runtime.generate(intent)
        assert isinstance(first, Succeeded), f"thinking tool turn did not succeed: {first!r}"
        continuation = first.response.continuation
        assert isinstance(continuation, Present), "thinking tool turn returned no continuation"
        reasoning = continuation.value.opaque_payload.get("reasoning_content")
        assert isinstance(reasoning, str) and reasoning.strip(), (
            "DeepSeek thinking tool continuation omitted reasoning_content"
        )
        content = first.response.content
        assert isinstance(content, TextContent), (
            f"thinking tool turn decoded {type(content).__name__}"
        )
        assert content.tool_calls, "thinking tool turn made no tool call"
        call = content.tool_calls[0]
        assert call.name == tool.name, f"unexpected tool called: {call.name!r}"

        second = await runtime.generate(
            replace(
                intent,
                messages=(
                    *intent.messages,
                    AssistantMessage(
                        text=content.text,
                        tool_calls=content.tool_calls,
                        continuation=continuation,
                    ),
                    ToolResultMessage(
                        call_id=call.id, output='{"temperature_c": 21}', is_error=False
                    ),
                ),
            )
        )
        assert isinstance(second, Succeeded), (
            f"thinking tool continuation did not succeed: {second!r}"
        )
        final = second.response.content
        assert isinstance(final, TextContent), (
            f"thinking tool continuation decoded {type(final).__name__}"
        )
        assert final.text.strip() and not final.tool_calls, (
            f"thinking tool continuation produced no final answer: {final!r}"
        )
        record["reasoning"] = "high"
        record["tool_called"] = call.name
        record.update(_meta_evidence(second.meta))


class _LiveJsonReply(pydantic.BaseModel):
    ok: bool
    word: str


@pytest.mark.parametrize("row", ROWS, ids=_ROW_IDS)
async def test_json_out_probe(row: ModelRow, evidence: _EvidenceRecorder) -> None:
    with evidence.probe(row, "json_out") as record:
        runtime = _runtime_for(row)
        intent = _intent(
            row,
            user=(
                "Reply with a JSON object with exactly two fields: "
                '"ok" (boolean, true) and "word" (the string "pong").'
            ),
        )
        reply = await runtime.json_out(_LiveJsonReply, intent)
        assert isinstance(reply, StructuredReply), (
            f"json_out did not return a typed reply: {reply!r}"
        )
        assert reply.value.ok is True, f"typed reply carries ok={reply.value.ok!r}"
        assert reply.value.word.strip().lower() == "pong", (
            f"typed reply carries word={reply.value.word!r}"
        )
        record["structured"] = row.structured
        record.update(_meta_evidence(reply.outcome.meta))


@pytest.mark.parametrize("row", ROWS, ids=_ROW_IDS)
async def test_continuation_probe(row: ModelRow, evidence: _EvidenceRecorder) -> None:
    with evidence.probe(row, "continuation") as record:
        runtime = _runtime_for(row)
        level = _replay_reasoning(row)
        if level is None:
            pytest.skip(f"row {row.ref} declares no reasoning level to replay")
        intent = _intent(
            row,
            user="What is 17 * 23? Think it through, then answer with just the number.",
            reasoning=level,
        )
        first = await runtime.generate(intent)
        assert isinstance(first, Succeeded), f"first reasoning turn did not succeed: {first!r}"
        continuation = first.response.continuation
        assert isinstance(continuation, Present), (
            "the first reasoning turn returned no continuation artifact to replay"
        )
        content = first.response.content
        assert isinstance(content, TextContent), (
            f"first reasoning turn decoded {type(content).__name__}"
        )

        second_intent = replace(
            intent,
            messages=(
                *intent.messages,
                AssistantMessage(
                    text=content.text,
                    tool_calls=content.tool_calls,
                    continuation=continuation,
                ),
                UserMessage(
                    blocks=(
                        PromptBlock(
                            text="Add 9 to your previous answer and reply with just the number."
                        ),
                    )
                ),
            ),
        )
        second = await runtime.generate(second_intent)
        assert isinstance(second, Succeeded), f"continuation replay did not succeed: {second!r}"
        record["replayed_level"] = level
        record.update(_meta_evidence(second.meta))
