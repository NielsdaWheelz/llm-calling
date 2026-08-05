"""Repo-hygiene negative gates over ``src/provider_runtime`` (spec §14).

These are source-scan invariants for things that must never come back: provider
SDK substrates, fallback policy, JSON repair, sampling knobs, hidden mutation,
stream-event rewriting, retry-schedule literals outside the planner, and
terminal costing that re-reads the catalog.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

from provider_runtime import (
    CATALOG,
    Dynamic,
    FinalizedProviderCall,
    GenerateIntent,
    GlobalScope,
    OperatorCertified,
    PromptBlock,
    Stable,
    SystemMessage,
    TextOutput,
    UserMessage,
    plan_generate,
)
from provider_runtime.agent_runtime.types import AGENT_ROUTES
from provider_runtime.catalog import Catalog, OpenRouterPrefixContract

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "provider_runtime"

CODEC_MODULES = ("openai", "anthropic", "gemini", "moonshot", "openrouter")


@dataclasses.dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    text: str

    def __str__(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.text}"


def _src_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _scan(pattern: str, files: list[Path] | None = None) -> list[Hit]:
    regex = re.compile(pattern)
    hits: list[Hit] = []
    for path in files if files is not None else _src_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if regex.search(line):
                hits.append(Hit(path=path, line=line_number, text=line.strip()))
    return hits


def _fmt(hits: list[Hit]) -> str:
    return "\n".join(f"  - {hit}" for hit in hits)


# ---------------------------------------------------------------------------
# Substrates and legacy names


def test_no_raw_provider_sdk_imports() -> None:
    pattern = (
        r"^\s*(from|import) (openai|anthropic)\b|"
        r"^\s*from google import genai\b|"
        r"^\s*(from|import) google\.(genai|generativeai)\b|"
        r"^\s*(from|import) pydantic_ai\b"
    )
    hits = _scan(pattern)
    assert not hits, f"raw provider SDK import present:\n{_fmt(hits)}"


# One allowlist per SDK name, never a shared one: a file audited for one vendor's SDK is
# not thereby audited for another's. The scan matches the bare module name anywhere, not
# just an import statement, because the audited adapter resolves its SDK through
# importlib.import_module("...") — a name in a string is a real dependency here.
_AGENT_SDK_ALLOWLIST: dict[str, frozenset[Path]] = {
    # The Claude Agent SDK is the pinned optional extra; only its adapter may name it.
    "claude_agent_sdk": frozenset({SRC / "agent_runtime" / "claude_sdk.py"}),
    # The pinned optional Codex SDK belongs only to its audited adapter.
    "openai_codex": frozenset({SRC / "agent_runtime" / "codex_sdk.py"}),
    # The SDK's public runtime package resolves the exact bundled executable used only by the
    # Codex adapter's environment-replacing launcher.
    "codex_cli_bin": frozenset({SRC / "agent_runtime" / "codex_sdk.py"}),
}


def test_agent_sdk_imports_are_confined_to_audited_adapters() -> None:
    hits: list[Hit] = []
    for module, allowed in _AGENT_SDK_ALLOWLIST.items():
        hits.extend(hit for hit in _scan(rf"\b{module}\b") if hit.path not in allowed)

    assert not hits, f"agent SDK name outside its audited adapter:\n{_fmt(hits)}"


def test_raw_agent_protocol_substrates_stay_deleted() -> None:
    agent_runtime = SRC / "agent_runtime"
    forbidden = (
        "codex_app_server.py",
        "codex_cli.py",
        "claude_cli.py",
        "_jsonrpc.py",
        "_jsonl.py",
    )

    present = [name for name in forbidden if (agent_runtime / name).exists()]
    assert not present, f"deleted raw agent substrates returned: {present}"
    assert AGENT_ROUTES == frozenset({("codex", "sdk"), ("claude", "sdk")})


# The package ships exactly two transports, one per backend. The gate below is what keeps
# that a fact about the code rather than a claim in a doc: a third lane added later joins
# this tuple, and until it does no file here may reach into another adapter at all.
_AGENT_ADAPTER_MODULES = (
    "codex_sdk",
    "claude_sdk",
)


def test_no_agent_adapter_imports_another_agent_adapter() -> None:
    """Each transport is an independent lane; one importing another fuses their lifecycles.

    `ClaudeSdkAdapter` once read the installed Claude Code version by constructing the CLI
    adapter and asking it for capabilities, which spawned `claude --version` *and*
    `claude auth status --json` and enforced that lane's version pin on this one. Two lanes
    that answer to different process lifecycles must not be able to do that to each other,
    whichever two they are: a shared backend fact belongs to the module that owns it, or to
    a helper both import, never to a sibling adapter.
    """
    agent_runtime = SRC / "agent_runtime"
    hits: list[Hit] = []
    for name in _AGENT_ADAPTER_MODULES:
        path = agent_runtime / f"{name}.py"
        siblings = "|".join(other for other in _AGENT_ADAPTER_MODULES if other != name)
        hits.extend(
            _scan(
                rf"^\s*(from|import)\s+\.?({siblings})\b"
                rf"|^\s*from\s+[\w.]*agent_runtime\.({siblings})\b"
                rf"|import_module\(\s*[\"'][\w.]*({siblings})[\"']",
                [path],
            )
        )
    assert not hits, f"agent adapter imports a sibling adapter:\n{_fmt(hits)}"


def test_release_certification_covers_every_shipped_route() -> None:
    """The live matrix must certify the whole route algebra, and this must be checkable in CI.

    The gate itself lives in ``tests/live/test_agent_matrix.py``, but that whole module carries
    the ``live_provider`` mark and ``addopts`` deselects it, so the one assertion in it that
    needs no provider — a pure in-process comparison of the matrix's route table against
    ``AGENT_ROUTES`` — could never fail in CI either. A release gate that cannot fail is not a
    gate, which is the exact defect class this file exists to prevent. Importing the module
    rather than restating its tables keeps a single owner: if a route is added to the algebra
    and the matrix is not taught to certify it, this fails deterministically.
    """
    matrix_path = ROOT / "tests" / "live" / "test_agent_matrix.py"
    spec = importlib.util.spec_from_file_location("_live_agent_matrix_gate", matrix_path)
    assert spec is not None and spec.loader is not None
    matrix = importlib.util.module_from_spec(spec)
    # dataclass() resolves its module from sys.modules while the body executes, so the
    # registration has to precede exec_module rather than follow it.
    sys.modules[spec.name] = matrix
    try:
        spec.loader.exec_module(matrix)
    finally:
        del sys.modules[spec.name]

    routes = matrix._ROUTES
    assert matrix._DEFAULT_ROUTES == frozenset(routes), (
        "release certification omits a shipped route by default"
    )
    assert {(route.backend, route.transport) for route in routes} == AGENT_ROUTES, (
        "the live route table drifted from the package's closed routing table"
    )


def test_no_legacy_llm_calling_name() -> None:
    hits = _scan(r"\bllm_calling\b")
    assert not hits, f"legacy llm_calling name present:\n{_fmt(hits)}"


def test_no_stateful_response_cursor() -> None:
    hits = _scan(r"\bprevious_response_id\b")
    assert not hits, f"stateful response cursor present:\n{_fmt(hits)}"


def test_no_json_repair() -> None:
    hits = _scan(r"json[_-]repair")
    assert not hits, f"JSON repair present:\n{_fmt(hits)}"
    assert "json-repair" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fallback: the only permitted mentions REJECT fallback


_ALLOWED_FALLBACK_LINE = re.compile(
    r"^\s*#"  # comment
    r"|\"allow_fallbacks\": False"  # openrouter routing block: fallbacks OFF
    r"|\"allowProviderModelFallback\": False"  # Codex thread policy: fallback OFF
    r"|fallback_arguments"  # openai decode: accumulated-arguments default, not provider fallback
    r"|fallbacks? (off|of any kind)"  # docstring prose rejecting fallback
)


def test_no_fallback_policy() -> None:
    hits = [hit for hit in _scan(r"(?i)fallback") if not _ALLOWED_FALLBACK_LINE.search(hit.text)]
    assert not hits, f"fallback policy present:\n{_fmt(hits)}"
    forbidden = _scan(r"\"allow_fallbacks\": True|fallback_model|model_fallback|provider_fallback")
    assert not forbidden, f"fallback enabling present:\n{_fmt(forbidden)}"


def test_openrouter_pins_fallbacks_off() -> None:
    assert '"allow_fallbacks": False' in (SRC / "openrouter.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sampling knobs never reappear in any codec body


def test_no_sampling_keys_in_source() -> None:
    hits = _scan(r"\"temperature\"|\"top_p\"|\"top_k\"")
    assert not hits, f"sampling knob present:\n{_fmt(hits)}"


# ---------------------------------------------------------------------------
# Hidden mutation and stream-event rewriting


def test_no_object_setattr() -> None:
    hits = _scan(r"object\.__setattr__")
    assert not hits, f"hidden mutation via object.__setattr__ present:\n{_fmt(hits)}"


def test_codecs_never_use_dataclasses_replace() -> None:
    files = [SRC / f"{name}.py" for name in CODEC_MODULES]
    files.append(SRC / "_chat_completions_wire.py")
    hits = _scan(r"\breplace\(", files)
    assert not hits, f"replace() in a codec module:\n{_fmt(hits)}"


def test_runtime_replace_targets_are_outcomes_never_stream_events() -> None:
    # runtime.py may rebuild terminal outcomes/meta (attempt-trace injection and
    # StructuredContent promotion) but must never replace() a stream-event value
    # (RuntimeStreamEvent/CodecStreamEvent are constructed once, seq stamped once).
    allowed_first_args = {
        "outcome",
        "outcome.meta",
        "outcome.response",
        "event.outcome",
        "event.outcome.meta",
    }
    text = (SRC / "runtime.py").read_text(encoding="utf-8")
    first_args = re.findall(r"\breplace\(\s*([A-Za-z_][\w.]*)", text)
    assert first_args, "expected the known outcome-rebuild replace() call sites in runtime.py"
    unexpected = [arg for arg in first_args if arg not in allowed_first_args]
    assert not unexpected, (
        f"replace() in runtime.py must target terminal outcome values only; got {unexpected}"
    )


# ---------------------------------------------------------------------------
# Ownership boundaries


def test_usage_module_never_imports_catalog_or_planning() -> None:
    hits = _scan(
        r"^\s*(from|import) provider_runtime\.(catalog|planning)\b|"
        r"^\s*from provider_runtime import .*\b(catalog|planning)\b",
        [SRC / "usage.py"],
    )
    assert not hits, f"terminal costing reads plan-time contracts:\n{_fmt(hits)}"


def test_retry_policy_literals_only_in_planning() -> None:
    files = [path for path in _src_files() if path.name != "planning.py"]
    hits = _scan(r"\bRetryPolicy\(", files)
    assert not hits, f"RetryPolicy constructed outside planning.py:\n{_fmt(hits)}"


def test_moonshot_and_openrouter_do_not_import_each_other() -> None:
    moonshot_text = (SRC / "moonshot.py").read_text(encoding="utf-8")
    openrouter_text = (SRC / "openrouter.py").read_text(encoding="utf-8")
    assert not re.search(
        r"^\s*(from|import) provider_runtime[. ].*\bopenrouter\b", moonshot_text, re.M
    )
    assert not re.search(
        r"^\s*(from|import) provider_runtime[. ].*\bmoonshot\b", openrouter_text, re.M
    )


# ---------------------------------------------------------------------------
# Codec seam completeness


def test_every_codec_defines_the_full_seam() -> None:
    for name in CODEC_MODULES:
        module = importlib.import_module(f"provider_runtime.{name}")
        assert isinstance(getattr(module, "CODEC_ID", None), str), f"{name}: CODEC_ID missing"
        for seam_function in (
            "encode",
            "finalize",
            "stream_request",
            "decode_response",
            "decode_stream",
            "classify_error",
        ):
            assert callable(getattr(module, seam_function, None)), (
                f"{name}: seam function {seam_function} missing"
            )


def test_openai_encode_sends_store_false() -> None:
    assert '"store": False' in (SRC / "openai.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Non-stream plans never carry the stream flag (streaming is stream_request's
# derivation, never intent state); exact bodies are covered by plan goldens.


def _minimal_intent(row_index: int) -> GenerateIntent:
    contract = CATALOG.chat[row_index]
    return GenerateIntent(
        target=contract.target,
        messages=(
            SystemMessage(
                blocks=(PromptBlock(text="Stable gate prefix.", stability=Stable(GlobalScope())),)
            ),
            UserMessage(blocks=(PromptBlock(text="Say ok.", stability=Dynamic()),)),
        ),
        max_output_tokens=16,
        reasoning=contract.reasoning.levels[0],
        tools=(),
        tool_choice="auto",
        output=TextOutput(),
    )


def test_planned_non_stream_bodies_have_no_stream_flag() -> None:
    for row_index, contract in enumerate(CATALOG.chat):
        if contract.protocol == "openrouter_chat":
            # OperatorUncertified in CATALOG (unusable by design); certify a copy.
            assert isinstance(contract.cache, OpenRouterPrefixContract)
            certified = dataclasses.replace(
                contract,
                certification=OperatorCertified(
                    certified_pinned_upstream=contract.cache.pinned_upstream,
                    certified_canonical_revision=contract.cache.canonical_revision,
                    evidence_revision="gate-probe",
                ),
            )
            catalog = Catalog(chat=(certified,), embeddings=(), transcriptions=())
            plan = plan_generate(_minimal_intent(row_index), catalog)
        else:
            plan = plan_generate(_minimal_intent(row_index))
        assert isinstance(plan, FinalizedProviderCall)
        body = json.loads(plan.request.body)
        assert "stream" not in body, f"{contract.protocol}: non-stream body carries stream flag"
