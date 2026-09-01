"""Repo-hygiene negative gates over ``src/provider_runtime`` (spec §11).

Source-scan invariants for things that must never come back or drift: provider
SDK imports outside the engine seam, agent SDK names outside their audited
adapters, retry-policy construction outside the single retry owner, environment
reads in the provider lane, unpinned OpenRouter routing, continuation payloads
in any repr, an unbounded facade, and deleted legacy modules returning.

Gates are cheap and deterministic: AST or line scanning over the checked-out
source plus direct assertions over the real registry rows — no subprocesses,
no network.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import provider_runtime
from provider_runtime.registry import _ROWS as ROWS
from provider_runtime.types import (
    ContinuationArtifact,
    ContinuationDelta,
    Present,
    ProviderTarget,
    RuntimeStreamEvent,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "provider_runtime"
AGENT_RUNTIME = SRC / "agent_runtime"
ENGINES = SRC / "engines"


@dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    text: str

    def __str__(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.text}"


def _src_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _provider_lane_files() -> list[Path]:
    """The provider lane: everything under src/provider_runtime except agent_runtime/."""
    return [path for path in _src_files() if AGENT_RUNTIME not in path.parents]


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


def _source_line(path: Path, line: int) -> str:
    return path.read_text(encoding="utf-8").splitlines()[line - 1].strip()


# ---------------------------------------------------------------------------
# SDK imports are confined to their audited seams


_PROVIDER_SDK_IMPORT = (
    r"^\s*(from|import) (openai|anthropic)\b"
    r"|^\s*from google import genai\b"
    r"|^\s*(from|import) google\.(genai|generativeai)\b"
)


def test_provider_sdk_imports_are_confined_to_engines_and_embeddings() -> None:
    """SDK types never cross the contract boundary (spec §1); the import sites prove it.

    The four protocol adapters under ``engines/`` own all provider SDK usage,
    plus the one spec-sanctioned exception: ``embeddings.py`` is rewritten on
    the ``openai`` SDK (spec §3) and lives outside the engines package.
    """
    allowed_outside_engines = {SRC / "embeddings.py"}
    hits = [
        hit
        for hit in _scan(_PROVIDER_SDK_IMPORT)
        if ENGINES not in hit.path.parents and hit.path not in allowed_outside_engines
    ]
    assert not hits, f"provider SDK import outside engines/ (or embeddings.py):\n{_fmt(hits)}"


# One allowlist per SDK name, never a shared one: a file audited for one vendor's SDK is
# not thereby audited for another's. The scan matches the bare module name anywhere, not
# just an import statement, because an audited adapter may resolve its SDK through
# importlib.import_module("...") — a name in a string is a real dependency here.
_AGENT_SDK_ALLOWLIST: dict[str, frozenset[Path]] = {
    # The Claude Agent SDK is the pinned optional extra; only its adapter may name it.
    "claude_agent_sdk": frozenset({AGENT_RUNTIME / "claude_sdk.py"}),
    # The pinned optional Codex SDK belongs only to its audited adapter.
    "openai_codex": frozenset({AGENT_RUNTIME / "codex_sdk.py"}),
    # The SDK's public runtime package resolves the exact bundled executable used only by
    # the Codex adapter's environment-replacing launcher.
    "codex_cli_bin": frozenset({AGENT_RUNTIME / "codex_sdk.py"}),
}


def test_agent_sdk_names_are_confined_to_their_audited_adapters() -> None:
    hits: list[Hit] = []
    for module, allowed in _AGENT_SDK_ALLOWLIST.items():
        hits.extend(hit for hit in _scan(rf"\b{module}\b") if hit.path not in allowed)
    assert not hits, f"agent SDK name outside its audited adapter:\n{_fmt(hits)}"


# ---------------------------------------------------------------------------
# One retry owner


def _construction_sites(path: Path, type_name: str) -> list[Hit]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[Hit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if called == type_name:
            hits.append(Hit(path=path, line=node.lineno, text=_source_line(path, node.lineno)))
    return hits


def test_retry_policy_is_constructed_only_in_retry() -> None:
    """`RetryPolicy(` has exactly one src construction site: retry.py's DEFAULT_RETRY.

    AST-based so prose mentioning the constructor (types.py's section header)
    never trips the gate — only a real call expression does.
    """
    retry_module = SRC / "retry.py"
    outside: list[Hit] = []
    inside: list[Hit] = []
    for path in _src_files():
        sites = _construction_sites(path, "RetryPolicy")
        (inside if path == retry_module else outside).extend(sites)
    assert not outside, f"RetryPolicy constructed outside retry.py:\n{_fmt(outside)}"
    assert len(inside) == 1, (
        "retry.py must hold exactly one RetryPolicy construction (DEFAULT_RETRY); "
        f"found {len(inside)}:\n{_fmt(inside)}"
    )


# ---------------------------------------------------------------------------
# Zero environment reads in the provider lane


def _environment_reads(path: Path) -> list[Hit]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[Hit] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in {"environ", "environb", "getenv"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ):
            hits.append(Hit(path=path, line=node.lineno, text=_source_line(path, node.lineno)))
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name in {"environ", "environb", "getenv"}:
                    hits.append(
                        Hit(path=path, line=node.lineno, text=_source_line(path, node.lineno))
                    )
    return hits


def test_provider_lane_reads_zero_environment_variables() -> None:
    """Credentials are values on the typed request, never ambient lookups (spec §5).

    Scoped to the provider lane only: ``agent_runtime/`` legitimately reads the
    parent environment — ``build_child_environment`` constructs the scrubbed
    child environment from it — so the agent lane is excluded here and its
    environment contract is enforced by its own tests.
    """
    hits = [hit for path in _provider_lane_files() for hit in _environment_reads(path)]
    assert not hits, f"environment read in the provider lane:\n{_fmt(hits)}"


# ---------------------------------------------------------------------------
# OpenRouter rows stay pinned


def test_every_openrouter_registry_row_pins_routing_with_fallbacks_off() -> None:
    """No unpinned OpenRouter passthrough (spec §7) — checked over the real ROWS."""
    openrouter_rows = [row for row in ROWS if row.provider == "openrouter"]
    assert openrouter_rows, "the registry ships no openrouter row; this gate would be vacuous"
    for row in openrouter_rows:
        assert isinstance(row.routing, Present), (
            f"openrouter row {row.ref!r} carries no routing pins"
        )
        routing = row.routing.value
        assert routing.allow_fallbacks is False, (
            f"openrouter row {row.ref!r} must pin allow_fallbacks=False; "
            f"got {routing.allow_fallbacks!r}"
        )


# ---------------------------------------------------------------------------
# Continuation payloads never appear in repr


def test_continuation_payload_never_appears_in_repr() -> None:
    sentinel = "OPAQUE-CONTINUATION-SENTINEL-b2ff41"
    artifact = ContinuationArtifact(
        target=ProviderTarget(provider="openai", model="gpt-5.6-sol"),
        codec_id="openai.v1",
        opaque_payload={"reasoning_item": sentinel},
    )
    assert sentinel not in repr(artifact), "ContinuationArtifact repr leaks its opaque payload"
    assert sentinel not in str(artifact), "ContinuationArtifact str leaks its opaque payload"

    envelope = RuntimeStreamEvent(seq=1, event=ContinuationDelta(artifact=artifact))
    assert sentinel not in repr(envelope), (
        "RuntimeStreamEvent(ContinuationDelta) repr leaks the artifact's opaque payload"
    )
    assert sentinel not in str(envelope), (
        "RuntimeStreamEvent(ContinuationDelta) str leaks the artifact's opaque payload"
    )


# ---------------------------------------------------------------------------
# Facade stays bounded


def test_facade_all_is_capped_and_every_name_importable() -> None:
    exported = provider_runtime.__all__
    assert len(exported) <= 40, (
        f"provider_runtime.__all__ is capped at 40 names; got {len(exported)}"
    )
    missing = [name for name in exported if not hasattr(provider_runtime, name)]
    assert not missing, f"__all__ names not importable from the package root: {missing}"


# ---------------------------------------------------------------------------
# Deleted modules stay dead


_DELETED_MODULES = (
    "catalog",
    "planning",
    "schema",
    "transport",
    "usage",
    "openai",
    "anthropic",
    "gemini",
    "moonshot",
    "openrouter",
    "_chat_completions_wire",
    "_signals",
)


def test_deleted_legacy_modules_stay_dead() -> None:
    alive = [
        name
        for name in _DELETED_MODULES
        if importlib.util.find_spec(f"provider_runtime.{name}") is not None
    ]
    assert not alive, f"deleted provider_runtime modules are importable again: {alive}"
