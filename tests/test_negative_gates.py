from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    text: str

    def __str__(self) -> str:
        rel = self.path.relative_to(ROOT)
        return f"{rel}:{self.line}: {self.text}"


def _scan(pattern: str) -> list[Hit]:
    regex = re.compile(pattern)
    hits: list[Hit] = []
    for path in sorted(SRC.rglob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if regex.search(line):
                hits.append(Hit(path=path, line=line_number, text=line.strip()))
    return hits


def _fmt(hits: list[Hit]) -> str:
    return "\n".join(f"  - {hit}" for hit in hits)


def test_legacy_llm_calling_import_path_is_absent() -> None:
    hits = _scan(r"\bllm_calling\b")
    assert not hits, f"legacy llm_calling package path present:\n{_fmt(hits)}"


def test_provider_sdk_substrate_imports_are_absent() -> None:
    pattern = (
        r"^(from|import) (openai|anthropic)\b|"
        r"^from google import genai\b|"
        r"^(from|import) google\.(genai|generativeai)\b|"
        r"^from pydantic_ai\.(models|providers)\b|"
        r"^import pydantic_ai\.(models|providers)\b"
    )
    hits = _scan(pattern)
    assert not hits, f"provider SDK substrate import present in provider_runtime:\n{_fmt(hits)}"


def test_response_cursor_and_cross_provider_fallback_policy_are_absent() -> None:
    pattern = (
        r"\bprevious_response_id\b|"
        r"cross[-_ ]?(model|provider).*fallback|"
        r"fallback_(model|provider)|"
        r"(model|provider)_fallback"
    )
    hits = _scan(pattern)
    assert not hits, (
        f"stateful response cursor or provider/model fallback policy present:\n{_fmt(hits)}"
    )
