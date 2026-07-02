#!/usr/bin/env python3
"""Audit tutorial notebooks for the jaxfne notebook grammar.

Checks are structural and text-based. They keep notebooks as setup/config/run/
objective/export documents while reusable numerical logic stays in package APIs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_GLOBS = ("tutorials/**/*.ipynb",)

SECTION_PATTERNS = {
    "setup": re.compile(r"\b(setup|install|import|runtime|version)\b", re.I),
    "config": re.compile(r"\b(config|configuration|editable|parameters|knobs)\b", re.I),
    "run": re.compile(r"\b(simulat\w*|construct|probe|readout|signals?)\b", re.I),
    "objective": re.compile(r"\b(objective|metric|score|gate|null|loss)\b", re.I),
    "export": re.compile(r"\b(export|manifest|receipt|asset|hash|png|json)\b", re.I),
}

LOCAL_ENGINE_PATTERNS = [
    re.compile(r"def\s+simulate\s*\(", re.I),
    re.compile(r"def\s+solve_.*field", re.I),
    re.compile(r"def\s+project_.*source", re.I),
    re.compile(r"def\s+objective\s*\(", re.I),
    re.compile(r"class\s+.*Solver", re.I),
]

PROXY_OVERCLAIM = re.compile(
    r"\b(real EEG|real MEG|calibrated CSD|solved PDE|Poisson solver|Maxwell solver|mechanism proof)\b",
    re.I,
)


def read_notebook_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in data.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            parts.append("".join(src))
        elif isinstance(src, str):
            parts.append(src)
    return "\n".join(parts)


def audit_one(path: Path) -> dict:
    text = read_notebook_text(path)
    lower = text.lower()
    has_jtfne = "import jaxfne as jtfne" in text
    sections = {name: bool(pattern.search(text)) for name, pattern in SECTION_PATTERNS.items()}
    local_engines = [pat.pattern for pat in LOCAL_ENGINE_PATTERNS if pat.search(text)]
    overclaims = [m.group(0) for m in PROXY_OVERCLAIM.finditer(text)]
    has_simulation = any(token in lower for token in ("simulate", "construct(", "jtfne.simulate", ".simulate"))
    problems: list[str] = []
    if not has_jtfne:
        problems.append("missing canonical import: import jaxfne as jtfne")
    for required in ("setup", "config", "export"):
        if not sections[required]:
            problems.append(f"missing {required} section marker")
    if has_simulation and not sections["run"]:
        problems.append("simulation/probe notebook lacks run section marker")
    if "objective" in lower or "metric" in lower or "score" in lower:
        if not sections["objective"]:
            problems.append("objective/metric notebook lacks objective section marker")
    if local_engines:
        problems.append("contains notebook-local engine definitions")
    if overclaims:
        problems.append("contains proxy-as-physical wording")
    return {
        "path": str(path.relative_to(ROOT)),
        "has_jtfne_import": has_jtfne,
        "sections": sections,
        "local_engine_patterns": local_engines,
        "proxy_overclaims": sorted(set(overclaims)),
        "problems": problems,
        "pass": not problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit nonzero on violations")
    args = parser.parse_args(argv)
    notebooks = sorted({p for glob in NOTEBOOK_GLOBS for p in ROOT.glob(glob)})
    results = [audit_one(path) for path in notebooks]
    summary = {
        "schema_version": "jaxfne.notebook_grammar_audit.v0.1.0",
        "notebook_count": len(results),
        "pass_count": sum(1 for r in results if r["pass"]),
        "fail_count": sum(1 for r in results if not r["pass"]),
        "results": results,
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    if args.check and summary["fail_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
