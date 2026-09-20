#!/usr/bin/env python3
"""Semantic-negative docs gate (Batch B7): fail on forbidden conflations.

Each pattern states a positive conflation the TFNE programme eliminated.
A match passes only if the same line carries an explicit negation marker
(not/never/n't/!=/≠/no/without/refused/inert/limitation/must not/cannot),
so doctrine pages that state the limitation keep passing.

Usage:
    python scripts/audit_semantic_negatives.py --check   # exit 1 on violation
    python scripts/audit_semantic_negatives.py           # human-readable report
"""

from __future__ import annotations

import argparse
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

PATTERNS = [
    ("delay-supported-by-tfne",
     r"delays?\s+(are\s+|is\s+)?(supported|executed|consumed)\s+(by|in)\s+tfne",
     "delay is refused (E_PARAM_UNSUPPORTED); nothing consumes it"),
    ("declared-geometry-executed",
     r"declared\s+geometry\s+(==|equals?|=)\s*executed|geometry\s+is\s+executed|"
     r"declared\s+range\s+(reaches|determines|controls)\s+(executed\s+)?positions",
     "declared geometry is inert at execution (PARAM-04)"),
    ("gaba-is-gaba-a",
     r"\bGABA\s*(==|equals?|treated\s+as|resolved\s+to)\s*GABA_A\b",
     "GABA is ambiguous, not an alias of GABA_A"),
    ("h-is-hdp",
     r"\bH\s*(==|equals?|is\s+simply|is\s+just)\s*HDP\b",
     "H (RBS container) != HDP (plasticity)"),
    ("proxy-is-physical",
     r"proxy\s*(==|equals?|is\s+simply|is\s+just)\s*(physical|calibrated)|"
     r"proxy\s+(lfp|csd|eeg|meg)\s+is\s+physical",
     "proxy != physical measurement"),
    ("source-order-is-realization-order",
     r"source\s+order\s*(==|equals?|determines?)\s*(realization|realized)\s+order|"
     r"enumeration\s+order\s+(carries|implies)\s+scientific\s+semantics",
     "implicit source order never carries scientific semantics (S20.1)"),
    ("tfne-fills-development",
     r"tfne\s+directly\s+fills\s+unspecified\s+developmental\s+choices|"
     r"tfne\s+realizes\s+(edges|positions)\s+directly\s+without\s+jdna",
     "TFNE constrains; JDNA completes under K_D"),
]

NEGATION = re.compile(
    r"not\b|never\b|n't\b|!=\b|≠|(?<!\w)no\s|without\b|refus|inert\b|"
    r"limitation\b|must\s+not\b|cannot\b|don't\b|doesn't\b|isn't\b|aren't\b",
    re.IGNORECASE,
)

# Files allowed to *discuss* a conflation (they state the rule, positively
# framed, with the refusal named nearby). Format: path -> [pattern ids].
ALLOWLIST = {
    "docs/doctrine/tfne_algebra.md": ["*"],
    "docs/doctrine/rbs_rbd_hdp.md": ["*"],
    "docs/doctrine/tfne_jdna_boundary.md": ["*"],
    "docs/doctrine/tfne_containment_architecture.md": ["*"],
    "docs/doctrine/relative_quantity_grammar.md": ["*"],
    "docs/source_field_equations.md": ["proxy-is-physical"],
    "docs/guides/hdp.md": ["h-is-hdp"],
    "docs/guides/model_inspection.md": ["*"],
    "docs/guides/atlas_suite.md": ["*"],
    "docs/api/neuronal_tensor.md": ["tfne-fills-development"],
}


def audit() -> list[dict]:
    violations = []
    compiled = [(pid, re.compile(rx, re.IGNORECASE), why)
                for pid, rx, why in PATTERNS]
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        allowed = ALLOWLIST.get(rel, [])
        for i, line in enumerate(lines, start=1):
            for pid, rx, why in compiled:
                if pid in allowed or "*" in allowed:
                    continue
                m = rx.search(line)
                if m and not NEGATION.search(line):
                    violations.append({"file": rel, "line": i,
                                       "pattern": pid,
                                       "text": line.strip()[:160],
                                       "rule": why})
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    violations = audit()
    if not violations:
        print(f"semantic negatives: pass ({len(PATTERNS)} patterns, "
              f"{len(ALLOWLIST)} allowlisted files)")
        return 0
    print(f"semantic negatives: {len(violations)} violations")
    for v in violations:
        print(f"  {v['file']}:{v['line']} [{v['pattern']}] {v['text']}")
        print(f"    rule: {v['rule']}")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
