#!/usr/bin/env python3
"""Audit public-facing docs/README for leaked internal doctrine and stale claim language.

Public docs (README.md, docs/**/*.md) are read by humans deciding whether to use
jaxfne. They must never quote or paraphrase agent-facing instructions (that content
belongs in AGENTS.md/skills/, not human docs), and every value/amplitude/unit claim
must use the public Relative/Absolute vocabulary, not ad hoc words like "calibrated"
or "validated" that don't map to a defined status.

Two independent checks:
  1. Denylist of exact phrases that have leaked from internal doctrine or read as
     comparison/bragging against other projects, checked against public docs.
  2. A source-code check (jaxfne/, scripts/, tests/) for string-concatenation
     identifier obfuscation -- two adjacent string literals joined with ``+`` to
     build a dict key/value at runtime, a pattern with no legitimate purpose
     other than hiding a name from grep-based audits, which this repo's whole
     verification culture depends on. This file's own detection regex is
     excluded from its own scan (see ``_SELF_PATH`` below).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_GLOBS = ("README.md", "CITATION.cff", "docs/**/*.md")
CODE_GLOBS = ("jaxfne/**/*.py", "scripts/**/*.py", "tests/**/*.py")

# Directories under docs/ that are explicitly dated snapshots or archived
# material, not live public-facing pages -- excluded from the language check.
EXEMPT_PREFIXES = (
    "docs/releases/",
    "docs/v047_refactor_audit.md",
    "docs/tutorials_v030/",
    "docs/changelog.md",
)

# Exact-phrase denylist: leaked agent doctrine, or comparison/bragging language.
# Keep this curated and precise -- broad word bans produce false positives on
# legitimate technical prose (e.g. guides/calibration.md is *about* calibration).
LEAKED_INSTRUCTION_PATTERNS = [
    re.compile(r"language to use", re.I),
    re.compile(r"avoid without receipts", re.I),
    re.compile(r"\bprefer:\s*\S.*\bavoid:", re.I | re.S),
    re.compile(r"do not use the (word|term|language)", re.I),
    re.compile(r"\bforbidden (word|term|language)\b", re.I),
    re.compile(r"\bbanned (word|term)\b", re.I),
]

COMPARISON_PATTERNS = [
    re.compile(r"compared to \[?jaxley", re.I),
    re.compile(r"\bunlike jaxley\b", re.I),
    re.compile(r"\bvs\.?\s+jaxley\b", re.I),
    re.compile(r"jaxley (is|builds) .{0,80}(while|whereas) jaxfne", re.I | re.S),
]

NEGATIVE_CLAIM_PATTERNS = [
    re.compile(r"does not claim", re.I),
    re.compile(r"\bnot a?n? ?calibrated\b", re.I),
    re.compile(r"\bnot a?n? ?validated\b", re.I),
    re.compile(r"\bnot an? absolute\b", re.I),
    re.compile(r"\bnot validated physical\b", re.I),
    re.compile(r"\bno substitute for\b", re.I),
]

# Semantic-escalation patterns (relative-quantity grammar): a relative/proxy
# quantity must not be described as though it were a physical measurement, and
# "effective"/"normalized" must not silently imply calibration or physicality.
# Curated to catch escalation, not legitimate technical prose (e.g. a page
# *about* calibration is fine, and "uncalibrated relative to mV" is a
# legitimate qualifier). Each pattern anchors on an UNQUALIFIED physical claim
# about a quantity the grammar declares relative/proxy/normalized/effective.
SEMANTIC_ESCALATION_PATTERNS = [
    # a relative/proxy quantity asserted to be a physical measurement
    re.compile(r"\b(relative|proxy|relative-value|relative value)\b[^.\n]{0,60}\bis a physical (measurement|amplitude|quantity)\b", re.I),
    # "effective" implies empirically calibrated (effective != calibrated)
    re.compile(r"\beffective\b[^.\n]{0,50}\b(therefore|thus|hence|implies)\b[^.\n]{0,30}\bcalibrated\b", re.I),
    # "calibrated" used as an unqualified synonym for "relative"/"normalized"
    re.compile(r"\bcalibrated (relative|proxy|normalized)\b", re.I),
    # "normalized" equated to "relative" as if universal synonyms
    re.compile(r"\bnormalized\b[^.\n]{0,30}\bsame as\b[^.\n]{0,20}\brelative\b", re.I),
    # "physical" applied to an uncalibrated proxy readout
    re.compile(r"\bphysical\b[^.\n]{0,30}\b(proxy|readout)\b[^.\n]{0,20}\b(measurement|amplitude)\b", re.I),
]

# A relative/proxy subject asserted to be "measured in" an absolute amplitude
# unit, without an in-sentence qualifier/caveat. Sentence-aware (handles a
# qualifier placed either before or after the unit).
_RELATIVE_WORD = re.compile(r"\b(relative|proxy|relative-value|relative value)\b", re.I)
_UNIT = re.compile(r"\bmeasured in\b[^.\n]{0,30}\b(µV|uV|mV|μA|nA|pA)\b", re.I)
_QUALIFIER = re.compile(r"\b(uncalibrated|not physical|caveat)\b", re.I)


def _find_measured_in_escalation(text: str) -> list[dict]:
    hits = []
    for m in _UNIT.finditer(text):
        s = text.rfind(".", 0, m.start())
        e = text.find(".", m.end())
        sentence = text[s + 1: e if e != -1 else len(text)]
        has_relative = bool(_RELATIVE_WORD.search(sentence))
        has_qualifier = bool(_QUALIFIER.search(sentence))
        if has_relative and not has_qualifier:
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append({"line": line_no, "match": m.group(0)[:80]})
    return hits

# String-concatenation identifier obfuscation: two adjacent quoted string
# literals joined with `+`, used to build a dict key/value at runtime instead
# of a plain literal. Legitimate f-strings/format calls don't match this shape.
OBFUSCATED_IDENTIFIER = re.compile(r'"[A-Za-z_]+"\s*\+\s*"[A-Za-z_]+"')

# This file's own source path -- excluded from the obfuscation scan so any
# future example strings added here (for documentation purposes) can't
# self-trigger a false positive.
_SELF_PATH = "scripts/audit_public_docs_language.py"


def is_exempt(rel_path: str) -> bool:
    return any(rel_path.startswith(prefix) for prefix in EXEMPT_PREFIXES)


def audit_docs() -> list[dict]:
    results = []
    paths = sorted({p for glob in DOC_GLOBS for p in ROOT.glob(glob)})
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if is_exempt(rel):
            continue
        text = path.read_text(encoding="utf-8")
        hits = []
        for label, patterns in (
            ("leaked_instruction", LEAKED_INSTRUCTION_PATTERNS),
            ("comparison_bragging", COMPARISON_PATTERNS),
            ("negative_claim", NEGATIVE_CLAIM_PATTERNS),
            ("semantic_escalation", SEMANTIC_ESCALATION_PATTERNS),
        ):
            for pat in patterns:
                for m in pat.finditer(text):
                    line_no = text.count("\n", 0, m.start()) + 1
                    hits.append({"category": label, "pattern": pat.pattern, "line": line_no, "match": m.group(0)[:80]})
        for h in _find_measured_in_escalation(text):
            hits.append({"category": "semantic_escalation",
                         "pattern": "measured-in-unit (sentence-aware)", **h})
        if hits:
            results.append({"path": rel, "hits": hits})
    return results


def audit_obfuscated_identifiers() -> list[dict]:
    results = []
    paths = sorted({p for glob in CODE_GLOBS for p in ROOT.glob(glob)})
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if rel == _SELF_PATH:
            continue
        text = path.read_text(encoding="utf-8")
        hits = []
        for m in OBFUSCATED_IDENTIFIER.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append({"line": line_no, "match": m.group(0)[:80]})
        if hits:
            results.append({"path": rel, "hits": hits})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit nonzero on violations")
    args = parser.parse_args(argv)

    doc_violations = audit_docs()
    code_violations = audit_obfuscated_identifiers()

    summary = {
        "schema_version": "jaxfne.public_docs_language_audit.v0.1.0",
        "doc_files_with_violations": len(doc_violations),
        "code_files_with_obfuscated_identifiers": len(code_violations),
        "doc_violations": doc_violations,
        "obfuscated_identifier_violations": code_violations,
        "pass": not doc_violations and not code_violations,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.check and not summary["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
