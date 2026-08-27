#!/usr/bin/env python3
"""Vocabulary audit for jaxfne.

Scans repository release-facing text (docs, README, docstrings, manifests,
scripts, tests) and reports:

  1. vocabulary statistics (total words, unique words, cluster frequencies);
  2. occurrences of discouraged/meta words by file and line;
  3. candidate synonym clusters (words the project may be using interchangeably);
  4. inconsistent terms for the same concept.

This is an AUDIT tool, not a rewriter. It does not replace text. Replacement is
context-aware and performed by a human/agent per the preferred-word map
(PREFERRED_WORDS). It supports an allowlist of mathematically/scientifically
necessary uses (PROTECTED_TERMS).

Usage:
    python3 scripts/audit_vocabulary.py                 # scan + report
    python3 scripts/audit_vocabulary.py --clusters      # list cluster words
    python3 scripts/audit_vocabulary.py --check         # exit 1 if discouraged words found outside allowlist
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Glob sources scanned. Docstrings and comments in package/tests/scripts are
# included so vocabulary stays consistent across release-facing surfaces.
SOURCES = (
    "README.md",
    "CITATION.cff",
    "docs/**/*.md",
    "jaxfne/**/*.py",
    "scripts/**/*.py",
    "tests/**/*.py",
)

# Docs-only sources used by --check: release-facing prose. Code identifiers,
# file names, and docstrings are excluded from the hard gate because they must
# not be renamed before freeze.
DOC_SOURCES = (
    "README.md",
    "CITATION.cff",
    "docs/**/*.md",
)

# Exempt: dated snapshots / archived / frozen evidence / generated material not
# live release-facing prose. docs/publication/, docs/etudes/, docs/doctrine/
# protocol_*.md, and docs/protocols/ are frozen evidence surfaces; docs/releases/
# are dated snapshots; inventory records are dated planning documents (developer
# roadmap archived to artifacts/developer/fullroadmap.md, gitignored).
EXEMPT_PREFIXES = (
    "docs/releases/",
    "docs/v047_refactor_audit.md",
    "docs/tutorials_v030/",
    "docs/changelog.md",
    "docs/_generated/",
    "docs/evidence_artifacts/",
    "docs/publication/",
    "docs/etudes/",
    "docs/doctrine/protocol_",
    "docs/protocols/",
    "docs/doctrine/rbs_rbd_hdp_inventory.md",
    "docs/HDP_REPORT.md",
)

# Protected scientific/technical terms that must never be merged away. These
# carry distinct, non-interchangeable meaning in jaxfne.
PROTECTED_TERMS = (
    # scientific quantities / operators
    "tfne", "emitter", "source", "field", "probe", "tensor", "geometry",
    "izhikevich", "plasticity", "adaptation", "habituation", "pseudogenome",
    "agsdr", "gsdr", "psd", "csd", "lfp", "eeg", "meg", "gradient", "jacobian",
    "calibration", "dimensional", "relative", "normalized", "effective",
    "base", "rbd", "hdp", "rbs",
    # mathematical concepts
    "invariant", "parameter", "variable", "equation", "set", "map",
    "function", "bound", "domain",
    # evidence levels
    "evidence", "test", "result", "validation", "verification",
    # provenance / schema
    "provenance", "schema",
    # NOTE: "protocol" is intentionally NOT protected here. It is allowed only
    # where it literally names a fixed scientific experimental protocol or an
    # existing identifier/file (see ALLOW_PROTOCOL_PREFIXES). Ordinary prose
    # uses "rule"/"condition"/"steps" instead.
)

# Synonym clusters to examine for redundant meta-language. Words in the same
# group are NOT automatically synonyms — the audit flags co-occurrence so a
# human can decide per context. The preferred map (PREFERRED_WORDS) states the
# intended single word when a distinction is unnecessary.
CLUSTERS = {
    "rule": ["contract", "protocol", "doctrine", "policy", "convention",
             "specification", "spec"],
    "condition": ["criterion", "requirement", "prerequisite", "condition"],
    "limit": ["restriction", "constraint", "limit", "bound", "boundary"],
    "types": ["taxonomy", "ontology", "classification", "categorization"],
    "system": ["framework", "system", "model"],
    "steps": ["pipeline", "workflow", "procedure", "protocol", "path", "steps"],
    "function": ["capability", "functionality", "feature", "function", "use"],
    "value": ["magnitude", "degree", "extent", "scope", "range", "value"],
    "state": ["state", "status", "regime", "condition"],
    "evidence": ["validation", "verification", "evidence", "test"],
}

# Preferred word per concept (derived from actual usage). Protected terms are
# retained; only pure meta-language is consolidated. This is the authoritative
# map for context-aware rewriting.
PREFERRED_WORDS = {
    # keep scientific distinctions; do not collapse
    "invariant": "invariant",  # mathematical
    "schema": "schema",        # JSON/data schema
    "mechanism": "mechanism",  # synaptic mechanism
    "provenance": "provenance",# metadata identity
    "evidence": "evidence",    # evidence level (distinct from test)
    "test": "test",            # executable check
    "validation": "validation",# check against spec
    "verification": "verification",  # check of implementation
    # consolidate pure meta-language where the distinction is unnecessary
    "criterion": "condition",
    "prerequisite": "condition",
    "taxonomy": "types",
    "ontology": "types",
    "capability": "function",
    "functionality": "function",
    "feature": "function",
    # meta-language reduction per the final-closure goal
    "contract": "rule",
    "doctrine": "rule",
    "framework": "model",
    "protocol": "steps",  # unless a fixed experiment name (allowlisted)
}

# Fixed scientific experiment / file / identifier names that legitimately keep
# "protocol". Ordinary prose "protocol" uses "steps"/"rule".
ALLOW_PROTOCOL_PREFIXES = (
    "protocol_", "protocols/", "docs/protocols/", "Protocol C", "Protocol D",
    "Protocol E", "Protocol H", "Protocol W", "Protocol B",
)

# Words to flag in --check mode (discouraged meta-language that has a preferred
# replacement, unless protected). Only pure-meta, non-scientific words with no
# API-parameter or mathematical-name collision. "criterion"/"feature"/
# "capability"/"ontology"/"taxonomy"/"prerequisite" are deliberately NOT in the
# hard --check list because they are legitimately used as an Objective.gate()
# parameter ("criterion"), as a scientific signal/statistical term ("feature",
# "feature dimensionality"), or as structural metadata labels ("operator
# ontology", "tutorial taxonomy"). The audit still REPORTS their counts for
# human review (see --clusters and the JSON report).
DISCOURAGED = (
    "functionality", "contract", "doctrine", "framework", "protocol",
)

# Exact phrases that legitimately use a discouraged word and must not be
# flagged (fixed technical phrases, not generalizable meta-language).
ALLOW_PHRASES = (
    "must not delete functionality",
    # the frozen release artifact and its file name
    "public surface contract",
    "public api contract",
    "public export contract",
    "frozen public export contract",
    # operator_doctrine.md defines the seven-stage "contract table" term
    "per-stage contract table",
    "per-operator domain/codomain contract table",
    "per-stage contract",
    "per-operator contract",
    "contract table",
    # established package technical terms (data-interface conventions)
    "metadata contract",
    "source contract",
    "trace contract",
    "shape contract",
    "solver contract",
    "probe operator contract",
    "probe contract",
    "output contract",
    "status contract",
    "recurrent continuation contract",
    "continuation contract",
    "paradigm contract",
    "computation contract",
    "mathematical contract",
    "reduction contract",
    "analytic contract",
    "discrete euler contract",
    "schema/contract",
    "contract description",
    "contract objects",
    "contract json files",
    "shadow contract",
    "scientific contract",
    # public-surface artifact shorthand
    "public contract",
    "compatibility contract",
    "0.4.13 contract",
    "0.2.5 contract",
    # source-schema stable data contract
    "stable contract",
    # package API terms (Izhikevich paradigm contract, JAX pytree protocol)
    "paradigm contract",
    "paradigm` contract",
    "vertical-slice` contract",
    "vertical-slice** contract",
    "pytree protocol",
    # baseline-drive reference fixed iteration procedure
    "iteration protocol",
    # protocol-ID table column headers (cells name fixed experiments)
    "| test | claim | protocol |",
    # operator_doctrine.md scope line and its own title
    "per-stage operator contract",
    "tfne operator doctrine",
    "operator doctrine",
    # tutorial output-link description
    "8-operator contract",
    # heading category label in references
    "framework-adjacent",
    # tutorial headings/terminology that name the standard working model
    "mathematical framework",
    "standard framework terminology",
    # bibliography titles (verbatim, cannot be rephrased)
    "unifying framework",
)


def iter_source_files() -> list[Path]:
    out: set[Path] = set()
    for glob in SOURCES:
        for p in ROOT.glob(glob):
            out.add(p)
    return sorted(out)


def iter_doc_files() -> list[Path]:
    out: set[Path] = set()
    for glob in DOC_SOURCES:
        for p in ROOT.glob(glob):
            out.add(p)
    return sorted(out)


def is_exempt(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix in EXEMPT_PREFIXES)


def scan(*, docs_only: bool = False) -> dict:
    total_words = Counter()
    cluster_hits = Counter()
    discouraged_hits = Counter()
    occurrence: dict[str, list[tuple[str, int]]] = {}
    word_re = re.compile(r"[A-Za-z][A-Za-z'-]*")
    cluster_pat = re.compile(
        r"\b(" + "|".join(sorted({w for grp in CLUSTERS.values() for w in grp})) + r")\b",
        re.I,
    )
    disc_pat = re.compile(
        r"\b(" + "|".join(DISCOURAGED) + r")\b", re.I
    )
    allow_pat = re.compile(
        "|".join(re.escape(p) for p in ALLOW_PHRASES), re.I
    )

    files = iter_doc_files() if docs_only else iter_source_files()
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if is_exempt(rel):
            continue
        if rel == "scripts/audit_vocabulary.py":
            continue  # the audit's own word lists must not flag themselves
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in word_re.finditer(text):
            total_words[m.group(0).lower()] += 1
        for m in cluster_pat.finditer(text):
            w = m.group(1).lower()
            cluster_hits[w] += 1
            occurrence.setdefault(w, []).append((rel, text.count("\n", 0, m.start()) + 1))
        # mask spans that are identifiers, not prose (docs-only): markdown
        # links `[text](dest)`, inline code `` `...` ``, and .md path tokens
        masked_spans: list[tuple[int, int]] = []
        if docs_only:
            masked_spans += [(mm.start(), mm.end()) for mm in re.finditer(r"``[^`]*``", text)]
            masked_spans += [(mm.start(), mm.end()) for mm in re.finditer(r"`[^`]*`", text)]
            masked_spans += [(mm.start(), mm.end()) for mm in re.finditer(r"\[[^\]]*\]\([^)]*\)", text)]
            masked_spans += [
                (mm.start(), mm.end())
                for mm in re.finditer(r"[A-Za-z0-9_.\-/]*\.[A-Za-z0-9_.\-/]*\.md\b", text, re.I)
            ]
        for m in disc_pat.finditer(text):
            w = m.group(1).lower()
            # skip discouraged words inside an allowed fixed phrase
            in_allow = any(
                ap.start() <= m.start() and ap.end() >= m.end()
                for ap in allow_pat.finditer(text)
            )
            # skip masked identifier spans (links, code, .md path tokens)
            in_link_or_code = any(
                s <= m.start() and e >= m.end() for s, e in masked_spans
            )
            # "protocol" is allowed when it names a fixed scientific experiment
            # or an existing identifier/file (protocol_C, Protocol D, ...)
            if w == "protocol":
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end = text.find("\n", m.end())
                line = text[line_start: line_end if line_end != -1 else len(text)]
                if any(pre.lower() in line.lower() for pre in ALLOW_PROTOCOL_PREFIXES):
                    continue
            if not in_allow and not in_link_or_code:
                discouraged_hits[w] += 1

    protected = set(PROTECTED_TERMS)
    reportable = {w: c for w, c in cluster_hits.items() if w not in protected}
    return {
        "total_words": sum(total_words.values()),
        "unique_words": len(total_words),
        "cluster_hits": dict(cluster_hits),
        "discouraged_hits": dict(discouraged_hits),
        "reportable_cluster_hits": reportable,
        "occurrences": {w: occ[:200] for w, occ in occurrence.items() if w in reportable},
        "protected_terms": sorted(protected),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    if args.clusters:
        print(json.dumps({k: v for k, v in CLUSTERS.items()}, indent=2))
        return 0

    data = scan(docs_only=args.check)
    if args.check:
        bad = data["discouraged_hits"]
        if bad:
            print(f"discouraged words outside allowlist: {dict(bad)}")
            return 1
        print("vocabulary check: pass")
        return 0

    print(json.dumps({
        "schema": "jaxfne.vocabulary_audit.v0.1.0",
        "total_words": data["total_words"],
        "unique_words": data["unique_words"],
        "reportable_cluster_hits": data["reportable_cluster_hits"],
        "discouraged_hits": data["discouraged_hits"],
        "protected_terms": data["protected_terms"],
        "occurrences": data["occurrences"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
