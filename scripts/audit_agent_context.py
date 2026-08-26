#!/usr/bin/env python3
"""Audit active repository context for drift-prone governance patterns.

The audit is read-only. It checks the repository-owned context surface; global
user-level mirrors are handled by ``skills/SYNC_GLOBAL.sh``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SET_ROOT = ROOT / "artifacts" / "project_sources"
EXPECTED_SOURCE_SET = (
    "1_global_rules_and_restrictions.md",
    "2_jaxfne_objective_grammar.md",
    "3_jaxfne_visualization_rules.md",
    "4_tfne_theory_and_neural_tensor.md",
    "5_docs_tutorials_etudes_and_suites.md",
    "6_other_important_notes.md",
)
OBSOLETE_GRAMMAR = re.compile(r"Config\s*(?:->|→)\s*Net")
VOLATILE_FACT = re.compile(
    r"(?:20\d{2}-\d{2}-\d{2}|"
    r"(?:version|release|current|verified|tested|added|moved|built|fixed|commit)"
    r".{0,60}(?:v?0\.\d+(?:\.\d+)?|[0-9a-f]{7,}|\b\d+ passed\b))",
    re.IGNORECASE,
)
RETIRED_STATUS_KEYS = (
    "truth_mode",
    "truth_safe_unverified",
    "laminar_proxy_no_pde",
    "proxy_readout_only",
    "physical_amplitude_claim_allowed",
)


def _active_files() -> list[Path]:
    paths = [ROOT / "artifacts" / "AGENTS.md", ROOT / "artifacts" / "skills" / "README.md"]
    paths.extend(
        sorted(
            path
            for path in (ROOT / "artifacts" / "skills").glob("*.md")
            if path.name not in {"PATCH.md", "ANTIGRAVITY_PROMPT.md", "FRICTIONS_STACK.md"}
        )
    )
    paths.extend(sorted((ROOT / "artifacts" / "skills").glob("*/SKILL.md")))
    return [path for path in paths if path.is_file()]


def build_report() -> dict:
    files = _active_files()
    contents = {str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in files}
    old_grammar_hits = {
        path: [line for line in text.splitlines() if OBSOLETE_GRAMMAR.search(line)]
        for path, text in contents.items()
    }
    old_grammar_hits = {path: lines for path, lines in old_grammar_hits.items() if lines}
    volatile_hits = {
        path: [line for line in text.splitlines() if VOLATILE_FACT.search(line)]
        for path, text in contents.items()
    }
    volatile_hits = {path: lines for path, lines in volatile_hits.items() if lines}
    retired_hits = {
        path: [
            key
            for key in RETIRED_STATUS_KEYS
            if key in text
        ]
        for path, text in contents.items()
    }
    retired_hits = {path: keys for path, keys in retired_hits.items() if keys}
    source_locations = {
        name: sorted(
            str(path.relative_to(ROOT))
            for path in SOURCE_SET_ROOT.glob(name)
            if path.is_file()
        )
        for name in EXPECTED_SOURCE_SET
    }
    source_set_missing = [
        name for name, locations in source_locations.items() if not locations
    ]

    checks = {
        "current_grammars": (
            "Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation"
            in contents.get("artifacts/AGENTS.md", "")
            and "CircuitSpec -> construct -> Model -> simulate -> Signals"
            in contents.get("artifacts/AGENTS.md", "")
        ),
        "obsolete_grammar_absent": not old_grammar_hits,
        "volatile_facts_absent": not volatile_hits,
        "retired_status_keys_absent": not retired_hits,
        "canonical_harden_skill": (ROOT / "artifacts/skills/jaxfne-harden/SKILL.md").is_file(),
        "mirror_direction_declared": (
            "canonical editable skill source" in contents.get("artifacts/skills/README.md", "").lower()
            and "--check" in (ROOT / "artifacts/skills/SYNC_GLOBAL.sh").read_text(encoding="utf-8")
            and "--apply" in (ROOT / "artifacts/skills/SYNC_GLOBAL.sh").read_text(encoding="utf-8")
        ),
        "semantic_delta_report_declared": (
            "API delta:" in contents.get("artifacts/AGENTS.md", "")
            and "Mathematical delta:" in contents.get("artifacts/AGENTS.md", "")
            and "Compatibility delta:" in contents.get("artifacts/AGENTS.md", "")
        ),
        "archived_context_marked": (
            "ARCHIVAL ONLY"
            in (ROOT / "artifacts/legacy/internal_docs/agent_context/claude/CLAUDE.md").read_text(
                encoding="utf-8"
            )
        ),
        "revised_source_set_present": not source_set_missing,
    }
    return {
        "schema_version": "jaxfne.agent_context_audit.v1",
        "active_file_count": len(files),
        "checks": checks,
        "source_locations": source_locations,
        "hits": {
            "obsolete_grammar": old_grammar_hits,
            "volatile_facts": volatile_hits,
            "retired_status_keys": retired_hits,
        },
        "pass": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(args.check and not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
