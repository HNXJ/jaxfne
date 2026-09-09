#!/usr/bin/env python3
"""W3 inventory: broad ``except Exception`` / bare ``except:`` handlers in jaxfne.

Emits ``artifacts/audit/w3_broad_handler_tally.json``. Classification only —
no code changes. Re-run after handler edits to refresh the tally.

Classes (roadmap W3):
  EXPECTED_OPTIONAL_CAPABILITY — narrow in 0.4.23; record unavailable status
  BEST_EFFORT_PRESENTATION     — keep fallback; expose reason in diagnostics
  IMPOSSIBLE_STATE_RAISE       — should raise; fix in 0.4.23+
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "jaxfne"
OUT = ROOT / "artifacts" / "audit" / "w3_broad_handler_tally.json"

HANDLER_RE = re.compile(
    r"^\s*except\s+(Exception|BaseException)(\s+as\s+\w+)?\s*:|^\s*except\s*:"
)

# Line-specific overrides for scientific paths (file, line) -> (class, note)
_OVERRIDES: dict[tuple[str, int], tuple[str, str]] = {
    ("jaxfne/_construct_connectivity.py", 592): (
        "BEST_EFFORT_PRESENTATION",
        "optional recurrent_backend=edge_list upgrade; construction continues",
    ),
    ("jaxfne/_model.py", 593): (
        "BEST_EFFORT_PRESENTATION",
        "neuron_table z position extraction; None when unavailable",
    ),
    ("jaxfne/_runtime_config.py", 307): (
        "BEST_EFFORT_PRESENTATION",
        "jaxlib version probe; returns unknown",
    ),
    ("jaxfne/bridges.py", 473): (
        "EXPECTED_OPTIONAL_CAPABILITY",
        "jaxley morphology position lookup when bridge optional",
    ),
    ("jaxfne/bridges.py", 709): (
        "EXPECTED_OPTIONAL_CAPABILITY",
        "jaxley compartment index enumeration",
    ),
    ("jaxfne/bridges.py", 778): (
        "EXPECTED_OPTIONAL_CAPABILITY",
        "jaxley recording state probe",
    ),
    ("jaxfne/neuronal_tensor.py", 913): (
        "BEST_EFFORT_PRESENTATION",
        "content hash fallback empty; never block construction",
    ),
    ("jaxfne/objectives.py", 481): (
        "SCIENTIFIC_COMPUTATION",
        "synchrony gate removed broad swallow; degenerate stats return 0.0 inside compute_synchrony_metric",
    ),
    ("jaxfne/_model_simulate.py", 592): (
        "EXPECTED_OPTIONAL_CAPABILITY",
        "opt-in simulate accessory must not break canonical simulate",
    ),
    ("jaxfne/_model_simulate.py", 1264): (
        "BEST_EFFORT_PRESENTATION",
        "post-sim diagnostics attachment",
    ),
    ("jaxfne/_model_evaluate.py", 297): (
        "BEST_EFFORT_PRESENTATION",
        "evaluate objective aggregation fallback",
    ),
    ("jaxfne/_model_tune.py", 652): (
        "SCIENTIFIC_COMPUTATION",
        "tune top-level failure returns REVISE + inf; does not fabricate ACCEPT",
    ),
    ("jaxfne/_model_evaluate.py", 297): (
        "SCIENTIFIC_COMPUTATION",
        "group rate evaluation failure sets all_gates_pass=False",
    ),
    ("jaxfne/validation.py", 276): (
        "VALIDATION",
        "eigenvalue failure returns is_valid=False with evidence",
    ),
    ("jaxfne/validation.py", 327): (
        "VALIDATION",
        "field array validation error sets *_finite=False and all_finite=False",
    ),
    ("jaxfne/validation.py", 337): (
        "VALIDATION",
        "field array validation error sets *_finite=False and all_finite=False",
    ),
    ("jaxfne/validation.py", 347): (
        "VALIDATION",
        "field array validation error sets *_finite=False and all_finite=False",
    ),
    ("jaxfne/validation.py", 875): (
        "VALIDATION",
        "SPD check failure returns (False, diagnostic)",
    ),
    ("jaxfne/validation.py", 923): (
        "VALIDATION",
        "conservation check failure returns (False, diagnostic, residual)",
    ),
    ("jaxfne/validation.py", 967): (
        "VALIDATION",
        "gauge check failure returns (False, diagnostic)",
    ),
    ("jaxfne/validation.py", 1177): (
        "VALIDATION",
        "BasisSpec dict normalization failure returns valid=False",
    ),
    ("jaxfne/optim/core.py", 224): (
        "PERSISTENCE",
        "AGSDRSpec.to_dict parameter serialization fallback",
    ),
    ("jaxfne/optim/core.py", 1489): (
        "SCIENTIFIC_COMPUTATION",
        "inner-loop surrogate loss failure -> inf (optimizer-domain rejection)",
    ),
    ("jaxfne/optim/core.py", 1601): (
        "EXPECTED_NUMERICAL_DOMAIN_FAILURE",
        "inner Adam step failure breaks inner loop; uses AGSDR candidate",
    ),
    ("jaxfne/optim/core.py", 1611): (
        "EXPECTED_NUMERICAL_DOMAIN_FAILURE",
        "candidate refinement failure falls back to unrefined AGSDR candidate",
    ),
    ("jaxfne/optim/core.py", 1625): (
        "EXPECTED_NUMERICAL_DOMAIN_FAILURE",
        "final scoring failure -> inf; REVISE unless finite best_score",
    ),
    ("jaxfne/optim/core.py", 1647): (
        "SCIENTIFIC_COMPUTATION",
        "matrix extraction failure counted in fallback_counts only",
    ),
    ("jaxfne/optim/core.py", 1692): (
        "SCIENTIFIC_COMPUTATION",
        "matrix diagnostics serialization failure counted only",
    ),
    ("jaxfne/optim/core.py", 1929): (
        "EXPECTED_NUMERICAL_DOMAIN_FAILURE",
        "scalar differentiable step failure -> nan loss; REVISE if no finite best",
    ),
    ("jaxfne/optim/manifests.py", 40): (
        "PERSISTENCE",
        "optimization manifest array serialization fallback to str",
    ),
    ("jaxfne/_construct_connectivity.py", 592): (
        "STATE_MUTATION",
        "optional recurrent_backend=edge_list upgrade; explicit contradiction raises before try",
    ),
}


def _default_class(rel: str) -> tuple[str, str]:
    if rel.startswith("jaxfne/vis/"):
        return "BEST_EFFORT_PRESENTATION", "visualization panel/plot fallback"
    if rel == "jaxfne/util.py":
        return "BEST_EFFORT_PRESENTATION", "config/tensor diff and summary coercion"
    if rel == "jaxfne/tutorial_utils.py":
        return "BEST_EFFORT_PRESENTATION", "tutorial helper fallback"
    if rel.startswith("jaxfne/fields/"):
        return "BEST_EFFORT_PRESENTATION", "field proxy diagnostic metric fallback to None"
    if rel.startswith("jaxfne/optim/"):
        return "BEST_EFFORT_PRESENTATION", "optimizer/manifest fallback with fallback_counts"
    if rel in ("jaxfne/paradigm.py", "jaxfne/runtime.py", "jaxfne/validation.py"):
        return "BEST_EFFORT_PRESENTATION", "runtime/validation/json_safe presentation fallback"
    return "BEST_EFFORT_PRESENTATION", "pending manual review"


def inventory() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, line in enumerate(lines, 1):
            if not HANDLER_RE.search(line):
                continue
            key = (rel, lineno)
            classification, note = _OVERRIDES.get(key, _default_class(rel))
            rows.append(
                {
                    "file": rel,
                    "line": lineno,
                    "handler": line.strip(),
                    "classification": classification,
                    "note": note,
                }
            )
    return rows


def main() -> int:
    rows = inventory()
    tally = {
        "audit": "w3_broad_handler_tally",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "package": "jaxfne",
        "handler_count": len(rows),
        "by_classification": {},
        "handlers": rows,
    }
    for row in rows:
        c = row["classification"]
        tally["by_classification"][c] = tally["by_classification"].get(c, 0) + 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tally, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUT.relative_to(ROOT)), **tally["by_classification"], "total": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
