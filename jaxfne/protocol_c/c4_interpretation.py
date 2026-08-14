"""C4 prospective interpretation freeze for Protocol C neural experiment."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from jaxfne.io import json_safe
from jaxfne.protocol_c.c3_execution import (
    C3_CONDITION_SUMMARY_PATH,
    C3_EXECUTION_RECEIPT_PATH,
    load_c3_execution_receipt,
)
from jaxfne.protocol_c.c3_protocol import c3_condition_ids, load_c3_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "artifacts" / "protocol_c"
C4_INTERPRETATION_RECEIPT_PATH = BUNDLE_ROOT / "c4_interpretation_receipt.json"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _count_classifications(cells: list[dict[str, Any]], condition_id: str) -> dict[str, int]:
    subset = [c for c in cells if c["condition_id"] == condition_id]
    counts = {"TRAVELING_WAVE": 0, "NO_WAVE": 0, "UNRESOLVED": 0}
    for c in subset:
        cls = c["estimator"]["classification"]
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _quality_distributions(cells: list[dict[str, Any]]) -> dict[str, Any]:
    coh = [float(c["estimator"]["spatial_coherence"]) for c in cells]
    r2 = [float(c["estimator"]["phase_fit_r2"]) for c in cells]
    nulls = [float(c["estimator"]["null_score"]) for c in cells]
    return {
        "spatial_coherence": {
            "min": float(np.min(coh)),
            "max": float(np.max(coh)),
            "mean": float(np.mean(coh)),
            "median": float(np.median(coh)),
        },
        "phase_fit_r2": {
            "min": float(np.min(r2)),
            "max": float(np.max(r2)),
            "mean": float(np.mean(r2)),
            "median": float(np.median(r2)),
        },
        "null_score": {
            "min": float(np.min(nulls)),
            "max": float(np.max(nulls)),
            "mean": float(np.mean(nulls)),
            "median": float(np.median(nulls)),
        },
    }


def _conditional_velocity_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    tw = [
        c for c in cells if c["estimator"]["classification"] == "TRAVELING_WAVE"
    ]
    if not tw:
        return {"n": 0, "phase_velocity": None, "direction": None}
    vels = [float(c["estimator"]["phase_velocity"]) for c in tw if np.isfinite(c["estimator"]["phase_velocity"])]
    dirs = [c["estimator"]["direction"] for c in tw]
    return {
        "n": len(tw),
        "phase_velocity": {
            "min": float(np.min(vels)) if vels else None,
            "max": float(np.max(vels)) if vels else None,
            "mean": float(np.mean(vels)) if vels else None,
        },
        "direction_examples": dirs[:5],
    }


def _assign_outcome_letter(condition_summaries: list[dict[str, Any]]) -> str:
    total_tw = sum(s["N_TW"] for s in condition_summaries)
    total_u = sum(s["N_U"] for s in condition_summaries)
    total_nw = sum(s["N_NW"] for s in condition_summaries)
    total = sum(s["N_total"] for s in condition_summaries)
    if total == 0:
        return "D"
    p_u = total_u / total
    p_w = total_tw / total
    if p_u > 0.5:
        return "D"
    if p_w == 0 and total_nw > 0:
        return "C"
    ordered_gd = next((s for s in condition_summaries if s["condition_id"] == "ordered_geometry_derived"), None)
    ordered_u = next((s for s in condition_summaries if s["condition_id"] == "ordered_uniform"), None)
    if ordered_gd and ordered_u and ordered_gd["p_W"] > ordered_u["p_W"]:
        return "A"
    if p_w > 0:
        return "B"
    return "C"


def summarize_conditions(execution: dict[str, Any], spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or load_c3_spec()
    cells = execution["cells"]
    n_seeds = int(spec["simulation_policy"]["n_seeds"])
    summaries = []
    for cid in c3_condition_ids(spec):
        counts = _count_classifications(cells, cid)
        n_tw = counts["TRAVELING_WAVE"]
        n_nw = counts["NO_WAVE"]
        n_u = counts["UNRESOLVED"]
        summaries.append(
            {
                "condition_id": cid,
                "N_TW": n_tw,
                "N_NW": n_nw,
                "N_U": n_u,
                "N_total": n_seeds,
                "p_W": float(n_tw) / float(n_seeds),
                "p_U": float(n_u) / float(n_seeds),
            }
        )
    by_id = {s["condition_id"]: s for s in summaries}
    delta_p_w_ordered = float(by_id["ordered_geometry_derived"]["p_W"] - by_id["ordered_uniform"]["p_W"])
    contrasts = {
        "delta_p_W_ordered_geometry_derived_minus_uniform": delta_p_w_ordered,
        "p_W_ordered_geometry_derived": by_id["ordered_geometry_derived"]["p_W"],
        "p_W_ordered_uniform": by_id["ordered_uniform"]["p_W"],
        "p_W_ordered_geometry_derived_vs_ordered_delay_shuffled": {
            "geometry_derived": by_id["ordered_geometry_derived"]["p_W"],
            "delay_shuffled": by_id["ordered_delay_shuffled"]["p_W"],
            "delta": float(
                by_id["ordered_geometry_derived"]["p_W"] - by_id["ordered_delay_shuffled"]["p_W"]
            ),
        },
        "p_W_ordered_vs_shuffled_geometry_geometry_derived": {
            "ordered": by_id["ordered_geometry_derived"]["p_W"],
            "shuffled": by_id["shuffled_geometry_derived"]["p_W"],
            "delta": float(
                by_id["ordered_geometry_derived"]["p_W"] - by_id["shuffled_geometry_derived"]["p_W"]
            ),
        },
    }
    directional = spec["directional_conjecture"]
    conjecture_observed = {
        "statement": directional["statement"],
        "required_for_success": directional["required_for_c3_success"],
        "observed_delta_p_W": delta_p_w_ordered,
        "conjecture_direction_supported": delta_p_w_ordered > 0,
    }
    return {
        "schema": "jaxfne.protocol_c.c3_condition_summary.v1",
        "per_condition": summaries,
        "contrasts": contrasts,
        "directional_conjecture": conjecture_observed,
        "quality_all_cells": _quality_distributions(cells),
        "conditional_TRAVELING_WAVE": _conditional_velocity_summary(cells),
        "v_c_diagnostic_cells": [
            {
                "condition_id": c["condition_id"],
                "seed": c["seed"],
                "v_c_diagnostic_ratio": c.get("v_c_diagnostic_ratio"),
                "phase_velocity": c["estimator"]["phase_velocity"],
            }
            for c in cells
            if c.get("v_c_diagnostic_ratio") is not None
        ],
    }


def build_c4_interpretation(
    execution: dict[str, Any],
    summary: dict[str, Any],
    *,
    package_head: str | None = None,
) -> dict[str, Any]:
    spec = load_c3_spec()
    outcome = _assign_outcome_letter(summary["per_condition"])
    outcomes = {
        "A": "wave evidence enriched by organized geometry/delays",
        "B": "waves occur but are insensitive to the tested organization",
        "C": "sufficient-quality neural activity yields predominantly NO_WAVE",
        "D": "activity is predominantly UNRESOLVED for the frozen estimator",
    }
    return json_safe(
        {
            "schema": "jaxfne.protocol_c.c4_interpretation_receipt.v1",
            "checkpoint": "C4",
            "status": "FROZEN",
            "write_once": True,
            "package_head": package_head or _git_head(),
            "execution_receipt": str(C3_EXECUTION_RECEIPT_PATH.relative_to(REPO_ROOT)),
            "condition_summary": str(C3_CONDITION_SUMMARY_PATH.relative_to(REPO_ROOT)),
            "outcome_letter": outcome,
            "outcome_description": outcomes[outcome],
            "outcomes_reference": outcomes,
            "directional_conjecture": summary["directional_conjecture"],
            "contrasts": summary["contrasts"],
            "evidence_level": {
                "estimator": "C1_VALIDATED_SYNTHETIC_ONLY",
                "neural_dynamics": "C3_PROSPECTIVE_OBSERVED",
                "wave_mechanism": "NOT_CLAIMED",
                "field_mediation": "NOT_CLAIMED",
                "inferential_statistics": "DESCRIPTIVE_COUNTS_ONLY",
            },
            "interpretation_rules": [
                "UNRESOLVED is not NO_WAVE",
                "v_phase != v_c is not automatic failure",
                "no_C3b_or_C5_followup_in_this_protocol",
                "protocol_C_closed_at_C4",
            ],
            "protocol_c_closed": True,
            "next_work": "0.4.17-D biological RBS (new protocol id for further wave science)",
        }
    )


def write_c4_freeze(
    execution: dict[str, Any] | None = None,
    *,
    package_head: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution = execution or load_c3_execution_receipt()
    summary = summarize_conditions(execution)
    C3_CONDITION_SUMMARY_PATH.write_text(json.dumps(json_safe(summary), indent=2) + "\n")
    c4 = build_c4_interpretation(execution, summary, package_head=package_head)
    C4_INTERPRETATION_RECEIPT_PATH.write_text(json.dumps(c4, indent=2) + "\n")
    return summary, c4


def load_c4_interpretation_receipt() -> dict[str, Any]:
    return json.loads(C4_INTERPRETATION_RECEIPT_PATH.read_text())
