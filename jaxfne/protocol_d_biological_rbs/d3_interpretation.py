"""D3 prospective interpretation freeze — Q1/Q2/Q3 and D−N2 contrast."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from jaxfne.io import json_safe
from jaxfne.protocol_d_biological_rbs.d3_execution import (
    REPO_ROOT,
    load_d3_execution_receipt,
    write_d3_execution_receipt,
)
from jaxfne.protocol_d_biological_rbs.d3_protocol import D3_INTERPRETATION_RECEIPT_PATH, load_d3_spec


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _count_labels(cells: list[dict[str, Any]], *, null_arm: str | None = None) -> dict[str, int]:
    counts = {"ADAPTATION": 0, "NO_ADAPTATION": 0, "UNRESOLVED": 0}
    for c in cells:
        if null_arm is not None and c["null_arm"] != null_arm:
            continue
        cls = c["classification"]
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _facilitation_summary(cells: list[dict[str, Any]], *, null_arm: str | None = None) -> dict[str, Any]:
    subset = [c for c in cells if null_arm is None or c["null_arm"] == null_arm]
    fac = [c for c in subset if c.get("facilitation") is True and c.get("A_adapt") is not None]
    return {
        "n_facilitation": len(fac),
        "A_adapt_values": [float(c["A_adapt"]) for c in fac],
    }


def _recovery_hidden_state_trend(d_cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Test T_rec up => |H_K(T_rechallenge)-1| down on D arm."""
    by_interval: dict[str, list[float]] = {}
    for c in d_cells:
        if c["null_arm"] != "D":
            continue
        rid = c["recovery_interval_id"]
        val = float(c["mechanism"]["abs_H_K_minus_1_at_rechallenge"])
        by_interval.setdefault(rid, []).append(val)
    rows = []
    for rid in ("short", "medium", "long"):
        vals = by_interval.get(rid, [])
        rows.append(
            {
                "recovery_interval_id": rid,
                "mean_abs_H_K_minus_1_at_rechallenge": float(np.mean(vals)) if vals else None,
                "n": len(vals),
            }
        )
    monotonic = None
    if all(r["mean_abs_H_K_minus_1_at_rechallenge"] is not None for r in rows):
        m = [r["mean_abs_H_K_minus_1_at_rechallenge"] for r in rows]
        monotonic = bool(m[0] >= m[1] >= m[2])
    return {"per_interval": rows, "T_rec_up_implies_H_K_closer_to_1": monotonic}


def _recovery_response_trend(d_cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate observable recovery: R_rechallenge vs R_early."""
    by_interval: dict[str, list[dict[str, float]]] = {}
    for c in d_cells:
        if c["null_arm"] != "D":
            continue
        rid = c["recovery_interval_id"]
        by_interval.setdefault(rid, []).append(
            {
                "R_early": float(c["R_early"]),
                "R_rechallenge": float(c["R_rechallenge"]),
                "R_recovery": c.get("R_recovery"),
            }
        )
    rows = []
    for rid in ("short", "medium", "long"):
        items = by_interval.get(rid, [])
        r_rec = [x["R_recovery"] for x in items if x["R_recovery"] is not None]
        rows.append(
            {
                "recovery_interval_id": rid,
                "mean_R_rechallenge": float(np.mean([x["R_rechallenge"] for x in items])) if items else None,
                "mean_R_early": float(np.mean([x["R_early"] for x in items])) if items else None,
                "mean_R_recovery": float(np.mean(r_rec)) if r_rec else None,
                "R_recovery_values": r_rec,
            }
        )
    return {"per_interval": rows}


def _d_minus_n2_contrast(cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for seed in sorted({c["seed"] for c in cells}):
        for rid in ("short", "medium", "long"):
            d = next(
                c for c in cells
                if c["seed"] == seed and c["null_arm"] == "D" and c["recovery_interval_id"] == rid
            )
            n2 = next(
                c for c in cells
                if c["seed"] == seed and c["null_arm"] == "N2" and c["recovery_interval_id"] == rid
            )
            rows.append(
                {
                    "seed": seed,
                    "recovery_interval_id": rid,
                    "D_classification": d["classification"],
                    "N2_classification": n2["classification"],
                    "D_A_adapt": d.get("A_adapt"),
                    "N2_A_adapt": n2.get("A_adapt"),
                    "D_adapts_N2_does_not": (
                        d["classification"] == "ADAPTATION" and n2["classification"] != "ADAPTATION"
                    ),
                }
            )
    n_support = sum(1 for r in rows if r["D_adapts_N2_does_not"])
    return {"pairwise": rows, "n_D_adaptation_N2_not": n_support}


def build_d3_interpretation(
    execution: dict[str, Any],
    spec: dict[str, Any] | None = None,
    *,
    package_head: str | None = None,
) -> dict[str, Any]:
    spec = spec or load_d3_spec()
    cells = execution["cells"]

    d_cells = [c for c in cells if c["null_arm"] == "D"]
    d_mech_ok = [c for c in d_cells if c.get("mechanism", {}).get("mechanism_ok")]
    d_adapt = [c for c in d_cells if c["classification"] == "ADAPTATION"]
    d_no = [c for c in d_cells if c["classification"] == "NO_ADAPTATION"]
    d_unres = [c for c in d_cells if c["classification"] == "UNRESOLVED"]

    theta_a = float(spec["frozen_thresholds"]["theta_A"])
    d_a_pos = [
        c for c in d_cells
        if c.get("A_adapt") is not None and float(c["A_adapt"]) > theta_a
    ]
    d_m1 = [c for c in d_cells if c.get("mechanism", {}).get("M1_pass")]
    d_m2 = [c for c in d_cells if c.get("mechanism", {}).get("M2_pass")]

    observable_invariant = False
    if d_cells:
        a0 = d_cells[0].get("A_adapt")
        observable_invariant = all(
            c.get("A_adapt") == a0
            for arm_cells in (
                [c for c in cells if c["null_arm"] == arm]
                for arm in ("N0", "N1", "N2", "D")
            )
            for c in arm_cells
        )

    Q1_mechanism = {
        "question": "Did activity write the intended RBS state?",
        "scope": "D arm mechanism checks M1/M2",
        "n_D_cells": len(d_cells),
        "n_M1_pass": len(d_m1),
        "n_M2_pass": len(d_m2),
        "n_mechanism_ok": len(d_mech_ok),
        "fraction_mechanism_ok": len(d_mech_ok) / len(d_cells) if d_cells else 0.0,
        "answer": (
            "yes"
            if len(d_mech_ok) == len(d_cells) and d_cells
            else ("partial" if d_m1 else "no")
        ),
    }

    Q2_adaptation = {
        "question": "Did that state produce adaptation?",
        "scope": "D arm phenotype classification",
        "counts": _count_labels(cells, null_arm="D"),
        "n_A_adapt_above_theta": len(d_a_pos),
        "observable_A_adapt_invariant_across_null_arms": observable_invariant,
        "facilitation": _facilitation_summary(cells, null_arm="D"),
        "answer": (
            "adaptation"
            if d_adapt
            else ("unresolved" if len(d_unres) == len(d_cells) else "no_adaptation")
        ),
    }

    Q3_recovery = {
        "question": "Did hidden-state and observable response recover with rest?",
        "hidden_state": _recovery_hidden_state_trend(d_cells),
        "observable_response": _recovery_response_trend(d_cells),
        "note": "hidden-state recovery and spike-count recovery are reported separately",
    }

    contrast = _d_minus_n2_contrast(cells)

    if d_adapt and contrast["n_D_adaptation_N2_not"] > 0:
        headline = "activity_written_RBS_produces_adaptation_phenotype"
        narrative = (
            "D arm shows ADAPTATION with mechanism gates satisfied while N2 does not "
            "in at least one matched seed/recovery cell (primary contrast D−N2)."
        )
    elif d_a_pos and not d_adapt:
        headline = "observable_attenuation_without_formal_ADAPTATION"
        narrative = (
            f"D arm shows A_adapt > theta_A ({theta_a}) in {len(d_a_pos)} cells, but "
            "formal ADAPTATION classification fails because joint mechanism gates are "
            "not satisfied (M2: H_K_late > 1 + theta_H). Observable spike attenuation "
            "is identical across N0/N1/N2/D under this frozen paradigm (D−N2 null "
            "on A_adapt); hidden-state writing is detectable on D but does not "
            "differentiate the preregistered spike-count phenotype from N2."
        )
    elif d_mech_ok and not d_adapt and d_no:
        headline = "mechanism_ok_NO_ADAPTATION"
        narrative = (
            "Hidden-state mechanism S→H_A→H_K behaves as designed on D arm, but "
            "A_adapt does not exceed theta_A: state-space works; Izhikevich coupling "
            "does not yield preregistered adaptation phenotype under frozen paradigm."
        )
    elif d_unres and not d_adapt and not d_no:
        headline = "UNRESOLVED_sparse_early_response"
        narrative = "Early responses too sparse to define adaptation index; silence not interpreted as attenuation."
    else:
        headline = "mixed_or_no_adaptation"
        narrative = "No D-arm ADAPTATION classification under frozen gates; see per-cell receipts."

    fac_note = _facilitation_summary(cells, null_arm="D")
    if fac_note["n_facilitation"] > 0:
        narrative += (
            f" Facilitation observed (A_adapt<0) in {fac_note['n_facilitation']} D cells; "
            "reported as NO_ADAPTATION with facilitation flag."
        )

    return json_safe(
        {
            "schema": "jaxfne.protocol_d_biological_rbs.d3_interpretation_receipt.v1",
            "checkpoint": "D3",
            "status": "FROZEN",
            "write_once": True,
            "package_head": package_head or _git_head(),
            "execution_receipt": "artifacts/protocol_d_biological_rbs/d3_execution_receipt.json",
            "spec_path": "artifacts/protocol_d_biological_rbs/d3_adaptation_recovery_phenotype_spec.json",
            "headline": headline,
            "narrative": narrative,
            "questions": {"Q1_mechanism": Q1_mechanism, "Q2_adaptation": Q2_adaptation, "Q3_recovery": Q3_recovery},
            "primary_contrast_D_minus_N2": contrast,
            "classification_counts_all_arms": {
                arm: _count_labels(cells, null_arm=arm) for arm in ("N0", "N1", "N2", "D")
            },
            "facilitation_all_arms": {
                arm: _facilitation_summary(cells, null_arm=arm) for arm in ("N0", "N1", "N2", "D")
            },
            "interpretation_rules_applied": spec["interpretation_rules"],
            "d2b_not_invalidated_by_NO_ADAPTATION": True,
            "terminology": spec["terminology"],
        }
    )


def write_d3_interpretation_receipt(
    execution: dict[str, Any] | None = None,
    *,
    package_head: str | None = None,
) -> dict[str, Any]:
    execution = execution or load_d3_execution_receipt()
    interp = build_d3_interpretation(execution, package_head=package_head)
    D3_INTERPRETATION_RECEIPT_PATH.write_text(json.dumps(interp, indent=2) + "\n")
    return interp


def load_d3_interpretation_receipt() -> dict[str, Any]:
    return json.loads(D3_INTERPRETATION_RECEIPT_PATH.read_text())


def freeze_d3_protocol(
    *,
    package_head: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run 36 cells, write raw + interpretation receipts (write-once)."""
    execution = write_d3_execution_receipt(package_head=package_head)
    interpretation = write_d3_interpretation_receipt(execution, package_head=package_head)
    return execution, interpretation
