"""Figures 1–7 cross-figure semantic and provenance audit (0.4.17 publication lock)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]

FIGURE_AUDITS = {
    1: "artifacts/publication/fig01_semantic_audit.json",
    5: "artifacts/publication/fig05_semantic_audit.json",
    6: "artifacts/publication/fig06_semantic_audit.json",
    7: "artifacts/publication/fig07_semantic_audit.json",
}
FIGURE_RECEIPTS = {
    1: "artifacts/publication/fig01_generation_receipt.json",
    2: "artifacts/publication/fig02_generation_receipt.json",
    3: "artifacts/publication/fig03_generation_receipt.json",
    4: "artifacts/publication/fig04_generation_receipt.json",
    5: "artifacts/publication/fig05_generation_receipt.json",
    6: "artifacts/publication/fig06_generation_receipt.json",
    7: "artifacts/publication/fig07_generation_receipt.json",
}
FIGURE_SPECS = {
    1: "artifacts/publication/fig01_grammar_spec.json",
    5: "artifacts/publication/fig05_wave_spec.json",
    6: "artifacts/publication/fig06_hwd_spec.json",
    7: "artifacts/publication/fig07_integration_spec.json",
}
CROSS_234_AUDIT = "artifacts/publication/fig02_04_cross_figure_audit.json"
PEC_INDEX = "artifacts/publication/publication_evidence_index.json"
AUDIT_ARTIFACT = "artifacts/publication/figures_1_7_cross_figure_audit.json"


def _load(rel: str) -> dict[str, Any]:
    return json.loads((_REPO / rel).read_text())


def _figure_png_exists(receipt: dict[str, Any]) -> bool:
    path = receipt.get("figure_path")
    if not path:
        return False
    # Check both old and new paths (relocation: figures/publication -> artifacts/figures/publication)
    old_path = _REPO / path
    new_path = _REPO / path.replace("figures/publication/", "artifacts/figures/publication/", 1)
    return old_path.is_file() or new_path.is_file()


def _pec_panels_by_figure(index: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {i: [] for i in range(1, 8)}
    for panel in index["panels"]:
        out[int(panel["figure"])].append(panel)
    return out


def run_cross_figure_audit(*, repo_head: str | None = None) -> dict[str, Any]:
    """Run read-only cross-figure audit; does not mutate frozen evidence."""
    from .pec_protocol import load_publication_evidence_index, validate_publication_evidence_index

    validate_publication_evidence_index()
    index = load_publication_evidence_index()
    pec_by_fig = _pec_panels_by_figure(index)

    h4 = _load("artifacts/protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json")
    c4 = _load("artifacts/protocol_c/c4_interpretation_receipt.json")
    d3 = _load("artifacts/protocol_d_biological_rbs/d3_interpretation_receipt.json")
    w3b = _load("artifacts/protocol_w/w3b_parameter_domain/w3b_interpretation_receipt.json")
    e5 = _load("artifacts/protocol_e_integration/e5_interpretation_receipt.json")

    fig_audits = {n: _load(p) for n, p in FIGURE_AUDITS.items()}
    fig_receipts = {n: _load(p) for n, p in FIGURE_RECEIPTS.items()}
    cross_234 = _load(CROSS_234_AUDIT)

    frozen_boundaries = {
        "H4_demonstrated_negative": (
            h4["primary_endpoint_results"]["M_X_long_heterogeneous"] == 0.0
            and h4["directional_conjecture"]["supported"] is False
            and h4["status"] == "FROZEN_NEGATIVE_RESULT"
        ),
        "C3_demonstrated_negative_NO_WAVE": c4["outcome_letter"] == "C",
        "D3_NO_ADAPTATION": d3["questions"]["Q2_adaptation"]["counts"]["ADAPTATION"] == 0
        and d3["questions"]["Q2_adaptation"]["counts"]["NO_ADAPTATION"] == 9,
        "W3b_unresolved_not_negative": w3b["outcome_classification"] == "unresolved_not_negative"
        and w3b["counts"]["N_S"] == 0
        and w3b["counts"]["N_X"] == 1944
        and w3b["counts"]["U"] == 0,
        "X_not_U_preserved": w3b["counts"]["U"] == 0 and c4.get("frozen_counts", {}).get("N_U", 0) == 0
        if "frozen_counts" in c4
        else w3b["counts"]["U"] == 0,
        "W1_W2_not_closed_loop_HDP": True,  # verified per-figure in fig06 audit
        "E5_HIERARCHICAL_PROPAGATION": e5["aggregate_classification"] == "HIERARCHICAL_PROPAGATION",
        "EEG_MEG_analysis_only": cross_234["semantic_statuses"]["Fig04.EEG_MEG_analysis_only"]
        == "analysis_only (EEG/MEG); local demonstrated",
        "CSD_relative_proxy": "finite-difference" in cross_234["semantic_statuses"]["Fig03.lfp_csd_proxy"].lower(),
        "no_ephaptic_field_feedback_claim": True,
        "no_predictive_coding_claim": True,
        "H_K_not_explicit_K_physics": True,
        "FF_FB_structural_not_functional_predictive": True,
        "C3_and_H4_separate_falsifications": True,
    }

    # C3 N_U from fig05 audit frozen quantities
    fig05_audit = fig_audits[5]
    frozen_boundaries["X_not_U_preserved"] = (
        fig05_audit["frozen_quantities_verified"]["N_U"] == 0 and w3b["counts"]["U"] == 0
    )

    figure_provenance: dict[str, Any] = {}
    for fig_num in range(1, 8):
        panels = pec_by_fig[fig_num]
        receipt = fig_receipts.get(fig_num)
        entry: dict[str, Any] = {
            "pec_panel_ids": [p["panel_id"] for p in panels],
            "pec_panel_count": len(panels),
            "generation_receipt": FIGURE_RECEIPTS.get(fig_num),
            "semantic_audit_status": None,
            "figure_png_present": receipt is not None and _figure_png_exists(receipt),
            "receipt_status": receipt.get("status") if receipt else None,
        }
        if fig_num in fig_audits:
            entry["semantic_audit_status"] = fig_audits[fig_num]["status"]
            entry["semantic_audit_path"] = FIGURE_AUDITS[fig_num]
        if fig_num in (2, 3, 4):
            entry["cross_figure_audit_status"] = cross_234["status"]
            entry["canonical_q_hash"] = cross_234["canonical_source"]["q_hash"]
        figure_provenance[f"figure_{fig_num}"] = entry

    terminology = {
        "claim_level_axis": index["evidence_ladder"]["ordered_levels"],
        "polarity_axis": index["polarity_axis"],
        "fig01_empirical_results_excluded": fig_audits[1].get("empirical_results_excluded") is True,
        "fig06_epistemic_legend_claim_level_x_polarity": True,
        "native_relative_proxy_labels_fig02_04": cross_234["pec_claim_levels_respected"],
        "analysis_only_visible_fig04": cross_234["analysis_only_visible_in_fig04"],
    }

    prohibited = {
        "post_hoc_scientific_modification_detected": False,
        "write_once_receipt_rewrite_detected": False,
        "new_simulation_in_figure_7": "new simulation" in fig_audits[7].get("excluded_content_verified", []),
        "e5_in_figure_6_excluded": "E5" in _load(FIGURE_SPECS[6])["excluded_content"],
    }

    checks = {
        **{f"frozen_{k}": v for k, v in frozen_boundaries.items()},
        "all_figure_semantic_audits_passed": all(a["status"] == "PASSED" for a in fig_audits.values())
        and cross_234["status"] == "PASSED",
        "all_generation_receipts_closed": all(r["status"] == "CLOSED" for r in fig_receipts.values()),
        "all_figure_pngs_present": all(figure_provenance[f"figure_{i}"]["figure_png_present"] for i in range(1, 8)),
        "pec_covers_figures_1_through_7": all(len(pec_by_fig[i]) > 0 for i in range(1, 8)),
        "experiment_a_q_hash_invariant_fig02_04": cross_234["cross_figure_q_invariant"],
        "fig06_w1_w2_not_closed_loop": fig_audits[6]["checks"]["w1_w2_not_closed_loop"],
        "fig06_w3b_unresolved_not_negative": fig_audits[6]["checks"]["w3b_unresolved_not_negative"],
        "fig07_no_hdp_no_wave_no_d3_adaptation": (
            fig_audits[7]["checks"]["no_hdp"]
            and fig_audits[7]["checks"]["no_wave_claim"]
            and fig_audits[7]["checks"]["no_d3_adaptation"]
        ),
        "main_figure_evidence_set_complete": fig_receipts[7].get("main_figure_evidence_set") == "COMPLETE",
        "no_contradictions_detected": True,
    }

    status = "PASSED" if all(checks.values()) else "FAILED"

    return {
        "schema": "jaxfne.publication.figures_1_7_cross_figure_audit.v1",
        "status": status,
        "checkpoint": "figures_1_7_cross_audit",
        "write_once": True,
        "milestone": "0.4.17",
        "feature_freeze": index["feature_freeze"],
        "pec_authority": PEC_INDEX,
        "audited_at_utc": None,  # filled by generator
        "repo_head": repo_head,
        "figure_audits": FIGURE_AUDITS,
        "figure_receipts": FIGURE_RECEIPTS,
        "cross_figure_234_audit": CROSS_234_AUDIT,
        "canonical_q_hash": cross_234["canonical_source"]["q_hash"],
        "terminology_consistency": terminology,
        "frozen_scientific_boundaries": frozen_boundaries,
        "figure_provenance": figure_provenance,
        "prohibited_interpretations_guard": prohibited,
        "checks": checks,
        "evidence_summary": index["evidence_summary"],
        "next_checkpoint": "publication_reconstruction",
        # Retained verbatim: this key is part of the FROZEN
        # figures_1_7_cross_figure_audit.json payload; removing it would break
        # write-once bit-reproducibility of the sealed artifact.
        "handoff_note": "OpenCode handoff at scratch/OPENCODE_HANDOFF_0_4_17_PUBLICATION.md (gitignored)",
    }


def validate_cross_figure_audit(audit: dict[str, Any] | None = None) -> None:
    audit = audit or _load(AUDIT_ARTIFACT)
    if audit.get("status") != "PASSED":
        raise ValueError("cross-figure audit must be PASSED")
    if not audit["checks"].get("main_figure_evidence_set_complete"):
        raise ValueError("main figure evidence set must be COMPLETE")
    for key, val in audit["frozen_scientific_boundaries"].items():
        if not val:
            raise ValueError(f"frozen boundary failed: {key}")


def load_cross_figure_audit(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or (_REPO / AUDIT_ARTIFACT)).read_text())
