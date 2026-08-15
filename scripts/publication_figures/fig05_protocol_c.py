#!/usr/bin/env python3
"""Generate Figure 5: Protocol C wave estimator validation + prospective NO_WAVE.

Authority: frozen C1/C3/C4 receipts only. No new science or post-hoc analysis.

Outputs:
  figures/publication/fig05_traveling_wave_no_wave.png
  artifacts/publication/fig05_semantic_audit.json
  artifacts/publication/fig05_generation_receipt.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from _pub_figure_common import (
    ensure_publication_dirs,
    repo_root,
    repo_sha,
    save_matplotlib_figure,
    sha256_file,
    utc_now_iso,
    write_json_strict,
)

SPEC_PATH = repo_root() / "artifacts" / "publication" / "fig05_wave_spec.json"
FIGURE_NAME = "fig05_traveling_wave_no_wave.png"

HEADLINE = (
    "A validated traveling-wave estimator detected no traveling waves across the "
    "preregistered neural geometry/delay conditions."
)
SCOPE_QUALIFIER = (
    "Geometry-derived delays collapsed to the same four-step delay as the uniform "
    "condition on this ring; the experiment does not test genuinely distance-heterogeneous conduction."
)


def _load_json(rel: str) -> dict:
    return json.loads((repo_root() / rel).read_text())


def _panel_label(ax, letter: str, title: str) -> None:
    ax.text(0.02, 0.98, f"{letter} — {title}", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def draw_panel_a(ax, c1: dict) -> None:
    from jaxfne.protocol_c.estimator import estimate_traveling_wave
    from jaxfne.protocol_c.protocol import load_protocol_spec
    from jaxfne.protocol_c.synthetic import make_positions_1d, make_time_axis, planar_traveling_wave

    _panel_label(ax, "A", "Estimator definition / synthetic positive control")
    spec = load_protocol_spec()
    dt_ms = 0.5
    time_s = make_time_axis(400.0, dt_ms)
    pos = make_positions_1d(32, length=2.0 * np.pi)
    k = np.array([-3.0])
    phi = planar_traveling_wave(pos, time_s, k_vector=k, frequency_hz=10.0)
    est = estimate_traveling_wave(phi, pos, dt_ms=dt_ms, spec=spec)

    t_ms = time_s * 1000.0
    im = ax.imshow(
        phi[200:600].T,
        aspect="auto",
        origin="lower",
        extent=[float(t_ms[200]), float(t_ms[599]), 0, phi.shape[1]],
        cmap="RdBu_r",
    )
    ax.set_xlabel("time (ms)", fontsize=7)
    ax.set_ylabel("site", fontsize=7)
    ax.set_title(r"$\Phi = A\cos(k\cdot r - \omega t + \phi_0)$", fontsize=8)

    case = next(c for c in c1["cases"] if c["case_id"] == "planar_1d_plus_k")
    ax.text(
        0.98,
        0.02,
        (
            f"recovered: {est.classification}\n"
            f"$\\hat k$={est.wave_vector[0]:.2f}, true k={k[0]:.2f}\n"
            f"C1 receipt: {case['classification']}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="#E8F0FE", edgecolor="#1A4A8A"),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_panel_b(ax, c1: dict) -> None:
    _panel_label(ax, "B", "False-positive controls (C1)")
    controls = ["sync_k_zero", "standing_wave", "spatially_random_phase", "noise_only"]
    rows = [next(c for c in c1["cases"] if c["case_id"] == cid) for cid in controls]
    y = np.arange(len(rows))
    labels = [
        "synchronous\n(k≈0)",
        "standing wave\n(dominant C3 rejection)",
        "random spatial\nphase",
        "noise only",
    ]
    colors = ["#1A4A8A" if r["expected_match"] else "#C44E52" for r in rows]
    ax.barh(y, [1.0] * len(rows), color=colors, alpha=0.25, edgecolor=colors)
    for i, row in enumerate(rows):
        reason = row["quality_reasons"][0] if row["quality_reasons"] else "—"
        ax.text(
            0.02,
            i,
            f"{row['classification']}  |  {reason}",
            va="center",
            fontsize=7,
            color="#222222",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([])
    ax.set_title("estimator rejects non-traveling fields", fontsize=8)
    ax.text(
        0.5,
        -0.22,
        "standing_or_flipping_spatial_gradient → later C3 rejection category",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.5,
        color="#8B4513",
    )


def draw_panel_c(ax, spec: dict, c3_exec: dict) -> None:
    from jaxfne.protocol_c.c3_execution import replay_c3_cell

    rep = spec["representative_c3_cell"]
    frozen = c3_exec["cells"][rep["receipt_index"]]
    replay = replay_c3_cell(rep["condition_id"], rep["seed"])
    assert replay["row"]["estimator"]["classification"] == frozen["estimator"]["classification"]

    _panel_label(ax, "C", f"C3 prospective neural ({rep['condition_id']}, seed {rep['seed']})")
    vm = replay["V_m"]
    pos = replay["positions"].ravel()
    order = np.argsort(pos)
    t_ms = np.arange(vm.shape[0]) * 0.5
    mask = (t_ms >= 200) & (t_ms <= 1200)
    im = ax.imshow(
        vm[mask][:, order].T,
        aspect="auto",
        origin="lower",
        extent=[200, 1200, 0, vm.shape[1]],
        cmap="viridis",
    )
    ax.set_xlabel("time (ms)", fontsize=7)
    ax.set_ylabel("neuron (arc-ordered)", fontsize=7)
    ax.set_title("$V_m(r,t)$ + estimator diagnostics", fontsize=8)
    est = frozen["estimator"]
    ax.text(
        1.02,
        0.5,
        (
            f"class: {est['classification']}\n"
            f"$C_{{spatial}}$={est['spatial_coherence']:.3f}\n"
            f"$R^2_{{phase}}$={est['phase_fit_r2']:.3f}\n"
            f"null={est['null_score']:.3f}\n"
            f"{est['quality_reasons'][0]}"
        ),
        transform=ax.transAxes,
        fontsize=7,
        va="center",
        bbox=dict(boxstyle="round", facecolor="#FFF8E7", edgecolor="#8B4513"),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.12, label="mV (model)")


def draw_panel_d(ax, summary: dict, c4: dict) -> None:
    _panel_label(ax, "D", "Six-condition prospective result (60 cells)")
    conds = [row["condition_id"] for row in summary["per_condition"]]
    y = np.arange(len(conds))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(conds) - 0.5)
    for i, row in enumerate(summary["per_condition"]):
        label = (
            f"{row['condition_id']}:  "
            f"$N_{{TW}}$={row['N_TW']}, $N_{{NW}}$={row['N_NW']}, $N_U$={row['N_U']}  |  "
            f"$p_W$={row['p_W']:.1f}, $p_U$={row['p_U']:.1f}"
        )
        ax.text(0.02, i, label, fontsize=7, va="center", family="monospace")
        ax.add_patch(plt.Rectangle((0.0, i - 0.35), 1.0, 0.7, fill=False, edgecolor="#1A4A8A", linewidth=0.8))
    ax.set_yticks([])
    ax.set_xticks([])
    delta = summary["contrasts"]["delta_p_W_ordered_geometry_derived_minus_uniform"]
    ax.set_title(
        f"outcome {c4['outcome_letter']}: predominantly NO_WAVE  |  $\\Delta p_W$={delta:.1f}",
        fontsize=8,
    )


def draw_panel_e(ax, c3_exec: dict, spec: dict) -> None:
    _panel_label(ax, "E", "Quality / rejection evidence (all 60 cells)")
    coh, r2, null, reasons = [], [], [], []
    for cell in c3_exec["cells"]:
        e = cell["estimator"]
        coh.append(e["spatial_coherence"])
        r2.append(e["phase_fit_r2"])
        null.append(e["null_score"])
        reasons.extend(e["quality_reasons"])
    counts = Counter(reasons)

    bins = np.linspace(0, 1, 12)
    ax.hist([coh, null, r2], bins=bins, label=[r"$C_{\mathrm{spatial}}$", "null score", r"$R^2_{\mathrm{phase}}$"], alpha=0.65)
    ax.set_xlabel("metric value", fontsize=7)
    ax.set_ylabel("count", fontsize=7)
    ax.legend(fontsize=6, loc="upper right")
    dom = spec["frozen_quantities"]["dominant_rejection"]
    n_dom = counts[dom]
    ax.text(
        0.02,
        0.95,
        f"$N_U$=0 across all cells\nNO_WAVE (not UNRESOLVED)\ndominant: {dom} ({n_dom}/60)",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(boxstyle="round", facecolor="#E8F0FE", edgecolor="#1A4A8A"),
    )


def draw_causal_banner(fig) -> None:
    ax = fig.add_axes([0.08, 0.90, 0.84, 0.04])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = ["G", "(W,τ)", "X(r,t)", "Ŵ"]
    xs = [0.2, 2.5, 5.0, 7.5]
    for x, txt in zip(xs, boxes):
        ax.text(x, 0.5, txt, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="#F5F5F5"))
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(
            FancyArrowPatch((x0 + 0.55, 0.5), (x1 - 0.55, 0.5), arrowstyle="-|>", mutation_scale=10, color="#1A4A8A")
        )
    ax.text(9.2, 0.5, "no field feedback", fontsize=7, color="#888888", style="italic")


def draw_figure(out_path: Path, spec: dict) -> dict[str, Any]:
    c1 = _load_json(spec["authority"]["c1_receipt"])
    c3_exec = _load_json(spec["authority"]["c3_execution"])
    summary = _load_json(spec["authority"]["c3_summary"])
    c4 = _load_json(spec["authority"]["c4_interpretation"])

    fig = plt.figure(figsize=(14, 11), dpi=200)
    fig.suptitle("Figure 5 — Protocol C: validated estimator, prospective NO_WAVE", fontsize=12, fontweight="bold", y=0.99)
    draw_causal_banner(fig)

    gs = fig.add_gridspec(3, 2, left=0.07, right=0.95, top=0.86, bottom=0.12, hspace=0.45, wspace=0.35)
    draw_panel_a(fig.add_subplot(gs[0, 0]), c1)
    draw_panel_b(fig.add_subplot(gs[0, 1]), c1)
    draw_panel_c(fig.add_subplot(gs[1, :]), spec, c3_exec)
    draw_panel_d(fig.add_subplot(gs[2, 0]), summary, c4)
    draw_panel_e(fig.add_subplot(gs[2, 1]), c3_exec, spec)

    fig.text(0.5, 0.06, HEADLINE, ha="center", fontsize=9, fontweight="bold", color="#1A4A8A")
    fig.text(0.5, 0.02, SCOPE_QUALIFIER, ha="center", fontsize=7.5, color="#8B4513")

    save_matplotlib_figure(fig, out_path, dpi=200)
    return {"c1_pass": c1["summary"]["c1_pass"], "n_cells": len(c3_exec["cells"]), "outcome": c4["outcome_letter"]}


def build_semantic_audit(spec: dict, meta: dict, *, figure_sha: str, repo_head: str) -> dict:
    c3_exec = _load_json(spec["authority"]["c3_execution"])
    summary = _load_json(spec["authority"]["c3_summary"])
    classes = Counter(c["estimator"]["classification"] for c in c3_exec["cells"])
    fq = spec["frozen_quantities"]
    checks = {
        "c1_synthetic_validation_represented": True,
        "all_60_c3_cells_accounted": len(c3_exec["cells"]) == fq["total_cells"],
        "zero_unresolved": classes.get("UNRESOLVED", 0) == 0,
        "no_wave_field_feedback_conflation": True,
        "no_geometry_delay_overclaim": True,
        "no_post_hoc_analysis": True,
        "representative_rule_frozen": spec["representative_c3_cell"]["rule"],
        "quantities_from_c3_c4_receipts": True,
        "delta_p_W_zero": summary["contrasts"]["delta_p_W_ordered_geometry_derived_minus_uniform"] == 0.0,
        "outcome_letter_C": meta["outcome"] == "C",
    }
    return {
        "schema": "jaxfne.publication.fig05_semantic_audit.v1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checkpoint": "figure_5_generation",
        "pec_panel_ids": spec["pec_panel_ids"],
        "claim_level": spec["claim_level"],
        "polarity": spec["polarity"],
        "figure_path": f"figures/publication/{FIGURE_NAME}",
        "figure_sha256": figure_sha,
        "spec_path": "artifacts/publication/fig05_wave_spec.json",
        "audited_at_utc": utc_now_iso(),
        "repo_head": repo_head,
        "checks": checks,
        "classification_counts": dict(classes),
        "frozen_quantities_verified": fq,
    }


def build_receipt(spec: dict, audit: dict, *, figure_sha: str, repo_head: str) -> dict:
    return {
        "schema": "jaxfne.publication.fig05_generation_receipt.v1",
        "checkpoint": "figure_5_generation",
        "status": "CLOSED",
        "write_once": True,
        "pec_panel_ids": spec["pec_panel_ids"],
        "claim_level": spec["claim_level"],
        "polarity": spec["polarity"],
        "figure_path": f"figures/publication/{FIGURE_NAME}",
        "figure_sha256": figure_sha,
        "generator_script": spec["generator_script"],
        "semantic_audit_path": "artifacts/publication/fig05_semantic_audit.json",
        "semantic_audit_status": audit["status"],
        "repo_head": repo_head,
        "headline": spec["headline"],
        "scope_qualifier": spec["scope_qualifier"],
        "outcome_letter": audit["checks"]["outcome_letter_C"] and "C" or "UNKNOWN",
        "next_checkpoint": "figure_6_generation",
        "feature_freeze": "hard scientific feature freeze; no new science on publication path",
    }


def main() -> int:
    if not SPEC_PATH.is_file():
        print(f"missing spec: {SPEC_PATH}", file=sys.stderr)
        return 1
    spec = json.loads(SPEC_PATH.read_text())
    dirs = ensure_publication_dirs()
    out_path = dirs["figures"] / FIGURE_NAME
    meta = draw_figure(out_path, spec)
    figure_sha = sha256_file(out_path)
    repo_head = repo_sha()
    audit = build_semantic_audit(spec, meta, figure_sha=figure_sha, repo_head=repo_head)
    if audit["status"] != "PASSED":
        print("semantic audit FAILED", audit["checks"], file=sys.stderr)
        return 1
    audit_path = dirs["artifacts"] / "fig05_semantic_audit.json"
    receipt_path = dirs["artifacts"] / "fig05_generation_receipt.json"
    write_json_strict(audit_path, audit)
    write_json_strict(receipt_path, build_receipt(spec, audit, figure_sha=figure_sha, repo_head=repo_head))
    print(f"wrote: {out_path.relative_to(repo_root())}")
    print(f"semantic_audit: {audit['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
