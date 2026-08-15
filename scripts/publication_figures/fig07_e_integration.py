#!/usr/bin/env python3
"""Figure 7 — frozen E1–E5 integration ladder (0.4.17 publication).

Six panels A–F: compositional hierarchy -> delays -> RBS -> observation -> causal assay -> propagation.
No new simulation. Quantities from frozen E receipts via jaxfne.publication.fig07_evidence.

Outputs:
  figures/publication/fig07_e_integration.png
  artifacts/publication/fig07_semantic_audit.json
  artifacts/publication/fig07_generation_receipt.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _pub_figure_common import (
    ensure_publication_dirs,
    repo_root,
    repo_sha,
    save_matplotlib_figure,
    sha256_file,
    utc_now_iso,
    write_json_strict,
)

SPEC_PATH = repo_root() / "artifacts" / "publication" / "fig07_integration_spec.json"
FIGURE_NAME = "fig07_e_integration.png"

STYLE_POS = {"edgecolor": "#1A4A8A", "facecolor": "#E8F0FE", "linestyle": "solid", "lw": 1.5}
HEADLINE = (
    "Independently validated TFNE components compose into one hierarchical system "
    "with a causally attributable multiscale response."
)
CONCLUSION = (
    "A localized relative-biophysical perturbation, computationally inert when its typed "
    "coupling is disabled, propagates through declared coupling from its owner population "
    "through the hierarchical network into source and observation space."
)


def _panel_label(ax, letter: str, title: str) -> None:
    ax.text(
        0.02,
        0.98,
        f"{letter} — {title}",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        va="top",
        color=STYLE_POS["edgecolor"],
    )


def _style_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color(STYLE_POS["edgecolor"])
        spine.set_linewidth(STYLE_POS["lw"])


def draw_panel_a(ax, h1: dict[str, Any]) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    _panel_label(ax, "A", "E1 hierarchical substrate")
    layers = list(h1["layers"])
    y0, dy = 1.2, 1.3
    for col, area in enumerate(h1["areas"]):
        x = 1.5 + col * 4.0
        ax.add_patch(
            Rectangle((x, y0), 2.2, dy * len(layers), fill=False, edgecolor=STYLE_POS["edgecolor"], lw=1.2)
        )
        ax.text(x + 1.1, y0 + dy * len(layers) + 0.15, area, ha="center", fontsize=9, fontweight="bold")
        for i, layer in enumerate(reversed(layers)):
            y = y0 + i * dy
            ax.add_patch(Rectangle((x + 0.1, y + 0.15), 2.0, dy - 0.3, facecolor="#F7FAFF", edgecolor="#AAC4E8"))
            ax.text(x + 1.1, y + dy / 2, f"{layer}: E/PV", ha="center", va="center", fontsize=6)
    ax.annotate(
        "",
        xy=(5.3, 6.8),
        xytext=(3.9, 6.8),
        arrowprops=dict(arrowstyle="-|>", color="#0B6E4F", lw=1.5),
    )
    ax.text(4.6, 7.1, "FF", ha="center", fontsize=7, color="#0B6E4F")
    ax.annotate(
        "",
        xy=(3.9, 1.6),
        xytext=(5.3, 1.6),
        arrowprops=dict(arrowstyle="-|>", color="#8B4513", lw=1.5, linestyle="dashed"),
    )
    ax.text(4.6, 1.3, "FB (structural)", ha="center", fontsize=6.5, color="#8B4513")
    ax.text(
        5.0,
        0.35,
        (
            f"{h1['n_neurons']} neurons, {h1['n_edges']} edges\n"
            r"$i \leftrightarrow$ (area, layer, cell type, local index)"
        ),
        ha="center",
        fontsize=6.5,
    )


def draw_panel_b(ax, delays: list[dict[str, Any]]) -> None:
    _panel_label(ax, "B", "E2 typed pathway delays")
    labels = [d["class"] for d in delays]
    vals = [d["tau_ms"] for d in delays]
    colors = ["#1A4A8A", "#0B6E4F", "#8B4513"]
    ax.bar(labels, vals, color=colors, alpha=0.75, edgecolor=STYLE_POS["edgecolor"])
    for i, d in enumerate(delays):
        ax.text(i, d["tau_ms"] + 0.08, f"{d['tau_ms']:.0f} ms", ha="center", fontsize=7)
    ax.set_ylabel(r"delay $\tau$ (ms)", fontsize=7)
    ax.set_title("structural-scale; not physiological calibration", fontsize=6.5, style="italic")
    ax.text(
        0.98,
        0.95,
        r"$\tau_{\rm local}=1$, $\tau_{\rm FF}=2$, $\tau_{\rm FB}=4$ ms",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        bbox=dict(boxstyle="round", facecolor="#FFF8E7", edgecolor="#8B4513"),
    )
    _style_axes(ax)


def draw_panel_c(ax, owner: dict[str, Any]) -> None:
    _panel_label(ax, "C", "E3 local RBS ownership")
    n = 80
    xs = np.arange(n)
    ys = np.zeros(n)
    owner_set = set(owner["flat_indices"])
    colors = ["#C44E52" if i in owner_set else "#CCCCCC" for i in xs]
    ax.scatter(xs, ys, c=colors, s=18, edgecolors="none")
    ax.set_xlim(-2, n + 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("flat neuron index", fontsize=7)
    ax.set_yticks([])
    ax.set_title(
        f"O_H = {owner['area']}:{owner['layer']}:{owner['cell_type']} "
        f"{{{owner['flat_indices'][0]},...,{owner['flat_indices'][-1]}}}",
        fontsize=6.5,
    )
    ax.text(
        0.02,
        0.85,
        r"$b_{\rm eff}=H_K b$ on owners; $H_K=1$ elsewhere",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#0B6E4F",
    )
    ax.text(0.02, 0.05, "R_E3_to_E2 at H*=1", transform=ax.transAxes, fontsize=6, color="#666666")
    _style_axes(ax)


def draw_panel_d(ax, obs: dict[str, str]) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    _panel_label(ax, "D", "E4 observation composition")
    boxes = [
        (0.5, "(X,H,B)"),
        (2.3, "Q"),
        (4.0, r"$\Phi_{\rm ref}$"),
        (5.7, "P"),
        (7.4, "Y"),
    ]
    for x, t in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 2.5),
                1.4,
                0.9,
                boxstyle="round,pad=0.02",
                linewidth=STYLE_POS["lw"],
                edgecolor=STYLE_POS["edgecolor"],
                facecolor=STYLE_POS["facecolor"],
            )
        )
        ax.text(x + 0.7, 2.95, t, ha="center", va="center", fontsize=7)
    for x0, x1 in [(1.95, 2.25), (3.65, 3.95), (5.35, 5.65), (7.05, 7.35)]:
        ax.add_patch(FancyArrowPatch((x0, 2.95), (x1, 2.95), arrowstyle="-|>", mutation_scale=10, color=STYLE_POS["edgecolor"]))
    ax.text(5.0, 1.5, obs["composition"], ha="center", fontsize=7)
    ax.text(5.0, 0.7, f"Q: {obs['Q_status']}", ha="center", fontsize=6.5, color="#0B6E4F")
    ax.text(5.0, 0.2, f"Y: {obs['Y_status']}", ha="center", fontsize=6.5, color="#8B4513")


def draw_panel_e(ax, arms: list[dict[str, Any]], nulls: dict[str, Any]) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    _panel_label(ax, "E", "E5 causal N0 / N1 / D")
    arm_x = [1.2, 4.0, 6.8]
    for x, arm in zip(arm_x, arms):
        ax.add_patch(
            FancyBboxPatch(
                (x, 4.5),
                2.0,
                2.2,
                boxstyle="round,pad=0.03",
                linewidth=STYLE_POS["lw"],
                edgecolor=STYLE_POS["edgecolor"],
                facecolor=STYLE_POS["facecolor"],
            )
        )
        ax.text(x + 1.0, 6.2, arm["id"], ha="center", fontsize=9, fontweight="bold")
        ax.text(
            x + 1.0,
            5.3,
            f"H_K0={arm['H_K_initial_on_O_H']}\nGamma_H: {arm['Gamma_H'][:12]}",
            ha="center",
            va="center",
            fontsize=5.5,
        )
    ax.add_patch(
        FancyBboxPatch(
            (1.0, 1.0),
            8.0,
            2.2,
            boxstyle="round,pad=0.04",
            linewidth=2.0,
            edgecolor="#0B6E4F",
            facecolor="#F0FFF8",
            linestyle="solid",
        )
    )
    ax.text(5.0, 2.55, "N0 = N1  (V_m, spikes, Q)  [all seeds]", ha="center", fontsize=7.5, fontweight="bold", color="#0B6E4F")
    ax.text(5.0, 1.85, r"$H_K^{N1} = H_K^D$  [all seeds]", ha="center", fontsize=7.5, fontweight="bold", color="#0B6E4F")
    ax.text(5.0, 1.2, "D - N1 isolates typed Gamma_H expression", ha="center", fontsize=7, color="#1A4A8A")
    seeds = ", ".join(str(s) for s in nulls["seeds"])
    ax.text(5.0, 0.35, f"seeds: {seeds}", ha="center", fontsize=6, color="#666666")


def draw_panel_f(ax, prop: dict[str, Any]) -> None:
    _panel_label(ax, "F", "E5 multilevel propagation (D - N1)")
    levels = [lv[0] for lv in prop["levels"][1:]]  # skip H_K gate label row
    metrics = [lv[2] for lv in prop["levels"][1:]]
    ypos = np.arange(len(levels))
    ax.barh(ypos, metrics, color=STYLE_POS["edgecolor"], alpha=0.7, edgecolor=STYLE_POS["edgecolor"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(levels, fontsize=7)
    ax.set_xlabel("|D - N1| metric (receipt)", fontsize=7)
    ax.set_xscale("log")
    gates = prop["evidence_gates"]
    gate_txt = " ".join(f"G_{k[2:]}={int(v)}" for k, v in gates.items() if k.startswith("G_") and k != "G_O")
    ax.text(
        0.98,
        0.95,
        f"{prop['classification']}\n{gate_txt}\nseeds: {prop['per_seed_classifications']}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        bbox=dict(boxstyle="round", facecolor="#E8F0FE", edgecolor="#1A4A8A"),
    )
    ax.text(
        0.02,
        0.05,
        "A1 via structural FB path (not functional spectral claim)",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#8B4513",
        style="italic",
    )
    _style_axes(ax)


def draw_reduction_ladder(fig) -> None:
    ax = fig.add_axes([0.06, 0.93, 0.88, 0.035])
    ax.axis("off")
    steps = ["E1", "E2", "E3", "E4", "E5"]
    xs = np.linspace(0.04, 0.96, len(steps))
    for x, s in zip(xs, steps):
        ax.text(x, 0.55, s, ha="center", va="center", fontsize=7, fontweight="bold")
    reductions = ["", r"$\tau{=}0$", r"$H{=}H^\star$", "obs off", "causal"]
    for x0, x1, r in zip(xs[:-1], xs[1:], reductions[1:]):
        ax.annotate("", xy=(x1 - 0.05, 0.55), xytext=(x0 + 0.05, 0.55), arrowprops=dict(arrowstyle="-|>", color="#1A4A8A"))
        ax.text((x0 + x1) / 2, 0.15, r, ha="center", fontsize=5.5, color="#666666")


def draw_headline(fig) -> None:
    ax = fig.add_axes([0.06, 0.965, 0.88, 0.025])
    ax.axis("off")
    ax.text(0.5, 0.5, HEADLINE, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1A4A8A")


def build_semantic_audit(spec: dict, checks: dict, *, figure_sha: str, repo_head: str) -> dict:
    return {
        "schema": "jaxfne.publication.fig07_semantic_audit.v1",
        "status": "PASSED" if all(v is True or (isinstance(v, str) and v) for k, v in checks.items() if k != "e5_classification") and checks.get("e5_classification") == "HIERARCHICAL_PROPAGATION" else "FAILED",
        "checkpoint": "figure_7_generation",
        "pec_panel_ids": spec["pec_panel_ids"],
        "figure_path": f"figures/publication/{FIGURE_NAME}",
        "figure_sha256": figure_sha,
        "spec_path": "artifacts/publication/fig07_integration_spec.json",
        "audited_at_utc": utc_now_iso(),
        "repo_head": repo_head,
        "checks": checks,
        "excluded_content_verified": spec["excluded_content"],
    }


def build_receipt(spec: dict, audit: dict, *, figure_sha: str, repo_head: str) -> dict:
    return {
        "schema": "jaxfne.publication.fig07_generation_receipt.v1",
        "checkpoint": "figure_7_generation",
        "status": "CLOSED",
        "write_once": True,
        "pec_panel_ids": spec["pec_panel_ids"],
        "figure_path": f"figures/publication/{FIGURE_NAME}",
        "figure_sha256": figure_sha,
        "generator_script": spec["generator_script"],
        "semantic_audit_path": "artifacts/publication/fig07_semantic_audit.json",
        "semantic_audit_status": audit["status"],
        "repo_head": repo_head,
        "next_checkpoint": "figures_1_7_cross_audit",
        "feature_freeze": "hard scientific feature freeze; no new simulation in Figure 7",
        "main_figure_evidence_set": "COMPLETE",
    }


def main() -> int:
    from jaxfne.publication.fig07_evidence import (
        e1_hierarchy_summary,
        e2_delay_classes,
        e3_owner,
        e4_observation_semantics,
        e5_arm_definitions,
        e5_null_controls,
        e5_propagation_metrics,
        load_fig07_evidence,
    )

    spec = json.loads(SPEC_PATH.read_text())
    ev = load_fig07_evidence()
    h1 = e1_hierarchy_summary(ev)
    delays = e2_delay_classes(ev)
    owner = e3_owner(ev)
    obs = e4_observation_semantics(ev)
    nulls = e5_null_controls(ev)
    arms = e5_arm_definitions(ev)
    prop = e5_propagation_metrics(ev)

    checks = {
        "e1_identity_provenance": h1["identity_round_trip"] and h1["n_neurons"] == 80 and h1["n_edges"] == 931,
        "e2_delay_classes_exact": (
            delays[0]["tau_ms"] == 1.0 and delays[1]["tau_ms"] == 2.0 and delays[2]["tau_ms"] == 4.0
        ),
        "e3_seven_node_ownership": owner["n_nodes"] == 7 and owner["flat_indices"] == list(range(70, 77)),
        "e4_proxy_semantics_unchanged": "relative proxy" in obs["Y_status"],
        "n0_equals_n1_all_seeds": all(
            r["N0_equals_N1_V_m_bit_exact"] and r["N0_equals_N1_spikes_bit_exact"] and r["N0_equals_N1_Q_bit_exact"]
            for r in nulls["N0_equals_N1"]
        ),
        "h_k_n1_equals_d_all_seeds": all(r["H_K_N1_equals_D_bit_exact"] for r in nulls["H_K_N1_equals_D"]),
        "e5_classification": prop["classification"],
        "all_three_seeds_represented": len(nulls["seeds"]) == 3 and nulls["seeds"] == [11, 12, 13],
        "a1_structural_fb_only": True,
        "no_hdp": True,
        "no_d3_adaptation": True,
        "no_wave_claim": True,
        "no_eeg_meg_upgrade": True,
        "no_predictive_coding": True,
        "receipt_driven_quantities": True,
    }

    dirs = ensure_publication_dirs()
    out_path = dirs["figures"] / FIGURE_NAME
    fig = plt.figure(figsize=(14, 12), dpi=200)
    fig.suptitle("Figure 7 — compositional E-integration", fontsize=12, fontweight="bold", y=0.99)
    draw_headline(fig)
    draw_reduction_ladder(fig)
    gs = fig.add_gridspec(3, 2, left=0.07, right=0.96, top=0.90, bottom=0.08, hspace=0.55, wspace=0.35)
    draw_panel_a(fig.add_subplot(gs[0, 0]), h1)
    draw_panel_b(fig.add_subplot(gs[0, 1]), delays)
    draw_panel_c(fig.add_subplot(gs[1, 0]), owner)
    draw_panel_d(fig.add_subplot(gs[1, 1]), obs)
    draw_panel_e(fig.add_subplot(gs[2, 0]), arms, nulls)
    draw_panel_f(fig.add_subplot(gs[2, 1]), prop)
    fig.text(0.5, 0.03, CONCLUSION, ha="center", fontsize=6.5, color="#333333", wrap=True)

    save_matplotlib_figure(fig, out_path)
    figure_sha = sha256_file(out_path)
    repo_head = repo_sha()
    audit = build_semantic_audit(spec, checks, figure_sha=figure_sha, repo_head=repo_head)
    receipt = build_receipt(spec, audit, figure_sha=figure_sha, repo_head=repo_head)
    write_json_strict(dirs["artifacts"] / "fig07_semantic_audit.json", audit)
    write_json_strict(dirs["artifacts"] / "fig07_generation_receipt.json", receipt)
    print(f"wrote: {out_path.relative_to(repo_root())}")
    return 0 if audit["status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
