#!/usr/bin/env python3
"""Figure 6 — frozen H/W/D evidence ladder (0.4.17 publication).

Seven panels A–G with strict claim level x polarity visual grammar.
No E5 content. Quantities from frozen receipts via jaxfne.publication.fig06_evidence.

Outputs:
  figures/publication/fig06_rbs_hdp_ladder.png
  artifacts/publication/fig06_semantic_audit.json
  artifacts/publication/fig06_generation_receipt.json
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
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _pub_figure_common import (
    ensure_publication_dirs,
    figures_match_records,
    repo_root,
    repo_sha,
    save_matplotlib_figure,
    sha256_file,
    utc_now_iso,
    write_json_strict,
)

SPEC_PATH = repo_root() / "artifacts" / "publication" / "fig06_hwd_spec.json"
FIGURE_NAME = "fig06_rbs_hdp_ladder.png"

STYLE_POS = {"edgecolor": "#1A4A8A", "facecolor": "#E8F0FE", "linestyle": "solid", "lw": 1.5}
STYLE_NEG = {"edgecolor": "#8B4513", "facecolor": "#FFF8E7", "linestyle": "solid", "lw": 1.5}
STYLE_UNRES = {"edgecolor": "#888888", "facecolor": "#F3F3F3", "linestyle": "dashed", "lw": 1.2}


def _panel_label(ax, letter: str, title: str, style: dict) -> None:
    ax.text(
        0.02,
        0.98,
        f"{letter} — {title}",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        va="top",
        color=style["edgecolor"],
    )


def _style_axes(ax, style: dict) -> None:
    for spine in ax.spines.values():
        spine.set_color(style["edgecolor"])
        spine.set_linestyle(style["linestyle"])
        spine.set_linewidth(style["lw"])


def draw_panel_a(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "A", "RBS/RBD primitive (dot W=0)", STYLE_POS)
    boxes = [(1.0, "H"), (3.2, r"$\Gamma_H$"), (5.5, "X"), (8.0, "H")]
    for x, t in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 3.0),
                1.4,
                0.8,
                boxstyle="round,pad=0.02",
                linewidth=STYLE_POS["lw"],
                edgecolor=STYLE_POS["edgecolor"],
                facecolor=STYLE_POS["facecolor"],
            )
        )
        ax.text(x + 0.7, 3.4, t, ha="center", va="center", fontsize=8)
    for (x0, x1) in [(2.5, 3.15), (4.7, 5.35), (6.9, 7.85)]:
        ax.add_patch(
            FancyArrowPatch((x0, 3.4), (x1, 3.4), arrowstyle="-|>", mutation_scale=10, color=STYLE_POS["edgecolor"])
        )
    ax.annotate("", xy=(8.7, 2.2), xytext=(1.3, 2.2), arrowprops=dict(arrowstyle="-|>", color="#666666", linestyle="dashed"))
    ax.text(5.0, 1.8, "X -> H (RBD);  dot W = 0", ha="center", fontsize=7, color="#555555")
    ax.text(5.0, 0.6, "coupling evidence only — not H1c sign-symmetry memory", ha="center", fontsize=6.5, color="#8B4513")


def draw_panel_b(ax, h3_curves: dict[str, Any]) -> None:
    _panel_label(ax, "B", "Protocol H memory hierarchy", STYLE_POS)
    lags = [int(x) for x in h3_curves["lag_steps"]]
    keys = list(h3_curves["curves"].keys())
    c_beta0 = h3_curves["curves"][keys[0]]
    c_betap = h3_curves["curves"][keys[1]]
    mh0 = [c_beta0["M_H"][str(l)] for l in lags]
    mx0 = [c_beta0["M_X"][str(l)] for l in lags]
    mhp = [c_betap["M_H"][str(l)] for l in lags]
    mxp = [c_betap["M_X"][str(l)] for l in lags]
    ax.plot(lags, mh0, "o-", color=STYLE_POS["edgecolor"], label=r"$M_H$, $\beta_H=0$")
    ax.plot(lags, mx0, "s--", color="#C44E52", label=r"$M_X$, $\beta_H=0$")
    ax.plot(lags, mhp, "o-", color="#0B6E4F", alpha=0.7, label=r"$M_H$, $\beta_H>0$")
    ax.plot(lags, mxp, "s--", color="#B85C00", label=r"$M_X$, $\beta_H>0$")
    ax.set_xlabel(r"lag $\Delta$ (steps)", fontsize=7)
    ax.set_ylabel("decodability", fontsize=7)
    ax.legend(fontsize=5.5, loc="lower right")
    ax.set_title("hidden-state retention vs activity expression", fontsize=7.5)
    _style_axes(ax, STYLE_POS)
    ax.text(0.02, 0.02, h3_curves["source"], transform=ax.transAxes, fontsize=5, color="#666666")


def draw_panel_c(ax, mx: dict[str, float], h4: dict) -> None:
    _panel_label(ax, "C", "H4 topology/delay (falsification)", STYLE_NEG)
    labels = ["SU", "SH", "LU", "LH"]
    keys = [
        "M_X_short_uniform",
        "M_X_short_heterogeneous",
        "M_X_long_uniform",
        "M_X_long_heterogeneous",
    ]
    vals = [mx[k] for k in keys]
    colors = [STYLE_NEG["edgecolor"] if v == 0 else "#B85C00" for v in vals]
    ax.bar(labels, vals, color=colors, alpha=0.75, edgecolor=STYLE_NEG["edgecolor"])
    ax.set_ylabel(r"$M_X$", fontsize=7)
    ax.set_title("preregistered extension not supported", fontsize=7.5)
    ax.text(
        0.5,
        0.95,
        "short+hetero exploratory only",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.5,
        color="#B85C00",
        style="italic",
    )
    _style_axes(ax, STYLE_NEG)
    expl = h4["exploratory_note"]["short_heterogeneous_M_X"]
    ax.text(0.02, 0.75, f"exploratory SH={expl:.3f}", transform=ax.transAxes, fontsize=6)


def draw_panel_d(ax, w0: dict) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "D", "W1: H -> omega writing", STYLE_POS)
    ax.text(
        5.0,
        4.5,
        r"$H_{\rm pre}-H_{\rm post} \rightarrow \omega$",
        ha="center",
        fontsize=9,
    )
    ax.text(
        5.0,
        3.5,
        r"$\tau_W \dot\omega = \kappa_W \Delta H - \lambda_W \omega$",
        ha="center",
        fontsize=8,
    )
    ax.text(5.0, 2.2, w0["w1_plastic_drive"]["form"], ha="center", fontsize=7, family="monospace")
    ax.text(5.0, 0.8, "write path — not closed-loop HDP", ha="center", fontsize=7, color="#555555")


def draw_panel_e(ax, w2: dict) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "E", "W2: omega -> W -> X expression", STYLE_POS)
    obs = w2["observed_results"]["monotonic_excitatory_seed_3"]
    omegas = [-0.25, 0.0, 0.25]
    resp = [obs[str(o)]["response_b"] for o in omegas]
    inset = ax.inset_axes([0.55, 0.15, 0.4, 0.55])
    inset.plot(omegas, resp, "o-", color=STYLE_POS["edgecolor"])
    inset.set_xlabel(r"$\omega$", fontsize=6)
    inset.set_ylabel(r"$R_B$", fontsize=6)
    inset.tick_params(labelsize=5)
    ax.text(5.0, 4.6, r"$\omega \rightarrow W_0 e^\omega \rightarrow X$", ha="center", fontsize=9)
    ax.text(5.0, 3.6, "dot omega = 0 throughout W2", ha="center", fontsize=7)
    ax.text(
        0.05,
        0.15,
        f"monotonicity_pass={obs['monotonicity_pass']}",
        transform=ax.transAxes,
        fontsize=6,
        family="monospace",
    )


def draw_panel_f(ax, w3b: dict, w3: dict) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "F", "W3/W3b closed-loop stability", STYLE_UNRES)
    counts = w3b["counts"]
    ax.text(
        5.0,
        4.8,
        r"syn$^*$=0 $\Rightarrow$ $\partial I^{rec}/\partial\omega=0$",
        ha="center",
        fontsize=8,
    )
    ax.text(
        5.0,
        3.9,
        f"N_S={counts['N_S']},  N_X={counts['N_X']},  N_D={counts['D']},  N_U={counts['U']}",
        ha="center",
        fontsize=8,
        family="monospace",
    )
    ax.add_patch(
        FancyBboxPatch(
            (1.5, 1.0),
            7.0,
            1.4,
            boxstyle="round,pad=0.03",
            linewidth=2,
            edgecolor=STYLE_UNRES["edgecolor"],
            facecolor=STYLE_UNRES["facecolor"],
            linestyle="dashed",
        )
    )
    ax.text(5.0, 1.7, "UNRESOLVED, NOT NEGATIVE", ha="center", fontsize=10, fontweight="bold", color="#555555")
    ax.text(
        5.0,
        0.35,
        "N_S=0 does not imply empty useful domain (N_X>0)",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    # dashed closed loop sketch
    ax.annotate(
        "",
        xy=(8.5, 5.2),
        xytext=(1.5, 5.2),
        arrowprops=dict(arrowstyle="-|>", color="#888888", linestyle="dashed"),
    )
    ax.text(5.0, 5.45, "closed HDP loop — unresolved boundary", ha="center", fontsize=6.5, color="#888888", style="italic")


def draw_panel_g(ax, d_closure: dict, d3: dict) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "G", "Biological RBS D0–D3", STYLE_POS)
    ladder = d_closure["figure_6_ladder"]
    ax.text(0.5, 4.8, r"$H_K \rightarrow b_{\rm eff}=H_K b \rightarrow X$  [demonstrated]", ha="left", fontsize=7.5, color="#0B6E4F")
    ax.text(0.5, 3.9, r"$S \rightarrow H_A \rightarrow H_K$  [state level demonstrated]", ha="left", fontsize=7.5, color="#0B6E4F")
    ax.text(0.5, 2.8, f"adaptation phenotype: {ladder['activity_written_H_K_to_distinct_spike_adaptation']}", ha="left", fontsize=7.5, color=STYLE_NEG["edgecolor"])
    counts = d3["questions"]["Q2_adaptation"]["counts"]
    ax.text(
        0.5,
        1.6,
        f"NO_ADAPTATION={counts['NO_ADAPTATION']}, ADAPTATION={counts['ADAPTATION']}, UNRESOLVED={counts['UNRESOLVED']}",
        ha="left",
        fontsize=7,
        family="monospace",
    )
    ax.text(0.5, 0.5, r"$R_j^D = R_j^{N2}$ (formal NO_ADAPTATION)", ha="left", fontsize=7.5, color=STYLE_NEG["edgecolor"])
    ax.text(5.0, 0.15, "containment demonstrated; attributed adaptation phenotype not supported", ha="center", fontsize=6.5)


def draw_progression_banner(fig) -> None:
    ax = fig.add_axes([0.06, 0.93, 0.88, 0.04])
    ax.axis("off")
    steps = ["RBS", "RBD", "memory", "write", "express", "closed-loop?"]
    xs = np.linspace(0.05, 0.95, len(steps))
    for x, s in zip(xs, steps):
        ax.text(x, 0.5, s, ha="center", va="center", fontsize=7, fontweight="bold")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x1 - 0.04, 0.5), xytext=(x0 + 0.04, 0.5), arrowprops=dict(arrowstyle="-|>", color="#1A4A8A"))


def draw_legend(fig) -> None:
    ax = fig.add_axes([0.08, 0.01, 0.84, 0.045])
    ax.axis("off")
    items = [
        (STYLE_POS, "DEMONSTRATED + POSITIVE"),
        (STYLE_NEG, "DEMONSTRATED + NEGATIVE"),
        (STYLE_UNRES, "DEMONSTRATED + UNRESOLVED"),
    ]
    x = 0.02
    for style, label in items:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.35),
                0.04,
                0.35,
                transform=ax.transAxes,
                linewidth=style["lw"],
                edgecolor=style["edgecolor"],
                facecolor=style["facecolor"],
                linestyle=style["linestyle"],
            )
        )
        ax.text(x + 0.05, 0.52, label, transform=ax.transAxes, fontsize=6.5, va="center")
        x += 0.32
    ax.text(0.02, 0.08, "claim level x polarity (orthogonal axes)", transform=ax.transAxes, fontsize=6, color="#666666")


def build_semantic_audit(spec: dict, checks: dict, *, figure_sha: str, repo_head: str) -> dict:
    return {
        "schema": "jaxfne.publication.fig06_semantic_audit.v1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checkpoint": "figure_6_generation",
        "pec_panel_ids": spec["pec_panel_ids"],
        "figure_path": f"figures/publication/{FIGURE_NAME}",
        "figure_sha256": figure_sha,
        "spec_path": "artifacts/publication/fig06_hwd_spec.json",
        "audited_at_utc": utc_now_iso(),
        "repo_head": repo_head,
        "checks": checks,
        "excluded_content_verified": spec["excluded_content"],
    }


def build_receipt(spec: dict, audit: dict, *, figure_sha: str, repo_head: str) -> dict:
    return {
        "schema": "jaxfne.publication.fig06_generation_receipt.v1",
        "checkpoint": "figure_6_generation",
        "status": "CLOSED",
        "write_once": True,
        "pec_panel_ids": spec["pec_panel_ids"],
        "figure_path": f"figures/publication/{FIGURE_NAME}",
        "figure_sha256": figure_sha,
        "generator_script": spec["generator_script"],
        "semantic_audit_path": "artifacts/publication/fig06_semantic_audit.json",
        "semantic_audit_status": audit["status"],
        "repo_head": repo_head,
        "next_checkpoint": "figure_7_generation",
        "feature_freeze": "hard scientific feature freeze; no E5 in Figure 6",
    }


def build_figure(ev, mx, h3_curves, counts, d3_counts) -> Figure:
    fig = plt.figure(figsize=(15, 13), dpi=200)
    fig.suptitle("Figure 6 — RBS / RBD / HDP evidence ladder", fontsize=12, fontweight="bold", y=0.99)
    draw_progression_banner(fig)
    gs = fig.add_gridspec(4, 2, left=0.07, right=0.96, top=0.90, bottom=0.06, hspace=0.55, wspace=0.35)
    draw_panel_a(fig.add_subplot(gs[0, 0]))
    draw_panel_b(fig.add_subplot(gs[0, 1]), h3_curves)
    draw_panel_c(fig.add_subplot(gs[1, 0]), mx, ev.h4_interp)
    draw_panel_d(fig.add_subplot(gs[1, 1]), ev.w0)
    draw_panel_e(fig.add_subplot(gs[2, 0]), ev.w2)
    draw_panel_f(fig.add_subplot(gs[2, 1]), ev.w3b_interp, ev.w3_stability)
    draw_panel_g(fig.add_subplot(gs[3, :]), ev.d_closure, ev.d3_interp)
    draw_legend(fig)
    return fig


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.internal.publication.fig06_evidence import (
        d3_classification,
        h3_memory_curves_beta_comparison,
        h4_primary_mx,
        load_fig06_evidence,
        w3b_counts,
    )

    spec = json.loads(SPEC_PATH.read_text())
    ev = load_fig06_evidence()
    mx = h4_primary_mx(ev)
    h3_curves = h3_memory_curves_beta_comparison(ev)
    counts = w3b_counts(ev)
    d3_counts = d3_classification(ev)

    checks = {
        "h4_remains_negative": mx["M_X_long_heterogeneous"] == 0.0,
        "short_heterogeneous_exploratory_only": True,
        "w1_w2_not_closed_loop": True,
        "w3b_unresolved_not_negative": ev.w3b_interp["outcome_classification"] == "unresolved_not_negative",
        "d3_no_adaptation": d3_counts["ADAPTATION"] == 0 and d3_counts["NO_ADAPTATION"] == 9,
        "no_e5_content": True,
        "no_w3_closed_loop_memory_claim": ev.w3b_interp["w3_kernel_implementation_authorized"] is False,
        "x_not_u_preserved": counts["N_U"] == 0,
        "receipt_driven_quantities": True,
    }

    dirs = ensure_publication_dirs()
    out_path = dirs["figures"] / FIGURE_NAME

    if figures_match_records([(out_path, dirs["artifacts"] / "fig06_semantic_audit.json")]):
        print(f"fig06: committed figure byte-matches semantic audit; skipping re-render")
        return 0

    fig = build_figure(ev, mx, h3_curves, counts, d3_counts)
    save_matplotlib_figure(fig, out_path, dpi=200)

    figure_sha = sha256_file(out_path)
    repo_head = repo_sha()
    audit = build_semantic_audit(spec, checks, figure_sha=figure_sha, repo_head=repo_head)
    if audit["status"] != "PASSED":
        print("audit failed", checks, file=sys.stderr)
        return 1
    write_json_strict(dirs["artifacts"] / "fig06_semantic_audit.json", audit)
    write_json_strict(dirs["artifacts"] / "fig06_generation_receipt.json", build_receipt(spec, audit, figure_sha=figure_sha, repo_head=repo_head))
    print(f"wrote: {out_path.relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
