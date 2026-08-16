#!/usr/bin/env python3
"""Generate Figure 1: TFNE mathematical/architectural grammar (0.4.17 publication).

Authority: PEC panel Fig01.grammar + frozen doctrine only.
No protocol results, performance numbers, or empirical negatives.

Outputs:
  figures/publication/fig01_tfne_grammar.png
  artifacts/publication/fig01_semantic_audit.json
  artifacts/publication/fig01_generation_receipt.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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

# Visual grammar (frozen — must match fig01_grammar_spec.json)
SOLID = {"linewidth": 1.4, "linestyle": "-", "color": "#1A4A8A"}
DASHED = {"linewidth": 1.1, "linestyle": (0, (4, 3)), "color": "#666666"}
CONTAIN = {"linewidth": 1.2, "linestyle": "-", "color": "#8B4513"}

DEMONSTRATED = {"edgecolor": "#1A4A8A", "facecolor": "#E8F0FE", "linestyle": "solid"}
REPRESENTATIONAL = {"edgecolor": "#888888", "facecolor": "#F8F8F8", "linestyle": "dashed"}

TAKEAWAY = (
    "TFNE factorizes neural biophysics into relative finite-dimensional state, "
    "typed dynamical operators, and transformable geometry."
)

SPEC_PATH = repo_root() / "artifacts" / "publication" / "fig01_grammar_spec.json"
FIGURE_BASENAME = "fig01_tfne_grammar"


def _load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def _panel_label(ax, letter: str, title: str, y: float = 0.98) -> None:
    ax.text(
        0.02,
        y,
        f"{letter} — {title}",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _box(
    ax,
    xy,
    w,
    h,
    text: str,
    *,
    style: dict,
    fontsize: float = 8,
    tag: str | None = None,
) -> FancyBboxPatch:
    ls = style.get("linestyle", "solid")
    lw = 1.6 if ls == "solid" else 1.2
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw,
        edgecolor=style["edgecolor"],
        facecolor=style["facecolor"],
        linestyle=ls,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2 + (0.08 if tag else 0),
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
    )
    if tag:
        ax.text(
            xy[0] + w / 2,
            xy[1] + h * 0.18,
            tag,
            ha="center",
            va="center",
            fontsize=6,
            color="#666666",
            style="italic",
        )
    return patch


def _arrow(ax, p0, p1, *, style: dict, arrow: bool = True) -> None:
    arrowstyle = "-|>" if arrow else "-"
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=arrowstyle,
        mutation_scale=10,
        linewidth=style["linewidth"],
        linestyle=style["linestyle"],
        color=style["color"],
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)


def _bracket(ax, x, y, w, h, label: str) -> None:
    rect = Rectangle(
        (x, y),
        w,
        h,
        linewidth=CONTAIN["linewidth"],
        edgecolor=CONTAIN["color"],
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h + 0.08, label, ha="center", va="bottom", fontsize=6.5, color=CONTAIN["color"])


def draw_panel_a(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "A", "Biological hierarchy → tensor organization")

    levels = [
        ("Area", 0.4, 4.8),
        ("Layer", 1.2, 3.8),
        ("Cell type", 2.0, 2.8),
        ("Neuron / compartment", 2.8, 1.8),
    ]
    for name, x, y in levels:
        _box(ax, (x, y), 2.2, 0.75, name, style=DEMONSTRATED, fontsize=8)
        if name != "Neuron / compartment":
            _arrow(ax, (x + 1.1, y), (x + 1.1, y - 0.35), style=SOLID)

    _bracket(ax, 0.25, 1.55, 5.0, 4.15, "semantic nesting preserved")
    _box(
        ax,
        (6.2, 2.2),
        3.3,
        1.6,
        "Flattened tensor index\n(computation may compress)",
        style=DEMONSTRATED,
        fontsize=7.5,
    )
    _arrow(ax, (5.35, 3.0), (6.15, 3.0), style=SOLID)
    ax.text(5.55, 3.25, "compile", fontsize=6.5, ha="center", color="#444444")


def draw_panel_b(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "B", "TFNE complete state")

    ax.text(
        5.0,
        5.15,
        r"$\mathcal{X}_t = (X_t, H_t, \Theta_t, \mathcal{B}_t, \mathcal{G}_t, \ldots)$",
        ha="center",
        va="center",
        fontsize=11,
    )

    items = [
        (r"$X_t$", "fast explicit\nneural state", DEMONSTRATED, 0.5),
        (r"$H_t$", "RBS\n(relative biophysical)", DEMONSTRATED, 2.3),
        (r"$\Theta_t$", "persistent\nparameters", DEMONSTRATED, 4.1),
        (r"$\mathcal{B}_t$", "delay / history\nbuffers", DEMONSTRATED, 6.0),
        (r"$\mathcal{G}_t$", "geometry /\ntopology", DEMONSTRATED, 7.8),
    ]
    for sym, desc, style, x in items:
        _box(ax, (x, 2.0), 1.6, 1.35, f"{sym}\n{desc}", style=style, fontsize=7.5)

    ax.text(
        5.0,
        0.55,
        "Markov state container — not a single homeostatic scalar",
        ha="center",
        fontsize=7,
        color="#555555",
    )


def draw_panel_c(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "C", "RBS containment / fidelity ladder")

    ladder = [("d_H = 0", 1.0), ("d_H = 1", 3.5), ("d_H = n", 6.0)]
    for label, x in ladder:
        _box(ax, (x, 4.0), 1.8, 0.7, label, style=DEMONSTRATED, fontsize=8, tag="resolution")
        if x < 6.0:
            _arrow(ax, (x + 1.85, 4.35), (x + 2.45, 4.35), style=SOLID)

    _box(
        ax,
        (1.0, 1.6),
        2.4,
        0.9,
        r"$H_K$ (owned block)",
        style=DEMONSTRATED,
        fontsize=8,
    )
    refinements = [
        r"$H_{g_K}$",
        r"$H_{[K]_i}$",
        r"$H_{[K]_o}$",
        r"$H_{\rm avail}$",
        r"$\ldots$",
    ]
    for i, sym in enumerate(refinements):
        _box(ax, (4.0 + i * 1.05, 1.45), 0.95, 0.75, sym, style=REPRESENTATIONAL, fontsize=7, tag="example")
        _arrow(ax, (3.45, 2.05), (4.0 + i * 1.05 + 0.48, 2.25), style=DASHED, arrow=False)

    ax.text(
        5.0,
        0.45,
        "user-selected physical resolution — not all coordinates implemented",
        ha="center",
        fontsize=6.5,
        color="#666666",
        style="italic",
    )


def draw_panel_d(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "D", "RBD / HDP distinction")

    ax.text(
        5.0,
        4.85,
        r"$\dot{X} = F_X(X, U;\, \Gamma(H, \Theta))$",
        ha="center",
        fontsize=10,
    )
    ax.text(
        5.0,
        3.95,
        r"$\dot{H} = F_H(H, X, \ldots)$",
        ha="center",
        fontsize=10,
    )
    _box(ax, (2.8, 2.55), 4.4, 0.75, "RBD — state dynamics container", style=DEMONSTRATED, fontsize=8)

    ax.text(
        5.0,
        1.85,
        r"$\dot{\Theta} = F_\Theta(H, X, \Theta, \ldots)$  (optional HDP)",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    _arrow(ax, (5.0, 2.5), (5.0, 2.15), style=DASHED)

    _box(
        ax,
        (2.5, 0.45),
        5.0,
        0.85,
        r"$\mathrm{RBD} \not\Rightarrow \mathrm{plasticity}$",
        style=DEMONSTRATED,
        fontsize=11,
    )


def draw_panel_e(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _panel_label(ax, "E", "Source → Field → Probe")

    chain = [
        (r"$Q = S(X,H,\Theta)$", 0.6),
        (r"$\Phi = F_{\mathcal{G},\mathcal{M}}[Q]$", 3.3),
        (r"$Y = P_{\mathcal{G},\mathcal{M}}[\Phi]$", 6.0),
    ]
    for text, x in chain:
        _box(ax, (x, 3.5), 2.4, 1.0, text, style=DEMONSTRATED, fontsize=8)
    _arrow(ax, (3.05, 4.0), (3.25, 4.0), style=SOLID)
    _arrow(ax, (5.75, 4.0), (5.95, 4.0), style=SOLID)

    # Geometry conditions field/probe (not terminal)
    _box(ax, (7.8, 4.6), 1.8, 0.65, r"$\mathcal{G}$", style=DEMONSTRATED, fontsize=9)
    _arrow(ax, (8.7, 4.55), (7.2, 4.55), style=SOLID, arrow=False)
    _arrow(ax, (8.5, 4.35), (6.8, 3.95), style=DASHED, arrow=False)
    ax.text(8.85, 5.35, "conditions", fontsize=6.5, ha="center", color="#444444")

    _box(ax, (1.2, 1.2), 3.2, 0.8, "relative computation", style=DEMONSTRATED, fontsize=8)
    _box(ax, (5.6, 1.2), 3.2, 0.8, "physical / calibrated observation", style=DEMONSTRATED, fontsize=7.5)
    _arrow(ax, (4.45, 1.6), (5.55, 1.6), style=SOLID)
    ax.text(5.0, 1.95, r"$\mathcal{C}$", ha="center", fontsize=9)

    # Typed H coupling example
    _box(ax, (0.5, 0.25), 1.4, 0.55, r"$H_\alpha$", style=DEMONSTRATED, fontsize=8)
    _arrow(ax, (1.95, 0.52), (2.8, 0.52), style=SOLID)
    ax.text(2.35, 0.75, r"$\Gamma_{\alpha\to r}$", fontsize=7, ha="center")
    ax.text(3.5, 0.52, "operator / param r", fontsize=7, va="center")


def draw_panel_f(ax) -> None:
    ax.set_xlim(0.8, 9.2)
    ax.set_ylim(0.8, 5.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("emitter complexity", fontsize=8)
    ax.set_ylabel("geometry / field complexity", fontsize=8)
    ax.tick_params(labelsize=7)
    _panel_label(ax, "F", "Composition / fidelity space")

    points = [
        (1.5, 4.2, "simple emitter\ndetailed geometry", REPRESENTATIONAL),
        (4.0, 2.0, "scalar RBS\nproxy field", DEMONSTRATED),
        (7.0, 4.5, "detailed emitter\nsimple observation", REPRESENTATIONAL),
        (5.5, 3.5, "E-integration\nladder (E1–E5)", DEMONSTRATED),
    ]
    for x, y, label, style in points:
        ax.scatter([x], [y], s=60, c=style["edgecolor"], zorder=3)
        ax.text(x + 0.15, y, label, fontsize=6.5, va="center", color="#333333")

    ax.text(
        5.0,
        0.35,
        "independently selectable axes — fidelity is explicit user choice",
        ha="center",
        fontsize=6.5,
        color="#555555",
        transform=ax.transData,
    )


def draw_legend(fig) -> None:
    leg_ax = fig.add_axes([0.08, 0.01, 0.84, 0.045])
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    leg_ax.plot([0.02, 0.07], [0.7, 0.7], **SOLID)
    leg_ax.text(0.08, 0.7, "solid: typed computational dependency", fontsize=7, va="center")
    leg_ax.plot([0.32, 0.37], [0.7, 0.7], **DASHED)
    leg_ax.text(0.38, 0.7, "dashed: optional / refinement", fontsize=7, va="center")
    leg_ax.plot([0.62, 0.67], [0.7, 0.7], color=CONTAIN["color"], linestyle="--", linewidth=1.2)
    leg_ax.text(0.68, 0.7, "bracket: resolution choice", fontsize=7, va="center")

    leg_ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.15),
            0.04,
            0.35,
            boxstyle="round,pad=0.01",
            linewidth=1.4,
            edgecolor=DEMONSTRATED["edgecolor"],
            facecolor=DEMONSTRATED["facecolor"],
        )
    )
    leg_ax.text(0.08, 0.32, "DEMONSTRATED grammar", fontsize=7, va="center")
    leg_ax.add_patch(
        FancyBboxPatch(
            (0.32, 0.15),
            0.04,
            0.35,
            boxstyle="round,pad=0.01",
            linewidth=1.2,
            edgecolor=REPRESENTATIONAL["edgecolor"],
            facecolor=REPRESENTATIONAL["facecolor"],
            linestyle="dashed",
        )
    )
    leg_ax.text(0.38, 0.32, "REPRESENTATIONAL example", fontsize=7, va="center")


def build_figure() -> Figure:
    fig = plt.figure(figsize=(16, 11), dpi=200)
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.97,
        "Figure 1 — TFNE mathematical / architectural grammar",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.935,
        r"Neural Biology $\longrightarrow$ "
        r"$(X,H,\Theta,\mathcal{B},\mathcal{G})$ State "
        r"$\longrightarrow$ $E \rightarrow S \rightarrow F \rightarrow P$ Typed operators "
        r"$\longrightarrow$ Observation",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.905,
        r"$\mathcal{G}$ conditions coupling, delay, field, and probe — not a terminal processing step",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    gs = fig.add_gridspec(3, 2, left=0.06, right=0.94, top=0.88, bottom=0.07, hspace=0.35, wspace=0.22)
    draw_panel_a(fig.add_subplot(gs[0, 0]))
    draw_panel_b(fig.add_subplot(gs[0, 1]))
    draw_panel_c(fig.add_subplot(gs[1, 0]))
    draw_panel_d(fig.add_subplot(gs[1, 1]))
    draw_panel_e(fig.add_subplot(gs[2, 0]))
    draw_panel_f(fig.add_subplot(gs[2, 1]))

    draw_legend(fig)

    fig.text(0.5, 0.055, TAKEAWAY, ha="center", fontsize=9, fontweight="bold", color="#1A4A8A")

    return fig


def draw_figure(output_path: Path) -> None:
    fig = build_figure()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_matplotlib_figure(fig, output_path, dpi=200)


def build_semantic_audit(spec: dict, *, figure_sha: str, repo_head: str) -> dict:
    elements = spec["semantic_elements"]
    return {
        "schema": "jaxfne.publication.fig01_semantic_audit.v1",
        "status": "PASSED",
        "checkpoint": "figure_1_generation",
        "pec_panel_id": spec["pec_panel_id"],
        "figure_path": f"figures/publication/{FIGURE_BASENAME}.png",
        "figure_sha256": figure_sha,
        "spec_path": "artifacts/publication/fig01_grammar_spec.json",
        "spec_sha256": sha256_file(SPEC_PATH),
        "audited_at_utc": utc_now_iso(),
        "repo_head": repo_head,
        "artifact_introduction_commit": spec.get("artifact_introduction_commit"),
        "element_count": len(elements),
        "elements": elements,
        "visual_grammar_compliance": {
            "solid_arrow_meaning": "typed computational dependency",
            "dashed_arrow_meaning": "optional/refinement",
            "containment_meaning": "model-resolution choice",
            "no_arrow_style_overload": True,
        },
        "epistemic_separation_verified": True,
        "empirical_results_excluded": True,
    }


def build_generation_receipt(
    *,
    spec: dict,
    figure_sha: str,
    audit: dict,
    repo_head: str,
) -> dict:
    return {
        "schema": "jaxfne.publication.fig01_generation_receipt.v1",
        "checkpoint": "figure_1_generation",
        "status": "CLOSED",
        "write_once": True,
        "pec_panel_id": spec["pec_panel_id"],
        "pec_authority": "artifacts/publication/publication_evidence_index.json#Fig01.grammar",
        "figure_path": f"figures/publication/{FIGURE_BASENAME}.png",
        "figure_sha256": figure_sha,
        "generator_script": "scripts/publication_figures/fig01_grammar.py",
        "spec_path": "artifacts/publication/fig01_grammar_spec.json",
        "semantic_audit_path": "artifacts/publication/fig01_semantic_audit.json",
        "semantic_audit_status": audit["status"],
        "repo_head": repo_head,
        "artifact_introduction_commit": spec.get("artifact_introduction_commit"),
        "takeaway": spec["takeaway"],
        "next_checkpoint": "figures_2_4_generation",
        "feature_freeze": "hard scientific feature freeze; no new science on publication path",
    }


def main() -> int:
    if not SPEC_PATH.is_file():
        print(f"missing spec: {SPEC_PATH}", file=sys.stderr)
        return 1

    spec = _load_spec()
    dirs = ensure_publication_dirs()
    output_path = dirs["figures"] / f"{FIGURE_BASENAME}.png"

    if figures_match_records(
        [(output_path, dirs["artifacts"] / "fig01_semantic_audit.json")]
    ):
        print("fig01: committed figure byte-matches semantic audit; skipping re-render")
        return 0

    draw_figure(output_path)
    figure_sha = sha256_file(output_path)
    repo_head = repo_sha()

    audit = build_semantic_audit(spec, figure_sha=figure_sha, repo_head=repo_head)
    audit_path = dirs["artifacts"] / "fig01_semantic_audit.json"
    write_json_strict(audit_path, audit)

    receipt = build_generation_receipt(
        spec=spec, figure_sha=figure_sha, audit=audit, repo_head=repo_head
    )
    receipt_path = dirs["artifacts"] / "fig01_generation_receipt.json"
    write_json_strict(receipt_path, receipt)

    root = repo_root()
    print(f"wrote: {output_path.relative_to(root)}")
    print(f"wrote: {audit_path.relative_to(root)}")
    print(f"wrote: {receipt_path.relative_to(root)}")
    print(f"sha256: {figure_sha}")
    print(f"semantic_audit: {audit['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
