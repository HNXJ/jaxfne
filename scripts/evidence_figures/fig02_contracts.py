#!/usr/bin/env python3
"""Generate Figure 2: TFNE source-to-field contract and evidence ladder.

Outputs:
  figures/evidence/fig02_source_field_contracts.png
  outputs/evidence/fig02_source_field_contracts_manifest.json
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _figure_common import (
    save_matplotlib_figure,
    ensure_evidence_dirs,
    repo_root,
    save_figure_manifest,
    sha256_file,
)


SOURCE_FILES = [
    "scripts/evidence_figures/fig02_contracts.py",
    "scripts/evidence_figures/_figure_common.py",
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "fig02_source_field_contracts.png"

TITLE = "TFNE source-to-field contract and evidence ladder"

SCOPE_STATUS = (
    "Default jaxfne evidence path: laminar proxy readouts, not calibrated physical fields"
)

EQUATIONS_SHOWN = [
    "J_e = -sigma_e grad phi_e",
    "div J_e = q",
    "CSD = div J_e",
    "Y = S W^T  (laminar proxy / readout path)",
]

EVIDENCE_LADDER = [
    "P0: proxy projection (default jaxfne v0.3.x)",
    "P1: lead-field-like projection",
    "P2: boundary-normalized kernels",
    "P3: discrete volume-conductor solve",
    "P4: differentiable adjoint solve",
    "P5: external validation",
]


def _panel(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    facecolor: str = "#F8FAFF",
    edgecolor: str = "#1A4A8A",
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transData,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.12,
        y + h - 0.18,
        title,
        fontsize=9,
        fontweight="bold",
        va="top",
        color="#1A1A1A",
    )


def draw_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(7.0, 9.55, TITLE, ha="center", va="center", fontsize=15, fontweight="bold")

    # Panel A — source bookkeeping
    pa_x, pa_y, pa_w, pa_h = 0.45, 6.55, 6.35, 2.55
    _panel(ax, pa_x, pa_y, pa_w, pa_h, "Panel A — source bookkeeping")
    panel_a_lines = [
        "Emitter state X(t)",
        "Source bridge S[X]",
        "Source tensor q  or  S_k",
        "",
        "Mode A: total membrane-current / source proxy",
        "Mode B: decomposed source terms",
        "",
        "Gate: one source mode per run",
    ]
    for i, line in enumerate(panel_a_lines):
        weight = "bold" if line.startswith("Gate:") else "normal"
        ax.text(pa_x + 0.2, pa_y + pa_h - 0.55 - i * 0.28, line, fontsize=7.5, va="top", fontweight=weight)

    # Panel B — physical target equations
    pb_x, pb_y, pb_w, pb_h = 7.2, 6.55, 6.35, 2.55
    _panel(ax, pb_x, pb_y, pb_w, pb_h, "Panel B — physical target equations (reference)")
    equations = [
        r"$J_e = -\sigma_e \nabla \phi_e$",
        r"$\nabla \cdot J_e = q$",
        r"$\mathrm{CSD} = \nabla \cdot J_e$",
        r"$Y = S W^{T}$  (laminar proxy / readout)",
    ]
    for i, eq in enumerate(equations):
        ax.text(pb_x + 0.2, pb_y + pb_h - 0.55 - i * 0.42, eq, fontsize=9, va="top")
    ax.text(
        pb_x + 0.2,
        pb_y + 0.22,
        "Reference framing only; not solved in default jaxfne path",
        fontsize=6.5,
        va="bottom",
        color="#555555",
        style="italic",
    )

    # Panel C — default jaxfne v0.3.x path
    pc_x, pc_y, pc_w, pc_h = 0.45, 4.35, 13.1, 1.75
    _panel(ax, pc_x, pc_y, pc_w, pc_h, "Panel C — default jaxfne v0.3.x path", facecolor="#E8F5E9", edgecolor="#2E7D32")
    panel_c_items = [
        "laminar proxy field",
        "finite arrays",
        "JSON manifest",
        "probe report",
        "no PDE solve",
        "no calibrated amplitude",
    ]
    for i, item in enumerate(panel_c_items):
        col = i % 3
        row = i // 3
        ax.text(pc_x + 0.25 + col * 4.2, pc_y + pc_h - 0.55 - row * 0.38, f"• {item}", fontsize=8, va="top")

    # Panel D — evidence ladder
    pd_x, pd_y, pd_w, pd_h = 0.45, 1.55, 6.35, 2.45
    _panel(ax, pd_x, pd_y, pd_w, pd_h, "Panel D — evidence ladder")
    for i, step in enumerate(EVIDENCE_LADDER):
        color = "#1B5E20" if i == 0 else "#333333"
        weight = "bold" if i == 0 else "normal"
        ax.text(pd_x + 0.2, pd_y + pd_h - 0.5 - i * 0.32, step, fontsize=7.5, va="top", color=color, fontweight=weight)

    # Panel E — status gates
    pe_x, pe_y, pe_w, pe_h = 7.2, 1.55, 6.35, 2.45
    _panel(ax, pe_x, pe_y, pe_w, pe_h, "Panel E — status gates")
    panel_e_lines = [
        "source_calibration_status recorded",
        "field_solver_status recorded",
        "probe_status recorded",
        "boundary / gauge required for stronger status",
        "residual / convergence required for solver status",
        "physical_amplitude_calibrated: false (default)",
    ]
    for i, line in enumerate(panel_e_lines):
        weight = "bold" if "physical_amplitude" in line else "normal"
        ax.text(pe_x + 0.2, pe_y + pe_h - 0.5 - i * 0.32, f"• {line}", fontsize=7.5, va="top", fontweight=weight)

    # Scope label
    scope_box = FancyBboxPatch(
        (0.45, 0.35),
        13.1,
        0.95,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(scope_box)
    ax.text(0.6, 0.82, SCOPE_STATUS, fontsize=8, va="top", color="#333333")
    ax.text(
        0.6,
        0.52,
        "field_solver_status: linear_solver",
        fontsize=7,
        va="top",
        family="monospace",
        color="#555555",
    )

    fig.tight_layout(pad=0.35)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_matplotlib_figure(fig, FIGURE_PATH, dpi=150)


def main() -> int:
    draw_figure()
    manifest = save_figure_manifest(
        figure_id="fig02",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "equations_shown": list(EQUATIONS_SHOWN),
            "evidence_ladder": list(EVIDENCE_LADDER),
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/fig02_contracts.py",
        },
    )
    sha = sha256_file(FIGURE_PATH)
    root = repo_root()
    manifest_rel = f"outputs/evidence/{FIGURE_PATH.stem}_manifest.json"
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {manifest_rel}")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
