#!/usr/bin/env python3
"""Generate Figure 3: jaxfne computational backend and reproducibility boundary.

Outputs:
  figures/publication/fig03_jaxfne_backend.png
  outputs/publication/fig03_jaxfne_backend_manifest.json
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _figure_common import (
    ensure_publication_dirs,
    repo_root,
    save_figure_manifest,
    sha256_file,
)


SOURCE_FILES = [
    "scripts/publication/fig03_backend.py",
    "scripts/publication/_figure_common.py",
]

_dirs = ensure_publication_dirs()
FIGURE_PATH = _dirs["figures"] / "fig03_jaxfne_backend.png"

TITLE = "jaxfne computational backend and reproducibility boundary"

SCOPE_STATUS = (
    "JAX kernels execute simulated/proxy TFNE paths; reports record runtime, status, and artifacts"
)

BACKEND_CONTRACT = [
    "Config / Configuration -> construct() -> Model / Net",
    "Simulation / Paradigm -> simulate() -> Signals",
    "model.probe() / Signals.get() -> Objective report",
    "Optimizer / Trainer -> Manifest / validation",
    "import jaxfne as jtfne",
]

OPTIONAL_DEPENDENCY_POLICY = [
    "core import stays lightweight",
    "visualization extras optional (matplotlib/plotly)",
    "Jaxley bridge lazy-imported and guarded",
    "Optax optional for declared differentiable / surrogate paths",
    "no hard spike-reset differentiability claim without surrogate",
]

RUNTIME_REPRODUCIBILITY_FIELDS = [
    "seed",
    "dtype (float32 default)",
    "JAX backend / devices",
    "duration_ms / dt_ms",
    "finite output checks",
    "asset SHA256 hashes",
]

PUBLIC_FLOW = [
    ("Config / Configuration", "construct()"),
    ("Model / Net", "Simulation / Paradigm"),
    ("simulate()", "Signals"),
    ("model.probe() / Signals.get()", "Objective report"),
    ("Optimizer / Trainer", "Manifest / validation"),
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
    )
    ax.add_patch(box)
    ax.text(x + 0.12, y + h - 0.18, title, fontsize=9, fontweight="bold", va="top")


def draw_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(7.0, 9.55, TITLE, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(7.0, 9.1, "import jaxfne as jtfne", ha="center", fontsize=9, family="monospace", color="#444444")

    # Panel A — public object flow
    pa_x, pa_y, pa_w, pa_h = 0.45, 6.4, 13.1, 2.7
    _panel(ax, pa_x, pa_y, pa_w, pa_h, "Panel A — public object flow")
    box_w, box_h = 1.85, 0.55
    y_flow = pa_y + 1.35
    x_flow_start = pa_x + 0.25
    x_gap = 2.35
    for i, (left, right) in enumerate(PUBLIC_FLOW):
        x = x_flow_start + i * x_gap
        for j, label in enumerate((left, right)):
            yy = y_flow + (1 - j) * 0.65
            inner = FancyBboxPatch(
                (x, yy),
                box_w,
                box_h,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                linewidth=0.9,
                edgecolor="#5C6BC0",
                facecolor="#E8EAF6",
            )
            ax.add_patch(inner)
            ax.text(x + box_w / 2, yy + box_h / 2, label, ha="center", va="center", fontsize=6.5)
        if i < len(PUBLIC_FLOW) - 1:
            ax.text(x + box_w + 0.22, y_flow + 0.28, "->", fontsize=10, ha="center", color="#333333")

    # Panel B — numerical hot path
    pb_x, pb_y, pb_w, pb_h = 0.45, 4.55, 6.35, 1.55
    _panel(ax, pb_x, pb_y, pb_w, pb_h, "Panel B — numerical hot path (inside JIT)", facecolor="#FFF3E0", edgecolor="#E65100")
    hot_path = [
        "pure JAX kernels",
        "explicit PRNG keys",
        "lax.scan over time",
        "vmap over seeds / candidates / probes",
        "jit only after shapes / static config fixed",
    ]
    for i, line in enumerate(hot_path):
        ax.text(pb_x + 0.2, pb_y + pb_h - 0.48 - i * 0.22, f"• {line}", fontsize=7.2, va="top")

    # Panel C — outside-JIT layer
    pc_x, pc_y, pc_w, pc_h = 7.2, 4.55, 6.35, 1.55
    _panel(ax, pc_x, pc_y, pc_w, pc_h, "Panel C — outside-JIT layer", facecolor="#F3E5F5", edgecolor="#6A1B9A")
    outside_jit = [
        "plotting",
        "JSON / markdown / docs",
        "file I/O",
        "manifests",
        "notebook display",
    ]
    for i, line in enumerate(outside_jit):
        col = i % 2
        row = i // 2
        ax.text(pc_x + 0.2 + col * 3.0, pc_y + pc_h - 0.48 - row * 0.28, f"• {line}", fontsize=7.2, va="top")

    # Panel D — optional dependency boundary
    pd_x, pd_y, pd_w, pd_h = 0.45, 2.55, 6.35, 1.65
    _panel(ax, pd_x, pd_y, pd_w, pd_h, "Panel D — optional dependency boundary", facecolor="#E0F2F1", edgecolor="#00695C")
    for i, line in enumerate(OPTIONAL_DEPENDENCY_POLICY):
        ax.text(pd_x + 0.2, pd_y + pd_h - 0.48 - i * 0.26, f"• {line}", fontsize=7.2, va="top")

    # Panel E — reproducibility report
    pe_x, pe_y, pe_w, pe_h = 7.2, 2.55, 6.35, 1.65
    _panel(ax, pe_x, pe_y, pe_w, pe_h, "Panel E — reproducibility report fields", facecolor="#E8F5E9", edgecolor="#2E7D32")
    for i, line in enumerate(RUNTIME_REPRODUCIBILITY_FIELDS):
        col = i % 2
        row = i // 2
        ax.text(pe_x + 0.2 + col * 3.0, pe_y + pe_h - 0.48 - row * 0.28, f"• {line}", fontsize=7.2, va="top")

    # Boundary arrow between B and C
    arrow = FancyArrowPatch(
        (pb_x + pb_w / 2, pb_y - 0.05),
        (pc_x + pc_w / 2, pc_y + pc_h + 0.05),
        arrowstyle="<->",
        mutation_scale=10,
        linewidth=1.0,
        color="#888888",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    ax.text(7.0, 4.35, "JIT boundary", ha="center", fontsize=7, color="#666666", style="italic")

    # Scope label
    scope_box = FancyBboxPatch(
        (0.45, 0.35),
        13.1,
        1.85,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(scope_box)
    ax.text(0.6, 1.95, SCOPE_STATUS, fontsize=8, va="top", color="#333333")
    ax.text(
        0.6,
        1.55,
        "physical_amplitude_claim_allowed: false  |  field_solver_status: laminar_proxy_no_pde",
        fontsize=7,
        va="top",
        family="monospace",
        color="#555555",
    )
    ax.text(
        0.6,
        1.15,
        "Spike-reset paths are not claimed differentiable unless surrogate config is explicit.",
        fontsize=7,
        va="top",
        color="#555555",
        style="italic",
    )

    fig.tight_layout(pad=0.35)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    draw_figure()
    manifest = save_figure_manifest(
        figure_id="fig03",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "backend_contract": list(BACKEND_CONTRACT),
            "optional_dependency_policy": list(OPTIONAL_DEPENDENCY_POLICY),
            "runtime_reproducibility_fields": list(RUNTIME_REPRODUCIBILITY_FIELDS),
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/publication/fig03_backend.py",
        },
    )
    sha = sha256_file(FIGURE_PATH)
    root = repo_root()
    manifest_rel = f"outputs/publication/{FIGURE_PATH.stem}_manifest.json"
    print(f"wrote: {FIGURE_PATH.relative_to(root)}")
    print(f"wrote: {manifest_rel}")
    print(f"sha256: {sha}")
    print(f"figure_id: {manifest['figure_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
