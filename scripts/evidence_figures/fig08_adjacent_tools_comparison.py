#!/usr/bin/env python3
"""Generate Figure 8: adjacent tools positioning (capability comparison only).

Outputs:
  figures/evidence/fig08_adjacent_tools_comparison.png
  outputs/evidence/fig08_adjacent_tools_comparison_manifest.json

No speedup, accuracy, biological-validity, or solver-superiority claims.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _figure_common import (
    ensure_evidence_dirs,
    repo_root,
    save_figure_manifest,
    sha256_file,
)


SOURCE_FILES = [
    "scripts/evidence_figures/fig08_adjacent_tools_comparison.py",
    "scripts/evidence_figures/_figure_common.py",
]

TITLE = "Adjacent tools positioning (capability comparison only)"

SCOPE_STATUS = (
    "Compares roles and interfaces, not performance or biological superiority; "
    "jaxfne remains a compact TFNE scaffold with proxy readouts"
)

COMPARISON_ROWS = [
    (
        "JAX",
        "array transforms, JIT, vmap/scan",
        "numerical substrate for jaxfne kernels",
    ),
    (
        "Jaxley",
        "differentiable compartment/cell emitters",
        "optional detailed emitter bridge (lazy import)",
    ),
    (
        "PyNWB",
        "neurodata export schema",
        "artifact/export target for manifests and bundles",
    ),
    (
        "LFPy / forward tools",
        "biophysical extracellular modeling",
        "external comparison / future validation path",
    ),
    (
        "jaxfne",
        "emitter -> source -> field/probe scaffold",
        "compact tutorial and evidence evidence layer",
    ),
]

POSITIONING_NOTES = [
    "jaxfne does not replace full forward-modeling stacks",
    "proxy readouts are default; calibrated amplitude requires separate evidence",
    "Jaxley bridge is optional, not mandatory for core import",
    "comparison is capability-oriented, not benchmark-oriented",
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "fig08_adjacent_tools_comparison.png"


def draw_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7.0, 7.55, TITLE, ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(
        7.0,
        7.05,
        "import jaxfne as jtfne",
        ha="center",
        va="center",
        fontsize=9,
        family="monospace",
        color="#333333",
    )

    table_box = FancyBboxPatch(
        (0.55, 3.15),
        12.9,
        3.55,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.1,
        edgecolor="#1A4A8A",
        facecolor="#F8FAFF",
    )
    ax.add_patch(table_box)

    headers = ["Tool / layer", "Primary role", "jaxfne relation"]
    col_x = [0.75, 3.6, 8.2]
    y_header = 6.35
    for x, header in zip(col_x, headers):
        ax.text(x, y_header, header, fontsize=9, fontweight="bold", va="top")

    y = 5.85
    for tool, role, relation in COMPARISON_ROWS:
        weight = "bold" if tool == "jaxfne" else "normal"
        face = "#E8F5E9" if tool == "jaxfne" else "#FFFFFF"
        row_box = FancyBboxPatch(
            (0.65, y - 0.55),
            12.7,
            0.62,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.6,
            edgecolor="#CCCCCC",
            facecolor=face,
        )
        ax.add_patch(row_box)
        ax.text(col_x[0], y, tool, fontsize=8, fontweight=weight, va="top")
        ax.text(col_x[1], y, role, fontsize=7.5, va="top")
        ax.text(col_x[2], y, relation, fontsize=7.5, va="top")
        y -= 0.68

    notes_box = FancyBboxPatch(
        (0.55, 1.55),
        12.9,
        1.35,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#2E7D32",
        facecolor="#F1F8E9",
    )
    ax.add_patch(notes_box)
    ax.text(0.75, 2.65, "Positioning notes", fontsize=9, fontweight="bold", va="top")
    for i, note in enumerate(POSITIONING_NOTES):
        ax.text(0.85, 2.25 - i * 0.32, f"- {note}", fontsize=7.5, va="top", color="#222222")

    scope_box = FancyBboxPatch(
        (0.55, 0.45),
        12.9,
        0.9,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(scope_box)
    ax.text(0.75, 1.1, SCOPE_STATUS, fontsize=7.5, va="top", color="#333333")
    ax.text(
        0.75,
        0.62,
        "physical_amplitude_calibrated: false  |  capability_comparison_not_superiority",
        fontsize=7,
        va="top",
        family="monospace",
        color="#555555",
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    draw_figure()
    manifest = save_figure_manifest(
        figure_id="fig08",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/fig08_adjacent_tools_comparison.py",
            "claim_boundary": "capability_comparison_not_superiority",
            "comparison_rows": [
                {"tool": t, "primary_role": r, "jaxfne_relation": j}
                for t, r, j in COMPARISON_ROWS
            ],
            "positioning_notes": list(POSITIONING_NOTES),
            "performance_claims_allowed": False,
            "superiority_claims_allowed": False,
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
