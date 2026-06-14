#!/usr/bin/env python3
"""Generate Figure 1: TFNE source-to-field/readout architecture diagram.

Outputs:
  figures/evidence/fig01_tfne_architecture.png
  outputs/evidence/fig01_tfne_architecture_manifest.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _figure_common import (
    ensure_evidence_dirs,
    repo_root,
    save_figure_manifest,
    sha256_file,
)


SOURCE_FILES = [
    "scripts/evidence_figures/fig01_architecture.py",
    "scripts/evidence_figures/_figure_common.py",
]

SCOPE_STATUS = (
    "Computational scaffold; proxy readouts unless calibrated run evidence is supplied"
)

TITLE = "TFNE source-to-field/readout architecture"

PIPELINE_NODES = [
    ("Emitter", "Config / Configuration\nconstruct"),
    ("Source", "source tensors\nbookkeeping"),
    ("Field / proxy field", "simulate\nlaminar projection"),
    ("Probe", "Signals / model.probe\nproxy readouts"),
    ("Objective", "objectives\nmetrics / gates"),
    ("Optimizer", "optim\nAGSDR / optax"),
    ("Validation / manifest", "io / validation\nJSON / hash artifacts"),
]

METADATA_BANDS = [
    "seed / dtype / runtime",
    "source mode / calibration status",
    "field solver status: laminar_proxy_no_pde",
    "probe report / operator status",
    "finite outputs",
    "JSON-safe manifests / SHA256 hashes",
]

_dirs = ensure_evidence_dirs()
FIGURE_PATH = _dirs["figures"] / "fig01_tfne_architecture.png"


def draw_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(
        7.0,
        7.55,
        TITLE,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        7.0,
        7.05,
        "import jaxfne as jtfne",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        color="#333333",
    )

    node_w = 1.55
    node_h = 1.05
    y_node = 4.85
    x_start = 0.55
    x_gap = 1.85
    box_color = "#E8F0FE"
    edge_color = "#1A4A8A"

    centers: list[tuple[float, float]] = []
    for idx, (title, mapping) in enumerate(PIPELINE_NODES):
        x = x_start + idx * x_gap
        box = FancyBboxPatch(
            (x, y_node),
            node_w,
            node_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=edge_color,
            facecolor=box_color,
        )
        ax.add_patch(box)
        ax.text(
            x + node_w / 2,
            y_node + node_h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            x + node_w / 2,
            y_node + node_h * 0.28,
            mapping,
            ha="center",
            va="center",
            fontsize=7,
            color="#333333",
        )
        centers.append((x + node_w / 2, y_node + node_h / 2))

    for (x0, y0), (x1, y1) in zip(centers[:-1], centers[1:]):
        arrow = FancyArrowPatch(
            (x0 + node_w * 0.32, y0),
            (x1 - node_w * 0.32, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#444444",
        )
        ax.add_patch(arrow)

    meta_box = FancyBboxPatch(
        (0.55, 2.55),
        12.9,
        1.55,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="#666666",
        facecolor="#F7F7F7",
    )
    ax.add_patch(meta_box)
    ax.text(0.75, 3.75, "Metadata bands", fontsize=9, fontweight="bold", va="top")
    for i, band in enumerate(METADATA_BANDS):
        col = i % 3
        row = i // 3
        ax.text(
            0.85 + col * 4.2,
            3.35 - row * 0.42,
            f"• {band}",
            fontsize=7.5,
            va="top",
            color="#222222",
        )

    scope_box = FancyBboxPatch(
        (0.55, 0.55),
        12.9,
        1.55,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="#8B4513",
        facecolor="#FFF8E7",
    )
    ax.add_patch(scope_box)
    ax.text(0.75, 1.75, "Scope / status", fontsize=9, fontweight="bold", va="top")
    ax.text(0.85, 1.25, SCOPE_STATUS, fontsize=8, va="top", color="#333333")
    ax.text(
        0.85,
        0.85,
        "physical_amplitude_claim_allowed: false  |  truth_mode: truth_safe_unverified",
        fontsize=7.5,
        va="top",
        family="monospace",
        color="#555555",
    )

    fig.tight_layout(pad=0.4)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    draw_figure()
    manifest = save_figure_manifest(
        figure_id="fig01",
        title=TITLE,
        output_path=FIGURE_PATH,
        source_files=SOURCE_FILES,
        extra={
            "scope_status": SCOPE_STATUS,
            "generator_command": "python scripts/evidence_figures/fig01_architecture.py",
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
