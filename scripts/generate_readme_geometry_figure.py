#!/usr/bin/env python3
"""Static circuit-geometry panel for the README visual section.

Renders the default canonical laminar column (300-neuron proxy size) from
real package output: model.neuron_table() positions, colored by layer and
symbolized by cell type. Same API path as scripts/visualize_cylinder_cortex_1000.py
but static (matplotlib) for README embedding.

Usage:
  python scripts/generate_readme_geometry_figure.py [OUTPUT_PATH]
Writes: docs/assets/showcases/circuit_geometry_column.png (default)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import jaxfne as jtfne  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/assets/showcases/circuit_geometry_column.png"

CELL_COLORS = {"E": "#1f77b4", "PV": "#d62728", "SST": "#2ca02c", "VIP": "#9467bd"}
CELL_MARKERS = {"E": "o", "PV": "D", "SST": "s", "VIP": "^"}
LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6"]


def main() -> int:
    cfg = (
        jtfne.build_laminar_column("V1", n=300, ei_profile="canonical", layers=jtfne.CANONICAL_LAYERS_6L)
        .update_metadata(column_radius_mm=0.1, column_height_mm=1.0)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"])
        .field(domain="laminar_column", conductivity="proxy")
    )
    model = jtfne.construct(cfg)
    rows = model.neuron_table()

    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    for layer in LAYERS:
        sel = [r for r in rows if r["layer"] == layer]
        if not sel:
            continue
        xs = [r["x"] for r in sel]
        ys = [r["y"] for r in sel]
        zs = [r["z"] for r in sel]
        ax.scatter(xs, ys, zs, s=6, alpha=0.55, label=layer)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("Canonical laminar column — 300-neuron proxy\n(neuron_table() positions, layer-colored)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())