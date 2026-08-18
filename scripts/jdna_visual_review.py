"""JDNA visual review (campaign Block L): compare the canonical fixed
NeuronalTensor against phenotypes developed from the canonical PseudoGenome.

Run:  python3 scripts/jdna_visual_review.py [OUTPUT_DIR]
Writes: <OUTPUT_DIR>/jdna_visual_review.png (default: figures/)
Prints a numeric comparison table to stdout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import jaxfne as jtfne  # noqa: E402
from jaxfne.jdna import develop, load_canonical_pseudogenome  # noqa: E402

FIG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _tensor_layers(tensor):
    for area in tensor.areas:
        yield from area.layers


def layer_counts(tensor) -> dict[str, int]:
    return {l.name: l.n_neurons for l in _tensor_layers(tensor)}


def cell_type_rows(tensor) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for l in _tensor_layers(tensor):
        for nt in l.neuron_types:
            rows.append((l.name, nt.name, int(round(l.n_neurons * (nt.fraction or 0.0)))))
    return rows


def main() -> None:
    genome = load_canonical_pseudogenome("canonical-v1-column-1000n")
    canonical = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")

    t0 = time.perf_counter()
    developed = [develop(genome, seed=s) for s in range(3)]
    dev_time = (time.perf_counter() - t0) / 3.0

    print(f"canonical tensor: {canonical.name} n={sum(layer_counts(canonical).values())}")
    for i, tensor in enumerate(developed):
        print(f"developed seed={i}: n={sum(layer_counts(tensor).values())} "
              f"provenance={sorted(tensor.provenance) if tensor.provenance else None}")

    canon_counts = layer_counts(canonical)
    rows = cell_type_rows(canonical)
    print(f"\n{'layer':<4} {'cell':<5} {'canonical':>9} {'dev0':>5} {'dev1':>5} {'dev2':>5}")
    for layer, ct, cc in rows:
        line = f"{layer:<4} {ct:<5} {cc:>9}"
        for t in developed:
            sub = next(
                (int(round(l.n_neurons * (nt.fraction or 0.0)))
                 for l in _tensor_layers(t) for nt in l.neuron_types
                 if l.name == layer and nt.name == ct),
                0,
            )
            line += f" {sub:>5}"
        print(line)
    assert canon_counts == layer_counts(developed[0]), "dev0 must match canonical layer totals"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    layers = ["L1", "L2", "L3", "L4", "L5", "L6"]
    cells = ["E", "PV", "SST", "VIP"]

    comp = {layer: {c: 0 for c in cells} for layer in layers}
    for layer, ct, cc in rows:
        comp[layer][ct] = cc
    ax = axes[0]
    im = ax.imshow([[comp[l][c] for c in cells] for l in layers], cmap="viridis")
    ax.set_xticks(range(4), cells)
    ax.set_yticks(range(6), layers)
    ax.set_title("canonical fixed tensor (counts)")
    fig.colorbar(im, ax=ax)

    dev = {layer: {c: 0 for c in cells} for layer in layers}
    for layer, ct, cc in cell_type_rows(developed[0]):
        dev[layer][ct] = cc
    ax = axes[1]
    im = ax.imshow([[dev[l][c] for c in cells] for l in layers], cmap="viridis")
    ax.set_xticks(range(4), cells)
    ax.set_yticks(range(6), layers)
    ax.set_title("JDNA develop(seed=0) (counts)")
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow([[comp[l][c] - dev[l][c] for c in cells] for l in layers],
                   cmap="RdBu", vmin=-25, vmax=25)
    ax.set_xticks(range(4), cells)
    ax.set_yticks(range(6), layers)
    ax.set_title("canonical - dev0")
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"JDNA visual review — canonical-v1-column-1000n (develop {dev_time*1000:.1f} ms/phenotype)")
    fig.tight_layout()
    out = FIG_DIR / "jdna_visual_review.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
