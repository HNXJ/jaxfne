"""G_20: the synthetic 20-area hierarchy PseudoGenome for AT-10 (0.5.5 ATLAS 6).

Human decision (2026-09-26): G_20 is a synthetic declared hierarchy, not
data-driven; 50 neurons per area. Twenty areas sit at relative hierarchy
positions h = i / 19; between-area E -> {E, PV} projections follow JDNA's
``exponential_distance`` rule (p = p_max * exp(-|dh| / decay), delay_ms =
|dh| * traversal_ms, pairs with p < p_min omitted). Each area is one E/PV
layer with the four within-area E/PV motifs. Every value is a relative
scaffold value; nothing is calibrated against anatomy.

The module constants are the genome's inputs (the Atlas spec reads them by
name); ``genome_dict()`` builds the genome from them and the frozen JSON in
``genomes/`` must equal it. Import rule: top-level ``jaxfne`` only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jaxfne as J

G20_NAME = "g20-hierarchy-v1"
G20_N_AREAS = 20
G20_N_PER_AREA = 50
G20_CELL_TYPE_FRACTIONS = {"E": 0.8, "PV": 0.2}
G20_P_MAX = 0.1
G20_DECAY = 0.2
G20_TRAVERSAL_MS = 20.0
G20_P_MIN = 0.01
G20_DEV_SEED = 20
GENOME_PATH = Path(__file__).parent / "genomes" / f"{G20_NAME}.json"


def area_names() -> list[str]:
    return [f"H{i:02d}" for i in range(1, G20_N_AREAS + 1)]


def genome_dict() -> dict[str, Any]:
    """The G_20 PseudoGenome as JSON-safe data, built from the module constants."""
    names = area_names()
    layer = {"name": "L", "n_neurons": G20_N_PER_AREA, "depth_band": [0.0, 1.0],
             "cell_type_fractions": dict(G20_CELL_TYPE_FRACTIONS)}
    within = [{"source_layer": "L", "source_neuron_type": s, "target_layer": "L",
               "target_neuron_type": t, "mechanism": "AMPA" if s == "E" else "GABA_A"}
              for s in ("E", "PV") for t in ("E", "PV")]
    return {
        "schema_version": "pseudogenome_v1",
        "name": G20_NAME,
        "description": (
            "Synthetic 20-area hierarchy for Atlas S10/AT-10. Areas at relative "
            "hierarchy positions i/19; between-area E->{E,PV} AMPA projections by the "
            "exponential_distance rule; one E/PV layer per area with the four "
            "within-area motifs. Relative scaffold values; not calibrated."),
        "development_parameters": {"fraction_jitter_sigma": 0.0},
        "areas": [{"name": a, "layers": [layer], "inter_connections": within} for a in names],
        "area_connections": [],
        "area_connection_rules": [{
            "kind": "exponential_distance",
            "positions": {a: i / (G20_N_AREAS - 1) for i, a in enumerate(names)},
            "p_max": G20_P_MAX, "decay": G20_DECAY, "traversal_ms": G20_TRAVERSAL_MS,
            "p_min": G20_P_MIN,
            "source": {"layer": "L", "neuron_type": "E"},
            "targets": [{"layer": "L", "neuron_type": "E"}, {"layer": "L", "neuron_type": "PV"}],
            "mechanism": "AMPA",
        }],
    }


def write_genome(path: Path = GENOME_PATH) -> Path:
    """Freeze the genome JSON (write-once)."""
    if path.exists():
        raise FileExistsError(f"{path} is frozen; remove it deliberately to re-freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(genome_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_g20() -> Any:
    """The frozen G_20 PseudoGenome."""
    return J.load_pseudogenome(GENOME_PATH)


def develop_n20(seed: int = G20_DEV_SEED) -> Any:
    """N_20 = develop(G_20, K_D = seed)."""
    return J.develop(load_g20(), seed=seed)


if __name__ == "__main__":
    print(write_genome())
