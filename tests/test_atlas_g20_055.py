"""0.5.5 ATLAS 6: the frozen G_20 genome develops into the declared N_20."""

import json
import math

import jaxfne as J
from artifacts.atlas import g20_genome as G


def test_frozen_genome_equals_its_constants():
    assert json.loads(G.GENOME_PATH.read_text(encoding="utf-8")) == G.genome_dict()


def test_n20_realizes_the_declared_hierarchy():
    g = G.load_g20()
    t = G.develop_n20()
    assert [a.name for a in t.areas] == G.area_names()
    assert all(sum(L.n_neurons for L in a.layers) == G.G20_N_PER_AREA for a in t.areas)
    h = g.area_connection_rules[0]["positions"]
    by_pair = {}
    for c in t.area_connections:
        d = abs(h[c.source_area] - h[c.target_area])
        assert math.isclose(c.probability, G.G20_P_MAX * math.exp(-d / G.G20_DECAY))
        assert math.isclose(c.delay_ms, d * G.G20_TRAVERSAL_MS)
        assert c.probability >= G.G20_P_MIN
        by_pair[c.source_area, c.target_area] = c
    assert ("H01", "H02") in by_pair and ("H01", "H20") not in by_pair  # p_min prunes far pairs
    assert t.provenance["development_seed"] == G.G20_DEV_SEED


def test_n20_constructs_with_delayed_cross_area_edges():
    cfg = J.neuronal_tensor_to_configuration(G.develop_n20(), seed=5, duration_ms=10.0, dt_ms=0.5)
    model = J.construct(cfg)
    area = {r["neuron_id"]: r["area"] for r in model.neuron_table()}
    assert len(area) == G.G20_N_AREAS * G.G20_N_PER_AREA
    cross = [e for e in model.edge_table() if area[e["pre"]] != area[e["post"]]]
    assert cross and min(e["delay_steps"] for e in cross) >= 2  # adjacent: 20/19 ms at 0.5 ms
