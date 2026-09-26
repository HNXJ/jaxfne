"""0.5.5 ATLAS 6 groundwork: JDNA between-area probability, delay and the
compact exponential-distance rule (G_20 -> N_20 needs all three)."""

import math

import numpy as np
import pytest

import jaxfne as J
from jaxfne.jdna import (
    ORIGIN_DERIVED,
    develop,
    expand_area_connection_rules,
    genome_rules_hash,
    load_canonical_pseudogenome,
    pseudogenome_from_dict,
)
from jaxfne.neuronal_tensor import (
    load_neuronal_tensor,
    neuronal_tensor_to_configuration,
    save_neuronal_tensor,
)


def _genome(positions, *, n=10, p_max=0.5, decay=0.5, traversal_ms=20.0, **extra):
    layer = {"name": "L", "n_neurons": n, "depth_band": [0.0, 1.0],
             "cell_type_fractions": {"E": 1.0}}
    rule = {"kind": "exponential_distance", "positions": positions, "p_max": p_max,
            "decay": decay, "traversal_ms": traversal_ms,
            "source": {"layer": "L", "neuron_type": "E"},
            "targets": [{"layer": "L", "neuron_type": "E"}], "mechanism": "AMPA", **extra}
    return pseudogenome_from_dict({
        "name": "g-test", "areas": [{"name": a, "layers": [layer]} for a in positions],
        "area_connection_rules": [rule]})


def test_rule_expands_by_its_formula_and_marks_origins():
    g = _genome({"A": 0.0, "B": 0.5, "C": 1.0})
    got = {(c["source_area"], c["target_area"]): c for c in expand_area_connection_rules(g)}
    assert sorted(got) == [("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")]
    assert math.isclose(got["A", "C"]["probability"], 0.5 * math.exp(-2.0))
    assert got["A", "C"]["delay_ms"] == 20.0 and got["C", "B"]["delay_ms"] == 10.0
    t = develop(g, seed=0)
    assert [(c.probability, c.delay_ms) for c in t.area_connections] == [
        (c["probability"], c["delay_ms"]) for c in expand_area_connection_rules(g)]
    origins = [v for k, v in t.provenance["value_origins"].items() if k.startswith("area_connections.")]
    assert len(origins) == 6 and all(o["probability"] == ORIGIN_DERIVED for o in origins)
    pruned = _genome({"A": 0.0, "B": 0.5, "C": 1.0}, p_min=0.1)  # drops A<->C (0.068)
    assert len(expand_area_connection_rules(pruned)) == 4


def test_rule_free_genomes_keep_their_hashes():
    g = load_canonical_pseudogenome("canonical-v1-column-1000n")
    assert genome_rules_hash(g) == "07282b0928e9be9e49be5fa0a616da6fa65eaf72184976cd53a1cc6ce5dd0e76"
    assert develop(g, seed=0).provenance["phenotype_sha256"] == (
        "261f6997682b6b40d3aaa232cb4328e4eb0d4650d2b959d2bf2c15809c9ab49f")


def _cross_edges(tensor, dt_ms=0.5):
    cfg = neuronal_tensor_to_configuration(tensor, seed=3, duration_ms=5.0, dt_ms=dt_ms)
    model = J.construct(cfg)
    area = {r["neuron_id"]: r["area"] for r in model.neuron_table()}
    return [e for e in model.edge_table() if area[e["pre"]] != area[e["post"]]], area


def test_derived_probability_and_delay_reach_execution():
    full, _ = _cross_edges(develop(_genome({"A": 0.0, "B": 1.0}, p_max=1.0, decay=1e9), seed=0))
    assert len(full) == 2 * 10 * 10  # p = 1: every ordered cross pair, both directions
    assert {e["delay_steps"] for e in full} == {40}  # 20 ms at dt 0.5 ms
    half, area = _cross_edges(develop(_genome({"A": 0.0, "B": 1.0}, p_max=0.5, decay=1e9), seed=0))
    assert 60 <= len(half) <= 140  # Binomial(200, 0.5): mean 100, sd ~7
    ab = sum(area[e["pre"]] == "A" for e in half)
    assert 0 < ab < len(half)


def test_rule_gain_scales_cross_edge_weights():
    one, _ = _cross_edges(develop(_genome({"A": 0.0, "B": 1.0}, p_max=1.0, decay=1e9), seed=0))
    three, _ = _cross_edges(develop(_genome({"A": 0.0, "B": 1.0}, p_max=1.0, decay=1e9,
                                            w_mech=3.0), seed=0))
    w1 = np.abs([e["weight"] for e in one])
    w3 = np.abs([e["weight"] for e in three])
    assert len(w1) == len(w3) == 200 and np.allclose(w3, 3.0 * w1) and w1.min() > 0
    with pytest.raises(ValueError, match="w_mech must be finite and > 0"):
        expand_area_connection_rules(_genome({"A": 0.0, "B": 1.0}, w_mech=0.0))


def test_tensor_roundtrip_keeps_declared_delay_and_probability(tmp_path):
    t = develop(_genome({"A": 0.0, "B": 0.25}), seed=0)
    path = save_neuronal_tensor(t, tmp_path / "t.json")
    back = load_neuronal_tensor(path)
    assert [(c.probability, c.delay_ms) for c in back.area_connections] == [
        (c.probability, c.delay_ms) for c in t.area_connections]
    assert all(c.delay_ms == 5.0 for c in back.area_connections)


@pytest.mark.parametrize("bad, match", [
    ({"kind": "power_law"}, "kind must be one of"),
    ({"p_max": 0.0}, r"p_max must be in \(0"),
    ({"positions": {"A": 0.0, "B": 1.5}}, r"positions\['B'\] must be in"),
    ({"positions": {"A": 0.0, "Z": 1.0}}, "unknown (source|target)_area 'Z'"),
    ({"decay": -1.0}, "decay must be finite and > 0"),
    ({"w_mech": 0.0}, "w_mech must be finite and > 0"),
])
def test_invalid_rules_are_refused(bad, match):
    g = _genome({"A": 0.0, "B": 1.0})
    rule = {**g.area_connection_rules[0], **bad}
    g = pseudogenome_from_dict({**{"name": g.name, "areas": [
        {"name": a.name, "layers": [{"name": "L", "n_neurons": 10, "depth_band": [0.0, 1.0],
                                     "cell_type_fractions": {"E": 1.0}}]} for a in g.areas]},
        "area_connection_rules": [rule]})
    with pytest.raises(ValueError, match=match):
        develop(g, seed=0)
    with pytest.raises(ValueError, match="probability must be in"):
        J.neuronal_tensor.AreaConnection("A", "L", "E", "B", "L", "E", probability=1.5)


def test_explicit_entry_gain_is_validated():
    g = _genome({"A": 0.0, "B": 1.0})
    entry = {"source_area": "A", "source_layer": "L", "source_neuron_type": "E",
             "target_area": "B", "target_layer": "L", "target_neuron_type": "E", "w_mech": -2.0}
    g = pseudogenome_from_dict({"name": g.name, "areas": [
        {"name": a.name, "layers": [{"name": "L", "n_neurons": 10, "depth_band": [0.0, 1.0],
                                     "cell_type_fractions": {"E": 1.0}}]} for a in g.areas],
        "area_connections": [entry]})
    with pytest.raises(ValueError, match="w_mech must be finite and > 0"):
        develop(g, seed=0)
