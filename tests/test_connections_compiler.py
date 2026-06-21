"""The .connections() selector->edge compiler.

`Configuration.connections(...)` is no longer a declaration-only fence: construct()
compiles each rule into real edges in the EdgeList, honoring area/layer/cell_type
selectors, probability, weight, and sign.
"""
import numpy as np
import pandas as pd
import pytest

import jaxfne as jtfne


def _two_area_base():
    layers = {"L1": 10, "L2": 25, "L3": 20, "L4": 10, "L5": 20, "L6": 15}
    return (
        jtfne.Configuration()
        .areas(["V1", "V2"])
        .population(N=100, neurons=dict(layers), name="V1")
        .population(N=100, neurons=dict(layers), name="V2")
    )


def _build(cfg):
    cfg = (cfg.set_emitter("izhikevich", "cortical_eig")
              .probes(["spikes", "V_m"]).field(domain="laminar_column", conductivity="proxy"))
    model = jtfne.construct(cfg)
    el = model.params["edge_list"]
    pre = np.asarray(el.pre); post = np.asarray(el.post); w = np.asarray(el.weight)
    nt = pd.DataFrame(model.neuron_table())
    return model, pre, post, w, nt["area"].values, nt["layer"].values, nt["cell_type"].values


def _ff_rule(cfg, *, probability=1.0, weight=0.5, sign="excitatory"):
    return cfg.connections(
        name="ff", source={"area": "V1", "layers": ["L2", "L3"], "cell_type": "E"},
        target={"area": "V2", "layers": ["L4"]},
        probability=probability, weight=weight, sign=sign,
    )


def test_connections_probability_one_full_routing():
    model, pre, post, w, area, layer, cell = _build(_ff_rule(_two_area_base()))
    cross = (area[pre] == "V1") & (area[post] == "V2")
    assert int(cross.sum()) > 0
    assert set(layer[pre[cross]]) <= {"L2", "L3"}      # source L2/3
    assert set(layer[post[cross]]) == {"L4"}           # target L4
    assert set(cell[pre[cross]]) == {"E"}              # source E only
    # full connectivity: every source-E in L2/3 connects to every V2-L4 target
    n_src = int(((area == "V1") & np.isin(layer, ["L2", "L3"]) & (cell == "E")).sum())
    n_tgt = int(((area == "V2") & (layer == "L4")).sum())
    assert int(cross.sum()) == n_src * n_tgt


def test_connections_probability_zero_no_edges():
    model, pre, post, w, area, layer, cell = _build(_ff_rule(_two_area_base(), probability=0.0))
    cross = (area[pre] == "V1") & (area[post] == "V2")
    assert int(cross.sum()) == 0


def test_connections_probability_half_is_partial():
    model, pre, post, w, area, layer, cell = _build(_ff_rule(_two_area_base(), probability=0.5))
    cross = (area[pre] == "V1") & (area[post] == "V2")
    n_src = int(((area == "V1") & np.isin(layer, ["L2", "L3"]) & (cell == "E")).sum())
    n_tgt = int(((area == "V2") & (layer == "L4")).sum())
    full = n_src * n_tgt
    assert 0 < int(cross.sum()) < full                 # strictly between none and full


def test_connections_sign_excitatory_positive():
    model, pre, post, w, area, layer, cell = _build(_ff_rule(_two_area_base(), sign="excitatory"))
    cross = (area[pre] == "V1") & (area[post] == "V2")
    assert bool((w[cross] > 0).all())


def test_connections_sign_inhibitory_negative():
    model, pre, post, w, area, layer, cell = _build(_ff_rule(_two_area_base(), sign="inhibitory"))
    cross = (area[pre] == "V1") & (area[post] == "V2")
    assert bool((w[cross] < 0).all())


def test_connections_sign_signed_uses_presynaptic_sign():
    # source selector spans all cell types; "signed" -> weight sign follows pre neuron
    cfg = _two_area_base().connections(
        name="mixed", source={"area": "V1", "layers": ["L2", "L3"]},
        target={"area": "V2", "layers": ["L4"]}, probability=1.0, weight=0.5, sign="signed")
    model, pre, post, w, area, layer, cell = _build(cfg)
    cross = (area[pre] == "V1") & (area[post] == "V2")
    # excitatory presyn -> positive weight, inhibitory presyn -> negative
    assert bool((w[cross][cell[pre[cross]] == "E"] > 0).all())
    inh = cell[pre[cross]] != "E"
    if inh.any():
        assert bool((w[cross][inh] < 0).all())


def test_connections_status_flips_to_compiled():
    model, *_ = _build(_ff_rule(_two_area_base()))
    rule = model.cfg.metadata["circuit"]["connections"][-1]
    assert rule["status"] == "compiled"
    assert rule["compiled_n_edges"] > 0


def test_connections_no_match_marked_compiled_no_edges():
    # target a non-existent area -> rule compiles to zero edges, flagged honestly
    cfg = _two_area_base().connections(
        name="ghost", source={"area": "V1"}, target={"area": "V9"}, probability=1.0)
    model, pre, post, w, area, layer, cell = _build(cfg)
    rule = model.cfg.metadata["circuit"]["connections"][-1]
    assert rule["status"] == "compiled_no_matching_edges"
    assert rule["compiled_n_edges"] == 0


def test_connections_layer_merged_scheme_tolerant():
    # selector "L2/3" must match split L2/L3 columns (and vice versa)
    cfg = _two_area_base().connections(
        name="ff", source={"area": "V1", "layers": ["L2/3"], "cell_type": "E"},
        target={"area": "V2", "layers": ["L4"]}, probability=1.0, weight=0.5, sign="excitatory")
    model, pre, post, w, area, layer, cell = _build(cfg)
    cross = (area[pre] == "V1") & (area[post] == "V2")
    assert int(cross.sum()) > 0
    assert set(layer[pre[cross]]) <= {"L2", "L3"}
