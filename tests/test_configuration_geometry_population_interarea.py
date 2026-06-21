"""Tests for the Configuration fluent grammar: geometry / population verbs and
real inter-column (inter-area) edge wiring.

Covers the three gaps closed in core.py:
  1. population(N, neurons={...}) -> per-layer counts decoupled from thickness
  2. geometry(layer_thickness={...}) -> cumulative z-intervals
  3. inter_column_connectivity(...) -> materialized cross-area edges with
     anatomical routing (feedforward L2/3->L4, feedback L6->L1/L5) + override
"""
import numpy as np
import pandas as pd
import pytest

import jaxfne as jtfne


def _base(name="V1", N=100):
    return (
        jtfne.Configuration()
        .runtime(seed=0, dt_ms=0.1, duration_ms=1000.0)
        .geometry(layer_thickness={"L1": 0.10, "L2": 0.15, "L3": 0.15, "L4": 0.10, "L5": 0.30, "L6": 0.20})
        .population(N=N, neurons={"L1": 10, "L2": 25, "L3": 20, "L4": 10, "L5": 20, "L6": 15}, name=name)
        .cell_types({"E": 0.6, "PV": 0.2, "SST": 0.13, "VIP": 0.07})
        .emitter(kind="izhikevich")
        .field(kind="laminar_proxy")
        .probe(kind="lfp_proxy")
    )


def test_geometry_thickness_to_z_intervals():
    cfg = jtfne.Configuration().geometry(
        layer_thickness={"L1": 0.10, "L2": 0.15, "L3": 0.15, "L4": 0.10, "L5": 0.30, "L6": 0.20}
    )
    lf = cfg.metadata["layer_fractions"]
    assert lf["L1"] == [0.0, 0.10]
    assert lf["L2"] == [0.10, 0.25]
    assert lf["L5"][0] == pytest.approx(0.50)
    assert lf["L6"][1] == pytest.approx(1.0)


def test_geometry_does_not_clobber_cell_types():
    # geometry without layer_cell_types must not overwrite a prior cell-type decl
    cfg = jtfne.Configuration().cell_types({"E": 0.6, "PV": 0.4}).geometry()
    assert "layer_cell_types" not in cfg.metadata  # geometry left it untouched
    assert cfg.metadata["cell_types"]["E"] == pytest.approx(0.6)


def test_population_exact_per_layer_counts():
    model = jtfne.construct(_base())
    nt = pd.DataFrame(model.neuron_table())
    counts = nt.groupby("layer").size().to_dict()
    assert counts == {"L1": 10, "L2": 25, "L3": 20, "L4": 10, "L5": 20, "L6": 15}


def test_population_counts_decoupled_from_thickness():
    # L5 is the thickest (0.30) but must NOT get the most neurons
    model = jtfne.construct(_base())
    nt = pd.DataFrame(model.neuron_table())
    counts = nt.groupby("layer").size().to_dict()
    assert counts["L2"] > counts["L5"]  # thin-dense L2 beats thick-sparse L5


def test_population_single_N_change_scales_proportionally():
    model = jtfne.construct(_base(N=200))
    nt = pd.DataFrame(model.neuron_table())
    counts = nt.groupby("layer").size().to_dict()
    assert sum(counts.values()) == 200
    assert counts == {"L1": 20, "L2": 50, "L3": 40, "L4": 20, "L5": 40, "L6": 30}


def _interarea_model(p_ff, p_fb, l2l=None):
    cfg = (
        _base("V1")
        .population(N=100, neurons={"L1": 10, "L2": 25, "L3": 20, "L4": 10, "L5": 20, "L6": 15}, name="V4")
        .inter_column_connectivity(
            source_area="V1", target_area="V4",
            layer_to_layer_map=l2l, p_feedforward=p_ff, p_feedback=p_fb, seed=0,
        )
    )
    model = jtfne.construct(cfg)
    el = model.params["edge_list"]
    pre = np.asarray(el.pre); post = np.asarray(el.post); w = np.asarray(el.weight)
    nt = pd.DataFrame(model.neuron_table())
    area = nt["area"].values; layer = nt["layer"].values
    cross = (area[pre] == "V1") & (area[post] == "V4")
    return pre, post, w, layer, cross


def test_interarea_off_produces_no_cross_edges():
    pre, post, w, layer, cross = _interarea_model(0.0, 0.0)
    assert int(cross.sum()) == 0


def test_interarea_feedforward_routes_l23_to_l4():
    pre, post, w, layer, cross = _interarea_model(0.5, 0.0)
    assert int(cross.sum()) > 0
    src = set(layer[pre[cross]]); dst = set(layer[post[cross]])
    assert src <= {"L2", "L3"}      # feedforward source is L2/3
    assert dst == {"L4"}            # feedforward target is L4


def test_interarea_feedback_routes_l6_to_l1_l5():
    pre, post, w, layer, cross = _interarea_model(0.0, 0.5)
    assert int(cross.sum()) > 0
    src = set(layer[pre[cross]]); dst = set(layer[post[cross]])
    assert src == {"L6"}            # feedback source is L6
    assert dst <= {"L1", "L5"}      # feedback target is L1/L5


def test_interarea_override_map():
    pre, post, w, layer, cross = _interarea_model(0.5, 0.0, l2l={"L5": "L4"})
    assert int(cross.sum()) > 0
    assert set(layer[pre[cross]]) == {"L5"}
    assert set(layer[post[cross]]) == {"L4"}


def test_interarea_cross_edges_excitatory_positive():
    pre, post, w, layer, cross = _interarea_model(0.5, 0.5)
    assert np.all(w[cross] > 0.0)  # E source -> positive weight


def test_truth_gates_not_escalated():
    model = jtfne.construct(_interarea_cfg())
    man = model.manifest()
    assert man.get("claim_level", "computational_scaffold") == "computational_scaffold"
    assert man.get("physical_amplitude_calibrated", False) is False


def _interarea_cfg():
    return (
        _base("V1")
        .population(N=100, neurons={"L1": 10, "L2": 25, "L3": 20, "L4": 10, "L5": 20, "L6": 15}, name="V4")
        .inter_column_connectivity(source_area="V1", target_area="V4", p_feedforward=0.3, p_feedback=0.3, seed=0)
    )


@pytest.mark.parametrize("like,proxy", [
    ("lfp_like", "lfp_proxy"), ("csd_like", "csd_proxy"),
    ("eeg_like", "eeg_proxy"), ("meg_like", "meg_proxy"),
])
def test_probe_kind_rejects_retired_like(like, proxy):
    # `*_like` is fully retired (no aliases): proxy-only vocabulary.
    with pytest.raises(ValueError, match="retired"):
        jtfne.Configuration().probe(kind=like)
    assert jtfne.Configuration().probe(kind=proxy).probes[-1]["kind"] == proxy


def test_probe_modes_reject_retired_like():
    with pytest.raises(ValueError, match="retired"):
        jtfne.Configuration().probes(["spikes", "csd_like"])


def test_signal_get_rejects_retired_like():
    model = jtfne.construct(_base())
    sig = model.simulate(jtfne.simulation(duration_ms=20.0, dt_ms=0.5, seed=0))
    assert sig.get("lfp_proxy") is not None
    with pytest.raises(KeyError):
        sig.get("lfp_like")


def test_inter_column_connectivity_accumulates_specs():
    # Multiple calls accumulate (do not overwrite) -> a list of specs that all
    # materialize. Regression: previously each call overwrote the single key so
    # only the last adjacent pair wired.
    cfg = (
        jtfne.Configuration()
        .inter_column_connectivity(source_area="V1", target_area="V2", seed=0)
        .inter_column_connectivity(source_area="V2", target_area="V4", seed=0)
    )
    specs = cfg.metadata["inter_column_connectivity"]
    assert isinstance(specs, list) and len(specs) == 2
    assert {s["source_area"] for s in specs} == {"V1", "V2"}


def test_build_multi_area_columns_wires_bidirectional_hierarchy():
    # The full hierarchy must carry feedforward (lo->hi, L2/3->L4) AND genuine
    # top-down feedback (hi->lo, L6->L1/L5) for every adjacent pair.
    areas = ["V1", "V2", "V4"]
    cfg = (
        jtfne.build_multi_area_columns(areas, n_per_area=120, ei_profile="canonical",
                                       p_feedforward=0.3, p_feedback=0.2)
        .runtime(seed=0).set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"]).field(domain="laminar_column", conductivity="proxy")
    )
    model = jtfne.construct(cfg)
    nt = pd.DataFrame(model.neuron_table())
    area = nt["area"].values; layer = nt["layer"].values
    el = model.params["edge_list"]; pre = np.asarray(el.pre); post = np.asarray(el.post)
    rank = {a: i for i, a in enumerate(areas)}
    inter = area[pre] != area[post]
    rpre = np.array([rank[a] for a in area[pre]]); rpost = np.array([rank[a] for a in area[post]])
    ff = inter & (rpre < rpost); fb = inter & (rpre > rpost)
    assert int(ff.sum()) > 0 and int(fb.sum()) > 0          # both directions present
    assert set(layer[post[ff]]) == {"L4"}                   # feedforward targets L4
    assert set(layer[pre[ff]]) == {"L2/3"}                  # from superficial L2/3
    assert set(layer[pre[fb]]) == {"L6"}                    # feedback from deep L6
    assert set(layer[post[fb]]) <= {"L1", "L5"}             # to L1/L5
    # all adjacent pairs wired in both directions
    fwd_pairs = set(zip(area[pre[ff]].tolist(), area[post[ff]].tolist()))
    assert ("V1", "V2") in fwd_pairs and ("V2", "V4") in fwd_pairs
