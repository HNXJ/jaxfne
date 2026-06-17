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
    jtfne.enable_x64()
    model = jtfne.construct(_base())
    nt = pd.DataFrame(model.neuron_table())
    counts = nt.groupby("layer").size().to_dict()
    assert counts == {"L1": 10, "L2": 25, "L3": 20, "L4": 10, "L5": 20, "L6": 15}


def test_population_counts_decoupled_from_thickness():
    # L5 is the thickest (0.30) but must NOT get the most neurons
    jtfne.enable_x64()
    model = jtfne.construct(_base())
    nt = pd.DataFrame(model.neuron_table())
    counts = nt.groupby("layer").size().to_dict()
    assert counts["L2"] > counts["L5"]  # thin-dense L2 beats thick-sparse L5


def test_population_single_N_change_scales_proportionally():
    jtfne.enable_x64()
    model = jtfne.construct(_base(N=200))
    nt = pd.DataFrame(model.neuron_table())
    counts = nt.groupby("layer").size().to_dict()
    assert sum(counts.values()) == 200
    assert counts == {"L1": 20, "L2": 50, "L3": 40, "L4": 20, "L5": 40, "L6": 30}


def _interarea_model(p_ff, p_fb, l2l=None):
    jtfne.enable_x64()
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
    jtfne.enable_x64()
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
