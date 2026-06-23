"""Golden-output regression for connect(), captured BEFORE splitting it into
staged helpers (see core.connect's "# -- stage --" comments: input
validation, runtime reconciliation, namespace handling, merged emitter,
merged recurrence, merged positions, merged neuron metadata, cross-model
edges, merged cfg, merged static).

These assertions are deliberately concrete (exact values, not just shapes)
so a refactor that drops or mis-threads a stage's local state is caught.
Written against the CURRENT (pre-split) connect() implementation; must keep
passing unchanged after the split.
"""

import warnings

import numpy as np
import pytest

import jaxfne as jtfne


def _small_model(area, n=8, cell_types=None, seed=0, dt_ms=0.5):
    cfg = jtfne.laminar_cortex_config(
        seed=seed, duration_ms=20.0, dt_ms=dt_ms, areas=(area,), layers=("L4",),
        cell_types=cell_types or {"E": 0.5, "PV": 0.5}, n=n, emitter="izhikevich",
    )
    return jtfne.construct(cfg)


def test_connect_basic_two_model_merge_sizes_and_offsets():
    m1 = _small_model("V1", n=6)
    m2 = _small_model("V4", n=4)
    merged = jtfne.connect(m1, m2)

    n1 = m1.params["emitter"].n_neurons
    n2 = m2.params["emitter"].n_neurons
    assert merged.params["emitter"].n_neurons == n1 + n2

    el1 = m1.params["edge_list"]
    el2 = m2.params["edge_list"]
    merged_el = merged.params["edge_list"]
    assert merged_el.n_edges == el1.n_edges + el2.n_edges

    # member-2 edges must be offset by n1 (block-diagonal, no spurious cross-recurrence)
    pre2 = np.asarray(el2.pre)
    merged_pre = np.asarray(merged_el.pre)
    assert np.array_equal(merged_pre[el1.n_edges:], pre2 + n1)

    assert merged.params["positions"].shape[0] == n1 + n2
    assert len(merged.static["neuron_metadata"]) == n1 + n2


def test_connect_namespace_prefixes_areas_and_tags_model_index():
    m1 = _small_model("V1", n=5)
    m2 = _small_model("V1", n=3)  # same area name -- requires namespace
    merged = jtfne.connect(m1, m2, namespace=("A", "B"))

    rows = merged.static["neuron_metadata"]
    areas = {r["area"] for r in rows}
    assert areas == {"A/V1", "B/V1"}

    model_idx = [r["model"] for r in rows]
    assert model_idx == [0] * 5 + [1] * 3

    neuron_ids = [r["neuron_id"] for r in rows]
    assert neuron_ids == list(range(8))


def test_connect_area_collision_without_namespace_raises_under_strict():
    m1 = _small_model("V1", n=4)
    m2 = _small_model("V1", n=4)
    with pytest.raises(ValueError, match="area label collision"):
        jtfne.connect(m1, m2, strict=True)


def test_connect_area_collision_without_namespace_warns_when_not_strict():
    m1 = _small_model("V1", n=4)
    m2 = _small_model("V1", n=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merged = jtfne.connect(m1, m2, strict=False)
    assert any("area label collision" in str(w.message) for w in caught)
    assert merged.params["emitter"].n_neurons == 8


def test_connect_cross_model_edges_compiled_and_recorded_in_circuit():
    # NOTE: cell_types={"E": 1.0} does NOT guarantee a pure population --
    # laminar_cortex_config still fills remaining slots with other types
    # (same builder quirk noted in the prior session's custom-cfg pipeline
    # test). Count actual matches from neuron_table() rather than assuming
    # the full n is selected.
    m1 = _small_model("V1", n=6, cell_types={"E": 1.0})
    m2 = _small_model("V4", n=6, cell_types={"PV": 1.0})
    n1_e = sum(1 for r in m1.neuron_table() if r["cell_type"] == "E")
    n2_pv = sum(1 for r in m2.neuron_table() if r["cell_type"] == "PV")
    assert n1_e > 0 and n2_pv > 0

    edges = [dict(name="v1_to_v4", source={"model": 0, "cell_type": "E"},
                  target={"model": 1, "cell_type": "PV"}, probability=1.0,
                  weight=0.3, sign="excitatory")]
    merged = jtfne.connect(m1, m2, edges=edges)

    n1 = m1.params["emitter"].n_neurons
    el1 = m1.params["edge_list"]
    el2 = m2.params["edge_list"]
    merged_el = merged.params["edge_list"]
    expected_cross = n1_e * n2_pv  # exact E(m1) x PV(m2) product, probability=1.0
    assert merged_el.n_edges == el1.n_edges + el2.n_edges + expected_cross

    cross_pre = np.asarray(merged_el.pre)[el1.n_edges + el2.n_edges:]
    cross_post = np.asarray(merged_el.post)[el1.n_edges + el2.n_edges:]
    assert np.all(cross_pre < n1)
    assert np.all(cross_post >= n1)

    assert merged.cfg.metadata["ensemble"]["cross_model_edges"] == expected_cross
    cross_rules = [c for c in merged.cfg.metadata["circuit"]["connections"] if c.get("scope") == "cross_model"]
    assert len(cross_rules) == 1
    assert cross_rules[0]["compiled_n_edges"] == expected_cross
    assert cross_rules[0]["status"] == "compiled"


def test_connect_layout_offset_x_separates_columns_spatially():
    m1 = _small_model("V1", n=5)
    m2 = _small_model("V4", n=5)
    merged_offset = jtfne.connect(m1, m2, layout="offset_x")
    merged_keep = jtfne.connect(m1, m2, layout="keep")

    pos_offset = np.asarray(merged_offset.params["positions"])
    pos_keep = np.asarray(merged_keep.params["positions"])

    n1 = m1.params["emitter"].n_neurons
    # offset_x: member 2's x-range must not overlap member 1's x-range
    x1_max = pos_offset[:n1, 0].max()
    x2_min = pos_offset[n1:, 0].min()
    assert x2_min > x1_max

    # keep: positions are untouched (identical to the unmerged member arrays on x)
    p1_orig = np.asarray(m1.params["positions"])[:, 0]
    p2_orig = np.asarray(m2.params["positions"])[:, 0]
    np.testing.assert_allclose(pos_keep[:n1, 0], p1_orig)
    np.testing.assert_allclose(pos_keep[n1:, 0], p2_orig)


def test_connect_mismatched_dt_ms_raises_under_strict_warns_otherwise():
    m1 = _small_model("V1", n=4, dt_ms=0.5)
    m2 = _small_model("V4", n=4, dt_ms=1.0)
    with pytest.raises(ValueError, match="mismatched dt_ms"):
        jtfne.connect(m1, m2, strict=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merged = jtfne.connect(m1, m2, strict=False)
    assert any("mismatched dt_ms" in str(w.message) for w in caught)
    assert merged.params["emitter"].n_neurons == 8


def test_connect_requires_at_least_two_models():
    m1 = _small_model("V1", n=4)
    with pytest.raises(ValueError, match="at least two models"):
        jtfne.connect(m1)


def test_connect_rejects_non_model_argument():
    m1 = _small_model("V1", n=4)
    with pytest.raises(TypeError, match="not a Model"):
        jtfne.connect(m1, "not_a_model")


def test_connect_static_reconciles_operator_status_and_n_contacts():
    m1 = _small_model("V1", n=4)
    m2 = _small_model("V4", n=4)
    merged = jtfne.connect(m1, m2)

    assert merged.static["n_contacts"] == m1.static.get("n_contacts", 16)
    assert "operator_status" in merged.static
    assert merged.static["ensemble"]["n_models"] == 2
    assert merged.static["ensemble"]["model_sizes"] == [4, 4]
    assert merged.static["ensemble"]["n_neurons_total"] == 8


def test_connect_ensemble_name_recorded():
    m1 = _small_model("V1", n=4)
    m2 = _small_model("V4", n=4)
    merged = jtfne.connect(m1, m2, name="v1_v4_pair")
    assert merged.cfg.metadata["ensemble"]["name"] == "v1_v4_pair"
