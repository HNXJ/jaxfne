"""Stage B: PCL retinotopic topology on the jaxfne core (lane 2b).

B1 footprint connection rule with optional patch weight-sharing, tested
against the oracle cover tables for identical (pre, post) sets; B2 complex
pooling with the oracle's broadcast-sum semantics (every covering cell gets
the spike, no max); B3 same-step local inhibition in R rounds in declared
ascending depth order, tested against the oracle primitives on a hand-built
2-3 cell case. Each mechanism reads posed positions; every misdeclaration
is refused by name.

Oracle (read-only): ``artifacts/etudes/pcl/port/pcl_event_ref.py``,
a transcription of the authors' C++. ``PCLEventRef(directory=None)`` uses
the DefaultConfig.cpp layout with random weights, so no warehouse data is
needed here. Times are integer microseconds throughout.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.connectivity import compile_connection_rules
from jaxfne.pcl_topology import (
    accumulate_shared_update,
    expand_shared,
    footprint_pairs,
    pcl_grid_origins,
    resolve_local_inhibition,
    validate_footprint,
    validate_rounds,
    validate_weight_sharing,
)

pytest.importorskip("h5py", reason="the PCL oracle reads HDF5 event files")

from artifacts.etudes.pcl.port import pcl_event_ref as oracle  # noqa: E402

DT_MS = 0.5
DUR_MS = 20.0


def _nt(*args, **kwargs):
    return nt.NeuronType.make(*args, **kwargs)


def _ampa_static():
    return nt.StaticParams(dT_ms=5.0)


def _ref():
    return oracle.PCLEventRef(directory=None, learning=False)


# B1: the footprint rule reproduces the oracle cover tables.
def test_grid_origins_match_the_oracle():
    ref = _ref()
    assert pcl_grid_origins(150, 10, 3, 9) == list(ref.s_ox)
    assert pcl_grid_origins(100, 10, 3, 9) == list(ref.s_oy)
    assert pcl_grid_origins(0, 4, 3, 6) == list(ref.c_ox)
    assert pcl_grid_origins(0, 4, 3, 6) == list(ref.c_oy)


def _simple_origins(ref):
    """One post entry per simple (i, j): ids, origins, flat k=0 index."""
    ids, xy, flat = [], [], []
    for i in range(ref.NX):
        for j in range(ref.NY):
            ids.append((i * ref.NY + j) * ref.ND)
            xy.append((float(ref.s_ox[i]), float(ref.s_oy[j])))
            flat.append((i, j))
    return ids, np.array(xy, dtype=float), flat


def test_footprint_matches_cover_x_on_the_full_axis():
    ref = _ref()
    post_ids, post_xy, _ = _simple_origins(ref)
    y0 = 100
    assert list(ref.cover_y[y0]) == [0]
    for x in range(ref.VW):
        pairs, _ = footprint_pairs(
            [x], np.array([(float(x), float(y0))]),
            post_ids, post_xy,
            {"size": [float(ref.SX), float(ref.SY)]},
        )
        want = {
            (x, (i * ref.NY + j) * ref.ND)
            for i in ref.cover_x[x] for j in ref.cover_y[y0]
        }
        assert set(pairs) == want, f"pixel x={x}"


def test_footprint_matches_cover_y_on_the_full_axis():
    ref = _ref()
    post_ids, post_xy, _ = _simple_origins(ref)
    x0 = 150
    assert list(ref.cover_x[x0]) == [0]
    for y in range(ref.VH):
        pairs, _ = footprint_pairs(
            [y], np.array([(float(x0), float(y))]),
            post_ids, post_xy,
            {"size": [float(ref.SX), float(ref.SY)]},
        )
        want = {
            (y, (i * ref.NY + j) * ref.ND)
            for i in ref.cover_x[x0] for j in ref.cover_y[y]
        }
        assert set(pairs) == want, f"pixel y={y}"


def test_footprint_matches_cover_with_depths_and_sharing_groups():
    ref = _ref()
    y0 = 100
    xs = list(range(150, 214, 6))
    pre_ids = list(range(len(xs)))
    pre_xy = np.array([(float(x), float(y0)) for x in xs])
    post_ids, post_xy, groups_want = [], [], []
    for i in range(ref.NX):
        for j in range(ref.NY):
            for k in range(ref.ND):
                post_ids.append((i * ref.NY + j) * ref.ND + k)
                post_xy.append((float(ref.s_ox[i]), float(ref.s_oy[j])))
                groups_want.append(k)
    post_xy = np.array(post_xy, dtype=float)
    pairs, groups = footprint_pairs(
        pre_ids, pre_xy, post_ids, post_xy,
        {"size": [float(ref.SX), float(ref.SY)]},
    )
    want, want_groups = [], []
    for a, x in enumerate(xs):
        for i in ref.cover_x[x]:
            for j in ref.cover_y[y0]:
                for k in range(ref.ND):
                    want.append((a, (i * ref.NY + j) * ref.ND + k))
                    want_groups.append(k)
    # Contract is the identical (pre, post) set (B1); order is post-major
    # by construction, so compare order-insensitively with groups attached.
    assert sorted(pairs) == sorted(want)
    assert sorted(zip(pairs, groups)) == sorted(zip(want, want_groups))


# B2: complex pooling is broadcast-sum over the oracle's c_cover.
def test_complex_pooling_matches_c_cover_with_depth_groups():
    ref = _ref()
    sites = [(i, j) for i in range(0, 9, 2) for j in range(0, 9, 2)]
    pre_ids = list(range(len(sites)))
    pre_xy = np.array([(float(i), float(j)) for i, j in sites])
    post_ids, post_xy = [], []
    for cx in range(ref.CX):
        for cy in range(ref.CY):
            for cz in range(ref.CD):
                post_ids.append((cx * ref.CY + cy) * ref.CD + cz)
                post_xy.append((float(ref.c_ox[cx]), float(ref.c_oy[cy])))
    post_xy = np.array(post_xy, dtype=float)
    pairs, groups = footprint_pairs(
        pre_ids, pre_xy, post_ids, post_xy,
        {"size": [float(ref.CSX), float(ref.CSY)]},
    )
    want, want_groups = [], []
    for a, (i, j) in enumerate(sites):
        for cx, cy in ref.c_cover[(i, j)]:
            for cz in range(ref.CD):
                want.append((a, (cx * ref.CY + cy) * ref.CD + cz))
                want_groups.append(cz)
    # Same set contract as B1 (post-major order by construction).
    assert sorted(pairs) == sorted(want)
    assert sorted(zip(pairs, groups)) == sorted(zip(want, want_groups))


def test_complex_pooling_fans_out_to_every_covering_cell():
    """Pooling is a sum, not a max: one spike reaches all covering cells."""
    ref = _ref()
    post_ids = [(cx * ref.CY + cy) * ref.CD + cz
                for cx in range(ref.CX) for cy in range(ref.CY)
                for cz in range(ref.CD)]
    post_xy = np.array([(float(ref.c_ox[cx]), float(ref.c_oy[cy]))
                        for cx in range(ref.CX) for cy in range(ref.CY)
                        for cz in range(ref.CD)])
    pairs, _ = footprint_pairs(
        [0], np.array([(0.0, 0.0)]), post_ids, post_xy,
        {"size": [float(ref.CSX), float(ref.CSY)]},
    )
    assert len(pairs) == len(ref.c_cover[(0, 0)]) * ref.CD
    assert len(pairs) > 1


def test_shared_weights_gather_and_scatter_per_group():
    shared = np.array([[1.0, 2.0], [10.0, 20.0]])
    groups = [0, 1, 0, 1]
    np.testing.assert_array_equal(
        expand_shared(shared, groups),
        [[1.0, 2.0], [10.0, 20.0], [1.0, 2.0], [10.0, 20.0]],
    )
    out = accumulate_shared_update(
        shared, groups,
        np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]),
    )
    np.testing.assert_allclose(out, [[3.0, 2.0], [10.0, 23.0]], rtol=1e-12)


# B3: same-step local inhibition against the oracle primitives.
def _simple_cells(ref, cfg, v, t0):
    cells = [oracle.Cell(cfg.VTHRESH) for _ in range(len(v))]
    for cell, value in zip(cells, v):
        cell.V = float(value)
        cell.tlast = t0
    return cells


def test_local_inhibition_matches_oracle_simple_cells():
    ref = _ref()
    raw = dict(oracle.DEFAULT_SIMPLE)
    raw["ETA_SRA"] = 2.0  # exercise the adaptation term (default 0)
    cfg = oracle.CellConfig("simple", raw)
    t0, t = 2_000_000, 2_001_000
    v0 = np.array([28.0, 5.0, 29.0])
    inputs = np.array([5.0, 0.0, 5.0])
    wi = np.asarray(ref.wi_s[:3][:, :3], dtype=float)
    np.fill_diagonal(wi, 0.0)
    cells = _simple_cells(ref, cfg, v0, t0)
    winners = []
    for d in range(3):
        if oracle.excite_simple(cells[d], cfg, t, float(inputs[d])):
            winners.append(d)
            for e in range(3):
                if e != d:
                    oracle.inhibit_simple(cells[e], cfg, t, float(wi[e, d]))
    out = resolve_local_inhibition(
        v=v0, adapt=np.zeros(3), t_last=np.full(3, t0), t_spike=np.zeros(3),
        t_now=float(t), thr=cfg.VTHRESH, inputs=inputs, inhib=wi,
        tau_m=cfg.TAU_M, tau_adapt=cfg.TAU_SRA, delta_adapt=cfg.DELTA_SRA,
        delta_rp=cfg.DELTA_RP, tau_rp=cfg.TAU_RP, v_reset=cfg.VRESET,
        neg_limit=oracle.NEG_LIMIT, adaptive=True, rounds=3,
    )
    assert winners == np.flatnonzero(out["winners"]).tolist()
    assert out["missed"] == 0
    np.testing.assert_allclose(out["v_next"], [c.V for c in cells], rtol=1e-12)
    np.testing.assert_allclose(out["adapt_next"], [c.A for c in cells], rtol=1e-12)
    np.testing.assert_allclose(
        out["t_spike_next"], [float(c.t_spk) for c in cells], rtol=1e-12)


def test_local_inhibition_matches_oracle_complex_rows():
    ref = _ref()
    cfg = ref.ccfg
    t0, t = 2_000_000, 2_005_000
    v0 = np.array([2.0, 2.5])
    inputs = np.array([[0.0, 1.5], [1.5, 0.0]])
    wi = np.array([[0.0, 0.7], [0.7, 0.0]])
    cells = _simple_cells(ref, cfg, v0, t0)
    winners = []
    for row in inputs.tolist():
        for d in range(2):
            if oracle.excite_complex(cells[d], cfg, t, float(row[d])):
                winners.append(d)
                for e in range(2):
                    if e != d:
                        oracle.inhibit_complex(cells[e], cfg, t, float(wi[e, d]))
    out = resolve_local_inhibition(
        v=v0, adapt=None, t_last=np.full(2, t0), t_spike=np.zeros(2),
        t_now=float(t), thr=cfg.VTHRESH, inputs=inputs, inhib=wi,
        tau_m=cfg.TAU_M, tau_adapt=1.0, delta_adapt=0.0,
        delta_rp=cfg.DELTA_RP, tau_rp=cfg.TAU_RP, v_reset=cfg.VRESET,
        neg_limit=oracle.NEG_LIMIT, adaptive=False, rounds=2,
    )
    assert winners == np.flatnonzero(out["winners"]).tolist()
    assert out["adapt_next"] is None
    np.testing.assert_allclose(out["v_next"], [c.V for c in cells], rtol=1e-12)


def test_local_inhibition_order_change_alters_winners():
    kw = dict(
        v=np.array([28.0, 28.0]), adapt=np.zeros(2),
        t_last=np.zeros(2), t_spike=np.zeros(2), t_now=1000.0,
        thr=30.0, inputs=np.array([5.0, 5.0]), inhib=np.zeros((2, 2)),
        tau_m=18000.0, tau_adapt=100000.0, delta_adapt=0.0,
        delta_rp=10.0, tau_rp=5000.0, v_reset=-10.0,
        neg_limit=-20.0, adaptive=True, rounds=1,
    )
    asc = resolve_local_inhibition(**kw, order="ascending")
    desc = resolve_local_inhibition(**kw, order="descending")
    assert asc["winners"].tolist() == [True, False]
    assert desc["winners"].tolist() == [False, True]
    with pytest.raises(ValueError, match="order"):
        resolve_local_inhibition(**kw, order="sideways")


def test_local_inhibition_rounds_limit_winners_in_depth_order():
    out = resolve_local_inhibition(
        v=np.array([28.0, 28.0, 28.0]), adapt=np.zeros(3),
        t_last=np.zeros(3), t_spike=np.zeros(3), t_now=1000.0,
        thr=30.0, inputs=np.array([5.0, 5.0, 5.0]),
        inhib=np.zeros((3, 3)),
        tau_m=18000.0, tau_adapt=100000.0, delta_adapt=0.0,
        delta_rp=10.0, tau_rp=5000.0, v_reset=-10.0,
        neg_limit=-20.0, adaptive=True, rounds=1,
    )
    assert out["winners"].tolist() == [True, False, False]
    assert out["missed"] == 2
    out = resolve_local_inhibition(
        v=np.array([28.0, 28.0, 28.0]), adapt=np.zeros(3),
        t_last=np.zeros(3), t_spike=np.zeros(3), t_now=1000.0,
        thr=30.0, inputs=np.array([5.0, 5.0, 5.0]),
        inhib=np.zeros((3, 3)),
        tau_m=18000.0, tau_adapt=100000.0, delta_adapt=0.0,
        delta_rp=10.0, tau_rp=5000.0, v_reset=-10.0,
        neg_limit=-20.0, adaptive=True, rounds=3,
    )
    assert out["winners"].tolist() == [True, True, True]
    assert out["missed"] == 0


# Refusals, by name.
@pytest.mark.parametrize("bad", [
    {"offset": [0.0, 0.0]},  # size required
    {"size": [0.0, 2.0]},  # non-positive
    {"size": [2.0, np.inf]},  # non-finite
    {"size": [True, 2.0]},  # bool
    {"size": ["2", 2.0]},  # string
    {"size": [2.0]},  # not a pair
    {"size": [2.0, 2.0], "pooling": "max"},  # no pooling parameter exists
    {"size": [2.0, 2.0], "stride": 1.0},  # unknown key
    "patch",  # not a mapping
])
def test_bad_footprints_are_refused(bad):
    with pytest.raises(ValueError, match="footprint"):
        validate_footprint(bad, label="probe")


@pytest.mark.parametrize("bad", ["map", "", True, 0])
def test_bad_weight_sharing_is_refused(bad):
    with pytest.raises(ValueError, match="weight_sharing"):
        validate_weight_sharing(bad, label="probe")


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2", None])
def test_bad_rounds_are_refused(bad):
    with pytest.raises(ValueError, match="rounds"):
        validate_rounds(bad, label="probe")


def test_inhibition_refusals_name_themselves():
    v = np.array([0.0, 0.0])
    base = dict(v=v, adapt=None, t_last=np.zeros(2), t_spike=np.zeros(2),
                t_now=1.0, thr=1.0, inputs=np.zeros(2),
                tau_m=10.0, tau_adapt=10.0, delta_adapt=0.0,
                delta_rp=1.0, tau_rp=5.0, v_reset=0.0,
                neg_limit=-20.0, adaptive=False, rounds=2)
    with pytest.raises(ValueError, match="zero diagonal"):
        resolve_local_inhibition(inhib=np.eye(2), **base)
    bad_thr = dict(base, thr=np.zeros(3))
    with pytest.raises(ValueError, match="thr"):
        resolve_local_inhibition(inhib=np.zeros((2, 2)), **bad_thr)
    bad_inputs = dict(base, inputs=np.zeros(3))
    with pytest.raises(ValueError, match="inputs"):
        resolve_local_inhibition(inhib=np.zeros((2, 2)), **bad_inputs)
    bad_t = dict(base, t_now=-1.0)
    with pytest.raises(ValueError, match="t_now"):
        resolve_local_inhibition(inhib=np.zeros((2, 2)), **bad_t)
    bad_adapt = dict(base, adaptive=True, adapt=None)
    with pytest.raises(ValueError, match="adapt"):
        resolve_local_inhibition(inhib=np.zeros((2, 2)), **bad_adapt)


def test_shared_table_refusals_name_themselves():
    with pytest.raises(ValueError, match="2-D"):
        expand_shared(np.zeros((2, 2, 2)), [0])
    with pytest.raises(ValueError, match="group_of_edge"):
        expand_shared(np.zeros((2, 2)), [0, 5])
    with pytest.raises(ValueError, match="disagree"):
        accumulate_shared_update(np.zeros((2, 2)), [0], np.zeros((2, 2)))


def test_footprint_pairs_refuses_bad_coordinates():
    with pytest.raises(ValueError, match="pre_xy"):
        footprint_pairs([0], np.array([[np.nan, 0.0]]), [1],
                        np.array([[0.0, 0.0]]), {"size": [2.0, 2.0]})


# The compiled rule: deterministic pairs from posed positions.
def _rows():
    return [
        {"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E",
         "x": 0.0, "y": 0.0, "z": 0.0},
        {"neuron_id": 1, "area": "V1", "layer": "L4", "cell_type": "E",
         "x": 1.0, "y": 0.0, "z": 0.0},
        {"neuron_id": 2, "area": "V1", "layer": "L4", "cell_type": "E",
         "x": 2.0, "y": 1.0, "z": 0.0},
        {"neuron_id": 3, "area": "V1", "layer": "L2", "cell_type": "E",
         "x": 0.0, "y": 0.0, "z": 1.0},
        {"neuron_id": 4, "area": "V1", "layer": "L2", "cell_type": "E",
         "x": 0.0, "y": 0.0, "z": 1.0},
    ]


def _mech():
    return [{"name": "AMPA", "kind": "AMPA", "params": {"tau_ms": 5.0}}]


def test_compiler_footprint_rule_with_sharing_groups():
    res = compile_connection_rules(
        _rows(),
        [{"name": "fp",
          "source": {"area": "V1", "layer": "L4"},
          "target": {"area": "V1", "layer": "L2"},
          "mechanism": "AMPA",
          "footprint": {"size": [2.0, 2.0], "offset": [0.0, 0.0]},
          "weight_sharing": "patch"}],
        _mech(), seed=0)
    assert [tuple(e) for e in zip(np.asarray(res.edge_pre), np.asarray(res.edge_post))] == [
        (0, 3), (1, 3), (0, 4), (1, 4)]
    row = res.connection_table[0]
    assert row["footprint"] == {"size": [2.0, 2.0], "offset": [0.0, 0.0]}
    assert row["weight_sharing"] == "patch"
    assert row["sharing_group_of_edge"] == [0, 0, 1, 1]
    assert row["n_sharing_groups"] == 2
    assert row["probability"] is None


def test_compiler_footprint_refusals_name_themselves():
    base = {"name": "fp", "source": {"area": "V1", "layer": "L4"},
            "target": {"area": "V1", "layer": "L2"}, "mechanism": "AMPA"}
    bad_rules = [
        dict(base, weight_sharing="patch"),  # sharing needs footprint
        dict(base, footprint={"size": [2.0, 2.0]}, probability=1.0),
        dict(base, footprint={"size": [2.0, 2.0]}, max_in_degree=2),
        dict(base, footprint={"size": [2.0, 2.0]}, sampling="bernoulli"),
        dict(base, footprint={"size": [2.0, 2.0]}, weight_sharing="map"),
    ]
    for rule in bad_rules:
        with pytest.raises(ValueError, match="footprint|weight_sharing"):
            compile_connection_rules(_rows(), [rule], _mech(), seed=0)


def test_compiler_footprint_needs_posed_positions():
    rows = [dict(r) for r in _rows()]
    del rows[0]["x"]
    with pytest.raises(ValueError, match="x/y"):
        compile_connection_rules(
            rows,
            [{"name": "fp", "source": {"area": "V1", "layer": "L4"},
              "target": {"area": "V1", "layer": "L2"}, "mechanism": "AMPA",
              "footprint": {"size": [2.0, 2.0]}}],
            _mech(), seed=0)


# Tensor declarations: footprint and weight_sharing ride the bridge.
def _tensor(**conn_kwargs):
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(), **conn_kwargs)
    return nt.NeuronalTensor(
        name="pcl-topology",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=4, neuron_types=[_nt("E")])],
            inter_connections=[conn])])


def _construct(tensor, emitter="izhikevich"):
    runtime = nt.RuntimeConfiguration(
        seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter=emitter)
    return jtfne.construct(tensor, runtime)


def test_tensor_footprint_reaches_the_compiled_rule():
    tensor = _tensor(footprint={"size": [10.0, 10.0], "offset": [-5.0, -5.0]},
                     weight_sharing="patch", allow_self=True)
    model = _construct(tensor)
    rules = model.cfg.metadata["circuit"]["connections"]
    assert rules[0]["footprint"] == {"size": [10.0, 10.0], "offset": [-5.0, -5.0]}
    assert rules[0]["weight_sharing"] == "patch"
    assert rules[0]["probability"] is None
    rows = model.neuron_table()
    ids = [int(r["neuron_id"]) for r in rows]
    xy = np.array([(float(r["x"]), float(r["y"])) for r in rows])
    want, groups = footprint_pairs(ids, xy, ids, xy, {"size": [10.0, 10.0],
                                                       "offset": [-5.0, -5.0]})
    edges = model.params["edge_list"]
    got = [(int(a), int(b)) for a, b in zip(np.asarray(edges.pre),
                                            np.asarray(edges.post))]
    assert got == want
    # Per-edge sharing groups live on the compiler's connection-table row
    # (pinned by test_compiler_footprint_rule_with_sharing_groups above);
    # the model receipt carries the declaration (footprint + mode).


def test_undeclared_footprint_writes_no_rule_key():
    model = _construct(_tensor())
    assert "footprint" not in model.cfg.metadata["circuit"]["connections"][0]
    assert "weight_sharing" not in model.cfg.metadata["circuit"]["connections"][0]
    d = _tensor().to_dict()
    assert "footprint" not in d["areas"][0]["inter_connections"][0]
    assert "weight_sharing" not in d["areas"][0]["inter_connections"][0]


def test_declared_footprint_survives_to_dict():
    d = _tensor(footprint={"size": [10.0, 10.0]},
                weight_sharing="patch").to_dict()
    conn = d["areas"][0]["inter_connections"][0]
    assert conn["footprint"] == {"size": [10.0, 10.0], "offset": [0.0, 0.0]}
    assert conn["weight_sharing"] == "patch"


def test_tensor_footprint_declaration_refusals():
    with pytest.raises(ValueError, match="footprint"):
        _tensor(footprint={"size": [10.0, 10.0]}, max_in_degree=2)
    with pytest.raises(ValueError, match="footprint"):
        _tensor(footprint={"size": [10.0, 10.0]}, sampling="bernoulli")
    with pytest.raises(ValueError, match="weight_sharing"):
        _tensor(weight_sharing="patch")
    with pytest.raises(ValueError, match="footprint"):
        nt.AreaConnection(
            source_area="A", source_layer="L4", source_neuron_type="E",
            target_area="B", target_layer="L4", target_neuron_type="E",
            probability=0.5, footprint={"size": [10.0, 10.0]})


def test_footprint_refused_on_the_adex_route():
    tensor = nt.NeuronalTensor(
        name="pcl-topology-adex",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=4,
                             neuron_types=[_nt("E", emitter="adex")])],
            inter_connections=[nt.InterConnection(
                source_layer="L4", source_neuron_type="E",
                target_layer="L4", target_neuron_type="E",
                mechanism="AMPA",
                static=nt.StaticParams(
                    dT_ms=5.0, reversal_potentials_mV={"AMPA": 0.0}),
                delay_ms=1.0,
                footprint={"size": [10.0, 10.0]})])])
    with pytest.raises(ValueError, match="footprint"):
        _construct(tensor, emitter="adex")


# B3 declaration: R and the round order are part of A (Hamm 2026-10-14).
def test_local_inhibition_declaration_validates_and_normalizes():
    from jaxfne.pcl_topology import validate_local_inhibition

    assert validate_local_inhibition(None, label="probe") is None
    assert validate_local_inhibition({"rounds": 2}, label="probe") == {
        "rounds": 2, "order": "ascending"}
    assert validate_local_inhibition(
        {"rounds": 3, "order": "descending"}, label="probe") == {
        "rounds": 3, "order": "descending"}
    for bad in [0, -1, True, 1.5, "2", None]:
        with pytest.raises(ValueError, match="rounds"):
            validate_local_inhibition({"rounds": bad}, label="probe")
    with pytest.raises(ValueError, match="order"):
        validate_local_inhibition({"rounds": 2, "order": "depth"}, label="probe")
    with pytest.raises(ValueError, match="unknown keys"):
        validate_local_inhibition({"rounds": 2, "R": 2}, label="probe")
    with pytest.raises(ValueError, match="rounds"):
        validate_local_inhibition({"order": "ascending"}, label="probe")
    with pytest.raises(ValueError, match="local_inhibition"):
        _tensor(local_inhibition={"rounds": 0})


def test_undeclared_local_inhibition_writes_no_keys():
    model = _construct(_tensor())
    assert "local_inhibition" not in model.cfg.metadata["circuit"]["connections"][0]
    d = _tensor().to_dict()
    assert "local_inhibition" not in d["areas"][0]["inter_connections"][0]


def test_local_inhibition_reaches_the_compiled_row_and_lif_receipt():
    tensor = _tensor(local_inhibition={"rounds": 2, "order": "descending"})
    d = tensor.to_dict()
    assert d["areas"][0]["inter_connections"][0]["local_inhibition"] == {
        "rounds": 2, "order": "descending"}
    model = _construct(tensor, emitter="lif")
    rules = model.cfg.metadata["circuit"]["connections"]
    assert rules[0]["local_inhibition"] == {"rounds": 2, "order": "descending"}
    receipt = model.static["lif"]["local_inhibition"]
    assert receipt == [{
        "rule": "V1.L4.E->V1.L4.E", "rounds": 2, "order": "descending"}]


def test_local_inhibition_refused_on_adex_and_unexecuted_on_lif():
    tensor = _tensor(local_inhibition={"rounds": 2})
    with pytest.raises(ValueError, match="local_inhibition"):
        _construct(tensor, emitter="adex")
    model = _construct(tensor, emitter="lif")
    with pytest.raises(ValueError, match="local_inhibition"):
        jtfne.simulate(model)


def test_local_inhibition_delay_zero_refused_as_inside_step():
    tensor = _tensor(delay_ms=0.0, local_inhibition={"rounds": 2})
    model = _construct(tensor, emitter="lif")
    with pytest.raises(ValueError, match="inside the step"):
        jtfne.simulate(model)
