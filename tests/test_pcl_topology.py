"""Stage B: PCL retinotopic topology on the jaxfne core (lane 2b).

B1 footprint connection rule with optional patch weight-sharing, tested
against the oracle cover tables for identical (pre, post) sets; B2 complex
pooling with the oracle's broadcast-sum semantics (every covering cell gets
the spike, no max); B3 same-step local inhibition in R rounds in declared
ascending depth order, tested against the oracle primitives on a hand-built
2-3 cell case. Each mechanism reads posed positions; every misdeclaration
is refused by name.

Oracle (read-only): ``artifacts/etudes/papers/pcl/port/pcl_event_ref.py``,
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

from artifacts.etudes.papers.pcl.port import pcl_event_ref as oracle  # noqa: E402

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
                     allow_self=True)
    model = _construct(tensor)
    rules = model.cfg.metadata["circuit"]["connections"]
    assert rules[0]["footprint"] == {"size": [10.0, 10.0], "offset": [-5.0, -5.0]}
    assert rules[0]["probability"] is None
    rows = model.neuron_table()
    ids = [int(r["neuron_id"]) for r in rows]
    xy = {int(r["neuron_id"]): (float(r["x"]), float(r["y"])) for r in rows}
    # Hand-built window expectation (not footprint_pairs itself): a pair
    # (pre, post) is selected iff the pre position falls in the rectangle
    # [post + offset, post + offset + size) (allow_self=True keeps diagonals).
    sx, sy = 10.0, 10.0
    ox, oy = -5.0, -5.0
    want = {
        (a, b)
        for a in ids for b in ids
        if 0.0 <= xy[a][0] - (xy[b][0] + ox) < sx
        and 0.0 <= xy[a][1] - (xy[b][1] + oy) < sy
    }
    assert len(want) > 0, "fixture must select at least one footprint pair"
    edges = model.params["edge_list"]
    got = {(int(a), int(b)) for a, b in zip(np.asarray(edges.pre),
                                            np.asarray(edges.post))}
    assert got == want


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


def test_local_inhibition_resolution_defaults_and_echoes_when_declared():
    from jaxfne.pcl_topology import validate_local_inhibition

    assert validate_local_inhibition(
        {"rounds": 2, "resolution": "event_order"}, label="probe") == {
        "rounds": 2, "order": "ascending", "resolution": "event_order"}
    with pytest.raises(ValueError, match="resolution"):
        validate_local_inhibition(
            {"rounds": 2, "resolution": "per_event"}, label="probe")
    # Undeclared resolution echoes byte-identical static (default whole_step).
    tensor = _tensor(local_inhibition={"rounds": 2, "order": "descending"})
    d = tensor.to_dict()
    assert d["areas"][0]["inter_connections"][0]["local_inhibition"] == {
        "rounds": 2, "order": "descending"}
    model = _construct(tensor, emitter="lif")
    assert model.cfg.metadata["circuit"]["connections"][0][
        "local_inhibition"] == {"rounds": 2, "order": "descending"}
    # Declared resolution rides the same echo path everywhere.
    tensor = _tensor(local_inhibition={"rounds": 2, "resolution": "event_order"})
    d = tensor.to_dict()
    assert d["areas"][0]["inter_connections"][0]["local_inhibition"] == {
        "rounds": 2, "order": "ascending", "resolution": "event_order"}
    model = _construct(tensor, emitter="lif")
    assert model.cfg.metadata["circuit"]["connections"][0][
        "local_inhibition"] == {
        "rounds": 2, "order": "ascending", "resolution": "event_order"}
    assert model.static["lif"]["local_inhibition"] == [{
        "rule": "V1.L4.E->V1.L4.E", "rounds": 2, "order": "ascending",
        "resolution": "event_order"}]


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


def _adex_tensor(**conn_kwargs):
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(
            dT_ms=5.0, reversal_potentials_mV={"AMPA": 0.0}),
        delay_ms=1.0, **conn_kwargs)
    return nt.NeuronalTensor(
        name="pcl-topology-adex",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=4,
                             neuron_types=[_nt("E", emitter="adex")])],
            inter_connections=[conn])])


def _sharing_tensor(**conn_kwargs):
    return _tensor(
        footprint={"size": [10.0, 10.0], "offset": [-5.0, -5.0]},
        weight_sharing="patch", allow_self=True, **conn_kwargs)


def test_weight_sharing_refused_on_the_izhikevich_and_default_routes():
    tensor = _sharing_tensor()
    with pytest.raises(ValueError, match="weight_sharing"):
        _construct(tensor, emitter="izhikevich")
    with pytest.raises(ValueError, match="weight_sharing"):
        jtfne.construct(tensor, nt.RuntimeConfiguration(
            seed=0, duration_ms=DUR_MS, dt_ms=DT_MS))


def test_weight_sharing_refused_on_the_adex_route():
    tensor = _adex_tensor(
        footprint={"size": [10.0, 10.0]}, weight_sharing="patch")
    with pytest.raises(ValueError, match="weight_sharing"):
        _construct(tensor, emitter="adex")


def _sign_only_column():
    cfg = (
        jtfne.Configuration()
        .runtime(seed=1, dtype="float32", duration_ms=20.0, dt_ms=0.5)
        .column(name="c", layers=["L4"], n=20).cell_types({"E": 1.0})
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann")
    )
    return cfg


def test_weight_sharing_refused_on_the_sign_only_fallback():
    cfg = _sign_only_column().connections(
        name="rec", source={}, target={}, weight=0.03,
        footprint={"size": [10.0, 10.0]}, weight_sharing="patch")
    with pytest.raises(ValueError, match="weight_sharing"):
        jtfne.construct(cfg)


def test_weight_sharing_refused_on_the_default_simulate_path():
    mech = dict(name="ampa", kind="exponential",
                params={"tau_ms": 2.0, "receptor": "AMPA"})
    cfg = _sign_only_column().mechanisms(**mech).connections(
        name="rec", source={}, target={}, mechanism="ampa", weight=0.03,
        footprint={"size": [10.0, 10.0]}, weight_sharing="patch")
    model = jtfne.construct(cfg)
    with pytest.raises(ValueError, match="weight_sharing"):
        model.simulate(jtfne.Simulation(duration_ms=20.0, dt_ms=0.5))


def test_local_inhibition_refused_on_adex_izhikevich_and_default():
    tensor = _tensor(local_inhibition={"rounds": 2})
    with pytest.raises(ValueError, match="local_inhibition"):
        _construct(tensor, emitter="adex")
    with pytest.raises(ValueError, match="local_inhibition"):
        _construct(tensor, emitter="izhikevich")
    with pytest.raises(ValueError, match="local_inhibition"):
        jtfne.construct(tensor, nt.RuntimeConfiguration(
            seed=0, duration_ms=DUR_MS, dt_ms=DT_MS))


def test_local_inhibition_refused_on_the_adex_route():
    tensor = _adex_tensor(local_inhibition={"rounds": 2})
    with pytest.raises(ValueError, match="local_inhibition"):
        _construct(tensor, emitter="adex")


def test_local_inhibition_refused_on_the_sign_only_fallback():
    from jaxfne.connectivity import with_rule_local_inhibition

    cfg = with_rule_local_inhibition(
        _sign_only_column().connections(
            name="rec", source={}, target={}, weight=0.03),
        "rec", {"rounds": 2, "order": "ascending"})
    with pytest.raises(ValueError, match="local_inhibition"):
        jtfne.construct(cfg)


def test_local_inhibition_refused_on_the_default_simulate_path():
    from jaxfne.connectivity import with_rule_local_inhibition

    mech = dict(name="ampa", kind="exponential",
                params={"tau_ms": 2.0, "receptor": "AMPA"})
    cfg = with_rule_local_inhibition(
        _sign_only_column().mechanisms(**mech).connections(
            name="rec", source={}, target={}, mechanism="ampa",
            weight=0.03),
        "rec", {"rounds": 2, "order": "ascending"})
    model = jtfne.construct(cfg)
    with pytest.raises(ValueError, match="local_inhibition"):
        model.simulate(jtfne.Simulation(duration_ms=20.0, dt_ms=0.5))


def test_local_inhibition_delay_zero_runs_inside_the_step():
    tensor = _tensor(delay_ms=0.0, local_inhibition={"rounds": 2})
    model = _construct(tensor, emitter="lif")
    signals = model.simulate(jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
    assert signals.metadata["local_inhibition"] == {
        "rounds": 2, "order": "ascending"}
    assert int(signals.metadata["n_floored_to_one_step"]) > 0
    assert bool(np.isfinite(np.asarray(signals.V_m)).all())


def _two_inhibition_tensor(spec0, spec1):
    def _conn(layer, spec):
        return nt.InterConnection(
            source_layer=layer, source_neuron_type="E",
            target_layer=layer, target_neuron_type="E",
            mechanism="AMPA", static=_ampa_static(),
            local_inhibition=spec)
    return nt.NeuronalTensor(
        name="pcl-topology-two",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=4,
                             neuron_types=[_nt("E")]),
                    nt.Layer(name="L23", n_neurons=4,
                             neuron_types=[_nt("E")])],
            inter_connections=[
                _conn("L4", spec0), _conn("L23", spec1)])])


def test_distinct_local_inhibition_specs_refused_on_lif():
    tensor = _two_inhibition_tensor(
        {"rounds": 1}, {"rounds": 2, "order": "descending"})
    with pytest.raises(ValueError, match="local_inhibition"):
        _construct(tensor, emitter="lif")
    same = _two_inhibition_tensor({"rounds": 2}, {"rounds": 2})
    model = _construct(same, emitter="lif")
    assert len(model.static["lif"]["local_inhibition"]) == 2


def _two_position_tensor():
    """Two retinotopic positions: L4a holds 2 cells at one degenerate
    point, L4b 1 cell at another; the declaration rides L4a->L4b."""
    def _layer(name, n, point):
        return nt.Layer(
            name=name, n_neurons=n,
            neuron_types=[_nt("E", drive=0.0)],
            geometry=nt.Geometry3D(x_range=point, y_range=point))
    conn = nt.InterConnection(
        source_layer="L4a", source_neuron_type="E",
        target_layer="L4b", target_neuron_type="E",
        mechanism="AMPA", static=_ampa_static(), delay_ms=1.0,
        local_inhibition={"rounds": 1})
    return nt.NeuronalTensor(
        name="pcl-topology-two-positions",
        areas=[nt.Area(
            name="V1",
            layers=[_layer("L4a", 2, (0.0, 0.0)),
                    _layer("L4b", 1, (1.0, 1.0))],
            inter_connections=[conn])])


def test_local_inhibition_competes_per_position_on_the_lif_route():
    """Fail-first for Gap 2: grouping comes from A's posed geometry.

    All three cells driven above threshold with rounds=1: each position
    keeps its first crosser in depth (index) order, so the second
    position's cell spikes where the whole-population cap silenced it.
    """
    model = _construct(_two_position_tensor(), emitter="lif")
    assert model.static["lif"]["local_inhibition_group_of_neuron"] == [0, 0, 1]
    off = np.asarray(model.simulate(jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))).spikes)
    assert int(off.sum()) == 0, "fixture must be silent without drive"
    schedule = jtfne.stimulus_schedule(
        [{"onset_ms": 0.0, "duration_ms": DUR_MS, "amplitude": 60.0}],
        n_neurons=3)
    signals = model.simulate(jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=1,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
        paradigm=schedule)
    first = np.asarray(signals.spikes)[0]
    assert [bool(s) for s in first] == [True, False, True]


def test_pooling_executes_broadcast_sum_through_edges():
    """Gap 3 pooling execution: one simple spike reaches every covering cell.

    The (pre, post) sets already match ``c_cover`` above; here the kernel
    executes the fan-out: a driven simple spike arrives one step later with
    each covering complex cell's own ``w_c`` weight (a sum, no max). Cells
    start at rest with no refractory load, so each post voltage equals its
    weight exactly (characterization of the segment-sum path, green on
    baseline too; the new mapping is the S1 assembly in test_pcl_stage_c).
    """
    import jax.numpy as jnp

    from jaxfne.emitters_lif import make_lif_params, simulate_lif_edge

    ref = _ref()
    i = j0 = 0
    k = 3
    # Pooling positions are simple-grid indices (c_cover tests grid index
    # i against complex origins, oracle:423-428), not pixel coordinates.
    pre_xy = np.array([(float(i), float(j0))])
    post_ids, post_xy = [], []
    for cx, cy in ref.c_cover[(i, j0)]:
        for cz in range(ref.CD):
            post_ids.append(1 + (cx * ref.CY + cy) * ref.CD + cz)
            post_xy.append((float(ref.c_ox[cx]), float(ref.c_oy[cy])))
    pairs, _ = footprint_pairs(
        [0], pre_xy, post_ids, np.array(post_xy),
        {"size": [float(ref.CSX), float(ref.CSY)]},
    )
    assert len(pairs) == len(ref.c_cover[(i, j0)]) * ref.CD > 0
    n_post = max(post_ids)
    edge_pre, edge_post, edge_w = [], [], []
    for a, b in pairs:
        cx, cy = ref.c_cover[(i, j0)][(b - 1) // ref.CD]
        cz = (b - 1) % ref.CD
        edge_pre.append(a)
        edge_post.append(b)
        edge_w.append(float(ref.w_c[cz, i - ref.c_ox[cx], j0 - ref.c_oy[cy], k]))
    labels = ["simple"] + ["complex"] * n_post
    params = make_lif_params(labels)
    drive = np.zeros((3, n_post + 1), dtype=np.float32)
    drive[0, 0] = 30.0  # the simple cell spikes on step 0
    out = simulate_lif_edge(
        params, n_steps=3, dt_ms=DT_MS,
        edge_pre=np.array(edge_pre, dtype=np.int64),
        edge_post=np.array(edge_post, dtype=np.int64),
        edge_weight=np.array(edge_w, dtype=np.float64),
        edge_delay_steps=np.ones(len(edge_pre), dtype=np.int64),
        I_ext_pA=jnp.asarray(drive), record_v=True,
    )
    assert bool(out["spikes"][0, 0])
    got = np.asarray(out["V_mV"][1, 1:])
    want = np.array(edge_w, dtype=float)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def _feed_kwargs(ref):
    return dict(
        dt_ms=DT_MS, n_steps=40,
        origins_x=[float(v) for v in ref.s_ox],
        origins_y=[float(v) for v in ref.s_oy],
        size=(float(ref.SX), float(ref.SY)),
        weights=np.asarray(ref.w_s, dtype=float),
        nx=ref.NX, ny=ref.NY, nd=ref.ND,
    )


def test_dvs_feed_arrays_cover_the_oracle_tables():
    """Fail-first for Gap 3 feed: binned drive lands on the oracle's cells.

    Events at known pixels give nonzero current/arrivals on exactly the
    covered simple cells (oracle ``cover_x``/``cover_y``), with the
    oracle's weight per (depth, polarity, receptive-field offset) and one
    arrival each; a pixel no window covers contributes nothing.
    """
    from jaxfne.pcl_topology import dvs_feed_arrays

    ref = _ref()
    kw = _feed_kwargs(ref)
    x0, y0 = 160, 110
    t = np.array([0, 250, 500], dtype=np.int64)  # us; bins 0, 0, 1 at dt 0.5 ms
    cur, arr = dvs_feed_arrays(
        t, np.array([x0, x0, 7 * x0]), np.array([y0, y0, y0]),
        np.array([1, 0, 1]), **kw)
    assert cur.shape == (40, ref.NS) and arr.shape == (40, ref.NS)
    want_cells = {
        (i * ref.NY + j) * ref.ND + k
        for i in ref.cover_x[x0] for j in ref.cover_y[y0] for k in range(ref.ND)
    }
    assert {int(n) for n in np.flatnonzero(arr[0] > 0)} == want_cells
    assert {int(n) for n in np.flatnonzero(cur[0] != 0)} <= want_cells
    for i in ref.cover_x[x0]:
        for j in ref.cover_y[y0]:
            xr, yr = x0 - ref.s_ox[i], y0 - ref.s_oy[j]
            for k in range(ref.ND):
                n = (i * ref.NY + j) * ref.ND + k
                # Two events share bin 0 (polarities 1 and 0): weights add.
                assert arr[0, n] == 2.0
                assert abs(cur[0, n] - (
                    float(ref.w_s[k, 1, xr, yr]) + float(ref.w_s[k, 0, xr, yr]))) < 1e-6
    # Bin 1 holds the uncovered pixel only: nothing anywhere.
    assert int((arr[1] > 0).sum()) == 0 and float(np.abs(cur[1]).sum()) == 0.0
    # Single event, single polarity: exact weight and one arrival.
    cur1, arr1 = dvs_feed_arrays(
        np.array([1000]), np.array([x0]), np.array([y0]), np.array([1]), **kw)
    assert int(arr1.sum()) == len(want_cells)
    n = (ref.cover_x[x0][0] * ref.NY + ref.cover_y[y0][0]) * ref.ND
    assert abs(cur1[2, n] - float(ref.w_s[0, 1, x0 - ref.s_ox[ref.cover_x[x0][0]],
                                           y0 - ref.s_oy[ref.cover_y[y0][0]]])) < 1e-6


def test_event_order_builders_visit_feed_order_per_position():
    """Fail-first for the event-order fixed point plumbing.

    Synthetic 1x2 grid (depth 2, N=4): ranks carry each fed cell's first
    event offset in the step, the visit arrays sort fed cells by
    (rank, index): event order first, ascending index on ties, scoped to
    each retinotopic position, and the blocks hold the per-position
    inhibition magnitudes the cascade subtracts.
    """
    from jaxfne.pcl_topology import (
        dvs_feed_arrays,
        dvs_feed_ranks,
        feed_visit_order,
        inhibition_visit_order,
        local_inhibition_blocks,
    )

    w = np.ones((2, 2, 10, 10))
    kw = dict(dt_ms=0.5, n_steps=2, origins_x=[0.0], origins_y=[0.0, 10.0],
              size=(10.0, 10.0), weights=w, nx=1, ny=2, nd=2, t0_us=0)
    t = np.array([300, 100, 600], dtype=np.int64)
    x = np.array([5, 5, 5], dtype=np.int64)
    y = np.array([5, 5, 15], dtype=np.int64)
    p = np.array([1, 1, 1], dtype=np.int64)
    cur, arr = dvs_feed_arrays(t, x, y, p, **kw)
    ranks = dvs_feed_ranks(t, x, y, p, **kw)
    assert ranks.shape == arr.shape == (2, 4)
    # Bin 0 feeds position 0 (cells 0, 1) twice; first offset is 100 us.
    assert (arr[0] == [2, 2, 0, 0]).all()
    assert (ranks[0] == [100, 100, 0, 0]).all()
    # Bin 1 feeds position 1 (cells 2, 3) once at offset 100 us.
    assert (arr[1] == [0, 0, 1, 1]).all()
    assert (ranks[1] == [0, 0, 100, 100]).all()
    # Ranks agree with the current builder on which pairs the stream touches.
    assert ((ranks > 0) == (arr > 0)).all()
    assert ((cur != 0) == (arr > 0)).all()

    visit, lens = feed_visit_order(ranks, arr)
    assert lens.tolist() == [2, 2]
    assert visit[0, :2].tolist() == [0, 1]  # tie at 100 us: ascending index
    assert visit[1, :2].tolist() == [2, 3]

    groups = np.array([0, 0, 1, 1])
    blocks = local_inhibition_blocks(
        groups, li_pre=np.array([1, 0, 3, 2]),
        li_post=np.array([0, 1, 2, 3]),
        li_weight=np.array([8.0, 8.0, 8.0, 8.0]))
    assert blocks["members"].tolist() == [[0, 1], [2, 3]]
    assert bool(blocks["member_mask"].all())
    assert blocks["weights"][0].tolist() == [[0, 8.0], [8.0, 0]]
    assert blocks["weights"][1].tolist() == [[0, 8.0], [8.0, 0]]

    order = inhibition_visit_order(ranks, blocks)
    assert order.shape == (2, 2, 2)
    assert order[0].tolist() == [[0, 1], [0, 1]]  # unfed position: index order
    assert order[1].tolist() == [[0, 1], [0, 1]]

    # Earlier event reorders the visit: cell 1 at 50 us beats cell 0 at 400 us.
    ranks2 = np.array([[400.0, 50.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert inhibition_visit_order(ranks2, blocks)[0].tolist() == [[1, 0], [0, 1]]
    assert feed_visit_order(
        ranks2, np.array([[1, 1, 0, 0], [0, 0, 0, 0]]))[0][0].tolist() == [1, 0]

    for bad in (
        lambda: local_inhibition_blocks(
            groups, li_pre=np.array([1]), li_post=np.array([2]),
            li_weight=np.array([8.0])),  # crosses groups
        lambda: local_inhibition_blocks(
            groups, li_pre=np.array([1]), li_post=np.array([0]),
            li_weight=np.array([-8.0])),  # signed edge weight, not a magnitude
        lambda: feed_visit_order(ranks, -arr),  # negative arrivals
        lambda: inhibition_visit_order(np.full((2, 3), 0.0), blocks),  # N skew
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("event-order builder refusal must raise")


def test_dvs_feed_arrays_refusals_name_the_feed():
    """Every feed declaration changes the realized arrays or is refused."""
    from jaxfne.pcl_topology import dvs_feed_arrays

    ref = _ref()
    kw = _feed_kwargs(ref)
    bad_kw = [
        dict(kw, dt_ms=0.5005),  # fractional microseconds refused
        dict(kw, n_steps=0),
        dict(kw, size=(10.5, 10.0)),
        dict(kw, weights=np.zeros((ref.ND, 2, ref.SX + 1, ref.SY))),
    ]
    for bkw in bad_kw:
        try:
            dvs_feed_arrays(np.array([0]), np.array([160]), np.array([110]),
                            np.array([1]), **bkw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bkw} must raise")
    bad_events = [
        (np.array([-1]), np.array([160]), np.array([110]), np.array([1])),  # pre-grid
        (np.array([0]), np.array([-3]), np.array([110]), np.array([1])),  # off plane
        (np.array([0]), np.array([160]), np.array([110]), np.array([2])),  # polarity
        (np.array([0, 99999999]), np.array([160, 160]), np.array([110, 110]),
         np.array([1, 1])),  # past the last step
    ]
    for t, x, y, p in bad_events:
        try:
            dvs_feed_arrays(t, x, y, p, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"event {(t, x, y, p)} must raise")


def _scan_micro_tables():
    """Shared micro network for the event-scan builder tests (1x1 grid,
    depth 2, one complex cell; pixel (2, 2) covered, 4x4 plane)."""
    from jaxfne.pcl_topology import event_scan_tables

    w_s = np.zeros((2, 2, 1, 1), dtype=np.float64)
    w_s[1, 0, 0, 0] = 12.0
    w_s[0, 1, 0, 0] = 12.0
    tables = event_scan_tables(
        nx=1, ny=1, nd=2, cx=1, cy=1, cd=1,
        origins_x=[2.0], origins_y=[2.0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0], c_size=(1, 1),
        c_weights=np.zeros((1, 1, 1, 2), dtype=np.float64),
        vw=4, vh=4, inhib_range=(0, 0),
    )
    return tables


def test_event_scan_tables_cover_oracle_maps():
    """Fail-first for the event-scan tables: pixel cover, pooling fanout,
    and the empty lateral/top-down neighborhoods of the micro network."""
    tables = _scan_micro_tables()
    assert tables["pix_pos"].shape == (16, 1)
    assert tables["pix_n"].shape == (16,)
    # Only pixel (2, 2) is covered, by position 0 (oracle cover rule).
    assert int(tables["pix_n"].sum()) == 1
    assert int(tables["pix_n"][2 * 4 + 2]) == 1
    assert int(tables["pix_pos"][2 * 4 + 2, 0]) == 0
    # Both simple cells forward to the one complex cell (global id 2).
    assert tables["fwd_n"].tolist() == [1, 1]
    assert tables["fwd_idx"].tolist() == [[2], [2]]
    assert tables["fwd_w"].tolist() == [[0.0], [0.0]]
    # Empty neighborhoods: width-1 pads, zero counts.
    assert tables["lat_idx"].shape == (2, 1)
    assert tables["lat_n"].tolist() == [0, 0]
    assert tables["td_idx"].shape == (1, 1)
    assert tables["td_n"].tolist() == [0]
    assert tables["lat_on"] is False
    assert tables["td_on"] is False
    assert tables["activation"] is True


def test_event_scan_events_sort_stably_and_refuse():
    """The scan event list is file order within equal timestamps; off-plane
    pixels, bad polarities, fractional dt and out-of-grid times are refused."""
    from jaxfne.pcl_topology import event_scan_events

    kw = dict(dt_ms=0.5, n_steps=2, vw=4, vh=4)
    ev = event_scan_events(
        np.array([300, 100, 100]), np.array([2, 2, 2]),
        np.array([2, 2, 2]), np.array([1, 0, 1]), **kw)
    assert ev.shape == (3, 4) and ev.dtype == np.int64
    # Stable by time: the two t=100 events keep file order.
    assert ev[:, 0].tolist() == [100, 100, 300]
    assert ev[:, 3].tolist() == [0, 1, 1]
    bads = [
        # x outside the pixel plane (the scan indexes the cover table)
        (np.array([100]), np.array([4]), np.array([2]), np.array([0]),
         dict(dt_ms=0.5, n_steps=2, vw=4, vh=4)),
        (np.array([100]), np.array([2]), np.array([2]), np.array([2]),
         dict(dt_ms=0.5, n_steps=2, vw=4, vh=4)),
        (np.array([1000]), np.array([2]), np.array([2]), np.array([0]),
         dict(dt_ms=0.5, n_steps=2, vw=4, vh=4)),
        (np.array([100]), np.array([2]), np.array([2]), np.array([0]),
         dict(dt_ms=0.5005, n_steps=2, vw=4, vh=4)),
    ]
    for t, x, y, p, bkw in bads:
        try:
            event_scan_events(t, x, y, p, **bkw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"event {(t, x, y, p)} must raise")


def test_event_scan_tables_refusals_name_themselves():
    """Every scan-table declaration changes the tables or is refused."""
    from jaxfne.pcl_topology import event_scan_tables

    w_s = np.zeros((2, 2, 1, 1), dtype=np.float64)
    good = dict(
        nx=1, ny=1, nd=2, cx=1, cy=1, cd=1,
        origins_x=[2.0], origins_y=[2.0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0], c_size=(1, 1),
        c_weights=np.zeros((1, 1, 1, 2), dtype=np.float64),
        vw=4, vh=4, inhib_range=(0, 0),
    )
    event_scan_tables(**good)  # must not raise
    bads = [
        dict(good, c_cover={(0, 1): [(0, 0)]}),  # cover misses a position
        dict(good, c_cover={(0, 0): [(1, 0)]}),  # cover names no complex cell
        dict(good, c_weights=np.zeros((1, 1, 1, 3))),
        dict(good, weights=np.zeros((2, 2, 2, 1))),
        dict(good, inhib_range=(1, 0)),  # negative window
        dict(good, lateral_on=True,
             lateral_weights=np.zeros((3, 3))),  # (NS, NS) skew
        dict(good, topdown_on=True,
             topdown_weights=np.zeros((2, 1))),  # (NS, NC) skew
        dict(good, vw=0),
    ]
    for bkw in bads:
        try:
            event_scan_tables(**bkw)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("scan-table refusal must raise")


def _two_column_tables():
    """Two simple columns (nd 2) under one complex cell that spans both (csx 2).

    The complex cell sees ex = 0 and 1, and each forward slot reads its own
    entry of w_c. Returns ``(tables, w_c)``.
    """
    from jaxfne.pcl_topology import event_scan_tables

    w_s = np.zeros((2, 2, 1, 1), dtype=np.float64)
    w_s[1, 0, 0, 0] = 12.0
    w_s[0, 1, 0, 0] = 12.0
    w_c = np.arange(1.0, 9.0).reshape(2, 2, 1, 2)
    tables = event_scan_tables(
        nx=2, ny=1, nd=2, cx=1, cy=1, cd=2,
        origins_x=[0, 1], origins_y=[0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)], (1, 0): [(0, 0)]},
        c_ox=[0], c_oy=[0], c_size=(2, 1), c_weights=w_c,
        vw=2, vh=1, inhib_range=(0, 0),
    )
    return tables, w_c


def _oracle_forward_slots(*, nx, ny, nd, cy, cd, c_cover, c_ox, c_oy, ns, w_c):
    """Forward targets per simple cell, transcribed from ``pcl_event_ref.py:708-720``.

    The oracle's ``_forward`` visits complex cells in ascending index and
    excites each with ``w_c[cz, ex, ey, k]`` at the source's depth ``k``. The
    core numbers complex cells after the ``ns`` simple ones. Returns
    ``{n: [(target, (cz, ex, ey), weight), ...]}``.
    """
    slots = {}
    for i in range(nx):
        for j in range(ny):
            for k in range(nd):
                n = (i * ny + j) * nd + k
                row = []
                for cx, cj in c_cover[(i, j)]:
                    ex, ey = i - c_ox[cx], j - c_oy[cj]
                    base = (cx * cy + cj) * cd
                    for cz in range(cd):
                        row.append((ns + base + cz, (cz, ex, ey),
                                    float(w_c[cz, ex, ey, k])))
                slots[n] = row
    return slots


def _assert_fwd_key_is_oracle_forward_map(tables, **geometry):
    ns = int(tables["ns"])
    want = _oracle_forward_slots(ns=ns, **geometry)
    checked = 0
    for n in range(ns):
        rows = want[n]
        assert int(tables["fwd_n"][n]) == len(rows), f"fwd_n[{n}]"
        for f, (target, key, weight) in enumerate(rows):
            assert int(tables["fwd_idx"][n, f]) == target, (n, f)
            assert tuple(int(v) for v in tables["fwd_key"][n, f]) == key, (n, f)
            assert tables["fwd_w"][n, f] == weight, (n, f)
            checked += 1
    return checked


def test_event_scan_fwd_key_matches_oracle_forward_map():
    """Fail-first (PCL-L3a): fwd_key names the shared weight each forward slot reads.

    Checked on the micro network (its complex cell sees ex = ey = 0 only, so
    this pins the cover and the keys) and on the two-column geometry, whose
    complex cell sees ex = 0 and 1 with distinct w_c entries.
    """
    checked = _assert_fwd_key_is_oracle_forward_map(
        _scan_micro_tables(), nx=1, ny=1, nd=2, cy=1, cd=1,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0],
        w_c=np.zeros((1, 1, 1, 2), dtype=np.float64))
    wide, w_c = _two_column_tables()
    checked += _assert_fwd_key_is_oracle_forward_map(
        wide, nx=2, ny=1, nd=2, cy=1, cd=2,
        c_cover={(0, 0): [(0, 0)], (1, 0): [(0, 0)]},
        c_ox=[0], c_oy=[0], w_c=w_c)
    assert checked > 0, "fixture must hold forward slots"
    assert {int(v) for v in wide["fwd_key"][:, :, 1].ravel()} == {0, 1}, (
        "two-column fixture must exercise ex = 0 and 1")


def test_event_scan_refuses_fwd_w_that_disagrees_with_w_c():
    """A forward-weight table that disagrees with the shared w_c is refused, naming fwd_w."""
    import jax.numpy as jnp

    from jaxfne.emitters_lif import make_lif_params, simulate_lif_edge
    from jaxfne.pcl_topology import event_scan_events, local_inhibition_blocks

    tables, _ = _two_column_tables()
    events = event_scan_events(
        np.array([100]), np.array([0]), np.array([0]), np.array([0]),
        dt_ms=0.5, n_steps=1, vw=2, vh=1)
    p = make_lif_params(("simple",) * 4 + ("complex",) * 2)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    empty = np.zeros(0, dtype=np.int64)
    blocks = local_inhibition_blocks(
        groups, li_pre=empty, li_post=empty, li_weight=np.zeros(0))
    kw = dict(
        n_steps=1, dt_ms=0.5,
        local_inhibition_groups=jnp.asarray(groups),
        local_inhibition_resolution="event_scan",
        local_inhibition_blocks=blocks,
        event_scan_events=events,
    )
    simulate_lif_edge(p, event_scan_tables=tables, **kw)  # the declared tables run
    bad = dict(tables)
    bad["fwd_w"] = np.array(tables["fwd_w"], dtype=np.float64, copy=True)
    bad["fwd_w"][0, 1] += 1.0
    with pytest.raises(ValueError, match="fwd_w"):
        simulate_lif_edge(p, event_scan_tables=bad, **kw)


def test_validate_local_inhibition_event_scan_has_no_rounds():
    """Fail-first for Step 9: the scan replay resolves every event, no cap.

    ``rounds`` on an ``event_scan`` spec is refused (it would change
    nothing); a rounds-free ascending spec validates.
    """
    from jaxfne.pcl_topology import validate_local_inhibition

    spec = validate_local_inhibition(
        {"resolution": "event_scan"}, label="InterConnection")
    assert spec == {"order": "ascending", "resolution": "event_scan"}
    with pytest.raises(ValueError, match="rounds"):
        validate_local_inhibition(
            {"rounds": 4, "resolution": "event_scan"},
            label="InterConnection")
    with pytest.raises(ValueError, match="order"):
        validate_local_inhibition(
            {"order": "descending", "resolution": "event_scan"},
            label="InterConnection")
    # Other resolutions still require rounds (existing contract).
    with pytest.raises(ValueError, match="rounds"):
        validate_local_inhibition(
            {"resolution": "event_order"}, label="InterConnection")


def test_scan_inhibition_triplets_match_hand_meshgrid():
    """Fail-first for Step 9: LI triplets transcribe the per-position depths.

    Every ordered depth pair inside one retinotopic position carries the
    local-inhibition magnitude; nothing crosses positions.
    """
    from jaxfne.pcl_topology import scan_inhibition_triplets

    wi_s = np.array([[0.0, 0.5], [1.5, 0.0]])
    wi_c = np.array([[0.0]])
    pre, post, mag = scan_inhibition_triplets(
        1, 1, 2, 1, 1, 1, 2, wi_s, wi_c)
    assert pre.tolist() == [0, 1]
    assert post.tolist() == [1, 0]
    assert mag.tolist() == [1.5, 0.5]
    with pytest.raises(ValueError, match="scan_inhibition"):
        scan_inhibition_triplets(
            1, 1, 2, 1, 1, 1, 3, wi_s, wi_c)
