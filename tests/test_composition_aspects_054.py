"""0.5.4 item 1 (E1b) — ensemble structural aspects, one by one.

Cross-area edges (explicit, with realized delays), geometry (offset_x keeps
depth, separates x), probes/fields (run + finite), continuation (chunked ==
continuous on the ensemble, and equal to the members' solo trajectories),
HDP engagement, batch determinism. Batch draws one global stream per
replicate by design (statistics utility); member solo-equivalence is
claimed for simulate(), not for batch.
"""

import numpy as np

import jaxfne as jtfne

D_MS, DT_MS, SEED = 40.0, 0.5, 7
N1, N2 = 12, 8


def _col(name, n, seed, probes=("spikes", "V_m"), **runtime_kw):
    cfg = (
        jtfne.Configuration()
        .runtime(
            duration_ms=40, dt_ms=DT_MS, seed=seed, recurrent_backend="edge_list", **runtime_kw
        )
        .column(name, layers=["L2/3", "L4"], n=n)
        .cell_types({"E": 0.8, "PV": 0.2})
        .connectivity()
        .set_emitter(family="izhikevich")
        .probes(list(probes), n_contacts=8)
    )
    return jtfne.construct(cfg)


def _field_col(name, n, seed):
    cfg = (
        jtfne.Configuration()
        .runtime(duration_ms=40, dt_ms=DT_MS, seed=seed, recurrent_backend="edge_list")
        .column(name, layers=["L2/3", "L4"], n=n)
        .cell_types({"E": 0.8, "PV": 0.2})
        .connectivity()
        .set_emitter(family="izhikevich")
        .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    return jtfne.construct(cfg)


def _sim(model, seed):
    return jtfne.simulate(model, duration_ms=D_MS, dt_ms=DT_MS, seed=seed)


def test_cross_edges_explicit_with_realized_delay():
    m1, m2 = _col("V1", N1, 0), _col("V2", N2, 1)
    ens = jtfne.connect(
        m1,
        m2,
        namespace=("A", "B"),
        edges=[
            dict(
                source={"model": 0, "area": "V1", "cell_type": "E"},
                target={"model": 1, "area": "V2"},
                probability=0.5,
                weight=0.5,
                sign="excitatory",
                delay_ms=2.0,
            )
        ],
    )
    el = ens.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    cross = (pre < N1) & (post >= N1)
    assert int(cross.sum()) > 0
    assert int((~cross).sum()) > 0  # member-internal edges preserved
    assert sorted(set(np.asarray(el.delay_steps)[cross].tolist())) == [4]
    md = ens.cfg.metadata["ensemble"]
    assert md["cross_model_edges"] == int(cross.sum())
    assert md["dt_ms"] == DT_MS


def test_member_internal_delays_preserved():
    m1, m2 = _col("V1", N1, 0), _col("V2", N2, 1)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    el = ens.params["edge_list"]
    assert list(np.asarray(el.delay_steps)) == (
        list(np.asarray(m1.params["edge_list"].delay_steps))
        + list(np.asarray(m2.params["edge_list"].delay_steps))
    )


def test_geometry_offset_x_preserves_depth():
    m1, m2 = _col("V1", N1, 0), _col("V2", N2, 1)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    pos = np.asarray(ens.params["positions"])
    solo = np.concatenate(
        [np.asarray(m1.params["positions"]), np.asarray(m2.params["positions"])],
        axis=0,
    )
    assert np.array_equal(pos[:, 1:], solo[:, 1:])  # y/z untouched
    assert float(pos[:N1, 0].max()) < float(pos[N1:, 0].min())  # x disjoint


def test_field_probes_run_finite_on_ensemble():
    ens = jtfne.connect(_field_col("V1", N1, 0), _field_col("V2", N2, 1), namespace=("A", "B"))
    sig = _sim(ens, SEED)
    assert sig.field is not None
    assert bool(np.isfinite(np.asarray(sig.field.lfp_proxy)).all())


def test_continuation_matches_continuous_and_solo():
    m1, m2 = _col("V1", N1, 0), _col("V2", N2, 1)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    full = _sim(ens, SEED)
    seg1, st = jtfne.simulate(ens, duration_ms=20.0, dt_ms=DT_MS, seed=SEED, return_state=True)
    seg2 = jtfne.simulate(ens, duration_ms=20.0, dt_ms=DT_MS, seed=SEED, continuation=st)
    cat = np.concatenate([np.asarray(seg1.V_m), np.asarray(seg2.V_m)], axis=0)
    assert np.array_equal(np.asarray(full.V_m), cat)
    solo0 = _sim(m1, jtfne.ensemble_member_seed(SEED, 0, 2))
    assert np.array_equal(np.asarray(full.V_m)[:, :N1], np.asarray(solo0.V_m))


def test_hdp_engages_on_ensemble():
    ens = jtfne.connect(
        _col("V1", N1, 0, enable_hdp=True), _col("V2", N2, 1, enable_hdp=True), namespace=("A", "B")
    )
    sig = _sim(ens, SEED)
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())
    assert sig.metadata.get("hdp", {}).get("enabled") is True


def test_batch_deterministic_on_ensemble():
    m1, m2 = _col("V1", N1, 0), _col("V2", N2, 1)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    kw = dict(duration_ms=D_MS, dt_ms=DT_MS, seed=SEED)
    b1 = ens.simulate_batch(jtfne.Simulation(**kw), n_seeds=2)
    b2 = ens.simulate_batch(jtfne.Simulation(**kw), n_seeds=2)
    assert np.array_equal(np.asarray(b1["V_m"]), np.asarray(b2["V_m"]))
