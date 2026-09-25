"""0.5.4 item 1 — ensemble member RNG domains and solo-equivalence.

``connect(A, B)`` + ``simulate(seed=S)`` must reproduce each member run
alone: member ``m``'s slice of the ensemble trajectory is bit-identical to
a solo run of that member with ``seed=ensemble_member_seed(S, m, n)`` when
cross-model edges are zero, with no paradigm and no poisson_drive (both
draw global streams by design and are outside the identity contract).
"""

import numpy as np
import pytest

import jaxfne as jtfne

D_MS, DT_MS, SEED = 40.0, 0.5, 7


def _col(name, n, seed, **runtime_kw):
    cfg = (
        jtfne.Configuration()
        .runtime(
            duration_ms=40, dt_ms=DT_MS, seed=seed, recurrent_backend="edge_list", **runtime_kw
        )
        .column(name, layers=["L2/3", "L4"], n=n)
        .cell_types({"E": 0.8, "PV": 0.2})
        .connectivity()
        .set_emitter(family="izhikevich")
        .probes(["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _sim(model, seed):
    return jtfne.simulate(model, duration_ms=D_MS, dt_ms=DT_MS, seed=seed)


def test_ensemble_member_seed_contract():
    s0 = jtfne.ensemble_member_seed(SEED, 0, 2)
    s1 = jtfne.ensemble_member_seed(SEED, 1, 2)
    assert s0 == jtfne.ensemble_member_seed(SEED, 0, 2)
    assert s0 != s1
    assert all(isinstance(s, int) and s >= 0 for s in (s0, s1))
    with pytest.raises(ValueError):
        jtfne.ensemble_member_seed(SEED, 2, 2)
    with pytest.raises(ValueError):
        jtfne.ensemble_member_seed(SEED, 0, 1)


def test_ensemble_slice_matches_solo_eager():
    m1, m2 = _col("V1", 12, 0), _col("V2", 8, 1)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    assert ens.cfg.metadata["ensemble"]["model_sizes"] == [12, 8]
    sig = _sim(ens, SEED)
    n1 = 12
    solo0 = _sim(m1, jtfne.ensemble_member_seed(SEED, 0, 2))
    solo1 = _sim(m2, jtfne.ensemble_member_seed(SEED, 1, 2))
    assert bool(np.array_equal(np.asarray(sig.V_m)[:, :n1], np.asarray(solo0.V_m)))
    assert bool(np.array_equal(np.asarray(sig.V_m)[:, n1:], np.asarray(solo1.V_m)))
    assert bool(np.array_equal(np.asarray(sig.spikes)[:, :n1], np.asarray(solo0.spikes)))
    assert bool(np.array_equal(np.asarray(sig.spikes)[:, n1:], np.asarray(solo1.spikes)))


def test_ensemble_slice_matches_solo_jit():
    m1, m2 = _col("V1", 12, 0, jit=True), _col("V2", 8, 1, jit=True)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    sig = _sim(ens, SEED)
    n1 = 12
    solo0 = _sim(m1, jtfne.ensemble_member_seed(SEED, 0, 2))
    solo1 = _sim(m2, jtfne.ensemble_member_seed(SEED, 1, 2))
    assert bool(np.array_equal(np.asarray(sig.V_m)[:, :n1], np.asarray(solo0.V_m)))
    assert bool(np.array_equal(np.asarray(sig.V_m)[:, n1:], np.asarray(solo1.V_m)))
    assert bool(np.array_equal(np.asarray(sig.spikes)[:, :n1], np.asarray(solo0.spikes)))
    assert bool(np.array_equal(np.asarray(sig.spikes)[:, n1:], np.asarray(solo1.spikes)))
