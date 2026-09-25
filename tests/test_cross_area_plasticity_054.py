"""0.5.4 item 4 — cross-area plasticity owned by the projection.

`ensemble_edge_ownership` maps merged edge indices to member ranges and
per-rule cross ranges (W_12, W_21 separately). A `plasticity_mask` built
from those ranges scopes HDP to the cross-area projection (or freezes
it) with the 0.5.3 disable/replay semantics: masked edges are carried
exactly, replay is bit-identical.
"""

import numpy as np
import pytest
from dataclasses import replace

import jaxfne as jtfne

DT_MS = 0.5
GAINS = {
    "K_HDP": 0.2,
    "K_ctrl": 0.3,
    "K_w_ctrl": 0.002,
    "alpha": 0.05,
    "tau_0_ms": 5.0,
    "noise_scale": 0.0,
}


def _col(name, n, seed):
    cfg = (
        jtfne.Configuration()
        .runtime(duration_ms=40, dt_ms=DT_MS, seed=seed, recurrent_backend="edge_list")
        .column(name, layers=["L2/3", "L4"], n=n)
        .cell_types({"E": 0.8, "PV": 0.2})
        .connectivity()
        .set_emitter(family="izhikevich")
        .probes(["spikes", "V_m"], n_contacts=8)
    )
    return jtfne.construct(cfg)


def _ens():
    m1, m2 = _col("V1", 12, 0), _col("V2", 8, 1)
    return jtfne.connect(
        m1,
        m2,
        namespace=("A", "B"),
        edges=[
            dict(
                source={"model": 0, "area": "V1", "cell_type": "E"},
                target={"model": 1, "area": "V2"},
                probability=0.3,
                weight=0.5,
                sign="excitatory",
            ),
            dict(
                source={"model": 1, "area": "V2", "cell_type": "E"},
                target={"model": 0, "area": "V1"},
                probability=0.3,
                weight=0.5,
                sign="excitatory",
            ),
        ],
    )


def _run_masked(ens, mask):
    hp = dict(GAINS, plasticity_mask=np.asarray(mask, dtype=np.float32))
    model = replace(ens, cfg=ens.cfg.runtime(enable_hdp=True, hdp_params=hp))
    sig = jtfne.simulate(model, duration_ms=40.0, dt_ms=DT_MS, seed=7)
    return model, sig, np.asarray(model.last_hdp_diagnostics()["w_final"])


def test_ownership_ranges_and_fail_closed():
    ens = _ens()
    own = jtfne.ensemble_edge_ownership(ens)
    n = int(ens.params["edge_list"].n_edges)
    lo0, hi0 = own["member_ranges"][0]
    assert (lo0, hi0) == (0, hi0)
    (lo1, hi1), (c0lo, c0hi), (c1lo, c1hi) = (
        own["member_ranges"][1],
        own["cross_ranges"][0],
        own["cross_ranges"][1],
    )
    assert hi0 == lo1 and hi1 == c0lo and c0hi == c1lo and c1hi == n
    assert c0hi > c0lo and c1hi > c1lo  # both directions wired
    with pytest.raises(ValueError):
        jtfne.ensemble_edge_ownership(_col("V1", 12, 0))


def test_cross_only_mask_freezes_members():
    ens = _ens()
    own = jtfne.ensemble_edge_ownership(ens)
    n = int(ens.params["edge_list"].n_edges)
    w0 = np.asarray(ens.params["edge_list"].weight)
    mask = np.zeros(n)
    for lo, hi in own["cross_ranges"]:
        mask[lo:hi] = 1.0
    _, sig, wf = _run_masked(ens, mask)
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())
    mid = np.zeros(n, dtype=bool)
    for lo, hi in own["member_ranges"]:
        mid[lo:hi] = True
    assert bool(np.array_equal(wf[mid], w0[mid]))  # frozen exactly
    assert bool((np.abs(wf[~mid] - w0[~mid]) > 0).any())  # W_12/W_21 moved


def test_member_only_mask_freezes_cross():
    ens = _ens()
    own = jtfne.ensemble_edge_ownership(ens)
    n = int(ens.params["edge_list"].n_edges)
    w0 = np.asarray(ens.params["edge_list"].weight)
    mask = np.zeros(n)
    for lo, hi in own["member_ranges"]:
        mask[lo:hi] = 1.0
    _, _, wf = _run_masked(ens, mask)
    x = np.zeros(n, dtype=bool)
    for lo, hi in own["cross_ranges"]:
        x[lo:hi] = True
    assert bool(np.array_equal(wf[x], w0[x]))  # cross frozen exactly
    assert bool((np.abs(wf[~x] - w0[~x]) > 0).any())  # members moved


def test_zero_mask_disables_and_replay_is_exact():
    ens = _ens()
    n = int(ens.params["edge_list"].n_edges)
    w0 = np.asarray(ens.params["edge_list"].weight)
    _, sig1, wf1 = _run_masked(ens, np.zeros(n))
    assert bool(np.array_equal(wf1, w0))
    _, sig2, wf2 = _run_masked(ens, np.zeros(n))
    assert bool(np.array_equal(np.asarray(sig1.V_m), np.asarray(sig2.V_m)))
    assert bool(np.array_equal(wf1, wf2))
