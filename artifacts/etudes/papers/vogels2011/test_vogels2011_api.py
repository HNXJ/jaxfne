"""Tests for the vogels2011 lane API (P2: PORT_SPEC rows, G-IE/G-SC settled).

Passing tests pin exact port behaviour; gap tests fail on the baseline
implementation and carry ``xfail(strict=True)`` so a core change that closes
a gap turns them green loudly.
"""

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jaxfne as jtfne
assert "site-packages" not in jtfne.__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)
from artifacts.etudes.papers.vogels2011 import vogels2011_api as L

N_SMALL = 20
DT = 0.1


def _adex_model(n=N_SMALL, seed=0):
    from jaxfne import neuronal_tensor as nt

    return jtfne.construct(
        L.build_spec(n=n, seed=seed),
        nt.RuntimeConfiguration(seed=seed, duration_ms=50.0, dt_ms=DT,
                                emitter="adex"),
    )


def _edges(model):
    st = model.static["adex"]
    rows = model.neuron_table()
    lab = np.array([r["cell_type"] for r in rows])
    names = list(st["receptor_names"])
    return np.asarray(st["pre"]), np.asarray(st["post"]), \
        np.asarray(st["weight_nS"], dtype=float), lab, names


def _delay_steps(model):
    from jaxfne._construct_adex import adex_kernel_args

    return np.asarray(adex_kernel_args(model, DT)["edge_delay_steps"])


def test_build_spec_smoke():
    """The lane spec builds (fixture must build)."""
    spec = L.build_spec(n=20, seed=0)
    assert spec is not None, "fixture must build the lane spec"


def test_realized_cell_params():
    """Realized AdEx cells carry the paper leak/threshold/reset values (PORT_SPEC 1-3)."""
    from jaxfne._construct_adex import adex_kernel_args

    kw = adex_kernel_args(_adex_model(), DT)
    p = kw["params"]

    def uniq(field):
        return sorted(set(np.round(np.asarray(
            getattr(p, field)).astype(float), 6).tolist()))

    assert uniq("C_pF") == [200.0], "membrane capacitance must be 200 pF"
    assert uniq("gL_nS") == [10.0], "leak must be 10 nS"
    assert uniq("EL_mV") == [-60.0], "leak reversal must be -60 mV"
    assert uniq("VT_mV") == [-50.0], "threshold must be -50 mV"
    assert uniq("Vreset_mV") == [-60.0], "reset must return to -60 mV"
    assert uniq("Tref_ms") == [5.0], "refractory must be 5 ms"
    assert uniq("a_nS") == [0.0], "adaptation coupling must be zeroed"
    assert uniq("b_pA") == [0.0], "spike-triggered adaptation must be zeroed"
    assert uniq("Delta_mV") == [0.5, 2.0], \
        "sharpness stays at Table-1 defaults (D-CELL)"


def test_realized_weights():
    """Realized edges carry +0.3 (E) and -3.0 (I) nS (PORT_SPEC 7, 9)."""
    pre, _, weight, lab, _ = _edges(_adex_model())
    assert len(weight) > 0, "fixture must wire edges"
    w_e = weight[lab[pre] == "E"]
    w_i = weight[lab[pre] == "PV"]
    assert len(w_e) > 0 and len(w_i) > 0, "both sources must wire edges"
    assert np.allclose(w_e, 0.3, rtol=1e-6), "E edges must realize +0.3 nS"
    assert np.allclose(w_i, -3.0, rtol=1e-6), "I edges must realize -3.0 nS"


def test_delay_one_step():
    """Every edge realizes delay_ms=0.1 as exactly 1 step (PORT_SPEC 12)."""
    delay = _delay_steps(_adex_model())
    assert len(delay) > 0, "fixture must wire edges"
    assert set(delay.tolist()) == {1}, "delay_ms=0.1 at dt=0.1 is 1 step"


def test_drive_current():
    """Every cell realizes the 200 pA background current (PORT_SPEC 13)."""
    from jaxfne._construct_adex import adex_kernel_args

    kw = adex_kernel_args(_adex_model(), DT)
    got = sorted(set(np.asarray(kw["I_ext_pA"]).astype(float).tolist()))
    assert got == [200.0], "background current must be 200 pA on every cell"


def test_receptor_tau_reversal():
    """Receptors carry tau (5, 10) ms and reversals (0, -80) mV (PORT_SPEC 6)."""
    st = _adex_model().static["adex"]
    by_name = {}
    for name, tau, rev in zip(st["receptor_names"], st["receptor_tau_ms"],
                              st["receptor_reversal_mV"]):
        by_name[str(name).split("__")[0]] = (float(tau), float(rev))
    assert by_name["ampa"] == (5.0, 0.0), "AMPA tau/reversal"
    assert by_name["gaba"] == (10.0, -80.0), "GABA tau/reversal"


def test_density_exact_totals():
    """Density sampling realizes exactly round(0.02*n_pre*n_post) per rule."""
    pre, post, _, lab, _ = _edges(_adex_model())
    ne = int((lab == "E").sum())
    ni = int((lab == "PV").sum())
    assert (ne, ni) == (16, 4), "fractions must split 16/4 at N=20"
    for src, npre in (("E", ne), ("PV", ni)):
        for tgt in ("E", "PV"):
            npost = ne if tgt == "E" else ni
            got = int(((lab[pre] == src) & (lab[post] == tgt)).sum())
            assert got == round(0.02 * npre * npost), \
                "rule %s->%s must realize its exact density total" % (src, tgt)


def test_short_sim_spikes_finite():
    """A 50 ms N=20 run spikes with finite voltages (PORT_SPEC 13 probe)."""
    model = _adex_model()
    sim = jtfne.Simulation(duration_ms=50.0, dt_ms=DT, seed=0)
    signals = model.simulate(sim)
    spikes = np.asarray(signals.spikes, dtype=bool)
    assert spikes.shape[0] == 500, "raster must cover every 0.1 ms step"
    assert int(spikes.sum()) > 0, "driven network must spike"
    assert bool(np.isfinite(np.asarray(signals.V_m)).all()), \
        "V must stay finite"


def test_score_keys():
    """The lane scorer returns the analysis estimands (checklist E3)."""
    regime = L.score(np.zeros((100, 10), dtype=bool), DT)
    assert set(regime) == {"rate_hz", "cv_isi", "kappa", "active_fraction",
                           "spectral_peak_excess", "label"}, "scorer keys"
    assert regime["label"] == "silent", "empty raster must read silent"


@pytest.mark.xfail(strict=True)
def test_istdp_ie_gap():
    """Inhibitory presynaptic spikes must move I-to-E weights (gap G-IE)."""
    import jax.numpy as jnp

    from jaxfne.plasticity import stdp_weight_update

    n = 4
    W = jnp.full((n, n), 3.0)
    exc_mask = jnp.asarray([True, True, False, False])
    trace_pre = jnp.ones(n)
    trace_post = jnp.ones(n)
    spiked = jnp.asarray([False, False, True, False])
    out = np.asarray(stdp_weight_update(
        W, trace_pre, trace_post, spiked, exc_mask,
        A_plus=0.01, A_minus=0.01, plasticity_scale=1.0,
        w_min=0.0, w_max=100.0))
    assert float(out[0, 2]) != 3.0, \
        "inhibitory-presynaptic weight must update under iSTDP"


@pytest.mark.xfail(strict=True)
def test_per_channel_rates_gap():
    """Per-channel Poisson rates compile to distinct sources (gap G-SC)."""
    from jaxfne.inputs import compile_poisson_sources

    entries = compile_poisson_sources({
        "n_sources": 8,
        "rate_hz": [50.0, 100.0],
        "weight_nS": 0.3,
        "mechanism": "ampa",
        "target_cell_types": ["E"],
        "delay_ms": 0.1,
        "seed": 0,
    })
    assert [e.value for e in entries] == [50.0, 100.0], \
        "one source per channel rate"
