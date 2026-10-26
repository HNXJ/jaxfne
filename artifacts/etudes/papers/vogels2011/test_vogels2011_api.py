"""Tests for the vogels2011 lane API (P2: PORT_SPEC rows, G-IE closed).

Passing tests pin exact port behaviour; the remaining gap test fails on the
baseline implementation and carries ``xfail(strict=True)`` so a core change
that closes the gap turns it green loudly.
"""

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

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
    """Realized AdEx cells carry the paper leak/threshold/reset values (PORT_SPEC 1-4)."""
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
    assert uniq("Delta_mV") == [0.0], \
        "sharpness must be the LIF limit (exact LIF, PORT_SPEC 4)"


def test_realized_weights():
    """Realized edges carry +0.3 (E), -1e-10 (I-to-E init), -3.0 (I-to-I)."""
    # N=100: every rule wires (at N=20 the I-to-I rule rounds to 0 edges).
    pre, post, weight, lab, _ = _edges(_adex_model(n=100, seed=0))
    assert len(weight) > 0, "fixture must wire edges"
    w_e = weight[lab[pre] == "E"]
    w_ie = weight[(lab[pre] == "PV") & (lab[post] == "E")]
    w_ii = weight[(lab[pre] == "PV") & (lab[post] == "PV")]
    assert len(w_e) > 0 and len(w_ie) > 0 and len(w_ii) > 0, \
        "E, I-to-E and I-to-I rules must all wire edges"
    assert np.allclose(w_e, 0.3, rtol=1e-6), "E edges must realize +0.3 nS"
    assert np.allclose(w_ie, -1e-10, rtol=1e-6), \
        "I-to-E edges must realize the -1e-10 nS paper init"
    assert np.allclose(w_ii, -3.0, rtol=1e-6), "I-to-I edges must realize -3.0 nS"


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


def test_istdp_ie():
    """The registered iSTDP rule moves lane I-to-E weights, freezing the rest."""
    from jaxfne._construct_adex import adex_kernel_args
    from jaxfne.emitters_adex import simulate_adex_hdp_registered
    from jaxfne.istdp_vogels2011 import (
        ISTDP_RULE_NAME,
        W_MAX_NS,
        W_MIN_NS,
        istdp_plasticity_mask,
    )

    model = _adex_model()
    kw = adex_kernel_args(model, DT)
    rows = model.neuron_table()
    lab = np.array([r["cell_type"] for r in rows])
    pre = np.asarray(kw["edge_pre"])
    post = np.asarray(kw["edge_post"])
    mask = np.asarray(istdp_plasticity_mask(lab[pre] == "PV", lab[post] == "E"))
    w_init = np.abs(np.asarray(kw["edge_weight_nS"], dtype=float))
    out = simulate_adex_hdp_registered(
        **kw,
        n_steps=500,
        dt_ms=DT,
        hdp_rule=ISTDP_RULE_NAME,
        plasticity_mask=mask,
        seed=0,
    )
    spikes = np.asarray(out["spikes"], dtype=bool)
    assert int(spikes.sum()) > 0, "fixture must spike"
    w_final = np.asarray(out["w_final_nS"], dtype=float)
    assert bool(np.all(w_final[~mask] == w_init[~mask])), \
        "frozen E and I-to-I edges must stay bit-identical"
    assert bool(np.any(w_final[mask] != w_init[mask])), \
        "plastic I-to-E weights must move under the registered rule"
    assert bool(np.all((w_final >= W_MIN_NS) & (w_final <= W_MAX_NS))), \
        "plastic magnitudes must stay within the (0, 100) clip"


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
