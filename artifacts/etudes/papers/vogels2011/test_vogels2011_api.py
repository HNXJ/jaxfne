"""Tests for the vogels2011 lane API (P2: PORT_SPEC rows, G-IE closed).

Passing tests pin exact port behaviour; the remaining gap test fails on the
baseline implementation and carries ``xfail(strict=True)`` so a core change
that closes the gap turns it green loudly.
"""

import os
import sys
from pathlib import Path

import pytest

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


def test_plastic_rho0_changes_realized_weights(tmp_path):
    """rho0_hz=6 Hz (doubled alpha) moves the realized weights (P3 v2 B8)."""
    base = L.run_plastic_segment(0, "base", tmp_path / "base", n=200, T_s=0.3)
    doubled = L.run_plastic_segment(0, "dbl", tmp_path / "dbl", n=200,
                                    T_s=0.3, rho0_hz=6.0)
    assert doubled["rho0_hz"] == 6.0, "lesion run must record rho0 6 Hz"
    assert int(np.asarray(doubled["spikes"], dtype=bool).sum()) > 0, \
        "lesion fixture must spike"
    assert base["w_ie_final_mean_nS"] != doubled["w_ie_final_mean_nS"], \
        "doubled rho0 must change the realized I-to-E weights"


def test_plastic_eta_zero_freezes_weights(tmp_path):
    """eta=0 holds I-to-E at init while the default rule moves them (B8)."""
    frozen = L.run_plastic_segment(0, "eta0", tmp_path / "eta0", n=200,
                                   T_s=0.3, eta=0.0)
    assert frozen["eta"] == 0.0, "lesion run must record eta 0"
    assert frozen["w_ie_final_mean_nS"] == pytest.approx(1e-10), \
        "eta=0 must hold I-to-E at the 1e-10 nS paper init"
    moved = L.run_plastic_segment(0, "dflt", tmp_path / "dflt", n=200,
                                  T_s=0.3)
    assert moved["w_ie_final_mean_nS"] != pytest.approx(1e-10), \
        "default eta must move I-to-E off init in 0.3 s"


def test_plastic_defaults_unchanged(tmp_path):
    """Explicit prereg defaults are bit-identical to the unparameterized call."""
    a = L.run_plastic_segment(0, "imp", tmp_path / "imp", n=100, T_s=0.2)
    b = L.run_plastic_segment(0, "exp", tmp_path / "exp", n=100, T_s=0.2,
                              rho0_hz=3.0, eta=1e-2)
    assert (np.asarray(a["spikes"]) == np.asarray(b["spikes"])).all(), \
        "explicit defaults must reproduce the default raster bit-exactly"
    assert a["w_ie_final_mean_nS"] == b["w_ie_final_mean_nS"], \
        "explicit defaults must reproduce the default weights exactly"


def test_plastic_segment_skips_the_weight_trace(tmp_path, monkeypatch):
    """The plastic leg reads w_final only; the per-step |w| trace (n_steps x n_edges) is off, so the
    n = 10000, 9 s pilot fits in memory (the traced run reached 80 GB and was OOM-killed, 2026-10-31)."""
    import jaxfne.emitters_adex as E

    real = E.simulate_adex_hdp_registered
    seen = {}

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(E, "simulate_adex_hdp_registered", spy)
    out = L.run_plastic_segment(0, "notrace", tmp_path / "nt", n=100, T_s=0.1)
    assert seen.get("record_w_trace") is False, seen.keys()
    assert np.asarray(out["spikes"]).shape[0] == 1000, "fixture must run the full segment"


def test_plastic_refuses_unrealizable(tmp_path):
    """Values the API cannot realize are refused, never stored unused."""
    with pytest.raises(ValueError):
        L.run_plastic_segment(0, "neg", tmp_path / "neg", n=20, T_s=0.1,
                              eta=-1.0)
    with pytest.raises(ValueError):
        L.run_plastic_segment(0, "negrho", tmp_path / "negrho", n=20,
                              T_s=0.1, rho0_hz=-1.0)


def test_pilot_receipt_unscored(tmp_path):
    """The pilot writes the prereg estimands, marked unscored (B8)."""
    out = tmp_path / "pilot"
    receipt = L.pilot(0, 20, 0.2, out)
    assert receipt["verdict"] == "unscored pilot", "pilots are never scored"
    assert (out / "pilot_receipt.json").exists(), "receipt must land on disk"
    assert set(receipt) >= {"R-E", "O-DROP", "W-IE"}, "prereg estimands"
    assert receipt["R-E"]["window_clamped"] is True, \
        "a 0.2 s leg cannot hold the 1 s R-E window"
    assert receipt["R-E"]["e"] >= 0.0, "relative error must be finite-sided"
    assert receipt["O-DROP"]["ratio"] >= 0.0, "drop ratio must be finite-sided"


def test_plastic_manifest_records_w_trace_off(tmp_path):
    """The plastic manifest entry records that the per-step |w| trace is off."""
    import json

    out = tmp_path / "traceflag"
    L.run_plastic_segment(0, "flag", out, n=20, T_s=0.1)
    manifest = json.loads((out / "manifest.json").read_text())
    entries = [e for e in manifest["runs"]
               if e["file"] == "spikes_seed0_flag.npz"]
    assert len(entries) == 1, "fixture must write exactly one run entry"
    assert entries[0]["plastic"]["record_w_trace"] is False, \
        "the entry must record record_w_trace false"


def test_segmented_plastic_matches_unsegmented(tmp_path):
    """Two 1 s segments chained by state reproduce the 2 s leg exactly."""
    seg = L.run_plastic_segment(0, "sg", tmp_path / "sg", n=40, T_s=2.0,
                                plastic_segment_s=1.0)
    one = L.run_plastic_segment(0, "one", tmp_path / "one", n=40, T_s=2.0)
    assert int(np.asarray(one["spikes"]).sum()) > 0, "fixture must spike"
    assert (np.asarray(seg["spikes"])
            == np.asarray(one["spikes"])[-seg["spikes"].shape[0]:]).all(), \
        "the segmented leg must reproduce the run tail bit-exactly"
    assert seg["e_rate_per_s_hz"] == one["e_rate_per_s_hz"], \
        "the per-second E rate course must not depend on segmentation"
    assert seg["w_ie_final_mean_nS"] == one["w_ie_final_mean_nS"], \
        "the final I-to-E mean must not depend on segmentation"


def test_segmented_pilot_matches_unsegmented_estimands(tmp_path):
    """A short pilot with segments equals the unsegmented pilot's estimands."""
    a = L.pilot(0, 40, 2.0, tmp_path / "u")
    b = L.pilot(0, 40, 2.0, tmp_path / "s", plastic_segment_s=1.0)
    assert b["R-E"]["z_hz"] == a["R-E"]["z_hz"], "R-E z must match"
    assert b["R-E"]["e"] == a["R-E"]["e"], "R-E e must match"
    assert b["O-DROP"]["ratio"] == a["O-DROP"]["ratio"], "O-DROP must match"
    assert b["W-IE"]["w_ie_final_mean_nS"] == a["W-IE"]["w_ie_final_mean_nS"], \
        "W-IE must match"
    assert b["e_rate_per_s_hz"] == a["e_rate_per_s_hz"], "rate course must match"
    u = np.load(tmp_path / "u" / "spikes_seed0_segB.npz")["spikes"]
    s = np.load(tmp_path / "s" / "spikes_seed0_segB.npz")["spikes"]
    assert (u[-s.shape[0]:] == s).all(), \
        "the segmented raster must equal the unsegmented tail"
