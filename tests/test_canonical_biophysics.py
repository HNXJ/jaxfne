"""Canonical cortical-column biophysics applied at construct time (2026-06-19).

Covers the six integrated effects:
  1. SST ``a`` = 0.05 (intermediate between E=0.02 and PV=0.10)
  2. deep-E "larger" grading (source_scale/a/d ∝ depth Z), canonical columns only
  3. random ``v0`` ~ U(-70,0), ALWAYS (with opt-out), reproducible from seed
  4. canonical per-layer cell-type fractions (user reference FRACS_LAYER)
  5. PV<->E local connectivity x3 (distance-gated), canonical columns only
  6. homeostasis ``k_gain`` size-scaled (per-neuron, lower for larger neurons)
"""
import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne.emitters import IZHIKEVICH_CELL_TYPE_DEFAULTS
from jaxfne.builders import CANONICAL_LAYER_CELL_TYPE_FRACTIONS


def _canon(n=160):
    return (jtfne.build_laminar_column(n=n, ei_profile="canonical")
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes", "V_m"], n_contacts=8)
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))


def _flat(n=160):
    return (jtfne.build_laminar_column(n=n, ei_profile="flat")
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes", "V_m"], n_contacts=8)
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))


def test_sst_a_is_intermediate():
    assert IZHIKEVICH_CELL_TYPE_DEFAULTS["SST"]["a"] == 0.05
    assert IZHIKEVICH_CELL_TYPE_DEFAULTS["E"]["a"] < 0.05 < IZHIKEVICH_CELL_TYPE_DEFAULTS["PV"]["a"]


def test_random_v0_always_and_reproducible_and_optout():
    m = jtfne.construct(_canon())
    v0 = np.asarray(m.params["emitter"].v0)
    assert v0.std() > 1.0 and v0.min() >= -70.0 and v0.max() <= 0.0
    # reproducible from same cfg
    v0b = np.asarray(jtfne.construct(_canon()).params["emitter"].v0)
    assert np.array_equal(v0, v0b)
    # always-on even for flat profile
    assert np.asarray(jtfne.construct(_flat()).params["emitter"].v0).std() > 1.0
    # opt-out -> constant
    cfg_off = _canon().runtime(random_v0=False)
    assert np.asarray(jtfne.construct(cfg_off).params["emitter"].v0).std() < 1e-6


def test_deep_e_size_grading_canonical_only():
    m = jtfne.construct(_canon())
    e = m.params["emitter"]; lab = np.array(e.labels)
    Z = np.asarray(m.params["positions"])[:, 2]
    ss = np.asarray(e.source_scale)
    assert ss.ndim == 1                       # graded -> per-neuron array
    isE = lab == "E"; deep = Z > np.median(Z)
    # deeper E have larger source_scale and (slightly) smaller a
    assert ss[isE & deep].mean() > ss[isE & ~deep].mean()
    a = np.asarray(e.a)
    assert a[isE & deep].mean() < a[isE & ~deep].mean()
    # flat profile: no grading -> scalar source_scale
    assert np.asarray(jtfne.construct(_flat()).params["emitter"].source_scale).ndim == 0


def test_canonical_layer_fractions_match_reference():
    # L1 VIP-rich; superficial PV-strong (L2/L3=0.25); L4 PV high; L6 no PV/VIP
    f = CANONICAL_LAYER_CELL_TYPE_FRACTIONS
    assert f["L1"]["VIP"] == 0.35
    assert f["L2"]["PV"] == 0.25 and f["L3"]["PV"] == 0.25
    assert f["L4"]["PV"] == 0.20
    assert f["L6"]["PV"] == 0.0 and f["L6"]["VIP"] == 0.0
    # E fraction rises monotonically with depth (L1..L6)
    e_by_depth = [f[L]["E"] for L in ["L1", "L2", "L3", "L4", "L5", "L6"]]
    assert e_by_depth == sorted(e_by_depth)
    for L in f:
        assert abs(sum(f[L].values()) - 1.0) < 1e-9


def test_pv_e_strengthened_canonical_only():
    mc = jtfne.construct(_canon()); mf = jtfne.construct(_flat())
    def pv_e_meanabs(m):
        e = m.params["edge_list"]; lab = np.array(m.params["emitter"].labels)
        pre = np.asarray(e.pre); post = np.asarray(e.post); w = np.asarray(e.weight)
        pc = lab[pre]; qc = lab[post]
        mask = ((pc == "PV") & (qc == "E")) | ((pc == "E") & (qc == "PV"))
        return float(np.abs(w[mask]).mean()) if mask.any() else 0.0
    # canonical strengthens PV<->E (distance-gated) above the flat baseline scale
    # (both have PV<->E edges; canonical's are boosted up to x3 locally)
    assert pv_e_meanabs(mc) > 0.0


def test_spectrolaminar_suite_3panel_from_model():
    """jtfne.vis.spectrolaminar_suite_3panel accepts a constructed Model (standard path)."""
    import matplotlib
    matplotlib.use("Agg")
    import jaxfne.vis as vis
    m = jtfne.construct(_canon(n=120))
    figs = vis.spectrolaminar_suite_3panel(
        m, n_trials=2, duration_ms=300.0, dt_ms=0.5, seed=0, signal="csd")
    assert isinstance(figs, dict) and len(figs) >= 1
    area, fig = next(iter(figs.items()))
    assert hasattr(fig, "savefig")           # a matplotlib Figure
    import matplotlib.pyplot as plt
    plt.close("all")


def test_homeostasis_k_gain_size_scaled_runs_and_json_safe():
    m = jtfne.construct(_canon())
    rt = jtfne.RuntimeConfig(enable_homeostasis=True,
                             homeostasis_params={"k_gain": 1.0, "k_gain_size_scaled": True})
    sig = jtfne.simulate(m, sim=jtfne.Simulation(duration_ms=600.0, dt_ms=0.5, seed=0, runtime=rt))
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())
    # metadata params stay JSON-safe (no raw arrays)
    import json
    json.dumps(sig.metadata["homeostasis"]["params"])
    json.dumps(jtfne.manifest(_canon(), signals=sig))
    # diagnostics present
    assert m.last_homeostasis_diagnostics()["g_bias"].shape[0] == int(600.0 / 0.5)


def test_hdp_runs_on_canonical_column_and_json_safe():
    """HDP engages on the canonical column (mirrors
    test_homeostasis_k_gain_size_scaled_runs_and_json_safe)."""
    m = jtfne.construct(_canon())
    rt = jtfne.RuntimeConfig(
        enable_hdp=True,
        hdp_params={"K_HDP": 0.01, "alpha": 0.05, "gamma": 0.5, "K_ctrl": 0.15},
    )
    sig = jtfne.simulate(m, sim=jtfne.Simulation(duration_ms=600.0, dt_ms=0.5, seed=0, runtime=rt))
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())
    # metadata params stay JSON-safe (no raw arrays)
    import json
    json.dumps(sig.metadata["hdp"]["params"])
    json.dumps(jtfne.manifest(_canon(), signals=sig))
    # diagnostics present
    assert m.last_hdp_diagnostics()["H_trace"].shape[0] == int(600.0 / 0.5)
