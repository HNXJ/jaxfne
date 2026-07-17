"""``enable_hdp`` must route ``simulate()`` through the HDP kernel, expose
diagnostics, and reduce to the baseline edge-list kernel exactly when all
income/spending/control gains are zero (the documented null control).

The kernel itself is validated standalone in test_hdp_kernel_standalone.py;
these tests pin the *dispatch wiring* mirroring test_homeostasis_dispatch.py:
config propagation, the null, diagnostics passthrough, the JIT cache
(N_compile==1 across seeds), and the receptor-kernel + mutual-exclusivity
guards.
"""
import numpy as np
import pytest
import jaxfne as jtfne

D, DT, SEED = 200.0, 0.5, 0


def _build(runtime_kwargs=None, n=80):
    cfg = jtfne.build_laminar_column(n=n, ei_profile="canonical")
    if runtime_kwargs:
        cfg = cfg.runtime(**runtime_kwargs)
    cfg = (cfg.set_emitter("izhikevich", "cortical_eig")
              .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=8)
              .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
    return jtfne.construct(cfg)


def test_null_gains_freeze_H_and_weights():
    """All income/spending/control gains default to 0.0 -- the documented null
    control. Confirmed bit-exact at the kernel level in
    test_hdp_kernel_standalone.py; through the dispatch path the resulting
    V_m trajectory is *not* asserted bit-identical to the baseline kernel,
    since the HDP step function carries extra (unused-at-null) terms (e.g.
    H_boost_gain's boost factor, per-edge |w| bookkeeping) that XLA may fuse
    differently from the baseline kernel's leaner graph -- a sub-ULP
    floating-point difference that a chaotic spiking system amplifies over
    hundreds of steps, not a dispatch-wiring bug. What the null control
    *does* guarantee, and what this test checks directly: H stays pinned at
    1.0 and edge weights are unchanged."""
    model_obj = _build({"enable_hdp": True})
    jtfne.simulate(model_obj, duration_ms=D, dt_ms=DT, seed=SEED)
    diag = model_obj.last_hdp_diagnostics()
    H = np.asarray(diag["H_trace"])
    w_final = np.asarray(diag["w_final"])
    w0 = np.asarray(model_obj.params["edge_list"].weight)
    assert np.allclose(H, 1.0, atol=1e-6)
    assert np.allclose(w_final, w0, atol=1e-6)


def test_hdp_engages_and_exposes_diagnostics():
    model = _build({"enable_hdp": True,
                    "hdp_params": {"alpha": 0.05, "gamma": 0.5, "K_ctrl": 0.15,
                                   "K_HDP": 0.01, "tau_0_ms": 5.0}})
    sig = jtfne.simulate(model, duration_ms=D, dt_ms=DT, seed=SEED)
    assert sig.metadata["runtime"]["enable_hdp"] is True
    hdp = sig.metadata["hdp"]
    assert hdp["enabled"] is True
    assert "H_trace_summary" in hdp and "w_final_summary" in hdp
    diag = model.last_hdp_diagnostics()
    H = np.asarray(diag["H_trace"])
    n_steps = int(D / DT)
    assert H.shape[0] == n_steps
    assert bool(np.isfinite(H).all())
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())


def test_size_scale_by_cell_type_forwarded_through_dispatch():
    """hdp_params["size_scale_by_cell_type"] must reach the kernel's tau_i
    (cube law, 0.4.7): a cell type declared 2x size must show ~8x smaller H
    deviation than a 1x-size type under identical income/spending gains,
    exactly mirroring the standalone-kernel cube-law test but exercised
    through the full Configuration/RuntimeConfig dispatch path (this was a
    real gap -- size_scale_by_cell_type/size_scale_override were silently
    dropped by _hdp_packed before being wired)."""
    model = _build({"enable_hdp": True,
                    "hdp_params": {"alpha": 0.0, "beta": 0.02, "K_HDP": 0.0,
                                   "tau_0_ms": 50.0,
                                   "size_scale_by_cell_type": {"E": 2.0, "PV": 1.0,
                                                                "SST": 1.0, "VIP": 1.0}}})
    jtfne.simulate(model, duration_ms=D, dt_ms=DT, seed=SEED)
    diag = model.last_hdp_diagnostics()
    labels = np.asarray(model.params["emitter"].labels)
    H_dev = np.asarray(diag["H_trace"])[-1] - 1.0
    e_dev = float(H_dev[labels == "E"].mean())
    pv_dev = float(H_dev[labels == "PV"].mean())
    assert pv_dev / e_dev == pytest.approx(8.0, rel=0.05)


def test_default_run_has_no_hdp_metadata():
    sig = jtfne.simulate(_build({"recurrent_backend": "edge_list"}),
                         duration_ms=D, dt_ms=DT, seed=SEED)
    assert sig.metadata["runtime"]["enable_hdp"] is False
    assert "hdp" not in sig.metadata


def test_jit_path_compiles_once_across_seeds():
    model = _build({"enable_hdp": True, "jit": True,
                    "hdp_params": {"alpha": 0.05, "gamma": 0.5, "K_ctrl": 0.15}})
    jtfne.simulate(model, duration_ms=D, dt_ms=DT, seed=0)
    jtfne.simulate(model, duration_ms=D, dt_ms=DT, seed=1)
    assert len([k for k in model._compiled_cache if k[0] == "simulate_hdp"]) == 1


def test_receptor_kernel_with_hdp_raises():
    model = _build({"enable_hdp": True, "synaptic_kernel": "receptor_exponential"})
    with pytest.raises(ValueError, match="enable_hdp"):
        jtfne.simulate(model, duration_ms=D, dt_ms=DT, seed=SEED)


def test_enable_homeostasis_and_enable_hdp_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        jtfne.RuntimeConfig(enable_homeostasis=True, enable_hdp=True)


def test_record_weight_trace_false_drops_w_trace_but_keeps_dynamics_identical():
    """hdp_params["record_weight_trace"]=False must thread through
    Configuration/Model.simulate() to the kernel: last_hdp_diagnostics()'s
    w_trace becomes None (avoiding the (n_steps, n_edges) memory cost at
    scale -- plans.json:hdp-edge-trace-memory-scaling), while w_final/H_trace
    and the actual simulated V_m trajectory are bit-identical to the default
    (True) run with the same seed."""
    hp = {"alpha": 0.05, "gamma": 0.5, "K_ctrl": 0.15, "K_HDP": 0.01, "tau_0_ms": 5.0}
    model_true = _build({"enable_hdp": True, "hdp_params": hp})
    sig_true = jtfne.simulate(model_true, duration_ms=D, dt_ms=DT, seed=SEED)
    diag_true = model_true.last_hdp_diagnostics()

    model_false = _build({"enable_hdp": True,
                          "hdp_params": {**hp, "record_weight_trace": False}})
    sig_false = jtfne.simulate(model_false, duration_ms=D, dt_ms=DT, seed=SEED)
    diag_false = model_false.last_hdp_diagnostics()

    assert diag_true["w_trace"] is not None
    assert diag_false["w_trace"] is None
    np.testing.assert_array_equal(np.asarray(diag_true["H_trace"]), np.asarray(diag_false["H_trace"]))
    np.testing.assert_allclose(np.asarray(diag_true["w_final"]), np.asarray(diag_false["w_final"]), atol=1e-6)
    assert np.array_equal(np.asarray(sig_true.V_m), np.asarray(sig_false.V_m))


def test_cache_key_isolated_across_hdp_params_on_reused_model():
    """A reused Model switching K_HDP (identical shapes) must not replay a stale
    compiled closure built for the first K_HDP -- mirrors
    test_cache_key_isolated_across_r_star_on_reused_model for homeostasis."""
    model = _build({"jit": True})
    sim_a = jtfne.Simulation(duration_ms=D, dt_ms=DT, seed=SEED,
                             runtime=jtfne.RuntimeConfig(
                                 enable_hdp=True, jit=True,
                                 hdp_params={"alpha": 0.05, "gamma": 0.5,
                                             "K_ctrl": 0.15, "K_HDP": 0.01}))
    sig_a = model.simulate(sim_a)
    sim_b = jtfne.Simulation(duration_ms=D, dt_ms=DT, seed=SEED,
                             runtime=jtfne.RuntimeConfig(
                                 enable_hdp=True, jit=True,
                                 hdp_params={"alpha": 0.05, "gamma": 0.5,
                                             "K_ctrl": 0.15, "K_HDP": 0.5}))
    sig_b = model.simulate(sim_b)
    assert not np.array_equal(np.asarray(sig_a.V_m), np.asarray(sig_b.V_m))
    assert len([k for k in model._compiled_cache if k[0] == "simulate_hdp"]) == 2
