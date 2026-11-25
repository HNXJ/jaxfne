"""``enable_hdp`` must route ``simulate()`` through the HDP kernel and expose
diagnostics. These tests cover the H-state and weight-term null behavior; they
do not claim full-system equivalence with the baseline edge-list kernel.

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


def test_null_gains_match_baseline_bit_exact():
    """Null builtin HDP parameters route through the baseline kernel (23-HDP-01)."""
    baseline = _build({"enable_hdp": False})
    null_hdp = _build({"enable_hdp": True})
    v_base = np.asarray(jtfne.simulate(baseline, duration_ms=D, dt_ms=DT, seed=SEED).V_m)
    v_null = np.asarray(jtfne.simulate(null_hdp, duration_ms=D, dt_ms=DT, seed=SEED).V_m)
    assert np.array_equal(v_base, v_null)
    assert null_hdp.last_hdp_diagnostics() is None


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


def test_simulate_batch_engages_hdp_and_default_is_unaffected():
    """simulate_batch must dispatch to the HDP kernel when enable_hdp=True is
    passed via RuntimeConfig, mirroring test_batch_engages_homeostasis_and_null_matches
    in test_homeostasis_dispatch.py. Closes the gap documented in
    jaxfne/skills/jaxfne-neural-tensor/SKILL.md's HDP landmines section
    (2026-07-22): prior to this pass, simulate_batch never read enable_hdp at
    all, so a caller enabling HDP via simulate() and calling simulate_batch()
    on the same Model/config would silently diverge (HDP-active vs not)."""
    model = _build()
    sim_on = jtfne.Simulation(duration_ms=D, dt_ms=DT, seed=SEED,
                              runtime=jtfne.RuntimeConfig(
                                  enable_hdp=True,
                                  hdp_params={"alpha": 0.05, "gamma": 0.5,
                                              "K_ctrl": 0.15, "K_HDP": 0.01,
                                              "tau_0_ms": 5.0}))
    b_on = model.simulate_batch(sim_on, n_seeds=3)
    assert b_on["metadata"]["enable_hdp"] is True
    assert bool(np.isfinite(np.asarray(b_on["V_m"])).all())

    # Non-null HDP batch V_m must actually differ from the non-HDP baseline
    # batch (proves the HDP kernel really ran, not just that no error occurred).
    sim_base = jtfne.Simulation(duration_ms=D, dt_ms=DT, seed=SEED,
                                runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    b_base = model.simulate_batch(sim_base, n_seeds=3)
    assert not np.array_equal(np.asarray(b_on["V_m"]), np.asarray(b_base["V_m"]))

    # enable_hdp omitted (default False) must be unaffected -- matches the
    # non-HDP edge_list baseline bit-exactly when the same recurrent_backend
    # is used (RuntimeConfig()'s own default backend is "dense", a different
    # kernel entirely, so pin edge_list on both sides for a fair comparison).
    sim_default = jtfne.Simulation(duration_ms=D, dt_ms=DT, seed=SEED,
                                   runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    b_default = model.simulate_batch(sim_default, n_seeds=3)
    assert b_default["metadata"]["enable_hdp"] is False
    assert np.array_equal(np.asarray(b_default["V_m"]), np.asarray(b_base["V_m"]))


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


# --- H without HDP (plan R46): known negative, known positive, null and DEFAULT_HDP gains

N = 20
LEGACY = {"K_HDP": 0.05, "K_ctrl": 0.3, "K_w_ctrl": 0.002, "alpha": 0.05,
          "gamma": 0.05, "C_spike": 0.02, "noise_scale": 0.0}
PURE_RBD = {"K_HDP": 0.0, "K_w_ctrl": 0.0, "K_ctrl": 5.0, "noise_scale": 0.0}
NULL_GAINS_RBD = {"K_HDP": 0.0, "K_w_ctrl": 0.0, "K_ctrl": 0.0, "noise_scale": 0.0}
DEFAULT_HDP_RBD = {**jtfne.DEFAULT_HDP, "K_HDP": 0.0, "K_w_ctrl": 0.0}


def _model():
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=N, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _sim(hp=None):
    kw = {} if hp is None else {"enable_hdp": True, "hdp_params": hp}
    return jtfne.simulation(
        duration_ms=60.0, dt_ms=1.0, seed=3, record_sources=True, record_fields=False,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", jit=False, **kw),
    )


def _seeded_pair(model):
    n = int(model.params["emitter"].n_neurons)
    return (model.with_hdp_initial_state(H0=np.ones(n)),
            model.with_hdp_initial_state(H0=np.linspace(0.2, 4.0, n)))


def test_h_is_inert_when_hdp_is_off():
    model = _model()
    ones, spread = _seeded_pair(model)
    base = jtfne.simulate(model, _sim())
    assert float(np.asarray(base.spikes).sum()) > 0, "fixture must spike"
    for seeded in (ones, spread):
        other = jtfne.simulate(seeded, _sim())
        assert seeded.last_hdp_diagnostics() is None
        assert np.array_equal(np.asarray(base.spikes), np.asarray(other.spikes))
        assert np.array_equal(np.asarray(base.V_m), np.asarray(other.V_m))
        # sources: bitwise identical on CPU (6 repeats x 3 H0, measured). On the default
        # GPU backend (RTX A4000, JAX 0.10.2) fresh processes differ by up to 1.19e-6
        # absolute in 2-4 of 1200 entries (12 processes, HDP off, 60 steps x 20 neurons).
        # The source is the scatter-add in jax.ops.segment_sum; XLA_FLAGS=
        # --xla_gpu_deterministic_ops=true removes it. atol=2e-5 is ~17x that bound;
        # rtol=1e-5 scales with the data (CPU, HDP off: max |sources| ~26, median
        # nonzero |sources| ~5). H is not the source of the gap.
        np.testing.assert_allclose(np.asarray(other.sources), np.asarray(base.sources),
                                   rtol=1e-5, atol=2e-5)


def test_h_drives_state_when_hdp_is_on():
    ones, spread = _seeded_pair(_model())
    sig_a = jtfne.simulate(ones, _sim(LEGACY))
    sig_b = jtfne.simulate(spread, _sim(LEGACY))
    # t=0 differs by construction (the seeds); the claim is about the dynamics after it
    assert float(np.abs(np.asarray(sig_a.V_m) - np.asarray(sig_b.V_m)).max()) > 1e-3, (
        "HDP on must make the outputs depend on H")
    da, db = ones.last_hdp_diagnostics(), spread.last_hdp_diagnostics()
    assert float(np.abs(np.asarray(da["H_trace"])[1:] - np.asarray(db["H_trace"])[1:]).max()) > 1e-3
    w = np.asarray(da["w_trace"])
    assert float(np.abs(w - w[:1]).max()) > 0, "legacy HDP must move the weights (live counterpart)"


def test_pure_rbd_moves_h_and_freezes_weights():
    _, spread = _seeded_pair(_model())
    jtfne.simulate(spread, _sim(PURE_RBD))
    diag = spread.last_hdp_diagnostics()
    h = np.asarray(diag["H_trace"])
    w = np.asarray(diag["w_trace"])
    assert w.shape[1] > 0, "fixture must have edges"
    # compare each step with the seed: the seeded spread alone spans 3.8 at t=0
    assert float(np.abs(h - h[:1]).max()) > 1e-3, "H must evolve under RBD"
    assert np.array_equal(w, np.broadcast_to(w[:1], w.shape)), "weights must stay fixed"


def test_null_controller_gains_hold_h_at_its_seed_under_rbd():
    """Observed: with null controller gains (K_ctrl = K_w_ctrl = K_HDP = 0; K_ctrl = 0 is the
    RuntimeConfig default), H has no dynamics even with HDP engaged."""
    _, spread = _seeded_pair(_model())
    jtfne.simulate(spread, _sim(NULL_GAINS_RBD))
    h = np.asarray(spread.last_hdp_diagnostics()["H_trace"])
    assert np.array_equal(h, np.broadcast_to(h[:1], h.shape)), "null gains must freeze H"


def test_default_hdp_preset_moves_h_under_rbd():
    """Observed: the DEFAULT_HDP preset (K_ctrl = 5) moves H with K_HDP = K_w_ctrl = 0."""
    _, spread = _seeded_pair(_model())
    jtfne.simulate(spread, _sim(DEFAULT_HDP_RBD))
    h = np.asarray(spread.last_hdp_diagnostics()["H_trace"])
    assert float(np.abs(h - h[:1]).max()) > 1e-3, "DEFAULT_HDP's K_ctrl must move H"
