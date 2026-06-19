"""``enable_homeostasis`` must route ``simulate()``/``simulate_batch()`` through the
per-neuron homeostatic kernel, expose diagnostics, and reduce to the baseline
kernel exactly when ``k_gain=0`` (the null).

The kernel itself is validated in the commit that introduced it; these tests pin
the *dispatch wiring*: config propagation, the k=0 null, diagnostics passthrough,
the JIT cache (N_compile==1 across seeds), and the receptor-kernel guard.
"""
import numpy as np
import jax.numpy as jnp
import pytest
import jaxfne as jtfne

D, DT, SEED = 400.0, 0.5, 0


def _build(runtime_kwargs=None, n=160):
    cfg = jtfne.build_laminar_column(n=n, ei_profile="canonical")
    if runtime_kwargs:
        cfg = cfg.runtime(**runtime_kwargs)
    cfg = (cfg.set_emitter("izhikevich", "cortical_eig")
              .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=8)
              .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
    return jtfne.construct(cfg)


def test_kgain_zero_null_matches_baseline_edge_list():
    base = jtfne.simulate(_build({"recurrent_backend": "edge_list"}),
                          duration_ms=D, dt_ms=DT, seed=SEED)
    null = jtfne.simulate(_build({"enable_homeostasis": True,
                                  "homeostasis_params": {"k_gain": 0.0}}),
                          duration_ms=D, dt_ms=DT, seed=SEED)
    assert bool(jnp.array_equal(base.V_m, null.V_m))
    assert bool(jnp.array_equal(base.spikes, null.spikes))


def test_homeostasis_engages_and_exposes_diagnostics():
    model = _build({"enable_homeostasis": True})
    sig = jtfne.simulate(model, duration_ms=D, dt_ms=DT, seed=SEED)
    assert sig.metadata["runtime"]["enable_homeostasis"] is True
    homeo = sig.metadata["homeostasis"]
    assert homeo["enabled"] is True
    assert "g_bias_summary" in homeo and "r_trace_summary" in homeo
    diag = model.last_homeostasis_diagnostics()
    g, r = np.asarray(diag["g_bias"]), np.asarray(diag["r_trace"])
    n_steps = int(D / DT)
    assert g.shape[0] == n_steps and r.shape[0] == n_steps
    # bounds held (defaults g in [-12, 8], r in [0, 1])
    assert -12.0 - 1e-4 <= g.min() and g.max() <= 8.0 + 1e-4
    assert 0.0 <= r.min() and r.max() <= 1.0 + 1e-4
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())


def test_default_run_has_no_homeostasis_metadata():
    sig = jtfne.simulate(_build({"recurrent_backend": "edge_list"}),
                         duration_ms=D, dt_ms=DT, seed=SEED)
    assert sig.metadata["runtime"]["enable_homeostasis"] is False
    assert "homeostasis" not in sig.metadata


def test_jit_path_compiles_once_across_seeds():
    model = _build({"enable_homeostasis": True, "jit": True})
    jtfne.simulate(model, duration_ms=D, dt_ms=DT, seed=0)
    jtfne.simulate(model, duration_ms=D, dt_ms=DT, seed=1)
    # one cache entry => one compile for the homeostatic kernel (N_compile == 1)
    assert len([k for k in model._compiled_cache if k[0] == "simulate_homeostatic"]) == 1


def test_batch_engages_homeostasis_and_null_matches():
    model = _build()
    sim_on = jtfne.Simulation(duration_ms=D, dt_ms=DT, seed=SEED,
                              runtime=jtfne.RuntimeConfig(enable_homeostasis=True))
    b = model.simulate_batch(sim_on, n_seeds=3)
    assert b["metadata"]["enable_homeostasis"] is True
    assert bool(np.isfinite(np.asarray(b["V_m"])).all())
    sim_null = jtfne.Simulation(duration_ms=D, dt_ms=DT, seed=SEED,
                                runtime=jtfne.RuntimeConfig(enable_homeostasis=True,
                                                            homeostasis_params={"k_gain": 0.0}))
    sim_base = jtfne.Simulation(duration_ms=D, dt_ms=DT, seed=SEED,
                                runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    b_null = model.simulate_batch(sim_null, n_seeds=3)
    b_base = model.simulate_batch(sim_base, n_seeds=3)
    assert bool(np.array_equal(np.asarray(b_null["V_m"]), np.asarray(b_base["V_m"])))


def test_receptor_kernel_with_homeostasis_raises():
    model = _build({"enable_homeostasis": True, "synaptic_kernel": "receptor_exponential"})
    with pytest.raises(ValueError):
        jtfne.simulate(model, duration_ms=D, dt_ms=DT, seed=SEED)
