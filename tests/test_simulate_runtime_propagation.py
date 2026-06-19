"""The high-level ``simulate`` must inherit the runtime declared on the model's
Configuration via ``.runtime(...)`` — previously ``dtype``/``recurrent_backend``
set on the config were silently ignored (Simulation built a default RuntimeConfig).

x64 is toggled with save/restore so these tests don't pollute global JAX state.
"""
import numpy as np
import jax
import pytest
import jaxfne as jtfne


def _build(runtime_kwargs=None, n=200):
    cfg = jtfne.build_laminar_column(n=n, ei_profile="canonical")
    if runtime_kwargs:
        cfg = cfg.runtime(**runtime_kwargs)
    cfg = (cfg.set_emitter("izhikevich", "cortical_eig")
              .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=8)
              .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
    return jtfne.construct(cfg)


def test_config_dtype_float64_propagates_to_simulate():
    orig = jax.config.read("jax_enable_x64")
    try:
        jtfne.enable_x64()
        model = _build({"dtype": "float64"})
        sig = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)
        assert np.asarray(sig.V_m).dtype == np.float64
        assert sig.metadata["runtime"]["requested_dtype"] == "float64"
    finally:
        jax.config.update("jax_enable_x64", orig)


def test_dtype_kwarg_overrides_and_default_is_float32():
    orig = jax.config.read("jax_enable_x64")
    try:
        jtfne.enable_x64()
        model = _build()  # no runtime knobs on the config
        sig64 = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0, dtype="float64")
        assert np.asarray(sig64.V_m).dtype == np.float64
        sig32 = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)
        assert np.asarray(sig32.V_m).dtype == np.float32  # backward-compatible default
    finally:
        jax.config.update("jax_enable_x64", orig)


def test_recurrent_backend_propagates_and_is_finite():
    model = _build({"recurrent_backend": "edge_list"})
    sig = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)
    assert sig.metadata["recurrent_backend"] == "edge_list"
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())
    # default config stays on the dense backend
    dense = jtfne.simulate(_build(), duration_ms=200.0, dt_ms=0.5, seed=0)
    assert dense.metadata["recurrent_backend"] == "dense"


def test_runtime_and_dtype_keyword_conflict_raises():
    model = _build()
    with pytest.raises(ValueError):
        jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5,
                       runtime=jtfne.RuntimeConfig(dtype="float64"), dtype="float64")


def test_explicit_simulation_object_still_works():
    model = _build()
    sim = jtfne.Simulation(duration_ms=200.0, dt_ms=0.5, seed=1)
    sig = jtfne.simulate(model, sim)
    assert np.asarray(sig.V_m).shape[0] == int(200.0 / 0.5)
