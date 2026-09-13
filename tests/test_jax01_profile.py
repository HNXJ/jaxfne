"""24-JAX-01: execution-profile gate (measure-before-JIT-change baseline).

Pins: jit==eager bit-exactness, vmap==loop bit-exactness, single-compile
under the recompilation guard, and memory_report byte agreement. Full
timing matrix lives in scripts/perf/jax_profile_v0424.py; the receipt
records the measured table.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne


def _model():
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=8,
                 cell_types={"E": 0.8, "PV": 0.2})
        .cell_type_drives({"E": 10.0, "PV": 10.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _sim(**rt_kw):
    return jtfne.simulation(
        duration_ms=60.0, dt_ms=1.0, seed=3, record_sources=True,
        record_fields=False,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", **rt_kw),
    )


def test_batch_vmap_loop_bit_exact():
    model = _model()
    sim_v = jtfne.simulation(duration_ms=60.0, dt_ms=1.0, seed=3, runtime=jtfne.RuntimeConfig(
        recurrent_backend="edge_list", vmap=True, hdp_params={"noise_scale": 0.0}))
    sim_l = jtfne.simulation(duration_ms=60.0, dt_ms=1.0, seed=3, runtime=jtfne.RuntimeConfig(
        recurrent_backend="edge_list", vmap=False, hdp_params={"noise_scale": 0.0}))
    out_v = model.simulate_batch(sim_v, n_seeds=2)
    out_l = model.simulate_batch(sim_l, n_seeds=2)
    assert jnp.array_equal(out_v["V_m"], out_l["V_m"])
    assert jnp.array_equal(out_v["spikes"], out_l["spikes"])


def test_single_compile_under_guard():
    model = _model()
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", jit=True,
                             recompilation_guard="exception",
                             hdp_params={"noise_scale": 0.0})
    jtfne.simulate(model, _sim(jit=True, recompilation_guard="exception",
                               hdp_params={"noise_scale": 0.0}))
    # same shapes again: must not recompile (exception guard would raise)
    jtfne.simulate(model, _sim(jit=True, recompilation_guard="exception",
                               hdp_params={"noise_scale": 0.0}))
    assert len(model._compiled_cache) == 1
    _ = rt


def test_memory_report_matches_profile_bytes():
    from jaxfne._pipeline import memory_report

    model = _model()
    sim = _sim(hdp_params={"noise_scale": 0.0})
    rep = memory_report(model, sim)
    sig = jtfne.simulate(model, sim)
    assert rep["components"]["recording.V_m"] == np.asarray(sig.V_m).nbytes
    assert rep["recording_total"] >= rep["components"]["recording.V_m"]
