"""bfloat16 is a storage dtype (Hamm, 2026-10-28): dynamics integrate in float32.

With ``RuntimeConfig(dtype="bfloat16")`` every route that accepts the dtype integrates in
float32 and casts its recorded arrays to bfloat16 once, at the end. So a bfloat16 run's
voltages are the float32 run's voltages cast to bfloat16 exactly, its spikes equal the
float32 spikes, and its time axis is the float32 axis (time is not a state variable).
"""

from __future__ import annotations

from collections import namedtuple
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._model_simulate import simulate_adex_seed_batch, simulate_lif_seed_batch
from jaxfne._runtime_config import RuntimeConfig
from jaxfne.neuronal_tensor import make_minimal_ei_tensor

Run = namedtuple("Run", "V_m spikes time_ms")

# amplitude 0.3 is not representable in bfloat16 (0.30078125): a bfloat16 drive differs from float32.
LIF_POISSON = {"rate_hz": 200, "amplitude": 0.3, "target": "all", "seed": 11}
LIF_POISSON_MEMBER = {"rate_hz": 200, "amplitude": 0.3, "target": "all"}  # seeds ride the argument
ADEX_DRIVE = {
    "n_sources": 560, "rate_hz": 4.0, "weight_nS": 2.5, "mechanism": "ampa",
    "target_cell_types": ["E", "PV"], "delay_ms": 2.0, "seed": 7,
}


def _adex_config():
    p = 0.2
    return (
        jtfne.Configuration()
        .network(name="adex60", n=60, cell_types={"E": 0.8, "PV": 0.1, "SOM": 0.1})
        .emitter(family="adex", adex_decay="exp", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": 8.0, "reversal_mV": -80.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=p, weight=0.5, mechanism="ampa", delay_ms=2.0, sampling="bernoulli")
        .connections(name="e_pv", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=p, weight=1.0, mechanism="ampa", delay_ms=1.3, sampling="bernoulli")
        .connections(name="pv_e", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=p, weight=-2.0, mechanism="gaba", delay_ms=0.25, sampling="bernoulli")
        .connections(name="som_e", source={"cell_type": "SOM"}, target={"cell_type": "E"},
                     probability=p, weight=1.0, mechanism="gaba", delay_ms=0.8, sampling="bernoulli")
        .connectivity(edge_seed=11)
    )


def _hh_config():
    return (
        jtfne.Configuration()
        .network(name="mini", n=3, cell_types={"E": 2.0 / 3.0, "Ig": 1.0 / 3.0})
        .emitter(family="hh")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=None, weight=4.0, mechanism="ampa")
    ).connectivity(edge_seed=0)


def _homeostatic_ei_config():
    return (
        jtfne.configuration()
        .runtime(seed=0, duration_ms=100.0, dt_ms=0.5)
        .network(name="ei8", n=8)
        .set_emitter("homeostatic_ei")
        .field()
        .probe(modes=["vm"])
    )


@lru_cache(maxsize=None)
def _model(name):
    if name == "izhikevich":
        return jtfne.construct(jtfne.suite2_four_celltype_config(seed=0))
    if name == "lif":
        return jtfne.construct(make_minimal_ei_tensor(), jtfne.RuntimeConfiguration(
            duration_ms=50.0, dt_ms=0.5, seed=0, emitter="lif"))
    if name == "adex":
        return jtfne.construct(_adex_config())
    if name == "hh":
        return jtfne.construct(_hh_config())
    if name == "homeostatic_ei":
        return jtfne.construct(_homeostatic_ei_config())
    raise KeyError(name)


def _run_sig(model_name, **sim_kw):
    sig = _model(model_name).simulate(jtfne.Simulation(seed=0, **sim_kw))
    return Run(sig.V_m, sig.spikes, sig.time_ms)


def _izhikevich(dtype, **runtime_kw):
    return _run_sig("izhikevich", duration_ms=100.0, dt_ms=0.1,
                    runtime=RuntimeConfig(dtype=dtype, **runtime_kw))


def _seed_batch(run, model_name, sim_kw, seeds):
    out = run(_model(model_name), jtfne.Simulation(seed=0, **sim_kw), seeds)
    return Run(out["V_m"], out["spikes"], None)


# Route -> run(dtype) -> Run. Routes whose run has no time axis report time_ms None.
ROUTES = {
    "izhikevich_dense": lambda dtype: _izhikevich(dtype),
    "izhikevich_edge_list": lambda dtype: _izhikevich(dtype, recurrent_backend="edge_list"),
    "lif": lambda dtype: _run_sig(
        "lif", duration_ms=50.0, dt_ms=0.5, poisson_drive=LIF_POISSON,
        runtime=RuntimeConfig(dtype=dtype, recurrent_backend="edge_list")),
    "hh": lambda dtype: _run_sig(
        "hh", duration_ms=50.0, dt_ms=0.1, runtime=RuntimeConfig(dtype=dtype)),
    "homeostatic_ei": lambda dtype: _run_sig(
        "homeostatic_ei", duration_ms=200.0, dt_ms=0.5, runtime=RuntimeConfig(dtype=dtype)),
    "adex_seed_batch": lambda dtype: _seed_batch(
        simulate_adex_seed_batch, "adex",
        dict(duration_ms=50.0, dt_ms=0.1, poisson_drive=ADEX_DRIVE,
             runtime=RuntimeConfig(dtype=dtype, recurrent_backend="edge_list")),
        [7, 8]),
    "lif_seed_batch": lambda dtype: _seed_batch(
        simulate_lif_seed_batch, "lif",
        dict(duration_ms=50.0, dt_ms=0.5, poisson_drive=LIF_POISSON_MEMBER,
             runtime=RuntimeConfig(dtype=dtype, recurrent_backend="edge_list")),
        [3, 4]),
}
TIMED_ROUTES = ["izhikevich_dense", "izhikevich_edge_list", "lif", "hh", "homeostatic_ei"]


@lru_cache(maxsize=None)
def _pair(route):
    run = ROUTES[route]
    return run("float32"), run("bfloat16")


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_route_integrates_in_float32_then_casts_voltages_once(route):
    f, b = _pair(route)
    v32 = np.asarray(f.V_m, np.float32)
    assert np.ptp(v32) > 0, f"{route}: fixture voltages must move"
    assert float(np.asarray(f.spikes, np.float32).sum()) > 0, f"{route}: fixture must spike"
    assert b.V_m.dtype == jnp.bfloat16, f"{route}: recorded voltages are the storage dtype"
    expected = np.asarray(jnp.asarray(v32).astype(jnp.bfloat16), np.float32)
    np.testing.assert_array_equal(
        np.asarray(b.V_m, np.float32), expected,
        err_msg=f"{route}: bfloat16 voltages are not the float32 voltages cast once",
    )


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_route_spikes_are_the_float32_spikes(route):
    f, b = _pair(route)
    assert b.spikes.dtype == f.spikes.dtype, f"{route}: spikes keep their own dtype"
    np.testing.assert_array_equal(
        np.asarray(b.spikes, np.float32), np.asarray(f.spikes, np.float32),
        err_msg=f"{route}: bfloat16 run spikes differ from the float32 run",
    )


@pytest.mark.parametrize("route", TIMED_ROUTES)
def test_time_axis_is_float32_on_a_bfloat16_run(route):
    f, b = _pair(route)
    assert np.asarray(b.time_ms).shape[0] > 1, f"{route}: fixture must have a time axis"
    assert b.time_ms.dtype == jnp.float32, f"{route}: time is not a storage quantity"
    np.testing.assert_array_equal(np.asarray(b.time_ms), np.asarray(f.time_ms))


def test_bfloat16_time_axis_keeps_float32_precision_over_1000_steps():
    sig = _run_sig("izhikevich", duration_ms=100.0, dt_ms=0.1,
                   runtime=RuntimeConfig(dtype="bfloat16"))
    t = np.asarray(sig.time_ms)
    assert t.shape == (1000,), "fixture must run 1000 steps at dt 0.1"
    assert t.dtype == np.float32
    assert float(t[-1]) == pytest.approx(99.9, rel=1e-6)


def test_runtime_config_separates_storage_dtype_from_compute_dtype():
    rt = RuntimeConfig(dtype="bfloat16")
    assert rt.actual_dtype == "bfloat16", "the report keeps the requested and stored dtype"
    assert rt.jnp_dtype == jnp.bfloat16
    assert rt.compute_dtype == "float32", "bfloat16 is storage only; dynamics integrate in float32"
    assert rt.jnp_compute_dtype == jnp.float32
    rt32 = RuntimeConfig(dtype="float32")
    assert rt32.compute_dtype == rt32.actual_dtype == "float32"
    assert rt32.jnp_compute_dtype == rt32.jnp_dtype == jnp.float32
