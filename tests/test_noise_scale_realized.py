"""Izhikevich ``noise_scale`` is realized on every Model.simulate route (P-045).

``hdp_params['noise_scale']`` sets the stochastic-current coefficient of the
Izhikevich kernels. Before P-045 it was dropped silently on the dense default
update order and on ``synaptic_kernel='receptor_exponential'``. Each route below
must now move the voltage trace with the declared value, and a declaration of
the default 0.5 must equal no declaration exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxfne as jtfne

_UNDECLARED = object()

ROUTES = {
    "dense_default": {"recurrent_backend": "dense"},
    "edge_list_exponential": {"recurrent_backend": "edge_list"},
    "edge_list_receptor_exponential": {
        "recurrent_backend": "edge_list",
        "synaptic_kernel": "receptor_exponential",
    },
}


def _model():
    cfg = (
        jtfne.configuration()
        .network(name="noise_ei", kind="balanced_ei_population", n=20,
                 cell_types={"E": 0.5, "I": 0.5})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )
    return jtfne.construct(cfg)


def _run(model, route, noise_scale, jit="auto"):
    hdp_params = None if noise_scale is _UNDECLARED else {"noise_scale": noise_scale}
    runtime = jtfne.RuntimeConfig(hdp_params=hdp_params, jit=jit, **ROUTES[route])
    sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=0, runtime=runtime)
    return model.simulate(sim)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_noise_scale_moves_the_voltage_trace(route):
    model = _model()
    runs = {ns: _run(model, route, ns) for ns in (0.0, 0.5, 2.0)}
    V = {ns: np.asarray(s.V_m) for ns, s in runs.items()}
    assert V[0.5].shape[0] == 200, "fixture must run 200 steps"
    assert float(np.std(V[0.5])) > 0.0, "fixture must give a nonzero V spread"
    assert float(np.asarray(runs[0.5].spikes).sum()) > 0, "fixture must spike"
    for a, b in ((0.0, 0.5), (0.0, 2.0), (0.5, 2.0)):
        diff = float(np.max(np.abs(V[a] - V[b])))
        assert diff > 0.0, f"{route}: noise_scale {a} and {b} gave identical V traces"


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_jit_cache_follows_noise_scale(route):
    # One model object keeps its compiled kernels; a new noise_scale must not reuse a stale one.
    model = _model()
    V = {ns: np.asarray(_run(model, route, ns, jit=True).V_m) for ns in (0.0, 2.0)}
    assert float(np.std(V[2.0])) > 0.0, "fixture must give a nonzero V spread"
    assert float(np.max(np.abs(V[0.0] - V[2.0]))) > 0.0, f"{route}: jit reused the noise_scale=0.0 kernel"
    eager = np.asarray(_run(model, route, 2.0, jit=False).V_m)
    np.testing.assert_allclose(V[2.0], eager, rtol=0, atol=1e-4)


def _batch(model, route, noise_scale, jit="auto"):
    hdp_params = None if noise_scale is _UNDECLARED else {"noise_scale": noise_scale}
    runtime = jtfne.RuntimeConfig(hdp_params=hdp_params, jit=jit, **ROUTES[route])
    sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=0, runtime=runtime)
    return np.asarray(model.simulate_batch(sim, n_seeds=2)["V_m"])


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_simulate_batch_realizes_noise_scale(route):
    model = _model()
    V = {ns: _batch(model, route, ns, jit=True) for ns in (0.0, 0.5, 2.0)}
    assert V[0.5].shape[:2] == (2, 200), "fixture must run 2 seeds of 200 steps"
    assert float(np.std(V[0.5])) > 0.0, "fixture must give a nonzero V spread"
    for a, b in ((0.0, 0.5), (0.0, 2.0), (0.5, 2.0)):
        assert float(np.max(np.abs(V[a] - V[b]))) > 0.0, f"{route}: batch noise_scale {a} == {b}"
    np.testing.assert_array_equal(_batch(model, route, _UNDECLARED, jit=True), V[0.5])


def test_simulate_batch_homeostatic_realizes_noise_scale():
    model = _model()

    def run(ns):
        runtime = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_homeostasis=True,
                                      homeostasis_params={"noise_scale": ns})
        sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=0, runtime=runtime)
        return np.asarray(model.simulate_batch(sim, n_seeds=2)["V_m"])

    low, high = run(0.0), run(2.0)
    assert float(np.std(high)) > 0.0, "fixture must give a nonzero V spread"
    assert float(np.max(np.abs(low - high))) > 0.0


def test_simulate_batch_refuses_net_m():
    cfg = (
        jtfne.configuration()
        .network(name="nm", kind="balanced_ei_population", n=20,
                 cell_types={"E": 0.5, "I": 0.5})
        .emitter(family="izhikevich", izhikevich_update_order="net_m")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )
    model = jtfne.construct(cfg)
    runtime = jtfne.RuntimeConfig(recurrent_backend="dense", hdp_params={"noise_scale": 0.0})
    sim = jtfne.Simulation(duration_ms=10.0, dt_ms=1.0, seed=0, runtime=runtime)
    with pytest.raises(ValueError, match="izhikevich_update_order='net_m'"):
        model.simulate_batch(sim, n_seeds=2)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_declared_default_equals_undeclared(route):
    model = _model()
    undeclared = _run(model, route, _UNDECLARED)
    declared = _run(model, route, 0.5)
    np.testing.assert_array_equal(np.asarray(declared.V_m), np.asarray(undeclared.V_m))
    np.testing.assert_array_equal(np.asarray(declared.spikes), np.asarray(undeclared.spikes))
