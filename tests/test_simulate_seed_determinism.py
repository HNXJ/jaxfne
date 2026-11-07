"""Seed determinism on the simulate() routes that consume a seed (C2).

Per route: the same seed run twice gives bit-identical spikes and V_m, and seeds 0 and 1
give different V_m. Each fixture holds a seeded random input and is active: spikes > 0,
except the rate route, which emits no spikes, so its rate trace must vary.

Covered:
- izhikevich: per-cell gaussian_drive noise keyed by Simulation.seed.
- homeostatic_ei: default noise_scale keyed by Simulation.seed (the route passes none).
- lif: Poisson drive without a declared seed, so it uses Simulation.seed + 7919.
- rate: noise sigma keyed by Simulation.seed.
- adex: Simulation.seed must stay 0 (refused otherwise, tests/test_adex_route.py). The
  drive seed is poisson_drive["seed"], which this test varies.
- hh: Simulation.seed must stay 0 (refused otherwise,
  jaxfne/_model_simulate.py _simulate_hh); seed 0 run twice gives bit-identical
  V_m, and the V trace is non-constant (so the test cannot pass on an empty run).
"""

import inspect
from typing import Any, Callable, NamedTuple

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.emitters_homeostatic_ei import simulate_homeostatic_ei

DT_MS = 0.5
ADEX_DT_MS = 0.1
DUR_MS = 100.0
N_CELLS = 20
HEI_N_CELLS = 16


def _observe(signals):
    return np.asarray(signals.spikes), np.asarray(signals.V_m, dtype=float)


def _spiking(spikes, _v):
    return int(spikes.sum()) > 0


def _rate_varies(_spikes, v):
    return float(v.std()) > 0.0


# Izhikevich: per-cell gaussian noise on E and I, keyed by Simulation.seed.
def _izh_model():
    cfg = (
        jtfne.configuration()
        .network(name="c2_izh", kind="balanced_ei_population", n=N_CELLS,
                 cell_types={"E": 0.5, "I": 0.5})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )
    return jtfne.construct(cfg)


def _izh_sim(seed):
    return jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"),
        gaussian_drive={"std_by_cell_type": {"E": 3.0, "I": 3.0}},
    )


def _izh_input(_model, sim):
    return max(sim.gaussian_drive["std_by_cell_type"].values()) > 0.0


# Homeostatic EI: the route passes no noise_scale, so the function default applies.
def _hei_model():
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, duration_ms=DUR_MS, dt_ms=DT_MS)
        .network(name="c2_hei", n=HEI_N_CELLS)
        .set_emitter("homeostatic_ei")
        .field()
        .probe(modes=["vm"])
    )
    return jtfne.construct(cfg)


def _hei_sim(seed):
    return jtfne.Simulation(duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed)


def _hei_input(_model, _sim):
    default = inspect.signature(simulate_homeostatic_ei).parameters["noise_scale"].default
    return float(default) > 0.0


# LIF: Poisson drive with no declared seed, so the drive seed is Simulation.seed + 7919.
def _lif_model():
    tensor = nt.NeuronalTensor(
        name="c2-lif",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(
                name="L4", n_neurons=N_CELLS,
                neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")])],
            inter_connections=[nt.InterConnection(
                source_layer="L4", source_neuron_type="E",
                target_layer="L4", target_neuron_type="E",
                mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0))],
        )],
    )
    return jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=0, dt_ms=DT_MS, emitter="lif", duration_ms=DUR_MS))


def _lif_sim(seed):
    return jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))


def _lif_input(_model, sim):
    return float(sim.poisson_drive["rate_hz"]) > 0.0 and "seed" not in sim.poisson_drive


# Rate: noise sigma on every cell, keyed by Simulation.seed; no spikes are emitted.
def _rate_model():
    types = [
        nt.NeuronType.make(name, fraction=0.5, emitter="rate",
                           emitter_params={"sigma": 0.5})
        for name in ("E", "I")
    ]
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="I",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0), delay_ms=1.0)
    tensor = nt.NeuronalTensor(
        name="c2-rate",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=N_CELLS, neuron_types=types)],
            inter_connections=[conn])],
    )
    return jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=0, dt_ms=DT_MS, emitter="rate", duration_ms=DUR_MS))


def _rate_sim(seed):
    return jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=DT_MS, seed=seed,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))


def _rate_input(model, _sim):
    return float(np.max(np.asarray(model.params["emitter"].sigma))) > 0.0


# AdEx: the drive seed rides poisson_drive["seed"]; Simulation.seed stays 0.
def _adex_model():
    cfg = (
        jtfne.Configuration()
        .network(name="c2_adex", n=N_CELLS, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="adex", adex_decay="exp", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": 8.0, "reversal_mV": -80.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.2, weight=0.5, mechanism="ampa", delay_ms=2.0,
                     sampling="bernoulli")
        .connections(name="e_pv", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=0.2, weight=1.0, mechanism="ampa", delay_ms=1.3,
                     sampling="bernoulli")
        .connections(name="pv_e", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=0.2, weight=-2.0, mechanism="gaba", delay_ms=0.25,
                     sampling="bernoulli")
        .connectivity(edge_seed=11)
    )
    return jtfne.construct(cfg)


def _adex_sim(seed):
    return jtfne.Simulation(
        duration_ms=DUR_MS, dt_ms=ADEX_DT_MS,
        poisson_drive={"n_sources": 560, "rate_hz": 4.0, "weight_nS": 2.5,
                       "mechanism": "ampa", "target_cell_types": ["E", "PV"],
                       "delay_ms": 2.0, "seed": seed})


def _adex_input(_model, sim):
    return float(sim.poisson_drive["rate_hz"]) > 0.0 and sim.seed == 0


class Route(NamedTuple):
    build: Callable[[], Any]
    sim_for: Callable[[int], Any]
    seeded_input: Callable[[Any, Any], bool]
    active: Callable[[np.ndarray, np.ndarray], bool]

ROUTES = {
    "izhikevich": Route(_izh_model, _izh_sim, _izh_input, _spiking),
    "homeostatic_ei": Route(_hei_model, _hei_sim, _hei_input, _spiking),
    "lif": Route(_lif_model, _lif_sim, _lif_input, _spiking),
    "rate": Route(_rate_model, _rate_sim, _rate_input, _rate_varies),
    "adex": Route(_adex_model, _adex_sim, _adex_input, _spiking),
}


def _hh_model():
    """Smallest hh-family builder used in tests/ (same shape as
    tests/test_construct_golden_snapshot.py::_hh_test_config)."""
    cfg = (
        jtfne.configuration()
        .network(n=10, cell_types={"E": 0.6, "Ig": 0.2, "Il": 0.2})
        .emitter(family="hh")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0, "tau_r_ms": 0.2,
                            "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1})
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=None, weight=2.5, mechanism="ampa",
                     sign="excitatory")
        .connectivity(edge_seed=0)
    )
    return jtfne.construct(cfg)


def _hh_sim(seed):
    return jtfne.Simulation(duration_ms=2.0, dt_ms=DT_MS, seed=seed)


def test_hh_seed_nonzero_refused():
    """seed != 0 must raise: the hh route has no noise, so there is no keyed
    randomness for Simulation.seed to drive."""
    model = _hh_model()
    with pytest.raises(ValueError, match="does not consume Simulation.seed"):
        model.simulate(_hh_sim(1))


def test_hh_seed_zero_deterministic_and_non_constant():
    model = _hh_model()
    v_a = np.asarray(model.simulate(_hh_sim(0)).V_m, dtype=float)
    v_b = np.asarray(model.simulate(_hh_sim(0)).V_m, dtype=float)
    assert np.array_equal(v_a, v_b), "seed 0 run twice must give bit-identical V_m"
    assert float(v_a.std()) > 0.0, "V trace must be non-constant (else this passes on an empty run)"


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_same_seed_reproduces_and_seed_1_differs(route):
    spec = ROUTES[route]
    model = spec.build()
    assert spec.seeded_input(model, spec.sim_for(0)), "fixture must hold a seeded random input"
    spikes_a, v_a = _observe(model.simulate(spec.sim_for(0)))
    spikes_b, v_b = _observe(model.simulate(spec.sim_for(0)))
    spikes_c, v_c = _observe(model.simulate(spec.sim_for(1)))
    assert spec.active(spikes_a, v_a), "fixture must be active: spikes > 0 (rate: V_m varies)"
    assert np.array_equal(spikes_a, spikes_b), "the same seed must give identical spikes"
    assert np.array_equal(v_a, v_b), "the same seed must give identical V_m"
    assert not np.array_equal(v_a, v_c), "seeds differ: seed 0 and seed 1 must give different V_m"
