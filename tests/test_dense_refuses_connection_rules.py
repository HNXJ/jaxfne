"""Dense backend refuses connection rules it would ignore.

A ``connections()`` rule is compiled into ``params["edge_list"]`` at construct
time, but the dense kernel (``simulate_eig_izhikevich``) runs on ``emitter.W``
only. Running such a model on ``recurrent_backend="dense"`` must raise instead
of silently dropping the declared edges.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxfne as jtfne


def _build_model(edge_seed: int | None = None):
    cfg = (
        jtfne.configuration()
        .network(
            name="dense_refusal",
            kind="balanced_ei_population",
            n=50,
            cell_types={"E": 0.8, "I": 0.2},
        )
        .connections(
            name="rec_ee",
            source={"cell_type": "E"},
            target={"cell_type": "E"},
            probability=0.5,
            weight=1.0,
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="declared_proxy",
            gauge="mean_zero",
        )
        .probe(name="spike_probe", modes=["spikes"])
    )
    if edge_seed is not None:
        cfg = cfg.connectivity(edge_seed=int(edge_seed))
    return jtfne.construct(cfg)


def _build_plain_model():
    cfg = (
        jtfne.configuration()
        .network(
            name="plain",
            kind="balanced_ei_population",
            n=50,
            cell_types={"E": 0.8, "I": 0.2},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="declared_proxy",
            gauge="mean_zero",
        )
        .probe(name="spike_probe", modes=["spikes"])
    )
    return jtfne.construct(cfg)


def test_dense_refuses_connection_rules():
    model = _build_model(edge_seed=0)
    sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=0)
    with pytest.raises(ValueError, match="recurrent_backend='edge_list'"):
        model.simulate(sim)


def test_dense_batch_refuses_connection_rules():
    model = _build_model(edge_seed=0)
    sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=0)
    with pytest.raises(ValueError, match="recurrent_backend='edge_list'"):
        model.simulate_batch(sim, n_seeds=2, seed=1)


def _n_rule_edges(model):
    rules = model.cfg.metadata["circuit"]["connections"]
    return sum(int(rule.get("compiled_n_edges", 0) or 0) for rule in rules)


def test_edge_list_runs_and_seeds_differ():
    runtime = jtfne.RuntimeConfig(recurrent_backend="edge_list", seed=0)
    sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=0, runtime=runtime)
    spikes = []
    for edge_seed in (0, 1):
        model = _build_model(edge_seed=edge_seed)
        assert _n_rule_edges(model) > 0, "fixture must compile rule edges"
        signals = model.simulate(sim)
        array = np.asarray(signals.get("spikes"))
        assert array.sum() > 0, "fixture must spike"
        spikes.append(array)
    assert not np.array_equal(spikes[0], spikes[1])


def test_dense_without_rules_still_runs():
    model = _build_plain_model()
    sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=0)
    array = np.asarray(model.simulate(sim).get("spikes"))
    assert array.shape == (200, 50)
    assert array.sum() > 0, "fixture must spike"
