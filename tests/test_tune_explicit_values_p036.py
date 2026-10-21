"""Packet O-39a: tune stub verdict and explicit-zero generations/population_size."""

import pytest

import jaxfne as jtfne


def _model_and_sim(n=12, duration_ms=10.0):
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=0.1, seed=0)
    return model, sim


def test_optax_stub_no_eligible_path_reports_revise():
    """Stub (no eligible gradient path, optax present) must report REVISE."""
    optax = pytest.importorskip("optax")
    del optax
    model, sim = _model_and_sim()
    # drive_gain is in the soft-rate set but this objective has no
    # rate_targets gate; metric is not source_scale-linear -> stub path.
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    spec = jtfne.optax_adam(
        learning_rate=1e-3,
        differentiability_status="declared_surrogate",
        surrogate_status="declared",
    )
    result = model.tune(
        obj, optimizer=spec, steps=5, strict=False, simulation=sim, parameter="drive_gain"
    )
    assert result.summary["tuning_status"] == "optax_guarded_path_no_loop_v0.0.8"
    assert result.summary["acceptance_decision"] == "REVISE"
    assert result.best_parameters == {}


def test_generations_zero_raises():
    """Explicit generations=0 must raise ValueError naming generations."""
    model, sim = _model_and_sim()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    with pytest.raises(ValueError, match="generations"):
        model.tune(
            objectives=obj,
            optimizer="AGSDR",
            parameters={"source_scale": (0.5, 2.0)},
            generations=0,
            simulation=sim,
        )


def test_population_size_zero_raises():
    """Explicit population_size=0 must raise ValueError naming population_size."""
    model, sim = _model_and_sim()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    with pytest.raises(ValueError, match="population_size"):
        model.tune(
            objectives=obj,
            optimizer="AGSDR",
            parameters={"source_scale": (0.5, 2.0)},
            population_size=0,
            simulation=sim,
        )


def test_generations_negative_raises():
    """Explicit generations=-1 must raise ValueError naming generations."""
    model, sim = _model_and_sim()
    obj = jtfne.objective().loss("rate_loss", target=20.0, metric="spike_rate_hz_mean")
    with pytest.raises(ValueError, match="generations"):
        model.tune(
            objectives=obj,
            optimizer="AGSDR",
            parameters={"source_scale": (0.5, 2.0)},
            generations=-1,
            simulation=sim,
        )
