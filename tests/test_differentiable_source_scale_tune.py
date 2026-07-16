"""Regression tests for the real jax.grad-based source_scale differentiable
tune loop (jaxfne/optim/core.py::_tune_source_scale_optax).

Scope: source_proxy = source_scale * (current_native + GAIN * spikes) is a
linear post-processing rescale of already-simulated arrays -- genuinely
differentiable without any surrogate, since it never crosses the spike/reset
boundary. This is deliberately narrower than a general differentiable-tune
path (see jaxfne/_model.py::Model.tune()'s eligibility check) -- only a
single loss on a metric linear in source_scale is supported; everything else
still returns the pre-existing metadata-only stub.
"""
import jaxfne as jtfne
from jaxfne._model import _model_with_scalar_parameter


def _model():
    cfg = (
        jtfne.configuration()
        .network(n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def test_source_scale_real_gradient_descent_reduces_loss_and_hits_target():
    model = _model()
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=0)

    baseline = model.simulate(sim)
    baseline_val = float(abs(baseline.field.source_proxy).mean())
    target = baseline_val * 2.0

    objective = jtfne.objective().loss("hit_target", metric="source_proxy_abs_mean", target=target)
    optimizer = jtfne.optax_adam(learning_rate=0.05, differentiability_status="declared_surrogate")
    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim, steps=25)

    assert result.summary["tuning_status"] == "differentiable_source_scale_loop_v0.1"
    assert result.best_parameters, "best_parameters must not be empty (that's the old stub's behavior)"
    history = result.summary["gradient_history"]
    assert len(history) == 25
    assert any(h["gradient"] != 0.0 for h in history), "gradient must be nonzero -- confirms real backprop"
    assert history[-1]["loss"] < history[0]["loss"], "loss must decrease under real gradient descent"

    final_scale = result.best_parameters["source_scale"]
    tuned_model = _model_with_scalar_parameter(model, "source_scale", final_scale)
    tuned_val = float(abs(tuned_model.simulate(sim).field.source_proxy).mean())
    assert abs(tuned_val - target) < abs(baseline_val - target)


def test_source_scale_loop_falls_back_to_stub_for_unsupported_metric():
    model = _model()
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=0)
    objective = jtfne.objective().loss("hit_target", metric="mean_V_m", target=-50.0)
    optimizer = jtfne.optax_adam(learning_rate=0.05, differentiability_status="declared_surrogate")
    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim, steps=5)
    assert result.summary["tuning_status"] == "optax_guarded_path_no_loop_v0.0.8"
    assert result.best_parameters == {}


def test_source_scale_loop_falls_back_to_stub_for_multiple_losses():
    model = _model()
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=0)
    objective = (
        jtfne.objective()
        .loss("a", metric="source_proxy_abs_mean", target=1.0)
        .loss("b", metric="mean_V_m", target=-50.0)
    )
    optimizer = jtfne.optax_adam(learning_rate=0.05, differentiability_status="declared_surrogate")
    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim, steps=5)
    assert result.summary["tuning_status"] == "optax_guarded_path_no_loop_v0.0.8"
