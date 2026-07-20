"""Regression tests for the real jax.grad-based differentiable tune loop over
scalar parameters that DO cross the spike/reset boundary (drive_gain,
synaptic_gain, gAMPA) -- jaxfne/optim/core.py::_tune_scalar_soft_rate_optax.

Unlike _tune_source_scale_optax (a pure post-processing linear rescale),
these parameters feed into the recurrent Izhikevich dynamics before
simulate() runs. No change was made to jaxfne/emitters.py's step kernels --
JAX already differentiates through Model.simulate()'s hard jnp.where-based
spike-reset without a custom surrogate gradient on the kernel itself
(verified empirically: real nonzero gradients at the existing, unmodified
step dynamics). The soft-rate surrogate lives only at the loss level
(_evaluate_soft_rate_targets, reused from the already-tested AGSDR two-level
inner loop, not reinvented).
"""
import jaxfne as jtfne


def _model():
    cfg = (
        jtfne.configuration()
        .network(n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _sim():
    return jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=0)


def _rate_objective():
    return jtfne.rate_targets(groups={"all": list(range(20))}, targets_hz={"all": 15.0})


def _optax_adam():
    return jtfne.optax_adam(learning_rate=0.05, differentiability_status="declared_surrogate")


def test_drive_gain_real_gradient_descent_reduces_loss():
    model = _model()
    result = model.tune(
        objectives=_rate_objective(), optimizer=_optax_adam(), simulation=_sim(),
        parameter="drive_gain", steps=15,
    )
    assert result.summary["tuning_status"] == "differentiable_scalar_soft_rate_loop_v0.1"
    assert result.best_parameters, "best_parameters must not be empty (that's the old stub's behavior)"
    history = result.summary["gradient_history"]
    assert len(history) == 15
    assert any(h["gradient"] != 0.0 for h in history), "gradient must be nonzero -- confirms real backprop through spike/reset"
    assert history[-1]["loss"] <= history[0]["loss"]


def test_synaptic_gain_real_gradient_descent_works():
    model = _model()
    result = model.tune(
        objectives=_rate_objective(), optimizer=_optax_adam(), simulation=_sim(),
        parameter="synaptic_gain", steps=10,
    )
    assert result.summary["tuning_status"] == "differentiable_scalar_soft_rate_loop_v0.1"
    assert result.best_parameters


def test_gampa_real_gradient_descent_works():
    model = _model()
    result = model.tune(
        objectives=_rate_objective(), optimizer=_optax_adam(), simulation=_sim(),
        parameter="gAMPA", steps=10,
    )
    assert result.summary["tuning_status"] == "differentiable_scalar_soft_rate_loop_v0.1"
    assert result.best_parameters


def test_falls_back_to_stub_for_non_rate_targets_objective():
    model = _model()
    plain_objective = jtfne.objective().loss("hit_target", metric="mean_V_m", target=-50.0)
    result = model.tune(
        objectives=plain_objective, optimizer=_optax_adam(), simulation=_sim(),
        parameter="drive_gain", steps=5,
    )
    assert result.summary["tuning_status"] == "optax_guarded_path_no_loop_v0.0.8"
    assert result.best_parameters == {}


def test_falls_back_to_stub_for_unsupported_parameter():
    model = _model()
    result = model.tune(
        objectives=_rate_objective(), optimizer=_optax_adam(), simulation=_sim(),
        parameter="drive_scale_a", steps=5,
    )
    assert result.summary["tuning_status"] == "optax_guarded_path_no_loop_v0.0.8"


def test_gampa_scalar_parameter_edit_is_behaviorally_identical_to_prior_numpy_implementation():
    """Regression: gAMPA's implementation moved from numpy boolean-indexed
    assignment to jnp.where (to stay jax-traceable) -- must be bit-identical
    for concrete inputs."""
    import numpy as np
    from jaxfne._model import _model_with_scalar_parameter

    model = _model()
    W = model.params["emitter"].W
    value = 1.7
    W_native = np.asarray(W, dtype=W.dtype)
    expected = W_native.copy()
    expected[W_native > 0] = W_native[W_native > 0] * np.asarray(value, dtype=W.dtype)

    new_model = _model_with_scalar_parameter(model, "gAMPA", value)
    actual = np.asarray(new_model.params["emitter"].W)
    assert np.array_equal(expected, actual)
