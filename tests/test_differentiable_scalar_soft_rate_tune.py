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
import pytest

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
    history = result.summary["gradient_history"]
    assert any(h["gradient"] != 0.0 for h in history), "gAMPA must backpropagate (traced value)"


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


def test_zero_learning_rate_is_refused_not_replaced_by_default():
    """learning_rate=0.0 was silently replaced by 0.01; a zero step is refused."""
    import pytest

    optimizer = jtfne.optax_adam(learning_rate=0.0, differentiability_status="declared_surrogate")
    with pytest.raises(ValueError, match="learning_rate"):
        _model().tune(
            objectives=_rate_objective(), optimizer=optimizer, simulation=_sim(),
            parameter="drive_gain", steps=2,
        )


def _edge_list_authoritative_model():
    """Small edge-list-authoritative model: placeholder emitter.W, topology in edge_list."""
    from dataclasses import replace

    import jax.numpy as jnp

    from jaxfne._model import Model
    from jaxfne.emitters import is_placeholder_dense_W

    base = _model()
    emitter = base.params["emitter"]
    assert "edge_list" in base.params
    assert int(base.params["edge_list"].n_edges) > 0
    new_emitter = replace(emitter, W=jnp.zeros((0, 0), dtype=emitter.W.dtype))
    model = Model(cfg=base.cfg, params={**base.params, "emitter": new_emitter}, static=dict(base.static))
    assert is_placeholder_dense_W(model.params["emitter"].W, model.params["emitter"].n_neurons)
    return model


def test_edge_list_synaptic_gain_refuses_differentiable_loop():
    import pytest

    model = _edge_list_authoritative_model()
    with pytest.raises(ValueError) as excinfo:
        model.tune(
            objectives=_rate_objective(), optimizer=_optax_adam(), simulation=_sim(),
            parameter="synaptic_gain", steps=5,
        )
    assert "synaptic_gain" in str(excinfo.value)
    assert "edge_list" in str(excinfo.value)


def test_edge_list_gampa_refuses_differentiable_loop():
    import pytest

    model = _edge_list_authoritative_model()
    with pytest.raises(ValueError) as excinfo:
        model.tune(
            objectives=_rate_objective(), optimizer=_optax_adam(), simulation=_sim(),
            parameter="gAMPA", steps=5,
        )
    assert "gAMPA" in str(excinfo.value)
    assert "edge_list" in str(excinfo.value)


def test_concrete_edge_list_scaling_unchanged_at_asymmetric_values():
    """Concrete (non-traced) edge-list scaling keeps working at 0.7 and 1.3.

    This fixture's edge_list uses compact scalar-magnitude storage
    (``magnitude_times_presynaptic_sign`` with one ``weight_magnitude``), so
    both parameters scale every resolved weight by the factor -- positive-only
    gAMPA selectivity needs per-edge storage. The test pins that behavior.
    """
    import numpy as np

    from jaxfne._model import _model_with_scalar_parameter
    from jaxfne.emitters import resolve_edge_weight

    model = _edge_list_authoritative_model()
    sign = np.asarray(model.params["emitter"].sign)
    el0 = model.params["edge_list"]
    assert el0.weight_magnitude is not None
    base_weights = np.asarray(resolve_edge_weight(el0, el0.weight.dtype, presynaptic_sign=sign))

    seen = {}
    for parameter in ("synaptic_gain", "gAMPA"):
        for scale in (0.7, 1.3):
            scaled = _model_with_scalar_parameter(model, parameter, scale)
            assert np.asarray(scaled.params["emitter"].W).shape == (0, 0)
            el = scaled.params["edge_list"]
            weights = np.asarray(resolve_edge_weight(el, el.weight.dtype, presynaptic_sign=sign))
            np.testing.assert_allclose(weights, base_weights * scale, rtol=1e-6, atol=0)
            seen[(parameter, scale)] = weights
    assert not np.array_equal(seen[("synaptic_gain", 0.7)], seen[("synaptic_gain", 1.3)])
    assert not np.array_equal(seen[("gAMPA", 0.7)], seen[("gAMPA", 1.3)])


@pytest.mark.parametrize("bounds", [(0.0, 0.5), (1.5, 3.0)])
def test_returned_value_stays_inside_declared_bounds(bounds):
    """The start value 1.0 was not clipped to bounds and was returned when no step improved."""
    result = _model().tune(
        objectives=_rate_objective(), optimizer=_optax_adam(), simulation=_sim(),
        parameter="drive_gain", bounds=bounds, steps=2,
    )
    assert result.summary["tuning_status"] == "differentiable_scalar_soft_rate_loop_v0.1"
    value = result.best_parameters["drive_gain"]
    assert bounds[0] <= value <= bounds[1], f"best {value} outside declared bounds {bounds}"


def test_nondefault_seed_is_refused_on_the_scalar_soft_rate_loop():
    """seed=7 against a seed-0 simulation: the loop reads no seed of its own, so it is refused."""
    with pytest.raises(ValueError, match="seed"):
        _model().tune(
            objectives=_rate_objective(), optimizer=_optax_adam(), simulation=_sim(),
            parameter="drive_gain", steps=2, seed=7,
        )


def test_seed_without_simulation_builds_the_default_simulation_from_it():
    """tune(seed=7) with no simulation: the default simulation takes seed 7, and the summary records 7."""
    result = _model().tune(
        objectives=_rate_objective(), optimizer=_optax_adam(),
        parameter="drive_gain", steps=2, seed=7,
    )
    assert result.summary["tuning_status"] == "differentiable_scalar_soft_rate_loop_v0.1"
    assert result.summary["seed"] == 7


def test_seed_that_differs_from_the_simulation_seed_is_refused():
    """simulation.seed=3 with seed=7: seed 7 would apply to nothing, so it is refused."""
    sim3 = jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=3)
    with pytest.raises(ValueError, match="seed"):
        _model().tune(
            objectives=_rate_objective(), optimizer=_optax_adam(), simulation=sim3,
            parameter="drive_gain", steps=2, seed=7,
        )


def test_zero_seed_runs_on_a_seeded_simulation_and_records_its_seed():
    """simulation.seed=3 with seed=0: the loop runs, and the summary records the seed that applied (3)."""
    sim3 = jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=3)
    result = _model().tune(
        objectives=_rate_objective(), optimizer=_optax_adam(), simulation=sim3,
        parameter="drive_gain", steps=2,
    )
    assert result.summary["tuning_status"] == "differentiable_scalar_soft_rate_loop_v0.1"
    assert result.summary["seed"] == 3
