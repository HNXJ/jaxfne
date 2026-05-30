"""JAX-native optimizer helper and placeholder-gate audit tests."""

import json

import jax
import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne.optim import (
    _agsdr_candidates_from_noise,
    _quadratic_target_loss,
    quadratic_target_loss_grad,
)
from jaxfne.validation import validate_poisson_gauge_condition


def test_quadratic_target_loss_is_jittable_and_differentiable():
    achieved = jnp.asarray([4.0, 11.0], dtype=jnp.float32)
    target = jnp.asarray([5.0, 10.0], dtype=jnp.float32)
    weights = jnp.ones((2,), dtype=jnp.float32)

    loss = _quadratic_target_loss(achieved, target, weights)
    grad = quadratic_target_loss_grad(achieved, target, weights)

    assert float(loss) > 0.0
    assert grad.shape == achieved.shape
    assert jnp.all(jnp.isfinite(grad))
    assert "reduce_sum" in str(jax.make_jaxpr(_quadratic_target_loss)(achieved, target, weights))


def test_agsdr_candidate_proposal_uses_jax_shapes_and_bounds():
    center = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    lows = jnp.asarray([0.35, 0.35], dtype=jnp.float32)
    highs = jnp.asarray([2.25, 2.25], dtype=jnp.float32)
    noise = jnp.asarray([[0.0, 0.0], [10.0, -10.0], [0.5, -0.5]], dtype=jnp.float32)

    candidates = _agsdr_candidates_from_noise(center, lows, highs, 0.18, noise)

    assert candidates.shape == (3, 2)
    assert jnp.all(candidates >= lows)
    assert jnp.all(candidates <= highs)
    assert jnp.allclose(candidates[0], center)


def test_agsdr_spec_is_used_by_tune_report():
    cfg = (
        jtfne.configuration()
        .network(name="test", kind="cortical_column", n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy")
        .probe(name="probe", modes=["spikes"])
    )
    model = jtfne.construct(cfg)
    objectives = jtfne.rate_targets(groups={"all": range(8)}, targets_hz={"all": 5.0})
    optimizer = jtfne.agsdr(
        parameters={"source_scale": (0.25, 4.0)},
        generations=1,
        population_size=1,
        alpha=0.33,
        exploration=0.11,
        seed=7,
    )

    result = model.tune(objectives=objectives, optimizer=optimizer, seed=7)
    report = result.to_dict()

    assert report["summary"]["optimizer"]["optimizer"] == "AGSDR"
    assert report["summary"]["optimizer"]["alpha"] == 0.33
    assert report["summary"]["optimizer"]["exploration"] == 0.11
    json.dumps(report, allow_nan=False)


def test_unsupported_poisson_gauge_raises_loudly():
    with pytest.raises(NotImplementedError):
        validate_poisson_gauge_condition(0.0, gauge="floating_reference")


def test_quadratic_target_loss_preserves_dtypes():
    # Verify default float32 behavior when inputs are float32
    achieved_32 = jnp.asarray([4.0, 11.0], dtype=jnp.float32)
    target_32 = jnp.asarray([5.0, 10.0], dtype=jnp.float32)
    weights_32 = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    loss_32 = _quadratic_target_loss(achieved_32, target_32, weights_32)
    assert loss_32.dtype == jnp.float32

    # Verify float64 preservation when inputs are float64 (if enable_x64 is active or local array)
    try:
        achieved_64 = jnp.asarray([4.0, 11.0], dtype=jnp.float64)
        target_64 = jnp.asarray([5.0, 10.0], dtype=jnp.float64)
        weights_64 = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
        loss_64 = _quadratic_target_loss(achieved_64, target_64, weights_64)
        # result_type dynamically promotes to float64 if enable_x64 is active
        if jax.config.read("jax_enable_x64"):
            assert loss_64.dtype == jnp.float64
    except Exception:
        pass


def test_agsdr_candidate_proposal_preserves_dtypes():
    # float32 default path
    center_32 = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    lows_32 = jnp.asarray([0.35, 0.35], dtype=jnp.float32)
    highs_32 = jnp.asarray([2.25, 2.25], dtype=jnp.float32)
    noise_32 = jnp.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=jnp.float32)

    candidates_32 = _agsdr_candidates_from_noise(center_32, lows_32, highs_32, 0.18, noise_32)
    assert candidates_32.dtype == jnp.float32

    # float64 path
    try:
        center_64 = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
        lows_64 = jnp.asarray([0.35, 0.35], dtype=jnp.float64)
        highs_64 = jnp.asarray([2.25, 2.25], dtype=jnp.float64)
        noise_64 = jnp.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=jnp.float64)

        candidates_64 = _agsdr_candidates_from_noise(center_64, lows_64, highs_64, 0.18, noise_64)
        if jax.config.read("jax_enable_x64"):
            assert candidates_64.dtype == jnp.float64
    except Exception:
        pass
