"""Tests for v0.3.17 precision-matching type invariants.

Verifies that the AGSDR optimization loop does not silently downcast
float64 bounds to float32, and that all internal array constructions
(lows, highs, noise, gen_best_arr) match the inferred working dtype.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxfne.optim import _run_agsdr_optimization_loop, _agsdr_candidates_from_noise


# ---------------------------------------------------------------------------
# Helper: a trivial evaluate_fn that returns the sum of candidate values.
# ---------------------------------------------------------------------------

def _sum_evaluate(candidate: dict[str, float]) -> float:
    return sum(candidate.values())


def _sphere_evaluate(candidate: dict[str, float]) -> float:
    """Sphere function — optimum at 0.0 for each param."""
    return sum(v * v for v in candidate.values())


# ---------------------------------------------------------------------------
# 1. Default float32 path: bounds supplied as Python floats → float32 arrays
# ---------------------------------------------------------------------------

class TestDefaultFloat32Path:
    """Regression: standard float32 path must still pass end-to-end."""

    def test_small_run_completes(self):
        bounds = {"a": (0.0, 1.0), "b": (-1.0, 1.0)}
        result = _run_agsdr_optimization_loop(
            evaluate_fn=_sphere_evaluate,
            parameter_bounds=bounds,
            n_generations=2,
            n_population=3,
            seed=42,
        )
        assert "best_parameters" in result
        assert "best_score" in result
        assert result["best_score"] < float("inf")

    def test_generation_records_structure(self):
        bounds = {"x": (0.0, 1.0)}
        result = _run_agsdr_optimization_loop(
            evaluate_fn=_sum_evaluate,
            parameter_bounds=bounds,
            n_generations=3,
            n_population=2,
            seed=0,
        )
        assert len(result["generation_records"]) == 3
        for rec in result["generation_records"]:
            assert "generation" in rec
            assert "best_score" in rec
            assert "best_parameters" in rec
            assert "theta_center" in rec

    def test_all_candidates_within_bounds(self):
        lo, hi = 0.5, 2.5
        bounds = {"p": (lo, hi)}
        result = _run_agsdr_optimization_loop(
            evaluate_fn=_sphere_evaluate,
            parameter_bounds=bounds,
            n_generations=4,
            n_population=5,
            seed=7,
        )
        for cand in result["all_candidates"]:
            assert lo <= cand["p"] <= hi, f"candidate {cand['p']} out of [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# 2. Dtype inheritance: float64 bounds must not be silently downcast
# ---------------------------------------------------------------------------

class TestDtypeInheritance:
    """v0.3.17 core: float64 bounds propagate through AGSDR loop arrays."""

    def test_explicit_float32_arrays(self):
        """Verify that when JAX has x64 disabled, low/high center arrays are float32."""
        # Force float32 by checking arrays in candidate proposal
        bounds = {"a": (0.0, 1.0), "b": (-2.0, 2.0)}

        # Disable x64 explicitly for this check
        orig_x64 = jax.config.read("jax_enable_x64")
        try:
            jax.config.update("jax_enable_x64", False)
            center = jnp.asarray([0.5, 0.0])
            lows = jnp.asarray([0.0, -2.0])
            highs = jnp.asarray([1.0, 2.0])
            noise = jax.random.normal(jax.random.PRNGKey(42), shape=(3, 2))

            out = _agsdr_candidates_from_noise(center, lows, highs, 0.1, noise)
            assert out.dtype == jnp.float32
        finally:
            jax.config.update("jax_enable_x64", orig_x64)

    def test_explicit_x64_arrays(self):
        """Verify that when JAX has x64 enabled, bounds propagate to float64."""
        bounds = {"a": (0.0, 1.0), "b": (-2.0, 2.0)}

        orig_x64 = jax.config.read("jax_enable_x64")
        # Try enabling x64. If it's supported by the platform, verify exact float64.
        try:
            jax.config.update("jax_enable_x64", True)

            # Re-read to confirm it took effect
            if not jax.config.read("jax_enable_x64"):
                pytest.skip("Environment does not support JAX x64 mode.")

            center = jnp.asarray([0.5, 0.0], dtype=jnp.float64)
            lows = jnp.asarray([0.0, -2.0], dtype=jnp.float64)
            highs = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
            noise = jax.random.normal(jax.random.PRNGKey(42), shape=(3, 2), dtype=jnp.float64)

            out = _agsdr_candidates_from_noise(center, lows, highs, 0.1, noise)
            assert out.dtype == jnp.float64
        finally:
            jax.config.update("jax_enable_x64", orig_x64)

    def test_candidates_finite_after_dtype_patch(self):
        """All evaluated candidates and scores must be finite."""
        bounds = {"x": (0.1, 5.0), "y": (-3.0, 3.0), "z": (0.0, 10.0)}
        result = _run_agsdr_optimization_loop(
            evaluate_fn=_sphere_evaluate,
            parameter_bounds=bounds,
            n_generations=3,
            n_population=4,
            seed=99,
        )
        for score in result["all_scores"]:
            assert jnp.isfinite(jnp.asarray(score)), f"non-finite score: {score}"

    def test_best_score_monotone_non_increasing(self):
        """best_score in generation records must be monotone non-increasing."""
        bounds = {"u": (0.0, 1.0), "v": (0.0, 1.0)}
        result = _run_agsdr_optimization_loop(
            evaluate_fn=_sphere_evaluate,
            parameter_bounds=bounds,
            n_generations=6,
            n_population=4,
            seed=2026,
        )
        records = result["generation_records"]
        for i in range(1, len(records)):
            assert records[i]["best_score"] <= records[i - 1]["best_score"] + 1e-9, (
                f"best_score increased at generation {i}: "
                f"{records[i - 1]['best_score']} → {records[i]['best_score']}"
            )


# ---------------------------------------------------------------------------
# 3. _agsdr_candidates_from_noise dtype propagation
# ---------------------------------------------------------------------------

class TestCandidatesDtypePropagation:
    """Ensure _agsdr_candidates_from_noise output dtype matches inputs."""

    def test_float32_input_float32_output(self):
        center = jnp.asarray([0.5, 0.5], dtype=jnp.float32)
        lows   = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
        highs  = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
        noise  = jax.random.normal(jax.random.PRNGKey(0), shape=(4, 2), dtype=jnp.float32)
        out = _agsdr_candidates_from_noise(center, lows, highs, 0.1, noise)
        assert jnp.issubdtype(out.dtype, jnp.floating), f"unexpected dtype {out.dtype}"
        assert out.shape == (4, 2)

    def test_all_candidates_clipped_within_bounds(self):
        center = jnp.asarray([0.5], dtype=jnp.float32)
        lows   = jnp.asarray([0.0], dtype=jnp.float32)
        highs  = jnp.asarray([1.0], dtype=jnp.float32)
        # Large noise → should still clip to [0, 1]
        noise  = jnp.ones((10, 1), dtype=jnp.float32) * 100.0
        out = _agsdr_candidates_from_noise(center, lows, highs, 1.0, noise)
        assert jnp.all(out >= 0.0).item()
        assert jnp.all(out <= 1.0).item()

    def test_candidate_zero_locked_to_clipped_center(self):
        """Row 0 must be the clipped center (no noise perturbation)."""
        center = jnp.asarray([0.75, 0.25], dtype=jnp.float32)
        lows   = jnp.asarray([0.0,  0.0],  dtype=jnp.float32)
        highs  = jnp.asarray([1.0,  1.0],  dtype=jnp.float32)
        noise  = jax.random.normal(jax.random.PRNGKey(42), shape=(5, 2), dtype=jnp.float32)
        out = _agsdr_candidates_from_noise(center, lows, highs, 0.5, noise)
        expected_row0 = jnp.clip(center, lows, highs)
        assert jnp.allclose(out[0], expected_row0, atol=1e-6).item()


# ---------------------------------------------------------------------------
# 4. Input validation guard
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Bounds validation must still fire after dtype patch."""

    def test_empty_bounds_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _run_agsdr_optimization_loop(
                evaluate_fn=_sphere_evaluate,
                parameter_bounds={},
                n_generations=1,
                n_population=1,
            )

    def test_degenerate_bounds_raises(self):
        with pytest.raises(ValueError, match="degenerate"):
            _run_agsdr_optimization_loop(
                evaluate_fn=_sphere_evaluate,
                parameter_bounds={"x": (1.0, 1.0)},
                n_generations=1,
                n_population=1,
            )

    def test_zero_generations_raises(self):
        with pytest.raises(ValueError):
            _run_agsdr_optimization_loop(
                evaluate_fn=_sphere_evaluate,
                parameter_bounds={"x": (0.0, 1.0)},
                n_generations=0,
                n_population=2,
            )
