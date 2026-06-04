"""v0.3.30 regression: objective null generators must be reproducible.

Bug (audit finding F21): the spectrolaminar null-distribution generators in
``jaxfne/objectives.py`` drew from the global ``np.random`` state, so null
distributions could not be reproduced across runs. The fix threads an explicit
``rng: np.random.Generator`` through each generator and adds a ``null_seed``
parameter to ``spectrolaminar_objective``.

These are host-side statistical transforms (not JAX-traced kernels), so a NumPy
``Generator`` is the correct reproducibility primitive — no JAX PRNG needed.

Contracts verified here:
- same seed -> identical stochastic null output
- different seed -> different stochastic null output (where math allows)
- an explicit ``np.random.default_rng(seed)`` is reproducible
- dispatcher ``null_seed`` makes the null-score sequence reproducible
- old positional calls (no rng) still work
- output shape / keys / finiteness invariants preserved
- deterministic nulls ignore rng and stay exact
"""

import numpy as np
import pytest

from jaxfne.objectives import (
    spectrolaminar_objective,
    spectrolaminar_objective_factory,
    layer_shuffle_null,
    band_label_shuffle_null,
    uniform_gain_null,
    no_field_projection_null,
    phase_randomized_null,
    source_polarity_flip_null,
)

# The four generators that consume randomness.
STOCHASTIC_NULLS = [
    layer_shuffle_null,
    uniform_gain_null,
    no_field_projection_null,
    phase_randomized_null,
]

# The two that are deterministic by construction.
DETERMINISTIC_NULLS = [
    band_label_shuffle_null,
    source_polarity_flip_null,
]

ALL_NULLS = STOCHASTIC_NULLS + DETERMINISTIC_NULLS


def _readout():
    return {
        "alpha_beta": np.array([0.2, 0.3, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02]),
        "gamma": np.array([0.05, 0.03, 0.08, 0.15, 0.2, 0.18, 0.12, 0.08]),
    }


@pytest.mark.parametrize("null_fn", STOCHASTIC_NULLS, ids=lambda f: f.__name__)
def test_same_seed_identical_output(null_fn):
    """Same seed -> byte-identical null output for stochastic generators."""
    out_a = null_fn(_readout(), rng=np.random.default_rng(7))
    out_b = null_fn(_readout(), rng=np.random.default_rng(7))
    np.testing.assert_array_equal(out_a["alpha_beta_shuffled"], out_b["alpha_beta_shuffled"])
    np.testing.assert_array_equal(out_a["gamma_shuffled"], out_b["gamma_shuffled"])


@pytest.mark.parametrize("null_fn", STOCHASTIC_NULLS, ids=lambda f: f.__name__)
def test_different_seed_changes_output(null_fn):
    """Different seed -> different output where mathematically expected."""
    out_a = null_fn(_readout(), rng=np.random.default_rng(1))
    out_b = null_fn(_readout(), rng=np.random.default_rng(999))
    differs = not (
        np.array_equal(out_a["alpha_beta_shuffled"], out_b["alpha_beta_shuffled"])
        and np.array_equal(out_a["gamma_shuffled"], out_b["gamma_shuffled"])
    )
    assert differs, f"{null_fn.__name__} produced identical output for distinct seeds"


@pytest.mark.parametrize("null_fn", DETERMINISTIC_NULLS, ids=lambda f: f.__name__)
def test_deterministic_nulls_ignore_rng(null_fn):
    """Deterministic nulls produce the same result regardless of rng."""
    base = null_fn(_readout())
    seeded = null_fn(_readout(), rng=np.random.default_rng(123))
    np.testing.assert_array_equal(base["alpha_beta_shuffled"], seeded["alpha_beta_shuffled"])
    np.testing.assert_array_equal(base["gamma_shuffled"], seeded["gamma_shuffled"])


@pytest.mark.parametrize("null_fn", ALL_NULLS, ids=lambda f: f.__name__)
def test_positional_call_still_works(null_fn):
    """Legacy call without rng must still return a valid null readout."""
    out = null_fn(_readout())
    assert set(out) >= {"alpha_beta_shuffled", "gamma_shuffled", "null_type"}


@pytest.mark.parametrize("null_fn", ALL_NULLS, ids=lambda f: f.__name__)
def test_shape_and_finite_invariants(null_fn):
    """Output shapes match input and values are finite."""
    r = _readout()
    out = null_fn(r, rng=np.random.default_rng(3))
    assert out["alpha_beta_shuffled"].shape == r["alpha_beta"].shape
    assert out["gamma_shuffled"].shape == r["gamma"].shape
    assert np.all(np.isfinite(out["alpha_beta_shuffled"]))
    assert np.all(np.isfinite(out["gamma_shuffled"]))


def test_explicit_generator_is_reproducible():
    """Passing an explicit default_rng(seed) reproduces the draw sequence."""
    g1 = np.random.default_rng(2024)
    g2 = np.random.default_rng(2024)
    a = uniform_gain_null(_readout(), rng=g1)
    b = uniform_gain_null(_readout(), rng=g2)
    np.testing.assert_array_equal(a["alpha_beta_shuffled"], b["alpha_beta_shuffled"])


def test_dispatcher_null_seed_reproducible():
    """spectrolaminar_objective(null_seed=...) -> reproducible null scores."""
    readout = _readout()
    target_ab = np.array([0.25, 0.28, 0.12, 0.08, 0.06, 0.04, 0.02, 0.01])
    target_gamma = np.array([0.08, 0.05, 0.1, 0.18, 0.22, 0.15, 0.1, 0.05])
    nulls = ["layer_shuffle", "uniform_gain", "phase_randomized"]

    rep_a = spectrolaminar_objective(
        readout, target_ab, target_gamma, nulls=nulls, null_n_samples=8, null_seed=11
    )
    rep_b = spectrolaminar_objective(
        readout, target_ab, target_gamma, nulls=nulls, null_n_samples=8, null_seed=11
    )
    # The full report must be JSON-equal for the same seed.
    import json

    assert json.dumps(rep_a, default=float, sort_keys=True) == json.dumps(
        rep_b, default=float, sort_keys=True
    )


def test_factory_forwards_null_seed():
    """The factory must forward null_seed so factory-built objectives are reproducible."""
    import json

    readout = _readout()
    target_ab = np.array([0.25, 0.28, 0.12, 0.08, 0.06, 0.04, 0.02, 0.01])
    target_gamma = np.array([0.08, 0.05, 0.1, 0.18, 0.22, 0.15, 0.1, 0.05])
    nulls = ["layer_shuffle", "uniform_gain", "phase_randomized"]

    obj = spectrolaminar_objective_factory(
        target_ab, target_gamma, nulls=nulls, null_n_samples=8, null_seed=11
    )
    rep_factory = obj(readout)
    rep_direct = spectrolaminar_objective(
        readout, target_ab, target_gamma, nulls=nulls, null_n_samples=8, null_seed=11
    )
    # Factory path must match the direct path for the same seed.
    assert json.dumps(rep_factory, default=float, sort_keys=True) == json.dumps(
        rep_direct, default=float, sort_keys=True
    )


def test_dispatcher_distinct_seeds_differ():
    """Different null_seed should change the null-derived report."""
    readout = _readout()
    target_ab = np.array([0.25, 0.28, 0.12, 0.08, 0.06, 0.04, 0.02, 0.01])
    target_gamma = np.array([0.08, 0.05, 0.1, 0.18, 0.22, 0.15, 0.1, 0.05])
    nulls = ["layer_shuffle", "uniform_gain", "phase_randomized"]

    import json

    rep_a = spectrolaminar_objective(
        readout, target_ab, target_gamma, nulls=nulls, null_n_samples=8, null_seed=1
    )
    rep_b = spectrolaminar_objective(
        readout, target_ab, target_gamma, nulls=nulls, null_n_samples=8, null_seed=2
    )
    assert json.dumps(rep_a, default=float, sort_keys=True) != json.dumps(
        rep_b, default=float, sort_keys=True
    )
