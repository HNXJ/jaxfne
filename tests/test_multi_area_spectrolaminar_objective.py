"""Tests for spectrolaminar objective and scoring (Patch F)."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.fields import (
    spectrolaminar_similarity,
    spectrolaminar_objective,
)


def create_mock_readout(
    n_contacts: int = 5,
    area: str = "V1",
    alpha_beta_vals: "np.ndarray | None" = None,
    gamma_vals: "np.ndarray | None" = None,
) -> dict:
    """Create a mock spectrolaminar readout for testing."""
    if alpha_beta_vals is None:
        alpha_beta_vals = np.linspace(0.7, 0.3, n_contacts)
    if gamma_vals is None:
        gamma_vals = np.linspace(0.2, 0.6, n_contacts)

    return {
        "freq_hz": np.linspace(1.0, 150.0, 128),
        "pos_from_l4": np.linspace(-3, 3, n_contacts),
        "relative_power": np.random.randn(128, n_contacts).astype(np.float32),
        "alpha_beta": alpha_beta_vals,
        "gamma": gamma_vals,
        "contact_depths_m": np.linspace(0, 0.5, n_contacts),
        "n_contacts": n_contacts,
        "n_neurons": n_contacts * 10,
        "area": area,
    }


def test_spectrolaminar_similarity_perfect_match():
    """Test that identical profiles score high."""
    readout = create_mock_readout()
    target_alpha = readout["alpha_beta"]
    target_gamma = readout["gamma"]

    score = spectrolaminar_similarity(readout, target_alpha, target_gamma)

    # Perfect match should score high (>80)
    assert score > 80.0, f"Perfect match scored {score}, expected >80"


def test_spectrolaminar_similarity_no_targets():
    """Test that scoring works without target constraints."""
    readout = create_mock_readout()

    score = spectrolaminar_similarity(readout)

    # Should return a reasonable baseline score
    assert 0.0 <= score <= 100.0, f"Score {score} out of bounds"


def test_spectrolaminar_similarity_bad_match():
    """Test that dissimilar profiles score lower."""
    readout = create_mock_readout()
    target_alpha = np.ones(5) * 0.0  # Opposite of actual
    target_gamma = np.ones(5) * 1.0  # Opposite of actual

    score = spectrolaminar_similarity(readout, target_alpha, target_gamma)

    # Bad match should score lower
    assert score < 50.0, f"Bad match scored {score}, expected <50"


def test_spectrolaminar_similarity_bounds():
    """Test that score is always bounded [0, 100]."""
    readout = create_mock_readout()
    target_alpha = np.random.rand(5)
    target_gamma = np.random.rand(5)

    for _ in range(10):
        score = spectrolaminar_similarity(
            readout,
            target_alpha + np.random.randn(5) * 2.0,  # Random targets
            target_gamma + np.random.randn(5) * 2.0,
        )
        assert 0.0 <= score <= 100.0, f"Score {score} out of bounds"


def test_spectrolaminar_similarity_anticorrelation_bonus():
    """Test that anticorrelated profiles get bonus."""
    # Create profile with inverse relationship
    alpha_beta = np.array([0.8, 0.6, 0.4, 0.2, 0.1], dtype=np.float32)
    gamma = np.array([0.1, 0.2, 0.4, 0.6, 0.8], dtype=np.float32)

    readout = create_mock_readout(alpha_beta_vals=alpha_beta, gamma_vals=gamma)
    score = spectrolaminar_similarity(readout)

    # Anticorrelated profiles should score reasonably well
    assert score > 40.0, f"Anticorrelated profile scored {score}, expected >40"


def test_spectrolaminar_objective_single_area():
    """Test objective for single area."""
    readout = create_mock_readout(area="V1")
    target = {
        "alpha_beta": readout["alpha_beta"],
        "gamma": readout["gamma"],
    }

    obj = spectrolaminar_objective(target_profiles={"V1": target})
    score = obj.score({"V1": readout})

    # Perfect match should score high
    assert score > 80.0, f"Single area perfect match scored {score}, expected >80"


def test_spectrolaminar_objective_multi_area():
    """Test objective for multiple areas."""
    readout_v1 = create_mock_readout(area="V1")
    readout_pfc = create_mock_readout(area="PFC")

    target_v1 = {
        "alpha_beta": readout_v1["alpha_beta"],
        "gamma": readout_v1["gamma"],
    }
    target_pfc = {
        "alpha_beta": readout_pfc["alpha_beta"],
        "gamma": readout_pfc["gamma"],
    }

    obj = spectrolaminar_objective(
        target_profiles={"V1": target_v1, "PFC": target_pfc}
    )
    score = obj.score({"V1": readout_v1, "PFC": readout_pfc})

    # Perfect match should score high
    assert score > 80.0, f"Multi-area perfect match scored {score}, expected >80"


def test_spectrolaminar_objective_missing_areas():
    """Test that objective handles missing areas gracefully."""
    readout_v1 = create_mock_readout(area="V1")

    target_v1 = {
        "alpha_beta": readout_v1["alpha_beta"],
        "gamma": readout_v1["gamma"],
    }

    # Objective expects V1 and PFC, but only gets V1
    obj = spectrolaminar_objective(
        target_profiles={"V1": target_v1, "PFC": {"alpha_beta": None, "gamma": None}}
    )

    # Should handle gracefully (only score V1)
    score = obj.score({"V1": readout_v1})
    assert 0.0 <= score <= 100.0


def test_spectrolaminar_objective_no_targets():
    """Test objective with no predefined targets."""
    readout = create_mock_readout()

    obj = spectrolaminar_objective(target_profiles=None)
    score = obj.score({"V1": readout})

    # Should return baseline score even without targets
    assert 0.0 <= score <= 100.0


def test_spectrolaminar_objective_empty_readout():
    """Test objective with empty readout."""
    empty_readout = {
        "freq_hz": np.array([]),
        "pos_from_l4": np.array([]),
        "relative_power": np.zeros((128, 0)),
        "alpha_beta": np.array([]),
        "gamma": np.array([]),
        "contact_depths_m": np.array([]),
        "n_contacts": 0,
        "n_neurons": 0,
        "area": "V1",
    }

    obj = spectrolaminar_objective()
    score = obj.score({"V1": empty_readout})

    # Should handle empty gracefully
    assert 0.0 <= score <= 100.0


def test_spectrolaminar_similarity_different_sizes():
    """Test similarity handling readouts of different sizes."""
    readout_small = create_mock_readout(n_contacts=3)
    readout_large = create_mock_readout(n_contacts=5)

    # Should handle size mismatch gracefully
    score_small = spectrolaminar_similarity(readout_small)
    score_large = spectrolaminar_similarity(readout_large)

    assert 0.0 <= score_small <= 100.0
    assert 0.0 <= score_large <= 100.0


def test_spectrolaminar_objective_reproducibility():
    """Test that objective scoring is reproducible."""
    readout = create_mock_readout()
    target = {
        "alpha_beta": readout["alpha_beta"],
        "gamma": readout["gamma"],
    }

    obj = spectrolaminar_objective(target_profiles={"V1": target})

    score1 = obj.score({"V1": readout})
    score2 = obj.score({"V1": readout})

    assert score1 == score2, "Scores should be reproducible"


def test_spectrolaminar_objective_partial_targets():
    """Test objective with only one band as target."""
    readout = create_mock_readout()

    # Only provide alpha_beta target
    target = {"alpha_beta": readout["alpha_beta"], "gamma": None}

    obj = spectrolaminar_objective(target_profiles={"V1": target})
    score = obj.score({"V1": readout})

    assert 0.0 <= score <= 100.0, "Should handle partial targets"


def test_spectrolaminar_objective_inverse_profiles():
    """Test scoring of inverse profiles."""
    alpha_beta = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    gamma = np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)

    readout = create_mock_readout(alpha_beta_vals=alpha_beta, gamma_vals=gamma)

    # Target is opposite (good anticorrelation)
    target_alpha = np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    target_gamma = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    score_inverse = spectrolaminar_similarity(readout, target_alpha, target_gamma)

    # Inverse profiles (bad match) should score lower
    score_direct = spectrolaminar_similarity(readout, alpha_beta, gamma)

    assert score_direct > score_inverse, "Direct match should score better than inverse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
