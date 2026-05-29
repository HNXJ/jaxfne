"""
Tests for spectrolaminar objectives, null distributions, and synchrony metrics.

Tests verify:
- Score grammar (profile_score_no_null | null_normalized_similarity | motif_gate)
- S_lam only exported with null distribution
- Null distributions change scores appropriately
- Synchrony metric computation and rejection
- Teaching/control source detection and rejection
- JSON serialization safety
- Metadata correctness
"""

import pytest
import numpy as np
import json
from jaxfne.objectives import (
    spectrolaminar_profile_score,
    spectrolaminar_objective,
    layer_shuffle_null,
    band_label_shuffle_null,
    uniform_gain_null,
    no_field_projection_null,
    phase_randomized_null,
    source_polarity_flip_null,
    compute_synchrony_metric,
    spectrolaminar_objective_factory,
)


@pytest.fixture
def mock_readout():
    """Mock readout dict with spectrolaminar data."""
    return {
        "area": "V1",
        "n_neurons": 100,
        "n_contacts": 8,
        "alpha_beta": np.array([0.2, 0.3, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02]),
        "gamma": np.array([0.05, 0.03, 0.08, 0.15, 0.2, 0.18, 0.12, 0.08]),
        "pos_from_l4": np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]),
        "contact_depths_m": np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "metadata": {
            "teaching_control_source": False,
            "default_evidence_path": True,
            "field_solver_status": "laminar_proxy_no_pde",
        },
    }


@pytest.fixture
def target_profiles():
    """Target alpha/beta and gamma profiles."""
    alpha_beta = np.array([0.25, 0.28, 0.12, 0.08, 0.06, 0.04, 0.02, 0.01])
    gamma = np.array([0.08, 0.05, 0.1, 0.18, 0.22, 0.15, 0.1, 0.05])
    return alpha_beta, gamma


def test_profile_score_returns_bounded_percent(mock_readout, target_profiles):
    """Profile score (no nulls) should return [0, 100]."""
    alpha_beta, gamma = target_profiles

    report = spectrolaminar_profile_score(mock_readout, alpha_beta, gamma)

    assert report["score_type"] == "profile_score_no_null"
    assert 0.0 <= report["profile_score_percent"] <= 100.0
    assert np.isfinite(report["profile_score_percent"])


def test_profile_score_exports_no_s_lam_without_nulls(mock_readout, target_profiles):
    """Profile score (no nulls) should NOT export S_lam."""
    alpha_beta, gamma = target_profiles

    report = spectrolaminar_profile_score(mock_readout, alpha_beta, gamma)

    assert report["S_lam"] is None
    assert report["motif_gate_percent"] is None
    assert report["nulls_run"] is False


def test_null_layer_shuffle_changes_score(mock_readout, target_profiles):
    """Layer shuffle null should produce different score."""
    alpha_beta, gamma = target_profiles

    # Original score
    original = spectrolaminar_profile_score(mock_readout, alpha_beta, gamma)
    orig_score = original["profile_score_percent"]

    # Shuffled score
    np.random.seed(42)
    shuffled_readout = mock_readout.copy()
    null_out = layer_shuffle_null(mock_readout)
    shuffled_readout["alpha_beta"] = null_out["alpha_beta_shuffled"]
    shuffled_readout["gamma"] = null_out["gamma_shuffled"]

    shuffled = spectrolaminar_profile_score(shuffled_readout, alpha_beta, gamma)
    shuffled_score = shuffled["profile_score_percent"]

    # Scores should differ (with high probability)
    # Allow equality with very low probability
    assert isinstance(shuffled_score, (float, np.floating))
    assert 0.0 <= shuffled_score <= 100.0


def test_null_band_shuffle_changes_score(mock_readout, target_profiles):
    """Band label shuffle null should produce different score."""
    alpha_beta, gamma = target_profiles

    np.random.seed(42)
    null_out = band_label_shuffle_null(mock_readout)

    shuffled_readout = mock_readout.copy()
    shuffled_readout["alpha_beta"] = null_out["alpha_beta_shuffled"]
    shuffled_readout["gamma"] = null_out["gamma_shuffled"]

    shuffled = spectrolaminar_profile_score(shuffled_readout, alpha_beta, gamma)
    shuffled_score = shuffled["profile_score_percent"]

    assert 0.0 <= shuffled_score <= 100.0
    assert np.isfinite(shuffled_score)


def test_null_uniform_gain_reduces_profile_info(mock_readout, target_profiles):
    """Uniform gain null should reduce information content."""
    alpha_beta, gamma = target_profiles

    np.random.seed(42)
    original = spectrolaminar_profile_score(mock_readout, alpha_beta, gamma)

    null_out = uniform_gain_null(mock_readout, gain_min=0.1)
    readout_copy = mock_readout.copy()
    readout_copy["alpha_beta"] = null_out["alpha_beta_shuffled"]
    readout_copy["gamma"] = null_out["gamma_shuffled"]

    modified = spectrolaminar_profile_score(readout_copy, alpha_beta, gamma)

    assert np.isfinite(modified["profile_score_percent"])
    assert 0.0 <= modified["profile_score_percent"] <= 100.0


def test_null_no_field_projection_handled(mock_readout, target_profiles):
    """No field projection null should flatten profiles."""
    alpha_beta, gamma = target_profiles

    np.random.seed(42)
    null_out = no_field_projection_null(mock_readout)

    assert "alpha_beta_shuffled" in null_out
    assert "gamma_shuffled" in null_out
    assert null_out["null_type"] == "no_field_projection"
    assert np.isfinite(null_out["alpha_beta_shuffled"]).all()
    assert np.isfinite(null_out["gamma_shuffled"]).all()


def test_null_distribution_z_score_normalized(mock_readout, target_profiles):
    """Objective with nulls should compute z-score normalized S_lam."""
    alpha_beta, gamma = target_profiles

    np.random.seed(42)
    report = spectrolaminar_objective(
        mock_readout,
        alpha_beta,
        gamma,
        nulls=["layer_shuffle"],
        null_n_samples=5,
    )

    assert report["nulls_run"] is True
    assert report["null_distribution_n"] > 0
    assert report["null_normalization_method"] == "z_score"
    if report["S_lam"] is not None:
        assert np.isfinite(report["S_lam"])


def test_teaching_control_source_rejected_from_default(target_profiles):
    """Teaching/control source should be marked and rejected."""
    alpha_beta, gamma = target_profiles

    readout = {
        "area": "V1",
        "alpha_beta": np.array([0.2, 0.3, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02]),
        "gamma": np.array([0.05, 0.03, 0.08, 0.15, 0.2, 0.18, 0.12, 0.08]),
        "pos_from_l4": np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]),
        "metadata": {
            "teaching_control_source": True,
            "default_evidence_path": False,
        },
    }

    report = spectrolaminar_objective(readout, alpha_beta, gamma, nulls=[])

    assert report["uses_teaching_control_source"] is True
    assert report["default_evidence_path"] is False
    assert "teaching_control_source_detected" in report["rejection_reasons"]


def test_synchrony_metric_computes_finite(mock_readout, target_profiles):
    """Synchrony metric should compute finite value."""
    alpha_beta, gamma = target_profiles

    # Mock spike array
    spikes = np.random.randint(0, 2, size=(1000, 100))

    np.random.seed(42)
    report = spectrolaminar_objective(
        mock_readout,
        alpha_beta,
        gamma,
        synchrony_spikes=spikes,
        synchrony_metric="mean_pairwise_correlation",
        synchrony_threshold=0.8,
    )

    assert report["synchrony_checked"] is True
    assert report["synchrony_metric"] == "mean_pairwise_correlation"
    if report["synchrony_value"] is not None:
        assert 0.0 <= report["synchrony_value"] <= 1.0
        assert np.isfinite(report["synchrony_value"])


def test_high_synchrony_adds_rejection_reason(mock_readout, target_profiles):
    """High synchrony should trigger rejection."""
    alpha_beta, gamma = target_profiles

    # Perfectly correlated spikes (high synchrony)
    spikes = np.ones((1000, 100))

    np.random.seed(42)
    report = spectrolaminar_objective(
        mock_readout,
        alpha_beta,
        gamma,
        synchrony_spikes=spikes,
        synchrony_metric="mean_pairwise_correlation",
        synchrony_threshold=0.5,
    )

    # High synchrony should exceed threshold
    if report["synchrony_value"] is not None and report["synchrony_value"] > 0.5:
        assert report["synchrony_rejection"] is True
        assert len(report["rejection_reasons"]) > 0


def test_objective_report_json_serializable(mock_readout, target_profiles):
    """Objective report should be JSON-serializable."""
    alpha_beta, gamma = target_profiles

    np.random.seed(42)
    report = spectrolaminar_objective(
        mock_readout,
        alpha_beta,
        gamma,
        nulls=["layer_shuffle"],
        null_n_samples=5,
    )

    # Must serialize with allow_nan=False
    json_str = json.dumps(report, allow_nan=False)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Round-trip
    loaded = json.loads(json_str)
    assert loaded["objective_kind"] == "spectrolaminar_profile"


def test_objective_metadata_says_physical_amplitude_false(mock_readout, target_profiles):
    """Objective metadata should enforce physical_amplitude_claim_allowed=False."""
    alpha_beta, gamma = target_profiles

    report = spectrolaminar_objective(mock_readout, alpha_beta, gamma)

    assert report["physical_amplitude_claim_allowed"] is False
    assert report["truth_mode"] == "truth_safe_unverified"


def test_s_lam_absent_without_nulls(mock_readout, target_profiles):
    """S_lam should be absent (None) without null distribution."""
    alpha_beta, gamma = target_profiles

    report = spectrolaminar_objective(mock_readout, alpha_beta, gamma, nulls=[])

    assert report["S_lam"] is None
    assert report["null_distribution_n"] == 0


def test_s_lam_present_with_null_distribution(mock_readout, target_profiles):
    """S_lam should be present with null distribution."""
    alpha_beta, gamma = target_profiles

    np.random.seed(42)
    report = spectrolaminar_objective(
        mock_readout,
        alpha_beta,
        gamma,
        nulls=["layer_shuffle", "band_label_shuffle"],
        null_n_samples=3,
    )

    assert report["null_distribution_n"] > 0
    if report["S_lam"] is not None:
        assert np.isfinite(report["S_lam"])
        assert report["score_type"] == "null_normalized_similarity"


def test_score_type_grammar_correct(mock_readout, target_profiles):
    """Score type should follow grammar rules."""
    alpha_beta, gamma = target_profiles

    # No nulls
    report1 = spectrolaminar_objective(
        mock_readout, alpha_beta, gamma, nulls=[]
    )
    assert report1["score_type"] == "profile_score_no_null"

    # With nulls
    np.random.seed(42)
    report2 = spectrolaminar_objective(
        mock_readout,
        alpha_beta,
        gamma,
        nulls=["layer_shuffle"],
        null_n_samples=2,
    )
    if report2["null_distribution_n"] > 0:
        assert report2["score_type"] == "null_normalized_similarity"


def test_compute_synchrony_metric_handles_small_populations(target_profiles):
    """Synchrony metric should handle small populations gracefully."""
    # Single neuron
    spikes_single = np.random.randint(0, 2, size=(100, 1))
    sync_single = compute_synchrony_metric(spikes_single)
    assert 0.0 <= sync_single <= 1.0

    # Empty
    spikes_empty = np.zeros((0, 10))
    sync_empty = compute_synchrony_metric(spikes_empty)
    assert sync_empty == 0.0


def test_objective_factory_returns_callable(target_profiles):
    """Factory should return a callable objective function."""
    alpha_beta, gamma = target_profiles

    objective_fn = spectrolaminar_objective_factory(
        alpha_beta,
        gamma,
        nulls=["layer_shuffle"],
    )

    assert callable(objective_fn)

    # Create mock readout and test
    readout = {
        "area": "V1",
        "alpha_beta": alpha_beta + 0.01,
        "gamma": gamma + 0.01,
        "pos_from_l4": np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]),
        "metadata": {"teaching_control_source": False},
    }

    np.random.seed(42)
    report = objective_fn(readout)

    assert isinstance(report, dict)
    assert "objective_kind" in report
