"""Tests for Patch D: spectrolaminar source projectors.

Focus: Validate dynamics-derived vs teaching/control source paths.
"""

import json
import numpy as np
import pytest
import jax.numpy as jnp

from jaxfne.fields import (
    filtered_spike_source,
    teaching_control_spectrolaminar_resonance_source,
)


class MockNeuronsDataFrame:
    """Mock neuron dataframe for testing."""

    def __init__(self, n=20, areas=None, layers=None, cell_types=None):
        self.n = n
        self.data = {
            "area": areas or (["V1"] * (n // 2) + ["PFC"] * (n - n // 2)),
            "layer": layers or (["L4"] * (n // 2) + ["L5"] * (n - n // 2)),
            "cell_type": cell_types or (["E"] * int(0.75 * n) + ["PV"] * (n - int(0.75 * n))),
        }

    def __len__(self):
        return self.n

    def get(self, key, default=None):
        return self.data.get(key, default)


# ============================================================================
# Tests: filtered_spike_source (Dynamics-Derived)
# ============================================================================


def test_filtered_spike_source_returns_finite():
    """Test that filtered_spike_source returns finite source tensor."""
    spikes = np.random.randn(500, 10).astype(np.float32)
    neurons = MockNeuronsDataFrame(n=10)

    source, metadata = filtered_spike_source(spikes, neurons, tau_ms=5.0)

    assert np.all(np.isfinite(source)), "Source must be finite"
    assert source.shape == (500, 10), "Source shape must match spikes"


def test_filtered_spike_source_metadata_dynamics_derived():
    """Test that filtered_spike_source metadata says dynamics_derived=true."""
    spikes = np.random.randn(100, 5).astype(np.float32)
    neurons = MockNeuronsDataFrame(n=5)

    _, metadata = filtered_spike_source(spikes, neurons)

    assert metadata["dynamics_derived"] is True, "Must be marked as dynamics-derived"
    assert metadata["source_mode"] == "dynamics_derived_filtered_spike_source"


def test_filtered_spike_source_metadata_no_injection():
    """Test that filtered_spike_source says spectrolaminar_profile_injected=false."""
    spikes = np.random.randn(100, 5).astype(np.float32)
    neurons = MockNeuronsDataFrame(n=5)

    _, metadata = filtered_spike_source(spikes, neurons)

    assert metadata["spectrolaminar_profile_injected"] is False
    assert metadata["default_evidence_path"] is True


def test_filtered_spike_source_metadata_json_safe():
    """Test that filtered_spike_source metadata is JSON-serializable."""
    spikes = np.random.randn(100, 5).astype(np.float32)
    neurons = MockNeuronsDataFrame(n=5)

    _, metadata = filtered_spike_source(spikes, neurons)

    try:
        json_str = json.dumps(metadata, allow_nan=False)
        assert isinstance(json_str, str)
    except (TypeError, ValueError) as e:
        pytest.fail(f"Metadata not JSON-serializable: {e}")


def test_filtered_spike_source_applies_cell_type_signs():
    """Test that filtered_spike_source applies cell-type signs (E=+1, I=-1)."""
    # Create simple spikes: positive values
    spikes = np.ones((100, 2), dtype=np.float32)
    neurons = {
        "cell_type": ["E", "PV"],
        "area": ["V1", "V1"],
        "layer": ["L4", "L4"],
    }

    source, _ = filtered_spike_source(spikes, neurons)

    # E cell should have positive source, I cell should have negative
    assert np.all(source[:, 0] > 0), "E cell source should be positive"
    assert np.all(source[:, 1] < 0), "I cell source should be negative"


# ============================================================================
# Tests: teaching_control_spectrolaminar_resonance_source
# ============================================================================


def test_teaching_control_source_returns_finite():
    """Test that teaching_control source returns finite oscillations."""
    neurons = MockNeuronsDataFrame(n=15)

    source, metadata = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=500, dt_ms=0.1
    )

    assert np.all(np.isfinite(source)), "Source must be finite"
    assert source.shape == (500, 15), "Source shape must match n_steps x n_neurons"


def test_teaching_control_source_metadata_not_dynamics_derived():
    """Test that teaching/control source says dynamics_derived=false."""
    neurons = MockNeuronsDataFrame(n=10)

    _, metadata = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=100, dt_ms=0.1
    )

    assert metadata["dynamics_derived"] is False
    assert metadata["source_mode"] == "teaching_control_resonance_source"


def test_teaching_control_source_metadata_injection_marked():
    """Test that teaching/control source says spectrolaminar_profile_injected=true."""
    neurons = MockNeuronsDataFrame(n=10)

    _, metadata = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=100, dt_ms=0.1
    )

    assert metadata["spectrolaminar_profile_injected"] is True
    assert metadata["default_evidence_path"] is False


def test_teaching_control_source_metadata_excluded_from_default():
    """Test that teaching/control source is excluded from default evidence path."""
    neurons = MockNeuronsDataFrame(n=10)

    _, metadata = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=100, dt_ms=0.1
    )

    assert metadata["default_evidence_path"] is False
    assert metadata["teaching_control_source"] is True


def test_teaching_control_source_includes_warning():
    """Test that teaching/control source includes warning text."""
    neurons = MockNeuronsDataFrame(n=10)

    _, metadata = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=100, dt_ms=0.1
    )

    assert "warning" in metadata
    assert "teaching" in metadata["warning"].lower()
    assert "evidence" in metadata["warning"].lower()


def test_teaching_control_source_metadata_json_safe():
    """Test that teaching/control source metadata is JSON-serializable."""
    neurons = MockNeuronsDataFrame(n=10)

    _, metadata = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=100, dt_ms=0.1
    )

    try:
        json_str = json.dumps(metadata, allow_nan=False)
        assert isinstance(json_str, str)
    except (TypeError, ValueError) as e:
        pytest.fail(f"Metadata not JSON-serializable: {e}")


def test_teaching_control_source_hard_coded_frequencies():
    """Test that teaching/control source uses hard-coded frequencies."""
    neurons = MockNeuronsDataFrame(n=5)

    _, metadata = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=100, dt_ms=0.1
    )

    # Check that hard-coded frequencies are in metadata
    assert metadata["alpha_beta_freq_hz"] == 15.0
    assert metadata["gamma_freq_hz"] == 90.0


# ============================================================================
# Tests: Quarantine Gates (Exclusion from Default Evidence Path)
# ============================================================================


def test_teaching_control_source_excluded_from_default_objective():
    """Test that teaching/control source must be explicitly allowed to use in objectives."""
    neurons = MockNeuronsDataFrame(n=10)

    _, metadata = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=100, dt_ms=0.1
    )

    # This metadata should prevent default objective path from accepting it
    assert metadata["spectrolaminar_profile_injected"] is True
    assert metadata["default_evidence_path"] is False
    # Objective code should check these flags before using


def test_teaching_control_source_empty_population():
    """Test that teaching/control source handles empty neuron population gracefully."""
    neurons = {"area": [], "layer": [], "cell_type": []}

    source, metadata = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=100, dt_ms=0.1
    )

    assert source.shape[0] == 100
    assert source.shape[1] == 1  # Single dummy neuron
    assert "warning" in metadata
    assert "empty" in metadata["warning"].lower()


def test_source_path_distinction():
    """Test that we can distinguish dynamics-derived from teaching/control paths."""
    spikes = np.random.randn(100, 5).astype(np.float32)
    neurons = MockNeuronsDataFrame(n=5)

    # Dynamics-derived path
    source_dyn, meta_dyn = filtered_spike_source(spikes, neurons)
    assert meta_dyn["dynamics_derived"] is True
    assert meta_dyn["default_evidence_path"] is True

    # Teaching/control path
    source_teach, meta_teach = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps=100, dt_ms=0.1
    )
    assert meta_teach["dynamics_derived"] is False
    assert meta_teach["default_evidence_path"] is False

    # Both should be finite, but metadata differs
    assert np.all(np.isfinite(source_dyn))
    assert np.all(np.isfinite(source_teach))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
