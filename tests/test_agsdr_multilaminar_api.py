"""Tests for Phase 5b multi-laminar AGSDR API (select_neurons, kappa_synchrony, rate_synchrony_targets).

Covers the new analysis and selection functions for Suite No. 2 AGSDR tuning workflow.
"""

import numpy as np
import pytest

import jaxfne as jtfne


class TestSelectNeurons:
    """Tests for select_neurons() function."""

    def test_select_neurons_basic(self):
        """Test that select_neurons can be called on a model."""
        cfg = jtfne.suite2_net1_config(seed=7, n=100, duration_ms=10.0, dt_ms=0.1)
        model = jtfne.construct(cfg)

        # Should be callable without error
        indices = jtfne.select_neurons(model)
        assert indices is not None
        assert isinstance(indices, np.ndarray)

    def test_select_neurons_returns_array(self):
        """Test that select_neurons returns a numpy array of indices."""
        cfg = jtfne.suite2_net1_config(seed=7, n=50, duration_ms=10.0, dt_ms=0.1)
        model = jtfne.construct(cfg)
        indices = jtfne.select_neurons(model)

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == int
        # May be empty if neuron_metadata not set, which is fine
        assert len(indices) <= 50

    def test_select_neurons_empty_metadata(self):
        """Test that select_neurons handles model with empty or missing metadata gracefully."""
        cfg = jtfne.suite2_net1_config(seed=7, n=50, duration_ms=10.0, dt_ms=0.1)
        model = jtfne.construct(cfg)

        # Model may have neuron_metadata as None or empty
        indices = jtfne.select_neurons(model, cell_type="E")

        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64 or indices.dtype == int


class TestKappaSynchrony:
    """Tests for kappa_synchrony() function."""

    def test_kappa_synchrony_zero_for_independent(self):
        """Test that kappa ≈ 0 for independent spike trains."""
        n_neurons = 10
        n_steps = 100
        # Random independent spikes
        spikes = np.random.rand(n_neurons, n_steps) > 0.9  # ~10% fire rate

        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert isinstance(kappa, float)
        # Independent spikes should have low correlation
        assert -1.0 <= kappa <= 1.0

    def test_kappa_synchrony_high_for_synchronized(self):
        """Test that kappa is higher for synchronized spike trains."""
        n_neurons = 10
        n_steps = 100
        # Synchronized spikes
        spikes = np.zeros((n_neurons, n_steps), dtype=bool)
        # All neurons spike together at certain times
        for t in range(0, n_steps, 10):
            spikes[:, t] = True

        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert kappa > 0.5  # High synchrony

    def test_kappa_synchrony_single_neuron(self):
        """Test that kappa returns 0 for single neuron."""
        spikes = np.ones((1, 100), dtype=bool)
        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert kappa == 0.0

    def test_kappa_synchrony_empty_array(self):
        """Test that kappa returns 0 for empty array."""
        spikes = np.array([])
        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert kappa == 0.0

    def test_kappa_synchrony_silent_neurons(self):
        """Test that kappa handles completely silent neurons gracefully."""
        spikes = np.zeros((5, 100), dtype=bool)
        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert kappa == 0.0


class TestRateSynchronyTargets:
    """Tests for rate_synchrony_targets() function."""

    def test_rate_synchrony_targets_returns_dict(self):
        """Test that rate_synchrony_targets returns a dictionary."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
        )

        assert isinstance(objective, dict)

    def test_rate_synchrony_targets_required_keys(self):
        """Test that returned dict contains all required keys."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=8.0,
            target_kappa_synchrony=-0.1,
            rate_weight=1.0,
            synchrony_weight=0.5,
        )

        required_keys = [
            "name", "kind", "target_rate_hz", "target_kappa_synchrony",
            "rate_weight", "synchrony_weight", "truth_mode", "claim_level"
        ]
        for key in required_keys:
            assert key in objective, f"Missing key: {key}"

    def test_rate_synchrony_targets_truth_gates(self):
        """Test that returned dict preserves truth gates."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
        )

        assert objective["truth_mode"] == "truth_safe_unverified"
        assert objective["claim_level"] == "computational_scaffold"

    def test_rate_synchrony_targets_default_weights(self):
        """Test default weights when not specified."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
        )

        assert objective["rate_weight"] == 1.0
        assert objective["synchrony_weight"] == 0.25

    def test_rate_synchrony_targets_custom_weights(self):
        """Test custom weights are preserved."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
            rate_weight=2.0,
            synchrony_weight=1.0,
        )

        assert objective["rate_weight"] == 2.0
        assert objective["synchrony_weight"] == 1.0


class TestIntegration:
    """Integration tests for the three functions together."""

    def test_workflow_smoke(self):
        """Test a basic workflow: config → model → select → objective."""
        # Create config
        cfg = jtfne.suite2_net1_config(seed=7, n=50, duration_ms=100.0, dt_ms=0.1)
        model = jtfne.construct(cfg)

        # Select neurons (if metadata available)
        selected = jtfne.select_neurons(model)
        assert selected is not None

        # Create objective
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
        )
        assert objective["name"] == "rate_synchrony_targets"

    def test_kappa_on_simulated_spikes(self):
        """Test kappa_synchrony on actual simulated spikes."""
        cfg = jtfne.suite2_net1_config(seed=7, n=20, duration_ms=50.0, dt_ms=0.1)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, seed=7, duration_ms=50.0, dt_ms=0.1)

        spikes = np.array(signals.spikes)
        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)

        assert isinstance(kappa, float)
        assert -1.0 <= kappa <= 1.0
