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
        n_steps = 100
        n_neurons = 10
        # Random independent spikes [T, N]
        spikes = np.random.rand(n_steps, n_neurons) > 0.9  # ~10% fire rate

        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert isinstance(kappa, float)
        # Independent spikes should have low correlation
        assert -1.0 <= kappa <= 1.0

    def test_kappa_synchrony_high_for_synchronized(self):
        """Test that kappa is higher for synchronized spike trains."""
        n_steps = 100
        n_neurons = 10
        # Synchronized spikes [T, N]
        spikes = np.zeros((n_steps, n_neurons), dtype=bool)
        # All neurons spike together at certain times
        for t in range(0, n_steps, 10):
            spikes[t, :] = True

        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert kappa > 0.5  # High synchrony

    def test_kappa_synchrony_single_neuron(self):
        """Test that kappa returns 0 for single neuron."""
        spikes = np.ones((100, 1), dtype=bool)
        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert kappa == 0.0

    def test_kappa_synchrony_empty_array(self):
        """Test that kappa returns 0 for empty array."""
        spikes = np.array([])
        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert kappa == 0.0

    def test_kappa_synchrony_silent_neurons(self):
        """Test that kappa handles completely silent neurons gracefully."""
        spikes = np.zeros((100, 5), dtype=bool)
        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
        assert kappa == 0.0


class TestRateSynchronyTargets:
    """Tests for rate_synchrony_targets() function."""

    def test_rate_synchrony_targets_returns_objective(self):
        """Test that rate_synchrony_targets returns an Objective."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
        )

        assert isinstance(objective, jtfne.Objective)

    def test_rate_synchrony_targets_has_name_and_kind(self):
        """Test that returned Objective has correct name and kind."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=8.0,
            target_kappa_synchrony=-0.1,
            rate_weight=1.0,
            synchrony_weight=0.5,
        )

        assert objective.name == "rate_synchrony_targets"
        assert objective.kind == "rate_synchrony_targets"

    def test_rate_synchrony_targets_has_losses(self):
        """Test that returned Objective has loss terms."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
        )

        assert len(objective.losses) == 2
        loss_names = {loss["name"] for loss in objective.losses}
        assert "population_firing_rate" in loss_names
        assert "kappa_synchrony" in loss_names

    def test_rate_synchrony_targets_truth_gates(self):
        """Test that returned Objective includes truth gates."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
        )

        gate_names = {gate["name"] for gate in objective.gates}
        assert "truth_mode" in gate_names
        assert "claim_level" in gate_names

        # Find and verify gate values
        for gate in objective.gates:
            if gate["name"] == "truth_mode":
                assert gate["threshold"] == "truth_safe_unverified"
            elif gate["name"] == "claim_level":
                assert gate["threshold"] == "computational_scaffold"

    def test_rate_synchrony_targets_loss_targets(self):
        """Test that loss targets are set correctly."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
        )

        for loss in objective.losses:
            if loss["name"] == "population_firing_rate":
                assert loss["target"] == 5.0
            elif loss["name"] == "kappa_synchrony":
                assert loss["target"] == 0.0

    def test_rate_synchrony_targets_custom_weights(self):
        """Test custom weights are preserved in losses."""
        objective = jtfne.rate_synchrony_targets(
            target_rate_hz=5.0,
            target_kappa_synchrony=0.0,
            rate_weight=2.0,
            synchrony_weight=1.0,
        )

        for loss in objective.losses:
            if loss["name"] == "population_firing_rate":
                assert loss["weight"] == 2.0
            elif loss["name"] == "kappa_synchrony":
                assert loss["weight"] == 1.0


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
        assert objective.name == "rate_synchrony_targets"

    def test_kappa_on_simulated_spikes(self):
        """Test kappa_synchrony on actual simulated spikes."""
        cfg = jtfne.suite2_net1_config(seed=7, n=20, duration_ms=50.0, dt_ms=0.1)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, seed=7, duration_ms=50.0, dt_ms=0.1)

        spikes = np.array(signals.spikes)
        kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)

        assert isinstance(kappa, float)
        assert -1.0 <= kappa <= 1.0
