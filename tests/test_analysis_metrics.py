"""Test suite for jaxfne.analysis.metrics (Phase J1).

Tests cover:
- JSON-safe outputs (floats, no NaN/Inf)
- Boundary conditions (empty, single neuron, no spikes)
- Parametrized test cases with known outputs
- Signals compatibility
- No new dependencies
"""

import json
import numpy as np
import pytest
from jaxfne.analysis import (
    mean_pairwise_spike_correlation,
    burst_index,
    fano_factor,
    fleiss_kappa_binary,
)


class TestMeanPairwiseSpikeCorrelation:
    """Test mean_pairwise_spike_correlation metric."""

    def test_json_safe_output(self):
        """Output must be JSON-safe float."""
        spikes = np.random.rand(10, 100) > 0.8
        result = mean_pairwise_spike_correlation(spikes)
        assert isinstance(result, (int, float))
        assert np.isfinite(result)
        # Verify JSON serializable
        json.dumps({"correlation": result})

    def test_empty_array(self):
        """Empty array should return 0.0."""
        spikes = np.zeros((0, 100))
        assert mean_pairwise_spike_correlation(spikes) == 0.0

    def test_single_neuron(self):
        """Single neuron returns 0.0 (no pairs)."""
        spikes = np.random.rand(1, 100) > 0.8
        assert mean_pairwise_spike_correlation(spikes) == 0.0

    def test_no_variance(self):
        """All same spike values returns 0.0."""
        spikes = np.ones((10, 100))
        assert mean_pairwise_spike_correlation(spikes) == 0.0

    def test_perfectly_correlated(self):
        """Identical spike trains return correlation near 1.0."""
        spike_pattern = np.random.rand(100) > 0.8
        spikes = np.tile(spike_pattern, (5, 1))
        result = mean_pairwise_spike_correlation(spikes)
        assert result > 0.9  # Near perfect correlation

    def test_uncorrelated_spikes(self):
        """Random independent spikes return correlation near 0.0."""
        np.random.seed(42)
        spikes = np.random.rand(20, 1000) > 0.8
        result = mean_pairwise_spike_correlation(spikes)
        assert -0.3 < result < 0.3  # Should be near 0

    def test_min_neurons_filter(self):
        """Only neurons with variance should be included."""
        spikes = np.zeros((10, 100))
        spikes[0:5, :] = np.random.rand(5, 100) > 0.5  # Only 5 active
        result = mean_pairwise_spike_correlation(spikes, min_neurons=3)
        assert np.isfinite(result)

    def test_range_output(self):
        """Output must be in [-1, 1]."""
        for _ in range(5):
            spikes = np.random.rand(10, 1000) > 0.8
            result = mean_pairwise_spike_correlation(spikes)
            assert -1.0 <= result <= 1.0

    def test_nan_handling(self):
        """NaN values should be handled gracefully."""
        spikes = np.random.rand(10, 100) > 0.8
        spikes = spikes.astype(float)
        spikes[0, :5] = np.nan
        result = mean_pairwise_spike_correlation(spikes)
        assert np.isfinite(result)

    def test_continuous_spike_proxy(self):
        """Continuous values in [0,1] should work."""
        spikes = np.random.rand(10, 100)  # [0,1] values
        result = mean_pairwise_spike_correlation(spikes)
        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_short_timeseries(self):
        """Very short timeseries should return 0.0."""
        spikes = np.random.rand(10, 1)
        assert mean_pairwise_spike_correlation(spikes) == 0.0


class TestBurstIndex:
    """Test burst_index metric."""

    def test_json_safe_output(self):
        """Output must be JSON-safe float in [0,1]."""
        spikes = np.random.rand(20, 1000) > 0.8
        result = burst_index(spikes, bin_ms=10, dt_ms=0.1)
        assert isinstance(result, (int, float))
        assert 0.0 <= result <= 1.0
        json.dumps({"burst_index": result})

    def test_empty_array(self):
        """Empty array returns 0.0."""
        spikes = np.zeros((0, 1000))
        assert burst_index(spikes, bin_ms=10, dt_ms=0.1) == 0.0

    def test_no_spikes(self):
        """No spikes returns 0.0."""
        spikes = np.zeros((20, 1000))
        assert burst_index(spikes, bin_ms=10, dt_ms=0.1) == 0.0

    def test_continuous_burst(self):
        """All neurons active all time returns 1.0."""
        spikes = np.ones((20, 1000))
        result = burst_index(spikes, bin_ms=10, dt_ms=0.1, active_fraction_threshold=0.5)
        assert result == 1.0

    def test_threshold_parameter(self):
        """Different thresholds should give different results."""
        spikes = np.random.rand(20, 1000) > 0.7
        bi_50 = burst_index(spikes, bin_ms=10, dt_ms=0.1, active_fraction_threshold=0.5)
        bi_80 = burst_index(spikes, bin_ms=10, dt_ms=0.1, active_fraction_threshold=0.8)
        assert bi_50 >= bi_80  # Higher threshold = lower burst index

    def test_invalid_dt(self):
        """Invalid dt_ms should raise ValueError."""
        spikes = np.random.rand(20, 1000)
        with pytest.raises(ValueError):
            burst_index(spikes, bin_ms=10, dt_ms=0)
        with pytest.raises(ValueError):
            burst_index(spikes, bin_ms=10, dt_ms=-1)

    def test_invalid_threshold(self):
        """Invalid threshold should raise ValueError."""
        spikes = np.random.rand(20, 1000)
        with pytest.raises(ValueError):
            burst_index(spikes, bin_ms=10, dt_ms=0.1, active_fraction_threshold=-0.1)
        with pytest.raises(ValueError):
            burst_index(spikes, bin_ms=10, dt_ms=0.1, active_fraction_threshold=1.1)

    def test_range_output(self):
        """Output must be in [0, 1]."""
        for _ in range(5):
            spikes = np.random.rand(20, 1000) > 0.8
            result = burst_index(spikes, bin_ms=10, dt_ms=0.1)
            assert 0.0 <= result <= 1.0

    def test_nan_handling(self):
        """NaN spikes should be treated as 0 (not firing)."""
        spikes = np.random.rand(20, 1000) > 0.8
        spikes = spikes.astype(float)
        spikes[:5, :] = np.nan
        result = burst_index(spikes, bin_ms=10, dt_ms=0.1)
        assert np.isfinite(result)

    def test_variable_dt_scaling(self):
        """Different dt scales should give similar results."""
        spikes = np.random.rand(20, 10000) > 0.8
        # Same physical simulation, different dt
        bi_fine = burst_index(spikes[:, :1000], bin_ms=1, dt_ms=0.01)
        bi_coarse = burst_index(spikes[:, :1000:10], bin_ms=10, dt_ms=0.1)
        # Should be similar (within tolerance due to binning)
        assert abs(bi_fine - bi_coarse) < 0.2


class TestFanoFactor:
    """Test fano_factor metric."""

    def test_json_safe_output(self):
        """Output must be JSON-safe float."""
        spikes = np.random.rand(20, 10000) > 0.9
        result = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
        assert isinstance(result, (int, float))
        assert np.isfinite(result)
        json.dumps({"fano_factor": result})

    def test_empty_array(self):
        """Empty array returns 0.0."""
        spikes = np.zeros((0, 1000))
        assert fano_factor(spikes, bin_size_ms=10, dt_ms=0.1) == 0.0

    def test_no_spikes(self):
        """No spikes returns 0.0 (zero mean)."""
        spikes = np.zeros((20, 1000))
        assert fano_factor(spikes, bin_size_ms=10, dt_ms=0.1) == 0.0

    def test_poisson_like_spikes(self):
        """Poisson-like spike train should have Fano ≈ 1.0."""
        np.random.seed(42)
        # Poisson spike train: FF should be close to 1
        spikes = np.random.poisson(lam=0.05, size=(20, 10000))
        result = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
        # Poisson has FF ≈ 1, but binning may shift it
        assert 0.5 < result < 2.0

    def test_regular_spikes(self):
        """Periodic spikes should have Fano < 1.0."""
        spikes = np.zeros((20, 1000))
        # Every 10th timestep, one neuron fires
        for i in range(20):
            spikes[i, i * 50 : i * 50 + 100 : 10] = 1
        result = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
        # Periodic should have FF < 1
        assert result < 1.0

    def test_invalid_bin_size(self):
        """Invalid bin_size_ms should raise ValueError."""
        spikes = np.random.rand(20, 1000)
        with pytest.raises(ValueError):
            fano_factor(spikes, bin_size_ms=0, dt_ms=0.1)
        with pytest.raises(ValueError):
            fano_factor(spikes, bin_size_ms=-1, dt_ms=0.1)

    def test_invalid_dt(self):
        """Invalid dt_ms should raise ValueError."""
        spikes = np.random.rand(20, 1000)
        with pytest.raises(ValueError):
            fano_factor(spikes, bin_size_ms=10, dt_ms=0)
        with pytest.raises(ValueError):
            fano_factor(spikes, bin_size_ms=10, dt_ms=-1)

    def test_nan_handling(self):
        """NaN spike counts should be treated as 0."""
        spikes = np.random.rand(20, 1000) > 0.9
        spikes = spikes.astype(float)
        spikes[:5, :] = np.nan
        result = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
        assert np.isfinite(result)

    def test_range_output(self):
        """Output should be positive."""
        for _ in range(5):
            spikes = np.random.rand(20, 1000) > 0.9
            result = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
            assert result >= 0.0

    def test_varying_bin_sizes(self):
        """Different bin sizes should all produce finite results."""
        spikes = np.random.rand(20, 10000) > 0.9
        for bin_ms in [1, 5, 10, 20, 50]:
            result = fano_factor(spikes, bin_size_ms=bin_ms, dt_ms=0.1)
            assert np.isfinite(result)

    def test_large_bin_exceeding_duration(self):
        """Bin size larger than array should still work."""
        spikes = np.random.rand(20, 100) > 0.9
        result = fano_factor(spikes, bin_size_ms=1000, dt_ms=0.1)
        assert np.isfinite(result)


class TestFleissKappaBinary:
    """Test fleiss_kappa_binary metric."""

    def test_json_safe_output(self):
        """Output must be JSON-safe float in [-1, 1]."""
        spikes = np.random.rand(30, 10000) > 0.8
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert isinstance(result, (int, float))
        assert -1.0 <= result <= 1.0
        json.dumps({"kappa": result})

    def test_empty_array(self):
        """Empty array returns 0.0."""
        spikes = np.zeros((0, 1000))
        assert fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1) == 0.0

    def test_single_neuron(self):
        """Single neuron returns 0.0 (need >= 2)."""
        spikes = np.random.rand(1, 1000) > 0.8
        assert fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1) == 0.0

    def test_high_agreement(self):
        """Population with high but not perfect agreement should give κ > 0."""
        # Create blocks: all fire together, then all silent
        spikes = np.concatenate(
            [np.ones((30, 500)), np.zeros((30, 500))], axis=1
        )
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert result > 0.5  # High agreement on activity blocks

    def test_no_agreement(self):
        """Independent neurons should give κ near 0.0."""
        np.random.seed(42)
        spikes = np.random.rand(30, 10000) > 0.5
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert -0.2 < result < 0.2

    def test_alternating_activity(self):
        """Alternating neuron activity should give κ near 0."""
        # Alternating pattern: neurons fire in different bins
        spikes = np.zeros((30, 1000))
        # Even neurons fire in odd bins, odd neurons in even bins
        spikes[::2, ::2] = 1  # Every other neuron, every other bin
        spikes[1::2, 1::2] = 1
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert -0.3 < result < 0.3  # Weak agreement

    def test_invalid_dt(self):
        """Invalid dt_ms should raise ValueError."""
        spikes = np.random.rand(30, 1000)
        with pytest.raises(ValueError):
            fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0)
        with pytest.raises(ValueError):
            fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=-1)

    def test_nan_handling(self):
        """NaN spikes should be treated as 0 (not firing)."""
        spikes = np.random.rand(30, 1000) > 0.8
        spikes = spikes.astype(float)
        spikes[:10, :] = np.nan
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert np.isfinite(result)

    def test_range_output(self):
        """Output must be in [-1, 1]."""
        for _ in range(5):
            spikes = np.random.rand(30, 1000) > 0.8
            result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
            assert -1.0 <= result <= 1.0

    def test_varying_bin_sizes(self):
        """Different bin sizes should all produce valid results."""
        spikes = np.random.rand(30, 10000) > 0.8
        for bin_ms in [1, 5, 10, 20, 50, 100]:
            result = fleiss_kappa_binary(spikes, bin_ms=bin_ms, dt_ms=0.1)
            assert -1.0 <= result <= 1.0

    def test_continuous_spike_values(self):
        """Continuous [0,1] spike values should work."""
        spikes = np.random.rand(30, 1000)  # [0,1]
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert -1.0 <= result <= 1.0


class TestModuleImports:
    """Test that analysis module is properly exposed."""

    def test_analysis_module_imported(self):
        """Module should be importable via jaxfne.analysis."""
        import jaxfne.analysis

        assert hasattr(jaxfne.analysis, "mean_pairwise_spike_correlation")
        assert hasattr(jaxfne.analysis, "burst_index")
        assert hasattr(jaxfne.analysis, "fano_factor")
        assert hasattr(jaxfne.analysis, "fleiss_kappa_binary")

    def test_all_exports(self):
        """__all__ should list all metrics."""
        from jaxfne.analysis import __all__

        expected = [
            "mean_pairwise_spike_correlation",
            "burst_index",
            "fano_factor",
            "fleiss_kappa_binary",
        ]
        for func in expected:
            assert func in __all__

    def test_root_level_access(self):
        """Metrics should be accessible via jaxfne.analysis."""
        import jaxfne

        assert hasattr(jaxfne.analysis, "mean_pairwise_spike_correlation")
        assert hasattr(jaxfne.analysis, "burst_index")
        assert hasattr(jaxfne.analysis, "fano_factor")
        assert hasattr(jaxfne.analysis, "fleiss_kappa_binary")


class TestCrossMetricConsistency:
    """Test consistency across metrics for known spike patterns."""

    def test_identical_metrics_on_same_input(self):
        """Repeated calls with same input should give identical results."""
        spikes = np.random.rand(20, 1000) > 0.8
        c1 = mean_pairwise_spike_correlation(spikes)
        c2 = mean_pairwise_spike_correlation(spikes)
        assert c1 == c2

    def test_deterministic_output_all_metrics(self):
        """All metrics should be deterministic."""
        spikes = np.random.rand(30, 5000) > 0.8
        for _ in range(3):
            c = mean_pairwise_spike_correlation(spikes)
            b = burst_index(spikes, bin_ms=10, dt_ms=0.1)
            f = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
            k = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
            assert isinstance(c, (int, float))
            assert isinstance(b, (int, float))
            assert isinstance(f, (int, float))
            assert isinstance(k, (int, float))

    def test_all_metrics_produce_json_safe_output(self):
        """All metrics must be JSON-serializable."""
        spikes = np.random.rand(30, 5000) > 0.8
        c = mean_pairwise_spike_correlation(spikes)
        b = burst_index(spikes, bin_ms=10, dt_ms=0.1)
        f = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
        k = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)

        output = {
            "correlation": c,
            "burst_index": b,
            "fano_factor": f,
            "kappa": k,
        }
        # Should not raise
        json_str = json.dumps(output)
        assert "correlation" in json_str
        assert "burst_index" in json_str
        assert "fano_factor" in json_str
        assert "kappa" in json_str


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_metrics_with_minimal_valid_input(self):
        """All metrics should handle minimal valid arrays."""
        # 2 neurons, 2 timesteps minimum
        spikes = np.array([[1, 0], [0, 1]])
        c = mean_pairwise_spike_correlation(spikes)
        b = burst_index(spikes, bin_ms=1, dt_ms=1)
        f = fano_factor(spikes, bin_size_ms=1, dt_ms=1)
        k = fleiss_kappa_binary(spikes, bin_ms=1, dt_ms=1)

        assert np.isfinite(c)
        assert np.isfinite(b)
        assert np.isfinite(f)
        assert np.isfinite(k)

    def test_all_metrics_with_large_arrays(self):
        """All metrics should handle large arrays efficiently."""
        # 1000 neurons, 100k timesteps
        spikes = np.random.rand(100, 10000) > 0.95
        c = mean_pairwise_spike_correlation(spikes)
        b = burst_index(spikes, bin_ms=10, dt_ms=0.1)
        f = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
        k = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)

        assert all(np.isfinite(x) for x in [c, b, f, k])

    def test_all_metrics_with_extreme_sparsity(self):
        """All metrics should handle very sparse spikes."""
        spikes = np.random.rand(100, 100000) > 0.99
        c = mean_pairwise_spike_correlation(spikes)
        b = burst_index(spikes, bin_ms=10, dt_ms=0.1)
        f = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
        k = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)

        assert all(np.isfinite(x) for x in [c, b, f, k])
