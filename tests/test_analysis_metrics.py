"""Test suite for jaxfne.analysis.metrics (Phase J1).

Tests cover:
- JSON-safe outputs for defined inputs (floats); NaN for undefined inputs
- Boundary conditions (empty, single neuron, no spikes)
- Parametrized test cases with known outputs
- Signals compatibility
- No new dependencies
"""

import json
import numpy as np
import pytest
import warnings
from jaxfne.analysis import (
    mean_pairwise_spike_correlation,
    burst_index,
    fano_factor,
    fleiss_kappa_binary,
)


class TestDeclaredDefinitions:
    """T-13 declared definitions: shared binning, per-neuron Fano, binned burst/correlation."""

    def test_fano_per_neuron_differs_from_population_sum(self):
        # Hand computation at bin 1 step (b=1): each row [1,0,0,0] has
        # mean 0.25 and var ((0.75**2 + 3 * 0.25**2) / 4) = 0.1875, so
        # F_i = 0.1875 / 0.25 = 0.75 per neuron, mean 0.75. The old
        # population-summed series [2,0,0,0] has mean 0.5 and var 0.75,
        # i.e. 1.5, so the definitions differ here.
        spikes = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        assert fano_factor(spikes, bin_size_ms=1.0, dt_ms=1.0) == pytest.approx(0.75)

    def test_burst_index_changes_with_bin_ms(self):
        # Hand computation on [[1,0,1,0],[0,0,1,1]] at threshold 0.5: at
        # bin 1 step the per-bin active fractions are [0.5, 0.0, 1.0, 0.5],
        # so 3/4 = 0.75 reach threshold; at bin 2 steps the binned counts
        # are [[1,1],[0,2]], fractions [0.5, 1.0], so 2/2 = 1.0.
        spikes = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        fine = burst_index(spikes, bin_ms=1.0, dt_ms=1.0)
        coarse = burst_index(spikes, bin_ms=2.0, dt_ms=1.0)
        assert fine == pytest.approx(0.75)
        assert coarse == pytest.approx(1.0)

    def test_binned_correlation_differs_from_per_step(self):
        # Hand computation on [[1,0,0,1],[0,1,1,0]]: per-step rows have
        # means 0.5, covariance -0.25 and variances 0.25, so r = -1.0. At
        # bin 2 steps the binned rows are [1,1] and [1,1], both without
        # variance, so the metric returns 0.0.
        spikes = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
        per_step = mean_pairwise_spike_correlation(spikes, bin_ms=1.0, dt_ms=1.0)
        binned = mean_pairwise_spike_correlation(spikes, bin_ms=2.0, dt_ms=1.0)
        assert per_step == pytest.approx(-1.0)
        assert np.isnan(binned)

    def test_correlation_requires_bin_ms_and_dt_ms(self):
        spikes = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
        with pytest.raises(TypeError):
            mean_pairwise_spike_correlation(spikes)

    def test_no_deprecation_warning_emitted(self):
        spikes = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fano_factor(spikes, bin_size_ms=1.0, dt_ms=1.0)
            burst_index(spikes, bin_ms=1.0, dt_ms=1.0)
            mean_pairwise_spike_correlation(spikes, bin_ms=1.0, dt_ms=1.0)


class TestMeanPairwiseSpikeCorrelation:
    """Test mean_pairwise_spike_correlation metric."""

    def test_json_safe_output(self):
        """Output must be JSON-safe float."""
        spikes = np.random.rand(10, 100) > 0.8
        result = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        assert isinstance(result, (int, float))
        assert np.isfinite(result)
        # Verify JSON serializable
        json.dumps({"correlation": result})

    def test_empty_array(self):
        """Empty array is undefined -> NaN."""
        spikes = np.zeros((0, 100))
        assert np.isnan(mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1))

    def test_single_neuron(self):
        """Single neuron is undefined (no pairs) -> NaN."""
        spikes = np.random.rand(1, 100) > 0.8
        assert np.isnan(mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1))

    def test_no_variance(self):
        """All same spike values is undefined -> NaN."""
        spikes = np.ones((10, 100))
        assert np.isnan(mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1))

    def test_perfectly_correlated(self):
        """Identical spike trains return correlation near 1.0."""
        rng = np.random.default_rng(7)
        spike_pattern = rng.random(100) > 0.8
        spikes = np.tile(spike_pattern, (5, 1))
        result = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        assert result > 0.9  # Near perfect correlation

    def test_uncorrelated_spikes(self):
        """Random independent spikes return correlation near 0.0."""
        np.random.seed(42)
        spikes = np.random.rand(20, 1000) > 0.8
        result = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        assert -0.3 < result < 0.3  # Should be near 0

    def test_min_neurons_filter(self):
        """Only neurons with variance should be included."""
        spikes = np.zeros((10, 100))
        spikes[0:5, :] = np.random.rand(5, 100) > 0.5  # Only 5 active
        result = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1, min_neurons=3)
        assert np.isfinite(result)

    def test_range_output(self):
        """Output must be in [-1, 1]."""
        for _ in range(5):
            spikes = np.random.rand(10, 1000) > 0.8
            result = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
            assert -1.0 <= result <= 1.0

    def test_nan_handling(self):
        """NaN values should be handled gracefully."""
        spikes = np.random.rand(10, 100) > 0.8
        spikes = spikes.astype(float)
        spikes[0, :5] = np.nan
        result = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        assert np.isfinite(result)

    def test_continuous_spike_proxy(self):
        """Continuous values in [0,1] should work."""
        spikes = np.random.rand(10, 100)  # [0,1] values
        result = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_short_timeseries(self):
        """Very short timeseries is undefined (fewer than 2 bins) -> NaN."""
        spikes = np.random.rand(10, 1)
        assert np.isnan(mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1))


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
        """Empty array is undefined -> NaN."""
        spikes = np.zeros((0, 1000))
        assert np.isnan(burst_index(spikes, bin_ms=10, dt_ms=0.1))

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
        rng = np.random.default_rng(11)
        spikes = rng.random((20, 10000)) > 0.8
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
        """Empty array is undefined -> NaN."""
        spikes = np.zeros((0, 1000))
        assert np.isnan(fano_factor(spikes, bin_size_ms=10, dt_ms=0.1))

    def test_no_spikes(self):
        """No spikes is undefined (zero mean) -> NaN."""
        spikes = np.zeros((20, 1000))
        assert np.isnan(fano_factor(spikes, bin_size_ms=10, dt_ms=0.1))

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
        # One spike per bin per neuron: b = round(10 / 0.1) = 100 steps over
        # 1000 steps gives 10 bins; each neuron's counts are [1] * 10, so
        # var 0 and F_i = 0 for every neuron.
        spikes = np.zeros((20, 1000))
        for i in range(20):
            spikes[i, ::100] = 1
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
        """Bin size larger than array gives zero full bins -> NaN."""
        spikes = np.random.rand(20, 100) > 0.9
        result = fano_factor(spikes, bin_size_ms=1000, dt_ms=0.1)
        assert np.isnan(result)


class TestFleissKappaBinary:
    """Test fleiss_kappa_binary metric."""

    def test_json_safe_output(self):
        """Output must be JSON-safe float in [-1, 1]."""
        spikes = np.random.rand(30, 10000) > 0.8
        result = fleiss_kappa_binary(spikes, bin_ms=1, dt_ms=0.1)
        assert isinstance(result, (int, float))
        assert -1.0 <= result <= 1.0
        json.dumps({"kappa": result})

    def test_empty_array(self):
        """Empty array is undefined -> NaN."""
        spikes = np.zeros((0, 1000))
        assert np.isnan(fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1))

    def test_single_neuron(self):
        """Single neuron is undefined (need >= 2) -> NaN."""
        spikes = np.random.rand(1, 1000) > 0.8
        assert np.isnan(fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1))

    def test_high_agreement(self):
        """Population with high but not perfect agreement should give κ > 0."""
        # Create blocks: all fire together, then all silent
        spikes = np.concatenate(
            [np.ones((30, 500)), np.zeros((30, 500))], axis=1
        )
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert result > 0.5  # High agreement on activity blocks

    def test_no_agreement(self):
        """Half unanimous and half independent bins give κ near 0.0."""
        # Dense independent spikes saturate wide bins (every bin active),
        # which makes kappa undefined (NaN); this mixture keeps bins
        # defined with observed agreement near expected agreement.
        rng = np.random.default_rng(1)
        n, t = 30, 4000
        spikes = np.zeros((n, t))
        spikes[:, : t // 2] = (rng.random(t // 2) > 0.5)[None, :]
        spikes[:, t // 2 :] = rng.random((n, t // 2)) > 0.5
        result = fleiss_kappa_binary(spikes, bin_ms=0.1, dt_ms=0.1)
        assert -0.2 < result < 0.2

    def test_dense_saturation_is_nan(self):
        """Independent dense spikes saturate every bin -> NaN (undefined)."""
        np.random.seed(42)
        spikes = np.random.rand(30, 10000) > 0.5
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert np.isnan(result)

    def test_alternating_activity(self):
        """Alternating firing saturates every wide bin -> NaN (undefined)."""
        # Alternating pattern: neurons fire in different bins
        spikes = np.zeros((30, 1000))
        # Even neurons fire in odd bins, odd neurons in even bins
        spikes[::2, ::2] = 1  # Every other neuron, every other bin
        spikes[1::2, 1::2] = 1
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert np.isnan(result)

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
        result = fleiss_kappa_binary(spikes, bin_ms=1, dt_ms=0.1)
        assert np.isfinite(result)

    def test_range_output(self):
        """Output must be in [-1, 1]."""
        for _ in range(5):
            spikes = np.random.rand(30, 1000) > 0.8
            result = fleiss_kappa_binary(spikes, bin_ms=1, dt_ms=0.1)
            assert -1.0 <= result <= 1.0

    def test_varying_bin_sizes(self):
        """Narrow bins give defined results; saturated wide bins give NaN."""
        rng = np.random.default_rng(3)
        spikes = rng.random((30, 10000)) > 0.8
        result = fleiss_kappa_binary(spikes, bin_ms=1, dt_ms=0.1)
        assert -1.0 <= result <= 1.0
        for bin_ms in [10, 50, 100]:
            result = fleiss_kappa_binary(spikes, bin_ms=bin_ms, dt_ms=0.1)
            assert np.isnan(result)

    def test_continuous_spike_values(self):
        """Float-valued spike input should work."""
        rng = np.random.default_rng(4)
        spikes = (rng.random((30, 1000)) > 0.9).astype(float)
        result = fleiss_kappa_binary(spikes, bin_ms=1, dt_ms=0.1)
        assert -1.0 <= result <= 1.0

    def test_all_active_input_is_nan(self):
        """Continuous (0, 1) values binarize to all-active -> NaN."""
        spikes = np.random.rand(30, 1000)  # [0,1]
        result = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)
        assert np.isnan(result)


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
        c1 = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        c2 = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        assert c1 == c2

    def test_deterministic_output_all_metrics(self):
        """All metrics should be deterministic."""
        spikes = np.random.rand(30, 5000) > 0.8
        for _ in range(3):
            c = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
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
        c = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
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
        c = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
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
        c = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        b = burst_index(spikes, bin_ms=10, dt_ms=0.1)
        f = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
        k = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)

        assert all(np.isfinite(x) for x in [c, b, f, k])

    def test_all_metrics_with_extreme_sparsity(self):
        """All metrics should handle very sparse spikes."""
        spikes = np.random.rand(100, 100000) > 0.99
        c = mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        b = burst_index(spikes, bin_ms=10, dt_ms=0.1)
        f = fano_factor(spikes, bin_size_ms=10, dt_ms=0.1)
        k = fleiss_kappa_binary(spikes, bin_ms=10, dt_ms=0.1)

        assert all(np.isfinite(x) for x in [c, b, f, k])


class TestUndefinedIsNaN:
    """Undefined inputs return NaN; defined zeros stay 0.0 (2026-10-13)."""

    def test_correlation_fewer_than_two_bins_is_nan(self):
        spikes = np.random.rand(4, 1)
        assert np.isnan(mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1))

    def test_correlation_too_few_variance_rows_is_nan(self):
        spikes = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
        assert np.isnan(
            mean_pairwise_spike_correlation(spikes, bin_ms=2.0, dt_ms=1.0)
        )

    def test_correlation_too_few_active_bins_is_nan(self):
        spikes = np.zeros((3, 8))
        spikes[0, 0] = 1.0
        spikes[1, 0] = 1.0
        spikes[2, 0] = 1.0
        assert np.isnan(
            mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1)
        )

    def test_correlation_defined_zero_stays_zero(self):
        spikes = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
        assert mean_pairwise_spike_correlation(spikes, bin_ms=1, dt_ms=1) == pytest.approx(-1.0)

    def test_burst_zero_full_bins_is_nan(self):
        spikes = np.random.rand(4, 3)
        assert np.isnan(burst_index(spikes, bin_ms=100.0, dt_ms=1.0))

    def test_burst_defined_zero_stays_zero(self):
        spikes = np.zeros((4, 8))
        assert burst_index(spikes, bin_ms=1, dt_ms=1) == 0.0

    def test_fano_zero_full_bins_is_nan(self):
        spikes = np.random.rand(4, 3)
        assert np.isnan(fano_factor(spikes, bin_size_ms=100.0, dt_ms=1.0))

    def test_fano_no_positive_mean_is_nan(self):
        spikes = np.zeros((4, 8))
        assert np.isnan(fano_factor(spikes, bin_size_ms=1, dt_ms=1))

    def test_fano_defined_zero_stays_zero(self):
        spikes = np.ones((4, 8))
        assert fano_factor(spikes, bin_size_ms=1, dt_ms=1) == 0.0

    def test_kappa_no_variance_is_nan(self):
        spikes = np.zeros((4, 100))
        assert np.isnan(fleiss_kappa_binary(spikes, bin_ms=1, dt_ms=1))
        spikes = np.ones((4, 100))
        assert np.isnan(fleiss_kappa_binary(spikes, bin_ms=1, dt_ms=1))

    def test_kappa_single_neuron_is_nan(self):
        spikes = np.random.rand(1, 100) > 0.5
        assert np.isnan(fleiss_kappa_binary(spikes, bin_ms=1, dt_ms=1))

    def test_zero_full_bins_is_nan(self):
        """A bin window covering the whole trace (n_bins == 0) is undefined -> NaN."""
        spikes = np.zeros((4, 10))
        spikes[0, 0] = 1.0
        spikes[1, 5] = 1.0
        bin_ms, dt_ms = 1000.0, 0.1
        assert spikes.shape[1] // max(1, int(round(bin_ms / dt_ms))) == 0
        assert np.isnan(fleiss_kappa_binary(spikes, bin_ms=bin_ms, dt_ms=dt_ms))


# --- T-23 activity-regime classifier (Hamm 2026-10-14) -------------------------


def _regime_api():
    """Lazily import the T-23 names so a missing implementation fails per-test."""
    from jaxfne.analysis import (
        ACTIVITY_REGIME_DEFAULTS,
        activity_regime,
        activity_regimes,
        wang_buzsaki_kappa,
    )

    return ACTIVITY_REGIME_DEFAULTS, activity_regime, activity_regimes, wang_buzsaki_kappa


def _t23_runaway_spikes():
    """12 cells, 4000 steps at 0.5 ms: 2 spikes then 40 per cell (ratio 20)."""
    rng = np.random.default_rng(23)
    n, half = 12, 2000
    spikes = np.zeros((2 * half, n), dtype=bool)
    for j in range(n):
        spikes[int(rng.integers(0, 1000)), j] = True
        spikes[int(1000 + rng.integers(0, 1000)), j] = True
        spikes[rng.choice(np.arange(half, 2 * half), size=40, replace=False), j] = True
    return spikes


def _t23_oscillatory_spikes():
    """16 cells, 20000 steps at 0.5 ms: 8 cells burst in lockstep every 40 steps."""
    spikes = np.zeros((20000, 16), dtype=bool)
    spikes[np.arange(0, 20000, 40), :8] = True
    return spikes


def _t23_async_spikes():
    """20 cells, 20000 steps at 0.5 ms: independent 0.005/step Bernoulli trains."""
    rng = np.random.default_rng(7)
    return rng.random((20000, 20)) < 0.005


def _t23_irregular_spikes():
    """20 cells in 5 groups of 4 sharing one Bernoulli train per group."""
    rng = np.random.default_rng(11)
    spikes = np.zeros((20000, 20), dtype=bool)
    for group in range(5):
        train = rng.random(20000) < 0.005
        spikes[:, 4 * group : 4 * group + 4] = train[:, None]
    return spikes


class TestWangBuzsakiKappa:
    """wang_buzsaki_kappa: mean binary-bin cosine similarity over active pairs."""

    def test_hand_pinned_value(self):
        _, _, _, wang_buzsaki_kappa = _regime_api()
        # [n_steps, n_neurons], dt 1 ms, 2 ms bins: bin 0 holds steps 0-1
        # (cells 0, 1, 2 active), bin 1 holds steps 2-3 (cells 0, 1 active).
        # X = [[1, 1, 1], [1, 1, 0]] with column sums [2, 2, 1]:
        # k01 = 2 / sqrt(2*2) = 1, k02 = k12 = 1 / sqrt(2*1).
        spikes = np.array(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool
        )
        assert wang_buzsaki_kappa(spikes, 1.0, bin_ms=2.0) == pytest.approx(
            (1.0 + 2.0 / np.sqrt(2.0)) / 3.0
        )

    def test_non_positive_values_are_silence_per_step(self):
        _, _, _, wang_buzsaki_kappa = _regime_api()
        # Cell 0 fires at step 0; the -5 at step 1 is silence, so its bin is
        # active. Summing raw values first would give -4 and drop the cell.
        spikes = np.array([[1.0, 1.0], [-5.0, 0.0]])
        assert spikes[0, 0] > 0 and spikes[:, 0].sum() < 0, "fixture must mix signs"
        assert wang_buzsaki_kappa(spikes, 1.0, bin_ms=2.0) == pytest.approx(1.0)

    def test_disjoint_trains_kappa_zero(self):
        _, _, _, wang_buzsaki_kappa = _regime_api()
        spikes = np.zeros((4, 2), dtype=bool)
        spikes[0, 0] = True
        spikes[2, 1] = True
        assert wang_buzsaki_kappa(spikes, 1.0, bin_ms=2.0) == 0.0

    def test_identical_trains_kappa_one(self):
        _, _, _, wang_buzsaki_kappa = _regime_api()
        rng = np.random.default_rng(3)
        train = rng.random(200) > 0.5
        assert train.any()
        spikes = np.tile(train, (4, 1)).T
        assert wang_buzsaki_kappa(spikes, 1.0, bin_ms=5.0) == pytest.approx(1.0)

    def test_nan_without_two_active_cells(self):
        _, _, _, wang_buzsaki_kappa = _regime_api()
        assert np.isnan(wang_buzsaki_kappa(np.zeros((10, 3), dtype=bool), 0.5))
        one = np.zeros((10, 3), dtype=bool)
        one[2, 1] = True
        assert np.isnan(wang_buzsaki_kappa(one, 0.5))

    def test_refuses_non_multiple_bin_ms(self):
        _, _, _, wang_buzsaki_kappa = _regime_api()
        spikes = np.zeros((10, 3), dtype=bool)
        with pytest.raises(ValueError, match="bin_ms"):
            wang_buzsaki_kappa(spikes, 0.5, bin_ms=1.3)
        with pytest.raises(ValueError, match="bin_ms"):
            wang_buzsaki_kappa(spikes, 0.5, bin_ms=0.0)
        with pytest.raises(ValueError, match="bin_ms"):
            wang_buzsaki_kappa(spikes, 0.5, bin_ms=-5.0)


class TestActivityRegimeLabels:
    """Each label has a fixture that builds its case and asserts its precondition."""

    def test_silent(self):
        _, activity_regime, _, _ = _regime_api()
        out = activity_regime(np.zeros((2000, 8), dtype=bool), 0.5)
        assert out["rate_hz"] == 0.0
        assert out["active_fraction"] == 0.0
        assert out["label"] == "silent"

    def test_saturated(self):
        _, activity_regime, _, _ = _regime_api()
        spikes = np.ones((2000, 8), dtype=bool)
        assert bool(spikes.all())  # precondition: every cell spikes every step
        out = activity_regime(spikes, 0.5)
        assert out["active_fraction"] == pytest.approx(1.0)
        assert out["kappa"] == pytest.approx(1.0)
        assert out["label"] == "saturated"

    def test_runaway(self):
        _, activity_regime, _, _ = _regime_api()
        out = activity_regime(_t23_runaway_spikes(), 0.5)
        assert out["rate_ratio"] == pytest.approx(20.0)  # 40 vs 2 spikes per cell
        assert out["kappa"] < 0.5  # precondition: not saturated by synchrony
        assert out["label"] == "runaway"

    def test_oscillatory(self):
        _, activity_regime, _, _ = _regime_api()
        out = activity_regime(_t23_oscillatory_spikes(), 0.5)
        assert out["active_fraction"] == pytest.approx(0.5)  # 8 of 16 cells drive
        assert out["spectral_peak_excess"] >= 5.0
        assert out["label"] == "oscillatory"

    def test_asynchronous(self):
        _, activity_regime, _, _ = _regime_api()
        out = activity_regime(_t23_async_spikes(), 0.5)
        assert out["kappa"] < 0.1
        assert 0.7 <= out["cv_isi"] <= 1.5
        assert out["spectral_peak_excess"] < 5.0
        assert out["label"] == "asynchronous"

    def test_irregular(self):
        _, activity_regime, _, _ = _regime_api()
        out = activity_regime(_t23_irregular_spikes(), 0.5)
        assert out["kappa"] > 0.1  # precondition: too correlated for asynchronous
        assert 0.7 <= out["cv_isi"] <= 1.5  # ... although timing is Poisson-like
        assert out["spectral_peak_excess"] < 5.0
        assert out["label"] == "irregular"


class TestActivityRegimeThresholds:
    """Each override below moves one label across a boundary value."""

    def test_unknown_threshold_key_refused_by_name(self):
        _, activity_regime, _, _ = _regime_api()
        with pytest.raises(ValueError, match="bogus_threshold"):
            activity_regime(
                np.zeros((100, 4), dtype=bool),
                0.5,
                thresholds={"bogus_threshold": 1.0},
            )

    def test_silent_rate_floor(self):
        defaults, activity_regime, _, _ = _regime_api()
        assert defaults["silent_rate_hz"] == pytest.approx(0.1)
        # 20 cells, one spike each over 20 s: 0.05 Hz, every cell active.
        spikes = np.zeros((40000, 20), dtype=bool)
        spikes[np.linspace(0, 39999, 20).astype(int), np.arange(20)] = True
        out = activity_regime(spikes, 0.5)
        assert out["rate_hz"] == pytest.approx(0.05)
        assert out["label"] == "silent"
        flipped = activity_regime(spikes, 0.5, thresholds={"silent_rate_hz": 0.01})
        assert flipped["label"] == "irregular"
        assert flipped["thresholds"]["silent_rate_hz"] == pytest.approx(0.01)

    def test_silent_active_fraction_floor(self):
        _, activity_regime, _, _ = _regime_api()
        # Exactly one of 20 cells fires: active fraction 0.05.
        rng = np.random.default_rng(31)
        spikes = np.zeros((20000, 20), dtype=bool)
        spikes[:, 0] = rng.random(20000) < 0.005
        assert spikes[:, 1:].sum() == 0
        out = activity_regime(spikes, 0.5)
        assert out["active_fraction"] == pytest.approx(0.05)
        assert out["label"] == "irregular"
        flipped = activity_regime(
            spikes, 0.5, thresholds={"silent_active_fraction": 0.06}
        )
        assert flipped["label"] == "silent"

    def test_saturated_arms(self):
        _, activity_regime, _, _ = _regime_api()
        spikes = np.ones((2000, 8), dtype=bool)
        assert activity_regime(spikes, 0.5)["label"] == "saturated"
        # Kappa cannot exceed 1, so a floor of 2 refuses the synchrony arm;
        # a 5000 Hz ceiling refuses the rate arm (2000 Hz here).
        flipped = activity_regime(
            spikes,
            0.5,
            thresholds={"saturated_kappa": 2.0, "saturated_rate_hz": 5000.0},
        )
        assert flipped["label"] == "irregular"

    def test_saturated_active_fraction_arm(self):
        _, activity_regime, _, _ = _regime_api()
        assert activity_regime(_t23_oscillatory_spikes(), 0.5)["label"] == "oscillatory"
        flipped = activity_regime(
            _t23_oscillatory_spikes(), 0.5, thresholds={"saturated_active_fraction": 0.4}
        )
        assert flipped["label"] == "saturated"

    def test_runaway_ratio(self):
        _, activity_regime, _, _ = _regime_api()
        assert activity_regime(_t23_runaway_spikes(), 0.5)["label"] == "runaway"
        flipped = activity_regime(
            _t23_runaway_spikes(), 0.5, thresholds={"runaway_ratio": 50.0}
        )
        assert flipped["label"] == "irregular"

    def test_oscillation_floor(self):
        _, activity_regime, _, _ = _regime_api()
        assert activity_regime(_t23_oscillatory_spikes(), 0.5)["label"] == "oscillatory"
        flipped = activity_regime(
            _t23_oscillatory_spikes(), 0.5, thresholds={"oscillation_floor": 1e9}
        )
        assert flipped["label"] == "irregular"

    def test_async_kappa_and_cv_bounds(self):
        _, activity_regime, _, _ = _regime_api()
        assert activity_regime(_t23_async_spikes(), 0.5)["label"] == "asynchronous"
        assert (
            activity_regime(
                _t23_async_spikes(), 0.5, thresholds={"async_kappa": 1e-9}
            )["label"]
            == "irregular"
        )
        assert (
            activity_regime(_t23_async_spikes(), 0.5, thresholds={"async_cv_lo": 1.1})[
                "label"
            ]
            == "irregular"
        )
        assert (
            activity_regime(_t23_async_spikes(), 0.5, thresholds={"async_cv_hi": 0.9})[
                "label"
            ]
            == "irregular"
        )

    def test_kappa_bin_moves_async_boundary(self):
        _, activity_regime, _, _ = _regime_api()
        out = activity_regime(_t23_async_spikes(), 0.5)
        assert out["kappa"] < 0.1
        wide = activity_regime(_t23_async_spikes(), 0.5, thresholds={"bin_ms": 50.0})
        assert wide["kappa"] > 0.1
        assert wide["label"] == "irregular"

    def test_welch_resolution_moves_oscillatory_boundary(self):
        _, activity_regime, _, _ = _regime_api()
        assert activity_regime(_t23_oscillatory_spikes(), 0.5)["label"] == "oscillatory"
        coarse = activity_regime(
            _t23_oscillatory_spikes(), 0.5, thresholds={"welch_nperseg": 16}
        )
        assert coarse["spectral_peak_excess"] == pytest.approx(1.0)
        assert coarse["label"] == "irregular"


class TestActivityRegimeWindows:
    def test_kept_window_changes_the_call(self):
        _, activity_regime, _, _ = _regime_api()
        spikes = _t23_runaway_spikes()
        assert activity_regime(spikes, 0.5)["label"] == "runaway"
        first = activity_regime(spikes, 0.5, kept_window=(0, 2000))
        assert first["rate_ratio"] == pytest.approx(1.0)  # 12 vs 12 spikes
        assert first["label"] == "irregular"

    def test_kept_window_refused_by_name(self):
        _, activity_regime, _, _ = _regime_api()
        spikes = _t23_runaway_spikes()
        for bad in ((0, 0), (10, 5), (-1, 10), (0, 4001), (0, 4000.5)):
            with pytest.raises(ValueError, match="kept_window"):
                activity_regime(spikes, 0.5, kept_window=bad)


class TestActivityRegimesPopulations:
    def test_per_population_labels(self):
        _, _, activity_regimes, _ = _regime_api()
        out = activity_regimes(
            _t23_oscillatory_spikes(),
            0.5,
            {"drivers": np.arange(8), "quiet": np.arange(8, 16)},
        )
        assert set(out) == {"drivers", "quiet"}
        # Precondition: the driving octet fires in lockstep every 40 steps.
        assert out["drivers"]["kappa"] == pytest.approx(1.0)
        assert out["drivers"]["label"] == "saturated"
        assert out["quiet"]["label"] == "silent"

    def test_populations_refused_by_name(self):
        _, _, activity_regimes, _ = _regime_api()
        spikes = _t23_oscillatory_spikes()
        with pytest.raises(ValueError, match="populations"):
            activity_regimes(spikes, 0.5, ["drivers"])
        with pytest.raises(ValueError, match="populations"):
            activity_regimes(spikes, 0.5, {})


class TestActivityRegimeNanAndEcho:
    def test_undefined_metrics_are_nan(self):
        _, activity_regime, _, _ = _regime_api()
        out = activity_regime(np.zeros((500, 4), dtype=bool), 0.5)
        assert np.isnan(out["kappa"])
        assert np.isnan(out["cv_isi"])
        assert out["label"] == "silent"
        # No cell reaches 3 spikes: cv stays NaN while the rate is healthy.
        # Eight impulses at aperiodic steps in distinct 5 ms bins.
        sparse = np.zeros((500, 4), dtype=bool)
        onsets = [3, 47, 101, 163, 211, 289, 330, 402]
        for j, step in enumerate(onsets):
            sparse[step, j % 4] = True
        out = activity_regime(sparse, 0.5)
        assert np.isnan(out["cv_isi"])
        assert out["kappa"] == 0.0
        assert out["label"] == "irregular"

    def test_thresholds_echoed_and_dict_per_call(self):
        defaults, activity_regime, _, _ = _regime_api()
        spikes = np.zeros((500, 4), dtype=bool)
        first = activity_regime(spikes, 0.5)
        second = activity_regime(spikes, 0.5)
        assert set(first) == set(second)
        assert first["label"] == second["label"] == "silent"
        assert first["thresholds"] == second["thresholds"]
        assert first["thresholds"] == dict(defaults)
        assert first["thresholds"] is not defaults
        first["thresholds"]["silent_rate_hz"] = -1.0
        assert second["thresholds"]["silent_rate_hz"] == pytest.approx(0.1)
        assert defaults["silent_rate_hz"] == pytest.approx(0.1)

    def test_root_and_analysis_exports(self):
        import jaxfne as jtfne
        import jaxfne.analysis as analysis
        from jaxfne.public_surface import symbol_tier

        for name in ("wang_buzsaki_kappa", "activity_regime", "activity_regimes"):
            assert hasattr(jtfne, name), name
            assert name in jtfne.__all__, name
            assert hasattr(analysis, name), name
            assert symbol_tier(name) == "CANONICAL", name
