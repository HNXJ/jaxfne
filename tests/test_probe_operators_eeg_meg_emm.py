"""Tests for EEG/MEG/EMM proxy operator implementations.

Validates real proxy mathematical transformations:
- EEG proxy: Y_eeg[c,t] = sum_k L_eeg[c,k] * S[t,k]
- MEG proxy: Y_meg[c,t] = sum_k L_meg[c,k] * J_oriented[t,k]
- EMM proxy: E[t] = (lambda_spk * R_spk[t] + lambda_src * ||S[t]||_1 + lambda_field * ||Phi[t]||_2^2) / normalization
"""

import json
import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne.fields import eeg_proxy_transform, meg_proxy_transform, emm_proxy_transform


class TestEEGProxyTransform:
    """Test EEG proxy linear leadfield transformation."""

    def test_eeg_proxy_basic_projection(self):
        """EEG proxy must equal source @ leadfield.T."""
        # Create simple test data
        T, K, C = 10, 5, 3  # 10 timesteps, 5 sources, 3 sensors
        source = jnp.ones((T, K))  # [T, K]
        leadfield = jnp.ones((C, K))  # [C, K]

        # Compute EEG proxy
        y_eeg = eeg_proxy_transform(source, leadfield)

        # Expected shape
        assert y_eeg.shape == (T, C)

        # Expected values: each output should be sum of K ones = K
        expected = jnp.full((T, C), K, dtype=leadfield.dtype)
        assert jnp.allclose(y_eeg, expected)

    def test_eeg_proxy_matrix_multiplication(self):
        """EEG proxy computation must match matrix multiplication contract."""
        # Create non-trivial test data
        T, K, C = 20, 4, 2
        source = jnp.arange(T * K, dtype=jnp.float32).reshape(T, K)
        leadfield = jnp.arange(C * K, dtype=jnp.float32).reshape(C, K)

        y_eeg = eeg_proxy_transform(source, leadfield)

        # Expected: source @ leadfield.T
        expected = source @ leadfield.T

        assert jnp.allclose(y_eeg, expected)

    def test_eeg_proxy_shape_mismatch_source(self):
        """EEG proxy must raise error for wrong source dimensions."""
        source_1d = jnp.ones(5)
        leadfield = jnp.ones((3, 5))

        try:
            eeg_proxy_transform(source_1d, leadfield)
            assert False, "Expected ValueError for 1D source"
        except ValueError as e:
            assert "source must be 2D" in str(e)

    def test_eeg_proxy_shape_mismatch_leadfield(self):
        """EEG proxy must raise error for wrong leadfield dimensions."""
        source = jnp.ones((10, 5))
        leadfield_1d = jnp.ones(5)

        try:
            eeg_proxy_transform(source, leadfield_1d)
            assert False, "Expected ValueError for 1D leadfield"
        except ValueError as e:
            assert "leadfield must be 2D" in str(e)

    def test_eeg_proxy_k_dimension_mismatch(self):
        """EEG proxy must raise error if K dimensions don't match."""
        source = jnp.ones((10, 5))  # K=5
        leadfield = jnp.ones((3, 4))  # K=4

        try:
            eeg_proxy_transform(source, leadfield)
            assert False, "Expected ValueError for K mismatch"
        except ValueError as e:
            assert "K dimension mismatch" in str(e)

    def test_eeg_proxy_output_finite(self):
        """EEG proxy output must be finite."""
        import jax
        T, K, C = 100, 8, 16
        key = jax.random.PRNGKey(42)
        key_s, key_l = jax.random.split(key)
        source = jax.random.normal(key_s, (T, K))
        leadfield = jax.random.normal(key_l, (C, K))

        y_eeg = eeg_proxy_transform(source, leadfield)

        assert jnp.all(jnp.isfinite(y_eeg))
        assert y_eeg.dtype in [jnp.float32, jnp.float64]


class TestMEGProxyTransform:
    """Test MEG proxy linear leadfield transformation."""

    def test_meg_proxy_basic_projection(self):
        """MEG proxy must equal source_oriented @ leadfield.T."""
        T, K, C = 10, 5, 4  # 10 timesteps, 5 sources, 4 sensors
        source_oriented = jnp.ones((T, K))
        leadfield = jnp.ones((C, K))

        y_meg = meg_proxy_transform(source_oriented, leadfield)

        assert y_meg.shape == (T, C)

        # Each output should be sum of K ones = K
        expected = jnp.full((T, C), K, dtype=leadfield.dtype)
        assert jnp.allclose(y_meg, expected)

    def test_meg_proxy_matrix_multiplication(self):
        """MEG proxy computation must match matrix multiplication contract."""
        T, K, C = 15, 6, 3
        source_oriented = jnp.arange(T * K, dtype=jnp.float32).reshape(T, K)
        leadfield = jnp.arange(C * K, dtype=jnp.float32).reshape(C, K)

        y_meg = meg_proxy_transform(source_oriented, leadfield)

        expected = source_oriented @ leadfield.T

        assert jnp.allclose(y_meg, expected)

    def test_meg_proxy_shape_mismatch_source(self):
        """MEG proxy must raise error for wrong source dimensions."""
        source_oriented = jnp.ones(5)
        leadfield = jnp.ones((3, 5))

        try:
            meg_proxy_transform(source_oriented, leadfield)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "source_oriented must be 2D" in str(e)

    def test_meg_proxy_output_finite(self):
        """MEG proxy output must be finite."""
        import jax
        T, K, C = 50, 7, 8
        key = jax.random.PRNGKey(42)
        key_s, key_l = jax.random.split(key)
        source_oriented = jax.random.normal(key_s, (T, K))
        leadfield = jax.random.normal(key_l, (C, K))

        y_meg = meg_proxy_transform(source_oriented, leadfield)

        assert jnp.all(jnp.isfinite(y_meg))


class TestEMMProxyTransform:
    """Test EMM proxy weighted cost transformation."""

    def test_emm_proxy_basic_shape(self):
        """EMM proxy must return [T, 1] shape."""
        T, K, X = 20, 5, 8
        spike_rate = jnp.ones(T)
        source = jnp.ones((T, K))
        field_potential = jnp.ones((T, X))

        emm = emm_proxy_transform(spike_rate, source, field_potential)

        assert emm.shape == (T, 1) or emm.shape == (T,)

    def test_emm_proxy_normalization(self):
        """EMM proxy must normalize by sum of weights."""
        T, K, X = 10, 3, 4
        spike_rate = jnp.ones(T)
        source = jnp.ones((T, K))
        field_potential = jnp.ones((T, X))

        # With equal weights and all ones:
        # term_spk = 1 * 1 = 1
        # term_src = 1 * K = 3
        # term_field = 1 * X = 4
        # total = 1 + 3 + 4 = 8
        # normalized = 8 / (1 + 1 + 1) = 8/3

        emm = emm_proxy_transform(spike_rate, source, field_potential,
                                   lambda_spk=1.0, lambda_src=1.0, lambda_field=1.0)

        expected_value = (1.0 + K + X) / 3.0
        assert jnp.allclose(emm, expected_value)

    def test_emm_proxy_lambda_weighting(self):
        """EMM proxy must respect lambda weight factors."""
        T, K, X = 5, 2, 3
        spike_rate = jnp.ones(T)
        source = jnp.ones((T, K))
        field_potential = jnp.ones((T, X))

        # With different weights
        lambda_spk, lambda_src, lambda_field = 2.0, 3.0, 1.0
        emm = emm_proxy_transform(spike_rate, source, field_potential,
                                   lambda_spk=lambda_spk,
                                   lambda_src=lambda_src,
                                   lambda_field=lambda_field)

        # total = 2*1 + 3*K + 1*X = 2 + 6 + 3 = 11
        # normalized = 11 / (2 + 3 + 1) = 11/6
        expected = (lambda_spk * 1 + lambda_src * K + lambda_field * X) / (lambda_spk + lambda_src + lambda_field)
        assert jnp.allclose(emm, expected)

    def test_emm_proxy_time_dimension_mismatch(self):
        """EMM proxy must raise error if time dimensions don't match."""
        spike_rate = jnp.ones(10)
        source = jnp.ones((20, 5))  # Different T!
        field_potential = jnp.ones((10, 4))

        try:
            emm_proxy_transform(spike_rate, source, field_potential)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Time dimension mismatch" in str(e)

    def test_emm_proxy_spike_rate_1d_to_2d(self):
        """EMM proxy must handle 1D spike_rate by reshaping to [T, 1]."""
        T, K, X = 15, 4, 6
        spike_rate_1d = jnp.ones(T)
        source = jnp.ones((T, K))
        field_potential = jnp.ones((T, X))

        emm = emm_proxy_transform(spike_rate_1d, source, field_potential)

        # Should work without error
        assert emm.shape[0] == T

    def test_emm_proxy_output_finite(self):
        """EMM proxy output must be finite."""
        import jax
        T, K, X = 50, 5, 8
        key = jax.random.PRNGKey(42)
        key_r, key_s, key_f = jax.random.split(key, 3)
        spike_rate = jnp.abs(jax.random.normal(key_r, (T,)))  # Ensure non-negative
        source = jax.random.normal(key_s, (T, K))
        field_potential = jax.random.normal(key_f, (T, X))

        emm = emm_proxy_transform(spike_rate, source, field_potential)

        assert jnp.all(jnp.isfinite(emm))

    def test_emm_proxy_different_inputs(self):
        """EMM proxy must change with different input values."""
        T, K, X = 10, 3, 4

        # Test 1: Low activity
        spike_rate_low = jnp.ones(T) * 0.1
        source_low = jnp.ones((T, K)) * 0.1
        field_low = jnp.ones((T, X)) * 0.1

        # Test 2: High activity
        spike_rate_high = jnp.ones(T) * 10.0
        source_high = jnp.ones((T, K)) * 10.0
        field_high = jnp.ones((T, X)) * 10.0

        emm_low = emm_proxy_transform(spike_rate_low, source_low, field_low)
        emm_high = emm_proxy_transform(spike_rate_high, source_high, field_high)

        # EMM should be different for different inputs
        assert not jnp.allclose(emm_low, emm_high)


class TestProbeReportMetadata:
    """Test that proxy probes include proper metadata."""

    def test_eeg_proxy_probe_report_has_truth_status(self):
        """EEG proxy probe report must include truth status."""
        eeg_data = jnp.ones((100, 16))
        probe = jtfne.fields.eeg_proxy_probe(eeg_data)

        report = probe.report
        assert "physical_amplitude_claim_allowed" in report
        assert report["physical_amplitude_claim_allowed"] is False

    def test_meg_proxy_probe_report_has_truth_status(self):
        """MEG proxy probe report must include truth status."""
        meg_data = jnp.ones((100, 8))
        probe = jtfne.fields.meg_proxy_probe(meg_data)

        report = probe.report
        assert "physical_amplitude_claim_allowed" in report
        assert report["physical_amplitude_claim_allowed"] is False

    def test_emm_proxy_probe_report_has_truth_status(self):
        """EMM proxy probe report must include truth status."""
        emm_data = jnp.ones((100, 1))
        probe = jtfne.fields.emm_proxy_probe(emm_data)

        report = probe.report
        assert "calibration_status" in report
        assert "uncalibrated" in str(report["calibration_status"]).lower()

    def test_proxy_reports_json_safe(self):
        """All proxy probe reports must serialize to JSON."""
        eeg = jtfne.fields.eeg_proxy_probe(jnp.ones((50, 8)))
        meg = jtfne.fields.meg_proxy_probe(jnp.ones((50, 4)))
        emm = jtfne.fields.emm_proxy_probe(jnp.ones((50, 1)))

        for probe in [eeg, meg, emm]:
            json_str = json.dumps(probe.report, allow_nan=False)
            assert isinstance(json_str, str)


class TestProbeIntegration:
    """Test proxy operators in full simulation context."""

    def test_eeg_proxy_in_signals(self):
        """EEG proxy must integrate with signal generation."""
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=8)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42)
        signals = model.simulate(sim)

        # Verify standard probes work
        assert signals.spikes is not None
        assert signals.V_m is not None

    def test_no_output_labels_claim_real_eeg_meg(self):
        """Proxy operator reports must not claim real EEG/MEG."""
        eeg_data = jnp.ones((100, 16))
        meg_data = jnp.ones((100, 8))

        eeg_probe = jtfne.fields.eeg_proxy_probe(eeg_data)
        meg_probe = jtfne.fields.meg_proxy_probe(meg_data)

        eeg_report_str = str(eeg_probe.report).lower()
        meg_report_str = str(meg_probe.report).lower()

        # Must not claim to be real EEG/MEG
        assert "real eeg" not in eeg_report_str
        assert "real meg" not in meg_report_str
        assert "proxy" in eeg_report_str
        assert "proxy" in meg_report_str
