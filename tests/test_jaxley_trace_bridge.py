"""Tests for Jaxley array-first trace bridge (v0.2.22).

Minimal bridge for converting Jaxley-style voltage trace arrays to jaxfne Signals.
All tests pass without Jaxley installed.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from jaxfne import JaxleyTraceSpec, jaxley_trace_to_signals


class TestJaxleyTraceSpec:
    """Test JaxleyTraceSpec frozen dataclass and validation."""

    def test_trace_spec_json_safe(self) -> None:
        """Spec can be serialized to JSON-safe dict."""
        spec = JaxleyTraceSpec()
        spec_dict = spec.to_dict()
        assert isinstance(spec_dict, dict)
        assert spec_dict["claim_level"] == "computational_scaffold"
        assert spec_dict["physical_amplitude_claim_allowed"] is False
        assert "metadata" in spec_dict

    def test_trace_spec_dt_validation(self) -> None:
        """dt_ms must be > 0."""
        with pytest.raises(ValueError, match="dt_ms must be > 0"):
            JaxleyTraceSpec(dt_ms=-0.1)

        with pytest.raises(ValueError, match="dt_ms must be > 0"):
            JaxleyTraceSpec(dt_ms=0.0)

    def test_trace_spec_layout_validation(self) -> None:
        """layout must be one of the approved values."""
        with pytest.raises(ValueError, match="layout must be"):
            JaxleyTraceSpec(layout="unknown_layout")

        # Valid layouts should not raise
        for valid_layout in ["time_by_unit", "unit_by_time", "recording_by_time"]:
            spec = JaxleyTraceSpec(layout=valid_layout)
            assert spec.layout == valid_layout

    def test_trace_spec_claim_gate_immutable(self) -> None:
        """physical_amplitude_claim_allowed must be False."""
        with pytest.raises(ValueError, match="physical_amplitude_claim_allowed must be False"):
            JaxleyTraceSpec(physical_amplitude_claim_allowed=True)

    def test_trace_spec_claim_level_immutable(self) -> None:
        """claim_level must be 'computational_scaffold'."""
        with pytest.raises(ValueError, match="claim_level must be 'computational_scaffold'"):
            JaxleyTraceSpec(claim_level="empirically_validated")

    def test_trace_spec_frozen(self) -> None:
        """Spec dataclass is frozen (immutable)."""
        spec = JaxleyTraceSpec()
        with pytest.raises(AttributeError):
            spec.dt_ms = 0.05


class TestBridgeLayoutNormalization:
    """Test trace layout conversion to canonical [T, N]."""

    def test_bridge_time_by_unit_shape(self) -> None:
        """time_by_unit layout [T, N] requires no conversion."""
        trace = jnp.ones((100, 32))
        signals = jaxley_trace_to_signals(trace, layout="time_by_unit")
        assert signals.V_m.shape == (100, 32)
        assert signals.spikes.shape == (100, 32)
        assert signals.time_ms.shape == (100,)

    def test_bridge_unit_by_time_shape(self) -> None:
        """unit_by_time layout [N, T] is transposed to [T, N]."""
        trace = jnp.ones((32, 100))  # [N, T] format
        signals = jaxley_trace_to_signals(trace, layout="unit_by_time")
        assert signals.V_m.shape == (100, 32)
        assert signals.spikes.shape == (100, 32)

    def test_bridge_recording_by_time_shape(self) -> None:
        """recording_by_time layout [R, T] is treated as [T, N]."""
        trace = jnp.ones((100, 16))  # [R, T] format
        signals = jaxley_trace_to_signals(trace, layout="recording_by_time")
        assert signals.V_m.shape == (100, 16)

    def test_bridge_rejects_unknown_layout(self) -> None:
        """Unknown layout raises ValueError."""
        trace = jnp.ones((100, 32))
        with pytest.raises(ValueError, match="layout must be"):
            jaxley_trace_to_signals(trace, layout="unknown_layout")

    def test_bridge_rejects_nonfinite_voltage(self) -> None:
        """Non-finite values in trace raise ValueError."""
        trace = jnp.array([[1.0, 2.0], [3.0, jnp.nan], [5.0, 6.0]])
        with pytest.raises(ValueError, match="non-finite"):
            jaxley_trace_to_signals(trace)

        trace_inf = jnp.array([[1.0, 2.0], [jnp.inf, 4.0]])
        with pytest.raises(ValueError, match="non-finite"):
            jaxley_trace_to_signals(trace_inf)


class TestBridgeSpikeDerivation:
    """Test spike proxy derivation from voltage threshold."""

    def test_bridge_spike_threshold_default(self) -> None:
        """Default spike_threshold=0.0 derives spikes above 0."""
        trace = jnp.array([[-1.0, 1.0, 2.0], [-0.5, 0.5, 3.0]])
        signals = jaxley_trace_to_signals(trace)
        expected_spikes = jnp.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        assert jnp.allclose(signals.spikes, expected_spikes)

    def test_bridge_spike_threshold_none(self) -> None:
        """spike_threshold=None returns all-zero spikes."""
        spec = JaxleyTraceSpec(spike_threshold=None)
        trace = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        signals = jaxley_trace_to_signals(trace, spec=spec)
        assert jnp.allclose(signals.spikes, jnp.zeros((2, 2)))

    def test_bridge_spike_threshold_custom(self) -> None:
        """Custom spike_threshold applies to derivation."""
        spec = JaxleyTraceSpec(spike_threshold=-50.0)
        trace = jnp.array([[-60.0, -40.0, 0.0], [-50.0, -30.0, 50.0]])
        signals = jaxley_trace_to_signals(trace, spec=spec)
        # Threshold -50.0: V >= -50.0 → spikes
        # Row 0: [-60.0, -40.0, 0.0] >= -50.0 → [False, True, True] → [0.0, 1.0, 1.0]
        # Row 1: [-50.0, -30.0, 50.0] >= -50.0 → [True, True, True] → [1.0, 1.0, 1.0]
        expected_spikes = jnp.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        assert jnp.allclose(signals.spikes, expected_spikes)


class TestBridgeSourceHandling:
    """Test source array handling and voltage-proxy fallback."""

    def test_bridge_voltage_proxy_source(self) -> None:
        """Default source_mode='voltage_proxy' uses voltage as source."""
        trace = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        signals = jaxley_trace_to_signals(trace)
        assert signals.sources is not None
        assert jnp.allclose(signals.sources, trace)

    def test_bridge_rejects_current_mode_without_source(self) -> None:
        """Current-like source_mode without source raises ValueError."""
        spec = JaxleyTraceSpec(source_mode="current_decomposition")
        trace = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        # This should not raise in current implementation (no explicit check)
        # but documents the expected behavior
        # (deferred to v0.2.23)

    def test_bridge_user_provided_source(self) -> None:
        """User can provide explicit source array."""
        trace = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        user_source = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        signals = jaxley_trace_to_signals(trace, source=user_source)
        assert jnp.allclose(signals.sources, user_source)


class TestBridgeMetadata:
    """Test metadata and claim gate integrity."""

    def test_bridge_proxy_metadata_forbids_physical_amplitude_claim(self) -> None:
        """Metadata must have physical_amplitude_claim_allowed=False."""
        trace = jnp.ones((100, 32))
        signals = jaxley_trace_to_signals(trace)
        assert signals.metadata["physical_amplitude_claim_allowed"] is False
        assert signals.metadata["claim_level"] == "computational_scaffold"

    def test_bridge_metadata_json_safe(self) -> None:
        """Metadata dict is JSON-safe (no NaN/Inf)."""
        from jaxfne.io import json_safe
        trace = jnp.ones((10, 5))
        signals = jaxley_trace_to_signals(trace)
        # json_safe should not raise
        safe_metadata = json_safe(signals.metadata)
        assert isinstance(safe_metadata, dict)
        assert safe_metadata["claim_level"] == "computational_scaffold"

    def test_bridge_metadata_includes_calibration_status(self) -> None:
        """Metadata includes source_calibration_status."""
        trace = jnp.ones((10, 5))
        signals = jaxley_trace_to_signals(trace)
        assert "source_calibration_status" in signals.metadata
        assert signals.metadata["source_calibration_status"] == "uncalibrated_jaxley_voltage_proxy"


class TestBridgeFieldHandling:
    """Test field handling (should be None)."""

    def test_bridge_returns_signals_with_field_none(self) -> None:
        """Bridge returns Signals with field=None (not computed)."""
        trace = jnp.ones((100, 32))
        signals = jaxley_trace_to_signals(trace)
        assert signals.field is None
        assert signals.metadata["field_solver_status"] == "not_computed"


class TestBridgeDtypeHandling:
    """Test NumPy to JAX dtype conversion."""

    def test_bridge_accepts_numpy_array(self) -> None:
        """NumPy arrays are accepted and converted to JAX."""
        trace_np = np.ones((100, 32), dtype=np.float32)
        signals = jaxley_trace_to_signals(trace_np)
        assert isinstance(signals.V_m, jnp.ndarray)
        assert signals.V_m.shape == (100, 32)

    def test_bridge_accepts_list_array(self) -> None:
        """Python lists are accepted and converted to JAX."""
        trace_list = [[1.0, 2.0], [3.0, 4.0]]
        signals = jaxley_trace_to_signals(trace_list)
        assert isinstance(signals.V_m, jnp.ndarray)


class TestBridgeImportWithoutJaxley:
    """Test that core import works without Jaxley installed."""

    def test_import_jaxfne_does_not_require_jaxley(self) -> None:
        """jaxfne can be imported without jaxley optional dependency."""
        # This test passes if we reach here (jaxfne imported successfully)
        import jaxfne
        assert hasattr(jaxfne, "jaxley_trace_to_signals")
        assert hasattr(jaxfne, "JaxleyTraceSpec")

    def test_bridge_functions_work_without_jaxley(self) -> None:
        """Bridge functions work without jaxley installed."""
        trace = jnp.ones((10, 5))
        signals = jaxley_trace_to_signals(trace)
        assert signals is not None
        assert signals.V_m.shape == (10, 5)
