"""Tests for jaxfne.is_valid_signal() public API.

Validates that the signal validation helper works with both Signals objects
and dict-like signal containers, and correctly identifies finite/invalid arrays.
"""

import jax.numpy as jnp
import pytest
import jaxfne as jtfne


class TestIsValidSignalWithSignalsObject:
    """Test is_valid_signal with actual Signals dataclass."""

    def test_valid_signals_from_simulate(self):
        """Valid signals from a real simulation should pass."""
        cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)

        assert jtfne.is_valid_signal(signals) is True

    def test_signals_with_nan_in_vm(self):
        """Signals with NaN in V_m should fail."""
        # Create a mock signals-like object with NaN
        class MockSignals:
            def __init__(self):
                self.V_m = jnp.array([[1.0, jnp.nan], [2.0, 3.0]])
                self.spikes = jnp.zeros((2, 2))

        signals = MockSignals()
        assert jtfne.is_valid_signal(signals) is False

    def test_signals_with_inf_in_spikes(self):
        """Signals with Inf in spikes should fail."""
        class MockSignals:
            def __init__(self):
                self.V_m = jnp.zeros((2, 2))
                self.spikes = jnp.array([[0.0, jnp.inf], [0.0, 1.0]])

        signals = MockSignals()
        assert jtfne.is_valid_signal(signals) is False


class TestIsValidSignalWithDictInput:
    """Test is_valid_signal with dict-like signal containers."""

    def test_valid_dict_signals(self):
        """Valid dict with V_m and spikes should pass."""
        signals_dict = {
            "V_m": jnp.zeros((10, 5)),
            "spikes": jnp.ones((10, 5))
        }
        assert jtfne.is_valid_signal(signals_dict) is True

    def test_dict_with_nan(self):
        """Dict with NaN should fail."""
        signals_dict = {
            "V_m": jnp.array([1.0, jnp.nan, 3.0]),
            "spikes": jnp.zeros((3,))
        }
        assert jtfne.is_valid_signal(signals_dict) is False

    def test_dict_with_inf(self):
        """Dict with Inf should fail."""
        signals_dict = {
            "V_m": jnp.zeros((3,)),
            "spikes": jnp.array([0.0, jnp.inf, 0.0])
        }
        assert jtfne.is_valid_signal(signals_dict) is False

    def test_dict_missing_vm(self):
        """Dict missing V_m should fail."""
        signals_dict = {"spikes": jnp.zeros((3,))}
        assert jtfne.is_valid_signal(signals_dict) is False

    def test_dict_missing_spikes(self):
        """Dict missing spikes should fail."""
        signals_dict = {"V_m": jnp.zeros((3,))}
        assert jtfne.is_valid_signal(signals_dict) is False


class TestIsValidSignalEdgeCases:
    """Test edge cases and error handling."""

    def test_none_input(self):
        """None input should return False."""
        assert jtfne.is_valid_signal(None) is False

    def test_invalid_type(self):
        """Invalid input type should return False."""
        assert jtfne.is_valid_signal("not a signal") is False
        assert jtfne.is_valid_signal(123) is False
        assert jtfne.is_valid_signal([1.0, 2.0]) is False

    def test_object_missing_attributes(self):
        """Object with missing signal attributes should fail."""
        class PartialSignals:
            def __init__(self):
                self.V_m = jnp.zeros((3,))

        signals = PartialSignals()
        assert jtfne.is_valid_signal(signals) is False

    def test_object_with_none_arrays(self):
        """Object with None arrays should handle gracefully."""
        class BrokenSignals:
            def __init__(self):
                self.V_m = None
                self.spikes = jnp.zeros((3,))

        signals = BrokenSignals()
        assert jtfne.is_valid_signal(signals) is False

    def test_empty_arrays(self):
        """Empty but finite arrays should pass."""
        signals_dict = {
            "V_m": jnp.array([]),
            "spikes": jnp.array([])
        }
        # isfinite on empty array returns empty array, all() of empty is True
        assert jtfne.is_valid_signal(signals_dict) is True

    def test_single_element_arrays(self):
        """Single-element finite arrays should pass."""
        signals_dict = {
            "V_m": jnp.array([1.5]),
            "spikes": jnp.array([0.0])
        }
        assert jtfne.is_valid_signal(signals_dict) is True

    def test_large_finite_arrays(self):
        """Large arrays with all finite values should pass."""
        signals_dict = {
            "V_m": jnp.ones((1000, 1000)),
            "spikes": jnp.zeros((1000, 1000))
        }
        assert jtfne.is_valid_signal(signals_dict) is True


class TestIsValidSignalExportedInPublicAPI:
    """Verify is_valid_signal is properly exported."""

    def test_in_public_api(self):
        """is_valid_signal should be in jaxfne's public API."""
        assert hasattr(jtfne, "is_valid_signal")
        assert callable(jtfne.is_valid_signal)

    def test_in_all(self):
        """is_valid_signal should be in __all__."""
        assert "is_valid_signal" in jtfne.__all__


class TestIsValidSignalAliases:
    """Test is_valid_signal with vm/spk aliases."""

    def test_valid_dict_signals_aliases(self):
        """Valid dict with vm and spk should pass."""
        signals_dict = {
            "vm": jnp.zeros((10, 5)),
            "spk": jnp.ones((10, 5))
        }
        assert jtfne.is_valid_signal(signals_dict) is True

    def test_valid_object_signals_aliases(self):
        """Valid object with vm and spk should pass."""
        class MockSignals:
            def __init__(self):
                self.vm = jnp.zeros((5,))
                self.spk = jnp.zeros((5,))

        signals = MockSignals()
        assert jtfne.is_valid_signal(signals) is True

