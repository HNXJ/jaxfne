"""Tests for optional dependency checks.

Validates that missing optional dependencies raise clean, descriptive ImportErrors.
"""

import sys
from unittest.mock import patch
import pytest

from jaxfne.bridges import require_jaxley


def test_require_jaxley_raises_error_when_missing():
    """Verify require_jaxley raises ImportError when jaxley is missing in env."""
    # Mask jaxley to force an ImportError
    with patch.dict(sys.modules, {"jaxley": None}):
        with pytest.raises(ImportError, match="requires optional dependency 'jaxley'"):
            require_jaxley()


def test_hh_jaxley_reference_trace_optional_dependency():
    """Verify that hh_jaxley_reference_trace raises optional dependency error when jaxley is absent."""
    with patch.dict(sys.modules, {"jaxley": None}):
        with pytest.raises(ImportError, match="requires optional dependency 'jaxley'"):
            from jaxfne.bridges import hh_jaxley_reference_trace
            hh_jaxley_reference_trace(duration_ms=10.0, dt_ms=0.1)


def test_hh_numpy_reference_trace_runs_cleanly():
    """Verify that hh_numpy_reference_trace runs cleanly and returns expected shapes without Jaxley."""
    from jaxfne.bridges import hh_numpy_reference_trace
    t, V, I_inj = hh_numpy_reference_trace(duration_ms=10.0, dt_ms=0.1)
    
    assert t.ndim == 1
    assert V.ndim == 1
    assert I_inj.ndim == 1
    assert len(t) == 100
    assert len(V) == 100
    assert len(I_inj) == 100
