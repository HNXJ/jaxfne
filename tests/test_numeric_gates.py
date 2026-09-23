"""Unit tests for the Item-10 corrected cross-implementation gate + diagnostics determinism."""

from __future__ import annotations

import numpy as np
import pytest

from _numeric_gates import assert_cross_impl_close, cross_impl_close


def test_abs_leg_governs_near_zero():
    assert cross_impl_close(0.0, 5e-9)
    assert not cross_impl_close(0.0, 5e-8)


def test_rel_leg_governs_order_one():
    # 2.4e-07 at |v|~1: typical cross-implementation rounding, must pass.
    assert cross_impl_close(0.97649205, 0.97649229)
    assert not cross_impl_close(1.0, 1.0001)


def test_nan_never_close():
    assert not cross_impl_close(float("nan"), 0.0)
    with pytest.raises(AssertionError):
        assert_cross_impl_close(1.0, 1.0001, label="probe")


def test_conservation_diagnostics_deterministic():
    """Same input twice -> identical output (deterministic regression guard)."""
    from jaxfne import compute_conservation_proxy_diagnostics

    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 10)).astype(np.float32)
    d1 = compute_conservation_proxy_diagnostics(source=src)
    d2 = compute_conservation_proxy_diagnostics(source=np.array(src, copy=True))
    assert list(d1.keys()) == list(d2.keys())
    for k, v1 in d1.items():
        v2 = d2[k]
        if isinstance(v1, float):
            assert v1 == v2, k
        else:
            assert v1 == v2 or (v1 is None and v2 is None), k
