"""Tests for jaxfne/units.py -- dtype-keyed epsilon/dither defaults."""
import warnings

import jax.numpy as jnp
import pytest

from jaxfne import units


@pytest.mark.parametrize("dtype", ["float32", "float64", "bfloat16"])
def test_epsilon_known_dtypes(dtype):
    eps = units.epsilon(dtype)
    assert eps > 0.0
    assert jnp.isfinite(eps)


def test_epsilon_ordering_reflects_precision():
    # float64 tightest, float32 middle, bfloat16 loosest.
    assert units.epsilon("float64") < units.epsilon("float32") < units.epsilon("bfloat16")


@pytest.mark.parametrize("dtype", ["float32", "float64", "bfloat16"])
def test_dither_scale_known_dtypes(dtype):
    d = units.dither_scale(dtype)
    assert d > 0.0
    assert jnp.isfinite(d)


def test_dither_scale_ordering_reflects_precision():
    assert units.dither_scale("float64") < units.dither_scale("float32") < units.dither_scale("bfloat16")


def test_epsilon_accepts_jnp_dtype_object():
    assert units.epsilon(jnp.float32) == units.epsilon("float32")
    assert units.epsilon(jnp.bfloat16) == units.epsilon("bfloat16")


def test_unknown_dtype_raises():
    with pytest.raises(ValueError, match="No epsilon/dither default"):
        units.epsilon("int32")
    with pytest.raises(ValueError, match="No epsilon/dither default"):
        units.dither_scale("float16")


def test_warn_dt_dtype_mismatch_fires_for_too_fine_dt_at_bf16():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        units.warn_dt_dtype_mismatch(1.0e-6, "bfloat16", quantity_name="dt_ms")
        assert any("dt_ms" in str(w.message) for w in caught)
        assert any("not being auto-adjusted" in str(w.message) for w in caught)


def test_warn_dt_dtype_mismatch_silent_for_reasonable_dt_at_float32():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        units.warn_dt_dtype_mismatch(0.5, "float32", quantity_name="dt_ms")
        assert len(caught) == 0


def test_warn_dt_dtype_mismatch_does_not_mutate_dt():
    # Never auto-adjusts -- the function has no return value and takes dt_ms by value.
    dt_ms = 1.0e-6
    result = units.warn_dt_dtype_mismatch(dt_ms, "bfloat16")
    assert result is None
    assert dt_ms == 1.0e-6
