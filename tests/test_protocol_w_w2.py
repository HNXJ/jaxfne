"""Protocol W2 — frozen-omega parameter expression."""

from __future__ import annotations

import numpy as np
import pytest

from jaxfne.w2_parameter_expression import (
    FROZEN_W2_CONFIG,
    W2FrozenOmegaState,
    run_w2_expression,
    run_w2_monotonic_sweep,
    run_w2_w1b_memory_contrast,
)


def _cfg() -> W2ProtocolConfig:
    return FROZEN_W2_CONFIG


def test_omega_zero_gives_baseline_weight():
    w0 = 6.0
    omega = W2FrozenOmegaState(omega_ab=0.0, w0_ab=w0)
    out = run_w2_expression(omega, cfg=_cfg(), seed=1)
    assert out["effective_weight_ab"] == pytest.approx(w0)


def test_weight_product_symmetry():
    w0 = 6.0
    a = 0.25
    pos = W2FrozenOmegaState(omega_ab=a, w0_ab=w0)
    neg = W2FrozenOmegaState(omega_ab=-a, w0_ab=w0)
    assert pos.magnitude_ab() * neg.magnitude_ab() == pytest.approx(w0**2)


def test_omega_frozen_constant_in_receipt():
    omega = W2FrozenOmegaState(omega_ab=0.17, omega_ba=-0.03, w0_ab=6.0, w0_ba=6.0)
    out = run_w2_expression(omega, cfg=_cfg(), seed=2)
    assert out["omega_state"].omega_ab == pytest.approx(0.17)
    assert out["omega_state"].omega_ba == pytest.approx(-0.03)


def test_excitatory_monotonic_response():
    sweep = run_w2_monotonic_sweep(_cfg(), seed=3, sign_ab=1)
    r_neg = sweep["results"][-0.25]["response_b"]
    r_zero = sweep["results"][0.0]["response_b"]
    r_pos = sweep["results"][0.25]["response_b"]
    assert r_neg < r_zero < r_pos
    assert sweep["results"][-0.25]["effective_weight_ab"] < sweep["results"][0.0]["effective_weight_ab"]
    assert sweep["results"][0.0]["effective_weight_ab"] < sweep["results"][0.25]["effective_weight_ab"]


def test_inhibitory_sign_increases_inhibition_magnitude():
    w0 = 6.0
    a = 0.25
    low = run_w2_expression(
        W2FrozenOmegaState(omega_ab=-a, w0_ab=w0),
        cfg=_cfg(),
        seed=4,
        sign_ab=-1,
    )
    high = run_w2_expression(
        W2FrozenOmegaState(omega_ab=a, w0_ab=w0),
        cfg=_cfg(),
        seed=4,
        sign_ab=-1,
    )
    assert low["effective_weight_ab"] < 0.0
    assert high["effective_weight_ab"] < low["effective_weight_ab"]
    assert high["response_b"] > low["response_b"]


def test_w1b_derived_memory_differs_from_null():
    out = run_w2_w1b_memory_contrast(_cfg(), seed=5)
    assert abs(out["omega_star"]) > 1e-6
    r_mem = out["memory"]["response_b"]
    r_null = out["null"]["response_b"]
    assert r_mem != pytest.approx(r_null)
    np.testing.assert_array_equal(
        out["memory"]["drive_schedule"],
        out["null"]["drive_schedule"],
    )
