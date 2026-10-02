"""UNTESTED-exact rule-pair gate (0.5.5 stack item 5).

simulate_homeostatic_ei refuses named conductance x homeostasis pairings
outside the measured B-05/B-08 set. Tests pin the refusal list, the
allowed list, and the exemptions (callables, frozen axes).
"""

from __future__ import annotations

import jax
import pytest

from jaxfne.emitters_homeostatic_ei import (
    _MEASURED_RULE_PAIRS,
    _assert_measured_rule_pair,
    simulate_homeostatic_ei,
)

MEASURED = {
    ("hebbian", "linear"),
    ("hebbian", "logistic"),
    ("hebbian", "cubic_penalty"),
    ("hebbian", "cubic_penalty_coupled"),
    ("hebbian_pairwise", "linear"),
    ("hebbian_pairwise", "cubic_penalty"),
    ("linear", "cubic_penalty"),
    ("bcm", "cubic_penalty"),
}

UNMEASURED = sorted(
    set(
        (c, h)
        for c in ("hebbian", "hebbian_pairwise", "linear", "bcm")
        for h in ("linear", "logistic", "cubic_penalty", "cubic_penalty_coupled")
    )
    - MEASURED
)


def test_measured_set_is_the_b05_b08_cells():
    assert _MEASURED_RULE_PAIRS == MEASURED


def test_every_unmeasured_named_pairing_is_refused_by_name():
    for cond, homeo in UNMEASURED:
        with pytest.raises(ValueError, match=r"unmeasured conductance x homeostasis"):
            _assert_measured_rule_pair(cond, homeo)


def test_refusal_names_the_offending_pair_and_the_allowed_set():
    with pytest.raises(ValueError) as exc:
        _assert_measured_rule_pair("logistic", "linear")
    msg = str(exc.value)
    assert "('logistic' x 'linear')" in msg
    for cond, homeo in sorted(MEASURED):
        assert f"({cond} x {homeo})" in msg


def test_callables_bypass_the_gate_and_run():
    cond = lambda x, H, G, is_e: 0.1 * x[0]  # noqa: E731
    homeo = lambda x, H, is_e: 0.01 * x  # noqa: E731
    out = simulate_homeostatic_ei(
        _params(),
        n_steps=5,
        dt_ms=0.5,
        key=jax.random.PRNGKey(0),
        conductance_rule=cond,
        homeostasis_rule=homeo,
    )
    assert out[-1]["error"] in ("", None, False)


def _params():
    import jax.numpy as jnp

    from jaxfne.emitters_homeostatic_ei import HomeostaticEIParams

    return HomeostaticEIParams(
        x0=jnp.array([-60.0, -70.0]),
        G0=jnp.eye(2) * 0.1,
        H0=jnp.array([1.0, 1.0]),
        drive=jnp.array([5.0, 1.0]),
        tau_x_ms=jnp.array([10.0, 10.0]),
        tau_G_ms=jnp.array([50.0, 50.0]),
        tau_H_ms=jnp.array([500.0, 500.0]),
        G_min=jnp.array([0.0, 0.0]),
        G_max=jnp.array([10.0, 10.0]),
        H_min=jnp.array([0.1, 0.1]),
        H_max=jnp.array([10.0, 10.0]),
        source_scale=jnp.array([1.0, 1.0]),
    )


def test_refusal_reached_through_simulate_homeostatic_ei():
    # Kills a mutant that deletes the _assert_measured_rule_pair call site
    # inside simulate_homeostatic_ei (the direct-function tests above would
    # survive that mutant).
    with pytest.raises(ValueError, match=r"unmeasured conductance x homeostasis"):
        simulate_homeostatic_ei(
            _params(),
            n_steps=3,
            dt_ms=0.5,
            key=jax.random.PRNGKey(0),
            conductance_rule="bcm",
            homeostasis_rule="linear",
        )


def test_frozen_axes_do_not_gate():
    p = _params()
    # freeze_G=True: only the homeostasis axis runs; unmeasured (C,H) pair
    # must NOT refuse (stage-1-style surveys measure H alone)
    out = simulate_homeostatic_ei(
        p,
        n_steps=5,
        dt_ms=0.5,
        key=jax.random.PRNGKey(0),
        conductance_rule="linear",
        homeostasis_rule="logistic",
        freeze_G=True,
    )
    assert out[-1] is not None
    # freeze_H=True: only the conductance axis runs (G-adaptation tests)
    out = simulate_homeostatic_ei(
        p,
        n_steps=5,
        dt_ms=0.5,
        key=jax.random.PRNGKey(0),
        conductance_rule="bcm",
        homeostasis_rule="linear",
        freeze_H=True,
    )
    assert out[-1] is not None
