"""Opt-in membrane floor (``v_floor``) on the registrable HDP kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered as sim
from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.hdp_rule import (
    HDPRuleDescriptor,
    HDPRuleUpdate,
    list_registered_hdp_rules,
    register_hdp_rule,
)

_RULE = "test_null_wide_bounds"
if _RULE not in list_registered_hdp_rules():
    # No plasticity; wide weight bounds so strong inhibition is not clipped to the default ceiling.
    register_hdp_rule(
        HDPRuleDescriptor(name=_RULE, w_bounds=(0.0, 1e9)),
        lambda ctx: HDPRuleUpdate(dH=jnp.zeros_like(ctx.H)),
    )


def _run(w_inh, delay, **kw):
    def f(x):
        return jnp.full((2,), x, jnp.float32)

    params = IzhikevichParams(
        a=f(0.02),
        b=f(0.2),
        c=f(-65.0),
        d=f(8.0),
        drive=f(0.0),
        sign=f(1.0),
        W=jnp.zeros((2, 2)),
        v0=f(-65.0),
        u0=f(-13.0),
        source_scale=f(1.0),
        labels=("E", "E"),
        source_calibration_status="x",
    )
    edges = EdgeList(
        pre=jnp.array([0], jnp.int32),
        post=jnp.array([1], jnp.int32),
        weight=jnp.array([-w_inh], jnp.float32),
        receptor_index=jnp.array([1], jnp.int32),
        tau_ms=jnp.array([2.0], jnp.float32),
        source_calibration_status="x",
        delay_steps=jnp.array([delay], jnp.int32),
    )
    dt, n = 0.5, 80
    drive = jnp.zeros((n, 2), jnp.float32).at[:4, 0].set(30.0)  # neuron 0 fires once
    v, spikes, _, _ = sim(
        params,
        edges,
        n,
        dt,
        jax.random.PRNGKey(0),
        drive_schedule=drive,
        noise_scale=0.0,
        hdp_rule=_RULE,
        **kw,
    )
    return np.asarray(v), np.asarray(spikes)


@pytest.mark.parametrize("delay", [0, 3])
def test_v_floor_blocks_inhibition_induced_spikes(delay):
    _, s = _run(1000.0, delay)
    assert s[:, 0].sum() == 1 and s[:, 1].sum() > 0  # unclamped: inhibition alone fires the target
    v, s = _run(1000.0, delay, v_floor=-85.0)
    assert s[:, 1].sum() == 0 and v[:, 1].min() >= -85.0


@pytest.mark.parametrize("delay", [0, 3])
def test_non_binding_floor_is_bit_identical(delay):
    v0, s0 = _run(20.0, delay)
    v1, s1 = _run(20.0, delay, v_floor=-1e9)
    np.testing.assert_array_equal(v0, v1)
    np.testing.assert_array_equal(s0, s1)
