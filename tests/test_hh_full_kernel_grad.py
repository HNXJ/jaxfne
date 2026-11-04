"""jax.grad through the full HH kernel, checked against a central difference.

The HH unit tests differentiate one stepper (``graded_syn_step``). This test
differentiates a loss through ``simulate_hh`` over 3000 steps, so the membrane,
the gates m/h/n and the synaptic gating s all lie on the path. The model is built
through the public grammar (``Configuration`` with the hh emitter, then
``construct``); its kernel arguments come from ``hh_kernel_args``, the call
``Model.simulate()`` makes.

Differentiated input: a scalar synaptic weight scale on ``edge_weight``, which
``_check_edges`` accepts as a tracer. Construction and ``Model.simulate()`` take
no differentiable scalar, so the scale enters at the kernel call.

Loss: mean membrane voltage over all cells and steps (no spike term, so smooth).
Dtype: float32 (x64 off, the conftest default); step 1e-2 relative, rtol 5e-2.
CPU-only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._construct_hh import hh_kernel_args
from jaxfne.emitters_hh import simulate_hh

DT_MS = 0.1
N_STEPS = 3000
DRIVE_UA_CM2 = 2.0  # weak: below the repetitive-firing onset
H_REL = 1e-2
RTOL = 5e-2


def _kernel_args():
    cfg = (
        jtfne.Configuration()
        .network(name="mini", n=3, cell_types={"E": 2.0 / 3.0, "Ig": 1.0 / 3.0})
        .emitter(family="hh")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=None, weight=4.0, mechanism="ampa")
    ).connectivity(edge_seed=0)
    return hh_kernel_args(jtfne.construct(cfg), DT_MS)


def _mean_v(scale, kwargs, drive):
    kw = dict(kwargs)
    kw["edge_weight"] = scale * jnp.asarray(kwargs["edge_weight"])
    out = simulate_hh(**kw, n_steps=N_STEPS, dt_ms=DT_MS, I_ext=drive, record_v=True)
    return jnp.mean(out["V_mV"])


def test_full_kernel_grad_matches_central_difference():
    kwargs = _kernel_args()
    drive = np.full((N_STEPS, 3), DRIVE_UA_CM2, dtype=np.float32)

    def f(scale):
        return _mean_v(scale, kwargs, drive)

    s0 = jnp.float32(1.0)
    grad = float(jax.grad(f)(s0))
    loss0 = float(f(s0))
    s_plus = jnp.float32(1.0 + H_REL)
    s_minus = jnp.float32(1.0 - H_REL)
    h = float(s_plus) - float(s_minus)  # the step float32 actually realizes
    fd = (float(f(s_plus)) - float(f(s_minus))) / h

    # Preconditions: a finite, nonzero gradient, and a finite-difference slope
    # above float32 roundoff (eps * |loss| / step).
    noise = np.finfo(np.float32).eps * max(abs(loss0), 1.0) / h
    assert np.isfinite(grad) and grad != 0.0, f"grad must be finite and nonzero; got {grad!r}"
    assert np.isfinite(fd) and abs(fd) > 10.0 * noise, (
        f"FD must exceed float32 roundoff {noise:.3g}; got {fd!r}")
    np.testing.assert_allclose(grad, fd, rtol=RTOL, atol=0.0,
                               err_msg="grad vs central difference through simulate_hh")


@pytest.mark.x64
def test_full_kernel_runs_under_x64():
    # The held-step counters in the scan carry start as int32; a sum under x64 promotes to int64.
    assert jax.config.read("jax_enable_x64"), "fixture must enable x64"
    kwargs = dict(_kernel_args())
    kwargs.update(ou_tau=np.full(3, 5.0), ou_sigma=np.full(3, 0.5), ou_key=jax.random.PRNGKey(0))
    out = simulate_hh(**kwargs, n_steps=100, dt_ms=DT_MS,
                      I_ext=np.full((100, 3), DRIVE_UA_CM2), record_v=True)
    assert np.all(np.isfinite(np.asarray(out["V_mV"])))
    assert np.any(np.asarray(out["ou"]) != 0.0), "fixture must run the OU noise path"
    assert int(out["held_steps"]["s"]) == 0 and int(out["held_steps"]["ou"]) == 0
