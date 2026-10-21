"""P3b arm: JAX transcription of the paper's net.m update order.

Spec: ``PREREG_P3b.md`` (signed v4), Arm section. One ``jax.lax.scan`` step
is 1 ms, in net.m's order: threshold test on the v left by the previous
step, reset of the fired cells, ``I = thal[t] + S @ fired`` with no other
drive, two 0.5 ms forward-Euler v half-steps with the same I (no clamp, so
a first-half-step crossing overshoots in the second), then one u step on
the v after both half-steps. Row t of the output is ``fired`` from step 1
of step t. Extra u per spike is defined exactly as
``net_reference.simulate`` defines it: at each first-half-step crossing,
``a * b * (v_after_both_half_steps - 30)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _field(params, name):
    if isinstance(params, dict):
        return params[name]
    return getattr(params, name)


def simulate_net_m(params, S, thal, v0, u0, return_v=False):
    """Run the net.m-order update; dtype follows the inputs.

    ``params`` carries per-cell a, b, c, d (a dict with those keys, or an
    object with those attributes, e.g. the ``net`` dict from
    ``net_reference.build_network``). ``S`` is (n, n), ``thal`` is
    (n_steps, n), ``v0``/``u0`` are (n,). Returns a dict with ``spikes``
    (bool (n_steps, n)), ``extra_u`` and ``mean_extra_u_per_spike``; when
    ``return_v`` is set it also carries ``v_test`` (n_steps, n), the v each
    step's threshold test reads. The caller enables x64 before building any
    array when float64 is wanted.
    """
    a = jnp.asarray(_field(params, "a"))
    b = jnp.asarray(_field(params, "b"))
    c = jnp.asarray(_field(params, "c"))
    d = jnp.asarray(_field(params, "d"))
    mat = jnp.asarray(S)
    drive_thal = jnp.asarray(thal)
    v_init = jnp.asarray(v0)
    u_init = jnp.asarray(u0)
    v_dtype = v_init.dtype

    def _step(carry, thal_t):
        v, u = carry
        v_test = v
        fired = v >= 30.0
        v_r = jnp.where(fired, c, v)
        u_r = jnp.where(fired, u + d, u)
        cur = thal_t + jnp.matmul(mat, fired.astype(v_dtype))
        v1 = v_r + 0.5 * (0.04 * v_r * v_r + 5.0 * v_r + 140.0 - u_r + cur)
        will_fire = (v1 >= 30.0) & ~fired
        v2 = v1 + 0.5 * (0.04 * v1 * v1 + 5.0 * v1 + 140.0 - u_r + cur)
        extra_row = jnp.where(will_fire, a * b * (v2 - 30.0), jnp.zeros_like(v2))
        u_next = u_r + a * (b * v2 - u_r)
        return (v2, u_next), (fired, extra_row, will_fire, v_test)

    (_, _), (fired_rows, extra_rows, will_rows, vtest_rows) = jax.lax.scan(
        _step, (v_init, u_init), drive_thal)
    n_spikes = jnp.sum(will_rows)
    total = jnp.sum(jnp.where(will_rows, extra_rows, jnp.zeros_like(extra_rows)))
    mean_extra = float(total / n_spikes) if int(n_spikes) else 0.0
    out = {"spikes": fired_rows, "extra_u": extra_rows,
           "mean_extra_u_per_spike": mean_extra}
    if return_v:
        out["v_test"] = vtest_rows
    return out
