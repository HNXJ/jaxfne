"""Falsifiable test of the barrier-potential equilibrium condition derived in
docs/guides/hdp.md ("Barrier equilibrium"): the minimum of
C(H) = barrier_c/(H-H_min) + barrier_d/(H_max-H) sits exactly at H*=1 only
when barrier_d/barrier_c = ((H_max-1)/(1-H_min))**2 -- a property of the
barrier potential alone, isolated here from every other dH/dt term
(alpha=beta=gamma=delta=K_ctrl=rho_passive=0, so the barrier is the only
active force). This is a numerical-methods claim, independent of any
biological validation: given only barrier_c/barrier_d and the H_min/H_max
clamps, the resulting pure-barrier fixed point is exactly predictable.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jaxfne.emitters import (
    IzhikevichParams, EdgeList,
    simulate_edge_recurrent_izhikevich_hdp as hdp_kernel,
)


def _make_params_edges(N: int, ne: int, H0: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = IzhikevichParams(
        a=jnp.full((N,), 0.02), b=jnp.full((N,), 0.2), c=jnp.full((N,), -65.0),
        d=jnp.full((N,), 8.0), drive=jnp.full((N,), 6.0), sign=jnp.ones((N,)),
        W=jnp.zeros((N, N)), v0=jnp.full((N,), -65.0), u0=jnp.full((N,), -13.0),
        source_scale=jnp.ones((N,)), labels=tuple("E" for _ in range(N)),
        layer_labels=tuple("L4" for _ in range(N)), source_calibration_status="x")
    receptor_index = rng.integers(0, 2, ne)
    weight_mag = np.abs(rng.normal(0, 0.3, ne)).astype(np.float32)
    weight = np.where(receptor_index == 0, weight_mag, -weight_mag).astype(np.float32)
    edges = EdgeList(
        pre=jnp.asarray(rng.integers(0, N, ne), jnp.int32),
        post=jnp.asarray(rng.integers(0, N, ne), jnp.int32),
        weight=jnp.asarray(weight),
        receptor_index=jnp.asarray(receptor_index, jnp.int32),
        tau_ms=jnp.full((ne,), 5.0), source_calibration_status="x")
    init_state = {
        "v": p.v0, "u": p.u0,
        "prev_spikes": jnp.zeros((N,), dtype=jnp.float32),
        "syn_state": jnp.zeros((ne,), dtype=jnp.float32),
        "H_final": jnp.full((N,), H0, dtype=jnp.float32),
        "w_final": edges.weight,
    }
    return p, edges, init_state


# Pure-barrier kwargs: every other dH/dt term held at its null so the barrier
# is the sole active force. size_scale_override=1.0 makes tau_i = tau_0_ms
# directly (no cube-law confound).
_NULL_OTHER_TERMS = dict(
    alpha=0.0, beta=0.0, gamma=0.0, delta=0.0, C_spike=0.0,
    K_ctrl=0.0, rho_passive=0.0, K_HDP=0.0, noise_scale=0.0,
)


def _relax(barrier_c, barrier_d, H0, n_steps=4000, dt_ms=0.5, tau_0_ms=5.0, N=8, ne=32):
    p, edges, init_state = _make_params_edges(N, ne, H0)
    key = jax.random.PRNGKey(0)
    size_override = jnp.ones((N,), dtype=jnp.float32)
    _, _, _, diag = hdp_kernel(
        p, edges, n_steps, dt_ms, key,
        barrier_c=barrier_c, barrier_d=barrier_d, tau_0_ms=tau_0_ms,
        size_scale_override=size_override, init_state=init_state,
        **_NULL_OTHER_TERMS,
    )
    H_trace = np.asarray(diag["H_trace"])
    assert bool(np.isfinite(H_trace).all())
    return float(H_trace[-1].mean())


def test_ratio_100_relaxes_to_H_star_1_from_above():
    """barrier_d/barrier_c = 100 (the derived condition at H_min=0.1/H_max=10.0)
    -> starting away from 1 relaxes back to 1, confirming the barrier ALONE
    creates the H*=1 equilibrium when the ratio matches the derivation."""
    H_final = _relax(barrier_c=0.01, barrier_d=1.0, H0=3.0)
    assert abs(H_final - 1.0) < 0.05, f"expected H->1.0 at ratio=100, got {H_final}"


def test_ratio_100_relaxes_to_H_star_1_from_below():
    """Same equilibrium approached from the other side (H0 < 1), confirming
    it's a genuine attracting fixed point, not an artifact of approach direction."""
    H_final = _relax(barrier_c=0.01, barrier_d=1.0, H0=0.3)
    assert abs(H_final - 1.0) < 0.05, f"expected H->1.0 at ratio=100, got {H_final}"


def test_ratio_1_shipped_default_does_not_relax_to_1():
    """DEFAULT_HDP's shipped barrier_c=barrier_d=0.01 (ratio 1) does NOT
    satisfy the H*=1 condition -- solving barrier_c/(H-H_min)**2 =
    barrier_d/(H_max-H)**2 with equal coefficients gives H=(H_min+H_max)/2
    -style solution -> H=5.05 at H_min=0.1/H_max=10.0. This is the concrete,
    falsifiable claim that DEFAULT_HDP's real stabilization comes from
    K_ctrl, not the barrier (see docs/guides/hdp.md). The equilibrium
    LOCATION depends only on the barrier_d/barrier_c RATIO, not the absolute
    magnitude (the force-balance equation cancels a common scale factor) --
    barrier_c=barrier_d=1.0 (ratio 1, same as the shipped 0.01/0.01) is used
    here purely to converge within a practical step budget; DEFAULT_HDP's
    literal 0.01/0.01 magnitude converges to the identical 5.05 point, just
    far more slowly given tau_i=5ms and n_steps=4000 (verified separately,
    not re-asserted here to keep this test fast)."""
    H_final = _relax(barrier_c=1.0, barrier_d=1.0, H0=3.0)
    assert abs(H_final - 1.0) > 1.0, (
        f"expected H to NOT relax near 1.0 at ratio=1 (shipped DEFAULT_HDP barrier "
        f"ratio), got {H_final} -- if this now converges near 1.0, either the "
        f"barrier force sign/formula changed or the ratio assumption changed; "
        f"re-derive before trusting the doc's claim.")
    assert abs(H_final - 5.05) < 0.1, f"expected H->~5.05 at ratio=1, got {H_final}"
