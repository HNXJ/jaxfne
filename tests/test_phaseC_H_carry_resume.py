"""Phase C-04: HDP H(t) carry across pause/resume at 2000-step scale.

Gates the gap found in C-03: the 200-step kernel-level chunk-resume proof
(test_hdp_kernel_standalone.py::test_init_state_resume_matches_full_run)
exists, but no 2000-step reference with explicit H byte-equality and a
homeostasis-null control.

Positive continuity case: a continuous 2000-step run must be byte-identical
to two 1000-step chunks chained via init_state, including the H_trace, V,
spikes, and w_final -- not merely allclose. allclose is asserted separately
with a loose atol (1e-5), and byte identity with jnp.array_equal; the two
are different claims and must not be conflated.

Null control: K_HDP=0.0 -- the HDP kernel's actual homeostasis-disable
mechanism (its docstring: "K_HDP=0 disables HDP outright"), which is the
"k_gain=0" analog for this kernel. With the alpha=beta=gamma=delta=C_spike=
K_ctrl=0.0 defaults, H_i stays pinned at its 1.0 initial value forever, so
the null asserts H constant at 1.0 in the continuous run, chunk 1, and
chunk 2, and array-equality of the resumed vs continuous null H traces.

Determinism: noise_scale=0.0 (bulk noise is generated per-call from the
split key; only a zero coefficient makes the one-chunk and two-chunk paths
consume equivalent deterministic conditions -- see the handoff's noise
caveat).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jaxfne.emitters import (
    IzhikevichParams, EdgeList,
    simulate_edge_recurrent_izhikevich_hdp as hdp_kernel,
)

N, NE = 32, 256
DT_MS = 0.5
N_STEPS_FULL = 2000
N_STEPS_CHUNK = 1000

# Positive-case gains, matching the verified scaffold in
# test_hdp_kernel_standalone.py (nonzero gains move H and weights).
KW = dict(
    alpha=0.05, gamma=0.5, K_ctrl=0.15, K_HDP=0.01,
    barrier_c=0.01, barrier_d=0.01, tau_0_ms=5.0,
    w_floor=0.01, w_ceiling=10.0, noise_scale=0.0,
)


def _make_params_edges():
    """IzhikevichParams + EdgeList scaffold, mirroring
    test_hdp_kernel_standalone.py::_make_params_edges (private there; the
    repo convention is to inline fixtures rather than import between test
    modules). Weight signs follow the kernel's convention (receptor_index 0
    -> positive E, 1 -> negative I) so a K_HDP=0 null is a true no-op."""
    rng = np.random.default_rng(0)
    p = IzhikevichParams(
        a=jnp.full((N,), 0.02), b=jnp.full((N,), 0.2), c=jnp.full((N,), -65.0),
        d=jnp.full((N,), 8.0), drive=jnp.full((N,), 6.0), sign=jnp.ones((N,)),
        W=jnp.zeros((N, N)), v0=jnp.full((N,), -65.0), u0=jnp.full((N,), -13.0),
        source_scale=jnp.ones((N,)), labels=tuple("E" for _ in range(N)),
        layer_labels=tuple("L4" for _ in range(N)), source_calibration_status="x")
    receptor_index = rng.integers(0, 2, NE)
    weight_mag = np.abs(rng.normal(0, 0.3, NE)).astype(np.float32)
    weight = np.where(receptor_index == 0, weight_mag, -weight_mag).astype(np.float32)
    edges = EdgeList(
        pre=jnp.asarray(rng.integers(0, N, NE), jnp.int32),
        post=jnp.asarray(rng.integers(0, N, NE), jnp.int32),
        weight=jnp.asarray(weight),
        receptor_index=jnp.asarray(receptor_index, jnp.int32),
        tau_ms=jnp.full((NE,), 5.0), source_calibration_status="x")
    return p, edges


def _assert_six_final_state_keys(diag):
    """The diagnostics dict returned by the kernel must carry all six
    final-state fields required to seed init_state."""
    for name in ("v", "u", "prev_spikes", "syn_state", "H_final", "w_final"):
        assert name in diag, f"missing final-state key {name!r}"


def test_2000_step_resume_matches_continuous_byte_identically():
    """One 2000-step run == two 1000-step chunks chained via init_state:
    H_trace, V, and spikes byte-identical (jnp.array_equal), H allclose,
    w_final byte-identical, H finite throughout."""
    p, edges = _make_params_edges()
    key = jax.random.PRNGKey(5)

    # (1) one continuous 2000-step run
    V_full, S_full, _, d_full = hdp_kernel(p, edges, N_STEPS_FULL, DT_MS, key, **KW)
    H_full = d_full["H_trace"]

    # (2) chunk 1: 1000 steps
    V1, S1, _, d1 = hdp_kernel(p, edges, N_STEPS_CHUNK, DT_MS, key, **KW)
    H1 = d1["H_trace"]
    _assert_six_final_state_keys(d1)

    # (3) chunk 2: 1000 steps seeded from chunk 1's returned final state
    init_state = {name: d1[name] for name in
                  ("v", "u", "prev_spikes", "syn_state", "H_final", "w_final")}
    V2, S2, _, d2 = hdp_kernel(p, edges, N_STEPS_CHUNK, DT_MS, key,
                               init_state=init_state, **KW)
    H2 = d2["H_trace"]

    # Chunk 2 begins from chunk 1's H_final: the boundary value of the
    # continuous H trace (H after step 1000, i.e. index 999) is exactly
    # chunk 1's H_final, which is the H chunk 2 resumes from; chunk 2's
    # first recorded step therefore matches the continuous trace at index
    # 1000 (its next step after the boundary).
    assert jnp.array_equal(d1["H_final"], H_full[999])
    assert jnp.array_equal(init_state["H_final"], d1["H_final"])
    assert jnp.array_equal(H2[0], H_full[1000])
    assert jnp.array_equal(V2[0], V_full[1000])
    assert jnp.array_equal(S2[0], S_full[1000])

    # (4) continuous second 1000 steps vs chunk 2
    assert jnp.allclose(H_full[-N_STEPS_CHUNK:], H2, atol=1e-5)
    assert jnp.array_equal(H_full[-N_STEPS_CHUNK:], H2)
    assert jnp.array_equal(V_full[-N_STEPS_CHUNK:], V2)
    assert jnp.array_equal(S_full[-N_STEPS_CHUNK:], S2)
    assert jnp.array_equal(d_full["w_final"], d2["w_final"])

    # H finite everywhere; chunk bookkeeping shapes
    assert bool(jnp.isfinite(H_full).all())
    assert bool(jnp.isfinite(H1).all())
    assert bool(jnp.isfinite(H2).all())
    assert H_full.shape == (N_STEPS_FULL, N)
    assert H1.shape == (N_STEPS_CHUNK, N)
    assert H2.shape == (N_STEPS_CHUNK, N)


def test_null_control_K_HDP_zero_holds_H_at_1_resume_matches_continuous():
    """K_HDP=0.0 (the HDP kernel's 'k_gain=0' null mechanism) with the
    default zero homeostasis gains pins H_i at 1.0 forever; the resumed
    two-chunk H trace is array-equal to the continuous second half."""
    p, edges = _make_params_edges()
    key = jax.random.PRNGKey(5)
    kw_null = dict(K_HDP=0.0, noise_scale=0.0)

    V_full, S_full, _, d_full = hdp_kernel(p, edges, N_STEPS_FULL, DT_MS, key, **kw_null)
    H_full = d_full["H_trace"]

    V1, S1, _, d1 = hdp_kernel(p, edges, N_STEPS_CHUNK, DT_MS, key, **kw_null)
    H1 = d1["H_trace"]
    _assert_six_final_state_keys(d1)

    init_state = {name: d1[name] for name in
                  ("v", "u", "prev_spikes", "syn_state", "H_final", "w_final")}
    V2, S2, _, d2 = hdp_kernel(p, edges, N_STEPS_CHUNK, DT_MS, key,
                               init_state=init_state, **kw_null)
    H2 = d2["H_trace"]

    # H constant at 1.0 throughout: continuous, chunk 1, chunk 2
    assert bool(jnp.allclose(H_full, 1.0, atol=1e-6))
    assert bool(jnp.allclose(H1, 1.0, atol=1e-6))
    assert bool(jnp.allclose(H2, 1.0, atol=1e-6))
    # Weights untouched by the null (K_HDP=0 disables HDP outright)
    assert jnp.array_equal(d_full["w_final"], edges.weight)
    assert jnp.array_equal(d1["w_final"], edges.weight)
    assert jnp.array_equal(d2["w_final"], edges.weight)
    # Resumed and continuous null-control H traces array-equal
    assert jnp.array_equal(H_full[-N_STEPS_CHUNK:], H2)
    # V/spikes byte-identity holds under the null too
    assert jnp.array_equal(V_full[-N_STEPS_CHUNK:], V2)
    assert jnp.array_equal(S_full[-N_STEPS_CHUNK:], S2)
    # H finite everywhere
    assert bool(jnp.isfinite(H_full).all())
    assert bool(jnp.isfinite(H1).all())
    assert bool(jnp.isfinite(H2).all())
