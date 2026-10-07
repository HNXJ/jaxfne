"""Balance étude: learning consumes eta and the modulator; frozen agents keep W bit-identical."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import artifacts.etudes.hdp_balance.balance as B


def _run(conds, eta):
    W, plastic, S0, etas, perm = B.build_batch([3] * len(conds), conds, eta)
    carry = B.init_carry(W, jax.random.PRNGKey(0))
    carry, _ = B.run_phase(carry, etas, perm, plastic, S0, n_bins=1, bin_steps=300)
    return np.asarray(W), np.asarray(carry[7]), np.asarray(plastic)


def test_frozen_unchanged_learning_changes_and_budget_held():
    W0, W1, plastic = _run(["frozen", "full", "shuffled", "shuffled"], 1.0)
    assert np.array_equal(W0[0], W1[0])
    for k in (1, 2):
        assert not np.array_equal(W0[k], W1[k])
        np.testing.assert_allclose((W1[k] * plastic[k]).sum(-1), (W0[k] * plastic[k]).sum(-1), rtol=1e-4)
        assert np.array_equal(W1[k][~plastic[k]], W0[k][~plastic[k]])


def test_shuffled_modulator_differs_from_own():
    _, Wf, _ = _run(["full", "full"], 1.0)
    _, Ws, _ = _run(["shuffled", "shuffled"], 1.0)
    assert not np.array_equal(Wf[0], Ws[0])


def test_wired_control_is_correct_sign():
    W, _ = B.init_network(0, wired=True)
    assert B.motor_map_index(W) > 0
