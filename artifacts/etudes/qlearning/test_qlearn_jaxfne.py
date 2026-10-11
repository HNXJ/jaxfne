"""Tests for the Q-learning etude (lane QL). Each test asserts its precondition."""

import numpy as np

from qlearn_jaxfne import (
    A_LEFT,
    A_RIGHT,
    GROUP,
    N_E,
    features_for_seed,
    greedy_policy,
    phi_for_seed_state,
    q_star,
    q_values,
    run_q_learning,
)


def test_1_tabular_baseline_matches_q_star():
    qs = q_star()
    assert abs(qs[0, A_RIGHT] - 0.9**3) < 1e-12, "Q* precondition"
    assert abs(qs[3, A_RIGHT] - 1.0) < 1e-12, "Q* precondition"
    phi = np.eye(5, dtype=np.float64)
    out = run_q_learning(phi, seed=0, alpha=0.1, epsilon=0.1, n_episodes=2000)
    assert out["visits"][:4, :].min() > 0, "every non-terminal (s, a) visited"
    err = float(np.abs(q_values(out["w"], phi) - qs).max())
    print(f"\ntabular max |Q - Q*| = {err:.6f}")
    assert err < 1e-2, f"tabular Q off by {err}"


def test_2_jaxfne_features_learn_right_policy_all_seeds():
    for seed in range(5):
        phi = features_for_seed(seed)
        assert all(phi[s].sum() > 0 for s in range(5)), "every phi(s) spikes"
        out = run_q_learning(phi, seed=seed, alpha=0.1, epsilon=0.1,
                             n_episodes=1000)
        ep = out["first_policy_ep"]
        # Q* is specified on the non-terminal states 0..3 (state 4 is
        # terminal and never visited, so linear Q generalizes there).
        err = float(np.abs(q_values(out["w"], phi[:4]) - q_star()[:4]).max())
        print(f"\nseed {seed}: episodes-to-policy = {ep}, max |Q - Q*| = {err:.6f}")
        assert ep is not None and ep <= 1000, f"seed {seed} never learned right"
        assert greedy_policy(out["w"], phi[:4]) == [A_RIGHT] * 4


def test_3_identical_features_cannot_learn_policy():
    phi = features_for_seed(0)
    flat = np.stack([phi.mean(axis=0)] * 5)
    assert all(np.array_equal(flat[s], flat[0]) for s in range(5)), "control is flat"
    out = run_q_learning(flat, seed=0, alpha=0.1, epsilon=0.1, n_episodes=500)
    policy = greedy_policy(out["w"], flat)
    print(f"\ncontrol policy = {policy}")
    assert len(set(policy)) == 1, "flat features force one action everywhere"


def test_4_two_seeds_give_different_network_features():
    phi0 = features_for_seed(0)
    phi1 = features_for_seed(1)
    assert all(phi0[s].sum() > 0 and phi1[s].sum() > 0 for s in range(5))
    assert float(np.abs(phi0 - phi1).max()) > 0, "seeds must differ"
    for phi in (phi0, phi1):  # driven E group carries the max: network-made code
        for s in range(5):
            top = set(np.flatnonzero(phi[s] == phi[s].max()).tolist())
            assert top <= set(range(s * GROUP, s * GROUP + GROUP)), (s, sorted(top))


def test_5_alpha_zero_leaves_weights_unchanged():
    rng = np.random.default_rng(7)
    phi = rng.random((5, 6))
    out = run_q_learning(phi, seed=0, alpha=0.0, epsilon=0.1, n_episodes=50)
    assert out["attempts"] > 0, "loop must run update attempts"
    assert np.array_equal(out["w"], np.zeros_like(out["w"])), "alpha=0 froze w"
