"""Tests for the Q-learning etude (lane QL). Each test asserts its precondition."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from qlearn_jaxfne import (  # noqa: E402
    ALPHA,
    A_LEFT,
    A_RIGHT,
    CORRIDOR_TASK,
    corridor_features,
    driven_cells,
    episodes_to_policy,
    features_for_seed,
    greedy_policy,
    max_normalised_for_seed_state,
    optimal_actions,
    policy_states,
    q_star,
    q_values,
    run_corridor,
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
        ep = out["episodes_to_policy"]
        # Q* is specified on the non-terminal states 0..3 (state 4 is
        # terminal and never visited, so linear Q generalizes there).
        err = float(np.abs(q_values(out["w"], phi[:4]) - q_star()[:4]).max())
        final = greedy_policy(out["w"], phi[:4])
        shown = "never" if ep is None else ep
        print(f"\nseed {seed}: episodes-to-policy = {shown}, final greedy = {final}, "
              f"max |Q - Q*| = {err:.6f}")
        assert ep is not None and ep <= 1000, f"seed {seed} never learned right"
        assert final == [A_RIGHT] * 4


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
    for phi in (phi0, phi1):  # the driven cells carry the max: network-made code
        for s in range(5):
            top = set(np.flatnonzero(phi[s] == phi[s].max()).tolist())
            assert top <= set(driven_cells(s)), (s, sorted(top))


def test_5_alpha_zero_leaves_weights_unchanged():
    rng = np.random.default_rng(7)
    phi = rng.random((5, 6))
    out = run_q_learning(phi, seed=0, alpha=0.0, epsilon=0.1, n_episodes=50)
    assert out["attempts"] > 0, "loop must run update attempts"
    assert np.array_equal(out["w"], np.zeros_like(out["w"])), "alpha=0 froze w"


def test_6_unit_norm_caps_alpha_phi_squared_at_alpha():
    # The max-normalised features of the earlier design put alpha * ||phi||^2 above 2
    # (divergence). The unit-norm features must keep every state at alpha or below.
    for seed in range(5):
        old = np.stack(
            [max_normalised_for_seed_state(seed, s) for s in range(5)]
        ).astype(np.float64)
        old_sq = ALPHA * (old ** 2).sum(axis=1)
        assert old_sq.max() > 2, f"precondition: seed {seed} old features stay under 2"
        phi = features_for_seed(seed).astype(np.float64)
        new_sq = ALPHA * (phi ** 2).sum(axis=1)
        assert np.all(new_sq <= ALPHA * (1 + 1e-6)), (seed, new_sq.tolist())


def test_7_episodes_to_policy_is_first_episode_of_the_final_correct_run():
    # Constructed histories (True = greedy policy correct after that episode).
    # First-hit would answer 1 for the first one; the definition answers 3.
    assert episodes_to_policy([True, False, True, True]) == 3
    assert episodes_to_policy([False, True, False, True, True]) == 4
    assert episodes_to_policy([True, True, True]) == 1
    assert episodes_to_policy([True, False]) is None, "last episode wrong: never"
    assert episodes_to_policy([]) is None
    # Wiring: one entry per episode; alpha = 0 keeps w at zero, so the greedy
    # policy (ties go left) is wrong at every episode and the value is "never".
    rng = np.random.default_rng(7)
    out = run_q_learning(rng.random((5, 6)), seed=0, alpha=0.0, epsilon=0.1, n_episodes=20)
    assert len(out["policy_correct"]) == 20, "one correctness entry per episode"
    assert not any(out["policy_correct"]), "zero weights never choose right"
    assert out["episodes_to_policy"] is None


def _shown(out):
    return "never" if out["episodes_to_policy"] is None else out["episodes_to_policy"]


def test_8_corridor_flat_control_is_state_blind_and_never_reaches_policy():
    q_opt = CORRIDOR_TASK.q_star()
    opt = optimal_actions(q_opt)
    states = policy_states(q_opt)
    assert states == [1, 2, 3, 4, 5], "precondition: policy states are 1..5"
    left_only = [s for s in states if opt[s, A_LEFT] and not opt[s, A_RIGHT]]
    right_only = [s for s in states if opt[s, A_RIGHT] and not opt[s, A_LEFT]]
    assert left_only == [1, 2] and right_only == [4, 5], "precondition: corridor discriminates"
    flat = corridor_features("flat", 0)
    assert all(np.array_equal(flat[s], flat[0]) for s in range(len(flat))), "flat is constant"
    for seed in range(5):
        out = run_corridor("flat", seed)
        print(f"\nflat seed {seed}: episodes-to-policy = {_shown(out)}, "
              f"final greedy (states 1..5) = {out['greedy']}, max |Q - Q*| = {out['max_err']:.6f}")
        assert len(set(out["greedy"])) == 1, f"seed {seed}: flat greedy varies over states"
        assert not any(out["policy_correct"]), f"seed {seed}: flat reached the policy"
        assert out["episodes_to_policy"] is None, f"seed {seed}: flat has a policy episode"


def test_9_network_reaches_and_holds_policy_on_corridor_tabular_reported():
    # Every row is printed before the assertion, so a failing seed does not hide the others.
    # The network features must reach and hold the policy in every seed. Tabular is a control:
    # at alpha 0.1 and epsilon 0.1 a rarely visited (state, action) can stay under-trained within
    # 1000 episodes (seed 3: Q(2, left) 0.722 after 19 updates vs Q(2, right) about 0.729), so it
    # must reach the policy in at least 4 of 5 seeds.
    held = {"tabular": [], "network": []}
    for kind in ("tabular", "network"):
        for seed in range(5):
            out = run_corridor(kind, seed)
            print(f"\n{kind} seed {seed}: episodes-to-policy = {_shown(out)}, "
                  f"final greedy (states 1..5) = {out['greedy']}, "
                  f"max |Q - Q*| = {out['max_err']:.6f}")
            held[kind].append(out["episodes_to_policy"] is not None and out["final_correct"])
    assert all(held["network"]), f"network not reached and held: {held['network']}"
    assert sum(held["tabular"]) >= 4, f"tabular reached in fewer than 4 seeds: {held['tabular']}"
