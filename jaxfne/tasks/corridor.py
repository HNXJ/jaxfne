"""Seven-state corridor task: a discriminating entry of the private task registry.

States 0..6; actions 0 = left, 1 = right. Terminal states are 0 and 6. Entering a
terminal gives reward 1 and ends the episode. A terminal state is absorbing: every
action keeps it there with reward 0 and done = True. Non-terminal moves give reward
0. gamma = 0.9. ``reset`` draws the start uniformly from 1..5.

The optimal action depends on the state (left in 1 and 2, right in 4 and 5, either
in 3), so a feature set that cannot tell the states apart cannot reach the policy.
"""

from __future__ import annotations

import numpy as np

from ._registry import TaskDescriptor

N_STATES = 7
N_ACTIONS = 2
A_LEFT = 0
A_RIGHT = 1
TERMINALS = (0, 6)
GAMMA = 0.9

CORRIDOR7 = TaskDescriptor(
    name="corridor7",
    n_states=N_STATES,
    n_actions=N_ACTIONS,
    gamma=GAMMA,
    has_q_star=True,
)


def step_env(s: int, a: int) -> tuple[int, float, bool]:
    """One corridor step; returns (s_next, reward, done)."""
    if s in TERMINALS:
        return s, 0.0, True
    s2 = s - 1 if a == A_LEFT else s + 1
    if s2 in TERMINALS:
        return s2, 1.0, True
    return s2, 0.0, False


def q_star_table(gamma: float = GAMMA) -> np.ndarray:
    """Analytic Q*: go to the nearer end. V(s) = gamma**(d(s) - 1), d = distance to
    the nearer terminal; a move into a terminal pays 1 and ends the episode."""
    v = np.zeros(N_STATES, dtype=np.float64)
    for s in range(1, N_STATES - 1):
        v[s] = gamma ** (min(s, N_STATES - 1 - s) - 1)
    qs = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    for s in range(1, N_STATES - 1):
        for a, s2 in ((A_LEFT, s - 1), (A_RIGHT, s + 1)):
            qs[s, a] = 1.0 if s2 in TERMINALS else gamma * v[s2]
    return qs


class Corridor7:
    """Stochastic-start corridor: ``reset`` draws the start uniformly from 1..5."""

    def reset(self, rng: np.random.Generator) -> int:
        return int(rng.integers(1, N_STATES - 1))

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        return step_env(int(state), int(action))

    def q_star(self) -> np.ndarray:
        return q_star_table(GAMMA)


def make_corridor7() -> Corridor7:
    return Corridor7()
