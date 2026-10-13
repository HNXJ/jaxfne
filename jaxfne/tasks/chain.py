"""Five-state chain task: the first entry of the private task registry.

States 0..4; actions 0 = left, 1 = right. Start at 0. Moving right from 3 into
4 gives r = 1 and ends the episode. State 4 is terminal: every action keeps it
there with reward 0 and done = True, so the Bellman check holds on it. Left at
0 stays at 0. All other rewards are 0. gamma = 0.9.
"""

from __future__ import annotations

import numpy as np

from ._registry import TaskDescriptor

N_STATES = 5
N_ACTIONS = 2
A_LEFT = 0
A_RIGHT = 1
TERMINAL = 4
GAMMA = 0.9

CHAIN5 = TaskDescriptor(
    name="chain5",
    n_states=N_STATES,
    n_actions=N_ACTIONS,
    gamma=GAMMA,
    has_q_star=True,
)


def step_env(s: int, a: int) -> tuple[int, float, bool]:
    """One chain step; returns (s_next, reward, done)."""
    if s == TERMINAL:
        return TERMINAL, 0.0, True
    if a == A_RIGHT:
        if s >= 3:
            return TERMINAL, 1.0, True
        return s + 1, 0.0, False
    if s <= 0:
        return 0, 0.0, False
    return s - 1, 0.0, False


def q_star_table(gamma: float = GAMMA) -> np.ndarray:
    """Analytic Q*: Q*(s, right) = gamma**(3-s), Q*(s, left) from V*."""
    qs = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    qs[:4, A_RIGHT] = gamma ** (3 - np.arange(4))
    v = np.zeros(N_STATES, dtype=np.float64)
    v[:4] = qs[:4, A_RIGHT]
    for s in range(N_STATES - 1):
        qs[s, A_LEFT] = gamma * v[max(s - 1, 0)]
    # Row 4 is terminal: no outgoing action, Q*(4, .) = 0.
    return qs


class Chain5:
    """Deterministic 5-state chain: ``reset`` always returns state 0."""

    def reset(self, rng: np.random.Generator) -> int:
        return 0

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        return step_env(int(state), int(action))

    def q_star(self) -> np.ndarray:
        return q_star_table(GAMMA)


def make_chain5() -> Chain5:
    return Chain5()
