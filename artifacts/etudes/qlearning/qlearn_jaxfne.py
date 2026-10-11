"""Q-learning etude on a small jaxfne E/I network (lane QL).

Task: 5-state chain (states 0..4, actions left/right). Start at 0; moving
right from 3 into 4 gives r = 1 and ends the episode; left at 0 stays at 0;
all other rewards are 0. gamma = 0.9, episode cap 50 steps.

State code: a 40 E + 10 I jaxfne network (default izhikevich family, built and
simulated exactly as in examples/00_minimal_column.py and
examples/05_network_100_ei_multimodal.py). State s is a 200 ms drive to one
of 5 disjoint 8-cell E input groups. phi(s) = E spike counts over the window,
float32, divided by their max; simulated once per (seed, s) and cached.

Seeds 0..4 drive both the network realization and the agent: each seed fixes
a per-seed input gain of the drive schedule (the public Configuration path
builds fixed dense weights, so the seeded gain is what makes the realized
network response seed-dependent) and seeds the exploration RNG.

Agent: linear Q(s, a) = w_a . phi(s); Q-learning with epsilon = 0.1,
alpha = 0.1. Tabular baseline uses one-hot phi with the same loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import jaxfne as jtfne

N_STATES = 5
A_LEFT = 0
A_RIGHT = 1
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.1
EPISODE_CAP = 50

N_E = 40
N_I = 10
N_NEURONS = N_E + N_I
GROUP = 8
STIM_MS = 200.0
DT_MS = 0.5


def input_gain(seed: int) -> float:
    """Fixed per-seed drive gain: the seed-dependent network realization."""
    return 10.0 + 2.0 * float(seed)


def q_star(gamma: float = GAMMA) -> np.ndarray:
    """Analytic Q*: Q*(s, right) = gamma**(3-s), Q*(s, left) from V*."""
    qs = np.zeros((N_STATES, 2), dtype=np.float64)
    qs[:4, A_RIGHT] = gamma ** (3 - np.arange(4))
    v = np.zeros(N_STATES, dtype=np.float64)
    v[:4] = qs[:4, A_RIGHT]
    for s in range(N_STATES - 1):
        qs[s, A_LEFT] = gamma * v[max(s - 1, 0)]
    # Row 4 is terminal: no outgoing action, Q*(4, .) = 0.
    return qs


def step_env(s: int, a: int):
    """One chain step; returns (s_next, reward, done)."""
    if a == A_RIGHT:
        if s >= 3:
            return 4, 1.0, True
        return s + 1, 0.0, False
    if s <= 0:
        return 0, 0.0, False
    return s - 1, 0.0, False


def build_model():
    """40 E + 10 I network; same build pattern as examples 00/05."""
    cfg = (
        jtfne.configuration()
        .network(
            name="qlearn_chain",
            kind="balanced_ei_population",
            n=N_NEURONS,
            cell_types={"E": 0.8, "I": 0.2},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="declared_proxy",
            gauge="mean_zero",
        )
        .probe(name="spike_probe", modes=["spikes"])
    )
    return jtfne.construct(cfg)


_MODEL_CACHE: dict[int, object] = {}
_PHI_CACHE: dict[tuple[int, int], np.ndarray] = {}


def phi_for_seed_state(seed: int, s: int) -> np.ndarray:
    """E spike-count feature for state s under seed; cached, float32."""
    key = (int(seed), int(s))
    if key not in _PHI_CACHE:
        if key[0] not in _MODEL_CACHE:
            _MODEL_CACHE[key[0]] = build_model()
        model = _MODEL_CACHE[key[0]]
        group = list(range(s * GROUP, s * GROUP + GROUP))
        sched = jtfne.stimulus_schedule(
            [
                {
                    "onset_ms": 0.0,
                    "duration_ms": STIM_MS,
                    "amplitude": input_gain(seed),
                    "label": f"state_{s}",
                    "is_drive_event": True,
                    "target_indices": group,
                }
            ],
            N_NEURONS,
        )
        sim = jtfne.simulation(
            duration_ms=STIM_MS, dt_ms=DT_MS, seed=seed,
            record_sources=False, record_fields=False,
        )
        signals = model.simulate(sim, paradigm=sched)
        counts = np.asarray(signals.spikes)[:, :N_E].sum(axis=0)
        assert counts.sum() > 0, f"phi({seed}, {s}) has no spikes"
        peak = counts.max()
        _PHI_CACHE[key] = (counts / peak).astype(np.float32)
    return _PHI_CACHE[key]


def features_for_seed(seed: int) -> np.ndarray:
    """(5, 40) feature matrix; every row carries spikes."""
    phi = np.stack([phi_for_seed_state(seed, s) for s in range(N_STATES)])
    assert all(phi[s].sum() > 0 for s in range(N_STATES))
    return phi


def q_values(w: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Q(s, a) = w_a . phi(s); shape (n_states, 2)."""
    return np.stack([phi @ w[A_LEFT], phi @ w[A_RIGHT]], axis=1)


def greedy_policy(w: np.ndarray, phi: np.ndarray) -> list[int]:
    """Greedy action per state; ties (exact) go left."""
    q = q_values(w, phi)
    return [A_RIGHT if q[s, A_RIGHT] > q[s, A_LEFT] else A_LEFT
            for s in range(phi.shape[0])]


def run_q_learning(
    phi: np.ndarray,
    *,
    seed: int,
    alpha: float,
    epsilon: float,
    n_episodes: int,
    gamma: float = GAMMA,
    cap: int = EPISODE_CAP,
) -> dict:
    """Epsilon-greedy linear Q-learning from state 0; returns w and stats."""
    rng = np.random.default_rng(seed)
    n_states, n_feat = phi.shape
    w = np.zeros((2, n_feat), dtype=np.float64)
    visits = np.zeros((n_states, 2), dtype=np.int64)
    attempts = 0
    first_policy_ep: int | None = None
    for ep in range(1, n_episodes + 1):
        s = 0
        for _ in range(cap):
            q = q_values(w, phi[s : s + 1])[0]
            if rng.random() < epsilon:
                a = int(rng.integers(2))
            elif q[A_RIGHT] == q[A_LEFT]:
                a = int(rng.integers(2))  # exact tie: random, else ties trap left
            else:
                a = int(np.argmax(q))
            s2, r, done = step_env(s, a)
            target = r if done else r + gamma * float(np.max(q_values(w, phi[s2 : s2 + 1])[0]))
            delta = target - float(w[a] @ phi[s])
            w[a] += alpha * delta * phi[s]
            visits[s, a] += 1
            attempts += 1
            s = s2
            if done:
                break
        if first_policy_ep is None and all(
            a == A_RIGHT for a in greedy_policy(w, phi[:4])
        ):
            first_policy_ep = ep
    return {"w": w, "visits": visits, "attempts": attempts,
            "first_policy_ep": first_policy_ep}
