"""Q-learning etude on a small jaxfne E/I network (lane QL).

Task: the registered ``chain5`` task (``jaxfne.tasks``). 5 states (0..4), actions
left/right, start at 0. Moving right from 3 into 4 gives r = 1 and ends the
episode; gamma = 0.9; episode cap 50 steps.

State code: a 40 E + 10 I jaxfne network (izhikevich family, public Configuration
builder). The deterministic all-to-all base graph (make_eig) is unchanged. One
connections() rule adds E-to-E edges with probability 0.5 and weight 1.0, and
connectivity(edge_seed=seed) fixes that random subset per seed. The rule takes
effect only on the edge_list backend: the default dense backend ignores it, so the
seed does not reach phi there. State s is a 200 ms drive of amplitude 10 to two
neighbouring 8-cell E groups, s and (s + 1) mod 5, so neighbouring states share a
group. phi(s) = E spike counts over the window, divided by their max, then scaled
to unit L2 norm (float32); simulated once per (seed, s) and cached. The unit norm
keeps alpha * ||phi||^2 = alpha (see README).

Seeds 0..4 fix the network realization (edge_seed) and the simulation noise
(simulation seed), and seed the exploration RNG.

Agent: linear Q(s, a) = w_a . phi(s); Q-learning with epsilon = 0.1,
alpha = 0.1. Tabular baseline uses one-hot phi with the same loop.

Second task: the registered ``corridor7`` (states 0..6, start drawn from 1..5). Its
feature sets are the network (as above), a flat constant vector and one-hot tabular
features; see ``corridor_features`` and ``run_corridor``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np

import jaxfne as jtfne

assert "site-packages" not in jtfne.__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

from jaxfne.tasks import get_task  # noqa: E402
from jaxfne.tasks.chain import A_LEFT, A_RIGHT  # noqa: E402

TASK_DESC, TASK = get_task("chain5")
CORRIDOR_DESC, CORRIDOR_TASK = get_task("corridor7")
N_STATES = TASK_DESC.n_states
GAMMA = TASK_DESC.gamma
ALPHA = 0.1
EPSILON = 0.1
EPISODE_CAP = 50

N_E = 40
N_I = 10
N_NEURONS = N_E + N_I
GROUP = 8
N_GROUPS = N_E // GROUP
STIM_MS = 200.0
DT_MS = 0.5
DRIVE_GAIN = 10.0
RECURRENT_P = 0.5
RECURRENT_WEIGHT = 1.0


def driven_cells(s: int) -> list[int]:
    """E cells driven in state s: groups s and (s + 1) mod N_GROUPS (5 groups of 8).

    chain5 has 5 states, so its rule is unchanged; corridor7 has 7 states and wraps.
    """
    groups = (s % N_GROUPS, (s + 1) % N_GROUPS)
    return sorted({g * GROUP + i for g in groups for i in range(GROUP)})


def q_star() -> np.ndarray:
    """Analytic Q* of the registered chain task."""
    return TASK.q_star()


def build_model(seed: int):
    """40 E + 10 I network with one seeded E-to-E connections() rule.

    Without connectivity(edge_seed=...) the construct route is deterministic and
    the seed would not reach the network's code.
    """
    cfg = (
        jtfne.configuration()
        .network(
            name="qlearn_chain",
            kind="balanced_ei_population",
            n=N_NEURONS,
            cell_types={"E": 0.8, "I": 0.2},
        )
        .connections(
            name="rec_ee",
            source={"cell_type": "E"},
            target={"cell_type": "E"},
            probability=RECURRENT_P,
            weight=RECURRENT_WEIGHT,
        )
        .connectivity(edge_seed=int(seed))
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
_RAW_CACHE: dict[tuple[int, int], np.ndarray] = {}
_PHI_CACHE: dict[tuple[int, int], np.ndarray] = {}


def max_normalised_for_seed_state(seed: int, s: int) -> np.ndarray:
    """E spike counts for state s under seed, divided by their max; cached, float32.

    These are the features of the earlier design: alpha * ||phi||^2 is above 2 for
    them, so the learner diverges (see README).
    """
    key = (int(seed), int(s))
    if key not in _RAW_CACHE:
        if key[0] not in _MODEL_CACHE:
            _MODEL_CACHE[key[0]] = build_model(key[0])
        model = _MODEL_CACHE[key[0]]
        sched = jtfne.stimulus_schedule(
            [
                {
                    "onset_ms": 0.0,
                    "duration_ms": STIM_MS,
                    "amplitude": DRIVE_GAIN,
                    "label": f"state_{s}",
                    "is_drive_event": True,
                    "target_indices": driven_cells(key[1]),
                }
            ],
            N_NEURONS,
        )
        sim = jtfne.simulation(
            duration_ms=STIM_MS, dt_ms=DT_MS, seed=seed,
            record_sources=False, record_fields=False,
            runtime=jtfne.runtime(recurrent_backend="edge_list", seed=seed),
        )
        signals = model.simulate(sim, paradigm=sched)
        counts = np.asarray(signals.spikes)[:, :N_E].sum(axis=0)
        assert counts.sum() > 0, f"phi({seed}, {s}) has no spikes"
        peak = counts.max()
        _RAW_CACHE[key] = (counts / peak).astype(np.float32)
    return _RAW_CACHE[key]


def phi_for_seed_state(seed: int, s: int) -> np.ndarray:
    """phi(s): max-normalised counts scaled to unit L2 norm; cached, float32."""
    key = (int(seed), int(s))
    if key not in _PHI_CACHE:
        v = max_normalised_for_seed_state(seed, s).astype(np.float64)
        _PHI_CACHE[key] = (v / np.linalg.norm(v)).astype(np.float32)
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


def episodes_to_policy(policy_correct: list[bool]) -> int | None:
    """First episode E (1-based) after which the greedy policy is correct after E and
    after every later episode. None ("never") if the last episode is wrong."""
    if not policy_correct or not policy_correct[-1]:
        return None
    first = len(policy_correct)
    for i in range(len(policy_correct) - 1, -1, -1):
        if not policy_correct[i]:
            break
        first = i + 1
    return first


def optimal_actions(q_opt: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Boolean (n_states, n_actions): actions within tol of the row maximum of Q*."""
    return q_opt >= q_opt.max(axis=1, keepdims=True) - tol


def policy_states(q_opt: np.ndarray) -> list[int]:
    """Non-terminal states: rows of Q* with a positive value (terminal rows are zero)."""
    return [s for s in range(q_opt.shape[0]) if q_opt[s].max() > 0]


def greedy_is_correct(w: np.ndarray, phi: np.ndarray, opt: np.ndarray, states: list[int]) -> bool:
    """True when the greedy action (ties go left) is optimal in every listed state."""
    q = q_values(w, phi[states])
    greedy = [A_RIGHT if q[i, A_RIGHT] > q[i, A_LEFT] else A_LEFT for i in range(len(states))]
    return all(bool(opt[s, a]) for s, a in zip(states, greedy))


def run_q_learning(
    phi: np.ndarray,
    *,
    seed: int,
    alpha: float,
    epsilon: float,
    n_episodes: int,
    task_name: str = "chain5",
    gamma: float | None = None,
    cap: int = EPISODE_CAP,
) -> dict:
    """Epsilon-greedy linear Q-learning; each episode starts at ``task.reset(rng)``.

    The policy is correct when every policy state's greedy action (ties go left) is
    optimal under the task's analytic Q*; where Q* ties, either action counts. Policy
    states are the non-terminal states: the rows of Q* with a positive value.
    """
    desc, task = get_task(task_name)
    q_opt = task.q_star()
    opt = optimal_actions(q_opt)
    pol_states = policy_states(q_opt)
    gamma = desc.gamma if gamma is None else gamma
    rng = np.random.default_rng(seed)
    n_states, n_feat = phi.shape
    assert n_states == desc.n_states, "feature rows must match the task's states"
    w = np.zeros((2, n_feat), dtype=np.float64)
    visits = np.zeros((n_states, 2), dtype=np.int64)
    attempts = 0
    policy_correct: list[bool] = []
    for _ep in range(1, n_episodes + 1):
        s = task.reset(rng)
        for _ in range(cap):
            q = q_values(w, phi[s : s + 1])[0]
            if rng.random() < epsilon:
                a = int(rng.integers(2))
            elif q[A_RIGHT] == q[A_LEFT]:
                a = int(rng.integers(2))  # exact tie: random, else ties trap left
            else:
                a = int(np.argmax(q))
            s2, r, done = task.step(s, a)
            target = r if done else r + gamma * float(np.max(q_values(w, phi[s2 : s2 + 1])[0]))
            delta = target - float(w[a] @ phi[s])
            w[a] += alpha * delta * phi[s]
            visits[s, a] += 1
            attempts += 1
            s = s2
            if done:
                break
        # Checked after every episode: is the greedy policy correct now?
        policy_correct.append(greedy_is_correct(w, phi, opt, pol_states))
    return {"w": w, "visits": visits, "attempts": attempts,
            "policy_correct": policy_correct,
            "episodes_to_policy": episodes_to_policy(policy_correct)}


def corridor_features(kind: str, seed: int) -> np.ndarray:
    """Feature matrix (7, d) for corridor7.

    network: unit-norm E/I network rows for the seed; flat: one constant unit vector
    of length N_E for every state (state-blind); tabular: one-hot rows.
    """
    n = CORRIDOR_DESC.n_states
    if kind == "network":
        return np.stack([phi_for_seed_state(seed, s) for s in range(n)]).astype(np.float64)
    if kind == "flat":
        return np.full((n, N_E), 1.0 / np.sqrt(N_E))
    if kind == "tabular":
        return np.eye(n, dtype=np.float64)
    raise ValueError(f"unknown feature kind {kind!r}")


def run_corridor(kind: str, seed: int, n_episodes: int = 1000) -> dict:
    """Q-learning on corridor7 with one feature set; alpha 0.1, epsilon 0.1."""
    phi = corridor_features(kind, seed)
    out = run_q_learning(phi, seed=seed, alpha=ALPHA, epsilon=EPSILON,
                         n_episodes=n_episodes, task_name="corridor7")
    q_opt = CORRIDOR_TASK.q_star()
    states = policy_states(q_opt)
    w = out["w"]
    greedy_all = greedy_policy(w, phi)
    return {
        "kind": kind,
        "seed": seed,
        "episodes_to_policy": out["episodes_to_policy"],
        "final_correct": greedy_is_correct(w, phi, optimal_actions(q_opt), states),
        "greedy": [greedy_all[s] for s in states],
        "max_err": float(np.abs(q_values(w, phi)[states] - q_opt[states]).max()),
        "policy_correct": out["policy_correct"],
    }
