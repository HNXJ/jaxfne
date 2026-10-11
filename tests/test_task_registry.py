"""Contract tests for the private task registry (lane QL).

Every registered task passes the contract tests, which iterate ``list_tasks()``.
The registration tests run on a sandbox copy of the registry, so they leave
no entries behind for the contract tests to pick up.
"""

import numpy as np
import pytest

import jaxfne.tasks._registry as registry_mod
from jaxfne.tasks import CHAIN5, TaskDescriptor, get_task, list_tasks, make_chain5, register_task

TASKS = list_tasks()
CAP = 50
SEEDS = 16
# Tasks whose reset must not depend on the rng. Any other task must vary with the seed.
DETERMINISTIC_RESET_TASKS = frozenset({"chain5"})


def _last_action_policy(desc):
    def policy(_state):
        return desc.n_actions - 1

    return policy


def _rollout(task, seed, policy, cap=CAP):
    rng = np.random.default_rng(seed)
    state = task.reset(rng)
    traj = [("reset", state)]
    for _ in range(cap):
        action = policy(state)
        state, reward, done = task.step(state, action)
        traj.append((action, state, reward, done))
        if done:
            break
    return traj


def _distinct_reset_states(task):
    return len({int(task.reset(np.random.default_rng(s))) for s in range(SEEDS)})


def check_seed_dependence(name, task):
    """Assert the reset behaviour the name is declared to have; return its kind."""
    distinct = _distinct_reset_states(task)
    if name in DETERMINISTIC_RESET_TASKS:
        assert distinct == 1, f"{name} is declared deterministic but its reset varies with the seed"
        return "deterministic"
    assert distinct > 1, f"{name} is not declared deterministic but its reset ignores the seed"
    return "stochastic"


class _ToyTask:
    def reset(self, rng):
        return 0

    def step(self, state, action):
        return 0, 0.0, True


class _RandomStartTask:
    def reset(self, rng):
        return int(rng.integers(3))

    def step(self, state, action):
        return 0, 0.0, True


@pytest.fixture
def sandbox(monkeypatch):
    monkeypatch.setattr(registry_mod, "_REGISTRY", dict(registry_mod._REGISTRY))
    return registry_mod


def _desc(**overrides):
    fields = dict(name="toy", n_states=3, n_actions=2, gamma=0.5, has_q_star=False)
    fields.update(overrides)
    return TaskDescriptor(**fields)


def test_registry_has_registered_tasks():
    assert TASKS, "no tasks registered: the contract below would examine nothing"
    assert "chain5" in TASKS


@pytest.mark.parametrize("name", TASKS)
def test_reset_and_step_types_and_ranges(name):
    desc, task = get_task(name)
    s0 = task.reset(np.random.default_rng(0))
    assert isinstance(s0, (int, np.integer)) and not isinstance(s0, bool)
    assert 0 <= s0 < desc.n_states
    for s in range(desc.n_states):
        for a in range(desc.n_actions):
            out = task.step(s, a)
            assert isinstance(out, tuple) and len(out) == 3, (s, a, out)
            s2, reward, done = out
            assert isinstance(s2, (int, np.integer)) and 0 <= s2 < desc.n_states, (s, a, s2)
            assert np.isfinite(float(reward)), (s, a, reward)
            assert isinstance(done, (bool, np.bool_)), (s, a, done)


@pytest.mark.parametrize("name", TASKS)
def test_terminal_transition_exists_and_fixed_policy_reaches_it(name):
    desc, task = get_task(name)
    terminal = [
        (s, a)
        for s in range(desc.n_states)
        for a in range(desc.n_actions)
        if bool(task.step(s, a)[2])
    ]
    assert terminal, f"{name} has no terminal transition in its state-action table"
    traj = _rollout(task, 0, _last_action_policy(desc))
    assert len(traj) > 1, "precondition: the rollout takes at least one step"
    assert bool(traj[-1][3]), f"{name}: fixed policy did not reach done within {CAP} steps"


@pytest.mark.parametrize("name", TASKS)
def test_same_seed_gives_same_trajectory(name):
    desc, _ = get_task(name)
    policy = _last_action_policy(desc)
    first = _rollout(get_task(name)[1], 3, policy)
    second = _rollout(get_task(name)[1], 3, policy)
    assert len(first) > 1, "precondition: the rollout takes at least one step"
    assert first == second


@pytest.mark.parametrize("name", TASKS)
def test_reset_seed_dependence(name):
    _, task = get_task(name)
    if check_seed_dependence(name, task) == "deterministic":
        pytest.skip(f"{name} has a deterministic reset; the two-seeds-differ check does not apply")


def test_seed_contract_flags_unlisted_seed_ignoring_task(sandbox):
    register_task(_desc(name="toy_fixed_start"), _ToyTask)
    _, task = get_task("toy_fixed_start")
    with pytest.raises(AssertionError, match="ignores the seed"):
        check_seed_dependence("toy_fixed_start", task)


def test_seed_contract_accepts_unlisted_stochastic_task(sandbox):
    register_task(_desc(name="toy_random_start"), _RandomStartTask)
    _, task = get_task("toy_random_start")
    assert check_seed_dependence("toy_random_start", task) == "stochastic"


@pytest.mark.parametrize("name", TASKS)
def test_q_star_shape_and_bellman_consistency(name):
    desc, task = get_task(name)
    assert hasattr(task, "q_star") == desc.has_q_star, (
        f"{name}: has_q_star={desc.has_q_star} disagrees with the task object"
    )
    if not desc.has_q_star:
        pytest.skip(f"{name} declares no q_star")
    q = np.asarray(task.q_star(), dtype=np.float64)
    assert q.shape == (desc.n_states, desc.n_actions), q.shape
    assert np.all(np.isfinite(q))
    for s in range(desc.n_states):
        for a in range(desc.n_actions):
            s2, reward, done = task.step(s, a)
            target = float(reward) if done else float(reward) + desc.gamma * float(np.max(q[s2]))
            np.testing.assert_allclose(
                q[s, a], target, rtol=1e-12, atol=1e-12, err_msg=f"{name} Bellman at ({s}, {a})"
            )


def test_chain5_descriptor_and_factory():
    assert CHAIN5.name == "chain5"
    assert (CHAIN5.n_states, CHAIN5.n_actions) == (5, 2)
    assert CHAIN5.gamma == 0.9 and CHAIN5.has_q_star is True
    assert hasattr(make_chain5(), "q_star")


def test_list_tasks_is_sorted(sandbox):
    register_task(_desc(name="zz_toy"), _ToyTask)
    register_task(_desc(name="aa_toy"), _ToyTask)
    names = list_tasks()
    assert names == tuple(sorted(names))
    assert {"aa_toy", "chain5", "zz_toy"} <= set(names)


def test_get_task_refuses_unknown_name_and_lists_known():
    with pytest.raises(ValueError, match="known.*chain5"):
        get_task("no_such_task")


def test_register_refuses_duplicate_name(sandbox):
    with pytest.raises(ValueError, match="already registered"):
        register_task(CHAIN5, make_chain5)


@pytest.mark.parametrize("gamma", [1.0, -0.1])
def test_register_refuses_gamma_outside_unit_interval(sandbox, gamma):
    with pytest.raises(ValueError, match="gamma"):
        register_task(_desc(name="toy_gamma", gamma=gamma), _ToyTask)


def test_register_accepts_gamma_zero(sandbox):
    register_task(_desc(name="toy_gamma0", gamma=0.0), _ToyTask)
    assert "toy_gamma0" in list_tasks()


@pytest.mark.parametrize("field", ["n_states", "n_actions"])
def test_register_refuses_zero_sizes(sandbox, field):
    with pytest.raises(ValueError, match=field):
        register_task(_desc(name="toy_zero", **{field: 0}), _ToyTask)
