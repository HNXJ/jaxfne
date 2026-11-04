"""``simulate_batch`` member i is keyed ``split(PRNGKey(base_seed), n)[i]``, as its docstring states."""

from __future__ import annotations

import numpy as np

import jaxfne as jtfne


def _model_and_sim(seed: int):
    cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=40.0, dt_ms=1.0)
    return jtfne.construct(cfg), jtfne.simulation(duration_ms=40.0, dt_ms=1.0, seed=seed)


def test_batch_members_differ_from_serial_seeds():
    model, sim0 = _model_and_sim(0)
    _, sim1 = _model_and_sim(1)
    v0 = np.asarray(jtfne.simulate(model, sim0).V_m)
    v1 = np.asarray(jtfne.simulate(model, sim1).V_m)
    assert not np.array_equal(v0, v1), "fixture must be noisy: seeds 0 and 1 give the same V_m"

    batch = np.asarray(model.simulate_batch(sim0, n_seeds=2)["V_m"])
    assert not np.array_equal(batch[0], v0), "member 0 matches simulate(seed=base_seed)"
    assert not np.array_equal(batch[1], v1), "member 1 matches simulate(seed=1)"


def test_batch_members_follow_the_base_seed():
    model, sim = _model_and_sim(0)
    a = np.asarray(model.simulate_batch(sim, n_seeds=2, seed=3)["V_m"])
    b = np.asarray(model.simulate_batch(sim, n_seeds=2, seed=3)["V_m"])
    c = np.asarray(model.simulate_batch(sim, n_seeds=2, seed=4)["V_m"])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[0], a[1]), "members of one batch must differ"
    assert not np.array_equal(a, c), "a different base seed must change the members"
