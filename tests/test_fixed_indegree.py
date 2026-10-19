"""Exact fixed in-degree sampling ``in_degree`` / ``sampling="fixed_indegree"``.

Every post cell draws exactly ``in_degree`` distinct presynaptic partners
uniformly without replacement; a post cell with fewer available partners
raises rather than filling short. Declared with ``probability``,
``max_in_degree``, ``footprint`` or another sampling it is refused.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.connectivity import (
    _candidate_pairs_fixed_indegree,
    _validate_in_degree,
    compile_connection_rules,
)

NEURONS = [
    {"neuron_id": i, "area": "X", "layer": "L", "cell_type": "E"}
    for i in range(10)
]
MECHS = [{"name": "ampa", "kind": "synapse", "params": {"tau_ms": 5.0, "sign": 1.0}}]


def _compile(conns, **kw):
    return compile_connection_rules(NEURONS, conns, MECHS, **kw)


def _rule(**kw):
    rule = {"name": "rec", "source": {"area": "X"}, "target": {"area": "X"},
            "weight": 0.1, "mechanism": "ampa", "allow_self_connections": True}
    rule.update(kw)
    return [rule]


def test_validate_accepts_positive_int():
    assert _validate_in_degree("r", 3) == 3
    assert _validate_in_degree("r", None) is None


@pytest.mark.parametrize("bad", [True, False, 0, -2, 2.5, "3", (3,)])
def test_validate_refuses_bad_degree(bad):
    with pytest.raises(ValueError, match="in_degree"):
        _validate_in_degree("r", bad)


def test_sampler_draws_exact_per_post_count():
    r = _compile(_rule(sampling="fixed_indegree", in_degree=3), seed=0)
    post = np.asarray(r.edge_post)
    assert r.n_edges == 3 * 10, "every post cell must hold exactly in_degree edges"
    for i in range(10):
        kept = np.asarray(r.edge_pre)[post == i]
        assert len(kept) == 3, f"post cell {i} must hold exactly 3 partners"
        assert len(set(int(x) for x in kept)) == 3, \
            f"post cell {i} partners must be distinct"


def test_sampler_is_deterministic_for_a_seed():
    a = _compile(_rule(sampling="fixed_indegree", in_degree=3), seed=7)
    b = _compile(_rule(sampling="fixed_indegree", in_degree=3), seed=7)
    assert np.array_equal(np.asarray(a.edge_pre), np.asarray(b.edge_pre))
    assert np.array_equal(np.asarray(a.edge_post), np.asarray(b.edge_post))


def test_sampler_covers_the_pool():
    # 10 post cells x 9 draws from 10 presynaptic cells: every pre cell fires.
    r = _compile(_rule(sampling="fixed_indegree", in_degree=9), seed=0)
    assert set(int(x) for x in np.asarray(r.edge_pre)) == set(range(10)), \
        "near-complete draws must cover the presynaptic pool"


def test_degree_above_pool_raises():
    with pytest.raises(ValueError, match="exceeds"):
        _compile(_rule(sampling="fixed_indegree", in_degree=11), seed=0)


def test_small_pool_without_self_raises():
    # allow_self False removes the self pair first, so 10 cells admit at most 9.
    rule = _rule(sampling="fixed_indegree", in_degree=10)
    rule[0]["allow_self_connections"] = False
    with pytest.raises(ValueError, match="exceeds"):
        _compile(rule, seed=0)


def test_unit_sampler_without_self_holds_count():
    import jax

    pairs = _candidate_pairs_fixed_indegree(
        [0, 1, 2, 3], [0, 1, 2, 3], in_degree=2, allow_self=False,
        key=jax.random.PRNGKey(0))
    assert len(pairs) == 8, "every post cell must hold exactly in_degree edges"
    for post in range(4):
        kept = [a for a, b in pairs if b == post]
        assert len(kept) == 2 and post not in kept, \
            "self pair excluded first, count still holds"


def test_sampler_marginals_are_uniform():
    # Each of 20 pre cells is drawn by a post cell with probability 5/20; over
    # 4000 post cells the per-pre count is Binomial(4000, 0.25): sd ~27.4.
    import jax

    pairs = _candidate_pairs_fixed_indegree(
        list(range(20)), list(range(100, 4100)), in_degree=5, allow_self=False,
        key=jax.random.PRNGKey(1))
    counts = np.bincount([a for a, _ in pairs], minlength=20)
    assert counts.sum() == 4000 * 5, "fixture must hold every draw"
    assert np.all(np.abs(counts - 1000) < 6 * 27.4), counts


def test_sampler_scales_without_per_cell_device_calls():
    # 20000 post cells: a per-cell device draw costs seconds; one host pass is
    # milliseconds.
    import time

    import jax

    t0 = time.perf_counter()
    pairs = _candidate_pairs_fixed_indegree(
        list(range(50)), list(range(20000)), in_degree=10, allow_self=False,
        key=jax.random.PRNGKey(2))
    elapsed = time.perf_counter() - t0
    assert len(pairs) == 200000, "every post cell must hold exactly in_degree edges"
    assert elapsed < 3.0, f"sampler took {elapsed:.1f} s for 20000 post cells"


def test_sampling_without_degree_refused():
    with pytest.raises(ValueError, match="without"):
        _compile(_rule(sampling="fixed_indegree"), seed=0)


@pytest.mark.parametrize("extra", [
    {"probability": 0.1},
    {"max_in_degree": 4},
    {"footprint": {"size": (1.0, 1.0), "offset": [0.0, 0.0]}},
])
def test_degree_refuses_other_pair_selectors(extra):
    rule = _rule(sampling="fixed_indegree", in_degree=3)
    rule[0].update(extra)
    with pytest.raises(ValueError, match="in_degree"):
        _compile(rule, seed=0)


def test_degree_refuses_bernoulli_sampling():
    with pytest.raises(ValueError, match="in_degree"):
        _compile(_rule(sampling="capped_bernoulli", in_degree=3), seed=0)


def test_degree_without_sampling_uses_fixed_sampler():
    # in_degree alone (default sampling) selects the exact sampler: the count
    # contract holds either way, so the declaration order does not matter.
    r = _compile(_rule(in_degree=3), seed=0)
    post = np.asarray(r.edge_post)
    assert r.n_edges == 3 * 10
    for i in range(10):
        assert int((post == i).sum()) == 3, f"post cell {i} must hold 3 partners"


def test_row_echoes_degree_and_sampler():
    r = _compile(_rule(sampling="fixed_indegree", in_degree=3), seed=0)
    rows = [row for row in r.connection_table if row["name"] == "rec"]
    assert len(rows) == 1, "fixture must compile exactly one rule row"
    assert rows[0]["in_degree"] == 3
    assert rows[0]["sampling"] == "fixed_indegree"
    assert "probability" not in rows[0] or rows[0].get("probability") is None


def test_density_row_carries_no_degree():
    r = _compile(_rule(probability=0.2), seed=0)
    rows = [row for row in r.connection_table if row["name"] == "rec"]
    assert len(rows) == 1, "fixture must compile exactly one rule row"
    assert rows[0].get("in_degree") is None
    assert "sampling" not in rows[0], "density stays the unmarked default"


def test_tensor_inter_connection_routes_fixed_sampler():
    from jaxfne import neuronal_tensor as nt

    layer = nt.Layer(name="L1", n_neurons=10,
                     neuron_types=[nt.NeuronType.make("E", fraction=1.0)])
    tensor = nt.NeuronalTensor(
        name="fixed",
        areas=[nt.Area(name="net", layers=[layer], inter_connections=[
            nt.InterConnection("L1", "E", "L1", "E", "AMPA",
                               static=nt.StaticParams(dT_ms=2.0),
                               sampling="fixed_indegree", in_degree=3)])],
    )
    model = jtfne.construct(tensor, nt.RuntimeConfiguration(seed=0, duration_ms=2.0,
                                                            dt_ms=0.5))
    el = model.params["edge_list"]
    post = np.asarray(el.post)
    assert len(post) == 3 * 10, "every post cell must hold exactly in_degree edges"
    for i in range(10):
        assert int((post == i).sum()) == 3, f"post cell {i} must hold 3 partners"


def test_tensor_inter_connection_refuses_degree_with_max_degree():
    from jaxfne import neuronal_tensor as nt

    with pytest.raises(ValueError, match="in_degree"):
        nt.InterConnection("L1", "E", "L1", "E", "AMPA",
                           static=nt.StaticParams(dT_ms=2.0),
                           max_in_degree=4, in_degree=3)


def test_configuration_stores_degree_only_when_declared():
    cfg = jtfne.configuration().connections(name="dflt", source={}, target={})
    rules = cfg.metadata["circuit"]["connections"]
    assert "in_degree" not in rules[-1], \
        "unremarked rules must not grow an in_degree key"
    cfg = jtfne.configuration().connections(
        name="fix", source={}, target={}, sampling="fixed_indegree", in_degree=3)
    rules = cfg.metadata["circuit"]["connections"]
    assert rules[-1]["in_degree"] == 3
    assert rules[-1]["sampling"] == "fixed_indegree"


@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.5, "3"])
def test_configuration_refuses_bad_degree(bad):
    with pytest.raises(ValueError, match="in_degree"):
        jtfne.configuration().connections(name="x", source={}, target={}, in_degree=bad)


def test_configuration_refuses_fixed_without_degree():
    with pytest.raises(ValueError, match="in_degree"):
        jtfne.configuration().connections(
            name="x", source={}, target={}, sampling="fixed_indegree")


@pytest.mark.parametrize("kw", [
    {"probability": 0.1},
    {"max_in_degree": 4},
    {"sampling": "bernoulli"},
    {"sampling": "capped_bernoulli"},
])
def test_configuration_refuses_degree_with_other_modes(kw):
    with pytest.raises(ValueError, match="in_degree"):
        jtfne.configuration().connections(name="x", source={}, target={},
                                          in_degree=3, **kw)
