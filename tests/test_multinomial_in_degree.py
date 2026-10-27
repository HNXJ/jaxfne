"""``in_degree=k`` with ``sampling="multinomial"`` (issue P-049).

``k`` is the mean in-degree: the rule draws ``K = round_half_up(k * n_post)``
synapses with replacement, through the same draw as the ``probability`` form.
Covers the Configuration route (compiled edges), the NeuronalTensor route
(realized edge list of a constructed model), the refusals, and the unchanged
probability form.
"""

import math

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt

MECHS = [{"name": "ampa", "kind": "synapse", "params": {"tau_ms": 5.0, "sign": 1.0}}]
MECH_DECL = {"name": "ampa", "kind": "synapse", "params": {"tau_ms": 5.0, "sign": 1.0}}
K_FRACTIONAL = 37.4  # mean in-degree, non-integer and greater than n_pre below
DT_MS = 0.5
DUR_MS = 20.0


def _round_half_up(x):
    return int(math.floor(x + 0.5))


def _rows(n, area, cell="E", start=0, layer="L4"):
    return [
        {"neuron_id": start + i, "area": area, "layer": layer, "cell_type": cell}
        for i in range(n)
    ]


def _rule(**kw):
    rule = {"name": "m", "source": {"area": "PRE"}, "target": {"area": "POST"},
            "weight": 0.1, "mechanism": "ampa", "sampling": "multinomial"}
    rule.update(kw)
    return rule


def _pairs(result):
    return list(zip(np.asarray(result.edge_pre).tolist(),
                    np.asarray(result.edge_post).tolist()))


def _config_rule(**kw):
    """The rule entry a Configuration stores for ``connections(**kw)``."""
    cfg = jtfne.configuration().mechanisms(**MECH_DECL)
    cfg = cfg.connections(**{"name": "m", "source": {"area": "PRE"},
                             "target": {"area": "POST"}, "mechanism": "ampa",
                             "weight": 0.1, "sampling": "multinomial", **kw})
    (rule,) = cfg.metadata["circuit"]["connections"]
    return rule


def _area_conn(**kw):
    kwargs = dict(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(dT_ms=5.0, reversal_potentials_mV={"AMPA": 0.0}),
        delay_ms=1.0, sampling="multinomial",
    )
    kwargs.update(kw)
    return nt.AreaConnection(**kwargs)


def _two_area_tensor(n_pre, n_post, **conn_kw):
    types = [nt.NeuronType.make("E")]
    return nt.NeuronalTensor(
        name="p049-area",
        areas=[
            nt.Area(name="V1", layers=[nt.Layer(
                name="L4", n_neurons=n_pre, neuron_types=types)]),
            nt.Area(name="V4", layers=[nt.Layer(
                name="L4", n_neurons=n_post, neuron_types=types)]),
        ],
        area_connections=[_area_conn(**conn_kw)],
    )


def _inter_tensor(n, **conn_kw):
    types = [nt.NeuronType.make("E")]
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(dT_ms=5.0, reversal_potentials_mV={"AMPA": 0.0}),
        delay_ms=1.0, sampling="multinomial", **conn_kw,
    )
    return nt.NeuronalTensor(
        name="p049-inter",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=n, neuron_types=types)],
            inter_connections=[conn],
        )],
    )


def _construct(tensor):
    runtime = nt.RuntimeConfiguration(
        seed=0, duration_ms=DUR_MS, dt_ms=DT_MS, emitter="izhikevich")
    return jtfne.construct(tensor, runtime)


def _assert_k_exact_and_mean(pre, post, n_pre_ids, n_post_ids, expected_k, n_post):
    """Realized K equals the rounded count and the mean in-degree matches K/n_post."""
    lo_pre, hi_pre = n_pre_ids
    lo_post, hi_post = n_post_ids
    assert pre.size == expected_k, f"realized K {pre.size} != round_half_up(k*n_post) {expected_k}"
    assert (pre >= lo_pre).all() and (pre < hi_pre).all(), "pre ids outside the source pool"
    assert (post >= lo_post).all() and (post < hi_post).all(), "post ids outside the target pool"
    in_deg = np.bincount(post - lo_post, minlength=n_post)
    assert in_deg.size == n_post
    assert abs(in_deg.sum() / n_post - expected_k / n_post) <= 1e-12


# ---- K = round_half_up(k * n_post) on both routes ------------------------

def test_fixture_arithmetic_is_what_the_tests_assume():
    # Guards the literals below: k*n_post = 261.8 rounds half-up to 262 (floor gives 261).
    assert _round_half_up(K_FRACTIONAL * 7) == 262
    assert math.floor(K_FRACTIONAL * 7) == 261
    assert _round_half_up(2.5 * 5) == 13, "half-up, not banker's rounding (12)"


def test_configuration_in_degree_draws_round_half_up_k_times_n_post():
    n_pre, n_post = 5, 7
    rule = _config_rule(in_degree=K_FRACTIONAL)
    assert rule["in_degree"] == K_FRACTIONAL, "the declared mean in-degree survives Configuration"
    assert rule["sampling"] == "multinomial" and rule["probability"] is None
    neurons = _rows(n_pre, "PRE") + _rows(n_post, "POST", start=n_pre)
    r = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)
    _assert_k_exact_and_mean(
        np.asarray(r.edge_pre), np.asarray(r.edge_post), (0, n_pre), (n_pre, n_pre + n_post),
        _round_half_up(K_FRACTIONAL * n_post), n_post)
    assert r.n_edges == 262
    row = r.connection_table[0]
    assert row["in_degree"] == K_FRACTIONAL, "declared in_degree is recorded"
    assert row["realized_k"] == 262 and row["n_edges"] == 262, "realized K is recorded"


def test_configuration_in_degree_rounds_half_up_not_banker():
    n_pre, n_post = 5, 5
    rule = _config_rule(in_degree=2.5)
    neurons = _rows(n_pre, "PRE") + _rows(n_post, "POST", start=n_pre)
    r = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)
    assert r.n_edges == 13, "K = round_half_up(2.5 * 5) = 13"


@pytest.mark.parametrize("cls", ["inter", "area"])
def test_tensor_in_degree_realizes_round_half_up_k_times_n_post(cls):
    n_post = 7
    if cls == "area":
        tensor = _two_area_tensor(5, n_post, in_degree=K_FRACTIONAL)
        pre_pool, post_pool = (0, 5), (5, 5 + n_post)
    else:
        tensor = _inter_tensor(n_post, in_degree=K_FRACTIONAL)
        pre_pool, post_pool = (0, n_post), (0, n_post)
    model = _construct(tensor)
    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    expected = _round_half_up(K_FRACTIONAL * n_post)
    _assert_k_exact_and_mean(pre, post, pre_pool, post_pool, expected, n_post)
    if cls == "inter":
        assert (pre != post).all(), "a same-pool InterConnection drops self pairs"


# ---- multapses, self exclusion, the two count sources --------------------

def test_multapses_occur_when_k_exceeds_n_pre():
    n_pre, n_post = 5, 7
    assert K_FRACTIONAL > n_pre, "fixture: k must exceed n_pre for a repeat to be forced"
    rule = _config_rule(in_degree=K_FRACTIONAL)
    neurons = _rows(n_pre, "PRE") + _rows(n_post, "POST", start=n_pre)
    r = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=1)
    pairs = _pairs(r)
    assert len(set(pairs)) < len(pairs), "262 draws over 35 pairs must repeat a pair"


def test_self_pairs_excluded_when_allow_self_is_false():
    n = 7
    rule = _config_rule(in_degree=K_FRACTIONAL, source={"area": "V1"}, target={"area": "V1"})
    assert "allow_self_connections" not in rule, "fixture: allow_self left at its default"
    r = jtfne.compile_connection_rules(_rows(n, "V1"), [rule], MECHS, seed=0)
    assert r.n_edges == _round_half_up(K_FRACTIONAL * n) == 262, "K stays exact with self excluded"
    assert not np.any(np.asarray(r.edge_pre) == np.asarray(r.edge_post)), "self pairs dropped"


def test_in_degree_and_probability_share_one_draw():
    # Same K-generation path: in_degree=0.5 and probability=0.5 give the same edges.
    neurons = _rows(8, "PRE") + _rows(8, "POST", start=8)
    by_in_degree = jtfne.compile_connection_rules(
        neurons, [_rule(in_degree=0.5)], MECHS, seed=7)
    by_probability = jtfne.compile_connection_rules(
        neurons, [_rule(probability=0.5)], MECHS, seed=7)
    assert by_in_degree.n_edges == by_probability.n_edges == 4
    np.testing.assert_array_equal(np.asarray(by_in_degree.edge_pre),
                                  np.asarray(by_probability.edge_pre))
    np.testing.assert_array_equal(np.asarray(by_in_degree.edge_post),
                                  np.asarray(by_probability.edge_post))


# ---- refusals ------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "2"])
def test_configuration_refuses_bad_multinomial_in_degree(bad):
    with pytest.raises(ValueError, match="finite positive"):
        jtfne.configuration().connections(
            name="m", source={"area": "PRE"}, target={"area": "POST"},
            sampling="multinomial", in_degree=bad)


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "2"])
def test_compile_refuses_bad_multinomial_in_degree(bad):
    neurons = _rows(4, "PRE") + _rows(4, "POST", start=4)
    with pytest.raises(ValueError, match="finite positive"):
        jtfne.compile_connection_rules(neurons, [_rule(in_degree=bad)], MECHS, seed=0)


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True])
def test_area_connection_refuses_bad_multinomial_in_degree(bad):
    with pytest.raises(ValueError, match="finite positive"):
        _area_conn(in_degree=bad)


def test_in_degree_with_probability_is_refused_on_configuration():
    with pytest.raises(ValueError, match="declare one or neither"):
        jtfne.configuration().connections(
            name="m", source={"area": "PRE"}, target={"area": "POST"},
            sampling="multinomial", in_degree=2.5, probability=0.5)


def test_in_degree_with_probability_is_refused_at_compile():
    neurons = _rows(4, "PRE") + _rows(4, "POST", start=4)
    with pytest.raises(ValueError, match="declare one or neither"):
        jtfne.compile_connection_rules(
            neurons, [_rule(in_degree=2.5, probability=0.5)], MECHS, seed=0)


def test_in_degree_with_probability_is_refused_on_area_connection():
    with pytest.raises(ValueError, match="declare one or neither"):
        _area_conn(in_degree=2.5, probability=0.5)


@pytest.mark.parametrize("bad", [2.5, 0])
def test_fixed_indegree_still_refuses_a_non_integer_in_degree(bad):
    with pytest.raises(ValueError, match="positive integer"):
        jtfne.configuration().connections(
            name="f", source={"area": "PRE"}, target={"area": "POST"},
            sampling="fixed_indegree", in_degree=bad)
    with pytest.raises(ValueError, match="positive integer"):
        _area_conn(sampling="fixed_indegree", in_degree=bad)


# ---- probability form is unchanged ---------------------------------------

def test_probability_form_edges_are_unchanged_from_the_baseline():
    # Edges captured on the pre-change tree (baseline 3d7a5c9b) for these tiny cases.
    r = jtfne.compile_connection_rules(
        _rows(8, "PRE") + _rows(8, "POST", start=8), [_rule(probability=0.5)], MECHS, seed=7)
    assert np.asarray(r.edge_pre).tolist() == [0, 0, 1, 7]
    assert np.asarray(r.edge_post).tolist() == [9, 13, 10, 11]

    rec = {"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
           "probability": 0.5, "weight": 0.1, "mechanism": "ampa",
           "sampling": "multinomial"}
    r = jtfne.compile_connection_rules(_rows(5, "V1"), [rec], MECHS, seed=0)
    assert np.asarray(r.edge_pre).tolist() == [2, 4, 4]
    assert np.asarray(r.edge_post).tolist() == [0, 1, 2]

    r = jtfne.compile_connection_rules(
        _rows(1, "PRE") + _rows(2, "POST", start=1),
        [_rule(probability=1.0)], MECHS, seed=3)
    assert np.asarray(r.edge_pre).tolist() == [0, 0]
    assert np.asarray(r.edge_post).tolist() == [1, 2]
