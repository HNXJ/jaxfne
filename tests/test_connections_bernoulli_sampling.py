"""Independent per-pair Bernoulli sampling for ``Configuration.connections()``.

``sampling="density"`` (default) realizes an exact edge count
``round(p*n_pre*n_post)``; ``sampling="bernoulli"`` includes each (pre, post)
pair independently with probability ``p``. Unknown ``sampling`` values raise
``ValueError`` naming ``sampling``.
"""

import hashlib
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne

MECHS = [{"name": "ampa", "kind": "synapse", "params": {"tau_ms": 5.0, "sign": 1.0}}]

# Pins for the default "density" path: sha256 over edge_pre/edge_post bytes on
# the pre-change implementation (dev 360b1001). The default path must reproduce
# these exactly.
_FIXTURE_NEURONS = [
    {"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E"},
    {"neuron_id": 1, "area": "V1", "layer": "L4", "cell_type": "E"},
    {"neuron_id": 2, "area": "V1", "layer": "L4", "cell_type": "PV"},
    {"neuron_id": 3, "area": "V4", "layer": "L2", "cell_type": "E"},
]
_DENSITY_PINS = [
    (
        [{"name": "E_PV", "source": {"cell_type": "E", "area": "V1"},
          "target": {"cell_type": "PV"}, "weight": 0.1, "mechanism": "ampa"}],
        {},
        2,
        "dd840fb42246e1b7e4316bc51f7a24fa2bb285ceda848734a14822e9576f5da7",
    ),
    (
        [{"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
          "probability": 0.5, "weight": 0.1, "mechanism": "ampa",
          "allow_self_connections": True}],
        {"seed": 3},
        4,
        "d65baa24a1073220f8ea39af0b68b770036788a80272bc6050dd1fd4ac6b55c2",
    ),
    (
        [{"name": "rec", "source": {"area": "V1", "cell_type": "E"},
          "target": {"area": "V1", "cell_type": "E"}, "weight": 0.1,
          "mechanism": "ampa"}],
        {},
        2,
        "8bd2fa7c6873c97e24da3767da43702d8c85aadb7136ed816c324b1ebc6b26d2",
    ),
]


def _rows(n, area, cell="E", start=0, layer="L4"):
    return [
        {"neuron_id": start + i, "area": area, "layer": layer, "cell_type": cell}
        for i in range(n)
    ]


def _edge_hash(result):
    h = hashlib.sha256()
    h.update(np.asarray(result.edge_pre).tobytes())
    h.update(np.asarray(result.edge_post).tobytes())
    return h.hexdigest()


def _counts_200x300(sampling, seeds):
    neurons = _rows(200, "PRE") + _rows(300, "POST", start=200)
    rule = {"name": "feed", "source": {"area": "PRE"}, "target": {"area": "POST"},
            "probability": 0.1, "weight": 0.1, "mechanism": "ampa"}
    if sampling is not None:
        rule["sampling"] = sampling
    return [
        jtfne.compile_connection_rules(neurons, [dict(rule)], MECHS, seed=s).n_edges
        for s in seeds
    ]


def test_invalid_sampling_raises_value_error():
    with pytest.raises(ValueError, match="sampling"):
        jtfne.configuration().connections(
            name="x", source={}, target={}, sampling="bogus")
    neurons = _rows(4, "V1")
    with pytest.raises(ValueError, match="sampling"):
        jtfne.compile_connection_rules(
            neurons,
            [{"name": "x", "source": {}, "target": {}, "probability": 0.5,
              "weight": 0.1, "mechanism": "ampa", "sampling": "bogus"}],
            MECHS, seed=0)


def test_connections_stores_sampling():
    """The default stores no key, so existing config hashes do not change."""
    cfg = jtfne.configuration().connections(name="dflt", source={}, target={})
    rules = cfg.metadata["circuit"]["connections"]
    assert "sampling" not in rules[-1]
    cfg = jtfne.configuration().connections(
        name="bern", source={}, target={}, sampling="bernoulli")
    rules = cfg.metadata["circuit"]["connections"]
    assert rules[-1]["sampling"] == "bernoulli"


@pytest.mark.parametrize("bad", ["Bernoulli", ["bernoulli"], None, 0])
def test_connections_refuses_unknown_sampling(bad):
    with pytest.raises(ValueError, match="sampling"):
        jtfne.configuration().connections(name="x", source={}, target={}, sampling=bad)


def test_connections_refuses_bernoulli_with_max_in_degree():
    with pytest.raises(ValueError, match="max_in_degree"):
        jtfne.configuration().connections(
            name="x", source={}, target={}, max_in_degree=2, sampling="bernoulli")


def test_bernoulli_mean_variance_200x300():
    seeds = list(range(20))
    n, p = 200 * 300, 0.1
    bern = _counts_200x300("bernoulli", seeds)
    assert np.var(bern) > 0
    se_mean = float(np.sqrt(n * p * (1.0 - p) / len(seeds)))
    assert abs(float(np.mean(bern)) - n * p) <= 3.0 * se_mean
    dense = _counts_200x300(None, seeds)
    assert dense == [round(n * p)] * len(seeds)
    assert np.var(dense) == 0


def test_bernoulli_deterministic_given_seed():
    a = _counts_200x300("bernoulli", [7])
    b = _counts_200x300("bernoulli", [7])
    assert a == b


def test_bernoulli_indegree_varies_and_no_self():
    n, p = 200, 0.1
    neurons = _rows(n, "V1")
    rule = {"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
            "probability": p, "weight": 0.1, "mechanism": "ampa",
            "sampling": "bernoulli"}
    r = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=11)
    pre = np.asarray(r.edge_pre)
    post = np.asarray(r.edge_post)
    # Fixture must overlap: every pre id is also a post id.
    assert set(int(x) for x in pre) <= set(range(n))
    assert set(range(n)) == {row["neuron_id"] for row in neurons}
    assert not np.any(pre == post)
    indegree = np.bincount(post, minlength=n)
    assert len(set(int(x) for x in indegree)) > 1


def test_bernoulli_large_rule_completes():
    n, p = 3000, 0.01
    assert n * n > 4_000_000
    neurons = _rows(n, "PRE") + _rows(n, "POST", start=n)
    rule = {"name": "big", "source": {"area": "PRE"}, "target": {"area": "POST"},
            "probability": p, "weight": 0.1, "mechanism": "ampa",
            "sampling": "bernoulli"}
    r = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)
    expected = n * n * p
    se = float(np.sqrt(n * n * p * (1.0 - p)))
    assert abs(r.n_edges - expected) <= 5.0 * se


def test_default_density_edges_unchanged():
    for conns, kw, n_edges, pin in _DENSITY_PINS:
        r = jtfne.compile_connection_rules(_FIXTURE_NEURONS, conns, MECHS, **kw)
        assert r.n_edges == n_edges
        assert _edge_hash(r) == pin


def test_connection_table_records_bernoulli_and_keeps_density_rows_unchanged():
    neurons = _rows(20, "PRE") + _rows(30, "POST", start=20)
    rule = {"name": "feed", "source": {"area": "PRE"}, "target": {"area": "POST"},
            "probability": 0.2, "weight": 0.1, "mechanism": "ampa"}
    bern = jtfne.compile_connection_rules(neurons, [dict(rule, sampling="bernoulli")], MECHS, seed=0)
    dens = jtfne.compile_connection_rules(neurons, [dict(rule)], MECHS, seed=0)
    assert bern.connection_table[0]["status"] == "compiled", "fixture: the rule must compile"
    assert bern.connection_table[0]["sampling"] == "bernoulli"
    assert "sampling" not in dens.connection_table[0]


def _sign_only_labels(n):
    area = ["V1"] * n
    layer = ["L4"] * n
    cell = ["E"] * n
    sign = np.ones(n)
    return area, layer, cell, sign


def test_sign_only_bernoulli_large_is_exact():
    from jaxfne._construct_connectivity import _compile_connection_rules

    n, p = 3000, 0.01
    assert n * n > 4_000_000
    area, layer, cell, sign = _sign_only_labels(n)
    rules = [{"name": "big", "source": {}, "target": {}, "probability": p,
              "weight": 0.5, "sampling": "bernoulli"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        edges, counts = _compile_connection_rules(
            rules, area, layer, cell, sign, n, jnp.float32, 0)
    # Exact Bernoulli: no self edges, count within 5 SE of p*n*(n-1).
    expected = p * n * (n - 1)
    se = float(np.sqrt(n * (n - 1) * p * (1.0 - p)))
    assert abs(counts[0] - expected) <= 5.0 * se
    assert not np.any(np.asarray(edges.pre) == np.asarray(edges.post))


def test_sign_only_invalid_sampling_raises():
    from jaxfne._construct_connectivity import _compile_connection_rules

    n = 8
    area, layer, cell, sign = _sign_only_labels(n)
    rules = [{"name": "bad", "source": {}, "target": {}, "probability": 0.5,
              "weight": 0.5, "sampling": "bogus"}]
    with pytest.raises(ValueError, match="sampling"):
        _compile_connection_rules(rules, area, layer, cell, sign, n,
                                  jnp.float32, 0)


def test_bernoulli_with_max_in_degree_is_refused():
    """Localized sampling ignores `sampling`, so the pair must be refused, not dropped."""
    neurons = [dict(r, x=0.1 * i, y=0.0, z=0.0) for i, r in enumerate(_rows(6, "V1"))]
    rule = {"name": "loc", "source": {}, "target": {}, "weight": 0.1, "mechanism": "ampa",
            "max_in_degree": 2, "sampling": "bernoulli"}
    with pytest.raises(ValueError, match="max_in_degree"):
        jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)
    rule.pop("sampling")
    assert jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0).n_edges > 0


def test_bad_sampling_refused_before_skips():
    """An empty selector, a localized rule or p = 0 must not let a bad value through."""
    from jaxfne._construct_connectivity import _compile_connection_rules

    empty = {"name": "e", "source": {"area": "NONE"}, "target": {}, "weight": 0.1,
             "mechanism": "ampa", "sampling": "bogus"}
    with pytest.raises(ValueError, match="sampling"):
        jtfne.compile_connection_rules(_FIXTURE_NEURONS, [empty], MECHS, allow_empty=True)
    neurons = [dict(r, x=0.1 * i, y=0.0, z=0.0) for i, r in enumerate(_rows(6, "V1"))]
    loc = {"name": "loc", "source": {}, "target": {}, "weight": 0.1, "mechanism": "ampa",
           "max_in_degree": 2, "sampling": "bogus"}
    with pytest.raises(ValueError, match="sampling"):
        jtfne.compile_connection_rules(neurons, [loc], MECHS)
    area, layer, cell, sign = _sign_only_labels(8)
    zero_p = [{"name": "z", "source": {}, "target": {}, "probability": 0.0,
               "weight": 0.5, "sampling": "bogus"}]
    with pytest.raises(ValueError, match="sampling"):
        _compile_connection_rules(zero_p, area, layer, cell, sign, 8, jnp.float32, 0)


def test_bernoulli_tiles_draw_independently(monkeypatch):
    """Equal-shape tiles must not reuse one key (edge counts cannot see that)."""
    import jaxfne.connectivity as conn

    monkeypatch.setattr(conn, "_BERNOULLI_TILE_ELEMENTS", 400)
    n = 40  # 40 x 40 pairs in four 40 x 10 tiles
    neurons = _rows(n, "PRE") + _rows(n, "POST", start=n)
    rule = {"name": "t", "source": {"area": "PRE"}, "target": {"area": "POST"},
            "probability": 0.5, "weight": 0.1, "mechanism": "ampa", "sampling": "bernoulli"}
    r = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)
    mask = np.zeros((n, n), bool)
    mask[np.asarray(r.edge_pre), np.asarray(r.edge_post) - n] = True
    assert mask.sum() > 0, "fixture must draw edges"
    tiles = [mask[:, j:j + 10] for j in range(0, n, 10)]
    assert len({t.tobytes() for t in tiles}) == 4


# F7. capped_bernoulli: the Bernoulli draws truncated to int(n_pre * p) kept
# edges per post cell, scanning pre cells in order.
def _capped_rule(**kw):
    rule = {"name": "feed", "source": {"area": "PRE"}, "target": {"area": "POST"},
            "probability": 0.2, "weight": 0.1, "mechanism": "ampa",
            "sampling": "capped_bernoulli"}
    rule.update(kw)
    return rule


def _capped_fixture(n_pre=50, n_post=40, **kw):
    neurons = _rows(n_pre, "PRE") + _rows(n_post, "POST", start=n_pre)
    return neurons, _capped_rule(**kw)


def _indegrees(result, n_post, offset):
    return np.bincount(np.asarray(result.edge_post) - offset, minlength=n_post)


def test_capped_bernoulli_respects_per_post_cap():
    n_pre, n_post, p = 50, 40, 0.2
    neurons, rule = _capped_fixture(n_pre, n_post, probability=p)
    r = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)
    cap = int(n_pre * p)
    indeg = _indegrees(r, n_post, n_pre)
    assert indeg.max() <= cap, "no post cell keeps more than int(n_pre * p) edges"
    assert indeg.max() == cap, "the fixture must bind the cap on some post cell"
    bern = jtfne.compile_connection_rules(
        neurons, [dict(rule, sampling="bernoulli")], MECHS, seed=0)
    assert r.n_edges <= bern.n_edges, "truncation only removes Bernoulli draws"


def test_capped_nonbinding_equals_bernoulli():
    # n_pre = 10 without self pairs: 9 candidates per post cell, and
    # cap = int(10 * 0.95) = 9 covers every candidate.
    n, p = 10, 0.95
    neurons = _rows(n, "V1")
    base = {"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
            "probability": p, "weight": 0.1, "mechanism": "ampa"}
    capped = jtfne.compile_connection_rules(
        neurons, [dict(base, sampling="capped_bernoulli")], MECHS, seed=0)
    bern = jtfne.compile_connection_rules(
        neurons, [dict(base, sampling="bernoulli")], MECHS, seed=0)
    assert capped.n_edges > 0, "fixture must draw edges"
    np.testing.assert_array_equal(np.asarray(capped.edge_pre), np.asarray(bern.edge_pre))
    np.testing.assert_array_equal(
        np.asarray(capped.edge_post), np.asarray(bern.edge_post))


def test_capped_is_subset_of_bernoulli_and_deterministic():
    neurons, rule = _capped_fixture()
    a = jtfne.compile_connection_rules(neurons, [dict(rule)], MECHS, seed=4)
    b = jtfne.compile_connection_rules(neurons, [dict(rule)], MECHS, seed=4)
    np.testing.assert_array_equal(np.asarray(a.edge_pre), np.asarray(b.edge_pre))
    np.testing.assert_array_equal(np.asarray(a.edge_post), np.asarray(b.edge_post))
    bern = jtfne.compile_connection_rules(
        neurons, [dict(rule, sampling="bernoulli")], MECHS, seed=4)
    capped_pairs = set(
        zip(np.asarray(a.edge_pre).tolist(), np.asarray(a.edge_post).tolist()))
    bern_pairs = set(
        zip(np.asarray(bern.edge_pre).tolist(), np.asarray(bern.edge_post).tolist()))
    assert capped_pairs <= bern_pairs, "capped keeps a subset of the Bernoulli draws"
    assert a.n_edges < bern.n_edges, "the fixture must truncate something"


def test_connections_stores_capped_bernoulli():
    """Non-default sampling is stored, so the default path keeps no key."""
    cfg = jtfne.configuration().connections(
        name="cap", source={}, target={}, sampling="capped_bernoulli")
    assert cfg.metadata["circuit"]["connections"][-1]["sampling"] == "capped_bernoulli"


def test_connections_refuses_capped_bernoulli_with_max_in_degree():
    with pytest.raises(ValueError, match="max_in_degree"):
        jtfne.configuration().connections(
            name="x", source={}, target={}, max_in_degree=2,
            sampling="capped_bernoulli")


def test_connection_table_records_capped_and_keeps_density_rows_unchanged():
    neurons = _rows(20, "PRE") + _rows(30, "POST", start=20)
    rule = {"name": "feed", "source": {"area": "PRE"}, "target": {"area": "POST"},
            "probability": 0.2, "weight": 0.1, "mechanism": "ampa"}
    capped = jtfne.compile_connection_rules(
        neurons, [dict(rule, sampling="capped_bernoulli")], MECHS, seed=0)
    dens = jtfne.compile_connection_rules(neurons, [dict(rule)], MECHS, seed=0)
    assert capped.connection_table[0]["status"] == "compiled"
    assert capped.connection_table[0]["sampling"] == "capped_bernoulli"
    assert "sampling" not in dens.connection_table[0]


def test_capped_bernoulli_with_max_in_degree_is_refused():
    """Localized sampling ignores `sampling`, so the pair must be refused, not dropped."""
    neurons = [dict(r, x=0.1 * i, y=0.0, z=0.0) for i, r in enumerate(_rows(6, "V1"))]
    rule = {"name": "loc", "source": {}, "target": {}, "weight": 0.1, "mechanism": "ampa",
            "max_in_degree": 2, "sampling": "capped_bernoulli"}
    with pytest.raises(ValueError, match="max_in_degree"):
        jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)


def test_sign_only_capped_bernoulli_is_refused_while_full_path_compiles():
    """The sign-only fallback honours density/bernoulli only: capped stays
    fail-closed there, while the full compiler path accepts it."""
    from jaxfne._construct_connectivity import _compile_connection_rules

    neurons, rule = _capped_fixture(n_pre=8, n_post=8, probability=0.5)
    full = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)
    assert full.n_edges > 0, "the full path must compile capped_bernoulli"
    area, layer, cell, sign = _sign_only_labels(8)
    rules = [{"name": "c", "source": {}, "target": {}, "probability": 0.5,
              "weight": 0.5, "sampling": "capped_bernoulli"}]
    with pytest.raises(ValueError, match="sampling"):
        _compile_connection_rules(rules, area, layer, cell, sign, 8, jnp.float32, 0)


# Tensor sampling: InterConnection/AreaConnection declare the compiler mode
# (None = the density default); the bridges forward it into the compiled rule.
def _tensor_conn_kwargs():
    from jaxfne import neuronal_tensor as nt

    return dict(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(dT_ms=5.0, reversal_potentials_mV={"AMPA": 0.0}),
        delay_ms=1.0,
    )


def _tensor_conn(cls, **kw):
    from jaxfne import neuronal_tensor as nt

    kwargs = _tensor_conn_kwargs()
    kwargs.update(kw)
    if cls == "inter":
        return nt.InterConnection(**kwargs)
    return nt.AreaConnection(
        source_area="V1", target_area="V4", **kwargs
    )


@pytest.mark.parametrize("cls", ["inter", "area"])
def test_tensor_sampling_defaults_to_none_and_round_trips(cls):
    assert _tensor_conn(cls).sampling is None
    assert _tensor_conn(cls, sampling="capped_bernoulli").sampling == "capped_bernoulli"


@pytest.mark.parametrize("cls", ["inter", "area"])
@pytest.mark.parametrize("bad", ["bogus", "Bernoulli", 0, ["capped_bernoulli"]])
def test_tensor_sampling_refusals_name_sampling(cls, bad):
    with pytest.raises(ValueError, match="sampling"):
        _tensor_conn(cls, sampling=bad)


def _sampling_dict_tensor(cls, sampling):
    from jaxfne import neuronal_tensor as nt

    def area(name, conns):
        return nt.Area(
            name=name,
            layers=[nt.Layer(
                name="L4", n_neurons=2,
                neuron_types=[nt.NeuronType.make("E", emitter="adex")])],
            **({"inter_connections": conns} if cls == "inter" else {}),
        )

    conn = _tensor_conn(cls, sampling=sampling)
    if cls == "inter":
        return nt.NeuronalTensor(
            name="sampling-dict", areas=[area("V1", [conn])])
    return nt.NeuronalTensor(
        name="sampling-dict",
        areas=[area("V1", []), area("V4", [])],
        area_connections=[conn],
    )


@pytest.mark.parametrize("cls", ["inter", "area"])
def test_tensor_sampling_to_dict_omits_default_and_writes_set(cls):
    tensor = _sampling_dict_tensor(cls, None)
    if cls == "inter":
        assert "sampling" not in tensor.to_dict()["areas"][0]["inter_connections"][0]
        raw = _sampling_dict_tensor(
            cls, "capped_bernoulli").to_dict()["areas"][0]["inter_connections"][0]
    else:
        assert "sampling" not in tensor.to_dict()["area_connections"][0]
        raw = _sampling_dict_tensor(
            cls, "capped_bernoulli").to_dict()["area_connections"][0]
    assert raw["sampling"] == "capped_bernoulli"


def _sampling_tensor(sampling):
    from jaxfne import neuronal_tensor as nt

    types = [nt.NeuronType.make("E", emitter="adex")]
    conn = _tensor_conn("inter", sampling=sampling)
    return nt.NeuronalTensor(
        name="sampling",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=8, neuron_types=types)],
            inter_connections=[conn])],
    )


# F8. multinomial: exactly round(p*n_post) draws with replacement (multapses
# allowed). Each of the K draws picks pre and post independently and
# uniformly (SOURCE_P1.md section 1), so a (pre, post) pair may repeat.
def _mult_rule(**kw):
    rule = {"name": "feed", "source": {"area": "PRE"}, "target": {"area": "POST"},
            "probability": 0.5, "weight": 0.1, "mechanism": "ampa",
            "sampling": "multinomial"}
    rule.update(kw)
    return rule


def test_multinomial_exact_count_per_projection():
    neurons = _rows(8, "PRE") + _rows(8, "POST", start=8)
    for seed in range(5):
        r = jtfne.compile_connection_rules(neurons, [_mult_rule()], MECHS, seed=seed)
        assert r.n_edges == 4, "K = round(0.5 * n_post) exactly, every seed"


def test_multinomial_deterministic_given_seed():
    neurons = _rows(8, "PRE") + _rows(8, "POST", start=8)
    a = jtfne.compile_connection_rules(neurons, [_mult_rule()], MECHS, seed=7)
    b = jtfne.compile_connection_rules(neurons, [_mult_rule()], MECHS, seed=7)
    np.testing.assert_array_equal(np.asarray(a.edge_pre), np.asarray(b.edge_pre))
    np.testing.assert_array_equal(np.asarray(a.edge_post), np.asarray(b.edge_post))


def test_multinomial_draws_with_replacement():
    # 1 pre x 2 post, p = 1.0 -> K = 2 draws from 2 pairs. A
    # without-replacement sampler could never repeat a pair, so any repeated
    # pair across seeds proves replacement (P(none in 40 seeds) = 2**-40).
    neurons = _rows(1, "PRE") + _rows(2, "POST", start=1)
    seen_dup = False
    for seed in range(40):
        r = jtfne.compile_connection_rules(
            neurons, [_mult_rule(probability=1.0)], MECHS, seed=seed)
        assert r.n_edges == 2, "K = round(1.0 * n_post) stays exact"
        pairs = list(zip(np.asarray(r.edge_pre).tolist(),
                         np.asarray(r.edge_post).tolist()))
        if len(set(pairs)) < len(pairs):
            seen_dup = True
    assert seen_dup, "40 seeds must show a repeated (pre, post) pair"


def test_multinomial_recurrent_excludes_self_and_stays_exact():
    neurons = _rows(4, "V1")
    rule = {"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
            "probability": 0.5, "weight": 0.1, "mechanism": "ampa",
            "sampling": "multinomial"}
    r = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)
    assert r.n_edges == 2, "K = round(0.5 * n_post) exact with self excluded"
    assert not np.any(np.asarray(r.edge_pre) == np.asarray(r.edge_post)), \
        "allow_self=False drops self pairs"


def test_multinomial_with_in_degree_is_refused():
    neurons = _rows(4, "PRE") + _rows(4, "POST", start=4)
    rule = {"name": "m", "source": {"area": "PRE"}, "target": {"area": "POST"},
            "weight": 0.1, "mechanism": "ampa",
            "sampling": "multinomial", "in_degree": 2}
    with pytest.raises(ValueError, match="with in_degree"):
        jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)


def test_multinomial_recurrent_single_neuron_is_refused_not_empty():
    # 1 pre x 1 post, same id: every pair is a refused self pair, so a K>0
    # multinomial rule has an empty valid pair space -- refused, never a
    # silent empty edge list.
    neurons = _rows(1, "V1")
    rule = {"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
            "probability": 0.5, "weight": 0.1, "mechanism": "ampa",
            "sampling": "multinomial"}
    with pytest.raises(ValueError, match="empty valid pair space|refused self pair"):
        jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)


def test_multinomial_self_excluded_count_rounds_half_up():
    # 5 recurrent neurons, p = 0.5: K = round_half_up(2.5) = 3 (banker's
    # rounding would give 2). Exact K, no self pairs, every seed.
    neurons = _rows(5, "V1")
    rule = {"name": "rec", "source": {"area": "V1"}, "target": {"area": "V1"},
            "probability": 0.5, "weight": 0.1, "mechanism": "ampa",
            "sampling": "multinomial"}
    for seed in range(5):
        r = jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=seed)
        assert r.n_edges == 3, "half-up: K = 3 exactly, every seed"
        assert not np.any(
            np.asarray(r.edge_pre) == np.asarray(r.edge_post)
        ), "self pairs stay excluded"


def test_compile_refuses_probability_out_of_range():
    # The compiler owns the rule even when the builder is bypassed.
    neurons = _rows(4, "PRE") + _rows(4, "POST", start=4)
    for p in (1.5, -0.1):
        rule = _mult_rule(probability=p)
        with pytest.raises(ValueError, match="finite number in \\[0, 1\\]"):
            jtfne.compile_connection_rules(neurons, [rule], MECHS, seed=0)


def test_duplicate_guard_flags_cross_block_repeat():
    # The construct-time duplicate guard: an identity repeated entirely inside
    # ONE declared multinomial block is the declared with-replacement outcome;
    # the same identity repeating across two blocks (or spanning an exempt and
    # a non-exempt source) is an accident and raises.
    from jaxfne._construct_core import _reject_duplicate_explicit_edges
    from jaxfne.emitters import EdgeList

    def el(pairs):
        arr = np.asarray(pairs, dtype=np.int64)
        n = arr.shape[0]
        return EdgeList(
            pre=jnp.asarray(arr[:, 0]),
            post=jnp.asarray(arr[:, 1]),
            weight=jnp.ones(n),
            receptor_index=jnp.zeros(n, dtype=jnp.int32),
            tau_ms=jnp.ones(n),
        )

    both = el([(0, 1), (0, 1)])
    with pytest.raises(ValueError, match="duplicate executable synaptic identities"):
        _reject_duplicate_explicit_edges(both, exempt_ranges=[(0, 1)])
    _reject_duplicate_explicit_edges(both, exempt_ranges=[(0, 2)])
    with pytest.raises(ValueError, match="duplicate executable synaptic identities"):
        _reject_duplicate_explicit_edges(both)


def test_b2b_net_m_dense_model_consumes_the_order():
    import jaxfne as jtfne

    def cfg(order):
        return (jtfne.configuration().network(n=6)
                .emitter(family="izhikevich", izhikevich_update_order=order)
                .field().probe(name="p"))

    # noise_scale rides the Simulation-level hdp_params transport; the config
    # metadata form is dropped by resolved_runtime (observed 2026-10-22).
    sim = jtfne.Simulation(
        duration_ms=5.0, dt_ms=1.0, seed=0,
        runtime=jtfne.RuntimeConfig(hdp_params={"noise_scale": 0.0}))
    s_def = np.asarray(jtfne.construct(cfg("default")).simulate(sim).spikes)
    s_net = np.asarray(jtfne.construct(cfg("net_m")).simulate(sim).spikes)
    assert s_net.shape == s_def.shape
    assert not np.array_equal(s_net, s_def), \
        "update_order='net_m' must be a realized change at the same seed"


def test_b2b_net_m_simulate_refuses_dt_and_edge_list():
    import jaxfne as jtfne

    cfg = (jtfne.configuration().network(n=6)
           .emitter(family="izhikevich", izhikevich_update_order="net_m")
           .field().probe(name="p"))

    model = jtfne.construct(cfg)
    with pytest.raises(ValueError, match="dt_ms must be 1.0"):
        model.simulate(jtfne.Simulation(
            duration_ms=5.0, dt_ms=0.5, seed=0,
            runtime=jtfne.RuntimeConfig(hdp_params={"noise_scale": 0.0})))
    model_el = jtfne.construct(cfg.runtime(recurrent_backend="edge_list"))
    with pytest.raises(ValueError, match="dense"):
        model_el.simulate(jtfne.Simulation(
            duration_ms=5.0, dt_ms=1.0, seed=0,
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list",
                                        hdp_params={"noise_scale": 0.0})))


def test_connections_refuses_multinomial_with_in_degree():
    with pytest.raises(ValueError, match="with in_degree"):
        jtfne.configuration().connections(
            name="x", source={}, target={}, sampling="multinomial",
            in_degree=2, probability=None)


def test_sign_only_multinomial_is_refused_while_full_path_compiles():
    from jaxfne._construct_connectivity import _compile_connection_rules

    neurons = _rows(8, "PRE") + _rows(8, "POST", start=8)
    full = jtfne.compile_connection_rules(neurons, [_mult_rule()], MECHS, seed=0)
    assert full.n_edges == 4, "the full path must compile multinomial"
    area, layer, cell, sign = _sign_only_labels(8)
    rules = [{"name": "m", "source": {}, "target": {}, "probability": 0.5,
              "weight": 0.5, "sampling": "multinomial"}]
    with pytest.raises(ValueError, match="sampling"):
        _compile_connection_rules(rules, area, layer, cell, sign, 8, jnp.float32, 0)


def test_construct_sign_only_multinomial_is_refused():
    cfg = (jtfne.laminar_cortex_config(
        seed=0, duration_ms=50.0, dt_ms=0.5, areas=("V1",), layers=("L4",),
        cell_types={"E": 0.5, "PV": 0.5}, n=12, emitter="izhikevich")
        .connections(name="m", source={}, target={}, probability=0.5,
                     weight=0.5, sampling="multinomial"))
    with pytest.raises(ValueError, match="mechanism-aware"):
        jtfne.construct(cfg)


@pytest.mark.parametrize("cls", ["inter", "area"])
def test_tensor_multinomial_round_trips(cls):
    assert _tensor_conn(cls, sampling="multinomial").sampling == "multinomial"


@pytest.mark.parametrize("cls", ["inter", "area"])
def test_tensor_multinomial_with_in_degree_refused(cls):
    with pytest.raises(ValueError, match="with sampling"):
        _tensor_conn(cls, sampling="multinomial", in_degree=2)


def test_tensor_sampling_forwards_to_the_adex_rule():
    from jaxfne import neuronal_tensor as nt

    kw = dict(seed=0, duration_ms=20.0, dt_ms=0.5)
    cfg = nt.neuronal_tensor_to_adex_configuration(
        _sampling_tensor("capped_bernoulli"), **kw)
    rules = cfg.metadata["circuit"]["connections"]
    assert rules[-1]["sampling"] == "capped_bernoulli"
    plain = nt.neuronal_tensor_to_adex_configuration(_sampling_tensor(None), **kw)
    assert "sampling" not in plain.metadata["circuit"]["connections"][-1]
