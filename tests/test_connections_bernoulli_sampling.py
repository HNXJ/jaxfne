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
