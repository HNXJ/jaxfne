"""W13: realized connectivity must equal the request, or the request must raise.

Central invariant: C_requested --realize--> C_realized, or explicit error. Never a
silent substitution. Each case below encodes one previously-silent failure mode.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxfne as jtfne

N = 200
K_MAX = 10
DENSE_E = N * (N - 1)
MECH = dict(name="ampa", kind="exponential", params={"tau_ms": 2.0, "receptor": "AMPA"})


def _base():
    return jtfne.Configuration().runtime(seed=1, dtype="float32", duration_ms=20.0, dt_ms=0.5)


def _finish(cfg):
    return (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )


def _column(**conn):
    cfg = _base().column(name="c", layers=["L4"], n=N).cell_types(
        {"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03}
    )
    if conn:
        cfg = cfg.connectivity(**conn)
    return _finish(cfg)


def _realized(model):
    edge_list = model.params["edge_list"]
    post = np.asarray(edge_list.post)
    max_in = int(np.bincount(post, minlength=1).max()) if post.size else 0
    return int(edge_list.n_edges), max_in


# ── D3: zero is a request, not an unset value ────────────────────────────────
def test_p_connect_zero_yields_no_within_area_edges():
    edges, max_in = _realized(jtfne.construct(_column(p_connect=0.0)))
    assert edges == 0 and max_in == 0


def test_p_connect_fraction_is_realized_not_ignored():
    edges, _ = _realized(jtfne.construct(_column(p_connect=0.01)))
    assert edges < 0.5 * DENSE_E, "a 100x sparsity request must reduce realized edges"


# ── .network() route: honour the request or refuse it ────────────────────────
@pytest.mark.parametrize("spelling", ["connectivity", "network"])
def test_network_route_refuses_a_p_connect_it_cannot_honour(spelling):
    cfg = _base().network(**({"n": N, "p_connect": 0.01} if spelling == "network" else {"n": N}))
    if spelling == "connectivity":
        cfg = cfg.connectivity(p_connect=0.01)
    with pytest.raises(ValueError, match="cannot be honoured by this construction route"):
        jtfne.construct(_finish(cfg))


# ── D1: an explicit-None spatial_sigma must fall back, not crash ─────────────
def test_max_in_degree_does_not_require_an_undocumented_spatial_sigma():
    cfg = (
        _column(p_connect=0.0)
        .mechanisms(**MECH)
        .connections(
            name="rec", source={}, target={}, mechanism="ampa", weight=0.03, max_in_degree=K_MAX
        )
    )
    edges, max_in = _realized(jtfne.construct(cfg))
    assert max_in <= K_MAX
    assert edges == N * K_MAX


# ── D2: a cap the compiler cannot apply must refuse, never silently drop ─────
def test_max_in_degree_without_a_resolvable_mechanism_refuses():
    cfg = _column().connections(
        name="rec", source={}, target={}, weight=0.03, max_in_degree=K_MAX, spatial_sigma=0.1
    )
    with pytest.raises(ValueError, match="max_in_degree"):
        jtfne.construct(cfg)


def test_bounded_degree_is_linear_in_n_when_the_dense_baseline_is_suppressed():
    """E = N * K_max exactly -- the O(N*K) construction the reduction programme needs."""
    for n in (100, 200):
        cfg = (
            _finish(
                _base()
                .column(name="c", layers=["L4"], n=n)
                .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
                .connectivity(p_connect=0.0)
            )
            .mechanisms(**MECH)
            .connections(
                name="rec", source={}, target={}, mechanism="ampa", weight=0.03,
                max_in_degree=K_MAX, spatial_sigma=0.1,
            )
        )
        edges, max_in = _realized(jtfne.construct(cfg))
        assert edges == n * K_MAX and max_in == K_MAX
