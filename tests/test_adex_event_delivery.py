"""AdEx route: event-driven delayed-edge delivery (RuntimeConfig(edge_delivery="event"))."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._construct_adex import adex_external_drive_kwargs, adex_kernel_args
from jaxfne._model_simulate import simulate_adex_seed_batch
from jaxfne.emitters_adex import _event_delivery, make_adex_params, simulate_adex

DT_MS = 0.1


def _random_edges(seed, n_src, n_out, n_edges, sort_by_src):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_src, n_edges)
    seg = rng.integers(0, n_out, n_edges)
    if sort_by_src:
        order = np.argsort(src, kind="stable")
        src, seg = src[order], seg[order]
    w = rng.uniform(0.1, 5.0, n_edges)
    return src, seg, w


@pytest.mark.parametrize("gain", [False, True])
def test_event_delivery_equals_all_edges_sum_on_every_tier(gain):
    """Each tier, the overflow branch and the empty step give the all-edges segment sum bit-for-bit."""
    n_src, n_out, n_edges = 600, 400, 30000
    src, seg, w = _random_edges(0, n_src, n_out, n_edges, sort_by_src=True)
    mask = np.random.default_rng(1).random(n_edges) < 0.3 if gain else None
    deliver, info = _event_delivery(src, n_src, seg, w, mask, n_out, jnp.float32)
    assert info["order_exact"], "fixture: edges sorted by source give the all-edges order"
    assert info["tiers"] == [8, 64], "fixture: two tiers and the overflow branch"
    w32 = jnp.asarray(w, jnp.float32)
    g_t = jnp.asarray(0.37, jnp.float32)
    rng = np.random.default_rng(2)
    for n_active in (0, 1, 8, 9, 64, 65, n_src):
        active = np.zeros(n_src, bool)
        active[rng.choice(n_src, n_active, replace=False)] = True
        w_eff = w32 if mask is None else w32 * jnp.where(jnp.asarray(mask), g_t, 1.0)
        ref = jax.ops.segment_sum(
            jnp.where(jnp.asarray(active)[src], w_eff, 0.0), jnp.asarray(seg), num_segments=n_out)
        got = deliver(jnp.asarray(active), g_t if gain else None)
        assert np.array_equal(np.asarray(got), np.asarray(ref)), n_active


def test_event_delivery_order_flag_detects_unsorted_sources():
    src, seg, w = _random_edges(0, 600, 400, 30000, sort_by_src=False)
    _, info = _event_delivery(src, 600, seg, w, None, 400, jnp.float32)
    assert not info["order_exact"]


def _net(N=300, p=0.15, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    nE = int(0.8 * N)
    params = make_adex_params(["E"] * nE + ["PV"] * (N - nE))
    pre, post = np.nonzero(rng.random((N, N)) < p)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    if shuffle:
        order = rng.permutation(pre.size)
        pre, post = pre[order], post[order]
    rec = (pre >= nE).astype(np.int64)
    kw = dict(
        dt_ms=DT_MS, edge_pre=pre, edge_post=post, edge_weight_nS=np.where(rec == 0, 2.0, 5.0),
        edge_receptor=rec, edge_delay_steps=np.where(rec == 0, 20, 15) + (pre % 3),
        receptor_reversal_mV=(0.0, -80.0), receptor_tau_ms=(2.0, 8.0), decay="euler",
        ext_targets=np.arange(N), ext_lambda=0.5, ext_weight_nS=2.5, ext_receptor=0,
        ext_delay_steps=20, key=jax.random.PRNGKey(3),
    )
    return params, kw


def _active_counts(spikes, pre, lag):
    """Per step, the number of distinct (delay, pre) sources whose cell spiked delay steps earlier."""
    keys = np.unique(np.asarray(lag) * spikes.shape[1] + np.asarray(pre))
    s_pre, s_lag = keys % spikes.shape[1], keys // spikes.shape[1]
    t = np.arange(spikes.shape[0])[:, None] - s_lag[None, :]
    return np.where(t >= 0, spikes[np.clip(t, 0, None), s_pre], False).sum(axis=1)


def test_kernel_event_bit_identical_with_mixed_delays_and_overflow():
    params, kw = _net()
    n_steps = 3000
    # A 1 ms pulse at 150 ms fires most cells in one step, so the arrivals overflow every tier.
    pulse = np.zeros((n_steps, 300), np.float32)
    pulse[1500:1510] = 30000.0
    kw["I_ext_pA"] = pulse
    ref = simulate_adex(params, n_steps=n_steps, **kw, record_v=True)
    got = simulate_adex(params, n_steps=n_steps, **kw, record_v=True, edge_delivery="event")
    info = got["event_delivery"]
    assert info["order_exact"] and info["tiers"] == [8, 64], "fixture: exact order, two tiers"
    counts = _active_counts(np.asarray(ref["spikes"]), kw["edge_pre"], kw["edge_delay_steps"])
    assert (counts == 0).any() and ((counts > 0) & (counts <= 8)).any(), "fixture must run tier 8"
    assert ((counts > 8) & (counts <= 64)).any(), "fixture must run tier 64"
    assert (counts > 64).any(), "fixture must overflow to the all-edges branch"
    assert "event_delivery" not in ref
    for k in ("spikes", "V_mV", "g", "w", "V"):
        assert np.array_equal(np.asarray(got[k]), np.asarray(ref[k])), k


def test_kernel_event_with_edge_gain_bit_identical():
    params, kw = _net()
    n_steps = 1500
    n_edges = kw["edge_pre"].size
    gain = dict(edge_gain_mask=kw["edge_receptor"] == 0,
                edge_gain_schedule=np.linspace(0.5, 1.5, n_steps))
    ref = simulate_adex(params, n_steps=n_steps, **kw, **gain)
    got = simulate_adex(params, n_steps=n_steps, **kw, **gain, edge_delivery="event")
    assert np.asarray(ref["spikes"]).sum() > 0, "fixture must spike"
    assert gain["edge_gain_mask"].any() and n_edges > gain["edge_gain_mask"].sum()
    assert np.array_equal(np.asarray(got["spikes"]), np.asarray(ref["spikes"]))
    assert np.array_equal(np.asarray(got["g"]), np.asarray(ref["g"]))


def test_kernel_event_unsorted_edges_reports_inexact_order():
    params, kw = _net(shuffle=True)
    got = simulate_adex(params, n_steps=200, **kw, edge_delivery="event")
    assert got["event_delivery"]["order_exact"] is False


def test_kernel_refuses_unknown_delivery():
    params, kw = _net(N=20)
    with pytest.raises(ValueError, match="edge_delivery must be one of"):
        simulate_adex(params, n_steps=10, **kw, edge_delivery="sparse")


def _adex_model():
    cfg = (
        jtfne.Configuration()
        .network(name="adex200", n=200, cell_types={"E": 0.8, "PV": 0.1, "SOM": 0.1})
        .emitter(family="adex", adex_decay="euler")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": 8.0, "reversal_mV": -80.0})
        .connections(name="e_all", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.2, weight=2.0, mechanism="ampa", delay_ms=2.0, sampling="bernoulli")
        .connections(name="e_pv", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=0.2, weight=3.0, mechanism="ampa", delay_ms=2.0, sampling="bernoulli")
        .connections(name="pv_e", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=0.2, weight=5.0, mechanism="gaba", delay_ms=2.0, sampling="bernoulli")
        .connectivity(edge_seed=5)
    )
    return jtfne.construct(cfg)


_DRIVE = {"n_sources": 560, "rate_hz": 4.0, "weight_nS": 2.5, "mechanism": "ampa",
          "target_cell_types": ["E", "PV"], "delay_ms": 2.0, "seed": 7}


def _route_sim(delivery=None):
    runtime = None if delivery is None else jtfne.RuntimeConfig(
        recurrent_backend="edge_list", edge_delivery=delivery)
    return jtfne.Simulation(duration_ms=300.0, dt_ms=DT_MS, poisson_drive=_DRIVE, runtime=runtime)


def test_route_event_equals_default_and_records_delivery():
    model = _adex_model()
    ref = model.simulate(_route_sim())
    got = model.simulate(_route_sim("event"))
    assert np.asarray(ref.spikes).sum() > 0, "fixture must spike"
    assert np.array_equal(np.asarray(got.spikes), np.asarray(ref.spikes))
    assert "edge_delivery" not in ref.metadata
    rec = got.metadata["edge_delivery"]
    assert rec["mode"] == "event" and rec["order_exact"] is True


def test_route_default_delivery_is_all_edges():
    assert jtfne.RuntimeConfig().edge_delivery == "all_edges"
    with pytest.raises(ValueError, match="edge_delivery must be"):
        jtfne.RuntimeConfig(edge_delivery="sparse")


def test_event_refused_off_the_serial_adex_route():
    model = _adex_model()
    with pytest.raises(ValueError, match="serial AdEx simulate route only"):
        simulate_adex_seed_batch(model, _route_sim("event"), [1, 2])
    izh = jtfne.construct(
        jtfne.configuration().network(n=12, cell_types={"E": 0.75, "PV": 0.25})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy").probe(name="p", n_contacts=4))
    for backend in ("dense", "edge_list"):
        sim = jtfne.Simulation(duration_ms=5.0, dt_ms=DT_MS, runtime=jtfne.RuntimeConfig(
            recurrent_backend=backend, edge_delivery="event"))
        with pytest.raises(ValueError, match="serial AdEx simulate route only"):
            izh.simulate(sim)
        with pytest.raises(ValueError, match="serial AdEx simulate route only"):
            izh.simulate_batch(sim, n_seeds=2)


def test_route_kernel_args_match_direct_event_kernel():
    """The route passes edge_delivery to the kernel and nothing else changes."""
    model = _adex_model()
    sim = _route_sim("event")
    kw = adex_kernel_args(model, DT_MS, dtype="float32")
    direct = simulate_adex(**kw, **adex_external_drive_kwargs(model, sim), n_steps=sim.n_steps,
                           dt_ms=DT_MS, edge_delivery="event")
    routed = model.simulate(sim)
    assert np.array_equal(np.asarray(routed.spikes), np.asarray(direct["spikes"]))
    assert dataclasses.replace(sim.resolved_runtime, edge_delivery="all_edges") == \
        _route_sim("all_edges").resolved_runtime
