"""PCL LIF core kernel (jaxfne/emitters_lif.py): single-cell trajectory, floor, refractory, hook, refusals."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters_lif import (
    LIF_CELL_TYPE_DEFAULTS,
    lif_step,
    make_lif_params,
    simulate_lif_edge,
)


_LIF_FLOAT_FIELDS = (
    "tau_m_ms", "v_thresh_mV", "v_reset_mV", "v_min_mV", "eta_rp_mV",
    "tau_rp_ms", "tau_sra_ms", "delta_sra_mV",
)


def _as_f32(params):
    """Cast every float column of a LIFParams to float32 (labels untouched)."""
    return params._replace(
        **{f: jnp.asarray(getattr(params, f), jnp.float32) for f in _LIF_FLOAT_FIELDS}
    )


def _simple_f32():
    return _as_f32(make_lif_params(("simple",)))


def test_defaults_match_port_spec():
    d = LIF_CELL_TYPE_DEFAULTS["simple"]
    assert (d["tau_m_ms"], d["v_thresh_mV"], d["v_reset_mV"], d["v_min_mV"]) == (18.0, 10.0, -10.0, -20.0)
    assert (d["eta_rp_mV"], d["tau_rp_ms"]) == (10.0, 5.0)
    d = LIF_CELL_TYPE_DEFAULTS["complex"]
    assert (d["tau_m_ms"], d["v_thresh_mV"], d["v_reset_mV"], d["v_min_mV"]) == (50.0, 3.0, -10.0, -20.0)
    assert d["delta_sra_mV"] == 0.0


def test_make_lif_params_refusals():
    try:
        make_lif_params(("pyramidal",))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown cell type must raise")
    try:
        make_lif_params(("simple",), {"simple": {"no_such_field": 1.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown field must raise")
    try:
        make_lif_params(("simple",), {"complex": {"tau_m_ms": 9.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("override for absent type must raise")


def test_membrane_decay_closed_form():
    p = _simple_f32()
    out = simulate_lif_edge(p, n_steps=10, dt_ms=0.5, V0_mV=jnp.array([5.0], jnp.float32))
    got = float(out["V"][0])
    want = 5.0 * math.exp(-5.0 / 18.0)
    assert abs(got - want) < 1e-5, (got, want)
    assert not bool(out["spikes"].any())


def test_input_adds_weight_and_floor():
    p = _simple_f32()
    out = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        I_ext_pA=jnp.array([[7.0]], jnp.float32),
    )
    assert abs(float(out["V"][0]) - 7.0) < 1e-6
    out = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        I_ext_pA=jnp.array([[-50.0]], jnp.float32),
    )
    assert float(out["V"][0]) == -20.0  # v_min floor, Neuron.cpp:26
    assert not bool(out["spikes"].any())


def test_refractory_subtraction_per_arrival():
    p = _simple_f32()
    V = jnp.array([-10.0], jnp.float32)  # just spiked and reset
    A = jnp.array([0.0], jnp.float32)
    t_last = jnp.array([0.0], jnp.float32)
    (V1, A1, t1), spk = lif_step((V, A, t_last), jnp.array([0.0]), jnp.array([1.0]),
                                 0.5, p, 0.5)
    want = float(-10.0 * math.exp(-0.5 / 18.0) - 10.0 * math.exp(-0.5 / 5.0))
    assert abs(float(V1[0]) - want) < 1e-5, (float(V1[0]), want)
    assert not bool(spk[0])
    # No arrival: decay only, no refractory subtraction (event-driven C++ does nothing).
    (V2, _, _), _ = lif_step((V, A, t_last), jnp.array([0.0]), jnp.array([0.0]),
                             0.5, p, 0.5)
    assert abs(float(V2[0]) - float(-10.0 * math.exp(-0.5 / 18.0))) < 1e-6


def test_threshold_reset_and_adaptation_increment():
    p = _as_f32(make_lif_params(("simple",), {"simple": {"delta_sra_mV": 2.0}}))
    V = jnp.array([9.0], jnp.float32)
    A = jnp.array([0.0], jnp.float32)
    t_last = jnp.array([-1.0e30], jnp.float32)
    (V1, A1, t1), spk = lif_step((V, A, t_last), jnp.array([2.0]), jnp.array([1.0]),
                                 0.5, p, 0.5)
    assert bool(spk[0])
    assert float(V1[0]) == -10.0  # reset
    assert abs(float(A1[0]) - 2.0) < 1e-6  # spike-rate adaptation increment
    assert float(t1[0]) == 0.5


def test_delayed_drive_spikes_postsynaptic_cell():
    p = _as_f32(make_lif_params(("simple", "simple")))
    I_ext = np.zeros((3, 2), np.float32)
    I_ext[0, 0] = 30.0
    out = simulate_lif_edge(
        p, n_steps=3, dt_ms=0.5,
        edge_pre=np.array([0]), edge_post=np.array([1]),
        edge_weight=np.array([12.0]), edge_delay_steps=np.array([1]),
        I_ext_pA=jnp.asarray(I_ext),
    )
    spk = np.asarray(out["spikes"])
    assert spk[0, 0] and not spk[0, 1]
    assert spk[1, 1], spk  # 12 mV arrival crosses threshold 10
    assert not spk[1, 0]


def test_rule_hook_context_and_static_weights():
    from jaxfne.hdp_rule import (
        HDPRuleDescriptor,
        HDPRuleUpdate,
        list_registered_hdp_rules,
        register_hdp_rule,
    )

    seen = []

    def _probe(ctx):
        # Shapes and pre_sp presence are static (visible at trace time); the
        # per-step invocation itself is proven numerically via H_final below
        # (lax.scan traces body once, so a call counter would read 1).
        seen.append((int(ctx.n_neurons), float(ctx.dt), tuple(ctx.pre.shape),
                     tuple(ctx.post.shape), tuple(ctx.spikes.shape),
                     ctx.pre_sp is not None))
        return HDPRuleUpdate(dH=jnp.full_like(ctx.H, 0.4),
                             d_theta={"edge_weight": jnp.zeros_like(ctx.w)})

    name = "test_lif_probe_hook"
    if name not in list_registered_hdp_rules():
        register_hdp_rule(
            HDPRuleDescriptor(name=name, h_coords=("H",), theta_targets=("edge_weight",),
                              scope="node", default_params={}),
            _probe,
        )
    p = _simple_f32()
    w0 = np.array([5.0])
    out = simulate_lif_edge(
        p, n_steps=3, dt_ms=0.5,
        edge_pre=np.array([0]), edge_post=np.array([0]),
        edge_weight=w0, edge_delay_steps=np.array([1]),
        hdp_rule=name,
    )
    assert len(seen) == 1, seen
    n, dt, pre_sh, post_sh, spk_sh, has_pre_sp = seen[0]
    assert (n, pre_sh, post_sh, spk_sh, has_pre_sp) == (1, (1,), (1,), (1,), True)
    assert abs(dt - 0.5) < 1e-12
    assert float(np.asarray(out["w_final"])[0]) == 5.0  # zero drive: weights static
    # dH = 0.4/ms integrated every step: H = 1 + 3 * 0.5 * 0.4 proves the rule
    # observed all three steps.
    assert abs(float(np.asarray(out["H_final"])[0]) - 1.6) < 1e-6


def test_kernel_refusals():
    p = _simple_f32()
    bad = [
        dict(n_steps=0, dt_ms=0.5),
        dict(n_steps=3, dt_ms=0.0),
        dict(n_steps=3, dt_ms=-0.5),
    ]
    for kw in bad:
        try:
            simulate_lif_edge(p, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"must refuse {kw}")
    try:
        simulate_lif_edge(p, n_steps=2, dt_ms=0.5, edge_pre=np.array([0, 0]),
                          edge_post=np.array([0]), edge_weight=np.array([1.0]),
                          edge_delay_steps=np.array([1]))
    except ValueError:
        pass
    else:
        raise AssertionError("ragged edge arrays must raise")
    try:
        simulate_lif_edge(p, n_steps=2, dt_ms=0.5, edge_pre=np.array([0]),
                          edge_post=np.array([0]), edge_weight=np.array([1.0]),
                          edge_delay_steps=np.array([0]))
    except ValueError:
        pass
    else:
        raise AssertionError("delay < 1 must raise")
    try:
        simulate_lif_edge(p, n_steps=2, dt_ms=0.5, hdp_rule="no_such_rule")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown hdp_rule must raise")
    try:
        simulate_lif_edge(p, n_steps=2, dt_ms=0.5, hdp_rule_params={"eta": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("rule params without a rule must raise")


def test_grouped_cap_resolves_winners_per_position():
    """Fail-first for Gap 2: the same-step cap competes per position group.

    Four cells at two retinotopic positions, all above threshold: rounds=1
    keeps the first crosser of each group in the declared visit order
    (oracle SpikingNetwork.cpp:671-681 ascending depth visit).
    """
    p = _simple_f32()
    state = (jnp.zeros(4, jnp.float32), jnp.zeros(4, jnp.float32),
             jnp.full(4, -1.0e30, jnp.float32))
    drive = jnp.full(4, 60.0, jnp.float32)
    no_arr = jnp.zeros(4, jnp.float32)
    groups = jnp.array([0, 0, 1, 1])
    _, asc = lif_step(state, drive, no_arr, 0.5, p, 0.5,
                      cap_rounds=1, cap_groups=groups)
    assert [bool(s) for s in asc] == [True, False, True, False]
    _, desc = lif_step(state, drive, no_arr, 0.5, p, 0.5,
                       cap_rounds=1, cap_order="descending",
                       cap_groups=groups)
    assert [bool(s) for s in desc] == [False, True, False, True]
    _, two = lif_step(state, drive, no_arr, 0.5, p, 0.5,
                      cap_rounds=2, cap_groups=jnp.array([0, 0, 0, 1]))
    assert [bool(s) for s in two] == [True, True, False, True]
    # No groups: the whole-population cap (one winner total), unchanged.
    _, whole = lif_step(state, drive, no_arr, 0.5, p, 0.5, cap_rounds=1)
    assert [bool(s) for s in whole] == [True, False, False, False]


def test_feed_current_and_arrivals_carry_oracle_refractory_term():
    """Fail-first for Gap 3 feed: the DVS cover-table feed is part of x.

    ``feed_current``/``feed_arrivals`` are the per-step summed weights and
    event counts of the binned DVS feed. Each fed event subtracts one R + A,
    exactly like an edge arrival (oracle ``excite_simple``,
    ``pcl_event_ref.py:226-238``); the sustained ``I_ext`` channel stays
    bare current (no arrivals), so Gap-1 drive numerics are untouched.
    """
    p = _simple_f32()
    # Step 0 forces a spike (V -> reset, t_last stamped); step 1 probes
    # the feed with the refractory transient active.
    drive = jnp.array([[30.0], [0.0]], jnp.float32)
    feed_i = jnp.array([[0.0], [6.0]], jnp.float32)
    feed_n = jnp.array([[0.0], [1.0]], jnp.float32)
    fed = simulate_lif_edge(
        p, n_steps=2, dt_ms=0.5, I_ext_pA=drive,
        feed_current=feed_i, feed_arrivals=feed_n, record_v=True,
    )
    assert [bool(s) for s in fed["spikes"][:, 0]] == [True, False]
    r_half_ms = 10.0 * math.exp(-0.5 / 5.0)
    decayed_reset = -10.0 * math.exp(-0.5 / 18.0)
    assert abs(float(fed["V_mV"][1, 0]) - (decayed_reset + 6.0 - r_half_ms)) < 1e-4
    # Same current with no arrivals: bare current, no R + A subtraction.
    bare = simulate_lif_edge(
        p, n_steps=2, dt_ms=0.5, I_ext_pA=drive,
        feed_current=feed_i, feed_arrivals=jnp.zeros_like(feed_n),
        record_v=True,
    )
    assert abs(float(bare["V_mV"][1, 0]) - (decayed_reset + 6.0)) < 1e-6
    # And the historical I_ext-only path matches the zero-arrival feed path.
    hist = simulate_lif_edge(p, n_steps=2, dt_ms=0.5, I_ext_pA=drive + feed_i)
    assert bool((hist["spikes"] == bare["spikes"]).all())
    assert abs(float(hist["V"][0]) - float(bare["V"][0])) < 1e-6


def test_event_order_cascade_matches_oracle_per_event_winners():
    """Fail-first for the event-order intra-step fixed point.

    Two same-bin events at one position, each crossing on its own: the
    earlier event feeds cell 1, the later feeds cell 0, and mutual local
    inhibition (8 mV) suppresses the later candidate. The oracle resolves
    per event in ascending index (``pcl_event_ref.py:663-697``), so cell 1
    spikes and inhibits cell 0 before its event resolves; the whole-step
    cap instead keeps the ascending cell. The cascade must match the
    oracle's winner, not the cap's.
    """
    from jaxfne.pcl_topology import local_inhibition_blocks

    p = _as_f32(make_lif_params(("simple", "simple")))
    groups = np.array([0, 0], dtype=np.int64)
    blocks = local_inhibition_blocks(
        groups,
        li_pre=np.array([1, 0]), li_post=np.array([0, 1]),
        li_weight=np.array([8.0, 8.0]),
    )
    feed_i = jnp.array([[12.0, 12.0]], jnp.float32)
    feed_n = jnp.array([[1.0, 1.0]], jnp.float32)
    # Cell 1's event is earlier (100 us), cell 0's later (300 us).
    visit = np.zeros((1, 1, 2), dtype=np.int32)
    visit[0, 0, :] = [1, 0]
    out = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        feed_current=feed_i, feed_arrivals=feed_n,
        local_inhibition_rounds=1,
        local_inhibition_groups=jnp.asarray(groups),
        local_inhibition_resolution="event_order",
        local_inhibition_blocks=blocks,
        local_inhibition_visit_order=jnp.asarray(visit),
    )
    assert [bool(s) for s in out["spikes"][0]] == [False, True]
    assert abs(float(out["V"][0]) - 4.0) < 1e-4
    assert abs(float(out["V"][1]) - (-10.0)) < 1e-6
    # The whole-step cap on the same input keeps the ascending cell.
    capped = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        feed_current=feed_i, feed_arrivals=feed_n,
        local_inhibition_rounds=1,
        local_inhibition_groups=jnp.asarray(groups),
    )
    assert [bool(s) for s in capped["spikes"][0]] == [True, False]
    # The oracle primitives in event order pick the same winner: cell 1's
    # event resolves first and its inhibition holds cell 0 down.
    pytest.importorskip("h5py", reason="the PCL oracle reads HDF5 event files")
    from artifacts.etudes.papers.pcl.port import pcl_event_ref as oracle

    ref = oracle.PCLEventRef(directory=None, learning=False)
    c0, c1 = oracle.Cell(thr=10.0), oracle.Cell(thr=10.0)
    assert oracle.excite_simple(c1, ref.scfg, 100, 12.0) is True
    oracle.inhibit_simple(c0, ref.scfg, 100, 8.0)
    assert oracle.excite_simple(c0, ref.scfg, 300, 12.0) is False


def test_event_order_refusals_name_the_channel():
    """Every event-order parameter changes the realized object or is refused."""
    from jaxfne.pcl_topology import local_inhibition_blocks

    p = _as_f32(make_lif_params(("simple", "simple")))
    groups = jnp.asarray(np.array([0, 0]))
    blocks = local_inhibition_blocks(
        np.array([0, 0]),
        li_pre=np.array([1]), li_post=np.array([0]),
        li_weight=np.array([8.0]),
    )
    visit = jnp.asarray(np.array([[[1, 0]]], dtype=np.int32))
    good = dict(
        n_steps=1, dt_ms=0.5,
        feed_current=jnp.zeros((1, 2), jnp.float32),
        feed_arrivals=jnp.zeros((1, 2), jnp.float32),
        local_inhibition_rounds=1, local_inhibition_groups=groups,
        local_inhibition_resolution="event_order",
        local_inhibition_blocks=blocks,
        local_inhibition_visit_order=visit,
    )
    simulate_lif_edge(p, **good)  # must not raise
    bads = [
        {"local_inhibition_resolution": "per_event"},
        {"local_inhibition_resolution": "whole_step",
         "local_inhibition_blocks": blocks},
        {"local_inhibition_resolution": "whole_step",
         "local_inhibition_visit_order": visit},
        {"local_inhibition_resolution": "event_order",
         "local_inhibition_rounds": None},
        {"local_inhibition_resolution": "event_order",
         "local_inhibition_groups": None},
        {"local_inhibition_resolution": "event_order",
         "local_inhibition_blocks": None},
        {"local_inhibition_resolution": "event_order",
         "local_inhibition_visit_order": None},
        {"local_inhibition_resolution": "event_order",
         "local_inhibition_order": "descending"},
        {"local_inhibition_resolution": "event_order",
         "local_inhibition_visit_order": jnp.zeros((1, 1, 2), np.int32)},
    ]
    for kw in bads:
        cfg = dict(good)
        cfg.update(kw)
        try:
            simulate_lif_edge(p, **cfg)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"{sorted(kw)} must raise")
    neg = dict(blocks)
    neg["weights"] = -np.asarray(blocks["weights"])
    cfg = dict(good, local_inhibition_blocks=neg)
    try:
        simulate_lif_edge(p, **cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("negative block magnitudes must raise")
    # Explicit whole-step spells the default: identical spikes.
    a = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        feed_current=jnp.full((1, 2), 12.0, jnp.float32),
        feed_arrivals=jnp.ones((1, 2), jnp.float32),
        local_inhibition_rounds=1, local_inhibition_groups=groups)
    b = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        feed_current=jnp.full((1, 2), 12.0, jnp.float32),
        feed_arrivals=jnp.ones((1, 2), jnp.float32),
        local_inhibition_rounds=1, local_inhibition_groups=groups,
        local_inhibition_resolution="whole_step")
    assert bool((np.asarray(a["spikes"]) == np.asarray(b["spikes"])).all())


def test_event_scan_matches_oracle_event_for_event():
    """Fail-first for the event-scan intra-step fixed point.

    Three same-bin events at one two-cell position: e1 (100 us, polarity 0)
    feeds cell 1 over threshold, e2 (200 us, polarity 1) feeds cell 0 while
    still suppressed, e3 (300 us, polarity 1) feeds cell 0 over threshold.
    Mutual local inhibition is 8 mV. The oracle resolves per event in
    ascending index (``pcl_event_ref.py:663-697``): cell 1 spikes at 100 us
    and holds cell 0 down through e2, then cell 0 spikes at 300 us. The
    event-order rounds cascade instead keeps only the ascending winner, so
    the scan must match the oracle's two spikes, not the cascade's one.
    """
    from jaxfne.pcl_topology import (
        event_scan_events,
        event_scan_tables,
        local_inhibition_blocks,
    )

    w_s = np.zeros((2, 2, 1, 1), dtype=np.float64)
    w_s[1, 0, 0, 0] = 12.0
    w_s[0, 1, 0, 0] = 12.0
    tables = event_scan_tables(
        nx=1, ny=1, nd=2, cx=1, cy=1, cd=1,
        origins_x=[2.0], origins_y=[2.0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0], c_size=(1, 1),
        c_weights=np.zeros((1, 1, 1, 2), dtype=np.float64),
        vw=4, vh=4, inhib_range=(0, 0),
    )
    events = event_scan_events(
        np.array([100, 200, 300]), np.array([2, 2, 2]),
        np.array([2, 2, 2]), np.array([0, 1, 1]),
        dt_ms=0.5, n_steps=1, vw=4, vh=4,
    )
    p = _as_f32(make_lif_params(("simple", "simple", "complex")))
    groups = np.array([0, 0, 1], dtype=np.int64)
    blocks = local_inhibition_blocks(
        groups,
        li_pre=np.array([1, 0]), li_post=np.array([0, 1]),
        li_weight=np.array([8.0, 8.0]),
    )
    out = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        local_inhibition_groups=jnp.asarray(groups),
        local_inhibition_resolution="event_scan",
        local_inhibition_blocks=blocks,
        event_scan_events=events,
        event_scan_tables=tables,
    )
    assert [bool(s) for s in out["spikes"][0]] == [True, True, False]
    assert [int(round(float(t) * 1000)) for t in out["t_last"][:2]] == [300, 100]
    # The oracle primitives in event order pick the same spikes: cell 1 at
    # 100 us (holding cell 0 down), then cell 0 at 300 us.
    pytest.importorskip("h5py", reason="the PCL oracle reads HDF5 event files")
    from artifacts.etudes.papers.pcl.port import pcl_event_ref as oracle

    ref = oracle.PCLEventRef(directory=None, learning=False)
    c0, c1 = oracle.Cell(thr=10.0), oracle.Cell(thr=10.0)
    assert oracle.excite_simple(c0, ref.scfg, 100, 0.0) is False
    assert oracle.excite_simple(c1, ref.scfg, 100, 12.0) is True
    oracle.inhibit_simple(c0, ref.scfg, 100, 8.0)
    assert oracle.excite_simple(c0, ref.scfg, 200, 12.0) is False
    assert oracle.excite_simple(c1, ref.scfg, 200, 0.0) is False
    assert oracle.excite_simple(c0, ref.scfg, 300, 12.0) is True
    oracle.inhibit_simple(c1, ref.scfg, 300, 8.0)
    assert oracle.excite_simple(c1, ref.scfg, 300, 0.0) is False
    assert [c0.spikes, c1.spikes] == [[300], [100]]
    for got, want in zip(
        [float(out["V"][0]), float(out["V"][1])], [c0.V, c1.V]
    ):
        assert got == pytest.approx(want, rel=1e-9)
    # The event-order cascade on the same binned input keeps only the
    # ascending winner: one spike, not the oracle's two.
    visit = np.zeros((1, 2, 2), dtype=np.int32)
    visit[0, 0, :] = [0, 1]
    visit[0, 1, :] = [0, -1]
    capped = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        feed_current=jnp.array([[24.0, 12.0, 0.0]], jnp.float32),
        feed_arrivals=jnp.array([[3.0, 3.0, 0.0]], jnp.float32),
        local_inhibition_rounds=1,
        local_inhibition_groups=jnp.asarray(groups),
        local_inhibition_resolution="event_order",
        local_inhibition_blocks=blocks,
        local_inhibition_visit_order=jnp.asarray(visit),
    )
    assert [bool(s) for s in capped["spikes"][0]] == [True, False, False]


def test_event_scan_refusals_name_the_channel():
    """Every event-scan parameter changes the realized object or is refused."""
    from jaxfne.pcl_topology import (
        event_scan_events,
        event_scan_tables,
        local_inhibition_blocks,
    )

    w_s = np.zeros((2, 2, 1, 1), dtype=np.float64)
    w_s[1, 0, 0, 0] = 12.0
    w_s[0, 1, 0, 0] = 12.0
    tables = event_scan_tables(
        nx=1, ny=1, nd=2, cx=1, cy=1, cd=1,
        origins_x=[2.0], origins_y=[2.0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0], c_size=(1, 1),
        c_weights=np.zeros((1, 1, 1, 2), dtype=np.float64),
        vw=4, vh=4, inhib_range=(0, 0),
    )
    events = event_scan_events(
        np.array([100, 200]), np.array([2, 2]),
        np.array([2, 2]), np.array([0, 1]),
        dt_ms=0.5, n_steps=1, vw=4, vh=4,
    )
    p = _as_f32(make_lif_params(("simple", "simple", "complex")))
    groups = jnp.asarray(np.array([0, 0, 1]))
    blocks = local_inhibition_blocks(
        np.array([0, 0, 1]),
        li_pre=np.array([1]), li_post=np.array([0]),
        li_weight=np.array([8.0]),
    )
    good = dict(
        n_steps=1, dt_ms=0.5,
        local_inhibition_groups=groups,
        local_inhibition_resolution="event_scan",
        local_inhibition_blocks=blocks,
        event_scan_events=events,
        event_scan_tables=tables,
    )
    simulate_lif_edge(p, **good)  # must not raise
    bads = [
        {"local_inhibition_resolution": "per_scan"},
        {"local_inhibition_resolution": "event_scan",
         "local_inhibition_rounds": 1},
        {"local_inhibition_resolution": "event_scan",
         "local_inhibition_order": "descending"},
        {"local_inhibition_resolution": "event_scan",
         "local_inhibition_visit_order": jnp.zeros((1, 1, 2), np.int32)},
        {"local_inhibition_resolution": "event_scan",
         "local_inhibition_groups": None},
        {"local_inhibition_resolution": "event_scan",
         "local_inhibition_blocks": None},
        {"local_inhibition_resolution": "event_scan",
         "event_scan_events": None},
        {"local_inhibition_resolution": "event_scan",
         "event_scan_tables": None},
        {"local_inhibition_resolution": "event_scan",
         "feed_current": jnp.zeros((1, 3), jnp.float32)},
        {"local_inhibition_resolution": "event_scan",
         "feed_arrivals": jnp.zeros((1, 3), jnp.float32)},
        {"local_inhibition_resolution": "event_scan",
         "edge_pre": np.zeros((1,), np.int64),
         "edge_post": np.zeros((1,), np.int64),
         "edge_weight": np.zeros((1,), np.float64),
         "edge_delay_steps": np.ones((1,), np.int64)},
        {"local_inhibition_resolution": "event_scan",
         "I_ext_pA": jnp.zeros((3,), jnp.float32)},
        {"local_inhibition_resolution": "event_scan", "record_v": True},
        {"local_inhibition_resolution": "event_scan",
         "event_scan_events": events[::-1]},
        {"local_inhibition_resolution": "whole_step",
         "event_scan_events": events},
        {"local_inhibition_resolution": "whole_step",
         "event_scan_tables": tables},
    ]
    for kw in bads:
        cfg = dict(good)
        cfg.update(kw)
        try:
            simulate_lif_edge(p, **cfg)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"{sorted(kw)} must raise")


def test_feed_channel_refusals_name_the_channel():
    """Every feed parameter changes the realized object or is refused by name."""
    p = _simple_f32()
    for kw in (
        {"feed_current": jnp.zeros((3, 2), jnp.float32)},
        {"feed_arrivals": jnp.zeros((2, 3), jnp.float32)},
        {"feed_current": jnp.full((2, 1), float("nan"), jnp.float32)},
        {"feed_arrivals": jnp.full((2, 1), -1.0, jnp.float32)},
        {"feed_arrivals": jnp.full((2, 1), 0.5, jnp.float32)},
    ):
        try:
            simulate_lif_edge(p, n_steps=2, dt_ms=0.5, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{sorted(kw)} must raise")
