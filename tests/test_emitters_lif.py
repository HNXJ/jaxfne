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
    "tau_rp_ms", "tau_sra_ms", "delta_sra_mV", "tau_syn_ms",
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


def test_tau_syn_default_zero_means_unfiltered():
    """Absent tau_syn_ms defaults to 0 (the delta-summing path, unchanged)."""
    assert float(LIF_CELL_TYPE_DEFAULTS["simple"]["tau_syn_ms"]) == 0.0
    assert float(LIF_CELL_TYPE_DEFAULTS["complex"]["tau_syn_ms"]) == 0.0
    p = _simple_f32()
    assert bool((np.asarray(p.tau_syn_ms) == 0.0).all())


def _tau_drive_3x2():
    I_ext = np.zeros((3, 2), np.float32)
    I_ext[0, 0] = 30.0
    I_ext[1, 0] = 30.0
    return I_ext


def _tau_edge_kw(weight):
    return dict(
        n_steps=3, dt_ms=0.5,
        edge_pre=np.array([0]), edge_post=np.array([1]),
        edge_weight=np.array([weight]), edge_delay_steps=np.array([1]),
        I_ext_pA=jnp.asarray(_tau_drive_3x2()),
    )


def _simple_f32_two():
    return _as_f32(make_lif_params(("simple", "simple")))


def test_tau_syn_filter_decay_form():
    """G-TAUSYN decay form: a delta arrival sums fully, then decays as exp(-dt/tau_syn).

    Weight-4 arrivals at steps 1, 2 (cell 0 spikes steps 0, 1; delay 1):
    g_1 = 4.0, g_2 = 4.0*exp(-1) + 4.0, V_2 = V_1*exp(-dt/tau_m) + g_2.
    """
    p = _as_f32(make_lif_params(("simple", "simple"),
                                {"simple": {"tau_syn_ms": 0.5}}))
    out = simulate_lif_edge(p, **_tau_edge_kw(4.0), record_v=True)
    spk = np.asarray(out["spikes"])
    assert spk[:, 0].tolist() == [True, True, False]
    assert not bool(spk[:, 1].any())
    assert abs(float(out["V_mV"][1, 1]) - 4.0) < 1e-6
    want = (4.0 * math.exp(-0.5 / 18.0) + 4.0 * math.exp(-1.0) + 4.0)
    assert abs(float(out["V_mV"][2, 1]) - want) < 1e-5, (float(out["V_mV"][2, 1]), want)


def test_tau_syn_changes_spikes_at_fixed_seed():
    """The filter is consumed by the kernel: same seed, spikes differ."""
    kw = _tau_edge_kw(5.0)
    base = simulate_lif_edge(_simple_f32_two(), **kw)
    filt = simulate_lif_edge(
        _as_f32(make_lif_params(("simple", "simple"),
                                {"simple": {"tau_syn_ms": 0.5}})), **kw)
    assert not bool(np.asarray(base["spikes"])[:, 1].any()), "delta path stays silent"
    assert bool(np.asarray(filt["spikes"])[2, 1]), "filtered path spikes step 2"
    assert bool((np.asarray(base["spikes"]) != np.asarray(filt["spikes"])).any())


def test_tau_syn_refusals_name_the_field():
    """Declared tau_syn_ms must be a finite number > 0 in ms (fail-closed)."""
    for bad in (0.0, -1.0, float("nan"), float("inf"), "0.5", True, None):
        try:
            make_lif_params(("simple",), {"simple": {"tau_syn_ms": bad}})
        except ValueError as exc:
            assert "finite number > 0" in str(exc), str(exc)
        else:
            raise AssertionError(f"tau_syn_ms={bad!r} must raise")


def test_tau_syn_tensor_override_constructs_and_refuses():
    """Tensor-route plumbing carries tau_syn_ms per triple, fail-closed."""
    from jaxfne.emitters_lif import lif_network_params_from_labels

    p = lif_network_params_from_labels(
        ("E",), lif_overrides={"E": {"tau_syn_ms": 0.5}})
    assert abs(float(np.asarray(p.tau_syn_ms)[0]) - 0.5) < 1e-12
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        try:
            lif_network_params_from_labels(
                ("E",), lif_overrides={"E": {"tau_syn_ms": bad}})
        except ValueError as exc:
            assert "finite number > 0" in str(exc), str(exc)
        else:
            raise AssertionError(f"tau_syn_ms={bad!r} must raise")


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


def _spike_chain_fixture():
    """One simple cell driven by four same-pixel events, each forwarded to one complex cell.

    Simple weight 40 mV is far above the 10 mV simple threshold, so every event
    spikes the simple cell (reset to -10 mV; refractory 10 mV at tau_rp 5 ms
    moves the next crossing by well under a millivolt). Forward weight 40 mV is
    far above the 3 mV complex threshold, so each simple spike spikes the
    complex cell at the same event time.
    """
    from jaxfne.pcl_topology import (
        event_scan_events,
        event_scan_tables,
        local_inhibition_blocks,
    )

    w_s = np.zeros((1, 2, 1, 1), dtype=np.float64)
    w_s[0, 0, 0, 0] = 40.0
    tables = event_scan_tables(
        nx=1, ny=1, nd=1, cx=1, cy=1, cd=1,
        origins_x=[0], origins_y=[0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0], c_size=(1, 1),
        c_weights=np.full((1, 1, 1, 1), 40.0, dtype=np.float64),
        vw=1, vh=1, inhib_range=(0, 0),
    )
    zeros = np.zeros(4, dtype=np.int64)
    events = event_scan_events(
        np.array([100, 200, 300, 400]), zeros, zeros, zeros,
        dt_ms=0.5, n_steps=1, vw=1, vh=1,
    )
    p = _as_f32(make_lif_params(("simple", "complex")))
    groups = np.array([0, 1], dtype=np.int64)
    blocks = local_inhibition_blocks(
        groups,
        li_pre=np.zeros(0, dtype=np.int64), li_post=np.zeros(0, dtype=np.int64),
        li_weight=np.zeros(0, dtype=np.float64),
    )
    kw = dict(
        n_steps=1, dt_ms=0.5,
        local_inhibition_groups=jnp.asarray(groups),
        local_inhibition_resolution="event_scan",
        local_inhibition_blocks=blocks,
        event_scan_events=events,
        event_scan_tables=tables,
    )
    return p, kw


def test_event_scan_logs_every_spike_in_emission_order():
    """The event scan returns every spike's time and cell, in emission order.

    By hand: events at 100, 200, 300 and 400 us each spike the simple cell
    (cell 0), and each simple spike forwards to the complex cell (cell 1) at
    the same time. Emission order is simple then the complex it forwards.
    """
    p, kw = _spike_chain_fixture()
    out = simulate_lif_edge(p, **kw)
    counts = np.asarray(out["counts"])
    first = np.asarray(out["first_spike_us"])
    log_t = np.asarray(out["spike_times_us"])
    log_c = np.asarray(out["spike_cells"])
    assert counts.max() >= 2, "fixture must spike the simple cell twice"
    assert counts[1] >= 1, "fixture must forward a spike to the complex cell"
    assert log_t.shape == log_c.shape == (int(counts.sum()),)
    assert [int(t) for t in log_t] == [100, 100, 200, 200, 300, 300, 400, 400]
    assert [int(c) for c in log_c] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert bool(np.all(np.diff(log_t) >= 0))
    for cell in range(counts.shape[0]):
        mine = log_t[log_c == cell]
        assert mine.shape[0] == int(counts[cell])
        assert int(mine[0]) == int(first[cell])


def test_event_scan_spike_log_reruns_past_capacity(monkeypatch):
    """A log that overflows its first capacity is rerun at the logged size, unchanged."""
    import jaxfne.emitters_lif as lif

    p, kw = _spike_chain_fixture()
    base = simulate_lif_edge(p, **kw)
    assert int(np.asarray(base["counts"]).sum()) >= 3, "fixture must hold at least 3 spikes"
    calls = []

    def _tiny_capacity(n_events):
        calls.append(n_events)
        return 1

    monkeypatch.setattr(lif, "_spike_log_capacity", _tiny_capacity)
    rerun = simulate_lif_edge(p, **kw)
    assert calls == [4], "the patched capacity must drive the first pass (4 events)"
    for key in ("spike_times_us", "spike_cells", "counts", "first_spike_us"):
        assert np.array_equal(np.asarray(rerun[key]), np.asarray(base[key])), key


def _interval_fixture():
    """One simple cell (12 mV per event, 10 mV threshold) and a silent complex cell, one pixel.

    Events at 500, 1500 and 60000 us. eta_rp is zero so the refractory term
    never moves a crossing; delta_sra is zero so no adaptation term either.
    """
    from jaxfne.pcl_topology import (
        event_scan_events,
        event_scan_tables,
        local_inhibition_blocks,
    )

    w_s = np.zeros((1, 2, 1, 1), dtype=np.float64)
    w_s[0, 0, 0, 0] = 12.0
    tables = event_scan_tables(
        nx=1, ny=1, nd=1, cx=1, cy=1, cd=1,
        origins_x=[0], origins_y=[0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0], c_size=(1, 1),
        c_weights=np.zeros((1, 1, 1, 1), dtype=np.float64),
        vw=1, vh=1, inhib_range=(0, 0),
    )
    zeros = np.zeros(3, dtype=np.int64)
    events = event_scan_events(
        np.array([500, 1500, 60000]), zeros, zeros, zeros,
        dt_ms=0.5, n_steps=130, vw=1, vh=1,
    )
    p = _as_f32(make_lif_params(("simple", "complex")))
    p = p._replace(eta_rp_mV=jnp.zeros((2,), jnp.float32))
    groups = np.array([0, 1], dtype=np.int64)
    blocks = local_inhibition_blocks(
        groups,
        li_pre=np.zeros(0, dtype=np.int64), li_post=np.zeros(0, dtype=np.int64),
        li_weight=np.zeros(0, dtype=np.float64),
    )
    kw = dict(
        n_steps=130, dt_ms=0.5,
        local_inhibition_groups=jnp.asarray(groups),
        local_inhibition_resolution="event_scan",
        local_inhibition_blocks=blocks,
        event_scan_events=events,
        event_scan_tables=tables,
    )
    return p, kw


def _interval_h():
    """The fixture's H update: 1 ms interval, alpha 0.6, 4 mV steps toward 0.75 Hz."""
    return dict(
        measurement_interval_us=1000.0, rate_alpha=0.6,
        target_rate_hz=np.array([0.75, 0.0]),
        threshold_eta_mV=np.array([4.0, 0.0]),
        threshold_min_mV=np.array([4.0, 0.0]),
        last_update_us=0,
    )


def _h_update_bad_cases():
    """Each bad H update, paired with the key its refusal must name."""
    base = _interval_h()
    return [
        ("rate_gain", dict(base, rate_gain=1.0)),
        ("rate_alpha", {k: v for k, v in base.items() if k != "rate_alpha"}),
        ("measurement_interval_us", dict(base, measurement_interval_us=0.0)),
        ("rate_alpha", dict(base, rate_alpha=0.0)),
        ("rate_alpha", dict(base, rate_alpha=1.5)),
        ("rate_alpha", dict(base, rate_alpha=float("nan"))),
        ("last_update_us", dict(base, last_update_us=1.5)),
        ("last_update_us", dict(base, last_update_us=True)),
        ("target_rate_hz", dict(base, target_rate_hz=np.zeros(3))),
        ("threshold_eta_mV", dict(base, threshold_eta_mV=np.array([np.nan, 0.0]))),
        ("threshold_min_mV", dict(base, threshold_min_mV=np.zeros((1, 2)))),
        ("rate0_hz", dict(base, rate0_hz=np.zeros(3))),
        ("threshold0_mV", dict(base, threshold0_mV=np.array([np.inf, 3.0]))),
        ("h_update", [0.0, 1.0]),
    ]


def test_event_scan_interval_update_moves_threshold():
    """The periodic update raises a threshold, and a later crossing of the old one stops spiking.

    Plain-Python rule (pcl_event_ref.py:746-762): the 500 us spike is the one
    spike of the first interval. The 1500 us event is the first past the
    1000 us interval from 0, so the update runs after its cascade: the rate
    0.6 * 1e6 / 1500 Hz passes the 0.75 Hz target and the 10 mV threshold rises
    by 4 mV. The 60000 us event then carries 12 mV onto the decayed membrane,
    crossing the old 10 mV but not the raised threshold; its own update sees
    no spike and lowers the rate toward zero, still above target.
    """
    p, kw = _interval_fixture()
    alpha, eta, tgt, tau_m_us = 0.6, 4.0, 0.75, 18000.0
    out = simulate_lif_edge(p, event_scan_h_update=_interval_h(), **kw)

    v2 = -10.0 * math.exp(-1000.0 / tau_m_us) + 12.0
    rate2 = alpha * (1e6 / 1500.0) + (1.0 - alpha) * 0.0
    th2 = 10.0 + eta * (1.0 - math.exp(tgt - rate2))
    v3_pre = v2 * math.exp(-58500.0 / tau_m_us)
    rate3 = alpha * 0.0 + (1.0 - alpha) * rate2
    th3 = th2 + eta * (1.0 - math.exp(tgt - rate3))
    assert v2 < 10.0, "fixture: 1500 us must not spike"
    assert rate2 > tgt, "fixture: the first update must pass the target"
    assert v3_pre + 12.0 > 10.0, "fixture: 60000 us must cross the old 10 mV"
    assert v3_pre + 12.0 < th2, "fixture: 60000 us must stay under the raised threshold"
    assert rate3 > tgt, "fixture: the second update stays above target"

    assert [int(t) for t in np.asarray(out["spike_times_us"])] == [500]
    assert np.asarray(out["counts"]).tolist() == [1, 0]
    assert float(out["threshold_mV"][0]) == pytest.approx(th3, rel=1e-12)
    assert float(out["threshold_mV"][1]) == 3.0
    assert float(out["rate_hz"][0]) == pytest.approx(rate3, rel=1e-12)
    assert np.asarray(out["interval_count"]).tolist() == [0, 0]
    assert int(out["last_update_us"]) == 60000


def test_event_scan_without_h_update_is_unchanged():
    """With h_update None the thresholds stay at the params' 10 mV and the spikes follow the fixed rule.

    By hand: 500 us spikes from rest; 1500 us leaves the membrane at 2.54 mV,
    under 10 mV; the 60000 us event reaches 12.10 mV, spikes and resets to
    -10 mV with no adaptation.
    """
    p, kw = _interval_fixture()
    out = simulate_lif_edge(p, **kw)

    v2 = -10.0 * math.exp(-1000.0 / 18000.0) + 12.0
    assert v2 < 10.0
    assert v2 * math.exp(-58500.0 / 18000.0) + 12.0 > 10.0
    assert [int(t) for t in np.asarray(out["spike_times_us"])] == [500, 60000]
    assert [int(c) for c in np.asarray(out["spike_cells"])] == [0, 0]
    assert np.asarray(out["counts"]).tolist() == [2, 0]
    assert np.asarray(out["first_spike_us"]).tolist() == [500, -1]
    spikes = np.asarray(out["spikes"])
    assert int(spikes.sum()) == 2 and bool(spikes[1, 0]) and bool(spikes[120, 0])
    assert float(out["V"][0]) == pytest.approx(-10.0, rel=1e-12)
    assert float(out["A"][0]) == 0.0
    assert np.asarray(out["threshold_mV"]).tolist() == [10.0, 3.0]
    assert float(out["rate_hz"][0]) == 0.0
    assert int(out["last_update_us"]) == 0


def test_event_scan_h_update_refusals_name_the_key():
    """Each bad H update is refused by name, and an H update off the event scan is refused too."""
    p, kw = _interval_fixture()
    for key, bad in _h_update_bad_cases():
        with pytest.raises(ValueError, match=key):
            simulate_lif_edge(p, event_scan_h_update=bad, **kw)
    whole_step = dict(kw, local_inhibition_resolution="whole_step")
    with pytest.raises(ValueError, match="event_scan_h_update"):
        simulate_lif_edge(p, event_scan_h_update=_interval_h(), **whole_step)


def test_event_scan_drive_h_update_refusals_name_the_drive():
    """EventScanDrive validates h_update itself, naming the drive and the key."""
    from jaxfne.pcl_topology import EventScanDrive

    _, kw = _interval_fixture()
    args = (kw["event_scan_events"], kw["event_scan_tables"],
            kw["local_inhibition_blocks"], np.array([0, 1]))
    drive = EventScanDrive(*args, dt_ms=0.5, n_steps=130, h_update=_interval_h())
    assert drive.to_dict()["h_update"] is True
    assert EventScanDrive(*args, dt_ms=0.5, n_steps=130).to_dict()["h_update"] is False
    for key, bad in _h_update_bad_cases():
        with pytest.raises(ValueError, match="EventScanDrive") as exc:
            EventScanDrive(*args, dt_ms=0.5, n_steps=130, h_update=bad)
        assert key in str(exc.value), key


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


def test_event_order_differs_from_event_scan_on_order_sensitive_drive():
    """Negative control: the resolution key names a realized policy, not a label.

    Two same-bin events at one two-cell position (cell 1 fed 12 mV at
    100 us, cell 0 fed 20 mV at 400 us, mutual 8 mV inhibition): the exact
    ``event_scan`` replay spikes both cells (one immediate subtraction
    each), while the per-step ``event_order`` cascade with one round keeps
    only the first-visited winner. Same physical drive, different spike
    outcomes on the default seed.
    """
    from jaxfne.pcl_topology import (
        event_scan_events,
        event_scan_tables,
        local_inhibition_blocks,
    )

    w_s = np.zeros((2, 2, 1, 1), dtype=np.float64)
    w_s[1, 0, 0, 0] = 12.0
    w_s[0, 1, 0, 0] = 20.0
    tables = event_scan_tables(
        nx=1, ny=1, nd=2, cx=1, cy=1, cd=1,
        origins_x=[2.0], origins_y=[2.0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0], c_size=(1, 1),
        c_weights=np.zeros((1, 1, 1, 2), dtype=np.float64),
        vw=4, vh=4, inhib_range=(0, 0),
    )
    events = event_scan_events(
        np.array([100, 400]), np.array([2, 2]),
        np.array([2, 2]), np.array([0, 1]),
        dt_ms=0.5, n_steps=1, vw=4, vh=4,
    )
    p = _as_f32(make_lif_params(("simple", "simple", "complex")))
    groups = np.array([0, 0, 1], dtype=np.int64)
    blocks = local_inhibition_blocks(
        groups,
        li_pre=np.array([1, 0]), li_post=np.array([0, 1]),
        li_weight=np.array([8.0, 8.0]),
    )
    scan = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        local_inhibition_groups=jnp.asarray(groups),
        local_inhibition_resolution="event_scan",
        local_inhibition_blocks=blocks,
        event_scan_events=events,
        event_scan_tables=tables,
    )
    assert [bool(s) for s in scan["spikes"][0]] == [True, True, False]
    # The same two events binned: cell 1 fed 12 mV, cell 0 fed 20 mV.
    visit = np.zeros((1, 2, 2), dtype=np.int32)
    visit[0, 0, :] = [1, 0]
    visit[0, 1, :] = [0, -1]
    order = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        feed_current=jnp.array([[12.0, 20.0, 0.0]], jnp.float32),
        feed_arrivals=jnp.array([[1.0, 1.0, 0.0]], jnp.float32),
        local_inhibition_rounds=1,
        local_inhibition_groups=jnp.asarray(groups),
        local_inhibition_resolution="event_order",
        local_inhibition_blocks=blocks,
        local_inhibition_visit_order=jnp.asarray(visit),
    )
    assert [bool(s) for s in order["spikes"][0]] == [False, True, False]
    assert (
        np.asarray(scan["spikes"]) != np.asarray(order["spikes"])
    ).any(), "order-sensitive drive must separate the policies"


def test_resolution_refusals_list_the_supported_policies():
    """Unknown resolutions are refused with the supported list, never mapped."""
    from jaxfne.pcl_topology import LOCAL_INHIBITION_RESOLUTIONS

    p = _simple_f32()
    for bad in ("per_event", "PER_EVENT", "event-scan", ""):
        try:
            simulate_lif_edge(
                p, n_steps=1, dt_ms=0.5,
                feed_current=jnp.zeros((1, 1), jnp.float32),
                feed_arrivals=jnp.zeros((1, 1), jnp.float32),
                local_inhibition_rounds=1,
                local_inhibition_resolution=bad,
            )
        except ValueError as exc:
            for name in LOCAL_INHIBITION_RESOLUTIONS:
                assert name in str(exc), f"{bad!r} refusal must list {name}"
        else:
            raise AssertionError(f"{bad!r} must raise")


def test_policies_are_deterministic_per_seed_without_continuation():
    """Pin what exists: each policy repeats bit-identically on a fixed seed.

    The LIF edge route exposes no continuation (state enters only via
    ``V0_mV``; adaptation ``A`` and ``t_last`` always start fresh), so a
    chunked resume is not constructible here — continuation equality
    lives on the construct/simulate path (``jaxfne._pipeline``). What the
    route does guarantee is exact repeat determinism under every
    supported policy, pinned here on one fixed two-step drive.
    """
    from jaxfne.pcl_topology import local_inhibition_blocks

    p = _as_f32(make_lif_params(("simple", "simple")))
    groups = np.array([0, 0], dtype=np.int64)
    blocks = local_inhibition_blocks(
        groups,
        li_pre=np.array([1, 0]), li_post=np.array([0, 1]),
        li_weight=np.array([8.0, 8.0]),
    )
    visit = jnp.asarray(np.array([[[1, 0]], [[0, 1]]], dtype=np.int32))
    feed_i = jnp.array([[12.0, 12.0], [12.0, 0.0]], jnp.float32)
    feed_n = jnp.array([[1.0, 1.0], [1.0, 0.0]], jnp.float32)
    cfgs = [
        dict(local_inhibition_rounds=1,
             local_inhibition_groups=jnp.asarray(groups)),
        dict(local_inhibition_rounds=1,
             local_inhibition_groups=jnp.asarray(groups),
             local_inhibition_resolution="whole_step"),
        dict(local_inhibition_rounds=1,
             local_inhibition_groups=jnp.asarray(groups),
             local_inhibition_resolution="event_order",
             local_inhibition_blocks=blocks,
             local_inhibition_visit_order=visit),
    ]
    for cfg in cfgs:
        first = simulate_lif_edge(
            p, n_steps=2, dt_ms=0.5, feed_current=feed_i,
            feed_arrivals=feed_n, record_v=True, **cfg)
        second = simulate_lif_edge(
            p, n_steps=2, dt_ms=0.5, feed_current=feed_i,
            feed_arrivals=feed_n, record_v=True, **cfg)
        assert bool((np.asarray(first["spikes"])
                     == np.asarray(second["spikes"])).all())
        assert bool((np.asarray(first["V_mV"])
                     == np.asarray(second["V_mV"])).all())
