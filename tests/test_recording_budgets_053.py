"""0.5.3 ENGINE item 2: H/W recording budgets — declared stride/subset.

Status of the 0.5.1 opt-in selective/downsampled recording API: it does
not exist in jaxfne/ (no record_stride/record_subset/selective symbols;
0.5.1 B6 excluded item 2c since recording stayed below threshold). The
only stride machinery predates 0.5.1 (run_continuation_strided /
memory_report stride, 23-REC-01). This battery therefore implements the
budgets fresh: H and W at declared stride/subset via
hdp_params["record_stride"/"record_h_subset"/"record_w_subset"].

Contract: full recording stays the default (stride 1, subsets None);
kept frame j equals the full-trace frame at that index exactly;
dynamics are untouched (decimation is post-scan); invalid declarations
fail closed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne

N = 8
STEPS = 12  # 6.0 ms at dt 0.5


def _model():
    cfg = jtfne.suite2_net1_config(seed=11, n=N, duration_ms=20.0, dt_ms=0.5)
    return jtfne.construct(cfg)


def _rt(**over):
    hp = {
        "K_HDP": 0.01,
        "K_ctrl": 0.15,
        "K_w_ctrl": 0.001,
        "tau_0_ms": 20.0,
        "alpha": 0.01,
        "barrier_c": 0.01,
        "barrier_d": 0.01,
        "noise_scale": 0.0,
    }
    hp.update(over)
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params=hp,
    )


def _run(model, runtime, seed=17, duration_ms=6.0):
    sig = jtfne.simulate(
        model,
        duration_ms=duration_ms,
        dt_ms=0.5,
        seed=seed,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
    )
    return sig, model.last_hdp_diagnostics()


def _registered_rt(**over):
    hp = {
        "hdp_rule": "synthetic_presyn_gain",
        "hdp_rule_params": {"k_h": 0.01, "k_w": 0.02, "gamma": 0.0},
        "noise_scale": 0.0,
    }
    hp.update(over)
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params=hp,
    )


# --- default path bit-identical -------------------------------------------


def test_default_path_bit_identical_absent_vs_explicit():
    """Full recording stays the default: absent keys == explicit defaults."""
    model = _model()
    s1, d1 = _run(model, _rt())
    s2, d2 = _run(model, _rt(record_stride=1, record_h_subset=None, record_w_subset=None))
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(s1.spikes, s2.spikes)
    assert jnp.array_equal(s1.sources, s2.sources)
    assert jnp.array_equal(d1["H_trace"], d2["H_trace"])
    assert jnp.array_equal(d1["w_trace"], d2["w_trace"])
    assert jnp.array_equal(d1["H_final"], d2["H_final"])
    assert jnp.array_equal(d1["w_final"], d2["w_final"])


def test_budgets_leave_dynamics_untouched():
    """Stride/subset change only what is recorded, never the trajectory."""
    model = _model()
    s_full, _ = _run(model, _rt())
    s_bud, d_bud = _run(
        model,
        _rt(record_stride=3, record_h_subset=[0, 2, 5], record_w_subset=[0, 3]),
    )
    assert jnp.array_equal(s_full.V_m, s_bud.V_m)
    assert jnp.array_equal(s_full.spikes, s_bud.spikes)
    assert jnp.array_equal(s_full.sources, s_bud.sources)


# --- stride ----------------------------------------------------------------


def test_stride_kept_frames_equal_full_frames():
    model = _model()
    _, d_full = _run(model, _rt())
    _, d_k3 = _run(model, _rt(record_stride=3))
    H_full = np.asarray(d_full["H_trace"])
    w_full = np.asarray(d_full["w_trace"])
    H_k = np.asarray(d_k3["H_trace"])
    w_k = np.asarray(d_k3["w_trace"])
    assert H_k.shape == (4, N)
    assert w_k.shape[0] == 4
    np.testing.assert_array_equal(H_k, H_full[::3])
    np.testing.assert_array_equal(w_k, w_full[::3])


def test_stride_covers_uneven_tail():
    model = _model()
    _, d = _run(model, _rt(record_stride=5), duration_ms=6.0)
    assert np.asarray(d["H_trace"]).shape == (3, N)  # steps 0, 5, 10


def test_stride_applies_to_weight_trace_off_h_only():
    model = _model()
    _, d = _run(model, _rt(record_stride=2, record_weight_trace=False))
    assert d["w_trace"] is None
    assert np.asarray(d["H_trace"]).shape == (6, N)


# --- subset ----------------------------------------------------------------


def test_h_subset_frames_equal_full_at_indices():
    model = _model()
    _, d_full = _run(model, _rt())
    _, d_sub = _run(model, _rt(record_h_subset=[0, 2, 5]))
    H_full = np.asarray(d_full["H_trace"])
    H_sub = np.asarray(d_sub["H_trace"])
    assert H_sub.shape == (STEPS, 3)
    np.testing.assert_array_equal(H_sub, H_full[:, [0, 2, 5]])


def test_w_subset_frames_equal_full_at_indices():
    model = _model()
    _, d_full = _run(model, _rt())
    _, d_sub = _run(model, _rt(record_w_subset=[0, 3]))
    w_full = np.asarray(d_full["w_trace"])
    w_sub = np.asarray(d_sub["w_trace"])
    assert w_sub.shape == (STEPS, 2)
    np.testing.assert_array_equal(w_sub, w_full[:, [0, 3]])


def test_stride_and_subset_compose():
    model = _model()
    _, d_full = _run(model, _rt())
    _, d = _run(
        model,
        _rt(record_stride=3, record_h_subset=[1, 4], record_w_subset=[2]),
    )
    np.testing.assert_array_equal(
        np.asarray(d["H_trace"]), np.asarray(d_full["H_trace"])[::3][:, [1, 4]]
    )
    np.testing.assert_array_equal(
        np.asarray(d["w_trace"]), np.asarray(d_full["w_trace"])[::3][:, [2]]
    )


# --- registered rule path ---------------------------------------------------


def test_registered_rule_stride_subset_exact():
    model = _model()
    _, d_full = _run(model, _registered_rt())
    assert d_full is not None
    _, d = _run(
        model,
        _registered_rt(record_stride=4, record_h_subset=[0, 7], record_w_subset=[1]),
    )
    H_full = np.asarray(d_full["H_trace"])
    w_full = np.asarray(d_full["w_trace"])
    np.testing.assert_array_equal(np.asarray(d["H_trace"]), H_full[::4][:, [0, 7]])
    np.testing.assert_array_equal(np.asarray(d["w_trace"]), w_full[::4][:, [1]])


def test_registered_rule_default_bit_identical():
    model = _model()
    s1, d1 = _run(model, _registered_rt())
    s2, d2 = _run(
        model,
        _registered_rt(record_stride=1, record_h_subset=None, record_w_subset=None),
    )
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(d1["H_trace"], d2["H_trace"])
    assert jnp.array_equal(d1["w_trace"], d2["w_trace"])


# --- continuation path -------------------------------------------------------


def test_continuation_stride_matches_plain_stride():
    """Chunked strided H/W == continuous strided H/W (same frames)."""
    runtime = _rt(record_stride=3)
    common = dict(
        dt_ms=0.5,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
    )
    # Continuous unstrided reference frames (24 steps).
    m_ref = _model()
    _, d_ref = _run(m_ref, _rt(), duration_ms=12.0)
    ref_H = np.asarray(d_ref["H_trace"])

    m_full = _model()
    full, _ = jtfne.simulate(m_full, duration_ms=12.0, return_state=True, **common)
    d_full = m_full.last_hdp_diagnostics()
    np.testing.assert_array_equal(np.asarray(d_full["H_trace"]), ref_H[::3])

    m_seg = _model()
    first, first_state = jtfne.simulate(m_seg, duration_ms=6.0, return_state=True, **common)
    d_seg1 = m_seg.last_hdp_diagnostics()
    second, _ = jtfne.simulate(
        m_seg,
        duration_ms=6.0,
        continuation=first_state,
        return_state=True,
        **{**common, "seed": 999},
    )
    d_seg2 = m_seg.last_hdp_diagnostics()
    # Dynamics never strided: chunked V concatenates to continuous V.
    assert jnp.array_equal(full.V_m, jnp.concatenate((first.V_m, second.V_m), axis=0))
    # Strided H frames concatenate across chunks to the continuous frames.
    seg_H = np.concatenate(
        (np.asarray(d_seg1["H_trace"]), np.asarray(d_seg2["H_trace"])),
        axis=0,
    )
    np.testing.assert_array_equal(seg_H, np.asarray(d_full["H_trace"]))
    assert seg_H.shape == (8, N)  # 24 steps, stride 3


def test_continuation_subset_applies_per_step():
    model = _model()
    runtime = _rt(record_h_subset=[0, 1], record_w_subset=[0])
    sig, _ = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    d = model.last_hdp_diagnostics()
    assert np.asarray(d["H_trace"]).shape == (STEPS, 2)
    assert np.asarray(d["w_trace"]).shape == (STEPS, 1)
    assert sig.V_m.shape == (STEPS, N)


# --- fail closed ---------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3"])
def test_bad_stride_refused(bad):
    with pytest.raises(ValueError, match="record_stride"):
        _run(_model(), _rt(record_stride=bad))


@pytest.mark.parametrize(
    "kw",
    [
        {"record_h_subset": [0, 999]},
        {"record_h_subset": [-1]},
        {"record_h_subset": []},
        {"record_h_subset": [0.5]},
        {"record_w_subset": [0, 10**9]},
    ],
)
def test_bad_subset_refused(kw):
    with pytest.raises(ValueError, match="record_"):
        _run(_model(), _rt(**kw))


def test_w_subset_without_weight_trace_refused():
    with pytest.raises(ValueError, match="record_w_subset"):
        _run(
            _model(),
            _rt(record_weight_trace=False, record_w_subset=[0]),
        )


# --- budget volume ----------------------------------------------------------


def test_recorded_volume_matches_declared_budget():
    """Returned recording volume scales with the declared budget."""
    model = _model()
    n_edges = int(model.params["edge_list"].n_edges)
    _, d_full = _run(model, _rt())
    _, d_bud = _run(
        model,
        _rt(record_stride=3, record_h_subset=[0, 1], record_w_subset=[0]),
    )
    full_bytes = np.asarray(d_full["H_trace"]).nbytes + np.asarray(d_full["w_trace"]).nbytes
    bud_bytes = np.asarray(d_bud["H_trace"]).nbytes + np.asarray(d_bud["w_trace"]).nbytes
    assert bud_bytes < full_bytes
    assert np.asarray(d_bud["H_trace"]).shape == (4, 2)
    assert np.asarray(d_bud["w_trace"]).shape == (4, 1)
    assert n_edges > 1
