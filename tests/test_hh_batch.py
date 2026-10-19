"""Tests for ``simulate_hh_batch`` in ``jaxfne/emitters_hh.py``.

Fail-first: written for the batched kernel; on the baseline tree the name
does not exist, so importing it fails there. CPU-only, small grids.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jaxfne.emitters_hh import (
    make_hh_params,
    reference_4comp_morphology,
    simulate_hh,
    simulate_hh_batch,
)

DT_MS = 0.1
REC = dict(
    receptor_e_rev=(0.0,),
    receptor_tau_r=(0.2,),
    receptor_tau_d=(2.0,),
    receptor_slope=(5.0,),
    receptor_vth=(-20.0,),
)


def _two_cell_shared(**over):
    kw = dict(
        edge_pre=[0],
        edge_post=[1],
        edge_receptor=[0],
        **REC,
    )
    kw.update(over)
    return kw


def test_batch_single_path_matches_serial():
    params = make_hh_params(["E", "E"])
    n_steps = 2000
    i_ext = np.zeros((3, 2), dtype=np.float64)
    i_ext[:, 0] = [8.0, 10.0, 12.0]
    weights = np.asarray([[2.0], [4.0], [6.0]], dtype=np.float64)
    v0 = np.full((3, 2), -70.0) + np.asarray([[0.0, 0.0], [-2.0, 1.0], [2.0, -1.0]])
    shared = _two_cell_shared()
    out = simulate_hh_batch(
        params, n_steps=n_steps, dt_ms=DT_MS,
        batch={"I_ext": i_ext, "edge_weight": weights, "V0_mV": v0}, **shared)
    assert np.asarray(out["spikes"]).shape == (3, n_steps, 2)
    assert np.asarray(out["V_mV"]).shape == (3, n_steps, 2)
    counts = np.asarray(out["spikes"]).sum(axis=(1, 2))
    assert all(int(c) > 0 for c in counts), \
        "every batch element must spike, got %r" % (counts,)
    for b in range(3):
        ref = simulate_hh(params, n_steps=n_steps, dt_ms=DT_MS,
                          I_ext=i_ext[b], edge_weight=weights[b],
                          V0_mV=v0[b], **shared)
        np.testing.assert_array_equal(np.asarray(out["spikes"][b]),
                                      np.asarray(ref["spikes"]))
        np.testing.assert_allclose(np.asarray(out["V_mV"][b]),
                                   np.asarray(ref["V_mV"]),
                                   rtol=1e-4, atol=1e-4)


def test_batch_noise_matches_serial():
    params = make_hh_params(["E", "E"])
    n_steps = 500
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    gains = np.asarray([1.0, 0.5, 2.0])
    shared = _two_cell_shared(edge_weight=[2.0],
                              I_ext=np.asarray([6.0, 0.0]),
                              ou_sigma=[5.0, 5.0], ou_tau=[20.0, 20.0])
    out = simulate_hh_batch(params, n_steps=n_steps, dt_ms=DT_MS,
                            batch={"ou_key": keys, "noise_gain": gains},
                            **shared)
    assert np.asarray(out["spikes"]).shape == (3, n_steps, 2)
    counts = np.asarray(out["spikes"]).sum(axis=(1, 2))
    assert all(int(c) > 0 for c in counts), \
        "every noise batch element must spike, got %r" % (counts,)
    for b in range(3):
        ref = simulate_hh(params, n_steps=n_steps, dt_ms=DT_MS,
                          ou_key=keys[b], noise_gain=float(gains[b]), **shared)
        np.testing.assert_array_equal(np.asarray(out["spikes"][b]),
                                      np.asarray(ref["spikes"]))
        np.testing.assert_allclose(np.asarray(out["V_mV"][b]),
                                   np.asarray(ref["V_mV"]),
                                   rtol=1e-4, atol=1e-4)


def test_batch_multi_path_matches_serial():
    params = make_hh_params(["E", "E"])
    morph = reference_4comp_morphology()
    n_steps, dt = 800, 0.025
    nA = np.zeros((3, 2), dtype=np.float64)
    nA[:, 0] = [0.02, 0.025, 0.03]
    uS = np.asarray([[0.5], [1.0], [1.5]], dtype=np.float64)
    shared = dict(edge_pre=[0], edge_post=[1], edge_receptor=[0],
                  morphology=morph, stim_comp=2, record_comp=0, **REC)
    out = simulate_hh_batch(params, n_steps=n_steps, dt_ms=dt,
                            batch={"I_ext_nA": nA, "edge_weight_uS": uS},
                            **shared)
    assert np.asarray(out["spikes"]).shape == (3, n_steps, 2)
    assert np.asarray(out["V_mV"]).shape == (3, n_steps, 2, 4)
    counts = np.asarray(out["spikes"]).sum(axis=(1, 2))
    assert all(int(c) > 0 for c in counts), \
        "every multi batch element must spike, got %r" % (counts,)
    for b in range(3):
        ref = simulate_hh(params, n_steps=n_steps, dt_ms=dt,
                          I_ext_nA=nA[b], edge_weight_uS=uS[b], **shared)
        np.testing.assert_array_equal(np.asarray(out["spikes"][b]),
                                      np.asarray(ref["spikes"]))
        # float32 rounding near spike steps: the batch-vs-serial V difference
        # (at most about 0.012 mV here) equals the jit-vs-eager difference of
        # the serial call itself, and with JAX_ENABLE_X64 the serial-vs-batch
        # V difference is about 1e-12 mV.
        np.testing.assert_allclose(np.asarray(out["V_mV"][b]),
                                   np.asarray(ref["V_mV"]),
                                   rtol=0, atol=2e-2)


def test_batch_refuses_unknown_key():
    params = make_hh_params(["E", "E"])
    with pytest.raises(ValueError, match="unknown batch keys"):
        simulate_hh_batch(params, n_steps=10, dt_ms=DT_MS,
                          batch={"nope": np.zeros((2, 2))},
                          **_two_cell_shared())


def test_batch_refuses_mismatched_leading_size():
    params = make_hh_params(["E", "E"])
    with pytest.raises(ValueError, match="share one leading size"):
        simulate_hh_batch(
            params, n_steps=10, dt_ms=DT_MS,
            batch={"I_ext": np.zeros((2, 2)), "V0_mV": np.zeros((3, 2))},
            **_two_cell_shared())


def test_batch_refuses_duplicated_shared_key():
    params = make_hh_params(["E", "E"])
    with pytest.raises(ValueError, match="also given as shared"):
        simulate_hh_batch(params, n_steps=10, dt_ms=DT_MS,
                          batch={"I_ext": np.zeros((2, 2))},
                          I_ext=np.zeros(2), **_two_cell_shared())


def test_batch_refuses_mixed_weight_forms():
    params = make_hh_params(["E", "E"])
    with pytest.raises(ValueError, match="alternative input forms"):
        simulate_hh_batch(
            params, n_steps=10, dt_ms=DT_MS,
            batch={"edge_weight_uS": np.ones((2, 1))},
            edge_weight=[1.0],
            morphology=reference_4comp_morphology(),
            **_two_cell_shared())


def test_batch_refuses_mixed_current_forms():
    params = make_hh_params(["E", "E"])
    with pytest.raises(ValueError, match="alternative input forms"):
        simulate_hh_batch(
            params, n_steps=10, dt_ms=DT_MS,
            batch={"I_ext_nA": np.zeros((2, 2))},
            I_ext=np.zeros(2),
            morphology=reference_4comp_morphology(),
            **_two_cell_shared())


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize(("kind", "match"), [
    ("neg_weight", "edge_weight must be >= 0"),
    ("nan_weight", "edge_weight must be finite"),
    ("neg_gain", "noise_gain must be finite"),
    ("nan_gain", "noise_gain must be finite"),
    ("missing_tau", "ou_tau is required"),
])
def test_batch_refuses_bad_later_element(b, kind, match):
    """Elements after the first get the same value checks as element 0."""
    params = make_hh_params(["E", "E"])
    if kind in ("neg_weight", "nan_weight"):
        w = np.full((3, 1), 2.0)
        w[b, 0] = -1.0 if kind == "neg_weight" else np.nan
        with pytest.raises(ValueError, match=match):
            simulate_hh_batch(params, n_steps=10, dt_ms=DT_MS,
                              batch={"edge_weight": w}, **_two_cell_shared())
    elif kind in ("neg_gain", "nan_gain"):
        g = np.ones(3)
        g[b] = -1.0 if kind == "neg_gain" else np.nan
        shared = _two_cell_shared(edge_weight=[2.0],
                                  I_ext=np.asarray([6.0, 0.0]))
        with pytest.raises(ValueError, match=match):
            simulate_hh_batch(params, n_steps=10, dt_ms=DT_MS,
                              batch={"noise_gain": g}, **shared)
    else:
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        g = np.zeros(3)
        g[b] = 1.0
        shared = _two_cell_shared(edge_weight=[2.0],
                                  I_ext=np.asarray([6.0, 0.0]),
                                  ou_sigma=[5.0, 5.0])
        with pytest.raises(ValueError, match=match):
            simulate_hh_batch(params, n_steps=10, dt_ms=DT_MS,
                              batch={"ou_key": keys, "noise_gain": g},
                              **shared)


def test_serial_jitted_traced_gain_raises():
    """A user-jitted serial call with a traced bad gain raises, as at baseline.

    The batch path pre-validates each element on the host and takes the traced
    slice as-is under an internal flag; the public serial path keeps the
    baseline refusal for concrete and traced inputs alike.
    """
    params = make_hh_params(["E", "E"])
    shared = _two_cell_shared(edge_weight=[2.0],
                              I_ext=np.asarray([6.0, 0.0]))
    with pytest.raises(ValueError, match="noise_gain"):
        jax.jit(lambda g: simulate_hh(params, n_steps=10, dt_ms=DT_MS,
                                      noise_gain=g, **shared))(-1.0)
