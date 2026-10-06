"""Poisson drive under full-state continuation (0.5.5 R6 engine piece).

Small fixture only (suite2, 8 neurons): chunk 0 draws the declared Poisson
seed (bit-identical to a plain call), later chunks derive theirs from
(seed, chunk_index). No 20k run here; the long-T runner itself
(``run_r6_long``) is covered through its pure helpers.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from artifacts.atlas import at10_n20_055 as A

DT = 0.5
N = 8
POISSON = {"rate_hz": 200.0, "amplitude": 4.0, "target": "all", "seed": 11}


def _model():
    cfg = jtfne.suite2_net1_config(seed=11, n=N, duration_ms=20.0, dt_ms=DT)
    return jtfne.construct(cfg)


def _runtime(*, enable_hdp: bool):
    hp = {} if not enable_hdp else {
        "K_HDP": 0.01, "K_ctrl": 0.15, "K_w_ctrl": 0.001,
        "tau_0_ms": 20.0, "alpha": 0.01, "barrier_c": 0.01, "barrier_d": 0.01,
        "noise_scale": 0.0,
    }
    return jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list",
                               enable_hdp=enable_hdp, hdp_params=dict(hp))


def _sched():
    return jtfne.stimulus_schedule(
        [{"onset_ms": 1.0, "duration_ms": 1.0, "amplitude": 40.0,
          "target_indices": [0]},
         {"onset_ms": 3.0, "duration_ms": 1.0, "amplitude": 38.0,
          "target_indices": [1]}],
        N)


def _kw(**over):
    kw = dict(duration_ms=12.0, dt_ms=DT, seed=17,
              runtime=_runtime(enable_hdp=True),
              record_sources=True, record_fields=False,
              poisson_drive=dict(POISSON))
    kw.update(over)
    return kw


def test_chunk0_matches_plain_call_bit_exact():
    """(a) The first continued chunk draws the declared seed: spikes and V
    are bit-identical to a plain single call with the same Simulation."""
    model, sched = _model(), _sched()
    plain = jtfne.simulate(model, paradigm=sched, **_kw())
    first, st0 = jtfne.simulate(model, paradigm=sched, return_state=True,
                                **_kw())
    assert int(np.asarray(first.spikes).sum()) > 0, "fixture must spike"
    assert jnp.array_equal(plain.spikes, first.spikes)
    assert jnp.array_equal(plain.V_m, first.V_m)
    assert st0.step_index == 24 == int(np.asarray(first.spikes).shape[0])
    assert st0.chunk_index == 1


def test_chunk1_differs_and_reruns_identically():
    """(b) Chunk 1 draws its own realization (differs from chunk 0) and the
    continued segment is deterministic given the carried state."""
    model, sched = _model(), _sched()
    first, st0 = jtfne.simulate(model, paradigm=sched, return_state=True,
                                **_kw())
    second, st1 = jtfne.simulate(model, paradigm=sched, continuation=st0,
                                 return_state=True,
                                 **{k: v for k, v in _kw().items()
                                    if k != "duration_ms"},
                                 duration_ms=12.0)
    assert not jnp.array_equal(second.spikes, first.spikes)
    assert not jnp.array_equal(second.V_m, first.V_m)
    assert st1.chunk_index == 2
    rerun, st1b = jtfne.simulate(model, paradigm=sched, continuation=st0,
                                 return_state=True,
                                 **{k: v for k, v in _kw().items()
                                    if k != "duration_ms"},
                                 duration_ms=12.0)
    assert jnp.array_equal(rerun.spikes, second.spikes)
    assert jnp.array_equal(rerun.V_m, second.V_m)
    assert jnp.array_equal(st1b.dynamic.H, st1.dynamic.H)


def test_two_chunks_run_with_state_continuity():
    """(c) Two 6 ms chunks thread H/w/delays/PRNG: shapes, indices, HDP
    movement and whole-trajectory determinism."""
    model, sched = _model(), _sched()
    first, st0 = jtfne.simulate(model, paradigm=sched, return_state=True,
                                **{**_kw(), "duration_ms": 6.0})
    second, st1 = jtfne.simulate(model, paradigm=sched, continuation=st0,
                                 return_state=True,
                                 **{**_kw(), "duration_ms": 6.0})
    assert first.spikes.shape == second.spikes.shape == (12, N)
    assert st0.step_index == 12
    assert st1.step_index == 24
    assert (st0.chunk_index, st1.chunk_index) == (1, 2)
    assert bool(jnp.all(jnp.isfinite(second.V_m)))
    assert not jnp.array_equal(st1.dynamic.H, st0.dynamic.H)
    assert bool(jnp.any(st0.dynamic.H != 0))
    assert bool(jnp.any(st1.dynamic.H != 0))
    again_first, again_st0 = jtfne.simulate(
        model, paradigm=sched, return_state=True, **{**_kw(), "duration_ms": 6.0})
    again_second, again_st1 = jtfne.simulate(
        model, paradigm=sched, continuation=again_st0, return_state=True,
        **{**_kw(), "duration_ms": 6.0})
    assert jnp.array_equal(again_first.spikes, first.spikes)
    assert jnp.array_equal(again_second.spikes, second.spikes)
    assert jnp.array_equal(again_st1.dynamic.H, st1.dynamic.H)


def test_non_poisson_continuation_is_unchanged():
    """(d) Without Poisson noise, chunked == continuous bit-exactly: the
    engine change touches only the Poisson-seed path. Each chunk gets its
    own window's pulses (explicit schedules carry no cursor)."""
    model = _model()
    pulse = [(1.0, 40.0, [0]), (3.0, 38.0, [1])]
    full_sched = jtfne.stimulus_schedule(
        [{"onset_ms": o, "duration_ms": 1.0, "amplitude": a, "target_indices": i}
         for o, a, i in pulse + [(o + 6.0, a, i) for o, a, i in pulse]], N)
    half_sched = jtfne.stimulus_schedule(
        [{"onset_ms": o, "duration_ms": 1.0, "amplitude": a, "target_indices": i}
         for o, a, i in pulse], N)
    base = {k: v for k, v in _kw().items() if k != "poisson_drive"}
    full, full_state = jtfne.simulate(model, paradigm=full_sched,
                                      return_state=True,
                                      **{**base, "duration_ms": 12.0})
    first, st0 = jtfne.simulate(model, paradigm=half_sched, return_state=True,
                                **{**base, "duration_ms": 6.0})
    second, st1 = jtfne.simulate(model, paradigm=half_sched, continuation=st0,
                                 return_state=True, **{**base, "duration_ms": 6.0})
    assert jnp.array_equal(
        full.spikes, jnp.concatenate((first.spikes, second.spikes), axis=0))
    assert jnp.array_equal(
        full.V_m, jnp.concatenate((first.V_m, second.V_m), axis=0))
    assert jnp.array_equal(full_state.dynamic.H, st1.dynamic.H)
    assert jnp.array_equal(full_state.dynamic.w, st1.dynamic.w)
    assert st1.step_index == full_state.step_index == 24
    assert (st1.chunk_index, full_state.chunk_index) == (2, 1)


def test_poisson_refusal_is_gone_but_other_guards_hold():
    """(e) poisson_drive + continuation/return_state runs; ablation and
    kernel guards still refuse."""
    model, sched = _model(), _sched()
    signals, state = jtfne.simulate(model, paradigm=sched, return_state=True,
                                    **_kw())
    assert signals.spikes.shape == (24, N)
    assert state.chunk_index == 1
    with pytest.raises(ValueError, match="does not support ablation"):
        jtfne.simulate(model, paradigm=sched, return_state=True,
                       **{**_kw(), "ablation": "E_silence"})
    bad_rt = jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list",
                                 synaptic_kernel="receptor_exponential",
                                 enable_hdp=True,
                                 hdp_params=dict(_runtime(enable_hdp=True).hdp_params))
    with pytest.raises(ValueError, match="supports only synaptic_kernel"):
        jtfne.simulate(model, paradigm=sched, return_state=True,
                       **{**_kw(), "runtime": bad_rt})


def test_chunk_seed_zero_is_identity():
    from jaxfne._signals import poisson_chunk_seed
    assert poisson_chunk_seed(11, 0) == 11
    assert poisson_chunk_seed(11, -1) == 11


def test_chunk_seeds_differ_deterministically():
    from jaxfne._signals import poisson_chunk_seed
    s1 = poisson_chunk_seed(11, 1)
    s2 = poisson_chunk_seed(11, 2)
    assert s1 != 11 and s2 != 11 and s1 != s2
    assert poisson_chunk_seed(11, 1) == s1
    assert 0 <= s1 < 2_147_483_647


# --- run_r6_long pure helpers (synthetic input, no model) ---


def _fake_chunk(i: int, rate: float, h: float, w_ratio: float,
                clip: bool = False) -> dict:
    areas = ["H01", "H02"]
    return {"chunk": i, "start_ms": float(i), "duration_ms": 1.0,
            "chunk_index": i + 1, "wall_s": 1.5, "rss_mb": 10.0,
            "mean_rate_hz": rate,
            "per_area_mean_hz": {a: rate for a in areas},
            "h_mean": {a: h for a in areas}, "h_min": {a: h - 1.0 for a in areas},
            "h_max": {a: h + 1.0 for a in areas},
            "h_bounds": [0.1, 10.0], "clip_reached": clip,
            "w_mean_ratio": w_ratio}


def test_summarize_r6_chunks_on_synthetic_input():
    out = A.summarize_r6_chunks(
        [_fake_chunk(0, 2.0, 1.0, 0.9), _fake_chunk(1, 4.0, 3.0, 0.8, clip=True)])
    assert out["n_chunks"] == 2
    assert out["per_area_mean_hz"] == {"H01": 3.0, "H02": 3.0}
    assert out["h_mean"] == {"H01": 2.0, "H02": 2.0}
    assert out["h_min"] == {"H01": 0.0, "H02": 0.0}
    assert out["h_max"] == {"H01": 4.0, "H02": 4.0}
    assert out["clip_reached"] is True
    assert out["w_mean_ratio_last"] == 0.8
    assert out["wall_s"] == pytest.approx(3.0)
    json.dumps(out)


def test_summarize_r6_chunks_refuses_empty():
    with pytest.raises(ValueError, match="at least one chunk"):
        A.summarize_r6_chunks([])


def test_chunk_stimulus_slices_and_shifts_onsets():
    area = np.array(["H01"] * 4 + ["H02"] * 4)
    total, chunk = 1000.0, 500.0
    s0 = A._chunk_stimulus(area, 0.0, chunk, total)
    s1 = A._chunk_stimulus(area, chunk, chunk, total)
    on0 = sorted(e["onset_ms"] for e in s0.events)
    on1 = sorted(e["onset_ms"] for e in s1.events)
    assert on0 == [100.0, 300.0]
    assert on1 == [0.0, 200.0]
    assert all(e["target_indices"] == [0, 1, 2, 3] for e in s0.events)
    n0, n1 = int(chunk / A.DT_MS), int(total / A.DT_MS)
    arr0 = np.asarray(s0.to_array(n0, A.DT_MS))
    arr1 = np.asarray(s1.to_array(n1 - n0, A.DT_MS))
    assert arr0.shape == (n0, 8) and arr1.shape == (n1 - n0, 8)
    full = np.asarray(A._chunk_stimulus(area, 0.0, total, total).to_array(n1, A.DT_MS))
    assert np.array_equal(np.concatenate((arr0, arr1), axis=0), full)


def test_rss_mb_is_float_or_none():
    assert A._rss_mb() is None or isinstance(A._rss_mb(), float)
