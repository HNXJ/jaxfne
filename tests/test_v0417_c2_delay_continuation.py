"""0.4.17-C2 — canonical Model.simulate delay_state continuation.

Pre-registered tolerance: bit-exact (float32, max abs diff == 0) with
noise_scale=0 and deterministic drive schedules. No wave-specific behavior.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.core import StimulusSchedule


def _runtime_deterministic():
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=False,
        hdp_params={"noise_scale": 0.0},
    )


def _model_with_delays(
  *,
  n: int = 3,
  delay_steps: int | list[int] = 0,
  seed: int = 7,
):
    cfg = jtfne.suite2_net1_config(seed=seed, n=n, duration_ms=100.0, dt_ms=1.0)
    model = jtfne.construct(cfg)
    edges = model.params["edge_list"]
    if isinstance(delay_steps, int):
        ds = jnp.full((edges.n_edges,), int(delay_steps), dtype=jnp.int32)
    else:
        ds = jnp.asarray(delay_steps, dtype=jnp.int32)
        if ds.shape[0] != edges.n_edges:
            reps = int(np.ceil(edges.n_edges / ds.shape[0]))
            ds = jnp.tile(ds, reps)[: edges.n_edges]
    new_edges = replace(edges, delay_steps=ds)
    object.__setattr__(model, "params", {**model.params, "edge_list": new_edges})
    return model


def _build_drive(
    n_steps: int,
    n_neurons: int,
    pulses: list[tuple[int, int, float]],
) -> np.ndarray:
    arr = np.zeros((n_steps, n_neurons), dtype=np.float32)
    for step, neuron, amp in pulses:
        arr[int(step), int(neuron)] = float(amp)
    return arr


def _schedule_from_array(drive: np.ndarray, *, dt_ms: float = 1.0) -> StimulusSchedule:
    events = []
    n_steps, n_neurons = drive.shape
    for t in range(n_steps):
        for n in range(n_neurons):
            amp = float(drive[t, n])
            if amp != 0.0:
                events.append(
                    {
                        "onset_ms": float(t) * dt_ms,
                        "duration_ms": dt_ms,
                        "amplitude": amp,
                        "target_indices": [int(n)],
                        "is_drive_event": True,
                    }
                )
    return StimulusSchedule(events=tuple(events), n_neurons=n_neurons)


def _pulse_schedule(
    *,
    n_steps: int,
    n_neurons: int,
    pulses: list[tuple[int, int, float]],
) -> StimulusSchedule:
    """(step, neuron, amplitude) tuples -> StimulusSchedule."""
    events = []
    dt_ms = 1.0
    for step, neuron, amp in pulses:
        events.append(
            {
                "onset_ms": float(step) * dt_ms,
                "duration_ms": dt_ms,
                "amplitude": float(amp),
                "target_indices": [int(neuron)],
                "is_drive_event": True,
            }
        )
    return StimulusSchedule(events=tuple(events), n_neurons=n_neurons)


def _simulate(
    model,
    *,
    n_steps: int,
    schedule: StimulusSchedule | None = None,
    continuation=None,
    return_state: bool = False,
    seed: int = 0,
):
    sim = jtfne.Simulation(
        duration_ms=float(n_steps),
        dt_ms=1.0,
        seed=seed,
        runtime=_runtime_deterministic(),
        record_sources=True,
        record_fields=False,
    )
    out = model.simulate(
        sim,
        paradigm=schedule,
        continuation=continuation,
        return_state=return_state,
    )
    if return_state:
        return out
    return out


def _spike_times(spikes: jax.Array, dt_ms: float = 1.0) -> list[tuple[int, int]]:
    arr = np.asarray(spikes)
    idx = np.argwhere(arr > 0.5)
    return [(int(r[0]), int(r[1])) for r in idx]


def _assert_bit_exact_states(st_full, st_split, *, delayed: bool):
    for key in ("v", "u", "prev_spikes", "syn_state", "H", "w"):
        assert float(
            jnp.max(jnp.abs(getattr(st_full.dynamic, key) - getattr(st_split.dynamic, key)))
        ) == 0.0
    if delayed:
        assert st_full.delay_state is not None
        assert float(jnp.max(jnp.abs(st_full.delay_state - st_split.delay_state))) == 0.0
        assert int(np.asarray(st_full.step_index)) == int(np.asarray(st_split.step_index))
    else:
        assert st_full.delay_state is None
        assert st_split.delay_state is None


def test_c2_zero_delay_no_delay_buffer_on_return_state():
    """Zero-delay models carry delay_state=None (no buffer cost)."""
    model = _model_with_delays(delay_steps=0, n=8)
    n_steps = 20
    sched = _pulse_schedule(n_steps=n_steps, n_neurons=8, pulses=[(5, 0, 40.0)])
    _, state = _simulate(model, n_steps=n_steps, schedule=sched, return_state=True)
    assert state.delay_state is None
    sig = _simulate(model, n_steps=n_steps, schedule=sched)
    assert sig.V_m.shape == (n_steps, 8)


def test_c2_zero_delay_segmented_bit_exact():
    model = _model_with_delays(delay_steps=0, n=3)
    t1, t2 = 30, 25
    total = t1 + t2
    drive = _build_drive(total, 3, [(7, 0, 42.0), (22, 1, 38.0)])
    sched = _schedule_from_array(drive)
    full, st_full = _simulate(model, n_steps=total, schedule=sched, return_state=True)
    first, st1 = _simulate(model, n_steps=t1, schedule=_schedule_from_array(drive[:t1]), return_state=True)
    second, st2 = _simulate(
        model,
        n_steps=t2,
        schedule=_schedule_from_array(drive[t1:total]),
        continuation=st1,
        return_state=True,
    )
    assert jnp.array_equal(full.V_m, jnp.concatenate([first.V_m, second.V_m]))
    assert jnp.array_equal(full.spikes, jnp.concatenate([first.spikes, second.spikes]))
    _assert_bit_exact_states(st_full, st2, delayed=False)


def test_c2_uniform_delay_segmented_bit_exact():
    model = _model_with_delays(delay_steps=4, n=3)
    t1, t2 = 35, 30
    total = t1 + t2
    drive = _build_drive(total, 3, [(7, 0, 42.0), (22, 1, 38.0)])
    sched = _schedule_from_array(drive)
    full, st_full = _simulate(model, n_steps=total, schedule=sched, return_state=True)
    first, st1 = _simulate(model, n_steps=t1, schedule=_schedule_from_array(drive[:t1]), return_state=True)
    second, st2 = _simulate(
        model,
        n_steps=t2,
        schedule=_schedule_from_array(drive[t1:total]),
        continuation=st1,
        return_state=True,
    )
    assert float(jnp.max(jnp.abs(full.V_m - jnp.concatenate([first.V_m, second.V_m])))) == 0.0
    assert float(jnp.max(jnp.abs(full.spikes - jnp.concatenate([first.spikes, second.spikes])))) == 0.0
    _assert_bit_exact_states(st_full, st2, delayed=True)


def test_c2_heterogeneous_delay_segmented_bit_exact():
    model = _model_with_delays(delay_steps=[0, 3, 6], n=3)
    t1, t2 = 40, 35
    total = t1 + t2
    drive = _build_drive(total, 3, [(7, 0, 42.0), (22, 1, 38.0)])
    sched = _schedule_from_array(drive)
    full, st_full = _simulate(model, n_steps=total, schedule=sched, return_state=True)
    first, st1 = _simulate(model, n_steps=t1, schedule=_schedule_from_array(drive[:t1]), return_state=True)
    second, st2 = _simulate(
        model,
        n_steps=t2,
        schedule=_schedule_from_array(drive[t1:total]),
        continuation=st1,
        return_state=True,
    )
    assert float(jnp.max(jnp.abs(full.spikes - jnp.concatenate([first.spikes, second.spikes])))) == 0.0
    _assert_bit_exact_states(st_full, st2, delayed=True)


def test_c2_split_while_delay_buffer_inflight():
    """Split mid propagation so in-flight delayed events must survive."""
    model = _model_with_delays(delay_steps=6, n=3)
    pulse_step = 8
    t1 = pulse_step + 2
    t2 = 12
    total = t1 + t2
    drive = _build_drive(total, 3, [(pulse_step, 0, 50.0)])
    sched = _schedule_from_array(drive)
    full, st_full = _simulate(model, n_steps=total, schedule=sched, return_state=True)
    first, st1 = _simulate(model, n_steps=t1, schedule=_schedule_from_array(drive[:t1]), return_state=True)
    second, st2 = _simulate(
        model,
        n_steps=t2,
        schedule=_schedule_from_array(drive[t1:total]),
        continuation=st1,
        return_state=True,
    )
    assert float(jnp.max(jnp.abs(full.spikes - jnp.concatenate([first.spikes, second.spikes])))) == 0.0
    _assert_bit_exact_states(st_full, st2, delayed=True)


def test_c2_triple_segmentation():
    model = _model_with_delays(delay_steps=4, n=3)
    t1, t2, t3 = 20, 18, 22
    total = t1 + t2 + t3
    drive = _build_drive(total, 3, [(5, 0, 45.0), (30, 1, 40.0)])
    sched = _schedule_from_array(drive)
    full, st_full = _simulate(model, n_steps=total, schedule=sched, return_state=True)
    s1, st1 = _simulate(model, n_steps=t1, schedule=_schedule_from_array(drive[:t1]), return_state=True)
    s2, st2 = _simulate(
        model,
        n_steps=t2,
        schedule=_schedule_from_array(drive[t1 : t1 + t2]),
        continuation=st1,
        return_state=True,
    )
    s3, st3 = _simulate(
        model,
        n_steps=t3,
        schedule=_schedule_from_array(drive[t1 + t2 : total]),
        continuation=st2,
        return_state=True,
    )
    seg = jnp.concatenate([s1.spikes, s2.spikes, s3.spikes])
    assert jnp.array_equal(full.spikes, seg)
    _assert_bit_exact_states(st_full, st3, delayed=True)


def test_c2_quiescent_boundary_preserves_delayed_arrivals():
    """No spikes in segment 2; delayed presynaptic events must still land."""
    model = _model_with_delays(delay_steps=5, n=3)
    pulse_step = 10
    t1 = pulse_step + 1
    t2 = 12
    total = t1 + t2
    drive = _build_drive(total, 3, [(pulse_step, 0, 55.0)])
    sched = _schedule_from_array(drive)
    full, st_full = _simulate(model, n_steps=total, schedule=sched, return_state=True)
    first, st1 = _simulate(model, n_steps=t1, schedule=_schedule_from_array(drive[:t1]), return_state=True)
    assert float(jnp.sum(first.spikes[t1 - 3 : t1])) == 0.0
    second, st2 = _simulate(
        model,
        n_steps=t2,
        schedule=_schedule_from_array(drive[t1:total]),
        continuation=st1,
        return_state=True,
    )
    assert float(jnp.max(jnp.abs(full.spikes - jnp.concatenate([first.spikes, second.spikes])))) == 0.0
    _assert_bit_exact_states(st_full, st2, delayed=True)


def test_c2_rejects_delayed_continuation_without_delay_state():
    model = _model_with_delays(delay_steps=3, n=3)
    sched = _pulse_schedule(n_steps=10, n_neurons=3, pulses=[(3, 0, 40.0)])
    _, st1 = _simulate(model, n_steps=5, schedule=sched, return_state=True)
    bad = st1._replace(delay_state=None)
    with pytest.raises(ValueError, match="delay_state"):
        _simulate(
            model,
            n_steps=5,
            schedule=_pulse_schedule(n_steps=5, n_neurons=3, pulses=[]),
            continuation=bad,
        )


def test_c2_event_timing_exact_across_segmentation():
    model = _model_with_delays(delay_steps=4, n=3)
    t1, t2 = 28, 24
    total = t1 + t2
    drive = _build_drive(total, 3, [(6, 0, 48.0), (18, 2, 44.0)])
    sched = _schedule_from_array(drive)
    full, _ = _simulate(model, n_steps=total, schedule=sched, return_state=True)
    first, st1 = _simulate(model, n_steps=t1, schedule=_schedule_from_array(drive[:t1]), return_state=True)
    second, _ = _simulate(
        model,
        n_steps=t2,
        schedule=_schedule_from_array(drive[t1:total]),
        continuation=st1,
        return_state=True,
    )
    seg_spikes = jnp.concatenate([first.spikes, second.spikes], axis=0)
    assert jnp.array_equal(full.spikes, seg_spikes)
    assert _spike_times(full.spikes) == _spike_times(seg_spikes)


def test_c2_frozen_receipt_passes():
    from jaxfne.protocol_c.c2_validation import load_c2_receipt

    receipt = load_c2_receipt()
    assert receipt["c2_pass"] is True
    assert receipt["checkpoint"] == "C2"
