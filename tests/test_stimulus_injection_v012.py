import json
import math
import jax
import jax.numpy as jnp
import pytest
from dataclasses import replace

import jaxfne as jtfne
from jaxfne.core import StimulusSchedule, ParadigmCondition, ParadigmEvent


def _cfg(n=5):
    return (
        jtfne.configuration()
        .network(n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", n_contacts=16)
    )


def _simple_condition(name="test_cond"):
    return jtfne.ParadigmCondition(
        name=name,
        sequence=("A", "A", "A", "B"),
        events=(jtfne.ParadigmEvent(label="p1", onset_ms=5.0, code=101),),
    )


# Test A — stimulus schedule construction
def test_stimulus_schedule_construction():
    events = (
        {"label": "ev1", "onset_ms": 10.0, "duration_ms": 20.0, "amplitude": 5.0},
        {"label": "ev2", "onset_ms": 50.0, "duration_ms": 10.0, "amplitude": 0.0, "is_drive_event": False},
    )
    sched = jtfne.stimulus_schedule(events, n_neurons=10)
    assert isinstance(sched, StimulusSchedule)
    assert sched.n_neurons == 10
    assert len(sched.events) == 2
    
    d = sched.to_dict()
    assert d["stimulus_injection_status"] == "native_drive_schedule_v0.0.12"
    assert d["physical_amplitude_claim_allowed"] is False
    assert d["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
    
    # JSON safe
    json.dumps(d, allow_nan=False)


# Test B — to_array shape and timing
def test_stimulus_schedule_to_array_shape_and_timing():
    events = (
        {"label": "stim", "onset_ms": 10.0, "duration_ms": 20.0, "amplitude": 10.0},
    )
    n_neurons = 4
    sched = jtfne.stimulus_schedule(events, n_neurons=n_neurons)
    
    dt_ms = 1.0
    n_steps = 50
    arr = sched.to_array(n_steps=n_steps, dt_ms=dt_ms)
    
    assert arr.shape == (n_steps, n_neurons)
    # Onset 10ms -> index 10
    # Duration 20ms -> ends at 30ms -> index 30
    assert jnp.all(arr[:10, :] == 0)
    assert jnp.all(arr[10:30, :] == 10.0)
    assert jnp.all(arr[30:, :] == 0)


# Test C — zero amplitude or no drive event
def test_zero_amplitude_or_no_drive_event():
    events = (
        {"label": "zero_amp", "onset_ms": 10.0, "duration_ms": 10.0, "amplitude": 0.0},
        {"label": "no_drive", "onset_ms": 30.0, "duration_ms": 10.0, "amplitude": 5.0, "is_drive_event": False},
    )
    sched = jtfne.stimulus_schedule(events, n_neurons=2)
    arr = sched.to_array(n_steps=50, dt_ms=1.0)
    assert jnp.all(arr == 0)


# Test D — simulate with stimulus schedule runs
def test_simulate_with_stimulus_schedule_runs():
    model = jtfne.construct(_cfg(n=5))
    events = ({"label": "s1", "onset_ms": 5.0, "duration_ms": 5.0, "amplitude": 10.0},)
    sched = jtfne.stimulus_schedule(events, n_neurons=5)
    
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.1)
    signals = model.simulate(sim, paradigm=sched)
    
    assert signals.V_m.shape == (200, 5)
    assert signals.metadata["stimulus_injection_status"] == "native_drive_schedule_v0.0.12"
    assert "stimulus_schedule" in signals.metadata


# Test E — simulate with paradigm condition runs
def test_simulate_with_paradigm_condition_runs():
    model = jtfne.construct(_cfg(n=5))
    cond = _simple_condition("test_cond")
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.1)
    signals = model.simulate(sim, paradigm=cond)

    assert signals.metadata["condition_name"] == "test_cond"
    assert signals.metadata["stimulus_injection_status"] == "native_drive_schedule_v0.0.12"


# Test F — simulate_condition wrapper
def test_simulate_condition_wrapper():
    model = jtfne.construct(_cfg(n=5))
    cond = _simple_condition("wrap_cond")
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.1)
    # Using wrapper
    s1 = model.simulate_condition(sim, cond, drive_amplitude=15.0)

    # Using simulate directly with explicit schedule
    sched = jtfne.stimulus_schedule(cond.events, n_neurons=5, drive_amplitude=15.0)
    s2 = model.simulate(sim, paradigm=sched)

    assert jnp.allclose(s1.V_m, s2.V_m)
    assert s1.metadata["condition_name"] == "wrap_cond"


# Test G — default simulate backward compatibility
def test_default_simulate_backward_compatibility():
    model = jtfne.construct(_cfg(n=5))
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=42)
    
    # v0.0.11 behavior
    s1 = model.simulate(sim, paradigm=None)
    
    # Ensure no-schedule path doesn't crash and is stable
    assert s1.metadata["paradigm"] is None
    assert "stimulus_injection_status" not in s1.metadata
    
    # Fixed seed determinism for no-schedule path
    s2 = model.simulate(sim, paradigm=None)
    assert jnp.array_equal(s1.V_m, s2.V_m)


# Test H — stimulus injection reproducible
def test_stimulus_injection_reproducible():
    model = jtfne.construct(_cfg(n=5))
    cond = _simple_condition("repro")
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=123)
    
    s1 = model.simulate(sim, paradigm=cond)
    s2 = model.simulate(sim, paradigm=cond)
    
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(s1.spikes, s2.spikes)


# Test I — stimulus injection truth gates unchanged
def test_stimulus_injection_truth_gates_unchanged():
    model = jtfne.construct(_cfg(n=5))
    cond = _simple_condition("truth")
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1)
    signals = model.simulate(sim, paradigm=cond)

    meta = signals.metadata
    assert meta["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
    assert meta["field_claim_level"] == "proxy_readout_only"

    manifest = model.manifest(signals)
    assert manifest["truth_mode"] == "truth_safe_unverified"
    assert manifest["claim_level"] == "computational_scaffold"
    labels = manifest.get("v005_claim_labels", {})
    assert labels.get("physical_amplitude_claim_allowed", False) is False
    assert labels.get("empirical_validation_status", "not_empirically_validated") == "not_empirically_validated"
    assert labels.get("mechanism_claim_status", "not_claimed") == "not_claimed"


# Test J — stimulus manifest json safe
def test_stimulus_manifest_json_safe():
    model = jtfne.construct(_cfg(n=5))
    cond = _simple_condition("json")
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1)
    signals = model.simulate(sim, paradigm=cond)

    manifest = model.manifest(signals)
    json.dumps(manifest, allow_nan=False)


# Test K — batch path no condition support yet
def test_batch_path_no_condition_support_yet():
    # simulate_batch doesn't take paradigm yet, so it should run as None
    model = jtfne.construct(_cfg(n=5))
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1)
    res = model.simulate_batch(sim, n_seeds=2)
    
    # The batch metadata should not claim stimulus injection
    assert "stimulus_injection_status" not in res
