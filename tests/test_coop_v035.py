import jaxfne as jtfne
import numpy as np
import pytest

def test_coop_paradigm_api():
    # Verify both access paths exist
    assert hasattr(jtfne, "coop_omission_oddball_paradigm")
    assert hasattr(jtfne.paradigm, "coop_omission_oddball_paradigm")
    assert jtfne.coop_omission_oddball_paradigm is jtfne.paradigm.coop_omission_oddball_paradigm

    # Construct paradigm
    p = jtfne.coop_omission_oddball_paradigm(
        duration_ms=2000.0,
        dt_ms=0.5,
        freq_hz=5.0,
        omission_prob=0.2,
        standard_amplitude=4.0,
        pre_stimulus_buffer_ms=100.0,
        target_indices=[0, 1, 2],
        seed=123
    )

    assert p.name == "coop_omission_oddball"
    assert len(p.conditions) == 1
    cond = p.conditions[0]
    assert cond.name == "coop_condition"
    
    # Check that events have metadata with target_indices
    stim_events = [e for e in cond.events if e.onset_ms > 0]
    for e in stim_events:
        if "target_indices" in e.metadata:
            assert e.metadata["target_indices"] == [0, 1, 2]

def test_stimulus_schedule_backward_compatibility():
    # Old style event list using dicts (no target_indices)
    events_old = [
        {"onset_ms": 10.0, "duration_ms": 20.0, "amplitude": 5.0, "is_drive_event": True},
        {"onset_ms": 50.0, "duration_ms": 20.0, "amplitude": 10.0, "is_drive_event": True},
    ]

    sched_old = jtfne.stimulus_schedule(events_old, n_neurons=5)
    arr_old = sched_old.to_array(n_steps=200, dt_ms=0.5)

    # All 5 neurons should receive stimulation
    # onset 10ms -> step 20, duration 20ms -> 40 steps (step 20 to 60)
    assert np.all(arr_old[20:60, :] == 5.0)
    assert np.all(arr_old[60:100, :] == 0.0)
    assert np.all(arr_old[100:140, :] == 10.0)

def test_stimulus_schedule_targeted():
    # Create events with target_indices using ParadigmEvents
    e1 = jtfne.ParadigmEvent(
        label="pulse1",
        onset_ms=10.0,
        duration_ms=20.0,
        metadata={"drive_amplitude": 5.0, "event_duration_ms": 20.0, "target_indices": [1, 3]}
    )
    # Event without target_indices (should target all)
    e2 = jtfne.ParadigmEvent(
        label="pulse2",
        onset_ms=50.0,
        duration_ms=20.0,
        metadata={"drive_amplitude": 8.0, "event_duration_ms": 20.0}
    )

    sched = jtfne.stimulus_schedule([e1, e2], n_neurons=5)
    arr = sched.to_array(n_steps=200, dt_ms=0.5)

    # First event (step 20 to 60) should only target neurons 1 and 3
    assert np.all(arr[20:60, [1, 3]] == 5.0)
    assert np.all(arr[20:60, [0, 2, 4]] == 0.0)

    # Second event (step 100 to 140) should target all neurons
    assert np.all(arr[100:140, :] == 8.0)
