"""P-017: paradigm events are injected as declared.

One test per behaviour of the 2026-09-28 "fix the semantics" decision.
Model config pattern copied from tests/test_evoked_l4_drive.py
(``column(...).set_emitter(...).probes(...)``).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def _model(n=12):
    cfg = (
        jtfne.configuration()
        .column("V1", layers=["L2/3", "L4"], n=n)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _sim(**kw):
    args = dict(duration_ms=200.0, dt_ms=0.5, seed=7)
    args.update(kw)
    return jtfne.simulation(**args)


def _layers(model):
    return [row["layer"] for row in model.neuron_table()]


def test_condition_without_stimulus_injects_nothing():
    """A ParadigmCondition whose events carry no stimulus injects nothing."""
    model = _model()
    cond = jtfne.ParadigmCondition(
        name="markers_only",
        sequence=("a", "b"),
        events=(
            jtfne.ParadigmEvent(label="trial_start", onset_ms=0.0, duration_ms=50.0),
            jtfne.ParadigmEvent(label="post_stim", onset_ms=50.0, duration_ms=50.0),
        ),
    )
    sim = _sim()
    s_none = model.simulate(sim, paradigm=None)
    s_cond = model.simulate(sim, paradigm=cond)
    assert jnp.array_equal(s_cond.spikes, s_none.spikes)
    assert jnp.array_equal(s_cond.V_m, s_none.V_m)


def test_stimulus_event_uses_own_duration_amplitude_and_targets():
    """A stimulus event injects for its own duration at its metadata amplitude into targets only."""
    event = jtfne.ParadigmEvent(
        label="drive",
        onset_ms=10.0,
        duration_ms=20.0,
        stimulus="pulse",
        metadata={"drive_amplitude": 3.0, "target_indices": [0, 2]},
    )
    sched = jtfne.stimulus_schedule([event], n_neurons=6)
    arr = np.asarray(sched.to_array(n_steps=240, dt_ms=0.5))
    # onset 10ms -> step 20; own duration 20ms -> steps 20..60
    # (the 50ms default would run to step 120, the 5.0 default differs from 3.0).
    assert (arr[20:60][:, [0, 2]] == 3.0).all()
    assert (arr[20:60][:, [1, 3, 4, 5]] == 0.0).all()
    assert (arr[:20] == 0.0).all()
    assert (arr[60:] == 0.0).all()


def test_target_layer_resolution_and_refusals():
    """target_layer resolves to exactly the model's L4 neurons; misuse is refused."""
    model = _model()
    layers = _layers(model)
    l4 = [i for i, lab in enumerate(layers) if lab == "L4"]
    assert len(l4) > 0 and len(l4) < len(layers)
    event = jtfne.ParadigmEvent(
        label="drive",
        onset_ms=10.0,
        duration_ms=20.0,
        stimulus="pulse",
        metadata={"drive_amplitude": 2.0, "target_layer": "L4"},
    )
    sched = jtfne.stimulus_schedule([event], n_neurons=len(layers), layer_labels=layers)
    assert sorted(sched.events[0]["target_indices"]) == sorted(l4)
    arr = np.asarray(sched.to_array(n_steps=240, dt_ms=0.5))
    assert (arr[20:60][:, l4] == 2.0).all()
    others = [i for i in range(len(layers)) if i not in l4]
    assert (arr[:, others] == 0.0).all()

    def _schedule(meta, **kw):
        return jtfne.stimulus_schedule(
            [jtfne.ParadigmEvent(label="d", onset_ms=10.0, duration_ms=20.0,
                                 stimulus="x", metadata=meta)],
            n_neurons=len(layers),
            **kw,
        )

    with pytest.raises(ValueError):  # target_layer but no layer_labels
        _schedule({"target_layer": "L4"})
    with pytest.raises(ValueError):  # no neuron has that layer
        _schedule({"target_layer": "L9"}, layer_labels=layers)
    with pytest.raises(ValueError):  # both target_layer and target_indices
        _schedule({"target_layer": "L4", "target_indices": [0]}, layer_labels=layers)


def test_paradigm_resolution_and_refusals():
    """Single-condition Paradigm equals its condition; multi-condition and other types refused."""
    model = _model()
    cond = jtfne.ParadigmCondition(
        name="c1",
        sequence=("a",),
        events=(jtfne.ParadigmEvent(label="d", onset_ms=10.0, duration_ms=20.0, stimulus="pulse"),),
    )
    cond2 = jtfne.ParadigmCondition(
        name="c2",
        sequence=("b",),
        events=(jtfne.ParadigmEvent(label="d", onset_ms=10.0, duration_ms=20.0, stimulus="pulse"),),
    )
    sim = _sim()
    s_cond = model.simulate(sim, paradigm=cond)
    s_single = model.simulate(sim, paradigm=jtfne.Paradigm(name="solo", conditions=(cond,)))
    assert jnp.array_equal(s_single.spikes, s_cond.spikes)
    assert jnp.array_equal(s_single.V_m, s_cond.V_m)

    with pytest.raises(ValueError, match="pass one ParadigmCondition") as exc:
        model.simulate(sim, paradigm=jtfne.Paradigm(name="multi", conditions=(cond, cond2)))
    assert "multi" in str(exc.value) and "c1" in str(exc.value) and "c2" in str(exc.value)

    with pytest.raises(TypeError):
        model.simulate(sim, paradigm={"label": "bare_dict"})
    with pytest.raises(TypeError):
        model.simulate(sim, paradigm=42)


def test_standard_visual_omission_stimulus_injects_markers_do_not():
    """standard_visual_omission: p-events inject, the fx marker (and omissions) do not."""
    paradigm = jtfne.standard_visual_omission()
    arr = np.asarray(
        jtfne.stimulus_schedule(paradigm.condition("AAAB").events, n_neurons=4)
        .to_array(n_steps=1000, dt_ms=0.5)
    )
    # p1: onset 100ms -> step 200, default 50ms/5.0 -> steps 200..300 driven.
    assert (arr[200:300, :] == 5.0).all()
    # fx marker (onset 0, no stimulus) injects nothing.
    assert (arr[0:200, :] == 0.0).all()
    # AXAB p2 is an omission carrying a stimulus token: still silent.
    arr_omit = np.asarray(
        jtfne.stimulus_schedule(paradigm.condition("AXAB").events, n_neurons=4)
        .to_array(n_steps=1000, dt_ms=0.5)
    )
    assert (arr_omit[400:500, :] == 0.0).all()


def test_omission_oddball_stimulus_injects_markers_do_not():
    """omission_oddball_paradigm: the standard slot injects, markers and the omission do not."""
    paradigm = jtfne.omission_oddball_paradigm(deviant_drive_amplitude=10.0)
    arr = np.asarray(
        jtfne.stimulus_schedule(paradigm.condition("expected").events, n_neurons=4)
        .to_array(n_steps=2000, dt_ms=0.5)
    )
    # standard: onset 200ms -> step 400, own duration 100ms -> steps 400..600 at default 5.0.
    assert (arr[400:600, :] == 5.0).all()
    # trial_start marker and post_stimulus marker inject nothing.
    assert (arr[0:400, :] == 0.0).all()
    assert (arr[600:, :] == 0.0).all()
    arr_omitted = np.asarray(
        jtfne.stimulus_schedule(paradigm.condition("omitted").events, n_neurons=4)
        .to_array(n_steps=2000, dt_ms=0.5)
    )
    assert (arr_omitted[400:600, :] == 0.0).all()


def test_omission_oddball_default_call_refuses_p024():
    """P-024: a default call (both drives None) refuses."""
    with pytest.raises(ValueError, match="P-024"):
        jtfne.omission_oddball_paradigm()


@pytest.mark.parametrize("std, dev", [
    (10.0, 10.0),
    (None, 5.0),          # None realizes the simulator default drive_amplitude=5.0
    (5.0, None),
    ("10", 10.0),         # the schedule coerces with float()
    (float("nan"), 1.0),  # non-finite refused
])
def test_omission_oddball_equal_realized_amplitudes_refuse_p024(std, dev):
    """P-024: amplitudes that realize equal (or non-finite) drives refuse."""
    with pytest.raises(ValueError, match="P-024"):
        jtfne.omission_oddball_paradigm(
            standard_drive_amplitude=std, deviant_drive_amplitude=dev
        )


def test_omission_oddball_distinct_amplitudes_differ_in_simulation_p024():
    """P-024: distinct amplitudes build, and expected vs unexpected spikes differ."""
    paradigm = jtfne.omission_oddball_paradigm(
        standard_duration_ms=20.0,
        deviant_duration_ms=20.0,
        pre_stimulus_buffer_ms=20.0,
        post_stimulus_buffer_ms=50.0,
        deviant_drive_amplitude=10.0,
    )
    expected = paradigm.condition("expected")
    unexpected = paradigm.condition("unexpected")
    std_event = next(e for e in expected.events if e.stimulus == "standard_tone")
    dev_event = next(e for e in unexpected.events if e.stimulus == "deviant_tone")
    assert std_event.metadata == {"drive_amplitude": 5.0}  # None bound to the simulator default
    assert dev_event.metadata == {"drive_amplitude": 10.0}
    model = _model()
    sim = _sim()
    s_expected = model.simulate(sim, paradigm=expected)
    s_unexpected = model.simulate(sim, paradigm=unexpected)
    assert not jnp.array_equal(s_expected.spikes, s_unexpected.spikes)
    # a simulate-time drive_amplitude equal to the deviant must not re-equalize (peer finding #126)
    s_expected_x = model.simulate_condition(sim, expected, drive_amplitude=10.0)
    assert not jnp.array_equal(s_expected_x.spikes, s_unexpected.spikes)


def test_evoked_l4_drive_semantics():
    """Evoked differs from baseline, baseline equals no-paradigm, windows anchor at onset."""
    model = _model()
    paradigm = jtfne.evoked_l4_drive_paradigm()
    assert paradigm.analysis_windows["baseline"] == (0.0, 200.0)
    assert paradigm.analysis_windows["evoked"] == (200.0, 400.0)
    assert paradigm.analysis_windows["post_evoked"] == (400.0, 900.0)

    custom = jtfne.evoked_l4_drive_paradigm(
        l4_onset_ms=300.0,
        l4_duration_ms=100.0,
        l4_amplitude=2.0,
        pre_stimulus_buffer_ms=100.0,
        post_stimulus_buffer_ms=150.0,
    )
    evoked = custom.condition("evoked")
    assert evoked.events[0].onset_ms == 200.0  # onset - pre buffer
    assert evoked.events[0].duration_ms == 100.0
    assert evoked.events[1].onset_ms == 300.0
    assert evoked.events[1].stimulus == "l4_evoked"
    assert evoked.events[1].metadata == {"drive_amplitude": 2.0, "target_layer": "L4"}
    assert evoked.events[2].onset_ms == 400.0
    assert custom.analysis_windows["baseline"] == (200.0, 300.0)
    assert custom.analysis_windows["evoked"] == (300.0, 400.0)
    assert custom.analysis_windows["post_evoked"] == (400.0, 550.0)

    sim = _sim(duration_ms=1000.0)
    s_none = model.simulate(sim, paradigm=None)
    s_base = model.simulate(sim, paradigm=paradigm.condition("baseline"))
    s_ev = model.simulate(sim, paradigm=paradigm.condition("evoked"))
    assert jnp.array_equal(s_base.spikes, s_none.spikes)
    assert jnp.array_equal(s_base.V_m, s_none.V_m)
    assert not jnp.array_equal(s_ev.spikes, s_base.spikes)

    with pytest.raises(ValueError):
        jtfne.evoked_l4_drive_paradigm(l4_onset_ms=100.0, pre_stimulus_buffer_ms=200.0)
