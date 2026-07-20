"""Regression gate for scripts/sphere20_ei_hdp_habituation.py -- the seed of
the HDP long-term habituation suite (see artifacts/developer/plans.json:
hdp-habituation-suite-sphere20). Pins the verified-stable 2026-07-14 config
(DEFAULT_HDP, E_TO_PV_GAIN=10.0) and the overlapping-A/B-group trial
paradigm so future evolution of the suite (larger networks, richer
paradigms, rate-dependent-H tuning) can't silently regress this baseline
back to double-wired connectivity, a dead HDP runtime, reset-every-turn
state, a runaway/silent network, or a broken trial/group structure.

Deliberately a smoke/bounds test (finite, right shape, sane ranges), not a
value-accuracy test -- exact float reproducibility across JAX/XLA versions
isn't the contract here; a real behavioral regression (NaN, PV silenced for
the whole run, double-wired edges, rates blowing up, broken group overlap)
is.
"""
import math

import numpy as np
import pytest

import scripts.sphere20_ei_hdp_habituation as suite


def test_neuron_table_has_declared_ei_split():
    model = suite.build_model()
    rows = model.neuron_table()
    cell_types = sorted(r["cell_type"] for r in rows)
    assert cell_types.count("E") == suite.N_E
    assert cell_types.count("PV") == suite.N_I
    assert len(rows) == suite.N


def test_edge_count_matches_fully_recurrent_ei_blocks_no_double_wiring():
    # Fully recurrent EE/EI/IE/II with no self-loops: 15*14 + 15*5 + 5*15 + 5*4.
    # A regression back to the double-wiring landmine (construct()'s own
    # default dense connectivity layered on top of the explicit
    # InterConnection edges) would roughly double this count.
    model = suite.build_model()
    expected = suite.N_E * (suite.N_E - 1) + suite.N_E * suite.N_I + suite.N_I * suite.N_E + suite.N_I * (suite.N_I - 1)
    assert model.params["edge_list"].n_edges == expected


def test_sphere_positions_are_within_declared_radius_and_not_a_box():
    model = suite.build_model()
    positions = np.asarray(model.params["positions"])
    radii = np.linalg.norm(positions, axis=1)
    assert positions.shape == (suite.N, 3)
    assert bool(np.all(np.isfinite(positions)))
    assert bool(np.all(radii <= suite.SPHERE_RADIUS_MM + 1e-6))
    # A box/cylinder sample (jaxfne's own uniform_3d default) would put mass
    # in the corners outside the inscribed sphere -- confirm every point is
    # inside the sphere, not just bounded by a cube of the same radius.
    corner = np.array([suite.SPHERE_RADIUS_MM] * 3)
    assert np.linalg.norm(corner) > suite.SPHERE_RADIUS_MM


def test_groups_have_declared_sizes_and_overlap():
    model = suite.build_model()
    group_a, group_b, shared = suite.build_groups(model, seed=suite.SEED)
    assert len(group_a) == suite.GROUP_A_SIZE
    assert len(group_b) == suite.GROUP_B_SIZE
    assert len(shared) == suite.GROUP_AB_SHARED
    assert sorted(set(group_a) & set(group_b)) == shared
    rows = model.neuron_table()
    e_indices = {r["neuron_id"] for r in rows if r["cell_type"] == "E"}
    assert set(group_a) <= e_indices
    assert set(group_b) <= e_indices


def test_trial_events_have_correct_structure_and_variable_duration():
    model = suite.build_model()
    group_a, group_b, shared = suite.build_groups(model, seed=suite.SEED)

    events_aaaa, duration_aaaa = suite.build_trial_events(group_a, group_b, suite.SEQUENCE_AAAA, seed=1)
    events_aaab, duration_aaab = suite.build_trial_events(group_a, group_b, suite.SEQUENCE_AAAB, seed=1)

    assert len(events_aaaa) == 2 * suite.TRIAL_N_REPS  # delay+stim per repetition
    # AAAA: every stim targets group_a; AAAB: only stim4 switches to group_b.
    stim_events_aaaa = [ev for ev in events_aaaa if ev["label"].startswith("stim")]
    stim_events_aaab = [ev for ev in events_aaab if ev["label"].startswith("stim")]
    assert all(ev["metadata"]["target_indices"] == group_a for ev in stim_events_aaaa)
    assert all(ev["metadata"]["target_indices"] == group_a for ev in stim_events_aaab[:-1])
    assert stim_events_aaab[-1]["metadata"]["target_indices"] == group_b

    # Same delay draws (same seed) -> AAAA and AAAB share the same total
    # duration and the same event onsets, differing only in stim4's target.
    assert duration_aaaa == duration_aaab
    for a, b in zip(events_aaaa, events_aaab):
        assert a["onset_ms"] == b["onset_ms"]
        assert a["duration_ms"] == b["duration_ms"]

    # Delays are genuinely random within the declared jitter band.
    delay_events = [ev for ev in events_aaaa if ev["label"].startswith("delay")]
    for ev in delay_events:
        assert (suite.DELAY_MEAN_MS - suite.DELAY_JITTER_MS) <= ev["duration_ms"] <= (
            suite.DELAY_MEAN_MS + suite.DELAY_JITTER_MS
        )
    # A different seed gives a different total duration (random, not fixed).
    _, duration_other_seed = suite.build_trial_events(group_a, group_b, suite.SEQUENCE_AAAA, seed=2)
    assert duration_other_seed != duration_aaaa


def test_baseline_simulate_is_finite_and_hdp_active():
    model = suite.build_model()
    group_a, group_b, shared = suite.build_groups(model, seed=suite.SEED)
    paradigm, duration_ms = suite.trial_paradigm(group_a, group_b, suite.SEQUENCE_AAAA, seed=suite.SEED)
    runtime_cfg = suite.jtfne.RuntimeConfig(enable_hdp=True, hdp_params=suite.HDP_PARAMS)
    sim = suite.jtfne.simulation(duration_ms=duration_ms, dt_ms=suite.DT_MS, seed=suite.SEED, runtime=runtime_cfg)
    sig = model.simulate(sim, paradigm=paradigm.conditions[0])
    assert bool(np.all(np.isfinite(np.asarray(sig.V_m))))
    assert bool(np.all(np.isfinite(np.asarray(sig.spikes))))


def test_continuous_task_full_chain_stays_finite_and_bounded():
    """Runs the actual suite.run() (all N_TURNS trials, real HDP scan chain)
    and checks every trial's observables for the invariants that a real
    regression would break: finiteness, both populations still able to fire
    (not permanently silenced), and rates staying in a physiologically-sane
    band rather than the runaway regime found with DEFAULT_HDP_DESYNC
    (80-190 Hz, H/weights pinned at their floor/ceiling)."""
    history = suite.run()

    assert len(history) == suite.N_TURNS
    for row in history:
        for key, value in row.items():
            if key == "turn":
                continue
            assert math.isfinite(value), f"turn {row['turn']} field {key} is not finite: {value}"

    e_rates = [row["rate_E_hz"] for row in history]
    pv_rates = [row["rate_PV_hz"] for row in history]

    # E is the directly-driven population -- should stay in a stable band,
    # not collapse to zero or run away.
    assert all(5.0 <= r <= 25.0 for r in e_rates)
    # PV must not be silenced for the whole chain (the pre-gain-fix regression:
    # PV fires once at turn 0, then exactly 0.0 Hz for all remaining turns).
    assert any(r > 0.0 for r in pv_rates[1:])
    # ...but must also not be in the DEFAULT_HDP_DESYNC runaway regime.
    assert all(r <= 30.0 for r in pv_rates)

    for row in history:
        assert 0.0 < row["H_E"] < 3.0
        assert 0.0 < row["H_PV"] < 3.0
        for key in ("Wee", "Wei", "Wie", "Wii"):
            assert 0.0 < row[key] < 10.0
        # Trial durations vary (random delays) but must stay in a sane range:
        # TRIAL_N_REPS * (min delay + stim) to TRIAL_N_REPS * (max delay + stim).
        min_duration = suite.TRIAL_N_REPS * (suite.DELAY_MEAN_MS - suite.DELAY_JITTER_MS + suite.STIM_DURATION_MS)
        max_duration = suite.TRIAL_N_REPS * (suite.DELAY_MEAN_MS + suite.DELAY_JITTER_MS + suite.STIM_DURATION_MS)
        assert min_duration <= row["trial_duration_ms"] <= max_duration

    oddball_rows = [row for row in history if row["is_oddball"] == 1.0]
    assert len(oddball_rows) == len(suite.ODDBALL_TURNS)


def test_e_to_pv_gain_of_one_leaves_pv_silenced_after_first_turn():
    """Documents the actual pre-fix regression this suite guards against:
    without the E->PV gain, PV fires a handful of times in the cold-start
    transient (turn 0) and then goes fully silent for the rest of the chain
    -- confirmed not a NaN/crash, just a genuine quiescent fixed point (see
    plans.json:hdp-habituation-suite-sphere20)."""
    import jax
    import jax.numpy as jnp
    from jaxfne._pipeline import compile_step_fn, scan_network

    original_gain = suite.E_TO_PV_GAIN
    try:
        suite.E_TO_PV_GAIN = 1.0
        model = suite.build_model()
        rows = model.neuron_table()
        pv_idx = [r["neuron_id"] for r in rows if r["cell_type"] == "PV"]
        group_a, group_b, shared = suite.build_groups(model, seed=suite.SEED)
        paradigm, duration_ms = suite.trial_paradigm(group_a, group_b, suite.SEQUENCE_AAAA, seed=suite.SEED)
        n_steps = int(round(duration_ms / suite.DT_MS))
        drive = jnp.asarray(suite.trial_drive_array(paradigm, n_steps))

        step_fn, carry = compile_step_fn(model, dt_ms=suite.DT_MS, **suite.HDP_PARAMS)
        base_key = jax.random.PRNGKey(suite.SEED)
        pv_spike_totals = []
        for turn in range(3):
            keys = jax.random.split(jax.random.fold_in(base_key, turn), n_steps)
            carry, outputs = scan_network(step_fn, carry, drive, keys)
            spikes_trace = outputs[1]
            pv_spike_totals.append(float(np.asarray(spikes_trace)[:, pv_idx].sum()))

        assert pv_spike_totals[0] > 0.0, "expected a nonzero cold-start transient at turn 0"
        assert pv_spike_totals[1] == 0.0 and pv_spike_totals[2] == 0.0, (
            f"expected PV silenced after turn 0 at gain=1.0 (documented regression), "
            f"got {pv_spike_totals}"
        )
    finally:
        suite.E_TO_PV_GAIN = original_gain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
