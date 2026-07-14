"""Regression gate for scripts/sphere20_ei_hdp_habituation.py -- the seed of
the HDP long-term habituation suite (see artifacts/developer/plans.json:
hdp-habituation-suite-sphere20). Pins the verified-stable 2026-07-14 config
(DEFAULT_HDP, E_TO_PV_GAIN=10.0) so future evolution of the suite (larger
networks, richer paradigms, rate-dependent-H tuning) can't silently regress
this baseline back to double-wired connectivity, a dead HDP runtime, reset-
every-turn state, or a runaway/silent network.

Deliberately a smoke/bounds test (finite, right shape, sane ranges), not a
value-accuracy test -- exact float reproducibility across JAX/XLA versions
isn't the contract here; a real behavioral regression (NaN, PV silenced for
the whole run, double-wired edges, rates blowing up) is.
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


def test_baseline_simulate_is_finite_and_hdp_active():
    model = suite.build_model()
    stimulus, driven = suite.build_pulse_train(model, seed=suite.SEED)
    runtime_cfg = suite.jtfne.RuntimeConfig(enable_hdp=True, hdp_params=suite.HDP_PARAMS)
    sim = suite.jtfne.simulation(
        duration_ms=suite.TURN_DURATION_MS, dt_ms=suite.DT_MS, seed=suite.SEED, runtime=runtime_cfg
    )
    sig = model.simulate(sim, paradigm=stimulus)
    assert bool(np.all(np.isfinite(np.asarray(sig.V_m))))
    assert bool(np.all(np.isfinite(np.asarray(sig.spikes))))
    assert len(driven) == max(1, round(suite.N_E * suite.DRIVEN_FRACTION))


def test_continuous_task_full_chain_stays_finite_and_bounded():
    """Runs the actual suite.run() (all N_TURNS turns, real HDP scan chain) and
    checks every turn's observables for the invariants that a real regression
    would break: finiteness, both populations still able to fire (not
    permanently silenced), and rates staying in a physiologically-sane band
    rather than the runaway regime found with DEFAULT_HDP_DESYNC (80-190 Hz,
    H/weights pinned at their floor/ceiling)."""
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
    assert all(5.0 <= r <= 20.0 for r in e_rates)
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
        stimulus, _ = suite.build_pulse_train(model, seed=suite.SEED)
        n_steps = int(round(suite.TURN_DURATION_MS / suite.DT_MS))
        drive = jnp.asarray(stimulus.to_array(n_steps, suite.DT_MS))

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
