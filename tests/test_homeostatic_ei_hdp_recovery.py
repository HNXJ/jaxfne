"""Milestone 3 of the homeostatic_ei canonical HDP sanity emitter: full HDP
enabled (dG/dt and dH/dt both active). Verifies recovery after a transient
perturbation, stable equilibrium, and bounded H dynamics -- the canonical
2-neuron analogue of the H-recovery checks already run for the Izhikevich/
edge-list HDP scaffold (see tests/test_ei_hdp_hebbian_angle.py,
tests/test_ei_hdp_stability_rho_passive_sweep.py).
"""
import jax
import jax.numpy as jnp

from jaxfne.emitters_homeostatic_ei import HomeostaticEIParams, simulate_homeostatic_ei

N_STEPS = 8000
DT_MS = 0.5
PULSE_START = 2000
PULSE_END = 2200
PULSE_AMPLITUDE = 5.0
BASELINE_WINDOW = (1900, 2000)
DURING_WINDOW = (2050, 2150)
RECOVERY_TAIL = 500


def _canonical_params():
    return HomeostaticEIParams(
        x0=jnp.array([0.1, 0.1]),
        G0=jnp.array([[0.5, -0.5], [0.5, -0.5]]),
        H0=jnp.array([0.3, 0.3]),
        drive=jnp.array([0.5, 0.3]),
        tau_x_ms=jnp.array(5.0),
        tau_G_ms=jnp.array(200.0),
        tau_H_ms=jnp.array(1000.0),
        G_min=jnp.array(-5.0),
        G_max=jnp.array(5.0),
        H_min=jnp.array(0.1),
        H_max=jnp.array(10.0),
        source_scale=jnp.array([1.0, 1.0]),
    )


def test_hdp_full_system_stays_finite_and_h_bounded():
    params = _canonical_params()
    voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="linear",
    )
    assert not bool(diag["error"])
    assert bool(jnp.all(jnp.isfinite(H_hist)))
    assert bool(jnp.all(H_hist >= params.H_min - 1e-6)) and bool(jnp.all(H_hist <= params.H_max + 1e-6)), (
        "H_history must stay within [H_min, H_max] across the whole run"
    )


def test_hdp_recovers_after_transient_perturbation():
    params = _canonical_params()
    baseline_schedule = jnp.zeros((N_STEPS, 2))
    voltages_b, _, _, _, H_baseline, diag_b = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="linear",
        drive_schedule=baseline_schedule,
    )
    assert not bool(diag_b["error"])

    perturbed_schedule = baseline_schedule.at[PULSE_START:PULSE_END, :].set(PULSE_AMPLITUDE)
    voltages_p, _, _, _, H_perturbed, diag_p = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="linear",
        drive_schedule=perturbed_schedule,
    )
    assert not bool(diag_p["error"])
    assert bool(jnp.all(jnp.isfinite(H_perturbed))), "perturbed run must stay finite"
    assert bool(jnp.all(H_perturbed >= params.H_min - 1e-6)) and bool(jnp.all(H_perturbed <= params.H_max + 1e-6))

    baseline_H = H_baseline[BASELINE_WINDOW[0]:BASELINE_WINDOW[1]].mean(axis=0)
    during_H = H_perturbed[DURING_WINDOW[0]:DURING_WINDOW[1]].mean(axis=0)
    final_H = H_perturbed[-RECOVERY_TAIL:].mean(axis=0)

    # The perturbation must actually perturb H away from baseline during the pulse window.
    perturbation_gap = jnp.abs(during_H - baseline_H)
    assert bool(jnp.any(perturbation_gap > 0.005)), (
        f"perturbation produced no measurable H excursion: gap={perturbation_gap}"
    )
    # H must recover close to its pre-perturbation baseline by the end of the run.
    recovery_gap = jnp.abs(final_H - baseline_H)
    assert bool(jnp.all(recovery_gap < 0.08)), (
        f"H did not recover toward baseline: baseline={baseline_H}, final={final_H}, gap={recovery_gap}"
    )

    baseline_x_equilibrium = voltages_b[-RECOVERY_TAIL:].mean(axis=0)
    perturbed_x_equilibrium = voltages_p[-RECOVERY_TAIL:].mean(axis=0)
    x_recovery_gap = jnp.abs(perturbed_x_equilibrium - baseline_x_equilibrium)
    assert bool(jnp.all(x_recovery_gap < 0.3)), (
        f"post-perturbation x equilibrium did not match pre-perturbation baseline: "
        f"baseline={baseline_x_equilibrium}, perturbed={perturbed_x_equilibrium}"
    )
