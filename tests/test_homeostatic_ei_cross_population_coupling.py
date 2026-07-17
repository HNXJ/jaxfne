"""Small, fast tests for the cubic_penalty_coupled homeostasis_rule (jaxfne/
emitters_homeostatic_ei.py): adds an explicit E<->I cross-population coupling
term on top of cubic_penalty.

Motivation (reasoned through with the user, 2026-07-16, "the HDP problem of
two groups"): every other homeostasis_rule (linear/logistic/cubic_penalty)
gives each neuron's dH a term that depends ONLY on that neuron's own x -- the
reduced long-term system in (H_e, H_i) (after adiabatically eliminating the
fast x/intermediate G timescales) has a DIAGONAL Jacobian: stable only
because each population is independently damped, not because of any real
E/I interaction. cubic_penalty_coupled adds the missing off-diagonal term.

Scope: small N only (2 and 8 neurons), matching this repo's "no large
networks in tests" direction. Test A below is the sharpest, cheapest check --
a direct function-level sensitivity comparison, no simulation needed at all.
"""
import jax
import jax.numpy as jnp

from jaxfne.emitters_homeostatic_ei import (
    HomeostaticEIParams,
    simulate_homeostatic_ei,
    HOMEOSTASIS_RULES,
)

N_STEPS = int(6000 / 0.5)
DT_MS = 0.5


def _make_params(n: int, G_max: float | None = None):
    n_e = 1 if n <= 2 else max(1, round(n * 0.75))
    n_i = n - n_e
    is_e = jnp.array([True] * n_e + [False] * n_i)
    drive = jnp.where(is_e, 0.5, 0.3)
    col = jnp.where(is_e, 0.5 / n_e, -0.5 / max(n_i, 1))
    G0 = jnp.broadcast_to(col[None, :], (n, n))
    if G_max is None:
        G_max = 10.0 / n
    return HomeostaticEIParams(
        x0=jnp.full((n,), 0.1), G0=G0, H0=jnp.full((n,), 0.3), drive=drive,
        tau_x_ms=jnp.array(5.0), tau_G_ms=jnp.array(200.0), tau_H_ms=jnp.array(1000.0),
        G_min=jnp.array(-G_max), G_max=jnp.array(G_max), H_min=jnp.array(0.1), H_max=jnp.array(10.0),
        source_scale=jnp.full((n,), 1.0),
    )


def test_coupled_rule_makes_i_dH_sensitive_to_e_activity_uncoupled_does_not():
    """Direct rule-level sensitivity check, no simulation: holding H and I's
    own x fixed, change ONLY E's x and confirm I's dH changes under
    cubic_penalty_coupled but NOT under plain cubic_penalty -- the sharpest,
    cheapest possible test of "does the cross term exist"."""
    is_e = jnp.array([True, False])  # neuron 0 = E, neuron 1 = I
    H = jnp.array([1.0, 1.0])
    x_e_low = jnp.array([0.5, 1.0])   # E under-active
    x_e_high = jnp.array([2.5, 1.0])  # E over-active, I's own x unchanged

    dH_coupled_low = HOMEOSTASIS_RULES["cubic_penalty_coupled"](x_e_low, H, is_e)
    dH_coupled_high = HOMEOSTASIS_RULES["cubic_penalty_coupled"](x_e_high, H, is_e)
    i_gap_coupled = float(jnp.abs(dH_coupled_high[1] - dH_coupled_low[1]))
    assert i_gap_coupled > 0.05, (
        f"cubic_penalty_coupled: I's dH should change when only E's x changes, "
        f"got gap={i_gap_coupled}"
    )

    dH_plain_low = HOMEOSTASIS_RULES["cubic_penalty"](x_e_low, H)
    dH_plain_high = HOMEOSTASIS_RULES["cubic_penalty"](x_e_high, H)
    i_gap_plain = float(jnp.abs(dH_plain_high[1] - dH_plain_low[1]))
    assert i_gap_plain < 1e-9, (
        f"cubic_penalty (uncoupled): I's dH must NOT depend on E's x, got gap={i_gap_plain}"
    )


def test_coupled_rule_e_only_perturbation_moves_i_h_in_a_real_run():
    """Full-simulation contrast: a drive pulse targeted at the E neuron only
    must produce a measurable I-side H excursion under cubic_penalty_coupled
    -- the property cubic_penalty (uncoupled) structurally cannot have.

    Checked well after the 100ms pulse ends (500ms post-pulse-start), not
    during it -- H is the slow variable (tau_H_ms=1000ms), so the coupling
    term's effect on I's H accumulates over hundreds of ms, not tens.
    Verified empirically: I-side gap at pulse_start+100 steps (50ms in) is
    only ~0.0025 (still building), but reaches ~0.009-0.014 by
    pulse_start+1000..2000 steps (500ms-1000ms in) -- checking too early
    would read as "no coupling" when the effect is simply still developing."""
    params = _make_params(2)
    baseline_sched = jnp.zeros((N_STEPS, 2))
    pulse_start, pulse_end = N_STEPS // 2, N_STEPS // 2 + 200
    e_only_sched = baseline_sched.at[pulse_start:pulse_end, 0].set(5.0)  # neuron 0 = E only

    _, _, _, _, H_base, diag_b = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="cubic_penalty_coupled",
        drive_schedule=baseline_sched,
    )
    _, _, _, _, H_pert, diag_p = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="cubic_penalty_coupled",
        drive_schedule=e_only_sched,
    )
    assert not bool(diag_b["error"]) and not bool(diag_p["error"])
    assert bool(jnp.all(jnp.isfinite(H_pert)))

    i_gap_late = float(jnp.abs(H_pert[pulse_start + 1000, 1] - H_base[pulse_start + 1000, 1]))
    assert i_gap_late > 0.005, (
        f"E-only perturbation produced no measurable I-side H excursion under "
        f"cubic_penalty_coupled 500ms after the pulse: gap={i_gap_late}"
    )


def test_coupled_rule_still_reaches_interior_equilibrium_not_floor():
    """The added cross term must not break cubic_penalty's core two-sided
    restoring property -- same "not pinned at H_min" check as
    test_homeostatic_ei_cubic_penalty_rule.py, run against the coupled rule."""
    params = _make_params(2)
    _, _, _, _, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="cubic_penalty_coupled",
    )
    assert not bool(diag["error"])
    H_final = H_hist[-1]
    assert bool(jnp.all(H_final > float(params.H_min) + 0.05)), (
        f"H must not collapse to the H_min floor, got {H_final}"
    )
    assert bool(jnp.all(H_final < float(params.H_max) - 0.05))
