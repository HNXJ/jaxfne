"""Small, fast tests for the cubic_penalty homeostasis_rule (jaxfne/
emitters_homeostatic_ei.py): dH = -dK * (H - baseline)^3 / (H + baseline).

Motivation: the existing "linear"/"logistic" homeostasis rules are pure
x-driven rate-drain -- nothing pulls H back UP once it starts falling, so at
N=8 (verified this session) H monotonically drains to its H_min clip floor
and gets stuck there (bounded, but not a genuine interior attractor -- the
same failure mode documented for the Izhikevich/edge-list HDP kernel's
rho_passive/H^2 formula in skills/FRICTIONS_STACK.md's F-017). cubic_penalty
is a genuine two-sided restoring force: zero at H=baseline, grows cubically
with deviation in either direction.

Scope: small N only (2 and 8 neurons), matching this repo's "no large
networks in tests" direction -- complexity/scale questions belong to a
separate doubling-ladder check, not here; this file verifies the local
attractor mechanics are sound, nothing about macro/population-scale
behavior.
"""
import jax
import jax.numpy as jnp

from jaxfne.emitters_homeostatic_ei import HomeostaticEIParams, simulate_homeostatic_ei, HOMEOSTASIS_RULES

N_STEPS = int(6000 / 0.5)
DT_MS = 0.5


def _make_params(n: int, G_max: float | None = None):
    """G_max defaults to 10.0/n, not a flat constant. G0 itself is already
    column-normalized (each E/I column sums to +-0.5 regardless of n), but
    Hebbian adaptation clips each of the n entries in a row independently
    toward the same G_max ceiling -- so the aggregate per-row recurrent
    feedback `sum_j G_ij * x_j` can reach n * G_max in the worst case, which
    grows with n even though G0's own aggregate doesn't. Holding n * G_max
    constant (=10.0, matching the n=2 canonical default's 2*5.0=10.0) keeps
    that aggregate ceiling comparable across n. Verified empirically this
    session: n=8 -> G_max=10.0/8=1.25 keeps the cubic_penalty run finite,
    same order of magnitude as the earlier ad hoc G_max=1.0 finding."""
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


def test_h_reaches_interior_equilibrium_not_floor_at_n2():
    params = _make_params(2)
    _, _, _, _, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="cubic_penalty",
    )
    assert not bool(diag["error"])
    H_final = H_hist[-1]
    assert bool(jnp.all(H_final > float(params.H_min) + 0.05)), (
        f"H must not collapse to the H_min floor, got {H_final}"
    )
    assert bool(jnp.all(H_final < float(params.H_max) - 0.05))
    late_delta = float(jnp.max(jnp.abs(H_hist[-1] - H_hist[-500])))
    assert late_delta < 0.01, f"H should have converged by the end of the run, delta={late_delta}"


def test_h_reaches_interior_equilibrium_at_n8_with_tightened_g_bounds():
    """N=8 requires a tighter G_max than the N=2 canonical default (5.0) to
    stay finite under cubic_penalty -- verified empirically this session,
    not assumed. `_make_params`'s default G_max=10.0/n (1.25 at n=8) is a
    principled derivation (aggregate per-row feedback ceiling n*G_max held
    constant across n), not a change to the shipped construct.py default,
    which stays a flat 5.0 for the hebbian/linear combination it actually
    ships with."""
    params = _make_params(8)
    voltages, _, _, _, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="cubic_penalty",
    )
    assert not bool(diag["error"])
    assert bool(jnp.all(jnp.isfinite(voltages)))
    H_final = H_hist[-1]
    assert bool(jnp.all(H_final > float(params.H_min) + 0.05)), (
        f"H must not collapse to the H_min floor, got {H_final}"
    )


def test_h_final_state_is_a_genuine_fixed_point_of_the_rule():
    """Direct fixed-point check: evaluate the actual homeostasis_rule at the
    late-window (x, H) and assert its derivative is near zero -- a
    principled test of "this is a real equilibrium of the vector field",
    independent of H_min/H_max or an ad hoc epsilon-from-the-clip-boundary
    heuristic (the prior proxy used by the two tests above). Verified
    empirically: converged late-window max|dH| is ~0.01 (n=2) / ~0.002 (n=8),
    vs. ~0.2-0.27 during the early/mid transient -- the 0.03 tolerance below
    has real discriminating power, not just headroom padding."""
    for n in (2, 8):
        params = _make_params(n)
        voltages, _, _, _, H_hist, diag = simulate_homeostatic_ei(
            params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
            activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="cubic_penalty",
        )
        assert not bool(diag["error"])
        x_late = voltages[-500:].mean(axis=0)
        H_late = H_hist[-500:].mean(axis=0)
        dH = HOMEOSTASIS_RULES["cubic_penalty"](x_late, H_late)
        max_dH = float(jnp.max(jnp.abs(dH)))
        assert max_dH < 0.03, (
            f"n={n}: late-window (x, H) is not a fixed point of cubic_penalty "
            f"-- max|dH|={max_dH}, x_late={x_late}, H_late={H_late}"
        )


def test_cubic_penalty_recovers_toward_baseline_after_perturbation():
    """A transient H perturbation (via a large drive pulse, injected late in
    the run) must relax back toward the SAME equilibrium the unperturbed run
    reaches -- the actual "attraction point" property. Compares both runs'
    LATE windows at identical elapsed time (not an early-vs-late window
    within one run) -- the cubic restoring term has a long convergence tail
    near equilibrium (its own strength shrinks as H approaches baseline), so
    an early window is not yet settled even without any perturbation and is
    not a fair comparison point."""
    params = _make_params(2)
    baseline_sched = jnp.zeros((N_STEPS, 2))
    _, _, _, _, H_baseline, diag_b = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="cubic_penalty",
        drive_schedule=baseline_sched,
    )
    assert not bool(diag_b["error"])

    pulse_start, pulse_end = N_STEPS // 2, N_STEPS // 2 + 200
    perturbed_sched = baseline_sched.at[pulse_start:pulse_end, :].set(5.0)
    _, _, _, _, H_perturbed, diag_p = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="cubic_penalty",
        drive_schedule=perturbed_sched,
    )
    assert not bool(diag_p["error"])
    assert bool(jnp.all(jnp.isfinite(H_perturbed)))

    # The pulse must actually perturb H away from the unperturbed trajectory.
    during_gap = jnp.abs(H_perturbed[pulse_start + 100] - H_baseline[pulse_start + 100])
    assert bool(jnp.any(during_gap > 0.01)), f"pulse produced no measurable H excursion: gap={during_gap}"

    baseline_H = H_baseline[-500:].mean(axis=0)
    final_H = H_perturbed[-500:].mean(axis=0)
    recovery_gap = jnp.abs(final_H - baseline_H)
    assert bool(jnp.all(recovery_gap < 0.05)), (
        f"H did not recover to the same equilibrium after perturbation: "
        f"baseline={baseline_H}, final={final_H}, gap={recovery_gap}"
    )
