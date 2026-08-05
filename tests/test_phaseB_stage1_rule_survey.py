"""Phase B — Stage 1: Rule survey under freeze_G=True.

Verifies the expected qualitative behavior of each homeostasis_rule when G
is frozen, using the deterministic isolation configuration (noise_scale=0,
extended horizon) established in Phase B Stage 0. This is the null-control
suite that documents which rules reach a genuine interior equilibrium,
which collapse to H_min, and which saturate at H_max.

Expected outcomes (measured 2026-08-04, N=2 canonical params):
  - linear:        saturates-at-H_max (I neuron -> 10.0, E -> 4.3; late_delta ~ 0.22)
  - logistic:      collapses-to-H_min (both neurons -> 0.1; late_delta ~ 0.0)
  - cubic_penalty: converges-to-interior (H -> [2.4, 2.9]; late_delta < 0.05)
  - cubic_penalty_coupled: converges-to-interior (H -> [2.5, 2.8]; late_delta < 0.05)

The linear rule under frozen singular G0 has no restoring term; with the
asymmetric drive, x < 1 so -(x-1)*H > 0 and H grows until H_max clip.
The logistic rule's bounded drain collapses H to the H_min floor.
Only the cubic rules have a two-sided restoring force that yields a genuine
interior equilibrium.
"""
import jax
import jax.numpy as jnp
import pytest

from jaxfne.emitters_homeostatic_ei import make_minimal_ei_params, simulate_homeostatic_ei

# Deterministic isolation params (match Phase B Stage 0 gate)
N_STEPS = 30000
DT_MS = 0.5
CONVERGENCE_WINDOW = 600
CONVERGENCE_TOL = 0.05
H_MIN = 0.1
H_MAX = 10.0
NOISE_SCALE = 0.0


def _make_params():
    return make_minimal_ei_params(n=2, x0_value=0.1, H0_value=0.3, drive_e=0.5, drive_i=0.3)


def _run(rule: str):
    params = _make_params()
    x_h, _, _, G_hist, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian",
        homeostasis_rule=rule, freeze_G=True, freeze_H=False,
        noise_scale=NOISE_SCALE,
    )
    return H_hist, diag


def test_linear_rule_saturates_at_H_max() -> None:
    """Linear rule: no restoring term -> H grows until H_max clip (not settled)."""
    H_hist, diag = _run("linear")
    assert not bool(diag["error"]), "Non-finite value detected."
    assert bool(jnp.all(jnp.isfinite(H_hist))), "H_hist non-finite."
    # I neuron saturates at H_MAX
    assert float(H_hist[-1, 1]) >= H_MAX - 0.05, (
        f"I neuron H did not saturate at H_max: {float(H_hist[-1, 1]):.3f}"
    )
    # Not settled (late_delta > tolerance)
    late_delta = float(jnp.max(jnp.abs(H_hist[-1] - H_hist[-CONVERGENCE_WINDOW])))
    assert late_delta >= CONVERGENCE_TOL, (
        f"Linear rule unexpectedly settled: late_delta={late_delta:.4f}"
    )


def test_logistic_rule_collapses_to_H_min() -> None:
    """Logistic rule: bounded drain -> collapses to H_min floor (settled)."""
    H_hist, diag = _run("logistic")
    assert not bool(diag["error"]), "Non-finite value detected."
    assert bool(jnp.all(jnp.isfinite(H_hist))), "H_hist non-finite."
    # Both neurons collapse to H_MIN
    assert bool(jnp.all(H_hist[-1] <= H_MIN + 0.05)), (
        f"H did not collapse to H_min: {H_hist[-1].tolist()}"
    )
    # Settled (late_delta ~ 0)
    late_delta = float(jnp.max(jnp.abs(H_hist[-1] - H_hist[-CONVERGENCE_WINDOW])))
    assert late_delta < CONVERGENCE_TOL, (
        f"Logistic rule not settled: late_delta={late_delta:.4f}"
    )


def test_cubic_penalty_rule_converges_to_interior() -> None:
    """Cubic penalty: genuine interior equilibrium (settled, not at bounds)."""
    H_hist, diag = _run("cubic_penalty")
    assert not bool(diag["error"]), "Non-finite value detected."
    assert bool(jnp.all(jnp.isfinite(H_hist))), "H_hist non-finite."
    assert bool(jnp.all(H_hist >= H_MIN - 1e-5)), f"H below H_min: {float(H_hist.min()):.6f}"
    assert bool(jnp.all(H_hist <= H_MAX + 1e-5)), f"H above H_max: {float(H_hist.max()):.6f}"
    # Settled
    late_delta = float(jnp.max(jnp.abs(H_hist[-1] - H_hist[-CONVERGENCE_WINDOW])))
    assert late_delta < CONVERGENCE_TOL, (
        f"Cubic penalty not settled: late_delta={late_delta:.4f}"
    )
    # Interior (not collapsed to H_min, not at H_max)
    assert bool(jnp.all(H_hist[-1] > H_MIN + 0.05)), f"H collapsed: {H_hist[-1].tolist()}"
    assert bool(jnp.all(H_hist[-1] < H_MAX - 0.05)), f"H saturated: {H_hist[-1].tolist()}"


def test_cubic_penalty_coupled_rule_converges_to_interior() -> None:
    """Cubic penalty coupled: genuine interior equilibrium (settled, not at bounds)."""
    H_hist, diag = _run("cubic_penalty_coupled")
    assert not bool(diag["error"]), "Non-finite value detected."
    assert bool(jnp.all(jnp.isfinite(H_hist))), "H_hist non-finite."
    assert bool(jnp.all(H_hist >= H_MIN - 1e-5)), f"H below H_min: {float(H_hist.min()):.6f}"
    assert bool(jnp.all(H_hist <= H_MAX + 1e-5)), f"H above H_max: {float(H_hist.max()):.6f}"
    # Settled
    late_delta = float(jnp.max(jnp.abs(H_hist[-1] - H_hist[-CONVERGENCE_WINDOW])))
    assert late_delta < CONVERGENCE_TOL, (
        f"Cubic penalty coupled not settled: late_delta={late_delta:.4f}"
    )
    # Interior
    assert bool(jnp.all(H_hist[-1] > H_MIN + 0.05)), f"H collapsed: {H_hist[-1].tolist()}"
    assert bool(jnp.all(H_hist[-1] < H_MAX - 0.05)), f"H saturated: {H_hist[-1].tolist()}"
