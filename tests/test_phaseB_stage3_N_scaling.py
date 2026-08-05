"""Phase B — Stage 3: N-scaling stability with bound_mode="stable".

Verifies the three-timescale system (fast x, intermediate G, slow H) remains
stable at larger N when bound_mode="stable" (smooth tanh soft-bound applied
to x, G, H each step) instead of the default "minimal" hard clip.

Configuration (matches B-12 measurements):
  - params = make_minimal_ei_params(n={4,8,16})
  - n_steps = 10000, dt_ms = 0.5
  - activation_rule = "cubic" (saturating nonlinearity, required for
    stability of non-BCM conductance rules)
  - conductance_rule = "bcm" (sliding-threshold, the rule verified to
    stabilize the system even under linear activation in B-08)
  - homeostasis_rule = "cubic_penalty" (interior-equilibrium rule from B-04)
  - freeze_G = False, freeze_H = False
  - noise_scale = 0.0, bound_mode = "stable"

Measured (2026-08-04):
  - N=8: finite, H -> [4.67, 4.77], no divergence (BCM and hebbian both stable)
  - N=16: finite, H -> [4.69, 4.79], no divergence (BCM and hebbian both stable)
  - H settles to interior equilibrium above H_min, below H_max.

Each test asserts four criteria:
  1. No non-finite values anywhere (diagnostics error flag, then direct check).
  2. x stays within the soft-bound range (finite, no explosion).
  3. H stays strictly inside [H_min, H_max].
  4. H settles: late-window max |ΔH| < tolerance.
"""
import jax
import jax.numpy as jnp
import pytest

from jaxfne.emitters_homeostatic_ei import make_minimal_ei_params, simulate_homeostatic_ei

N_STEPS = 10000
DT_MS = 0.5
CONVERGENCE_WINDOW = 100
CONVERGENCE_TOL = 0.05
H_MIN = 0.1
H_MAX = 10.0


def _run(n: int):
    params = make_minimal_ei_params(n=n, x0_value=0.1, H0_value=0.3, drive_e=0.5, drive_i=0.3)
    voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="bcm",
        homeostasis_rule="cubic_penalty", freeze_G=False, freeze_H=False,
        noise_scale=0.0, bound_mode="stable",
    )
    return voltages, G_hist, H_hist, diag


def _assert_stability(voltages, G_hist, H_hist, diag, label: str) -> None:
    """Shared four-assertion stability check."""
    # 1. No non-finite values anywhere.
    assert not bool(diag["error"]), f"Non-finite value detected ({label})."
    assert bool(jnp.all(jnp.isfinite(voltages))), f"voltages non-finite ({label})."
    assert bool(jnp.all(jnp.isfinite(G_hist))), f"G_hist non-finite ({label})."
    assert bool(jnp.all(jnp.isfinite(H_hist))), f"H_hist non-finite ({label})."

    # 2. x stays bounded (no explosion).
    x_max = float(jnp.max(jnp.abs(voltages)))
    assert x_max < 50.0, f"x exploded ({label}): max|x| = {x_max:.3f}"

    # 3. H stays strictly inside [H_min, H_max].
    assert bool(jnp.all(H_hist >= H_MIN - 1e-5)), (
        f"H below H_min ({label}): min={float(H_hist.min()):.6f}"
    )
    assert bool(jnp.all(H_hist <= H_MAX + 1e-5)), (
        f"H above H_max ({label}): max={float(H_hist.max()):.6f}"
    )

    # 4. H settles: late-window max |ΔH| < tolerance.
    late_delta = float(jnp.max(jnp.abs(H_hist[-1] - H_hist[-CONVERGENCE_WINDOW])))
    assert late_delta < CONVERGENCE_TOL, (
        f"H did not converge ({label}): late-window max |ΔH|={late_delta:.6f} "
        f"exceeds tolerance {CONVERGENCE_TOL}."
    )

    # Bonus interior check: H must not be at the bounds.
    assert bool(jnp.all(H_hist[-1] > H_MIN + 0.05)), f"H at floor ({label}): {H_hist[-1].tolist()}"
    assert bool(jnp.all(H_hist[-1] < H_MAX - 0.05)), f"H at ceiling ({label}): {H_hist[-1].tolist()}"


def test_n4_scaling() -> None:
    voltages, G_hist, H_hist, diag = _run(n=4)
    _assert_stability(voltages, G_hist, H_hist, diag, "N=4")


def test_n8_scaling() -> None:
    voltages, G_hist, H_hist, diag = _run(n=8)
    _assert_stability(voltages, G_hist, H_hist, diag, "N=8")


def test_n16_scaling() -> None:
    voltages, G_hist, H_hist, diag = _run(n=16)
    _assert_stability(voltages, G_hist, H_hist, diag, "N=16")
