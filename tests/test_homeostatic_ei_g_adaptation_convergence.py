"""Milestone 2 of the homeostatic_ei canonical HDP sanity emitter: conductance
adaptation only (dG/dt = f_G(x, H), H frozen at its initial value).

Canonical parameters (drive=[0.5, 0.3], H0=[0.3, 0.3], activation_rule="cubic")
were chosen empirically before writing this test: with the "linear" activation
rule (Milestone 1's default), enabling G-adaptation with the "hebbian"
conductance rule diverges to NaN within a few thousand steps -- the cubic
rule's self-damping (-x^3 term) is required once G is allowed to grow. This
is a real, reproducible finding (verified with a direct eager run before this
test was written), not a hypothetical caveat, and is why this milestone's
canonical config differs from Milestone 1's.
"""
import jax
import jax.numpy as jnp

from jaxfne.emitters_homeostatic_ei import HomeostaticEIParams, simulate_homeostatic_ei

N_STEPS = 6000
DT_MS = 0.5
CONVERGENCE_WINDOW = 500
CONVERGENCE_TOL = 0.05


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


def test_hebbian_g_adaptation_converges_bounded_and_stable():
    params = _canonical_params()
    voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic", conductance_rule="hebbian", freeze_H=True,
    )
    assert not bool(diag["error"])
    assert bool(jnp.all(jnp.isfinite(G_hist))), "G_history must be finite throughout"
    assert bool(jnp.all(G_hist >= params.G_min - 1e-6)) and bool(jnp.all(G_hist <= params.G_max + 1e-6)), (
        "G_history must stay within [G_min, G_max]"
    )
    late_delta = float(jnp.max(jnp.abs(G_hist[-1] - G_hist[-CONVERGENCE_WINDOW])))
    assert late_delta < CONVERGENCE_TOL, (
        f"G did not converge: late-window max change {late_delta} exceeds tolerance {CONVERGENCE_TOL}"
    )
    # H must be genuinely unchanged when frozen.
    assert bool(jnp.allclose(H_hist[0], H_hist[-1]))


def test_g_adaptation_stays_finite_under_bcm_and_linear_rules():
    """Secondary coverage confirming rule-swappability doesn't break numerical
    stability for the two other registered conductance rules -- not asserting
    interior convergence (BCM saturates at G_max under these canonical
    parameters, a real observed property of unmodified BCM without an
    explicit normalization term, not a bug)."""
    params = _canonical_params()
    for rule in ("bcm", "linear"):
        voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
            params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
            activation_rule="cubic", conductance_rule=rule, freeze_H=True,
        )
        assert not bool(diag["error"]), f"conductance_rule={rule} produced non-finite output"
        assert bool(jnp.all(G_hist >= params.G_min - 1e-6)) and bool(jnp.all(G_hist <= params.G_max + 1e-6)), (
            f"conductance_rule={rule}: G left [G_min, G_max]"
        )
