"""Small, fast dt-invariance check for simulate_homeostatic_ei's noise term
(jaxfne/emitters_homeostatic_ei.py). Guards the Euler-Maruyama sqrt(dt_x)
scaling fix: noise enters additively into `u`, which is itself multiplied by
dt_x inside every activation rule -- without the sqrt(dt_x) correction, the
net stochastic contribution to x_next scales as dt_x rather than sqrt(dt_x),
making the simulated process's stationary statistics an artifact of whatever
dt_ms was chosen rather than a property of the dynamical system. This test
runs the same N=2 canonical-shaped circuit at three step sizes (a ladder, not
production scale -- consistent with this repo's "no large networks in tests"
direction, applied here to dt instead of N) and checks the late-window H
equilibrium stays close across all three, seed-averaged to reduce single-seed
sampling noise.
"""
import jax
import jax.numpy as jnp

from jaxfne.emitters_homeostatic_ei import HomeostaticEIParams, simulate_homeostatic_ei

DURATION_MS = 6000.0
DT_LADDER_MS = (1.0, 0.5, 0.25)
SEEDS = (0, 1, 2)
DT_INVARIANCE_TOL = 0.15  # generous vs. the ~0.05 seed-to-seed spread observed


def _make_params():
    return HomeostaticEIParams(
        x0=jnp.full((2,), 0.1), G0=jnp.array([[0.5, -0.5], [0.5, -0.5]]), H0=jnp.full((2,), 0.3),
        drive=jnp.array([0.5, 0.3]), tau_x_ms=jnp.array(5.0), tau_G_ms=jnp.array(200.0), tau_H_ms=jnp.array(1000.0),
        G_min=jnp.array(-5.0), G_max=jnp.array(5.0), H_min=jnp.array(0.1), H_max=jnp.array(10.0),
        source_scale=jnp.full((2,), 1.0),
    )


def _seed_averaged_late_h(dt_ms: float) -> jnp.ndarray:
    n_steps = int(DURATION_MS / dt_ms)
    window = int(1000.0 / dt_ms)
    late_means = []
    for seed in SEEDS:
        params = _make_params()
        _, _, _, _, H_hist, diag = simulate_homeostatic_ei(
            params, n_steps=n_steps, dt_ms=dt_ms, key=jax.random.PRNGKey(seed),
            activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="cubic_penalty",
        )
        assert not bool(diag["error"]), f"blowup at dt_ms={dt_ms}, seed={seed}"
        late_means.append(H_hist[-window:].mean(axis=0))
    return jnp.stack(late_means).mean(axis=0)


def test_h_equilibrium_is_dt_invariant_under_corrected_noise_scaling():
    results = {dt_ms: _seed_averaged_late_h(dt_ms) for dt_ms in DT_LADDER_MS}
    reference = results[DT_LADDER_MS[0]]
    for dt_ms in DT_LADDER_MS[1:]:
        gap = float(jnp.max(jnp.abs(results[dt_ms] - reference)))
        assert gap < DT_INVARIANCE_TOL, (
            f"H equilibrium at dt_ms={dt_ms} ({results[dt_ms]}) diverged from "
            f"dt_ms={DT_LADDER_MS[0]} ({reference}) by {gap} -- stationary "
            f"statistics should not depend on the step size after the "
            f"sqrt(dt_x) noise-scaling fix"
        )
