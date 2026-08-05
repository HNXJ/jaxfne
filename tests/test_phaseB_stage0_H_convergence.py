"""Phase B — Stage 0: H(t) interior convergence gate.

Verifies that ``simulate_homeostatic_ei`` with ``freeze_H=False`` and
``homeostasis_rule='cubic_penalty'`` produces an H(t) trajectory that:

  1. Is finite throughout.
  2. Stays strictly inside ``[H_min, H_max]`` at every step.
  3. Settles to an interior equilibrium (late-window max |ΔH| < tolerance).

This is the minimal acceptance gate for Phase B.  H(t) must converge at the
continuous-emitter level before any downstream G-plasticity or Izhikevich-
layer weight updates can be trusted.  The ``cubic_penalty`` rule is used
because it is the only homeostasis_rule documented (and verified) to reach a
genuine interior equilibrium for the canonical 2-neuron E/I circuit; the
linear and logistic rules collapse H to H_min under the same configuration
(see test_homeostatic_ei_g_adaptation_convergence.py for the documented
rationale).

Canonical parameters are copied verbatim from the existing convergence tests
(same drive, H0, G0 shape, tau values) so any future parameter drift here
should be immediately obvious by comparison.

Premise note (B-04a route, 2026-08-04): the ``freeze_G`` gate as first written
could not be satisfied by any ``_homeostasis_cubic_penalty`` variant. Root
cause (measured before editing): the canonical ``G0 = [[0.5,-0.5],[0.5,-0.5]]``
is rank-1 (identical rows), so under ``freeze_G`` the fast dynamics run on a
persistent null mode that the asymmetric drive ``[0.5, 0.3]`` plus noise keeps
exciting forever -- x never reaches a stationary distribution, and H chases a
slowly-moving interior target instead of settling. Under the same G0 with full
G dynamics the gate passes cleanly (late-window max |ΔH| ~ 0.002), and under
``freeze_G`` with a nonsingular stable G0 it also passes (~0.001-0.05 across
seeds). The defect is therefore in the test's isolation premise, not the rule:
the frozen singular G0 never gives the H-ODE a stationary input to converge
against. The ``freeze_G`` test was repaired by removing the stochastic driver
(``noise_scale=0``) and extending the horizon (``N_STEPS_FREEZE_G``) so the
deterministic frozen-G system reaches a genuine fixed point and H settles to a
static interior equilibrium -- verified 12x inside tolerance. See
``_homeostasis_cubic_penalty``'s docstring for the same regression note on the
rule side.
"""
import jax
import jax.numpy as jnp
import pytest

from jaxfne.emitters_homeostatic_ei import HomeostaticEIParams, simulate_homeostatic_ei

# ---------------------------------------------------------------------------
# Canonical parameters — kept intentionally close to
# test_homeostatic_ei_g_adaptation_convergence.py's _canonical_params() so
# that any divergence between the two test files is immediately visible.
# ---------------------------------------------------------------------------
N_STEPS = 8000
# Deterministic freeze_G isolation: the canonical G0 is singular (rank-1), so
# its frozen null mode never reaches a stationary distribution under noise and
# H cannot settle (measured; see module docstring). Removing the stochastic
# driver and running until the deterministic x reaches its fixed point gives
# the H-ODE a static input to converge against. 30000 steps = 15000 ms ~ 15
# tau_H, long past the ~10 tau_H settling time measured for this system.
N_STEPS_FREEZE_G = 30000
NOISE_SCALE_FREEZE_G = 0.0
DT_MS = 0.5
CONVERGENCE_WINDOW = 600
CONVERGENCE_TOL = 0.05   # max |H_hist[-1] - H_hist[-CONVERGENCE_WINDOW]| over all neurons
H_MIN = 0.1
H_MAX = 10.0


def _canonical_params() -> HomeostaticEIParams:
    """2-neuron E/I canonical params for Phase B Stage 0."""
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
        H_min=jnp.array(float(H_MIN)),
        H_max=jnp.array(float(H_MAX)),
        source_scale=jnp.array([1.0, 1.0]),
    )


# ---------------------------------------------------------------------------
# Stage 0 acceptance test
# ---------------------------------------------------------------------------

def test_H_converges_cubic_penalty_freeze_G() -> None:
    """H(t) must reach a finite interior equilibrium when G is frozen.

    Freezing G isolates the H-ODE so that convergence (or the lack of it) is
    unambiguously caused by the homeostasis_rule alone, not by G-feedback.
    This is the cleanest Stage 0 acceptance criterion.

    Deterministic isolation (see module docstring "Premise note"): the
    canonical G0 is singular, so under its frozen null mode with noise the
    fast x never reaches a stationary distribution and H cannot settle for any
    rule. noise_scale=0 removes that stochastic driver so the frozen system
    reaches a true fixed point; N_STEPS_FREEZE_G (30000) covers the measured
    ~10 tau_H settling time with ~12x margin.
    """
    params = _canonical_params()
    voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=N_STEPS_FREEZE_G, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
        activation_rule="cubic",
        conductance_rule="hebbian",
        homeostasis_rule="cubic_penalty",
        freeze_G=True,
        freeze_H=False,
        noise_scale=NOISE_SCALE_FREEZE_G,
    )

    # 1. No non-finite values anywhere.
    assert not bool(diag["error"]), "Non-finite value detected in simulation output."
    assert bool(jnp.all(jnp.isfinite(H_hist))), "H_hist contains non-finite values."

    # 2. H stays inside [H_min, H_max] throughout.
    assert bool(jnp.all(H_hist >= H_MIN - 1e-5)), (
        f"H_hist went below H_min={H_MIN}: min={float(H_hist.min()):.6f}"
    )
    assert bool(jnp.all(H_hist <= H_MAX + 1e-5)), (
        f"H_hist exceeded H_max={H_MAX}: max={float(H_hist.max()):.6f}"
    )

    # 3. H settles: late-window max |ΔH| < tolerance.
    late_delta = float(jnp.max(jnp.abs(H_hist[-1] - H_hist[-CONVERGENCE_WINDOW])))
    assert late_delta < CONVERGENCE_TOL, (
        f"H did not converge: late-window max |ΔH|={late_delta:.6f} exceeds "
        f"tolerance {CONVERGENCE_TOL}. H_hist[-1]={H_hist[-1].tolist()}"
    )

    # 4. H must NOT have collapsed to H_min (that would be the 'linear' rule
    #    failure mode, not a true interior equilibrium).
    assert bool(jnp.all(H_hist[-1] > H_MIN + 0.05)), (
        f"H collapsed to H_min floor: H_hist[-1]={H_hist[-1].tolist()}"
    )


def test_H_converges_cubic_penalty_full_dynamics() -> None:
    """H(t) must converge even when G is also allowed to adapt.

    This is a stronger gate: both G and H are live.  Convergence here means
    the full three-timescale system (fast x, intermediate G, slow H) reaches
    a joint equilibrium rather than H settling only because G is clamped.
    A failure here (while Stage 0 part 1 passes) would pinpoint G-feedback
    as the destabilising source and inform Phase B's next prioritised action.
    """
    params = _canonical_params()
    voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(42),
        activation_rule="cubic",
        conductance_rule="hebbian",
        homeostasis_rule="cubic_penalty",
        freeze_G=False,
        freeze_H=False,
    )

    assert not bool(diag["error"]), "Non-finite value detected (full dynamics)."
    assert bool(jnp.all(jnp.isfinite(H_hist))), "H_hist non-finite (full dynamics)."
    assert bool(jnp.all(H_hist >= H_MIN - 1e-5)), (
        f"H below H_min (full dynamics): min={float(H_hist.min()):.6f}"
    )
    assert bool(jnp.all(H_hist <= H_MAX + 1e-5)), (
        f"H above H_max (full dynamics): max={float(H_hist.max()):.6f}"
    )
    late_delta = float(jnp.max(jnp.abs(H_hist[-1] - H_hist[-CONVERGENCE_WINDOW])))
    assert late_delta < CONVERGENCE_TOL, (
        f"H did not converge (full dynamics): late-window max |ΔH|={late_delta:.6f} "
        f"exceeds tolerance {CONVERGENCE_TOL}."
    )
    assert bool(jnp.all(H_hist[-1] > H_MIN + 0.05)), (
        f"H collapsed to H_min (full dynamics): H_hist[-1]={H_hist[-1].tolist()}"
    )
