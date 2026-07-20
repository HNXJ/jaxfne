"""Milestone 1 of the homeostatic_ei canonical HDP sanity emitter
(see artifacts/developer/plans.json follow-up entry
homeostatic-ei-milestones-4-6-regime-sweep for the deferred parameter-sweep
regression that generalizes this check).

Holds G and H fixed (freeze_G=True, freeze_H=True) and verifies the fast
neuronal dynamics alone (dx/dt = f(x, G, u)) are numerically stable across a
few drive levels and a rotational G. Per the original spec, this milestone
does not assert a specific regime a priori -- it classifies whichever regime
is actually observed and asserts that classification holds across seeds.

Empirically (this file's own parameter points, verified before writing this
test): every point tested lands in a noisy-equilibrium regime -- finite,
seed-consistent late-window statistics, no runaway growth, no oscillation
detected via FFT. No oscillatory or bistable regime was found reachable at
these particular points; Milestone 5's deferred parameter sweep is where a
broader regime search (including oscillatory/bistable) belongs.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters_homeostatic_ei import HomeostaticEIParams, simulate_homeostatic_ei

N_STEPS = 4000
DT_MS = 0.5
N_SEEDS = 5


def _make_params(G, drive):
    return HomeostaticEIParams(
        x0=jnp.array([0.1, 0.1]),
        G0=jnp.asarray(G),
        H0=jnp.array([0.3, 0.3]),
        drive=jnp.asarray(drive),
        tau_x_ms=jnp.array(5.0),
        tau_G_ms=jnp.array(200.0),
        tau_H_ms=jnp.array(1000.0),
        G_min=jnp.array(-5.0),
        G_max=jnp.array(5.0),
        H_min=jnp.array(0.1),
        H_max=jnp.array(10.0),
        source_scale=jnp.array([1.0, 1.0]),
    )


def _classify_regime(voltages: np.ndarray) -> str:
    """Classify the late-window regime: equilibrium, oscillatory, or runaway.

    equilibrium: late-window std stays bounded and doesn't grow across the
    window (first-half vs second-half std ratio near 1).
    oscillatory: a dominant non-zero FFT frequency with power well above the
    noise floor.
    runaway: late-window |mean| grows an order of magnitude beyond the drive
    scale, or non-finite values appear.
    """
    if not np.all(np.isfinite(voltages)):
        return "runaway_nonfinite"
    late = voltages[-2000:]
    first_half, second_half = late[:1000], late[1000:]
    std_ratio = np.std(second_half) / max(np.std(first_half), 1e-8)
    if std_ratio > 3.0 or float(np.max(np.abs(late))) > 50.0:
        return "runaway_growth"
    centered = late - late.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0))
    dc_removed = spectrum[1:]
    if dc_removed.size and float(dc_removed.max()) > 8.0 * float(np.median(dc_removed)):
        return "oscillatory"
    return "equilibrium"


PARAMETER_POINTS = [
    ("low_drive", [[0.5, -0.5], [0.5, -0.5]], [0.2, 0.1]),
    ("canonical_drive", [[0.5, -0.5], [0.5, -0.5]], [0.5, 0.3]),
    ("high_drive", [[0.5, -0.5], [0.5, -0.5]], [1.0, 0.6]),
    ("rotational_g", [[0.2, -1.5], [1.5, 0.2]], [0.5, 0.3]),
]


@pytest.mark.parametrize("name,G,drive", PARAMETER_POINTS)
def test_fixed_g_regime_is_finite_and_seed_consistent(name, G, drive):
    params = _make_params(G, drive)
    regimes = []
    for seed in range(N_SEEDS):
        voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
            params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(seed),
            activation_rule="linear", freeze_G=True, freeze_H=True,
        )
        assert not bool(diag["error"]), f"{name} seed={seed}: non-finite output flagged"
        assert bool(jnp.all(jnp.isfinite(voltages))), f"{name} seed={seed}: non-finite voltages"
        # G/H must be genuinely unchanged when frozen.
        assert bool(jnp.allclose(G_hist[0], G_hist[-1])), f"{name} seed={seed}: G changed despite freeze_G"
        assert bool(jnp.allclose(H_hist[0], H_hist[-1])), f"{name} seed={seed}: H changed despite freeze_H"
        regimes.append(_classify_regime(np.asarray(voltages)))

    assert all(r == regimes[0] for r in regimes), (
        f"{name}: regime classification not seed-consistent, got {regimes}"
    )
    assert regimes[0] not in ("runaway_nonfinite", "runaway_growth"), (
        f"{name}: fixed-G dynamics diverged, regime={regimes[0]}"
    )
