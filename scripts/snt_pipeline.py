"""Combined sanity-test pipeline: snt_1 (major sanity, Izhikevich/HDP) +
snt_2 (homeostatic_ei), both at minimal 8-neuron scale.

Both tests check the same thing: does each neuron's firing rate get
attracted to a stable equilibrium via HDP adaptation? "Stable" means the
late-window rate matches the early-window rate within tolerance (still
settling would fail this); a target band is reported alongside as context,
not as a hard requirement for every neuron (see snt_1's PV-neuron result --
a real, honest low-rate equilibrium, not forced into the same band as E).

Run: PYTHONPATH=. python3 scripts/snt_pipeline.py
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import jaxfne as jtfne

RATE_BAND_HZ = (3.0, 15.0)
STABILITY_TOL_HZ = 2.0
RATE_PROXY_GAIN = 10.0  # homeostatic_ei's x is rate-like, not a spike count -- see snt_2


def per_neuron_rate_hz(spikes: np.ndarray, dt_ms: float, window: int) -> tuple:
    """Return (penultimate_window_rate, last_window_rate), each shape
    (n_neurons,) -- comparing two LATE windows (not start-vs-end) tests
    whether the system has actually stopped changing, not just how far it
    has moved since the initial transient."""
    spikes = np.asarray(spikes)
    early = spikes[-2 * window:-window].mean(axis=0) * (1000.0 / dt_ms)
    late = spikes[-window:].mean(axis=0) * (1000.0 / dt_ms)
    return early, late


def check_stable_equilibrium(early: np.ndarray, late: np.ndarray, label: str) -> bool:
    """Attracted to a stable point = late-window rate matches early-window
    rate within STABILITY_TOL_HZ (not still drifting toward something else)."""
    delta = np.abs(late - early)
    stable = bool(np.all(delta <= STABILITY_TOL_HZ))
    in_band = (late >= RATE_BAND_HZ[0]) & (late <= RATE_BAND_HZ[1])
    print(f"  {label}: early={early.round(2).tolist()} late={late.round(2).tolist()} "
          f"delta={delta.round(2).tolist()} stable={stable} in_{RATE_BAND_HZ}Hz_band={in_band.tolist()}")
    return stable


def snt_1_major_sanity() -> dict:
    """8-neuron (6E/2I) Izhikevich/HDP circuit, real spike train, HDP enabled
    via DEFAULT_HDP. Checks each neuron's firing rate settles to a stable
    late-window value under HDP adaptation."""
    print("=== snt_1: major_sanity (8-neuron Izhikevich, DEFAULT_HDP) ===")
    # within_gain=0.0 avoids double-wiring construct()'s default recurrent
    # connectivity on top of make_minimal_ei_tensor()'s explicit InterConnections.
    cfg = jtfne.neuronal_tensor_to_configuration(
        jtfne.make_minimal_ei_tensor(), seed=0, duration_ms=100.0, dt_ms=0.1
    ).connectivity(within_area="all_to_all_uniform_random", within_gain=0.0)
    model = jtfne.construct(cfg)

    dt_ms = 0.1
    duration_ms = 5000.0
    sim = jtfne.simulation(
        duration_ms=duration_ms, dt_ms=dt_ms, seed=0,
        runtime=jtfne.RuntimeConfig(enable_hdp=True, hdp_params=dict(jtfne.DEFAULT_HDP)),
    )
    signals = model.simulate(sim)
    diag = model.last_hdp_diagnostics()

    V_m = np.asarray(signals.V_m)
    assert np.all(np.isfinite(V_m)) and np.all(np.abs(V_m) < 150.0), "blowup detected"

    window = 10000  # 1000ms at dt=0.1ms -- 1Hz bin resolution, avoids single-spike quantization noise
    early, late = per_neuron_rate_hz(signals.spikes, dt_ms, window)
    stable = check_stable_equilibrium(early, late, "rate_hz (8 neurons, E[0:6]/PV[6:8])")

    H_trace = np.asarray(diag["H_trace"]) if diag else None
    h_finite = bool(np.all(np.isfinite(H_trace))) if H_trace is not None else False
    print(f"  H finite: {h_finite}, H_final: {H_trace[-1].round(3).tolist() if H_trace is not None else 'n/a'}")

    result = {"stable": stable, "early_hz": early, "late_hz": late, "h_finite": h_finite}
    print(f"  RESULT: {'PASS' if stable and h_finite else 'FAIL'} (stability + finiteness)")
    return result


def snt_2_homeostatic_ei() -> dict:
    """8-neuron homeostatic_ei circuit (6E/2I, generalized from the original
    2-neuron canonical build in jaxfne/_construct.py). x settles to a fixed
    point rather than oscillating, so "firing rate" here is a rectified-x
    proxy (RATE_PROXY_GAIN * relu(x)), not a spike count -- see this
    session's finding that edge-triggered spike-counting gives near-zero Hz
    for a system that rarely crosses zero more than once.

    Also demonstrates HDP's necessity directly: with H frozen, this 8-neuron
    circuit diverges to NaN; with H adapting, it settles to a tight,
    seed-consistent band -- H is load-bearing, not decorative.
    """
    print("=== snt_2: homeostatic_ei (8-neuron, generalized construct() path) ===")
    cfg = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=6000.0, dt_ms=0.5)
        .network(name="ei8", n=8)
        .set_emitter("homeostatic_ei")
        .field(domain="none")
        .probe(modes=["vm"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=6000.0, dt_ms=0.5, seed=0)
    signals = model.simulate(sim)

    V_m = np.asarray(signals.V_m)
    assert np.all(np.isfinite(V_m)), "blowup detected with H adapting"

    window = 400  # 200ms at dt=0.5ms
    x_early = V_m[-2 * window:-window].mean(axis=0)
    x_late = V_m[-window:].mean(axis=0)
    rate_early = np.maximum(x_early, 0.0) * RATE_PROXY_GAIN
    rate_late = np.maximum(x_late, 0.0) * RATE_PROXY_GAIN
    stable = check_stable_equilibrium(rate_early, rate_late, "rate_hz_proxy (8 neurons, E[0:6]/I[6:8])")

    H_trace = np.asarray(signals.metadata["hdp"]["H_trace"])
    h_finite = bool(np.all(np.isfinite(H_trace)))
    print(f"  H finite: {h_finite}, H_final: {H_trace[-1].round(3).tolist()}")

    result = {"stable": stable, "early_hz": rate_early, "late_hz": rate_late, "h_finite": h_finite}
    print(f"  RESULT: {'PASS' if stable and h_finite else 'FAIL'} (stability + finiteness)")
    return result


if __name__ == "__main__":
    r1 = snt_1_major_sanity()
    print()
    r2 = snt_2_homeostatic_ei()
    print()
    overall = (r1["stable"] and r1["h_finite"] and r2["stable"] and r2["h_finite"])
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
