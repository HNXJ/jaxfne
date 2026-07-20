"""homeostatic_ei stress test: DC current + 10Hz DC-pulse adaptation over a
100000ms (100s / 200,000-step at dt=0.5ms) simulation, across N=2/4/8 and
both bound_mode ("minimal" vs "stable").

100s of simulated time is a stress-test workload, not a fast unit test --
this script, not tests/, matching the existing scripts/sphere20_ei_hdp_habituation.py
/ scripts/hdp_k_w_ctrl_sweep.py convention in this repo.

Run: PYTHONPATH=. python3 scripts/homeostatic_ei_dc_pulse_adaptation.py
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.emitters_homeostatic_ei import make_minimal_ei_params, simulate_homeostatic_ei

DT_MS = 0.5
DURATION_MS = 100_000.0
N_STEPS = int(DURATION_MS / DT_MS)  # 200,000
DC_AMPLITUDE = 1.0
PULSE_AMPLITUDE = 2.0
PULSE_PERIOD_MS = 100.0  # 10Hz
PULSE_WIDTH_MS = 5.0
LATE_WINDOW_STEPS = 20000  # 10000ms/window -- two-window comparison near the end


def make_dc_schedule(n_steps: int, n: int, amplitude: float) -> jnp.ndarray:
    return jnp.full((n_steps, n), amplitude)


def make_pulse_schedule(n_steps: int, n: int, amplitude: float, dt_ms: float,
                         period_ms: float, width_ms: float) -> jnp.ndarray:
    t_steps = jnp.arange(n_steps)
    period_steps = int(round(period_ms / dt_ms))
    width_steps = int(round(width_ms / dt_ms))
    active = (t_steps % period_steps) < width_steps
    return jnp.where(active[:, None], amplitude, 0.0) * jnp.ones((1, n))


STIMULI = {
    "dc_constant": lambda n: make_dc_schedule(N_STEPS, n, DC_AMPLITUDE),
    "dc_pulses_10hz": lambda n: make_pulse_schedule(N_STEPS, n, PULSE_AMPLITUDE, DT_MS, PULSE_PERIOD_MS, PULSE_WIDTH_MS),
}


def check_adaptation(x_hist: np.ndarray, H_hist: np.ndarray) -> dict:
    all_finite = bool(np.all(np.isfinite(x_hist)) and np.all(np.isfinite(H_hist)))
    if not all_finite:
        return {"all_finite": False, "x_stable": None, "H_stable": None}
    w = LATE_WINDOW_STEPS
    x_early = x_hist[-2 * w:-w].mean(axis=0)
    x_late = x_hist[-w:].mean(axis=0)
    H_early = H_hist[-2 * w:-w].mean(axis=0)
    H_late = H_hist[-w:].mean(axis=0)
    x_gap = float(np.max(np.abs(x_late - x_early)))
    H_gap = float(np.max(np.abs(H_late - H_early)))
    return {
        "all_finite": True,
        "x_stable": x_gap < 0.5,
        "H_stable": H_gap < 0.1,
        "x_gap": x_gap, "H_gap": H_gap,
        "x_late": x_late.tolist(), "H_late": H_late.tolist(),
    }


def main() -> list[dict]:
    results = []
    for n in (2, 4, 8):
        for bound_mode in ("minimal", "stable"):
            params = make_minimal_ei_params(n)
            for stim_name, stim_fn in STIMULI.items():
                drive_schedule = stim_fn(n)
                t0 = time.perf_counter()
                V_m, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
                    params, n_steps=N_STEPS, dt_ms=DT_MS, key=jax.random.PRNGKey(0),
                    activation_rule="cubic", conductance_rule="hebbian", homeostasis_rule="linear",
                    drive_schedule=drive_schedule, bound_mode=bound_mode,
                )
                jax.block_until_ready((V_m, H_hist))
                elapsed = time.perf_counter() - t0
                x_hist = np.asarray(V_m)
                H_hist_np = np.asarray(H_hist)
                error = bool(diag["error"])
                adapt = check_adaptation(x_hist, H_hist_np) if not error else {"all_finite": False}
                row = {"n": n, "bound_mode": bound_mode, "stimulus": stim_name,
                       "error": error, "elapsed_s": round(elapsed, 2), **adapt}
                results.append(row)
                status = "ERROR(non-finite)" if error else (
                    "PASS" if adapt.get("x_stable") and adapt.get("H_stable") else "FINITE-BUT-NOT-SETTLED"
                )
                print(f"n={n:2d} bound_mode={bound_mode:7s} stim={stim_name:15s} "
                      f"elapsed={elapsed:6.2f}s status={status}"
                      + (f" x_gap={adapt.get('x_gap'):.3f} H_gap={adapt.get('H_gap'):.4f}" if not error else ""))
    return results


if __name__ == "__main__":
    results = main()
    n_error = sum(1 for r in results if r["error"])
    n_pass = sum(1 for r in results if not r["error"] and r.get("x_stable") and r.get("H_stable"))
    print()
    print(f"SUMMARY: {len(results)} runs, {n_pass} PASS (finite + settled), "
          f"{n_error} diverged (non-finite), {len(results) - n_pass - n_error} finite-but-not-settled")
