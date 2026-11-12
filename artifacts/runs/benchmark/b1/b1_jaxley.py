#!/usr/bin/env python3
"""B1 M3 Jaxley baseline (timing + spike-count receipt, run_cell API).

Mirrors b1_brian2.run_cell receipt fields; engine differences are explicit in
``mismatch``/``unavailable``. Lane sources: jaxfne/emitters_hh.py (:42-55 HH
densities gNa=120/gK=36/gL=0.3 mS/cm2, E_Na=50/E_K=-77/E_L=-54.3 mV in lane
vs Jaxley native HHChannel eLeak=-54.378 mV; comp area 62.83 um2; :300-317 OU
update theta=1/tau formulation, sigma*sqrt(2*theta); :72 spike readout at
-20 mV crossings).

N independent classic-HH single compartments (harness M3_SCALE_RULE), each
driven by its own OU current series (numpy PCG64 stream, seed-derived);
stimulus mapped density->nA through the 62.83 um2 default-comp area and
inserted via data_stimulate (whose return value feeds jx.integrate as
data_stimuli). Jaxley solves in float32 (dtype requested vs used explicit).

Grid default matches PREREG_B1_0510.md (dt=0.1 ms, duration=1000.0 ms, 10,000 steps);
earlier pilot ran at dt=0.025 / 500 ms. Drive is unified with harness M3:
constant mu=10.0 uA/cm2 + OU noise (tau=20 ms, sigma=0.05 unscaled nominal;
realized std ~0.05 uA/cm2; unscaled vs 1000x scaled 50 uA/cm2 documented).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREREG = ROOT / "artifacts" / "runs" / "benchmark" / "prereg" / "PREREG_B1_0510.md"
LANE_HH = "jaxfne/emitters_hh.py"
HARNESS = "artifacts/runs/benchmark/b1/b1_harness.py"

GRID_DURATION_MS = 1000.0
GRID_DT_MS = 0.1

HH = {"C_m_uF_cm2": 1.0, "gNa_mS_cm2": 120.0, "gK_mS_cm2": 36.0,
      "gL_mS_cm2": 0.3, "E_Na_mV": 50.0, "E_K_mV": -77.0,
      "E_L_mV": -54.378, "E_L_mV_lane": -54.3, "area_um2": 62.83}
OU = {"mu": 10.0, "sigma": 0.05, "theta": 0.05,
      "tau_ms": 20.0, "sigma_unscaled": 0.05, "sigma_scaled_1000x": 50.0}
M3_SCALE_RULE = ("N independent classic-HH single compartments; OU drive "
                 "mu=10 uA/cm2, tau=20 ms (theta=0.05 ms^-1), sigma=0.05 uA/cm2 "
                 "(matching harness unscaled HH_OU_SIGMA_CLIP; lane SDE)")


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def _thread_env() -> dict:
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "XLA_FLAGS", "JAX_PLATFORMS", "JAX_NUM_THREADS")
    return {k: os.environ.get(k, "unset") for k in keys}


class _PeakRSS:
    """10 ms peak-hold host-RSS sampler (Linux ru_maxrss; no psutil needed)."""

    def __init__(self) -> None:
        self.samples: list = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _read_mb(self):
        try:
            import resource

            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1e3
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                v = self._read_mb()
                if v is not None:
                    self.samples.append((time.time(), v))
            except Exception:
                pass
            time.sleep(0.01)

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=5)

    def peak_between(self, t0: float, t1: float):
        vals = [r for (t, r) in self.samples if t0 <= t <= t1]
        return max(vals) if vals else None


def ou_current(rng, steps: int, dt_ms: float, mu=10.0, sigma=5.0,
               theta=1.0) -> np.ndarray:
    """OU series matching the lane theta/dt update (emitters_hh.py:300-317)."""
    x = np.zeros(steps)
    lam = math.exp(-theta * dt_ms)
    s = sigma * math.sqrt(max(0.0, 1.0 - lam * lam))
    for t in range(1, steps):
        x[t] = lam * x[t - 1] + s * rng.standard_normal()
    return mu + x


def _run_single(seed: int, steps: int, dt_ms: float, duration_ms: float):
    import jaxley as jx
    from jaxley.channels import HH as HHChannel

    rng = np.random.default_rng(int(seed))
    ou = ou_current(rng, steps, dt_ms, mu=OU["mu"], sigma=OU["sigma"], theta=OU["theta"])
    area_cm2 = HH["area_um2"] * 1e-8
    stim_nA = ou * area_cm2 * 1e3
    cell = jx.Cell()
    cell.branch(0).comp(0).insert(HHChannel())
    cell.branch(0).comp(0).record("v")
    cell.init_states()
    data_stim = cell.data_stimulate(stim_nA)
    v = np.asarray(jx.integrate(cell, data_stimuli=data_stim,
                                t_max=steps * dt_ms,
                                delta_t=dt_ms), dtype=float).reshape(-1)
    thr = -20.0
    spike_steps = [t for t in range(1, len(v))
                   if v[t - 1] < thr <= v[t]]
    return spike_steps, float(np.mean(v))


def run_cell(spec: dict) -> dict:
    """Run one benchmark cell in this process; returns the receipt dict."""
    import jaxley as jx

    t_cell = time.time()
    sampler = _PeakRSS()
    sampler.start()
    model_id = spec["model"]
    if model_id != "M3":
        raise ValueError(f"unknown model {model_id!r} (want M3)")
    n = int(spec["n"])
    dtype = spec.get("dtype", "float32")
    recording = spec.get("recording", "off")
    seeds = [int(s) for s in spec["seeds"]]
    duration_ms = float(spec.get("duration_ms", GRID_DURATION_MS))
    dt_ms = float(spec.get("dt_ms", GRID_DT_MS))
    build_seed = int(spec.get("build_seed", seeds[0]))
    warmup_seed = int(spec.get("warmup_seed", 0))
    if warmup_seed in seeds:
        warmup_seed = max(seeds) + 1
    n_steps = int(round(duration_ms / dt_ms))

    sim_version = getattr(jx, "__version__", None)
    rec: dict = {"cell": spec.get(
        "cell_id", f"M3_N{n}_{dtype}_jaxley_{recording}"),
        "status": "MEASURED"}
    notes: list = []
    mismatch = [
        {"item": "drive waveform",
         "lane": "I_ext=10 uA/cm2 constant + OU noise (tau=20 ms, sigma=0.05)",
         "here": f"OU mu={OU['mu']} uA/cm2, sigma={OU['sigma']}, tau={OU['tau_ms']} ms (theta={OU['theta']} ms^-1) per cell",
         "action": "unified with harness: both use I_ext/mu=10, tau=20 ms, sigma=0.05 (realized std ~0.05 uA/cm2; unscaled vs 1000x scaled 50 uA/cm2 documented)"},
        {"item": "channel kinetics and leak reversal E_L vs lane",
         "lane": "jaxfne hand HH (emitters_hh.py:42-55, EL_mV=-54.3)",
         "here": f"Jaxley built-in HHChannel() (Sterratt et al.; same densities, eLeak={HH['E_L_mV']} vs lane {HH['E_L_mV_lane']})",
         "action": "declared class difference: Jaxley uses its native channel library model class per prereg ('Jaxley\\'s model class')"},
        {"item": "grid time step and duration vs prereg",
         "lane": "prereg grid dt=0.1 ms, duration=1000 ms (10,000 steps)",
         "here": f"dt={dt_ms} ms, duration={duration_ms} ms (default matches prereg 0.1/1000; pilot was 0.025/500)",
         "action": "matched: default grid settings match PREREG_B1_0510"},
        {"item": "stimulus unit mapping",
         "lane": "current density uA/cm2",
         "here": "density->nA via default-comp area 62.83 um2",
         "action": "none; exact unit conversion for this geometry"},
    ]
    params = [{"name": "hh", "value": dict(HH),
               "source": f"{LANE_HH}:42-55"},
              {"name": "ou", "value": dict(OU),
               "source": f"{LANE_HH}:300-317"},
              {"name": "v_init_mV", "value": -70.0,
               "source": "baseline default"},
              {"name": "threshold_mV", "value": -20.0,
               "source": f"{LANE_HH}:72"}]
    equations = {"M3": ("N independent jx.Cell single-comp HH "
                        f"(gNa={HH['gNa_mS_cm2']},gK={HH['gK_mS_cm2']},gL={HH['gL_mS_cm2']} mS/cm2); OU drive "
                        f"mu={OU['mu']} sigma={OU['sigma']} tau={OU['tau_ms']}ms; readout -20mV crossings; "
                        f"delta_t={dt_ms}ms.")}

    def run_once(seed: int):
        t0 = time.perf_counter()
        counts = [0] * n
        total = 0
        for i in range(n):
            steps_i, _ = _run_single(1000000 * int(build_seed) + int(seed)
                                     + i, n_steps, dt_ms, duration_ms)
            counts[i] = len(steps_i)
            total += len(steps_i)
        return time.perf_counter() - t0, counts, total

    try:
        run_once(warmup_seed)
        run_s, last_total = [], 0
        for s in seeds:
            dt_run, _counts, total = run_once(s)
            run_s.append(dt_run)
            last_total = total
        import numpy as _np

        q1, med, q3 = [float(x) for x in
                       _np.percentile(_np.asarray(run_s), [25, 50, 75])]
        agreement = {"spike_counts": {"E": int(last_total)},
                     "mean_rate_hz": {"E": float(last_total / n /
                                                 (duration_ms / 1000.0))
                                      if n and duration_ms > 0 else 0.0},
                     "total_spikes": int(last_total), "n_steps": n_steps}
        e_rate = float(agreement["mean_rate_hz"]["E"])
        valid = bool(last_total > 0 and _np.isfinite(e_rate) and e_rate > 0)
        peak = sampler.peak_between(t_cell, time.time())
        rec.update({
            "model": model_id, "n": n, "n_steps": n_steps,
            "duration_ms": duration_ms, "dt_ms": dt_ms, "dtype": dtype,
            "device_requested": "cpu", "device_used": "cpu-float32",
            "backend": "jaxley", "batch": 1, "recording": recording,
            "seeds": seeds, "warmup_seed": warmup_seed,
            "build_seed": build_seed, "compile_s": 0.0,
            "construct_s": 0.0, "warmup_s": 0.0,
            "run_s": [float(x) for x in run_s],
            "run_s_median": med, "run_s_q1": q1, "run_s_q3": q3,
            "run_s_iqr": float(q3 - q1),
            "throughput_neuron_steps_s": float(n * n_steps / med)
            if med > 0 else None, "ensemble_s": None,
            "ensemble_mode": "not_run",
            "peak_gpu_mem": {"peak_bytes_in_use": None,
                             "bytes_reserved": None, "bytes_limit": None,
                             "raw": None,
                             "note": "CPU-only Jaxley runs; no device memory"},
            "peak_host_rss_mb": peak, "transfer_s": None,
            "numerical_agreement": agreement,
            "validity": {"spikes_gt_zero": bool(last_total > 0),
                         "mean_E_rate_hz": e_rate, "valid": valid,
                         "note": "in-harness check only; the prereg 20%-vs-"
                                 "reference comparison needs the timed grid"},
            "env": {"sha": _git_sha(), "prereg_sha256": _prereg_sha256(),
                    "jax": None, "jaxlib": None, "jaxfne_version": None,
                    "device_string": "cpu", "dtype": dtype,
                    "threads": _thread_env(), "seed_list": seeds,
                    "platform": platform.platform(),
                    "nvidia_smi_before": "not_requested",
                    "nvidia_smi_after": "not_requested"},
            "scaling_rule": M3_SCALE_RULE,
            "lane_refs": {"M1": "n/a", "M2": "n/a", "M3": LANE_HH},
            "prereg": "artifacts/runs/benchmark/prereg/PREREG_B1_0510.md "
                      "(signed, frozen)",
            "notes": notes, "simulator": "jaxley",
            "simulator_version": sim_version,
            "codegen_target": "jax-jit (jx.integrate)",
            "parameters": params, "equations": equations,
            "mismatch": mismatch,
            "unavailable": [
                {"field": "ensemble_s",
                 "reason": "baselines run batch=1 only; no ensemble path"},
                {"field": "peak_gpu_mem.peak_bytes_in_use",
                 "reason": "CPU-only runs; no device memory to report"},
                {"field": "peak_gpu_mem.bytes_reserved",
                 "reason": "CPU-only runs; no device memory to report"},
                {"field": "peak_gpu_mem.bytes_limit",
                 "reason": "CPU-only runs; no device memory to report"},
                {"field": "peak_gpu_mem.raw",
                 "reason": "CPU-only runs; no device memory to report"},
                {"field": "transfer_s",
                 "reason": "Jaxley runs keep arrays host-side; no "
                           "host-device transfer exists"},
                {"field": "env.jax",
                 "reason": "baseline process imports no jaxfne code"},
                {"field": "env.jaxlib",
                 "reason": "baseline process imports no jaxfne code"},
                {"field": "env.jaxfne_version",
                 "reason": "baseline process imports no jaxfne code"},
                {"field": "compile_s/construct_s/warmup_s",
                 "reason": "single-process loop; not separated for M3"},
            ]})
        if sim_version is None:
            rec["unavailable"].append(
                {"field": "simulator_version",
                 "reason": "jaxley exposes no __version__ here"})
    except Exception as e:
        rec["status"] = "FAILED"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["notes"] = notes
        if "simulator" not in rec:
            rec["simulator"] = "jaxley"
            rec["simulator_version"] = sim_version
    finally:
        sampler.stop()
    rec["wall_s"] = time.time() - t_cell
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run smoke verification")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--duration-ms", type=float, default=GRID_DURATION_MS)
    ap.add_argument("--dt-ms", type=float, default=GRID_DT_MS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.smoke:
        print("=== b1_jaxley smoke ===")
        print(f"Model: M3, N: {args.n}, dt_ms: {args.dt_ms}, duration_ms: {args.duration_ms}")
        print(f"Drive: OU mu={OU['mu']} uA/cm2, tau={OU['tau_ms']} ms (theta={OU['theta']} ms^-1), "
              f"sigma={OU['sigma']} uA/cm2 (unscaled nominal; 1000x scaled is {OU['sigma_scaled_1000x']} uA/cm2)")
        print(f"Parameters: E_L={HH['E_L_mV']} mV (Jaxley default) vs lane {HH['E_L_mV_lane']} mV; native HHChannel kinetics")
        spec = {"model": "M3", "n": args.n, "seeds": [1],
                "duration_ms": min(args.duration_ms, 10.0), "dt_ms": args.dt_ms}
        rec = run_cell(spec)
        print(f"Simulator: jaxley {rec.get('simulator_version')}, status={rec['status']}")
        print(json.dumps({k: rec.get(k) for k in ("cell", "status", "validity", "numerical_agreement")}, indent=2))
        return 0
    rec = run_cell({"model": "M3", "n": args.n,
                    "seeds": [int(s) for s in args.seeds.split(",")],
                    "duration_ms": args.duration_ms, "dt_ms": args.dt_ms})
    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
    print(json.dumps({k: rec.get(k) for k in
                      ("cell", "status", "simulator",
                       "numerical_agreement", "wall_s")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
