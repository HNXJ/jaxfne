#!/usr/bin/env python3
"""B1 M1/M2 NEST baselines (timing + spike-count receipts, run_cell API).

Mirrors b1_brian2.run_cell receipt fields; engine differences are explicit in
``mismatch``/``unavailable``. Lane sources: brunel2000_api.py (:32 J=0.1mV,
:33 D=1.5ms, :35 tau_m=20ms, :36 theta=20mV, :37 V_r=10mV, :38 p=0.1,
:39 nuthr=10Hz, :41 E fraction 0.8, :48 point-D g=4.5 eta=0.9, :82 N_REF=12500,
:121 w=J, :124-125 CE/CI fan-in, :165-179 Poisson drive); run_izh.py
(:235-249 a/b/c/d formulas, W ranges, noise 5.0/2.0); core/run_izh_core.py
(:57 NOISE_STD=5.0, :58 W_RANGE, :61 TAU null = instantaneous).

NEST notes: kernel is float64 (dtype requested vs used is explicit);
resolution=dt_ms (min 0.1ms); delay 0 (lane/brian2) is unrealizable so the
NEST floor (one step) is used; external drive is the pinned N_REF lane drive
(Cext=1000 independent 9 Hz trains per cell); iaf_psc_delta weights are in mV
(probed: weight 0.1 -> +0.1 mV jump); M2 noise is a white-current
noise_generator proxy for the lane OU-FIR input.
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

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREREG = ROOT / "artifacts" / "runs" / "benchmark" / "prereg" / "PREREG_B1_0510.md"
LANE_BRUNEL = "artifacts/etudes/papers/brunel2000/brunel2000_api.py"
LANE_IZH = "artifacts/etudes/papers/izhikevich2003/run_izh.py"
LANE_IZH_CORE = "artifacts/etudes/papers/izhikevich2003/core/run_izh_core.py"
HARNESS = "artifacts/runs/benchmark/b1/b1_harness.py"

GRID_DURATION_MS = 1000.0
GRID_DT_MS = 0.1

M1_SCALE_RULE = ("CE=round(0.1*NE), CI=round(0.1*NI) at each N; w_E=J=0.1mV, "
                 "w_I=-gJ (g=4.5 point D); external drive pinned at N_REF rate "
                 "(Cext=1000 independent nuext=9Hz trains per cell); delay "
                 "1.5ms recurrent, one-step floor on drive")
M2_SCALE_RULE = ("4:1 E:I resample of the lane Fig.3 draw at each N (same "
                 "a/b/c/d formulas, E w~U[0,1) I w~-U[0,2) all-to-all incl "
                 "self, noise std E/I 5.0/2.0 as white-current proxy for the "
                 "lane OU-FIR input); recurrent delay one-step floor")


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


def _m1_params(n: int) -> tuple[dict, list]:
    n = int(n)
    ne = int(round(0.8 * n))
    ni = n - ne
    ce = int(round(0.1 * ne))
    ci = int(round(0.1 * ni))
    vals = {"J_mV": 0.1, "g": 4.5, "eta": 0.9, "tau_ms": 20.0,
            "theta_mV": 20.0, "v_reset_mV": 10.0, "t_ref_ms": 2.0,
            "delay_ms": 1.5, "connectivity_p": 0.1, "nuthr_hz": 10.0,
            "n_E": ne, "n_I": ni, "CE": ce, "CI": ci,
            "w_E_mV": 0.1, "w_I_mV": -4.5 * 0.1,
            "nuext_hz": 0.9 * 10.0, "ext_rate_hz": 1000.0 * 0.9 * 10.0}
    src = [("J_mV", 0.1, f"{LANE_BRUNEL}:32"),
           ("g,eta point D", (4.5, 0.9), f"{LANE_BRUNEL}:48"),
           ("tau_ms", 20.0, f"{LANE_BRUNEL}:35"),
           ("theta_mV", 20.0, f"{LANE_BRUNEL}:36"),
           ("v_reset_mV", 10.0, f"{LANE_BRUNEL}:37"),
           ("t_ref_ms", 2.0, f"{LANE_BRUNEL}:63"),
           ("delay_ms", 1.5, f"{LANE_BRUNEL}:33,151"),
           ("connectivity_p", 0.1, f"{LANE_BRUNEL}:38"),
           ("nuthr_hz", 10.0, f"{LANE_BRUNEL}:39"),
           ("E_FRACTION", 0.8, f"{LANE_BRUNEL}:41"),
           ("N_REF", 12500, f"{LANE_BRUNEL}:82"),
           ("w=J", None, f"{LANE_BRUNEL}:121"),
           ("CE/CI fan-in", (ce, ci), f"{LANE_BRUNEL}:124-125"),
           ("external nuext", None, f"{LANE_BRUNEL}:165-166,179"),
           ("B1 Cext pin / scale rule", None, f"{HARNESS}:52-56")]
    return vals, [{"name": k, "value": v, "source": s} for k, v, s in src]


def _m2_params(n: int) -> tuple[dict, list]:
    n = int(n)
    n_e = int(round(0.8 * n))
    vals = {"n_E": n_e, "n_I": n - n_e}
    src = [("E a/b/c/d formulas", None, f"{LANE_IZH}:235-238"),
           ("W=2S, E|w|~U[0,1) I|w|~U[0,2)", None, f"{LANE_IZH}:239-242"),
           ("per-cell noise_scale E/I", (5.0, 2.0), f"{LANE_IZH}:243-244"),
           ("W_RANGE", None, f"{LANE_IZH_CORE}:58"),
           ("NOISE_STD", None, f"{LANE_IZH_CORE}:57"),
           ("INSTANTANEOUS_TAU_MS", 0.0, f"{LANE_IZH_CORE}:61"),
           ("v0/u0/drive", (-65.0, "b*-65", 0.0), f"{HARNESS}:246-249"),
           ("B1 4:1 resample rule", None, f"{HARNESS}:58-59")]
    return vals, [{"name": k, "value": v, "source": s} for k, v, s in src]


def _build_M1(nest, n: int, p: dict, dt_ms: float):
    nodes = nest.Create("iaf_psc_delta", n,
                        params={"C_m": 250.0, "tau_m": p["tau_ms"],
                                "t_ref": p["t_ref_ms"], "V_th": p["theta_mV"],
                                "V_reset": p["v_reset_mV"], "E_L": 0.0,
                                "V_m": 0.0})
    e_pool = nodes[:p["n_E"]]
    i_pool = nodes[p["n_E"]:]
    if p["CE"]:
        nest.Connect(e_pool, nodes,
                     conn_spec={"rule": "fixed_indegree",
                                "indegree": p["CE"],
                                "allow_autapses": True,
                                "allow_multapses": True},
                     syn_spec={"weight": p["w_E_mV"],
                               "delay": p["delay_ms"]})
    if p["CI"]:
        nest.Connect(i_pool, nodes,
                     conn_spec={"rule": "fixed_indegree",
                                "indegree": p["CI"],
                                "allow_autapses": True,
                                "allow_multapses": True},
                     syn_spec={"weight": p["w_I_mV"],
                               "delay": p["delay_ms"]})
    # Lane drive (PORT_SPEC.md row 12): Cext independent nuext-Hz trains per
    # cell. One poisson_generator at Cext*nuext sends each target its own
    # Poisson train, so each cell gets Poisson(Cext*nuext*dt) per step
    # (variance 0.9 at dt 0.1 ms; the lane's Binomial(1000, 9e-4) has 0.899).
    # Brian2's PoissonGroup per-step Bernoulli cap does not apply here.
    pg = nest.Create("poisson_generator", 1,
                     params={"rate": p["ext_rate_hz"]})
    nest.Connect(pg, nodes, syn_spec={"weight": p["w_E_mV"], "delay": dt_ms})
    sr = nest.Create("spike_recorder")
    nest.Connect(nodes, sr)
    return nodes, sr


def _build_M2(nest, n: int, p: dict, dt_ms: float, rng):
    import numpy as np

    n_e = p["n_E"]
    r = rng.random(n)
    a = np.concatenate([np.full(n_e, 0.02), 0.02 + 0.08 * r[n_e:]])
    b = np.concatenate([np.full(n_e, 0.2), 0.25 - 0.05 * r[n_e:]])
    c = np.concatenate([-65.0 + 15.0 * r[:n_e] ** 2, np.full(n - n_e, -65.0)])
    d = np.concatenate([8.0 - 6.0 * r[:n_e] ** 2, np.full(n - n_e, 2.0)])
    nodes = nest.Create("izhikevich", n)
    nest.SetStatus(nodes, [{"a": float(a[i]), "b": float(b[i]),
                            "c": float(c[i]), "d": float(d[i]), "I_e": 0.0,
                            "V_m": -65.0, "U_m": float(b[i] * -65.0)}
                           for i in range(n)])
    e_pool = nest.NodeCollection(
        [nodes[i].get("global_id") for i in range(n_e)])
    i_pool = nest.NodeCollection(
        [nodes[i].get("global_id") for i in range(n_e, n)])
    nest.Connect(e_pool, nodes, conn_spec={"rule": "all_to_all"},
                 syn_spec={"weight": 1.0, "delay": dt_ms})
    conns_e = nest.GetConnections(e_pool, nodes)
    conns_e.set(weight=rng.uniform(0.0, 1.0, len(conns_e)))
    if n - n_e:
        nest.Connect(i_pool, nodes, conn_spec={"rule": "all_to_all"},
                     syn_spec={"weight": -1.0, "delay": dt_ms})
        conns_i = nest.GetConnections(i_pool, nodes)
        conns_i.set(weight=-rng.uniform(0.0, 2.0, len(conns_i)))
    ng_e = nest.Create("noise_generator",
                       params={"mean": 0.0,
                               "std": 5.0 / math.sqrt(dt_ms),
                               "dt": dt_ms})
    nest.Connect(ng_e, e_pool)
    if n - n_e:
        ng_i = nest.Create("noise_generator",
                           params={"mean": 0.0,
                                   "std": 2.0 / math.sqrt(dt_ms),
                                   "dt": dt_ms})
        nest.Connect(ng_i, i_pool)
    sr = nest.Create("spike_recorder")
    nest.Connect(nodes, sr)
    return nodes, sr


def _agreement(spk_counts: dict, n_pops: dict, duration_ms: float,
               dt_ms: float) -> dict:
    t_s = float(duration_ms) / 1000.0
    rates = {k: float(spk_counts.get(k, 0) / (n_pops[k] * t_s))
             if n_pops.get(k, 0) and t_s > 0 else 0.0 for k in n_pops}
    total = int(sum(spk_counts.values()))
    return {"spike_counts": {k: int(v) for k, v in spk_counts.items()},
            "mean_rate_hz": rates, "total_spikes": total,
            "n_steps": int(round(float(duration_ms) / float(dt_ms)))}


def _spike_counts(nest, nodes, sr, n_e: int):
    import numpy as np

    events = nest.GetStatus(sr, "events")[0]
    senders = np.asarray(events["senders"])
    base = int(nodes[0].get("global_id"))
    idx = senders - base
    names = ["E", "I"]
    pop_of = np.zeros(len(idx), dtype=int)
    pop_of[idx >= n_e] = 1
    return {names[k]: int((pop_of == k).sum()) for k in (0, 1)}


def run_cell(spec: dict) -> dict:
    """Run one benchmark cell in this process; returns the receipt dict."""
    import numpy as np

    t_cell = time.time()
    sampler = _PeakRSS()
    sampler.start()
    model_id = spec["model"]
    n = int(spec["n"])
    dtype = spec.get("dtype", "float64")
    recording = spec.get("recording", "off")
    seeds = [int(s) for s in spec["seeds"]]
    duration_ms = float(spec.get("duration_ms", GRID_DURATION_MS))
    dt_ms = float(spec.get("dt_ms", GRID_DT_MS))
    if dt_ms < 0.1:
        raise ValueError(f"NEST resolution floor is 0.1 ms, got {dt_ms!r}")
    build_seed = int(spec.get("build_seed", seeds[0]))
    warmup_seed = int(spec.get("warmup_seed", 0))
    if warmup_seed in seeds:
        warmup_seed = max(seeds) + 1
    n_steps = int(round(duration_ms / dt_ms))

    import nest

    sim_version = nest.__version__
    rec: dict = {"cell": spec.get(
        "cell_id", f"{model_id}_N{n}_{dtype}_nest_off"),
        "status": "MEASURED"}
    notes: list = []
    mismatch: list = []
    if model_id == "M1":
        p, params = _m1_params(n)
        scale_rule = M1_SCALE_RULE
        equations = {"M1": ("iaf_psc_delta C_m=250pF tau_m=20ms t_ref=2ms "
                            "V_th=20mV V_reset=10mV E_L=0mV; recurrent "
                            "fixed in-degree (CE, CI) weights w_E=+0.1mV "
                            "w_I=-0.45mV delay 1.5ms; external 1000 "
                            "independent 9 Hz poisson trains per cell "
                            "(lane per-train Bernoulli), weight 0.1mV, "
                            "delay one step; resolution 0.1ms.")}
        mismatch.append({"item": "drive delay floor",
                         "lane": "delay 0 on the external projection",
                         "here": f"one-step delay ({dt_ms} ms, NEST minimum)",
                         "action": "none; sub-step timing below resolution"})
    elif model_id == "M2":
        p, params = _m2_params(n)
        scale_rule = M2_SCALE_RULE
        equations = {"M2": ("izhikevich a/b/c/d per cell (lane Fig.3 "
                            "formulas), V_th=30, I_e=0, v0=-65, u0=b*v0; "
                            "recurrent all-to-all incl self, E w~U[0,1) "
                            "I w~-U[0,2), one-step delay; noise_generator "
                            "white current with per-step std sigma/sqrt(dt) "
                            "(E 5, I 2) as the lane OU-FIR SDE Euler step; "
                            "resolution 0.1ms.")}
        mismatch.append({"item": "noise input stream",
                         "lane": "OU-FIR noise via JAX keys (brian2: numpy "
                                 "PCG64 network_operation)",
                         "here": "NEST noise_generator white current",
                         "action": "verify spectra before scored use"})
        mismatch.append({"item": "recurrent delay floor",
                         "lane": "delay 0 (instantaneous W)",
                         "here": f"one-step delay ({dt_ms} ms, NEST minimum)",
                         "action": "none; sub-step timing below resolution"})
    else:
        raise ValueError(f"unknown model {model_id!r} (want M1 or M2)")
    notes.append("NEST kernel is float64; dtype records the request, "
                 "device_used records the actual precision.")

    def run_once(seed: int):
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": dt_ms, "print_time": False,
                              "rng_seed": int(seed) + 1})
        rng = np.random.default_rng(int(build_seed))
        t0 = time.perf_counter()
        if model_id == "M1":
            nodes, sr = _build_M1(nest, n, p, dt_ms)
        else:
            nodes, sr = _build_M2(nest, n, p, dt_ms, rng)
        construct_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        nest.Simulate(duration_ms)
        dt_run = time.perf_counter() - t1
        return dt_run, construct_s, _spike_counts(nest, nodes, sr, p["n_E"])

    try:
        _, _, spikes_w = run_once(warmup_seed)
        t_warm = 0.0
        run_s, last_counts = [], None
        construct_s = 0.0
        for s in seeds:
            dt_run, construct_s, counts = run_once(s)
            run_s.append(dt_run)
            last_counts = counts
        import numpy as _np

        q1, med, q3 = [float(x) for x in
                       _np.percentile(_np.asarray(run_s), [25, 50, 75])]
        pops = {"E": p["n_E"], "I": n - p["n_E"]}
        agreement = _agreement(last_counts, pops, duration_ms, dt_ms)
        agreement["n_steps"] = n_steps
        e_rate = float(agreement["mean_rate_hz"]["E"])
        valid = bool(agreement["total_spikes"] > 0 and np.isfinite(e_rate)
                     and e_rate > 0)
        peak = sampler.peak_between(t_cell, time.time())
        rec.update({
            "model": model_id, "n": n, "n_steps": n_steps,
            "duration_ms": duration_ms, "dt_ms": dt_ms, "dtype": dtype,
            "device_requested": "cpu", "device_used": "cpu-float64",
            "backend": "nest", "batch": 1, "recording": recording,
            "seeds": seeds, "warmup_seed": warmup_seed,
            "build_seed": build_seed, "compile_s": 0.0,
            "construct_s": float(construct_s), "warmup_s": float(t_warm),
            "run_s": [float(x) for x in run_s],
            "run_s_median": med, "run_s_q1": q1, "run_s_q3": q3,
            "run_s_iqr": float(q3 - q1),
            "throughput_neuron_steps_s": float(n * n_steps / med)
            if med > 0 else None, "ensemble_s": None,
            "ensemble_mode": "not_run",
            "peak_gpu_mem": {"peak_bytes_in_use": None,
                             "bytes_reserved": None, "bytes_limit": None,
                             "raw": None,
                             "note": "CPU-only NEST runs; no device memory"},
            "peak_host_rss_mb": peak, "transfer_s": None,
            "numerical_agreement": agreement,
            "validity": {"spikes_gt_zero": bool(agreement["total_spikes"] > 0),
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
            "scaling_rule": scale_rule,
            "lane_refs": {"M1": LANE_BRUNEL, "M2": LANE_IZH, "M3": "n/a"},
            "prereg": "artifacts/runs/benchmark/prereg/PREREG_B1_0510.md "
                      "(signed, frozen)",
            "notes": notes, "simulator": "nest",
            "simulator_version": sim_version,
            "codegen_target": "n/a (NEST exact C++ kernel)",
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
                 "reason": "NEST runs keep arrays host-side; no "
                           "host-device transfer exists"},
                {"field": "env.jax",
                 "reason": "NEST baseline does not use JAX"},
                {"field": "env.jaxlib",
                 "reason": "NEST baseline does not use JAX"},
                {"field": "env.jaxfne_version",
                 "reason": "baseline process imports no jaxfne code"},
                {"field": "compile_s",
                 "reason": "NEST has no codegen compile step; set to 0.0"},
            ]})
    except Exception as e:
        rec["status"] = "FAILED"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["notes"] = notes
        if "simulator" not in rec:
            rec["simulator"] = "nest"
            rec["simulator_version"] = sim_version
    finally:
        sampler.stop()
    rec["wall_s"] = time.time() - t_cell
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--duration-ms", type=float, default=100.0)
    ap.add_argument("--dt-ms", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rec = run_cell({"model": args.model, "n": args.n,
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
