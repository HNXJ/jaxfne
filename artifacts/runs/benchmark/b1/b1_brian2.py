#!/usr/bin/env python3
"""B1 Brian2 baselines (code only, no scored runs).

Covers M1 (Brunel 2000 sparse LIF, SI regime) and M2 (Izhikevich 2003
network) of the signed prereg ``artifacts/runs/benchmark/prereg/PREREG_B1_0510.md``
(frozen; this file never edits it). Same model equations, parameters,
in-degree rule, external drive, dt (0.1 ms) and duration as ``_build_M1`` /
``_build_M2`` in ``artifacts/runs/benchmark/b1/b1_harness.py``; every
parameter below is read from the lane files those builders load, never from
memory, and each receipt lists its source ``path:line``.

Usage:
    python b1_brian2.py --cell --model M1 --n 100 --dtype float32 --target numpy \\
        --seeds 1,2 --recording full --out receipt.json

``--cell`` runs one warm-up then the timed seeds in this process and writes
one receipt JSON with the SAME field names as ``b1_harness.py`` receipts.
Metrics Brian2 cannot report are present as null with an ``unavailable``
list naming them and why; never a made-up value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

M1_SCALE_RULE = (
    "CE=round(0.1*NE), CI=round(0.1*NI) at each N (lane paper default, "
    "in_degree=None); w=J*sqrt(N) with kernel 1/sqrt(N) normalization; "
    "external Poisson drive pinned at the lane N_REF=12500 rate "
    "(Cext=1000 trains at nuext, point D eta=0.9; lane S1 finite-size "
    "convention: the N-scaled Cext silences N=100)"
)
M2_SCALE_RULE = (
    "4:1 E:I resample of the lane Fig.3 draw at each N (same a/b/c/d "
    "formulas, W=2S rule via the lane all-to-all weight ranges, same "
    "noise stds); lane scores N=1000 only"
)
RECORD_FLAGS = {"full", "reduced", "off"}


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
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

    def _read_mb(self) -> float | None:
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

    def peak_between(self, t0: float, t1: float) -> float | None:
        vals = [r for (t, r) in self.samples if t0 <= t <= t1]
        return max(vals) if vals else None


def _m1_params(n: int) -> tuple[dict, list]:
    """Brunel point-D numbers from the lane file; returns (values, sources)."""
    n = int(n)
    ne = int(round(0.8 * n))
    ni = n - ne
    ce = int(round(0.1 * ne))
    ci = int(round(0.1 * ni))
    cext_ref = int(round(0.1 * int(round(0.8 * 12500))))
    vals = {
        "J_mV": 0.1, "g": 4.5, "eta": 0.9, "tau_ms": 20.0,
        "theta_mV": 20.0, "v_reset_mV": 10.0, "t_ref_ms": 2.0,
        "delay_ms": 1.5, "connectivity_p": 0.1, "nuthr_hz": 10.0,
        "n_E": ne, "n_I": ni, "CE": ce, "CI": ci,
        "w_E_mV": 0.1, "w_I_mV": -4.5 * 0.1,
        "nuext_hz": 0.9 * 10.0, "cext_ref": cext_ref,
        "ext_rate_hz": float(cext_ref) * 0.9 * 10.0,
    }
    src = [
        ("J_mV", 0.1, f"{LANE_BRUNEL}:32"),
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
        ("w=J*sqrt(N)", None, f"{LANE_BRUNEL}:121"),
        ("CE/CI fan-in", (ce, ci), f"{LANE_BRUNEL}:124-125"),
        ("external nuext/cext", None, f"{LANE_BRUNEL}:165-166,179"),
        ("B1 Cext pin / scale rule", None, f"{HARNESS}:52-56"),
    ]
    params = [{"name": k, "value": v, "source": s} for k, v, s in src]
    return vals, params


def _m2_params(n: int) -> tuple[dict, list]:
    n = int(n)
    n_e = int(round(0.8 * n))
    vals = {"n_E": n_e, "n_I": n - n_e}
    src = [
        ("E a/b/c/d formulas", None, f"{LANE_IZH}:235-238"),
        ("W=2S, E|w|~U[0,1) I|w|~U[0,2)", None, f"{LANE_IZH}:239-242"),
        ("per-cell noise_scale E/I", (5.0, 2.0), f"{LANE_IZH}:243-244"),
        ("W_RANGE", None, f"{LANE_IZH_CORE}:58"),
        ("NOISE_STD", None, f"{LANE_IZH_CORE}:57"),
        ("INSTANTANEOUS_TAU_MS", 0.0, f"{LANE_IZH_CORE}:61"),
        ("v0/u0/drive", (-65.0, "b*-65", 0.0), f"{HARNESS}:246-249"),
        ("B1 4:1 resample rule", None, f"{HARNESS}:58-59"),
    ]
    params = [{"name": k, "value": v, "source": s} for k, v, s in src]
    return vals, params


def _build_M1(brian2, n: int, p: dict, build_seed: int, dt_ms: float):
    """Brunel network; returns (net, groups, monitors)."""
    from brian2 import (NeuronGroup, PoissonInput, Synapses, SpikeMonitor,
                        Network, ms, mV, Hz, defaultclock)

    defaultclock.dt = dt_ms * ms
    import numpy as np

    rng = np.random.default_rng(int(build_seed))
    ne, ce, ci = p["n_E"], p["CE"], p["CI"]
    eqs = "dv/dt = -v/tau : volt (unless refractory)"
    ns = {"tau": p["tau_ms"] * ms, "V_th": p["theta_mV"] * mV,
          "V_reset": p["v_reset_mV"] * mV}
    g = NeuronGroup(n, model=eqs, threshold="v > V_th", reset="v = V_reset",
                    refractory=p["t_ref_ms"] * ms, method="euler", namespace=ns)
    g.v = 0 * mV
    # Lane drive (PORT_SPEC.md row 12): Cext independent trains per cell at
    # nuext Hz each. PoissonInput draws Binomial(Cext, nuext*dt) per cell per
    # step, the lane's count, in O(N) memory. A single Cext*nuext PoissonGroup
    # train is not used: at 9000 Hz x 0.1 ms it emits Bernoulli(0.9) per step
    # (variance 0.09 vs the lane's 0.9), which silenced M1 in testing.
    cext_ref = int(p["cext_ref"])
    ext = PoissonInput(g, "v", N=cext_ref, rate=p["nuext_hz"] * Hz,
                       weight=p["w_E_mV"] * mV)
    e_idx = list(range(ne))
    i_idx = list(range(ne, n))
    tgt = list(range(n))
    blocks = []
    for pool, k, w in ((e_idx, ce, p["w_E_mV"]), (i_idx, ci, p["w_I_mV"])):
        if k <= 0:
            continue
        pre = rng.choice(pool, size=(n * k,))
        post = np.repeat(tgt, k)
        s = Synapses(g, g, model="W : volt", on_pre="v_post += W")
        s.connect(i=pre, j=post)
        s.W = w * mV
        s.delay = p["delay_ms"] * ms
        blocks.append(s)
    spk = SpikeMonitor(g)
    objs = [g, ext, *blocks, spk]
    return Network(*objs), {"g": g, "spk": spk}, "lif"


def _build_M2(brian2, n: int, p: dict, build_seed: int, dt_ms: float,
              noise_on: bool = True):
    """Izhikevich network; returns (net, groups, monitors, noise_state)."""
    import numpy as np

    from brian2 import (NeuronGroup, Synapses, SpikeMonitor,
                        Network, ms, network_operation, defaultclock)

    defaultclock.dt = dt_ms * ms
    rng = np.random.default_rng(int(build_seed))
    n_e = p["n_E"]
    r = rng.random(n)
    a = np.concatenate([np.full(n_e, 0.02), 0.02 + 0.08 * r[n_e:]])
    b = np.concatenate([np.full(n_e, 0.2), 0.25 - 0.05 * r[n_e:]])
    c = np.concatenate([-65.0 + 15.0 * r[:n_e] ** 2, np.full(n - n_e, -65.0)])
    d = np.concatenate([8.0 - 6.0 * r[:n_e] ** 2, np.full(n - n_e, 2.0)])
    eqs = ("dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn + I_noise)/ms : 1\n"
           "du/dt = a*(b*v - u)/ms : 1\n"
           "dI_syn/dt = -I_syn/(100*ms) : 1\n"
           "I_noise : 1\n"
           "a : 1 (constant)\n"
           "b : 1 (constant)\n"
           "c_c : 1 (constant)\n"
           "d_d : 1 (constant)")
    ns = {"a": a, "b": b, "c_c": c, "d_d": d}
    g = NeuronGroup(n, model=eqs, threshold="v >= 30",
                    reset="v = c_c; u = u + d_d", method="euler", namespace=ns)
    g.v = -65.0
    g.a = a
    g.b = b
    g.c_c = c
    g.d_d = d
    g.u = b * -65.0
    g.I_syn = 0.0
    # Instantaneous W @ prev_spikes in the forward-Euler lane: a one-step
    # current becomes the v jump w*(dt/ms); the slow I_syn decay below only
    # clears numerical residue and never carries charge across steps.
    w_e = rng.random(n_e * n)
    w_i = -rng.random((n - n_e) * n) * 2.0
    pre_e = np.repeat(np.arange(n_e), n)
    post_e = np.tile(np.arange(n), n_e)
    pre_i = np.repeat(np.arange(n_e, n), n)
    post_i = np.tile(np.arange(n), n - n_e)
    pre = np.concatenate([pre_e, pre_i])
    post = np.concatenate([post_e, post_i])
    w = np.concatenate([w_e, w_i])
    s = Synapses(g, g, model="w_syn : 1", on_pre="v_post += w_syn*(dt/ms)")
    s.connect(i=pre, j=post)
    s.w_syn = w
    s.delay = 0 * ms
    sig = np.concatenate([np.full(n_e, 5.0 * np.sqrt(2.0)),
                          np.full(n - n_e, 2.0 * np.sqrt(2.0))])
    coef = float(np.sqrt(0.5 / dt_ms))
    state = {"rng": np.random.default_rng(1000000 * int(build_seed) + 7),
             "sig": sig, "coef": coef, "on": bool(noise_on)}

    @network_operation(dt=defaultclock.dt, when="start")
    def inject(t):
        if state["on"]:
            g.I_noise = state["sig"] * state["coef"] * state["rng"].standard_normal(n)
        else:
            g.I_noise = 0.0

    spk = SpikeMonitor(g)
    return Network(g, s, inject, spk), {"g": g, "spk": spk}, "izh"


def _agreement(spk_counts: dict, n_pops: dict, duration_ms: float) -> dict:
    t_s = float(duration_ms) / 1000.0
    rates = {k: float(spk_counts.get(k, 0) / (n_pops[k] * t_s))
             if n_pops.get(k, 0) and t_s > 0 else 0.0 for k in n_pops}
    total = int(sum(spk_counts.values()))
    return {"spike_counts": {k: int(v) for k, v in spk_counts.items()},
            "mean_rate_hz": rates, "total_spikes": total,
            "n_steps": int(round(float(duration_ms) / 0.1))}


def run_cell(spec: dict) -> dict:
    """Run one benchmark cell in this process; returns the receipt dict."""
    import numpy as np

    t_cell = time.time()
    sampler = _PeakRSS()
    sampler.start()
    model_id = spec["model"]
    n = int(spec["n"])
    dtype = spec.get("dtype", "float32")
    target = spec.get("target", "numpy")
    recording = spec.get("recording", "full")
    seeds = [int(s) for s in spec["seeds"]]
    duration_ms = float(spec.get("duration_ms", GRID_DURATION_MS))
    dt_ms = float(spec.get("dt_ms", GRID_DT_MS))
    build_seed = int(spec.get("build_seed", seeds[0]))
    warmup_seed = int(spec.get("warmup_seed", 0))
    if warmup_seed in seeds:
        warmup_seed = max(seeds) + 1
    n_steps = int(round(duration_ms / dt_ms))

    import brian2

    brian2.prefs.codegen.target = target
    if dtype == "float32":
        brian2.prefs.core.default_float_dtype = np.float32
    elif dtype == "float64":
        brian2.prefs.core.default_float_dtype = np.float64
    else:
        raise ValueError(f"dtype must be float32|float64, got {dtype!r}")

    rec: dict = {"cell": spec.get(
        "cell_id", f"{model_id}_N{n}_{dtype}_brian2-{target}_{recording}"),
        "status": "MEASURED"}
    notes: list[str] = []
    mismatch: list = []
    sim_version = brian2.__version__
    try:
        if model_id == "M1":
            p, params = _m1_params(n)
            scale_rule = M1_SCALE_RULE
            lane_refs = {"M1": LANE_BRUNEL, "M2": LANE_IZH, "M3": "n/a"}
            equations = {
                "M1": ("dv/dt = -v/tau (unless refractory 2 ms); "
                       "threshold v > 20 mV; reset v = 10 mV; "
                       "recurrent: v_post += w on spike (w_E=+0.1 mV, "
                       "w_I=-0.45 mV), delay 1.5 ms, fixed in-degree "
                       "(CE, CI) drawn with replacement via numpy PCG64; "
                       "external: 1000 independent 9 Hz Poisson trains per cell "
                       "(the lane per-train Bernoulli, PORT_SPEC row 12; "
                       "Binomial(1000, nuext*dt) per cell per step), "
                       "v_post += 0.1 mV, delay 0; forward Euler, dt 0.1 ms.")}
            mismatch.append({
                "item": "recurrent sampling replacement rule",
                "lane": "jaxfne fixed_indegree sampler (replacement "
                        "convention not pinned for this path)",
                "here": "numpy choice with replacement per target cell",
                "action": "verify against the lane sampler before scored use",
            })
            notes.append("external drive uses 1000 independent 9 Hz trains "
                         "per cell (the lane per-train Bernoulli); V frozen "
                         "during refractory ((unless refractory)).")
        elif model_id == "M2":
            p, params = _m2_params(n)
            scale_rule = M2_SCALE_RULE
            lane_refs = {"M1": LANE_BRUNEL, "M2": LANE_IZH, "M3": "n/a"}
            equations = {
                "M2": ("dv/dt = (0.04 v^2 + 5 v + 140 - u + I_syn + I_noise)/ms; "
                       "du/dt = a (b v - u)/ms; threshold v >= 30; "
                       "reset v = c, u += d; recurrent: v_post += w*(dt/ms) "
                       "on spike (E w~U[0,1), I w~-U[0,2), all-to-all incl "
                       "self, delay 0 = prev-spike coupling under the "
                       "thresholds->synapses->resets schedule); I_noise = "
                       "sigma*sqrt(0.5/dt)*N(0,1) per step via "
                       "network_operation (numpy PCG64 stream, not the lane "
                       "JAX key stream; same SDE). Per-cell a/b/c/d, v0=-65, "
                       "u0=b*v0, drive 0; forward Euler, dt 0.1 ms.")}
            notes.append("recurrent coupling is a one-step current expressed "
                         "as the exactly equivalent v jump under forward "
                         "Euler; noise stream differs from the lane JAX keys "
                         "by construction (same distribution).")
        else:
            raise ValueError(f"unknown model {model_id!r} (want M1 or M2)")

        build_t0 = time.perf_counter()
        if model_id == "M1":
            net, G, _ = _build_M1(brian2, n, p, build_seed, dt_ms)
        else:
            net, G, _ = _build_M2(brian2, n, p, build_seed, dt_ms)
        construct_s = time.perf_counter() - build_t0
        vmon = None
        if recording == "full":
            from brian2 import StateMonitor

            max_elem = 2e7
            rec_n = n if n * n_steps <= max_elem else min(n, 1000)
            vmon = StateMonitor(G["g"], "v", record=list(range(rec_n)))
            net.add(vmon)
            if rec_n < n:
                notes.append(f"full V recording capped at {rec_n} cells "
                             f"(n*n_steps>{max_elem:.0e}).")

        def run_once(seed: int):
            brian2.seed(int(seed))
            if model_id == "M1":
                net2, G2, _ = _build_M1(brian2, n, p, build_seed, dt_ms)
            else:
                net2, G2, _ = _build_M2(brian2, n, p, build_seed, dt_ms)
            vm2 = None
            if recording == "full":
                from brian2 import StateMonitor

                rec_n = n if n * n_steps <= 2e7 else min(n, 1000)
                vm2 = StateMonitor(G2["g"], "v", record=list(range(rec_n)))
                net2.add(vm2)
            t0 = time.perf_counter()
            net2.run(duration_ms * brian2.ms)
            dt_run = time.perf_counter() - t0
            idx = np.asarray(G2["spk"].i)
            return dt_run, idx, vm2

        t0 = time.perf_counter()
        brian2.seed(int(warmup_seed))
        net.run(duration_ms * brian2.ms)
        t_warm = time.perf_counter() - t0
        del net

        run_s, last_idx = [], None
        for s in seeds:
            dt_run, idx, _ = run_once(s)
            run_s.append(dt_run)
            last_idx = idx

        import numpy as _np

        q1, med, q3 = [float(x) for x in _np.percentile(_np.asarray(run_s), [25, 50, 75])]
        compile_s = max(0.0, float(t_warm) - med)
        notes.append("compile_s = warm-up minus median timed run "
                     "(Brian2 codegen+compile on first run).")

        pops = {"E": list(range(p["n_E"])), {True: "PV", False: "I"}[model_id == "M1"]: list(range(p["n_E"], n))}
        pop_of = _np.zeros(n, dtype=int)
        pop_of[p["n_E"]:] = 1
        names = list(pops)
        counts = {names[k]: int((pop_of[last_idx] == k).sum()) for k in (0, 1)}
        n_pops = {names[0]: p["n_E"], names[1]: n - p["n_E"]}
        agreement = _agreement(counts, n_pops, duration_ms)
        agreement["n_steps"] = n_steps
        e_pop = names[0]
        e_rate = float(agreement["mean_rate_hz"][e_pop])
        valid = bool(agreement["total_spikes"] > 0 and np.isfinite(e_rate) and e_rate > 0)

        peak = sampler.peak_between(t_cell, time.time())
        rec.update({
            "model": model_id, "n": n, "n_steps": n_steps,
            "duration_ms": duration_ms, "dt_ms": dt_ms, "dtype": dtype,
            "device_requested": "cpu", "device_used": "cpu",
            "backend": f"brian2-{target}", "batch": 1, "recording": recording,
            "seeds": seeds, "warmup_seed": warmup_seed, "build_seed": build_seed,
            "compile_s": float(compile_s), "construct_s": float(construct_s),
            "warmup_s": float(t_warm), "run_s": [float(x) for x in run_s],
            "run_s_median": med, "run_s_q1": q1, "run_s_q3": q3,
            "run_s_iqr": float(q3 - q1),
            "throughput_neuron_steps_s": float(n * n_steps / med) if med > 0 else None,
            "ensemble_s": None, "ensemble_mode": "not_run",
            "peak_gpu_mem": {"peak_bytes_in_use": None, "bytes_reserved": None,
                             "bytes_limit": None, "raw": None,
                             "note": "CPU-only Brian2 runs; no device memory"},
            "peak_host_rss_mb": peak,
            "transfer_s": None,
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
            "scaling_rule": scale_rule, "lane_refs": lane_refs,
            "prereg": "artifacts/runs/benchmark/prereg/PREREG_B1_0510.md (signed, frozen)",
            "notes": notes,
            "simulator": "brian2", "simulator_version": sim_version,
            "codegen_target": target,
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
                 "reason": "runtime/codegen runs keep arrays host-side; "
                           "no host-device transfer exists"},
                {"field": "env.jax",
                 "reason": "Brian2 baseline does not use JAX"},
                {"field": "env.jaxlib",
                 "reason": "Brian2 baseline does not use JAX"},
                {"field": "env.jaxfne_version",
                 "reason": "baseline process imports no jaxfne code"},
            ],
        })
    except Exception as e:
        rec["status"] = "FAILED"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["notes"] = notes
        if "simulator" not in rec:
            rec["simulator"] = "brian2"
            rec["simulator_version"] = sim_version
    finally:
        sampler.stop()
    rec["wall_s"] = time.time() - t_cell
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--model", default="M1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--target", default="numpy", choices=["numpy", "cython"])
    ap.add_argument("--recording", default="full")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--duration-ms", type=float, default=GRID_DURATION_MS)
    ap.add_argument("--dt-ms", type=float, default=GRID_DT_MS)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.cell:
        ap.error("pass --cell")
    if args.recording not in RECORD_FLAGS:
        ap.error(f"--recording must be one of {sorted(RECORD_FLAGS)}")
    spec = {"model": args.model, "n": args.n, "dtype": args.dtype,
            "target": args.target, "recording": args.recording,
            "seeds": [int(s) for s in args.seeds.split(",") if s.strip()],
            "duration_ms": args.duration_ms, "dt_ms": args.dt_ms}
    rec = run_cell(spec)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"wrote {args.out} status={rec['status']}")
    else:
        print(json.dumps({k: rec.get(k) for k in
                          ("cell", "status", "run_s_median", "validity")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
