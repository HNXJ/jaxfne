#!/usr/bin/env python3
"""B1 benchmark harness, jaxfne side (pre-scored; smoke only).

Covers one benchmark cell of the signed prereg
``artifacts/runs/benchmark/PREREG_B1_0510.md`` (frozen; this file never edits
it). Pattern: ``scripts/benchmark_055_matrix.py`` (fresh process per cell,
``block_until_ready`` timing, RSS peak sampler).

Usage:
    python b1_harness.py --cell --model M1 --n 100 --dtype float32 \\
        --device cpu --batch 1 --recording full --seeds 1,2 --out receipt.json
    python b1_harness.py --driver [--receipt-dir DIR] [--plan-only]

``--cell`` runs one warm-up then the timed seeds in this process and writes
one receipt JSON. ``--driver`` writes the prereg grid to a plan JSON and runs
each missing cell in a subprocess (resumable: receipts that exist are kept).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
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

PREREG = ROOT / "artifacts" / "runs" / "benchmark" / "PREREG_B1_0510.md"
BRUNEL_API = ROOT / "artifacts" / "etudes" / "papers" / "brunel2000" / "brunel2000_api.py"
IZH_RUN = ROOT / "artifacts" / "etudes" / "papers" / "izhikevich2003" / "run_izh.py"

GRID_MODELS = {"M1": [100, 1000, 10000, 100000], "M2": [100, 1000, 10000, 100000],
               "M3": [1, 10, 100]}
GRID_DTYPES = ["float32", "float64"]
GRID_DEVICES = ["cpu", "gpu"]
GRID_BATCH = [1, 8, 32]
GRID_RECORDING = ["full", "reduced", "off"]
GRID_SEEDS = [1, 2, 3, 4, 5]
GRID_DURATION_MS = 1000.0
GRID_DT_MS = 0.1

M1_SCALE_RULE = ("CE=round(0.1*NE), CI=round(0.1*NI) at each N (lane paper default, "
                 "in_degree=None); w=J*sqrt(N) with kernel 1/sqrt(N) normalization; "
                 "external Poisson drive pinned at the lane N_REF=12500 rate "
                 "(Cext=1000 trains at nuext, point D eta=0.9; lane S1 finite-size "
                 "convention: the N-scaled Cext silences N=100)")
M2_SCALE_RULE = ("4:1 E:I resample of the lane Fig.3 draw at each N (same a/b/c/d "
                 "formulas, W=2S rule via the lane all-to-all weight ranges, same "
                 "noise stds); lane scores N=1000 only")
M3_SCALE_RULE = "N independent classic-HH single compartments, I_ext=10 uA/cm2 constant"

# Recording maps to the prereg memory controls (simulation-state vs trajectory).
RECORD_FLAGS = {"full": (True, True), "reduced": (True, False), "off": (False, False)}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _nvidia_smi() -> str:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        txt = (out.stdout or "") + (out.stderr or "")
        return txt[:4000] if txt.strip() else "nvidia-smi: no output"
    except Exception as e:
        return f"nvidia-smi unavailable: {type(e).__name__}: {e}"


def _rss_mb() -> float:
    import psutil

    return float(psutil.Process().memory_info().rss) / 1e6


class _PeakSampler:
    """10 ms peak-hold RSS sampler (pattern: benchmark_055_matrix.py)."""

    def __init__(self) -> None:
        self.samples: list = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), _rss_mb()))
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


def _sync(spikes, v, sources) -> None:
    import jax

    arrs = [spikes, v]
    if sources is not None:
        arrs.append(sources)
    jax.block_until_ready(arrs)


def _gpu_mem() -> dict:
    """Peak GPU memory via the device's memory_stats (None where absent)."""
    out = {"peak_bytes_in_use": None, "bytes_reserved": None, "bytes_limit": None,
           "raw": None, "note": ""}
    try:
        import jax

        devs = jax.devices()
        stats = devs[0].memory_stats()
        raw = {k: (float(v) if isinstance(v, (int, float)) else str(v))
               for k, v in dict(stats).items()}
        out["raw"] = raw
        for key in ("peak_bytes_in_use", "bytes_reserved", "bytes_limit"):
            if key in raw:
                out[key] = raw[key]
        if out["peak_bytes_in_use"] is None:
            out["note"] = "device memory_stats has no peak_bytes_in_use on this backend"
    except Exception as e:
        out["note"] = f"memory_stats unavailable: {type(e).__name__}: {e}"
    return out


def _pop_ids(model, model_id: str, n: int) -> dict[str, list[int]]:
    """Column index ranges per population, in neuron-table order."""
    if model_id in ("M1", "M2"):
        table = model.neuron_table()
        pops: dict[str, list[int]] = {}
        for i, row in enumerate(table):
            key = str(row.get("cell_type", "all"))
            pops.setdefault(key, []).append(i)
        return pops
    return {"cell": list(range(n))}


def _agreement(spikes, pops: dict, duration_ms: float) -> dict:
    import numpy as np

    arr = np.asarray(spikes)
    t_s = float(duration_ms) / 1000.0
    counts, rates = {}, {}
    for pop, cols in pops.items():
        c = int(arr[:, cols].sum()) if cols else 0
        counts[pop] = c
        rates[pop] = float(c / (len(cols) * t_s)) if cols and t_s > 0 else 0.0
    return {"spike_counts": counts, "mean_rate_hz": rates,
            "total_spikes": int(arr.sum()), "n_steps": int(arr.shape[0])}


def _build_M1(n: int, dtype: str, build_seed: int, duration_ms: float, dt_ms: float):
    """Brunel 2000 SI-regime network from the frozen lane builder (point D)."""
    import jaxfne as J
    from jaxfne import neuronal_tensor as nt

    api = _load_module("b1_brunel2000_api", BRUNEL_API)
    tensor = api.build_spec(point="D", n=int(n))
    model = J.construct(
        tensor,
        nt.RuntimeConfiguration(seed=int(build_seed), duration_ms=float(duration_ms),
                               dt_ms=float(dt_ms), emitter="lif", dtype=dtype),
    )
    base = api.build_sim(point="D", n=int(n), T_s=float(duration_ms) / 1000.0,
                         dt_ms=float(dt_ms), seed=int(build_seed), dtype=dtype)
    # Small-N drive pin (lane S1 finite-size convention): the lane default
    # scales Cext with N, which silences N=100 (0.14 mV mean drive vs 20 mV
    # threshold); the external Poisson drive is pinned at the lane N_REF rate
    # while recurrent fan-in follows the paper rule at each N.
    cext_ref = int(api._cext(api.N_REF, None))
    drive = dict(base.poisson_drive)
    drive["counts"] = cext_ref
    base = dataclasses.replace(base, poisson_drive=drive)

    def make_sim(seed: int, recording: str):
        rec_s, rec_f = RECORD_FLAGS[recording]
        return dataclasses.replace(base, seed=int(seed), record_sources=rec_s,
                                   record_fields=rec_f)

    return model, make_sim, M1_SCALE_RULE


def _build_M2(n: int, dtype: str, build_seed: int, duration_ms: float, dt_ms: float):
    """Izhikevich 2003 Fig.3 network resampled at N (lane draw, lane rules)."""
    import jax
    import jax.numpy as jnp

    import jaxfne as J
    from jaxfne.connectivity import with_rule_allow_self_connections

    lane = _load_module("b1_izh_run", IZH_RUN)
    core = _load_module("b1_izh_core",
                        IZH_RUN.parent / "core" / "run_izh_core.py")

    n = int(n)
    n_e = int(round(0.8 * n))
    n_i = n - n_e
    k_r, _, _ = lane.fig3_keys(int(build_seed))
    r = jax.random.uniform(k_r, (n,))
    r_e, r_i = r[:n_e], r[n_e:]
    e = {"a": jnp.full((n_e,), 0.02), "b": jnp.full((n_e,), 0.2),
         "c": -65.0 + 15.0 * r_e ** 2, "d": 8.0 - 6.0 * r_e ** 2}
    i = {"a": 0.02 + 0.08 * r_i, "b": 0.25 - 0.05 * r_i,
         "c": jnp.full((n_i,), -65.0), "d": jnp.full((n_i,), 2.0)}
    payload = {}
    for name, block, m in (("E", e, n_e), ("I", i, n_i)):
        block["v0"] = jnp.full((m,), -65.0)
        block["u0"] = block["b"] * -65.0
        block["drive"] = jnp.zeros((m,))
        payload[name] = {k: np_list(block[k]) for k in core.CELL_KEYS}

    cfg = (
        J.configuration()
        .network(name="izh2003_fig3_b1", kind="balanced_ei_population", n=n,
                 cell_types={"E": 0.8, "I": 0.2})
        .mechanisms(name="instantaneous", kind="receptor",
                    params={"tau_ms": core.INSTANTANEOUS_TAU_MS})
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=1.0, mechanism="instantaneous",
                     weight_range=core.W_RANGE["E"])
        .connections(name="i_all", source={"cell_type": "I"}, target={},
                     probability=1.0, mechanism="instantaneous",
                     weight_range=core.W_RANGE["I"])
        .connectivity(edge_seed=int(build_seed))
        .cell_params({"cell_type": "E"}, payload["E"])
        .cell_params({"cell_type": "I"}, payload["I"])
        .update_metadata(connectivity_mode="explicit")
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
        .runtime(random_v0=False)
    )
    cfg = with_rule_allow_self_connections(cfg, "e_all", True)
    cfg = with_rule_allow_self_connections(cfg, "i_all", True)
    model = J.construct(cfg)

    def make_sim(seed: int, recording: str):
        rec_s, rec_f = RECORD_FLAGS[recording]
        return J.Simulation(
            duration_ms=float(duration_ms), dt_ms=float(dt_ms), seed=int(seed),
            runtime=J.RuntimeConfig(recurrent_backend="edge_list", dtype=dtype),
            gaussian_drive={"std_by_cell_type": dict(core.NOISE_STD)},
            record_sources=rec_s, record_fields=rec_f,
        )

    return model, make_sim, M2_SCALE_RULE


def np_list(a):
    import numpy as np

    return np.asarray(a).tolist()


def _run_M3(n: int, dtype: str, seed: int, n_steps: int, dt_ms: float, recording: str):
    """Classic-HH cells (jaxfne.emitters_hh); returns (spikes, v_or_None, put_like)."""
    import jax
    import jax.numpy as jnp

    from jaxfne import emitters_hh as hh

    key = jax.random.PRNGKey(int(seed))
    k_ou, _ = jax.random.split(key, 2)
    params = hh.make_hh_params(["E"] * int(n))
    jdtype = jnp.float32 if dtype == "float32" else jnp.float64
    i_ext = jnp.full((int(n),), 10.0, dtype=jdtype)
    record_v = recording != "off"
    out = hh.simulate_hh(
        params, n_steps=int(n_steps), dt_ms=float(dt_ms), I_ext=i_ext,
        ou_tau=jnp.full((int(n),), 20.0),
        ou_sigma=jnp.full((int(n),), hh.HH_OU_SIGMA_CLIP),
        ou_key=k_ou, record_v=record_v,
    )
    spikes = out["spikes"]
    v = out.get("V_mV", None)
    jax.block_until_ready(spikes if v is None else [spikes, v])
    return spikes, v


def run_cell(spec: dict) -> dict:
    """Run one cell in-process; returns the receipt dict (writes nothing)."""
    import numpy as np

    t_cell = time.time()
    sampler = _PeakSampler()
    sampler.start()
    model_id = spec["model"]
    n = int(spec["n"])
    dtype = spec.get("dtype", "float32")
    device = spec.get("device", "cpu")
    batch = int(spec.get("batch", 1))
    recording = spec.get("recording", "full")
    seeds = [int(s) for s in spec["seeds"]]
    duration_ms = float(spec.get("duration_ms", GRID_DURATION_MS))
    dt_ms = float(spec.get("dt_ms", GRID_DT_MS))
    build_seed = int(spec.get("build_seed", seeds[0]))
    warmup_seed = int(spec.get("warmup_seed", 0))
    if warmup_seed in seeds:
        warmup_seed = max(seeds) + 1
    n_steps = int(round(duration_ms / dt_ms))

    if device == "cpu" and "jax" not in sys.modules:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jaxfne as J

    have = {d.platform for d in jax.devices()}
    if device == "gpu" and not ({"cuda", "gpu"} & have):
        raise RuntimeError(f"device 'gpu' requested but jax platforms are {sorted(have)}")
    smi_before = _nvidia_smi() if device == "gpu" else "not_requested"

    rec: dict = {"cell": spec.get("cell_id", f"{model_id}_N{n}_{dtype}_{device}"
                                                f"_B{batch}_{recording}"),
                 "status": "MEASURED"}
    notes: list[str] = []
    try:
        if model_id not in ("M1", "M2", "M3"):
            raise ValueError(f"unknown model {model_id!r} (want M1, M2 or M3)")
        if model_id == "M3":
            scale_rule = M3_SCALE_RULE
            t0 = time.perf_counter()
            _run_M3(n, dtype, warmup_seed, n_steps, dt_ms, recording)
            t_warm = time.perf_counter() - t0
            construct_s = 0.0
            run_s, last = [], None
            for s in seeds:
                t0 = time.perf_counter()
                last = _run_M3(n, dtype, s, n_steps, dt_ms, recording)
                run_s.append(time.perf_counter() - t0)
            spikes_np = np.asarray(last[0])
            v_np = (np.asarray(last[1]) if last[1] is not None
                    else np.zeros((n_steps, n)))
            sources_np = None
            pops = {"cell": list(range(n))}
            notes.append("M3 OU noise (tau 20 ms, sigma 0.05) keyed per seed; "
                         "warm-up doubles as the compile probe (eager kernel)")
        else:
            t0 = time.perf_counter()
            if model_id == "M1":
                model, make_sim, scale_rule = _build_M1(n, dtype, build_seed,
                                                       duration_ms, dt_ms)
            elif model_id == "M2":
                model, make_sim, scale_rule = _build_M2(n, dtype, build_seed,
                                                       duration_ms, dt_ms)
            construct_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            warm_sim = make_sim(warmup_seed, recording)
            wsignals = model.simulate(warm_sim)
            _sync(np.asarray(wsignals.spikes), np.asarray(wsignals.V_m),
                  np.asarray(wsignals.sources)
                  if wsignals.sources is not None else None)
            t_warm = time.perf_counter() - t0
            run_s, last_signals = [], None
            for s in seeds:
                sim = make_sim(s, recording)
                t0 = time.perf_counter()
                sig = model.simulate(sim)
                _sync(np.asarray(sig.spikes), np.asarray(sig.V_m),
                      np.asarray(sig.sources) if sig.sources is not None else None)
                run_s.append(time.perf_counter() - t0)
                last_signals = sig
            spikes_np = np.asarray(last_signals.spikes)
            try:
                v_np = np.asarray(last_signals.V_m)
            except Exception:
                v_np = np.zeros_like(spikes_np, dtype=float)
            sources_np = (np.asarray(last_signals.sources)
                          if last_signals.sources is not None else None)
            pops = _pop_ids(model, model_id, n)
            notes.append("construct covers lane build_spec + construct; timed sims "
                         "share the one model (wiring seed fixed)")

        import numpy as _np

        q1, med, q3 = [float(x) for x in _np.percentile(_np.asarray(run_s), [25, 50, 75])]
        compile_s = max(0.0, float(t_warm) - med)

        t0 = time.perf_counter()
        put_arr = jax.device_put(_np.zeros((n_steps, n), dtype=_np.float32))
        jax.block_until_ready(put_arr)
        put_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        _sync(spikes_np, v_np, sources_np)
        get_s = time.perf_counter() - t0

        ensemble_s, ensemble_mode = None, "not_run"
        if batch > 1:
            notes.append("batch>1 runs a sequential per-seed loop; simulate_batch "
                         "refuses the lane drives (poisson/gaussian), so no vmap path "
                         "applies to lane-faithful M1/M2")
            t0 = time.perf_counter()
            for i in range(batch):
                s = 1000 + i
                if model_id == "M3":
                    _run_M3(n, dtype, s, n_steps, dt_ms, recording)
                else:
                    sig = model.simulate(make_sim(s, recording))
                    _sync(np.asarray(sig.spikes), np.asarray(sig.V_m),
                          np.asarray(sig.sources) if sig.sources is not None else None)
            ensemble_s = time.perf_counter() - t0
            ensemble_mode = "sequential_loop"

        agreement = _agreement(spikes_np, pops, duration_ms)
        e_pop = next((k for k in ("E", "cell") if k in agreement["mean_rate_hz"]), None)
        e_rate = float(agreement["mean_rate_hz"][e_pop]) if e_pop else 0.0
        valid = bool(agreement["total_spikes"] > 0 and np.isfinite(e_rate) and e_rate > 0)

        import importlib.metadata as md

        try:
            jaxlib_v = md.version("jaxlib")
        except Exception:
            jaxlib_v = "unknown"
        peak = sampler.peak_between(t_cell, time.time())
        rec.update({
            "model": model_id, "n": n, "n_steps": n_steps,
            "duration_ms": duration_ms, "dt_ms": dt_ms, "dtype": dtype,
            "device_requested": device, "device_used": jax.devices()[0].platform,
            "backend": jax.default_backend(), "batch": batch, "recording": recording,
            "seeds": seeds, "warmup_seed": warmup_seed, "build_seed": build_seed,
            "compile_s": float(compile_s), "construct_s": float(construct_s),
            "warmup_s": float(t_warm), "run_s": [float(x) for x in run_s],
            "run_s_median": med, "run_s_q1": q1, "run_s_q3": q3,
            "run_s_iqr": float(q3 - q1),
            "throughput_neuron_steps_s": float(n * n_steps / med) if med > 0 else None,
            "ensemble_s": ensemble_s, "ensemble_mode": ensemble_mode,
            "ensemble_throughput_neuron_steps_s": (
                float(batch * n * n_steps / ensemble_s)
                if ensemble_s else None),
            "peak_gpu_mem": _gpu_mem(), "peak_host_rss_mb": peak,
            "transfer_s": {"put_s": float(put_s), "get_s": float(get_s),
                           "array_shape": [n_steps, n], "array_dtype": "float32",
                           "note": "device_put of a zeros (steps,N) probe + "
                                   "block_until_ready over the timed outputs"},
            "numerical_agreement": agreement,
            "validity": {"spikes_gt_zero": bool(agreement["total_spikes"] > 0),
                         "mean_E_rate_hz": e_rate, "valid": valid,
                         "note": "in-harness check only; the prereg 20%-vs-reference "
                                 "comparison needs the baseline tools"},
            "env": {"sha": _git_sha(), "prereg_sha256": _prereg_sha256(),
                    "jax": jax.__version__, "jaxlib": jaxlib_v,
                    "jaxfne_version": J.__version__,
                    "device_string": str([str(d) for d in jax.devices()]),
                    "dtype": dtype, "threads": _thread_env(), "seed_list": seeds,
                    "platform": platform.platform(),
                    "nvidia_smi_before": (smi_before if device == "gpu"
                                          else "not_requested"),
                    "nvidia_smi_after": (_nvidia_smi() if device == "gpu"
                                         else "not_requested")},
            "scaling_rule": scale_rule,
            "lane_refs": {"M1": str(BRUNEL_API.relative_to(ROOT)),
                          "M2": str(IZH_RUN.relative_to(ROOT)),
                          "M3": "jaxfne/emitters_hh.py"}[model_id],
            "prereg": "artifacts/runs/benchmark/PREREG_B1_0510.md (signed, frozen)",
            "notes": notes,
        })
    except Exception as e:
        rec["status"] = "FAILED"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["notes"] = notes
    finally:
        sampler.stop()
    rec["wall_s"] = time.time() - t_cell
    return rec


def cell_id(model: str, n: int, dtype: str, device: str, batch: int,
            recording: str) -> str:
    return f"{model}_N{n}_{dtype}_{device}_B{batch}_{recording}"


def grid_cells() -> list[dict]:
    cells = []
    for model, ns in GRID_MODELS.items():
        for n in ns:
            for dtype in GRID_DTYPES:
                for device in GRID_DEVICES:
                    for batch in GRID_BATCH:
                        for recording in GRID_RECORDING:
                            cells.append({
                                "cell_id": cell_id(model, n, dtype, device, batch,
                                                   recording),
                                "model": model, "n": n, "dtype": dtype,
                                "device": device, "batch": batch,
                                "recording": recording, "seeds": list(GRID_SEEDS),
                                "duration_ms": GRID_DURATION_MS, "dt_ms": GRID_DT_MS,
                            })
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--driver", action="store_true")
    ap.add_argument("--model", default="M1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--recording", default="full")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--duration-ms", type=float, default=GRID_DURATION_MS)
    ap.add_argument("--dt-ms", type=float, default=GRID_DT_MS)
    ap.add_argument("--out", default="")
    ap.add_argument("--receipt-dir", default=str(ROOT / "artifacts" / "runs"
                                                / "benchmark" / "b2_0510"))
    ap.add_argument("--plan-out", default="")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    if args.driver:
        receipt_dir = Path(args.receipt_dir)
        receipt_dir.mkdir(parents=True, exist_ok=True)
        plan_out = Path(args.plan_out) if args.plan_out else receipt_dir / "plan.json"
        cells = grid_cells()
        for c in cells:
            c["receipt"] = str(receipt_dir / (c["cell_id"] + ".json"))
        plan_out.write_text(json.dumps({"prereg_sha256": _prereg_sha256(),
                                        "cells": cells}, indent=1), encoding="utf-8")
        print(f"wrote plan {plan_out} ({len(cells)} cells)")
        if args.plan_only:
            return 0
        ran, skipped = 0, 0
        for c in cells:
            if Path(c["receipt"]).exists():
                skipped += 1
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), "--cell",
                   "--model", c["model"], "--n", str(c["n"]),
                   "--dtype", c["dtype"], "--device", c["device"],
                   "--batch", str(c["batch"]), "--recording", c["recording"],
                   "--seeds", ",".join(str(s) for s in c["seeds"]),
                   "--duration-ms", str(c["duration_ms"]),
                   "--dt-ms", str(c["dt_ms"]), "--out", c["receipt"]]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            print(f"{c['cell_id']}: rc={r.returncode} "
                  f"{(r.stdout or '').strip()[-200:]} {(r.stderr or '').strip()[-200:]}",
                  flush=True)
            ran += 1
        print(f"driver done: ran={ran} skipped={skipped}")
        return 0

    if args.cell:
        spec = {"model": args.model, "n": args.n, "dtype": args.dtype,
                "device": args.device, "batch": args.batch,
                "recording": args.recording,
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

    ap.error("pass --cell or --driver")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
