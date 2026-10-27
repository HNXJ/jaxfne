#!/usr/bin/env python3
"""Precision profile of one Izhikevich E/PV circuit at float64, float32 and bfloat16.

Each cell of the grid (dtype, n, steps, mode) runs in its own process, so the device's
peak-memory counter starts from zero. Per cell: compile time, steady wall time of the
compiled step, XLA ``memory_analysis`` and ``cost_analysis``, peak device memory, the
loss, the gradient (``mode="grad"``) and the spike count. The loss is the soft-rate MSE
on ``V_m`` that ``tune_handles`` differentiates; the parameter is the per-neuron
Izhikevich ``a`` (``cells.izhikevich.a[all]``), so the gradient runs back through the
whole recurrent scan.

Usage (WSL GPU, from the repo root):
    python scripts/perf/precision_profile.py --sweep --out artifacts/_temp/precision/grid.json
    python scripts/perf/precision_profile.py --one --dtype float32 --n 100 --steps 1000 --mode grad
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DTYPES = ("float64", "float32", "bfloat16")
DT_MS = 0.1
TARGET_HZ = 5.0
HANDLE = "cells.izhikevich.a[all]"


def _model(n: int):
    import jaxfne as jtfne

    cfg = (
        jtfne.configuration()
        .network(n=n, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _cost(compiled) -> dict:
    cost = compiled.cost_analysis()
    if isinstance(cost, (list, tuple)):
        cost = cost[0] if cost else {}
    keys = ("flops", "bytes accessed", "transcendentals")
    return {k.replace(" ", "_"): float(cost[k]) for k in keys if cost and k in cost}


def _memory(compiled) -> dict:
    mem = compiled.memory_analysis()
    names = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
             "generated_code_size_in_bytes", "alias_size_in_bytes")
    return {k: int(getattr(mem, k)) for k in names if mem is not None and hasattr(mem, k)}


def run_one(dtype: str, n: int, steps: int, mode: str, repeats: int, trace_dir: str | None) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    import jaxfne as jtfne
    from jaxfne._model_evaluate import _evaluate_soft_rate_targets

    if dtype == "float64" and not jax.config.jax_enable_x64:
        raise SystemExit("float64 needs JAX_ENABLE_X64=1 before jax is imported")
    device = jax.devices()[0]
    duration_ms = steps * DT_MS
    model = _model(n)
    jdtype = {"float64": jnp.float64, "float32": jnp.float32, "bfloat16": jnp.bfloat16}[dtype]
    theta0 = jnp.asarray(np.asarray(jtfne.get_parameter(model, HANDLE), dtype=np.float64), dtype=jdtype)
    groups = {"all": list(range(n))}

    def loss(theta):
        candidate = jtfne.set_parameter(model, HANDLE, theta)
        sig = jtfne.simulate(candidate, duration_ms=duration_ms, dt_ms=DT_MS, seed=0, dtype=dtype,
                             record_fields=False)
        return _evaluate_soft_rate_targets(V_m=sig.V_m, groups=groups, targets_hz={"all": TARGET_HZ},
                                           duration_ms=duration_ms, dt_ms=DT_MS)

    fn = jax.jit(jax.value_and_grad(loss) if mode == "grad" else loss)
    stats0 = device.memory_stats() or {}
    t0 = time.perf_counter()
    compiled = fn.lower(theta0).compile()
    t_compile = time.perf_counter() - t0
    out = jax.block_until_ready(compiled(theta0))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(theta0))
        times.append(time.perf_counter() - t0)
    if trace_dir:
        jax.profiler.start_trace(trace_dir)
        for _ in range(3):
            jax.block_until_ready(compiled(theta0))
        jax.profiler.stop_trace()
    stats1 = device.memory_stats() or {}

    loss_val, grad = (out if mode == "grad" else (out, None))
    sig = jtfne.simulate(model, duration_ms=duration_ms, dt_ms=DT_MS, seed=0, dtype=dtype, record_fields=False)
    rec = {
        "dtype": dtype, "n": n, "steps": steps, "dt_ms": DT_MS, "mode": mode,
        "backend": jax.default_backend(), "device": str(device), "jax": jax.__version__,
        "jaxfne": jtfne.__version__, "python": platform.python_version(),
        "compile_s": t_compile, "run_s_median": float(np.median(times)), "run_s_all": times,
        "memory_analysis": _memory(compiled), "cost_analysis": _cost(compiled),
        "peak_bytes_in_use": stats1.get("peak_bytes_in_use"),
        "bytes_in_use_before_compile": stats0.get("bytes_in_use"),
        "bytes_limit": stats1.get("bytes_limit"),
        "loss": float(np.asarray(loss_val, dtype=np.float64)),
        "spikes_total": int(np.asarray(sig.spikes).sum()),
        "V_m_dtype": str(np.asarray(sig.V_m).dtype),
    }
    if grad is not None:
        g = np.asarray(grad, dtype=np.float64)
        rec["grad_l2"] = float(np.linalg.norm(g))
        rec["grad_finite"] = bool(np.isfinite(g).all())
        if n <= 400:
            rec["grad"] = g.tolist()
    return rec


def _grid(quick: bool) -> list[tuple[str, int, int, str]]:
    ns = (100, 400) if quick else (100, 200, 400, 800, 1600, 3200)
    steps = (500, 1000) if quick else (250, 500, 1000, 2000, 4000)
    cells = []
    for dtype in DTYPES:
        for mode in ("fwd", "grad"):
            cells += [(dtype, n, 1000, mode) for n in ns]
            cells += [(dtype, 100, s, mode) for s in steps if s != 1000]
    return cells


def sweep(out: Path, quick: bool, repeats: int, timeout_s: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for dtype, n, steps, mode in _grid(quick):
        env = dict(os.environ, JAX_ENABLE_X64="1" if dtype == "float64" else "0")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one", "--dtype", dtype, "--n", str(n),
               "--steps", str(steps), "--mode", mode, "--repeats", str(repeats)]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
            last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            rec = json.loads(last) if proc.returncode == 0 and last.startswith("{") else {
                "dtype": dtype, "n": n, "steps": steps, "mode": mode, "error": proc.stderr.strip()[-1500:],
                "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            rec = {"dtype": dtype, "n": n, "steps": steps, "mode": mode, "error": f"timeout {timeout_s} s"}
        rec["wall_s"] = time.perf_counter() - t0
        records.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("dtype", "n", "steps", "mode", "run_s_median",
                                                  "peak_bytes_in_use", "error")}), flush=True)
        out.write_text(json.dumps({"records": records}, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dtype", choices=DTYPES, default="float32")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--mode", choices=("fwd", "grad"), default="grad")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "_temp" / "precision" / "grid.json")
    args = ap.parse_args(argv)
    if args.one:
        print(json.dumps(run_one(args.dtype, args.n, args.steps, args.mode, args.repeats, args.trace_dir)))
        return 0
    if args.sweep:
        sweep(args.out, args.quick, args.repeats, args.timeout)
        return 0
    ap.error("pass --one or --sweep")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
