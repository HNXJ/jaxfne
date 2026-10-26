#!/usr/bin/env python
"""Smoke-scale throughput harness for jaxfne (arXiv plan L3: C6, S6, F4).

Measures, per network size N, on the public path
``build_laminar_column -> construct -> simulate``:

* compile_s: the first ``simulate`` call, seed 0 (compile plus run), outputs blocked;
* run_s: the median of ``--seeds`` run-only ``simulate`` calls on seeds 1..k,
  never used before in the process; the spikes of consecutive timed runs are
  compared, and ``seed_changes_output`` records whether they differ;
* neuron_steps_per_s: N_units * n_steps / run_s;
* jit: the effective jit decision per call shape (``RuntimeConfig.resolve_jit``);
* ensemble: ``Model.simulate_batch`` over ``--ensemble`` PRNG keys (jax.vmap).
  The first call (base seed 0) is compile plus run; the timed second call uses
  base seed 1000, and its spikes must differ from the first batch. Or the string
  ``"unsupported"`` if the model has no ``simulate_batch``.

``--inventory`` prints whether brian2, nest (pynest) and jaxley import, and
their versions. It installs nothing.

Claim status: smoke receipt only. No signed protocol, no statistics over
trials, no claim about any hardware beyond the one JSON records.

Usage:
    python artifacts/programme/benchmarks/bench_scaling.py --out <path.json>
    python artifacts/programme/benchmarks/bench_scaling.py --inventory
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]

INVENTORY = {
    # label: (import name, candidate distribution names for the version)
    "brian2": ("brian2", ("brian2",)),
    "nest": ("nest", ("nest-simulator", "nest")),
    "jaxley": ("jaxley", ("jaxley", "Jaxley")),
}


def _dist_version(names: tuple[str, ...]) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def inventory() -> dict:
    out = {}
    for label, (module_name, dist_names) in INVENTORY.items():
        entry = {"module": module_name, "imports": False, "version": None, "error": None}
        try:
            module = importlib.import_module(module_name)
            entry["imports"] = True
            entry["version"] = getattr(module, "__version__", None) or _dist_version(dist_names)
        except Exception as exc:  # report only; nothing is installed here
            entry["error"] = f"{type(exc).__name__}: {exc}"
        out[label] = entry
    return out


def _git_sha() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _block(jax, *arrays) -> None:
    jax.block_until_ready(arrays)


def build_model(jtfne, n_neurons: int):
    cfg = (
        jtfne.build_laminar_column("V1", n=n_neurons)
        .emitter(family="izhikevich", preset="cortical_eig")
        .probe(name="laminar_probe", modes=["spikes", "V_m"])
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
    )
    return jtfne.construct(cfg)


def measure_size(jax, jtfne, n_neurons, duration_ms, dt_ms, seeds, ensemble):
    t0 = time.perf_counter()
    model = build_model(jtfne, n_neurons)
    construct_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    sig = jtfne.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=0)
    _block(jax, sig.V_m, sig.spikes, sig.time_ms)
    compile_s = time.perf_counter() - t0

    n_steps = int(sig.V_m.shape[0])
    n_units = int(sig.V_m.shape[1])
    dtype = str(sig.V_m.dtype)
    flags: list[str] = []

    # Seeds 1..k are first used here, so no timed call can repeat the compile call.
    run_times = []
    run_spikes = []
    for seed in range(1, seeds + 1):
        t0 = time.perf_counter()
        sig = jtfne.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
        _block(jax, sig.V_m, sig.spikes, sig.time_ms)
        run_times.append(time.perf_counter() - t0)
        run_spikes.append(np.asarray(sig.spikes))
    run_s = statistics.median(run_times)

    spike_totals = [int(s.sum()) for s in run_spikes]
    if len(run_spikes) < 2:
        seed_changes_output = None
        flags.append("seed check not run: fewer than two timed runs")
    else:
        seed_changes_output = all(
            not np.array_equal(a, b) for a, b in zip(run_spikes, run_spikes[1:])
        )
        if not seed_changes_output:
            flags.append("seed_changes_output false: consecutive timed runs gave equal spikes")
    if spike_totals and min(spike_totals) == 0:
        flags.append("a timed run had zero spikes; the seed check is vacuous for it")

    # Effective jit decision, from the same public method the path calls.
    jit_runtime = jtfne.Simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=0).resolved_runtime
    jit_info = {
        "single_simulate": bool(jit_runtime.resolve_jit(n_steps, n_units, batch=1)),
        "simulate_batch": bool(jit_runtime.resolve_jit(n_steps, n_units, batch=ensemble)),
        "rule": "RuntimeConfig.resolve_jit: jit='auto' is True iff n_steps*n_units*batch > 50000",
    }

    if hasattr(model, "simulate_batch"):
        def batch(base_seed: int):
            sim = jtfne.Simulation(
                duration_ms=duration_ms,
                dt_ms=dt_ms,
                seed=base_seed,
                runtime=jtfne.RuntimeConfig(vmap=True),
            )
            res = model.simulate_batch(sim, n_seeds=ensemble)
            _block(jax, res["V_m"], res["spikes"], res["sources"])
            return res

        t0 = time.perf_counter()
        first = batch(0)
        ens_compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        second = batch(1000)
        ens_run_s = time.perf_counter() - t0
        first_spikes = np.asarray(first["spikes"])
        second_spikes = np.asarray(second["spikes"])
        ens_changes = not np.array_equal(first_spikes, second_spikes)
        if not ens_changes:
            flags.append("ensemble seed_changes_output false: base seed 1000 gave equal spikes")
        ensemble_entry = {
            "method": "Model.simulate_batch (jax.vmap over PRNG keys)",
            "batch_execution_mode": second["metadata"]["batch_execution_mode"],
            "n_seeds": ensemble,
            "compile_base_seed": 0,
            "timed_base_seed": 1000,
            "seed_changes_output": bool(ens_changes),
            "spike_total_first": int(first_spikes.sum()),
            "spike_total_timed": int(second_spikes.sum()),
            "compile_s": ens_compile_s,
            "run_s": ens_run_s,
            "neuron_steps_per_s": ensemble * n_units * n_steps / ens_run_s,
        }
    else:
        ensemble_entry = "unsupported"

    return {
        "size_requested": n_neurons,
        "n_units": n_units,
        "n_steps": n_steps,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "dtype": dtype,
        "construct_s": construct_s,
        "compile_s": compile_s,
        "run_s": run_s,
        "run_seeds": list(range(1, seeds + 1)),
        "run_s_per_seed": run_times,
        "run_spike_totals": spike_totals,
        "seed_changes_output": seed_changes_output,
        "neuron_steps_per_s": n_units * n_steps / run_s,
        "jit": jit_info,
        "ensemble": ensemble_entry,
        "flags": flags,
    }


def run_scaling(args) -> dict:
    import jax
    import jaxfne as jtfne

    jaxfne_file = Path(jtfne.__file__).resolve()
    if REPO_ROOT.resolve() not in jaxfne_file.parents:
        raise SystemExit(f"jaxfne resolves outside the repo root: {jaxfne_file}")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    results = []
    for n in sizes:
        print(f"size {n}: measuring", flush=True)
        entry = measure_size(jax, jtfne, n, args.duration_ms, args.dt_ms, args.seeds, args.ensemble)
        print(
            f"size {n}: compile_s={entry['compile_s']:.3f} run_s={entry['run_s']:.4f} "
            f"neuron_steps_per_s={entry['neuron_steps_per_s']:.4g} "
            f"jit_single={entry['jit']['single_simulate']} seed_changes={entry['seed_changes_output']}",
            flush=True,
        )
        for flag in entry["flags"]:
            print(f"FLAG size {n}: {flag}", flush=True)
        results.append(entry)

    return {
        "schema": "jaxfne-bench-scaling-smoke-v0",
        "claim_status": "smoke_receipt_only",
        "protocol": "none; smoke scale. Scored benchmark needs a signed protocol.",
        "git_sha": _git_sha(),
        "jaxfne_file": str(jaxfne_file),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "devices": [str(d) for d in jax.devices()],
        "dtype": results[0]["dtype"] if results else None,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "args": {
            "sizes": sizes,
            "duration_ms": args.duration_ms,
            "dt_ms": args.dt_ms,
            "seeds": args.seeds,
            "ensemble": args.ensemble,
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", default="100,300,1000", help="comma-separated neuron counts")
    parser.add_argument("--duration-ms", type=float, default=200.0)
    parser.add_argument("--dt-ms", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=2, help="run-only repeats per size")
    parser.add_argument("--ensemble", type=int, default=4, help="seeds in the vmap batch")
    parser.add_argument("--out", type=Path, default=None, help="output JSON path (required unless --inventory)")
    parser.add_argument("--inventory", action="store_true", help="report brian2, nest, jaxley import status")
    args = parser.parse_args(argv)

    if args.inventory:
        print(json.dumps(inventory(), indent=2))
        return 0

    if args.out is None:
        parser.error("--out is required unless --inventory is given")

    report = run_scaling(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out} with {len(report['results'])} size entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
