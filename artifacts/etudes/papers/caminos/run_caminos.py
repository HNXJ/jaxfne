"""L2 run driver for the CAMINOS etude (PREREG_P3.md "Runs").

One ``caminos_jax.simulate`` call per (p, seed); per-step population
counts ``[steps, 3]`` saved as int32 ``.npy`` plus a ``manifest.json``
(git SHA, args, jax version and devices, start time, wall time per run).

Scored runs (any nonzero seed, which covers seeds 1-5 and the R4 p-grid;
seed 0 is excluded from scoring) are refused unless ``--signed`` is
passed, because PREREG_P3.md is still unsigned. Seed 0 and ``--pilot``
(tiny net) run without ``--signed``. G0 single-cell checks and timing
pilots are not scored.

Portable: no host paths are hardcoded (the output directory is an
argument), so this runs unchanged on WSL2 CUDA. Writes are atomic
(temp file + replace). The manifest merges entries across invocations,
keyed by file name, and never drops an earlier run. An existing output is
reused only when its manifest entry matches the request (seed, p, dtype,
dt, T, pilot, sizes); one that does not match is refused, never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent

SCORED_SEEDS = (1, 2, 3, 4, 5)  # prereg Runs: seeds 1-5 for the network and drive
# prereg R4: p = 0.02, 0.03, ..., 0.10, seeds 1-3 each.
R4_P_GRID = tuple(round(0.02 + 0.01 * i, 2) for i in range(9))
DTYPE_CHOICES = ("float32", "float64")


def repo_root():
    """Repository root (directory holding ``pyproject.toml``)."""
    root = _HERE
    while root.parent != root and not (root / "pyproject.toml").exists():
        root = root.parent
    return root


def git_sha():
    """Git SHA via ``git -C <repo root> rev-parse HEAD``."""
    out = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("git rev-parse HEAD failed: %s" % out.stderr.strip())
    return out.stdout.strip()


def is_scored_run(seeds, p, pilot):
    """True when the request is a scored run (needs ``--signed``).

    Scored = any nonzero seed (covers prereg seeds 1-5 and the R4 p-grid
    at seeds 1-3). Seed 0 is excluded from scoring (prereg Prior
    exposure) and ``--pilot`` is a timing/G0-style check, not scored.
    """
    if pilot or all(s == 0 for s in seeds):
        return False
    return True


def build_parser():
    ap = argparse.ArgumentParser(description="Run the CAMINOS reference net (PREREG_P3.md).")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--p", type=float, default=0.07)
    ap.add_argument("--T", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=DTYPE_CHOICES, default="float32")
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="tiny net (NE=80, NPV=8, NSOM=8), not scored")
    ap.add_argument("--indegree", choices=("bernoulli", "fixed"), default="bernoulli",
                    help="S3: exact round(p*N_pre) partners per post cell (default bernoulli)")
    ap.add_argument("--ext", choices=("private", "shared"), default="private",
                    help="S4: shared 560-train drive instead of private sources")
    ap.add_argument("--step-order", choices=("default", "brian2"), default="default",
                    help="S5: 21-step ring and 49-step clamp (default 20 and 50)")
    ap.add_argument("--ext-sources", default="fixed",
                    help="exploratory: 'fixed' (560), 'pNE' (round(p*NE)), or a positive int")
    return ap


def _resolve_ext_sources(value):
    """Normalise ``--ext-sources`` to 'fixed', 'pNE' or a positive int (refused otherwise)."""
    if value in ("fixed", "pNE"):
        return value
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise SystemExit("error: --ext-sources must be 'fixed', 'pNE' or a positive int, "
                         "got %r" % (value,))
    if n <= 0:
        raise SystemExit("error: --ext-sources must be 'fixed', 'pNE' or a positive int, "
                         "got %r" % (value,))
    return n


def run_filename(seed, p, dtype, dt, indegree="bernoulli", ext="private",
                 step_order="default", ext_sources="fixed"):
    """File name carries dtype and dt, so runs at other precision or step never share a file.

    Non-default variants extend the name, so a variant run never shares a file
    with a reference run.
    """
    base = "counts_seed%d_p%g_%s_dt%.0e.npy" % (seed, p, dtype, dt)
    tag = ""
    if indegree != "bernoulli":
        tag += "_indeg-%s" % indegree
    if ext != "private":
        tag += "_ext-%s" % ext
    if step_order != "default":
        tag += "_order-%s" % step_order
    if str(ext_sources) != "fixed":
        tag += "_src-%s" % ext_sources
    return base[:-4] + tag + ".npy" if tag else base


def _reusable(old, seed, args, sizes):
    """True when a manifest entry was made by this request, so its file may be reused.

    The file name carries seed, p, dtype and dt; T, pilot and sizes are not in
    it, so they are compared too. Variant values default for entries written
    before the variants existed.
    """
    return (old.get("seed") == seed
            and abs(float(old.get("p", -1.0)) - args.p) < 1e-12
            and old.get("dtype") == args.dtype
            and abs(float(old.get("dt", -1.0)) - args.dt) < 1e-15
            and abs(float(old.get("T", -1.0)) - args.T) < 1e-12
            and bool(old.get("pilot")) == bool(args.pilot)
            and old.get("sizes") == sizes
            and old.get("indegree", "bernoulli") == getattr(args, "indegree", "bernoulli")
            and old.get("ext", "private") == getattr(args, "ext", "private")
            and old.get("step_order", "default") == getattr(args, "step_order", "default")
            and str(old.get("ext_sources", "fixed")) == str(getattr(args, "ext_sources", "fixed")))


def _write_manifest(out, manifest):
    """Atomic manifest write (temp file + replace)."""
    mtmp = out / "manifest.json.tmp"
    mtmp.write_text(json.dumps(manifest, indent=2))
    os.replace(mtmp, out / "manifest.json")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if is_scored_run(args.seeds, args.p, args.pilot) and not args.signed:
        print("Refusing: PREREG_P3.md is unsigned. Scored runs (seeds 1-5 "
              "or the R4 p-grid) need --signed; seed 0 and --pilot do not.")
        raise SystemExit(2)
    args.ext_sources = _resolve_ext_sources(args.ext_sources)

    import jax
    import numpy as np

    sys.path.insert(0, str(_HERE))
    import caminos_jax as cj

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = getattr(jax.numpy, args.dtype)
    if args.pilot:
        sizes = {"E": 80, "PV": 8, "SOM": 8}
    else:
        sizes = {"E": cj.PARAMS["NE"], "NPV": cj.PARAMS["NPV"], "NSOM": cj.PARAMS["NSOM"]}
        sizes = {"E": sizes["E"], "PV": sizes["NPV"], "SOM": sizes["NSOM"]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc).isoformat()
    try:
        sha = git_sha()
    except RuntimeError:
        sha = "unknown"
    prior = None
    if (out / "manifest.json").exists():
        prior = json.loads((out / "manifest.json").read_text())
    entries = {e["file"]: e for e in prior["runs"]} if prior else {}
    invocations = list(prior.get("invocations", [])) if prior else []
    # Decide every seed before simulating: a file this request cannot match is refused.
    plan = []
    for seed in args.seeds:
        fname = run_filename(seed, args.p, args.dtype, args.dt, args.indegree,
                             args.ext, args.step_order, args.ext_sources)
        fpath = out / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            old = entries.get(fname)
            if old is None or not _reusable(old, seed, args, sizes):
                print("Refusing to overwrite %s: its manifest entry is missing or its "
                      "seed, p, dtype, dt, T, pilot, sizes or variants differ from this request."
                      % fpath)
                raise SystemExit(2)
            plan.append((seed, fname, True))
        else:
            plan.append((seed, fname, False))
    invocation = {"start_time": start_time, "git_sha": sha,
                  "args": {"seeds": list(args.seeds), "p": args.p, "T": args.T,
                           "dt": args.dt, "dtype": args.dtype, "pilot": args.pilot,
                           "signed": bool(args.signed), "indegree": args.indegree,
                           "ext": args.ext, "step_order": args.step_order,
                           "ext_sources": args.ext_sources},
                  "reused": [], "computed": []}
    invocations.append(invocation)
    manifest = {
        "git_sha": sha,
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "start_time": start_time,
        "invocations": invocations,
        "runs": list(entries.values()),
    }
    for seed, fname, reuse in plan:
        if reuse:
            invocation["reused"].append(fname)
            continue
        fpath = out / fname
        t0 = time.time()
        key = jax.random.PRNGKey(seed)
        counts, _final = cj.simulate(key, args.T, dt=args.dt, p=args.p,
                                     dtype=dtype, NE=sizes["E"],
                                     NPV=sizes["PV"], NSOM=sizes["SOM"],
                                     indegree=args.indegree, ext=args.ext,
                                     step_order=args.step_order,
                                     ext_sources=args.ext_sources)
        counts = np.asarray(counts, dtype=np.int32)
        tmp = out / (fname + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, counts)
        os.replace(tmp, fpath)
        entries[fname] = {"seed": seed, "p": args.p, "file": fname,
                          "dtype": args.dtype, "dt": args.dt, "T": args.T,
                          "pilot": bool(args.pilot), "sizes": sizes,
                          "indegree": args.indegree, "ext": args.ext,
                          "step_order": args.step_order,
                          "ext_sources": args.ext_sources,
                          "ext_sources_n": cj.ext_source_count(args.p, sizes["E"],
                                                              args.ext_sources),
                          "git_sha": sha, "start_time": start_time,
                          "wall_s": time.time() - t0,
                          "steps": int(counts.shape[0])}
        invocation["computed"].append(fname)
        manifest["runs"] = list(entries.values())
        _write_manifest(out, manifest)  # after each run, so a crash keeps finished runs
    manifest["runs"] = list(entries.values())
    _write_manifest(out, manifest)
    return manifest


if __name__ == "__main__":
    main()
