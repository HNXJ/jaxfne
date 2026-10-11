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
(temp file + replace) and resumable (existing outputs are skipped).
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
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=DTYPE_CHOICES, default="float32")
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="tiny net (NE=80, NPV=8, NSOM=8), not scored")
    return ap


def run_filename(seed, p):
    return "counts_seed%d_p%g.npy" % (seed, p)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if is_scored_run(args.seeds, args.p, args.pilot) and not args.signed:
        print("Refusing: PREREG_P3.md is unsigned. Scored runs (seeds 1-5 "
              "or the R4 p-grid) need --signed; seed 0 and --pilot do not.")
        raise SystemExit(2)

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
    manifest = {
        "git_sha": sha,
        "args": {"seeds": list(args.seeds), "p": args.p, "T": args.T,
                 "dt": args.dt, "dtype": args.dtype, "pilot": args.pilot,
                 "signed": bool(args.signed)},
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "start_time": start_time,
        "dt": args.dt,
        "T": args.T,
        "sizes": sizes,
        "runs": [],
    }
    for seed in args.seeds:
        fname = run_filename(seed, args.p)
        fpath = out / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            manifest["runs"].append({"seed": seed, "p": args.p, "file": fname,
                                     "wall_s": 0.0, "skipped": True})
            continue
        t0 = time.time()
        key = jax.random.PRNGKey(seed)
        counts, _final = cj.simulate(key, args.T, dt=args.dt, p=args.p,
                                     dtype=dtype, NE=sizes["E"],
                                     NPV=sizes["PV"], NSOM=sizes["SOM"])
        counts = np.asarray(counts, dtype=np.int32)
        tmp = out / (fname + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, counts)
        os.replace(tmp, fpath)
        manifest["runs"].append({"seed": seed, "p": args.p, "file": fname,
                                 "wall_s": time.time() - t0, "skipped": False,
                                 "steps": int(counts.shape[0])})
    mpath = out / "manifest.json"
    mtmp = out / "manifest.json.tmp"
    mtmp.write_text(json.dumps(manifest, indent=2))
    os.replace(mtmp, mpath)
    return manifest


if __name__ == "__main__":
    main()
