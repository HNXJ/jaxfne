"""GSDR 2026 first-in-process outlier repro (no fitting).

CALIBRATION.md §6.6 (end) and §6.7 (end): twice, the first ``run_trial``
call in a fresh process at g* returned an outlier (once kappa 0.0; once
5.19 Hz, kappa 0.0, empty stimulus window) while later calls in the same
process gave ~21 Hz, kappa ~1.0.

One invocation = one fresh process: it runs K calls (default 3) of
``run_trial`` at g*, stim 0.6, seed 1, and appends one JSON line per call
to the output file. Run many fresh processes (GPU under
``flock /tmp/jaxfne_gpu.lock``, CPU with ``JAX_PLATFORMS=cpu``) and count
outliers per backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN_DIR = _HERE.parent / "run"

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_RUN_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_DIR))

import run_gsdr as rg  # noqa: E402

GSTAR_DEFAULT = ("/mnt/f/warehouse/gsdr/lane4f_runs_2026-10-14/gsdr_fit500/"
                 "g_star_uS_float32.npy")
OUT_DEFAULT = "/mnt/f/warehouse/gsdr/first_call_repro_2026-10-17.jsonl"

STIM_AMP = 0.6
SEED = 1
N_CALLS = 3

T_L_MS, T_R_MS = 500.0, 1000.0


def window_edges(n_steps, t_l_ms, t_r_ms, dt_ms=rg.DT_MS):
    i0 = max(0, min(int(t_l_ms / dt_ms), n_steps - 2))
    i1 = max(i0 + 2, min(int(t_r_ms / dt_ms), n_steps - 1))
    return i0, i1


def build_parser():
    ap = argparse.ArgumentParser(
        description="GSDR first-call outlier repro (no fitting).")
    ap.add_argument("--gstar", default=GSTAR_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--process", type=int, default=0,
                    help="process index recorded on every line")
    ap.add_argument("--calls", type=int, default=N_CALLS,
                    help="run_trial calls in this process")
    ap.add_argument("--stim", type=float, default=STIM_AMP)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main(argv=None):
    import numpy as np

    import jax

    args = build_parser().parse_args(argv)
    with open(args.gstar, "rb") as f:
        gstar = np.asarray(np.load(f), dtype=np.float32)
    expect = (rg.N_AMPA_EDGES + rg.N_GABAA_GLOBAL_EDGES
              + rg.N_GABAA_LOCAL_EDGES)
    if gstar.shape != (expect,):
        raise ValueError("gstar must hold %d reference-uS values, got %s"
                         % (expect, gstar.shape,))
    backend = str(jax.default_backend())
    print("process %d backend %s gstar %s stim %g seed %d calls %d"
          % (args.process, backend, args.gstar, float(args.stim),
             int(args.seed), int(args.calls)), flush=True)

    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        for call in range(int(args.calls)):
            result, blobs = rg.run_trial(
                float(args.stim), g_seed=int(args.seed),
                ou_seed=int(args.seed), g_override=gstar)
            v = np.asarray(blobs["v"], dtype=np.float32)
            spikes = rg.detect_spikes(v)
            n_steps = v.shape[0]
            i0, i1 = window_edges(n_steps, T_L_MS, T_R_MS)
            k_fit = float(rg.kappa_of_onsets(spikes[i0:i1, :]))
            win = spikes[i0:i1, :]
            rates = np.asarray(result["rates_hz"], dtype=np.float64)
            line = {
                "process": int(args.process),
                "call": int(call),
                "backend": backend,
                "stim_amp": float(args.stim),
                "seed": int(args.seed),
                "mean_rate_hz": float(result["mean_rate_hz"]),
                "active_frac": float((rates > 0).mean()),
                "kappa_fit": k_fit,
                "n_spikes_window": int(win.sum()),
                "v_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
                "wall_s": float(result["wall_s"]),
            }
            fh.write(json.dumps(line) + "\n")
            fh.flush()
            print("process %(p)d call %(c)d: rate %(r).2f Hz active "
                  "%(a).2f k_fit %(k).4f n_win %(n)d sha %(s)s wall "
                  "%(w).1fs"
                  % {"p": line["process"], "c": line["call"],
                     "r": line["mean_rate_hz"], "a": line["active_frac"],
                     "k": line["kappa_fit"], "n": line["n_spikes_window"],
                     "s": line["v_sha256"][:12],
                     "w": line["wall_s"]}, flush=True)
    print("process %d: appended %d lines to %s"
          % (args.process, int(args.calls), args.out), flush=True)


if __name__ == "__main__":
    main()
