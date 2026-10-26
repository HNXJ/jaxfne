"""Held-out scoring of Fig. 6 estimands (CPU-only, from stored V traces).

Reads stim-trial traces (default: the Step-3 ``bistab_s{4..8}_main`` runs at
the paper/code parameters) and scores model z against the data-label y
(``y_labels.json``, data outside the repo):

- PSD peak Hz (argmax of the 15-60 Hz model PSD),
- gamma (35-60) and beta (13-30) band power (``data_labels.band_powers``),
- mean onset rate (``run_gsdr.firing_rates_hz``),
- kappa (``run_gsdr.kappa_of_onsets``, pairwise Fleiss 5 ms on the stim
  [500, 1000] ms onsets, the same statistic as the y kappa).

Comparison is proxy-to-label: the model PSD is the mean-voltage PSD while
the labels are mean-spike-trace PSDs (PORT_SPEC.md section 9); the table says
so. Per-estimand ``e = |z - y| / |y|``; band is close iff every e <= 0.05
(RESIDUAL_PROTOCOL.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN_DIR = _HERE.parent / "run"

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

Y_LABELS_DEFAULT = "/home/nejath/gsdr_src/y_labels.json"
HELD_OUT_SEEDS = (4, 5, 6, 7, 8)


def score_traces(trace_dirs, y, rg, dl):
    """Score each (seed, dir) stim trial; returns rows of y/z/e."""
    import numpy as np

    grid = np.asarray(y["grid"], dtype=np.float64)
    rows = []
    for seed, d in trace_dirs:
        d = Path(d)
        manifest = json.loads((d / "manifest.json").read_text())
        entry = manifest["trials"][0]
        v = np.load(d / entry["file"])
        psd = np.asarray(entry["psd"], dtype=np.float64)
        rates = np.asarray(entry["rates_hz"], dtype=np.float64)
        spikes = rg.detect_spikes(v)
        n_steps = v.shape[0]
        i0 = min(int(500.0 / rg.DT_MS), n_steps - 2)
        i1 = min(int(1000.0 / rg.DT_MS), n_steps - 1)
        z = {
            "psd_peak_hz": float(grid[int(np.argmax(psd))]),
            "bands": dl.band_powers(psd, grid),
            "mean_rate_hz": float(rates.mean()),
            "active_frac": float((rates > 0).mean()),
            "kappa": float(rg.kappa_of_onsets(spikes[i0:i1, :])),
        }
        yrow = {
            "psd_peak_hz": float(y["stim_peak_hz"]),
            "gamma": float(y["stim_bands"]["gamma"]),
            "beta": float(y["stim_bands"]["beta"]),
            "mean_rate_hz": float(y["stim_rates_hz"]["mean"]),
            "kappa": float(y["kappa_stim"]),
        }
        zrow = {"psd_peak_hz": z["psd_peak_hz"], "gamma": z["bands"]["gamma"],
                "beta": z["bands"]["beta"],
                "mean_rate_hz": z["mean_rate_hz"], "kappa": z["kappa"]}
        erow = {k: abs(zrow[k] - yrow[k]) / abs(yrow[k]) for k in yrow}
        rows.append({"seed": int(seed), "dir": str(d), "y": yrow, "z": zrow,
                     "e": erow, "active_frac": z["active_frac"],
                     "band": "close" if all(v <= 0.05 for v in erow.values()) else "off"})
    return rows


def main(argv=None):
    import numpy as np

    ap = argparse.ArgumentParser(description="Held-out Fig. 6 scoring (CPU).")
    ap.add_argument("--traces-root",
                    default=str(_repo_root / "artifacts" / "_temp" / "gsdr2026"))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(HELD_OUT_SEEDS))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--y-labels", default=Y_LABELS_DEFAULT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_RUN_DIR))
    import run_gsdr as rg
    from fit_gsdr import load_y_labels

    import importlib.util

    dl_path = _RUN_DIR / "data_labels.py"
    spec = importlib.util.spec_from_file_location("gsdr2026_dl_score", dl_path)
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)

    y = load_y_labels(args.y_labels)
    root = Path(args.traces_root)
    dirs = [(s, root / ("bistab_s%d_%s" % (s, args.tag))) for s in args.seeds]
    missing = [str(d) for _, d in dirs if not (d / "manifest.json").exists()]
    if missing:
        raise SystemExit("missing trace dirs: %s" % missing)
    rows = score_traces(dirs, y, rg, dl)
    for r in rows:
        print("seed %(seed)d active %(active_frac).2f band %(band)s | "
              "peak y %(y_psd).2f z %(z_psd).2f e %(e_psd).3f | "
              "gamma y %(y_g).4f z %(z_g).4f e %(e_g).3f | "
              "beta y %(y_b).4f z %(z_b).4f e %(e_b).3f | "
              "rate y %(y_r).2f z %(z_r).2f e %(e_r).3f | "
              "kappa y %(y_k).5f z %(z_k).4f e %(e_k).3f"
              % {"seed": r["seed"], "active_frac": r["active_frac"],
                 "band": r["band"],
                 "y_psd": r["y"]["psd_peak_hz"], "z_psd": r["z"]["psd_peak_hz"],
                 "e_psd": r["e"]["psd_peak_hz"],
                 "y_g": r["y"]["gamma"], "z_g": r["z"]["gamma"],
                 "e_g": r["e"]["gamma"],
                 "y_b": r["y"]["beta"], "z_b": r["z"]["beta"],
                 "e_b": r["e"]["beta"],
                 "y_r": r["y"]["mean_rate_hz"], "z_r": r["z"]["mean_rate_hz"],
                 "e_r": r["e"]["mean_rate_hz"],
                 "y_k": r["y"]["kappa"], "z_k": r["z"]["kappa"],
                 "e_k": r["e"]["kappa"]}, flush=True)
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps({"y_labels": args.y_labels, "rows": rows},
                                  indent=2))
        print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
