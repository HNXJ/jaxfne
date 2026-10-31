"""Paper-scale sweeps for Figures 3, 4 and S3 of the CAMINOS (Cell Reports 2025) replication.

Usage: python run_figA.py <outdir> <fractions csv> <seeds csv> <what>
  what: silence_pv | silence_som | replace_pv | replace_som | s3

Writes <outdir>/<what>.json: one row per (fraction, seed) with scalars and short series.
With replace_*, each row also keeps the LFP magnitude spectrum up to 60 Hz (psd_freqs_hz, psd_mag).
Silencing and replacement use the paired control of each seed (fraction 0 is the baseline).
Exit 1 on an unknown ``what``. Each run is about 170 s on the local GPU (3 seeds x 5 fractions = 15 runs).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_repo_root = Path(__file__).resolve().parents[4]  # P-025: the checkout's jaxfne, never site-packages
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_HERE))

import cellrep_figures as cf  # noqa: E402

assert "site-packages" not in cf.__file__, "run_figA must run the checkout's cellrep_figures"


def scalarize(row):
    out = {}
    for key, val in row.items():
        if isinstance(val, (int, float, str, bool)) or val is None:
            out[key] = val
        elif isinstance(val, np.generic):
            out[key] = val.item()
        elif isinstance(val, list):
            out[key] = val
        elif isinstance(val, np.ndarray) and val.size <= 64:
            out[key] = val.astype(float).tolist()
    return out


def keep_spectrum(rows, fmax_hz=60.0):
    """Keep the LFP magnitude spectrum up to fmax_hz on each row (peak-amplitude check)."""
    for row in rows:
        sp = row.get("spectrum")
        if sp:
            f = np.asarray(sp["freqs_hz"], float)
            m = np.asarray(sp["magnitude"], float)
            assert m.ndim == 1 and m.shape == f.shape, "spectrum must be 1-D over frequency"
            k = f <= fmax_hz
            row["psd_freqs_hz"] = f[k].tolist()
            row["psd_mag"] = m[k].tolist()
    return rows


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    outdir, fractions_csv, seeds_csv, what = sys.argv[1:5]
    fractions = [float(x) for x in fractions_csv.split(",")]
    seeds = [int(x) for x in seeds_csv.split(",")]
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    rows = []
    if what in ("silence_pv", "silence_som"):
        pop = "PV" if what == "silence_pv" else "SOM"
        for s in seeds:
            for f in fractions:
                rows += cf.silence_sweep(pop, [f], [s])
                print("run", what, "seed", s, "fraction", f, flush=True)
    elif what in ("replace_pv", "replace_som"):
        pop = "PV" if what == "replace_pv" else "SOM"
        col = {"E": 0, "PV": 1, "SOM": 2}[pop]
        rates = []
        for s in seeds:
            res = cf.run_cell(s)
            keep = np.asarray(res["counts_kept"])[:, col]
            rates.append(float(keep.sum()) / (int(res["sizes"][pop]) * float(res["dt_s"]) * keep.shape[0]))
        # paired baseline mean rate of the group (Figure 4 protocol, mmc2 1772-1774)
        rate = float(np.mean(rates))
        print("baseline", pop, "rate_hz", round(rate, 4), flush=True)
        for s in seeds:
            for f in fractions:
                rows += keep_spectrum(cf.replace_sweep(pop, [f], [s], rate_hz=rate))
                print("run", what, "seed", s, "fraction", f, flush=True)
    elif what == "s3":
        for s in seeds:
            res = cf.s3_paired(s)
            rows.append({
                "seed": s,
                "onset_s": res["onset_s"],
                "t_window_s": np.asarray(res["t_window_s"]).tolist(),
                "e_rate_control_hz": np.asarray(res["e_rate_control_hz"]).tolist(),
                "e_rate_silenced_hz": np.asarray(res["e_rate_silenced_hz"]).tolist(),
            })
            print("run s3 seed", s, flush=True)
    else:
        raise SystemExit("unknown what: %s" % what)
    payload = {
        "what": what,
        "fractions": fractions,
        "seeds": seeds,
        "elapsed_s": round(time.time() - t0, 1),
        "rows": [scalarize(r) if isinstance(r, dict) else r for r in rows],
    }
    path = os.path.join(outdir, "%s.json" % what)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote", path, "rows", len(rows), "elapsed", payload["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
