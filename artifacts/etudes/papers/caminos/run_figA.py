"""Paper-scale sweeps for the CAMINOS (Cell Reports 2025) replication: Figures 2, 3, 4, 5, 6A and S3.

Usage: python run_figA.py <outdir> <values csv> <seeds csv> <what>
  what: silence_pv | silence_som | replace_pv | replace_som | s3
        | fig2a | fig2b | fig6a | fig5a | fig5b | fig5c
  values: fractions (silence/replace), Hz (fig2a, fig2b), ratios (fig6a), nS (fig5a, fig5b);
          ignored for s3 and fig5c.
Env FIGA_T_S sets the run length in seconds (default 12; smoke runs use 3).

Writes <outdir>/<what>.json: one row per (value, seed) with scalars and short series.
Rows that carry a spectrum keep its magnitude up to 60 Hz (psd_freqs_hz, psd_mag).
Silencing and replacement use the paired control of each seed (fraction 0 is the baseline).
Each run is about 170 s on the local GPU at T_s 12 (3 seeds x 5 values = 15 runs per sweep).
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


_SKIP = object()  # marks a value too large to store (a full LFP or count trace)


def plain(val):
    """JSON-ready copy of a result value; arrays over 4096 elements are skipped."""
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist() if val.size <= 4096 else _SKIP
    if isinstance(val, dict):
        kept = {str(k): plain(v) for k, v in val.items()}
        return {k: v for k, v in kept.items() if v is not _SKIP}
    if isinstance(val, (list, tuple)):
        return [v for v in (plain(x) for x in val) if v is not _SKIP]
    return val


def scalarize(row):
    kept = {k: plain(v) for k, v in row.items()}
    return {k: v for k, v in kept.items() if v is not _SKIP}


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
    outdir, values_csv, seeds_csv, what = sys.argv[1:5]
    values = [float(x) for x in values_csv.split(",")]
    seeds = [int(x) for x in seeds_csv.split(",")]
    t_s = float(os.environ.get("FIGA_T_S", "12"))
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    rows = []
    if what in ("silence_pv", "silence_som"):
        pop = "PV" if what == "silence_pv" else "SOM"
        for s in seeds:
            for f in values:
                rows += cf.silence_sweep(pop, [f], [s], T_s=t_s)
                print("run", what, "seed", s, "fraction", f, flush=True)
    elif what in ("replace_pv", "replace_som"):
        pop = "PV" if what == "replace_pv" else "SOM"
        col = {"E": 0, "PV": 1, "SOM": 2}[pop]
        rates = []
        for s in seeds:
            res = cf.run_cell(s, T_s=t_s)
            keep = np.asarray(res["counts_kept"])[:, col]
            rates.append(float(keep.sum()) / (int(res["sizes"][pop]) * float(res["dt_s"]) * keep.shape[0]))
        # paired baseline mean rate of the group (Figure 4 protocol, mmc2 1772-1774)
        rate = float(np.mean(rates))
        print("baseline", pop, "rate_hz", round(rate, 4), flush=True)
        for s in seeds:
            for f in values:
                rows += keep_spectrum(cf.replace_sweep(pop, [f], [s], rate_hz=rate, T_s=t_s))
                print("run", what, "seed", s, "fraction", f, flush=True)
    elif what in ("fig2a", "fig2b", "fig6a", "fig5a", "fig5b"):
        # one value at a time, so each finished run is on disk (rows stream to the log)
        for v in values:
            if what == "fig2a":
                res = cf.fig2a_sweep(rates_hz=[v], seeds=seeds, T_s=t_s)
            elif what == "fig2b":
                res = cf.fig2b_sweep(tdrates_hz=[v], seeds=seeds, T_s=t_s)
            elif what == "fig6a":
                res = cf.fig6a_sweep(ratios=[v], seeds=seeds, T_s=t_s)
            else:
                kind = "PV_SOM" if what == "fig5a" else "SOM_SOM"
                res = cf.coupling_sweep(kind, values_nS=[v], seeds=seeds, T_s=t_s)
            rows += keep_spectrum(res)
            print("run", what, "value", v, "rows", len(res), flush=True)
    elif what == "fig5c":
        for s in seeds:
            res = cf.fig5c_swap(s, T_s=t_s)
            rows += keep_spectrum([res])
            print("run fig5c seed", s, flush=True)
    elif what == "s3":
        for s in seeds:
            res = cf.s3_paired(s, T_s=t_s)
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
        "values": values,
        "seeds": seeds,
        "T_s": t_s,
        "elapsed_s": round(time.time() - t0, 1),
        "rows": [scalarize(r) for r in rows],
    }
    path = os.path.join(outdir, "%s.json" % what)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote", path, "rows", len(rows), "elapsed", payload["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
