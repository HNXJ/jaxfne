"""Seed-mean summary of the PLoS CB 2026 Figs 2-4 network sweeps (run_figA.py, FIGA_PARAM_SET=default).

Usage: python score_plos_figs.py <figP_out dir> <out json>
Reads p2a/silence_pv.json, p2b/silence_som.json, p3a/replace_pv.json, p3b/replace_som.json, p4a/fig6a.json.
Per value: f_pk (25-45 Hz argmax), band power and peak |X| (5-60 Hz) as ratios to the seed's paired value at
fraction 0 (Fig 4A: ratio 0.5), max E rate. Writes the summary with each input's sha256. Exit 0, or 2 on usage.
"""
import hashlib
import json
import os
import sys

import numpy as np

PANELS = [
    ("2A", "p2a/silence_pv.json", "fraction", 0.0),
    ("2B", "p2b/silence_som.json", "fraction", 0.0),
    ("3A", "p3a/replace_pv.json", "fraction", 0.0),
    ("3B", "p3b/replace_som.json", "fraction", 0.0),
    ("4A", "p4a/fig6a.json", "ratio", 0.5),
]


def peak_mag(row, lo=5.0, hi=60.0):
    f = np.asarray(row["psd_freqs_hz"])
    m = np.asarray(row["psd_mag"])
    sel = (f >= lo) & (f <= hi)
    return float(m[sel].max())


def summarize(rows, key, ref):
    base = {r["seed"]: r for r in rows if r[key] == ref}
    out = []
    for v in sorted({r[key] for r in rows}):
        sel = [r for r in rows if r[key] == v]
        fk = np.array([r["f_pk_hz"] for r in sel])
        bp = np.array([r["band_power_25_45"] / base[r["seed"]]["band_power_25_45"] for r in sel])
        pk = np.array([peak_mag(r) / peak_mag(base[r["seed"]]) for r in sel])
        me = np.array([r["max_e_rate_hz"] for r in sel])
        out.append({
            key: v, "n": len(sel),
            "f_pk_hz": [round(float(fk.mean()), 2), round(float(fk.std(ddof=1)), 2)],
            "band_power_ratio": [round(float(bp.mean()), 3), round(float(bp.std(ddof=1)), 3)],
            "peak_mag_ratio": [round(float(pk.mean()), 3), round(float(pk.std(ddof=1)), 3)],
            "max_e_rate_hz": [round(float(me.mean()), 1), round(float(me.std(ddof=1)), 1)],
            # population mean rates over all cells of each type (silenced cells count as 0 Hz)
            "rates_hz": {p: round(float(np.mean([r["rates_Hz"][p] for r in sel])), 3) for p in ("E", "PV", "SOM")},
        })
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    root, out_path = sys.argv[1:3]
    result = {"inputs": {}, "panels": {}}
    for panel, rel, key, ref in PANELS:
        path = os.path.join(root, rel)
        with open(path, "rb") as fh:
            raw = fh.read()
        result["inputs"][rel] = hashlib.sha256(raw).hexdigest()
        d = json.loads(raw)
        assert d["param_set"] == "default", (rel, d["param_set"])
        result["panels"][panel] = summarize(d["rows"], key, ref)
        for s in result["panels"][panel]:
            print(panel, s)
    with open(out_path, "w", newline="\n") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
