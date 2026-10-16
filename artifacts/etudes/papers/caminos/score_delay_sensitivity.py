"""Score the declared delay sensitivity (option a: synaptic and drive delay 0.1 ms = 1 step).

Reads count arrays only. The sensitivity runs (seeds 1-8, one code SHA, one directory) are scored
next to the primary 2 ms runs with the same functions, so the two rows compare like with like:
the primary scored seeds 1-3 and held-out seeds 4-8 come from the P6 v2 receipts and the calib
directory, as in score_lag_ref42.py. Per seed: the ref. 42 delay at +-5 Hz (declared) and +-3, +-8 Hz
(sensitivity), the P3 cross-correlation lag (caminos_analysis.analyze_run), f_pk on the E rate
(port R1 definition) and on the LFP proxy (ref. 42), population rates, and the oscillation call.
Groups: seeds 1-3 and 4-8. Each lag is compared with y = 8.5 ms (PLoS CB) and 7 ms (preprint,
dispatcher reading). Writes one JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

import caminos_analysis as an
import caminos_api as ca
import score_lag_ref42 as s

SIZES = {"E": ca.NE, "PV": ca.NPV, "SOM": ca.NSOM}
SENS_DELAY_MS = 0.1  # declared: one step at dt = 0.1 ms (the core minimum)
SENS_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
GROUPS = (("seeds_1_3", (1, 2, 3)), ("seeds_4_8", (4, 5, 6, 7, 8)))
SENS_DIR = s.CAL_DIR / "sens_delay0p1_p007"


def seed_record(path):
    """One run through the ref. 42 record (f_pk on the LFP, delays per band) and analyze_run."""
    counts = np.load(path)
    rec = s.run_record(path)
    a = an.analyze_run(counts, s.DT_S, SIZES)
    rec.update({
        "f_peak_E_Hz": a["peak_freq_E_Hz"],
        "oscillates": a["oscillates"],
        "lag_P3_ms": a["lag_PV_SOM_ms"],
        "rates_Hz": a["rates_Hz"],
        "spikes_kept": a["spikes_kept"],
        "undefined": a["undefined"],
    })
    return rec


def check_manifest(d, want_delay_ms):
    """Every seed present once, at the declared delay, one code SHA."""
    runs = json.loads((Path(d) / "manifest.json").read_text())["runs"]
    seeds = sorted(r["seed"] for r in runs)
    if seeds != sorted(SENS_SEEDS):
        raise SystemExit("error: %s holds seeds %r, expected %r" % (d, seeds, list(SENS_SEEDS)))
    bad = [r["seed"] for r in runs if abs(float(r["delay_ms"]) - want_delay_ms) > 1e-12]
    if bad:
        raise SystemExit("error: %s has runs at another delay (seeds %r)" % (d, bad))
    return {"git_sha": sorted({r["git_sha"] for r in runs}), "seeds": seeds,
            "delay_ms": want_delay_ms}


def group_summary(recs):
    out = {}
    for b in s.BANDS_HZ:
        for y_name, y in (("y8.5", s.Y_DECLARED_MS), ("y7", s.Y_CONTEXT_MS)):
            out["ref42_half%g|%s" % (b, y_name)] = s.summarize(
                [r["delay_ms_half%g" % b] for r in recs], y)
    for y_name, y in (("y8.5", s.Y_DECLARED_MS), ("y7", s.Y_CONTEXT_MS)):
        out["P3|%s" % y_name] = s.summarize([r["lag_P3_ms"] for r in recs], y)
    rates = {p: float(np.mean([r["rates_Hz"][p] for r in recs])) for p in ("E", "PV", "SOM")}
    out["stats"] = {
        "n_seeds": len(recs),
        "f_peak_E_Hz_per_seed": [r["f_peak_E_Hz"] for r in recs],
        "f_peak_LFP_Hz_per_seed": [r["f_peak_Hz"] for r in recs],
        "rates_Hz_mean": rates,
        "rates_Hz_per_seed": [r["rates_Hz"] for r in recs],
        "oscillating_seeds": int(sum(bool(r["oscillates"]) for r in recs)),
        "oscillates_all": bool(recs) and all(bool(r["oscillates"]) for r in recs),
        "lag_P3_ms_per_seed": [r["lag_P3_ms"] for r in recs],
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the declared 0.1 ms delay sensitivity.")
    ap.add_argument("--sens-dir", default=str(SENS_DIR))
    ap.add_argument("--p6-dir", default=str(s.P6_DIR), help="P6 v2 receipts (seeds 1-3)")
    ap.add_argument("--heldout-dir", default=str(s.HELDOUT_DIR), help="primary held-out seeds 4-8")
    ap.add_argument("--out", default=None, help="JSON path (default: <cal-dir>/sens_delay0p1_score.json)")
    args = ap.parse_args(argv)
    sens_dir, p6_dir, held_dir = Path(args.sens_dir), Path(args.p6_dir), Path(args.heldout_dir)
    out_path = Path(args.out) if args.out else Path(args.sens_dir).parent / "sens_delay0p1_score.json"

    manifest = check_manifest(sens_dir, SENS_DELAY_MS)
    sens = {}
    for label, seeds in GROUPS:
        sens[label] = [seed_record(s.seed_path(sens_dir, k)) for k in seeds]
    primary = {}
    primary["seeds_1_3"] = [seed_record(s.seed_path(p6_dir, k)) for k in (1, 2, 3)]
    primary["seeds_4_8"] = [seed_record(s.seed_path(held_dir, k)) for k in (4, 5, 6, 7, 8)]

    doc = {
        "sensitivity": {"synaptic_delay_ms": SENS_DELAY_MS, "drive_delay_ms": SENS_DELAY_MS,
                        "steps_at_dt_0p1": 1, "p": s.P_REF, "T_s": 12.0,
                        "manifest": manifest, "dir": str(sens_dir)},
        "primary": {"synaptic_delay_ms": 2.0, "dir": "P6 v2 receipts (1-3), calib heldout (4-8)"},
        "declared_bands_Hz": list(s.BANDS_HZ),
        "y_ms": {"declared_PLoS_CB": s.Y_DECLARED_MS, "context_preprint": s.Y_CONTEXT_MS},
        "sensitivity_summary": {lab: group_summary(recs) for lab, recs in sens.items()},
        "primary_summary": {lab: group_summary(recs) for lab, recs in primary.items()},
        "sensitivity_per_seed": {lab: recs for lab, recs in sens.items()},
        "primary_per_seed": {lab: recs for lab, recs in primary.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, default=float)
        fh.write("\n")
    for lab in ("seeds_1_3", "seeds_4_8"):
        for tag, summ in (("sens", doc["sensitivity_summary"][lab]), ("prim", doc["primary_summary"][lab])):
            st = summ["stats"]
            print(lab, tag, "P3 mean_ms", summ["P3|y8.5"]["mean_ms"],
                  "osc", st["oscillating_seeds"], "/", st["n_seeds"])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
