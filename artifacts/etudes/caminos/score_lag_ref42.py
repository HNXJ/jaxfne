"""Re-score the PV->SOM lag with the ref. 42 phase method (dispatcher ruling, 2026-10-13).

Reads count arrays only (``counts_seed<s>_p0.07_float32_dt1e-04.npy``, per-step [E | PV | SOM]
spike counts from the core route). Scored seeds 1-3 come from the P6 v2 receipts (code a5539d34).
Held-out seeds 4-8 come from one calib directory (one code SHA). Seeds 1, 4 and 5 are also
compared with the receipts as an identity check. Writes one JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

import caminos_analysis as an
import caminos_api as ca

DT_S = 1e-4
P_REF = 0.07
Y_DECLARED_MS = 8.5  # PLoS CB Results: "around 8.5 ms" (declared target)
Y_CONTEXT_MS = 7.0  # context only: preprint "about 7 ms" (dispatcher reading, not checked by this lane)
DECLARED_HALF_BAND_HZ = 5.0
SENSITIVITY_HALF_BANDS_HZ = (3.0, 8.0)
BANDS_HZ = (DECLARED_HALF_BAND_HZ,) + SENSITIVITY_HALF_BANDS_HZ
BAND_CLOSE_E = 0.05  # protocol: close when e <= 0.05
P6_DIR = Path("F:/warehouse/caminos/p6_scored_v2")
CAL_DIR = Path("F:/warehouse/caminos/calib")
HELDOUT_DIR = CAL_DIR / "heldout_p007"
IDENTITY_DIR = CAL_DIR / "identity_p007"


def seed_path(root, seed):
    return Path(root) / ("counts_seed%d_p%g_float32_dt1e-04.npy" % (int(seed), P_REF))


def run_record(path):
    """f_pk of the LFP proxy, then the delay at each band."""
    counts = np.load(path)
    lfp = ca.ref42_lfp(counts, DT_S)
    f_pk = float(an.spectral_peak(lfp, DT_S)["peak_freq_Hz"])
    rec = {"file": path.name, "f_peak_Hz": f_pk}
    for b in BANDS_HZ:
        r = ca.ref42_phase_delay(counts, DT_S, f_pk, half_band_hz=b)
        rec["delay_ms_half%g" % b] = r["delay_ms"]
        if b == DECLARED_HALF_BAND_HZ:
            rec["n_spikes_PV"] = r["n_spikes_PV"]
            rec["n_spikes_SOM"] = r["n_spikes_SOM"]
    return rec


def summarize(delays_ms, y_ms):
    """Mean over seeds, e = |mean - y| / y, and the protocol band. Any undefined seed: 'undefined'."""
    vals = list(delays_ms)
    if any(v is None for v in vals):
        return {"mean_ms": None, "y_ms": y_ms, "e": None, "band": "undefined"}
    mean = sum(vals) / len(vals)
    e = abs(mean - y_ms) / abs(y_ms)
    return {"mean_ms": mean, "y_ms": y_ms, "e": e,
            "band": "close" if e <= BAND_CLOSE_E else "off"}


def same_counts(a, b):
    return bool(np.array_equal(np.load(a), np.load(b)))


def manifest_sha(d):
    return json.loads((Path(d) / "manifest.json").read_text())["git_sha"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-score the lag with the ref. 42 phase method.")
    ap.add_argument("--p6-dir", default=str(P6_DIR), help="P6 v2 receipts (code a5539d34)")
    ap.add_argument("--cal-dir", default=str(CAL_DIR), help="calib root with identity_p007/ and heldout_p007/")
    ap.add_argument("--out", default=None, help="JSON path (default: <cal-dir>/lag_ref42_p007.json)")
    args = ap.parse_args(argv)
    p6_dir = Path(args.p6_dir)
    cal_dir = Path(args.cal_dir)
    heldout_dir = cal_dir / "heldout_p007"
    identity_dir = cal_dir / "identity_p007"
    out_path = Path(args.out) if args.out else cal_dir / "lag_ref42_p007.json"

    scored = [run_record(seed_path(p6_dir, s)) for s in (1, 2, 3)]
    heldout = [run_record(seed_path(heldout_dir, s)) for s in (4, 5, 6, 7, 8)]
    identity = {
        "seed1_calib_vs_p6v2": same_counts(seed_path(identity_dir, 1), seed_path(p6_dir, 1)),
        "seed4_calib_vs_p6v2": same_counts(seed_path(heldout_dir, 4), seed_path(p6_dir, 4)),
        "seed5_calib_vs_p6v2": same_counts(seed_path(heldout_dir, 5), seed_path(p6_dir, 5)),
    }
    summaries = {}
    for label, recs in (("scored_seeds_1_3", scored), ("heldout_seeds_4_8", heldout)):
        for b in BANDS_HZ:
            for y_name, y in (("declared_8.5ms", Y_DECLARED_MS), ("context_7.0ms", Y_CONTEXT_MS)):
                key = "%s|half%g|%s" % (label, b, y_name)
                summaries[key] = summarize([r["delay_ms_half%g" % b] for r in recs], y)
    doc = {
        "method": "ref. 42 phase method (dispatcher ruling 2026-10-13); band and filter declared",
        "declared": {"drop_s": ca.REF42_DROP_S, "keep_s": ca.REF42_KEEP_S,
                     "lfp_sigma_ms": ca.REF42_LFP_SIGMA_MS,
                     "butter_order": ca.REF42_BUTTER_ORDER, "half_band_Hz": DECLARED_HALF_BAND_HZ,
                     "sensitivity_half_band_Hz": list(SENSITIVITY_HALF_BANDS_HZ),
                     "f_peak": "caminos_analysis.spectral_peak on the LFP window"},
        "code_sha": {"scored_p6_v2_receipts": manifest_sha(p6_dir),
                     "heldout_calib": manifest_sha(heldout_dir)},
        "scored": scored, "heldout": heldout, "identity": identity, "summary": summaries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    for key, s in summaries.items():
        print("%s mean=%s e=%s band=%s" % (key, s["mean_ms"], s["e"], s["band"]))
    print("identity", identity)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
