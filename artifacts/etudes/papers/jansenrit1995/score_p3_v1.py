"""Score Jansen-Rit P3 v1 (prereg/P3_v1_signed.md) from fig3_seed<S>.json receipts.

Usage: python score_p3_v1.py RECEIPT_DIR OUT_JSON [--seeds 10 11 12 13 14]
Exit codes: 0 scored (any verdict), 2 a receipt is missing or lacks a C row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

C_ORDER = (68.0, 128.0, 135.0, 270.0, 675.0, 1350.0)
E1_EXPECTED = ("noise", "alpha", "alpha", "spike-wave", "spike-wave", "noise")
OSC_MIN, SW_SD_MV = 0.4, 5.0


def classify(row: dict) -> str:
    """Prereg classes: noise if osc < 0.4; alpha if 8-13 Hz; spike-wave if < 8 Hz and SD >= 5 mV; else other."""
    if row["osc_index"] < OSC_MIN:
        return "noise"
    if 8.0 <= row["f_peak_hz"] <= 13.0:
        return "alpha"
    if row["f_peak_hz"] < 8.0 and row["eeg_std_mv"] >= SW_SD_MV:
        return "spike-wave"
    return "other"


def score_seed(rec: dict) -> dict:
    cs = [float(r["C"]) for r in rec["rows"]]
    if len(cs) != len(set(cs)):
        raise KeyError("duplicate C rows: %r" % (sorted(cs),))
    rows = {float(r["C"]): r for r in rec["rows"]}
    missing = [c for c in C_ORDER if c not in rows or not rows[c]["finite"]]
    if missing:
        raise KeyError("missing or non-finite C rows: %r" % (missing,))
    classes = [classify(rows[c]) for c in C_ORDER]
    e = {
        "E1": classes == list(E1_EXPECTED),
        "E2": rows[135.0]["osc_index"] > rows[128.0]["osc_index"],
        "E3": rows[675.0]["f_peak_hz"] < rows[270.0]["f_peak_hz"],
        "E4": 5.0 <= rows[135.0]["eeg_mean_mv"] <= 10.0,
    }
    return {"seed": rec["seed"], "git_sha": rec.get("git_sha"), "classes": classes, "criteria": e,
            "pass": all(e.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt_dir", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    a = ap.parse_args(argv)
    seeds, receipts = [], {}
    for s in a.seeds:
        p = a.receipt_dir / ("fig3_seed%d.json" % s)
        if not p.is_file():
            print("missing receipt %s" % p, file=sys.stderr)
            return 2
        b = p.read_bytes()
        receipts[p.name] = hashlib.sha256(b).hexdigest()
        try:
            seeds.append(score_seed(json.loads(b)))
        except KeyError as exc:
            print("%s: %s" % (p.name, exc), file=sys.stderr)
            return 2
    n_pass = sum(r["pass"] for r in seeds)
    out = {"prereg": "artifacts/etudes/papers/jansenrit1995/prereg/P3_v1_signed.md", "seeds": seeds,
           "n_pass": n_pass, "n": len(seeds), "verdict": "PASS" if n_pass >= 4 else "pending",
           "receipt_sha256": receipts}
    a.out_json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for r in seeds:
        print(r["seed"], "PASS" if r["pass"] else "no", r["classes"], r["criteria"])
    print("verdict", out["verdict"], "%d of %d" % (n_pass, len(seeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
