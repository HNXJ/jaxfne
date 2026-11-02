"""Score the five P3 v4 signed seeds (CONV, R-E, O-DROP; W-IE, AI reported).

Usage: python score_p3.py <receipts dir> <out.json>

Reads the five ``score_receipt*.json`` files (seeds 4-8) under the receipts
dir, refuses a missing, duplicate, non-signed or mixed-SHA set, and writes
the verdict: the lane "closes" only when CONV (P3 v4 signed, Change from v3:
least-squares slope b of the per-second E rate over 130-150 s, not converged
only when |b| >= 0.01 Hz/s and |b|/se > 2.101) holds on every seed, R-E
(P3 v2: e = |z - 3 Hz| / 3 Hz <= 0.20, z the seed mean of the 149-150 s E
rate) holds, and O-DROP (P3 v2: the seed-mean 0.8-1.0 s static rate over the
seed-mean z, ratio >= 2) holds. W-IE and AI are reported, never in the
verdict (P3 v2/v3 secondary rows).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from artifacts.etudes.papers.vogels2011 import vogels2011_api as L

SCORED_SEEDS = (4, 5, 6, 7, 8)
N_PLASTIC_S = 150


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refuse(msg, *args):
    raise SystemExit("error: " + (msg % args))


def _load_receipts(d):
    d = Path(d)
    files = sorted(d.rglob("score_receipt*.json"))
    if not files:
        _refuse("no score_receipt*.json under %s", d)
    recs = []
    for f in files:
        try:
            recs.append((f, json.loads(f.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            _refuse("unreadable receipt %s (%s)", f, exc)
    return recs


def score_dir(d):
    """Reduce the five signed receipts under ``d`` to the lane verdict."""
    recs = _load_receipts(d)
    for f, r in recs:
        if r.get("status") != "signed":
            _refuse("non-signed receipt %s (status %r)", f, r.get("status"))
    seeds = sorted(int(r["seed"]) for _, r in recs)
    if seeds != list(SCORED_SEEDS):
        _refuse("need seeds 4-8 exactly once, got %r", seeds)
    shas = {r.get("prereg_sha256") for _, r in recs}
    if len(shas) != 1 or None in shas:
        _refuse("mixed or missing prereg_sha256 %r", sorted(map(str, shas)))
    by_seed = {}
    for _, r in recs:
        by_seed[int(r["seed"])] = r
    conv_seeds = {}
    for s in SCORED_SEEDS:
        rates = by_seed[s].get("e_rate_per_s_hz")
        if not isinstance(rates, list) or len(rates) != N_PLASTIC_S:
            _refuse("seed %d needs %d one-second bins for 130-150 s, got %r",
                    s, N_PLASTIC_S,
                    len(rates) if isinstance(rates, list) else rates)
        fit = L.conv_fit(rates)
        conv_seeds[str(s)] = {"b_hz_s": fit["b"], "se_hz_s": fit["se"],
                              "t": fit["t"], "t_crit": fit["t_crit"],
                              "converged": fit["converged"]}
    conv_pass = bool(all(v["converged"] for v in conv_seeds.values()))
    z = {s: float(by_seed[s]["R-E"]["z_hz"]) for s in SCORED_SEEDS}
    z_mean = sum(z.values()) / len(z)
    e = abs(z_mean - L.R_E_Y_HZ) / L.R_E_Y_HZ
    re_pass = bool(e <= L.R_E_BAND)
    before = {s: float(by_seed[s]["O-DROP"]["before_hz"]) for s in SCORED_SEEDS}
    before_mean = sum(before.values()) / len(before)
    if z_mean > 0.0:
        ratio = before_mean / z_mean
    elif before_mean > 0.0:
        ratio = float("inf")
    else:
        ratio = float("nan")
    od_pass = bool(ratio >= L.O_DROP_RATIO)
    off = (([] if conv_pass else ["CONV"]) + ([] if re_pass else ["R-E"])
           + ([] if od_pass else ["O-DROP"]))
    verdict = {
        "verdict": "closes" if not off else "off (%s)" % "+".join(off),
        "prereg": L.PREREG_SIGNED_REL,
        "prereg_sha256": next(iter(shas)),
        "seeds": list(SCORED_SEEDS),
        "CONV": {"criterion": ("every scored seed converged: not converged "
                               "only when |b| >= 0.01 Hz/s and |b|/se > "
                               "2.101 (P3 v4 signed)"),
                 "per_seed": conv_seeds, "pass": conv_pass},
        "R-E": {"criterion": "e <= %.2f (P3 v2)" % L.R_E_BAND,
                "z_hz": z_mean, "e": e, "pass": re_pass,
                "per_seed_z_hz": {str(s): z[s] for s in SCORED_SEEDS}},
        "O-DROP": {"criterion": "ratio >= %.1f (P3 v2)" % L.O_DROP_RATIO,
                   "before_hz": before_mean, "z_hz": z_mean,
                   "ratio": ratio, "pass": od_pass},
        "W-IE": {"per_seed_nS": {str(s): float(
            by_seed[s]["W-IE"]["w_ie_final_mean_nS"]) for s in SCORED_SEEDS}},
        "AI": {"per_seed": {str(s): by_seed[s].get("AI")
                            for s in SCORED_SEEDS}},
        "inputs": {str(f.relative_to(d)) if _is_under(f, d) else str(f):
                   _sha256_file(f) for f, _ in recs},
    }
    verdict["W-IE"]["mean_nS"] = (
        sum(verdict["W-IE"]["per_seed_nS"].values())
        / len(verdict["W-IE"]["per_seed_nS"]))
    return verdict


def _is_under(f, d):
    try:
        f.relative_to(d)
        return True
    except ValueError:
        return False


def main(argv=None):
    """Entry: ``score_p3.py <receipts dir> <out.json>``."""
    import argparse

    p = argparse.ArgumentParser(description="score the P3 v4 signed seeds")
    p.add_argument("receipts_dir")
    p.add_argument("out_json")
    args = p.parse_args(argv)
    verdict = score_dir(args.receipts_dir)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True),
                   encoding="utf-8")
    print("%s | CONV %s R-E e=%.3f O-DROP ratio=%.3f" % (
        verdict["verdict"], verdict["CONV"]["pass"],
        verdict["R-E"]["e"], verdict["O-DROP"]["ratio"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
