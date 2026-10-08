"""Gate K1h-nf (README): H-state rule against the float32 noise floor of pcl_stdp.

Reads confirm/k1nf_{base,nudge,h}_seed{13,14,15}.json and prints the verdict.
"""
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent / "confirm"
SEEDS = (13, 14, 15)
FIELDS = [("pcl", "simple"), ("pcl", "complex"), ("pcl", "both"),
          ("no_inh", "simple"), ("no_inh", "complex"), ("no_inh", "both"), ("random", "simple")]


def verdicts(r):
    """A1-A3 as in Gate K1 (A3 on simple-cell decoding)."""
    osi_t, osi_u = r["osi_trained_median"], r["osi_untrained_median"]
    a1 = osi_t is not None and osi_t >= 0.3 and (osi_u is None or osi_t > osi_u)
    a2 = r["spikes"]["pcl"][0] <= 0.9 * r["spikes"]["no_inh"][0]
    a3 = r["acc"]["pcl"]["simple"] > r["acc"]["random"]["simple"]
    return a1, a2, a3


def gate(runs):
    """runs[seed][arm] -> result dict; returns (verdict, T, max |H - base|, per-seed verdict match)."""
    def diffs(arm):
        return [abs(runs[s][arm]["acc"][g][c] - runs[s]["base"]["acc"][g][c]) for s in SEEDS for g, c in FIELDS]

    t, dh = max(diffs("nudge")), max(diffs("h"))
    same = {s: verdicts(runs[s]["h"]) == verdicts(runs[s]["base"]) for s in SEEDS}
    if t == 0:
        return "ERROR", t, dh, same
    return ("PASS" if dh <= t and all(same.values()) else "FAIL"), t, dh, same


def main():
    runs = {s: {a: json.loads((D / f"k1nf_{a}_seed{s}.json").read_text()) for a in ("base", "nudge", "h")}
            for s in SEEDS}
    for s in SEEDS:
        assert runs[s]["nudge"]["params"]["nudge_ulp"] == 1 and runs[s]["base"]["params"]["nudge_ulp"] == 0, s
        assert runs[s]["h"]["params"]["rule"] == "pcl_stdp_h2" and runs[s]["base"]["params"]["rule"] == "pcl_stdp", s
    v, t, dh, same = gate(runs)
    for s in SEEDS:
        print(f"seed {s}: A1-A3 base {verdicts(runs[s]['base'])} h {verdicts(runs[s]['h'])} match={same[s]}")
    print(f"T (noise floor) = {t:.4f}; max |H - base| = {dh:.4f}")
    print(f"Gate K1h-nf: {v}")
    return 0 if v == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
