"""OCMA stages F-G input: Krippendorff alpha per hypothesis group vs a shuffled baseline, and the consensus table.

Usage: python artifacts/etudes/meta/ocma/tools/agreement.py <council run dir> [--draws 1000] [--seed 0]
The raw ratings live in the ocma repository (HNXJ/ocma, `cases/jaxfne/council/full`); pass that directory.
Reads r?.json (round 1) and r?.rev.json (feedback round). A score that fails validate.py's citation rule is set
to null before anything is computed (counted in the output). Alpha: interval metric over units (paper, factor),
raters as coders, missing = null. Baseline: each rater's paper labels permuted independently (factor kept),
`--draws` times; reported as the 95th percentile. Consensus: per factor the median of the valid feedback-round
scores; groups and action as in glossary.md (J24.1) via summarize.py.
Writes agreement.json and consensus.json to this folder's `council/` (the dashboard reads them there). Exit 0, 2 on missing input.
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize import action, groups  # noqa: E402
from validate import KEYS, PAPER_OK, cite_ok  # noqa: E402

GROUPS = {"H1": KEYS[0:8], "H2": KEYS[8:16], "H3": KEYS[16:24]}


def load(path):
    """{line: {key: value or None}} with uncited non-zero scores set to None; also the count of such."""
    out, dropped = {}, 0
    for p in json.loads(Path(path).read_text(encoding="utf-8"))["papers"]:
        s, ev = p["scores"], p.get("evidence", {})
        clean = {}
        for k in KEYS:
            v = s.get(k)
            if v is not None and v != 0 and k not in PAPER_OK and not cite_ok(ev.get(k)):
                v, dropped = None, dropped + 1
            clean[k] = v
        out[p["line"]] = clean
    return out, dropped


def alpha(matrix):
    """Krippendorff alpha, interval metric. matrix: list of units, each a list of values (None = missing)."""
    units = [[v for v in u if v is not None] for u in matrix]
    units = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in units)
    if n < 2:
        return None
    # sum over ordered pairs i != j of (a_i - a_j)^2 = 2 (m * sum a^2 - (sum a)^2)
    def pairsq(xs):
        return 2 * (len(xs) * sum(x * x for x in xs) - sum(xs) ** 2)
    do = sum(pairsq(u) / (len(u) - 1) for u in units) / n
    vals = [v for u in units for v in u]
    de = pairsq(vals) / (n * (n - 1))
    return None if de == 0 else 1 - do / de


def group_alpha(raters, lines, keys, perm=None):
    m = []
    for line_i, line in enumerate(lines):
        for k in keys:
            row = []
            for r_i, r in enumerate(raters):
                src = lines[perm[r_i][line_i]] if perm else line
                row.append(r.get(src, {}).get(k))
            m.append(row)
    return alpha(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.dir)
    rounds = {"round1": sorted(d.glob("r?.json")), "feedback": sorted(d.glob("r?.rev.json"))}
    if not rounds["round1"] or not rounds["feedback"]:
        print("missing r?.json or r?.rev.json")
        return 2
    rng = random.Random(a.seed)
    report = {"seed": a.seed, "draws": a.draws, "rounds": {}}
    loaded = {}
    for name, files in rounds.items():
        raters, dropped = zip(*(load(f) for f in files))
        loaded[name] = raters
        lines = sorted(set().union(*raters))
        res = {"raters": [f.name for f in files], "papers": len(lines), "uncited_set_null": sum(dropped)}
        for g, keys in GROUPS.items():
            obs = group_alpha(raters, lines, keys)
            null = []
            for _ in range(a.draws):
                perm = [rng.sample(range(len(lines)), len(lines)) for _ in raters]
                v = group_alpha(raters, lines, keys, perm)
                if v is not None:
                    null.append(v)
            null.sort()
            p95 = null[int(0.95 * (len(null) - 1))] if null else None
            res[g] = {"alpha": obs, "shuffle_p95": p95, "shuffle_mean": statistics.fmean(null) if null else None,
                      "above_p95": obs is not None and p95 is not None and obs > p95}
            print(f"{name:8s} {g}: alpha {obs:.3f}  shuffle p95 {p95:.3f}  mean {res[g]['shuffle_mean']:.3f}")
        print(f"{name:8s} uncited scores set null: {sum(dropped)}")
        report["rounds"][name] = res
    (HERE.parent / "council" / "agreement.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    inv = {r["line"]: r for r in json.loads((HERE.parent / "data" / "inventory.json").read_text(encoding="utf-8"))}
    raters = loaded["feedback"]
    cons = []
    for line in sorted(set().union(*raters)):
        med = {}
        for k in KEYS:
            vals = [r[line][k] for r in raters if line in r and r[line][k] is not None]
            med[k] = statistics.median(vals) if vals else None
        h = groups(med)
        cons.append({"line": line, "first_author": inv[line]["first_author"], "year": inv[line]["year"],
                     "section": inv[line]["section"], "on_dashboard": bool(inv[line]["dashboard"]),
                     **{g: (round(v, 3) if v is not None else None) for g, v in h.items()},
                     "action": action(h, bool(inv[line]["dashboard"])), "scores": med})
    (HERE.parent / "council" / "consensus.json").write_text(json.dumps(cons, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {}
    for c in cons:
        counts[c["action"]] = counts.get(c["action"], 0) + 1
    print("consensus actions:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
