"""Score a core-route sweep against the frozen oracle sweep (RESULT_FIGURES.md, "Core-route equivalence").

Both inputs are run_figA.py JSON (``rows``: one per value and seed). Criterion, declared before the core
output was read: per value and metric, the core seed mean lies inside the oracle per-seed [min, max].
Cells outside report their distance from the range in oracle-mean percent and in range widths.
The null is the inside probability when both routes draw from one distribution: the mean of n_core
normal draws inside the [min, max] of n_oracle draws (Monte Carlo, fixed seed).

Attribution A1 (random streams; declared in RESULT_FIGURES.md before its runs): ``--pool-oracle`` and
``--pool-core`` add seeds to each route; per value in ``--values`` and metric, the core minus oracle mean
with its 95% Welch interval and p, Holm over the cells. Holm p < 0.05 marks a systematic offset.

Equivalence test (Hamm 2026-11-02, declared in RESULT_FIGURES.md before its runs): ``--tost-margin-pct 5``
runs a Welch TOST per value and metric over the pooled seeds; a cell is equivalent when its 90% interval lies
inside +-5% of the oracle mean, and the routes are equivalent when every cell is.

Usage: python score_core_equivalence.py <oracle json> <core json> --out <json>
       [--pool-oracle <json>... --pool-core <json>... --values 0,0.4,0.7,0.9 [--tost-margin-pct 5]]
Exit codes: 0 scored (whatever the verdict), 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("f_pk_hz", "band_power_25_45", "max_e_rate_hz", "rate_E", "rate_PV", "rate_SOM")
NULL_DRAWS = 200_000
NULL_SEED = 0


def by_value(rows):
    """{value: {metric: [per-seed values]}} from run_figA rows."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = round(float(r.get("fraction", r.get("value"))), 4)
        rec = {"f_pk_hz": r["f_pk_hz"], "band_power_25_45": r["band_power_25_45"],
               "max_e_rate_hz": r["max_e_rate_hz"]}
        rec.update({"rate_" + k: x for k, x in r["rates_Hz"].items()})
        for m in METRICS:
            out[v][m].append(float(rec[m]))
    return out


def seed_range_equivalence(oracle_rows, core_rows):
    """One cell per (value, metric) present in both sweeps."""
    orc, core = by_value(oracle_rows), by_value(core_rows)
    cells = []
    for v in sorted(set(orc) & set(core)):
        for m in METRICS:
            lo, hi = min(orc[v][m]), max(orc[v][m])
            mu = float(np.mean(core[v][m]))
            dist = lo - mu if mu < lo else (mu - hi if mu > hi else 0.0)
            ref = float(np.mean(orc[v][m]))
            cells.append({"value": v, "metric": m, "oracle_min": lo, "oracle_max": hi,
                          "n_oracle": len(orc[v][m]), "core_mean": mu, "n_core": len(core[v][m]),
                          "inside": dist == 0.0, "distance": dist,
                          "distance_pct_of_oracle_mean": 100.0 * dist / abs(ref) if ref else None,
                          "distance_range_widths": dist / (hi - lo) if hi > lo else None})
    return cells


def null_inside_probability(n_oracle, n_core, draws=NULL_DRAWS, seed=NULL_SEED):
    """P(mean of n_core draws in [min, max] of n_oracle draws), one normal for both."""
    rng = np.random.default_rng(seed)
    o = rng.standard_normal((draws, n_oracle))
    mu = rng.standard_normal((draws, n_core)).mean(axis=1)
    return float(np.mean((mu >= o.min(axis=1)) & (mu <= o.max(axis=1))))


def welch_cells(oracle_rows, core_rows, values):
    """Per (value, metric): core minus oracle mean, its 95% Welch interval and two-sided p; Holm-adjusted p."""
    from scipy import stats

    orc, core = by_value(oracle_rows), by_value(core_rows)
    cells = []
    for v in values:
        for m in METRICS:
            o, c = np.asarray(orc[v][m]), np.asarray(core[v][m])
            diff = float(c.mean() - o.mean())
            se2 = o.var(ddof=1) / o.size + c.var(ddof=1) / c.size
            if se2 == 0.0:
                p, lo, hi = (1.0 if diff == 0.0 else 0.0), diff, diff
            else:
                df = se2 ** 2 / ((o.var(ddof=1) / o.size) ** 2 / (o.size - 1)
                                 + (c.var(ddof=1) / c.size) ** 2 / (c.size - 1))
                t = diff / np.sqrt(se2)
                p = float(2.0 * stats.t.sf(abs(t), df))
                q = float(stats.t.ppf(0.975, df)) * np.sqrt(se2)
                lo, hi = diff - q, diff + q
            ref = float(o.mean())
            cells.append({"value": v, "metric": m, "n_oracle": int(o.size), "n_core": int(c.size),
                          "diff": diff, "ci95": [float(lo), float(hi)], "p": p,
                          "diff_pct_of_oracle_mean": 100.0 * diff / abs(ref) if ref else None})
    order = np.argsort([c["p"] for c in cells])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(cells) - rank) * cells[i]["p"]))
        cells[i]["p_holm"] = running
        cells[i]["systematic"] = running < 0.05
    return cells


def tost_cells(oracle_rows, core_rows, values, margin_pct, alpha=0.05):
    """Per (value, metric): Welch TOST with margin margin_pct of the oracle mean.

    Equivalent when the (1 - 2 alpha) Welch interval of core minus oracle lies inside (-margin, +margin),
    i.e. both one-sided p < alpha. Every cell must pass (intersection-union), so no multiplicity correction.
    """
    from scipy import stats

    orc, core = by_value(oracle_rows), by_value(core_rows)
    cells = []
    for v in values:
        for m in METRICS:
            o, c = np.asarray(orc[v][m]), np.asarray(core[v][m])
            diff = float(c.mean() - o.mean())
            ref = float(o.mean())
            margin = margin_pct / 100.0 * abs(ref)
            vo, vc = o.var(ddof=1) / o.size, c.var(ddof=1) / c.size
            se = float(np.sqrt(vo + vc))
            if se == 0.0:
                p_lo = p_hi = 0.0 if abs(diff) < margin else 1.0
                lo = hi = diff
            else:
                df = (vo + vc) ** 2 / (vo ** 2 / (o.size - 1) + vc ** 2 / (c.size - 1))
                p_lo = float(stats.t.sf((diff + margin) / se, df))
                p_hi = float(stats.t.cdf((diff - margin) / se, df))
                q = float(stats.t.ppf(1.0 - alpha, df)) * se
                lo, hi = diff - q, diff + q
            p = max(p_lo, p_hi)
            cells.append({"value": v, "metric": m, "n_oracle": int(o.size), "n_core": int(c.size),
                          "diff": diff, "margin": margin, "ci": [float(lo), float(hi)],
                          "ci_level": 1.0 - 2.0 * alpha, "p_tost": p, "equivalent": p < alpha,
                          "diff_pct_of_oracle_mean": 100.0 * diff / abs(ref) if ref else None,
                          "ci_pct_of_oracle_mean": ([100.0 * lo / abs(ref), 100.0 * hi / abs(ref)]
                                                    if ref else None)})
    return cells


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Core-route equivalence against the frozen oracle sweep.")
    ap.add_argument("oracle")
    ap.add_argument("core")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-oracle", action="append", help="repeatable")
    ap.add_argument("--pool-core", action="append", help="repeatable")
    ap.add_argument("--values", help="comma-separated values for the attribution")
    ap.add_argument("--tost-margin-pct", type=float,
                    help="also run the TOST equivalence test over --values at this margin (percent of oracle mean)")
    a = ap.parse_args(argv)
    if a.tost_margin_pct is not None and not a.values:
        ap.error("--tost-margin-pct needs --pool-oracle, --pool-core and --values")
    if bool(a.pool_oracle) != bool(a.pool_core) or bool(a.pool_oracle) != bool(a.values):
        ap.error("--pool-oracle, --pool-core and --values go together")
    od = json.loads(Path(a.oracle).read_text(encoding="utf-8"))
    cd = json.loads(Path(a.core).read_text(encoding="utf-8"))
    cells = seed_range_equivalence(od["rows"], cd["rows"])
    if not cells:
        ap.error("no value is present in both sweeps")
    n_o = {c["n_oracle"] for c in cells}
    n_c = {c["n_core"] for c in cells}
    p_in = null_inside_probability(min(n_o), min(n_c)) if len(n_o) == len(n_c) == 1 else None
    n_in = sum(c["inside"] for c in cells)
    rec = {"oracle": {"path": a.oracle, "sha256": _sha256(a.oracle)},
           "core": {"path": a.core, "sha256": _sha256(a.core)},
           "criterion": "core seed mean inside the oracle per-seed [min, max], every cell",
           "inside": n_in, "cells_total": len(cells), "equivalent": n_in == len(cells),
           "null_inside_probability": p_in,
           "null_expected_outside": None if p_in is None else len(cells) * (1.0 - p_in),
           "null": f"one normal, {NULL_DRAWS} draws, seed {NULL_SEED}", "cells": cells}
    if a.pool_oracle:
        po = [r for p in a.pool_oracle for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]
        pc = [r for p in a.pool_core for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]
        pools = {"pool_oracle": [{"path": p, "sha256": _sha256(p)} for p in a.pool_oracle],
                 "pool_core": [{"path": p, "sha256": _sha256(p)} for p in a.pool_core]}
        values = [round(float(x), 4) for x in a.values.split(",")]
        if a.tost_margin_pct is not None:
            tc = tost_cells(od["rows"] + po, cd["rows"] + pc, values, a.tost_margin_pct)
            n_eq = sum(c["equivalent"] for c in tc)
            rec["tost"] = {**pools, "margin_pct": a.tost_margin_pct, "alpha": 0.05,
                           "test": "Welch TOST, 90% interval inside +-margin, every cell (intersection-union)",
                           "equivalent_cells": n_eq, "cells_total": len(tc), "equivalent": n_eq == len(tc),
                           "cells": tc}
            for c in tc:
                print(f"TOST {c['value']:<5} {c['metric']:<17} diff {c['diff_pct_of_oracle_mean']:+.2f}% "
                      f"90% CI [{c['ci_pct_of_oracle_mean'][0]:+.2f}, {c['ci_pct_of_oracle_mean'][1]:+.2f}]% "
                      f"p {c['p_tost']:.3g} n {c['n_oracle']}/{c['n_core']} {'EQ' if c['equivalent'] else 'NOT'}")
            print(f"TOST equivalent {n_eq} of {len(tc)} at +-{a.tost_margin_pct}%")
        att = welch_cells(od["rows"] + po, cd["rows"] + pc, values)
        rec["attribution_a1"] = {
            **pools,
            "test": "Welch two-sided, 95% interval, Holm over cells; systematic when p_holm < 0.05",
            "cells": att, "systematic": sum(c["systematic"] for c in att)}
        for c in att:
            print(f"A1 {c['value']:<5} {c['metric']:<17} diff {c['diff']:+.4g} "
                  f"({c['diff_pct_of_oracle_mean']:+.2f}%) CI [{c['ci95'][0]:+.3g}, {c['ci95'][1]:+.3g}] "
                  f"p {c['p']:.3g} holm {c['p_holm']:.3g} n {c['n_oracle']}/{c['n_core']}"
                  f"{'  SYSTEMATIC' if c['systematic'] else ''}")
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    for c in cells:
        tag = "IN " if c["inside"] else "OUT"
        extra = "" if c["inside"] else (f"  d {c['distance_pct_of_oracle_mean']:+.2f}% of mean, "
                                        f"{c['distance_range_widths']:+.2f} widths")
        print(f"{c['value']:<5} {c['metric']:<17} [{c['oracle_min']:.4g}, {c['oracle_max']:.4g}] "
              f"core {c['core_mean']:.4g} {tag}{extra}")
    print(f"inside {n_in} of {len(cells)}; null P(inside) {p_in}; "
          f"null expected outside {rec['null_expected_outside']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
