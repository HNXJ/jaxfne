"""Score a core-route sweep against the frozen oracle sweep (RESULT_FIGURES.md, "Core-route equivalence").

Both inputs are run_figA.py JSON (``rows``: one per value and seed). Criterion, declared before the core
output was read: per value and metric, the core seed mean lies inside the oracle per-seed [min, max].
Cells outside report their distance from the range in oracle-mean percent and in range widths.
The null is the inside probability when both routes draw from one distribution: the mean of n_core
normal draws inside the [min, max] of n_oracle draws (Monte Carlo, fixed seed).

Usage: python score_core_equivalence.py <oracle json> <core json> --out <json>
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


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Core-route equivalence against the frozen oracle sweep.")
    ap.add_argument("oracle")
    ap.add_argument("core")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
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
