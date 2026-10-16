"""T2 scorer of the signed PCL Fig. 3 preregistration v5.

Implements exactly ``artifacts/etudes/pcl/prereg/P3_fig3_v5.md`` sections
"Reference curves", "T2 procedure" and "T2 criterion". Synthetic data only:
nothing here may be run on the scored outputs until the preregistration is
signed; the CLI refuses ``F:/warehouse/pcl/runs/*`` unless ``--signed`` is
passed (same guard as fig3_analysis.main).

This file does not import jaxfne (repo-root guard: nothing to guard).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fig3_analysis as fa

# T2 criterion constants (P3_fig3_v5.md "T2 criterion").
EPS = 0.04
Z = 2.0
FRAC = 0.8
GUARD = 0.5

# Phase sets (P3_fig3_v5.md A1; default is the even phases).
EVEN_PHASES = (0, 2, 4, 6, 8)
ALL_PHASES = tuple(range(10))
FIRST5_PHASES = (0, 1, 2, 3, 4)

PANEL_SIMPLE = "c_simple"
PANEL_COMPLEX = "d_complex"


def reference_path():
    """Path of the digitized reference (P3_fig3_v5.md "Reference curves")."""
    return _HERE.parent / "reference" / "fig3_freq_digitized.json"


def load_reference(path=None):
    """The digitized Fig. 3c,d reference JSON as a dict."""
    with open(reference_path() if path is None else path, encoding="utf-8") as fh:
        return json.load(fh)


def _key(x):
    """Bin key: round(K/k, 1), formatted for stable JSON roundtrips."""
    return f"{float(np.round(x, 1)):.1f}"


def fit_prefs(counts, phases=EVEN_PHASES):
    """Preferred (ori*, sf*) per cell (P3_fig3_v5.md T2 steps 1-2).

    ``counts`` is int64 [ori 8, sf 18, phase 10, trial 2, cell] as returned
    by :func:`fig3_analysis._load_grid`. r is the trial mean restricted to
    ``phases``; m is r meaned over those phases. Per sf the max over ori is
    taken; sf* is the argmax over sf (first index on ties, np.argmax) and
    ori* the argmax over ori at sf* (first on ties). This two-step rule is
    not a flattened global argmax: ties across (ori, sf) resolve to the
    smallest sf first, then the smallest ori at that sf.
    """
    ph = [int(p) for p in phases]
    r = np.asarray(counts, dtype=np.float64).mean(axis=3)[:, :, ph, :]
    m = r.mean(axis=2)  # [ori, sf, cell]
    f_star = np.argmax(m.max(axis=0), axis=0).astype(int)
    n = m.shape[2]
    o_star = np.empty(n, dtype=int)
    for i in range(n):
        o_star[i] = int(np.argmax(m[:, int(f_star[i]), i]))
    return o_star, f_star


def tuning_curve(counts, kind, phases=EVEN_PHASES):
    """Pooled normalized sf curve of one cell set (P3_fig3_v5.md T2 steps 3-6).

    Simple cells use phase* (argmax over ``phases`` of r at (ori*, sf*),
    first on ties) only; complex cells average all of ``phases``. Per cell
    g(i) = g_raw(i) / g_raw(sf*); a cell with g_raw(sf*) == 0 is skipped
    and counted. x(i) = (20 - sf*) / (20 - i); pooled over cells, binned by
    round(x, 1), keys 0.4..2.0 kept, value is the mean, SE is std(ddof=0)
    over sqrt(n). Returns ``(curve, n_skipped)`` with curve mapping key to
    ``{"mean", "se", "n"}``.
    """
    if kind not in ("simple", "complex"):
        raise ValueError(f"unknown kind {kind!r}")
    ph = [int(p) for p in phases]
    c = np.asarray(counts, dtype=np.float64)
    n_cells = c.shape[4]
    r = c.mean(axis=3)  # trial mean, [ori, sf, phase, cell]
    rph = r[:, :, ph, :]
    o_star, f_star = fit_prefs(counts, phases)
    bins: dict[str, list[float]] = {}
    n_skipped = 0
    for i in range(n_cells):
        o, f = int(o_star[i]), int(f_star[i])
        if kind == "simple":
            p_star = int(np.argmax(rph[o, f, :, i]))
            g_raw = r[o, :, ph[p_star], i]
        else:
            g_raw = rph[o, :, :, i].mean(axis=1)
        den = float(g_raw[f])
        if den == 0.0:
            n_skipped += 1
            continue
        g = g_raw / den
        for sf_i in range(fa.N_SF):
            key = _key((20 - f) / (20 - sf_i))
            if 0.4 <= float(key) <= 2.0:
                bins.setdefault(key, []).append(float(g[sf_i]))
    curve = {}
    for key in sorted(bins, key=float):
        v = np.asarray(bins[key], dtype=np.float64)
        curve[key] = {
            "mean": float(v.mean()),
            "se": float(v.std(ddof=0) / math.sqrt(v.size)),
            "n": int(v.size),
        }
    return curve, n_skipped


def _scored_points(ref, panel):
    """(K/k, fig mean, fig se) triples scored for ``panel`` (K/k != 1, non-null)."""
    ks = ref["K_over_k"]
    means = ref["panels"][panel]["mean"]
    ses = ref["panels"][panel]["se"]
    return [(float(k), float(m), float(s)) for k, m, s in zip(ks, means, ses)
            if k != 1.0 and m is not None]


def apply_criterion(curve, panel, ref):
    """T2 criterion of one pooled curve against one figure panel.

    Per scored point ok_j = |ours - fig| <= EPS + Z*sqrt(se_fig^2 +
    se_ours^2); a key missing from ``curve`` fails. frac is the passing
    fraction, pass_frac is frac >= FRAC. The shape guard compares sse
    (missing points contribute (0 - fig)^2) against GUARD * sse_flat, where
    sse_flat uses m = mean of fig over the scored points. t2_pass is
    pass_frac and guard. Returns the points table and all flags.
    """
    pts = _scored_points(ref, panel)
    m = float(np.mean([fm for _, fm, _ in pts]))
    points = []
    n_ok = 0
    sse = 0.0
    sse_flat = 0.0
    for k, fm, fs in pts:
        entry = curve.get(f"{k:.1f}")
        sse_flat += (m - fm) ** 2
        if entry is None:
            tol = EPS + Z * fs
            points.append({"j": k, "ours": None, "fig": fm, "tol": tol, "ok": False})
            sse += (0.0 - fm) ** 2
            continue
        ours = float(entry["mean"])
        se_ours = float(entry["se"])
        tol = EPS + Z * math.sqrt(fs**2 + se_ours**2)
        ok = bool(abs(ours - fm) <= tol)
        n_ok += ok
        points.append({"j": k, "ours": ours, "fig": fm, "tol": tol, "ok": ok})
        sse += (ours - fm) ** 2
    frac = float(n_ok / len(pts)) if pts else 0.0
    pass_frac = bool(frac >= FRAC)
    guard = bool(sse <= GUARD * sse_flat)
    return {
        "points": points,
        "frac": frac,
        "sse": float(sse),
        "sse_flat": float(sse_flat),
        "pass_frac": pass_frac,
        "guard": guard,
        "t2_pass": bool(pass_frac and guard),
    }


def score_set(counts, kind, panel, phases, ref, scored=True, column=None):
    """Full T2 output dict of one cell set: curve, counts, criterion."""
    curve, n_skipped = tuning_curve(counts, kind, phases)
    res = {
        "kind": kind,
        "panel": panel,
        "scored": bool(scored),
        "curve": curve,
        "n_cells": int(np.asarray(counts).shape[4]),
        "n_skipped": int(n_skipped),
        **apply_criterion(curve, panel, ref),
    }
    if column is not None:
        res["column"] = [int(column[0]), int(column[1])]
    return res


def score(run_dir, phases=EVEN_PHASES, ref=None):
    """T2 over the preregistered sets of a tuning run directory.

    Simple: the 64 cells of (4, 4) against panel c; complex: the 128 cells
    of the four columns against panel d, plus each 32-cell column alone
    (same dict, ``scored`` False, not scored). Records the phase set and
    the reference sha256 copied from the JSON.
    """
    phases = tuple(int(p) for p in phases)
    ref = load_reference() if ref is None else ref
    sets = fa.cell_sets()
    counts_s, _, counts_c, _ = fa._load_grid(run_dir, sets["simple"], sets["complex"])
    n_col = counts_c.shape[4] // len(fa.COMPLEX_COLS)
    by_col = []
    for k, col in enumerate(fa.COMPLEX_COLS):
        by_col.append(score_set(counts_c[..., k * n_col:(k + 1) * n_col],
                                "complex", PANEL_COMPLEX, phases, ref,
                                scored=False, column=col))
    return {
        "phases": list(phases),
        "reference_sha256": ref["sha256"],
        "simple": score_set(counts_s, "simple", PANEL_SIMPLE, phases, ref),
        "complex": score_set(counts_c, "complex", PANEL_COMPLEX, phases, ref),
        "complex_by_column": by_col,
    }


def _parse_phases(name):
    if name == "even":
        return EVEN_PHASES
    if name == "all":
        return ALL_PHASES
    if name == "0-4":
        return FIRST5_PHASES
    raise ValueError(f"unknown phases {name!r}")


def main(argv=None):
    """CLI: score a tuning run directory against the Fig. 3 reference."""
    ap = argparse.ArgumentParser(description="Score a tuning run: P3_fig3_v5.md T2.")
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--phases", choices=("even", "all", "0-4"), default="even",
                    help="A1 sensitivity: even phases 0,2,4,6,8 (default), "
                         "all 10 phases, or phases 0-4. Recorded as phases.")
    args = ap.parse_args(argv)
    if not args.signed and fa._refuses(args.run_dir):
        print(
            f"refusing {args.run_dir}: it lies under {fa._RUNS_ROOT}, the scored runs tree, "
            "and the Fig. 3 preregistration (P3_fig3_v5.md) is unsigned. Analyzing scored "
            "outputs now would void the preregistration. Use synthetic or pilot data, "
            "or re-run with --signed after Hamm signs.",
            file=sys.stderr,
        )
        return fa._REFUSAL_CODE
    res = score(args.run_dir, _parse_phases(args.phases))
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} simple_t2={res['simple']['t2_pass']} "
          f"complex_t2={res['complex']['t2_pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
