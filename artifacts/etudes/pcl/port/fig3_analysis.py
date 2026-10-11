"""Fig. 3 analysis: preregistered simple/complex tuning tests T1-T4.

Implements exactly the estimands of ``artifacts/etudes/pcl/prereg/P3_fig3.md``
(draft v3, cited per function as P3 sections). Synthetic data only: nothing
here may be run on the scored outputs until the preregistration is signed; the
CLI refuses ``F:/warehouse/pcl/runs/*`` unless ``--signed`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Stimulus axes (P3 "Network, stimuli, data"; ori names from run_tuning.py).
ORIS = ("0", "23", "45", "68", "90", "113", "135", "158")
N_ORI, N_SF, N_PHASE, N_TRIAL = 8, 18, 10, 2
EVEN_PHASES = (0, 2, 4, 6, 8)
ODD_PHASES = (1, 3, 5, 7, 9)

# Population geometry (P3 "Cell sets"; sizes from the run format in
# run_tuning.py: simple_counts 5184 = 9*9*64, complex_counts 1152 = 6*6*32).
NX, NY, ND = 9, 9, 64
CX, CY, CD = 6, 6, 32
SIMPLE_COL = (4, 4)
COMPLEX_COLS = ((2, 2), (2, 3), (3, 2), (3, 3))

# Criteria (P3 table rows T1-T4 and "Inclusion").
T1_R_MIN = 2.0
T2_R_MIN = 2.0
NULL_PCTL = 99.0
T3_PMI_MIN_SIMPLE = 0.3
T3_COMPLEX_FRAC = 0.5
MIN_SPIKES = 20
MAX_EXCLUDE_FRAC = 0.5

# omega search band (P3 "T4 frequency doubling": searched over 1-3 Hz).
_OM_FMIN = 1.0
_OM_FMAX = 3.0
_OM_STEP = 0.001

# Scored-data guard for the CLI.
_RUNS_ROOT = "F:/warehouse/pcl/runs"
_REFUSAL_CODE = 2

# File-read parallelism for the tuning grid (I/O-bound; results are assembled
# by grid index, so the output is identical to a serial load).
_LOAD_WORKERS = 12


def _read_one(job):
    """Read one grid file; return (oi, f, p, s, counts-col, pos, t)."""
    oi, f, p, s, path, keys, cs = job
    ckey, tukey, tckey = keys
    with np.load(path, allow_pickle=False) as z:
        col = z[ckey]
        tu = z[tukey]
        tc = z[tckey]
    hit = np.isin(tc, cs)
    sel = np.flatnonzero(hit)
    return oi, f, p, s, col, tc[sel], tu[sel]


def cell_sets():
    """Flat cell ids of the two preregistered sets (P3 "Cell sets").

    Simple: the 64 cells of simple column (4,4). Complex: the 128 cells of
    complex columns (2,2), (2,3), (3,2), (3,3). Ids are computed from the
    grid sizes with the port's flat index (i*NY+j)*ND+k, not typed literally.
    """
    s_base = (SIMPLE_COL[0] * NY + SIMPLE_COL[1]) * ND
    simple = s_base + np.arange(ND, dtype=np.int64)
    parts = []
    for cx, cy in COMPLEX_COLS:
        parts.append((cx * CY + cy) * CD + np.arange(CD, dtype=np.int64))
    complex_ = np.concatenate(parts)
    return {"simple": simple, "complex": complex_}


def _load_grid(run_dir, cells_s, cells_c=None):
    """Read the <ori>/<sf>/<phase>/<trial>.npz grid (run_tuning.py format).

    Only phases 0-9 are read (P3: phase 10 repeats phase 0). Returns
    ``(counts_s, times_s, counts_c, times_c)``; the ``_c`` pair is None when
    ``cells_c`` is None. ``counts`` is int64 [ori 8, sf 18, phase 10, trial 2,
    cell]; ``times[(o, f, p, s)]`` is ``(pos, t)`` with cell positions into
    the ``cells`` order (int32) and spike times in seconds (float64).
    """
    run_dir = Path(run_dir)
    sets = [("simple", np.asarray(cells_s, dtype=np.int64))]
    if cells_c is not None:
        sets.append(("complex", np.asarray(cells_c, dtype=np.int64)))
    out = {}
    for population, cells in sets:
        if population == "simple":
            keys = ("simple_counts", "simple_times_us", "simple_times_cell")
        else:
            keys = ("complex_counts", "complex_times_us", "complex_times_cell")
        order = np.argsort(cells, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(cells))
        counts = np.zeros((N_ORI, N_SF, N_PHASE, N_TRIAL, len(cells)), dtype=np.int64)
        times = {}
        cs = cells[order]
        jobs = [
            (oi, f, p, s, str(run_dir / ori / str(f) / str(p) / f"{s}.npz"), keys, cs)
            for oi, ori in enumerate(ORIS)
            for f in range(N_SF)
            for p in range(N_PHASE)
            for s in range(N_TRIAL)
        ]
        with ThreadPoolExecutor(max_workers=_LOAD_WORKERS) as ex:
            for oi, f, p, s, col, tc_sel, tu_sel in ex.map(_read_one, jobs):
                counts[oi, f, p, s, :] = col[cells]
                pos = inv[np.searchsorted(cs, tc_sel)]
                t = tu_sel.astype(np.float64) / 1e6
                times[(oi, f, p, s)] = (pos.astype(np.int32), t)
        out[population] = (counts, times)
    if cells_c is None:
        return out["simple"]
    return out["simple"][0], out["simple"][1], out["complex"][0], out["complex"][1]


def load_run(run_dir, cells, population="simple"):
    """Load one preregistered cell set from a tuning run directory.

    P3 "Network, stimuli, data" with the run_tuning.py output format: per
    file ``<ori>/<sf>/<phase>/<trial>.npz`` holding counts and spike times of
    every cell. ``cells`` are flat ids into the ``population`` arrays;
    output cell axis follows the order of ``cells``.
    """
    if population not in ("simple", "complex"):
        raise ValueError(f"unknown population {population!r}")
    return _load_grid(run_dir, cells)


def omega_from_input(npz_path, step=_OM_STEP):
    """Contrast-reversal frequency from a stimulus file (P3 "T4 frequency doubling").

    Fundamental of the ON-event train of the single most active pixel,
    searched over 1-3 Hz on a fine grid (not FFT bins). Timestamps are
    microseconds (run_tuning.py feed convention).
    """
    with np.load(npz_path, allow_pickle=False) as z:
        t = z["arr_0"].astype(np.int64)
        x = z["arr_1"].astype(np.int64)
        y = z["arr_2"].astype(np.int64)
        p = z["arr_3"].astype(np.int64)
    on = p == 1
    if not np.any(on):
        on = np.ones_like(p, dtype=bool)
    pix = np.stack([x[on], y[on]], axis=1)
    _, idx, n = np.unique(pix, axis=0, return_inverse=True, return_counts=True)
    t_hot = t[on][idx == int(np.argmax(n))].astype(np.float64) / 1e6
    freqs = np.arange(_OM_FMIN, _OM_FMAX + step / 2, step)
    ang = 2.0 * np.pi * np.outer(t_hot, freqs)
    amp = np.hypot(np.cos(ang).sum(axis=0), np.sin(ang).sum(axis=0))
    return float(freqs[int(np.argmax(amp))])


def _ratio(num, den):
    """Peak/flank ratio with a zero-denominator rule (0/0 -> 1, x/0 -> inf)."""
    num, den = float(num), float(den)
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    if den == 0.0:
        return float("inf") if num > 0.0 else 1.0
    return num / den


def _passes(r, thr, p99):
    """Pass rule shared by T1 and T2 (P3 table: R >= 2 and above the null 99th percentile)."""
    return bool(r >= thr and r > p99)


def _ori_curve(flat, o_star, f_star):
    """Population orientation curve aligned to each cell's preference.

    ``flat`` is scored-fold totals reshaped to [(ori, sf) 144, cell];
    orientation wraps circularly (8 steps of ~22.5 deg, so d=4 is 90 deg).
    """
    n_cells = flat.shape[1]
    d = np.arange(N_ORI)[:, None]
    idx = ((o_star[None, :] + d) % N_ORI) * N_SF + f_star[None, :]
    return np.take_along_axis(flat, idx, axis=0).mean(axis=1)


def _sf_curve(flat, o_star, f_star):
    """Population sf curve aligned to each cell's preference.

    Offsets run linearly (-17..17); cells whose preferred sf plus the offset
    falls outside 0..17 do not contribute to that offset (no wraparound:
    spatial frequency is not circular).
    """
    n_cells = flat.shape[1]
    cols = np.arange(n_cells)
    curve = np.full(2 * (N_SF - 1) + 1, np.nan)
    for k, d in enumerate(range(-(N_SF - 1), N_SF)):
        f = f_star + d
        ok = (f >= 0) & (f < N_SF)
        if np.any(ok):
            curve[k] = flat[o_star[ok] * N_SF + f[ok], cols[ok]].mean()
    return curve


def t1_t2(counts, rng, n_null=1000):
    """Folded orientation/sf tuning with the selection-matched null.

    P3 "Preference folds" and "Null for T1 and T2": preferences fit on even
    phases, scored on odd phases (trials pooled throughout); the null
    shuffles the scored fold's (ori, sf) labels within each cell and reruns
    the fit-then-score procedure (the argmax fit on the even phases is
    unaffected by the shuffle, so the same selection step is retained).
    """
    pooled = np.asarray(counts).sum(axis=3)  # trials pooled, never folds
    n_cells = pooled.shape[3]
    fit = pooled[:, :, list(EVEN_PHASES), :].sum(axis=2)
    sco = pooled[:, :, list(ODD_PHASES), :].sum(axis=2)
    lin = fit.reshape(N_ORI * N_SF, n_cells).argmax(axis=0)
    o_star, f_star = lin // N_SF, lin % N_SF

    flat = sco.reshape(N_ORI * N_SF, n_cells).astype(np.float64)
    curve_ori = _ori_curve(flat, o_star, f_star)
    r_ori = _ratio(curve_ori[0], curve_ori[N_ORI // 2])
    curve_sf = _sf_curve(flat, o_star, f_star)
    flank = curve_sf[np.abs(np.arange(len(curve_sf)) - (N_SF - 1)) >= 6]
    r_sf = _ratio(curve_sf[N_SF - 1], np.nanmean(flank))

    cols = np.arange(n_cells)
    null_ori = np.empty(n_null)
    null_sf = np.empty(n_null)
    for b in range(int(n_null)):
        order = np.argsort(rng.random((N_ORI * N_SF, n_cells)), axis=0)
        sh = flat[order, np.broadcast_to(cols, (N_ORI * N_SF, n_cells))]
        co = _ori_curve(sh, o_star, f_star)
        null_ori[b] = _ratio(co[0], co[N_ORI // 2])
        cs = _sf_curve(sh, o_star, f_star)
        fl = cs[np.abs(np.arange(len(cs)) - (N_SF - 1)) >= 6]
        null_sf[b] = _ratio(cs[N_SF - 1], np.nanmean(fl))
    p99_ori, p99_sf = float(np.percentile(null_ori, NULL_PCTL)), float(np.percentile(null_sf, NULL_PCTL))
    return {
        "o_star": o_star.astype(int).tolist(),
        "f_star": f_star.astype(int).tolist(),
        "r_ori": float(r_ori),
        "r_ori_null99": p99_ori,
        "t1_pass": _passes(r_ori, T1_R_MIN, p99_ori),
        "r_sf": float(r_sf),
        "r_sf_null99": p99_sf,
        "t2_pass": _passes(r_sf, T2_R_MIN, p99_sf),
        "n_null": int(n_null),
    }


def fit_prefs_all(counts):
    """Preferences on all phases, trials pooled (P3: T3 and T4 fit on all phases)."""
    m = np.asarray(counts).sum(axis=(2, 3))
    n_cells = m.shape[2]
    lin = m.reshape(N_ORI * N_SF, n_cells).argmax(axis=0)
    o_star, f_star = (lin // N_SF).astype(int), (lin % N_SF).astype(int)
    cell = np.arange(n_cells)
    v = np.asarray(counts)[o_star, f_star, :, :, cell].sum(axis=2).T  # [phase 10, cell]
    p_star = v.argmax(axis=0).astype(int)
    return o_star, f_star, p_star


def _pmi_set(counts):
    """Per-cell PMI at the all-phase preference with the 20-spike floor (P3 T3)."""
    o_star, f_star, _ = fit_prefs_all(counts)
    cell = np.arange(np.asarray(counts).shape[4])
    v = np.asarray(counts)[o_star, f_star, :, :, cell].sum(axis=2).T.astype(np.float64)
    total = v.sum(axis=0)
    included = total >= MIN_SPIKES
    pmi = np.full(v.shape[1], np.nan)
    ok = included & (v.max(axis=0) + v.min(axis=0) > 0)
    mx, mn = v.max(axis=0), v.min(axis=0)
    pmi[ok] = (mx[ok] - mn[ok]) / (mx[ok] + mn[ok])
    return pmi, included


def t3(counts_s, counts_c):
    """Phase tuning of both sets (P3 row T3 and "Inclusion").

    PMI = (max-min)/(max+min) of the count over phases 0-9 at (ori*, sf*).
    A set with more than half its cells below the 20-spike floor fails.
    """
    pmi_s, incl_s = _pmi_set(counts_s)
    pmi_c, incl_c = _pmi_set(counts_c)
    med_s = float(np.median(pmi_s[incl_s])) if np.any(incl_s) else float("nan")
    med_c = float(np.median(pmi_c[incl_c])) if np.any(incl_c) else float("nan")
    excl_s = float(1.0 - incl_s.mean()) if len(incl_s) else 1.0
    excl_c = float(1.0 - incl_c.mean()) if len(incl_c) else 1.0
    t3_simple_pass = bool(excl_s <= MAX_EXCLUDE_FRAC and med_s >= T3_PMI_MIN_SIMPLE)
    t3_complex_pass = bool(excl_c <= MAX_EXCLUDE_FRAC and med_c <= T3_COMPLEX_FRAC * med_s)
    return {
        "pmi_simple": pmi_s.tolist(),
        "pmi_complex": pmi_c.tolist(),
        "median_pmi_simple": med_s,
        "median_pmi_complex": med_c,
        "excluded_frac_simple": excl_s,
        "excluded_frac_complex": excl_c,
        "t3_simple_pass": t3_simple_pass,
        "t3_complex_pass": t3_complex_pass,
    }


def _amp_at(spikes, f):
    """A(f) = |sum exp(-2 pi i f t)| over spike times in seconds (P3 T4).

    The duration division in the spec cancels in the log ratio, so it is
    omitted here.
    """
    spikes = np.asarray(spikes, dtype=np.float64)
    if spikes.size == 0:
        return 0.0
    ang = 2.0 * np.pi * f * spikes
    return float(np.hypot(np.cos(ang).sum(), np.sin(ang).sum()))


def _log_ratio(spikes, omega):
    """L = log(A(2w)/A(w)) evaluated at exactly w and 2w (P3 T4)."""
    a_lo = _amp_at(spikes, omega)
    a_hi = _amp_at(spikes, 2.0 * omega)
    if a_lo <= 0.0:
        return float("inf") if a_hi > 0.0 else float("nan")
    if a_hi <= 0.0:
        return float("-inf")
    return float(math.log(a_hi / a_lo))


def _cell_Ls(times, pref, omega):
    """Per-cell L at (ori*, sf*, phase*), trials pooled, with the floor (P3 T3/T4)."""
    o_star, f_star, p_star = pref
    n_cells = len(o_star)
    Ls = np.full(n_cells, np.nan)
    incl = np.zeros(n_cells, dtype=bool)
    for i in range(n_cells):
        tot = 0
        for p in range(N_PHASE):
            for s in range(N_TRIAL):
                pos, t = times[(int(o_star[i]), int(f_star[i]), p, s)]
                tot += int((pos == i).sum())
        incl[i] = tot >= MIN_SPIKES
        if incl[i]:
            sel = []
            for s in range(N_TRIAL):
                pos, t = times[(int(o_star[i]), int(f_star[i]), int(p_star[i]), s)]
                sel.append(np.asarray(t)[pos == i])
            pooled = np.concatenate(sel) if sel else np.zeros(0)
            Ls[i] = _log_ratio(pooled, omega)
    return Ls, incl


def t4(times, omega, prefs, rng, n_null=1000):
    """Frequency doubling of both sets (P3 row T4 and "T4 frequency doubling").

    ``times`` is ``(times_simple, times_complex)`` as returned by
    :func:`load_run`; ``prefs`` is ``(pref_simple, pref_complex)`` from
    :func:`fit_prefs_all`. The null permutes cells between the two sets.
    """
    Ls, incl_s = _cell_Ls(times[0], prefs[0], float(omega))
    Lc, incl_c = _cell_Ls(times[1], prefs[1], float(omega))
    vs = Ls[incl_s & np.isfinite(Ls)]
    vc = Lc[incl_c & np.isfinite(Lc)]
    med_s = float(np.median(vs)) if vs.size else float("nan")
    med_c = float(np.median(vc)) if vc.size else float("nan")
    diff = float(med_c - med_s) if vs.size and vc.size else float("nan")
    pool = np.concatenate([vs, vc]) if vs.size and vc.size else np.zeros(0)
    null_diff = np.empty(int(n_null))
    for b in range(int(n_null)):
        perm = rng.permutation(len(pool))
        gs, gc = perm[: len(vs)], perm[len(vs):]
        null_diff[b] = float(np.median(pool[gc]) - np.median(pool[gs]))
    p99 = float(np.percentile(null_diff, NULL_PCTL)) if pool.size else float("nan")
    excl_s = float(1.0 - (incl_s & np.isfinite(Ls)).mean()) if len(Ls) else 1.0
    excl_c = float(1.0 - (incl_c & np.isfinite(Lc)).mean()) if len(Lc) else 1.0
    t4_pass = bool(excl_s <= MAX_EXCLUDE_FRAC and excl_c <= MAX_EXCLUDE_FRAC
                   and med_s < 0.0 and 0.0 < med_c and diff > p99)
    return {
        "L_simple": Ls.tolist(),
        "L_complex": Lc.tolist(),
        "median_L_simple": med_s,
        "median_L_complex": med_c,
        "diff": diff,
        "diff_null99": p99,
        "excluded_frac_simple": excl_s,
        "excluded_frac_complex": excl_c,
        "t4_pass": t4_pass,
        "n_null": int(n_null),
    }


def _probe_paths(input_dir):
    """Three probe stimulus files used for the global omega (P3: 1.98-1.99 Hz in test7 files)."""
    input_dir = Path(input_dir)
    return [input_dir / ORIS[0] / str(f) / "0" / "0.npz" for f in (0, 9, 17)]


def score(run_dir, input_dir, seed=0, n_null=1000):
    """All T1-T4 statistics and pass flags for a tuning run directory."""
    sets = cell_sets()
    counts_s, times_s, counts_c, times_c = _load_grid(run_dir, sets["simple"], sets["complex"])
    rng = np.random.default_rng(seed)
    r_s = t1_t2(counts_s, rng, n_null)
    r_c = t1_t2(counts_c, rng, n_null)
    prefs_s = fit_prefs_all(counts_s)
    prefs_c = fit_prefs_all(counts_c)
    omegas = [omega_from_input(p) for p in _probe_paths(input_dir)]
    omega = float(np.median(omegas))
    res3 = t3(counts_s, counts_c)
    res4 = t4((times_s, times_c), omega, (prefs_s, prefs_c), rng, n_null)
    return {
        "seed": int(seed),
        "n_null": int(n_null),
        "omega": omega,
        "omega_probes": [float(o) for o in omegas],
        "t1_r_ori_simple": r_s["r_ori"],
        "t1_r_ori_simple_null99": r_s["r_ori_null99"],
        "t1_simple_pass": r_s["t1_pass"],
        "t1_r_ori_complex": r_c["r_ori"],
        "t1_r_ori_complex_null99": r_c["r_ori_null99"],
        "t1_complex_pass": r_c["t1_pass"],
        "t2_r_sf_simple": r_s["r_sf"],
        "t2_r_sf_simple_null99": r_s["r_sf_null99"],
        "t2_simple_pass": r_s["t2_pass"],
        "t2_r_sf_complex": r_c["r_sf"],
        "t2_r_sf_complex_null99": r_c["r_sf_null99"],
        "t2_complex_pass": r_c["t2_pass"],
        "median_pmi_simple": res3["median_pmi_simple"],
        "median_pmi_complex": res3["median_pmi_complex"],
        "excluded_frac_simple": res3["excluded_frac_simple"],
        "excluded_frac_complex": res3["excluded_frac_complex"],
        "t3_simple_pass": res3["t3_simple_pass"],
        "t3_complex_pass": res3["t3_complex_pass"],
        "median_L_simple": res4["median_L_simple"],
        "median_L_complex": res4["median_L_complex"],
        "t4_diff": res4["diff"],
        "t4_diff_null99": res4["diff_null99"],
        "t4_pass": res4["t4_pass"],
        "all_pass": bool(
            r_s["t1_pass"] and r_c["t1_pass"] and r_s["t2_pass"] and r_c["t2_pass"]
            and res3["t3_simple_pass"] and res3["t3_complex_pass"] and res4["t4_pass"]
        ),
    }


def _refuses(path):
    """True when ``path`` lies under the scored runs tree."""
    root = os.path.normcase(os.path.abspath(_RUNS_ROOT))
    p = os.path.normcase(os.path.abspath(path))
    return p == root or p.startswith(root + os.sep)


def main(argv=None):
    """CLI: score a tuning run directory and write the result as JSON."""
    ap = argparse.ArgumentParser(description="Score a Fig. 3 tuning run (P3_fig3.md T1-T4).")
    ap.add_argument("run_dir")
    ap.add_argument("input_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--signed", action="store_true")
    args = ap.parse_args(argv)
    if not args.signed and _refuses(args.run_dir):
        print(
            f"refusing {args.run_dir}: it lies under {_RUNS_ROOT}, the scored runs tree, "
            "and the Fig. 3 preregistration (P3_fig3.md) is unsigned. Analyzing scored "
            "outputs now would void the preregistration. Use synthetic or pilot data, "
            "or re-run with --signed after Hamm signs.",
            file=sys.stderr,
        )
        return _REFUSAL_CODE
    res = score(args.run_dir, args.input_dir, seed=args.seed, n_null=args.n_null)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} all_pass={res['all_pass']}")
    return 0


def _sha256_of_this_file():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
