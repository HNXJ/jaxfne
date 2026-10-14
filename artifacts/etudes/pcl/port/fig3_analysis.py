"""Fig. 3 analysis: preregistered simple/complex tuning tests T1-T4.

Implements exactly the estimands of ``artifacts/etudes/pcl/prereg/P3_fig3.md``
(draft v4, cited per function as P3 sections). Synthetic data only: nothing
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

# Criteria (P3 table rows T1-T4 and "Inclusion", draft v4).
T1_R_MIN = 2.0
T2_R_MIN = 2.0
NULL_PCTL = 99.0
T3_X_MIN_SIMPLE = 0.15
T3_COMPLEX_FRAC = 0.5
MIN_SPIKES_FIT = 20
T3_MIN_SPIKES = 50
T4_MIN_SPIKES = 10
MAX_EXCLUDE_FRAC = 0.5
N_PMI_DRAWS = 1000

# omega search band (P3 "T4 frequency doubling": searched over 1-3 Hz).
_OM_FMIN = 1.0
_OM_FMAX = 3.0
_OM_STEP = 0.001
# Per-file omega acceptance band and trial-agreement tolerance (P3 v4 T4).
_OM_ACCEPT_LO = 1.9
_OM_ACCEPT_HI = 2.1
_OM_TRIAL_DIFF_TOL = 0.01

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


def cell_sets(simple_col=None):
    """Flat cell ids of the two preregistered sets (P3 "Cell sets").

    Simple: the 64 cells of simple column ``simple_col`` (default
    ``SIMPLE_COL`` (4, 4); S3 passes another central column). Complex: the
    128 cells of complex columns (2,2), (2,3), (3,2), (3,3). Ids are computed
    from the grid sizes with the port's flat index (i*NY+j)*ND+k, not typed
    literally. A column outside the NX x NY grid raises ValueError.
    """
    col = SIMPLE_COL if simple_col is None else tuple(int(v) for v in simple_col)
    if len(col) != 2 or not (0 <= col[0] < NX and 0 <= col[1] < NY):
        raise ValueError(f"simple column {simple_col!r} outside the {NX}x{NY} grid")
    s_base = (col[0] * NY + col[1]) * ND
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


def _input_path(input_dir, o, f, p, s):
    """Stimulus-file path for one grid point (run_tuning.py feed layout)."""
    return Path(input_dir) / ORIS[int(o)] / str(int(f)) / str(int(p)) / f"{int(s)}.npz"


def _ratio(num, den):
    """Peak/flank ratio with a zero-denominator rule (0/0 -> 1, x/0 -> inf)."""
    num, den = float(num), float(den)
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    if den == 0.0:
        return float("inf") if num > 0.0 else 1.0
    return num / den


def _passes(r, thr, p99_label, p99_pref):
    """Pass rule shared by T1 and T2 (P3 v4 table: R >= 2 and above both null 99th percentiles)."""
    return bool(r >= thr and r > p99_label and r > p99_pref)


def _ori_curve(flat, o_star, f_star):
    """Population orientation curve aligned to each cell's preference.

    ``flat`` is scored-fold totals reshaped to [(ori, sf) 144, cell];
    orientation wraps circularly (8 steps of ~22.5 deg, so d=4 is 90 deg).
    """
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


def _sf_flank_per_cell(flat, o_star, f_star):
    """Per-cell sf flank (P3 v4: each cell's mean response at |dsf| >= 6).

    Only offsets landing inside 0..17 contribute to a cell's mean; the
    caller averages the per-cell means over cells.
    """
    n_cells = flat.shape[1]
    fl = np.full(n_cells, np.nan)
    sfs = np.arange(N_SF)
    for i in range(n_cells):
        sel = np.abs(sfs - int(f_star[i])) >= 6
        if np.any(sel):
            fl[i] = flat[int(o_star[i]) * N_SF + sfs[sel], i].mean()
    return fl


def t1_t2(counts, rng, n_null=1000, fit_phases=None, score_phases=None):
    """Folded orientation/sf tuning with both selection-matched nulls.

    P3 v4 "Preference folds", "Nulls for T1 and T2" and "Inclusion":
    preferences fit on even phases, scored on odd phases (trials pooled
    throughout). ``fit_phases``/``score_phases`` override the fold (default
    EVEN_PHASES/ODD_PHASES); S2 passes all phases 0-9 for both, i.e. the
    authors' no-fold procedure. N-label shuffles the scored fold's (ori, sf) labels within
    each cell and reruns fit-then-score; N-pref permutes the fitted
    (ori*, sf*) pairs across the cells of the set and recomputes the scored
    curves. Cells with fewer than 20 spikes in the fit fold are excluded; if
    more than half the set is excluded the test fails for the set. The fitted
    preference distribution (counts per ori*, per sf*) is reported.
    """
    pooled = np.asarray(counts).sum(axis=3)  # trials pooled, never folds
    n_cells = pooled.shape[3]
    if fit_phases is None:
        fit_phases = EVEN_PHASES
    if score_phases is None:
        score_phases = ODD_PHASES
    fit = pooled[:, :, list(fit_phases), :].sum(axis=2)
    sco = pooled[:, :, list(score_phases), :].sum(axis=2)
    fit_tot = fit.sum(axis=(0, 1))
    incl = fit_tot >= MIN_SPIKES_FIT
    n_excluded = int((~incl).sum())
    excluded_frac = float(n_excluded / n_cells) if n_cells else 1.0
    keep = np.flatnonzero(incl)
    base = {
        "n_null": int(n_null),
        "n_excluded": n_excluded,
        "excluded_frac": excluded_frac,
        "pref_ori_counts": [0] * N_ORI,
        "pref_sf_counts": [0] * N_SF,
    }
    if keep.size == 0:
        nan = float("nan")
        return {
            **base,
            "o_star": [],
            "f_star": [],
            "r_ori": nan,
            "r_ori_null99": nan,
            "r_ori_npref99": nan,
            "t1_pass": False,
            "r_sf": nan,
            "r_sf_null99": nan,
            "r_sf_npref99": nan,
            "t2_pass": False,
        }

    fit_k = fit[:, :, keep]
    sco_k = sco[:, :, keep]
    lin = fit_k.reshape(N_ORI * N_SF, keep.size).argmax(axis=0)
    o_star, f_star = lin // N_SF, lin % N_SF

    flat = sco_k.reshape(N_ORI * N_SF, keep.size).astype(np.float64)
    curve_ori = _ori_curve(flat, o_star, f_star)
    r_ori = _ratio(curve_ori[0], curve_ori[N_ORI // 2])
    curve_sf = _sf_curve(flat, o_star, f_star)
    flank = _sf_flank_per_cell(flat, o_star, f_star)
    r_sf = _ratio(curve_sf[N_SF - 1], np.nanmean(flank))

    cols = np.arange(keep.size)
    null_ori_label = np.empty(int(n_null))
    null_sf_label = np.empty(int(n_null))
    null_ori_pref = np.empty(int(n_null))
    null_sf_pref = np.empty(int(n_null))
    for b in range(int(n_null)):
        order = np.argsort(rng.random((N_ORI * N_SF, keep.size)), axis=0)
        sh = flat[order, np.broadcast_to(cols, (N_ORI * N_SF, keep.size))]
        co = _ori_curve(sh, o_star, f_star)
        null_ori_label[b] = _ratio(co[0], co[N_ORI // 2])
        cs = _sf_curve(sh, o_star, f_star)
        fl = _sf_flank_per_cell(sh, o_star, f_star)
        null_sf_label[b] = _ratio(cs[N_SF - 1], np.nanmean(fl))
        perm = rng.permutation(keep.size)
        op, fp = o_star[perm], f_star[perm]
        po = _ori_curve(flat, op, fp)
        null_ori_pref[b] = _ratio(po[0], po[N_ORI // 2])
        ps = _sf_curve(flat, op, fp)
        pf = _sf_flank_per_cell(flat, op, fp)
        null_sf_pref[b] = _ratio(ps[N_SF - 1], np.nanmean(pf))
    p99_ori_lab = float(np.percentile(null_ori_label, NULL_PCTL))
    p99_sf_lab = float(np.percentile(null_sf_label, NULL_PCTL))
    p99_ori_pref = float(np.percentile(null_ori_pref, NULL_PCTL))
    p99_sf_pref = float(np.percentile(null_sf_pref, NULL_PCTL))
    over_excluded = excluded_frac > MAX_EXCLUDE_FRAC
    return {
        **base,
        "o_star": o_star.astype(int).tolist(),
        "f_star": f_star.astype(int).tolist(),
        "pref_ori_counts": np.bincount(o_star, minlength=N_ORI).tolist(),
        "pref_sf_counts": np.bincount(f_star, minlength=N_SF).tolist(),
        "r_ori": float(r_ori),
        "r_ori_null99": p99_ori_lab,
        "r_ori_npref99": p99_ori_pref,
        "t1_pass": bool(_passes(r_ori, T1_R_MIN, p99_ori_lab, p99_ori_pref) and not over_excluded),
        "r_sf": float(r_sf),
        "r_sf_null99": p99_sf_lab,
        "r_sf_npref99": p99_sf_pref,
        "t2_pass": bool(_passes(r_sf, T2_R_MIN, p99_sf_lab, p99_sf_pref) and not over_excluded),
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


def _pmi_of_rows(v):
    """PMI = (max-min)/(max+min) per row of an [..., 10] array (0/0 -> 0)."""
    mx = v.max(axis=-1)
    mn = v.min(axis=-1)
    tot = mx + mn
    out = np.zeros_like(tot, dtype=np.float64)
    ok = tot > 0
    out[ok] = (mx[ok] - mn[ok]) / tot[ok]
    return out


def _pmi_set(counts, rng, n_draws=N_PMI_DRAWS):
    """Per-cell excess modulation X = PMI - PMI0 with the 50-spike floor (P3 v4 T3).

    Per trial, PMI is (max-min)/(max+min) of the count over phases 0-9 at
    (ori*, sf*), and PMI0 is the mean PMI of ``n_draws`` draws of 10 Poisson
    counts at that trial's mean count per phase. X, PMI and PMI0 are means
    over the two trials. Per-trial nulls stay unbiased when trials are
    correlated; pooling near-identical trials doubles the variance a pooled
    Poisson null assumes. The floor counts both trials. Cells below it are
    excluded (X is nan).
    """
    o_star, f_star, _ = fit_prefs_all(counts)
    n_cells = np.asarray(counts).shape[4]
    cell = np.arange(n_cells)
    v = np.asarray(counts)[o_star, f_star, :, :, cell].astype(np.float64)  # [cell, phase, trial]
    total = v.sum(axis=(1, 2))
    included = total >= T3_MIN_SPIKES
    pmi = np.full(n_cells, np.nan)
    pmi0 = np.full(n_cells, np.nan)
    x = np.full(n_cells, np.nan)
    for i in np.flatnonzero(included):
        per_trial = v[i].T  # [trial, phase]
        pmi_t = _pmi_of_rows(per_trial)
        pmi0_t = np.array([_pmi_of_rows(rng.poisson(lam, size=(int(n_draws), N_PHASE))
                                        .astype(np.float64)).mean()
                           for lam in per_trial.mean(axis=1)])
        pmi[i] = float(pmi_t.mean())
        pmi0[i] = float(pmi0_t.mean())
        x[i] = float(pmi[i] - pmi0[i])
    return x, pmi, pmi0, included


def t3(counts_s, counts_c, rng=None, n_pmi=N_PMI_DRAWS):
    """Phase tuning of both sets (P3 v4 row T3 and "Inclusion").

    Excess modulation X = PMI - PMI0 per cell; a set passes when its median
    X clears its bar (simple >= 0.15; complex <= 0.5 x simple median). A set
    with more than half its cells below the 50-spike floor fails.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x_s, pmi_s, pmi0_s, incl_s = _pmi_set(counts_s, rng, n_pmi)
    x_c, pmi_c, pmi0_c, incl_c = _pmi_set(counts_c, rng, n_pmi)
    med_xs = float(np.median(x_s[incl_s])) if np.any(incl_s) else float("nan")
    med_xc = float(np.median(x_c[incl_c])) if np.any(incl_c) else float("nan")
    med_ps = float(np.median(pmi_s[incl_s])) if np.any(incl_s) else float("nan")
    med_pc = float(np.median(pmi_c[incl_c])) if np.any(incl_c) else float("nan")
    n_excl_s = int((~incl_s).sum())
    n_excl_c = int((~incl_c).sum())
    excl_s = float(n_excl_s / len(incl_s)) if len(incl_s) else 1.0
    excl_c = float(n_excl_c / len(incl_c)) if len(incl_c) else 1.0
    t3_simple_pass = bool(excl_s <= MAX_EXCLUDE_FRAC and med_xs >= T3_X_MIN_SIMPLE)
    t3_complex_pass = bool(excl_c <= MAX_EXCLUDE_FRAC and med_xc <= T3_COMPLEX_FRAC * med_xs)
    return {
        "X_simple": x_s.tolist(),
        "X_complex": x_c.tolist(),
        "pmi_simple": pmi_s.tolist(),
        "pmi_complex": pmi_c.tolist(),
        "pmi0_simple": pmi0_s.tolist(),
        "pmi0_complex": pmi0_c.tolist(),
        "median_X_simple": med_xs,
        "median_X_complex": med_xc,
        "median_pmi_simple": med_ps,
        "median_pmi_complex": med_pc,
        "n_excluded_simple": n_excl_s,
        "n_excluded_complex": n_excl_c,
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


def _omega_for(omegas, key):
    """Omega of one stimulus file: a scalar holds for all files, else a dict lookup."""
    if isinstance(omegas, dict):
        return float(omegas[tuple(int(v) for v in key)])
    return float(omegas)


def _cell_Ls(times, pref, omegas):
    """Per-cell L at (ori*, sf*, phase*), trials pooled, with floors (P3 v4 T4/"Inclusion").

    ``omegas`` is one float for every file or a dict mapping
    ``(ori, sf, phase, trial)`` to that file's measured omega. A cell's L
    uses the omegas of the two files its spikes come from; trials whose
    omegas differ by more than 0.01 Hz are reported and averaged. A cell is
    excluded when it has fewer than 10 spikes at (ori*, sf*, phase*) over
    both trials, when either file's omega falls outside 1.9-2.1 Hz, or when
    its L is not finite. Every exclusion is counted by cause.
    """
    o_star, f_star, p_star = pref
    n_cells = len(o_star)
    Ls = np.full(n_cells, np.nan)
    used = np.full(n_cells, np.nan)
    count_ok = np.zeros(n_cells, dtype=bool)
    band_ok = np.zeros(n_cells, dtype=bool)
    bad_files = []
    mismatches = []
    n_spikes = np.zeros(n_cells, dtype=int)
    for i in range(n_cells):
        o, f, p = int(o_star[i]), int(f_star[i]), int(p_star[i])
        per_trial = []
        for s in range(N_TRIAL):
            pos, t = times[(o, f, p, s)]
            per_trial.append(np.asarray(t)[pos == i])
        tot = sum(len(v) for v in per_trial)
        n_spikes[i] = tot
        count_ok[i] = tot >= T4_MIN_SPIKES
        w0 = _omega_for(omegas, (o, f, p, 0))
        w1 = _omega_for(omegas, (o, f, p, 1))
        ok0 = _OM_ACCEPT_LO <= w0 <= _OM_ACCEPT_HI
        ok1 = _OM_ACCEPT_LO <= w1 <= _OM_ACCEPT_HI
        band_ok[i] = bool(ok0 and ok1)
        if not ok0:
            bad_files.append([o, f, p, 0, w0])
        if not ok1:
            bad_files.append([o, f, p, 1, w1])
        if abs(w0 - w1) > _OM_TRIAL_DIFF_TOL:
            mismatches.append({"key": [o, f, p], "w0": w0, "w1": w1})
        w = (w0 + w1) / 2.0
        used[i] = w
        if count_ok[i] and band_ok[i]:
            pooled = np.concatenate(per_trial) if per_trial else np.zeros(0)
            Ls[i] = _log_ratio(pooled, w)
    finite = np.isfinite(Ls)
    included = count_ok & band_ok & finite
    return {
        "Ls": Ls,
        "used": used,
        "included": included,
        "count_ok": count_ok,
        "band_ok": band_ok,
        "finite": finite,
        "bad_files": bad_files,
        "mismatches": mismatches,
        "n_spikes": n_spikes,
    }


def t4(times, omega, prefs, rng, n_null=1000):
    """Frequency doubling of both sets (P3 v4 row T4, "T4 frequency doubling", "Inclusion").

    ``times`` is ``(times_simple, times_complex)`` as returned by
    :func:`load_run`; ``prefs`` is ``(pref_simple, pref_complex)`` from
    :func:`fit_prefs_all`. ``omega`` is one float for every stimulus file or
    a dict mapping ``(ori, sf, phase, trial)`` to that file's omega. The null
    permutes cells between the two sets.
    """
    rs = _cell_Ls(times[0], prefs[0], omega)
    rc = _cell_Ls(times[1], prefs[1], omega)
    Ls, Lc = rs["Ls"], rc["Ls"]
    incl_s, incl_c = rs["included"], rc["included"]
    vs = Ls[incl_s]
    vc = Lc[incl_c]
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
    n_excl_s = int((~incl_s).sum())
    n_excl_c = int((~incl_c).sum())
    excl_s = float(n_excl_s / len(Ls)) if len(Ls) else 1.0
    excl_c = float(n_excl_c / len(Lc)) if len(Lc) else 1.0
    t4_pass = bool(excl_s <= MAX_EXCLUDE_FRAC and excl_c <= MAX_EXCLUDE_FRAC
                   and med_s < 0.0 and 0.0 < med_c and diff > p99)
    bad = sorted({tuple(e[:4]) for e in rs["bad_files"] + rc["bad_files"]})
    return {
        "L_simple": Ls.tolist(),
        "L_complex": Lc.tolist(),
        "median_L_simple": med_s,
        "median_L_complex": med_c,
        "diff": diff,
        "diff_null99": p99,
        "n_excluded_simple": n_excl_s,
        "n_excluded_complex": n_excl_c,
        "n_low_count_simple": int((~rs["count_ok"]).sum()),
        "n_low_count_complex": int((~rc["count_ok"]).sum()),
        "n_bad_omega_simple": int((~rs["band_ok"]).sum()),
        "n_bad_omega_complex": int((~rc["band_ok"]).sum()),
        "n_nonfinite_simple": int((rs["count_ok"] & rs["band_ok"] & ~rs["finite"]).sum()),
        "n_nonfinite_complex": int((rc["count_ok"] & rc["band_ok"] & ~rc["finite"]).sum()),
        "excluded_frac_simple": excl_s,
        "excluded_frac_complex": excl_c,
        "omega_bad_files": [list(k) for k in bad],
        "omega_mismatches": rs["mismatches"] + rc["mismatches"],
        "used_omega_simple": rs["used"].tolist(),
        "used_omega_complex": rc["used"].tolist(),
        "t4_pass": t4_pass,
        "n_null": int(n_null),
    }


def _omega_map(input_dir, prefs_s, prefs_c, omegas=None):
    """Per-file omegas for the files T4 cells draw spikes from (P3 v4 T4).

    ``omegas`` overrides the file reads: a float holds for every file, a
    dict maps ``(ori, sf, phase, trial)`` to omega. Otherwise each needed
    stimulus file is measured once with :func:`omega_from_input`.
    """
    keys = set()
    for pref in (prefs_s, prefs_c):
        o_star, f_star, p_star = pref
        for i in range(len(o_star)):
            for s in range(N_TRIAL):
                keys.add((int(o_star[i]), int(f_star[i]), int(p_star[i]), int(s)))
    if omegas is None:
        out = {}
        for key in sorted(keys):
            o, f, p, s = key
            out[key] = omega_from_input(_input_path(input_dir, o, f, p, s))
        return out
    if isinstance(omegas, dict):
        return {tuple(int(v) for v in k): float(v_) for k, v_ in omegas.items()}
    return {k: float(omegas) for k in keys}


def score(run_dir, input_dir, seed=0, n_null=1000, omegas=None,
          s2_all_phases=False, simple_col=None):
    """All T1-T4 statistics and pass flags for a tuning run directory.

    ``omegas`` overrides the per-file omega measurement (a float for every
    file or a dict); it exists so synthetic runs can be scored without
    stimulus files. ``s2_all_phases`` adds the S2 sensitivity (P3
    "Sensitivity"): T1/T2 refit with preferences on all phases 0-9 and
    scored on the same phases, under ``s2_``-prefixed keys beside the normal
    keys; T3 and T4 are unchanged. ``simple_col`` is the S3 sensitivity (P3
    "Sensitivity"): the (I, J) simple column replacing ``SIMPLE_COL`` for
    this run (default (4, 4)); the complex set is unchanged, and the column
    used is recorded as ``simple_col``. With both options off the result is
    exactly the preregistered dict (no ``s2_`` or ``simple_col`` keys).
    """
    sets = cell_sets(simple_col=simple_col)
    counts_s, times_s, counts_c, times_c = _load_grid(run_dir, sets["simple"], sets["complex"])
    rng = np.random.default_rng(seed)
    r_s = t1_t2(counts_s, rng, n_null)
    r_c = t1_t2(counts_c, rng, n_null)
    prefs_s = fit_prefs_all(counts_s)
    prefs_c = fit_prefs_all(counts_c)
    omap = _omega_map(input_dir, prefs_s, prefs_c, omegas)
    res3 = t3(counts_s, counts_c, rng)
    res4 = t4((times_s, times_c), omap, (prefs_s, prefs_c), rng, n_null)
    flat_vals = list(omap.values())
    omega_med = float(np.median(flat_vals)) if flat_vals else float("nan")
    res = {
        "seed": int(seed),
        "n_null": int(n_null),
        "omega": omega_med,
        "omega_bad_files": res4["omega_bad_files"],
        "omega_mismatches": res4["omega_mismatches"],
        "t1_r_ori_simple": r_s["r_ori"],
        "t1_r_ori_simple_null99": r_s["r_ori_null99"],
        "t1_r_ori_simple_npref99": r_s["r_ori_npref99"],
        "t1_simple_pass": r_s["t1_pass"],
        "t1_r_ori_complex": r_c["r_ori"],
        "t1_r_ori_complex_null99": r_c["r_ori_null99"],
        "t1_r_ori_complex_npref99": r_c["r_ori_npref99"],
        "t1_complex_pass": r_c["t1_pass"],
        "t2_r_sf_simple": r_s["r_sf"],
        "t2_r_sf_simple_null99": r_s["r_sf_null99"],
        "t2_r_sf_simple_npref99": r_s["r_sf_npref99"],
        "t2_simple_pass": r_s["t2_pass"],
        "t2_r_sf_complex": r_c["r_sf"],
        "t2_r_sf_complex_null99": r_c["r_sf_null99"],
        "t2_r_sf_complex_npref99": r_c["r_sf_npref99"],
        "t2_complex_pass": r_c["t2_pass"],
        "pref_ori_counts_simple": r_s["pref_ori_counts"],
        "pref_sf_counts_simple": r_s["pref_sf_counts"],
        "pref_ori_counts_complex": r_c["pref_ori_counts"],
        "pref_sf_counts_complex": r_c["pref_sf_counts"],
        "n_excluded_t12_simple": r_s["n_excluded"],
        "n_excluded_t12_complex": r_c["n_excluded"],
        "median_X_simple": res3["median_X_simple"],
        "median_X_complex": res3["median_X_complex"],
        "median_pmi_simple": res3["median_pmi_simple"],
        "median_pmi_complex": res3["median_pmi_complex"],
        "n_excluded_t3_simple": res3["n_excluded_simple"],
        "n_excluded_t3_complex": res3["n_excluded_complex"],
        "excluded_frac_simple": res3["excluded_frac_simple"],
        "excluded_frac_complex": res3["excluded_frac_complex"],
        "t3_simple_pass": res3["t3_simple_pass"],
        "t3_complex_pass": res3["t3_complex_pass"],
        "median_L_simple": res4["median_L_simple"],
        "median_L_complex": res4["median_L_complex"],
        "t4_diff": res4["diff"],
        "t4_diff_null99": res4["diff_null99"],
        "n_excluded_t4_simple": res4["n_excluded_simple"],
        "n_excluded_t4_complex": res4["n_excluded_complex"],
        "t4_pass": res4["t4_pass"],
        "all_pass": bool(
            r_s["t1_pass"] and r_c["t1_pass"] and r_s["t2_pass"] and r_c["t2_pass"]
            and res3["t3_simple_pass"] and res3["t3_complex_pass"] and res4["t4_pass"]
        ),
    }
    if simple_col is not None:
        res["simple_col"] = [int(simple_col[0]), int(simple_col[1])]
    if s2_all_phases:
        # Same nulls and floors as the folded run, fit and scored on all
        # phases 0-9 (P3 S2). Drawn after the main T1-T4 nulls so enabling S2
        # never changes the main keys.
        all_phases = list(range(N_PHASE))
        r2_s = t1_t2(counts_s, rng, n_null, fit_phases=all_phases, score_phases=all_phases)
        r2_c = t1_t2(counts_c, rng, n_null, fit_phases=all_phases, score_phases=all_phases)
        for tag, r2 in (("simple", r2_s), ("complex", r2_c)):
            res[f"s2_t1_r_ori_{tag}"] = r2["r_ori"]
            res[f"s2_t1_r_ori_{tag}_null99"] = r2["r_ori_null99"]
            res[f"s2_t1_r_ori_{tag}_npref99"] = r2["r_ori_npref99"]
            res[f"s2_t1_{tag}_pass"] = r2["t1_pass"]
            res[f"s2_t2_r_sf_{tag}"] = r2["r_sf"]
            res[f"s2_t2_r_sf_{tag}_null99"] = r2["r_sf_null99"]
            res[f"s2_t2_r_sf_{tag}_npref99"] = r2["r_sf_npref99"]
            res[f"s2_t2_{tag}_pass"] = r2["t2_pass"]
            res[f"s2_pref_ori_counts_{tag}"] = r2["pref_ori_counts"]
            res[f"s2_pref_sf_counts_{tag}"] = r2["pref_sf_counts"]
            res[f"s2_n_excluded_t12_{tag}"] = r2["n_excluded"]
    return res


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
    ap.add_argument("--s2-all-phases", action="store_true",
                    help="S2 sensitivity (P3): refit T1/T2 preferences on all "
                         "phases 0-9 and score on the same phases, beside the "
                         "folded result (keys prefixed s2_). T3/T4 unchanged.")
    ap.add_argument("--simple-col", nargs=2, type=int, default=None, metavar=("I", "J"),
                    help="S3 sensitivity (P3): simple column replacing (4, 4) "
                         "for this run; the complex set is unchanged. The "
                         "column used is recorded as simple_col.")
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
    res = score(args.run_dir, args.input_dir, seed=args.seed, n_null=args.n_null,
                s2_all_phases=args.s2_all_phases,
                simple_col=tuple(args.simple_col) if args.simple_col is not None else None)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} all_pass={res['all_pass']}")
    return 0


def _sha256_of_this_file():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
