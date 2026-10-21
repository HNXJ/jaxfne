"""Unscored diagnostic: sensitivities S6 and S7 of the v5 T2 scorer.

Declared in artifacts/etudes/pcl/prereg/P3_fig3_v5.md: A2 gives S6 (K/k from the
measured stimulus periods), A3 gives S7 (response window from 62.5 ms). Reads the
scored run tree and the stimulus tree (np.load, allow_pickle=False). Writes one
thing: a new section appended once to
artifacts/etudes/pcl/prereg/P3_fig3_diagnostic.md. Nothing is scored; the scorer,
the result file and scores/ are not touched. Run with -I. A second run refuses to
append a duplicate section.

Repo-root guard (P-025): parents[4] of this file is the repo root.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PORT = HERE.parent / "port"
if str(PORT) not in sys.path:
    sys.path.insert(0, str(PORT))

import numpy as np  # noqa: E402

import fig3_analysis as fa  # noqa: E402
import fig3_v5_t2 as v5  # noqa: E402

RUN_DIR = "F:/warehouse/pcl/runs/fig3_cSTDP3"
INPUT_DIR = "F:/warehouse/pcl/data/PCL_datasets_and_networks/grating/counterphase_gratingComplex/test7"
SCORED_JSON = HERE.parent / "prereg" / "scores" / "fig3_v5_t2_even.json"
DIAG_MD = HERE.parent / "prereg" / "P3_fig3_diagnostic.md"
TITLE = "v5 T2 sensitivities S6 and S7 (unscored, 2026-10-13)"

T_WINDOW_S = 62500 / 1e6  # notebook start, read as an absolute event timestamp (us)
PERIOD_MIN_PX = 2.0
PERIOD_MAX_PX = 20.0
FREQ_STEP = 1e-4  # cycles per px, fine grid (not FFT bins)
PRIMARY_ORI = "0"
CHECK_ORI = "90"


def default_map(f, i):
    """The scorer's K/k map: (20 - sf*) / (20 - i)."""
    return (20 - f) / (20 - i)


def stim_x(ori_name, sf_i):
    """Event x coordinates of ori, sf index, phase 0, trial 0 (read only)."""
    path = fa._input_path(INPUT_DIR, fa.ORIS.index(ori_name), sf_i, 0, 0)
    with np.load(path, allow_pickle=False) as z:
        return z["arr_1"].astype(np.int64)


def spatial_period(x):
    """Dominant period (px) of the per-column event count over the patch.

    The patch is the x span of the events. The profile is mean-subtracted and
    evaluated on a fine frequency grid over 1/20 to 1/2 cycles per px. A peak
    at the 1/2 edge (2 px) is flagged: sub-2 px structure cannot be resolved
    on an integer pixel grid and folds into that edge.
    """
    cols = x - x.min()
    width = int(cols.max()) + 1
    if width < 3:
        raise ValueError(f"patch width {width} px is too small for a period")
    prof = np.bincount(cols, minlength=width).astype(np.float64)
    prof -= prof.mean()
    pos = np.arange(width, dtype=np.float64)
    f_lo, f_hi = 1.0 / PERIOD_MAX_PX, 1.0 / PERIOD_MIN_PX
    n_step = int(round((f_hi - f_lo) / FREQ_STEP))
    freqs = np.linspace(f_lo, f_hi, n_step + 1)
    amp = np.abs(np.exp(-2j * np.pi * np.outer(freqs, pos)) @ prof)
    k = int(np.argmax(amp))
    edge = ""
    if k == len(freqs) - 1:
        edge = "2 px edge"
    elif k == 0:
        edge = "20 px edge"
    return {"period": float(1.0 / freqs[k]), "amp": float(amp[k]), "width": width,
            "edge": edge, "n_events": int(x.size)}


def periods(ori_name):
    return [spatial_period(stim_x(ori_name, i)) for i in range(fa.N_SF)]


def recount(times, shape, t_min):
    """Spike counts per cell from the spike times with t >= t_min (seconds)."""
    out = np.zeros(shape, dtype=np.int64)
    n_cells = shape[4]
    for (oi, f, p, s), (pos, t) in times.items():
        keep = t >= t_min
        out[oi, f, p, s, :] = np.bincount(pos[keep], minlength=n_cells)
    return out


def tuning_curve(counts, kind, xmap, phases=v5.EVEN_PHASES):
    """v5 T2 steps 3-6 with the K/k map ``xmap(sf_star, i)``.

    Same code path as ``fig3_v5_t2.tuning_curve``; only the K/k map differs.
    fit_prefs (steps 1-2) is the scorer's own.
    """
    ph = [int(p) for p in phases]
    c = np.asarray(counts, dtype=np.float64)
    n_cells = c.shape[4]
    r = c.mean(axis=3)
    rph = r[:, :, ph, :]
    o_star, f_star = v5.fit_prefs(counts, phases)
    bins = {}
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
            key = v5._key(xmap(f, sf_i))
            if 0.4 <= float(key) <= 2.0:
                bins.setdefault(key, []).append(float(g[sf_i]))
    curve = {}
    for key in sorted(bins, key=float):
        v = np.asarray(bins[key], dtype=np.float64)
        curve[key] = {"mean": float(v.mean()),
                      "se": float(v.std(ddof=0) / np.sqrt(v.size)),
                      "n": int(v.size)}
    return curve, n_skipped


def score_map(counts_s, counts_c, xmap, ref):
    out = {}
    for name, counts, kind, panel in (
        ("simple", counts_s, "simple", v5.PANEL_SIMPLE),
        ("complex", counts_c, "complex", v5.PANEL_COMPLEX),
    ):
        curve, n_skip = tuning_curve(counts, kind, xmap)
        crit = v5.apply_criterion(curve, panel, ref)
        out[name] = {"curve": curve, "n_skipped": int(n_skip),
                     "n_cells": int(counts.shape[4]), **crit}
    return out


def curve_gap(curve, ref_curve):
    if set(curve) != set(ref_curve):
        return float("inf")
    return max(max(abs(curve[k]["mean"] - ref_curve[k]["mean"]),
                   abs(curve[k]["se"] - ref_curve[k]["se"]),
                   0.0 if curve[k]["n"] == ref_curve[k]["n"] else 1.0)
               for k in curve)


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def row_summary(set_name, sensitivity, s):
    n_ok = sum(p["ok"] for p in s["points"])
    return (f"| {set_name} | {sensitivity} | {s['frac']:.3f} ({n_ok}/{len(s['points'])}) "
            f"| {s['sse'] / s['sse_flat']:.2f} | {s['t2_pass']} | {s['n_cells']} ({s['n_skipped']}) |")


def point_table(res):
    rows = ["| K/k | fig | ours | ok |", "|---|---|---|---|"]
    for p in res["points"]:
        rows.append(f"| {p['j']:.1f} | {p['fig']:.3f} | {fmt(p['ours'])} | {p['ok']} |")
    return rows


def main():
    ref = v5.load_reference()
    scored = json.loads(SCORED_JSON.read_text(encoding="utf-8"))
    sets = fa.cell_sets()
    counts_s, times_s, counts_c, times_c = fa._load_grid(RUN_DIR, sets["simple"], sets["complex"])

    # Guard: spike times must reproduce the counts before any window is applied.
    full_s = recount(times_s, counts_s.shape, -np.inf)
    full_c = recount(times_c, counts_c.shape, -np.inf)
    if not (np.array_equal(full_s, counts_s) and np.array_equal(full_c, counts_c)):
        raise SystemExit("recount from spike times does not reproduce counts; stopped, nothing appended")

    # Guard: the default-map curves must reproduce the scored JSON, else stop.
    default = score_map(counts_s, counts_c, default_map, ref)
    gaps = {}
    for name in ("simple", "complex"):
        gaps[name] = curve_gap(default[name]["curve"], scored[name]["curve"])
        for key in ("frac", "sse", "sse_flat"):
            if abs(default[name][key] - scored[name][key]) > 1e-9:
                raise SystemExit(f"default {name} {key} does not reproduce the scored JSON; stopped")
    if max(gaps.values()) > 1e-9:
        raise SystemExit(f"default curves do not reproduce the scored JSON: {gaps}; stopped")

    # S6: K/k = period(sf*) / period(i), periods measured from the stimulus.
    per_0 = periods(PRIMARY_ORI)
    per_90 = periods(CHECK_ORI)
    p0 = [p["period"] for p in per_0]
    s6 = score_map(counts_s, counts_c, lambda f, i: p0[f] / p0[i], ref)

    # S7: response window from 62.5 ms, default K/k map.
    win_s = recount(times_s, counts_s.shape, T_WINDOW_S)
    win_c = recount(times_c, counts_c.shape, T_WINDOW_S)
    s7 = score_map(win_s, win_c, default_map, ref)

    lines = [f"## {TITLE}", ""]
    lines.append("Script: `artifacts/etudes/pcl/diag/q6_v5_s6_s7.py`. Unscored; the verdict, the scorer "
                 "and `scores/` are unchanged. Even phases 0, 2, 4, 6, 8; reference and criterion from "
                 "`port/fig3_v5_t2.py`. Steps 1 to 4 and 6 are the prereg's; only the K/k map differs.")
    lines.append("")
    lines.append(f"Check: the default map reproduces `prereg/scores/fig3_v5_t2_even.json` "
                 f"(max curve gap simple {gaps['simple']:.1e}, complex {gaps['complex']:.1e}; "
                 f"frac, SSE and flat match).")
    lines.append("")
    lines.append("### S6 periods (px per cycle; DFT over 2 to 20 px, step 1e-4 cycles/px)")
    lines.append("")
    lines.append(f"| sf index | 20 - i | ori {PRIMARY_ORI}: period | width | edge | "
                 f"ori {CHECK_ORI}: period | width | edge |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i in range(fa.N_SF):
        a, b = per_0[i], per_90[i]
        lines.append(f"| {i} | {20 - i} | {a['period']:.2f} | {a['width']} | {a['edge'] or '-'} "
                     f"| {b['period']:.2f} | {b['width']} | {b['edge'] or '-'} |")
    lines.append("")
    lines.append("| Set | Sensitivity | frac (pass) | SSE/flat | T2 pass | cells (skipped) |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(row_summary("simple", "default", default["simple"]))
    lines.append(row_summary("simple", "S6", s6["simple"]))
    lines.append(row_summary("simple", "S7", s7["simple"]))
    lines.append(row_summary("complex", "default", default["complex"]))
    lines.append(row_summary("complex", "S6", s6["complex"]))
    lines.append(row_summary("complex", "S7", s7["complex"]))
    lines.append("")
    lines.append("Simple, per point under S6 (K/k from measured periods):")
    lines.append("")
    lines.extend(point_table(s6["simple"]))
    lines.append("")
    lines.append("Simple, per point under S7 (window from 62.5 ms):")
    lines.append("")
    lines.extend(point_table(s7["simple"]))
    lines.append("")

    text = "\n".join(lines) + "\n"
    existing = DIAG_MD.read_bytes()
    if TITLE.encode("utf-8") in existing:
        raise SystemExit(f"section already present in {DIAG_MD}; nothing appended")
    sep = b"\n" if existing.endswith(b"\n") else b"\n\n"
    DIAG_MD.write_bytes(existing + sep + text.encode("utf-8"))

    print(f"appended to {DIAG_MD}")
    print(text)
    print(f"q6_v5_s6_s7 DONE simple_s6={s6['simple']['t2_pass']} complex_s6={s6['complex']['t2_pass']} "
          f"simple_s7={s7['simple']['t2_pass']} complex_s7={s7['complex']['t2_pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
