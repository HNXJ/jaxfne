"""R1-R4 estimators and scorer for the L2 CAMINOS etude.

Spec: ``artifacts/etudes/caminos/PREREG_P3.md`` ("Runs" for the signal and
kept window, "Estimands and criteria" for the estimators and pass bands).

Input is per-step population spike counts ``[steps, 3]`` in [E | PV | SOM]
order as returned by ``caminos_jax.simulate`` (int-like), with step ``dt``
in seconds and population ``sizes`` as ``{"E": .., "PV": .., "SOM": ..}``.

Estimators (prereg text):

* Rates (R3): mean rate per population over the kept window (first 2 s
  discarded).
* Spectrum (R1, R4; v4 criterion): E population rate in ``dt`` bins, minus
  its mean, Welch 1 s Hann windows with 50 % overlap. Aperiodic fit
  log10 P = a - log10(k + f**chi) on 10-150 Hz; f_pk is the largest residual
  in 20-80 Hz; the background is refit with f_pk +- 8 Hz excluded; peak excess
  PE = P(f_pk) / background(f_pk). A run "oscillates" when PE >= pe_threshold
  (default 4, the v4 value; the v6 calibration gate sets it) and f_pk is not
  within 2 Hz of 20 or 80 Hz. v6 dropped the PSD local-maximum test; ``local_max``
  is reported only.
* Lag (R2): PV and SOM rates smoothed with a 2 ms Gaussian, peak of the
  cross-correlation over -14..+14 ms; positive means SOM after PV.
* Undefined rules (prereg text): a population with zero spikes in the
  kept window leaves its ratio and its lags undefined; the run counts
  as not oscillating.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# P-025 repo-root guard: pull jaxfne from this checkout, never site-packages.
_ROOT = str(Path(__file__).resolve().parents[4])
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)

import jaxfne as jtfne  # noqa: E402

assert "site-packages" not in jtfne.__file__, "caminos_analysis must run the checkout's jaxfne"

from jaxfne.analysis import (  # noqa: E402
    aperiodic_log10,
    aperiodic_spectral_peak,
    fit_aperiodic,
    oscillation_call,
    pairwise_phase_consistency,
    population_rates,
    rate_welch_psd,
    welch_numpy,
    welch_spectrum,
    xcorr_peak_lag,
)

import numpy as np

DISCARD_S = 2.0  # prereg Runs: first 2 s discarded (4 x tau_w)
WIN_S = 1.0  # prereg: Welch 1 s Hann windows
OVERLAP_FRAC = 0.5  # prereg: 50 % overlap
FIT_LO_HZ = 10.0  # v4: aperiodic fit band
FIT_HI_HZ = 150.0
BAND_LO_HZ = 20.0  # v4: peak search band
BAND_HI_HZ = 80.0
EXCL_HALF_HZ = 8.0  # v4: refit excludes f_pk +- 8 Hz
PE_THRESHOLD = 4.0  # v4: oscillates when peak excess >= 4
EDGE_HZ = 2.0  # v4: f_pk not within 2 Hz of 20 or 80 Hz
CHI_BOUNDS = (0.0, 5.0)  # v4: exponent bounds
K_MIN = 0.0  # v4: offset lower bound
INIT_K = 1.0  # v4: fit start, k0
INIT_CHI = 1.0  # v4: fit start, chi0 (a0 is set from the data)
LAG_WINDOW_MS = 14.0  # prereg R2: search -14..+14 ms
SMOOTH_MS = 2.0  # prereg R2: 2 ms Gaussian smoothing
POPS = ("E", "PV", "SOM")
REF_P = 0.07  # prereg Runs: reference connectivity
REF_SEEDS = (1, 2, 3, 4, 5)  # prereg Runs: R1-R3 seeds
R4_P_GRID = tuple(round(0.02 + 0.01 * i, 2) for i in range(9))  # prereg R4
R4_LOW_P = 0.02  # prereg R4: must be silent
R4_SEEDS = (1, 2, 3)  # prereg R4: seeds per p
REF_DTYPE = "float32"  # prereg Runs
REF_DT = 1e-4  # s, prereg Runs Euler step
REF_T = 12.0  # s, prereg Runs duration (v6; kept window is REF_T - DISCARD_S)
REF_STEPS = int(round(REF_T / REF_DT))


def _as_float_counts(counts):
    arr = np.asarray(counts)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("counts must be [steps, 3], got shape %r" % (arr.shape,))
    return arr


def kept_slice(n_steps, dt):
    """Row slice after discarding the first 2 s (prereg Runs)."""
    cut = int(round(DISCARD_S / dt))
    if cut >= n_steps:
        raise ValueError("run shorter than the 2 s discard: %d steps at dt=%g" % (n_steps, dt))
    return slice(cut, None)


def e_welch(e_rate, dt):
    """Welch PSD of the E rate as the scorer computes it: mean removed, 1 s Hann, 50 % overlap."""
    return rate_welch_psd(e_rate, dt, win_s=WIN_S, overlap_frac=OVERLAP_FRAC)


def spectral_peak(e_rate, dt, pe_threshold=PE_THRESHOLD):
    """Peak excess and oscillation call for the E rate (R1, R4; v4).

    ``e_rate`` is the E population rate in ``dt`` bins; the mean is removed
    here (prereg Runs). ``pe_threshold`` is the peak excess that counts as
    oscillating (default 4, the v4 value; the v6 calibration sets it). Returns
    a dict: ``peak_freq_Hz`` (f_pk), ``peak_excess`` (PE), ``background_Hz``
    (refit background at f_pk), ``local_max``, ``edge_ok``, ``oscillates``, the
    refit parameters ``fit`` and the first fit's parameters ``fit_pass1``.
    """
    freqs, pxx = e_welch(e_rate, dt)
    pk = aperiodic_spectral_peak(
        freqs, pxx,
        fit_lo_hz=FIT_LO_HZ, fit_hi_hz=FIT_HI_HZ,
        band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ,
        excl_half_hz=EXCL_HALF_HZ, edge_hz=EDGE_HZ,
    )
    f_pk = pk["peak_freq_Hz"]
    pe = pk["peak_excess"]
    background = pk["background_Hz"]
    local_max = pk["local_max"]
    edge_ok = pk["edge_ok"]
    pars1 = pk["fit_pass1"]
    pars2 = pk["fit"]
    return {
        "peak_freq_Hz": f_pk,
        "peak_excess": pe,
        "background_Hz": background,
        "local_max": local_max,
        "edge_ok": edge_ok,
        "oscillates": oscillation_call(pe, edge_ok, pe_threshold),
        "fit": pars2,
        "fit_pass1": pars1,
    }


def _gaussian_kernel(sigma_steps):
    radius = int(3 * sigma_steps)
    g = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (g / sigma_steps) ** 2)
    return k / k.sum()


def pv_som_lag(counts, dt, sizes):
    """PV->SOM rate cross-correlation peak lag in ms (R2).

    Rates smoothed with a 2 ms Gaussian; peak searched over -14..+14 ms.
    Positive means SOM after PV.
    """
    arr = _as_float_counts(counts)
    keep = arr[kept_slice(arr.shape[0], dt)]
    dt_ms = dt * 1000.0
    pv = keep[:, 1] / dt / sizes["PV"]
    som = keep[:, 2] / dt / sizes["SOM"]
    kernel = _gaussian_kernel(SMOOTH_MS / dt_ms)
    pv_s = np.convolve(pv, kernel, mode="same")
    som_s = np.convolve(som, kernel, mode="same")
    lag_steps = xcorr_peak_lag(som_s, pv_s, dt_ms, window_ms=LAG_WINDOW_MS)
    lag_ms = lag_steps * dt_ms  # positive: SOM after PV
    return float(lag_ms)


def analyze_run(counts, dt, sizes, pe_threshold=PE_THRESHOLD):
    """Full R1-R4 estimator set for one run; undefined per prereg text.

    ``pe_threshold`` is passed to :func:`spectral_peak` (default 4, v4).
    """
    arr = _as_float_counts(counts)
    keep = arr[kept_slice(arr.shape[0], dt)]
    spikes = {p: int(keep[:, i].sum()) for i, p in enumerate(POPS)}
    rates = population_rates(
        arr, sizes=sizes, pops=POPS, dt_ms=dt * 1000.0, discard_ms=DISCARD_S * 1000.0
    )
    out = {
        "rates_Hz": rates,
        "spikes_kept": spikes,
        "peak_freq_E_Hz": None,
        "peak_excess_E": None,
        "background_E_Hz": None,
        "peak_local_max": None,
        "peak_edge_ok": None,
        "fit_E": None,
        "oscillates": False,
        "lag_PV_SOM_ms": None,
        "undefined": [],
    }
    if spikes["E"] == 0:
        out["undefined"].append("peak")
    else:
        pk = spectral_peak(keep[:, 0] / dt / sizes["E"], dt, pe_threshold)
        out["peak_freq_E_Hz"] = pk["peak_freq_Hz"]
        out["peak_excess_E"] = pk["peak_excess"]
        out["background_E_Hz"] = pk["background_Hz"]
        out["peak_local_max"] = pk["local_max"]
        out["peak_edge_ok"] = pk["edge_ok"]
        out["fit_E"] = {"refit": list(pk["fit"]), "pass1": list(pk["fit_pass1"])}
        out["oscillates"] = pk["oscillates"]
    if spikes["PV"] == 0 or spikes["SOM"] == 0:
        out["undefined"].append("lag")
    else:
        out["lag_PV_SOM_ms"] = pv_som_lag(arr, dt, sizes)
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def ppc_by_type(phase_groups, types):
    """Pairwise phase consistency (PPC) per neuron, summarised by cell type.

    Per-neuron values come from :func:`jaxfne.analysis.pairwise_phase_consistency`
    (PPC; Vinck et al. 2012, paper ref. 36): the mean of ``cos(phi_i - phi_j)``
    over unordered spike-phase pairs. A neuron with fewer than 2 spikes has no
    PPC and gets NaN; a type whose neurons all lack a value maps to NaN.
    """
    groups = list(phase_groups)
    labels = list(types)
    if len(groups) != len(labels):
        raise ValueError("phase_groups and types differ in length: %d vs %d"
                         % (len(groups), len(labels)))
    per, _ = pairwise_phase_consistency(groups)
    by_type = {}
    for lab in labels:
        if lab not in by_type:
            vals = [v for v, t in zip(per, labels) if t == lab and not np.isnan(v)]
            by_type[lab] = float(np.mean(vals)) if vals else float("nan")
    return {"per_neuron": per, "by_type": by_type}


def _same(a, b):
    """Equal p values (floats from JSON) within 1e-9."""
    return abs(float(a) - float(b)) < 1e-9


def _rows_at(per_run, p, seeds):
    return [r for r in per_run if _same(r["p"], p) and r["seed"] in seeds]


def _is_reference(entry, ext_sources="fixed"):
    """Scored configuration: seed not 0, not a pilot, float32, dt = 1e-4 (prereg).

    Seed 0 is the unscored pilot (prereg Prior exposure). Sensitivity runs
    (float64, other dt, the S3-S5 variants) and pilots may share a directory;
    they are listed as not scored and never enter R1-R4. Entries written
    before the variants existed carry no variant keys and count as defaults.
    ``ext_sources`` is the drive the scored set must use: "fixed" (v6) or
    "pNE" (amendment v7, R4).
    """
    return (entry.get("seed") != 0
            and not entry.get("pilot", False)
            and entry.get("dtype") == REF_DTYPE
            and entry.get("dt") is not None
            and abs(float(entry["dt"]) - REF_DT) < 1e-12
            and entry.get("indegree", "bernoulli") == "bernoulli"
            and entry.get("ext", "private") == "private"
            and entry.get("step_order", "default") == "default"
            and str(entry.get("ext_sources", "fixed")) == ext_sources)


def _t_ok(entry, steps):
    """Run lasted T = REF_T (12 s) at dt = 1e-4 (manifest value and saved step count)."""
    return entry.get("T") is not None and abs(float(entry["T"]) - REF_T) < 1e-12 and steps == REF_STEPS


def score(run_dir, pe_threshold=PE_THRESHOLD, ext_sources="fixed"):
    """Score a run directory written by ``run_caminos.py`` (R1-R4, prereg v6).

    ``ext_sources="pNE"`` scores the amendment v7 drive instead (R4v7 uses
    the same R4 rule); the default scores v6.

    Only reference-configuration runs are scored. A result is incomplete
    (pass False) unless R1-R3 have exactly seeds 1-5 at p = 0.07 (seed 0
    excluded), R4 has seeds 1-3 at all nine p, and every run in those sets
    has T = 12 s and one 40-character git SHA.

    R1: every seed oscillates with f_pk in 30-40 Hz. R2: lag positive in
    every seed, mean 6-11 ms. R3: PV > SOM > E in every seed, mean E rate
    1-3 Hz. R4: no seed oscillates at p = 0.02, and every seed oscillates
    at p = 0.07. ``pe_threshold`` is the oscillation threshold (default 4).
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if ext_sources not in ("fixed", "pNE"):
        raise ValueError("ext_sources must be 'fixed' or 'pNE', got %r" % (ext_sources,))
    selected = [e for e in manifest["runs"] if _is_reference(e, ext_sources)]
    not_scored = [e["file"] for e in manifest["runs"] if not _is_reference(e, ext_sources)]
    per_run = []
    not_t = []
    for entry in selected:
        counts = np.load(run_dir / entry["file"])
        if not _t_ok(entry, int(counts.shape[0])):
            not_t.append(entry["file"])  # listed as incomplete, never analysed
            continue
        res = analyze_run(counts, float(entry["dt"]), entry["sizes"], pe_threshold)
        res["seed"] = entry["seed"]
        res["p"] = entry["p"]
        res["file"] = entry["file"]
        res["T"] = entry["T"]
        res["git_sha"] = entry.get("git_sha")
        res["steps"] = int(counts.shape[0])
        per_run.append(res)

    ref_rows = _rows_at(per_run, REF_P, REF_SEEDS)
    shas = {e.get("git_sha") for e in selected}
    common = []
    if not (len(shas) == 1 and all(isinstance(s, str) and len(s) == 40 for s in shas)):
        common.append("scored runs do not share one 40-character git SHA (%d distinct)"
                      % len(shas))
    if not_t:
        common.append("runs not at T = %g s and dt = 1e-4: %s"
                      % (REF_T, ", ".join(sorted(not_t))))

    ref_seeds = sorted(r["seed"] for r in per_run if _same(r["p"], REF_P))
    ref_reasons = list(common)
    if ref_seeds != list(REF_SEEDS):
        ref_reasons.append("p = 0.07 seeds are %s, need exactly 1-5" % ref_seeds)
    ref_complete = not ref_reasons

    grid_reasons = list(common)
    for g in R4_P_GRID:
        seeds_g = sorted(r["seed"] for r in _rows_at(per_run, g, R4_SEEDS))
        if seeds_g != list(R4_SEEDS):
            grid_reasons.append("p = %g seeds are %s, need 1-3" % (g, seeds_g))
    grid_complete = not grid_reasons

    band_every = bool(ref_rows) and all(
        r["oscillates"] and r["peak_freq_E_Hz"] is not None
        and 30.0 <= r["peak_freq_E_Hz"] <= 40.0 for r in ref_rows)
    r1 = {
        "n_runs": len(ref_rows),
        "mean_peak_freq_Hz": _mean([r["peak_freq_E_Hz"] for r in ref_rows]),
        "all_oscillate": bool(ref_rows) and all(r["oscillates"] for r in ref_rows),
        "complete": ref_complete,
        "reasons": ref_reasons,
    }
    r1["pass"] = bool(ref_complete and band_every)

    lags = [r["lag_PV_SOM_ms"] for r in ref_rows]
    all_pos = bool(lags) and all(v is not None and v > 0 for v in lags)
    mean_lag = _mean(lags)
    r2 = {"mean_lag_ms": mean_lag, "all_positive": all_pos,
          "complete": ref_complete, "reasons": ref_reasons}
    r2["pass"] = bool(ref_complete and all_pos and mean_lag is not None
                      and 6.0 <= mean_lag <= 11.0)

    order_ok = bool(ref_rows) and all(
        r["rates_Hz"]["PV"] > r["rates_Hz"]["SOM"] > r["rates_Hz"]["E"] for r in ref_rows)
    mean_e = _mean([r["rates_Hz"]["E"] for r in ref_rows])
    r3 = {"order_ok_every_seed": order_ok, "mean_E_Hz": mean_e,
          "complete": ref_complete, "reasons": ref_reasons}
    r3["pass"] = bool(ref_complete and order_ok and mean_e is not None
                      and 1.0 <= mean_e <= 3.0)

    low = _rows_at(per_run, R4_LOW_P, R4_SEEDS)
    high = _rows_at(per_run, REF_P, R4_SEEDS)
    low_silent = bool(low) and not any(r["oscillates"] for r in low)
    high_all = bool(high) and all(r["oscillates"] for r in high)
    p_osc = {}
    for g in R4_P_GRID:
        rows = _rows_at(per_run, g, R4_SEEDS)
        p_osc[g] = bool(rows) and all(r["oscillates"] for r in rows)
    osc_ps = [g for g in R4_P_GRID if p_osc[g]]
    r4 = {"p_oscillates": {"%g" % g: p_osc[g] for g in R4_P_GRID},
          "smallest_oscillating_p": osc_ps[0] if osc_ps else None,
          "low_p_silent": low_silent, "ref_p_all_oscillate": high_all,
          "complete": grid_complete, "reasons": grid_reasons}
    r4["pass"] = bool(grid_complete and low_silent and high_all)

    return {"per_run": per_run, "not_scored": not_scored,
            "r1": r1, "r2": r2, "r3": r3, "r4": r4}


def main(argv=None):
    """CLI: score a run directory, write JSON (default ``score.json``)."""
    ap = argparse.ArgumentParser(description="Score a CAMINOS run directory (PREREG_P3.md R1-R4).")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ext-sources", choices=("fixed", "pNE"), default="fixed",
                    help="drive of the scored set: fixed 560 (v6) or p*NE (amendment v7)")
    args = ap.parse_args(argv)
    result = score(args.run_dir, ext_sources=args.ext_sources)
    out = Path(args.out) if args.out else Path(args.run_dir) / "score.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)
    return result


if __name__ == "__main__":
    main()
