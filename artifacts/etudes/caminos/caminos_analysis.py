"""R1-R4 estimators and scorer for the L2 CAMINOS etude.

Spec: ``artifacts/etudes/caminos/PREREG_P3.md`` ("Runs" for the signal and
kept window, "Estimands and criteria" for the estimators and pass bands).

Input is per-step population spike counts ``[steps, 3]`` in [E | PV | SOM]
order as returned by ``caminos_jax.simulate`` (int-like), with step ``dt``
in seconds and population ``sizes`` as ``{"E": .., "PV": .., "SOM": ..}``.

Estimators (prereg text):

* Rates (R3): mean rate per population over the kept window (first 2 s
  discarded).
* Spectrum (R1, R4): E population rate in ``dt`` bins, minus its mean,
  Welch 1 s Hann windows with 50 % overlap; peak frequency in 20-80 Hz
  and peak ratio (peak power / median power in 20-80 Hz). A run
  "oscillates" when the ratio is at least 5.
* Lag (R2): PV and SOM rates smoothed with a 2 ms Gaussian, peak of the
  cross-correlation over -14..+14 ms; positive means SOM after PV.
* Undefined rules (prereg text): a population with zero spikes in the
  kept window leaves its ratio and its lags undefined; the run counts
  as not oscillating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DISCARD_S = 2.0  # prereg Runs: first 2 s discarded (4 x tau_w)
WIN_S = 1.0  # prereg: Welch 1 s Hann windows
OVERLAP_FRAC = 0.5  # prereg: 50 % overlap
BAND_LO_HZ = 20.0  # prereg: 20-80 Hz band
BAND_HI_HZ = 80.0
RATIO_THRESHOLD = 5.0  # prereg: oscillates when peak ratio >= 5
LAG_WINDOW_MS = 14.0  # prereg R2: search -14..+14 ms
SMOOTH_MS = 2.0  # prereg R2: 2 ms Gaussian smoothing
POPS = ("E", "PV", "SOM")


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


def population_rates(counts, dt, sizes):
    """Mean rate (Hz) per population over the kept window (R3)."""
    arr = _as_float_counts(counts)
    keep = arr[kept_slice(arr.shape[0], dt)]
    dur = keep.shape[0] * dt
    return {p: float(keep[:, i].sum() / dur / sizes[p]) for i, p in enumerate(POPS)}


def welch_numpy(x, fs, nperseg, noverlap):
    """Welch power density with Hann windows (fallback, no scipy needed).

    Same convention as ``scipy.signal.welch`` with ``detrend=False`` and
    ``scaling="density"``: segments demeaned here, scale
    ``1 / (fs * sum(win**2))``, one-sided doubling off DC/Nyquist.
    """
    x = np.asarray(x, dtype=float)
    step = nperseg - noverlap
    win = np.hanning(nperseg)
    scale = 1.0 / (fs * float(np.sum(win**2)))
    acc = None
    n = 0
    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start:start + nperseg]
        seg = seg - seg.mean()
        spec = np.fft.rfft(win * seg)
        pxx = (spec.real**2 + spec.imag**2) * scale
        if nperseg % 2 == 0:
            pxx[1:-1] *= 2.0
        else:
            pxx[1:] *= 2.0
        acc = pxx if acc is None else acc + pxx
        n += 1
    if n == 0:
        raise ValueError("signal shorter than one window: %d < %d" % (len(x), nperseg))
    return np.fft.rfftfreq(nperseg, 1.0 / fs), acc / n


def welch_spectrum(x, fs, nperseg, noverlap):
    """Welch density via scipy when present, else :func:`welch_numpy`."""
    try:
        from scipy.signal import welch as _welch
    except ImportError:
        return welch_numpy(x, fs, nperseg, noverlap)
    return _welch(x, fs=fs, window="hann", nperseg=nperseg,
                  noverlap=noverlap, detrend=False, scaling="density")


def spectral_peak(e_rate, dt):
    """Peak frequency and peak ratio of the E rate in 20-80 Hz (R1, R4).

    ``e_rate`` is the E population rate in ``dt`` bins; the mean is
    removed here (prereg Runs). Returns ``(peak_freq_Hz, peak_ratio)``.
    """
    x = np.asarray(e_rate, dtype=float)
    x = x - x.mean()
    fs = 1.0 / dt
    nperseg = int(round(WIN_S * fs))
    freqs, pxx = welch_spectrum(x, fs, nperseg, nperseg // 2)
    band = (freqs >= BAND_LO_HZ) & (freqs <= BAND_HI_HZ)
    if not band.any():
        raise ValueError("no frequency bins in 20-80 Hz")
    fb, pb = freqs[band], pxx[band]
    i = int(np.argmax(pb))
    return float(fb[i]), float(pb[i] / np.median(pb))


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
    cc = np.correlate(som_s - som_s.mean(), pv_s - pv_s.mean(), mode="full")
    lags = np.arange(-(len(pv_s) - 1), len(pv_s))
    max_lag = int(round(LAG_WINDOW_MS / dt_ms))
    sel = np.abs(lags) <= max_lag
    lag_steps = int(lags[sel][int(np.argmax(cc[sel]))])
    lag_ms = lag_steps * dt_ms  # positive: SOM after PV
    return float(lag_ms)


def analyze_run(counts, dt, sizes):
    """Full R1-R4 estimator set for one run; undefined per prereg text."""
    arr = _as_float_counts(counts)
    keep = arr[kept_slice(arr.shape[0], dt)]
    spikes = {p: int(keep[:, i].sum()) for i, p in enumerate(POPS)}
    rates = population_rates(arr, dt, sizes)
    out = {
        "rates_Hz": rates,
        "spikes_kept": spikes,
        "peak_freq_E_Hz": None,
        "peak_ratio_E": None,
        "oscillates": False,
        "lag_PV_SOM_ms": None,
        "undefined": [],
    }
    if spikes["E"] == 0:
        out["undefined"].append("ratio")
    else:
        freq, ratio = spectral_peak(keep[:, 0] / dt / sizes["E"], dt)
        out["peak_freq_E_Hz"] = freq
        out["peak_ratio_E"] = ratio
        out["oscillates"] = bool(ratio >= RATIO_THRESHOLD)
    if spikes["PV"] == 0 or spikes["SOM"] == 0:
        out["undefined"].append("lag")
    else:
        out["lag_PV_SOM_ms"] = pv_som_lag(arr, dt, sizes)
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def score(run_dir):
    """Score a run directory written by ``run_caminos.py`` (R1-R4 table).

    R1: mean peak 30-40 Hz and every seed oscillates. R2: lag positive
    in every seed, mean 6-11 ms. R3: PV > SOM > E in every seed, mean E
    rate 1-3 Hz. R4: per-p groups; a p "oscillates" when every seed at
    that p oscillates; pass needs p = 0.02 silent and p = 0.07
    oscillating, else ``pass`` is None.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    dt = float(manifest["dt"])
    sizes = manifest["sizes"]
    per_run = []
    for entry in manifest["runs"]:
        counts = np.load(run_dir / entry["file"])
        res = analyze_run(counts, dt, sizes)
        res["seed"] = entry["seed"]
        res["p"] = entry["p"]
        per_run.append(res)

    # R1-R3 score the reference model only: p = 0.07, seeds 1-5 (prereg Runs);
    # the R4 grid and the seed-0 pilot may share the directory.
    ref = [r for r in per_run if abs(r["p"] - 0.07) < 1e-12 and r["seed"] in (1, 2, 3, 4, 5)]
    r1 = {
        "n_runs": len(ref),
        "mean_peak_freq_Hz": _mean([r["peak_freq_E_Hz"] for r in ref]),
        "all_oscillate": bool(ref) and all(r["oscillates"] for r in ref),
    }
    r1["pass"] = (r1["mean_peak_freq_Hz"] is not None
                  and 30.0 <= r1["mean_peak_freq_Hz"] <= 40.0
                  and r1["all_oscillate"])

    lags = [r["lag_PV_SOM_ms"] for r in ref]
    r2 = {
        "mean_lag_ms": _mean(lags),
        "all_positive": all(v is not None and v > 0 for v in lags),
    }
    r2["pass"] = (r2["mean_lag_ms"] is not None and bool(lags)
                  and all(v is not None and v > 0 for v in lags)
                  and 6.0 <= r2["mean_lag_ms"] <= 11.0)

    r3 = {
        "order_ok_every_seed": bool(ref) and all(
            r["rates_Hz"]["PV"] > r["rates_Hz"]["SOM"] > r["rates_Hz"]["E"]
            for r in ref),
        "mean_E_Hz": _mean([r["rates_Hz"]["E"] for r in ref]),
    }
    r3["pass"] = (r3["order_ok_every_seed"] and r3["mean_E_Hz"] is not None
                  and 1.0 <= r3["mean_E_Hz"] <= 3.0)

    by_p = {}
    for r in per_run:
        if r["seed"] in (1, 2, 3):  # prereg R4: seeds 1-3 at each p
            by_p.setdefault(r["p"], []).append(r["oscillates"])
    p_osc = {p: (len(v) > 0 and all(v)) for p, v in by_p.items()}
    lo = next((p for p in by_p if abs(p - 0.02) < 1e-12), None)
    hi = next((p for p in by_p if abs(p - 0.07) < 1e-12), None)
    osc_ps = sorted(p for p, o in p_osc.items() if o)
    r4 = {"p_oscillates": {str(p): o for p, o in sorted(p_osc.items())},
          "smallest_oscillating_p": osc_ps[0] if osc_ps else None}
    if lo is None or hi is None:
        r4["pass"] = None
    else:
        r4["pass"] = (not p_osc[lo]) and p_osc[hi]

    return {"per_run": per_run, "r1": r1, "r2": r2, "r3": r3, "r4": r4}


def main(argv=None):
    """CLI: score a run directory, write JSON (default ``score.json``)."""
    ap = argparse.ArgumentParser(description="Score a CAMINOS run directory (PREREG_P3.md R1-R4).")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    result = score(args.run_dir)
    out = Path(args.out) if args.out else Path(args.run_dir) / "score.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)
    return result


if __name__ == "__main__":
    main()
