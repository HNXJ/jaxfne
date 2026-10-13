"""Detectors and estimators for the Izhikevich 2003 P3 study.

Spec: ``artifacts/etudes/izhikevich2003/PREREG_P3.md`` ("Fig. 2" time and window
rules, "Fig. 3", "Peak excess", "Detector calibration").

Times are integer kernel rows. Row k is stamped (k + 1) dt, the input of row k
acts during step k, so an event whose input starts at row e is at e dt. Every
window, ISI and time comparison below is in rows; times are converted to ms only
for reporting.

The aperiodic model and its bounded fit are imported from CAMINOS, not copied.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.signal import welch

_CAMINOS = Path(__file__).resolve().parents[1] / "caminos" / "caminos_analysis.py"
_spec = importlib.util.spec_from_file_location("caminos_analysis", _CAMINOS)
_caminos = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_caminos)
aperiodic_log10 = _caminos.aperiodic_log10
fit_aperiodic = _caminos.fit_aperiodic

BURST_ISI_MS = 10.0  # Fig. 2 definitions: inner ISIs < 10 ms
RATE_WINDOW_MS = 300.0  # Fig. 2 definitions: analysis window after the event
C8_MIN_PROMINENCE_MV = 0.05  # Fig. 2 C8 local maxima
WELCH_NPERSEG = 2000  # Peak excess step 1: 0.5 Hz bins at fs = 1000
WELCH_NOVERLAP = 1000


def spike_rows(spikes_bool_1d):
    """Row indices of the True entries (kernel threshold crossings), ascending ints."""
    return [int(r) for r in np.flatnonzero(np.asarray(spikes_bool_1d, dtype=bool))]


def bursts(rows, dt_ms):
    """Maximal runs of >= 2 consecutive spikes whose inner ISIs are all < 10 ms.

    The comparison is in rows against round(10 / dt_ms), so an ISI of exactly
    10.0 ms is not inside a burst.
    """
    thr = round(BURST_ISI_MS / dt_ms)
    out = []
    run = []
    for r in rows:
        r = int(r)
        if run and r - run[-1] < thr:
            run.append(r)
            continue
        if len(run) >= 2:
            out.append(run)
        run = [r]
    if len(run) >= 2:
        out.append(run)
    return out


def window_rows(rows, event_row, dt_ms):
    """Spikes in the 300 ms after the event row.

    Row k is stamped (k + 1) dt and the event at row e is at e dt, so the spikes
    stamped in (e dt, e dt + 300 ms] are the rows e <= k < e + 300 / dt.
    """
    n = round(RATE_WINDOW_MS / dt_ms)
    return [int(r) for r in rows if event_row <= int(r) < event_row + n]


def rate_hz(window):
    """Spike count in the window divided by 0.3 s."""
    return len(window) / (RATE_WINDOW_MS / 1000.0)


def _isis(window):
    return np.diff(np.asarray(window, dtype=np.int64))


def isi_ratio(window):
    """Last ISI / first ISI, in rows. None with fewer than 2 ISIs (3 spikes)."""
    isi = _isis(window)
    if isi.size < 2:
        return None
    return float(isi[-1] / isi[0])


def isi_cv(window):
    """std(ISI, ddof=0) / mean(ISI). None with fewer than 2 ISIs (3 spikes)."""
    isi = _isis(window).astype(float)
    if isi.size < 2:
        return None
    return float(np.std(isi, ddof=0) / np.mean(isi))


def c2_burst_then_tail(window, dt_ms):
    """C2: an initial burst (the burst holding the window's first spike), then a tail.

    The tail is every spike after the initial burst; it passes with >= 3 spikes and
    every tail ISI >= 10 ms (in rows). Returns False when there is no initial burst.
    """
    window = [int(r) for r in window]
    found = bursts(window, dt_ms)
    if not window or not found or found[0][0] != window[0]:
        return False
    tail = window[len(found[0]):]
    if len(tail) < 3:
        return False
    return bool(np.all(_isis(tail) >= round(BURST_ISI_MS / dt_ms)))


def burst_onset_frequency_hz(window, dt_ms):
    """C3: 1 / mean interval between burst onsets, in Hz. None with fewer than 2 bursts."""
    onsets = [b[0] for b in bursts(window, dt_ms)]
    if len(onsets) < 2:
        return None
    return float(1000.0 / (np.mean(np.diff(onsets)) * dt_ms))


def c8_frequency_hz(maxima, dt_ms):
    """C8 reported frequency: 1 / time between the first two maxima, in Hz. None below 2."""
    if len(maxima) < 2:
        return None
    return float(1000.0 / ((maxima[1] - maxima[0]) * dt_ms))


def c8_maxima(v, start_row):
    """C8 local maxima of v in the window v[start_row:], as absolute row indices.

    Interior samples only (the window's first and last sample excluded), with
    v_i > v_{i-1}, v_i >= v_{i+1} and prominence >= 0.05 mV. Prominence is v_i minus
    the larger of two minima: the minimum of v between v_i and the nearest
    higher-or-equal sample on the left (to the window start if there is none), and
    the same on the right (to the window end if there is none). All searches stay
    inside the window.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    out = []
    for i in range(start_row + 1, n - 1):
        if not (v[i] > v[i - 1] and v[i] >= v[i + 1]):
            continue
        j = i - 1
        while j > start_row and v[j] < v[i]:
            j -= 1
        k = i + 1
        while k < n - 1 and v[k] < v[i]:
            k += 1
        prominence = v[i] - max(v[j:i + 1].min(), v[i:k + 1].min())
        if prominence >= C8_MIN_PROMINENCE_MV:
            out.append(i)
    return out


def _no_peak(f_pk):
    return {"f_pk": f_pk, "pe": None, "local_max": False, "edge_ok": False}


def peak_excess(signal, fs, band, fit_hz=(2, 200), excl_hz=2, edge_hz=1):
    """Peak excess of a population signal in ``band`` (PREREG_P3 "Peak excess", steps 1-6).

    Returns a dict with f_pk (Hz), pe (PE, or None when a fit does not converge),
    local_max (f_pk is a strict local maximum of the PSD; true by step 3) and edge_ok
    (f_pk is at least ``edge_hz`` from both band edges). A non-converging fit, a PSD
    with a zero bin in the fit range (no log), or a band with no strict PSD local
    maximum returns pe None.
    Use ``holds(result, theta)`` for the band call.
    """
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    freqs, pxx = welch(x, fs=fs, window="hann", nperseg=WELCH_NPERSEG,
                       noverlap=WELCH_NOVERLAP, detrend=False, scaling="density")
    fit_band = (freqs >= fit_hz[0]) & (freqs <= fit_hz[1])
    lo, hi = band
    in_band = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    if in_band.size == 0:
        raise ValueError("no frequency bins in band %r" % (band,))
    if not np.all(pxx[fit_band] > 0):
        return _no_peak(None)
    try:
        pars1 = fit_aperiodic(freqs, pxx, fit_band)
    except RuntimeError:
        return _no_peak(None)
    resid = np.log10(pxx) - aperiodic_log10(freqs, *pars1)
    # Step 3 (v7): candidates are the in-band bins that are strict local maxima of the PSD.
    cand = in_band[(in_band > 0) & (in_band < pxx.size - 1)]
    cand = cand[(pxx[cand] > pxx[cand - 1]) & (pxx[cand] > pxx[cand + 1])]
    if cand.size == 0:
        return _no_peak(None)
    i = int(cand[np.argmax(resid[cand])])
    f_pk = float(freqs[i])
    try:
        pars2 = fit_aperiodic(freqs, pxx, fit_band & (np.abs(freqs - f_pk) > excl_hz))
    except RuntimeError:
        return _no_peak(f_pk)
    background = float(10.0 ** aperiodic_log10(f_pk, *pars2))
    pe = float(pxx[i]) / background
    local_max = bool(0 < i < pxx.size - 1 and pxx[i] > pxx[i - 1] and pxx[i] > pxx[i + 1])
    edge_ok = bool(abs(f_pk - lo) >= edge_hz and abs(f_pk - hi) >= edge_hz)
    return {"f_pk": f_pk, "pe": pe, "local_max": local_max, "edge_ok": edge_ok}


def holds(result, theta):
    """Band call (PREREG_P3 step 6): PE >= theta, f_pk a local maximum, f_pk >= 1 Hz from both edges."""
    return bool(result["pe"] is not None and result["pe"] >= theta
                and result["local_max"] and result["edge_ok"])
