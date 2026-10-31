"""Cell Reports 2025 figure runners for the CAMINOS etude (Figures 3, 4, S3).

Packet FIG-A. Protocols from the Cell Reports 2025 supplement (mmc2.txt):

* Figure 3 (results lines 370-385, legend lines 439-445): silencing —
  a fraction of PV or SOM cells is removed from the network (Methods
  "Causal manipulations", lines 1770-1772); the LFP spectrum, the 25-45 Hz
  band power and the maximum E firing rate are reported per fraction.
* Figure 4 (results lines 420-435, legend lines 500-504): spike-time
  disruption — a fraction of PV or SOM cells is replaced by independent
  Poissonian spike trains at the group's mean rate (Methods lines
  1772-1774), so the mean rate is preserved while the timing is randomised.
* Figure S3 (legend lines 1852-1860): SOM is fully silenced from about
  6.04 s; the control and silenced E-rate traces are compared over the
  5.9-6.2 s window around the onset.

Every run discards the first 2 s and analyses the following 10 s window
(Methods lines 1788-1790; Figure 3 legend line 445, Figure 4 legend line
509). The network is the ``param_set="cellrep"`` set of
``caminos_jax.py`` (NE 8000, NPV 800, NSOM 600, p 0.017) with the
``ext_sources="pNE"`` drive, dt = 1e-4 s, T = 12 s; ``sizes`` and other
``simulate`` keywords may be overridden for small smoke nets. Full-size
sweeps over seed grids are the dispatcher's job, not this module's.

The LFP proxy is the ref. 42 method of ``jaxfne.analysis.ref42_lfp`` (E
spikes per bin, Gaussian-smoothed over the whole run so the window edges
carry no truncated kernel); the colour-map magnitude comes from
``caminos_analysis.colour_map_spectrum``, which applies the 2 s discard
and the paper's normalisation to that proxy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_dir = str(Path(__file__).resolve().parent)
_root = str(Path(__file__).resolve().parents[4])
for _p in (_dir, _root):  # P-025: the checkout's jaxfne, and the sibling port
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import numpy as np

import caminos_analysis as cca
import caminos_jax as cj
from jaxfne.analysis import ref42_lfp

PARAM_SET = "cellrep"  # Cell Reports 2025 notebook set throughout
DT_S = 1e-4  # s, Euler step (cellrep set)
DISCARD_S = cca.DISCARD_S  # 2 s, Methods lines 1788-1790
F_PK_LO_HZ, F_PK_HI_HZ = 5.0, 100.0  # f_pk search band for the colour maps
S3_ONSET_S = 6.04  # s, Figure S3 legend line 1855 ("~ 6.04 s")
S3_WINDOW_S = (5.9, 6.2)  # s, Figure S3 axis range (lines 1849-1850)


def run_cell(seed, *, T_s=12.0, silence=None, replace=None, replace_rate_hz=None,
             silence_onset_s=None, sizes=None, raster_window=None,
             ext_sources="pNE", dt_s=DT_S, **sim_kw):
    """One cellrep simulation plus the Figure 3/4 estimators (mmc2 lines 439-445, 500-509).

    Runs :func:`caminos_jax.simulate` with ``param_set="cellrep"``,
    ``record="raster"`` and ``ext_sources="pNE"`` at dt = 1e-4 s, under the
    optional ``silence`` / ``replace`` / ``replace_rate_hz`` /
    ``silence_onset_s`` manipulations (dicts such as ``{"PV": 0.4}``,
    caminos_jax.py lines 544-549; fraction 0 is passed as ``None`` by the
    sweeps). ``sizes`` is ``{"NE": .., "NPV": .., "NSOM": ..}`` or None for
    the cellrep defaults; ``sim_kw`` forwards further ``simulate`` keywords
    (smoke nets only). ``raster_window`` is an optional ``(t0_s, t1_s)``
    pair; the boolean raster rows over that window are returned as
    ``raster_window`` (the full raster never leaves this function).

    Returns a dict: ``seed``, ``T_s``, ``dt_s``, ``sizes`` (E/PV/SOM),
    ``counts_kept`` (per-step ``[kept_steps, 3]`` counts after the 2 s
    discard), ``counts`` (per-type totals over the kept window),
    ``rates_Hz`` (mean rate per cell over the kept window, mmc2 Eq. 5,
    lines 1711-1720), ``lfp`` (the ref. 42 proxy over the whole run:
    ``ref42_lfp`` with ``drop_s=0.0, keep_s=T_s`` — the same whole-run
    smoothing as its default call, returned unwindowed so the 2 s discard
    is applied exactly once, by :func:`caminos_analysis.colour_map_spectrum`),
    ``spectrum`` (``{"freqs_hz", "magnitude"}`` from the colour-map
    estimator), ``f_pk_hz`` (magnitude argmax over 5-100 Hz),
    ``band_power_25_45`` (25-45 Hz integral, mmc2 lines 1741-1753),
    ``max_e_rate_hz`` (peak smoothed E rate, Figures 3C/4 lines 442-443,
    503-504) and, when asked, ``raster_window``.
    """
    NE, NPV, NSOM = _sizes_tuple(sizes)
    key = jax.random.PRNGKey(seed)
    raster, _ = cj.simulate(
        key, T_s, dt=dt_s, record="raster", param_set=PARAM_SET,
        ext_sources=ext_sources, NE=NE, NPV=NPV, NSOM=NSOM,
        silence=silence, replace=replace, replace_rate_hz=replace_rate_hz,
        silence_onset_s=silence_onset_s, **sim_kw)
    r = np.asarray(raster)
    counts = np.stack([r[:, :NE].sum(axis=1), r[:, NE:NE + NPV].sum(axis=1),
                       r[:, NE + NPV:].sum(axis=1)], axis=1).astype(np.int64)
    keep = counts[cca.kept_slice(counts.shape[0], dt_s)]
    n_tot = NE + NPV + NSOM
    out = {
        "seed": int(seed), "T_s": float(T_s), "dt_s": float(dt_s),
        "sizes": {"E": NE, "PV": NPV, "SOM": NSOM},
        "counts_kept": keep,
        "counts": {p: int(keep[:, i].sum()) for i, p in enumerate(cca.POPS)},
        "rates_Hz": {p: float(keep[:, i].mean() / (n * dt_s))
                     for i, (p, n) in enumerate(zip(cca.POPS, (NE, NPV, NSOM)))},
        "lfp": ref42_lfp(counts, dt_s, drop_s=0.0, keep_s=float(T_s)),
    }
    freqs, mag = cca.colour_map_spectrum(out["lfp"], dt_s)
    band = (freqs >= F_PK_LO_HZ) & (freqs <= F_PK_HI_HZ)
    if not np.any(band):
        raise ValueError("no frequency bin in [%g, %g] Hz" % (F_PK_LO_HZ, F_PK_HI_HZ))
    out["spectrum"] = {"freqs_hz": freqs, "magnitude": mag}
    out["f_pk_hz"] = float(freqs[band][int(np.argmax(mag[band]))])
    out["band_power_25_45"] = cca.band_power(mag, freqs, 25.0, 45.0)
    out["max_e_rate_hz"] = cca.max_rate_over_time(keep[:, 0], dt_s, n_e=NE)
    if raster_window is not None:
        out["raster_window"] = _raster_rows(r, raster_window, dt_s, T_s, n_tot)
    return out


def _sizes_tuple(sizes):
    """NE/NPV/NSOM from a ``{"NE": ..}`` dict, or the cellrep defaults."""
    if sizes is None:
        P = cj.CELLREP_PARAMS
        return P["NE"], P["NPV"], P["NSOM"]
    missing = {"NE", "NPV", "NSOM"} - set(sizes)
    if missing:
        raise ValueError("sizes must have the keys NE, NPV, NSOM; missing %s"
                         % sorted(missing))
    return int(sizes["NE"]), int(sizes["NPV"]), int(sizes["NSOM"])


def _raster_rows(raster, window_s, dt_s, T_s, n_tot):
    """Raster rows over a ``[t0_s, t1_s)`` window of one finished run."""
    t0, t1 = (float(x) for x in window_s)
    if not (0.0 <= t0 < t1 <= T_s):
        raise ValueError("raster_window must satisfy 0 <= t0 < t1 <= T_s=%g, "
                         "got (%g, %g)" % (T_s, t0, t1))
    i0, i1 = int(round(t0 / dt_s)), int(round(t1 / dt_s))
    return np.asarray(raster[i0:i1])


def _e_rate_series(e_counts, dt_s, n_e, sigma_ms=1.0):
    """Smoothed E rate in Hz per cell: the :func:`caminos_analysis.max_rate_over_time`
    kernel and normalisation (mmc2 Eq. 5, lines 1711-1720), kept as a series."""
    kernel = cca._gaussian_kernel(float(sigma_ms) / (float(dt_s) * 1000.0))
    return np.convolve(np.asarray(e_counts, dtype=float), kernel, mode="same") \
        / (float(n_e) * float(dt_s))


def _manip_rows(sweep, pop, fractions, seeds, run_kw, extra):
    """Shared sweep loop; fraction 0 passes the manipulation as None."""
    if pop not in ("PV", "SOM"):
        raise ValueError("pop must be 'PV' or 'SOM', got %r" % (pop,))
    rows = []
    for fraction in fractions:
        frac = float(fraction)
        if not 0.0 <= frac <= 1.0:
            raise ValueError("fraction must be in [0, 1], got %r" % (fraction,))
        spec = None if frac == 0.0 else {pop: frac}
        for seed in seeds:
            res = run_cell(seed, **{**run_kw, sweep: spec}, **extra)
            res.update({"pop": pop, "fraction": frac, "seed": int(seed), **extra})
            rows.append(res)
    return rows


def silence_sweep(pop, fractions, seeds, **kw):
    """Figure 3 sweep: silence ``fractions`` of ``pop`` (mmc2 lines 439-445, 1770-1772).

    For each fraction and seed, :func:`run_cell` with ``silence={pop: f}``;
    fraction 0 passes ``silence=None`` (the paired baseline). ``kw`` goes to
    :func:`run_cell`. Returns one result dict per (fraction, seed), each
    carrying ``pop``, ``fraction`` and ``seed``.
    """
    return _manip_rows("silence", pop, fractions, seeds, kw, {})


def replace_sweep(pop, fractions, seeds, rate_hz, **kw):
    """Figure 4 sweep: replace ``fractions`` of ``pop`` by Poisson spikes at
    ``rate_hz`` per cell (mmc2 lines 500-504, 1772-1774).

    ``rate_hz`` is supplied by the caller: the paired baseline's mean rate
    of that group, so the mean modulatory effect is preserved while the
    spike times are randomised. For each fraction and seed, :func:`run_cell`
    with ``replace={pop: f}`` and ``replace_rate_hz={pop: rate_hz}``;
    fraction 0 passes ``replace=None``. ``kw`` goes to :func:`run_cell`.
    Rows carry ``pop``, ``fraction``, ``seed``, ``rate_hz`` and ``replace_rate_hz``.
    """
    rows = _manip_rows("replace", pop, fractions, seeds, kw,
                       {"replace_rate_hz": {pop: float(rate_hz)}})
    for row in rows:
        row["rate_hz"] = float(rate_hz)
    return rows


def s3_paired(seed, *, onset_s=S3_ONSET_S, T_s=12.0, **kw):
    """Figure S3 pair (mmc2 lines 1852-1860): SOM fully silenced from ``onset_s``
    (default 6.04 s) against the unmanipulated control, same seed.

    The window is half-open [5.9, 6.2). Returns a dict with ``t_window_s``,
    ``e_rate_control_hz`` and ``e_rate_silenced_hz`` (E rate in Hz per cell
    over the window, smoothed with the
    :func:`caminos_analysis.max_rate_over_time` kernel: the raster is taken
    padded by 0.05 s on each side so the kernel has full support, smoothed,
    then sliced back to the window), and the two full :func:`run_cell`
    dicts as ``control`` / ``silenced`` (each with its 5.9-6.2 s
    ``raster_window``). ``kw`` goes to :func:`run_cell`. Raises ValueError
    when ``onset_s >= T_s``, when the run ends before the Figure S3 window
    does, or when the padded window [5.85, 6.25) s exceeds the run.
    """
    onset = float(onset_s)
    T = float(T_s)
    if onset >= T:
        raise ValueError("onset_s must be < T_s, got onset=%g, T=%g" % (onset, T))
    t0, t1 = S3_WINDOW_S
    if t1 > T:
        raise ValueError("the Figure S3 window ends at %g s; T_s=%g is too short"
                         % (t1, T))
    p0, p1 = t0 - 0.05, t1 + 0.05
    if not 0.0 <= p0 or not p1 <= T:
        raise ValueError("the padded Figure S3 window [%g, %g) s exceeds [0, %g) s"
                         % (p0, p1, T))
    win = {"raster_window": (p0, p1)}
    control = run_cell(seed, T_s=T, **win, **kw)
    silenced = run_cell(seed, T_s=T, silence={"SOM": 1.0}, silence_onset_s=onset,
                        **win, **kw)
    dt = control["dt_s"]
    i0, i1, i0p = int(round(t0 / dt)), int(round(t1 / dt)), int(round(p0 / dt))
    off = i0 - i0p
    t = (i0 + np.arange(i1 - i0)) * dt
    out = {"onset_s": onset, "T_s": T, "dt_s": dt, "t_window_s": t,
           "control": control, "silenced": silenced}
    for name, run in (("e_rate_control_hz", control),
                      ("e_rate_silenced_hz", silenced)):
        n_e = run["sizes"]["E"]
        counts_pad = np.asarray(run["raster_window"])[:, :n_e].sum(axis=1)
        out[name] = _e_rate_series(counts_pad, dt, n_e)[off:off + (i1 - i0)]
    for run in (control, silenced):
        run["raster_window"] = run["raster_window"][off:off + (i1 - i0)]
    return out