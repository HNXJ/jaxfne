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

_repo_root = Path(__file__).resolve().parents[4]  # P-025: the checkout's jaxfne
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import jax
import numpy as np

import caminos_analysis as cca
import caminos_jax as cj
import jaxfne as jtfne

assert "site-packages" not in jtfne.__file__, "cellrep_figures must run the checkout's jaxfne"

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


# --- Cell Reports Figure 2 / Figure 6A runners (packet FIG-2/6A) ---

FIG2A_RATES_HZ = tuple(float(i) for i in range(11))  # Fig 2A grid (mmc2 lines 283-292)
FIG2B_TDRATES_HZ = tuple(float(i) for i in range(11))  # Fig 2B top grid (lines 312-320)
FIG6A_RATIOS = (0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # Fig 6A grid (lines 533-570)


def _attach_phase_delay(res, half_band_hz=5.0):
    """Ref. 42 PV->SOM phase at the run's ``f_pk_hz`` (Figures 2, 6A, S1 style).

    Calls :func:`jaxfne.analysis.ref42_phase_delay` on ``counts_kept`` with
    ``half_band_hz``, ``drop_s=0.0`` and ``keep_s = T_s - 2`` (the kept
    window starts after the 2 s discard, as in :func:`s1_sweep`). Sets
    ``phase_delay`` (the full dict) and ``phase_delay_rad`` (wrapped
    SOM-minus-PV, None when a group is silent). Both stay None when the
    spectrum is degenerate (``f_pk_hz <= half_band_hz``, e.g. a silent E
    group), where the band itself is refused.
    """
    from jaxfne.analysis import ref42_phase_delay

    f_pk = float(res["f_pk_hz"])
    half = float(half_band_hz)
    delay, rad = None, None
    if f_pk > half:
        dur = float(res["T_s"]) - DISCARD_S
        delay = ref42_phase_delay(np.asarray(res["counts_kept"]), res["dt_s"],
                                  f_pk, half_band_hz=half,
                                  drop_s=0.0, keep_s=dur)
        phi_pv, phi_som = delay["phase_PV_rad"], delay["phase_SOM_rad"]
        if phi_pv is not None and phi_som is not None:
            rad = float(np.angle(np.exp(1j * (phi_som - phi_pv))))
    res.update({"phase_delay": delay, "phase_delay_rad": rad})
    return res


def fig2a_sweep(rates_hz=None, seeds=None, **kw):
    """Figure 2A sweep: tonic E/PV feedforward rate 0-10 Hz (mmc2 lines 283-292,
    legend lines 394-399).

    For each rate and seed this runs :func:`run_cell` with the constant E/PV
    feedforward rate set to that value, via ``ext_rate_schedule`` (a constant
    per-step Hz array; ``caminos_jax.simulate`` replaces its constant rate
    with the schedule, docstring lines 561-562). ``rates_hz`` defaults to the
    Figure's grid; every rate must be >= 0. Rows carry ``rate_hz``, ``seed``
    and the :func:`run_cell` outputs plus ``phase_delay`` / ``phase_delay_rad``
    (:func:`_attach_phase_delay`). ``kw`` goes to :func:`run_cell`.
    """
    if rates_hz is None:
        rates_hz = FIG2A_RATES_HZ
    if seeds is None:
        raise ValueError("seeds must be given, got None")
    t_s = float(kw.get("T_s", 12.0))
    dt_s = float(kw.get("dt_s", DT_S))
    steps = int(round(t_s / dt_s))
    rows = []
    for rate in rates_hz:
        try:
            rv = float(rate)
        except (TypeError, ValueError):
            raise ValueError("rate_hz must be a number >= 0, got %r" % (rate,))
        if not (np.isfinite(rv) and rv >= 0.0):
            raise ValueError("rate_hz must be >= 0, got %r" % (rate,))
        sched = np.full(steps, rv)
        for seed in seeds:
            res = run_cell(seed, **{**kw, "ext_rate_schedule": sched})
            _attach_phase_delay(res)
            res.update({"rate_hz": rv, "seed": int(seed)})
            rows.append(res)
    return rows


def fig2b_sweep(tdrates_hz=None, seeds=None, **kw):
    """Figure 2B top sweep: top-down SOM drive 0-10 Hz at 4 Hz feedforward
    (mmc2 lines 312-320, legend line 400).

    As :func:`fig2a_sweep`, with the E/PV feedforward at its 4 Hz default and
    ``som_ext_rate_hz`` set to each value (``caminos_jax.simulate`` lines
    559-560). ``tdrates_hz`` defaults to the Figure's grid; every value must
    be >= 0. Rows carry ``tdrate_hz``, ``seed`` and the :func:`run_cell`
    outputs plus ``phase_delay`` / ``phase_delay_rad``.
    """
    if tdrates_hz is None:
        tdrates_hz = FIG2B_TDRATES_HZ
    if seeds is None:
        raise ValueError("seeds must be given, got None")
    rows = []
    for rate in tdrates_hz:
        try:
            rv = float(rate)
        except (TypeError, ValueError):
            raise ValueError("tdrate_hz must be a number >= 0, got %r" % (rate,))
        if not (np.isfinite(rv) and rv >= 0.0):
            raise ValueError("tdrate_hz must be >= 0, got %r" % (rate,))
        for seed in seeds:
            res = run_cell(seed, **{**kw, "som_ext_rate_hz": rv})
            _attach_phase_delay(res)
            res.update({"tdrate_hz": rv, "seed": int(seed)})
            rows.append(res)
    return rows


def fig6a_sweep(ratios=None, seeds=None, total=1400, **kw):
    """Figure 6A sweep: SOM/PV density ratio at fixed interneuron total
    (mmc2 lines 533-548, 558-570, legend lines 636-646).

    The x-axis is the SOM/PV density ratio ``r = NSOM/NPV`` with
    ``NPV + NSOM = total`` (the cellrep defaults 800/600 sum to 1400):
    ``NPV = round(total / (1 + r))``, ``NSOM = total - NPV``. E stays at
    ``NE = 8000`` (overridable as ``NE=...`` for smoke nets; ``sizes`` itself
    is built here and refused in ``kw``). For each ratio and seed this runs
    :func:`run_cell` with ``sizes = {"NE": NE, "NPV": NPV, "NSOM": NSOM}``.
    ``ratios`` defaults to the Figure's grid; every ratio must be > 0 and
    ``total`` a positive integer. Rows carry ``ratio``, ``NPV``, ``NSOM``,
    ``seed`` and the outputs plus ``phase_delay_rad``.
    """
    if ratios is None:
        ratios = FIG6A_RATIOS
    if seeds is None:
        raise ValueError("seeds must be given, got None")
    if isinstance(total, bool) or not isinstance(total, (int, np.integer)) \
            or int(total) <= 0:
        raise ValueError("total must be a positive integer, got %r" % (total,))
    total = int(total)
    if "sizes" in kw:
        raise ValueError("fig6a builds sizes from ratio/total; pass NE=... "
                         "to override the E count, got sizes=%r" % (kw["sizes"],))
    ne = kw.pop("NE", 8000)
    if isinstance(ne, bool) or not isinstance(ne, (int, np.integer)) \
            or int(ne) <= 0:
        raise ValueError("NE must be a positive integer, got %r" % (ne,))
    rows = []
    for ratio in ratios:
        try:
            rv = float(ratio)
        except (TypeError, ValueError):
            raise ValueError("ratio must be > 0, got %r" % (ratio,))
        if not (np.isfinite(rv) and rv > 0.0):
            raise ValueError("ratio must be > 0, got %r" % (ratio,))
        npv = int(round(total / (1.0 + rv)))
        nsom = total - npv
        if npv < 1 or nsom < 1:
            raise ValueError("ratio %r splits total %d into NPV=%d, NSOM=%d; "
                             "both must be >= 1" % (ratio, total, npv, nsom))
        sizes = {"NE": int(ne), "NPV": npv, "NSOM": nsom}
        for seed in seeds:
            res = run_cell(seed, **{**kw, "sizes": sizes})
            _attach_phase_delay(res)
            res.update({"ratio": rv, "NPV": npv, "NSOM": nsom,
                        "seed": int(seed)})
            rows.append(res)
    return rows


# --- Cell Reports Figure 2 bottom / Figure 6B runners (packet FIG-2B6B) ---

FIG2B_BOTTOM_AMP_HZ = 4.0  # Gaussian peak rate (mmc2 Methods lines 1688-1708)
FIG2B_BOTTOM_WIDTH_S = 0.1  # Gaussian width D (mmc2 Methods lines 1688-1708)


def _gaussian_drive_schedule(t0_s, T_s=12.0, dt_s=DT_S,
                             amp_hz=FIG2B_BOTTOM_AMP_HZ,
                             width_s=FIG2B_BOTTOM_WIDTH_S):
    """Gaussian drive ``amp_hz * exp(-(t - t0)**2 / (2 * width_s**2))`` (Hz).

    Sampled on the step grid (``t = i * dt_s``, length ``steps``) so it fits
    the ``ext_rate_schedule`` / ``som_ext_rate_schedule`` length check of
    :func:`caminos_jax.simulate` (Figure 2 bottom, mmc2 Methods 1688-1708).
    """
    steps = int(round(float(T_s) / float(dt_s)))
    t = np.arange(steps) * float(dt_s)
    return (float(amp_hz)
            * np.exp(-((t - float(t0_s)) ** 2) / (2.0 * float(width_s) ** 2)))


def fig2b_bottom_run(seed, *, t0_s, **kw):
    """Figure 2 bottom run: non-stationary Gaussian input (mmc2 Methods 1688-1708,
    text lines 313-320).

    Both the feedforward E/PV rate and the top-down SOM rate are the Gaussian
    ``4 * exp(-(t - t0)**2 / (2 * D**2))`` Hz with ``D = 0.1`` s, sampled on
    the step grid and passed as ``ext_rate_schedule`` / ``som_ext_rate_schedule``
    to :func:`run_cell` (with ``param_set="cellrep"``, ``ext_sources="pNE"``,
    dt 1e-4 s, T 12 s, 2 s discard unless overridden in ``kw``). ``t0_s`` is
    the caller-supplied peak time. Returns the :func:`run_cell` dict plus
    ``phase_delay`` / ``phase_delay_rad`` (:func:`_attach_phase_delay`).
    """
    if seed is None:
        raise ValueError("seed must be given, got None")
    try:
        t0 = float(t0_s)
    except (TypeError, ValueError):
        raise ValueError("t0_s must be a finite time in [0, T_s], got %r" % (t0_s,))
    if not np.isfinite(t0):
        raise ValueError("t0_s must be finite, got %r" % (t0_s,))
    t_s = float(kw.get("T_s", 12.0))
    if not 0.0 <= t0 <= t_s:
        raise ValueError("t0_s must be in [0, T_s=%g], got %r" % (t_s, t0_s))
    dt_s = float(kw.get("dt_s", DT_S))
    sched = _gaussian_drive_schedule(t0, T_s=t_s, dt_s=dt_s)
    kw = dict(kw)
    kw.setdefault("ext_sources", "pNE")
    res = run_cell(seed, **{**kw, "ext_rate_schedule": sched,
                            "som_ext_rate_schedule": sched})
    _attach_phase_delay(res)
    res.update({"t0_s": t0})
    return res


def fig6b_run(seed, *, gain, t_start_s, dur_s=0.3, **kw):
    """Figure 6B run: localized E-to-E increase for ``dur_s`` s
    (mmc2 lines 558-560, legend lines 646-649).

    ``ee_gain_schedule`` is 1.0 everywhere except ``gain`` on the steps with
    ``t_start_s <= t < t_start_s + dur_s``; the magnitude is a caller argument
    (the text states none). Same :func:`run_cell` arguments as
    :func:`fig2b_bottom_run` (T 12 s, 2 s discard unless overridden in ``kw``).
    Returns the :func:`run_cell` dict plus ``e_rate_hz_pre`` / ``e_rate_hz_window`` /
    ``e_rate_hz_post`` (mean E rate per cell over the three equal-length spans
    before, inside and after the window).
    """
    if seed is None:
        raise ValueError("seed must be given, got None")
    try:
        g = float(gain)
    except (TypeError, ValueError):
        raise ValueError("gain must be a number >= 0, got %r" % (gain,))
    if not (np.isfinite(g) and g >= 0.0):
        raise ValueError("gain must be finite and >= 0, got %r" % (gain,))
    try:
        dur = float(dur_s)
    except (TypeError, ValueError):
        raise ValueError("dur_s must be a number > 0, got %r" % (dur_s,))
    if not dur > 0.0:
        raise ValueError("dur_s must be > 0, got %r" % (dur_s,))
    try:
        t_start = float(t_start_s)
    except (TypeError, ValueError):
        raise ValueError("t_start_s must be a number, got %r" % (t_start_s,))
    t_s = float(kw.get("T_s", 12.0))
    dt_s = float(kw.get("dt_s", DT_S))
    if not (DISCARD_S <= t_start and t_start + dur <= t_s):
        raise ValueError("window [%g, %g) s must lie inside [discard=%g, T_s=%g] s, "
                         "got t_start_s=%r, dur_s=%r"
                         % (t_start, t_start + dur, DISCARD_S, t_s, t_start_s, dur_s))
    steps = int(round(t_s / dt_s))
    t = np.arange(steps) * dt_s
    sched = np.ones(steps)
    sched[(t >= t_start) & (t < t_start + dur)] = g
    kw = dict(kw)
    kw.setdefault("ext_sources", "pNE")
    res = run_cell(seed, **{**kw, "ee_gain_schedule": sched})
    keep = np.asarray(res["counts_kept"])[:, 0]
    cut = int(round(DISCARD_S / dt_s))
    n_e = res["sizes"]["E"]

    def _span_hz(a_s, b_s):
        i0 = max(int(round(a_s / dt_s)) - cut, 0)
        i1 = min(int(round(b_s / dt_s)) - cut, keep.shape[0])
        seg = keep[i0:i1]
        return float(seg.mean() / (n_e * dt_s)) if seg.size else 0.0

    res.update({
        "gain": g,
        "t_start_s": t_start,
        "dur_s": dur,
        "e_rate_hz_pre": _span_hz(t_start - dur, t_start),
        "e_rate_hz_window": _span_hz(t_start, t_start + dur),
        "e_rate_hz_post": _span_hz(t_start + dur, t_start + 2.0 * dur),
    })
    return res


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


# --- Cell Reports Figure 5 / S1 runners (packet FIG-5S1) ---

FIG5A_VALUES_NS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)  # Fig 5A grid (mmc2 lines 473-478)
FIG5B_VALUES_NS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)  # Fig 5B grid (mmc2 lines 479-480)
S1_PROBABILITIES = (0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25,
                    0.30)  # Fig S1 grid (legend mmc2 lines 1808-1812)
FIG5C_RASTER_WINDOW_S = (DISCARD_S, DISCARD_S + 0.3)  # 0.3 s after the discard


def coupling_sweep(kind, values_nS=None, seeds=None, **kw):
    """Figure 5A/5B coupling sweeps (mmc2 lines 473-480, legends 573-577).

    ``kind`` is ``"PV_SOM"`` (Figure 5A: PV-to-SOM swept 0-10 nS, colour-map
    LFP spectrum plus 25-45 Hz band power per value) or ``"SOM_SOM"``
    (Figure 5B: SOM-to-SOM swept 0-10 nS with PV-to-SOM held at zero, mean
    over the 10 s window); any other kind is refused. For each value and
    seed this runs :func:`run_cell` with ``coupling={kind: v * 1e-9}``;
    value 0 passes ``coupling=None`` (the paired baseline). The nS value is
    the per-connection conductance, scaled by 1e-9 to S, consistent with
    the cellrep ``Q_ext_S = 1.25e-9`` convention (caminos_jax.py
    CELLREP_PARAMS). Rows carry ``kind``, ``value_nS``, ``seed`` and the
    :func:`run_cell` outputs. ``values_nS`` defaults to the Figure's grid
    for that kind; every value must be >= 0.
    """
    if kind not in ("PV_SOM", "SOM_SOM"):
        raise ValueError("kind must be 'PV_SOM' or 'SOM_SOM', got %r" % (kind,))
    if values_nS is None:
        values_nS = FIG5A_VALUES_NS if kind == "PV_SOM" else FIG5B_VALUES_NS
    if seeds is None:
        raise ValueError("seeds must be given, got None")
    rows = []
    for value in values_nS:
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValueError("value_nS must be a number >= 0, got %r" % (value,))
        if not (np.isfinite(val) and val >= 0.0):
            raise ValueError("value_nS must be >= 0, got %r" % (value,))
        coup = None if val == 0.0 else {kind: val * 1e-9}
        for seed in seeds:
            res = run_cell(seed, **{**kw, "coupling": coup})
            res.update({"kind": kind, "value_nS": val, "seed": int(seed)})
            rows.append(res)
    return rows


def fig5c_swap(seed, **kw):
    """Figure 5C swap run (mmc2 lines 481-496, Methods 1775-1778).

    Runs :func:`run_cell` with ``swap=True`` (PV and SOM neuronal parameters
    exchanged, connectivity unchanged, the 4 Hz drive staying on the PV
    indices; caminos_jax.py ``swap``). ``raster_window`` passes through to
    :func:`run_cell` exactly as in the FIG-A runners; when the caller gives
    none, the first 0.3 s after the 2 s discard is used (fits both smoke
    T_s = 3 s and full T_s = 12 s runs; override via ``raster_window``).
    Adds ``locking_by_type`` (the :func:`caminos_analysis.locking_by_type`
    result over the window), ``ppc_by_type`` (its per-type PPC part) and
    ``preferred_phase_by_type`` (its per-type circular-mean spike phase).
    Returns the :func:`run_cell` dict with those keys added.
    """
    return _locking_run(seed, True, **kw)


def fig1c_control(seed, **kw):
    """Figure 1C control run (mmc2 lines 254-268): the unswapped network with the
    spike-LFP locking of :func:`fig5c_swap`, so Figure 5C has its comparison."""
    return _locking_run(seed, False, **kw)


def _locking_run(seed, swap, **kw):
    """:func:`run_cell` (``swap`` as given) plus per-type spike-LFP locking over the window."""
    kw = dict(kw)
    if kw.get("raster_window", None) is None:
        t0, t1 = FIG5C_RASTER_WINDOW_S
        if not t1 <= float(kw.get("T_s", 12.0)):
            raise ValueError("default Figure 5C window (%g, %g) s exceeds T_s=%g; "
                             "pass raster_window explicitly" % (t0, t1, kw.get("T_s", 12.0)))
        kw["raster_window"] = (t0, t1)
    res = run_cell(seed, **{**kw, "swap": swap})
    t0, t1 = (float(x) for x in kw["raster_window"])
    dt = res["dt_s"]
    i0, i1 = int(round(t0 / dt)), int(round(t1 / dt))
    lfp_win = np.asarray(res["lfp"])[i0:i1]
    lock = cca.locking_by_type(np.asarray(res["raster_window"]), res["sizes"],
                               lfp_win, dt)
    res["locking_by_type"] = lock
    res["ppc_by_type"] = lock["ppc"]
    res["preferred_phase_by_type"] = lock["mean_phase"]
    return res


def s1_sweep(probabilities=None, seeds=None, **kw):
    """Figure S1 connection-probability sweep (legend mmc2 lines 1808-1812).

    For each p and seed this runs :func:`run_cell` with
    ``ext_sources="fixed"`` and ``p=p`` (``run_cell`` forwards ``p`` through
    ``**sim_kw`` to ``caminos_jax.simulate``, which takes ``p`` under that
    name). The external source count is held fixed so only p changes; the
    Methods do not say how the drive scales with p (inferred, decided here).
    Rows carry ``p``, ``seed`` and the :func:`run_cell` outputs (colour-map
    spectrum and max E rate per p) plus ``phase_delay`` (the full
    :func:`jaxfne.analysis.ref42_phase_delay` dict on the kept PV/SOM
    counts at the run's ``f_pk_hz``) and ``phase_delay_rad`` (its wrapped
    SOM-minus-PV phase, None when a group is silent).
    ``probabilities`` defaults to the Figure's grid; every p must be in
    (0, 1).
    """
    from jaxfne.analysis import ref42_phase_delay

    if probabilities is None:
        probabilities = S1_PROBABILITIES
    if seeds is None:
        raise ValueError("seeds must be given, got None")
    rows = []
    for p in probabilities:
        try:
            pp = float(p)
        except (TypeError, ValueError):
            raise ValueError("p must be in (0, 1), got %r" % (p,))
        if not 0.0 < pp < 1.0:
            raise ValueError("p must be in (0, 1), got %r" % (p,))
        for seed in seeds:
            res = run_cell(seed, **{**kw, "ext_sources": "fixed", "p": pp})
            f_pk = float(res["f_pk_hz"])
            dur = float(res["T_s"]) - DISCARD_S
            delay = ref42_phase_delay(np.asarray(res["counts_kept"]), res["dt_s"],
                                      f_pk, half_band_hz=5.0,
                                      drop_s=0.0, keep_s=dur)
            phi_pv, phi_som = delay["phase_PV_rad"], delay["phase_SOM_rad"]
            if phi_pv is None or phi_som is None:
                phase_rad = None
            else:
                phase_rad = float(np.angle(np.exp(1j * (phi_som - phi_pv))))
            res.update({"p": pp, "seed": int(seed), "phase_delay": delay,
                        "phase_delay_rad": phase_rad})
            rows.append(res)
    return rows