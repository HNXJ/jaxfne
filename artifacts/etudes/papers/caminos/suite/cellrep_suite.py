"""Cell Reports CAMINOS dashboard suite: one panel function per paper panel.

Lane B of the CAMINOS figure suites (packet: Cell Reports figures 1-6, S1, S3).
Each function draws one paper panel from the run-JSON receipts under the
warehouse root (``F:/warehouse/caminos``: ``figB_out/<mode>.json``,
``figC_out/<dir>/<mode>.json``, ``s1_95d2161b/s1.json``) or from a raster
trace ``.npz`` (the shared trace format of ``run_plos_rasters.py`` and the
new ``run_cellrep_rasters.py``). No panel takes a hidden global or a
hard-coded warehouse path; the notebook passes the root and each panel's
source file, which every figure states in a small grey footnote together
with the seed statistic ("mean +/- SD over N seeds").

Panel -> data map (verdicts cited from ``../RESULT_FIGURES.md``):

======================  =======================================================
``fig1b_raster``        ``cellrep_rasters/seed1/baseline.npz``
``fig1c``               ``figC_out/c1/fig1c.json`` (PPC + preferred phase)
``fig1c_raster``        ``cellrep_rasters/seed1/baseline.npz``
``fig2a_top``           ``figB_out/fig2a.json`` (swept key ``rate_hz``)
``fig2b_top``           ``figB_out/fig2b.json`` (swept key ``tdrate_hz``)
``fig3a`` / ``fig3b``   ``figC_out/c3a|c3b/silence_som|silence_pv.json``
``fig3c``               the same two files (max E rate, top/bottom)
``fig4a`` / ``fig4b``   ``figC_out/c4a|c4b/replace_som|replace_pv.json``
``fig5a`` / ``fig5b``   ``figB_out/fig5a|fig5b.json`` (``value_nS``)
``fig5c``               ``cellrep_rasters/seed1/<swap raster>.npz``
``fig6a_bottom``        ``figB_out/fig6a.json`` (swept key ``ratio``)
``fig_s1``              ``s1_95d2161b/s1.json`` (swept key ``p``)
``fig3_raster``         ``cellrep_rasters/seed1/silence_*.npz``
``fig4_raster``         ``cellrep_rasters/seed1/replace_*.npz``
======================  =======================================================

Not run in the replication (stated, no code): 1A, 1D, 2A bottom, 2B bottom,
6A top, 6B; S3 beyond 6.2 s (``RESULT_FIGURES.md`` lines 200-201, 206-209).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

POPS = ("E", "PV", "SOM")
POP_COLOURS = {"E": "#d62728", "PV": "#1f77b4", "SOM": "#2ca02c"}
RASTER_SUBSAMPLE = {"E": 200, "PV": 50, "SOM": 50}  # at most, for legibility
TRACE_DT_S = 1e-4  # s, the shared trace format's step
DEFAULT_T0_S = 11.5  # T_s 12 s minus the 0.5 s raster window, when no manifest says T_s
MAP_GRID_HZ = (0.0, 100.0, 201)  # the maps' x axis: 0-100 Hz, as the paper plots it
SMOOTH_HZ = 1.0  # boxcar width over linear |X| before the log, Hz
CLIM_PERCENTILES = (2.0, 99.5)  # shared colour limits over all four maps

SWEEP_LABELS = {
    "rate_hz": "feedforward drive rate (Hz)",
    "tdrate_hz": "top-down drive rate to SOM (Hz)",
    "value_nS": "added coupling (nS)",
    "ratio": "SOM/PV density ratio",
    "p": "connection probability p",
    "fraction": "fraction manipulated",
}

# QuadMeshes of the drawn colour maps, in draw order; :func:`share_map_colours`
# gives them one common colour scale and empties the list.
_MAP_IMAGE_REGISTRY = []


# ---------------------------------------------------------------- loading

def load_rows(path):
    """Load a run-JSON receipt and return its rows list.

    The receipt format (``run_figA.py`` docstring) is a dict with top keys
    ``what, param_set, values, seeds, T_s, elapsed_s, rows``; each row carries
    the swept value under its own key (``rate_hz``, ``tdrate_hz``,
    ``value_nS``, ``ratio``, ``p``, ``fraction``), ``rates_Hz`` per
    population, ``f_pk_hz``, ``band_power_25_45``, ``max_e_rate_hz`` and the
    stored spectrum ``psd_freqs_hz`` / ``psd_mag``.
    """
    with open(os.fspath(path), "r", encoding="utf-8") as fh:
        return json.load(fh)["rows"]


def _manifest_t0_s(trace_path):
    """Raster-window start for a trace without ``t0_s``: T_s - 0.5 from the
    run's ``manifest.json`` (folder of the seed folder, then the seed folder
    itself), else 11.5."""
    for folder in (Path(trace_path).parent, Path(trace_path).parent.parent):
        manifest = folder / "manifest.json"
        if manifest.is_file():
            with open(manifest, "r", encoding="utf-8") as fh:
                t_s = json.load(fh).get("T_s")
            if t_s:
                return float(t_s) - 0.5
    return DEFAULT_T0_S


def load_trace(path):
    """Load one raster trace ``.npz`` into a plain dict.

    Shared trace format: ``raster`` bool (steps, neurons) over the window at
    dt 1e-4 s; ``type_slices`` int (3, 2) = [start, stop) columns for E, PV,
    SOM; ``rate_hz`` float (bins, 3) per 1 ms bin, Hz per cell; scalars
    ``f_pk_hz``, ``band_power_25_45``, ``max_e_rate_hz``. Cell Reports files
    add ``lfp`` float (steps,) and ``t0_s``; PLoS files lack them, and then
    ``t0_s`` comes from :func:`_manifest_t0_s`.
    """
    with np.load(os.fspath(path), allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    out["t0_s"] = float(out["t0_s"]) if "t0_s" in out else _manifest_t0_s(path)
    out["lfp"] = np.asarray(out["lfp"], dtype=float) if "lfp" in out else None
    return out


def load_trace_or_none(path):
    """:func:`load_trace`, or None when the trace file does not exist yet."""
    return load_trace(path) if Path(path).is_file() else None


def _as_trace(trace):
    return trace if isinstance(trace, dict) else load_trace(trace)


# ---------------------------------------------------------------- helpers

def _group_by(rows, swept_key):
    """Group rows by swept value, sorted ascending: [(value, rows), ...]."""
    groups = {}
    for row in rows:
        groups.setdefault(row[swept_key], []).append(row)
    return sorted(groups.items(), key=lambda kv: float(kv[0]))


def _n_seeds(rows):
    """Number of distinct seeds in ``rows`` (0 when rows carry no seed key)."""
    seeds = {r.get("seed") for r in rows}
    seeds.discard(None)
    return len(seeds)


def _mean_sd(per_group):
    """Seed mean and mean +/- SD per swept value (NaN where a group has no
    usable rows, e.g. every seed's delay is None or the group is masked)."""
    mean = np.array([np.mean(v) if len(v) else np.nan for v in per_group])
    sd = np.array([np.std(v) if len(v) else np.nan for v in per_group])
    return mean, mean - sd, mean + sd


def _plot_mean_spread(ax, xs, per_group, colour, label):
    """One curve with its mean +/- SD seed band on ``ax`` (NaN breaks the line)."""
    mean, lo, hi = _mean_sd(per_group)
    ax.plot(xs, mean, "-o", color=colour, label=label, markersize=4)
    if len(xs) > 1:
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.18, linewidth=0)


def _row_oscillates(row):
    """A row oscillates when its band power is finite and its E rate is not 0."""
    bp = row.get("band_power_25_45")
    e_rate = (row.get("rates_Hz") or {}).get("E")
    bp_ok = bp is not None and np.isfinite(bp)
    rate_ok = e_rate is not None and float(e_rate) != 0.0
    return bool(bp_ok and rate_ok)


def _masked_per_group(groups, get):
    """Per-group values of ``get`` from oscillating rows only.

    Returns ``(per_group, silent_xs)``: one list of values per swept value
    (empty where no row oscillates) and the swept values of the silent
    groups, which the panels draw as open markers at the axis floor.
    """
    per_group, silent = [], []
    for value, rws in groups:
        vals = []
        for row in rws:
            if not _row_oscillates(row):
                continue
            v = get(row)
            if v is not None and np.isfinite(v):
                vals.append(v)
        per_group.append(vals)
        if not vals:
            silent.append(float(value))
    return per_group, silent


def _floor_markers(ax, silent_xs, label="no oscillation"):
    """Open markers at the axis floor for swept values without oscillation."""
    if not silent_xs:
        return
    floor = ax.get_ylim()[0]
    ax.plot(silent_xs, [floor] * len(silent_xs), ls="none", marker="o",
            markersize=5, markerfacecolor="none", markeredgecolor="#7f7f7f",
            label=label)


def _footnote(fig, source="", seeds=0, extra=""):
    """Small grey footnote at the figure bottom: data file, seed statistic, note."""
    parts = []
    if source:
        parts.append(source)
    if seeds:
        parts.append(f"mean \u00b1 SD over {seeds} seeds")
    if extra:
        parts.append(extra)
    if parts:
        fig.text(0.01, 0.01, "; ".join(parts), fontsize=6, color="#777777",
                 va="bottom")


def _mean_over_psd(groups):
    """Seed-mean ``psd_mag`` per swept value on the first row's freq grid."""
    freqs = np.asarray(groups[0][1][0]["psd_freqs_hz"], dtype=float)
    mag_rows = []
    for _, rws in groups:
        per_seed = [np.asarray(r["psd_mag"], dtype=float) for r in rws]
        on_grid = [np.interp(freqs, np.asarray(r["psd_freqs_hz"], dtype=float), m)
                   for r, m in zip(rws, per_seed)]
        mag_rows.append(np.mean(on_grid, axis=0))
    return freqs, np.array(mag_rows)


def _peak_mag(row, lo_hz=5.0, hi_hz=60.0):
    """Peak |X| of a row's stored spectrum over [lo_hz, hi_hz] (NaN if empty)."""
    freqs = np.asarray(row.get("psd_freqs_hz") or [], dtype=float)
    mags = np.asarray(row.get("psd_mag") or [], dtype=float)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    return float(np.max(mags[band])) if band.any() else float("nan")


def _rates_and_twin(rows, swept_key, title, twin_label, twin_of_row,
                    twin_unit="", rates=True, source=""):
    """Shared body of the swept-input panels: population rates on the left
    axis and one companion quantity (f_pk, band power, peak |X|) on a twin
    axis — or, when no rates are plotted, on the left axis itself. Companion
    values without oscillation (band power NaN or E rate 0) are masked and
    drawn as open markers at the companion axis floor."""
    groups = _group_by(rows, swept_key)
    xs = [float(v) for v, _ in groups]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    companion = ax.twinx() if rates else ax
    if rates:
        for pop in POPS:
            per_group = [[r["rates_Hz"][pop] for r in rws] for _, rws in groups]
            _plot_mean_spread(ax, xs, per_group, POP_COLOURS[pop], pop)
        ax.set_ylabel("rate (Hz per cell)")
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left")
    per_group, silent = _masked_per_group(groups, twin_of_row)
    _plot_mean_spread(companion, xs, per_group, "#7f7f7f", twin_label)
    _floor_markers(companion, silent)
    companion.set_ylabel(f"{twin_label} ({twin_unit})" if twin_unit else twin_label)
    companion.legend(loc="upper right" if rates else "upper left")
    ax.set_xlabel(SWEEP_LABELS.get(swept_key, swept_key))
    ax.set_title(title)
    _footnote(fig, source, _n_seeds(rows),
              extra="open markers at the floor: no oscillation "
                    "(band power NaN or E rate 0)" if silent else "")
    return fig


# ------------------------------------------- swept-input panels (Figs 2/5/6)

def fig2a_top(rows, swept="rate_hz", source=""):
    """Fig 2A top: f_pk and the PV-to-SOM delay vs the feedforward drive.

    Frequency (Hz, left axis) and delay (ms, right axis) get separate y axes.
    Drives without oscillation (band power NaN or E rate 0) are masked out of
    the f_pk curve and drawn as open markers at the axis floor; the footnote
    says so. Verdicts ``RESULT_FIGURES.md`` lines 48-49: trend pass, range
    14-63 Hz.
    """
    groups = _group_by(rows, swept)
    xs = [float(v) for v, _ in groups]
    fig, ax_f = plt.subplots(figsize=(7.0, 4.0))
    ax_d = ax_f.twinx()
    delay_groups = [[_row_delay_ms(r) for r in rws if _row_delay_ms(r) is not None]
                    for _, rws in groups]
    _plot_mean_spread(ax_d, xs, delay_groups, "#ff7f0e", "PV-to-SOM delay")
    ax_d.set_ylabel("PV-to-SOM delay (ms)")
    ax_d.legend(loc="upper right")
    fpk_groups, silent = _masked_per_group(groups, lambda r: r["f_pk_hz"])
    _plot_mean_spread(ax_f, xs, fpk_groups, "#7f7f7f", "f_pk")
    _floor_markers(ax_f, silent)
    ax_f.set_ylabel("f_pk (Hz)")
    ax_f.set_xlabel(SWEEP_LABELS.get(swept, swept))
    ax_f.set_title("Fig 2A (top)")
    ax_f.legend(loc="upper left")
    _footnote(fig, source, _n_seeds(rows),
              extra="open markers at the floor: no oscillation "
                    "(band power NaN or E rate 0)")
    return fig


def _row_delay_ms(row):
    delay = (row.get("phase_delay") or {}).get("delay_ms")
    return None if delay is None else float(delay)


def fig2b_top(rows, swept="tdrate_hz", source=""):
    """Fig 2B top: population rates and f_pk vs the top-down drive to SOM.

    The paper also plots the amplitude (peak |X|), which the markdown cell
    cites; verdict ``RESULT_FIGURES.md`` line 50: pass.
    """
    return _rates_and_twin(rows, swept, "Fig 2B (top)",
                           "f_pk", lambda r: r["f_pk_hz"], twin_unit="Hz",
                           source=source)


def fig5a(rows, swept="value_nS", source=""):
    """Fig 5A: population rates and 25-45 Hz band power vs PV-to-SOM coupling.

    The paper's colour map shows the amplitude; the packet plots the rates and
    the paper metric. Verdict ``RESULT_FIGURES.md`` line 51: partial.
    """
    return _rates_and_twin(rows, swept, "Fig 5A",
                           "band power 25-45 Hz", lambda r: r["band_power_25_45"],
                           source=source)


def fig5b(rows, swept="value_nS", source=""):
    """Fig 5B: population rates and 25-45 Hz band power vs SOM-to-SOM coupling.

    Verdict ``RESULT_FIGURES.md`` line 52: partial.
    """
    return _rates_and_twin(rows, swept, "Fig 5B",
                           "band power 25-45 Hz", lambda r: r["band_power_25_45"],
                           source=source)


def fig6a_bottom(rows, swept="ratio", source=""):
    """Fig 6A bottom: peak |X| vs the SOM/PV density ratio.

    The paper plots frequency (about 40 to 15 Hz) with the amplitude rising;
    verdict ``RESULT_FIGURES.md`` line 56: pass.
    """
    return _rates_and_twin(rows, swept, "Fig 6A (bottom)",
                           "peak |X|", _peak_mag, rates=False, source=source)


# ------------------------------------------------- colour-map panels (Figs 3/4)

def _map_log_mag(groups):
    """Seed-mean |X| resampled to the shared 0-100 Hz grid, as log10.

    The receipts store the spectrum over their own frequency grid (today up
    to 60 Hz); grid points beyond the stored band stay blank (NaN), so
    nothing is invented. Rows stay in swept-value order.
    """
    freqs, mag_rows = _mean_over_psd(groups)
    grid = np.linspace(MAP_GRID_HZ[0], MAP_GRID_HZ[1], MAP_GRID_HZ[2])
    log_mag = np.full((mag_rows.shape[0], grid.size), np.nan)
    inside = (grid >= freqs[0]) & (grid <= freqs[-1])
    df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0
    width = max(1, int(round(SMOOTH_HZ / df)))
    kernel = np.ones(width) / width
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, mag in enumerate(mag_rows):
            smooth = np.convolve(mag, kernel, mode="same")
            log_mag[i, inside] = np.log10(np.interp(grid[inside], freqs, smooth))
    return grid, log_mag


def _colour_map_figure(rows, swept_key, title, source=""):
    """One paper colour map plus its band-power and f_pk side panels.

    The map draws one equal-height row per swept fraction (``pcolormesh`` on
    the row index, ``aspect="auto"``), y tick labels = the fractions as
    percent, x = 0-100 Hz, colour = log10 |X| (seed mean); the QuadMesh is
    registered so :func:`share_map_colours` can give all four maps one
    common colour scale.
    """
    groups = _group_by(rows, swept_key)
    fracs = [float(v) for v, _ in groups]
    grid, log_mag = _map_log_mag(groups)
    fig, (ax_map, ax_band, ax_fpk) = plt.subplots(
        1, 3, figsize=(11.5, 3.4),
        gridspec_kw={"width_ratios": [2.0, 1.0, 1.0], "wspace": 0.45})
    mesh = ax_map.pcolormesh(grid, np.arange(len(fracs)),
                             np.ma.masked_invalid(log_mag),
                             shading="nearest", cmap="viridis")
    ax_map.set_aspect("auto")
    ax_map.set_xlim(MAP_GRID_HZ[0], MAP_GRID_HZ[1])
    ax_map.set_yticks(np.arange(len(fracs)))
    ax_map.set_yticklabels([f"{f * 100:g}%" for f in fracs])
    ax_map.set_xlabel("frequency (Hz)")
    ax_map.set_ylabel(SWEEP_LABELS.get(swept_key, swept_key))
    ax_map.set_title(title)
    fig.colorbar(mesh, ax=ax_map, label="log10 |X| (seed mean, 1 Hz boxcar)")
    _MAP_IMAGE_REGISTRY.append(mesh)
    band = [[r["band_power_25_45"] for r in rws] for _, rws in groups]
    _plot_mean_spread(ax_band, fracs, band, "#7f7f7f", "band power")
    ax_band.set_ylabel("band power 25-45 Hz")
    ax_band.set_xlabel(SWEEP_LABELS.get(swept_key, swept_key))
    fpk = [[r["f_pk_hz"] for r in rws] for _, rws in groups]
    _plot_mean_spread(ax_fpk, fracs, fpk, "#7f7f7f", "f_pk")
    ax_fpk.set_ylabel("f_pk (Hz)")
    ax_fpk.set_xlabel(SWEEP_LABELS.get(swept_key, swept_key))
    _footnote(fig, source, _n_seeds(rows))
    return fig


def share_map_colours():
    """Give every registered colour map one common log10 |X| colour scale.

    Called once after the four maps (Figs 3A, 3B, 4A, 4B) are drawn, so the
    amplitudes compare across the maps; a no-op when nothing is registered.
    """
    meshes = list(_MAP_IMAGE_REGISTRY)
    if not meshes:
        return
    values = []
    for mesh in meshes:
        data = np.ma.filled(np.ma.asanyarray(mesh.get_array(), dtype=float), np.nan)
        values.append(data[np.isfinite(data)].ravel())
    pooled = np.concatenate(values) if values else np.array([])
    lo, hi = np.inf, -np.inf
    if pooled.size:
        # Percentile limits: a single near-zero bin (log10 -> -12) must not set the scale.
        lo, hi = (float(v) for v in np.percentile(pooled, CLIM_PERCENTILES))
    if np.isfinite(lo) and np.isfinite(hi):
        if lo == hi:  # degenerate single-value data: keep the map readable
            lo, hi = lo - 1.0, hi + 1.0
        for mesh in meshes:
            mesh.set_clim(lo, hi)
    _MAP_IMAGE_REGISTRY.clear()


def fig3a(rows, swept="fraction", source=""):
    """Fig 3A: SOM silencing colour map with band power and f_pk beside it.

    Verdicts ``RESULT_FIGURES.md`` lines 27-29 and 69.
    """
    return _colour_map_figure(rows, swept, "Fig 3A", source=source)


def fig3b(rows, swept="fraction", source=""):
    """Fig 3B: PV silencing colour map with band power and f_pk beside it.

    Verdicts ``RESULT_FIGURES.md`` lines 30-32 and 70-72.
    """
    return _colour_map_figure(rows, swept, "Fig 3B", source=source)


def fig4a(rows, swept="fraction", source=""):
    """Fig 4A: SOM replacement colour map with band power and f_pk beside it.

    Verdicts ``RESULT_FIGURES.md`` lines 33 and 73.
    """
    return _colour_map_figure(rows, swept, "Fig 4A", source=source)


def fig4b(rows, swept="fraction", source=""):
    """Fig 4B: PV replacement colour map with band power and f_pk beside it.

    Verdicts ``RESULT_FIGURES.md`` lines 34-36 and 74.
    """
    return _colour_map_figure(rows, swept, "Fig 4B", source=source)


def fig3c(rows_som, rows_pv, swept="fraction", source=""):
    """Fig 3C top/bottom: max E rate vs fraction (top SOM, bottom PV).

    Both panels come from the same runs as Figures 3A and 3B, arranged as the
    paper arranges them. Verdicts ``RESULT_FIGURES.md`` lines 27-32.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.0, 6.0), sharex=True)
    for ax, rows, label in ((ax_top, rows_som, "SOM silenced"),
                            (ax_bot, rows_pv, "PV silenced")):
        groups = _group_by(rows, swept)
        xs = [float(v) for v, _ in groups]
        per_group = [[r["max_e_rate_hz"] for r in rws] for _, rws in groups]
        _plot_mean_spread(ax, xs, per_group, POP_COLOURS["E"], "max E rate")
        ax.set_ylabel(f"max E rate (Hz)\n{label}")
    ax_bot.set_xlabel(SWEEP_LABELS.get(swept, swept))
    fig.suptitle("Fig 3C")
    _footnote(fig, source, _n_seeds(rows_som))
    return fig


# --------------------------------------------------------- locking (Fig 1C)

def fig1c(rows, source=""):
    """Fig 1C: spike-LFP phase locking (PPC) and preferred phase per type.

    One coloured marker per seed per population (E red, PV blue, SOM green)
    with the seed mean as a short bar. Data: ``c1/fig1c.json`` rows
    ``ppc_by_type`` and ``preferred_phase_by_type``. Verdict
    ``RESULT_FIGURES.md`` line 54: partial (PV highest in every seed, the
    full order in 1 of 3 seeds).
    """
    fig, (ax_ppc, ax_phase) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    ppc_of = lambda r, pop: r["ppc_by_type"]["by_type"][pop]  # noqa: E731
    phase_of = lambda r, pop: r["preferred_phase_by_type"][pop]  # noqa: E731
    xs = np.arange(len(POPS), dtype=float)
    for ax, ylabel, get in ((ax_ppc, "PPC (ref. 36)", ppc_of),
                            (ax_phase, "preferred phase (rad)", phase_of)):
        for i, pop in enumerate(POPS):
            vals = np.array([get(r, pop) for r in rows], dtype=float)
            ax.plot(np.full(vals.size, xs[i]), vals, "o", markersize=5,
                    color=POP_COLOURS[pop],
                    label=pop if ax is ax_ppc else None)
            ax.plot([xs[i]], [float(np.mean(vals))], "_", markersize=14,
                    color=POP_COLOURS[pop])
        ax.set_xticks(xs, POPS)
        ax.set_ylabel(ylabel)
    ax_ppc.set_title("Fig 1C")
    ax_ppc.legend(loc="upper right")
    _footnote(fig, source, _n_seeds(rows))
    return fig


# ------------------------------------------------------- S1 (connection prob)

def fig_s1(rows, swept="p", source=""):
    """Fig S1: f_pk (spectrum peak) and population rates vs connection
    probability p. f_pk is masked where there is no oscillation (band power
    NaN or E rate 0), drawn as open markers at the axis floor.
    Verdicts ``RESULT_FIGURES.md`` lines 107-110."""
    groups = _group_by(rows, swept)
    xs = [float(v) for v, _ in groups]
    fig, ax_fpk = plt.subplots(figsize=(7.0, 4.0))
    ax_rate = ax_fpk.twinx()
    for pop in POPS:
        rate_groups = [[r["rates_Hz"][pop] for r in rws] for _, rws in groups]
        _plot_mean_spread(ax_rate, xs, rate_groups, POP_COLOURS[pop], pop)
    ax_rate.set_ylabel("rate (Hz per cell)")
    ax_rate.legend(loc="upper left")
    fpk_groups, silent = _masked_per_group(groups, lambda r: r["f_pk_hz"])
    _plot_mean_spread(ax_fpk, xs, fpk_groups, "#7f7f7f", "f_pk")
    _floor_markers(ax_fpk, silent)
    ax_fpk.set_ylabel("f_pk (Hz)")
    ax_fpk.set_xlabel(SWEEP_LABELS.get(swept, swept))
    ax_fpk.set_title("Fig S1")
    ax_fpk.legend(loc="upper right")
    _footnote(fig, source, _n_seeds(rows),
              extra="open markers at the floor: no oscillation "
                    "(band power NaN or E rate 0)" if silent else "")
    return fig


# ------------------------------------------------------------ raster panels

def _subsample_indices(slice_start, slice_stop, cap):
    """Evenly spaced neuron indices within [start, stop), at most ``cap``."""
    n = slice_stop - slice_start
    if n <= cap:
        return np.arange(slice_start, slice_stop)
    pick = np.linspace(0, n - 1, cap).round().astype(int)
    return slice_start + np.unique(pick)


def _raster_figure(trace, title, source=""):
    """Raster + population rate + (when present) the LFP proxy on one time axis.

    ``trace`` is a loaded :func:`load_trace` dict or a ``.npz`` path. Neurons
    sit on the y axis grouped E/PV/SOM, subsampled to at most 200 E, 50 PV,
    50 SOM (the subsample is stated in the footnote); time is in ms. Under the
    raster: the 1 ms population-rate bins and, for Cell Reports files, the
    LFP proxy over the same window.
    """
    trace = _as_trace(trace)
    raster = np.asarray(trace["raster"], dtype=bool)
    slices = np.asarray(trace["type_slices"], dtype=int)
    t0_s = float(trace["t0_s"])
    time_ms = (t0_s + np.arange(raster.shape[0]) * TRACE_DT_S) * 1000.0
    n_axes = 3 if trace.get("lfp") is not None else 2
    fig, axes = plt.subplots(n_axes, 1, sharex=True, figsize=(8.0, 2.1 * n_axes),
                             gridspec_kw={"height_ratios": [3, 1] + ([1] if n_axes == 3 else [])})
    ax_r, ax_rate = axes[0], axes[1]
    shown = []
    for pop, (start, stop) in zip(POPS, slices):
        idx = _subsample_indices(int(start), int(stop), RASTER_SUBSAMPLE[pop])
        shown.append(len(idx))
        ts, ys = [], []
        for row, neuron in enumerate(idx):
            steps = np.flatnonzero(raster[:, neuron])
            if steps.size:
                ts.append(time_ms[steps])
                ys.append(np.full(steps.size, float(row)))
        ax_r.plot(np.concatenate(ts) if ts else [], np.concatenate(ys) if ys else [],
                  ls="none", marker="|", markersize=3, color=POP_COLOURS[pop], label=pop)
    ax_r.set_yticks([])
    ax_r.set_ylabel("neurons (E/PV/SOM)")
    ax_r.invert_yaxis()
    n_e, n_pv, n_som = shown
    ax_r.set_title(title)
    ax_r.legend(loc="upper right")
    rate_hz = np.asarray(trace["rate_hz"], dtype=float)
    bin_ms = (t0_s * 1000.0) + (np.arange(rate_hz.shape[0]) + 0.5) * 1.0  # 1 ms bins
    for i, pop in enumerate(POPS):
        ax_rate.plot(bin_ms, rate_hz[:, i], color=POP_COLOURS[pop], label=pop)
    ax_rate.set_ylabel("rate (Hz)")
    if n_axes == 3:
        axes[2].plot(time_ms, trace["lfp"], color="#7f7f7f", linewidth=0.6)
        axes[2].set_ylabel("LFP proxy")
        axes[2].set_xlabel("time (ms)")
    else:
        ax_rate.set_xlabel("time (ms)")
    _footnote(fig, source,
              extra=f"subsample shown: {n_e} E / {n_pv} PV / {n_som} SOM")
    return fig


def trace_from_spikes(spikes, sizes, window_s=0.5):
    """The shared trace dict from a jaxfne ``Signals.spikes`` array ``[steps, N]`` (E | PV | SOM order,
    step ``TRACE_DT_S``): the last ``window_s`` of the raster, the type slices and the 1 ms
    population rates (Hz), with no LFP. ``sizes`` is ``(n_E, n_PV, n_SOM)``."""
    raster = np.asarray(spikes, dtype=bool)
    n_e, n_pv, n_som = (int(s) for s in sizes)
    if raster.ndim != 2 or raster.shape[1] != n_e + n_pv + n_som:
        raise ValueError(f"spikes must be [steps, {n_e + n_pv + n_som}], got {raster.shape}")
    keep = int(round(window_s / TRACE_DT_S))
    if keep > raster.shape[0]:
        raise ValueError(f"window_s {window_s} is longer than the run ({raster.shape[0] * TRACE_DT_S} s)")
    raster = raster[-keep:]
    slices = np.array([[0, n_e], [n_e, n_e + n_pv], [n_e + n_pv, n_e + n_pv + n_som]])
    per_ms = int(round(1e-3 / TRACE_DT_S))
    n_bins = keep // per_ms
    rate = np.stack([raster[:n_bins * per_ms, a:b].reshape(n_bins, per_ms, b - a).sum(axis=(1, 2))
                     / ((b - a) * 1e-3) for a, b in slices], axis=1)
    t0_s = (np.asarray(spikes).shape[0] - keep) * TRACE_DT_S
    return {"raster": raster, "type_slices": slices, "rate_hz": rate, "t0_s": t0_s}


def core_run_figure(spikes, sizes, title, window_s=0.5, source=""):
    """Raster + 1 ms population rates of a jaxfne core run (the last ``window_s``)."""
    return _raster_figure(trace_from_spikes(spikes, sizes, window_s), title, source=source)


def fig1b_raster(trace, source=""):
    """Fig 1B: baseline example raster + rate + LFP."""
    return _raster_figure(trace, "Fig 1B", source=source)


def fig1c_raster(trace, source=""):
    """Fig 1C raster part: baseline raster under the locking panel."""
    return _raster_figure(trace, "Fig 1C (raster)", source=source)


def fig3_raster(trace, source=""):
    """Fig 3 example raster: one silenced-PV or silenced-SOM trace."""
    return _raster_figure(trace, "Fig 3", source=source)


def fig4_raster(trace, source=""):
    """Fig 4 example raster: one replaced-PV or replaced-SOM trace."""
    return _raster_figure(trace, "Fig 4", source=source)


def fig5c(trace, source=""):
    """Fig 5C raster: the swap condition (drawn from whatever trace the
    notebook passes; the paper panel shows the swapped-parameter cells)."""
    return _raster_figure(trace, "Fig 5C", source=source)