"""PLoS Comput Biol 2026 CAMINOS figure suite (Lane A): one function per paper panel.

Panels and their data (functions take paths; the notebook passes the warehouse
root, nothing here hard-codes it):

- ``fig_plos_1d``           mean-field AdEx rates, ``caminos_meanfield.simulate_meanfield`` (CPU, 4 s).
- ``fig_plos_6a/6b/6c``     linear mean-field model, ``caminos_meanfield`` (CPU, closed form).
- ``fig_plos_1e``           mean rate vs connection probability p, counts
                            ``scored/v7_20261012`` (manifest runs or ``counts_seed*_p*.npy``);
                            example 1 s steady-state trace (first 2 s dropped), all populations.
- ``fig_plos_2a/2b/3a/3b/4a`` sweep panels from ``figP_out/{p2a,p2b,p3a,p3b,p4a}/*.json``
                            (run-JSON rows; f_pk, band-power ratio, max E rate, spectra 5-100 Hz).
- ``fig_plos_2c..2f, 3c..3f`` raster + rate panels from ``plosr_7f7a660e/seed<s>/<condition>.npz``,
                            mean-field trace overlaid where ``meanfield_<condition>.npz`` exists.

Trace files: ``raster`` bool (steps, neurons), ``type_slices`` int (3, 2), ``rate_hz`` float
(bins, 3) per 1 ms bin, scalars ``f_pk_hz``, ``band_power_25_45``, ``max_e_rate_hz``; PLoS
files carry no ``lfp``/``t0_s`` (then t0 = T_s - 0.5, T_s from the folder's ``manifest.json``,
else 11.5), Cell Reports files add both; both are accepted.

Colours fixed per population: E #d62728, PV #1f77b4, SOM #2ca02c.

Titles carry the paper panel label only (``Fig 2C: PV silenced 40%``); the data
file, seed and raster subsample go in a small grey footnote under the figure.
Sweep panels show seed mean +- SD and say so in the footnote.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def _repo_root(start: Path) -> Path:
    """Nearest ancestor that holds pyproject.toml (P-025: the checkout's modules, never site-packages)."""
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError(f"no pyproject.toml above {start}")


_HERE = Path(__file__).resolve().parent          # .../caminos/suite
_CAMINOS = _HERE.parent                          # .../caminos: run/score scripts live here
_REPO_ROOT = _repo_root(_HERE)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_CAMINOS) not in sys.path:
    sys.path.insert(0, str(_CAMINOS))

POP_NAMES = ("E", "PV", "SOM")
COLORS = {"E": "#d62728", "PV": "#1f77b4", "SOM": "#2ca02c"}
RASTER_DT_S = 1e-4          # raster step of the stored traces
RASTER_WINDOW_S = 0.5       # stored window length (run_plos_rasters raster_window)
BIN_S = 1e-3                # 1 ms rate bins
DISCARD_S = 2.0             # adaptation settle, PREREG_P3 kept window
DEFAULT_T_S = 11.5          # T_s when no manifest.json states it
RASTER_MAX_CELLS = {"E": 200, "PV": 50, "SOM": 50}
# Swept-value key and paper title per run-JSON family; score_plos_figs.py PANELS fixes them.
SWEEP_REFS = {"2A": ("p2a/silence_pv.json", "fraction", 0.0),
              "2B": ("p2b/silence_som.json", "fraction", 0.0),
              "3A": ("p3a/replace_pv.json", "fraction", 0.0),
              "3B": ("p3b/replace_som.json", "fraction", 0.0),
              "4A": ("p4a/fig6a.json", "ratio", 0.5)}
SWEEP_TITLES = {"2A": "PV silencing", "2B": "SOM silencing", "3A": "PV randomization",
                "3B": "SOM randomization", "4A": "SOM/PV density ratio"}
# Conditions with a mean-field trace (run_plos_rasters.MF_CONDITIONS).
MF_CONDITIONS = ("baseline", "pv_silenced_0.4", "pv_silenced_0.9",
                 "som_silenced_0.4", "som_silenced_0.9")
REGIME_COLORS = {"pure oscillation": "k", "damped": "#1f77b4",
                 "growing": "#d62728", "positive real": "#d62728", "real": "#7f7f7f"}


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_trace(path, folder=None):
    """Load a stored raster trace file; returns a dict (see module docstring)."""
    path, folder = Path(path), Path(folder) if folder is not None else None
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        out = {
            "raster": np.asarray(z["raster"]).astype(bool),
            "type_slices": np.asarray(z["type_slices"]).astype(int),
            "rate_hz": np.asarray(z["rate_hz"]).astype(float),
            "f_pk_hz": float(np.asarray(z["f_pk_hz"])),
            "band_power_25_45": float(np.asarray(z["band_power_25_45"])),
            "max_e_rate_hz": float(np.asarray(z["max_e_rate_hz"])),
            "lfp": np.asarray(z["lfp"]).astype(float) if "lfp" in keys else None,
        }
        if "t0_s" in keys:
            out["t0_s"] = float(np.asarray(z["t0_s"]))
        else:  # PLoS files: window start from the folder's manifest, else 11.5
            t_s = DEFAULT_T_S
            if folder is not None and (folder / "manifest.json").is_file():
                t_s = float(json.loads((folder / "manifest.json").read_text()).get("T_s", t_s))
            out["t0_s"] = t_s - RASTER_WINDOW_S
    return out


def _counts_entries(counts_dir, seeds):
    """Manifest runs of ``counts_dir`` for the given seeds (caminos_analysis reads the same)."""
    manifest = json.loads((Path(counts_dir) / "manifest.json").read_text())
    runs = [e for e in manifest.get("runs", [])
            if e.get("seed") in seeds and e.get("p") is not None]
    if runs:
        return runs
    return _counts_entries_glob(counts_dir, seeds)


def _counts_entries_glob(counts_dir, seeds):
    """Fallback: counts_seed<s>_p<p>*.npy with sizes from a folder JSON."""
    sizes = None
    for jf in sorted(Path(counts_dir).glob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except ValueError:
            continue
        if isinstance(d, dict) and "sizes" in d:
            sizes = d["sizes"]
            break
    out = []
    for f in sorted(Path(counts_dir).glob("counts_seed*_p*.npy")):
        m = re.match(r"counts_seed(\d+)_p([0-9.]+)", f.name)
        if m and int(m.group(1)) in seeds:
            out.append({"file": f.name, "seed": int(m.group(1)), "p": float(m.group(2)),
                        "dt": RASTER_DT_S, "sizes": sizes})
    return out


def _pop_sizes(sizes):
    """sizes as a dict entry or a [E, PV, SOM] list -> (n_E, n_PV, n_SOM)."""
    if isinstance(sizes, dict):
        return tuple(float(sizes[k]) for k in POP_NAMES)
    return tuple(float(v) for v in list(sizes)[:3])


def mean_rates(counts, dt_s, sizes, discard_s=DISCARD_S):
    """Mean rate per population over the kept window (first ``discard_s`` discarded)."""
    counts = np.asarray(counts, dtype=float)
    keep = counts[int(round(discard_s / dt_s)):]
    return [float(keep[:, i].mean()) / dt_s / n for i, n in enumerate(_pop_sizes(sizes))]


def smooth_rate(count_col, dt_s, n_cells, sigma_ms=1.0):
    """1 ms-sigma Gaussian-smoothed population rate per cell (Hz), as caminos_analysis."""
    counts = np.asarray(count_col, dtype=float)
    sigma = sigma_ms * 1e-3 / dt_s
    half = max(1, int(round(3.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    rate = np.convolve(counts, kernel, mode="same")
    return rate / (float(n_cells) * dt_s)


# ---------------------------------------------------------------------------
# raster panels (Figs 2C-F, 3C-F)
# ---------------------------------------------------------------------------
def _panel_map():
    """condition -> (panel label, title) from score_plos_rasters FIG2/FIG3_PANELS."""
    import score_plos_rasters as spr
    return {cond: (panel, title)
            for panel, cond, title in list(spr.FIG2_PANELS) + list(spr.FIG3_PANELS)}


def _bin_meanfield(t, nu, bin_s=BIN_S):
    """Average mean-field samples into ``bin_s`` bins (score_plos_rasters.bin_meanfield)."""
    t = np.asarray(t, dtype=float).ravel()
    nu = np.asarray(nu, dtype=float)
    t0 = float(t.min())
    idx = np.clip(np.floor((t - t0) / bin_s).astype(int), 0, None)
    n_bins = max(1, int(round((float(t.max()) - t0) / bin_s)))
    idx = np.clip(idx, 0, n_bins - 1)
    sums = np.zeros((n_bins, nu.shape[1]))
    counts = np.zeros(n_bins)
    tsum = np.zeros(n_bins)
    np.add.at(sums, idx, nu)
    np.add.at(counts, idx, 1)
    np.add.at(tsum, idx, t)
    safe = np.where(counts > 0, counts, 1.0)
    centers = np.where(counts > 0, tsum / safe, t0 + (np.arange(n_bins) + 0.5) * bin_s)
    return centers, sums / safe[:, None]


def _raster_axis(ax, raster, slices, t0_s, title, subsample=RASTER_MAX_CELLS):
    """Raster with neurons on y grouped E/PV/SOM, subsampled per population; time in ms.

    Returns the number of neurons shown per population (for the figure footnote).
    Markers are sized to stay visible at print size (>= 2 px stroke).
    """
    n_steps = raster.shape[0]
    times = (t0_s + (np.arange(n_steps) + 0.5) * RASTER_DT_S) * 1e3  # ms
    shown = []
    row = 0
    for i, name in enumerate(POP_NAMES):
        s0, s1 = int(slices[i][0]), int(slices[i][1])
        stride = max(1, int(math.ceil((s1 - s0) / float(subsample[name]))))
        shown.append(len(range(s0, s1, stride)))
        for c in range(s0, s1, stride):
            hits = np.flatnonzero(raster[:, c])
            if hits.size:
                ax.scatter(times[hits], np.full(hits.shape, row), s=10,
                           c=COLORS[name], marker="|", linewidths=1.5, label=name)
            row += 1
    ax.set_xlim(t0_s * 1e3, (t0_s + RASTER_WINDOW_S) * 1e3)
    ax.set_ylim(-1, max(row, 1))
    ax.set_ylabel("cell")
    ax.set_title(title, fontsize=10)
    return shown


def _rate_axis(ax, centers_s, rate, lfp=None, mf=None):
    """Population rate traces (1 ms bins) and, when present, LFP and mean-field overlay.

    The y-axis is clipped at the 99.5th percentile of the drawn rates; when the
    maximum sits above that (one spike dominating), it is printed in the corner.
    A legend marks network (solid) vs mean field (dashed) when both are drawn.
    """
    t_ms = np.asarray(centers_s) * 1e3
    rate = np.asarray(rate, dtype=float)
    for i, name in enumerate(POP_NAMES):
        ax.plot(t_ms, rate[:, i], c=COLORS[name], lw=1.0, label=name)
    if mf is not None:
        for i, name in enumerate(POP_NAMES):
            ax.plot(np.asarray(mf[0]) * 1e3, np.asarray(mf[1])[:, i], c=COLORS[name],
                    lw=1.0, ls="--", alpha=0.7)
        ax.legend(handles=[plt.Line2D([], [], c="#333333", lw=1.0, label="network"),
                           plt.Line2D([], [], c="#333333", lw=1.0, ls="--", label="mean field")],
                  fontsize=7, loc="upper right")
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
    vals = rate if mf is None else np.vstack([rate, np.asarray(mf[1], dtype=float)])
    p995 = float(np.percentile(vals, 99.5))
    vmax = float(np.max(vals))
    ax.set_ylim(top=max(p995 * 1.05, 1e-6))
    if vmax > p995 * 1.05:
        ax.text(0.99, 0.97, "max %.0f Hz" % vmax, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="0.3")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("rate (Hz)")
    if lfp is not None:
        ax2 = ax.twinx()
        ax2.plot(t_ms[:len(lfp)], np.asarray(lfp, dtype=float), c="#7f7f7f", lw=0.8)
        ax2.set_ylabel("LFP proxy")
    return ax


def raster_figure(run_dir, condition, seed=1, panel=None):
    """One paper raster panel: mean field (when present), raster, rate (+LFP, +MF overlay).

    Rows share one time axis in ms; the population legend appears once per
    figure (on the top row); the data file, seed and subsample go in a grey
    footnote; the title carries the paper panel label only.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    entry = manifest["conditions"][str(seed)][condition]
    trace = load_trace(run_dir / entry["file"], run_dir)
    if panel is None:
        panel, title = _panel_map().get(condition, (condition.upper(), condition))
    else:
        title = _panel_map().get(condition, (condition, condition))[1]
    n_steps = trace["raster"].shape[0]
    window_s = n_steps * RASTER_DT_S
    rate = trace["rate_hz"]
    centers = trace["t0_s"] + (np.arange(rate.shape[0]) + 0.5) * (window_s / rate.shape[0])
    mf = None
    if condition in MF_CONDITIONS:
        mf_path = run_dir / ("meanfield_%s.npz" % condition)
        if mf_path.is_file():
            with np.load(mf_path, allow_pickle=False) as z:
                mf = _bin_meanfield(np.asarray(z["t_s"]), np.asarray(z["nu_hz"]))
    rows = 3 if mf is not None else 2
    fig = plt.figure(figsize=(8.0, 2.2 * rows + 0.8))
    grid = fig.add_gridspec(rows, 1, height_ratios=([1, 2, 1.4] if mf is not None else [2, 1.4]),
                            hspace=0.35, top=0.94, bottom=0.12)
    r = 0
    prev = None
    title_txt = "Fig %s: %s" % (panel, title)
    if mf is not None:
        ax = fig.add_subplot(grid[r])
        for i, name in enumerate(POP_NAMES):
            ax.plot(mf[0] * 1e3, mf[1][:, i], c=COLORS[name], lw=1.0, label=name)  # ms
        ax.set_ylabel("MF rate (Hz)")
        ax.set_title(title_txt, fontsize=10)
        ax.legend(fontsize=7, loc="upper right", ncol=3)   # the one population legend
        ax.tick_params(labelbottom=False)
        r += 1
        prev = ax
    rax = fig.add_subplot(grid[r], sharex=prev) if prev is not None else fig.add_subplot(grid[r])
    shown = _raster_axis(rax, trace["raster"], trace["type_slices"], trace["t0_s"],
                         title_txt if mf is None else "")
    if prev is None:
        rax.legend(fontsize=7, loc="upper right", ncol=3)  # the one population legend
    rax.tick_params(labelbottom=False)
    _rate_axis(fig.add_subplot(grid[r + 1], sharex=rax), centers, rate,
               lfp=trace["lfp"], mf=mf)
    slices = trace["type_slices"]
    fig.text(0.01, 0.005,
             "data: %s (seed %s; shown %d/%d/%d of %s cells, E/PV/SOM)" % (
                 run_dir / entry["file"], seed, shown[0], shown[1], shown[2],
                 "/".join(str(int(s[1] - s[0])) for s in slices)),
             fontsize=6, color="0.45", va="bottom")
    fig.align_ylabels()
    return fig


def _raster_panel_fn(panel, condition):
    def fn(run_dir, seed=1):
        return raster_figure(run_dir, condition, seed=seed, panel=panel)
    fn.__name__ = "fig_plos_%s" % panel.lower()
    fn.__doc__ = "Paper panel %s from %s seed %s traces." % (panel, condition, "seed arg")
    return fn


fig_plos_2c = _raster_panel_fn("2C", "pv_silenced_0.4")
fig_plos_2d = _raster_panel_fn("2D", "som_silenced_0.4")
fig_plos_2e = _raster_panel_fn("2E", "pv_silenced_0.9")
fig_plos_2f = _raster_panel_fn("2F", "som_silenced_0.9")
fig_plos_3c = _raster_panel_fn("3C", "pv_randomized_0.4")
fig_plos_3d = _raster_panel_fn("3D", "som_randomized_0.4")
fig_plos_3e = _raster_panel_fn("3E", "pv_randomized_0.9")
fig_plos_3f = _raster_panel_fn("3F", "som_randomized_0.9")


# ---------------------------------------------------------------------------
# sweep panels (Figs 2A, 2B, 3A, 3B, 4A bottom)
# ---------------------------------------------------------------------------
def _sweep_rows(json_path, value_key=None):
    d = json.loads(Path(json_path).read_text())
    rows = d["rows"]
    if value_key is None:
        value_key = next(k for k in ("fraction", "ratio") if k in rows[0])
    return rows, value_key


def _mean_sd(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None, None
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return float(v.mean()), sd


def _moving_average_1hz(f, mag, half_width_hz=0.5):
    """Moving average of the (seed-averaged) linear power over a ~1 Hz window."""
    f = np.asarray(f, dtype=float)
    mag = np.asarray(mag, dtype=float)
    out = np.empty(mag.shape)
    for i, fi in enumerate(f):
        w = (f >= fi - half_width_hz) & (f <= fi + half_width_hz)
        out[i] = mag[w].mean() if w.any() else mag[i]
    return out


def _sweep_figure(json_path, ref_value, title, value_key=None):
    rows, key = _sweep_rows(json_path, value_key)
    base = {r["seed"]: r for r in rows if r[key] == ref_value}
    values = sorted({r[key] for r in rows})
    stats = {"f_pk": [], "band": [], "maxe": []}
    for v in values:
        sel = [r for r in rows if r[key] == v]
        stats["f_pk"].append(_mean_sd([r["f_pk_hz"] for r in sel]))
        stats["band"].append(_mean_sd([r["band_power_25_45"] / base[r["seed"]]["band_power_25_45"]
                                       for r in sel if r["seed"] in base]))
        stats["maxe"].append(_mean_sd([r["max_e_rate_hz"] for r in sel]))
    n_seeds = len({r["seed"] for r in rows})
    fig = plt.figure(figsize=(9.5, 7.2))
    fig.suptitle(title, fontsize=11)
    grid = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, top=0.93, bottom=0.09)
    panels = [("f$_{pk}$ (Hz)", stats["f_pk"], "frequency"),
              ("band power 25-45 Hz ratio", stats["band"], "band power ratio"),
              ("max E rate (Hz)", stats["maxe"], "max E rate")]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(values), max(values)) if len(values) > 1 else None
    for k, (ylabel, st, _kind) in enumerate(panels):
        ax = fig.add_subplot(grid[k // 2, k % 2])
        mean = [s[0] for s in st]
        sd = [s[1] for s in st]
        ax.errorbar(values, mean, yerr=sd if any(sd) else None, marker="o", c="#333333",
                    capsize=3, lw=1.0)
        ax.set_xlabel("%s" % ("SOM/PV ratio" if key == "ratio" else "fraction"))
        ax.set_ylabel(ylabel)
    ax = fig.add_subplot(grid[1, 1])
    for v in values:
        sel = [r for r in rows if r[key] == v]
        psds = [np.asarray(r["psd_mag"], dtype=float) for r in sel]
        f = np.asarray(sel[0]["psd_freqs_hz"], dtype=float)
        mag = np.mean(psds, axis=0) if all(p.shape == psds[0].shape for p in psds) else psds[0]
        b = (f >= 5.0) & (f <= 100.0)
        mag = _moving_average_1hz(f[b], mag[b])   # smooth the seed average
        color = cmap(norm(v)) if norm is not None else cmap(0.5)
        ax.semilogy(f[b], np.maximum(mag, 1e-12), c=color, lw=0.8, alpha=0.8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (a.u., 1 Hz moving average)")
    if norm is not None:
        import matplotlib.cm as cm  # noqa: PLC0415
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=key)
    fig.text(0.01, 0.005, "data: %s · mean ± SD over %d seeds" % (json_path, n_seeds),
             fontsize=6, color="0.45", va="bottom")
    return fig


def _sweep_panel_fn(panel):
    rel, key, ref = SWEEP_REFS[panel]

    def fn(json_path, value_key=None):
        return _sweep_figure(json_path, ref, "Fig %s: %s" % (panel, SWEEP_TITLES[panel]),
                             value_key or key)
    fn.__name__ = "fig_plos_%s" % panel.lower()
    fn.__doc__ = "Paper panel %s from the run-JSON %s." % (panel, rel)
    return fn


fig_plos_2a = _sweep_panel_fn("2A")
fig_plos_2b = _sweep_panel_fn("2B")
fig_plos_3a = _sweep_panel_fn("3A")
fig_plos_3b = _sweep_panel_fn("3B")
fig_plos_4a = _sweep_panel_fn("4A")


# ---------------------------------------------------------------------------
# Fig 1E: rate vs connection probability p (v7 counts)
# ---------------------------------------------------------------------------
def fig_plos_1e(counts_dir, critical_p=0.04, seeds=(1, 2, 3)):
    """Mean rate per population vs p (3 seeds, mean +- sd) and an example E trace at critical_p.

    ``critical_p`` is the smallest p at which every seed oscillates
    (RESULT_P3.md:38, amendment v7); the example traces (all populations) are
    seed <first> there, 1 s of steady state after the 2 s discard window.
    """
    counts_dir = Path(counts_dir)
    runs = _counts_entries(counts_dir, tuple(seeds))
    by_p = {}
    for e in runs:
        counts = np.asarray(np.load(counts_dir / e["file"]), dtype=float)
        rates = mean_rates(counts, float(e.get("dt", RASTER_DT_S)), e.get("sizes"))
        by_p.setdefault(float(e["p"]), {})[int(e["seed"])] = rates
    ps = sorted(by_p)
    fig = plt.figure(figsize=(9.5, 3.8))
    grid = fig.add_gridspec(1, 2, wspace=0.3)
    ax = fig.add_subplot(grid[0])
    for i, name in enumerate(POP_NAMES):
        mean = np.array([_mean_sd([by_p[p][s][i] for s in by_p[p]])[0] for p in ps])
        sd = np.array([_mean_sd([by_p[p][s][i] for s in by_p[p]])[1] for p in ps])
        ax.errorbar(ps, mean, yerr=sd if sd.any() else None, marker="o", capsize=3,
                    c=COLORS[name], label=name, lw=1.0)
    ax.set_xlabel("connection probability p")
    ax.set_ylabel("mean rate (Hz/cell)")
    ax.set_title("Fig 1E: mean rate vs p", fontsize=10)
    ax.legend(fontsize=8)
    ax2 = fig.add_subplot(grid[1])
    example = min(by_p.get(float(critical_p), {}), default=None)
    if example is not None:
        e = next(r for r in runs if float(r["p"]) == float(critical_p)
                 and int(r["seed"]) == example)
        counts = np.asarray(np.load(counts_dir / e["file"]), dtype=float)
        dt = float(e.get("dt", RASTER_DT_S))
        sizes = _pop_sizes(e.get("sizes"))
        t = np.arange(counts.shape[0]) * dt
        keep = (t >= DISCARD_S) & (t < DISCARD_S + 1.0)  # drop 2 s, show 1 s steady state
        for i, name in enumerate(POP_NAMES):
            ax2.plot(t[keep], smooth_rate(counts[:, i], dt, sizes[i])[keep],
                     c=COLORS[name], lw=0.8, label=name)
        ax2.set_xlim(DISCARD_S, DISCARD_S + 1.0)
        ax2.set_title("Fig 1E: example, seed %d, p = %g" % (example, critical_p), fontsize=9)
        ax2.legend(fontsize=7)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("rate (Hz/cell)")
    return fig


# ---------------------------------------------------------------------------
# mean-field panels (Figs 1D, 6A-C): caminos_meanfield, small CPU
# ---------------------------------------------------------------------------
def _meanfield():
    import caminos_meanfield as cmf
    return cmf


def fig_plos_1d(t_s=4.0):
    """Fig 1D: nonlinear AdEx mean-field population rates (E, PV, SOM)."""
    cmf = _meanfield()
    t, nu, _ = cmf.simulate_meanfield(t_s)
    sel = t >= 0.5 * t_s
    fig, ax = _new_ax(figsize=(6.0, 3.6))
    for i, name in enumerate(POP_NAMES):
        ax.plot(t[sel] * 1e3, nu[sel, i], c=COLORS[name], lw=1.0,
                label="%s (%.2f Hz mean)" % (name, float(nu[sel, i].mean())))
    e = nu[sel, 0]
    spec = np.abs(np.fft.rfft(e - e.mean()))
    freqs = np.fft.rfftfreq(e.size, d=t[1] - t[0])
    band = (freqs >= 5.0) & (freqs <= 100.0)
    f_pk = float(freqs[band][int(np.argmax(spec[band]))]) if band.any() else float("nan")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title("Fig 1D mean field: f$_{pk}$ %.1f Hz (2nd half of a %.0f s run)" % (f_pk, t_s),
                 fontsize=10)
    ax.legend(fontsize=8)
    return fig


def fig_plos_6a():
    """Fig 6A: linear-model eigenmode, frequency and PV/E leads over SOM."""
    cmf = _meanfield()
    m = cmf.oscillatory_mode(cmf.coupling_matrix())
    f = m["freq_hz"]
    phase = {"SOM": 0.0}
    for name in ("E", "PV"):
        phase[name] = m["lead_over_som"][name]["ms"] * 2.0 * np.pi * f / 1e3
    t_ms = np.linspace(0.0, 2000.0 / f, 800)
    fig, ax = _new_ax(figsize=(6.0, 3.6))
    for name in POP_NAMES:
        ax.plot(t_ms, np.sin(2.0 * np.pi * f * t_ms * 1e-3 + phase[name]),
                c=COLORS[name], lw=1.2,
                label="%s (leads SOM %.2f ms)" % (name, m["lead_over_som"].get(name, {"ms": 0.0})["ms"]))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("relative rate (a.u.)")
    ax.set_title("Fig 6A linear model: %.2f Hz eigenmode" % f, fontsize=10)
    ax.legend(fontsize=8)
    return fig


def fig_plos_6b():
    """Fig 6B: regime (pure oscillation / damped / growing) around the red line."""
    cmf = _meanfield()
    fig, ax = _new_ax(figsize=(6.0, 3.6))
    seen = set()
    for q_se in (2.0, 4.0, 6.0):
        line = cmf.red_line(q_se)
        xs, ys, cs = [], [], []
        for off in (-0.5, 0.0, 0.5):
            label = cmf.regime(cmf.coupling_matrix(SOME=q_se, SOMPV=line + off))
            xs.append(q_se)
            ys.append(line + off)
            cs.append(REGIME_COLORS[label])
            seen.add(label)
        q_grid = np.linspace(1.0, 7.0, 50)
        if q_se == 2.0:
            ax.plot(q_grid, cmf.red_line(q_grid), c="#d62728", lw=1.2, label="red line")
        ax.scatter(xs, ys, c=cs, s=40, zorder=3)
    ax.set_xlabel("Q$_{SOM\\to E}$")
    ax.set_ylabel("Q$_{SOM\\to PV}$")
    ax.set_title("Fig 6B: regimes around the red line", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", c=REGIME_COLORS[k], label=k) for k in sorted(seen)]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], fontsize=8, loc="best")
    return fig


def fig_plos_6c():
    """Fig 6C: left, det(A) vs Q_PV->SOM with the 220/59 boundary; right, Re vs Q_SOM->SOM."""
    cmf = _meanfield()
    fig = plt.figure(figsize=(9.0, 3.6))
    grid = fig.add_gridspec(1, 2, wspace=0.3)
    ax = fig.add_subplot(grid[0])
    q = np.linspace(0.0, 5.0, 60)
    dets = [np.linalg.det(cmf.coupling_matrix(PVSOM=v)) for v in q]
    ax.plot(q, dets, c="#333333", lw=1.2)
    ax.axhline(0.0, c="#7f7f7f", lw=0.8, ls="--")
    ax.axvline(220.0 / 59.0, c="#d62728", lw=0.8, ls=":")
    ax.set_xlabel("Q$_{PV\\to SOM}$")
    ax.set_ylabel("det(A)")
    ax.set_title("Fig 6C left: damped below 220/59, positive real above", fontsize=9)
    ax = fig.add_subplot(grid[1])
    qs = (0.0, 0.5, 1.0, 2.0, 4.0)
    res = [cmf.oscillatory_mode(cmf.coupling_matrix(SOMSOM=v))["real"] for v in qs]
    ax.plot(qs, res, marker="o", c="#333333", lw=1.2)
    ax.set_xlabel("Q$_{SOM\\to SOM}$")
    ax.set_ylabel("Re (1/s)")
    ax.set_title("Fig 6C right: damping grows with Q$_{SOM\\to SOM}$", fontsize=9)
    return fig


def fig_plos_5a(json_path):
    """Fig 5A: added PV->SOM and SOM->SOM couplings (0-3 nS) on the nonlinear mean field.

    Reads a ``fig5a_sweep`` JSON ({coupling: sweep}). Top row: oscillation frequency at unstable points.
    Bottom row: E-rate fixed point (solid stable, dashed unstable) and the limit-cycle min/max.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    fig = plt.figure(figsize=(9.0, 5.6))
    grid = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)
    labels = {"PVSOM": "PV$\\to$SOM", "SOMSOM": "SOM$\\to$SOM"}
    for col, name in enumerate(("PVSOM", "SOMSOM")):
        rows = data[name]["results"]
        g = np.array([r["g_ns"] for r in rows])
        unstable = np.array([not r["stable"] for r in rows])
        ax = fig.add_subplot(grid[0, col])
        ax.plot(g[unstable], [r["freq_hz"] for r, u in zip(rows, unstable) if u], "o-", c="#333333",
                ms=3, lw=1.0)
        ax.set_xlim(g.min(), g.max())
        ax.set_ylabel("frequency (Hz)")
        ax.set_title("Fig 5A %s: oscillation frequency" % labels[name], fontsize=9)
        ax = fig.add_subplot(grid[1, col])
        fp = np.array([r["fixed_E_hz"] for r in rows])
        ax.plot(np.where(~unstable, g, np.nan), np.where(~unstable, fp, np.nan), c=COLORS["E"], lw=1.4,
                label="fixed point, stable")
        ax.plot(np.where(unstable, g, np.nan), np.where(unstable, fp, np.nan), c=COLORS["E"], lw=1.2,
                ls="--", label="fixed point, unstable")
        lo = [r["min_E_hz"] if u else np.nan for r, u in zip(rows, unstable)]
        hi = [r["max_E_hz"] if u else np.nan for r, u in zip(rows, unstable)]
        ax.plot(g, lo, c="#7f7f7f", lw=1.0, label="limit cycle min/max")
        ax.plot(g, hi, c="#7f7f7f", lw=1.0)
        ax.set_xlabel("added %s conductance (nS)" % labels[name])
        ax.set_ylabel("E rate (Hz)")
        ax.legend(fontsize=7)
    return fig


def _new_ax(figsize):
    fig = plt.figure(figsize=figsize)
    return fig, fig.add_subplot(1, 1, 1)