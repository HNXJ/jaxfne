"""Per-paper figure suites for the dashboard Figures tab, drawn from run receipts only.

Each Panel names the receipt files it reads, root-relative: ``warehouse/`` is $JAXFNE_WAREHOUSE
(default F:/warehouse), ``receipts/`` is $JAXFNE_RECEIPTS (default C:/workspace/jaxfne_receipts),
``repo/`` is this repository. Its draw function returns the caption, every number in it read from
those files. scripts/build_dashboard_figures.py --suites renders the panels and records each
source's sha256 in dashboard/figures/manifest.json. Where a receipt does not state a raster's column
order, the draw function proves the order against counts or rates the receipt does state and
refuses on a mismatch.

No jaxfne import, so the repo-root guard (P-025) is not needed.
"""

import glob
import hashlib
import io
import json
import os
from collections import namedtuple
from pathlib import Path

try:
    import numpy as np
except ImportError:  # --check runs on a bare interpreter; only drawing needs numpy
    np = None

ROOT = Path(__file__).resolve().parents[1]
ROOTS = {
    "warehouse": Path(os.environ.get("JAXFNE_WAREHOUSE", "F:/warehouse")),
    "receipts": Path(os.environ.get("JAXFNE_RECEIPTS", "C:/workspace/jaxfne_receipts")),
    "repo": ROOT,
}

# sources: root-relative paths or glob patterns.
Panel = namedtuple("Panel", "paper id title sources draw")

THEMES = {
    False: {"E": "#1565c0", "I": "#e53935", "SOM": "#00897b", "ink": "#222222", "gold": "#b8963e",
            "series": ["#1565c0", "#ff9800", "#4caf50", "#e53935", "#8e24aa", "#00acc1", "#6d4c41", "#546e7a"]},
    True: {"E": "#64b5f6", "I": "#ef9a9a", "SOM": "#4dd0e1", "ink": "#dddddd", "gold": "#cfb87c",
           "series": ["#64b5f6", "#ffb74d", "#81c784", "#ef9a9a", "#ce93d8", "#4dd0e1", "#bcaaa4", "#b0bec5"]},
}


def resolve(source):
    """Root-relative path or glob -> sorted list of paths."""
    root, _, rest = source.partition("/")
    full = (ROOTS[root] / rest).as_posix()
    if any(c in rest for c in "*?["):
        return sorted(Path(p) for p in glob.glob(full))
    return [Path(full)]


def label(path):
    """File path -> its root-relative label, as the manifest records it."""
    p = Path(path).as_posix()
    for name, root in ROOTS.items():
        r = root.as_posix().rstrip("/") + "/"
        if p.startswith(r):
            return f"{name}/{p[len(r):]}"
    raise ValueError(f"{p} lies under no suite root")


def unlabel(text):
    root, _, rest = text.partition("/")
    return ROOTS[root] / rest


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pick(files, suffix):
    hits = [f for f in files if f.as_posix().endswith(suffix)]
    if len(hits) != 1:
        raise ValueError(f"expected one source ending {suffix}, found {len(hits)}")
    return hits[0]


def binned_rate(spikes, bin_steps, dt_s):
    """Population rate (Hz) of a (steps, cells) raster in bins of bin_steps steps."""
    n_bins = spikes.shape[0] // bin_steps
    counts = spikes[: n_bins * bin_steps].reshape(n_bins, bin_steps, -1).sum(axis=(1, 2))
    return counts / (spikes.shape[1] * bin_steps * dt_s)


def raster(ax, spikes, cols, t0, dt_ms, color, y0=0, ms=1.0):
    """Dots for spikes[:, cols], rows from y0 up; t0 (ms) is the time of row 0. Returns the next free row."""
    steps, idx = np.nonzero(spikes[:, cols])
    ax.plot(t0 + steps * dt_ms, y0 + idx, linestyle="none", marker=".", ms=ms, mew=0, color=color)
    return y0 + len(cols)


def psd(x, fs, nperseg):
    from scipy.signal import welch

    return welch(np.asarray(x, float) - np.mean(x), fs=fs, nperseg=nperseg)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def seed_num(text):
    return int("".join(c for c in text if c.isdigit()))


# ---- CAMINOS ----

CAM_SCORED = "warehouse/caminos/scored/caminos_scored"
CAM_PLOSR = "warehouse/caminos/plosr_7f7a660e"
CAM_POPS = ("E", "PV", "SOM")
CAM_COLOR = {"E": "E", "PV": "I", "SOM": "SOM"}


def caminos_counts(files, seed, p):
    """Counts (steps, 3) of one scored run; columns proven E, PV, SOM against score.json spikes_kept."""
    manifest = load_json(pick(files, "caminos_scored/manifest.json"))
    score = load_json(pick(files, "caminos_scored/score.json"))
    run = next(r for r in manifest["runs"] if r["seed"] == seed and r["p"] == p)
    entry = next(r for r in score["per_run"] if r["seed"] == seed and r["p"] == p)
    counts = np.load(pick(files, run["file"]))
    kept = counts.shape[0] - int(round(10.0 / run["dt"]))  # the score keeps the last 10 s
    totals = counts[kept:].sum(axis=0).tolist()
    want = [entry["spikes_kept"][k] for k in CAM_POPS]
    if totals != want:
        raise ValueError(f"CAMINOS seed {seed} p {p}: column totals {totals} != spikes_kept {want}")
    return counts, run, entry


def cam_raster(fig, files, th):
    m = load_json(pick(files, "plosr_7f7a660e/manifest.json"))
    with np.load(pick(files, "seed1/baseline.npz")) as z:
        r, slices, rate, f_pk = z["raster"], z["type_slices"].tolist(), z["rate_hz"], float(z["f_pk_hz"])
    window_s = 0.5  # run_plos_rasters.py stores the last 0.5 s
    for i, (a, b) in enumerate(slices):
        if not np.isclose(r[:, a:b].sum() / ((b - a) * window_s), rate[:, i].mean(), rtol=1e-9):
            raise ValueError("CAMINOS raster slices do not reproduce rate_hz")
    t0 = (float(m["T_s"]) - window_s) * 1000.0
    dt_ms = window_s * 1000.0 / r.shape[0]
    ax = fig.add_subplot(111)
    y = 0
    for name, (a, b), n in zip(CAM_POPS, slices, (200, 50, 50)):
        y0 = y
        y = raster(ax, r, np.arange(a, b, (b - a) // n)[:n], t0, dt_ms, th[CAM_COLOR[name]], y0, ms=1.8)
        ax.text(t0 + window_s * 1000.0 + 4, (y0 + y) / 2, name, va="center", fontsize=8)
    ax.set_xlim(t0, t0 + window_s * 1000.0)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("cell (sample)")
    tidy(ax)
    return (f"Baseline, seed 1, last {window_s:g} s of T = {m['T_s']:g} s (PLoS raster run {m['git_sha'][:8]}); "
            f"200 of {slices[0][1] - slices[0][0]} E, 50 PV and 50 SOM cells; E rate peak {f_pk:g} Hz.")


def cam_rates(fig, files, th):
    counts, run, entry = caminos_counts(files, 1, 0.07)
    dt = run["dt"]
    bin_steps = int(round(1e-3 / dt))
    win = counts[-int(round(0.5 / dt)):]
    n_bins = win.shape[0] // bin_steps
    t = (run["T"] - 0.5) * 1000.0 + np.arange(n_bins) * bin_steps * dt * 1000.0
    ax = fig.add_subplot(111)
    for i, name in enumerate(CAM_POPS):
        y = win[: n_bins * bin_steps, i].reshape(n_bins, bin_steps).sum(axis=1) / (run["sizes"][name] * bin_steps * dt)
        ax.plot(t, y, color=th[CAM_COLOR[name]], lw=0.9, label=name)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("population rate (Hz, 1 ms bins)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    tidy(ax)
    r = entry["rates_Hz"]
    return (f"Seed 1, p = 0.07, last 0.5 s of the scored run; mean rates over the scored 10 s: "
            f"E {r['E']:.2f}, PV {r['PV']:.2f}, SOM {r['SOM']:.2f} Hz.")


def cam_psd(fig, files, th):
    score = load_json(pick(files, "caminos_scored/score.json"))
    seeds = sorted(r["seed"] for r in score["per_run"] if r["p"] == 0.07)
    ax = fig.add_subplot(111)
    for k, seed in enumerate(seeds):
        counts, run, entry = caminos_counts(files, seed, 0.07)
        dt = run["dt"]
        bin_steps = int(round(1e-3 / dt))
        x = counts[-int(round(10.0 / dt)):, 0].reshape(-1, bin_steps).sum(axis=1) / (run["sizes"]["E"] * bin_steps * dt)
        f, pxx = psd(x, 1.0 / (bin_steps * dt), 1024)
        keep = f <= 100
        ax.semilogy(f[keep], pxx[keep], color=th["series"][k % 8], lw=0.9,
                    label=f"seed {seed}: peak {entry['peak_freq_E_Hz']:g} Hz")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("E rate PSD (Hz²/Hz)")
    ax.legend(frameon=False, fontsize=7)
    tidy(ax)
    return (f"Welch PSD of the E population rate (1 ms bins, 1.024 s segments) over the scored 10 s, p = 0.07, "
            f"seeds {seeds[0]}-{seeds[-1]}; R1 mean peak {score['r1']['mean_peak_freq_Hz']:g} Hz.")


def cam_conditions(fig, files, th):
    m = load_json(pick(files, "plosr_7f7a660e/manifest.json"))
    seeds = [str(s) for s in m["seeds"]]
    conds = list(m["conditions"][seeds[0]])
    means = np.zeros((len(seeds), len(conds), 3))
    for i, s in enumerate(seeds):
        for j, c in enumerate(conds):
            with np.load(pick(files, m["conditions"][s][c]["file"])) as z:
                means[i, j] = z["rate_hz"].mean(axis=0)
    ax = fig.add_subplot(111)
    x = np.arange(len(conds))
    for k, name in enumerate(CAM_POPS):
        ax.bar(x + (k - 1) * 0.27, means[:, :, k].mean(axis=0), 0.27, yerr=means[:, :, k].std(axis=0),
               color=th[CAM_COLOR[name]], label=name, error_kw={"lw": 0.7, "ecolor": th["ink"]})
    ax.set_xticks(x, [c.replace("_", " ") for c in conds], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("mean rate, last 0.5 s (Hz)")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)
    return (f"PLoS Figs. 2-3 conditions, mean ± sd over seeds {', '.join(seeds)}; T = {m['T_s']:g} s, "
            f"rate over the stored last 0.5 s.")


# ---- CAMINOS Cell Reports figures ----

CAM_S1 = "warehouse/caminos/s1_95d2161b/s1.json"


def s1_rows(files):
    d = load_json(files[0])
    return d, sorted(d["rows"], key=lambda r: (r["p"], r["seed"]))


def s1_sweep(fig, files, th):
    d, rows = s1_rows(files)
    axes = fig.subplots(1, 3)
    for ax, get, ylabel in ((axes[0], lambda r: r["f_pk_hz"], "E peak frequency (Hz)"),
                            (axes[1], lambda r: r["max_e_rate_hz"], "max E rate (Hz)"),
                            (axes[2], lambda r: r["phase_delay"]["delay_ms"], "PV-to-SOM delay (ms)")):
        for k, seed in enumerate(d["seeds"]):
            mine = [r for r in rows if r["seed"] == seed]
            ax.plot([100 * r["p"] for r in mine], [get(r) for r in mine], marker="o", ms=3, lw=0.8,
                    color=th["series"][k % 8], label=f"seed {seed}")
        ax.set_xlabel("connection probability p (%)", fontsize=7)
        ax.set_ylabel(ylabel)
        tidy(ax)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7)
    return (f"S1 sweep ({d['param_set']} parameters, T = {d['T_s']:g} s): {len(d['values'])} values of p from "
            f"{100 * d['values'][0]:g} to {100 * d['values'][-1]:g} %, seeds {d['seeds'][0]}-{d['seeds'][-1]}, "
            f"{len(rows)} runs.")


def s1_spectra(fig, files, th):
    d, rows = s1_rows(files)
    ax = fig.add_subplot(111)
    for k, p in enumerate(d["values"]):
        mine = [r for r in rows if r["p"] == p]
        f = np.asarray(mine[0]["psd_freqs_hz"])
        mag = np.mean([r["psd_mag"] for r in mine], axis=0)
        step = int(round(1.0 / (f[1] - f[0])))  # 1 Hz bins
        n = (len(f) // step) * step
        fb, mb = f[:n].reshape(-1, step).mean(axis=1), mag[:n].reshape(-1, step).mean(axis=1)
        keep = (fb > 1) & (fb <= 120)
        ax.semilogy(fb[keep], mb[keep], color=th["series"][k] if k < 8 else th["ink"], lw=0.9, label=f"p = {100 * p:g} %")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("E spectrum (receipt psd_mag)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    tidy(ax)
    return (f"E population spectra of the S1 sweep, mean over seeds {d['seeds'][0]}-{d['seeds'][-1]}, "
            f"averaged into 1 Hz bins.")


# ---- Destexhe 2009 ----

DX = "warehouse/destexhe2009/P3v3_signed_2d17488b/scored"


def dx_runs(files):
    """(seed, tag, score, spikes path) of every scored run among files."""
    runs = []
    for f in files:
        if f.name.startswith("score_seed"):
            s = load_json(f)
            runs.append((s["seed"], s["tag"], s, pick(files, f"spikes_seed{s['seed']}_{s['tag']}.npz")))
    return sorted(runs, key=lambda r: (r[1], r[0]))


def dx_load(path, score):
    with np.load(path) as z:
        sp, dt_ms = z["spikes"], float(z["dt_ms"])
    if not np.isclose(sp.sum() / (sp.shape[1] * sp.shape[0] * dt_ms / 1000.0), score["mean_rate_hz"], rtol=1e-6):
        raise ValueError(f"{path.name} does not reproduce its mean_rate_hz")
    return sp, dt_ms


def dx_tag(tag):
    return "Fig. " + tag.split("_")[0][3:].upper()


def dx_raster(fig, files, th):
    seed, tag, s, path = dx_runs(files)[0]
    sp, dt_ms = dx_load(path, s)
    ax = fig.add_subplot(111)
    raster(ax, sp, np.arange(0, sp.shape[1], 10), 0.0, dt_ms, th["E"], ms=1.2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("cell (every 10th)")
    tidy(ax)
    return (f"Seed {seed}, {dx_tag(tag)} continuation, {sp.shape[1]} cells over {sp.shape[0] * dt_ms / 1000:g} s "
            f"(the raster stores no RS/FS split); mean rate {s['mean_rate_hz']:.2f} Hz, CV ISI {s['cv_isi']:.2f}.")


def dx_rates(fig, files, th):
    runs = dx_runs(files)
    ax = fig.add_subplot(111)
    for k, (seed, tag, s, path) in enumerate(runs):
        sp, dt_ms = dx_load(path, s)
        y = binned_rate(sp, int(round(50 / dt_ms)), dt_ms / 1000.0)
        ax.plot(np.arange(len(y)) * 0.05, y, color=th["series"][k % 8], lw=0.9,
                linestyle="--" if "6a" in tag else "-", label=f"{dx_tag(tag)} seed {seed}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("population rate (Hz, 50 ms bins)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    tidy(ax)
    alive =sum(bool(r[2]["alive_last_s"]) for r in runs)
    return f"Every scored run (P3 v3 signed); activity alive in the last second in {alive} of {len(runs)} runs."


def dx_stats(fig, files, th):
    runs = dx_runs(files)
    ax = fig.add_subplot(111)
    for tag, marker, color in (("fig6a_cont", "s", th["I"]), ("fig6b_cont", "o", th["E"])):
        rs = [r for r in runs if r[1] == tag]
        ax.scatter([r[2]["cc"] for r in rs], [r[2]["cv_isi"] for r in rs], marker=marker, color=color, label=dx_tag(tag))
        for r in rs:
            ax.annotate(str(r[0]), (r[2]["cc"], r[2]["cv_isi"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("spike-count correlation CC")
    ax.set_ylabel("CV ISI")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)
    return f"CV ISI against pairwise count correlation for each of the {len(runs)} scored runs, labelled by seed."


# ---- GSDR 2026 ----

GS = "warehouse/gsdr/P3_fit_8516a005"


def gs_fit(fig, files, th):
    ms = sorted((load_json(f) for f in files if f.name == "manifest.json"), key=lambda m: m["seed"])
    ax = fig.add_subplot(111)
    for k, m in enumerate(ms):
        tr = m["trajectory"]
        ax.plot([t["generation"] for t in tr], [t["best_loss"] for t in tr], color=th["series"][k % 8], lw=1,
                label=f"seed {m['seed']}")
    ax.set_xlabel("generation")
    ax.set_ylabel("best loss")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)
    w = ms[0]["loss_weights"]
    return (f"Fit history per seed ({ms[0]['optimizer']['name']}; population {ms[0]['population']}, "
            f"{ms[0]['generations']} generations; loss weights psd {w['psd']:g}, rate {w['rate']:g}, kappa {w['kappa']:g}).")


def gs_gstar(fig, files, th):
    gs = sorted((f for f in files if f.name == "g_star_uS_float32.npy"), key=lambda f: seed_num(f.parent.name))
    bins = np.linspace(0, 10, 41)
    ax = fig.add_subplot(111)
    n = 0
    for k, f in enumerate(gs):
        g = np.load(f)
        n = g.size
        ax.hist(g, bins=bins, histtype="step", color=th["series"][k % 8], lw=1.1, label=f"seed {seed_num(f.parent.name)}")
    ax.set_xlabel("fitted conductance g* (µS)")
    ax.set_ylabel("edges")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)
    return f"Fitted conductances, all {n} trainable edges per seed."


def gs_eval(fig, files, th):
    ev = [f for f in files if f.name.startswith("eval_seed")]
    seeds = sorted({seed_num(f.name.split("_")[1]) for f in ev})
    ax = fig.add_subplot(111)
    for k, s in enumerate(seeds):
        rows = [load_json(f) for f in ev if f.name.startswith(f"eval_seed{s}_")]
        ax.scatter([r["mean_rate_hz"] for r in rows], [r["kappa_500_1000"] for r in rows],
                   color=th["series"][k % 8], s=18, label=f"seed {s}")
    ax.set_xlabel("mean rate (Hz)")
    ax.set_ylabel("kappa, 500-1000 ms")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)
    return f"Held-out evaluation trials at g*: {len(ev)} trials over {len(seeds)} fitted seeds."


# ---- Izhikevich 2003 ----

IZ = "warehouse/izh2003/scored_v7"


def iz_fig2(fig, files, th):
    m = load_json(pick(files, "fig2/manifest.json"))
    names = sorted(m["runs"])
    axes = fig.subplots(2, (len(names) + 1) // 2, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        run = m["runs"][name]
        with np.load(pick(files, f"fig2/{run['file']}")) as z:
            v = z["v"]
        ax.plot(np.arange(len(v)) * run["dt_ms"], v, color=th["E"], lw=0.7)
        ax.set_title(f"{name} {run['type']}", fontsize=8)
        ax.tick_params(labelsize=6)
        tidy(ax)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    axes[1, 0].set_xlabel("time (ms)", fontsize=7)
    axes[0, 0].set_ylabel("v (mV)", fontsize=7)
    return f"Membrane potential of the {len(names)} Fig. 2 single-cell runs (dt {m['dt_ms']:g} ms, {m['kernel']})."


def iz_fig3(fig, files, th):
    m = load_json(pick(files, "fig3/manifest.json"))
    with np.load(pick(files, "fig3/1.npz")) as z:
        rows, cols, dt_ms, n_cells = z["spike_rows"], z["spike_cols"], float(z["dt_ms"]), int(z["n_cells"])
    t = rows * dt_ms
    sel = (t >= 1000) & (t < 2000)
    ne = int(round(0.8 * n_cells))  # Izhikevich 2003: excitatory cells first, 800 of 1000
    ax = fig.add_subplot(111)
    for mask, color in ((cols < ne, th["E"]), (cols >= ne, th["I"])):
        ax.plot(t[sel & mask], cols[sel & mask], linestyle="none", marker=".", ms=1.2, mew=0, color=color)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("cell")
    tidy(ax)
    return (f"Seed 1, {n_cells}-cell network, 1-2 s of T = {m['T_s']:g} s (dt {m['dt_ms']:g} ms); "
            f"{m['runs']['1']['n_spikes']} spikes in the run.")


def iz_psd(fig, files, th):
    m = load_json(pick(files, "fig3/manifest.json"))
    score = load_json(pick(files, "scored_v7/score.json"))
    ax = fig.add_subplot(111)
    for k, seed in enumerate(m["seeds"]):
        run = m["runs"][str(seed)]
        with np.load(pick(files, f"fig3/{run['file']}")) as z:
            rows, n_steps, n_cells = z["spike_rows"], int(z["n_steps"]), int(z["n_cells"])
        bin_steps = int(round(1.0 / run["dt_ms"]))
        n_bins = n_steps // bin_steps
        x = np.bincount(rows // bin_steps, minlength=n_bins)[:n_bins] / (n_cells * 1e-3)
        f, pxx = psd(x, 1000.0, 1024)
        keep = f <= 80
        s = score["fig3"]["seeds"][str(seed)]
        ax.semilogy(f[keep], pxx[keep], color=th["series"][k % 8], lw=0.9,
                    label=f"seed {seed}: {s['N3']['numbers']['f_pk']:g} / {s['N4']['numbers']['f_pk']:g} Hz")
    for band in (score["alpha_band"], score["gamma_band"]):
        ax.axvspan(*band, color=th["gold"], alpha=0.25, lw=0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("population rate PSD (Hz²/Hz)")
    ax.legend(frameon=False, fontsize=7, title="alpha / gamma peak", title_fontsize=7)
    tidy(ax)
    a, g = score["alpha_band"], score["gamma_band"]
    return (f"Welch PSD of the population rate (1 ms bins, 1.024 s segments); shaded: the P3 v7 alpha "
            f"({a[0]:g}-{a[1]:g} Hz) and gamma ({g[0]:g}-{g[1]:g} Hz) bands.")


# ---- Brunel 2000 ----

BR = "receipts/brunel2000_p3v4/scored"


def br_runs(files):
    return sorted(load_json(pick(files, "scored/manifest.json"))["runs"], key=lambda r: (r["point"], r["seed"]))


def br_load(files, run):
    with np.load(pick(files, run["file"])) as z:
        sp = z["spikes"]
    burn = int(round(run["burn_in_ms"] / run["dt_ms"]))
    rate = sp[burn:].sum() / (sp.shape[1] * (sp.shape[0] - burn) * run["dt_ms"] / 1000.0)
    if not np.isclose(rate, run["regime"]["rate_hz"], rtol=1e-4):
        raise ValueError(f"{run['file']}: rate after burn-in {rate} != regime rate_hz {run['regime']['rate_hz']}")
    return sp, burn


def br_rasters(fig, files, th):
    runs = [r for r in br_runs(files) if r["seed"] == 4]
    axes = fig.subplots(2, len(runs), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for j, r in enumerate(runs):
        sp, _ = br_load(files, r)
        dt = r["dt_ms"]
        win = sp[-int(round(200 / dt)):]
        t0 = r["duration_ms"] - 200
        raster(axes[0, j], win, np.arange(50), t0, dt, th["E"], ms=1.8)
        y = binned_rate(win, int(round(1 / dt)), dt / 1000.0)
        axes[1, j].plot(t0 + np.arange(len(y)), y, color=th["E"], lw=0.7)
        axes[0, j].set_title(f"{r['point']}: {r['regime']['label']}", fontsize=8)
        for ax in axes[:, j]:
            ax.tick_params(labelsize=6)
            tidy(ax)
    axes[0, 0].set_ylabel("cell", fontsize=7)
    axes[1, 0].set_ylabel("rate (Hz)", fontsize=7)
    axes[1, 0].set_xlabel("time (ms)", fontsize=7)
    return (f"Fig. 8 points A-D, seed 4, last 200 ms of {runs[0]['duration_ms']:g} ms: 50 cells and the population "
            f"rate of all {runs[0]['n']} cells (1 ms bins). Titles: the receipts' regime labels.")


def br_psd(fig, files, th):
    runs = br_runs(files)
    ax = fig.add_subplot(111)
    for k, point in enumerate(sorted({r["point"] for r in runs})):
        acc, peaks = [], []
        for r in (r for r in runs if r["point"] == point):
            sp, burn = br_load(files, r)
            x = binned_rate(sp[burn:], int(round(1 / r["dt_ms"])), r["dt_ms"] / 1000.0)
            del sp
            f, pxx = psd(x, 1000.0, 1000)
            acc.append(pxx)
            peaks.append(r["regime"]["spectral_peak_hz"])
        keep = (f > 0) & (f <= 250)
        ax.semilogy(f[keep], np.mean(acc, axis=0)[keep], color=th["series"][k % 8], lw=0.9,
                    label=f"{point}: receipt peaks {min(peaks):g}-{max(peaks):g} Hz")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("population rate PSD (Hz²/Hz)")
    ax.legend(frameon=False, fontsize=7)
    tidy(ax)
    seeds = sorted({r["seed"] for r in runs})
    return (f"Welch PSD of the population rate after the {runs[0]['burn_in_ms']:g} ms burn-in, 1 ms bins, "
            f"mean over seeds {seeds[0]}-{seeds[-1]}.")


# ---- Potjans and Diesmann 2014 ----

PD = "warehouse/potjans2014/scored_P3v2_b23c9b49/scored"


def pd_runs(files):
    return sorted((load_json(f)["runs"][0] for f in files if f.name == "manifest.json"), key=lambda r: r["seed"])


def pd_raster(fig, files, th):
    run = pd_runs(files)[0]
    with np.load(pick(files, f"seed{run['seed']}/{run['file']}")) as z:
        sp = z["spikes"]
    dt_s = run["dt_ms"] / 1000.0
    # The receipt names population sizes, not the column order: prove contiguous blocks in the listed order.
    blocks, a = [], 0
    for name, p in run["pops"].items():
        b = a + int(p["n"])
        if not np.isclose(sp[:, a:b].sum() / ((b - a) * sp.shape[0] * dt_s), p["rate_hz"], rtol=1e-6):
            raise ValueError(f"Potjans block {name} [{a}, {b}) does not reproduce its rate_hz")
        blocks.append((name, a, b))
        a = b
    if a != sp.shape[1]:
        raise ValueError("Potjans population sizes do not cover the raster")
    win = sp[-int(round(0.2 / dt_s)):]
    t0 = run["duration_ms"] - 200
    ax = fig.add_subplot(111)
    y = 0
    for name, a, b in reversed(blocks):
        y0 = y
        y = raster(ax, win, np.arange(a, b, 10), t0, run["dt_ms"], th["E"] if name.endswith("E") else th["I"], y0, ms=2.6)
        ax.text(run["duration_ms"] + 2, (y0 + y) / 2, name, va="center", fontsize=7)
    ax.set_xlim(t0, run["duration_ms"])
    ax.set_yticks([])
    ax.set_xlabel("time (ms)")
    tidy(ax)
    return (f"Seed {run['seed']}, scale {run['scale']:g} ({run['n']} cells), last 200 ms of "
            f"{run['duration_ms'] / 1000:g} s; every 10th cell, layer 2/3 on top.")


def pd_bars(key, ylabel, what):
    def draw(fig, files, th):
        runs = pd_runs(files)
        pops = list(runs[0]["pops"])
        vals = np.array([[r["pops"][p][key] for p in pops] for r in runs])
        ax = fig.add_subplot(111)
        ax.bar(range(len(pops)), vals.mean(axis=0), yerr=vals.std(axis=0),
               color=[th["E"] if p.endswith("E") else th["I"] for p in pops], error_kw={"lw": 0.8, "ecolor": th["ink"]})
        for i in range(len(pops)):
            ax.plot([i] * len(runs), vals[:, i], linestyle="none", marker="o", ms=2.5, color=th["ink"])
        ax.set_xticks(range(len(pops)), pops, fontsize=8)
        ax.set_ylabel(ylabel)
        tidy(ax)
        return (f"{what} per population, mean ± sd with each of seeds {runs[0]['seed']}-{runs[-1]['seed']} "
                f"({runs[0]['duration_ms'] / 1000:g} s scored runs).")
    return draw


# ---- Vogels et al. 2011 ----

VG = "warehouse/vogels2011/scored_P3v4_b23c9b49"


def vg_receipts(files):
    return sorted((load_json(f) for f in files if f.name == "score_receipt.json"), key=lambda r: r["seed"])


def vg_history(fig, files, th):
    rs = vg_receipts(files)
    score = load_json(pick(files, "score_p3.json"))
    ax = fig.add_subplot(111)
    for k, r in enumerate(rs):
        y = r["e_rate_per_s_hz"]
        ax.semilogy(np.arange(1, len(y) + 1), y, color=th["series"][k % 8], lw=0.9, label=f"seed {r['seed']}")
    ax.axhline(rs[0]["rho0_hz"], color=th["gold"], linestyle="--", lw=1.5, label=f"ρ0 = {rs[0]['rho0_hz']:g} Hz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("E rate (Hz, 1 s bins)")
    ax.legend(frameon=False, fontsize=7)
    tidy(ax)
    return (f"E rate under inhibitory STDP (η = {rs[0]['eta']:g}, T = {rs[0]['T_s']:g} s). "
            f"R-E: e = {score['R-E']['e']:.2f} against {score['R-E']['criterion']}; "
            f"CONV {'holds' if score['CONV']['pass'] else 'does not hold'}.")


def vg_raster(fig, files, th):
    r = vg_receipts(files)[0]
    ne = int(round(0.8 * r["n"]))  # E columns first (vogels2011_api.py), proven below
    steps_s = int(round(1000.0 / r["dt_ms"]))
    axes = fig.subplots(1, 2, sharey=True)
    for ax, key, title in ((axes[0], "static_file", "static, first 1 s"), (axes[1], "plastic_file", "plastic, last 1 s")):
        with np.load(pick(files, f"seed{r['seed']}/{r[key]}")) as z:
            sp = z["spikes"]
        win = sp[-steps_s:] if key == "plastic_file" else sp[:steps_s]
        if key == "plastic_file" and not np.isclose(win[:, :ne].sum() / ne, r["e_rate_per_s_hz"][-1], rtol=1e-6):
            raise ValueError("Vogels E columns do not reproduce the last e_rate_per_s_hz")
        y = raster(ax, win, np.arange(0, ne, 40), 0.0, r["dt_ms"], th["E"], ms=2.0)
        raster(ax, win, np.arange(ne, r["n"], 40), 0.0, r["dt_ms"], th["I"], y, ms=2.0)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("time in window (ms)")
        tidy(ax)
    axes[0].set_ylabel("cell (every 40th; I on top)")
    return f"Seed {r['seed']}, {r['n']} cells, before plasticity and at the end of the {r['T_s']:g} s run."


def vg_weights(fig, files, th):
    score = load_json(pick(files, "score_p3.json"))
    seeds = [str(s) for s in score["seeds"]]
    conv = score["CONV"]["per_seed"]
    axes = fig.subplots(1, 2)
    axes[0].bar(range(len(seeds)), [score["W-IE"]["per_seed_nS"][s] for s in seeds], color=th["I"])
    axes[0].set_ylabel("final mean w_IE (nS)")
    axes[1].errorbar(range(len(seeds)), [conv[s]["b_hz_s"] for s in seeds], yerr=[conv[s]["se_hz_s"] for s in seeds],
                     linestyle="none", marker="o", color=th["E"], ecolor=th["ink"])
    axes[1].axhline(0, color=th["ink"], lw=0.6)
    axes[1].set_ylabel("rate slope b (Hz/s) ± se")
    for ax in axes:
        ax.set_xticks(range(len(seeds)), [f"seed {s}" for s in seeds], fontsize=7)
        tidy(ax)
    return (f"Learned inhibitory weight (mean {score['W-IE']['mean_nS']:.2f} nS) and the end-of-run E rate slope "
            f"per seed; CONV: {score['CONV']['criterion']}.")


# ---- Predictive coding light ----

PCL3 = "warehouse/pcl/scores/fig3_v4_full.json"
PCL2 = "repo/artifacts/etudes/papers/pcl"


def pcl_pref(key, xlabel, what):
    def draw(fig, files, th):
        s = load_json(files[0])
        axes = fig.subplots(1, 2, sharey=True)
        for ax, kind in zip(axes, ("simple", "complex")):
            c = s[f"pref_{key}_counts_{kind}"]
            ax.bar(range(len(c)), c, color=th["E"] if kind == "simple" else th["SOM"])
            ax.set_title(f"{kind} cells (n = {sum(c)})", fontsize=8)
            ax.set_xlabel(xlabel)
            tidy(ax)
        axes[0].set_ylabel("cells")
        return f"Preferred {what} of each cell in the scored Fig. 3 network (v4 full run, seed {s['seed']})."
    return draw


def pcl_fig2_weights(fig, files, th):
    axes = fig.subplots(1, 2)
    docs = []
    for ax, name, title, key in ((axes[0], "pcl/fig2.json", "standalone (P0)", "w_hist"),
                                 (axes[1], "pcl/fig2_hdp_avg_s012.json", "jaxfne HDP kernel (K0b), mV", "w_hist_mV")):
        d = load_json(pick(files, name))
        docs.append(d)
        for n, run in enumerate(d["runs"]):
            w = np.asarray(run[key])
            for i in range(w.shape[1]):
                ax.plot(np.arange(w.shape[0]), w[:, i], color=th["series"][i], lw=0.9, alpha=0.8,
                        label=f"input {i + 1}" if n == 0 else None)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("epoch")
        tidy(ax)
    axes[0].set_ylabel("inhibitory weight")
    axes[1].legend(frameon=False, fontsize=7)
    sup = [", ".join(f"{x:.2f}" for x in d["suppression_mean"]) for d in docs]
    return (f"Inhibitory weight per epoch for inputs 1-3, seeds {docs[0]['params']['seeds']}; mean suppression "
            f"{sup[0]} (P0) and {sup[1]} (K0b).")


SUITES = [
    Panel("caminos", "raster", "Spike raster", [f"{CAM_PLOSR}/manifest.json", f"{CAM_PLOSR}/seed1/baseline.npz"], cam_raster),
    Panel("caminos", "rates", "Population rates", [f"{CAM_SCORED}/manifest.json", f"{CAM_SCORED}/score.json",
          f"{CAM_SCORED}/counts_seed1_p0.07_float32_dt1e-04.npy"], cam_rates),
    Panel("caminos", "psd", "E rate spectra", [f"{CAM_SCORED}/manifest.json", f"{CAM_SCORED}/score.json",
          f"{CAM_SCORED}/counts_seed*_p0.07_float32_dt1e-04.npy"], cam_psd),
    Panel("caminos", "conditions", "Silencing and randomization", [f"{CAM_PLOSR}/manifest.json", f"{CAM_PLOSR}/seed*/*.npz"],
          cam_conditions),
    Panel("caminos-cellrep", "s1_sweep", "S1: connectivity sweep", [CAM_S1], s1_sweep),
    Panel("caminos-cellrep", "s1_spectra", "S1: E spectra across p", [CAM_S1], s1_spectra),
    Panel("destexhe2009", "raster", "Spike raster", [f"{DX}/score_seed9_6b.json", f"{DX}/spikes_seed9_fig6b_cont.npz"],
          dx_raster),
    Panel("destexhe2009", "rates", "Rate histories", [f"{DX}/score_seed*.json", f"{DX}/spikes_seed*.npz"], dx_rates),
    Panel("destexhe2009", "stats", "Irregularity and correlation", [f"{DX}/score_seed*.json", f"{DX}/spikes_seed*.npz"],
          dx_stats),
    Panel("gsdr2026", "fit", "Fit history", [f"{GS}/seed*/manifest.json"], gs_fit),
    Panel("gsdr2026", "gstar", "Fitted conductances", [f"{GS}/seed*/g_star_uS_float32.npy"], gs_gstar),
    Panel("gsdr2026", "eval", "Held-out trials", [f"{GS}/eval_seed*_trial*.json"], gs_eval),
    Panel("izhikevich2003", "fig2", "Single-cell voltage traces", [f"{IZ}/fig2/manifest.json", f"{IZ}/fig2/C*.npz"], iz_fig2),
    Panel("izhikevich2003", "raster", "Network raster", [f"{IZ}/fig3/manifest.json", f"{IZ}/fig3/1.npz"], iz_fig3),
    Panel("izhikevich2003", "psd", "Population rate spectra", [f"{IZ}/score.json", f"{IZ}/fig3/manifest.json",
          f"{IZ}/fig3/*.npz"], iz_psd),
    Panel("brunel2000", "rasters", "States A-D", [f"{BR}/manifest.json", f"{BR}/spikes_seed4_*.npz"], br_rasters),
    Panel("brunel2000", "psd", "Population rate spectra", [f"{BR}/manifest.json", f"{BR}/spikes_seed*.npz"], br_psd),
    Panel("potjans2014", "raster", "Spike raster", [f"{PD}/seed4/manifest.json", f"{PD}/seed4/spikes_seed4_scored.npz"],
          pd_raster),
    Panel("potjans2014", "rates", "Population rates", [f"{PD}/seed*/manifest.json"], pd_bars("rate_hz", "rate (Hz)", "Firing rate")),
    Panel("potjans2014", "cv", "Spike irregularity", [f"{PD}/seed*/manifest.json"], pd_bars("cv_isi", "CV ISI", "CV ISI")),
    Panel("vogels2011", "history", "Rate history", [f"{VG}/score_p3.json", f"{VG}/scored/seed*/score_receipt.json"], vg_history),
    Panel("vogels2011", "raster", "Before and after plasticity", [f"{VG}/scored/seed4/score_receipt.json",
          f"{VG}/scored/seed4/spikes_seed4_seg*.npz"], vg_raster),
    Panel("vogels2011", "weights", "Weights and convergence", [f"{VG}/score_p3.json"], vg_weights),
    Panel("pcl-fig3", "pref_ori", "Orientation preference", [PCL3], pcl_pref("ori", "orientation index", "orientation")),
    Panel("pcl-fig3", "pref_sf", "Spatial-frequency preference", [PCL3],
          pcl_pref("sf", "spatial-frequency index", "spatial frequency")),
    Panel("pcl-fig2", "weights", "Inhibitory weight trajectories", [f"{PCL2}/fig2.json", f"{PCL2}/fig2_hdp_avg_s012.json"],
          pcl_fig2_weights),
]


def png_names(panel):
    stem = f"suite-{panel.paper}-{panel.id}"
    return f"{stem}.png", f"{stem}.dark.png"


def sources_of(panel):
    """Resolved source files of a panel; a declared path or glob that finds no file raises."""
    files = []
    for s in panel.sources:
        hit = [p for p in resolve(s) if p.is_file()]
        if not hit:
            raise FileNotFoundError(f"{panel.paper}/{panel.id}: no receipt at {s}")
        files += hit
    return sorted(set(files))


def sha256_file(path, _cache={}):
    key = Path(path).resolve()
    if key not in _cache:
        h = hashlib.sha256()
        with open(key, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        _cache[key] = h.hexdigest()
    return _cache[key]


def check(entries, figures):
    """Problems of the manifest's suite entries against SUITES, the PNGs and any receipt reachable here.

    Returns (problems, sources verified, sources offline). A source that is not on this machine (CI)
    is offline: its PNGs are still checked against their recorded sha256.
    """
    problems, verified, offline = [], 0, 0
    declared = {f"{p.paper}/{p.id}": p for p in SUITES}
    recorded = {f"{e['paper']}/{e['id']}": e for e in entries}
    for key in sorted(set(declared) - set(recorded)):
        problems.append(f"{key}: declared panel not rendered")
    for key in sorted(set(recorded) - set(declared)):
        problems.append(f"{key}: rendered panel no longer declared")
    for key in sorted(set(declared) & set(recorded)):
        panel, entry = declared[key], recorded[key]
        if set(entry["png_sha256"]) != set(png_names(panel)):
            problems.append(f"{key}: PNG names {sorted(entry['png_sha256'])} != {list(png_names(panel))}")
        for name, digest in entry["png_sha256"].items():
            path = Path(figures) / name
            if not path.is_file():
                problems.append(f"{key}: {name} missing")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                problems.append(f"{key}: {name} differs from its recorded sha256")
        for text, digest in entry["sources"].items():
            path = unlabel(text)
            if not path.is_file():
                offline += 1
            elif sha256_file(path) != digest:
                problems.append(f"{key}: receipt {text} changed since the render")
            else:
                verified += 1
        try:
            now = {label(f) for f in sources_of(panel)}
        except FileNotFoundError:
            continue
        if now != set(entry["sources"]):
            problems.append(f"{key}: receipt set differs (new {sorted(now - set(entry['sources']))}, "
                            f"gone {sorted(set(entry['sources']) - now)})")
    return problems, verified, offline


def build(panel, figures):
    """Render a panel's light and dark PNGs into figures; return its manifest entry."""
    names = png_names(panel)
    captions = [render(panel, Path(figures) / name, dark) for name, dark in zip(names, (False, True))]
    if captions[0] != captions[1]:
        raise ValueError(f"{panel.paper}/{panel.id}: light and dark captions differ")
    return {
        "paper": panel.paper, "id": panel.id, "title": panel.title, "caption": captions[0],
        "png": {"light": names[0], "dark": names[1]},
        "png_sha256": {n: hashlib.sha256((Path(figures) / n).read_bytes()).hexdigest() for n in names},
        "sources": {label(f): sha256_file(f) for f in sources_of(panel)},
    }


def render(panel, path, dark):
    """Draw one panel to path as a palette PNG; return its caption."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # Styles are global in pyplot: reset each draw so a dark figure never restyles the next light one.
    plt.style.use("dark_background" if dark else "default")
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    fig = plt.figure(figsize=(6.4, 3.4), dpi=100)
    caption = panel.draw(fig, sources_of(panel), THEMES[dark])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), metadata={"Software": None})
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    img.save(path, format="PNG", optimize=True)
    return caption
