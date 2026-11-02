"""Seed-mean scorer for CAMINOS PLoS CB Figs 2C-F / 3C-F raster runs.

Usage: python score_plos_rasters.py <run dir> <out json> [--figdir DIR]

Reads the run directory written by ``run_plos_rasters.py`` (per-seed
``seed<s>/<condition>.npz``, ``meanfield_<condition>.npz``, ``manifest.json``),
computes the estimands and verdicts declared in ``RESULT_PLOS_FIGS.md``
section "Panels 2C-F and 3C-F: criteria declared before scoring", writes one
JSON summary, and optionally renders ``fig2_CF.png`` / ``fig3_CF.png``.

Exit codes: 0 ok; 2 on usage errors or missing/corrupt input (the message
names the file).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_repo_root = Path(__file__).resolve().parents[4]  # P-025: the checkout's jaxfne, never site-packages
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_HERE))

POP_NAMES = ("E", "PV", "SOM")
BIN_S = 1e-3  # s, network rate_hz bins and mean-field rebinning

CONDITIONS_9 = [
    "baseline",
    "pv_silenced_0.4",
    "pv_silenced_0.9",
    "som_silenced_0.4",
    "som_silenced_0.9",
    "pv_randomized_0.4",
    "pv_randomized_0.9",
    "som_randomized_0.4",
    "som_randomized_0.9",
]

# Fig 2 panels need these mean-field traces; Fig 3 (randomization) has none.
MF_CONDITIONS = [
    "baseline",
    "pv_silenced_0.4",
    "pv_silenced_0.9",
    "som_silenced_0.4",
    "som_silenced_0.9",
]

FIG2_PANELS = [
    ("2C", "pv_silenced_0.4", "PV silenced 40%"),
    ("2D", "som_silenced_0.4", "SOM silenced 40%"),
    ("2E", "pv_silenced_0.9", "PV silenced 90%"),
    ("2F", "som_silenced_0.9", "SOM silenced 90%"),
]
FIG3_PANELS = [
    ("3C", "pv_randomized_0.4", "PV randomized 40%"),
    ("3D", "som_randomized_0.4", "SOM randomized 40%"),
    ("3E", "pv_randomized_0.9", "PV randomized 90%"),
    ("3F", "som_randomized_0.9", "SOM randomized 90%"),
]

# Direction reference: which silencing condition each randomization panel
# follows ("like silencing at the same fraction").
_DIRECTION_REF = {
    "3C": "pv_silenced_0.4",
    "3E": "pv_silenced_0.9",
    "3D": "som_silenced_0.4",
    "3F": "som_silenced_0.9",
}

_SILENCED_POP = {"pv_silenced": 1, "som_silenced": 2}

COLORS = {"E": "#1f77b4", "PV": "#d62728", "SOM": "#2ca02c"}


class _InputError(Exception):
    """Missing or corrupt run input; the message names the file."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def estimate_f(trace, dt=BIN_S, lo=5.0, hi=100.0):
    """Frequency of argmax |FFT| of the mean-subtracted trace over lo-hi Hz."""
    x = np.asarray(trace, dtype=float).ravel() - np.asarray(trace, dtype=float).ravel().mean()
    if x.size < 2 or not np.any(x):
        return 0.0
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=dt)
    sel = (freqs >= lo) & (freqs <= hi)
    if not np.any(sel) or not np.any(spec[sel]):
        return 0.0
    return float(freqs[sel][int(np.argmax(spec[sel]))])


def estimate_amplitude(trace):
    """Peak-to-peak amplitude of a trace."""
    x = np.asarray(trace, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    return float(x.max() - x.min())


def bin_meanfield(t, nu, bin_s=BIN_S):
    """Average mean-field samples into 1 ms bins; returns (centers, means)."""
    t = np.asarray(t, dtype=float).ravel()
    nu = np.asarray(nu, dtype=float)
    t0 = float(t.min())
    idx = np.floor((t - t0) / bin_s).astype(int)
    n_bins = max(1, int(round((float(t.max()) - t0) / bin_s)))
    idx = np.clip(idx, 0, n_bins - 1)
    sums = np.zeros((n_bins, nu.shape[1]))
    counts = np.zeros(n_bins)
    tsum = np.zeros(n_bins)
    np.add.at(sums, idx, nu)
    np.add.at(counts, idx, 1)
    np.add.at(tsum, idx, t)
    safe = np.where(counts > 0, counts, 1.0)
    means = sums / safe[:, None]
    centers = np.where(counts > 0, tsum / safe, t0 + (np.arange(n_bins) + 0.5) * bin_s)
    return centers, means


def silenced_population(condition):
    """(population index, fraction) silenced by a `*_silenced_*` condition."""
    for prefix, index in _SILENCED_POP.items():
        if condition.startswith(prefix + "_"):
            try:
                return index, float(condition[len(prefix) + 1:])
            except ValueError:
                return None, None
    return None, None


def corrected_means(rate, condition, type_slices):
    """Per-population mean rates; a silenced population is per unsilenced cell.

    ``rate`` is Hz per cell over all cells of the type; silencing masks
    ``round(f N)`` cells (`caminos_jax.manip_mask`), so the network mean is
    ``rate * N / (N - round(f N))`` to match the mean field's per-active-cell
    rate. Returns (corrected dict, raw dict, n_cells dict).
    """
    slices = [(int(s0), int(s1)) for s0, s1 in np.asarray(type_slices).tolist()]
    raw, out, cells = {}, {}, {}
    for i, name in enumerate(POP_NAMES):
        n = slices[i][1] - slices[i][0]
        cells[name] = n
        raw[name] = float(np.asarray(rate[:, i], dtype=float).mean())
        out[name] = raw[name]
    index, frac = silenced_population(condition)
    if index is not None:
        name = POP_NAMES[index]
        n = cells[name]
        masked = int(round(frac * n))
        if masked >= n:
            raise _InputError("condition %s masks all %d %s cells" % (condition, n, name))
        out[name] = raw[name] * n / (n - masked)
    return out, raw, cells


def _ratio(value, base):
    if base is None:
        return None
    if base == 0.0:
        return 1.0 if value == 0.0 else float("inf")
    return float(value) / float(base)


def direction(ratio):
    """Sign of (ratio - 1): 1 above baseline, -1 below, 0 on, None unknown."""
    if ratio is None or not np.isfinite(ratio):
        if ratio is not None and ratio == float("inf"):
            return 1
        return None
    if ratio > 1.0:
        return 1
    if ratio < 1.0:
        return -1
    return 0


def _rel_within(a, b, tol):
    if b == 0.0:
        return bool(a == 0.0)
    return bool(abs(a - b) <= tol * abs(b))


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _load_condition(path, rel):
    try:
        with np.load(path, allow_pickle=False) as z:
            return dict(z)
    except (OSError, ValueError) as exc:
        raise _InputError("cannot read %s: %s" % (rel, exc))


def _network_estimands(z, condition):
    rate = np.asarray(z["rate_hz"], dtype=float)
    slices = np.asarray(z["type_slices"])
    mean, raw, cells = corrected_means(rate, condition, slices)
    return {
        "n_cells": cells,
        "raw_mean_rate_hz": raw,
        "mean_rate_hz": mean,
        "f_hz": estimate_f(rate[:, 0]),
        "e_amplitude": estimate_amplitude(rate[:, 0]),
        "f_pk_hz": float(np.asarray(z["f_pk_hz"])),
        "band_power_25_45": float(np.asarray(z["band_power_25_45"])),
        "max_e_rate_hz": float(np.asarray(z["max_e_rate_hz"])),
    }


def _mf_estimands(z):
    centers, binned = bin_meanfield(np.asarray(z["t_s"]), np.asarray(z["nu_hz"]))
    mean = binned.mean(axis=0)
    e = binned[:, 0]
    cv = float(e.std() / mean[0]) if mean[0] > 0 else None
    return {
        "centers_s": centers,
        "binned_hz": binned,
        "mean_rate_hz": dict(zip(POP_NAMES, (float(v) for v in mean))),
        "f_hz": estimate_f(e),
        "e_amplitude": estimate_amplitude(e),
        "e_cv": cv,
    }


def _verify_manifest(run_dir):
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise _InputError("missing manifest.json in %s" % run_dir)
    raw_manifest = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except ValueError as exc:
        raise _InputError("cannot parse manifest.json: %s" % exc)
    for key in ("seeds", "conditions", "meanfield"):
        if key not in manifest:
            raise _InputError("manifest.json lacks key %r" % key)
    inputs = {}
    want = {}
    for conds in manifest["conditions"].values():
        for entry in conds.values():
            want[entry["file"]] = entry["sha256"]
    for entry in manifest["meanfield"].values():
        want[entry["file"]] = entry["sha256"]
    for rel, digest in want.items():
        path = Path(run_dir) / rel
        if not path.is_file():
            raise _InputError("missing input file %s" % rel)
        if _sha256(path) != digest:
            raise _InputError("sha256 mismatch for %s" % rel)
        inputs[rel] = digest
    return manifest, _sha256(manifest_path), inputs


def _mean_across(values):
    vals = [v for v in values if v is not None]
    if len(vals) != len(values) or not vals:
        return None
    return float(sum(vals) / len(vals))


def score_run(run_dir):
    """Score a verified run directory; returns the JSON-able summary dict."""
    manifest, manifest_sha, inputs = _verify_manifest(run_dir)
    seeds = [str(s) for s in manifest["seeds"]]
    per_seed, mf = {}, {}
    for cond in MF_CONDITIONS:
        entry = manifest["meanfield"].get(cond)
        if entry is None:
            raise _InputError("manifest lacks mean-field entry meanfield_%s.npz" % cond)
        path = Path(run_dir) / entry["file"]
        if not path.is_file():
            raise _InputError("missing input file %s" % entry["file"])
        mf[cond] = _mf_estimands(_load_condition(path, entry["file"]))
    for s in seeds:
        per_seed[s] = {}
        conds = manifest["conditions"].get(s, {})
        if "baseline" not in conds:
            raise _InputError("manifest lacks conditions entry %s/baseline.npz" % s)
        base = _network_estimands(
            _load_condition(Path(run_dir) / conds["baseline"]["file"],
                            conds["baseline"]["file"]), "baseline")
        for cond in CONDITIONS_9:
            if cond not in conds:
                raise _InputError("manifest lacks conditions entry %s/%s.npz" % (s, cond))
            est = base if cond == "baseline" else _network_estimands(
                _load_condition(Path(run_dir) / conds[cond]["file"],
                                conds[cond]["file"]), cond)
            est["e_rate_ratio"] = _finite(_ratio(est["mean_rate_hz"]["E"], base["mean_rate_hz"]["E"]))
            est["f_pk_ratio"] = _finite(_ratio(est["f_pk_hz"], base["f_pk_hz"]))
            est["band_ratio"] = _finite(_ratio(est["band_power_25_45"],
                                               base["band_power_25_45"]))
            per_seed[s][cond] = est
    seed_means = {}
    for cond in CONDITIONS_9:
        first = per_seed[seeds[0]][cond]
        seed_means[cond] = {
            "n_cells": first["n_cells"],
            "mean_rate_hz": {p: _mean_across([per_seed[s][cond]["mean_rate_hz"][p]
                                              for s in seeds]) for p in POP_NAMES},
            "f_hz": _mean_across([per_seed[s][cond]["f_hz"] for s in seeds]),
            "e_amplitude": _mean_across([per_seed[s][cond]["e_amplitude"] for s in seeds]),
            "e_rate_ratio": _mean_across([per_seed[s][cond]["e_rate_ratio"] for s in seeds]),
            "f_pk_ratio": _mean_across([per_seed[s][cond]["f_pk_ratio"] for s in seeds]),
            "band_ratio": _mean_across([per_seed[s][cond]["band_ratio"] for s in seeds]),
        }
    panels = _score_panels(seed_means, mf)
    return {
        "run_dir": str(run_dir),
        "manifest_sha256": manifest_sha,
        "inputs": inputs,
        "seeds": manifest["seeds"],
        "per_seed": {s: {c: _public_est(e) for c, e in conds.items()}
                     for s, conds in per_seed.items()},
        "meanfield": {c: _public_mf(m) for c, m in mf.items()},
        "seed_means": seed_means,
        "panels": panels,
    }


def _public_est(est):
    out = dict(est)
    return out


def _public_mf(m):
    return {k: v for k, v in m.items() if k not in ("centers_s", "binned_hz")}


def _verdict(criteria):
    flags = [c["pass"] for c in criteria]
    if all(flags):
        return "pass"
    if not any(flags):
        return "fail"
    return "partial"


def _score_panels(means, mf):
    """Criteria and verdicts per the declared table (values on seed means)."""
    panels = {}

    def net_e_ratio(cond):
        return means[cond]["e_rate_ratio"]  # per-seed ratio to the seed's baseline, then the seed mean

    # 2C: PV silenced 40 %: network within 20 % of the mean field per
    # population; f within 10 %.
    cond = "pv_silenced_0.4"
    pops = {}
    for p in POP_NAMES:
        net, ref = means[cond]["mean_rate_hz"][p], mf[cond]["mean_rate_hz"][p]
        pops[p] = {"network_hz": net, "meanfield_hz": ref,
                   "rel_err": abs(net - ref) / abs(ref) if ref else (0.0 if net == 0 else None),
                   "pass": _rel_within(net, ref, 0.20)}
    crit_rates = {"name": "network mean rate within 20% of mean field, per population",
                  "values": pops, "threshold": 0.20,
                  "pass": all(v["pass"] for v in pops.values())}
    net_f, ref_f = means[cond]["f_hz"], mf[cond]["f_hz"]
    crit_f = {"name": "network f within 10% of mean-field f",
              "values": {"network_f_hz": net_f, "meanfield_f_hz": ref_f,
                         "rel_err": abs(net_f - ref_f) / abs(ref_f) if ref_f
                         else (0.0 if net_f == 0 else None)},
              "threshold": 0.10, "pass": _rel_within(net_f, ref_f, 0.10)}
    panels["2C"] = {"condition": cond, "criteria": [crit_rates, crit_f],
                    "verdict": _verdict([crit_rates, crit_f])}

    # 2D: SOM silenced 40 %: band power below 1 and above 0.10; mean-field
    # E amplitude below the mean-field baseline.
    cond = "som_silenced_0.4"
    band = means[cond]["band_ratio"]
    crit_band = {"name": "band power ratio below 1 and above 0.10",
                 "values": {"band_ratio": band},
                 "threshold": {"lower": 0.10, "upper": 1.0},
                 "pass": bool(band is not None and 0.10 < band < 1.0)}
    amp, amp_base = mf[cond]["e_amplitude"], mf["baseline"]["e_amplitude"]
    crit_mf = {"name": "mean-field E amplitude below mean-field baseline",
               "values": {"meanfield_e_amplitude": amp,
                          "meanfield_baseline_e_amplitude": amp_base},
               "threshold": "below baseline",
               "pass": bool(amp < amp_base)}
    panels["2D"] = {"condition": cond, "criteria": [crit_band, crit_mf],
                    "verdict": _verdict([crit_band, crit_mf])}

    # 2E: PV silenced 90 %: two claims -- the network is seizure-like (E at
    # 10x the seed baseline) and the mean field sits at a fixed point
    # (E mean at least 100 Hz, CV over time below 0.05).
    cond = "pv_silenced_0.9"
    ratio, mf_mean, mf_cv = net_e_ratio(cond), mf[cond]["mean_rate_hz"]["E"], mf[cond]["e_cv"]
    crit_net = {"name": "network E at 10x baseline",
                "values": {"network_e_ratio": ratio},
                "threshold": {"network_e_ratio_min": 10},
                "pass": bool(ratio is not None and ratio >= 10)}
    crit_fp = {"name": "mean-field E fixed point at 100+ Hz",
               "values": {"mf_e_mean_hz": mf_mean, "mf_e_cv": mf_cv},
               "threshold": {"mf_e_mean_min_hz": 100, "mf_e_cv_max": 0.05},
               "pass": bool(mf_mean is not None and mf_mean >= 100
                            and mf_cv is not None and mf_cv < 0.05)}
    panels["2E"] = {"condition": cond, "criteria": [crit_net, crit_fp],
                    "verdict": _verdict([crit_net, crit_fp])}

    # 2F: SOM silenced 90 %: band power below 1; mean-field E amplitude
    # below the mean-field baseline.
    cond = "som_silenced_0.9"
    band = means[cond]["band_ratio"]
    crit_band = {"name": "band power ratio below 1",
                 "values": {"band_ratio": band}, "threshold": {"upper": 1.0},
                 "pass": bool(band is not None and band < 1.0)}
    amp, amp_base = mf[cond]["e_amplitude"], mf["baseline"]["e_amplitude"]
    crit_mf = {"name": "mean-field E amplitude below mean-field baseline",
               "values": {"meanfield_e_amplitude": amp,
                          "meanfield_baseline_e_amplitude": amp_base},
               "threshold": "below baseline",
               "pass": bool(amp < amp_base)}
    panels["2F"] = {"condition": cond, "criteria": [crit_band, crit_mf],
                    "verdict": _verdict([crit_band, crit_mf])}

    # 3C-F: like silencing at the same fraction -- f_pk and band power move
    # from baseline in the same direction (sign of ratio - 1). The joint
    # direction check is one claim; 3E additionally meets the 2E network bar.
    for panel, rcond in (("3C", "pv_randomized_0.4"), ("3D", "som_randomized_0.4"),
                         ("3E", "pv_randomized_0.9"), ("3F", "som_randomized_0.9")):
        ref = _DIRECTION_REF[panel]
        got_f, want_f = direction(means[rcond]["f_pk_ratio"]), direction(means[ref]["f_pk_ratio"])
        got_b, want_b = direction(means[rcond]["band_ratio"]), direction(means[ref]["band_ratio"])
        crit_dir = {"name": "f_pk and band power move the same way as %s" % ref,
                    "values": {"f_pk": {"ratio": means[rcond]["f_pk_ratio"],
                                        "direction": got_f,
                                        "reference_ratio": means[ref]["f_pk_ratio"],
                                        "reference_direction": want_f},
                               "band_power": {"ratio": means[rcond]["band_ratio"],
                                              "direction": got_b,
                                              "reference_ratio": means[ref]["band_ratio"],
                                              "reference_direction": want_b}},
                    "threshold": "equal directions",
                    "pass": bool(got_f is not None and got_b is not None
                                 and got_f == want_f and got_b == want_b)}
        if panel == "3E":
            ratio = net_e_ratio(rcond)
            crit_net = {"name": "network E at 10x baseline (2E network bar)",
                        "values": {"network_e_ratio": ratio},
                        "threshold": {"network_e_ratio_min": 10},
                        "pass": bool(ratio is not None and ratio >= 10)}
            criteria = [crit_dir, crit_net]
        else:
            criteria = [crit_dir]
        panels[panel] = {"condition": rcond, "criteria": criteria,
                         "verdict": _verdict(criteria)}
    return panels


def _raster_panel(ax, raster, slices, t_end, title):
    """Seed raster: every 40th E, every 5th PV/SOM cell, one color per type."""
    n_steps = raster.shape[0]
    step_s = 0.5 / n_steps
    times = t_end - 0.5 + (np.arange(n_steps) + 0.5) * step_s
    row = 0
    handles = []
    for i, name in enumerate(POP_NAMES):
        s0, s1 = int(slices[i][0]), int(slices[i][1])
        stride = 40 if i == 0 else 5
        for c in range(s0, s1, stride):
            hits = np.flatnonzero(raster[:, c])
            if hits.size:
                h = ax.scatter(times[hits], np.full(hits.shape, row), s=4,
                               c=COLORS[name], marker="|", linewidths=0.5)
                if not any(hh.get_label() == name for hh in handles):
                    h.set_label(name)
                    handles.append(h)
            row += 1
    ax.set_xlim(t_end - 0.5, t_end)
    ax.set_ylim(-1, max(row, 1))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cell")
    ax.set_title(title)


def _rate_axis(ax, centers, traces, ylabel):
    for i, name in enumerate(POP_NAMES):
        ax.plot(centers, np.asarray(traces)[:, i], c=COLORS[name], lw=1.0, label=name)
    ax.set_xlim(float(centers[0]), float(centers[-1]))
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)


def render_figs(summary, run_dir, manifest, figdir, seed):
    """Render fig2_CF.png / fig3_CF.png (2x2: C, D top; E, F bottom)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_end = float(manifest.get("T_s", 12.0))
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for fname, panels, with_mf in (
            ("fig2_CF.png", FIG2_PANELS, True),
            ("fig3_CF.png", FIG3_PANELS, False)):
        rows = 3 if with_mf else 2
        fig = plt.figure(figsize=(12, 4.5 * rows / 1.5))
        outer = fig.add_gridspec(2, 2, hspace=0.7, wspace=0.25)
        for k, (panel, cond, title) in enumerate(panels):
            inner = outer[k].subgridspec(rows, 1,
                                         height_ratios=([1, 2, 1] if with_mf else [2, 1]),
                                         hspace=0.4)
            entry = manifest["conditions"][str(seed)][cond]
            z = _load_condition(Path(run_dir) / entry["file"], entry["file"])
            raster = np.asarray(z["raster"], dtype=bool)
            slices = np.asarray(z["type_slices"])
            rate = np.asarray(z["rate_hz"], dtype=float)
            n_bins = rate.shape[0]
            centers = t_end - 0.5 + (np.arange(n_bins) + 0.5) * (0.5 / n_bins)
            ax_i = 0
            if with_mf:
                ax_mf = fig.add_subplot(inner[0])
                m = manifest["meanfield"][cond]
                mz = _load_condition(Path(run_dir) / m["file"], m["file"])
                mc, mb = bin_meanfield(np.asarray(mz["t_s"]), np.asarray(mz["nu_hz"]))
                _rate_axis(ax_mf, mc, mb, "MF (Hz)")
                ax_mf.set_xlabel("")
                ax_mf.set_title("%s: %s" % (panel, title))
                ax_i = 1
            ax_ra = fig.add_subplot(inner[ax_i])
            _raster_panel(ax_ra, raster, slices, t_end,
                          "%s: %s" % (panel, title) if not with_mf else "")
            ax_ra.set_xlabel("")
            _rate_axis(fig.add_subplot(inner[ax_i + 1]), centers, rate, "rate (Hz)")
        fig.align_ylabels()
        fig.suptitle("network: seed %s, last 0.5 s; raster every 40th E, every 5th PV and SOM cell; "
                     "rate per cell of the type" % seed, fontsize=9)
        fig.legend(handles=[
            plt.Line2D([], [], color=COLORS[p], lw=2, label=p) for p in POP_NAMES],
            loc="upper right", ncols=3)
        path = figdir / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out[fname] = str(path)
    summary["figures"] = out


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = list(argv)
    figdir = None
    if "--figdir" in args:
        k = args.index("--figdir")
        if k + 1 >= len(args):
            print("error: --figdir needs a directory, got nothing", file=sys.stderr)
            return 2
        figdir = args[k + 1]
        del args[k:k + 2]
    if len(args) != 2 or any(a.startswith("-") for a in args):
        print(__doc__, file=sys.stderr)
        return 2
    run_dir, out_path = args
    try:
        summary = score_run(run_dir)
    except _InputError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    if figdir is not None:
        manifest_raw = json.loads((Path(run_dir) / "manifest.json").read_text())
        seeds = [str(s) for s in manifest_raw["seeds"]]
        seed = 1 if "1" in seeds else seeds[0]
        render_figs(summary, run_dir, manifest_raw, figdir, seed)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
