"""Izhikevich 2003 P3 calibration gate: false-positive and detection rates of the band call.

Spec: ``PREREG_P3.md`` "Calibration gate". Synthetic signals only; no model run.
Writes one JSON receipt with every row's rate, the chosen θ and whether the alpha band widened.

    python calibrate_detector.py --out calibration.json [--workers 8] [--draw-scale 1.0]

``--draw-scale`` below 1 is for smoke runs only (fewer draws; the receipt records it).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("izh_analysis", HERE / "izh_analysis.py")
izh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(izh)

ROOT_SEED = 20261013
N_BINS = 10_000  # 10 s at 1 ms
FS = 1000.0
MEAN_COUNT = 8.0  # N = 1000 at 8 Hz, 1 ms bins
N_NULL, N_DET, N_REF = 1000, 500, 200
THETAS = (2.0, 3.0, 4.0, 6.0)
BANDS = {"alpha": (8.0, 13.0), "gamma": (30.0, 50.0)}
ALPHA_WIDE = (7.0, 14.0)
TARGETS = {10.0: "alpha", 40.0: "gamma"}
FORMS = ("sine", "ar2")
FWHM_HZ = 3.0
BURN_IN = 1000
F_TOL_HZ = 1.5
FPR_MAX, DET_MIN = 0.05, 0.95

# Null floors, in prereg order: white, then tau-major lognormal AR(1) (tau in bins, multiple m).
FLOORS = [("white", None, None)] + [(f"ar1_tau{t}_m{m}", t, m) for t in (8, 30) for m in (0.5, 2.0, 8.0)]


def floor_draw(rng, floor):
    _, tau, m = floor
    if tau is None:
        return rng.poisson(MEAN_COUNT, N_BINS).astype(float)
    phi = math.exp(-1.0 / tau)
    sigma = math.sqrt(math.log(1.0 + m / MEAN_COUNT))
    g = rng.standard_normal(N_BINS)  # g[0] = z_0, g[1:] = eps_t
    z = np.empty(N_BINS)
    z[0] = g[0]
    c = math.sqrt(1.0 - phi * phi)
    for t in range(1, N_BINS):
        z[t] = phi * z[t - 1] + c * g[t]
    lam = MEAN_COUNT * np.exp(sigma * z - sigma * sigma / 2.0)
    return rng.poisson(lam).astype(float)


def component_draw(rng, f0, form):
    """Unit-amplitude component (sinusoid amplitude 1; AR(2) with N(0, 1) innovations)."""
    if form == "sine":
        phase = rng.uniform(0.0, 2.0 * math.pi)
        return np.sin(2.0 * math.pi * f0 * np.arange(N_BINS) / FS + phase)
    r = math.exp(-math.pi * FWHM_HZ / FS)
    a1, a2 = 2.0 * r * math.cos(2.0 * math.pi * f0 / FS), -r * r
    e = rng.standard_normal(N_BINS + BURN_IN)
    x = np.zeros(N_BINS + BURN_IN)
    for t in range(2, x.size):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
    return x[BURN_IN:]


def episode_mask(rng):
    slots = rng.choice(20, 4, replace=False)
    mask = np.zeros(N_BINS)
    for s in slots:
        mask[s * 500:(s + 1) * 500] = 1.0
    return mask


def rng_for(i, j):
    return np.random.default_rng([ROOT_SEED, i, j])


def welch_psd(x):
    x = x - x.mean()
    return izh.welch(x, fs=FS, window="hann", nperseg=izh.WELCH_NPERSEG,
                     noverlap=izh.WELCH_NOVERLAP, detrend=False, scaling="density")


# --- rows -------------------------------------------------------------------------------------

def build_rows():
    """Every row in prereg order with its index i."""
    rows = [{"i": i, "kind": "null", "floor": k} for i, k in enumerate(range(len(FLOORS)))]

    def det_block(kind, scale):
        for k in range(len(FLOORS)):
            for f0 in (10.0, 40.0):
                for form in FORMS:
                    rows.append({"i": len(rows), "kind": kind, "floor": k, "f0": f0,
                                 "form": form, "scale": scale})

    det_block("detect", 3.0)
    det_block("report", 1.5)
    det_block("report", 2.0)
    for k in (0, FLOORS.index(("ar1_tau30_m8.0", 30, 8.0))):
        for f0 in (10.0, 40.0):
            for form in FORMS:
                rows.append({"i": len(rows), "kind": "intermittent", "floor": k, "f0": f0,
                             "form": form, "scale": 3.0})
    return rows


# --- workers ----------------------------------------------------------------------------------

def _null_task(args):
    i, k, j, bands = args
    x = floor_draw(rng_for(i, j), FLOORS[k])
    return [izh.peak_excess(x, FS, b) for b in bands]


def _ref_floor_task(args):
    i, k, j = args
    return welch_psd(floor_draw(rng_for(i, j), FLOORS[k]))[1]


def _ref_comp_task(args):
    i, f0, form, j = args
    return welch_psd(component_draw(rng_for(1000 + i, j), f0, form))[1]


def _det_task(args):
    i, k, f0, form, j, amps, band, intermittent = args
    rng = rng_for(i, j)
    floor = floor_draw(rng, FLOORS[k])
    comp = component_draw(rng, f0, form)
    if intermittent:
        comp = comp * episode_mask(rng)
    out = []
    for a in amps:
        out.append(izh.peak_excess(floor + a * comp, FS, band))
    return out


def detected(res, theta, f0):
    return izh.holds(res, theta) and abs(res["f_pk"] - f0) <= F_TOL_HZ


# --- main -------------------------------------------------------------------------------------

def run(out_path, workers, draw_scale):
    n_null = max(2, int(N_NULL * draw_scale))
    n_det = max(2, int(N_DET * draw_scale))
    n_ref = max(2, int(N_REF * draw_scale))
    rows = build_rows()
    freqs = welch_psd(np.zeros(N_BINS) + 1.0)[0]
    tbin = {f0: int(np.argmin(np.abs(freqs - f0))) for f0 in TARGETS}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # Reference floor PSD F: null draws j = 0..n_ref-1 of the floor's null row.
        F = {}
        for k in range(len(FLOORS)):
            psds = list(ex.map(_ref_floor_task, [(k, k, j) for j in range(n_ref)], chunksize=8))
            arr = np.array(psds)
            F[k] = {f0: (float(arr[:, b].mean()), float(arr[:, b].std(ddof=1) / math.sqrt(len(arr))))
                    for f0, b in tbin.items()}
        # Unit component PSD C per detection-type row (seeded 1000 + i).
        C = {}
        for r in rows:
            if r["kind"] == "null":
                continue
            psds = np.array(list(ex.map(_ref_comp_task,
                                        [(r["i"], r["f0"], r["form"], j) for j in range(n_ref)],
                                        chunksize=8)))
            b = tbin[r["f0"]]
            C[r["i"]] = (float(psds[:, b].mean()), float(psds[:, b].std(ddof=1) / math.sqrt(len(psds))))

        detect_row = {(r["floor"], r["f0"], r["form"]): r for r in rows if r["kind"] == "detect"}

        def amp(r, theta):
            if r["kind"] == "intermittent":  # the continuous 3θ row's amplitude
                r = detect_row[(r["floor"], r["f0"], r["form"])]
            f_mean = F[r["floor"]][r["f0"]][0]
            return math.sqrt((r["scale"] * theta - 1.0) * f_mean / C[r["i"]][0])

        def null_rates(alpha_band):
            bands = [alpha_band, BANDS["gamma"]]
            out = {}
            for r in rows:
                if r["kind"] != "null":
                    continue
                res = list(ex.map(_null_task, [(r["i"], r["floor"], j, bands) for j in range(n_null)],
                                  chunksize=8))
                out[r["i"]] = {theta: {name: float(np.mean([izh.holds(x[n], theta) for x in res]))
                                       for n, name in enumerate(("alpha", "gamma"))}
                               for theta in THETAS}
            return out

        def det_rates(kind, thetas, alpha_band):
            out = {}
            for r in rows:
                if r["kind"] != kind:
                    continue
                band = alpha_band if TARGETS[r["f0"]] == "alpha" else BANDS["gamma"]
                amps = [amp(r, t) for t in thetas]
                res = list(ex.map(_det_task, [(r["i"], r["floor"], r["f0"], r["form"], j, amps, band,
                                               kind == "intermittent") for j in range(n_det)],
                                  chunksize=8))
                out[r["i"]] = {t: float(np.mean([detected(x[n], t, r["f0"]) for x in res]))
                               for n, t in enumerate(thetas)}
            return out

        def choose(nulls, dets):
            det_rows = [r for r in rows if r["kind"] == "detect"]
            ok_i = {t: all(nulls[i][t][b] <= FPR_MAX for i in nulls for b in ("alpha", "gamma"))
                    for t in THETAS}
            ok_g = {t: all(dets[r["i"]][t] >= DET_MIN for r in det_rows if r["f0"] == 40.0) for t in THETAS}
            ok_a = {t: all(dets[r["i"]][t] >= DET_MIN for r in det_rows if r["f0"] == 10.0) for t in THETAS}
            chosen = next((t for t in THETAS if ok_i[t] and ok_g[t] and ok_a[t]), None)
            alpha_only = chosen is None and any(ok_i[t] and ok_g[t] for t in THETAS)
            return chosen, alpha_only, {"fpr_ok": ok_i, "gamma_ok": ok_g, "alpha_ok": ok_a}

        alpha_band, widened = BANDS["alpha"], False
        nulls = null_rates(alpha_band)
        dets = det_rates("detect", THETAS, alpha_band)
        theta, alpha_only, checks = choose(nulls, dets)
        first_pass = {"nulls": nulls, "detect": dets, "checks": checks}
        if theta is None and alpha_only:
            alpha_band, widened = ALPHA_WIDE, True
            nulls = null_rates(alpha_band)
            dets = det_rates("detect", THETAS, alpha_band)
            theta, _, checks = choose(nulls, dets)

        reported = {}
        if theta is not None:
            for kind_scale in (1.5, 2.0):
                sub = [r for r in rows if r["kind"] == "report" and r["scale"] == kind_scale]
                band_rows = det_rates_subset(ex, sub, theta, alpha_band, amp, n_det)
                reported[str(kind_scale)] = band_rows
            reported["intermittent"] = det_rates("intermittent", [theta], alpha_band)

    sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    receipt = {
        "spec": "PREREG_P3.md Calibration gate",
        "git_sha": sha,
        "draw_scale": draw_scale,
        "n_null": n_null, "n_det": n_det, "n_ref": n_ref,
        "rows": rows,
        "theta": theta,
        "alpha_band": list(alpha_band),
        "alpha_widened": widened,
        "checks": _jsonable(checks),
        "null_fpr": _jsonable(nulls),
        "detect_3theta": _jsonable(dets),
        "first_pass": _jsonable(first_pass) if widened else None,
        "reported": _jsonable(reported),
        "F_target": _jsonable(F),
        "C_target": _jsonable(C),
    }
    Path(out_path).write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return receipt


def det_rates_subset(ex, sub, theta, alpha_band, amp, n_det):
    out = {}
    for r in sub:
        band = alpha_band if TARGETS[r["f0"]] == "alpha" else BANDS["gamma"]
        a = amp(r, theta)
        res = list(ex.map(_det_task, [(r["i"], r["floor"], r["f0"], r["form"], j, [a], band, False)
                                      for j in range(n_det)], chunksize=8))
        out[r["i"]] = float(np.mean([detected(x[0], theta, r["f0"]) for x in res]))
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--draw-scale", type=float, default=1.0)
    a = ap.parse_args()
    r = run(a.out, a.workers, a.draw_scale)
    print(f"theta={r['theta']} alpha_band={r['alpha_band']} widened={r['alpha_widened']} "
          f"draw_scale={r['draw_scale']}")


if __name__ == "__main__":
    main()
