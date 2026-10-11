"""v5 calibration gate for the CAMINOS oscillation criterion (PREREG_P3.md).

Spec: ``PREREG_P3.md``, "Calibration gate (v5)". Synthetic signals only; no
network data. Each draw is a 3 s E rate at NE = 8000, R0 = 1.2 Hz, dt = 1e-4 s,
Poisson sampled. The gate calls the scorer's own functions
(``caminos_analysis.spectral_peak``, ``e_welch`` and ``oscillation_call``), so
the rule is not restated here.

* Null floors (7): white Poisson; Poisson on a lognormal AR(1) rate with tau 8
  and 30 ms at 0.5x, 2x and 8x the Poisson count variance. AR(1): unit variance,
  phi = exp(-dt / tau). Rate R0 * exp(sigma * x - sigma**2 / 2) with
  sigma**2 = ln(1 + c / (R0 * NE * dt)), so the modulated count variance is c
  times the Poisson count variance.
* theta: the smallest of THETA_GRID at which every floor's FP rate is <= 5 %.
  If none qualifies the script refuses; the fit is revised before signature.
* Detection (14): each floor's null draws plus an additive 35 Hz component on the
  sampled E rate, as a sinusoid (uniform random phase) or an AR(2) bump (centre
  35 Hz, FWHM 5 Hz). Scaled per floor so the mean component PSD at 35 Hz is
  (2 theta - 1) x the floor PSD at 35 Hz (mean over the 200 null draws).
  Realised (component + floor) / floor on those same draws must lie within 10 %
  of 2 theta. Detection = oscillation call at theta and
  |f_pk - 35| <= 2 Hz; gate >= 95 % on all 14 rows.
* Power curve (not gated): true PE 2, 3, 4, 6, 8 on every floor and form.

Run: python calibrate_oscillation.py --out DIR [--draws N] [--power-draws M]
Smoke: --draws 10 --power-draws 10 (prints SMOKE; not a gate).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caminos_analysis as ca  # noqa: E402

NE = 8000
DT = 1e-4  # s
R0 = 1.2  # Hz, E baseline rate
BURN_S = 1.0  # s, discarded so the AR(1) floor and the AR(2) component are stationary
KEPT_S = 3.0  # s, prereg kept window
MOD_HZ = 35.0  # component centre
AR2_FWHM_HZ = 5.0  # AR(2) component full width at half maximum
FREQ_TOL_HZ = 2.0  # detection: |f_pk - 35| <= 2 Hz
TAUS_MS = (8.0, 30.0)
VAR_RATIOS = (0.5, 2.0, 8.0)
THETA_GRID = (4.0, 5.0, 6.0, 8.0)
FORMS = ("sinusoid", "ar2")
POWER_PE = (2.0, 3.0, 4.0, 6.0, 8.0)
DRAWS = 200
POWER_DRAWS = 100
FP_MAX = 0.05
DETECT_MIN = 0.95
REALISED_TOL = 0.10  # realised (component + floor) / floor within 10 % of its target
FWHM_S = 600.0  # s, long AR(2) realisation for the FWHM check
COUNT_LAG_MS = 1.0  # lag of the realised-autocorrelation precondition in tests
STAGE_NULL, STAGE_DETECT, STAGE_POWER, STAGE_FWHM = 1, 2, 3, 4  # seed streams
ROWS_NAME = "calibration_v5_rows.jsonl"
MD_NAME = "calibration_v5.md"


def sigma_for_ratio(c):
    """Lognormal sigma giving a modulated-count variance of c x the Poisson variance."""
    return math.sqrt(math.log(1.0 + c / (R0 * NE * DT)))


def ar1_unit(rng, n, tau_ms):
    """Unit-variance AR(1) sequence with phi = exp(-dt / tau)."""
    phi = math.exp(-DT / (tau_ms * 1e-3))
    eps = rng.standard_normal(n)
    return lfilter([math.sqrt(1.0 - phi * phi)], [1.0, -phi], eps)


def synth_e(rng, tau_ms, c, kept_s=KEPT_S):
    """E spike counts and rate (Hz) for one kept window of a null floor.

    ``tau_ms`` None is the white floor (constant rate R0); otherwise the floor
    is the lognormal AR(1) with variance ratio ``c``. Returns ``(counts, e_rate_Hz)``.
    """
    n_burn = int(round(BURN_S / DT))
    n_kept = int(round(kept_s / DT))
    n = n_burn + n_kept
    if tau_ms is None:
        rate = np.full(n, R0)
    else:
        sig = sigma_for_ratio(c)
        x = ar1_unit(rng, n, tau_ms)
        rate = R0 * np.exp(sig * x - 0.5 * sig * sig)
    rate = rate[n_burn:]
    counts = rng.poisson(NE * DT * rate)
    return counts, counts / (NE * DT)


def ar2_params(centre_hz=MOD_HZ, fwhm_hz=AR2_FWHM_HZ, dt=DT):
    """AR(2) coefficients (a1, a2) of x[n] = a1 x[n-1] + a2 x[n-2] + e[n].

    The poles are r exp(+/- i w0) with w0 = 2 pi f0 dt and r = 1 - pi FWHM dt.
    Near a pole the half-power full width of |H|^2 is 2 (1 - r) rad per sample,
    that is (1 - r) / (pi dt) Hz, so this r gives the requested FWHM. The
    long-realisation test checks it.
    """
    r = 1.0 - math.pi * fwhm_hz * dt
    w0 = 2.0 * math.pi * centre_hz * dt
    return 2.0 * r * math.cos(w0), -r * r


def ar2_unit(rng, n):
    """Unit-innovation narrowband AR(2) sequence (centre MOD_HZ, FWHM AR2_FWHM_HZ)."""
    a1, a2 = ar2_params()
    return lfilter([1.0], [1.0, -a1, -a2], rng.standard_normal(n))


def component_unit(form, rng, kept_s=KEPT_S):
    """Unit-amplitude 35 Hz component on the kept window.

    ``sinusoid``: uniform random phase. ``ar2``: the AR(2) bump after BURN_S.
    """
    n_kept = int(round(kept_s / DT))
    if form == "sinusoid":
        t = np.arange(n_kept) * DT
        return np.sin(2.0 * np.pi * MOD_HZ * t + rng.uniform(0.0, 2.0 * np.pi))
    if form == "ar2":
        n_burn = int(round(BURN_S / DT))
        return ar2_unit(rng, n_burn + n_kept)[n_burn:]
    raise ValueError("unknown component form %r" % (form,))


def measure_fwhm(x, centre_hz=MOD_HZ, seg_s=10.0, half_band_hz=10.0):
    """(FWHM Hz, peak Hz) of the spectral peak near ``centre_hz`` in a long realisation.

    Welch PSD with ``seg_s``-second Hann windows (50 % overlap). A Lorentzian
    plus a constant is fitted by least squares within ``half_band_hz`` of the
    centre, and the FWHM is 2 g. The fit is used because a half-maximum taken
    from the noisy maximum bin reads low (about 4.4 to 4.8 Hz for a 5 Hz AR(2)).
    """
    fs = 1.0 / DT
    nperseg = int(round(seg_s * fs))
    freqs, pxx = ca.welch_spectrum(np.asarray(x, dtype=float), fs, nperseg, nperseg // 2)
    band = np.abs(freqs - centre_hz) <= half_band_hz
    fb = freqs[band]
    yb = pxx[band] / pxx[band].max()

    def lorentz(f, amp, f0, g, base):
        return amp / ((f - f0) ** 2 + g ** 2) + base

    i = int(np.argmax(yb))
    p0 = [(1.0 - float(np.median(yb))) * 2.5 ** 2, fb[i], 2.5, float(np.median(yb))]
    popt, _ = curve_fit(lorentz, fb, yb, p0=p0,
                        bounds=([0.0, centre_hz - 5.0, 0.1, 0.0],
                                [np.inf, centre_hz + 5.0, 20.0, np.inf]),
                        maxfev=20000)
    return float(2.0 * abs(popt[2])), float(popt[1])


def count_acf(counts, lag_ms=COUNT_LAG_MS):
    """Sample autocorrelation of the spike counts at ``lag_ms``."""
    k = int(round(lag_ms * 1e-3 / DT))
    x = np.asarray(counts, dtype=float)
    return float(np.corrcoef(x[:-k], x[k:])[0, 1])


def expected_count_acf(tau_ms, c, lag_ms=COUNT_LAG_MS):
    """Population value of ``count_acf`` for the AR(1) floor (Poisson adds no covariance)."""
    sig2 = sigma_for_ratio(c) ** 2
    rho = math.exp(-lag_ms * 1e-3 / (tau_ms * 1e-3))
    corr_rate = (math.exp(sig2 * rho) - 1.0) / (math.exp(sig2) - 1.0)
    return c / (1.0 + c) * corr_rate


def floors():
    """(label, tau_ms, c) for the 7 null floors; tau None is the white floor."""
    out = [("white Poisson", None, None)]
    for tau in TAUS_MS:
        for c in VAR_RATIOS:
            out.append(("AR(1) tau %g ms, var %gx" % (tau, c), tau, c))
    return out


def psd35(e_rate):
    """Welch PSD of the E rate (scorer settings) at the 35 Hz bin."""
    freqs, pxx = ca.e_welch(e_rate, DT)
    i = int(np.argmin(np.abs(freqs - MOD_HZ)))
    if abs(freqs[i] - MOD_HZ) > 1e-9:
        raise ValueError("no Welch bin at %g Hz" % MOD_HZ)
    return float(pxx[i])


def oscillating(res, theta):
    """Per-draw oscillation call at ``theta`` from the stored peak fields, by the scorer's rule."""
    return np.array([ca.oscillation_call(p, lm, eo, theta)
                     for p, lm, eo in zip(res["pe"], res["local_max"], res["edge_ok"])],
                    dtype=bool)


def detected(res, theta):
    """Detection per draw: oscillation call at ``theta`` and |f_pk - 35| <= FREQ_TOL_HZ."""
    return oscillating(res, theta) & (np.abs(res["f_pk"] - MOD_HZ) <= FREQ_TOL_HZ)


def fp_rates(null_res):
    """False-positive rate of one null floor at each theta in THETA_GRID."""
    return {th: float(oscillating(null_res, th).mean()) for th in THETA_GRID}


def select_theta(fp_table, thetas=THETA_GRID, fp_max=FP_MAX):
    """Smallest theta at which every null floor's FP rate is <= ``fp_max``.

    ``fp_table`` maps floor label to {theta: rate}. Refuses with ValueError when
    no theta qualifies: the fit is then revised before signature, not here.
    """
    for th in sorted(thetas):
        if all(rates[th] <= fp_max + 1e-12 for rates in fp_table.values()):
            return th
    raise ValueError("no theta in %s keeps every null floor at FP <= %g %%"
                     % (tuple(thetas), 100.0 * fp_max))


def _peak_fields(res, d, pk):
    """Store one draw's peak fields (theta is applied later, from these fields)."""
    res["pe"][d] = pk["peak_excess"]
    res["local_max"][d] = pk["local_max"]
    res["edge_ok"][d] = pk["edge_ok"]
    res["f_pk"][d] = pk["peak_freq_Hz"]


def _empty_fields(label, draws):
    return {"label": label, "pe": np.zeros(draws), "local_max": np.zeros(draws, dtype=bool),
            "edge_ok": np.zeros(draws, dtype=bool), "f_pk": np.zeros(draws)}


def run_null(row, label, tau, c, draws):
    """Null floor: peak fields per draw and the mean floor PSD at 35 Hz over the draws."""
    res = _empty_fields(label, draws)
    psd_sum = 0.0
    for d in range(draws):
        _, e_rate = synth_e(np.random.default_rng([STAGE_NULL, row, d]), tau, c)
        _peak_fields(res, d, ca.spectral_peak(e_rate, DT))
        psd_sum += psd35(e_rate)
    res["floor_psd35"] = psd_sum / draws
    return res


def component_draws(row, tau, c, form, stage, key, draws):
    """(floor E rate, unit component) per draw.

    The floor is null draw ``d`` of its row (the seeds of ``run_null``), so each
    detection draw pairs with the null draw that sets the floor PSD. The component
    seeds depend only on (stage, key, draw).
    """
    pairs = []
    for d in range(draws):
        _, e_floor = synth_e(np.random.default_rng([STAGE_NULL, row, d]), tau, c)
        unit = component_unit(form, np.random.default_rng([stage, *key, d]))
        pairs.append((e_floor, unit))
    return pairs


def run_component(row, tau, c, form, pe_target, floor_psd35, draws, stage, key, peaks=True):
    """Floor plus a 35 Hz component whose mean PSD at 35 Hz is (pe_target - 1) x floor_psd35.

    ``floor_psd35`` is the floor's mean PSD at 35 Hz over the 200 null draws (the
    spec's reference). ``realised_pe`` is the mean PSD at 35 Hz of (component +
    floor) over that reference. ``peaks=False`` skips the spectral-peak fits
    (scaling checks only).
    """
    pairs = component_draws(row, tau, c, form, stage, key, draws)
    unit_psd = float(np.mean([psd35(u) for _, u in pairs]))
    scale = math.sqrt((pe_target - 1.0) * floor_psd35 / unit_psd)
    res = _empty_fields(form, draws)
    res.update({"form": form, "pe_target": pe_target, "scale": scale})
    det_sum = 0.0
    comp_sum = 0.0
    for d, (e_floor, unit) in enumerate(pairs):
        comp = scale * unit
        e_det = e_floor + comp
        if peaks:
            _peak_fields(res, d, ca.spectral_peak(e_det, DT))
        det_sum += psd35(e_det)
        comp_sum += psd35(comp)
    res["realised_pe"] = det_sum / draws / floor_psd35
    # Measured mean component PSD at 35 Hz over the floor PSD: the scale check (= pe_target - 1).
    res["comp_over_floor"] = comp_sum / draws / floor_psd35
    return res


def component_record(label, res, theta, draws):
    """Summary of one component row at ``theta``: hits, rate, realised check, median f_pk."""
    hits = int(detected(res, theta).sum())
    rec = {"floor": label, "form": res["form"], "pe_target": res["pe_target"],
           "scale": res["scale"], "comp_over_floor": res["comp_over_floor"],
           "realised_pe": res["realised_pe"], "hits": hits,
           "draws": draws, "rate": hits / draws,
           "median_f_pk": float(np.median(res["f_pk"]))}
    rec["realised_ok"] = bool(abs(rec["realised_pe"] / rec["pe_target"] - 1.0) <= REALISED_TOL)
    return rec


def render_md(meta, gate, null, fp_table, theta, detect, power, fwhm):
    """calibration_v5.md: the tables the prereg cites, with the gate line."""
    lines = ["# CAMINOS calibration gate v5", "",
             "Spec: `artifacts/etudes/caminos/PREREG_P3.md`, \"Calibration gate (v5)\". "
             "Synthetic only; no network data.", "", meta, "", "Gate: `%s`" % gate, "",
             "## 1. Null floors: FP rate by theta (hits/draws)", "",
             "| Null floor | Floor PSD at 35 Hz | FP theta 4 | FP theta 5 | FP theta 6 | FP theta 8 |",
             "|---|---|---|---|---|---|"]
    for r in null:
        fp = fp_table[r["label"]]
        draws = len(r["pe"])
        cells = ["%.3f (%d/%d)" % (fp[th], round(fp[th] * draws), draws) for th in THETA_GRID]
        lines.append("| %s | %.4g | %s |" % (r["label"], r["floor_psd35"], " | ".join(cells)))
    lines.append("")
    if theta is None:
        lines.append("No theta in %s keeps every floor at FP <= 5 %%. Detection and power are "
                     "not run; the fit is revised before signature." % (THETA_GRID,))
        return "\n".join(lines) + "\n"
    lines += ["theta = %g: the smallest with every floor at FP <= 5 %%." % theta, "",
              "## 2. Detection rows (gated): true PE = 2 theta = %g" % (2 * theta), "",
              "| Null floor | Form | Scale | Component / floor (target 2 theta - 1) | Realised PE "
              "| Within 10 % | Detection (hits/draws) | Median f_pk (Hz) | Pass |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in detect:
        lines.append("| %s | %s | %.4g | %.4f | %.3f | %s | %.3f (%d/%d) | %.2f | %s |" % (
            r["floor"], r["form"], r["scale"], r["comp_over_floor"], r["realised_pe"],
            "yes" if r["realised_ok"] else "no", r["rate"], r["hits"], r["draws"],
            r["median_f_pk"], "PASS" if r["pass"] else "FAIL"))
    n_power = power[0]["draws"] if power else 0
    lines += ["", "## 3. Power curve (not gated): detection at theta, hits/draws", "",
              "Power rows use the first %d of each floor's null draws; their realised PE "
              "includes that subset's floor sampling. Not gated." % n_power, "",
              "| Null floor | Form | PE 2 | PE 3 | PE 4 | PE 6 | PE 8 |",
              "|---|---|---|---|---|---|---|"]
    for i in range(0, len(power), len(POWER_PE)):
        grp = power[i:i + len(POWER_PE)]
        cells = ["%.2f (%d/%d)" % (g["rate"], g["hits"], g["draws"]) for g in grp]
        lines.append("| %s | %s | %s |" % (grp[0]["floor"], grp[0]["form"], " | ".join(cells)))
    lines += ["", "## 4. AR(2) component FWHM", "",
              "Poles r exp(+/- i 2 pi 35 Hz dt) with r = 1 - pi 5 Hz dt, dt = 1e-4 s.",
              "Measured FWHM: %.3f Hz (fitted peak %.2f Hz) on a %g s realisation; target 5 Hz."
              % (fwhm[0], fwhm[1], FWHM_S), ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="v5 calibration gate for the CAMINOS criterion "
                                             "(synthetic only).")
    ap.add_argument("--out", required=True, help="directory for the rows JSONL and calibration_v5.md")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--power-draws", type=int, default=POWER_DRAWS)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if (args.draws, args.power_draws) == (DRAWS, POWER_DRAWS) else "SMOKE, not a gate: "
    meta = ("NE=%d, dt=%g s, kept %g s, R0=%g Hz; draws=%d (power %d); theta grid %s; "
            "FP <= %g %%; detection >= %g %% with |f_pk - 35| <= %g Hz; realised within %g %%."
            % (NE, DT, KEPT_S, R0, args.draws, args.power_draws, THETA_GRID,
               100 * FP_MAX, 100 * DETECT_MIN, FREQ_TOL_HZ, 100 * REALISED_TOL))
    print(tag + meta, flush=True)
    with open(out / ROWS_NAME, "w") as rows:
        def emit(rec):
            rows.write(json.dumps(rec) + "\n")
            rows.flush()
            print(json.dumps(rec), flush=True)

        null = []
        for i, (label, tau, c) in enumerate(floors()):
            res = run_null(i, label, tau, c, args.draws)
            null.append(res)
            emit({"stage": "null", "floor": label, "draws": args.draws,
                  "floor_psd35": res["floor_psd35"],
                  "fp": {"%g" % th: rate for th, rate in fp_rates(res).items()}})
        fp_table = {r["label"]: fp_rates(r) for r in null}
        try:
            theta = select_theta(fp_table)
        except ValueError as exc:
            gate = "%sGate v5: FAIL, %s" % (tag, exc)
            (out / MD_NAME).write_text(render_md(meta, gate, null, fp_table, None, [], [], None))
            print(gate, flush=True)
            return 1

        detect = []
        for i, (label, tau, c) in enumerate(floors()):
            for j, form in enumerate(FORMS):
                res = run_component(i, tau, c, form, 2.0 * theta, null[i]["floor_psd35"],
                                    args.draws, STAGE_DETECT, (i, j))
                rec = component_record(label, res, theta, args.draws)
                rec["stage"] = "detect"
                rec["pass"] = bool(rec["rate"] >= DETECT_MIN - 1e-12 and rec["realised_ok"])
                detect.append(rec)
                emit(rec)
        power = []
        for i, (label, tau, c) in enumerate(floors()):
            for j, form in enumerate(FORMS):
                for k, pe in enumerate(POWER_PE):
                    res = run_component(i, tau, c, form, pe, null[i]["floor_psd35"],
                                        args.power_draws, STAGE_POWER, (i, j, k))
                    rec = component_record(label, res, theta, args.power_draws)
                    rec["stage"] = "power"
                    power.append(rec)
                    emit(rec)
        x = ar2_unit(np.random.default_rng([STAGE_FWHM]), int(round(FWHM_S / DT)))
        fwhm = measure_fwhm(x[int(round(BURN_S / DT)):])
        emit({"stage": "fwhm", "fwhm_hz": fwhm[0], "peak_hz": fwhm[1],
              "target_hz": AR2_FWHM_HZ, "seconds": FWHM_S})

    failed = [r for r in detect if not r["pass"]]
    if failed:
        reasons = ["%s, %s: detection %.3f, realised PE %.3f (target %g)" % (
            r["floor"], r["form"], r["rate"], r["realised_pe"], r["pe_target"]) for r in failed]
        gate = "%sGate v5: FAIL (theta = %g; %d of %d detection rows fail): %s" % (
            tag, theta, len(failed), len(detect), "; ".join(reasons))
    else:
        gate = ("%sGate v5: PASS (theta = %g; %d of %d detection rows >= 95 %%, "
                "realised within 10 %%)" % (tag, theta, len(detect), len(detect)))
    (out / MD_NAME).write_text(render_md(meta, gate, null, fp_table, theta, detect, power, fwhm))
    print(gate, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
