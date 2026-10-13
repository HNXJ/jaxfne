"""v4 calibration gate for the CAMINOS oscillation criterion (PREREG_P3.md).

Synthetic signals only; no network data. Each row draws ``--draws`` realisations
of the E population spike counts (3 s kept window, NE = 8000, dt = 1e-4 s,
baseline rate R0 = 1.2 Hz) and applies ``caminos_analysis.spectral_peak``, the
function the scorer uses.

Constructions (the packet fixes the variance ratios and taus; these are ours):
* AR(1) floor: x(t) is unit-variance AR(1) with phi = exp(-dt / tau). The rate
  is R0 * exp(sigma * x - sigma**2 / 2), a lognormal with mean R0, so it stays
  positive at 8x variance. sigma**2 = ln(1 + c / (R0 * NE * dt)), which makes
  the variance of the modulated expected count c times the Poisson count
  variance R0 * NE * dt.
* Modulation: rate *= 1 + 0.1 * sin(2 pi 35 t), on the white or AR(1) floor.
* Rows: white Poisson (FP); 6 AR(1) floors (FP); white + 35 Hz (detection);
  6 AR(1) floors + 35 Hz (detection). Gate: FP <= 5 %, detection >= 95 %.
  The threshold is not tuned here; a failing row is reported.

Run: python calibrate_oscillation.py [--draws N]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caminos_analysis as ca  # noqa: E402

NE = 8000
DT = 1e-4  # s
R0 = 1.2  # Hz, E baseline rate
BURN_S = 1.0  # s, discarded so the AR(1) floor is stationary
KEPT_S = 3.0  # s, prereg kept window
MOD_HZ = 35.0
MOD_DEPTH = 0.1
TAUS_MS = (8.0, 30.0)
VAR_RATIOS = (0.5, 2.0, 8.0)
DRAWS = 200
FP_MAX = 0.05
DETECT_MIN = 0.95
COUNT_LAG_MS = 1.0  # lag of the realised-autocorrelation precondition in tests


def sigma_for_ratio(c):
    """Lognormal sigma giving a modulated-count variance of c x the Poisson variance."""
    return math.sqrt(math.log(1.0 + c / (R0 * NE * DT)))


def ar1_unit(rng, n, tau_ms):
    """Unit-variance AR(1) sequence with phi = exp(-dt / tau)."""
    phi = math.exp(-DT / (tau_ms * 1e-3))
    eps = rng.standard_normal(n)
    return lfilter([math.sqrt(1.0 - phi * phi)], [1.0, -phi], eps)


def synth_e(rng, tau_ms, c, mod, kept_s=KEPT_S):
    """E spike counts and rate (Hz) for one kept window (3 s by default).

    ``tau_ms`` None is the white floor (constant rate R0); otherwise the floor
    is the lognormal AR(1) with variance ratio ``c``. ``mod`` adds the 35 Hz
    depth-0.1 modulation. Returns ``(counts, e_rate_Hz)``.
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
    if mod:
        t = np.arange(n_kept) * DT
        rate = rate * (1.0 + MOD_DEPTH * np.sin(2.0 * np.pi * MOD_HZ * t))
    counts = rng.poisson(NE * DT * rate)
    return counts, counts / (NE * DT)


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


def gate_rows():
    """(label, kind, tau_ms, c, mod) for the 14 gate rows; kind is FP or DET."""
    rows = [("white Poisson", "FP", None, None, False)]
    for tau in TAUS_MS:
        for c in VAR_RATIOS:
            rows.append(("AR(1) tau %g ms, var %gx" % (tau, c), "FP", tau, c, False))
    rows.append(("white + 35 Hz depth 0.1", "DET", None, None, True))
    for tau in TAUS_MS:
        for c in VAR_RATIOS:
            rows.append(("AR(1) tau %g ms, var %gx + 35 Hz depth 0.1" % (tau, c),
                         "DET", tau, c, True))
    return rows


def run_row(index, kind, tau, c, mod, draws):
    """Draws for one row: count of oscillation calls, median peak excess, pass flag."""
    osc = np.zeros(draws, dtype=bool)
    pes = np.zeros(draws)
    for d in range(draws):
        rng = np.random.default_rng([index, d])
        _, e_rate = synth_e(rng, tau, c, mod)
        pk = ca.spectral_peak(e_rate, DT)
        osc[d] = pk["oscillates"]
        pes[d] = pk["peak_excess"]
    hits = int(osc.sum())
    rate = hits / draws
    ok = rate <= FP_MAX if kind == "FP" else rate >= DETECT_MIN
    return {"hits": hits, "rate": rate, "median_pe": float(np.median(pes)), "ok": bool(ok)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="v4 calibration gate (synthetic only).")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    print("Synthetic calibration, no network data. NE=%d, dt=%g s, kept %g s, "
          "R0=%g Hz, draws=%d per row." % (NE, DT, KEPT_S, R0, args.draws))
    print("sigma per variance ratio: " + ", ".join(
        "%gx = %.4f" % (c, sigma_for_ratio(c)) for c in VAR_RATIOS))
    print()
    print("| Row | FP or detection rate (hits/draws) | Median PE | Gate | Result |")
    print("|---|---|---|---|---|")
    all_ok = True
    for i, (label, kind, tau, c, mod) in enumerate(gate_rows()):
        r = run_row(i, kind, tau, c, mod, args.draws)
        all_ok = all_ok and r["ok"]
        gate = "FP <= 5 %" if kind == "FP" else "detection >= 95 %"
        print("| %s | %.3f (%d/%d) | %.2f | %s | %s |" % (
            label, r["rate"], r["hits"], args.draws, r["median_pe"], gate,
            "PASS" if r["ok"] else "FAIL"))
    print()
    print("Gate: %s" % ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
