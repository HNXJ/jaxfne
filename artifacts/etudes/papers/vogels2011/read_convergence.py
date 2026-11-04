"""Convergence read of a segmented Vogels pilot: rate course, sliding slopes, exponential asymptote, cell spread.

Usage: python read_convergence.py <pilot dir> [t0_s]
Reads pilot_receipt.json (`e_rate_per_s_hz`, R-E) and spikes_seed<S>_segB.npz (the saved last-1 s window).
Slope: least squares over sliding 10 s windows. Fit: r(t) = r_inf + A exp(-t/tau) on t >= t0_s (default 20).
Exit 0, or 2 on usage.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

WINDOW_S = 10


def _approach(t, rinf, a, tau):
    return rinf + a * np.exp(-t / tau)


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    d = Path(sys.argv[1])
    t0 = float(sys.argv[2]) if len(sys.argv) == 3 else 20.0
    rec = json.loads((d / "pilot_receipt.json").read_text(encoding="utf-8"))
    r = np.asarray(rec["e_rate_per_s_hz"], dtype=float)
    t = np.arange(1, r.size + 1, dtype=float)
    print("T %d s; R-E z %.3f Hz e %.3f; O-DROP ratio %.2f" % (r.size, rec["R-E"]["z_hz"], rec["R-E"]["e"],
                                                             rec["O-DROP"]["ratio"]))
    first = None
    for end in range(WINDOW_S, r.size + 1):
        slope = np.polyfit(np.arange(WINDOW_S), r[end - WINDOW_S:end], 1)[0]
        if end % 20 == 0:
            print("10 s window ending %4d s: mean %.3f Hz, slope %+.4f Hz/s" % (end, r[end - WINDOW_S:end].mean(), slope))
        if first is None and abs(slope) < 0.1:
            first = end
    print("first 10 s window with |slope| < 0.1 Hz/s ends at %s s" % first)
    sel = t >= t0
    p, cov = curve_fit(_approach, t[sel], r[sel], p0=(float(r[-1]), float(r[sel][0] - r[-1]), 30.0), maxfev=20000)
    sd = np.sqrt(np.diag(cov))
    print("fit t >= %g s: r_inf %.3f +- %.3f Hz, tau %.1f +- %.1f s, residual sd %.3f Hz"
          % (t0, p[0], sd[0], p[2], sd[2], float((r[sel] - _approach(t[sel], *p)).std())))
    seed = int(rec.get("seed", 0))
    raster = d / ("spikes_seed%d_segB.npz" % seed)
    if raster.exists():
        sp = np.load(raster)["spikes"]
        dur = sp.shape[0] * float(rec["dt_ms"]) / 1000.0
        n_e = int(round(0.8 * sp.shape[1]))  # E columns first, NE_FRAC 0.8
        rates = sp[:, :n_e].sum(axis=0) / dur
        q = np.percentile(rates, [10, 50, 90])
        print("last %.1f s, E cells: mean %.3f, 10/50/90 %% %.1f/%.1f/%.1f Hz, top 10 %% share %.3f"
              % (dur, rates.mean(), q[0], q[1], q[2], np.sort(rates)[::-1][: n_e // 10].sum() / rates.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
