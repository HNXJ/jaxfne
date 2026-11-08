#!/usr/bin/env python3
"""Phase diagnosis for the reader STDP leg (O-60), read-only.

Loads a saved run JSON with a raster (one P trial), computes:
  - reader first-spike latency per carrier cycle,
  - reader spike count per cycle,
  - kernel-weighted STDP balance per afferent class (pattern vs carrier-only),
    dw = A+ * sum_{post>pre} exp(-(post-pre)/tau+) - A- * sum_{pre>post} exp(-(pre-post)/tau-),
    summed over reader spikes (all readers pooled, weights initial).

No simulation, no jaxfne import.
"""
import json
import sys

import numpy as np

DT_MS = 0.05
N_A = 100
N_I = 20
K_PAT = 20
TAU_P = 17.0
TAU_M = 34.0
A_P = 0.005
A_M = 0.00525


def main(path):
    with open(path) as f:
        d = json.load(f)
    r = d["raster_first"]
    spk = np.asarray(r["spikes"], float)  # bins x neurons
    peaks = np.asarray(r["peaks_ms"], float)
    dur = spk.shape[0] * DT_MS
    t = np.arange(spk.shape[0]) * DT_MS
    r_spk = spk[:, N_A + N_I:]  # readers
    a_spk = spk[:, :N_A]
    pat_spk = a_spk[:, :K_PAT]
    car_spk = a_spk[:, K_PAT:]

    r_times = [t[np.flatnonzero(r_spk[:, j])] for j in range(r_spk.shape[1])]

    # per-cycle reader latency and count (first spike of any reader)
    lat = []
    cnt = []
    all_r_phases = []
    for k in range(len(peaks)):
        p0 = peaks[k]
        p1 = peaks[k + 1] if k + 1 < len(peaks) else dur
        if p0 < 50.0:
            continue
        m = (t >= p0) & (t < p1)
        c = int(r_spk[m].sum())
        firsts = []
        r_phases = []
        for rt in r_times:
            rin = rt[(rt >= p0) & (rt < p1)]
            if rin.size:
                firsts.append(rin[0] - p0)
                r_phases.extend((rin - p0).tolist())
        lat.append(min(firsts) if firsts else float("nan"))
        cnt.append(c)
        all_r_phases.extend(r_phases)
    lat = np.asarray(lat)
    cnt = np.asarray(cnt)
    print(f"cycles analysed: {len(lat)} (burn-in cut)")
    print(f"reader spikes/cycle: mean {cnt.mean():.2f} "
          f"(median {np.median(cnt):.0f}, max {cnt.max()})")
    print(f"reader first-spike latency after peak (ms): "
          f"median {np.nanmedian(lat):.2f}, mean {np.nanmean(lat):.2f}, "
          f"p10 {np.nanpercentile(lat, 10):.2f}, p90 {np.nanpercentile(lat, 90):.2f}, "
          f"silent cycles {int(np.isnan(lat).sum())}")
    rp = np.asarray(all_r_phases)
    rhist, _ = np.histogram(rp, bins=np.arange(0.0, 25.01, 1.0))
    print("reader spike phase after peak (ms, 1 ms bins):")
    print("  " + " ".join(f"{v:d}" for v in rhist))

    # per-afferent STDP balance vs pooled reader spikes
    def balance(mask):
        vals = []
        for i in np.flatnonzero(mask):
            pre = t[np.flatnonzero(a_spk[:, i])]
            posts = np.concatenate(r_times)
            posts.sort()
            dw = 0.0
            for p in pre:
                # LTP: all post spikes within tau+ AFTER the pre spike
                hi = np.searchsorted(posts, p + TAU_P, side="right")
                lo = np.searchsorted(posts, p, side="right")
                if hi > lo:
                    dw += A_P * float(np.exp(-(posts[lo:hi] - p) / TAU_P).sum())
                # LTD: all post spikes within tau- BEFORE the pre spike
                lo2 = np.searchsorted(posts, p - TAU_M, side="left")
                hi2 = np.searchsorted(posts, p, side="left")
                if hi2 > lo2:
                    dw -= A_M * float(np.exp(-(p - posts[lo2:hi2]) / TAU_M).sum())
            vals.append(dw)
        return np.asarray(vals)

    pat = balance(np.concatenate([np.ones(K_PAT), np.zeros(N_A - K_PAT)]).astype(bool))
    car = balance(np.concatenate([np.zeros(K_PAT), np.ones(N_A - K_PAT)]).astype(bool))
    print(f"pattern afferents  (n={pat.size}): mean dw {pat.mean():+.5f} "
          f"(min {pat.min():+.5f}, max {pat.max():+.5f}, neg frac "
          f"{float((pat < 0).mean()):.2f})")
    print(f"carrier afferents  (n={car.size}): mean dw {car.mean():+.5f} "
          f"(min {car.min():+.5f}, max {car.max():+.5f}, neg frac "
          f"{float((car < 0).mean()):.2f})")

    # pattern spike phase relative to peak, and relative to the reader's first spike
    phases = []
    after = 0
    before = 0
    for i in range(K_PAT):
        pt = t[np.flatnonzero(pat_spk[:, i])]
        # pattern bins: the afferent's spikes that are NOT poisson cannot be
        # separated here; take all spikes of pattern afferents in each cycle
        # that lie within [p0+d_min-3*sigma, p0+8] as pattern candidates is
        # ambiguous, so report the full phase histogram of pattern-afferent
        # spikes instead
        for x in pt:
            k = int(np.searchsorted(peaks, x, side="right") - 1)
            if k < 0 or x - peaks[k] > 25.0:
                continue
            rel = x - peaks[k]
            phases.append(rel)
    phases = np.asarray(phases)
    hist, edges = np.histogram(phases, bins=np.arange(0.0, 25.01, 1.0))
    print("pattern-afferent spike phase after peak (ms, 1 ms bins):")
    print("  " + " ".join(f"{v:d}" for v in hist))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "out/smoke.json"))