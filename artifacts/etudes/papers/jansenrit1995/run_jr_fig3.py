"""Jansen-Rit 1995 Fig. 3 C sweep on the jaxfne emitter: per C, spectrum peak, oscillation index, EEG level.

Usage: python run_jr_fig3.py --seeds 0 1 2 --out DIR [--c 68 128 135 270 675 1350] [--t-s 10] [--dt-s 1e-4]
       [--hold-s 2e-3] [--discard-s 1]
Writes DIR/fig3_seed<S>.json per seed. Exit 0 on success, 2 on bad arguments.

Protocol (SOURCE_P1.md): standard set A 3.25 mV, B 22 mV, a 100/s, b 50/s, e0 2.5/s, v0 6 mV, r 0.56/mV,
C1..C4 = C, 0.8C, 0.25C, 0.25C; p(t) uniform 120-320 pulses/s, one draw held per ``hold_s`` (paper: unknown;
2 ms is the Fig. 3 sample interval, 500 units = 1 s, inferred). Fixed-step RK4 at ``dt_s`` (paper: RKF45).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import jaxfne
from jaxfne.emitters_jansen_rit import simulate_jansen_rit

STANDARD = {"A": 3.25, "B": 22.0, "a": 100.0, "b": 50.0, "e0": 2.5, "v0": 6.0, "r": 0.56}
FIG3_C = (68.0, 128.0, 135.0, 270.0, 675.0, 1350.0)
P_LOW, P_HIGH = 120.0, 320.0
BAND_HZ = (1.0, 30.0)
PEAK_HALF_WIDTH_HZ = 1.0


def drive(seed: int, n_steps: int, dt_s: float, hold_s: float) -> np.ndarray:
    """Uniform 120-320 pulses/s, one draw held per ``hold_s``."""
    per = max(1, int(round(hold_s / dt_s)))
    n_draw = -(-n_steps // per)
    rng = np.random.default_rng(seed)
    return np.repeat(rng.uniform(P_LOW, P_HIGH, n_draw), per)[:n_steps]


def spectrum_metrics(eeg: np.ndarray, dt_s: float) -> dict:
    """Peak frequency in BAND_HZ, oscillation index (power within ±1 Hz of the peak over band power)."""
    x = eeg - eeg.mean()
    seg = int(round(4.0 / dt_s))
    step = seg // 2
    win = np.hanning(seg)
    acc = None
    for s in range(0, len(x) - seg + 1, step):
        p = np.abs(np.fft.rfft(x[s:s + seg] * win)) ** 2
        acc = p if acc is None else acc + p
    f = np.fft.rfftfreq(seg, dt_s)
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    fb, pb = f[band], acc[band]
    k = int(np.argmax(pb))
    f_peak = float(fb[k])
    near = np.abs(fb - f_peak) <= PEAK_HALF_WIDTH_HZ
    total = float(pb.sum())
    return {"f_peak_hz": f_peak, "osc_index": float(pb[near].sum() / total) if total > 0 else 0.0}


def run_c(c: float, seed: int, t_s: float, dt_s: float, hold_s: float, discard_s: float) -> dict:
    n = int(round(t_s / dt_s))
    out = simulate_jansen_rit(
        STANDARD["A"], STANDARD["B"], STANDARD["a"], STANDARD["b"],
        c, 0.8 * c, 0.25 * c, 0.25 * c,
        STANDARD["e0"], STANDARD["v0"], STANDARD["r"], dt_s, n, drive(seed, n, dt_s, hold_s),
    )
    eeg = out["eeg"][int(round(discard_s / dt_s)):]
    if not np.all(np.isfinite(eeg)):
        return {"C": c, "finite": False}
    m = spectrum_metrics(eeg, dt_s)
    return {"C": c, "finite": True, "eeg_mean_mv": float(eeg.mean()), "eeg_std_mv": float(eeg.std()),
            "eeg_min_mv": float(eeg.min()), "eeg_max_mv": float(eeg.max()), **m}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c", type=float, nargs="+", default=list(FIG3_C))
    ap.add_argument("--t-s", type=float, default=10.0)
    ap.add_argument("--dt-s", type=float, default=1e-4)
    ap.add_argument("--hold-s", type=float, default=2e-3)
    ap.add_argument("--discard-s", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.discard_s + 4.0 > a.t_s or a.dt_s <= 0 or a.hold_s <= 0:
        print("need t_s >= discard_s + 4 and positive dt_s, hold_s", file=sys.stderr)
        return 2
    jaxfne.enable_x64()
    a.out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        rows = [run_c(c, seed, a.t_s, a.dt_s, a.hold_s, a.discard_s) for c in a.c]
        rec = {"seed": seed, "t_s": a.t_s, "dt_s": a.dt_s, "hold_s": a.hold_s, "discard_s": a.discard_s,
               "standard": STANDARD, "rows": rows}
        (a.out / f"fig3_seed{seed}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        for r in rows:
            print(seed, json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
