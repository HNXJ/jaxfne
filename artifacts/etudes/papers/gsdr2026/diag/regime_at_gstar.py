"""GSDR 2026 regime diagnostic at g*: one stim trial, CALIBRATION.md §6.2 quantities.

At the fitted g* (in particular the ``--morphology reference`` fit's g*,
500 generations) this runs one stim trial through the lane's trial function
(``run_gsdr.run_trial``) and writes the §6.2 quantities: gamma cycles and
window (period from the spike-trace PSD peak), per-cycle E participation
mean/std/min/max, E rate quantiles, fraction silent, fraction above 10 Hz,
stim-window E mean, kappa over [500, 1000] ms — plus k_fit (the fit's
``kappa_of_onsets`` on the same window) and spikes-per-third as in §6.11.
Participation comes from ``diag_participation.cycle_participation``; no
simulation lives here beyond the single ``run_trial`` call. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN_DIR = _HERE.parent / "run"

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_RUN_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_DIR))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_gsdr as rg  # noqa: E402
from diag_participation import cycle_participation  # noqa: E402

T_L_MS, T_R_MS = 500.0, 1000.0
THIRD_EDGES_MS = (0.0, 500.0, 1000.0, 1500.0)
RATE_HI_HZ = 10.0


def window_indices(n_steps, t_l_ms, t_r_ms, dt_ms=rg.DT_MS):
    """Clamped [i0, i1) sample indices for ``[t_l, t_r)`` ms.

    Same convention as ``run_gsdr.window_rates_hz``: ``int(t / dt)``
    truncated, clamped into the trace.
    """
    i0 = max(0, min(int(t_l_ms / float(dt_ms)), n_steps - 2))
    i1 = max(i0 + 2, min(int(t_r_ms / float(dt_ms)), n_steps - 1))
    return i0, i1


def rate_quantiles(rates):
    """min / p10 / p25 / median / p75 / p90 / max of per-cell rates (Hz)."""
    import numpy as np

    r = np.asarray(rates, dtype=np.float64)
    q = np.quantile(r, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {"min": float(q[0]), "p10": float(q[1]), "p25": float(q[2]),
            "median": float(q[3]), "p75": float(q[4]), "p90": float(q[5]),
            "max": float(q[6])}


def thirds_counts(onsets, n_steps_v, dt_ms=rg.DT_MS):
    """Total onset counts per 500 ms third over the full trial (all cells)."""
    import numpy as np

    b = np.asarray(onsets, dtype=bool)
    edges = [max(0, min(int(t / float(dt_ms)), n_steps_v - 2))
             for t in THIRD_EDGES_MS]
    return [int(b[edges[i]:edges[i + 1], :].sum()) for i in range(3)]


def spike_psd_peak_hz(v, dt_ms=rg.DT_MS, t_l_ms=T_L_MS, t_r_ms=T_R_MS):
    """Spike-trace PSD peak (Hz) over the window + gamma period (ms)."""
    import numpy as np

    psd = np.asarray(rg.spike_trace_psd(v, dt_ms=dt_ms, t_l_ms=t_l_ms,
                                        t_r_ms=t_r_ms))
    grid = np.asarray(rg.target_grid(), dtype=np.float64)
    peak = float(grid[int(np.argmax(psd))])
    if not peak > 0:
        raise ValueError("spike PSD peak is not positive: %r" % (peak,))
    return peak, 1000.0 / peak, [float(x) for x in psd]


def regime_from_onsets(onsets, *, dt_ms, t_l_ms, t_r_ms, period_ms, n_e):
    """§6.2 quantities from a binary onset matrix (time, cells).

    E cells are the first ``n_e`` columns (the lane builds E first,
    ``run_gsdr.build_spec``). Participation cycles come from
    ``diag_participation.cycle_participation`` on the E slice; kappa
    (k_fit) is the fit's ``kappa_of_onsets`` over all cells on the
    same window.
    """
    import numpy as np

    b = np.asarray(onsets, dtype=bool)
    if b.ndim != 2 or b.shape[1] < int(n_e):
        raise ValueError("onsets must be (time, cells>=n_e), got %s"
                         % (b.shape,))
    part = cycle_participation(np.ascontiguousarray(b[:, :int(n_e)]),
                               dt_ms=dt_ms, t_l_ms=t_l_ms, t_r_ms=t_r_ms,
                               period_ms=period_ms)
    participation = np.asarray(part["participation"], dtype=np.float64)
    n_steps = b.shape[0]
    i0, i1 = window_indices(n_steps, t_l_ms, t_r_ms, dt_ms)
    dur_s = (i1 - i0) * float(dt_ms) / 1000.0
    e_rates = b[i0:i1, :int(n_e)].sum(axis=0) / dur_s
    e_rates = np.asarray(e_rates, dtype=np.float64)
    k_fit = float(rg.kappa_of_onsets(b[i0:i1, :], dt_ms=dt_ms))
    return {
        "window_ms": [float(t_l_ms), float(t_r_ms)],
        "gamma_period_ms": float(period_ms),
        "n_gamma_cycles": int(part["n_cycles"]),
        "participation_mean": float(participation.mean()),
        "participation_std": float(participation.std()),
        "participation_min": float(participation.min()),
        "participation_max": float(participation.max()),
        "participation_per_cycle": [float(x) for x in participation],
        "e_rates_hz": rate_quantiles(e_rates),
        "e_frac_silent": float((e_rates == 0).mean()),
        "e_frac_above_10hz": float((e_rates > RATE_HI_HZ).mean()),
        "e_stim_window_mean_hz": float(e_rates.mean()),
        "kappa_500_1000": k_fit,
        "k_fit": k_fit,
    }


def regime_from_v(v, dt_ms=rg.DT_MS, n_e=rg.N_E):
    """Full diagnostic from a V trace: PSD peak/period + §6.2 + thirds."""
    import numpy as np

    v = np.asarray(v, dtype=np.float32)
    onsets = rg.detect_spikes(v)
    peak, period_ms, _ = spike_psd_peak_hz(v, dt_ms=dt_ms)
    rec = regime_from_onsets(onsets, dt_ms=dt_ms, t_l_ms=T_L_MS,
                             t_r_ms=T_R_MS, period_ms=period_ms, n_e=n_e)
    rec["spike_psd_peak_hz"] = peak
    rec["spikes_per_third"] = thirds_counts(onsets, v.shape[0], dt_ms)
    return rec


def run_diag(gstar, *, stim_amp, seed, morphology):
    """One stim trial at g* through ``run_gsdr.run_trial`` + diagnostic."""
    import numpy as np

    import jax

    backend = str(jax.default_backend())
    if backend != "cpu":
        raise SystemExit("error: CPU only, backend is %s" % backend)
    g = np.asarray(gstar, dtype=np.float32)
    expect = (rg.N_AMPA_EDGES + rg.N_GABAA_GLOBAL_EDGES
              + rg.N_GABAA_LOCAL_EDGES)
    if g.shape != (expect,):
        raise ValueError("gstar must hold %d reference-uS values, got %s"
                         % (expect, g.shape,))
    t0 = time.time()
    result, blobs = rg.run_trial(float(stim_amp), g_seed=int(seed),
                                 ou_seed=int(seed), g_override=g,
                                 morphology=morphology)
    trial_wall_s = time.time() - t0
    v = np.asarray(blobs["v"], dtype=np.float32)
    rec = regime_from_v(v)
    rec.update({
        "stim_amp": float(stim_amp),
        "seed": int(seed),
        "morphology": morphology,
        "backend": backend,
        "git_sha": rg.git_sha(),
        "mean_rate_hz": float(result["mean_rate_hz"]),
        "active_frac": float((np.asarray(result["rates_hz"]) > 0).mean()),
        "trial_wall_s": float(result["wall_s"]),
        "diag_wall_s": float(trial_wall_s),
    })
    return rec


def build_parser():
    ap = argparse.ArgumentParser(
        description="GSDR regime diagnostic at g* (one stim trial, CPU).")
    ap.add_argument("--gstar", required=True,
                    help="path to a g* .npy (2230 reference-uS values)")
    ap.add_argument("--morphology", choices=("single", "reference"),
                    default="reference")
    ap.add_argument("--stim", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output JSON path")
    return ap


def main(argv=None):
    import numpy as np

    args = build_parser().parse_args(argv)
    with open(args.gstar, "rb") as f:
        gstar = np.asarray(np.load(f), dtype=np.float32)
    rec = run_diag(gstar, stim_amp=args.stim, seed=args.seed,
                   morphology=args.morphology)
    rec["gstar"] = str(args.gstar)
    out = Path(args.out)
    if out.parent and str(out.parent):
        out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit("error: refusing to overwrite %s" % out)
    tmp = out.parent / (out.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    tmp.replace(out)
    print("morphology %(m)s stim %(s)g seed %(d)d: peak %(p).2f Hz "
          "period %(per).2f ms cycles %(c)d part %(pm).4f+/-%(ps).4f "
          "[%(pmin).2f,%(pmax).2f] Emean %(e).2f Hz silent %(sil).2f "
          ">10Hz %(hi).2f k_fit %(k).4f thirds %(t)s wall %(w).1fs -> %(o)s"
          % {"m": rec["morphology"], "s": rec["stim_amp"],
             "d": rec["seed"], "p": rec["spike_psd_peak_hz"],
             "per": rec["gamma_period_ms"], "c": rec["n_gamma_cycles"],
             "pm": rec["participation_mean"], "ps": rec["participation_std"],
             "pmin": rec["participation_min"],
             "pmax": rec["participation_max"],
             "e": rec["e_stim_window_mean_hz"], "sil": rec["e_frac_silent"],
             "hi": rec["e_frac_above_10hz"], "k": rec["k_fit"],
             "t": rec["spikes_per_third"], "w": rec["diag_wall_s"],
             "o": out}, flush=True)


if __name__ == "__main__":
    main()
