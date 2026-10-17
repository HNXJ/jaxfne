"""GSDR 2026 structural sweep at g*: E/I co-scaling and stim amplitude (no fitting).

Step-6 diagnosis follow-up (CALIBRATION.md §6.4–6.6): at the fitted g* every
cell fires every gamma cycle (active 1.00, kappa ~0.95–1.0) while the data are
sparse (2.14 Hz, kappa 0.0156). GABAa 1–16x, OU noise 1–4x and the kappa
estimator were tested and rejected. This sweep asks whether the saturation is
structural: co-scale E with I, or change the 120 Hz stimulus drive.

Three sweeps, seeds 1 and 2 each, as in CALIBRATION §6.4. The seed drives
the OU noise only: ``run_trial`` ignores ``g_seed`` when ``g_override`` is
given (``run_gsdr.py:417-421``).
  A ``all``:  every conductance x {0.25, 0.5, 1, 2}, stim 0.6.
  B ``ampa``: AMPA only x {0, 0.1, 0.25, 0.5}, stim 0.6.
  C ``stim``: g* unscaled, stim amplitude {0, 0.15, 0.3, 0.6, 1.2}.

Per run: the reference fields (mean rate, active fraction, psd term, penalty,
kappa_fit, kappa term, wall) plus the core classifier
``jaxfne.analysis.activity_regimes`` on the E, global-I and local-I
populations over [500, 1000] ms with default thresholds (label and measured
values per population). Synchrony uses the core ``wang_buzsaki_kappa`` —
no local copy. Stim-0 rows score against the baseline labels over [0, 200] ms
(like the fit's baseline trial); all other rows against the stim labels over
[500, 1000] ms. The JSON writes incrementally (after each run), so a crash
keeps finished rows. GPU: run under ``flock /tmp/jaxfne_gpu.lock``.
"""

from __future__ import annotations

import argparse
import json
import os
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

import run_gsdr as rg  # noqa: E402

GSTAR_DEFAULT = ("/mnt/f/warehouse/gsdr/lane4f_runs_2026-10-14/gsdr_fit500/"
                 "g_star_uS_float32.npy")
LABELS_DEFAULT = "/mnt/f/warehouse/gsdr/lane4f_runs_2026-10-14/y_labels.json"
OUT_DEFAULT = "/mnt/f/warehouse/gsdr/diag_structural_2026-10-17.json"

SWEEP_ALL = "all"
SWEEP_AMPA = "ampa"
SWEEP_STIM = "stim"
SWEEPS = (SWEEP_ALL, SWEEP_AMPA, SWEEP_STIM)

ALL_SCALES = (0.25, 0.5, 1.0, 2.0)
AMPA_SCALES = (0.0, 0.1, 0.25, 0.5)
STIM_AMPS = (0.0, 0.15, 0.3, 0.6, 1.2)
STIM_REF_AMP = 0.6
SEEDS = (1, 2)

T_L_MS, T_R_MS = 500.0, 1000.0
BASE_T_L_MS, BASE_T_R_MS = 0.0, 200.0


def scale_conductances(g, factor, scope, n_ampa=rg.N_AMPA_EDGES):
    """Scale a reference-uS conductance vector; AMPA entries come first.

    ``scope`` is ``"all"`` (every conductance x factor) or ``"ampa"``
    (first ``n_ampa`` entries x factor, GABAa untouched). Returns a new
    float32 array; the input is never mutated.
    """
    import numpy as np

    if scope not in (SWEEP_ALL, SWEEP_AMPA):
        raise ValueError("scope must be 'all' or 'ampa', got %r" % (scope,))
    vec = np.asarray(g, dtype=np.float32).copy()
    if vec.ndim != 1 or vec.shape[0] <= int(n_ampa):
        raise ValueError("g must hold more than n_ampa entries, got %s"
                         % (vec.shape,))
    factor = float(factor)
    if scope == SWEEP_ALL:
        vec = (vec * factor).astype(np.float32)
    else:
        vec[: int(n_ampa)] = (vec[: int(n_ampa)] * factor).astype(np.float32)
    return vec


def build_parser():
    ap = argparse.ArgumentParser(
        description="GSDR structural sweep at g* (no fitting).")
    ap.add_argument("--gstar", default=GSTAR_DEFAULT)
    ap.add_argument("--labels", default=LABELS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--sweep", choices=list(SWEEPS), default=None,
                    help="run one sweep only (default: all three)")
    ap.add_argument("--smoke", action="store_true",
                    help="one row only: sweep C stim 0.6 seed 1")
    ap.add_argument("--ampa-scales", type=float, nargs="+", default=None,
                    help="AMPA factors for the ampa sweep (default %s)"
                    % (AMPA_SCALES,))
    return ap


def plan_rows(sweep=None, smoke=False, ampa_scales=None):
    """Return the (sweep, scale, seed) plan in run order."""
    if smoke:
        return [(SWEEP_STIM, STIM_REF_AMP, 1)]
    names = [sweep] if sweep is not None else list(SWEEPS)
    ampa = tuple(ampa_scales) if ampa_scales is not None else AMPA_SCALES
    plan = []
    for name in names:
        if name == SWEEP_ALL:
            plan += [(name, s, seed) for s in ALL_SCALES for seed in SEEDS]
        elif name == SWEEP_AMPA:
            plan += [(name, s, seed) for s in ampa for seed in SEEDS]
        elif name == SWEEP_STIM:
            plan += [(name, a, seed) for a in STIM_AMPS for seed in SEEDS]
        else:  # pragma: no cover — argparse choices refuse these first
            raise ValueError("unknown sweep %r" % (name,))
    return plan


def window_edges(n_steps, t_l_ms, t_r_ms, dt_ms=rg.DT_MS):
    i0 = max(0, min(int(t_l_ms / dt_ms), n_steps - 2))
    i1 = max(i0 + 2, min(int(t_r_ms / dt_ms), n_steps - 1))
    return i0, i1


def run_row(gstar, y, sweep, scale, seed):
    """One sweep row: simulate, score, classify. No fitting."""
    import numpy as np

    from jaxfne.analysis import activity_regimes, wang_buzsaki_kappa

    stim_amp = float(scale) if sweep == SWEEP_STIM else STIM_REF_AMP
    if sweep == SWEEP_STIM:
        g = np.asarray(gstar, dtype=np.float32)
        g_label = "gstar"
    else:
        g = scale_conductances(gstar, scale, sweep)
        g_label = "%s x %g" % (sweep, float(scale))
    baseline = stim_amp == 0.0
    lab_psd = np.asarray(y["baseline_psd"] if baseline else y["stim_psd"],
                         dtype=np.float64)
    ky = float(y["kappa_baseline"] if baseline else y["kappa_stim"])

    result, blobs = rg.run_trial(stim_amp, g_seed=int(seed),
                                 ou_seed=int(seed), g_override=g)
    v = blobs["v"]
    spikes = rg.detect_spikes(v)
    n_steps = v.shape[0]
    if baseline:
        i0, i1 = window_edges(n_steps, BASE_T_L_MS, BASE_T_R_MS)
    else:
        i0, i1 = window_edges(n_steps, T_L_MS, T_R_MS)
    k_fit = float(rg.kappa_of_onsets(spikes[i0:i1, :]))
    win = spikes[i0:i1, :]
    wb = float(wang_buzsaki_kappa(win, rg.DT_MS, bin_ms=5.0))
    rates = np.asarray(result["rates_hz"], dtype=np.float64)

    pops = {"E": np.arange(0, rg.N_E),
            "I_global": np.arange(rg.N_E, rg.N_E + rg.N_IG),
            "I_local": np.arange(rg.N_E + rg.N_IG, rg.N_CELLS)}
    regimes = activity_regimes(spikes, rg.DT_MS, pops,
                               kept_window=(i0, i1))
    regimes_out = {}
    for name, rep in regimes.items():
        regimes_out[name] = {
            "label": rep["label"],
            "rate_hz": float(rep["rate_hz"]),
            "active_fraction": float(rep["active_fraction"]),
            "kappa": float(rep["kappa"]),
            "cv_isi": float(rep["cv_isi"]),
            "rate_ratio": float(rep["rate_ratio"]),
            "spectral_peak_excess": float(rep["spectral_peak_excess"]),
        }
    return {
        "sweep": sweep,
        "scale": float(scale),
        "g": g_label,
        "seed": int(seed),
        "stim_amp": stim_amp,
        "mean_rate_hz": float(result["mean_rate_hz"]),
        "active_frac": float((rates > 0).mean()),
        "psd_term": float(rg.psd_loss(np.asarray(result["psd"],
                                                 dtype=np.float64), lab_psd)),
        "penalty": float(result["rate_penalty"]),
        "kappa_fit": k_fit,
        "kappa_term": abs(k_fit - ky),
        "kappa_target": ky,
        "wb_kappa_5ms": wb,
        "regimes": regimes_out,
        "wall_s": float(result["wall_s"]),
    }


def write_incremental(out_path, payload):
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2))
    os.replace(tmp, out_path)


def _json_safe(obj):
    """Replace non-finite floats with None so the JSON stays strict."""
    import math

    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def main(argv=None):
    import numpy as np

    args = build_parser().parse_args(argv)
    with open(args.gstar, "rb") as f:
        gstar = np.asarray(np.load(f), dtype=np.float32)
    expect = (rg.N_AMPA_EDGES + rg.N_GABAA_GLOBAL_EDGES
              + rg.N_GABAA_LOCAL_EDGES)
    if gstar.shape != (expect,):
        raise ValueError("gstar must hold %d reference-uS values, got %s"
                         % (expect, gstar.shape,))
    with open(args.labels) as f:
        y = json.load(f)
    for key in ("baseline_psd", "stim_psd", "kappa_baseline", "kappa_stim"):
        if key not in y:
            raise ValueError("labels %s miss %r" % (args.labels, key))
    print("gstar GABAa range uS: %.3f .. %.3f"
          % (float(gstar[rg.N_AMPA_EDGES:].min()),
             float(gstar[rg.N_AMPA_EDGES:].max())), flush=True)

    payload = {"gstar": str(args.gstar), "labels": str(args.labels),
               "rows": []}
    t0 = time.time()
    for sweep, scale, seed in plan_rows(sweep=args.sweep, smoke=args.smoke,
                                        ampa_scales=args.ampa_scales):
        row = run_row(gstar, y, sweep, scale, seed)
        payload["rows"].append(row)
        write_incremental(args.out, payload)
        reg = row["regimes"]
        print("%(sweep)s %(scale)g seed %(seed)d: rate %(r).2f Hz active "
              "%(a).2f psd %(p).4f pen %(pen).4f k_fit %(k).4f k_term "
              "%(kt).4f wb %(wb).4f E:%(e)s Ig:%(ig)s Il:%(il)s wall "
              "%(w).1fs"
              % {"sweep": sweep, "scale": scale, "seed": seed,
                 "r": row["mean_rate_hz"], "a": row["active_frac"],
                 "p": row["psd_term"], "pen": row["penalty"],
                 "k": row["kappa_fit"], "kt": row["kappa_term"],
                 "wb": row["wb_kappa_5ms"], "e": reg["E"]["label"],
                 "ig": reg["I_global"]["label"], "il": reg["I_local"]["label"],
                 "w": row["wall_s"]}, flush=True)
    print("wrote %s (%d rows, wall %.1fs)"
          % (args.out, len(payload["rows"]), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
