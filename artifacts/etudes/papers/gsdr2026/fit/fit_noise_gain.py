"""Lane 4e residual fit: scalar noise-gain X on seeds 1-3 (GPU, flock).

X is the ``noise_gain`` residual in ``jaxfne.emitters_hh.simulate_hh``
(the noise level; 1.0 = reference). theta is fixed at the Step-4 gstar
conductances. Loss on the like-for-like held-out estimands
(``rescore_4e`` z vs y stim labels)::

    L = mean over seeds 1-3 of sum(e^2), e = |z - y| / |y|

over (spike-PSD peak, gamma, beta, stim-window rate, kappa). Optimizer:
``jaxfne.optim.agsdr`` spec values (alpha, exploration, deselect_factor)
driving the same generational loop as ``fit_gsdr.py`` on the scalar
``a`` in [0, 3]. NaN losses are never selected.

D (declared before fitting, unstated values only): the OU sigma clip upper
bound. The Fig. 6 run code clips ``U(-0.1, 0.1)`` to ``[0, 0.05]``
(``main_fig6.py:71-72``) while the packaged builder clips to ``[0, 0.1]``
(PORT_SPEC.md D5) -- the source itself attests the range [0.05, 0.1], i.e.
gain a in [1, 2] absorbs into D by moving the clip bound to ``0.05 * a``.
A fitted a outside [1, 2] is a step-6 finding (``needs X``).

Outputs land in the caller's directory (``scratch/gsdr2026/``), never under
a ``scored`` path. GPU: run under ``flock /tmp/jaxfne_gpu.lock``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN_DIR = _HERE.parent / "run"

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

Y_LABELS_DEFAULT = "/home/nejath/gsdr_src/y_labels.json"
GSTAR_DEFAULT = str(_repo_root / "scratch" / "gsdr2026" / "gsdr_fit"
                    / "g_star_uS_float32.npy")
FIT_SEEDS_DEFAULT = (1, 2, 3)
STIM_AMP = 0.6
A_LO, A_HI = 0.0, 3.0


def _refuse_scored_dir(out_dir):
    parts = Path(out_dir).resolve().parts
    if any("scored" in part.lower() for part in parts):
        raise SystemExit("error: refusing to write under a directory containing "
                         "'scored': %s" % out_dir)


def git_sha():
    out = subprocess.run(["git", "-C", str(_repo_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip()


def estimand_errors(v, y, rg, dl):
    """Like-for-like e vector of one stim trial (5 estimands)."""
    import numpy as np

    grid = np.asarray(y["grid"], dtype=np.float64)
    z_spike = np.asarray(rg.spike_trace_psd(v), dtype=np.float64)
    bands = dl.band_powers(z_spike, grid)
    spikes = rg.detect_spikes(v)
    n_steps = v.shape[0]
    i0 = min(int(500.0 / rg.DT_MS), n_steps - 2)
    i1 = min(int(1000.0 / rg.DT_MS), n_steps - 1)
    z = {
        "spike_peak_hz": float(grid[int(np.argmax(z_spike))]),
        "spike_gamma": float(bands["gamma"]),
        "spike_beta": float(bands["beta"]),
        "rate_stimwin_hz": float(
            np.asarray(rg.window_rates_hz(v, 500.0, 1000.0)).mean()),
        "kappa": float(rg.kappa_of_onsets(spikes[i0:i1, :])),
    }
    yrow = {"spike_peak_hz": float(y["stim_peak_hz"]),
            "spike_gamma": float(y["stim_bands"]["gamma"]),
            "spike_beta": float(y["stim_bands"]["beta"]),
            "rate_stimwin_hz": float(y["stim_rates_hz"]["mean"]),
            "kappa": float(y["kappa_stim"])}
    return {k: abs(z[k] - yrow[k]) / abs(yrow[k]) for k in yrow}, z


def candidate_loss(a, g_us, rg, dl, y, *, fit_seeds=FIT_SEEDS_DEFAULT):
    """Mean over fit seeds of sum(e^2) at noise gain ``a`` (theta = gstar)."""
    import numpy as np

    tot, comp = 0.0, []
    for s in fit_seeds:
        result, blobs = rg.run_trial(STIM_AMP, g_seed=s, ou_seed=s,
                                     g_override=np.asarray(
                                         g_us, dtype=np.float32),
                                     noise_gain=float(a))
        e, z = estimand_errors(blobs["v"], y, rg, dl)
        tot += sum(v * v for v in e.values())
        comp.append({"seed": int(s), "e": e, "z": z,
                     "mean_rate": result["mean_rate_hz"],
                     "wall_s": result["wall_s"]})
    return float(tot / len(fit_seeds)), comp


def run_fit(out_dir, *, generations=5, population=4, seed=0,
            y_labels_path=Y_LABELS_DEFAULT, gstar_path=GSTAR_DEFAULT,
            fit_seeds=FIT_SEEDS_DEFAULT):
    """Generational AGSDR fit of the scalar noise gain; returns (a, manifest)."""
    import numpy as np

    import jaxfne.optim as jopt

    _refuse_scored_dir(out_dir)
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_RUN_DIR))
    import run_gsdr as rg
    from fit_gsdr import load_y_labels

    import importlib.util

    dl_path = _RUN_DIR / "data_labels.py"
    spec = importlib.util.spec_from_file_location("gsdr2026_dl_fitx",
                                                  dl_path)
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)

    y = load_y_labels(y_labels_path)
    with open(gstar_path, "rb") as f:
        g_us = np.load(f).astype(np.float32)
    aspec = jopt.agsdr(alpha=0.7, exploration=0.2, deselect_factor=3.0)
    alpha = float(aspec.alpha)
    step = float(aspec.exploration) * (A_HI - A_LO)

    rng = np.random.default_rng(int(seed))
    center = 1.0
    best = {"loss": float("inf"), "a": None, "components": None}
    trajectory = []
    total_wall = 0.0

    def evaluate(a, tag):
        t0 = time.time()
        loss, comp = candidate_loss(a, g_us, rg, dl, y,
                                    fit_seeds=tuple(fit_seeds))
        wall = time.time() - t0
        finite = bool(np.isfinite(loss))
        print("%s a %.3f loss %.3f e-seeds %s wall %.1fs"
              % (tag, a, loss,
                 ",".join("%.1f" % sum(c["e"].values()) for c in comp),
                 wall), flush=True)
        return loss if finite else float("inf"), comp, wall

    loss0, comp0, wall0 = evaluate(center, "init")
    total_wall += wall0
    best = {"loss": loss0, "a": center, "components": comp0}
    trajectory.append({"generation": -1, "best_loss": best["loss"],
                       "best_a": best["a"]})
    for gen in range(int(generations)):
        cands = [center]
        for _ in range(int(population) - 1):
            cands.append(float(np.clip(
                center + step * rng.standard_normal(), A_LO, A_HI)))
        gen_best_loss, gen_best_a, gen_best_comp = best["loss"], None, None
        for i, cand in enumerate(cands):
            loss, comp, wall = evaluate(cand, "g%da%d" % (gen, i))
            total_wall += wall
            if loss < gen_best_loss:
                gen_best_loss, gen_best_a, gen_best_comp = loss, cand, comp
        if gen_best_a is not None:
            center = float(np.clip(center + alpha * (gen_best_a - center),
                                   A_LO, A_HI))
            if gen_best_loss < best["loss"]:
                best = {"loss": gen_best_loss, "a": gen_best_a,
                        "components": gen_best_comp}
        trajectory.append({"generation": gen, "best_loss": best["loss"],
                           "best_a": best["a"]})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    manifest = {
        "git_sha": sha,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "optimizer": {"name": "jaxfne.optim.agsdr", "alpha": alpha,
                      "exploration": float(aspec.exploration),
                      "deselect_factor": float(aspec.deselect_factor)},
        "residual_X": {"form": "noise_gain on ou_sigma",
                       "module": "jaxfne.emitters_hh.simulate_hh",
                       "bounds": [A_LO, A_HI], "theta": "gstar fixed",
                       "gstar": str(gstar_path)},
        "D": {"ou_sigma_clip_upper": {
            "range": [0.05, 0.1],
            "reason": "run code clips to 0.05, packaged builder to 0.1 "
                      "(PORT_SPEC.md D5); gain a absorbs as 0.05*a"}},
        "y_labels": str(y_labels_path),
        "fit_seeds": [int(s) for s in fit_seeds],
        "loss": "mean_seed sum(e^2) over 5 like-for-like estimands",
        "generations": int(generations), "population": int(population),
        "seed": int(seed),
        "best_loss": best["loss"], "best_a": best["a"],
        "best_components": best["components"],
        "trajectory": trajectory,
        "total_wall_s": float(total_wall),
    }
    try:
        import jax as _jax

        manifest["jax_version"] = _jax.__version__
        manifest["jax_devices"] = [str(d) for d in _jax.devices()]
    except Exception:
        pass
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("X FIT DONE a %.4f loss %.4f wall %.1fs -> %s"
          % (best["a"], best["loss"], total_wall, out), flush=True)
    return best["a"], manifest


def build_parser():
    ap = argparse.ArgumentParser(description="Lane 4e residual noise-gain fit.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--y-labels", default=Y_LABELS_DEFAULT)
    ap.add_argument("--gstar", default=GSTAR_DEFAULT)
    ap.add_argument("--fit-seeds", type=int, nargs="*", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_fit(args.out, generations=args.generations,
            population=args.population, seed=args.seed,
            y_labels_path=args.y_labels, gstar_path=args.gstar,
            fit_seeds=tuple(args.fit_seeds) if args.fit_seeds
            else FIT_SEEDS_DEFAULT)


if __name__ == "__main__":
    main()
