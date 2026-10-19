"""GSDR fit of the Fig. 6 conductances on the jaxfne core HH route (P4).

Loss: the Fig. 6 batch loss with data-label PSD terms. y comes from
``/home/nejath/gsdr_src/y_labels.json`` (data, outside the repo: stim/base
label PSDs on ``linspace(15, 60, 40)``, kappa, onset rates). z comes only
from the jaxfne core (``run_gsdr.run_trial`` on ``emitters_hh`` /
``_construct_hh``). Batch per candidate: amp 0.0 (ou 1) against the baseline
labels + amp 0.6 (ou seeds 1-3, the fit seeds) against the stim labels::

    L = 990 * mean(psd_loss) + 9 * mean(rate_penalty) + 100 * mean(|k - ky|)

``psd_loss``/``rate_penalty`` are ``run_gsdr``'s (PORT_SPEC.md section 9,
``misc/main.py: loss``, read as data); the kappa term follows the packaged
``get_loss_fn`` variant (``gsdr/pipeline.py``, 100 * |kappa|, read as data),
with kappa on both sides the pairwise-Fleiss 5 ms statistic (model side:
``run_gsdr.kappa_of_onsets`` on the stim [500, 1000] / baseline [0, 200] ms
windows, matching the y windows' lengths).

Trainable set (read as text from ``/home/nejath/gsdr_src/main_fig6.py``):
all ``gAMPA`` + ``gGABAa`` conductances, 1800 + 430 = 2230 (``:210-211``
``make_trainable("gAMPA")`` / ``make_trainable("gGABAa")``), init
``clip(U(-4, 10), 0, 8)`` / ``clip(U(2, 10), 0, 10)`` (``:191-192``), clamps
[0, 10] (``:194-198``). Optimizer: ``jaxfne.optim.gsdr`` with the paper run
values (``:671-681``: inner SDR lr 0.1 / momentum 0.4, ``deselection_threshold``
3.0 carried as ``deselect_factor``, ``a_init = 0.0``); generational loop with
an elitist anchor, Gaussian proposals at the spec's ``exploration`` scale,
and the delta-rule center update with the spec's ``alpha``. NaN losses are
never selected (the paper reverts to the best state on NaN).

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
FIT_SEEDS_DEFAULT = (1, 2, 3)
STIM_AMP = 0.6
KAPPA_WEIGHT = 100.0
BASE_OU_SEED = 1


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


def load_y_labels(path=Y_LABELS_DEFAULT):
    with open(path) as f:
        d = json.load(f)
    for key in ("grid", "baseline_psd", "stim_psd", "kappa_baseline",
                "kappa_stim", "baseline_rates_hz", "stim_rates_hz",
                "baseline_peak_hz", "stim_peak_hz",
                "baseline_bands", "stim_bands"):
        if key not in d:
            raise ValueError("y labels %s miss %r" % (path, key))
    return d


def candidate_loss(g_us, rg, y, *, fit_seeds=FIT_SEEDS_DEFAULT,
                   morphology="single"):
    """Fig. 6 batch loss of one conductance vector (reference-uS)."""
    import numpy as np

    grid = np.asarray(y["grid"], dtype=np.float64)
    base_psd = np.asarray(y["baseline_psd"], dtype=np.float64)
    stim_psd = np.asarray(y["stim_psd"], dtype=np.float64)
    trials = [(0.0, BASE_OU_SEED, base_psd, float(y["kappa_baseline"]))]
    trials += [(STIM_AMP, s, stim_psd, float(y["kappa_stim"])) for s in fit_seeds]
    psd_terms, pens, kappas, comp = [], [], [], []
    for amp, ou, lab, ky in trials:
        result, blobs = rg.run_trial(amp, g_seed=ou, ou_seed=ou,
                                     g_override=np.asarray(g_us, dtype=np.float32),
                                     morphology=morphology)
        v = blobs["v"]
        z_psd = np.asarray(result["psd"], dtype=np.float64)
        psd_terms.append(rg.psd_loss(z_psd, lab))
        pens.append(result["rate_penalty"])
        spikes = rg.detect_spikes(v)
        n_steps = v.shape[0]
        i0 = int(0.0 / rg.DT_MS) if amp == 0.0 else int(500.0 / rg.DT_MS)
        i1 = int(200.0 / rg.DT_MS) if amp == 0.0 else int(1000.0 / rg.DT_MS)
        i0 = max(0, min(i0, n_steps - 2))
        i1 = max(i0 + 2, min(i1, n_steps - 1))
        k = rg.kappa_of_onsets(spikes[i0:i1, :])
        kappas.append(abs(float(k) - ky))
        comp.append({"amp": amp, "ou": ou, "mean_rate": result["mean_rate_hz"],
                     "active_frac": float((np.asarray(result["rates_hz"]) > 0).mean()),
                     "kappa": float(k), "wall_s": result["wall_s"]})
    loss = float(np.mean(psd_terms) * rg.PSD_WEIGHT
                 + np.mean(pens) * rg.RATE_WEIGHT
                 + np.mean(kappas) * KAPPA_WEIGHT)
    return loss, {"mean_psd_term": float(np.mean(psd_terms)),
                  "mean_penalty": float(np.mean(pens)),
                  "mean_kappa_term": float(np.mean(kappas)),
                  "trials": comp}


def run_fit(out_dir, *, generations=2, population=3, seed=0,
            y_labels_path=Y_LABELS_DEFAULT, fit_seeds=FIT_SEEDS_DEFAULT):
    """Generational GSDR fit; returns (best_g_us, manifest)."""
    import numpy as np

    import jaxfne.optim as jopt

    _refuse_scored_dir(out_dir)
    sys.path.insert(0, str(_RUN_DIR))
    import run_gsdr as rg

    y = load_y_labels(y_labels_path)
    spec = jopt.gsdr(alpha=0.7, exploration=0.05, deselect_factor=3.0)
    alpha = float(spec.alpha)
    exploration = float(spec.exploration)
    span = 10.0  # clamp bounds [0, 10] width (main_fig6.py:194-198)

    import jax

    g0_density = np.asarray(
        rg.init_conductances(jax.random.PRNGKey(1)), dtype=np.float64)
    center = np.clip(g0_density / rg.US_TO_MSCM2, 0.0, 10.0)
    n_params = int(center.shape[0])

    rng = np.random.default_rng(int(seed))
    trajectory = []
    best = {"loss": float("inf"), "g": None, "components": None}

    def evaluate(g, tag):
        t0 = time.time()
        loss, comp = candidate_loss(g, rg, y, fit_seeds=tuple(fit_seeds))
        wall = time.time() - t0
        finite = bool(np.isfinite(loss))
        print("%s loss %.4f (psd %.4f pen %.4f kap %.4f) active %s wall %.1fs"
              % (tag, loss, comp["mean_psd_term"], comp["mean_penalty"],
                 comp["mean_kappa_term"],
                 ",".join("%.2f" % t["active_frac"] for t in comp["trials"]),
                 wall), flush=True)
        return loss if finite else float("inf"), comp, wall

    total_wall = 0.0
    loss0, comp0, wall0 = evaluate(center, "init")
    total_wall += wall0
    if loss0 < best["loss"]:
        best = {"loss": loss0, "g": center.copy(), "components": comp0}
    trajectory.append({"generation": -1, "best_loss": best["loss"],
                       "components": comp0})
    for gen in range(int(generations)):
        cands = [center.copy()]
        for _ in range(int(population) - 1):
            cands.append(np.clip(
                center + exploration * span * rng.standard_normal(n_params),
                0.0, 10.0))
        gen_best_loss, gen_best_g, gen_best_comp = best["loss"], None, None
        for i, cand in enumerate(cands):
            loss, comp, wall = evaluate(
                cand, "g%dc%d" % (gen, i))
            total_wall += wall
            if loss < gen_best_loss:
                gen_best_loss, gen_best_g, gen_best_comp = loss, cand, comp
        if gen_best_g is not None:
            center = np.clip(center + alpha * (gen_best_g - center), 0.0, 10.0)
            if gen_best_loss < best["loss"]:
                best = {"loss": gen_best_loss, "g": gen_best_g.copy(),
                        "components": gen_best_comp}
        trajectory.append({"generation": gen, "best_loss": best["loss"],
                           "components": best["components"]})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    gname = "g_star_uS_float32.npy"
    tmp = out / (gname + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(best["g"], dtype=np.float32))
    os.replace(tmp, out / gname)
    manifest = {
        "git_sha": sha,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "optimizer": {"name": "lane delta-rule center update (gradient-free)",
                      "alpha": alpha, "exploration": exploration,
                      "mcdp_applied": False,
                      "deselect_factor_consumed": False,
                      "authors_optimizer_settings_not_used": {
                          "deselect_factor": float(spec.deselect_factor),
                          "inner_SDR_lr": 0.1, "inner_SDR_momentum": 0.4,
                          "a_init": 0.0, "mcdp": True}},
        "trainable": {"gAMPA_edges": rg.N_AMPA_EDGES,
                      "gGABAa_edges": rg.N_GABAA_GLOBAL_EDGES + rg.N_GABAA_LOCAL_EDGES,
                      "bounds_uS": [0.0, 10.0],
                      "paper_code_ref": "main_fig6.py:191-192,194-198,210-211,671-681"},
        "y_labels": str(y_labels_path),
        "fit_seeds": [int(s) for s in fit_seeds],
        "loss_weights": {"psd": rg.PSD_WEIGHT, "rate": rg.RATE_WEIGHT,
                         "kappa": KAPPA_WEIGHT},
        "generations": int(generations), "population": int(population),
        "seed": int(seed), "n_params": n_params,
        "best_loss": best["loss"],
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
    print("FIT DONE best %.4f wall %.1fs -> %s" % (best["loss"], total_wall, out),
          flush=True)
    return np.asarray(best["g"], dtype=np.float32), manifest


def build_parser():
    ap = argparse.ArgumentParser(description="GSDR fit of Fig. 6 conductances.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--population", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--y-labels", default=Y_LABELS_DEFAULT)
    ap.add_argument("--fit-seeds", type=int, nargs="*", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_fit(args.out, generations=args.generations, population=args.population,
            seed=args.seed, y_labels_path=args.y_labels,
            fit_seeds=tuple(args.fit_seeds) if args.fit_seeds else FIT_SEEDS_DEFAULT)


if __name__ == "__main__":
    main()
