"""Step-6 fit at the authors' epoch budget (GPU, flock, checkpointed).

Same loss, init, seeds and generational mechanics as Step 4
(``fit_gsdr.py``: ``candidate_loss`` on baseline + stim seeds 1-3, center0
= seed-1 init, Gaussian proposals at exploration x span, elitist anchor,
delta-rule center update with alpha, NaN never selected). The only
differences: ``--generations`` may run to 500 (the authors'
``net_trainer_temp1(net, optimizer, dataloader, 500)``,
``main_fig6.py:684``), and every 10 generations the driver writes a
checkpoint (center + best g + rng advance count) and a partial manifest
with the loss trajectory, so a partial run stays usable. ``--resume``
continues from the latest checkpoint.
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

Y_LABELS_DEFAULT = "/home/nejath/gsdr_src/y_labels.json"
FIT_SEEDS_DEFAULT = (1, 2, 3)
CKPT_EVERY = 10


def _ckpt_path(out_dir, gen):
    return Path(out_dir) / ("ckpt_gen%04d.npz" % int(gen))


def _latest_ckpt(out_dir):
    cands = sorted(Path(out_dir).glob("ckpt_gen*.npz")) if Path(out_dir).exists() else []
    return cands[-1] if cands else None


def run_fit_500(out_dir, *, generations=500, population=6, seed=0,
                y_labels_path=Y_LABELS_DEFAULT,
                fit_seeds=FIT_SEEDS_DEFAULT, resume=False, morphology="single"):
    """Generational GSDR fit with checkpoints; returns (best_g_us, manifest)."""
    import numpy as np

    import jaxfne.optim as jopt

    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_RUN_DIR))
    import run_gsdr as rg
    from fit_gsdr import (_refuse_scored_dir, candidate_loss, git_sha,
                          load_y_labels)

    _refuse_scored_dir(out_dir)
    y = load_y_labels(y_labels_path)
    spec = jopt.gsdr(alpha=0.7, exploration=0.05, deselect_factor=3.0)
    alpha = float(spec.alpha)
    exploration = float(spec.exploration)
    span = 10.0

    import jax

    g0_density = np.asarray(
        rg.init_conductances(jax.random.PRNGKey(1)), dtype=np.float64)
    center = np.clip(g0_density / rg.US_TO_MSCM2, 0.0, 10.0)
    n_params = int(center.shape[0])

    rng = np.random.default_rng(int(seed))
    trajectory = []
    best = {"loss": float("inf"), "g": None, "components": None}
    total_wall = 0.0
    start_gen = 0

    def evaluate(g, tag):
        t0 = time.time()
        loss, comp = candidate_loss(g, rg, y, fit_seeds=tuple(fit_seeds),
                                    morphology=morphology)
        wall = time.time() - t0
        finite = bool(np.isfinite(loss))
        print("%s loss %.4f (psd %.4f pen %.4f kap %.4f) active %s wall %.1fs"
              % (tag, loss, comp["mean_psd_term"], comp["mean_penalty"],
                 comp["mean_kappa_term"],
                 ",".join("%.2f" % t["active_frac"] for t in comp["trials"]),
                 wall), flush=True)
        return loss if finite else float("inf"), comp, wall

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if resume:
        ck = _latest_ckpt(out)
        if ck is None:
            raise SystemExit("error: --resume with no checkpoint in %s" % out)
        z = np.load(ck)
        start_gen = int(z["next_gen"])
        center = np.asarray(z["center"], dtype=np.float64)
        best = {"loss": float(z["best_loss"]),
                "g": np.asarray(z["best_g"], dtype=np.float64),
                "components": json.loads(str(z["best_components_json"]))}
        total_wall = float(z["total_wall_s"])
        traj_path = out / "manifest_partial.json"
        prior = json.loads(traj_path.read_text()) if traj_path.exists() else {}
        trajectory = prior.get("trajectory", [])
        # The checkpoint's own record wins; checkpoints written before it carried one fall
        # back to the partial manifest, and fits older than the flag were all single.
        if "morphology" in z.files:
            prior_morph = str(z["morphology"])
        else:
            prior_morph = prior.get("morphology", "single")
        if prior_morph != morphology:
            raise SystemExit("error: --resume with morphology %r over a %r fit"
                             % (morphology, prior_morph))
        for _ in range(start_gen * (int(population) - 1)):
            rng.standard_normal(n_params)
        print("RESUMED from %s at gen %d best %.4f" % (ck, start_gen, best["loss"]),
              flush=True)
    else:
        if _latest_ckpt(out) is not None:
            raise SystemExit("error: refusing to overwrite checkpoints in %s "
                             "(pass --resume)" % out)
        loss0, comp0, wall0 = evaluate(center, "init")
        total_wall += wall0
        if loss0 < best["loss"]:
            best = {"loss": loss0, "g": center.copy(), "components": comp0}
        trajectory.append({"generation": -1, "best_loss": best["loss"],
                           "components": comp0})

    def write_ckpt(next_gen):
        # NOTE: np.savez appends ".npz" unless the name already ends with
        # it, so the tmp name must keep the suffix (a bare ".tmp" names a
        # ".tmp.npz" file and the replace below misses).
        tmp = out / ("ckpt_gen%04d.tmp.npz" % int(next_gen - 1))
        np.savez(tmp, center=np.asarray(center, dtype=np.float64),
                 best_g=np.asarray(best["g"], dtype=np.float64),
                 best_loss=np.float64(best["loss"]),
                 best_components_json=np.array(
                     json.dumps(best["components"])),
                 total_wall_s=np.float64(total_wall),
                 next_gen=np.int64(next_gen),
                 morphology=np.array(morphology))
        os.replace(tmp, _ckpt_path(out, next_gen - 1))
        (out / "manifest_partial.json").write_text(json.dumps(
            {"git_sha": git_sha(), "morphology": morphology,
             "completed_generations": int(next_gen),
             "best_loss": best["loss"], "best_components": best["components"],
             "trajectory": trajectory, "total_wall_s": float(total_wall)},
            indent=2))

    for gen in range(start_gen, int(generations)):
        cands = [center.copy()]
        for _ in range(int(population) - 1):
            cands.append(np.clip(
                center + exploration * span * rng.standard_normal(n_params),
                0.0, 10.0))
        gen_best_loss, gen_best_g, gen_best_comp = best["loss"], None, None
        for i, cand in enumerate(cands):
            loss, comp, wall = evaluate(cand, "g%dc%d" % (gen, i))
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
        if (gen + 1) % CKPT_EVERY == 0:
            write_ckpt(gen + 1)
            print("CKPT gen %d best %.4f wall %.1fs" % (gen + 1, best["loss"], total_wall),
                  flush=True)

    gname = "g_star_uS_float32.npy"
    tmp = out / (gname + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(best["g"], dtype=np.float32))
    os.replace(tmp, out / gname)
    sha = git_sha()
    manifest = {
        "git_sha": sha,
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
        "budget": {"authors_epochs": 500,
                   "paper_call": "main_fig6.py:684",
                   "lane_generations": int(generations),
                   "population": int(population)},
        "morphology": morphology,
        "y_labels": str(y_labels_path),
        "fit_seeds": [int(s) for s in fit_seeds],
        "loss_weights": {"psd": rg.PSD_WEIGHT, "rate": rg.RATE_WEIGHT,
                         "kappa": 100.0},
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
    print("FIT500 DONE best %.4f wall %.1fs -> %s" % (best["loss"], total_wall, out),
          flush=True)
    return np.asarray(best["g"], dtype=np.float32), manifest


def build_parser():
    ap = argparse.ArgumentParser(description="Step-6 GSDR fit at author budget.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=500)
    ap.add_argument("--population", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--y-labels", default=Y_LABELS_DEFAULT)
    ap.add_argument("--fit-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--morphology", choices=("single", "reference"), default="single")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_fit_500(args.out, generations=args.generations,
                population=args.population, seed=args.seed,
                y_labels_path=args.y_labels,
                fit_seeds=tuple(args.fit_seeds) if args.fit_seeds else FIT_SEEDS_DEFAULT,
                resume=args.resume, morphology=args.morphology)


if __name__ == "__main__":
    main()
