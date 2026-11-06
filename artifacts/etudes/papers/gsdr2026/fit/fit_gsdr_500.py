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

MCDP mode (``--mcdp``): the generational center update above is replaced by
``jaxfne.optim.gsdr.gsdr_mcdp_transform`` with the authors' settings (inner
SDR learning_rate 0.1 / momentum 0.4, deselection_threshold 3.0, a_init 0.0
dynamic, lambda_d 1.0, checkpoint_n 5; PORT_SPEC section 10). The gradient
slot takes an antithetic evolution-strategies estimate from the generation's
own candidates: P must be even (P = 2k pairs ``L(c + s*eps_i)`` /
``L(c - s*eps_i)`` evaluated with the same ``evaluate``; ``s`` is
``--es-sigma``, default the Gaussian proposal scale ``exploration * span``).
Pairs with a non-finite loss are dropped; no finite pair voids the epoch
(the authors void an epoch on NaN). MCDP factors come from the center's
batch traces (same 4 trials as ``candidate_loss``) through the core
``mcdp_factors`` helper, per AMPA / GABAa group. The loss and its selectors
stay bit-identical; the manifest declares the ES estimator as a deviation
from the authors' reverse-mode gradient. Without ``--mcdp`` every number,
key and trajectory is bit-identical to before.

``--batch`` evaluates each generation's candidates as one
``simulate_hh_batch`` call per trial (``fit_gsdr.candidate_losses_batch``)
instead of a serial ``candidate_loss`` loop. It is an opt-in fast path for
pilots and sweeps (Hamm 2026-11-06): losses agree with the serial path within
relative 1e-2 with the same ranking, spike counts within 2 per trial
(``test_fit_gsdr_batch.py``); float32 vmap rounding leaves the serial trajectory
after about 200 steps. The flag defaults off; signed and scored fits stay serial.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
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


def es_gradient(eps_list, l_plus, l_minus, sigma):
    """Antithetic ES gradient estimate: ``sum (L+ - L-) eps_i / (2 s n)``.

    ``eps_list`` holds the surviving (finite-loss) directions, ``l_plus``
    / ``l_minus`` their losses, ``sigma`` the sampling scale ``s``. All in
    the center's units. Needs at least one pair; the caller voids the epoch
    otherwise.
    """
    import numpy as np

    n = len(eps_list)
    if n == 0:
        raise ValueError("es_gradient needs at least one finite pair")
    if not (np.isfinite(float(sigma)) and float(sigma) > 0.0):
        raise ValueError("es_gradient needs a finite positive sigma, got %r"
                         % (sigma,))
    acc = np.zeros_like(np.asarray(eps_list[0], dtype=np.float64))
    for e, lp, lm in zip(eps_list, l_plus, l_minus):
        acc = acc + (float(lp) - float(lm)) * np.asarray(e, dtype=np.float64)
    return acc / (2.0 * float(sigma) * float(n))


def center_mcdp_factors(center, rg, fit_seeds, morphology="single"):
    """Per-edge MCDP factors of the center's batch traces (helper, not oracle).

    Runs the same 4 trials as ``candidate_loss`` (baseline + stim seeds) at
    ``center``, stacks their voltage traces, and z-scores per group with the
    core ``jaxfne.optim.gsdr.mcdp_factors`` (shift +10 steps, no smoothing;
    PORT_SPEC section 10). Edges come from the public build (same
    connectivity seeds as ``run_trial`` defaults); the receptor layout must
    match the trainable order (AMPA block, then GABAa), else this stops
    instead of guessing an order.
    """
    import numpy as np

    import jaxfne as jtfne
    from jaxfne.optim.gsdr import mcdp_factors

    from fit_gsdr import BASE_OU_SEED, STIM_AMP

    v_list = []
    for amp, ou in [(0.0, int(BASE_OU_SEED))] + [(float(STIM_AMP), int(s))
                                                for s in fit_seeds]:
        _, blobs = rg.run_trial(amp, g_seed=0, ou_seed=ou,
                                g_override=np.asarray(center, dtype=np.float32),
                                morphology=morphology)
        v_list.append(np.asarray(blobs["v"], dtype=np.float64))
    batch = np.stack(v_list, axis=0)

    cfg, _ = rg.build_spec()
    st = jtfne.construct(cfg).static["hh"]
    pre = np.asarray(st["pre"])
    post = np.asarray(st["post"])
    rec = np.asarray(st["receptor"])
    n_ampa = int(rg.N_AMPA_EDGES)
    if not (rec.shape[0] == pre.shape[0] == post.shape[0] == int(center.shape[0])
            and bool((rec[:n_ampa] == 0).all())
            and bool((rec[n_ampa:] != 0).all())
            and int((rec == 0).sum()) == n_ampa):
        raise SystemExit("error: receptor layout does not match the "
                         "[gAMPA, gGABAa] trainable order; refusing to guess "
                         "factor positions")
    fa = np.asarray(mcdp_factors(batch, pre[rec == 0], post[rec == 0], 10),
                    dtype=np.float64)
    fg = np.asarray(mcdp_factors(batch, pre[rec != 0], post[rec != 0], 10),
                    dtype=np.float64)
    return np.concatenate([fa, fg])


def _pack_state(obj):
    import numpy as np

    return np.array(bytearray(pickle.dumps(obj)), dtype=np.uint8)


def _unpack_state(arr):
    import numpy as np

    return pickle.loads(bytes(np.asarray(arr, dtype=np.uint8).tolist()))


def run_fit_500(out_dir, *, generations=500, population=6, seed=0,
                y_labels_path=Y_LABELS_DEFAULT,
                fit_seeds=FIT_SEEDS_DEFAULT, resume=False, morphology="single",
                mcdp=False, es_sigma=None, batch=False):
    """Generational GSDR fit with checkpoints; returns (best_g_us, manifest)."""
    import numpy as np

    import jaxfne.optim as jopt

    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_RUN_DIR))
    import run_gsdr as rg
    from fit_gsdr import (_refuse_scored_dir, candidate_loss,
                          candidate_losses_batch, git_sha, load_y_labels)

    _refuse_scored_dir(out_dir)
    y = load_y_labels(y_labels_path)
    spec = jopt.gsdr(alpha=0.7, exploration=0.05, deselect_factor=3.0)
    alpha = float(spec.alpha)
    exploration = float(spec.exploration)
    span = 10.0
    mcdp = bool(mcdp)
    if mcdp and int(population) % 2 != 0:
        raise SystemExit("error: --mcdp needs an even population (P = 2k "
                         "antithetic pairs), got %d" % int(population))
    if es_sigma is None:
        es_s = exploration * span
    else:
        es_s = float(es_sigma)
        if not (np.isfinite(es_s) and es_s > 0.0):
            raise SystemExit("error: --es-sigma must be finite and positive, "
                             "got %r" % (es_sigma,))

    import jax

    g0_density = np.asarray(
        rg.init_conductances(jax.random.PRNGKey(1)), dtype=np.float64)
    center = np.clip(g0_density / rg.US_TO_MSCM2, 0.0, 10.0)
    n_params = int(center.shape[0])

    rng = np.random.default_rng(int(seed))
    trajectory = []
    best = {"loss": float("inf"), "g": None, "components": None}
    total_wall = 0.0
    es_clipped_total = 0
    start_gen = 0
    a_traj = []
    opt_state = None
    transform = None
    if mcdp:
        from jaxfne.optim.gsdr import gsdr_mcdp_transform

        transform = gsdr_mcdp_transform(
            deselection_threshold=3.0, a_init=0.0, a_dynamic=True,
            lambda_d=1.0, checkpoint_n=5)
        opt_state = transform.init(
            jax.numpy.asarray(center, dtype=jax.numpy.float32))

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
        required = ["seed", "population"]
        if mcdp:
            required += ["es_sigma", "es_clipped_total"]
        for _key in required:
            if _key not in z.files:
                raise SystemExit("error: --resume over a checkpoint without "
                                 "%r: the checkpoint predates the "
                                 "sampling-state record" % (_key,))
        if int(z["seed"]) != int(seed):
            raise SystemExit("error: --resume with seed %d over a checkpoint "
                             "with seed %d" % (int(seed), int(z["seed"])))
        if int(z["population"]) != int(population):
            raise SystemExit("error: --resume with population %d over a "
                             "checkpoint with population %d"
                             % (int(population), int(z["population"])))
        if mcdp and float(z["es_sigma"]) != float(es_s):
            raise SystemExit("error: --resume with es_sigma %r over a "
                             "checkpoint with es_sigma %r"
                             % (float(es_s), float(z["es_sigma"])))
        ckpt_mcdp = "mcdp_gsdr_state" in z.files
        if bool(mcdp) != bool(ckpt_mcdp):
            raise SystemExit("error: --resume mode mismatch: checkpoint is "
                             "%s but --mcdp %s"
                             % ("mcdp" if ckpt_mcdp else "delta-rule",
                                "on" if mcdp else "off"))
        if mcdp:
            for _ in range(start_gen * (int(population) // 2)):
                rng.standard_normal(n_params)
            opt_state = (_unpack_state(z["mcdp_gsdr_state"]),
                         _unpack_state(z["mcdp_inner_state"]))
            a_traj = [float(a) for a in
                      json.loads(str(z["mcdp_a_traj_json"]))]
            es_clipped_total = int(z["es_clipped_total"])
        else:
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
        ckpt = {"center": np.asarray(center, dtype=np.float64),
                "best_g": np.asarray(best["g"], dtype=np.float64),
                "best_loss": np.float64(best["loss"]),
                "best_components_json": np.array(
                    json.dumps(best["components"])),
                "total_wall_s": np.float64(total_wall),
                "next_gen": np.int64(next_gen),
                "morphology": np.array(morphology),
                "seed": np.int64(int(seed)),
                "population": np.int64(int(population)),
                "es_sigma": np.float64(float(es_s)),
                "es_clipped_total": np.int64(int(es_clipped_total))}
        if mcdp:
            gsdr_state, inner_state = opt_state
            ckpt["mcdp_gsdr_state"] = _pack_state(gsdr_state)
            ckpt["mcdp_inner_state"] = _pack_state(inner_state)
            ckpt["mcdp_a_traj_json"] = np.array(json.dumps(a_traj))
        np.savez(tmp, **ckpt)
        os.replace(tmp, _ckpt_path(out, next_gen - 1))
        partial = {"git_sha": git_sha(), "morphology": morphology,
                 "completed_generations": int(next_gen),
                 "best_loss": best["loss"], "best_components": best["components"],
                 "trajectory": trajectory, "total_wall_s": float(total_wall)}
        if mcdp:
            partial["a_trajectory"] = [float(a) for a in a_traj]
            partial["es_clipped_coords"] = int(es_clipped_total)
        (out / "manifest_partial.json").write_text(json.dumps(partial, indent=2))

    for gen in range(start_gen, int(generations)):
        if mcdp:
            k = int(population) // 2
            key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), gen)
            dirs = [rng.standard_normal(n_params) for _ in range(k)]
            cands = [center.copy()]
            es_clipped = 0
            for e in dirs:
                for sgn in (1.0, -1.0):
                    raw = center + sgn * es_s * e
                    cand = np.clip(raw, 0.0, 10.0)
                    es_clipped += int((cand != raw).sum())
                    cands.append(cand)
            es_clipped_total += es_clipped
            losses = []
            if batch:
                t0 = time.time()
                bloss, bcomp = candidate_losses_batch(
                    cands, rg, y, fit_seeds=tuple(fit_seeds),
                    morphology=morphology)
                bwall = time.time() - t0
                total_wall += bwall
                for i, (loss, comp) in enumerate(zip(bloss, bcomp)):
                    finite = bool(np.isfinite(loss))
                    print("g%dc%d loss %.4f (psd %.4f pen %.4f kap %.4f) "
                          "active %s wall %.1fs (batch)" % (
                              gen, i, loss, comp["mean_psd_term"],
                              comp["mean_penalty"], comp["mean_kappa_term"],
                              ",".join("%.2f" % t["active_frac"]
                                        for t in comp["trials"]),
                              bwall / len(cands)), flush=True)
                    loss = loss if finite else float("inf")
                    losses.append(loss)
                    if loss < best["loss"]:
                        best = {"loss": loss, "g": cands[i].copy(),
                                "components": comp}
            else:
                for i, cand in enumerate(cands):
                    loss, comp, wall = evaluate(cand, "g%dc%d" % (gen, i))
                    total_wall += wall
                    losses.append(loss)
                    if loss < best["loss"]:
                        best = {"loss": loss, "g": cand.copy(),
                                "components": comp}
            keep_dirs, keep_lp, keep_lm = [], [], []
            for j, e in enumerate(dirs):
                lp, lm = losses[1 + 2 * j], losses[1 + 2 * j + 1]
                if bool(np.isfinite(lp)) and bool(np.isfinite(lm)):
                    keep_dirs.append(e)
                    keep_lp.append(lp)
                    keep_lm.append(lm)
            if keep_dirs and bool(np.isfinite(losses[0])):
                grad = es_gradient(keep_dirs, keep_lp, keep_lm, es_s)
                factors = center_mcdp_factors(
                    center, rg, tuple(fit_seeds), morphology)
                params_j = jax.numpy.asarray(center, dtype=jax.numpy.float32)
                updates, opt_state = transform.update(
                    jax.numpy.asarray(grad, dtype=jax.numpy.float32),
                    opt_state, params=params_j, loss=float(losses[0]),
                    key=key,
                    mcdp=jax.numpy.asarray(factors, dtype=jax.numpy.float32))
                center = np.clip(
                    np.asarray(np.asarray(params_j) + np.asarray(updates),
                               dtype=np.float64), 0.0, 10.0)
            else:
                # mcdp=None voids the epoch: gsdr_mcdp_transform.update
                # returns zero updates with state unchanged
                # (jaxfne/optim/gsdr.py:299-300).
                zero = jax.numpy.zeros_like(
                    jax.numpy.asarray(center, dtype=jax.numpy.float32))
                _, opt_state = transform.update(
                    zero, opt_state,
                    params=jax.numpy.asarray(center, dtype=jax.numpy.float32),
                    loss=float(losses[0]) if bool(np.isfinite(losses[0]))
                    else float("nan"),
                    key=key, mcdp=None)
            a_traj.append(float(opt_state[0].a))
            trajectory.append({"generation": gen, "best_loss": best["loss"],
                               "components": best["components"],
                               "es_clipped_coords": int(es_clipped)})
            if (gen + 1) % CKPT_EVERY == 0:
                write_ckpt(gen + 1)
                print("CKPT gen %d best %.4f wall %.1fs"
                      % (gen + 1, best["loss"], total_wall), flush=True)
            continue
        cands = [center.copy()]
        for _ in range(int(population) - 1):
            cands.append(np.clip(
                center + exploration * span * rng.standard_normal(n_params),
                0.0, 10.0))
        gen_best_loss, gen_best_g, gen_best_comp = best["loss"], None, None
        if batch:
            t0 = time.time()
            bloss, bcomp = candidate_losses_batch(
                cands, rg, y, fit_seeds=tuple(fit_seeds),
                morphology=morphology)
            bwall = time.time() - t0
            total_wall += bwall
            for i, (loss, comp) in enumerate(zip(bloss, bcomp)):
                finite = bool(np.isfinite(loss))
                print("g%dc%d loss %.4f (psd %.4f pen %.4f kap %.4f) "
                      "active %s wall %.1fs (batch)" % (
                          gen, i, loss, comp["mean_psd_term"],
                          comp["mean_penalty"], comp["mean_kappa_term"],
                          ",".join("%.2f" % t["active_frac"]
                                    for t in comp["trials"]),
                          bwall / len(cands)), flush=True)
                loss = loss if finite else float("inf")
                if loss < gen_best_loss:
                    gen_best_loss, gen_best_g, gen_best_comp = loss, cands[i], comp
        else:
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
    if mcdp:
        optimizer = {
            "name": "GSDR-MCDP center update (antithetic-ES gradient)",
            "mcdp_applied": True,
            "deselect_factor_consumed": True,
            "inner": {"name": "SDR", "learning_rate": 0.1,
                      "momentum": 0.4},
            "deselection_threshold": 3.0, "a_init": 0.0,
            "a_dynamic": True, "lambda_d": 1.0, "checkpoint_n": 5,
            "stochastic_scale": 1.0}
    else:
        optimizer = {
            "name": "lane delta-rule center update (gradient-free)",
            "alpha": alpha, "exploration": exploration,
            "mcdp_applied": False,
            "deselect_factor_consumed": False,
            "authors_optimizer_settings_not_used": {
                "deselect_factor": float(spec.deselect_factor),
                "inner_SDR_lr": 0.1, "inner_SDR_momentum": 0.4,
                "a_init": 0.0, "mcdp": True}}
    manifest = {
        "git_sha": sha,
        "optimizer": optimizer,
        "trainable": {"gAMPA_edges": rg.N_AMPA_EDGES,
                      "gGABAa_edges": rg.N_GABAA_GLOBAL_EDGES + rg.N_GABAA_LOCAL_EDGES,
                      "bounds_uS": [0.0, 10.0],
                      "paper_code_ref": "main_fig6.py:191-192,194-198,210-211,671-681"},
        "budget": {"authors_epochs": 500,
                   "paper_call": "main_fig6.py:684",
                   "lane_generations": int(generations),
                   "population": int(population)},
        "morphology": morphology,
        "batch": bool(batch),
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
    if mcdp:
        manifest["gradient_estimator"] = {
            "method": "antithetic evolution strategies (finite differences)",
            "k": int(population) // 2, "sigma": float(es_s),
            "deviation": "authors use a reverse-mode gradient through "
                         "their loss; ruled 2026-10-31"}
        manifest["a_trajectory"] = [float(a) for a in a_traj]
        manifest["es_clipped_coords"] = int(es_clipped_total)
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
    ap.add_argument("--mcdp", action="store_true",
                    help="update the center with gsdr_mcdp_transform on an "
                         "antithetic-ES gradient instead of the delta rule")
    ap.add_argument("--es-sigma", type=float, default=None,
                    help="ES sampling scale s (default exploration * span)")
    ap.add_argument("--batch", action="store_true",
                    help="evaluate each generation as one simulate_hh_batch "
                         "call per trial; pilots and sweeps only, losses within "
                         "rel 1e-2 of serial (default: serial candidate_loss)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_fit_500(args.out, generations=args.generations,
                population=args.population, seed=args.seed,
                y_labels_path=args.y_labels,
                fit_seeds=tuple(args.fit_seeds) if args.fit_seeds else FIT_SEEDS_DEFAULT,
                resume=args.resume, morphology=args.morphology,
                mcdp=args.mcdp, es_sigma=args.es_sigma, batch=args.batch)


if __name__ == "__main__":
    main()
