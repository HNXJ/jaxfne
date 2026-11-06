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

Outputs land in the caller's directory (``artifacts/_temp/gsdr2026/``), never under
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
                   morphology="single", t_max_ms=None):
    """Fig. 6 batch loss of one conductance vector (reference-uS)."""
    import numpy as np

    base_psd = np.asarray(y["baseline_psd"], dtype=np.float64)
    stim_psd = np.asarray(y["stim_psd"], dtype=np.float64)
    trials = [(0.0, BASE_OU_SEED, base_psd, float(y["kappa_baseline"]))]
    trials += [(STIM_AMP, s, stim_psd, float(y["kappa_stim"])) for s in fit_seeds]
    psd_terms, pens, kappas, comp = [], [], [], []
    t_kw = {} if t_max_ms is None else {"t_max_ms": float(t_max_ms)}
    for amp, ou, lab, ky in trials:
        result, blobs = rg.run_trial(amp, g_seed=ou, ou_seed=ou,
                                     g_override=np.asarray(g_us, dtype=np.float32),
                                     morphology=morphology, **t_kw)
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


def batch_trial_vs(g_us_list, rg, amp, ou_seed, *, morphology="single",
                   t_max_ms=None):
    """Stacked V traces of candidates for one (amp, ou_seed) trial.

    Same construction as ``rg.run_trial`` (spec seeds, conversions,
    stimulus, OU draws) with only the conductance vector differing per
    candidate, run as one ``simulate_hh_batch`` call. Returns ``(B,
    n_steps, N)`` float32 in input order (``N`` cells at the record
    compartment on the reference morphology). Topology, morphology and
    ``n_steps`` are shared, so a candidate that changed any of them is
    refused by the batch kernel rather than run here.
    """
    import numpy as np

    import jax

    import jaxfne as jtfne
    from jaxfne._construct_hh import construct_hh_model, hh_kernel_args
    from jaxfne.emitters_hh import simulate_hh_batch

    if morphology not in ("single", "reference"):
        raise ValueError(
            "morphology must be 'single' or 'reference'; got %r" % (morphology,))
    gs = [np.asarray(g, dtype=np.float32) for g in g_us_list]
    if not gs:
        raise ValueError("g_us_list must hold at least one candidate")
    t = rg.T_MAX_MS if t_max_ms is None else float(t_max_ms)
    n_steps = int(round(t / rg.DT_MS)) + 1
    ou_key = jax.random.fold_in(jax.random.PRNGKey(int(ou_seed)), 0)
    ou_tau, ou_sigma = rg.ou_hyperparams(ou_key)
    if morphology == "reference":
        from jaxfne.emitters_hh import reference_4comp_morphology

        cfg, _ = rg.build_spec(weight_unit="uS")
        model = construct_hh_model(
            cfg, morphology=reference_4comp_morphology(),
            edge_pre_comp=0, edge_post_comp=3, stim_comp=2, record_comp=0,
            weight_unit="uS", gate_init=0.2)
        kwargs = hh_kernel_args(model, rg.DT_MS)
        stacked = np.stack(gs, axis=0)
        n_e = int(np.asarray(kwargs["edge_pre"]).shape[0])
        if stacked.shape != (len(gs), n_e):
            raise ValueError(
                "each candidate must hold %d reference-uS values, got %s"
                % (n_e, (stacked.shape,)))
        out = simulate_hh_batch(
            kwargs["params"], n_steps=n_steps, dt_ms=rg.DT_MS,
            batch={"edge_weight_uS": stacked},
            edge_pre=kwargs["edge_pre"], edge_post=kwargs["edge_post"],
            edge_receptor=kwargs["edge_receptor"],
            receptor_e_rev=kwargs["receptor_e_rev"],
            receptor_tau_r=kwargs["receptor_tau_r"],
            receptor_tau_d=kwargs["receptor_tau_d"],
            receptor_slope=kwargs["receptor_slope"],
            receptor_vth=kwargs["receptor_vth"],
            receptor_s0=kwargs["receptor_s0"],
            morphology=kwargs["morphology"],
            edge_pre_comp=kwargs["edge_pre_comp"],
            edge_post_comp=kwargs["edge_post_comp"],
            stim_comp=kwargs["stim_comp"], record_comp=kwargs["record_comp"],
            gate_init=kwargs.get("gate_init"),
            ou_tau=np.asarray(ou_tau), ou_sigma=np.asarray(ou_sigma),
            ou_key=ou_key,
            I_ext_nA=rg.stimulus_matrix(amp, n_steps=n_steps, unit="nA"),
            record_v=True)
        jax.block_until_ready(out["V_mV"])
        return np.ascontiguousarray(
            np.asarray(out["V_mV"], dtype=np.float32)[:, :, :, 0])
    cfg, _ = rg.build_spec()
    model = jtfne.construct(cfg)
    kwargs = hh_kernel_args(model, rg.DT_MS)
    stacked = np.stack(
        [(g * np.float32(rg.US_TO_MSCM2)).astype(np.float32) for g in gs],
        axis=0)
    n_e = int(np.asarray(kwargs["edge_pre"]).shape[0])
    if stacked.shape != (len(gs), n_e):
        raise ValueError(
            "each candidate must hold %d reference-uS values, got %s"
            % (n_e, (stacked.shape,)))
    out = simulate_hh_batch(
        kwargs["params"], n_steps=n_steps, dt_ms=rg.DT_MS,
        batch={"edge_weight": stacked},
        edge_pre=kwargs["edge_pre"], edge_post=kwargs["edge_post"],
        edge_receptor=kwargs["edge_receptor"],
        receptor_e_rev=kwargs["receptor_e_rev"],
        receptor_tau_r=kwargs["receptor_tau_r"],
        receptor_tau_d=kwargs["receptor_tau_d"],
        receptor_slope=kwargs["receptor_slope"],
        receptor_vth=kwargs["receptor_vth"],
        receptor_s0=kwargs["receptor_s0"],
        ou_tau=np.asarray(ou_tau), ou_sigma=np.asarray(ou_sigma),
        ou_key=ou_key,
        I_ext=rg.stimulus_matrix(amp, n_steps=n_steps), record_v=True)
    jax.block_until_ready(out["V_mV"])
    return np.ascontiguousarray(np.asarray(out["V_mV"], dtype=np.float32))


def candidate_losses_batch(g_us_list, rg, y, *, fit_seeds=FIT_SEEDS_DEFAULT,
                           morphology="single", t_max_ms=None):
    """Fig. 6 batch losses of a candidate list through ``simulate_hh_batch``.

    One batched call per trial (same trial list and loss code as
    ``candidate_loss``); returns ``(losses, comps)`` in input order.
    """
    import time

    import numpy as np

    gs = list(g_us_list)
    base_psd = np.asarray(y["baseline_psd"], dtype=np.float64)
    stim_psd = np.asarray(y["stim_psd"], dtype=np.float64)
    trials = [(0.0, BASE_OU_SEED, base_psd, float(y["kappa_baseline"]))]
    trials += [(STIM_AMP, s, stim_psd, float(y["kappa_stim"])) for s in fit_seeds]
    b = len(gs)
    psd_terms = [[] for _ in range(b)]
    pens = [[] for _ in range(b)]
    kappas = [[] for _ in range(b)]
    comp = [[] for _ in range(b)]
    for amp, ou, lab, ky in trials:
        t0 = time.time()
        vb = np.asarray(batch_trial_vs(gs, rg, amp, ou, morphology=morphology,
                                       t_max_ms=t_max_ms))
        wall = time.time() - t0
        for i in range(b):
            v = np.ascontiguousarray(vb[i])
            z_psd = np.asarray(rg.psd_mean_voltage(v), dtype=np.float64)
            psd_terms[i].append(rg.psd_loss(z_psd, lab))
            rates = rg.firing_rates_hz(
                v, t_max_ms=(rg.T_MAX_MS if t_max_ms is None
                             else float(t_max_ms)))
            pens[i].append(rg.rate_penalty(rates))
            spikes = rg.detect_spikes(v)
            n_steps = v.shape[0]
            i0 = int(0.0 / rg.DT_MS) if amp == 0.0 else int(500.0 / rg.DT_MS)
            i1 = int(200.0 / rg.DT_MS) if amp == 0.0 else int(1000.0 / rg.DT_MS)
            i0 = max(0, min(i0, n_steps - 2))
            i1 = max(i0 + 2, min(i1, n_steps - 1))
            k = rg.kappa_of_onsets(spikes[i0:i1, :])
            kappas[i].append(abs(float(k) - ky))
            comp[i].append({"amp": amp, "ou": ou,
                            "mean_rate": float(np.mean(rates)),
                            "active_frac": float((np.asarray(rates) > 0).mean()),
                            "kappa": float(k), "wall_s": float(wall)})
    losses = [float(np.mean(psd_terms[i]) * rg.PSD_WEIGHT
                    + np.mean(pens[i]) * rg.RATE_WEIGHT
                    + np.mean(kappas[i]) * KAPPA_WEIGHT)
              for i in range(b)]
    comps = [{"mean_psd_term": float(np.mean(psd_terms[i])),
              "mean_penalty": float(np.mean(pens[i])),
              "mean_kappa_term": float(np.mean(kappas[i])),
              "trials": comp[i]} for i in range(b)]
    return losses, comps


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
