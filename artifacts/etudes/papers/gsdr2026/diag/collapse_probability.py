"""GSDR 2026 collapse probability on CPU: controlled perturbations at g* (no fitting).

CALIBRATION.md §6.8: on GPU about 1 call in 10 at g* (stim 0.6, seed 1)
collapses — all spikes early, silent in [500, 1000) ms, kappa 0.0, ~5.2 Hz.
CPU runs are bit-identical. This script asks how sensitive the collapse is
to small, controlled perturbations, and whether other noise seeds collapse.

One invocation = one process on CPU: it first reproduces ``run_trial``'s V
trace bit-identically (sha256) with its own trial builder (same kwargs to
``simulate_hh``), then runs two axes and appends one JSON line per run:

- axis P: per-cell initial-V offsets ``eps * N(0, 1)`` mV, eps in
  {1e-6, 1e-4, 1e-2}, draws from numpy seed d, d = 0..D-1, OU seed 1.
  The initial-V entry is ``simulate_hh``'s ``V0_mV`` (``None`` = rest at
  -70 mV; ``hh_kernel_args`` leaves it to the caller, ``_construct_hh.py``).
- axis S: no perturbation, OU seeds 1..D.

``collapsed`` = no spikes in [500, 1000) ms. Thirds split the full 1500 ms
trial into [0, 500) / [500, 1000) / [1000, 1500) ms spike counts.
"""

from __future__ import annotations

import argparse
import hashlib
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

import run_gsdr as rg  # noqa: E402

GSTAR_DEFAULT = ("/mnt/f/warehouse/gsdr/lane4f_runs_2026-10-14/gsdr_fit500/"
                 "g_star_uS_float32.npy")
OUT_DEFAULT = "/mnt/f/warehouse/gsdr/collapse_probability_2026-10-17.jsonl"

STIM_AMP = 0.6
SEED = 1
EPSILONS = (1e-6, 1e-4, 1e-2)
D_MIN, D_MAX = 10, 40

T_L_MS, T_R_MS = 500.0, 1000.0
THIRD_EDGES_MS = (0.0, 500.0, 1000.0, 1500.0)


def ms_index(n_steps, t_ms, dt_ms=rg.DT_MS):
    return max(0, min(int(t_ms / dt_ms), n_steps - 2))


def trial(gstar, *, ou_seed, v0_mv=None):
    """One g* trial, built exactly as ``run_gsdr.run_trial`` (same kwargs).

    ``ou_seed`` drives the OU noise key only (``g_override`` fixes the
    conductances, so ``g_seed`` is inert); ``v0_mv`` is ``None`` (rest) or
    a per-cell initial-V vector in mV passed as ``V0_mV`` to
    ``simulate_hh``. Returns ``(v, wall_s)`` with ``v`` float32.
    """
    import jax
    import numpy as np

    from jaxfne._construct_hh import construct_hh_model, hh_kernel_args
    from jaxfne.emitters_hh import simulate_hh

    n_steps = int(round(rg.T_MAX_MS / rg.DT_MS)) + 1
    cfg, _ = rg.build_spec(local_seed=1, edge_seed=0)
    model = construct_hh_model(cfg)
    kwargs = hh_kernel_args(model, rg.DT_MS)
    kwargs["edge_weight"] = (np.asarray(gstar, dtype=np.float32)
                             * np.float32(rg.US_TO_MSCM2)).astype(np.float32)
    ou_key = jax.random.fold_in(jax.random.PRNGKey(int(ou_seed)), 0)
    ou_tau, ou_sigma = rg.ou_hyperparams(ou_key)
    i_ext = rg.stimulus_matrix(STIM_AMP, n_steps=n_steps)
    t0 = time.time()
    out = simulate_hh(**kwargs, n_steps=n_steps, dt_ms=rg.DT_MS, I_ext=i_ext,
                      ou_tau=np.asarray(ou_tau), ou_sigma=np.asarray(ou_sigma),
                      ou_key=ou_key, noise_gain=1.0, record_v=True,
                      V0_mV=v0_mv)
    jax.block_until_ready(out["V_mV"])
    wall_s = time.time() - t0
    return np.asarray(out["V_mV"], dtype=np.float32), wall_s


def metrics(v):
    """Rate, active fraction, window spikes, thirds, kappa_fit for one V trace."""
    import numpy as np

    spikes = rg.detect_spikes(v)
    n_steps = v.shape[0]
    i0 = ms_index(n_steps, T_L_MS)
    i1 = max(i0 + 2, min(int(T_R_MS / rg.DT_MS), n_steps - 1))
    win = spikes[i0:i1, :]
    edges = [ms_index(n_steps, t) for t in THIRD_EDGES_MS]
    thirds = [int(spikes[edges[i]:edges[i + 1], :].sum()) for i in range(3)]
    rates = np.asarray(rg.firing_rates_hz(v, t_max_ms=rg.T_MAX_MS))
    return {
        "mean_rate_hz": float(np.mean(rates)),
        "active_frac": float((rates > 0).mean()),
        "kappa_fit": float(rg.kappa_of_onsets(win)),
        "n_spikes_window": int(win.sum()),
        "thirds": thirds,
    }


def build_parser():
    ap = argparse.ArgumentParser(
        description="GSDR collapse probability on CPU (no fitting).")
    ap.add_argument("--gstar", default=GSTAR_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--draws", type=int, default=D_MAX,
                    help="D draws per group (%d..%d)" % (D_MIN, D_MAX))
    ap.add_argument("--smoke", action="store_true",
                    help="anchor + one P row + one S row only")
    return ap


def main(argv=None):
    import numpy as np

    import jax

    args = build_parser().parse_args(argv)
    if args.smoke:
        n_draws = 1
    else:
        n_draws = int(args.draws)
        if not D_MIN <= n_draws <= D_MAX:
            raise ValueError("draws must be %d..%d, got %r"
                             % (D_MIN, D_MAX, args.draws))
    backend = str(jax.default_backend())
    if backend != "cpu":
        raise SystemExit("error: CPU only, backend is %s" % backend)
    with open(args.gstar, "rb") as f:
        gstar = np.asarray(np.load(f), dtype=np.float32)
    expect = (rg.N_AMPA_EDGES + rg.N_GABAA_GLOBAL_EDGES
              + rg.N_GABAA_LOCAL_EDGES)
    if gstar.shape != (expect,):
        raise ValueError("gstar must hold %d reference-uS values, got %s"
                         % (expect, gstar.shape,))

    from jaxfne.emitters_hh import HH_V_REST_MV

    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(line):
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    def run_row(axis, ou_seed, eps=None, draw=None, v0_mv=None):
        v, wall_s = trial(gstar, ou_seed=ou_seed, v0_mv=v0_mv)
        met = metrics(v)
        line = {
            "axis": axis,
            "eps_mv": None if eps is None else float(eps),
            "draw": draw,
            "ou_seed": int(ou_seed),
            "stim_amp": STIM_AMP,
            "backend": backend,
            "mean_rate_hz": met["mean_rate_hz"],
            "active_frac": met["active_frac"],
            "kappa_fit": met["kappa_fit"],
            "n_spikes_window": met["n_spikes_window"],
            "thirds": met["thirds"],
            "collapsed": bool(met["n_spikes_window"] == 0),
            "v_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            "wall_s": float(wall_s),
        }
        emit(line)
        print("%(a)s eps=%(e)s draw=%(d)s seed=%(s)d: rate %(r).2f Hz "
              "active %(af).2f k_fit %(k).4f n_win %(n)d thirds %(t)s "
              "collapsed %(c)s sha %(h)s wall %(w).1fs"
              % {"a": axis, "e": eps, "d": draw, "s": line["ou_seed"],
                 "r": line["mean_rate_hz"], "af": line["active_frac"],
                 "k": line["kappa_fit"], "n": line["n_spikes_window"],
                 "t": line["thirds"], "c": line["collapsed"],
                 "h": line["v_sha256"][:12], "w": line["wall_s"]},
              flush=True)
        return line

    ref_result, ref_blobs = rg.run_trial(
        STIM_AMP, g_seed=SEED, ou_seed=SEED, g_override=gstar)
    ref_v = np.asarray(ref_blobs["v"], dtype=np.float32)
    ref_sha = hashlib.sha256(ref_v.tobytes()).hexdigest()
    own_v, _ = trial(gstar, ou_seed=SEED)
    own_sha = hashlib.sha256(own_v.tobytes()).hexdigest()
    if own_sha != ref_sha:
        raise SystemExit("error: anchor mismatch: run_trial %s... != "
                         "own builder %s..." % (ref_sha[:12], own_sha[:12]))
    print("anchor OK: unperturbed CPU run reproduces run_trial %s... "
          "(rate %.2f Hz)" % (ref_sha[:12], ref_result["mean_rate_hz"]),
          flush=True)
    emit({"axis": "ref", "eps_mv": None, "draw": None, "ou_seed": SEED,
          "stim_amp": STIM_AMP, "backend": backend,
          "mean_rate_hz": float(ref_result["mean_rate_hz"]),
          "active_frac": float((np.asarray(ref_result["rates_hz"]) > 0).mean()),
          "kappa_fit": float(metrics(ref_v)["kappa_fit"]),
          "n_spikes_window": int(metrics(ref_v)["n_spikes_window"]),
          "thirds": metrics(ref_v)["thirds"],
          "collapsed": bool(metrics(ref_v)["n_spikes_window"] == 0),
          "v_sha256": ref_sha, "wall_s": float(ref_result["wall_s"])})

    n_cells = int(rg.N_CELLS)
    rest = float(HH_V_REST_MV)
    eps_list = list(EPSILONS) if not args.smoke else [EPSILONS[0]]
    for eps in eps_list:
        for d in range(n_draws):
            off = (np.asarray(np.random.default_rng(d).standard_normal(n_cells),
                              dtype=np.float32) * np.float32(eps))
            v0 = (np.full((n_cells,), np.float32(rest), dtype=np.float32)
                  + off).astype(np.float32)
            run_row("P", SEED, eps=float(eps), draw=int(d), v0_mv=v0)
    for s in range(1, n_draws + 1):
        if args.smoke and s > 1:
            break
        run_row("S", s)
    print("done: appended anchor + %d P rows + %d S rows to %s"
          % (len(eps_list) * n_draws, n_draws if not args.smoke else 1,
             args.out), flush=True)


if __name__ == "__main__":
    main()
