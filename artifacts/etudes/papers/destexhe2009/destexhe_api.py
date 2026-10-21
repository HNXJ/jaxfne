"""Destexhe 2009 replication, library-path runner (PORT_SPEC.md).

Builds the cortical self-sustained AI network (Fig. 6B, S2) from the jaxfne
public API and runs it through the AdEx route exactly as
``artifacts/etudes/caminos/caminos_api.py`` does: ``construct`` the model,
swap in the Destexhe cell parameters (``jaxfne.adex_destexhe``, a core
component), then call the model's compiled ``adex_kernel_args`` /
``adex_external_drive_kwargs`` through the core kernel
``jaxfne.emitters_adex.simulate_adex``. No scored run: outputs land in the
caller's directory, never under a ``scored`` path.

Every biophysical value comes from ``SOURCE_P1.md``; carriers, gaps and
fidelity notes are named in ``PORT_SPEC.md``. Units are jaxfne's
(mV, ms, pF, nS, pA).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts import lane_kit

_HERE = Path(__file__).resolve().parent

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# SOURCE_P1.md sections 1-3: the S2 cortical network (Fig. 6B).
DT_MS = 0.1
N_RS = 1600  # PY, weak adaptation (b = 5 pA)
N_FS = 400  # IN
N = N_RS + N_FS
P = 0.02  # all four blocks
TAU_E_MS = 5.0  # tau_E
TAU_I_MS = 10.0  # tau_I
REV_E_MV = 0.0
REV_I_MV = -80.0
GE_NS = 6.0  # excitatory quantum
GI_NS = 67.0  # inhibitory quantum
DELAY_MS = 0.1  # no paper delays; one kernel step (fidelity note F1)
KICK_MS = 50.0  # initiation window
KICK_RATE_HZ = 300.0  # 200-400 Hz (gap G5)
KICK_SOURCES = 1  # private sources per target cell
FREE_T_S = 10.0  # S2 free run scored for "alive at 10 s"

# Carrier labels on the adex route (PORT_SPEC.md): E carries RS, PV carries FS.
CARRIERS = ("E", "PV")
S2_MAPPING = {"E": "RSweak", "PV": "FS"}

RULES = (
    ("rs_rs", "E", "E", GE_NS, "ampa"),
    ("rs_fs", "E", "PV", GE_NS, "ampa"),
    ("fs_rs", "PV", "E", GI_NS, "gaba"),
    ("fs_fs", "PV", "PV", GI_NS, "gaba"),
)


def build_spec(n_rs=N_RS, n_fs=N_FS, seed=1, probability=P,
               rs_type="RSweak", strong_adaptation=False):
    """The S2 cortical network built only from the public API.

    ``strong_adaptation`` selects the Fig. 6A control (RS b = 40 pA); the
    default is the Fig. 6B weak-adaptation network (b = 5 pA).
    """
    import jaxfne as jtfne

    n = int(n_rs) + int(n_fs)
    mapping = {"E": ("RS" if strong_adaptation else rs_type), "PV": "FS"}
    frac_e = n_rs / n
    cfg = (
        jtfne.Configuration()
        .network(name="destexhe-cortex", n=n,
                 cell_types={"E": frac_e, "PV": 1.0 - frac_e})
        .emitter(family="adex", adex_decay="exp", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": TAU_E_MS, "reversal_mV": REV_E_MV})
        .mechanisms(name="gaba", kind="receptor",
                    params={"tau_ms": TAU_I_MS, "reversal_mV": REV_I_MV})
    )
    for name, pre, post, weight, mechanism in RULES:
        cfg = cfg.connections(
            name=name,
            source={"cell_type": pre},
            target={"cell_type": post},
            probability=float(probability),
            weight=weight,
            mechanism=mechanism,
            delay_ms=DELAY_MS,
            sampling="capped_bernoulli",
        )
    cfg = cfg.connectivity(edge_seed=int(seed))
    model = jtfne.construct(cfg)
    from jaxfne.adex_destexhe import apply_destexhe_params

    return apply_destexhe_params(model, mapping)


def kick_drive(seed, rate_hz=KICK_RATE_HZ):
    """The 50 ms initiation kick as ``Simulation(poisson_drive=...)`` (gap G5).

    Private sources at ``rate_hz`` with ge-sized kicks to the RS (E)
    population; FS cells get no drive (fidelity note F5: the kernel targets
    whole types; fractions via ``subset_drive_kwargs``).
    """
    return {
        "n_sources": KICK_SOURCES,
        "rate_hz": float(rate_hz),
        "weight_nS": GE_NS,
        "mechanism": "ampa",
        "target_cell_types": ["E"],
        "delay_ms": DELAY_MS,
        "seed": int(seed),
    }


def build_edges(n=N, seed=1):
    """Wiring of the S2 network read off the constructed core model.

    Returns the compiled edge arrays plus per-block possible-pair counts and
    realized densities. Capped-Bernoulli at p = 0.02, autapses excluded
    (gap G10).
    """
    import jaxfne as jtfne

    n_rs = int(round(0.8 * n))
    model = build_spec(n_rs=n_rs, n_fs=n - n_rs, seed=seed)
    st = model.static["adex"]
    labels = list(model.params["emitter"].labels)
    pre = [labels[i] for i in st["pre"].tolist()]
    post = [labels[i] for i in st["post"].tolist()]
    assert all(p != q for p, q in zip(st["pre"].tolist(), st["post"].tolist()))
    n_e = labels.count("E")
    n_pv = labels.count("PV")
    possible = {
        ("E", "E"): n_e * (n_e - 1),
        ("E", "PV"): n_e * n_pv,
        ("PV", "E"): n_pv * n_e,
        ("PV", "PV"): n_pv * (n_pv - 1),
    }
    realized = {k: 0 for k in possible}
    for a, b in zip(pre, post):
        realized[a, b] += 1
    density = {k: realized[k] / possible[k] for k in possible}
    return {
        "pre": st["pre"],
        "post": st["post"],
        "n": n,
        "labels": labels,
        "block_counts": possible,
        "block_density": density,
        "model": model,
    }


def single_cell(dtype, protocol="depol", dt_ms=DT_MS):
    """One Destexhe cell under a ±250 pA step (Fig. 1) through the core kernel.

    ``protocol`` is ``"depol"`` (+250 pA) or ``"hyper"`` (-250 pA), applied
    100-600 ms of a 1200 ms run. Returns the boolean spike vector.
    """
    import numpy as np

    from jaxfne.adex_destexhe import DESTEXHE_CELL_DEFAULTS
    from jaxfne.emitters_adex import AdExParams, simulate_adex
    import jax.numpy as jnp

    if dtype not in DESTEXHE_CELL_DEFAULTS:
        raise ValueError(f"unknown Destexhe cell type {dtype!r}")
    if protocol not in ("depol", "hyper"):
        raise ValueError(f"protocol must be 'depol' or 'hyper', got {protocol!r}")
    row = dict(DESTEXHE_CELL_DEFAULTS[dtype])
    fields = ("C_pF", "gL_nS", "EL_mV", "VT_mV", "Delta_mV", "Vreset_mV",
              "tauw_ms", "a_nS", "b_pA", "Tref_ms", "Vpeak_mV")
    cols = {k: jnp.asarray([row[k]], dtype=jnp.float32) for k in fields}
    params = AdExParams(labels=(dtype,), **cols)
    steps = int(round(1200.0 / dt_ms))
    lo, hi = int(round(100.0 / dt_ms)), int(round(600.0 / dt_ms))
    current = 250.0 if protocol == "depol" else -250.0
    stim = np.zeros((steps, 1), dtype=np.float32)
    stim[lo:hi, 0] = current
    out = simulate_adex(params, n_steps=steps, dt_ms=dt_ms,
                        receptor_reversal_mV=(), receptor_tau_ms=(),
                        decay="exp", I_ext_pA=stim, record_v=False)
    return np.asarray(out["spikes"])[:, 0]


def epsp_peak(weight_nS, dt_ms=DT_MS):
    """Peak first-EPSP (mV) on a resting RSweak cell for one 6/12 nS kick.

    Two RSweak cells, one edge, a 30 ms pre-cell kick; the peak is measured
    between the first and second pre spike, so the ratio tests the
    conductance jump, not summation.
    """
    import numpy as np

    from jaxfne.adex_destexhe import DESTEXHE_CELL_DEFAULTS
    from jaxfne.emitters_adex import AdExParams, simulate_adex
    import jax.numpy as jnp

    fields = ("C_pF", "gL_nS", "EL_mV", "VT_mV", "Delta_mV", "Vreset_mV",
              "tauw_ms", "a_nS", "b_pA", "Tref_ms", "Vpeak_mV")
    rows = [dict(DESTEXHE_CELL_DEFAULTS["RSweak"])] * 2
    cols = {k: jnp.asarray([r[k] for r in rows], dtype=jnp.float32) for k in fields}
    params = AdExParams(labels=("E", "E"), **cols)
    steps = 400
    stim = np.zeros((steps, 2), dtype=np.float32)
    stim[50:350, 0] = 400.0
    out = simulate_adex(params, n_steps=steps, dt_ms=dt_ms,
                        edge_pre=np.array([0]), edge_post=np.array([1]),
                        edge_weight_nS=np.array([float(weight_nS)]),
                        edge_receptor=np.array([0]),
                        edge_delay_steps=np.array([1]),
                        receptor_reversal_mV=(REV_E_MV,),
                        receptor_tau_ms=(TAU_E_MS,),
                        decay="exp", I_ext_pA=stim, record_v=True)
    import numpy as _np

    v = _np.asarray(out["V_mV"])
    pre = _np.flatnonzero(_np.asarray(out["spikes"])[:, 0])
    assert pre.size, "fixture must spike the pre cell"
    base = v[:50, 1].mean()
    return float(v[int(pre[0]) + 1:280, 1].max() - base)


def cv_isi(spikes, dt_ms):
    """Mean over spiking cells of sigma_ISI / mean_ISI (paper Eq. 3).

    Cells with fewer than 3 spikes carry no ISI variance and are excluded;
    the included fraction is reported alongside by the caller.
    """
    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    cvs = []
    for col in range(arr.shape[1]):
        t = np.flatnonzero(arr[:, col]).astype(float) * dt_ms
        if t.size < 3:
            continue
        isi = np.diff(t)
        if isi.mean() <= 0:
            continue
        cvs.append(float(isi.std() / isi.mean()))
    if not cvs:
        return float("nan")
    return float(np.mean(cvs))


def pairwise_cc(spikes, dt_ms, seed=0, bin_ms=5.0, n_pairs=500):
    """Mean Pearson correlation over disjoint pairs in 5 ms bins (Eq. 4)."""
    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    width = max(1, int(round(bin_ms / dt_ms)))
    cut = (arr.shape[0] // width) * width
    binned = arr[:cut].reshape(cut // width, width, arr.shape[1]).sum(axis=1)
    n = arr.shape[1]
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    order = order[: (len(order) // 2) * 2].reshape(-1, 2)
    order = order[:n_pairs]
    rs = []
    for i, j in order:
        x = binned[:, i].astype(float)
        y = binned[:, j].astype(float)
        if x.std() == 0 or y.std() == 0:
            continue
        rs.append(float(np.corrcoef(x, y)[0, 1]))
    if not rs:
        return float("nan")
    return float(np.mean(rs))


def mean_rate_hz(spikes, dt_ms):
    """Population mean firing rate (Hz) over the whole raster."""
    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    seconds = arr.shape[0] * dt_ms / 1000.0
    return float(arr.sum() / (arr.shape[1] * seconds))


_refuse_scored_dir = lane_kit.refuse_scored_dir
_refuse_existing = lane_kit.refuse_existing


def run_filename(seed, tag):
    """Spike-raster file name for a run."""
    return "spikes_seed%d_%s.npz" % (int(seed), tag)


def write_run(out_dir, *, seed, tag, spikes, dt_ms, meta_extra=None,
              signed=False):
    """Write the boolean raster plus the manifest the scorer reads."""
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = run_filename(seed, tag)
    sha = lane_kit.git_sha(_repo_root)
    prior = _refuse_existing(out, fname, sha, signed=signed)
    entries = {e["file"]: e for e in prior["runs"]} if prior else {}
    spikes = np.asarray(spikes, dtype=bool)
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, spikes=spikes, dt_ms=np.float64(dt_ms))
    os.replace(tmp, out / fname)
    entry = {"seed": int(seed), "tag": tag, "file": fname,
             "dtype": "float32", "dt_ms": float(dt_ms),
             "steps": int(spikes.shape[0]), "n": int(spikes.shape[1]),
             "git_sha": sha,
             "start_time": datetime.now(timezone.utc).isoformat()}
    if meta_extra:
        entry.update(meta_extra)
    entries[fname] = entry
    manifest = {"git_sha": sha,
                "runs": list(entries.values())}
    lane_kit.write_manifest(out, "manifest.json", manifest, sha=sha)
    return out / fname


def run(seed, tag, out_dir, *, free_t_s=FREE_T_S, n_rs=N_RS, n_fs=N_FS,
        signed=False, strong_adaptation=False):
    """Two-phase S2 run through the core AdEx route (caminos route-c pattern).

    Phase 1: 50 ms with the Poisson kick. Phase 2: ``free_t_s`` from phase 1's
    final V with no drive (w/g/ref restart; fidelity note F2). Writes the
    boolean raster and the manifest; refuses ``scored`` output unless signed.
    """
    import numpy as np

    _refuse_scored_dir(out_dir, signed)
    _refuse_existing(out_dir, run_filename(seed, tag),
                     lane_kit.git_sha(_repo_root), signed=signed)  # before the compute
    import jaxfne as jtfne
    from jaxfne._construct_adex import adex_external_drive_kwargs, adex_kernel_args
    from jaxfne.emitters_adex import simulate_adex

    start = time.time()
    model = build_spec(n_rs=n_rs, n_fs=n_fs, seed=seed,
                       strong_adaptation=strong_adaptation)
    sim = jtfne.Simulation(duration_ms=KICK_MS, dt_ms=DT_MS,
                           poisson_drive=kick_drive(seed))
    runtime_cfg = sim.resolved_runtime
    if str(runtime_cfg.actual_dtype) != "float32":
        raise SystemExit("error: destexhe runs are float32, runtime resolved %s"
                         % runtime_cfg.actual_dtype)
    kwargs = adex_kernel_args(model, sim.dt_ms, dtype=runtime_cfg.actual_dtype)
    drive_kwargs = adex_external_drive_kwargs(model, sim)
    kick = simulate_adex(**kwargs, **drive_kwargs, n_steps=sim.n_steps,
                         dt_ms=sim.dt_ms, record_v=False)
    import jax

    jax.block_until_ready(kick["spikes"])
    v0 = np.asarray(kick["V"], dtype=np.float32)
    free_steps = int(round(free_t_s * 1000.0 / DT_MS))
    free = simulate_adex(**kwargs, n_steps=free_steps, dt_ms=DT_MS,
                         V0_mV=v0, record_v=False)
    jax.block_until_ready(free["spikes"])
    spikes = np.asarray(free["spikes"])
    wall_s = time.time() - start
    write_run(out_dir, seed=seed, tag=tag, spikes=spikes, dt_ms=DT_MS,
              meta_extra={"free_t_s": float(free_t_s),
                          "strong_adaptation": bool(strong_adaptation),
                          "wall_s": float(wall_s)},
              signed=signed)
    return {"spikes": spikes, "wall_s": wall_s, "steps": free_steps}


def run_continuous(seed, tag, out_dir, *, free_t_s=FREE_T_S, n_rs=N_RS,
                   n_fs=N_FS, signed=False, strong_adaptation=False,
                   kick_fraction=1.0, kick_rate_hz=KICK_RATE_HZ,
                   v0_seed=None, ongoing=False, probability=P):
    """One continuous S2 run: the kick drive gated to the first 50 ms (F2 test).

    Same model, kick and free length as :func:`run`, but a single
    ``simulate_adex`` call over kick + free steps with the core
    ``ext_t_start_steps``/``ext_t_stop_steps`` drive window
    (``t_start_ms=0``/``t_stop_ms=50`` on the run's ``Simulation``), so
    adaptation ``w``, conductances ``g``, refractory ``ref`` and in-flight
    spikes carry through the kick-to-free boundary where :func:`run`
    restarts them. Writes the free portion only (kick steps dropped), so the
    scorer reads the same window as a two-phase run. ``kick_fraction`` (D, F5)
    keeps a seeded fraction of the RS kick targets; 1.0 is the
    whole-population kick. ``v0_seed`` (diagnostic only, not D) replaces the
    uniform EL start with a seeded uniform V0 in [-70, -50] mV to test
    attractor existence; None starts at EL. ``kick_fraction=0`` is the
    no-kick attractor probe (plain core run from V0, no drive; the first
    50 ms are still dropped as settle). ``ongoing=True`` (residual probe
    X_bg, not D) runs the Poisson drive on every step instead of windowing
    it to the first 50 ms.
    """
    import numpy as np

    _refuse_scored_dir(out_dir, signed)
    _refuse_existing(out_dir, run_filename(seed, tag),
                     lane_kit.git_sha(_repo_root), signed=signed)  # before the compute
    import jaxfne as jtfne
    from jaxfne._construct_adex import adex_external_drive_kwargs, adex_kernel_args
    from jaxfne.adex_destexhe import subset_drive_kwargs
    from jaxfne.emitters_adex import simulate_adex

    start = time.time()
    model = build_spec(n_rs=n_rs, n_fs=n_fs, seed=seed,
                       strong_adaptation=strong_adaptation,
                       probability=float(probability))
    runtime_cfg = jtfne.Simulation(
        duration_ms=KICK_MS, dt_ms=DT_MS,
        poisson_drive=kick_drive(seed, rate_hz=kick_rate_hz)).resolved_runtime
    if str(runtime_cfg.actual_dtype) != "float32":
        raise SystemExit("error: destexhe runs are float32, runtime resolved %s"
                         % runtime_cfg.actual_dtype)
    kwargs = adex_kernel_args(model, DT_MS, dtype=runtime_cfg.actual_dtype)
    kick_steps = int(round(KICK_MS / DT_MS))
    free_steps = int(round(free_t_s * 1000.0 / DT_MS))
    total_steps = kick_steps + free_steps
    frac = float(kick_fraction)
    if frac == 0.0:
        drive_kwargs = None
    else:
        drive = kick_drive(seed, rate_hz=kick_rate_hz)
        if not ongoing:
            drive = dict(drive, t_start_ms=0.0, t_stop_ms=KICK_MS)
        sim = jtfne.Simulation(duration_ms=KICK_MS + free_t_s * 1000.0,
                               dt_ms=DT_MS, poisson_drive=drive)
        if frac == 1.0:
            drive_kwargs = adex_external_drive_kwargs(model, sim)
        else:
            drive_kwargs = subset_drive_kwargs(model, sim, frac)
    if v0_seed is None:
        v0 = None
    else:
        rng = np.random.default_rng(int(v0_seed))
        v0 = rng.uniform(-70.0, -50.0, size=n_rs + n_fs).astype(np.float32)
    if drive_kwargs is None:
        out = simulate_adex(**kwargs,
                            n_steps=total_steps,
                            dt_ms=DT_MS, V0_mV=v0, record_v=False)
    else:
        out = simulate_adex(**kwargs, **drive_kwargs,
                            n_steps=total_steps,
                            dt_ms=DT_MS,
                            V0_mV=v0,
                            record_v=False)
    import jax

    jax.block_until_ready(out["spikes"])
    spikes = np.asarray(out["spikes"])[kick_steps:]
    wall_s = time.time() - start
    write_run(out_dir, seed=seed, tag=tag, spikes=spikes, dt_ms=DT_MS,
              meta_extra={"free_t_s": float(free_t_s),
                          "strong_adaptation": bool(strong_adaptation),
                          "continuous": True,
                          "kick_fraction": float(kick_fraction),
                          "kick_rate_hz": float(kick_rate_hz),
                          "v0_seed": None if v0_seed is None else int(v0_seed),
                          "ongoing": bool(ongoing),
                          "probability": float(probability),
                          "n_edges": int(model.static["adex"]["pre"].shape[0]),
                          "wall_s": float(wall_s)},
              signed=signed)
    return {"spikes": spikes, "wall_s": wall_s, "steps": free_steps,
            "n_edges": int(model.static["adex"]["pre"].shape[0])}


def build_parser():
    ap = argparse.ArgumentParser(description="Run the Destexhe 2009 cortical "
                                 "AI network through the jaxfne AdEx route.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="fig6b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--free-t-s", type=float, default=FREE_T_S)
    ap.add_argument("--n-rs", type=int, default=N_RS)
    ap.add_argument("--n-fs", type=int, default=N_FS)
    ap.add_argument("--strong-adaptation", action="store_true",
                    help="Fig. 6A control (RS b = 40 pA)")
    ap.add_argument("--signed", action="store_true",
                    help="allow a path containing 'scored' (signed runs only)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args.seed, args.tag, args.out, free_t_s=args.free_t_s,
        n_rs=args.n_rs, n_fs=args.n_fs, signed=args.signed,
        strong_adaptation=args.strong_adaptation)


if __name__ == "__main__":
    main()
