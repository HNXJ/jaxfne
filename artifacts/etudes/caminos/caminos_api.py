"""CAMINOS P6 library-path runner (PREREG_P6.md, signed v2).

Builds the CAMINOS E-PV-SOM network only from the jaxfne public API and runs
it through B0 route (c): the model's compiled ``adex_kernel_args`` /
``adex_external_drive_kwargs`` called through the ADVANCED kernel
``jaxfne.emitters_adex.simulate_adex`` with ``record_v=False``. No scored run:
outputs land in the caller's directory, never under a ``scored`` path.

Every biophysical value comes from ``PORT_SPEC.md`` Tables 1-2; nothing is
invented. Units are jaxfne's (mV, ms, pF, nS, pA).
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

_repo_root = _HERE
while _repo_root.parent != _repo_root and not (_repo_root / "pyproject.toml").exists():
    _repo_root = _repo_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# PORT_SPEC.md Table 2 and Methods text.
NE = 8000
NPV = 750
NSOM = 750
N = NE + NPV + NSOM
DT_MS = 0.1
DELAY_MS = 2.0  # Table 2, tau_d, all synapses
TAU_E_MS = 2.0  # Table 2, tau_E
TAU_I_MS = 8.0  # Table 2, tau_I
REV_E_MV = 0.0  # Table 2, E_rev (E)
REV_I_MV = -80.0  # Table 2, E_rev (I)
Q_E_E = 2.0  # Table 2, Q E -> E (nS)
Q_E_INH = 3.0  # Table 2, Q E -> PV, SOM (nS)
Q_INH = 5.0  # Table 2, Q PV, SOM -> E, PV (nS)
Q_EXT = 2.5  # Table 2, Q Ext -> E, PV (nS)
EXT_RATE_HZ = 4.0  # Methods, rate per external source
DT_S = 1e-4  # Methods, Euler step in seconds

# The seven realized blocks: PV -> SOM and SOM -> SOM are absent in the
# reference model, so no rules are declared for them.
RULES = (
    ("e_e", "E", "E", Q_E_E, "ampa"),
    ("e_pv", "E", "PV", Q_E_INH, "ampa"),
    ("e_som", "E", "SOM", Q_E_INH, "ampa"),
    ("pv_e", "PV", "E", Q_INH, "gaba"),
    ("pv_pv", "PV", "PV", Q_INH, "gaba"),
    ("som_e", "SOM", "E", Q_INH, "gaba"),
    ("som_pv", "SOM", "PV", Q_INH, "gaba"),
)

POPS = ("E", "PV", "SOM")


def ext_source_count(p):
    """Private Poisson sources per E and PV cell: ``round(p * NE)`` (PREREG_P6.md)."""
    return int(round(float(p) * NE))


def drive_spec(p, seed):
    """The ``Simulation(poisson_drive=...)`` dict for the CAMINOS drive."""
    return {
        "n_sources": ext_source_count(p),
        "rate_hz": EXT_RATE_HZ,
        "weight_nS": Q_EXT,
        "mechanism": "ampa",
        "target_cell_types": ["E", "PV"],
        "delay_ms": DELAY_MS,
        "seed": int(seed),
    }


def build_spec(p, seed):
    """The CAMINOS network built only from the public API (PREREG_P6.md table)."""
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .network(name="caminos", n=N,
                 cell_types={"E": NE / N, "PV": NPV / N, "SOM": NSOM / N})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
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
            probability=float(p),
            weight=weight,
            mechanism=mechanism,
            delay_ms=DELAY_MS,
            sampling="bernoulli",
        )
    return cfg.connectivity(edge_seed=int(seed))


def build_sim(p, seed, T_s):
    """The ``Simulation`` for a run: dt 0.1 ms, the ``drive_spec`` drive.

    ``Simulation(seed=)`` stays at its default: the route refuses it, and the
    run's randomness comes from ``connectivity(edge_seed=)`` and the drive
    seed (PREREG_P6.md).
    """
    import jaxfne as jtfne

    return jtfne.Simulation(duration_ms=float(T_s) * 1000.0, dt_ms=DT_MS,
                            poisson_drive=drive_spec(p, seed))


def spikes_to_counts(spikes, sizes):
    """Per-step population counts ``[steps, 3]`` in [E | PV | SOM] order.

    ``spikes`` is a boolean ``[steps, N]`` raster whose first ``sizes["E"]``
    columns are E, next ``sizes["PV"]`` PV, rest SOM (the ``build_spec``
    label order). Returns int32, the port's count-array format.
    """
    import numpy as np

    arr = np.asarray(spikes, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("spikes must be [steps, N], got shape %r" % (arr.shape,))
    n_e, n_pv, n_som = int(sizes["E"]), int(sizes["PV"]), int(sizes["SOM"])
    if arr.shape[1] != n_e + n_pv + n_som:
        raise ValueError("spikes has %d cells but sizes sum to %d"
                         % (arr.shape[1], n_e + n_pv + n_som))
    out = np.empty((arr.shape[0], 3), dtype=np.int32)
    out[:, 0] = arr[:, :n_e].sum(axis=1)
    out[:, 1] = arr[:, n_e:n_e + n_pv].sum(axis=1)
    out[:, 2] = arr[:, n_e + n_pv:].sum(axis=1)
    return out


def _refuse_scored_dir(out_dir, signed=False):
    if signed:
        return
    parts = Path(out_dir).resolve().parts
    if any("scored" in part.lower() for part in parts):
        raise SystemExit("error: refusing to write under a directory containing "
                         "'scored': %s" % out_dir)


def git_sha():
    """Git SHA via ``git rev-parse HEAD`` at the repository root."""
    out = subprocess.run(["git", "-C", str(_repo_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip()


def run_filename(seed, p):
    """Count-array file name, in the ``run_caminos.py`` style."""
    return "counts_seed%d_p%g_float32_dt1e-04.npy" % (int(seed), float(p))


def _refuse_existing(out_dir, fname, sha):
    """Refuse a directory holding another code SHA or this run's file; return its manifest."""
    out = Path(out_dir)
    prior = None
    if (out / "manifest.json").exists():
        prior = json.loads((out / "manifest.json").read_text())
        if prior.get("git_sha") != sha:
            raise SystemExit("error: %s holds runs from code %s; this run is %s (one code SHA "
                             "per directory)" % (out, prior.get("git_sha"), sha))
        if any(e["file"] == fname for e in prior["runs"]):
            raise SystemExit("error: refusing to overwrite %s" % (out / fname))
    if (out / fname).exists():
        raise SystemExit("error: refusing to overwrite %s" % (out / fname))
    return prior


def write_run(out_dir, *, p, seed, T_s, counts, sizes, wall_s, start_time, sha):
    """Write the count array plus the manifest ``caminos_analysis.score`` reads."""
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = run_filename(seed, p)
    prior = _refuse_existing(out, fname, sha)
    entries = {e["file"]: e for e in prior["runs"]} if prior else {}
    counts = np.asarray(counts, dtype=np.int32)
    if counts.ndim != 2 or counts.shape[1] != 3:
        raise ValueError("counts must be [steps, 3], got shape %r" % (counts.shape,))
    tmp = out / (fname + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, counts)
    os.replace(tmp, out / fname)
    entry = {"seed": int(seed), "p": float(p), "file": fname,
             "dtype": "float32", "dt": DT_S, "T": float(T_s),
             "pilot": False, "sizes": dict(sizes),
             "indegree": "bernoulli", "ext": "private",
             "step_order": "default", "ext_sources": "pNE",
             "ext_sources_n": ext_source_count(p),
             "backend": "edge_list",
             "git_sha": sha, "start_time": start_time,
             "wall_s": float(wall_s), "steps": int(counts.shape[0])}
    entries[fname] = entry
    manifest = {"git_sha": sha, "start_time": prior["start_time"] if prior else start_time,
                "runs": list(entries.values())}
    try:
        import jax

        manifest["jax_version"] = jax.__version__
        manifest["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    mtmp = out / "manifest.json.tmp"
    mtmp.write_text(json.dumps(manifest, indent=2))
    os.replace(mtmp, out / "manifest.json")
    return out / fname


def run(p, seed, T_s, out_dir, signed=False):
    """B0 route (c): construct the model, run the kernel with ``record_v=False``.

    Writes the port's per-step population count array and ``manifest.json``
    exactly as ``caminos_analysis.score`` reads them. Refuses any ``out_dir``
    with a path part containing "scored" unless ``signed``.
    """
    import numpy as np

    _refuse_scored_dir(out_dir, signed)
    _refuse_existing(out_dir, run_filename(seed, p), git_sha())  # before the compute
    import jaxfne as jtfne
    from jaxfne._construct_adex import adex_external_drive_kwargs, adex_kernel_args
    from jaxfne.emitters_adex import simulate_adex

    start_time = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    model = jtfne.construct(build_spec(p, seed))
    sim = build_sim(p, seed, T_s)
    runtime_cfg = sim.resolved_runtime
    if str(runtime_cfg.actual_dtype) != "float32":
        raise SystemExit("error: P6 runs are float32, runtime resolved %s"
                         % runtime_cfg.actual_dtype)
    kwargs = adex_kernel_args(model, sim.dt_ms, dtype=runtime_cfg.actual_dtype)
    drive_kwargs = adex_external_drive_kwargs(model, sim)
    out = simulate_adex(**kwargs, **drive_kwargs, n_steps=sim.n_steps,
                        dt_ms=sim.dt_ms, record_v=False)
    import jax

    jax.block_until_ready(out["spikes"])
    labels = tuple(model.params["emitter"].labels)
    sizes = {t: labels.count(t) for t in POPS}
    counts = spikes_to_counts(np.asarray(out["spikes"]), sizes)
    wall_s = time.time() - t0
    write_run(out_dir, p=p, seed=seed, T_s=T_s, counts=counts, sizes=sizes,
              wall_s=wall_s, start_time=start_time, sha=git_sha())
    return {"counts": counts, "sizes": sizes, "steps": sim.n_steps,
            "wall_s": wall_s}


def build_parser():
    ap = argparse.ArgumentParser(description="Run the CAMINOS network through the "
                                 "jaxfne public API (PREREG_P6.md B0 route (c)).")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--T", type=float, required=True,
                    help="duration in seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--signed", action="store_true",
                    help="allow a path containing 'scored' (signed P6 runs only)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args.p, args.seed, args.T, args.out, signed=args.signed)


if __name__ == "__main__":
    main()
