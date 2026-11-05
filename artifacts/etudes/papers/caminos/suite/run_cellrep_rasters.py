"""Paper-scale Cell Reports Figs 3/4 raster + LFP trace runner (suite lane C).

Usage: python run_cellrep_rasters.py <outdir> <seeds, e.g. 1,2,3>
           [--sizes NE,NPV,NSOM] [--conditions a,b] [--smoke-drive]

Nine conditions per seed (baseline, SOM silenced at 0.5 and 1.0, PV silenced
at 0.5, 0.75 and 1.0, SOM replaced at 0.5, PV replaced at 0.5 and 1.0) run
through the same ``cellrep_figures`` calls as ``run_figA.py``, each with
``raster_window=(T_s - 0.5, T_s)``. Replacement runs at the paired-baseline
group rate: the mean over all given seeds' baselines, as ``run_figA.py``
computes it (Figure 4 protocol, mmc2 1772-1774). Env ``FIGA_T_S`` sets the
run length in seconds (default 12; the estimators need T_s > 2, the fixed
2 s discard of ``caminos_analysis.kept_slice``). ``FIGA_PARAM_SET`` must be
``cellrep`` (anything else is refused). ``--conditions`` runs a subset of
the nine conditions (unknown names are refused). ``--sizes`` overrides the
network sizes; ``--smoke-drive`` is a test-only flag adding the documented
small-net drive (``ext_sources=100``, ``som_ext_rate_hz=4.0``); without it
the calls match ``run_figA.py`` exactly. Seeding comes only from ``seed``.

Writes ``seed<s>/<condition>.npz`` per seed and condition: ``raster`` bool
over the last 0.5 s, ``type_slices``, ``rate_hz`` per 1 ms bin in Hz per
cell, ``f_pk_hz``, ``band_power_25_45``, ``max_e_rate_hz``, plus the
Cell Reports extras ``lfp`` (the ref. 42 proxy over the same window, sliced
from the whole-run ``lfp`` of :func:`cellrep_figures.run_cell`, whose
whole-run smoothing leaves no truncated kernel at the window edges) and
``t0_s`` (the window start). ``manifest.json`` records the seeds, T_s,
param set, git SHA, backend, wall time, the baseline/group replacement
rates and the sha256 of every file. An existing non-empty outdir is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

def _repo_root(start: Path) -> Path:
    """Nearest ancestor that holds pyproject.toml (P-025: the checkout's jaxfne, never site-packages)."""
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError(f"no pyproject.toml above {start}")


_HERE = Path(__file__).resolve().parent
_CAMINOS = _HERE.parent
_REPO_ROOT = _repo_root(_HERE)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_CAMINOS) not in sys.path:
    sys.path.insert(0, str(_CAMINOS))

import cellrep_figures as cf  # noqa: E402
import jax  # noqa: E402

assert "site-packages" not in cf.__file__, "run_cellrep_rasters must run the checkout's cellrep_figures"

CONDITIONS = [
    "baseline",
    "silence_som_0.5",
    "silence_som_1.0",
    "silence_pv_0.5",
    "silence_pv_0.75",
    "silence_pv_1.0",
    "replace_som_0.5",
    "replace_pv_0.5",
    "replace_pv_1.0",
]

_SILENCE_OF = {
    "silence_pv_0.5": ("PV", 0.5),
    "silence_pv_0.75": ("PV", 0.75),
    "silence_pv_1.0": ("PV", 1.0),
    "silence_som_0.5": ("SOM", 0.5),
    "silence_som_1.0": ("SOM", 1.0),
}

_REPLACE_OF = {
    "replace_pv_0.5": ("PV", 0.5),
    "replace_pv_1.0": ("PV", 1.0),
    "replace_som_0.5": ("SOM", 0.5),
}

WINDOW_S = 0.5  # s, raster and LFP window at the end of the run
BIN_S = 1e-3  # s, rate_hz bins


def repo_root():
    """Repository root (directory holding ``pyproject.toml``)."""
    root = _HERE
    while root.parent != root and not (root / "pyproject.toml").exists():
        root = root.parent
    return root


def git_sha():
    """Git SHA via ``git -C <repo root> rev-parse HEAD``."""
    out = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("git rev-parse HEAD failed: %s" % out.stderr.strip())
    return out.stdout.strip()


def _parse_sizes(text):
    try:
        ne, npv, nsom = (int(x) for x in text.split(","))
    except (TypeError, ValueError):
        raise SystemExit("error: --sizes must be 'NE,NPV,NSOM' of positive ints, got %r" % (text,))
    if min(ne, npv, nsom) < 1:
        raise SystemExit("error: --sizes must be 'NE,NPV,NSOM' of positive ints, got %r" % (text,))
    return {"NE": ne, "NPV": npv, "NSOM": nsom}


def parse_args(argv):
    """Split ``[outdir, seeds, ...flags]`` into outdir, seeds, sizes, conditions, smoke-drive."""
    if len(argv) < 2:
        raise SystemExit(__doc__)
    outdir, seed_txt = argv[0], argv[1]
    try:
        seeds = [int(s) for s in seed_txt.split(",")]
    except (TypeError, ValueError):
        raise SystemExit("error: seeds must be comma-separated ints, got %r" % (seed_txt,))
    if len(set(seeds)) != len(seeds):
        raise SystemExit("error: duplicate seed in %r" % (seed_txt,))
    sizes = None
    conditions = list(CONDITIONS)
    smoke_drive = False
    rest = list(argv[2:])
    i = 0
    while i < len(rest):
        flag = rest[i]
        if flag == "--sizes":
            if i + 1 >= len(rest):
                raise SystemExit("error: --sizes needs a value, got nothing")
            sizes = _parse_sizes(rest[i + 1])
            i += 2
        elif flag.startswith("--sizes="):
            sizes = _parse_sizes(flag.split("=", 1)[1])
            i += 1
        elif flag == "--conditions":
            if i + 1 >= len(rest):
                raise SystemExit("error: --conditions needs a value, got nothing")
            conditions = _parse_conditions(rest[i + 1])
            i += 2
        elif flag.startswith("--conditions="):
            conditions = _parse_conditions(flag.split("=", 1)[1])
            i += 1
        elif flag == "--smoke-drive":
            smoke_drive = True
            i += 1
        else:
            raise SystemExit("error: unknown flag %r" % (flag,))
    return outdir, seeds, sizes, conditions, smoke_drive


def _parse_conditions(text):
    """Condition names from a comma-separated list; unknown names are refused."""
    names = [x for x in (s.strip() for s in text.split(",")) if x]
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        raise SystemExit("error: unknown condition(s) %s; valid: %s"
                         % (unknown, ", ".join(CONDITIONS)))
    if not names:
        raise SystemExit("error: --conditions got an empty list")
    return names


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _baseline_rate_hz(res, pop):
    """Kept-window mean rate of ``pop`` in one baseline run (``run_figA.py`` lines 95-101)."""
    keep = np.asarray(res["counts_kept"])[:, {"E": 0, "PV": 1, "SOM": 2}[pop]]
    return float(keep.sum()) / (int(res["sizes"][pop]) * float(res["dt_s"]) * keep.shape[0])


def _condition_results(seed, baseline, group_rate, t_s, win, extra, conditions):
    """The requested condition dicts of one seed through the ``run_figA.py`` call pattern.

    ``group_rate`` is the paired-baseline mean rate over all seeds of the run
    (``run_figA.py`` lines 95-101; mmc2 1772-1774), shared by every seed.
    """
    out = {}
    if "baseline" in conditions:
        out["baseline"] = baseline
    for name in conditions:
        if name in out:
            continue
        if name in _SILENCE_OF:
            pop, frac = _SILENCE_OF[name]
            out[name] = cf.silence_sweep(pop, [frac], [seed], T_s=t_s,
                                         raster_window=win, **extra)[0]
        else:
            pop, frac = _REPLACE_OF[name]
            out[name] = cf.replace_sweep(pop, [frac], [seed], rate_hz=group_rate[pop],
                                         T_s=t_s, raster_window=win, **extra)[0]
    return out


def _window_indices(res):
    """(first, end) step indices of the last-0.5 s window, as ``_raster_rows`` computes them."""
    dt = float(res["dt_s"])
    t0 = float(res["T_s"]) - WINDOW_S
    return int(round(t0 / dt)), int(round(float(res["T_s"]) / dt))


def _condition_npz(res):
    """On-disk arrays for one condition: raster, slices, binned rates, scalars, window LFP."""
    raster = np.asarray(res["raster_window"], dtype=bool)
    sizes = res["sizes"]
    ne, npv = int(sizes["E"]), int(sizes["PV"])
    n_tot = ne + npv + int(sizes["SOM"])
    assert raster.shape[1] == n_tot, (raster.shape, sizes)
    dt = float(res["dt_s"])
    bin_steps = int(round(BIN_S / dt))
    if bin_steps < 1 or raster.shape[0] % bin_steps:
        raise ValueError("raster steps %d do not split into %g ms bins at dt=%g s"
                         % (raster.shape[0], BIN_S * 1000.0, dt))
    n_bins = raster.shape[0] // bin_steps
    blocked = raster.reshape(n_bins, bin_steps, n_tot).sum(axis=1).astype(float)
    bounds = (0, ne, ne + npv, n_tot)
    rate = np.empty((n_bins, 3))
    for i in range(3):
        n_cells = bounds[i + 1] - bounds[i]
        rate[:, i] = blocked[:, bounds[i]:bounds[i + 1]].sum(axis=1) / (n_cells * BIN_S)
    i0, i1 = _window_indices(res)
    lfp = np.asarray(res["lfp"], dtype=float)[i0:i1]
    assert lfp.shape == (raster.shape[0],), (lfp.shape, raster.shape)
    return {
        "raster": raster,
        "type_slices": np.array([[0, ne], [ne, ne + npv], [ne + npv, n_tot]]),
        "rate_hz": rate,
        "f_pk_hz": np.asarray(float(res["f_pk_hz"])),
        "band_power_25_45": np.asarray(float(res["band_power_25_45"])),
        "max_e_rate_hz": np.asarray(float(res["max_e_rate_hz"])),
        "lfp": lfp,
        "t0_s": np.asarray(float(res["T_s"]) - WINDOW_S),
    }


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    outdir, seeds, sizes, conditions, smoke_drive = parse_args(list(argv))
    t_s = float(os.environ.get("FIGA_T_S", "12"))
    param_set = os.environ.get("FIGA_PARAM_SET", "cellrep")
    if param_set != "cellrep":
        raise SystemExit("FIGA_PARAM_SET must be cellrep, got %r" % (param_set,))
    cf.PARAM_SET = "cellrep"
    out = Path(outdir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing non-empty outdir %s" % outdir)
    out.mkdir(parents=True, exist_ok=True)
    win = (t_s - WINDOW_S, t_s)
    extra = {}
    if sizes is not None:
        extra["sizes"] = sizes
    if smoke_drive:
        extra["ext_sources"] = 100
        extra["som_ext_rate_hz"] = 4.0
    total_start = time.time()
    start = time.time()
    baselines = {s: cf.run_cell(s, T_s=t_s, raster_window=win, **extra) for s in seeds}
    baseline_rate = {s: {pop: _baseline_rate_hz(baselines[s], pop) for pop in ("PV", "SOM")}
                     for s in seeds}
    group_rate = {pop: float(np.mean([baseline_rate[s][pop] for s in seeds]))
                  for pop in ("PV", "SOM")}
    print("paired-baseline group rate_hz", group_rate, flush=True)
    cond_entries, run_wall = {}, {"baselines": time.time() - start}
    for s in seeds:
        start = time.time()
        results = _condition_results(s, baselines[s], group_rate, t_s, win, extra, conditions)
        run_wall["seed%d" % s] = time.time() - start
        (out / ("seed%d" % s)).mkdir()
        cond_entries[str(s)] = {}
        for name in conditions:
            rel = "seed%d/%s.npz" % (s, name)
            np.savez(out / rel, **_condition_npz(results[name]))
            cond_entries[str(s)][name] = {"file": rel, "sha256": _sha256(out / rel)}
            print("wrote", out / rel, flush=True)
    manifest = {
        "seeds": seeds,
        "conditions_run": conditions,
        "baseline_rate_hz": {str(s): baseline_rate[s] for s in seeds},
        "group_rate_hz": group_rate,
        "run_wall_s": run_wall,
        "wall_s": time.time() - total_start,
        "T_s": t_s,
        "window_s": [t_s - WINDOW_S, t_s],
        "param_set": "cellrep",
        "backend": jax.default_backend(),
        "git_sha": git_sha(),
        "sizes": sizes,
        "smoke_drive": smoke_drive,
        "conditions": cond_entries,
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("wrote", out / "manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    main()