"""Paper-scale PLoS CB Figs 2C-F / 3C-F raster and mean-field data runner.

Usage: python run_plos_rasters.py <outdir> <seeds, e.g. 1,2,3> [--sizes NE,NPV,NSOM] [--smoke-drive]

Nine conditions per seed (baseline, PV/SOM silenced at 0.4 and 0.9, PV/SOM
randomized at 0.4 and 0.9) run through the same ``cellrep_figures`` calls as
``run_figA.py``, each with ``raster_window=(T_s - 0.5, T_s)``. Randomization
runs at the paired-baseline group rate: the mean over all given seeds'
baselines, as ``run_figA.py`` computes it. Env ``FIGA_T_S`` sets the run
length in seconds (default 12); ``FIGA_PARAM_SET`` must be ``default``
(anything else is refused). ``--sizes`` overrides the network sizes;
``--smoke-drive`` is a test-only flag adding the documented small-net drive
(``ext_sources=100``, ``som_ext_rate_hz=4.0``); without it the calls match
``run_figA.py`` exactly.

Writes ``seed<s>/<condition>.npz`` per seed and condition (``raster`` bool over the last 0.5 s,
``type_slices``, ``rate_hz`` per 1 ms bin in Hz per cell, ``f_pk_hz``,
``band_power_25_45``, ``max_e_rate_hz``), ``meanfield_<condition>.npz`` for
the baseline and the four silencing conditions (last-0.5 s ``t_s``/``nu_hz``
traces of ``caminos_meanfield.simulate_meanfield``), and a ``manifest.json``
(seed, T_s, param set, git SHA, backend, per-file sha256 and wall time).
An existing non-empty outdir is refused. Seeding comes only from ``seed``.
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

_HERE = Path(__file__).resolve().parent
_repo_root = Path(__file__).resolve().parents[4]  # P-025: the checkout's jaxfne, never site-packages
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_HERE))

import cellrep_figures as cf  # noqa: E402
import caminos_meanfield as cmf  # noqa: E402
import jax  # noqa: E402

assert "site-packages" not in cf.__file__, "run_plos_rasters must run the checkout's cellrep_figures"

CONDITIONS = [
    "baseline",
    "pv_silenced_0.4",
    "pv_silenced_0.9",
    "som_silenced_0.4",
    "som_silenced_0.9",
    "pv_randomized_0.4",
    "pv_randomized_0.9",
    "som_randomized_0.4",
    "som_randomized_0.9",
]

MF_CONDITIONS = [
    "baseline",
    "pv_silenced_0.4",
    "pv_silenced_0.9",
    "som_silenced_0.4",
    "som_silenced_0.9",
]

MF_ABSENT = [c for c in CONDITIONS if c not in MF_CONDITIONS]

_SILENCE_OF = {
    "pv_silenced_0.4": ("PV", 0.4),
    "pv_silenced_0.9": ("PV", 0.9),
    "som_silenced_0.4": ("SOM", 0.4),
    "som_silenced_0.9": ("SOM", 0.9),
}

_REPLACE_OF = {
    "pv_randomized_0.4": ("PV", 0.4),
    "pv_randomized_0.9": ("PV", 0.9),
    "som_randomized_0.4": ("SOM", 0.4),
    "som_randomized_0.9": ("SOM", 0.9),
}

_MF_SILENCE = {
    "baseline": (0.0, 0.0),
    "pv_silenced_0.4": (0.4, 0.0),
    "pv_silenced_0.9": (0.9, 0.0),
    "som_silenced_0.4": (0.0, 0.4),
    "som_silenced_0.9": (0.0, 0.9),
}

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
    """Split ``[outdir, seeds, ...flags]`` into outdir, seeds, sizes and smoke-drive."""
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
        elif flag == "--smoke-drive":
            smoke_drive = True
            i += 1
        else:
            raise SystemExit("error: unknown flag %r" % (flag,))
    return outdir, seeds, sizes, smoke_drive


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _baseline_rate_hz(res, pop):
    """Kept-window mean rate of ``pop`` in one baseline run (``run_figA.py`` lines 97-99)."""
    keep = np.asarray(res["counts_kept"])[:, {"E": 0, "PV": 1, "SOM": 2}[pop]]
    return float(keep.sum()) / (int(res["sizes"][pop]) * float(res["dt_s"]) * keep.shape[0])


def _condition_results(seed, baseline, group_rate, t_s, win, extra):
    """The nine condition dicts of one seed through the ``run_figA.py`` call pattern.

    ``group_rate`` is the paired-baseline mean rate over all seeds of the run
    (``run_figA.py`` lines 95-101; mmc2 1772-1774), shared by every seed.
    """
    out = {"baseline": baseline}
    for name, (pop, frac) in _SILENCE_OF.items():
        out[name] = cf.silence_sweep(pop, [frac], [seed], T_s=t_s,
                                     raster_window=win, **extra)[0]
    for name, (pop, frac) in _REPLACE_OF.items():
        out[name] = cf.replace_sweep(pop, [frac], [seed], rate_hz=group_rate[pop],
                                     T_s=t_s, raster_window=win, **extra)[0]
    return out


def _condition_npz(res):
    """On-disk arrays for one condition: raster, slices, binned rates, scalars."""
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
    return {
        "raster": raster,
        "type_slices": np.array([[0, ne], [ne, ne + npv], [ne + npv, n_tot]]),
        "rate_hz": rate,
        "f_pk_hz": np.asarray(float(res["f_pk_hz"])),
        "band_power_25_45": np.asarray(float(res["band_power_25_45"])),
        "max_e_rate_hz": np.asarray(float(res["max_e_rate_hz"])),
    }


def _meanfield_npz(t_s, pv_silence, som_silence):
    """Last-0.5 s mean-field traces for one silencing pair."""
    t, nu, w = cmf.simulate_meanfield(t_s, pv_silence=pv_silence, som_silence=som_silence)
    sel = np.asarray(t) >= t_s - 0.5
    return {
        "t_s": np.asarray(t)[sel],
        "nu_hz": np.asarray(nu)[sel],
        "w": np.asarray(w)[sel],
        "pv_silence": np.asarray(float(pv_silence)),
        "som_silence": np.asarray(float(som_silence)),
    }


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    outdir, seeds, sizes, smoke_drive = parse_args(list(argv))
    t_s = float(os.environ.get("FIGA_T_S", "12"))
    param_set = os.environ.get("FIGA_PARAM_SET", "default")
    if param_set != "default":
        raise SystemExit("FIGA_PARAM_SET must be default, got %r" % (param_set,))
    cf.PARAM_SET = "default"
    out = Path(outdir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing non-empty outdir %s" % outdir)
    out.mkdir(parents=True, exist_ok=True)
    win = (t_s - 0.5, t_s)
    extra = {}
    if sizes is not None:
        extra["sizes"] = sizes
    if smoke_drive:
        extra["ext_sources"] = 100
        extra["som_ext_rate_hz"] = 4.0
    start = time.time()
    baselines = {s: cf.run_cell(s, T_s=t_s, raster_window=win, **extra) for s in seeds}
    baseline_rate = {s: {pop: _baseline_rate_hz(baselines[s], pop) for pop in ("PV", "SOM")}
                     for s in seeds}
    group_rate = {pop: float(np.mean([baseline_rate[s][pop] for s in seeds]))
                  for pop in ("PV", "SOM")}
    print("paired-baseline group rate_hz", group_rate, flush=True)
    cond_entries, mf_entries, run_wall = {}, {}, {"baselines": time.time() - start}
    for s in seeds:
        start = time.time()
        results = _condition_results(s, baselines[s], group_rate, t_s, win, extra)
        run_wall["seed%d" % s] = time.time() - start
        (out / ("seed%d" % s)).mkdir()
        cond_entries[str(s)] = {}
        for name in CONDITIONS:
            rel = "seed%d/%s.npz" % (s, name)
            np.savez(out / rel, **_condition_npz(results[name]))
            cond_entries[str(s)][name] = {"file": rel, "sha256": _sha256(out / rel)}
            print("wrote", out / rel, flush=True)
    for name in MF_CONDITIONS:
        start = time.time()
        pv_sil, som_sil = _MF_SILENCE[name]
        path = out / ("meanfield_" + name + ".npz")
        np.savez(path, **_meanfield_npz(t_s, pv_sil, som_sil))
        mf_entries[name] = {"file": "meanfield_" + name + ".npz", "sha256": _sha256(path),
                            "wall_s": time.time() - start}
        print("wrote", path, flush=True)
    manifest = {
        "seeds": seeds,
        "baseline_rate_hz": {str(s): baseline_rate[s] for s in seeds},
        "group_rate_hz": group_rate,
        "run_wall_s": run_wall,
        "T_s": t_s,
        "param_set": "default",
        "backend": jax.default_backend(),
        "git_sha": git_sha(),
        "sizes": sizes,
        "smoke_drive": smoke_drive,
        "conditions": cond_entries,
        "meanfield": mf_entries,
        "meanfield_absent": MF_ABSENT,
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("wrote", out / "manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    main()
