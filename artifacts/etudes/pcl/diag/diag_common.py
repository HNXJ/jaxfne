"""Shared setup for the unscored PCL Fig. 3 FAIL diagnostic.

Reads only. Writes only under C:/workspace/pcl_diag/. The scorer is imported
from fig3_analysis and its loaders and statistics are reused unchanged.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

DIAG = Path("C:/workspace/pcl_diag")
SCORER = Path("E:/repos/jaxfne-wt-pcl/artifacts/etudes/pcl/port")
RUN = "F:/warehouse/pcl/runs/fig3_cSTDP3"
INPUT = "F:/warehouse/pcl/data/PCL_datasets_and_networks/grating/counterphase_gratingComplex/test7"
SCORED_JSON = Path("E:/repos/jaxfne-wt-pcl/artifacts/etudes/pcl/prereg/scores/fig3_v4_full.json")
CACHE = DIAG / "_temp_grid_cache.pkl"

EVEN = [0, 2, 4, 6, 8]
ODD = [1, 3, 5, 7, 9]

for _p in (str(SCORER), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fig3_analysis as fa  # noqa: E402  scorer under test, reused unchanged


def load_grid():
    """Both cell sets via fig3_analysis._load_grid, cached as a pickle under DIAG."""
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    sets = fa.cell_sets()
    cs, ts, cc, tc = fa._load_grid(RUN, sets["simple"], sets["complex"])
    data = {"counts_s": cs, "times_s": ts, "counts_c": cc, "times_c": tc,
            "cells_s": sets["simple"], "cells_c": sets["complex"]}
    with open(CACHE, "wb") as fh:
        pickle.dump(data, fh, protocol=4)
    return data


def fold_state(counts):
    """T1/T2 fold as the scorer does it: fit on even phases, score on odd, trials pooled.

    Mirrors the first lines of fig3_analysis.t1_t2 (lines 260-301) so that
    per-cell arrays can be built; q2 checks the result against t1_t2 itself.
    """
    pooled = np.asarray(counts).sum(axis=3)
    fit = pooled[:, :, EVEN, :].sum(axis=2)
    sco = pooled[:, :, ODD, :].sum(axis=2)
    keep = np.flatnonzero(fit.sum(axis=(0, 1)) >= fa.MIN_SPIKES_FIT)
    fit_k = fit[:, :, keep]
    sco_k = sco[:, :, keep]
    lin = fit_k.reshape(fa.N_ORI * fa.N_SF, keep.size).argmax(axis=0)
    o_star = (lin // fa.N_SF).astype(int)
    f_star = (lin % fa.N_SF).astype(int)
    flat = sco_k.reshape(fa.N_ORI * fa.N_SF, keep.size).astype(np.float64)
    return keep, o_star, f_star, flat


def scored():
    """Scored values from the signed run (read-only)."""
    with open(SCORED_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def sets_of(data):
    """Yield (name, counts, times, cells) for both sets."""
    yield "simple", data["counts_s"], data["times_s"], data["cells_s"]
    yield "complex", data["counts_c"], data["times_c"], data["cells_c"]
