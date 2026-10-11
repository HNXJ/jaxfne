"""Validate the emulated libstdc++ order of the trained lateral (li) and top-down (tdi) maps.

Read-only. Each simple neuron s holds one file per map with one value per key, in the authors' map
iteration order (WeightMap.cpp:131-137). Three candidate orders are scored against a structural
prediction, one Pearson r per neuron and order:
  E  emulated libstdc++ iteration order (libstdcxx_umap.iteration_order on the insertion sequence)
  A  ascending key (the insertion sequence itself, since keys are inserted in ascending index)
  D  descending key
tdi prediction (from the port packet): the top-down inhibitory weight from complex cell c to simple
cell s correlates positively with c's excitatory weight on s, weights/1/<c>.npy at (x - i, y - j, z).
li prediction (inferred, not from the paper): the lateral weight from simple cell m to simple cell n
correlates positively with the cosine similarity of their depths' shared excitatory weights
(weights/0/<k>.npy), i.e. similarly tuned neighbours inhibit each other more.

CRITERION, fixed before the run. A candidate family "clearly wins" for a map when all hold:
  (a) mean r(E) - max(mean r(A), mean r(D)) >= 0.10,
  (b) mean r(E) >= 0.20,
  (c) E is the strict best of the three for at least 75 % of the neurons with finite r.

Outcome (umap_order_validation.json): neither map met the criterion. The dispatcher then accepted the
emulated order as a declared assumption (libstdcxx_umap.py docstring), and the loader uses it. This file
is kept as an informational record of the statistics, not as the validation of that assumption.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _repo_root(start: Path) -> Path:
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError(f"no pyproject.toml above {start}")


_ROOT = str(_repo_root(HERE))
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)
if str(HERE) not in sys.path:
    sys.path.insert(1, str(HERE))

import libstdcxx_umap as um  # noqa: E402

W = Path("F:/warehouse/pcl/data/PCL_datasets_and_networks/add_simulations_cSTDP/cSTDP3/weights")
OUT = HERE / "umap_order_validation.json"

NX, NY, ND = 9, 9, 64  # simple layer (layerSizes[0]), network_config.json
CX, CY, CD = 6, 6, 32  # complex layer (layerSizes[1])
RANGE = 4  # neuronInhibitionRange
N_S = NX * NY * ND
N_C = CX * CY * CD
CRIT_GAP, CRIT_MEAN, CRIT_BEST = 0.10, 0.20, 0.75


def simple_xyz(s: int) -> tuple[int, int, int]:
    return s // (NY * ND), (s // ND) % NY, s % ND


def lateral_keys(x0: int, y0: int) -> list[int]:
    """Keys of lateralDynamicInhibitionConnection (SpikingNetwork.cpp:409-442), in insertion order."""
    keys = [x * NY * ND + y * ND + z
            for x in range(max(0, x0 - RANGE), min(NX, x0 + RANGE + 1))
            for y in range(max(0, y0 - RANGE), min(NY, y0 + RANGE + 1))
            for z in range(ND) if (x, y) != (x0, y0)]
    assert keys == sorted(keys), "lateral insertion order must be ascending index"
    return keys


def topdown_keys(x0: int, y0: int) -> list[int]:
    """Keys of topDownConnection (SpikingNetwork.cpp:368-400) on simple cell (x0, y0), in insertion order.

    A complex cell (i, j, k) covers simple x when i <= x <= i + 3 (offset stride 1, RF 4), and all depths.
    """
    ii = [i for i in range(CX) if i <= x0 <= i + 3]
    jj = [j for j in range(CY) if j <= y0 <= j + 3]
    keys = [(i * CY + j) * CD + k for i in ii for j in jj for k in range(CD)]
    assert keys == sorted(keys), "top-down insertion order must be ascending complex index"
    return keys


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def emulated_positions(keys: list[int]) -> np.ndarray:
    """posE[q]: where keys[q] sits in the emulated iteration order (file index of its value)."""
    order = um.iteration_order(keys)
    pos = {k: t for t, k in enumerate(order)}
    return np.array([pos[k] for k in keys], dtype=np.int64)


def summarize(name: str, rs: np.ndarray, n_mismatch: int) -> dict:
    fin = np.all(np.isfinite(rs), axis=1)
    mE, mA, mD = (float(v) for v in rs[fin].mean(axis=0))
    best = (rs[:, 0] > rs[:, 1]) & (rs[:, 0] > rs[:, 2]) & fin
    frac = float(best.sum() / fin.sum())
    gap = mE - max(mA, mD)
    clear = bool(gap >= CRIT_GAP and mE >= CRIT_MEAN and frac >= CRIT_BEST)
    return {
        "map": name,
        "neurons": int(rs.shape[0]),
        "finite": int(fin.sum()),
        "length_mismatch": int(n_mismatch),
        "mean_r_emulated": mE,
        "mean_r_ascending": mA,
        "mean_r_descending": mD,
        "gap_E_minus_best_alternative": gap,
        "fraction_E_strict_best": frac,
        "clearly_wins": clear,
    }


def main() -> None:
    t0 = time.time()
    W1 = np.stack([np.load(W / "1" / f"{c}.npy") for c in range(N_C)])  # (1152, 4, 4, 64)
    print(f"complex excitatory weights {W1.shape} loaded in {time.time() - t0:.1f} s", flush=True)

    # Depth similarity from the shared simple excitatory weights of each depth (patch sharing).
    V = np.stack([np.load(W / "0" / f"{k}.npy").ravel() for k in range(ND)])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T  # (64, 64) cosine similarity of depth weights

    # Emulated order depends only on (x, y): the key set does not depend on depth.
    cache: dict[tuple[str, int, int], np.ndarray] = {}

    rows_t, mis_t = [], 0
    for s in range(N_S):
        x0, y0, z0 = simple_xyz(s)
        K = topdown_keys(x0, y0)
        f = np.load(W / "0" / f"{s}tdi.npy")
        if f.shape[0] != len(K):
            mis_t += 1
            continue
        Kc = np.asarray(K)
        ci, cj = Kc // (CY * CD), (Kc // CD) % CY
        e = W1[Kc, x0 - ci, y0 - cj, z0]
        key = ("tdi", x0, y0)
        if key not in cache:
            cache[key] = emulated_positions(K)
        aE, aA, aD = f[cache[key]], f, f[::-1]
        rows_t.append((pearson(aE, e), pearson(aA, e), pearson(aD, e)))
    print(f"tdi: {len(rows_t)} neurons scored, {mis_t} length mismatches, {time.time() - t0:.1f} s", flush=True)

    rows_l, mis_l = [], 0
    for n in range(N_S):
        x0, y0, z0 = simple_xyz(n)
        L = lateral_keys(x0, y0)
        f = np.load(W / "0" / f"{n}li.npy")
        if f.shape[0] != len(L):
            mis_l += 1
            continue
        Lc = np.asarray(L)
        pred = S[z0, Lc % ND]
        key = ("li", x0, y0)
        if key not in cache:
            cache[key] = emulated_positions(L)
        aE, aA, aD = f[cache[key]], f, f[::-1]
        rows_l.append((pearson(aE, pred), pearson(aA, pred), pearson(aD, pred)))
        if n % 1024 == 0:
            print(f"li: {n} of {N_S}, {time.time() - t0:.1f} s", flush=True)
    print(f"li: {len(rows_l)} neurons scored, {mis_l} length mismatches, {time.time() - t0:.1f} s", flush=True)

    result = {
        "criterion": {"gap_min": CRIT_GAP, "mean_r_min": CRIT_MEAN, "strict_best_fraction_min": CRIT_BEST},
        "tdi": summarize("tdi", np.asarray(rows_t), mis_t),
        "li": summarize("li", np.asarray(rows_l), mis_l),
        "emulated_order_cache_entries": len(cache),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for key in ("tdi", "li"):
        r = result[key]
        print(f"{key}: finite {r['finite']}/{r['neurons']}  mean r E {r['mean_r_emulated']:.4f}  "
              f"A {r['mean_r_ascending']:.4f}  D {r['mean_r_descending']:.4f}  "
              f"E strict best {r['fraction_E_strict_best']:.4f}  clearly_wins {r['clearly_wins']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
