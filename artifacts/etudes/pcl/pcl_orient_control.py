"""PCL control O1: the C1 column with hand-set oriented simple-cell kernels, no learning.

Feature f = 2k + q has bar orientation k*pi/8; its ON lobe lies one pixel ahead of the
receptive-field centre along the bar normal and its OFF lobe one pixel behind (q = 0),
or the reverse (q = 1), so the pair covers both drift directions. Lobes are Gaussian
across the bar (sigma 0.5 px) and flat along it; each kernel is normalized like the
initial weights. Local inhibition keeps its random initial weights; distant and
top-down inhibition are off, as in the A1 measurement. Protocol: README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl_column as C  # noqa: E402

SIGMA = 0.5  # px, lobe width across the bar


def oriented_kernels():
    """(FS, RF*RF*2) magnitudes in pcl_column's (dy, dx, polarity) order, before normalization."""
    dy, dx = np.meshgrid(np.arange(C.RF), np.arange(C.RF), indexing="ij")
    c = (C.RF - 1) / 2
    k = np.zeros((C.FS, C.RF, C.RF, 2))
    for f in range(C.FS):
        ori, q = divmod(f, 2)
        th = ori * np.pi / C.N_ORI + np.pi / 2  # bar normal, as in stimulus_step
        dn = (dx - c) * np.cos(th) + (dy - c) * np.sin(th)
        ahead, behind = np.exp(-(dn - 1) ** 2 / (2 * SIGMA**2)), np.exp(-(dn + 1) ** 2 / (2 * SIGMA**2))
        k[f, ..., 0], k[f, ..., 1] = (ahead, behind) if q == 0 else (behind, ahead)
    return k.reshape(C.FS, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-test-per-ori", type=int, default=20)
    ap.add_argument("--stim", choices=("bar", "segment"), default="bar")
    ap.add_argument("--length", type=float, default=C.STIM["length"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    C.STIM["kind"], C.STIM["length"] = a.stim, a.length
    t0 = time.time()
    key = jax.random.PRNGKey(a.seed)
    k_init, _, _, _, k_test = jax.random.split(key, 5)  # same split as pcl_column.main
    W0 = C.init_weights(k_init)
    W_or = dict(W0, s_exc=C.normalize(oriented_kernels(), C.MASKS["s_exc"], "s_exc"))
    labels = np.repeat(np.arange(C.N_ORI), a.n_test_per_ori)
    off = (False, False, 0.0)
    res = dict(params=dict(seed=a.seed, n_test_per_ori=a.n_test_per_ori, stim=C.STIM, sigma_px=SIGMA))
    for name, W in (("oriented", W_or), ("untrained", W0)):
        _, s, _, _ = C.run_block(W, k_test, 0, off, labels=labels)
        osi, rate = C.osi(s, labels)
        resp = rate >= 0.5
        res[name] = dict(osi_median=float(np.median(osi[resp])) if resp.any() else None,
                         n_resp=int(resp.sum()), spikes=int(s.sum()), acc_simple=C.decode(s, labels, a.seed))
    res["wall_s"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
