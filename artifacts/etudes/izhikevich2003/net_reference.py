"""S1 reference arm: float64 NumPy transcription of the paper's net.m.

Spec: ``SOURCE_P1.md`` (Fig. 3 network) and ``PORT_SPEC.md`` (Reference arm).
Update order per 1 ms step: spike test on the v left by the previous step,
reset of the fired cells, thalamic input plus the fired weights, two 0.5 ms
forward-Euler v half-steps, then one 1 ms u step on the v after both
half-steps. A cell crossing 30 mV on the first half-step overshoots on the
second and is detected at the next step; that v enters the u step, and the
per-spike extra u increment this adds (against a capped v of 30 mV) is
recorded.

No download and no authors' code: every line below is written from the
SOURCE_P1/PORT_SPEC description. Outputs share the run_izh.py Fig. 3 layout
(one ``<seed>.npz`` with spike indices plus ``manifest.json``) with
``dt_ms = 1.0`` recorded in the manifest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

DT_MS = 1.0
N_E, N_I = 800, 200
V_INIT = -65.0


def build_network(rng, n_e=N_E, n_i=N_I):
    """Draw the per-cell parameters and the all-to-all weight matrix S.

    r ~ U[0, 1] per cell; excitatory (a, b) = (0.02, 0.2) and
    (c, d) = (-65, 8) + (15, -6) r^2; inhibitory (a, b) =
    (0.02, 0.25) + (0.08, -0.05) r and (c, d) = (-65, 2). S is
    0.5 U[0, 1] from excitatory columns and -U[0, 1] from inhibitory
    columns, self-connections included. All float64.
    """
    n = n_e + n_i
    r = np.asarray(rng.uniform(0.0, 1.0, n), dtype=np.float64)
    a = np.empty(n)
    b = np.empty(n)
    c = np.empty(n)
    d = np.empty(n)
    a[:n_e], b[:n_e] = 0.02, 0.2
    c[:n_e] = -65.0 + 15.0 * r[:n_e] ** 2
    d[:n_e] = 8.0 - 6.0 * r[:n_e] ** 2
    a[n_e:] = 0.02 + 0.08 * r[n_e:]
    b[n_e:] = 0.25 - 0.05 * r[n_e:]
    c[n_e:], d[n_e:] = -65.0, 2.0
    u = np.asarray(rng.uniform(0.0, 1.0, (n, n)), dtype=np.float64)
    s = np.where(np.arange(n)[None, :] < n_e, 0.5 * u, -u)
    return {"a": a, "b": b, "c": c, "d": d, "S": s, "n_e": n_e, "n_i": n_i}


def simulate(net, n_steps, rng, noise_on=True, drive=0.0):
    """Run the reference update for ``n_steps`` 1 ms steps.

    ``drive`` is a constant baseline current (the study uses 0.0).
    Returns a dict with ``spikes`` (bool (n_steps, n) array: row t holds the
    cells fired at the test opening step t), ``extra_u`` (the overshoot extra
    u increment recorded at each first-half-step crossing) and
    ``mean_extra_u_per_spike``.
    """
    a, b, c, d, s = net["a"], net["b"], net["c"], net["d"], net["S"]
    n = a.size
    n_e = int(net["n_e"])
    v = np.full(n, V_INIT)
    u = b * v
    spikes = np.zeros((n_steps, n), dtype=bool)
    extra = np.zeros((n_steps, n))
    will = np.zeros((n_steps, n), dtype=bool)
    for t in range(n_steps):
        if noise_on:
            thal = np.concatenate([5.0 * rng.standard_normal(n_e),
                                   2.0 * rng.standard_normal(n - n_e)])
        else:
            thal = np.zeros(n)
        fired = v >= 30.0
        spikes[t] = fired
        v = np.where(fired, c, v)
        u = np.where(fired, u + d, u)
        cur = thal + drive + (s[:, fired].sum(axis=1) if fired.any() else 0.0)
        v = v + 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + cur)
        # Cells crossing 30 on the first half-step overshoot on the second and
        # are detected at the next step; that v enters the u step below.
        will_fire = (v >= 30.0) & ~fired
        will[t] = will_fire
        v = v + 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + cur)
        # Extra u increment from the overshoot: the u step sees v, not 30 mV.
        extra[t, will_fire] = a[will_fire] * b[will_fire] * (v[will_fire] - 30.0)
        u = u + a * (b * v - u)
    mean_extra = float(extra[will].mean()) if will.any() else 0.0
    return {"spikes": spikes, "extra_u": extra,
            "mean_extra_u_per_spike": mean_extra}


def _repo_root():
    p = Path(__file__).resolve()
    for q in (p,) + tuple(p.parents):
        if (q / ".git").exists():
            return q
    return Path(__file__).resolve().parents[3]


def _git_sha():
    try:
        out = subprocess.run(["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        return out or "unknown"
    except OSError:
        return "unknown"


def write_seed(out_dir, seed, t_s):
    """Run one reference seed and write ``<seed>.npz``; returns the manifest entry."""
    rng = np.random.default_rng(seed)
    net = build_network(rng)
    n_steps = int(round(t_s * 1000.0 / DT_MS))
    res = simulate(net, n_steps, rng, noise_on=True)
    rows, cols = np.flatnonzero(res["spikes"])
    path = Path(out_dir) / f"{seed}.npz"
    np.savez_compressed(path, spike_rows=rows.astype(np.int64),
                        spike_cols=cols.astype(np.int64),
                        n_steps=np.int64(n_steps), n_cells=np.int64(net["a"].size),
                        dt_ms=np.float64(DT_MS))
    return {"seed": seed, "file": path.name, "n_steps": n_steps, "dt_ms": DT_MS,
            "n_spikes": int(rows.size),
            "mean_extra_u_per_spike": res["mean_extra_u_per_spike"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if "scored" in str(a.out):
        raise SystemExit("refusing --out containing 'scored' without --signed")
    out.mkdir(parents=True, exist_ok=True)
    entries = [write_seed(out, s, a.T) for s in a.seeds]
    manifest = {"prereg": "PREREG_P3.md signed v7, sensitivity S1",
                "code_sha": _git_sha(), "backend": "numpy_reference_float64",
                "dtype": "float64", "dt_ms": DT_MS, "T_s": a.T,
                "seeds": list(a.seeds), "sensitivity": "S1", "runs": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(entries)} reference seeds to {out}")


if __name__ == "__main__":
    main()
