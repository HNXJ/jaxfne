"""P3b equivalence gate, steps 1-3 (PREREG_P3b.md signed v4).

Runs the noise-free small case and the full-network case in float64 with
x64 enabled before any array is built, plus the float32 precision report.
Prints a JSON report to stdout, writes it to --out, and exits 0 when steps
1 and 2 PASS, 1 otherwise. A failure is a finding; nothing here is tuned.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

import numpy as np  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_root():
    for q in (HERE,) + tuple(HERE.parents):
        if (q / ".git").exists():
            return q
    return HERE.parents[3]


def _git_sha():
    try:
        out = subprocess.run(["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        return out or "unknown"
    except OSError:
        return "unknown"


def _close(a, b, rtol=1e-9):
    return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=0.0)


def _first_divergent_row(ref_spikes, ker_spikes):
    same = np.asarray(ref_spikes) == np.asarray(ker_spikes)
    bad = np.flatnonzero(~same.all(axis=1))
    return None if bad.size == 0 else int(bad[0])


def _flip_bar(div_rows, ref_spikes, ker_spikes, ref_v, ker_v, tol=1e-9):
    """Every differing cell of every divergent row is within tol of 30 mV in both arms."""
    ref_spikes = np.asarray(ref_spikes)
    ker_spikes = np.asarray(ker_spikes)
    ref_v = np.asarray(ref_v, dtype=float)
    ker_v = np.asarray(ker_v, dtype=float)
    for row in div_rows:
        cells = np.flatnonzero(ref_spikes[row] != ker_spikes[row])
        if cells.size == 0:
            return False
        if not bool((np.abs(ref_v[row, cells] - 30.0) < tol).all()
                    and (np.abs(ker_v[row, cells] - 30.0) < tol).all()):
            return False
    return True


def _count_ok(ref_n, ker_n, frac=0.01):
    if ref_n == 0:
        return ker_n == 0
    return abs(int(ker_n) - int(ref_n)) / int(ref_n) <= frac


def main():
    import jax
    import jax.numpy as jnp

    if not bool(jax.config.read("jax_enable_x64")):
        jax.config.update("jax_enable_x64", True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    nref = _load("p3b_net_reference", HERE / "net_reference.py")
    kval = _load("p3b_kernel_net_m", HERE / "kernel_net_m.py")

    def _run_both(net, thal, n_steps):
        ref = nref.simulate(net, n_steps, None, thal=thal, return_v=True)
        jnet = {k: jnp.asarray(net[k], dtype=jnp.float64) for k in ("a", "b", "c", "d")}
        smat = jnp.asarray(net["S"], dtype=jnp.float64)
        thal_k = jnp.asarray(thal, dtype=jnp.float64)
        v0 = jnp.full((net["a"].size,), -65.0, dtype=jnp.float64)
        u0 = jnet["b"] * -65.0
        ker = kval.simulate_net_m(jnet, smat, thal_k, v0, u0, return_v=True)
        return ref, ker

    # --- step 1: small, noise-free -------------------------------------------------
    net1 = nref.build_network(np.random.default_rng(20261013), n_e=40, n_i=10)
    n1 = 50
    steps1 = 1000
    thal1 = np.full((steps1, n1), 6.0, dtype=np.float64)
    ref1, ker1 = _run_both(net1, thal1, steps1)
    sp_r1 = np.asarray(ref1["spikes"])
    sp_k1 = np.asarray(ker1["spikes"])
    first1 = _first_divergent_row(sp_r1, sp_k1)
    div1 = sorted(map(int, np.flatnonzero(~(sp_r1 == sp_k1).all(axis=1)).tolist()))
    extra_mean_ok1 = _close(ref1["mean_extra_u_per_spike"], ker1["mean_extra_u_per_spike"])
    extra_arr_ok1 = bool(np.allclose(np.asarray(ker1["extra_u"]), np.asarray(ref1["extra_u"]),
                                    rtol=1e-9, atol=0.0))
    count_ok1 = _count_ok(int(sp_r1.sum()), int(sp_k1.sum()))
    flip_ok1 = None
    if first1 is None:
        step1_pass = bool(extra_mean_ok1 and extra_arr_ok1)
    else:
        flip_ok1 = bool(_flip_bar(div1, sp_r1, sp_k1, ref1["v_test"],
                                 np.asarray(ker1["v_test"])))
        step1_pass = bool(flip_ok1 and count_ok1)
    step1 = {"n_e": 40, "n_i": 10, "n_steps": steps1, "thal": "6.0 const",
             "ref_spikes": int(sp_r1.sum()), "kernel_spikes": int(sp_k1.sum()),
             "mean_extra_u_ref": float(ref1["mean_extra_u_per_spike"]),
             "mean_extra_u_kernel": float(ker1["mean_extra_u_per_spike"]),
             "extra_mean_agree_rtol1e-9": bool(extra_mean_ok1),
             "extra_array_agree_rtol1e-9": bool(extra_arr_ok1),
             "identical": first1 is None, "first_divergent_row": first1,
             "n_divergent_rows": len(div1),
             "rounding_flip_bar": flip_ok1, "count_within_1pct": bool(count_ok1),
             "verdict": "PASS" if step1_pass else "FAIL"}

    # --- step 2: full network ------------------------------------------------------
    net2 = nref.build_network(np.random.default_rng(1))
    n2 = 1000
    steps2 = 1000
    rng2 = np.random.default_rng(2)
    thal2 = np.empty((steps2, n2), dtype=np.float64)
    for t in range(steps2):
        thal2[t] = np.concatenate([5.0 * rng2.standard_normal(800),
                                   2.0 * rng2.standard_normal(200)])
    ref2, ker2 = _run_both(net2, thal2, steps2)
    sp_r2 = np.asarray(ref2["spikes"])
    sp_k2 = np.asarray(ker2["spikes"])
    first2 = _first_divergent_row(sp_r2, sp_k2)
    prefix = slice(None, first2) if first2 is not None else slice(None)
    extra_prefix_ok2 = bool(np.allclose(np.asarray(ker2["extra_u"])[prefix],
                                       np.asarray(ref2["extra_u"])[prefix],
                                       rtol=1e-9, atol=0.0))
    count_ok2 = _count_ok(int(sp_r2.sum()), int(sp_k2.sum()))
    step2_pass = bool(extra_prefix_ok2 and count_ok2)
    step2 = {"seed_net": 1, "seed_thal": 2, "n_steps": steps2,
             "ref_spikes": int(sp_r2.sum()), "kernel_spikes": int(sp_k2.sum()),
             "mean_extra_u_ref": float(ref2["mean_extra_u_per_spike"]),
             "mean_extra_u_kernel": float(ker2["mean_extra_u_per_spike"]),
             "first_divergent_row": first2,
             "extra_prefix_agree_rtol1e-9": bool(extra_prefix_ok2),
             "count_within_1pct": bool(count_ok2),
             "verdict": "PASS" if step2_pass else "FAIL"}

    # --- step 3: precision report (reported only) -----------------------------------
    fnet = {k: jnp.asarray(net2[k], dtype=jnp.float32) for k in ("a", "b", "c", "d")}
    ker32 = kval.simulate_net_m(fnet, jnp.asarray(net2["S"], dtype=jnp.float32),
                                jnp.asarray(thal2, dtype=jnp.float32),
                                jnp.full((n2,), -65.0, dtype=jnp.float32),
                                fnet["b"] * jnp.asarray(-65.0, dtype=jnp.float32))
    sp_32 = np.asarray(ker32["spikes"])
    sp_64 = np.asarray(ker2["spikes"])
    n64, n32 = int(sp_64.sum()), int(sp_32.sum())
    step3 = {"count_f64": n64, "count_f32": n32, "count_diff": n32 - n64,
             "rel_diff": (abs(n32 - n64) / n64) if n64 else None,
             "first_divergent_row": _first_divergent_row(sp_64, sp_32)}

    verdict = "PASS" if (step1_pass and step2_pass) else "FAIL"
    report = {"prereg": "PREREG_P3b.md signed v4, equivalence gate steps 1-3",
              "code_sha": _git_sha(), "dtype": "float64",
              "x64": bool(jax.config.read("jax_enable_x64")),
              "backend": str(jax.devices()[0].platform),
              "step1": step1, "step2": step2, "step3_precision_report": step3,
              "verdict": verdict}
    text = json.dumps(report, indent=1)
    Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
