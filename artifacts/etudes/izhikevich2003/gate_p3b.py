"""P3b equivalence gate, steps 1-3 (PREREG_P3b.md signed v4, gate amendment v5).

Runs the noise-free small case and the full-network case in float64 with
x64 enabled before any array is built, plus the float32 precision report.
Steps 1 and 2 use the v5 bar: the kernel against the reference up to R, the
first row where the reference's own 1-ulp control (thal moved by one ulp)
differs by more than 1e-6 in v (``v5_step``). The v4 receipt is
``scores/p3b_gate_v4.json``.
Prints a JSON report to stdout, writes it to --out, and exits 0 when steps
1 and 2 PASS, 1 otherwise. A failure is a finding; nothing here is tuned.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def _first_divergent_row(ref_spikes, ker_spikes):
    same = np.asarray(ref_spikes) == np.asarray(ker_spikes)
    bad = np.flatnonzero(~same.all(axis=1))
    return None if bad.size == 0 else int(bad[0])


def _count_ok(ref_n, ker_n, frac=0.01):
    if ref_n == 0:
        return ker_n == 0
    return abs(int(ker_n) - int(ref_n)) / int(ref_n) <= frac


def ulp_row(ref_v, ctrl_v, tol=1e-6):
    """R: first row where max |v_test(ref) - v_test(ulp control)| exceeds tol, else n_steps."""
    d = np.abs(np.asarray(ref_v, dtype=float) - np.asarray(ctrl_v, dtype=float)).max(axis=1)
    over = np.flatnonzero(d > tol)
    return int(over[0]) if over.size else int(d.size)


def v5_step(ref, ker, ctrl):
    """Amendment v5 bar for one step: rasters identical and extra u at rtol 1e-6 on rows < R,
    total counts within 1 %. Each argument is a simulate dict with spikes, extra_u, v_test."""
    sp_r, sp_k, sp_c = (np.asarray(x["spikes"]) for x in (ref, ker, ctrl))
    r = ulp_row(ref["v_test"], ctrl["v_test"])
    raster_ok = bool(np.array_equal(sp_r[:r], sp_k[:r]))
    extra_ok = bool(np.allclose(np.asarray(ker["extra_u"])[:r], np.asarray(ref["extra_u"])[:r],
                                rtol=1e-6, atol=0.0))
    count_ok = _count_ok(int(sp_r.sum()), int(sp_k.sum()))
    ok = raster_ok and extra_ok and count_ok
    return {"R": r, "rasters_identical_before_R": raster_ok,
            "extra_u_rtol1e-6_before_R": extra_ok, "count_within_1pct": bool(count_ok),
            "ref_spikes": int(sp_r.sum()), "kernel_spikes": int(sp_k.sum()),
            "control_spikes": int(sp_c.sum()),
            "kernel_first_divergent_row": _first_divergent_row(sp_r, sp_k),
            "control_first_divergent_row": _first_divergent_row(sp_r, sp_c),
            "mean_extra_u_ref": float(ref["mean_extra_u_per_spike"]),
            "mean_extra_u_kernel": float(ker["mean_extra_u_per_spike"]),
            "verdict": "PASS" if ok else "FAIL"}


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
        ker = {k: (np.asarray(v) if k != "mean_extra_u_per_spike" else v) for k, v in ker.items()}
        ctrl = nref.simulate(net, n_steps, None, thal=np.nextafter(thal, np.inf), return_v=True)
        return ref, ker, ctrl

    # --- step 1: small, noise-free -------------------------------------------------
    net1 = nref.build_network(np.random.default_rng(20261013), n_e=40, n_i=10)
    n1 = 50
    steps1 = 1000
    thal1 = np.full((steps1, n1), 6.0, dtype=np.float64)
    step1 = dict({"n_e": 40, "n_i": 10, "n_steps": steps1, "thal": "6.0 const"},
                 **v5_step(*_run_both(net1, thal1, steps1)))

    # --- step 2: full network ------------------------------------------------------
    net2 = nref.build_network(np.random.default_rng(1))
    n2 = 1000
    steps2 = 1000
    rng2 = np.random.default_rng(2)
    thal2 = np.empty((steps2, n2), dtype=np.float64)
    for t in range(steps2):
        thal2[t] = np.concatenate([5.0 * rng2.standard_normal(800),
                                   2.0 * rng2.standard_normal(200)])
    ref2, ker2, ctrl2 = _run_both(net2, thal2, steps2)
    step2 = dict({"seed_net": 1, "seed_thal": 2, "n_steps": steps2},
                 **v5_step(ref2, ker2, ctrl2))

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

    verdict = "PASS" if (step1["verdict"] == step2["verdict"] == "PASS") else "FAIL"
    report = {"prereg": "PREREG_P3b.md signed v4, gate amendment v5, steps 1-3",
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
