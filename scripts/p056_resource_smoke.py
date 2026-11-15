"""P-056 resource smoke (spec: artifacts/programme/design/plans/p056_diagnostic_spec_v1.md, section 5b).

A generic E/PV/SOM AdEx network (N = 1000, iSTDP on I-to-E edges) that measures cost, numerical stability and
input engagement. It is NOT a P-056 measurement: weights and drive are arbitrary and carry no scientific meaning.

    python scripts/p056_resource_smoke.py OUT.json

Refuses to overwrite OUT.json. Budget, configuration and checks are declared in the spec before the first run and
are copied into the receipt; a failed budget or check is reported, never retuned.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jaxfne.emitters_adex import make_adex_params, simulate_adex_hdp_registered  # noqa: E402
from jaxfne.istdp_vogels2011 import ISTDP_RULE_NAME, istdp_plasticity_mask  # noqa: E402

CONFIG = dict(n_e=800, n_pv=100, n_som=100, dt_ms=0.1, n_steps=10000, seed=0, p_connect=0.1, w_e_nS=1.0, w_i_nS=3.0,
              delay_steps=20, drive_pA=300.0, receptor_reversal_mV=(0.0, -80.0), receptor_tau_ms=(2.0, 5.0),
              precision="float32")
BUDGET = dict(compile_s=180.0, execute_s=120.0, peak_rss_gb=4.0, output_mb=50.0)


def build():
    c = CONFIG
    labels = ("E",) * c["n_e"] + ("PV",) * c["n_pv"] + ("SOM",) * c["n_som"]
    n = len(labels)
    rng = np.random.default_rng(c["seed"])
    mask = rng.random((n, n)) < c["p_connect"]  # [post, pre]
    np.fill_diagonal(mask, False)
    post, pre = np.nonzero(mask)
    pre_inh = np.array([labels[i] != "E" for i in pre])
    post_exc = np.array([labels[i] == "E" for i in post])
    weight = np.where(pre_inh, c["w_i_nS"], c["w_e_nS"]).astype(np.float64)
    receptor = pre_inh.astype(int)
    return labels, pre, post, pre_inh, post_exc, weight, receptor


def run(labels, pre, post, pre_inh, post_exc, weight, receptor, drive_scale_e=1.0):
    c = CONFIG
    n = len(labels)
    i_ext = np.zeros(n)
    i_ext[: c["n_e"]] = c["drive_pA"] * drive_scale_e
    i_ext[c["n_e"]:] = c["drive_pA"]
    t0 = time.perf_counter()
    out = simulate_adex_hdp_registered(
        make_adex_params(labels), n_steps=c["n_steps"], dt_ms=c["dt_ms"],
        edge_pre=pre, edge_post=post, edge_weight_nS=weight, edge_receptor=receptor,
        edge_delay_steps=np.full(len(pre), c["delay_steps"]),
        receptor_reversal_mV=c["receptor_reversal_mV"], receptor_tau_ms=c["receptor_tau_ms"], decay="exp",
        I_ext_pA=i_ext, record_v=False, record_w_trace=False,
        hdp_rule=ISTDP_RULE_NAME, plasticity_mask=istdp_plasticity_mask(pre_inh, post_exc), seed=c["seed"])
    jax.block_until_ready(out["spikes"])
    return out, time.perf_counter() - t0


def rates_hz(out, labels):
    spikes = np.asarray(out["spikes"], dtype=bool)
    lab = np.array(labels)
    t_s = spikes.shape[0] * CONFIG["dt_ms"] / 1000.0
    return {k: float(spikes[:, lab == k].sum() / (t_s * (lab == k).sum())) for k in ("E", "PV", "SOM")}


def finite_checks(out, labels):
    spikes = np.asarray(out["spikes"])
    counts = spikes.astype(np.int64) @ np.stack([(np.array(labels) == k).astype(np.int64) for k in ("E", "PV", "SOM")], 1)
    return dict(
        spikes_bool=bool(spikes.dtype == bool),
        counts_shape=list(counts.shape),
        counts_nonnegative_integer=bool((counts >= 0).all()),
        w_final_finite=bool(np.isfinite(np.asarray(out["w_final_nS"], dtype=float)).all()),
        H_final_finite=bool(np.isfinite(np.asarray(out["H_final"], dtype=float)).all()),
        populations_present=bool((counts.shape[1] == 3)),
    )


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python scripts/p056_resource_smoke.py OUT.json", file=sys.stderr)
        return 2
    dest = Path(argv[0])
    if dest.exists():
        print(f"REFUSING: {dest} exists; nothing written", file=sys.stderr)
        return 2
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True,
                                check=True).stdout.strip())
    net = build()
    labels = net[0]
    out1, t_first = run(*net)
    out2, t_second = run(*net)
    comp = max(t_first - t_second, 0.0)
    rec = {"recurrent": rates_hz(out2, labels)}
    zero_w = np.zeros_like(net[5])
    z_base, t_zb = run(labels, net[1], net[2], net[3], net[4], zero_w, net[6], 1.0)
    z_e2, t_ze = run(labels, net[1], net[2], net[3], net[4], zero_w, net[6], 2.0)
    rec["zero_recurrence_base"], rec["zero_recurrence_E_drive_x2"] = rates_hz(z_base, labels), rates_hz(z_e2, labels)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2)
    checks = finite_checks(out2, labels)
    checks["deterministic_repeat"] = bool(np.array_equal(np.asarray(out1["spikes"]), np.asarray(out2["spikes"])))
    receipt = {
        "schema": "jaxfne.p056_resource_smoke.v1",
        "label": "resource smoke only: generic E/PV/SOM network, arbitrary weights and drive; says nothing about P-056",
        "config": CONFIG, "budget": BUDGET, "n_edges": int(len(net[1])),
        "timing_s": {"first_call": t_first, "second_call": t_second, "compile_estimate": comp, "execute": t_second,
                     "zero_recurrence_base": t_zb, "zero_recurrence_E_x2": t_ze},
        "peak_rss_gb": rss_gb,
        "rates_hz": rec, "checks": checks,
        "engagement": {"E_rate_ratio_x2_over_base_zero_recurrence": rec["zero_recurrence_E_drive_x2"]["E"] /
                       max(rec["zero_recurrence_base"]["E"], 1e-12),
                       "PV_rate_ratio": rec["zero_recurrence_E_drive_x2"]["PV"] / max(rec["zero_recurrence_base"]["PV"], 1e-12),
                       "SOM_rate_ratio": rec["zero_recurrence_E_drive_x2"]["SOM"] / max(rec["zero_recurrence_base"]["SOM"], 1e-12)},
        "within_budget": {"compile": comp <= BUDGET["compile_s"], "execute": t_second <= BUDGET["execute_s"],
                          "rss": rss_gb <= BUDGET["peak_rss_gb"]},
        "env": {"python": platform.python_version(), "platform": platform.platform(), "jax": metadata.version("jax"),
                "jaxlib": metadata.version("jaxlib"), "numpy": metadata.version("numpy"),
                "x64": bool(jax.config.jax_enable_x64), "git_sha": sha, "git_dirty": dirty},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as f:
        f.write(json.dumps(receipt, indent=1) + "\n")
    print(f"receipt: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
