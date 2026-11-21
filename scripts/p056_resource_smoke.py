"""P-056 resource smoke v2 (spec: artifacts/programme/design/capabilities/p056_diagnostic_spec_v1.md, sections 5b and 5d).

A generic E/PV/SOM AdEx network (N = 2000, iSTDP on I-to-E edges) that measures cost, numerical stability and
input engagement. It is NOT a P-056 measurement: weights and drive are arbitrary and carry no scientific meaning.

    python scripts/p056_resource_smoke.py OUT.json

Refuses to overwrite OUT.json. Limits, configuration and checks are declared in the spec before the first run and
are copied into the receipt; a failed limit or check is reported, never retuned.
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

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

CONFIG = dict(n_e=1600, n_pv=200, n_som=200, dt_ms=0.1, n_steps=10000, floor_steps=10, seed=0, p_connect=0.1,
              w_e_nS=1.0, w_i_nS=3.0, delay_steps=20, drive_pA=300.0, receptor_reversal_mV=(0.0, -80.0),
              receptor_tau_ms=(2.0, 5.0), precision="float32")
LIMITS = dict(execute_s=120.0, peak_rss_gb=4.0, output_mb=50.0)
PREDICTION = dict(execute_s_if_cost_follows_N=12.0, execute_s_if_cost_follows_edges=24.0, rss_gb_below=2.0)
CLASSES = ("E", "PV", "SOM")


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
    return dict(labels=labels, pre=pre, post=post, pre_inh=pre_inh, post_exc=post_exc, weight=weight,
                receptor=pre_inh.astype(int))


def run(net, *, recurrent=True, scale=None, n_steps=None):
    c = CONFIG
    labels = np.array(net["labels"])
    scale = scale or {}
    i_ext = np.array([c["drive_pA"] * scale.get(k, 1.0) for k in labels])
    weight = net["weight"] if recurrent else np.zeros_like(net["weight"])
    t0 = time.perf_counter()
    out = simulate_adex_hdp_registered(
        make_adex_params(net["labels"]), n_steps=n_steps or c["n_steps"], dt_ms=c["dt_ms"],
        edge_pre=net["pre"], edge_post=net["post"], edge_weight_nS=weight, edge_receptor=net["receptor"],
        edge_delay_steps=np.full(len(net["pre"]), c["delay_steps"]),
        receptor_reversal_mV=c["receptor_reversal_mV"], receptor_tau_ms=c["receptor_tau_ms"], decay="exp",
        I_ext_pA=i_ext, record_v=False, record_w_trace=False,
        hdp_rule=ISTDP_RULE_NAME, plasticity_mask=istdp_plasticity_mask(net["pre_inh"], net["post_exc"]),
        seed=c["seed"])
    jax.block_until_ready(out["spikes"])
    return out, time.perf_counter() - t0


def rates_hz(out, labels):
    spikes = np.asarray(out["spikes"], dtype=bool)
    lab = np.array(labels)
    t_s = spikes.shape[0] * CONFIG["dt_ms"] / 1000.0
    return {k: float(spikes[:, lab == k].sum() / (t_s * (lab == k).sum())) for k in CLASSES}


def counts_and_checks(out, labels):
    spikes = np.asarray(out["spikes"])
    onehot = np.stack([(np.array(labels) == k).astype(np.int64) for k in CLASSES], 1)
    counts = spikes.astype(np.int64) @ onehot
    return dict(spikes_bool=bool(spikes.dtype == bool), counts_shape=list(counts.shape),
                counts_nonnegative_integer=bool((counts >= 0).all()),
                w_final_finite=bool(np.isfinite(np.asarray(out["w_final_nS"], dtype=float)).all()),
                H_final_finite=bool(np.isfinite(np.asarray(out["H_final"], dtype=float)).all()),
                populations_present=bool(counts.shape[1] == len(CLASSES)))


def same_state(a, b):
    return {k: bool(np.array_equal(np.asarray(a[k]), np.asarray(b[k]))) for k in ("spikes", "H_final", "w_final_nS")}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python scripts/p056_resource_smoke.py OUT.json", file=sys.stderr)
        return 2
    dest = Path(argv[0])
    if dest.exists():
        print(f"REFUSING: {dest} exists; nothing written", file=sys.stderr)
        return 2

    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout

    sha, dirty = git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain").strip())
    net = build()
    labels = net["labels"]
    _, t_floor = run(net, n_steps=CONFIG["floor_steps"])
    out1, t1 = run(net)
    out2, t2 = run(net)
    rates = {"recurrent_base": rates_hz(out2, labels)}
    timings = {"floor_10_steps": t_floor, "call_1": t1, "call_2": t2}
    rates["zero_recurrence_base"] = rates_hz(run(net, recurrent=False)[0], labels)
    for k in CLASSES:
        z, tz = run(net, recurrent=False, scale={k: 2.0})
        r, tr = run(net, recurrent=True, scale={k: 2.0})
        rates[f"zero_recurrence_{k}_x2"], rates[f"recurrent_{k}_x2"] = rates_hz(z, labels), rates_hz(r, labels)
        timings[f"zero_recurrence_{k}_x2"], timings[f"recurrent_{k}_x2"] = tz, tr
    base = rates["zero_recurrence_base"]
    engagement = {k: {"own_rate_ratio": rates[f"zero_recurrence_{k}_x2"][k] / max(base[k], 1e-12),
                      "other_rate_ratios": {o: rates[f"zero_recurrence_{k}_x2"][o] / max(base[o], 1e-12)
                                            for o in CLASSES if o != k}} for k in CLASSES}
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2)
    receipt = {
        "schema": "jaxfne.p056_resource_smoke.v2",
        "label": "resource smoke only: generic E/PV/SOM network, arbitrary weights and drive; says nothing about P-056",
        "config": CONFIG, "limits": LIMITS, "prediction": PREDICTION, "n_edges": int(len(net["pre"])),
        "timing_s": timings, "peak_rss_gb": rss_gb, "rates_hz": rates,
        "checks": {**counts_and_checks(out2, labels), "state_identical_across_two_runs": same_state(out1, out2)},
        "engagement": engagement,
        "env": {"python": platform.python_version(), "platform": platform.platform(), "jax": metadata.version("jax"),
                "jaxlib": metadata.version("jaxlib"), "numpy": metadata.version("numpy"),
                "x64": bool(jax.config.jax_enable_x64), "git_sha": sha, "git_dirty": dirty},
    }
    receipt["output_mb"] = len(json.dumps(receipt)) / 1e6
    receipt["within_limits"] = {"execute": t2 <= LIMITS["execute_s"], "rss": rss_gb <= LIMITS["peak_rss_gb"],
                                "output": receipt["output_mb"] <= LIMITS["output_mb"]}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as f:
        f.write(json.dumps(receipt, indent=1) + "\n")
    print(f"receipt: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
