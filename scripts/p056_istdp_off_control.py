"""P-056 iSTDP-off control with seed spread (spec: artifacts/programme/design/capabilities/p056_diagnostic_spec_v1.md, 5e and 5f).

Zero initial weights on every edge; default iSTDP versus hdp_rule_params={"eta": 0.0}; drives base, PV x2, SOM x2;
seeds 0-4 (network draw and simulation seed). Resource-smoke network: arbitrary weights and drive, says nothing about P-056.

    python scripts/p056_istdp_off_control.py OUT.json

Refuses to overwrite OUT.json. Criteria C1-C4 are declared in the spec before the first run; failures are reported, never retuned.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p056_resource_smoke as m  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
DRIVES = {"base": {}, "PV_x2": {"PV": 2.0}, "SOM_x2": {"SOM": 2.0}}
RULES = {"default_iSTDP": None, "eta_0": {"eta": 0.0}}


def one(net, scale, params):
    c = m.CONFIG
    labels = np.array(net["labels"])
    i_ext = np.array([c["drive_pA"] * scale.get(k, 1.0) for k in labels])
    out = m.simulate_adex_hdp_registered(
        m.make_adex_params(net["labels"]), n_steps=c["n_steps"], dt_ms=c["dt_ms"],
        edge_pre=net["pre"], edge_post=net["post"], edge_weight_nS=np.zeros_like(net["weight"]),
        edge_receptor=net["receptor"], edge_delay_steps=np.full(len(net["pre"]), c["delay_steps"]),
        receptor_reversal_mV=c["receptor_reversal_mV"], receptor_tau_ms=c["receptor_tau_ms"], decay="exp",
        I_ext_pA=i_ext, record_v=False, record_w_trace=False, hdp_rule=m.ISTDP_RULE_NAME, hdp_rule_params=params,
        plasticity_mask=m.istdp_plasticity_mask(net["pre_inh"], net["post_exc"]), seed=c["seed"])
    jax.block_until_ready(out["spikes"])
    return m.rates_hz(out, net["labels"]), np.asarray(out["w_final_nS"], dtype=float)


def main(argv):
    if len(argv) != 1:
        print("usage: python scripts/p056_istdp_off_control.py OUT.json", file=sys.stderr)
        return 2
    dest = Path(argv[0])
    if dest.exists():
        print(f"REFUSING: {dest} exists; nothing written", file=sys.stderr)
        return 2

    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout

    sha, dirty = git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain").strip())
    runs = {}
    for s in SEEDS:
        m.CONFIG["seed"] = s
        net = m.build()
        plastic = np.asarray(m.istdp_plasticity_mask(net["pre_inh"], net["post_exc"]))
        for rule, params in RULES.items():
            for dname, scale in DRIVES.items():
                r, w = one(net, scale, params)
                runs[f"{s}/{rule}/{dname}"] = dict(rates_hz=r, w_plastic_mean=float(w[plastic].mean()), w_max_all=float(w.max()))
                print(s, rule, dname, runs[f"{s}/{rule}/{dname}"], flush=True)

    def e(s, rule, d):
        return runs[f"{s}/{rule}/{d}"]["rates_hz"]["E"]

    crit = {
        "C1_eta0_weights_exactly_zero": all(runs[f"{s}/eta_0/{d}"]["w_max_all"] == 0.0 for s in SEEDS for d in DRIVES),
        "C2_eta0_E_equals_base": all(e(s, "eta_0", d) == e(s, "eta_0", "base") for s in SEEDS for d in ("PV_x2", "SOM_x2")),
        "C4_default_weights_grow": all(runs[f"{s}/default_iSTDP/{d}"]["w_plastic_mean"] > 0 for s in SEEDS for d in DRIVES),
    }
    c3 = {}
    for d in ("PV_x2", "SOM_x2"):
        r = np.array([e(s, "default_iSTDP", d) / e(s, "default_iSTDP", "base") for s in SEEDS])
        sd = float(r.std(ddof=1))
        c3[d] = dict(ratios=r.tolist(), mean=float(r.mean()), sd=sd, larger_than_2sd=bool(abs(r.mean() - 1) > 2 * sd))
    receipt = {"schema": "jaxfne.p056_istdp_off_control.v1", "label": "resource-smoke network; says nothing about P-056",
               "config": m.CONFIG, "seeds": list(SEEDS), "runs": runs, "criteria": crit, "C3_default_E_ratio": c3,
               "env": {"jax": jax.__version__, "numpy": np.__version__, "git_sha": sha, "git_dirty": dirty}}
    with dest.open("x", encoding="utf-8") as f:
        f.write(json.dumps(receipt, indent=1) + "\n")
    print("criteria", crit, "C3", c3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
