"""ED9 — homeostasis / adaptation controller evidence bundle.

Null + ablation + repeated-seed evidence that the minimal homeostatic
resource/adaptation controller (intrinsic excitability bias g_i + optional
homeostatic synaptic scaling) STABILIZES firing rates and reduces runaway/imbalance
in the jaxfne proxy scaffold.

This is COMPUTATIONAL-METHOD evidence only — NOT a biological-mechanism validation.
Truth gates stay conservative (claim_level=computational_scaffold,
biological_learning_claim=False, mechanism_claim_status=not_claimed). Objective
success here does not imply mechanism support.

Ablation grid (each controller component on/off):
  null       : controller OFF (k_gain=0, eta=0)          -> the null control
  intrinsic  : intrinsic homeostasis only (k_gain>0)
  synaptic   : homeostatic synaptic plasticity only (eta>0)
  both       : intrinsic + synaptic

Each condition is run over N repeated seeds on a deliberately IMBALANCED column
(a hyperactive group + a quiet group), so the stabilizing effect is measurable.
Metrics (mean +/- std across seeds): mean firing rate, hyper-vs-quiet rate spread,
synchrony kappa, V_m finiteness. A manifest + SHA256 receipt are written.

Usage:
    python scripts/ed9_homeostasis_evidence.py --n 200 --seeds 5 --duration-ms 2000
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne


def build_imbalanced_model(n: int, hi_drive: float, lo_drive: float):
    """Canonical column with a hyperactive (first half) and quiet (second half) group."""
    cfg = (
        jtfne.build_laminar_column(n=n, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    model = jtfne.construct(cfg)
    n_real = int(model.params["emitter"].v0.shape[0])
    drive = np.where(np.arange(n_real) < n_real // 2, hi_drive, lo_drive).astype(np.float32)
    model = jtfne.with_emitter_parameters(model, drive_per_neuron=jnp.asarray(drive))
    return model, n_real


def _runtime(cond: dict, k_gain: float, eta: float) -> jtfne.RuntimeConfig:
    if not cond["enable_homeostasis"]:
        return jtfne.RuntimeConfig(recurrent_backend="edge_list")
    return jtfne.RuntimeConfig(
        enable_homeostasis=True,
        homeostasis_params={"k_gain": cond["k_gain"], "eta": cond["eta"],
                            "w_min": -3.0, "w_max": 3.0},
    )


def metrics(sig, hyper: slice, quiet: slice, duration_ms: float, dt_ms: float) -> dict:
    spikes = np.asarray(sig.spikes)
    rate = spikes.sum(0) / (duration_ms / 1000.0)
    return {
        "mean_rate_hz": float(rate.mean()),
        "rate_spread_hz": float(rate[hyper].mean() - rate[quiet].mean()),
        "kappa_synchrony": float(jtfne.kappa_synchrony(sig.spikes, dt_ms)),
        "vm_finite": bool(np.isfinite(np.asarray(sig.V_m)).all()),
    }


def _agg(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {"mean": float(arr.mean()), "std": sd,
            "ci95_halfwidth": float(1.96 * sd / max(n, 1) ** 0.5), "n": n}


def run(n=200, seeds=5, duration_ms=2000.0, dt_ms=0.5, k_gain=8.0, eta=0.05,
        hi_drive=12.0, lo_drive=3.0, out_dir="outputs/ed9_homeostasis_evidence"):
    conditions = {
        "null":      {"enable_homeostasis": False, "k_gain": 0.0, "eta": 0.0},
        "intrinsic": {"enable_homeostasis": True,  "k_gain": k_gain, "eta": 0.0},
        "synaptic":  {"enable_homeostasis": True,  "k_gain": 0.0, "eta": eta},
        "both":      {"enable_homeostasis": True,  "k_gain": k_gain, "eta": eta},
    }
    model, n_real = build_imbalanced_model(n, hi_drive, lo_drive)  # built ONCE, reused
    hyper, quiet = slice(0, n_real // 2), slice(n_real // 2, n_real)

    per_condition = {}
    for name, cond in conditions.items():
        runs = []
        for s in range(int(seeds)):
            sig = jtfne.simulate(model, sim=jtfne.Simulation(
                duration_ms=duration_ms, dt_ms=dt_ms, seed=s, runtime=_runtime(cond, k_gain, eta)))
            runs.append(metrics(sig, hyper, quiet, duration_ms, dt_ms))
        per_condition[name] = {
            "config": cond,
            "mean_rate_hz": _agg([r["mean_rate_hz"] for r in runs]),
            "rate_spread_hz": _agg([r["rate_spread_hz"] for r in runs]),
            "kappa_synchrony": _agg([r["kappa_synchrony"] for r in runs]),
            "all_vm_finite": all(r["vm_finite"] for r in runs),
        }

    null_spread = per_condition["null"]["rate_spread_hz"]["mean"]
    both_spread = per_condition["both"]["rate_spread_hz"]["mean"]
    bundle = {
        "schema_version": "jaxfne.ed9_homeostasis_evidence.v0.1.0",
        "evidence_kind": "computational_method_stabilization_null_ablation_seeds",
        "claim_status": "computational_control_proxy_not_biological_mechanism",
        "biological_learning_claim": False,
        "mechanism_claim_status": "not_claimed",
        "note": ("Method evidence only: controller stabilizes rates / reduces imbalance. "
                 "Mechanism support requires empirical comparison beyond this bundle."),
        "setup": {"n": n_real, "seeds": int(seeds), "duration_ms": duration_ms, "dt_ms": dt_ms,
                  "k_gain": k_gain, "eta": eta, "hi_drive": hi_drive, "lo_drive": lo_drive},
        "conditions": per_condition,
        "summary": {
            "null_rate_spread_hz": null_spread,
            "both_rate_spread_hz": both_spread,
            "spread_reduction_fraction": (None if null_spread == 0
                                          else float(1.0 - both_spread / null_spread)),
            "all_conditions_finite": all(c["all_vm_finite"] for c in per_condition.values()),
        },
    }

    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / "ed9_evidence.json"
    jtfne.save_json(jtfne.json_safe(bundle), str(bundle_path))
    digest = jtfne.sha256_file(str(bundle_path))
    receipt = {"artifact": "ed9_evidence.json", "sha256": digest,
               "schema_version": bundle["schema_version"],
               "truth_status": "method_evidence_not_mechanism_validation"}
    jtfne.save_json(receipt, str(out / "ed9_receipt.json"))
    return bundle, digest, out


def main():
    ap = argparse.ArgumentParser(description="ED9 homeostasis controller evidence bundle")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--duration-ms", type=float, default=2000.0)
    ap.add_argument("--dt-ms", type=float, default=0.5)
    ap.add_argument("--k-gain", type=float, default=8.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--out-dir", default="outputs/ed9_homeostasis_evidence")
    args = ap.parse_args()
    bundle, digest, out = run(n=args.n, seeds=args.seeds, duration_ms=args.duration_ms,
                              dt_ms=args.dt_ms, k_gain=args.k_gain, eta=args.eta, out_dir=args.out_dir)
    print(f"ED9 evidence written to {out}/ (sha256={digest[:16]}...)")
    for name, c in bundle["conditions"].items():
        print(f"  {name:9} spread={c['rate_spread_hz']['mean']:6.2f}±{c['rate_spread_hz']['std']:.2f} Hz"
              f"  mean={c['mean_rate_hz']['mean']:5.1f} Hz  kappa={c['kappa_synchrony']['mean']:.3f}"
              f"  finite={c['all_vm_finite']}")
    s = bundle["summary"]
    print(f"  spread reduction (both vs null): "
          f"{(s['spread_reduction_fraction'] or 0)*100:.0f}%   all_finite={s['all_conditions_finite']}")


if __name__ == "__main__":
    main()
