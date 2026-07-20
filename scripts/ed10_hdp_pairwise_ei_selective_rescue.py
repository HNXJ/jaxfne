"""ED10 -- pairwise E/I HDP selective-rescue evidence bundle.

Tests the framing from plans.json's 2026-07-17 novelty brainstorm
(novelty::hdp-pairwise-ei-selective-rescue-framing): can independently-gained
E/I conductance updates (make_hebbian_pairwise_rule's k_ee/k_ei/k_ie/k_ii)
rescue an inhibition-weakened E/I circuit's excitatory activity toward a
healthy reference level via disproportionate growth of the inhibitory-onto-
excitatory pathway (k_ie boosted), while perturbing the E-E/E-I pathways less
than a uniform (flat-gain, plain "hebbian") rescue would?

This is COMPUTATIONAL-METHOD evidence only -- NOT a biological-mechanism
validation, and NOT a claim that this reproduces any specific disorder model.
No literature target was identified for a direct qualitative comparison (see
plans.json/progress.json -- the disorder-relevance framing from the original
brainstorm is explicitly NOT asserted here, only the narrower, checkable
mechanism claim: does asymmetric-gain HDP behave differently from uniform-gain
HDP in the way the framing predicted).

Three conditions, same "damaged" (inhibition-weakened) starting network:
  no_rescue : conductance frozen (freeze_G=True)                    -- damaged baseline
  uniform   : conductance_rule="hebbian" (flat gain, all pairs equal) -- standard rescue
  selective : conductance_rule=make_hebbian_pairwise_rule(k_ie boosted, others damped)

Plus a fourth, undamaged reference condition (healthy) for a real rescue target,
not an arbitrary number.

Usage:
    python scripts/ed10_hdp_pairwise_ei_selective_rescue.py --n 8 --seeds 20 --duration-ms 5000
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne.emitters_homeostatic_ei import (
    make_minimal_ei_params, simulate_homeostatic_ei, make_hebbian_pairwise_rule,
)

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _ed9_common import _agg, _significance_test


def _run_one(params, n_steps, dt_ms, seed, *, conductance_rule="hebbian", freeze_G=False):
    key = jax.random.PRNGKey(seed)
    voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps, dt_ms, key,
        conductance_rule=conductance_rule, freeze_G=freeze_G, noise_scale=0.1,
        bound_mode="stable",
    )
    return voltages, G_hist, diag


def _settled_mean(arr, window_frac=0.2):
    """Mean over the last window_frac of the trajectory (post-adaptation)."""
    n = arr.shape[0]
    start = int(n * (1.0 - window_frac))
    return np.asarray(arr[start:]).mean(axis=0)


def run(n=8, e_fraction=0.75, seeds=20, duration_ms=5000.0, dt_ms=0.5,
        drive_i_healthy=0.3, drive_i_damaged=0.03,
        k_ie_boost=2.5, k_other_damp=0.4,
        out_dir="outputs/ed10_pairwise_ei_selective_rescue"):
    n_steps = int(round(duration_ms / dt_ms))

    params_healthy = make_minimal_ei_params(n=n, e_fraction=e_fraction, drive_i=drive_i_healthy)
    params_damaged = make_minimal_ei_params(n=n, e_fraction=e_fraction, drive_i=drive_i_damaged)
    is_e_mask = np.array([lbl == "E" for lbl in params_damaged.labels])
    G0 = np.asarray(params_damaged.G0)

    selective_rule = make_hebbian_pairwise_rule(
        k_ee=k_other_damp, k_ei=k_other_damp, k_ie=k_ie_boost, k_ii=k_other_damp)

    conditions = {
        "healthy_reference": dict(params=params_healthy, conductance_rule="hebbian", freeze_G=False),
        "damaged_no_rescue": dict(params=params_damaged, conductance_rule="hebbian", freeze_G=True),
        "damaged_uniform_rescue": dict(params=params_damaged, conductance_rule="hebbian", freeze_G=False),
        "damaged_selective_rescue": dict(params=params_damaged, conductance_rule=selective_rule, freeze_G=False),
    }

    per_condition = {}
    raw_e_activity = {}
    raw_g_ie_change = {}
    for name, cfg in conditions.items():
        e_activities, g_ie_changes = [], []
        for s in range(int(seeds)):
            voltages, G_hist, diag = _run_one(
                cfg["params"], n_steps, dt_ms, s,
                conductance_rule=cfg["conductance_rule"], freeze_G=cfg["freeze_G"])
            e_act = float(_settled_mean(voltages)[is_e_mask].mean())
            g_final = np.asarray(G_hist[-1])
            g_ie_mean_change = float(np.abs(g_final[np.ix_(is_e_mask, ~is_e_mask)]
                                             - G0[np.ix_(is_e_mask, ~is_e_mask)]).mean())
            g_ee_mean_change = float(np.abs(g_final[np.ix_(is_e_mask, is_e_mask)]
                                             - G0[np.ix_(is_e_mask, is_e_mask)]).mean())
            e_activities.append(e_act)
            g_ie_changes.append(g_ie_mean_change)
            if s == 0:
                per_condition.setdefault(name, {})["g_ee_change_seed0"] = g_ee_mean_change
            assert bool(np.isfinite(np.asarray(voltages)).all()), f"{name} seed={s} non-finite"
        raw_e_activity[name] = e_activities
        raw_g_ie_change[name] = g_ie_changes
        per_condition[name]["e_activity"] = _agg(e_activities)
        per_condition[name]["g_ie_mean_abs_change"] = _agg(g_ie_changes)

    healthy_e = per_condition["healthy_reference"]["e_activity"]["mean"]
    damaged_e = per_condition["damaged_no_rescue"]["e_activity"]["mean"]
    uniform_e = per_condition["damaged_uniform_rescue"]["e_activity"]["mean"]
    selective_e = per_condition["damaged_selective_rescue"]["e_activity"]["mean"]

    def _rescue_fraction(rescued_e):
        denom = damaged_e - healthy_e
        return None if abs(denom) < 1e-9 else float(1.0 - (rescued_e - healthy_e) / denom)

    sig_uniform_vs_selective = _significance_test(
        raw_e_activity["damaged_uniform_rescue"], raw_e_activity["damaged_selective_rescue"])
    sig_gie_uniform_vs_selective = _significance_test(
        raw_g_ie_change["damaged_uniform_rescue"], raw_g_ie_change["damaged_selective_rescue"])

    bundle = {
        "schema_version": "jaxfne.ed10_pairwise_ei_selective_rescue.v0.1.0",
        "evidence_kind": "computational_method_pairwise_gain_mechanism_comparison",
        "claim_status": "computational_control_proxy_not_biological_mechanism",
        "biological_learning_claim": False,
        "mechanism_claim_status": "not_claimed",
        "disorder_relevance_claim": False,
        "note": (
            "Method evidence only: compares uniform-gain vs. asymmetric-gain (k_ie "
            "boosted) HDP conductance rescue of an inhibition-weakened E/I circuit. "
            "No disorder model, no literature qualitative target -- this checks the "
            "narrow mechanism claim only (does asymmetric-gain HDP rescue E activity "
            "via disproportionate I->E growth vs. uniform-gain HDP), not any clinical "
            "or biological-validity claim."
        ),
        "setup": {
            "n": n, "e_fraction": e_fraction, "seeds": int(seeds), "duration_ms": duration_ms,
            "dt_ms": dt_ms, "drive_i_healthy": drive_i_healthy, "drive_i_damaged": drive_i_damaged,
            "k_ie_boost": k_ie_boost, "k_other_damp": k_other_damp,
        },
        "conditions": per_condition,
        "summary": {
            "healthy_e_activity": healthy_e,
            "damaged_e_activity": damaged_e,
            "uniform_rescue_e_activity": uniform_e,
            "selective_rescue_e_activity": selective_e,
            "uniform_rescue_fraction": _rescue_fraction(uniform_e),
            "selective_rescue_fraction": _rescue_fraction(selective_e),
            "significance_uniform_vs_selective_e_activity": sig_uniform_vs_selective,
            "significance_uniform_vs_selective_g_ie_change": sig_gie_uniform_vs_selective,
            "selective_perturbs_g_ee_less_than_uniform": (
                per_condition["damaged_selective_rescue"]["g_ee_change_seed0"]
                < per_condition["damaged_uniform_rescue"]["g_ee_change_seed0"]
            ),
        },
    }

    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / "ed10_evidence.json"
    jtfne.save_json(jtfne.json_safe(bundle), str(bundle_path))
    digest = jtfne.sha256_file(str(bundle_path))
    receipt = {"artifact": "ed10_evidence.json", "sha256": digest,
               "schema_version": bundle["schema_version"],
               "truth_status": "method_evidence_not_mechanism_validation"}
    jtfne.save_json(receipt, str(out / "ed10_receipt.json"))
    return bundle, digest, out


def main():
    ap = argparse.ArgumentParser(description="ED10 pairwise E/I HDP selective-rescue evidence bundle")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--e-fraction", type=float, default=0.75)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--duration-ms", type=float, default=5000.0)
    ap.add_argument("--dt-ms", type=float, default=0.5)
    ap.add_argument("--drive-i-healthy", type=float, default=0.3)
    ap.add_argument("--drive-i-damaged", type=float, default=0.03)
    ap.add_argument("--k-ie-boost", type=float, default=2.5)
    ap.add_argument("--k-other-damp", type=float, default=0.4)
    ap.add_argument("--out-dir", default="outputs/ed10_pairwise_ei_selective_rescue")
    args = ap.parse_args()
    bundle, digest, out = run(
        n=args.n, e_fraction=args.e_fraction, seeds=args.seeds, duration_ms=args.duration_ms,
        dt_ms=args.dt_ms, drive_i_healthy=args.drive_i_healthy, drive_i_damaged=args.drive_i_damaged,
        k_ie_boost=args.k_ie_boost, k_other_damp=args.k_other_damp, out_dir=args.out_dir)
    print(f"ED10 pairwise E/I evidence written to {out}/ (sha256={digest[:16]}...)")
    s = bundle["summary"]
    print(f"  healthy_e={s['healthy_e_activity']:.4f}  damaged_e={s['damaged_e_activity']:.4f}")
    print(f"  uniform_rescue_e={s['uniform_rescue_e_activity']:.4f}"
          f"  (rescue_fraction={s['uniform_rescue_fraction']})")
    print(f"  selective_rescue_e={s['selective_rescue_e_activity']:.4f}"
          f"  (rescue_fraction={s['selective_rescue_fraction']})")
    sig = s["significance_uniform_vs_selective_e_activity"]
    if sig["p_value"] is not None:
        print(f"  significance (uniform vs selective, e_activity): p={sig['p_value']:.4g},"
              f" Cohen's d={sig['cohens_d']:.2f}")
    print(f"  selective perturbs G_ee less than uniform: {s['selective_perturbs_g_ee_less_than_uniform']}")


if __name__ == "__main__":
    main()
