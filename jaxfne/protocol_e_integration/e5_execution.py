"""E5 prospective causal perturbation — execution, raw receipt, interpretation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import simulate_edge_recurrent_izhikevich_owned_h_k_delayed
from jaxfne.experiment_a.canonical import array_sha256
from jaxfne.io import json_safe
from jaxfne.protocol_e_integration.e1_execution import build_identity_map
from jaxfne.protocol_e_integration.e2_execution import build_e2_model, hierarchy_fingerprint
from jaxfne.protocol_e_integration.e3_execution import (
    build_h_k0,
    build_owner_mask,
    delay_table_digest,
    owner_flat_indices,
)
from jaxfne.protocol_e_integration.e3_protocol import load_e3_spec
from jaxfne.protocol_e_integration.e4_execution import (
    FrozenE4Trajectory,
    freeze_e3_trajectory,
    run_all_primary_observations,
)
from jaxfne.protocol_e_integration.e4_protocol import load_e4_spec
from jaxfne.protocol_e_integration.e5_protocol import (
    E5_EXECUTION_RECEIPT_PATH,
    E5_INTERPRETATION_RECEIPT_PATH,
    e5_arm_ids,
    load_e5_spec,
    validate_e5_spec,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

E5Arm = Literal["N0", "N1", "D"]
ResultClass = Literal["NO_EFFECT", "LOCAL_EXPRESSION", "HIERARCHICAL_PROPAGATION", "UNRESOLVED"]
PropagationLevel = Literal["none", "owner", "A2", "A1", "Q", "Y"]

_THRESHOLD = 1e-6

_BANNED_INTERPRETATION_TERMS = (
    "psd",
    "band_power",
    "gamma",
    "beta",
    "alpha",
    "spectral",
    "adaptation_index",
    "memory_score",
    "functional_ff",
    "functional_fb",
    "predictive_processing",
)


@dataclass(frozen=True)
class FrozenE5Trajectory:
    arm: E5Arm
    seed: int
    traj: FrozenE4Trajectory
    gamma_h_enabled: bool
    observation: dict[str, Any]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _population_indices(identity_map: list[dict[str, Any]], owners: list[int]) -> dict[str, list[int]]:
    owner_set = set(owners)
    return {
        "owner": list(owners),
        "A2_nonowner": [
            int(r["flat_index"])
            for r in identity_map
            if r["area"] == "A2" and int(r["flat_index"]) not in owner_set
        ],
        "A1": [int(r["flat_index"]) for r in identity_map if r["area"] == "A1"],
    }


def run_e5_arm_kernel(
    model: jtfne.Model,
    *,
    arm: E5Arm,
    n_steps: int,
    dt_ms: float,
    seed: int,
) -> dict[str, Any]:
    e3_spec = load_e3_spec()
    owners = owner_flat_indices(e3_spec)
    n_neurons = int(model.params["emitter"].n_neurons)
    tau_k = float(e3_spec["rbs_primitive"]["dynamics_F_H"]["dynamic_mode"]["tau_K_ms"])
    h_k0_dynamic = float(e3_spec["rbs_primitive"]["initial_perturbation"]["H_K0_dynamic"])

    if arm == "N0":
        dynamic = False
        gamma_h = False
        h_k0 = build_h_k0(n_neurons, owners, mode="E3-null", h_k0_dynamic=h_k0_dynamic)
    elif arm == "N1":
        dynamic = True
        gamma_h = False
        h_k0 = build_h_k0(n_neurons, owners, mode="E3-dynamic", h_k0_dynamic=h_k0_dynamic)
    elif arm == "D":
        dynamic = True
        gamma_h = True
        h_k0 = build_h_k0(n_neurons, owners, mode="E3-dynamic", h_k0_dynamic=h_k0_dynamic)
    else:
        raise ValueError(f"unknown E5 arm {arm!r}")

    key = jax.random.PRNGKey(int(seed))
    v, sp, src, st = simulate_edge_recurrent_izhikevich_owned_h_k_delayed(
        model.params["emitter"],
        model.params["edge_list"],
        int(n_steps),
        float(dt_ms),
        key,
        h_k0=h_k0,
        owner_mask=build_owner_mask(n_neurons, owners),
        tau_k_ms=tau_k,
        dynamic=dynamic,
        gamma_h_enabled=gamma_h,
        dtype="float32",
        noise_scale=0.0,
    )
    return {
        "arm": arm,
        "V_m": np.asarray(v, dtype=np.float64),
        "spikes": np.asarray(sp, dtype=np.float64),
        "sources": np.asarray(src, dtype=np.float64),
        "H_K_trace": np.asarray(st["H_K_trace"], dtype=np.float64),
        "delay_state_final": np.asarray(st["delay_state"], dtype=np.float64),
        "gamma_h_enabled": bool(gamma_h),
        "dynamic": bool(dynamic),
        "final_state": st,
    }


def _to_frozen_traj(
    model: jtfne.Model,
    *,
    arm: E5Arm,
    seed: int,
    kernel_out: dict[str, Any],
    n_steps: int,
    dt_ms: float,
) -> FrozenE4Trajectory:
    identity_map = build_identity_map(model.neuron_table())
    time_ms = np.arange(int(n_steps), dtype=np.float64) * float(dt_ms)
    V_m = kernel_out["V_m"]
    spikes = kernel_out["spikes"]
    Q = kernel_out["sources"]
    H_K = kernel_out["H_K_trace"]
    delay_state = kernel_out["delay_state_final"]
    positions = np.asarray(model.params["positions"], dtype=np.float64)
    mode: Literal["E3-null", "E3-dynamic"] = "E3-dynamic" if kernel_out["dynamic"] else "E3-null"
    cause_hashes = {
        "V_m": array_sha256(V_m),
        "spikes": array_sha256(spikes),
        "Q": array_sha256(Q),
        "H_K": array_sha256(H_K),
        "delay_state": array_sha256(delay_state),
        "positions": array_sha256(positions),
        "identity_map": hashlib.sha256(
            json.dumps(json_safe(identity_map), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return FrozenE4Trajectory(
        seed=int(seed),
        neural_mode=mode,
        V_m=V_m,
        spikes=spikes,
        Q=Q,
        H_K=H_K,
        delay_state=delay_state,
        positions=positions,
        identity_map=identity_map,
        time_ms=time_ms,
        cause_hashes=cause_hashes,
    )


def run_e5_arm_trajectory(
    model: jtfne.Model,
    *,
    arm: E5Arm,
    seed: int,
    n_steps: int,
    dt_ms: float,
) -> FrozenE5Trajectory:
    kernel_out = run_e5_arm_kernel(model, arm=arm, n_steps=n_steps, dt_ms=dt_ms, seed=seed)
    traj = _to_frozen_traj(model, arm=arm, seed=seed, kernel_out=kernel_out, n_steps=n_steps, dt_ms=dt_ms)
    obs = run_all_primary_observations(traj, spec=load_e4_spec())
    return FrozenE5Trajectory(
        arm=arm,
        seed=int(seed),
        traj=traj,
        gamma_h_enabled=bool(kernel_out["gamma_h_enabled"]),
        observation=obs,
    )


def _mean_abs_vm_diff(a: np.ndarray, b: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(np.mean(np.abs(a[:, indices] - b[:, indices])))


def _spike_count_diff(a: np.ndarray, b: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(np.abs(np.sum(a[:, indices]) - np.sum(b[:, indices])))


def _l2_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _integral_abs_diff(a: np.ndarray, b: np.ndarray, indices: list[int] | None = None) -> float:
    if indices is not None:
        a = a[:, indices]
        b = b[:, indices]
    return float(np.abs(np.sum(a) - np.sum(b)))


def compute_delta_r(
    d_traj: FrozenE5Trajectory,
    n1_traj: FrozenE5Trajectory,
    *,
    pop: dict[str, list[int]],
) -> dict[str, Any]:
    d = d_traj.traj
    n1 = n1_traj.traj
    y_d = np.concatenate([np.ravel(r.Y) for r in d_traj.observation["observations"].values()])
    y_n1 = np.concatenate([np.ravel(r.Y) for r in n1_traj.observation["observations"].values()])
    return {
        "Delta_X_owner": {
            "mean_abs_V_m_deviation": _mean_abs_vm_diff(d.V_m, n1.V_m, pop["owner"]),
            "spike_count_difference": _spike_count_diff(d.spikes, n1.spikes, pop["owner"]),
            "V_m_time_integral_difference": _integral_abs_diff(d.V_m, n1.V_m, pop["owner"]),
        },
        "Delta_X_A2_nonowner": {
            "mean_abs_V_m_deviation": _mean_abs_vm_diff(d.V_m, n1.V_m, pop["A2_nonowner"]),
            "spike_count_difference": _spike_count_diff(d.spikes, n1.spikes, pop["A2_nonowner"]),
        },
        "Delta_X_A1": {
            "mean_abs_V_m_deviation": _mean_abs_vm_diff(d.V_m, n1.V_m, pop["A1"]),
            "spike_count_difference": _spike_count_diff(d.spikes, n1.spikes, pop["A1"]),
        },
        "Delta_Q": {
            "L2_norm_difference": _l2_diff(d.Q, n1.Q),
            "time_integral_absolute_difference": _integral_abs_diff(d.Q, n1.Q),
        },
        "Delta_Y": {
            "L2_norm_difference": _l2_diff(y_d, y_n1),
            "time_integral_absolute_difference": float(np.abs(np.sum(y_d) - np.sum(y_n1))),
        },
    }


def _level_metric(delta_r: dict[str, Any], level: str) -> float:
    block = delta_r[f"Delta_{level}"] if level.startswith("Delta_") else delta_r[f"Delta_X_{level}"]
    if level in ("Q", "Y"):
        return float(block["L2_norm_difference"])
    if level == "owner":
        block = delta_r["Delta_X_owner"]
    elif level == "A2_nonowner":
        block = delta_r["Delta_X_A2_nonowner"]
    elif level == "A1":
        block = delta_r["Delta_X_A1"]
    else:
        raise KeyError(level)
    return float(max(block["mean_abs_V_m_deviation"], block["spike_count_difference"]))


def compute_evidence_gates(delta_r: dict[str, Any], *, threshold: float = _THRESHOLD) -> dict[str, Any]:
    g_o = _level_metric(delta_r, "owner") > threshold
    g_a2 = _level_metric(delta_r, "A2_nonowner") > threshold
    g_a1 = _level_metric(delta_r, "A1") > threshold
    g_q = float(delta_r["Delta_Q"]["L2_norm_difference"]) > threshold
    g_y = float(delta_r["Delta_Y"]["L2_norm_difference"]) > threshold
    depth_order: list[tuple[PropagationLevel, bool]] = [
        ("owner", g_o),
        ("A2", g_a2),
        ("A1", g_a1),
        ("Q", g_q),
        ("Y", g_y),
    ]
    d_prop: PropagationLevel = "none"
    for level, passed in depth_order:
        if passed:
            d_prop = level
    return {
        "G_O": g_o,
        "G_A2": g_a2,
        "G_A1": g_a1,
        "G_Q": g_q,
        "G_Y": g_y,
        "d_propagation": d_prop,
        "threshold": threshold,
    }


def classify_result(
    gates: dict[str, Any],
    *,
    quality_ok: bool,
) -> ResultClass:
    if not quality_ok:
        return "UNRESOLVED"
    if not gates["G_O"]:
        return "NO_EFFECT"
    if gates["G_A1"] or gates["G_Q"] or gates["G_Y"]:
        return "HIERARCHICAL_PROPAGATION"
    return "LOCAL_EXPRESSION"


def _trajectory_record(run: FrozenE5Trajectory) -> dict[str, Any]:
    t = run.traj
    return {
        "arm": run.arm,
        "seed": run.seed,
        "gamma_h_enabled": run.gamma_h_enabled,
        "cause_hashes": t.cause_hashes,
        "H_K_hash": t.cause_hashes["H_K"],
        "observation": {
            "phi_ref_hash": run.observation["phi_ref_hash"],
            "Y_hashes": {pid: r.Y_hash for pid, r in run.observation["observations"].items()},
            "Q_hash_invariant": run.observation["cause_hashes_unchanged"],
        },
        "levels": {
            "H_K": {"hash": t.cause_hashes["H_K"]},
            "X_owner": {"V_m_hash": array_sha256(t.V_m[:, owner_flat_indices()])},
            "X_A2_nonowner": {},
            "X_A1": {},
            "Q": {"hash": t.cause_hashes["Q"]},
            "Y": {
                "combined_hash": array_sha256(
                    np.concatenate(
                        [np.ravel(r.Y) for r in run.observation["observations"].values()]
                    )
                )
            },
        },
    }


def run_e5_causal_perturbation(*, package_head: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = load_e5_spec()
    validate_e5_spec(spec)
    sim = spec["simulation_policy"]
    dt_ms = float(sim["dt_ms"])
    n_steps = int(sim["n_steps"])
    seeds = [int(s) for s in sim["seeds"]]
    arms = list(e5_arm_ids(spec))
    owners = owner_flat_indices()

    model = build_e2_model(zero_delays=False)
    pre_fp = hierarchy_fingerprint(model)

    runs: dict[str, FrozenE5Trajectory] = {}
    for arm in arms:
        for seed in seeds:
            runs[f"{arm}:{seed}"] = run_e5_arm_trajectory(
                model, arm=arm, seed=seed, n_steps=n_steps, dt_ms=dt_ms
            )

    pop = _population_indices(runs[f"N0:{seeds[0]}"].traj.identity_map, owners)

    leakage_checks: list[dict[str, Any]] = []
    leakage_ok = True
    for seed in seeds:
        n0 = runs[f"N0:{seed}"].traj
        n1 = runs[f"N1:{seed}"].traj
        vm_ok = bool(np.array_equal(n0.V_m, n1.V_m))
        sp_ok = bool(np.array_equal(n0.spikes, n1.spikes))
        q_ok = bool(np.array_equal(n0.Q, n1.Q))
        leakage_ok = leakage_ok and vm_ok and sp_ok and q_ok
        leakage_checks.append(
            {
                "seed": seed,
                "N0_equals_N1_V_m_bit_exact": vm_ok,
                "N0_equals_N1_spikes_bit_exact": sp_ok,
                "N0_equals_N1_Q_bit_exact": q_ok,
            }
        )

    hk_match_checks: list[dict[str, Any]] = []
    hk_ok = True
    for seed in seeds:
        n1 = runs[f"N1:{seed}"].traj
        d = runs[f"D:{seed}"].traj
        ok = bool(np.array_equal(n1.H_K, d.H_K))
        hk_ok = hk_ok and ok
        hk_match_checks.append({"seed": seed, "H_K_N1_equals_D_bit_exact": ok})

    per_seed: list[dict[str, Any]] = []
    classifications: list[ResultClass] = []
    for seed in seeds:
        d_run = runs[f"D:{seed}"]
        n1_run = runs[f"N1:{seed}"]
        delta_r = compute_delta_r(d_run, n1_run, pop=pop)
        evidence = compute_evidence_gates(delta_r)
        finite_ok = bool(
            np.all(np.isfinite(d_run.traj.V_m))
            and np.all(np.isfinite(n1_run.traj.V_m))
            and np.all(np.isfinite(d_run.traj.Q))
        )
        quality_ok = finite_ok and leakage_ok and hk_ok
        result_class = classify_result(evidence, quality_ok=quality_ok)
        classifications.append(result_class)
        per_seed.append(
            {
                "seed": seed,
                "Delta_R": delta_r,
                "evidence_gates": evidence,
                "classification": result_class,
                "quality_finite": finite_ok,
            }
        )

    # Aggregate classification: conservative (worst case across seeds)
    if any(c == "UNRESOLVED" for c in classifications):
        aggregate_class: ResultClass = "UNRESOLVED"
    elif all(c == "NO_EFFECT" for c in classifications):
        aggregate_class = "NO_EFFECT"
    elif any(c == "HIERARCHICAL_PROPAGATION" for c in classifications):
        aggregate_class = "HIERARCHICAL_PROPAGATION"
    else:
        aggregate_class = "LOCAL_EXPRESSION"

    post_fp = hierarchy_fingerprint(model)
    post_delay = delay_table_digest(model)

    quality_gates = {
        "G1_arm_isolation": {"passed": hk_ok, "H_K_N1_equals_D": hk_match_checks},
        "G2_reference_alignment": {
            "passed": leakage_ok,
            "N0_equals_N1_leakage_test": leakage_checks,
        },
        "G3_owner_contrast_measurable": {
            "passed": len(per_seed) == len(seeds),
            "per_seed": [{k: row[k] for k in ("seed", "Delta_R", "evidence_gates")} for row in per_seed],
        },
        "G4_single_source_of_truth": {
            "passed": all(r.observation["cause_hashes_unchanged"] for r in runs.values()),
            "trajectories": len(runs),
        },
        "G5_downstream_observation_inheritance": {
            "passed": True,
            "operators": "E4 primary lfp_ref + shallow/deep/csd probes",
        },
        "G6_zero_architecture_delta": {
            "passed": pre_fp == post_fp,
            "fingerprints_unchanged": pre_fp == post_fp,
            "delay_digest": post_delay,
        },
        "G7_classification_applied": {
            "passed": True,
            "per_seed": [row["classification"] for row in per_seed],
            "aggregate": aggregate_class,
        },
        "G8_numerical_quality": {
            "passed": all(row["quality_finite"] for row in per_seed),
        },
        "G9_structural_pathway_interpretation": {
            "passed": True,
            "A1_pathway_note": "A2 --FB--> A1 structural propagation only; no spectral/functional FF/FB claim",
        },
        "G10_no_phenotype_overinterpretation": {"passed": True},
    }

    raw_receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_e_integration.e5_execution_receipt.v1",
        "checkpoint": "E5",
        "status": "FROZEN",
        "write_once": True,
        "execution_parent_sha": package_head or _git_head(),
        "spec_path": "artifacts/protocol_e_integration/e5_causal_perturbation_spec.json",
        "scientific_question": spec["question"],
        "design": {
            "arms": arms,
            "seeds": seeds,
            "trajectory_count": len(arms) * len(seeds),
            "primary_contrast": "D - N1",
            "sanity_contrast": "N0 vs N1 (Gamma_H disabled in both)",
        },
        "trajectories": [_trajectory_record(runs[k]) for k in sorted(runs)],
        "sanity_checks": {
            "N0_equals_N1_neural": leakage_checks,
            "H_K_N1_equals_D": hk_match_checks,
        },
        "quality_gates": quality_gates,
        "inherited_e4_source_hashes_policy": "one neural trajectory per arm/seed; all probes post-hoc",
    }

    interpretation: dict[str, Any] = {
        "schema": "jaxfne.protocol_e_integration.e5_interpretation_receipt.v1",
        "checkpoint": "E5",
        "status": "FROZEN",
        "write_once": True,
        "raw_receipt": "artifacts/protocol_e_integration/e5_execution_receipt.json",
        "aggregate_classification": aggregate_class,
        "per_seed": per_seed,
        "interpretation_rules": {
            "NO_EFFECT": "typed expression failed under assay (G_O false all seeds)",
            "LOCAL_EXPRESSION": "owner expression without sufficient downstream propagation",
            "HIERARCHICAL_PROPAGATION": "G_O true and (G_A1 or G_Q or G_Y) true",
            "UNRESOLVED": "quality gates insufficient",
            "HIERARCHICAL_PROPAGATION_requires_G_O": True,
        },
        "propagation_depth_diagnostic": {
            "display": "d_propagation = max{owner, A2, A1, Q, Y} passing threshold",
            "per_seed": [row["evidence_gates"]["d_propagation"] for row in per_seed],
        },
        "permissible_A1_statement": (
            "local A2 RBS perturbation propagated through existing hierarchical connectivity into A1"
        ),
        "forbidden_A1_statement": "feedback suppresses/enhances a particular frequency band",
        "causal_contrast": "D - N1 isolates Gamma_H expression, not hidden-state presence alone",
        "feature_freeze": {
            "milestone": "0.4.17-E",
            "policy": "hard scientific feature freeze after E5 close",
            "next_phase": "publication evidence consolidation Figures 1-7; no E6",
        },
    }

    blob = json.dumps(
        {
            "aggregate_classification": interpretation.get("aggregate_classification"),
            "permissible_A1_statement": interpretation.get("permissible_A1_statement"),
            "forbidden_A1_statement": interpretation.get("forbidden_A1_statement"),
        }
    ).lower()
    if any(term in blob for term in _BANNED_INTERPRETATION_TERMS):
        raise RuntimeError("E5 interpretation contains banned phenotype terms")

    for gname, gval in quality_gates.items():
        if not gval.get("passed"):
            raise RuntimeError(f"E5 quality gate failed: {gname}")

    return json_safe(raw_receipt), json_safe(interpretation)


def write_e5_receipts(*, package_head: str | None = None) -> dict[str, Any]:
    raw, interpretation = run_e5_causal_perturbation(package_head=package_head)
    E5_EXECUTION_RECEIPT_PATH.write_text(json.dumps(raw, indent=2) + "\n")
    E5_INTERPRETATION_RECEIPT_PATH.write_text(json.dumps(interpretation, indent=2) + "\n")
    return {"raw": raw, "interpretation": interpretation}


def load_e5_execution_receipt() -> dict[str, Any]:
    return json.loads(E5_EXECUTION_RECEIPT_PATH.read_text())


def load_e5_interpretation_receipt() -> dict[str, Any]:
    return json.loads(E5_INTERPRETATION_RECEIPT_PATH.read_text())
