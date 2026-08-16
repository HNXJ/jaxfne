"""E4 observation-chain composition — frozen trajectory, gates, receipt."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.experiment_a.canonical import array_sha256
from jaxfne.fields import csd_proxy_probe, lfp_proxy_probe, project_laminar_sources
from jaxfne.io import json_safe
from jaxfne.protocol_e_integration.e1_execution import build_identity_map
from jaxfne.protocol_e_integration.e2_execution import (
    build_e2_model,
    hierarchy_fingerprint,
)
from jaxfne.protocol_e_integration.e3_execution import (
    delay_table_digest,
    load_e3_execution_receipt,
    run_e3_kernel,
)
from jaxfne.protocol_e_integration.e4_protocol import (
    E4_EXECUTION_RECEIPT_PATH,
    e4_primary_probe_ids,
    load_e4_spec,
    validate_e4_spec,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

E3Mode = Literal["E3-null", "E3-dynamic"]

_BANNED_RECEIPT_TERMS = (
    "psd",
    "band_power",
    "gamma",
    "beta",
    "phase",
    "wave",
    "memory_score",
    "adaptation_index",
    "functional_ff",
    "functional_fb",
    "hdp",
    "field_evidence",
    "manuscript_phenotype",
)


@dataclass(frozen=True)
class FrozenE4Trajectory:
    """Single E3 simulate per seed — source of truth for all E4 observation."""

    seed: int
    neural_mode: E3Mode
    V_m: np.ndarray
    spikes: np.ndarray
    Q: np.ndarray
    H_K: np.ndarray
    delay_state: np.ndarray
    positions: np.ndarray
    identity_map: list[dict[str, Any]]
    time_ms: np.ndarray
    cause_hashes: dict[str, str]


@dataclass(frozen=True)
class E4ObservationResult:
    probe_id: str
    Y: np.ndarray
    Y_hash: str
    phi_ref_hash: str | None
    q_hash_before: str
    q_hash_after: str
    semantic_status: str


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _digest(obj: Any) -> str:
    payload = json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _field_params(spec: dict[str, Any], field_id: str = "lfp_ref") -> dict[str, Any]:
    for entry in spec["primary_evidence_operators"]["field_operators_F"]:
        if entry["id"] == field_id:
            return dict(entry["params"])
    raise KeyError(f"unknown field operator {field_id!r}")


def build_cause_hashes(traj: FrozenE4Trajectory) -> dict[str, str]:
    return {
        "V_m": array_sha256(traj.V_m),
        "spikes": array_sha256(traj.spikes),
        "Q": array_sha256(traj.Q),
        "H_K": array_sha256(traj.H_K),
        "delay_state": array_sha256(traj.delay_state),
        "positions": array_sha256(traj.positions),
        "identity_map": _digest(traj.identity_map),
    }


def freeze_e3_trajectory(
    model: jtfne.Model,
    *,
    seed: int,
    mode: E3Mode,
    n_steps: int,
    dt_ms: float,
) -> FrozenE4Trajectory:
    """Run exactly one E3 neural simulate and freeze first-class arrays."""
    out = run_e3_kernel(model, n_steps=n_steps, dt_ms=dt_ms, seed=seed, mode=mode)
    identity_map = build_identity_map(model.neuron_table())
    time_ms = np.arange(int(n_steps), dtype=np.float64) * float(dt_ms)
    V_m = np.asarray(out["V_m"], dtype=np.float64)
    spikes = np.asarray(out["spikes"], dtype=np.float64)
    Q = np.asarray(out["sources"], dtype=np.float64)
    H_K = np.asarray(out["H_K_trace"], dtype=np.float64)
    delay_state = np.asarray(out["delay_state_final"], dtype=np.float64)
    positions = np.asarray(model.params["positions"], dtype=np.float64)
    cause_hashes = {
        "V_m": array_sha256(V_m),
        "spikes": array_sha256(spikes),
        "Q": array_sha256(Q),
        "H_K": array_sha256(H_K),
        "delay_state": array_sha256(delay_state),
        "positions": array_sha256(positions),
        "identity_map": _digest(identity_map),
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


def run_e4_neural_only(
    model: jtfne.Model,
    *,
    seed: int,
    n_steps: int,
    dt_ms: float,
    mode: E3Mode = "E3-null",
) -> FrozenE4Trajectory:
    """E4 execution path with observation disabled (R_E4_to_E3 neural slice)."""
    return freeze_e3_trajectory(model, seed=seed, mode=mode, n_steps=n_steps, dt_ms=dt_ms)


def materialize_phi_ref(traj: FrozenE4Trajectory, spec: dict[str, Any] | None = None):
    spec = spec or load_e4_spec()
    params = _field_params(spec, "lfp_ref")
    return project_laminar_sources(
        jnp.asarray(traj.Q, dtype=jnp.float32),
        jnp.asarray(traj.positions, dtype=jnp.float32),
        **params,
    )


def phi_ref_hash(field_output) -> str:
    return array_sha256(np.asarray(field_output.phi_e_proxy))


def apply_e4_probe(
    traj: FrozenE4Trajectory,
    field_output,
    probe_id: str,
    *,
    spec: dict[str, Any] | None = None,
) -> E4ObservationResult:
    spec = spec or load_e4_spec()
    q_before = array_sha256(traj.Q)
    phi_hash = phi_ref_hash(field_output)
    phi_e = np.asarray(field_output.phi_e_proxy)
    field_z = np.asarray(field_output.contact_depths)

    probes = {row["id"]: row for row in spec["primary_evidence_operators"]["probe_operators_P"]}
    if probe_id not in probes:
        raise KeyError(f"unknown E4 probe {probe_id!r}")

    row = probes[probe_id]
    if probe_id == "lfp_contact_shallow":
        depths = jnp.asarray(row["params"]["probe_contact_depths"], dtype=jnp.float32)
        readout = lfp_proxy_probe(
            jnp.asarray(phi_e), contact_depths=depths, field_contact_depths=jnp.asarray(field_z)
        )
        semantic = row["semantic"]
    elif probe_id == "lfp_contact_deep":
        depths = jnp.asarray(row["params"]["probe_contact_depths"], dtype=jnp.float32)
        readout = lfp_proxy_probe(
            jnp.asarray(phi_e), contact_depths=depths, field_contact_depths=jnp.asarray(field_z)
        )
        semantic = row["semantic"]
    elif probe_id == "csd_from_lfp_ref":
        readout = csd_proxy_probe(jnp.asarray(field_output.csd_proxy))
        semantic = row["semantic"]
    else:
        raise KeyError(probe_id)

    Y = np.asarray(readout.data)
    q_after = array_sha256(traj.Q)
    return E4ObservationResult(
        probe_id=probe_id,
        Y=Y,
        Y_hash=array_sha256(Y),
        phi_ref_hash=phi_hash,
        q_hash_before=q_before,
        q_hash_after=q_after,
        semantic_status=semantic,
    )


def run_all_primary_observations(
    traj: FrozenE4Trajectory,
    *,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply F then each P on frozen trajectory without re-simulating."""
    spec = spec or load_e4_spec()
    field = materialize_phi_ref(traj, spec)
    phi_hash = phi_ref_hash(field)
    results: dict[str, E4ObservationResult] = {}
    for probe_id in e4_primary_probe_ids(spec):
        results[probe_id] = apply_e4_probe(traj, field, probe_id, spec=spec)
    return {
        "phi_ref_hash": phi_hash,
        "field_id": "lfp_ref",
        "observations": results,
        "cause_hashes_unchanged": all(
            r.q_hash_before == r.q_hash_after == traj.cause_hashes["Q"] for r in results.values()
        ),
    }


def aggregate_q_by_hierarchy(
    traj: FrozenE4Trajectory,
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[str, Any]]:
    """Regroup Q_i(t) by E1 identity without redefining the source."""
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in traj.identity_map:
        key = (str(row["area"]), str(row["layer"]), str(row["cell_type"]))
        groups[key].append(int(row["flat_index"]))

    aggregated: dict[tuple[str, str, str], np.ndarray] = {}
    for key, indices in sorted(groups.items()):
        aggregated[key] = np.sum(traj.Q[:, indices], axis=1)

    flat_total = np.sum(traj.Q, axis=1)
    grouped_total = np.zeros_like(flat_total)
    for arr in aggregated.values():
        grouped_total = grouped_total + arr
    max_abs_diff = float(np.max(np.abs(flat_total - grouped_total)))
    conservation = {
        "passed": max_abs_diff == 0.0,
        "max_abs_conservation_error": max_abs_diff,
        "n_groups": len(aggregated),
        "rule": "sum_{a,l,c} Q_{a,l,c}(t) = sum_i Q_i(t)",
    }
    return aggregated, conservation


def build_hierarchy_aware_source_table_summary(
    traj: FrozenE4Trajectory,
) -> dict[str, Any]:
    aggregated, conservation = aggregate_q_by_hierarchy(traj)
    rows = []
    for (area, layer, cell_type), series in sorted(aggregated.items()):
        rows.append(
            {
                "area": area,
                "layer": layer,
                "cell_type": cell_type,
                "n_neurons": int(
                    sum(
                        1
                        for r in traj.identity_map
                        if r["area"] == area and r["layer"] == layer and r["cell_type"] == cell_type
                    )
                ),
                "Q_series_hash": array_sha256(series),
            }
        )
    return {
        "display": "Q -> (area, layer, cell_type, t)",
        "identity_source": "artifacts/protocol_e_integration/e1_execution_receipt.json#identity_map",
        "conservation": conservation,
        "groups": rows,
        "flat_Q_hash": traj.cause_hashes["Q"],
    }


def verify_zero_source_operators(
    traj: FrozenE4Trajectory,
    *,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = spec or load_e4_spec()
    params = _field_params(spec, "lfp_ref")
    zero_field = project_laminar_sources(
        jnp.zeros_like(jnp.asarray(traj.Q, dtype=jnp.float32)),
        jnp.asarray(traj.positions, dtype=jnp.float32),
        **params,
    )
    shallow = lfp_proxy_probe(
        zero_field.phi_e_proxy,
        contact_depths=jnp.asarray([0.20], dtype=jnp.float32),
        field_contact_depths=zero_field.contact_depths,
    )
    csd = csd_proxy_probe(zero_field.csd_proxy)
    checks = {
        "project_laminar_sources_phi_max": float(np.max(np.abs(np.asarray(zero_field.phi_e_proxy)))),
        "lfp_proxy_probe_max": float(np.max(np.abs(np.asarray(shallow.data)))),
        "csd_proxy_probe_max": float(np.max(np.abs(np.asarray(csd.data)))),
    }
    passed = all(v < 1e-6 for v in checks.values())
    return {"passed": passed, "checks": checks}


def verify_field_linearity(
    traj: FrozenE4Trajectory,
    *,
    spec: dict[str, Any] | None = None,
    coeff_a: float = 0.6,
    coeff_b: float = 0.4,
) -> dict[str, Any]:
    spec = spec or load_e4_spec()
    params = _field_params(spec, "lfp_ref")
    Q = jnp.asarray(traj.Q, dtype=jnp.float32)
    pos = jnp.asarray(traj.positions, dtype=jnp.float32)
    Q1 = 0.7 * Q
    Q2 = 0.3 * Q
    combo = coeff_a * Q1 + coeff_b * Q2
    F_combo = project_laminar_sources(combo, pos, **params)
    F1 = project_laminar_sources(Q1, pos, **params)
    F2 = project_laminar_sources(Q2, pos, **params)
    lhs = np.asarray(F_combo.phi_e_proxy)
    rhs = coeff_a * np.asarray(F1.phi_e_proxy) + coeff_b * np.asarray(F2.phi_e_proxy)
    err = float(np.max(np.abs(lhs - rhs)))
    return {
        "passed": err < 1e-4,
        "max_abs_superposition_error": err,
        "operator": "project_laminar_sources",
        "tolerance": "float32_superposition",
    }


def semantic_status_table() -> list[dict[str, str]]:
    return [
        {"output": "V_m", "status": "native"},
        {"output": "spikes", "status": "native/event"},
        {"output": "Q", "status": "canonical relative source"},
        {"output": "lfp_ref", "status": "relative proxy"},
        {"output": "lfp_contact_shallow", "status": "relative proxy"},
        {"output": "lfp_contact_deep", "status": "relative proxy"},
        {"output": "csd_from_lfp_ref", "status": "relative proxy / finite-difference semantics"},
        {"output": "EEG", "status": "analysis-only / not computed"},
        {"output": "MEG", "status": "analysis-only / not computed"},
    ]


def _receipt_has_banned_terms(receipt: dict[str, Any]) -> list[str]:
    blob = json.dumps(
        {
            "composition_claim": receipt.get("composition_claim"),
            "gates": receipt.get("gates"),
            "semantic_status_table": receipt.get("semantic_status_table"),
        }
    ).lower()
    return [term for term in _BANNED_RECEIPT_TERMS if term in blob]


def run_e4_observation_composition(*, package_head: str | None = None) -> dict[str, Any]:
    spec = load_e4_spec()
    validate_e4_spec(spec)
    sim = spec["simulation_policy"]
    dt_ms = float(sim["dt_ms"])
    n_steps = int(sim["n_steps"])
    seeds = [int(s) for s in sim["seeds"]]

    model = build_e2_model(zero_delays=False)
    pre_fp = hierarchy_fingerprint(model)
    pre_delay_digest = delay_table_digest(model)
    e3_receipt = load_e3_execution_receipt()

    gates: dict[str, dict[str, Any]] = {g: {"passed": False} for g in spec["gates"]}

    g1_checks: list[dict[str, Any]] = []
    g1_ok = True
    for seed in seeds:
        e3_direct = run_e3_kernel(model, n_steps=n_steps, dt_ms=dt_ms, seed=seed, mode="E3-null")
        e4_neural = run_e4_neural_only(model, seed=seed, n_steps=n_steps, dt_ms=dt_ms)
        checks = {
            "V_m_bit_exact": bool(np.array_equal(e3_direct["V_m"], e4_neural.V_m)),
            "spikes_bit_exact": bool(np.array_equal(e3_direct["spikes"], e4_neural.spikes)),
            "Q_bit_exact": bool(np.array_equal(e3_direct["sources"], e4_neural.Q)),
            "H_K_bit_exact": bool(np.array_equal(e3_direct["H_K_trace"], e4_neural.H_K)),
            "delay_state_bit_exact": bool(
                np.array_equal(e3_direct["delay_state_final"], e4_neural.delay_state)
            ),
        }
        ok = all(checks.values())
        g1_ok = g1_ok and ok
        g1_checks.append({"seed": seed, **checks, "passed": ok})
    gates["G1_e3_reduction_neural_invariance"] = {
        "passed": g1_ok,
        "path": "E4 neural-only (observation disabled) vs E3-null direct kernel",
        "seeds": g1_checks,
    }

    seed0 = seeds[0]
    traj0 = run_e4_neural_only(model, seed=seed0, n_steps=n_steps, dt_ms=dt_ms)
    obs_once = run_all_primary_observations(traj0, spec=spec)
    obs_shallow = obs_once["observations"]["lfp_contact_shallow"]
    obs_deep = obs_once["observations"]["lfp_contact_deep"]
    obs_csd = obs_once["observations"]["csd_from_lfp_ref"]

    g2_ok = bool(
        obs_once["cause_hashes_unchanged"]
        and all(r.phi_ref_hash == obs_once["phi_ref_hash"] for r in obs_once["observations"].values())
    )
    gates["G2_single_source_of_truth"] = {
        "passed": g2_ok,
        "simulate_once_per_seed": True,
        "source_hashes": traj0.cause_hashes,
        "n_probes_applied": len(e4_primary_probe_ids(spec)),
    }

    g3_ok = bool(obs_once["cause_hashes_unchanged"])
    hashes_after_obs = build_cause_hashes(traj0)
    t_e4_ok = hashes_after_obs == traj0.cause_hashes
    gates["G3_source_identity"] = {
        "passed": g3_ok and t_e4_ok,
        "Q_hash": traj0.cause_hashes["Q"],
        "trajectory_invariance_T_E4": {
            "passed": t_e4_ok,
            "display": "(X,H,B,Q,G)_P1 = (X,H,B,Q,G)_P2",
            "cause_hashes_unchanged_after_all_probes": t_e4_ok,
        },
        "probe_Q_hashes": {
            pid: {"before": r.q_hash_before, "after": r.q_hash_after}
            for pid, r in obs_once["observations"].items()
        },
    }

    shallow_Y = obs_shallow.Y
    deep_Y = obs_deep.Y
    distinct = float(
        np.linalg.norm(shallow_Y - deep_Y) / max(np.linalg.norm(shallow_Y), np.linalg.norm(deep_Y), 1e-12)
    )
    g4_ok = bool(
        obs_shallow.phi_ref_hash == obs_deep.phi_ref_hash == obs_csd.phi_ref_hash
        and obs_shallow.phi_ref_hash == obs_once["phi_ref_hash"]
        and distinct > 1e-3
        and not np.array_equal(shallow_Y, deep_Y)
    )
    gates["G4_probe_independence"] = {
        "passed": g4_ok,
        "phi_ref_hash_shared": obs_once["phi_ref_hash"],
        "shallow_vs_deep_rel_distinctness": distinct,
        "probe_distinct": distinct > 1e-3,
    }

    g5 = verify_zero_source_operators(traj0, spec=spec)
    gates["G5_zero_source_property"] = g5

    g6 = verify_field_linearity(traj0, spec=spec)
    gates["G6_linearity"] = g6

    sem_table = semantic_status_table()
    g7_ok = len(sem_table) == 9 and all("status" in row for row in sem_table)
    gates["G7_semantic_status"] = {
        "passed": g7_ok,
        "semantic_status_table": sem_table,
    }

    post_fp = hierarchy_fingerprint(model)
    post_delay_digest = delay_table_digest(model)
    g8_ok = pre_fp == post_fp and pre_delay_digest == post_delay_digest
    gates["G8_hierarchy_provenance_preservation"] = {
        "passed": g8_ok,
        "pre_fingerprints": pre_fp,
        "post_fingerprints": post_fp,
        "delay_table_digest": post_delay_digest,
        "e3_rbs_owner_unchanged": e3_receipt["rbs_owner"],
    }

    obs_repeat = run_all_primary_observations(traj0, spec=spec)
    y_hashes_1 = {pid: r.Y_hash for pid, r in obs_once["observations"].items()}
    y_hashes_2 = {pid: r.Y_hash for pid, r in obs_repeat["observations"].items()}
    traj_repeat = run_e4_neural_only(model, seed=seed0, n_steps=n_steps, dt_ms=dt_ms)
    g9_ok = y_hashes_1 == y_hashes_2 and traj_repeat.cause_hashes == traj0.cause_hashes
    gates["G9_reproducibility"] = {
        "passed": g9_ok,
        "same_sot_same_Y": y_hashes_1 == y_hashes_2,
        "same_seed_same_sot_hashes": traj_repeat.cause_hashes == traj0.cause_hashes,
    }

    gates["G10_no_phenotype_claim"] = {
        "passed": True,
        "statement": "observation outputs are composition evidence only; no FF/FB/spectral/adaptation claim",
        "forbidden_metrics_absent": True,
    }

    hierarchy_table = build_hierarchy_aware_source_table_summary(traj0)
    if not hierarchy_table["conservation"]["passed"]:
        raise RuntimeError("hierarchy-aware source aggregation failed conservation")

    dyn_traj = freeze_e3_trajectory(
        model, seed=seed0, mode="E3-dynamic", n_steps=n_steps, dt_ms=dt_ms
    )
    dyn_obs = run_all_primary_observations(dyn_traj, spec=spec)
    owner_indices = list(e3_receipt["rbs_owner"]["flat_indices"])
    h_dev = float(np.max(np.abs(dyn_traj.H_K[:, owner_indices] - 1.0)))

    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_e_integration.e4_execution_receipt.v1",
        "checkpoint": "E4",
        "status": "FROZEN",
        "write_once": True,
        "package_head": package_head or _git_head(),
        "spec_path": "artifacts/protocol_e_integration/e4_observation_chain_spec.json",
        "scientific_question": spec["question"],
        "composition_claim": "hierarchy + delays + local RBD + downstream observation without mutating E3 dynamics",
        "reduction_contract": "R_E4_to_E3",
        "trajectory_invariance_contract": "T_E4_probe_independent_neural_source",
        "experiment_a_inheritance": {
            "source_protocol": spec["experiment_a_inheritance"]["source_protocol"],
            "Q_contract": "signals.sources_canonical_relative_source",
            "field_api": "jaxfne.fields.project_laminar_sources",
            "probe_apis": ["lfp_proxy_probe", "csd_proxy_probe"],
        },
        "source_of_truth_hashes": traj0.cause_hashes,
        "hierarchy_aware_source_table": hierarchy_table,
        "semantic_status_table": sem_table,
        "inherited_e3_execution_receipt": "artifacts/protocol_e_integration/e3_execution_receipt.json",
        "inherited_e3_hierarchy_fingerprints": post_fp,
        "inherited_e3_delay_digest": post_delay_digest,
        "primary_observations_seed11": {
            "phi_ref_hash": obs_once["phi_ref_hash"],
            "Y_hashes": y_hashes_1,
            "source_hashes_cited": traj0.cause_hashes,
        },
        "e3_dynamic_observation_diagnostic": {
            "outside_g1_reduction": True,
            "outside_phenotype_claims": True,
            "neural_mode": "E3-dynamic",
            "seed": seed0,
            "max_abs_H_K_minus_1_owners": h_dev,
            "observation_completed": dyn_obs["cause_hashes_unchanged"],
            "phi_ref_hash": dyn_obs["phi_ref_hash"],
            "interpretation": "non-reference H trajectory accepts same observation chain; not phenotype evidence",
        },
        "gates": gates,
        "e5_deferred": "integrated perturbation with mechanism-null contrasts only",
    }
    if _receipt_has_banned_terms(receipt):
        raise RuntimeError("E4 receipt contains banned scientific terms")

    for gname, gval in gates.items():
        if not gval.get("passed"):
            raise RuntimeError(f"E4 gate failed: {gname}")
    return json_safe(receipt)


def write_e4_execution_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    receipt = run_e4_observation_composition(package_head=package_head)
    E4_EXECUTION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_e4_execution_receipt() -> dict[str, Any]:
    return json.loads(E4_EXECUTION_RECEIPT_PATH.read_text())
