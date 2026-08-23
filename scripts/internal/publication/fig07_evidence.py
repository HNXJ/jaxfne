"""Frozen E1–E5 evidence loader for Figure 7 (receipt-driven quantities only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]


def _load(rel: str) -> dict[str, Any]:
    return json.loads((_REPO / rel).read_text())


@dataclass(frozen=True)
class Fig07Evidence:
    e1: dict[str, Any]
    e1_protocol: dict[str, Any]
    e2: dict[str, Any]
    e2_protocol: dict[str, Any]
    e3: dict[str, Any]
    e3_protocol: dict[str, Any]
    e4: dict[str, Any]
    e4_protocol: dict[str, Any]
    e5: dict[str, Any]
    e5_interp: dict[str, Any]
    e5_spec: dict[str, Any]


def load_fig07_evidence() -> Fig07Evidence:
    return Fig07Evidence(
        e1=_load("artifacts/protocol_e_integration/e1_execution_receipt.json"),
        e1_protocol=_load("artifacts/protocol_e_integration/e1_protocol_receipt.json"),
        e2=_load("artifacts/protocol_e_integration/e2_execution_receipt.json"),
        e2_protocol=_load("artifacts/protocol_e_integration/e2_protocol_receipt.json"),
        e3=_load("artifacts/protocol_e_integration/e3_execution_receipt.json"),
        e3_protocol=_load("artifacts/protocol_e_integration/e3_protocol_receipt.json"),
        e4=_load("artifacts/protocol_e_integration/e4_execution_receipt.json"),
        e4_protocol=_load("artifacts/protocol_e_integration/e4_protocol_receipt.json"),
        e5=_load("artifacts/protocol_e_integration/e5_execution_receipt.json"),
        e5_interp=_load("artifacts/protocol_e_integration/e5_interpretation_receipt.json"),
        e5_spec=_load("artifacts/protocol_e_integration/e5_causal_perturbation_spec.json"),
    )


def e1_hierarchy_summary(ev: Fig07Evidence) -> dict[str, Any]:
    g1 = ev.e1["gates"]["G1_construction"]
    g2 = ev.e1["gates"]["G2_identity_recovery"]
    return {
        "n_neurons": int(g1["n_neurons"]),
        "n_edges": int(g1["n_edges"]),
        "n_identity_rows": int(g2["n_identity_rows"]),
        "identity_round_trip": bool(g2["identity_round_trip"]),
        "areas": tuple(g1["areas"]),
        "layers": tuple(g1["layers"]),
        "cell_types": tuple(g1["cell_types"]),
        "edge_class_counts": dict(ev.e1["edge_provenance_summary"]["edge_class_counts"]),
    }


def e2_delay_classes(ev: Fig07Evidence) -> list[dict[str, Any]]:
    """Aggregate local / FF / FB delay classes from frozen typed_delay_table."""
    table = ev.e2["typed_delay_table"]
    local_ms = float(table[0]["delay_ms"])
    ff_ms = float(next(r["delay_ms"] for r in table if "FF" in r["edge_class"]))
    fb_ms = float(next(r["delay_ms"] for r in table if "FB" in r["edge_class"]))
    return [
        {"class": "local", "tau_ms": local_ms, "symbol": r"$\tau_{\rm local}$"},
        {"class": "FF", "tau_ms": ff_ms, "symbol": r"$\tau_{\rm FF}$"},
        {"class": "FB", "tau_ms": fb_ms, "symbol": r"$\tau_{\rm FB}$"},
    ]


def e3_owner(ev: Fig07Evidence) -> dict[str, Any]:
    owner = ev.e3_protocol["rbs_owner"]
    return {
        "area": owner["area"],
        "layer": owner["layer"],
        "cell_type": owner["cell_type"],
        "n_nodes": int(owner["n_nodes"]),
        "flat_indices": list(owner["flat_indices"]),
    }


def e4_observation_semantics(ev: Fig07Evidence) -> dict[str, str]:
    chain = ev.e4.get("observation_semantics", {})
    if not chain:
        chain = {
            "Q": ev.e4_protocol.get("trajectory_invariance", "T_E4_probe_independent_neural_source"),
            "Y": ev.e4_protocol.get("experiment_a_semantics", "relative proxy"),
        }
    return {
        "trajectory_invariance": str(ev.e4_protocol.get("trajectory_invariance", "T_E4")),
        "Q_status": "canonical relative source",
        "Y_status": "relative proxy (not calibrated EEG/MEG)",
        "composition": "(X,H,B) -> Q -> Phi_ref -> P -> Y",
    }


def e5_null_controls(ev: Fig07Evidence) -> dict[str, Any]:
    sanity = ev.e5["sanity_checks"]
    return {
        "N0_equals_N1": list(sanity["N0_equals_N1_neural"]),
        "H_K_N1_equals_D": list(sanity["H_K_N1_equals_D"]),
        "seeds": list(ev.e5["design"]["seeds"]),
        "arms": list(ev.e5["design"]["arms"]),
    }


def e5_propagation_metrics(ev: Fig07Evidence) -> dict[str, Any]:
    """Per-level |D-N1| metrics averaged across seeds (identical in frozen receipt)."""
    per_seed = ev.e5_interp["per_seed"]
    seed0 = per_seed[0]["Delta_R"]
    return {
        "levels": [
            ("H_K", "owner gate", seed0.get("Delta_X_owner", {}).get("mean_abs_V_m_deviation", 0.0)),
            ("X_owner", "mean |V_m|", seed0["Delta_X_owner"]["mean_abs_V_m_deviation"]),
            ("X_A2", "mean |V_m|", seed0["Delta_X_A2_nonowner"]["mean_abs_V_m_deviation"]),
            ("X_A1", "mean |V_m|", seed0["Delta_X_A1"]["mean_abs_V_m_deviation"]),
            ("Q", "L2 norm", seed0["Delta_Q"]["L2_norm_difference"]),
            ("Y", "L2 norm", seed0["Delta_Y"]["L2_norm_difference"]),
        ],
        "classification": ev.e5_interp["aggregate_classification"],
        "evidence_gates": per_seed[0]["evidence_gates"],
        "per_seed_classifications": [s["classification"] for s in per_seed],
        "permissible_A1_statement": ev.e5_interp["permissible_A1_statement"],
    }


def e5_arm_definitions(ev: Fig07Evidence) -> list[dict[str, Any]]:
    return list(ev.e5_spec["experimental_arms"])
