"""Frozen Protocol E4 observation-chain composition specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_e_integration.e0_protocol import PROTOCOL_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
E4_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e4_observation_chain_spec.json"
E4_EXECUTION_RECEIPT_PATH = (
    _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e4_execution_receipt.json"
)


def load_e4_spec(path: Path | None = None) -> dict[str, Any]:
    spec_path = path or E4_SPEC_PATH
    return json.loads(spec_path.read_text())


def e4_gate_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e4_spec()
    return tuple(spec["gates"])


def e4_primary_field_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e4_spec()
    rows = spec["primary_evidence_operators"]["field_operators_F"]
    return tuple(str(row["id"]) for row in rows)


def e4_primary_probe_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e4_spec()
    rows = spec["primary_evidence_operators"]["probe_operators_P"]
    return tuple(str(row["id"]) for row in rows)


def validate_e4_spec(spec: dict[str, Any] | None = None) -> None:
    spec = spec or load_e4_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("E4 spec must have status FROZEN")
    if spec.get("checkpoint") != "E4":
        raise ValueError("E4 spec must have checkpoint E4")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    if spec["reduction_contract"]["id"] != "R_E4_to_E3":
        raise ValueError("E4 must declare R_E4_to_E3 reduction contract")

    traj = spec["trajectory_invariance_contract"]
    if traj["id"] != "T_E4_probe_independent_neural_source":
        raise ValueError("E4 must declare probe-independent neural/source trajectory contract")
    if "Q" not in traj["probe_selection_invariant_fields"]:
        raise ValueError("trajectory invariance must include Q")

    inherit = spec["experiment_a_inheritance"]
    if "b0_protocol_spec.json" not in inherit["source_protocol"]:
        raise ValueError("E4 must inherit Experiment A b0 protocol")
    if "do not create" not in inherit.get("rule", "").lower():
        raise ValueError("E4 must prohibit integrated-model source variant")
    q_contract = inherit["inherited_source_contract"]["Q_recorded"]
    if q_contract != "signals.sources_canonical_relative_source":
        raise ValueError("E4 must reuse Experiment A canonical relative source contract")

    primary_fields = e4_primary_field_ids(spec)
    primary_probes = e4_primary_probe_ids(spec)
    if primary_fields != ("lfp_ref",):
        raise ValueError("E4 primary evidence must freeze lfp_ref field operator")
    if primary_probes != ("lfp_contact_shallow", "lfp_contact_deep", "csd_from_lfp_ref"):
        raise ValueError("E4 primary probes must match conservative Experiment A subset")

    deferred = spec["deferred_operators"]["analysis_only_excluded_from_primary_evidence"]
    deferred_ids = {row["id"] for row in deferred}
    if not {"eeg_superficial", "eeg_deep", "meg_relative"}.issubset(deferred_ids):
        raise ValueError("EEG/MEG must be deferred analysis_only operators")

    gates = e4_gate_ids(spec)
    if len(gates) != 10:
        raise ValueError("E4 requires exactly ten gates G1–G10")
    if gates[0] != "G1_e3_reduction_neural_invariance":
        raise ValueError("G1 must be E3 reduction / neural invariance gate")
    if gates[-1] != "G10_no_phenotype_claim":
        raise ValueError("G10 must prohibit phenotype claims")

    if spec["neural_execution"]["reference_mode"] != "E3-null":
        raise ValueError("E4 primary neural path must use E3-null reference mode")

    sim = spec["simulation_policy"]
    if sim["neural_mode"] != "E3-null":
        raise ValueError("E4 simulation_policy neural_mode must be E3-null")
    if int(sim["n_steps"]) != 2000:
        raise ValueError("E4 must inherit E3 n_steps=2000")

    if not spec["execution_authorization"].get("specification_only"):
        raise ValueError("E4 must be specification_only at freeze")
    if spec["execution_authorization"].get("implementation_authorized") is not False:
        raise ValueError("E4 must not authorize implementation at specification freeze")

    banned = set(spec["explicit_prohibitions"])
    if "no_new_source_definition" not in banned:
        raise ValueError("must prohibit new source definition at E4")
    if "no_observation_feedback_into_E3_dynamics" not in banned:
        raise ValueError("must prohibit observation feedback into E3 dynamics")
    if "no_EEG_MEG_in_primary_evidence" not in banned:
        raise ValueError("must exclude EEG/MEG from primary evidence")
