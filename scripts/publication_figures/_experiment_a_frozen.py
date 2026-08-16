"""Load frozen Experiment A canonical dataset for publication figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jaxfne.experiment_a.canonical import CanonicalDataset, array_sha256, load_frozen_canonical_dataset
from jaxfne.experiment_a.observe import apply_independent_probe, materialize_field, verify_b2_invariants
from jaxfne.experiment_a.protocol import load_protocol_spec

_REPO = Path(__file__).resolve().parents[2]
B1_RECEIPT = _REPO / "artifacts" / "etudes" / "experiment_a" / "b1_canonical_receipt.json"
B2_RECEIPT = _REPO / "artifacts" / "etudes" / "experiment_a" / "b2_probe_receipt.json"
CANONICAL_NPZ = _REPO / "artifacts" / "etudes" / "experiment_a" / "canonical_source.npz"


@dataclass(frozen=True)
class ExperimentAFrozenBundle:
    dataset: CanonicalDataset
    spec: dict[str, Any]
    q_hash: str
    b1_receipt: dict[str, Any]
    b2_receipt: dict[str, Any]
    field_output: Any
    shallow: Any
    deep: Any
    csd: Any
    b2_invariants: dict[str, Any]


def load_experiment_a_bundle() -> ExperimentAFrozenBundle:
    spec = load_protocol_spec()
    dataset = load_frozen_canonical_dataset(npz_path=CANONICAL_NPZ, receipt_path=B1_RECEIPT)
    q_hash = array_sha256(dataset.Q)
    b1 = json.loads(B1_RECEIPT.read_text())
    b2 = json.loads(B2_RECEIPT.read_text())
    field = materialize_field(dataset, "lfp_ref", spec=spec)
    shallow = apply_independent_probe(dataset, field, "lfp_contact_shallow", spec)
    deep = apply_independent_probe(dataset, field, "lfp_contact_deep", spec)
    csd = apply_independent_probe(dataset, field, "csd_from_lfp_ref", spec)
    invariants = verify_b2_invariants(dataset, spec=spec)
    return ExperimentAFrozenBundle(
        dataset=dataset,
        spec=spec,
        q_hash=q_hash,
        b1_receipt=b1,
        b2_receipt=b2,
        field_output=field,
        shallow=shallow,
        deep=deep,
        csd=csd,
        b2_invariants=invariants,
    )


def pick_display_units(positions: np.ndarray, n: int = 4) -> list[int]:
    """Pick neuron indices spanning laminar depth for compact panels."""
    z = positions[:, 2] if positions.shape[1] >= 3 else positions[:, 0]
    order = np.argsort(z)
    idx = np.linspace(0, len(order) - 1, n, dtype=int)
    return [int(order[i]) for i in idx]


def time_window_mask(time_ms: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (time_ms >= t0) & (time_ms <= t1)
