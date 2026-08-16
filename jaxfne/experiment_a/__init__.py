"""Experiment A (0.4.17-B): canonical multiscale observation étude."""

from .protocol import (
    PROTOCOL_ID,
    PROTOCOL_SPEC_PATH,
    load_protocol_spec,
)
from .canonical import (
    CanonicalDataset,
    build_experiment_a_config,
    freeze_canonical_dataset,
    load_frozen_canonical_dataset,
    write_b1_receipt,
    write_canonical_npz,
)
from .observe import (
    FactorizedObservation,
    apply_independent_probe,
    materialize_field,
    verify_b2_invariants,
    write_b2_receipt,
)

__all__ = [
    "PROTOCOL_ID",
    "PROTOCOL_SPEC_PATH",
    "load_protocol_spec",
    "CanonicalDataset",
    "build_experiment_a_config",
    "freeze_canonical_dataset",
    "load_frozen_canonical_dataset",
    "write_b1_receipt",
    "write_canonical_npz",
    "FactorizedObservation",
    "apply_independent_probe",
    "materialize_field",
    "verify_b2_invariants",
    "write_b2_receipt",
]
