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
    write_b1_receipt,
    write_canonical_npz,
)

__all__ = [
    "PROTOCOL_ID",
    "PROTOCOL_SPEC_PATH",
    "load_protocol_spec",
    "CanonicalDataset",
    "build_experiment_a_config",
    "freeze_canonical_dataset",
    "write_b1_receipt",
    "write_canonical_npz",
]
