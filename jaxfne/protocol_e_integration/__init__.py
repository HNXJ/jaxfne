"""Protocol E (0.4.17-E) — integrated TFNE composition."""

from .e0_protocol import (
    E0_SPEC_PATH,
    PROTOCOL_ID,
    load_e0_spec,
    validate_e0_spec,
)
from .e0_1_protocol import (
    E0_1_SPEC_PATH,
    e0_1_ladder_ids,
    load_e0_1_spec,
    validate_e0_1_spec,
)
from .e1_protocol import (
    E1_EXECUTION_RECEIPT_PATH,
    E1_SPEC_PATH,
    load_e1_spec,
    validate_e1_spec,
)
from .e1_execution import (
    build_edge_provenance_table,
    build_e1_configuration,
    build_identity_map,
    identity_round_trip_ok,
    load_e1_execution_receipt,
    run_e1_hierarchy_runtime,
    verify_connectivity_ownership,
    write_e1_execution_receipt,
)

__all__ = [
    "PROTOCOL_ID",
    "E0_SPEC_PATH",
    "E0_1_SPEC_PATH",
    "E1_SPEC_PATH",
    "E1_EXECUTION_RECEIPT_PATH",
    "load_e0_spec",
    "validate_e0_spec",
    "load_e0_1_spec",
    "validate_e0_1_spec",
    "e0_1_ladder_ids",
    "load_e1_spec",
    "validate_e1_spec",
    "build_e1_configuration",
    "build_identity_map",
    "build_edge_provenance_table",
    "identity_round_trip_ok",
    "verify_connectivity_ownership",
    "run_e1_hierarchy_runtime",
    "write_e1_execution_receipt",
    "load_e1_execution_receipt",
]
