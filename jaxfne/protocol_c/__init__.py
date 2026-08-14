"""Protocol C (0.4.17-C) — wave inference specification and receipts."""

from .protocol import (
    PROTOCOL_ID,
    PROTOCOL_SPEC_PATH,
    load_protocol_spec,
)
from .estimator import WaveEstimate, estimate_traveling_wave
from .c1_validation import run_c1_synthetic_validation, write_c1_receipt
from .c3_protocol import (
    C3_SPEC_PATH,
    c3_condition_ids,
    c3_total_cells,
    load_c3_spec,
    validate_c3_spec,
)

__all__ = [
    "PROTOCOL_ID",
    "PROTOCOL_SPEC_PATH",
    "load_protocol_spec",
    "WaveEstimate",
    "estimate_traveling_wave",
    "run_c1_synthetic_validation",
    "write_c1_receipt",
    "C3_SPEC_PATH",
    "load_c3_spec",
    "validate_c3_spec",
    "c3_condition_ids",
    "c3_total_cells",
]
