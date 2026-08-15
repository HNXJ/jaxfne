"""0.4.17-D Protocol D closure at D3."""

from __future__ import annotations

import json

from jaxfne.protocol_d_biological_rbs.d_closure import (
    D_CLOSURE_RECEIPT_PATH,
    load_d_closure_receipt,
    validate_d_closure_receipt,
)
from jaxfne.protocol_d_biological_rbs.d3_protocol import D3_SPEC_PATH


def test_d_closure_receipt_frozen():
    receipt = load_d_closure_receipt()
    assert receipt["status"] == "FROZEN"
    assert receipt["protocol_d_closed"] is True
    assert receipt["closed_at_checkpoint"] == "D3"


def test_d_closure_D4_not_authorized():
    receipt = load_d_closure_receipt()
    assert receipt["D4_status"] == "not_authorized"
    validate_d_closure_receipt(receipt)


def test_d_closure_figure_6_ladder():
    ladder = load_d_closure_receipt()["figure_6_ladder"]
    assert ladder["static_H_K_to_X"] == "demonstrated"
    assert ladder["activity_written_H_K_to_distinct_spike_adaptation"] == "not_supported"


def test_d_closure_next_milestone_E():
    receipt = load_d_closure_receipt()
    assert receipt["next_milestone"] == "0.4.17-E"
    assert receipt["next_checkpoint"] == "E0_specification"


def test_d3_protocol_receipt_points_to_closure():
    proto = json.loads((D3_SPEC_PATH.parent / "d3_protocol_receipt.json").read_text())
    assert proto["protocol_d_closed"] is True
    assert proto["D4_status"] == "not_authorized"
    assert proto["next_milestone"] == "0.4.17-E"
