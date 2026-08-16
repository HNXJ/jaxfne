"""0.4.17-E E1 — hierarchy/runtime implementation tests."""

from __future__ import annotations

import json

import jaxfne as jtfne
import numpy as np
import pytest

from jaxfne.protocol_e_integration.e1_execution import (
    build_edge_provenance_table,
    build_e1_configuration,
    build_identity_map,
    identity_round_trip_ok,
    load_e1_execution_receipt,
    run_e1_hierarchy_runtime,
    verify_connectivity_ownership,
    write_e1_execution_receipt,
)
from jaxfne.protocol_e_integration.e1_protocol import (
    E1_EXECUTION_RECEIPT_PATH,
    E1_SPEC_PATH,
    load_e1_spec,
    validate_e1_spec,
)


def test_e1_build_configuration_compiles():
    cfg = build_e1_configuration(include_inter_area=True)
    model = jtfne.construct(cfg)
    table = model.neuron_table()
    assert len(table) == 80
    assert sorted({r["area"] for r in table}) == ["A1", "A2"]


def test_e1_identity_round_trip():
    model = jtfne.construct(build_e1_configuration())
    table = model.neuron_table()
    identity = build_identity_map(table)
    assert len(identity) == len(table)
    assert identity_round_trip_ok(identity)
    for i, row in enumerate(table):
        assert int(row["neuron_id"]) == i
        assert identity[i]["flat_index"] == i
        assert identity[i]["area"] == row["area"]
        assert identity[i]["layer"] == row["layer"]
        assert identity[i]["cell_type"] == row["cell_type"]


def test_e1_edge_provenance_classes_and_ff_fb_semantics():
    model = jtfne.construct(build_e1_configuration())
    provenance = build_edge_provenance_table(model)
    g3 = verify_connectivity_ownership(provenance)
    assert g3["passed"]
    classes = set(g3["edge_class_counts"])
    assert classes == {"local_A1", "local_A2", "FF_A1_to_A2", "FB_A2_to_A1"}
    for row in provenance:
        if row["edge_class"] == "FF_A1_to_A2":
            assert row["pre_area"] == "A1"
            assert row["pre_layer"] in ("L2", "L3")
            assert row["pre_cell_type"] == "E"
            assert row["post_area"] == "A2"
            assert row["post_layer"] == "L4"
        if row["edge_class"] == "FB_A2_to_A1":
            assert row["pre_area"] == "A2"
            assert row["pre_layer"] == "L5"
            assert row["pre_cell_type"] == "E"
            assert row["post_area"] == "A1"
            assert row["post_layer"] in ("L2", "L3")


def test_e1_zero_delay_explicit_connectivity():
    model = jtfne.construct(build_e1_configuration())
    delays = np.asarray(model.params["edge_list"].delay_steps, dtype=np.int32)
    assert np.all(delays == 0)
    cc = model.cfg.metadata["connectivity_compilation"]
    assert cc["connectivity_mode"] == "explicit"
    assert cc["default_edge_count"] == 0


def test_e1_g6_inter_area_disabled_baseline():
    full = jtfne.construct(build_e1_configuration(include_inter_area=True))
    baseline = jtfne.construct(build_e1_configuration(include_inter_area=False))
    assert build_identity_map(full.neuron_table()) == build_identity_map(baseline.neuron_table())
    prov = build_edge_provenance_table(baseline)
    assert not any(r["pre_area"] != r["post_area"] for r in prov)
    assert {r["edge_class"] for r in prov} <= {"local_A1", "local_A2"}


def test_e1_finite_reproducible_simulation():
    model = jtfne.construct(build_e1_configuration())
    sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=11)
    a = model.simulate(sim)
    b = model.simulate(sim)
    va = np.asarray(a.V_m)
    vb = np.asarray(b.V_m)
    assert np.isfinite(va).all()
    assert np.array_equal(va, vb)


def test_e1_run_gates_and_write_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jaxfne.protocol_e_integration.e1_execution.E1_EXECUTION_RECEIPT_PATH",
        tmp_path / "e1_execution_receipt.json",
    )
    receipt = write_e1_execution_receipt()
    assert receipt["checkpoint"] == "E1"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])
    assert receipt["edge_provenance_summary"]["n_edges"] > 0
    loaded = json.loads((tmp_path / "e1_execution_receipt.json").read_text())
    assert loaded["schema"].endswith("e1_execution_receipt.v1")


def test_e1_validate_spec_passes():
    validate_e1_spec()


def test_e1_frozen_spec_unchanged():
    spec = load_e1_spec()
    assert spec["status"] == "FROZEN"
    assert spec["rbs"]["enabled"] is False
    assert int(spec["connectivity"]["delay_steps"]) == 0


def test_e1_frozen_execution_receipt_present():
    receipt = load_e1_execution_receipt()
    assert receipt["status"] == "FROZEN"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])
