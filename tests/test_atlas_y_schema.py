"""Frozen Atlas measurement vector (0.5.5 ENGINE item 1): schema v4 contract.

No simulation: runs in the CI Fast Atlas step. Real-run records are pinned in
tests/test_atlas_v4_records.py.
"""

from __future__ import annotations

import ast
import math
import pathlib

import pytest

from artifacts.atlas import y_schema as Y

V3_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "artifacts/atlas/at08_at09_054.py"


def _v3_keys() -> tuple[str, ...]:
    """Y_KEYS_V3 read statically, so this module needs no jaxfne import."""
    tree = ast.parse(V3_SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "Y_KEYS_V3":
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("Y_KEYS_V3 not found")


def test_v4_freezes_v3_names_in_order():
    assert Y.SCHEMA_VERSION == "v4"
    assert Y.Y_KEYS_V4 == _v3_keys()
    assert len(Y.Y_STABLE) == 13
    assert set(Y.NOTES) == set(Y.Y_KEYS_V4)


def test_record_derives_state_from_values():
    rec = Y.record(
        values={"SPK": 42, "T_compute": 1.5, "H": None},
        refused={"Phi_B": "no calibrated Phi_B"},
        notes={"SPK": "count"},
    )
    assert Y.validate(rec) == []
    assert rec["SPK"] == {
        "state": "IMPLEMENTED",
        "value": 42,
        "level": "RELATIVE_PROXY",
        "note": "count",
    }
    assert rec["H"]["state"] == "OMITTED" and rec["H"]["value"] is None
    assert rec["Phi_B"]["state"] == "REFUSED"
    assert sum(c["state"] == "IMPLEMENTED" for c in rec.values()) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": {"Nope": 1}, "refused": {}},  # name outside the schema
        {"values": {"Phi_B": 1.0}, "refused": {"Phi_B": "x"}},  # refused with a value
        {"values": {"SPK": math.nan}, "refused": {}},  # non-finite value
        {"values": {"W": object()}, "refused": {}},  # not JSON
    ],
)
def test_record_refuses_contract_breaks(kwargs):
    with pytest.raises(ValueError):
        Y.record(**kwargs)


def test_validate_catches_tampered_records():
    rec = Y.record(values={"SPK": 3}, refused={})
    omitted_with_value = {**rec, "H": {**rec["H"], "value": 0.0}}
    implemented_empty = {**rec, "SPK": {**rec["SPK"], "value": None}}
    reordered = dict(reversed(list(rec.items())))
    assert Y.validate(omitted_with_value) == ["H: OMITTED must carry value None"]
    assert Y.validate(implemented_empty) == ["SPK: IMPLEMENTED needs a finite JSON value"]
    assert Y.validate(reordered)[0].startswith("keys must be Y_KEYS_V4")


def test_measure_v4_reads_only_what_the_run_wrote():
    raw = {
        "scenario": "AT-07",
        "status": "OK",
        "wall_s": 2.0,
        "arms": {"fixed": {"spike_count": 5, "H_final_mean": 1.0}, "hebbian": {"spike_count": 7}},
    }
    rec = Y.measure_v4(raw)
    assert Y.validate(rec) == []
    assert rec["SPK"]["value"] == {"fixed": 5, "hebbian": 7}
    assert rec["H"]["value"] == {"fixed": 1.0}  # hebbian produced no H: left out
    assert rec["W"]["state"] == "OMITTED"  # no arm wrote W
    assert rec["T_compute"]["value"] == 2.0
    assert rec["Phi_B"]["state"] == "REFUSED"


def test_measure_v4_error_run_keeps_only_t_compute():
    rec = Y.measure_v4(
        {"scenario": "AT-03", "status": "ERROR", "wall_s": 0.1, "pair": {"n_spikes": 3}}
    )
    implemented = [k for k, c in rec.items() if c["state"] == "IMPLEMENTED"]
    assert implemented == ["T_compute"]


def test_measure_v4_unknown_scenario_raises():
    with pytest.raises(KeyError):
        Y.measure_v4({"scenario": "AT-99", "status": "OK"})
