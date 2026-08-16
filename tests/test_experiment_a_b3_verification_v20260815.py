"""Experiment A B3 regeneration verification semantics (write-once receipts)."""

from __future__ import annotations

import pytest

from jaxfne.experiment_a.receipt import _compare_value, _tracked_write, _verify_canonical_hashes


def test_compare_tolerates_committed_only_annotation_and_head_drift():
    eq, rel, diff = _compare_value(
        {
            "test_evidence": {"initial": 19, "passed": 18},
            "protocol": "experiment_a_v0417_b",
            "package_head": "5206c54",
        },
        {"protocol": "experiment_a_v0417_b", "package_head": "de62bc7"},
    )
    assert eq
    assert diff == ""
    assert rel == 0.0


def test_compare_float_tail_drift_within_rtol_passes():
    eq, rel, diff = _compare_value(
        {"spectral_centroid_hz": {"Q": 46.70724533480392}},
        {"spectral_centroid_hz": {"Q": 46.707244762767466}},
    )
    assert eq
    assert rel < 1e-7
    assert diff == ""


def test_compare_rejects_real_float_drift():
    eq, rel, diff = _compare_value({"v": 43.8}, {"v": 44.5})
    assert not eq
    assert "v:" in diff


def test_compare_rejects_missing_regenerated_key():
    eq, rel, diff = _compare_value({"cause_hashes": {"Q": "a"}}, {"cause_hashes": {}})
    assert not eq
    assert "cause_hashes.Q" in diff


def test_compare_hash_strings_exact():
    eq, rel, diff = _compare_value({"q": "02f4..."}, {"q": "02f5..."})
    assert not eq
    assert diff.startswith(".q:")


def test_tracked_write_verifies_without_rewriting(tmp_path):
    target = tmp_path / "receipt.json"
    committed = {
        "status": "FROZEN",
        "package_head": "5206c54",
        "spectral_centroid_hz": {"Q": 46.70724533480392},
    }
    target.write_text(
        __import__("json").dumps(committed, indent=2, sort_keys=True) + "\n"
    )
    before = target.read_bytes()
    ver = _tracked_write(
        target,
        {
            "status": "FROZEN",
            "package_head": "de62bc7",
            "spectral_centroid_hz": {"Q": 46.707244762767466},
        },
        name="receipt.json",
    )
    assert ver["status"] == "verified"
    assert target.read_bytes() == before


def test_tracked_write_raises_on_real_drift(tmp_path):
    target = tmp_path / "receipt.json"
    target.write_text(
        __import__("json").dumps({"n_neurons": 40}, indent=2) + "\n"
    )
    with pytest.raises(RuntimeError, match="drift"):
        _tracked_write(target, {"n_neurons": 41}, name="receipt.json")


def test_verify_canonical_hashes_against_committed(tmp_path):
    b1 = tmp_path / "b1.json"
    b1.write_text(
        __import__("json").dumps(
            {"cause_hashes": {"Q": "expected", "H": "other"}}, indent=2
        )
        + "\n"
    )
    result = _verify_canonical_hashes(
        {"Q": "expected", "H": "changed", "V_m": "extra"}, b1
    )
    assert result == {"Q": True, "H": False, "V_m": False}