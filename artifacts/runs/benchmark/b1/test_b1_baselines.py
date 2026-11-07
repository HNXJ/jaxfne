"""B1 simulator baselines: receipt shape, null-with-reason, spikes>0.

Brian2/NEST/Jaxley run_cell receipts mirror b1_harness.py fields; every
null leaf carries a reason in ``unavailable``.
"""
import sys

import pytest

sys.path.insert(0, "artifacts/runs/benchmark/b1")


def check_receipt(rec):
    assert rec["status"] == "MEASURED", rec.get("error")
    assert rec["simulator_version"], "simulator_version must be reported"
    agr = rec["numerical_agreement"]
    assert agr["total_spikes"] > 0
    assert agr["total_spikes"] == sum(agr["spike_counts"].values())
    assert rec["validity"]["spikes_gt_zero"] is True
    assert rec["validity"]["valid"] is True
    for entry in rec["unavailable"]:
        assert set(entry) >= {"field", "reason"}, entry
        assert entry["reason"], entry
    assert rec["wall_s"] > 0


def test_brian2_m1():
    pytest.importorskip("brian2")
    from b1_brian2 import run_cell
    rec = run_cell({"model": "M1", "n": 100, "dtype": "float32",
                    "target": "numpy", "recording": "off", "seeds": [7],
                    "duration_ms": 100.0, "dt_ms": 0.1})
    check_receipt(rec)


def test_brian2_m2():
    pytest.importorskip("brian2")
    from b1_brian2 import run_cell
    rec = run_cell({"model": "M2", "n": 100, "dtype": "float32",
                    "target": "numpy", "recording": "off", "seeds": [7],
                    "duration_ms": 100.0, "dt_ms": 0.1})
    check_receipt(rec)


def test_nest_m1():
    pytest.importorskip("nest")
    from b1_nest import run_cell
    rec = run_cell({"model": "M1", "n": 100, "dtype": "float64",
                    "recording": "off", "seeds": [7],
                    "duration_ms": 100.0, "dt_ms": 0.1})
    check_receipt(rec)


def test_nest_m2():
    pytest.importorskip("nest")
    from b1_nest import run_cell
    rec = run_cell({"model": "M2", "n": 100, "dtype": "float64",
                    "recording": "off", "seeds": [7],
                    "duration_ms": 100.0, "dt_ms": 0.1})
    check_receipt(rec)


def test_jaxley_m3():
    pytest.importorskip("jaxley")
    from b1_jaxley import run_cell
    rec = run_cell({"model": "M3", "n": 1, "dtype": "float32",
                    "recording": "off", "seeds": [7],
                    "duration_ms": 100.0, "dt_ms": 0.025})
    check_receipt(rec)
