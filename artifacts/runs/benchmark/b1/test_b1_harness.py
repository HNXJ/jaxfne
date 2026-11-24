"""B1 harness smoke: one tiny cell in-process; every receipt field present/finite."""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "b1_harness", Path(__file__).with_name("b1_harness.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def _spec(recording: str) -> dict:
    return {"model": "M1", "n": 100, "dtype": "float32", "device": "cpu",
            "batch": 1, "recording": recording, "seeds": [1],
            "duration_ms": 10.0, "dt_ms": 0.1}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def test_tiny_cell_receipt_fields():
    rec = H.run_cell(_spec("full"))
    assert rec["status"] == "MEASURED", rec.get("error")
    for key in ("cell", "model", "n", "n_steps", "duration_ms", "dt_ms",
                "dtype", "device_requested", "device_used", "backend",
                "batch", "recording", "seeds", "warmup_seed", "build_seed",
                "compile_s", "construct_s", "warmup_s", "run_s",
                "run_s_median", "run_s_q1", "run_s_q3", "run_s_iqr",
                "throughput_neuron_steps_s", "ensemble_s", "ensemble_mode",
                "peak_gpu_mem", "peak_host_rss_mb", "transfer_s",
                "numerical_agreement", "validity", "env", "scaling_rule",
                "lane_refs", "prereg", "notes", "wall_s"):
        assert key in rec, f"missing receipt field {key}"
    assert rec["n_steps"] == 100
    for key in ("compile_s", "construct_s", "warmup_s", "run_s_median",
                "run_s_q1", "run_s_q3", "run_s_iqr",
                "throughput_neuron_steps_s", "peak_host_rss_mb", "wall_s"):
        assert _finite(rec[key]), f"{key}={rec[key]!r} not finite"
    assert len(rec["run_s"]) == 1 and all(_finite(x) for x in rec["run_s"])
    put, get = rec["transfer_s"]["put_s"], rec["transfer_s"]["get_s"]
    assert _finite(put) and _finite(get), (put, get)
    ag = rec["numerical_agreement"]
    assert ag["total_spikes"] >= 0 and _finite(ag["total_spikes"])
    assert set(ag["spike_counts"]) == set(ag["mean_rate_hz"]) >= {"E", "PV"}
    for pop, rate in ag["mean_rate_hz"].items():
        assert _finite(rate), (pop, rate)
    assert rec["validity"]["spikes_gt_zero"] == (ag["total_spikes"] > 0)
    env = rec["env"]
    for key in ("sha", "prereg_sha256", "jax", "jaxlib", "jaxfne_version",
                "device_string", "dtype", "threads", "seed_list", "platform",
                "nvidia_smi_before", "nvidia_smi_after"):
        assert key in env, f"missing env field {key}"
    assert len(env["prereg_sha256"]) == 64
    assert env["seed_list"] == [1]


def test_m3_tiny_cell_receipt_fields():
    spec = {"model": "M3", "n": 1, "dtype": "float32", "device": "cpu",
            "batch": 1, "recording": "full", "seeds": [1],
            "duration_ms": 10.0, "dt_ms": 0.1}
    rec = H.run_cell(spec)
    assert rec["status"] == "MEASURED", rec.get("error")
    for key in ("cell", "model", "n", "n_steps", "duration_ms", "dt_ms",
                "dtype", "device_requested", "device_used", "backend",
                "batch", "recording", "seeds", "warmup_seed", "build_seed",
                "compile_s", "construct_s", "warmup_s", "run_s",
                "run_s_median", "run_s_q1", "run_s_q3", "run_s_iqr",
                "throughput_neuron_steps_s", "ensemble_s", "ensemble_mode",
                "peak_gpu_mem", "peak_host_rss_mb", "transfer_s",
                "numerical_agreement", "validity", "env", "scaling_rule",
                "lane_refs", "prereg", "notes", "wall_s"):
        assert key in rec, f"missing receipt field {key}"
    assert rec["n_steps"] == 100
    ag = rec["numerical_agreement"]
    assert ag["total_spikes"] > 0, "I_ext=10 must drive HH spikes"
    assert rec["validity"]["spikes_gt_zero"] is True
    assert rec["validity"]["mean_E_rate_hz"] > 0


class _DeviceLeaf:
    """Stands in for a device array: waitable, and refuses a host copy."""

    def __init__(self):
        self.waited = 0

    def block_until_ready(self):
        self.waited += 1
        return self

    def __array__(self, *args, **kwargs):
        raise AssertionError("host copy inside the timed window")


class _Signals:
    def __init__(self, sources):
        self.spikes, self.V_m = _DeviceLeaf(), _DeviceLeaf()
        self.sources = _DeviceLeaf() if sources else None


def test_timed_wait_makes_no_host_copy():
    """Prereg E1: run_s is the compiled run's wall time; the device-to-host copy is timed apart (get_s)."""
    for with_sources in (True, False):
        sig = _Signals(with_sources)
        H._block_signals(sig)
        leaves = [sig.spikes, sig.V_m] + ([sig.sources] if with_sources else [])
        assert all(leaf.waited == 1 for leaf in leaves), "every device leaf must be waited on"


def test_cell_timeout_is_reported_not_raised():
    import sys

    out = H._run_cell_process([sys.executable, "-c", "import time; time.sleep(30)"], 0.5)
    assert out.startswith("TIMEOUT"), out
    ok = H._run_cell_process([sys.executable, "-c", "print('done')"], 30)
    assert ok.startswith("rc=0") and "done" in ok, ok


def test_recording_off_rss_reported():
    full = H.run_cell(_spec("full"))
    off = H.run_cell(_spec("off"))
    assert full["status"] == "MEASURED" and off["status"] == "MEASURED"
    # Presence (not a magnitude): both modes report peak host RSS.
    assert _finite(full["peak_host_rss_mb"])
    assert _finite(off["peak_host_rss_mb"])


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tiny_plan(tmp_path):
    """Plan in the driver's format (first three grid cells); receipts live in tmp_path."""
    cells = H.grid_cells()[:3]
    for c in cells:
        c["receipt"] = str(tmp_path / (c["cell_id"] + ".json"))
    plan = tmp_path / "plan.json"
    plan.write_bytes(json.dumps({"prereg_sha256": H._prereg_sha256(), "cells": cells},
                                indent=1).encode("utf-8"))
    return plan, cells


def _receipt(tmp_path, cell, status="MEASURED") -> bytes:
    data = json.dumps({"cell": cell["cell_id"], "status": status, "run_s": [0.5, 0.6]},
                      indent=1).encode("utf-8")
    (tmp_path / (cell["cell_id"] + ".json")).write_bytes(data)
    return data


def test_manifest_pins_present_receipts(tmp_path):
    plan, cells = _tiny_plan(tmp_path)
    assert len(cells) == 3
    data = [_receipt(tmp_path, cells[0]), _receipt(tmp_path, cells[1])]
    m = H.write_manifest(tmp_path, plan)
    assert m["n_cells"] == 3 and m["n_present"] == 2
    assert m["missing"] == [cells[2]["cell_id"]]
    ids = sorted([cells[0]["cell_id"], cells[1]["cell_id"]])
    assert [c["cell_id"] for c in m["cells"]] == ids
    by_id = {c["cell_id"]: c for c in m["cells"]}
    for cid, blob in zip((cells[0]["cell_id"], cells[1]["cell_id"]), data):
        entry = by_id[cid]
        assert entry["receipt"] == cid + ".json"
        assert entry["sha256"] == _sha(blob)
        assert entry["bytes"] == len(blob)
        assert entry["status"] == "MEASURED"
    assert m["plan_sha256"] == _sha(plan.read_bytes())
    assert m["prereg_sha256"] == H._prereg_sha256()
    assert m["git_sha"] == H._git_sha()
    on_disk = (tmp_path / "manifest.json").read_bytes()
    assert json.loads(on_disk) == m
    assert on_disk == (json.dumps(m, indent=1, sort_keys=True) + "\n").encode("utf-8")


def test_manifest_bytes_are_stable(tmp_path):
    plan, cells = _tiny_plan(tmp_path)
    _receipt(tmp_path, cells[0])
    H.write_manifest(tmp_path, plan)
    first = (tmp_path / "manifest.json").read_bytes()
    H.write_manifest(tmp_path, plan)
    assert (tmp_path / "manifest.json").read_bytes() == first
    assert first.endswith(b"\n") and b"\r" not in first


def test_manifest_records_plan_prereg_and_check_flags_a_change(tmp_path):
    plan, cells = _tiny_plan(tmp_path)
    old = "0" * 64
    assert old != H._prereg_sha256(), "fixture must differ from the current prereg"
    blob = json.loads(plan.read_bytes())
    blob["prereg_sha256"] = old
    plan.write_bytes(json.dumps(blob, indent=1).encode("utf-8"))
    _receipt(tmp_path, cells[0])
    assert H.write_manifest(tmp_path, plan)["prereg_sha256"] == old
    assert H.check_manifest(tmp_path) == ["prereg sha mismatch: PREREG_B1_0510.md changed since the plan"]


def test_check_untouched_dir_is_clean(tmp_path):
    plan, cells = _tiny_plan(tmp_path)
    _receipt(tmp_path, cells[0])
    _receipt(tmp_path, cells[1])
    H.write_manifest(tmp_path, plan)
    assert H.check_manifest(tmp_path) == []


def test_manifest_tampered_receipt_names_cell(tmp_path):
    plan, cells = _tiny_plan(tmp_path)
    _receipt(tmp_path, cells[0])
    _receipt(tmp_path, cells[1])
    H.write_manifest(tmp_path, plan)
    p = tmp_path / (cells[1]["cell_id"] + ".json")
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0x01
    p.write_bytes(bytes(data))
    problems = H.check_manifest(tmp_path)
    assert len(problems) == 1, problems
    assert cells[1]["cell_id"] in problems[0]
    assert cells[0]["cell_id"] not in problems[0]


def test_manifest_deleted_receipt_is_reported(tmp_path):
    plan, cells = _tiny_plan(tmp_path)
    _receipt(tmp_path, cells[0])
    _receipt(tmp_path, cells[1])
    H.write_manifest(tmp_path, plan)
    (tmp_path / (cells[0]["cell_id"] + ".json")).unlink()
    problems = H.check_manifest(tmp_path)
    assert len(problems) == 1, problems
    assert cells[0]["cell_id"] in problems[0] and "absent" in problems[0]


def test_manifest_unlisted_receipt_is_reported(tmp_path):
    plan, cells = _tiny_plan(tmp_path)
    _receipt(tmp_path, cells[0])
    H.write_manifest(tmp_path, plan)
    _receipt(tmp_path, cells[2])  # appears after pinning: must not pass silently
    problems = H.check_manifest(tmp_path)
    assert len(problems) == 1, problems
    assert cells[2]["cell_id"] in problems[0]


def test_manifest_plan_change_is_reported(tmp_path):
    plan, cells = _tiny_plan(tmp_path)
    _receipt(tmp_path, cells[0])
    H.write_manifest(tmp_path, plan)
    plan.write_bytes(plan.read_bytes() + b" ")
    problems = H.check_manifest(tmp_path)
    assert len(problems) == 1, problems
    assert "plan" in problems[0]


def test_manifest_stray_receipt_raises(tmp_path):
    plan, cells = _tiny_plan(tmp_path)
    _receipt(tmp_path, cells[0])
    (tmp_path / "stray_M9_N1_float32_cpu_B1_full.json").write_bytes(b"{}")
    try:
        H.write_manifest(tmp_path, plan)
    except ValueError as e:
        assert "stray_M9_N1_float32_cpu_B1_full.json" in str(e), e
    else:
        raise AssertionError("stray receipt was pinned silently")
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_cli_exit_codes(tmp_path):
    script = Path(__file__).with_name("b1_harness.py")
    plan, cells = _tiny_plan(tmp_path)
    _receipt(tmp_path, cells[0])
    _receipt(tmp_path, cells[1])
    base = [sys.executable, str(script), "--receipt-dir", str(tmp_path)]
    w = subprocess.run(base + ["--manifest"], capture_output=True, text=True, timeout=60)
    assert w.returncode == 0, w.stderr
    ok = subprocess.run(base + ["--check-manifest"], capture_output=True, text=True, timeout=60)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "manifest: cells=3 present=2 missing=1 problems=0" in ok.stdout, ok.stdout
    p = tmp_path / (cells[0]["cell_id"] + ".json")
    p.write_bytes(p.read_bytes() + b"\n")
    bad = subprocess.run(base + ["--check-manifest"], capture_output=True, text=True, timeout=60)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "manifest: cells=3 present=2 missing=1 problems=1" in bad.stdout, bad.stdout
